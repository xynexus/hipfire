// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! `gemv_oq_compact_multicol` against a CPU oracle.
//!
//! The small-batch twin of the compact WMMA GEMM: reads each weight row once and
//! accumulates B columns. This checks it directly on tiny shapes and PRINTS the
//! first few values on mismatch, because a pass/fail verdict says nothing about
//! whether the error is a scale, a permutation, or garbage.
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemv_oq_compact_multicol

use hipfire_rdna::{DType, Gpu};

const GROUP: usize = 256;

/// Interleaved on-disk block -> split planes, as the loader does.
fn split_planes(data: &[u8], n_blocks: usize, stride: usize) -> Vec<u8> {
    let nib = GROUP / 2;
    let side = stride - nib;
    let mut out = vec![0u8; data.len()];
    let side_base = n_blocks * nib;
    for b in 0..n_blocks {
        let src = b * stride;
        out[b * nib..(b + 1) * nib].copy_from_slice(&data[src + 2..src + 2 + nib]);
        let d = side_base + b * side;
        out[d..d + 2].copy_from_slice(&data[src..src + 2]);
        out[d + 2..d + side].copy_from_slice(&data[src + 2 + nib..src + stride]);
    }
    out
}

fn f16_to_f32(bits: u16) -> f32 {
    let s = ((bits >> 15) & 1) as u32;
    let e = ((bits >> 10) & 0x1f) as u32;
    let m = (bits & 0x3ff) as u32;
    let v = if e == 0 {
        (s << 31) | (m << 13)
    } else {
        (s << 31) | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(v)
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut seed = 0x2468_ACE1u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        (seed >> 16) as u32
    };
    let mut fail = 0usize;
    println!("gemv_oq_compact_multicol vs CPU oracle\n");
    println!("      M      K   B  N_out    max|rel|   verdict");

    for &(m, k, b, n_out) in &[
        (8usize, 256usize, 1usize, 1usize),
        (8, 256, 2, 1),
        (16, 512, 8, 3),
        (64, 1024, 16, 3),
        (512, 5120, 8, 3),
    ] {
        let ng = k / GROUP;
        let stride = 2 + GROUP / 2 + 2 * n_out;
        let nblk = m * ng;
        let mut blocks = vec![0u8; nblk * stride];
        for blk in 0..nblk {
            let off = blk * stride;
            // f16 scale in a sane range
            let bits = (((14 + rnd() % 3) as u16) << 10) | (rnd() % 1024) as u16;
            blocks[off..off + 2].copy_from_slice(&bits.to_le_bytes());
            for i in 0..GROUP / 2 {
                blocks[off + 2 + i] = (rnd() & 0xff) as u8;
            }
            let hdr = 2 + GROUP / 2;
            let mut used = [false; GROUP];
            for s in 0..n_out {
                let mut idx = (rnd() % GROUP as u32) as usize;
                while used[idx] {
                    idx = (idx + 1) % GROUP;
                }
                used[idx] = true;
                blocks[off + hdr + 2 * s] = idx as u8;
                blocks[off + hdr + 2 * s + 1] = (rnd() & 0xff) as u8;
                // The loader ZEROES the bulk nibble under every overlay.
                let nb = &mut blocks[off + 2 + idx / 2];
                *nb &= if idx % 2 == 0 { 0xf0 } else { 0x0f };
            }
        }

        // int8 activations + per-(b, group) scales, as the GEMM consumes them.
        let xq: Vec<i8> = (0..b * k).map(|_| (rnd() % 255) as i8).collect();
        let xs: Vec<f32> = (0..b * ng)
            .map(|_| (rnd() % 1000) as f32 * 1e-5 + 1e-4)
            .collect();

        // CPU oracle: decode bulk, overlay REPLACES, accumulate per group.
        let mut want = vec![0f32; b * m];
        for row in 0..m {
            for g in 0..ng {
                let off = (row * ng + g) * stride;
                let sw = f16_to_f32(u16::from_le_bytes([blocks[off], blocks[off + 1]]));
                let mut q = [0i8; GROUP];
                for i in 0..GROUP / 2 {
                    let p = blocks[off + 2 + i];
                    q[2 * i] = ((p & 0xf) as i8) << 4 >> 4;
                    q[2 * i + 1] = ((p >> 4) as i8) << 4 >> 4;
                }
                let hdr = 2 + GROUP / 2;
                for s in 0..n_out {
                    q[blocks[off + hdr + 2 * s] as usize] = blocks[off + hdr + 2 * s + 1] as i8;
                }
                for bb in 0..b {
                    let mut acc = 0i32;
                    for i in 0..GROUP {
                        acc += q[i] as i32 * xq[bb * k + g * GROUP + i] as i32;
                    }
                    want[bb * m + row] += acc as f32 * sw * xs[bb * ng + g];
                }
            }
        }

        let dev = split_planes(&blocks, nblk, stride);
        let wb = gpu.upload_raw(&dev, &[dev.len()]).expect("w");
        let xqb = gpu
            .upload_raw(
                unsafe { std::slice::from_raw_parts(xq.as_ptr() as *const u8, xq.len()) },
                &[xq.len()],
            )
            .expect("xq");
        let xsb = gpu.upload_f32(&xs, &[xs.len()]).expect("xs");
        let yb = gpu.alloc_tensor(&[b * m], DType::F32).expect("y");
        gpu.gemv_oq_compact_multicol(&wb, &xqb, &xsb, &yb, m, k, b, stride)
            .expect("launch");
        gpu.device_synchronize().expect("sync");
        let got = gpu.download_f32(&yb).expect("dl");

        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-30);
        let rel = want
            .iter()
            .zip(&got)
            .fold(0f32, |a, (p, q)| a.max((p - q).abs()))
            / scale;
        let ok = rel < 1e-4;
        if !ok {
            fail += 1;
        }
        println!(
            "  {m:>5}  {k:>5}  {b:>2}  {n_out:>5}   {rel:>9.2e}   {}",
            if ok { "PASS" } else { "FAIL" }
        );
        if !ok {
            for i in 0..6.min(want.len()) {
                println!("       [{i}] want {:>14.5}   got {:>14.5}", want[i], got[i]);
            }
        }
        for t in [wb, xqb, xsb, yb] {
            let _ = gpu.free_tensor(t);
        }
    }
    if fail == 0 {
        println!("\nparity_gemv_oq_compact_multicol: PASS");
    } else {
        println!("\nparity_gemv_oq_compact_multicol: FAIL ({fail} shape(s))");
        std::process::exit(1);
    }
}
