// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! `gemm_oq_compact_ladder` against a CPU oracle computed on the ORIGINAL
//! int8 activations — the point being that two iu4 passes reproduce int8
//! activation precision exactly, not approximately.
//!
//! Digit split: `u = v + 128; hi = (u>>4) - 8; lo = u & 15`, so
//! `v = 16*hi + lo` with `hi` in [-8,7] (signed int4) and `lo` in [0,15]
//! (unsigned int4). The kernel recombines in i32 before any f32 scaling, so the
//! only error against the oracle is f32 rounding of the per-group rescale.
//!
//! Bulk term only — the sparse weight overlay is a separate pass, as for the
//! 1-pass twin.
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemm_oq_compact_iu4x2

use hipfire_rdna::{DType, Gpu};

fn f16_to_f32(bits: u16) -> f32 {
    let s = ((bits >> 15) & 1) as u32;
    let e = ((bits >> 10) & 0x1f) as u32;
    let m = (bits & 0x3ff) as u32;
    let out = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while (m2 & 0x400) == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            m2 &= 0x3ff;
            (s << 31) | (e2 << 23) | (m2 << 13)
        }
    } else if e == 0x1f {
        (s << 31) | 0x7f80_0000 | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(out)
}

#[inline]
fn sext4(v: u8) -> i32 {
    (((v & 0xf) as i32) << 28) >> 28
}

fn split_planes(blocks: &[u8], nblk: usize, stride: usize, group: usize) -> Vec<u8> {
    let nib = group / 2;
    let side = stride - nib;
    let mut out = vec![0u8; nblk * stride];
    for b in 0..nblk {
        let src = b * stride;
        out[b * nib..b * nib + nib].copy_from_slice(&blocks[src + 2..src + 2 + nib]);
        let d = nblk * nib + b * side;
        out[d..d + 2].copy_from_slice(&blocks[src..src + 2]);
        out[d + 2..d + side].copy_from_slice(&blocks[src + 2 + nib..src + stride]);
    }
    out
}

/// Launch geometry for the CURRENT rung. Update together with the kernel.
const BM: usize = 16;   // rung 0
const BN: usize = 128;
const THREADS: usize = 256;

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut seed = 0xC0FFEE11u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        (seed >> 16) as u32
    };
    let mut fail = 0usize;
    println!("gemm_oq_compact_ladder vs int8-activation CPU oracle\n");
    println!("      M      K    B  N_out    G    max|rel|   verdict");

    for &(m, k, b, n_out, group) in &[
        (16usize, 256usize, 16usize, 1usize, 256usize),
        (64, 512, 32, 3, 256),
        (256, 1024, 64, 3, 256),
        (512, 5120, 128, 3, 256),
        (272, 2560, 256, 7, 256),
    ] {
        let ng = k / group;
        let stride = 2 + group / 2 + 2 * n_out;
        let nblk = m * ng;
        let mut blocks = vec![0u8; nblk * stride];
        for blk in 0..nblk {
            let off = blk * stride;
            let bits = (((14 + rnd() % 3) as u16) << 10) | (rnd() % 1024) as u16;
            blocks[off..off + 2].copy_from_slice(&bits.to_le_bytes());
            for i in 0..group / 2 {
                blocks[off + 2 + i] = (rnd() & 0xff) as u8;
            }
            let hdr = 2 + group / 2;
            let mut used = vec![false; group];
            for s in 0..n_out {
                let mut idx = (rnd() % group as u32) as usize;
                while used[idx] {
                    idx = (idx + 1) % group;
                }
                used[idx] = true;
                blocks[off + hdr + 2 * s] = idx as u8;
                blocks[off + hdr + 2 * s + 1] = (rnd() & 0xff) as u8;
                let nb = &mut blocks[off + 2 + idx / 2];
                *nb &= if idx % 2 == 0 { 0xf0 } else { 0x0f };
            }
        }

        // int8 activations, exactly as quantize_act_oq8 leaves them. The kernel
        // does the radix-16 digit split itself while staging into LDS.
        let x8: Vec<i8> = (0..b * k).map(|_| (rnd() % 256) as u8 as i8).collect();
        let x8u: Vec<u8> = x8.iter().map(|&v| v as u8).collect();
        let xs: Vec<f32> = (0..b * ng)
            .map(|_| (rnd() % 1000) as f32 * 1e-5 + 1e-4)
            .collect();

        let mut want = vec![0f32; b * m];
        for row in 0..m {
            for g in 0..ng {
                let off = (row * ng + g) * stride;
                let sw = f16_to_f32(u16::from_le_bytes([blocks[off], blocks[off + 1]]));
                for bb in 0..b {
                    let mut acc = 0i32;
                    for i in 0..group {
                        let p = blocks[off + 2 + i / 2];
                        let qw = if i % 2 == 0 {
                            sext4(p & 0xf)
                        } else {
                            sext4(p >> 4)
                        };
                        acc += qw * x8[bb * k + g * group + i] as i32;
                    }
                    want[bb * m + row] += acc as f32 * sw * xs[bb * ng + g];
                }
            }
        }

        let dev = split_planes(&blocks, nblk, stride, group);
        let wb = gpu.upload_raw(&dev, &[dev.len()]).expect("w");
        let x8b = gpu.upload_raw(&x8u, &[x8u.len()]).expect("x8");
        let xsb = gpu.upload_f32(&xs, &[xs.len()]).expect("xs");
        let yb = gpu.alloc_tensor(&[b * m], DType::F32).expect("y");
        gpu.gemm_oq_compact_ladder(&wb, &x8b, &xsb, &yb, m, k, b, stride, BM, BN, THREADS)
            .expect("launch");
        let got = gpu.download_f32(&yb).expect("dl");

        let mut max_abs = 0f32;
        let mut max_ref = 0f32;
        for i in 0..b * m {
            max_abs = max_abs.max((got[i] - want[i]).abs());
            max_ref = max_ref.max(want[i].abs());
        }
        let worst = max_abs / max_ref.max(1e-30);
        let ok = worst < 1e-5;
        if !ok {
            fail += 1;
        }
        println!(
            "  {m:>5} {k:>6} {b:>4} {n_out:>6} {group:>4}    {worst:>8.2e}   {}",
            if ok { "PASS" } else { "FAIL" }
        );
        for t in [wb, x8b, xsb, yb] {
            let _ = gpu.free_tensor(t);
        }
    }
    if fail == 0 {
        println!("\nparity_gemm_oq_compact_iu4x2: PASS");
    } else {
        println!("\nparity_gemm_oq_compact_iu4x2: FAIL ({fail} shape(s))");
        std::process::exit(1);
    }

    // Timing at the real gate/up shape, only reached if every parity shape passed.
    if fail == 0 {
        let (m, k, b, n_out, group) = (17408usize, 5120usize, 256usize, 3usize, 256usize);
        let ng = k / group;
        let stride = 2 + group / 2 + 2 * n_out;
        let mut blocks = vec![0u8; m * ng * stride];
        for (i, v) in blocks.iter_mut().enumerate() {
            *v = (i * 7 + 1) as u8;
        }
        let dev = split_planes(&blocks, m * ng, stride, group);
        let x8u: Vec<u8> = (0..b * k).map(|i| (i * 13 + 5) as u8).collect();
        let xs: Vec<f32> = (0..b * ng).map(|i| (i % 997) as f32 * 1e-5 + 1e-4).collect();
        let wb = gpu.upload_raw(&dev, &[dev.len()]).expect("w");
        let x8b = gpu.upload_raw(&x8u, &[x8u.len()]).expect("x");
        let xsb = gpu.upload_f32(&xs, &[xs.len()]).expect("xs");
        let yb = gpu.alloc_tensor(&[b * m], DType::F32).expect("y");
        gpu.gemm_oq_compact_ladder(&wb, &x8b, &xsb, &yb, m, k, b, stride, BM, BN, THREADS)
            .expect("warm");
        gpu.device_synchronize().expect("sync");
        let iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            gpu.gemm_oq_compact_ladder(&wb, &x8b, &xsb, &yb, m, k, b, stride, BM, BN, THREADS)
                .expect("run");
        }
        gpu.device_synchronize().expect("sync");
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let tops = 2.0 * 2.0 * m as f64 * k as f64 * b as f64 / (ms * 1e-3) / 1e12;
        println!("\ngate/up 17408x5120 B=256: {ms:.3} ms, {tops:.1} TOPS iu4 issue");
        println!("  reference: shipping wave64 1.661 ms / 54.9 TOPS");
    }
}
