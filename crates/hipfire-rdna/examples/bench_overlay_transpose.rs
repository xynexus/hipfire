// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Head-to-head: compact iu8 (`gemm_oq_compact_grouped_wmma`) vs compact iu4
//! K-MAJOR overlay correction vs the shipped b-major one. Same math, same
//! iu4 arm feeds the bulk nibbles to the matrix core raw and takes int4
//! activations, halving the activation traffic.
//!
//! This kernel is 85.1% of a 2048-token prefill profile and the whole prefill
//! runs at ~14% of the ~56 TOPS int8 peak, which is why prefill is FLAT at
//! ~160 tok/s and declining with length instead of amortizing upward.
//! Reports effective TOPS, since this path is compute-bound, not
//! bandwidth-bound like the B<=16 GEMV.
//!
//!   cargo run --release -p hipfire-rdna --example bench_overlay_transpose

use hipfire_rdna::{DType, Gpu};
use std::time::Instant;

const GROUP: usize = 256;

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut seed = 0x1357_9BDFu32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        (seed >> 16) as u32
    };
    let n_out = 3usize;
    let stride = 2 + GROUP / 2 + 2 * n_out; // 136 at n_out=3 => 4.25 bits
    let iters = 10usize;

    println!("gemv_oq_compact_multicol at Qwen3.8-27B shapes (n_out={n_out}, stride={stride})");
    println!("weight bytes only; 233 GB/s is the measured achievable ceiling\n");
    println!("  proj             M      K    B    b-major_ms   transpose_ms   corr_T_ms   total_T   speedup   max|diff|");

    for &(name, m, k, b) in &[
        ("gate/up", 17408usize, 5120usize, 256usize),
        ("down", 5120, 17408, 256),
        ("qkv", 6144, 5120, 256),
        ("wo", 5120, 4096, 256),
        ("gate/up B=128", 17408, 5120, 128),
        ("gate/up B=512", 17408, 5120, 512),
    ] {
        let ng = k / GROUP;
        let nblk = m * ng;
        let bytes = nblk * stride;
        let mut blocks = vec![0u8; bytes];
        for blk in 0..nblk {
            let off = blk * stride;
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
                let nb = &mut blocks[off + 2 + idx / 2];
                *nb &= if idx % 2 == 0 { 0xf0 } else { 0x0f };
            }
        }
        // Split planes: all nibble groups first, then all [f16 scale][table].
        let side = stride - GROUP / 2;
        let mut dev = vec![0u8; bytes];
        for blk in 0..nblk {
            let src = blk * stride;
            dev[blk * (GROUP / 2)..blk * (GROUP / 2) + GROUP / 2]
                .copy_from_slice(&blocks[src + 2..src + 2 + GROUP / 2]);
            let d = nblk * (GROUP / 2) + blk * side;
            dev[d..d + 2].copy_from_slice(&blocks[src..src + 2]);
            dev[d + 2..d + side].copy_from_slice(&blocks[src + 2 + GROUP / 2..src + stride]);
        }

        let xq: Vec<i8> = (0..b * k).map(|_| (rnd() % 255) as i8).collect();
        // Packed signed int4 activations, [B, K/2], byte = k_even | k_odd<<4.
        let x4: Vec<u8> = (0..b * k / 2).map(|_| (rnd() & 0xff) as u8).collect();
        // Two digit planes for the 8-bit-via-2-passes arm (same total bytes as
        // one int8 activation, which is the point).
        let xhi: Vec<u8> = (0..b * k / 2).map(|_| (rnd() & 0xff) as u8).collect();
        let xlo: Vec<u8> = (0..b * k / 2).map(|_| (rnd() & 0xff) as u8).collect();
        let xs: Vec<f32> = (0..b * ng)
            .map(|_| (rnd() % 1000) as f32 * 1e-5 + 1e-4)
            .collect();
        let wb = gpu.upload_raw(&dev, &[dev.len()]).expect("w");
        let xqb = gpu
            .upload_raw(
                unsafe { std::slice::from_raw_parts(xq.as_ptr() as *const u8, xq.len()) },
                &[xq.len()],
            )
            .expect("xq");
        let xsb = gpu.upload_f32(&xs, &[xs.len()]).expect("xs");
        let x4b = gpu.upload_raw(&x4, &[x4.len()]).expect("x4");
        let _ = (&xqb, &xhi, &xlo);

        // outputs: one per arm, both accumulate in place from zero
        let zeros0 = vec![0f32; b * m];
        let y0 = gpu.upload_f32(&zeros0, &[b * m]).expect("y0");
        let y1 = gpu.upload_f32(&zeros0, &[b * m]).expect("y1");
        // k-major staging
        let xt = gpu.alloc_tensor(&[k * b], DType::Raw).expect("xt");
        let xst = gpu.alloc_tensor(&[ng * b], DType::F32).expect("xst");

        let iters = 10usize;

        // --- arm A: shipped b-major correction ---
        gpu.gemv_oq_compact_overlay_correct(&wb, &x4b, &xsb, &y0, m, k, b, GROUP, stride)
            .expect("warm0");
        gpu.device_synchronize().expect("s");
        let t0 = Instant::now();
        for _ in 0..iters {
            gpu.gemv_oq_compact_overlay_correct(&wb, &x4b, &xsb, &y0, m, k, b, GROUP, stride)
                .expect("a0");
        }
        gpu.device_synchronize().expect("s");
        let ms0 = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

        // --- arm B: transpose + k-major correction ---
        gpu.oq_compact_x4_transpose(&x4b, &xsb, &xt, &xst, b, k, ng)
            .expect("warmT");
        gpu.oq_compact_overlay_correct_t(&wb, &xt, &xst, &y1, m, k, b, GROUP, stride)
            .expect("warm1");
        gpu.device_synchronize().expect("s");
        let tt = Instant::now();
        for _ in 0..iters {
            gpu.oq_compact_x4_transpose(&x4b, &xsb, &xt, &xst, b, k, ng)
                .expect("t");
        }
        gpu.device_synchronize().expect("s");
        let mst = tt.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let t1 = Instant::now();
        for _ in 0..iters {
            gpu.oq_compact_overlay_correct_t(&wb, &xt, &xst, &y1, m, k, b, GROUP, stride)
                .expect("a1");
        }
        gpu.device_synchronize().expect("s");
        let ms1 = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;

        // --- correctness: both arms ran `iters+1` accumulations from zero ---
        let d0 = gpu.download_f32(&y0).expect("d0");
        let d1 = gpu.download_f32(&y1).expect("d1");
        let mut maxd = 0f32;
        for i in 0..d0.len() {
            let e = (d0[i] - d1[i]).abs();
            let s = d0[i].abs().max(d1[i].abs()).max(1e-6);
            if e / s > maxd {
                maxd = e / s;
            }
        }
        let total_t = mst + ms1;
        println!(
            "  {name:<14} {m:>6} {k:>6} {b:>4} {ms0:>11.3} {mst:>14.3} {ms1:>11.3} {total_t:>9.3} {:>9.2}x {maxd:>11.2e}",
            ms0 / total_t
        );
    }
}
