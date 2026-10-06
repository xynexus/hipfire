// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! `gemm_q8_0_batched_chunked` at prefill widths (the 16x64 WMMA tile for the
//! 64-aligned rows, the scalar kernel for the rest) against the scalar
//! `gemm_q8_0_batched` alone, at the Qwen3.6-35B-A3B router (256x2048) and
//! shared-expert gate (1x2048) shapes, with B both a multiple of 64 and not. The
//! WMMA tile reads activations and weights as f16, so the gate is error relative to
//! the output scale at the f16 floor; rows past the aligned prefix must be
//! bit-identical (they take the scalar kernel either way).
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemm_q8_0_x64

use hipfire_rdna::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut seed = 0x2468_ACE1u32;
    let mut rnd = move || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        seed >> 8
    };
    let mut ok = true;
    for &(m, k, b) in &[
        (256usize, 2048usize, 1024usize),
        (256, 2048, 1000),
        (1, 2048, 4096),
        (1, 2048, 300),
    ] {
        // Q8_0: per 32 weights, an f16 scale then 32 int8.
        let mut w = vec![0u8; m * (k / 32) * 34];
        for blk in w.chunks_mut(34) {
            let n = 64 + rnd() % 512; // scale n/16384, exact in f16
            let e = 31 - n.leading_zeros();
            let bits = (((e as i32 - 14 + 15) as u16) << 10) | (((n << (10 - e)) & 0x3ff) as u16);
            blk[..2].copy_from_slice(&bits.to_le_bytes());
            for q in &mut blk[2..] {
                *q = ((rnd() % 255) as i32 - 127) as i8 as u8;
            }
        }
        let x: Vec<f32> = (0..b * k)
            .map(|_| (rnd() % 2000) as f32 * 1e-3 - 1.0)
            .collect();
        let wd = gpu.upload_raw(&w, &[w.len()]).unwrap();
        let xd = gpu.upload_f32(&x, &[x.len()]).unwrap();
        let (yr, yg) = (
            gpu.alloc_tensor(&[b * m], DType::F32).unwrap(),
            gpu.alloc_tensor(&[b * m], DType::F32).unwrap(),
        );
        let mut off = 0;
        while off < b {
            let take = (b - off).min(64);
            let (xs, ys) = (
                xd.sub_offset(off * k, take * k),
                yr.sub_offset(off * m, take * m),
            );
            gpu.gemm_q8_0_batched(&wd, &xs, &ys, m, k, take).unwrap();
            off += take;
        }
        gpu.gemm_q8_0_batched_chunked(&wd, &xd, &yg, m, k, b)
            .unwrap();
        let (r, g) = (
            gpu.download_f32(&yr).unwrap(),
            gpu.download_f32(&yg).unwrap(),
        );
        let scale = r.iter().fold(0f32, |a, v| a.max(v.abs()));
        let err = r
            .iter()
            .zip(&g)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()))
            / scale;
        let tail = b / 64 * 64 * m;
        let tail_exact = r[tail..]
            .iter()
            .zip(&g[tail..])
            .all(|(x, y)| x.to_bits() == y.to_bits());
        let pass = err <= 2e-3 && tail_exact;
        ok &= pass;
        println!(
            "M={m} K={k} B={b}: max_err/scale={err:.2e} tail_bit_exact={tail_exact} -> {}",
            if pass { "PASS" } else { "FAIL" }
        );
        for t in [xd, yr, yg, wd] {
            let _ = gpu.free_tensor(t);
        }
    }
    if !ok {
        std::process::exit(1);
    }
    println!("ALL PASS");
}
