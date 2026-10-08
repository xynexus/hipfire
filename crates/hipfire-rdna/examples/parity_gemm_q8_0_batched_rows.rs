// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! `gemm_q8_0_batched` at 2..64 rows (the row-split kernel) must be
//! BIT-IDENTICAL to the same rows run one at a time (the one-wave-per-row
//! kernel), at the Qwen3.6-35B-A3B MoE router (256x2048) and shared-expert gate
//! (1x2048) shapes. Also prints warm timings.
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemm_q8_0_batched_rows

use hipfire_rdna::{DType, Gpu};

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            s as f32 / 1_073_741_824.0 - 1.0
        })
        .collect()
}

/// Random Q8_0 rows: per 32-block an f16 scale and 32 int8 codes.
fn q8_weights(m: usize, k: usize) -> Vec<u8> {
    let r = lcg(7, m * k);
    let mut out = Vec::with_capacity(m * k / 32 * 34);
    for (i, blk) in r.chunks(32).enumerate() {
        // An f16 scale around 2^-6, varied per block (bits 0x2400 + i % 512).
        out.extend_from_slice(&(0x2400u16 + (i % 512) as u16).to_le_bytes());
        out.extend(blk.iter().map(|v| (v * 127.0) as i8 as u8));
    }
    out
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    let mut ok = true;
    for &(m, k) in &[(256usize, 2048usize), (1, 2048)] {
        let a = gpu
            .upload_raw(&q8_weights(m, k), &[m * k / 32 * 34])
            .unwrap();
        for &n in &[2usize, 3, 4, 5, 8, 13, 16, 32, 33, 64] {
            let x: Vec<f32> = lcg(3 + n as u32, n * k);
            // F32, not raw: sub_offset counts elements of the tensor's dtype.
            let xd = gpu.upload_f32(&x, &[n * k]).unwrap();
            let y_new = gpu.zeros(&[n * m], DType::F32).unwrap();
            let y_ref = gpu.zeros(&[n * m], DType::F32).unwrap();
            gpu.gemm_q8_0_batched(&a, &xd, &y_new, m, k, n).unwrap();
            // Reference: one row per call -- the one-wave-per-row kernel.
            for off in 0..n {
                let take = 1;
                let xs = xd.sub_offset(off * k, take * k);
                let ys = y_ref.sub_offset(off * m, take * m);
                gpu.gemm_q8_0_batched(&a, &xs, &ys, m, k, take).unwrap();
            }
            gpu.device_synchronize().unwrap();
            let (yn, yr) = (
                gpu.download_f32(&y_new).unwrap(),
                gpu.download_f32(&y_ref).unwrap(),
            );
            let bad = yn
                .iter()
                .zip(&yr)
                .filter(|(p, q)| p.to_bits() != q.to_bits())
                .count();
            if let Some(i) = (0..yn.len()).find(|&i| yn[i].to_bits() != yr[i].to_bits()) {
                let maxd = yn
                    .iter()
                    .zip(&yr)
                    .fold(0f32, |a, (p, q)| a.max((p - q).abs()));
                println!(
                    "    first mismatch at {i} (row {}): {} vs {}  max|d| {maxd:.3e}",
                    i / m,
                    yn[i],
                    yr[i]
                );
            }
            let time = |gpu: &mut Gpu, rows: usize| {
                let t = std::time::Instant::now();
                for _ in 0..50 {
                    for off in (0..n).step_by(rows) {
                        let take = (n - off).min(rows);
                        let xs = xd.sub_offset(off * k, take * k);
                        let ys = y_ref.sub_offset(off * m, take * m);
                        gpu.gemm_q8_0_batched(&a, &xs, &ys, m, k, take).unwrap();
                    }
                }
                gpu.device_synchronize().unwrap();
                t.elapsed().as_secs_f64() * 1e6 / 50.0
            };
            let (t_new, t_old1) = (time(&mut gpu, n), time(&mut gpu, 1));
            println!(
                "  M={m} K={k} n={n}: {bad} bit mismatches of {}  row-split {t_new:7.1} us  (1-row calls {t_old1:7.1} us)",
                yn.len()
            );
            // FNV-1a of the batched output, to compare builds of the kernel.
            let hash = yn.iter().fold(0xcbf29ce484222325u64, |h, v| {
                (h ^ v.to_bits() as u64).wrapping_mul(0x100000001b3)
            });
            println!("    output hash {hash:016x}");
            ok &= bad == 0;
        }
    }
    println!(
        "parity_gemm_q8_0_batched_rows -> {}",
        if ok { "PASS" } else { "FAIL" }
    );
    if !ok {
        std::process::exit(1);
    }
}
