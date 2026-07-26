// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for the OQ shared-expert-down kernels
//! (`gemv_oq{4,8}g256_residual_sigmoid_scaled_gpu_batched`) vs a CPU oracle.
//!
//! Each checks the N-batched W4A16 / W8A16 dot with a fused sigmoid-scaled
//! residual add, exactly as the qwen35 MoE shared-expert down accumulator uses:
//!   y_batch[t, row] = y0[t, row] + sigmoid(c[t]) · Σ_g sc[row,g]·Σ qw·x_batch[t]
//! Dense OQ layout: OQ4 W=[M,K/2] packed int4 + Ws=[M,K/256] f32; OQ8 W=[M,K] int8.
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemv_oq_residual_sigmoid_scaled [M K N]

use hipfire_rdna::Gpu;

fn lcgf(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            -1.0 + (s as f32 / 2_147_483_648.0) * 2.0
        })
        .collect()
}
fn lcgu(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s >> 11) as u8
        })
        .collect()
}
fn sext4(nib: u8) -> i32 {
    let v = (nib & 0xf) as i32;
    (v << 28) >> 28
}
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    let n: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let group = 256usize;
    assert_eq!(k % group, 0);
    let ng = k / group;

    let mut gpu = Gpu::init().unwrap();

    // Shared inputs.
    let x = lcgf(3, n * k); // [N, K]
    let c = lcgf(7, n); // [N] gate logits
    let scales: Vec<f32> = lcgf(0x21, m * ng)
        .iter()
        .map(|v| 0.01 + v.abs() * 0.25)
        .collect();
    let y0 = lcgf(0x99, n * m); // initial residual

    let xd = gpu.upload_raw(&fbytes(&x), &[n, k]).unwrap();
    let cd = gpu.upload_raw(&fbytes(&c), &[n]).unwrap();
    let wsd = gpu.upload_raw(&fbytes(&scales), &[m, ng]).unwrap();

    let mut worst = 0.0f32;

    // ── OQ4: W = [M, K/2] packed signed int4 ───────────────────────────────
    {
        let wnib = lcgu(1, m * k / 2); // packed nibbles
        let wd = gpu.upload_raw(&wnib, &[m * k / 2]).unwrap();
        let yd = gpu.upload_raw(&fbytes(&y0), &[n, m]).unwrap();
        gpu.gemv_oq4g256_residual_sigmoid_scaled_gpu_batched(
            &wd, &wsd, &xd, &yd, &cd, m, k, group, n,
        )
        .unwrap();
        gpu.device_synchronize().unwrap();
        let got = gpu.download_f32(&yd).unwrap();
        let (ma, mag) = oracle_check(&got, &y0, &c, &scales, &x, m, k, ng, n, |row, col| {
            let byte = wnib[row * (k / 2) + col / 2];
            let nib = if col & 1 == 0 { byte & 0xf } else { byte >> 4 };
            sext4(nib) as f32
        });
        let tol = 1e-3 * mag.max(1.0);
        worst = worst.max(ma);
        println!(
            "  oq4 M={m} K={k} N={n}: max_abs={ma:.5} (mag={mag:.2}) tol={tol:.5} -> {}",
            if ma <= tol { "PASS" } else { "FAIL" }
        );
        assert!(ma <= tol, "oq4 parity FAIL");
    }

    // ── OQ8: W = [M, K] signed int8 ────────────────────────────────────────
    {
        let wi8 = lcgu(2, m * k); // int8 bytes (interpreted signed)
        let wd = gpu.upload_raw(&wi8, &[m * k]).unwrap();
        let yd = gpu.upload_raw(&fbytes(&y0), &[n, m]).unwrap();
        gpu.gemv_oq8g256_residual_sigmoid_scaled_gpu_batched(
            &wd, &wsd, &xd, &yd, &cd, m, k, group, n,
        )
        .unwrap();
        gpu.device_synchronize().unwrap();
        let got = gpu.download_f32(&yd).unwrap();
        let (ma, mag) = oracle_check(&got, &y0, &c, &scales, &x, m, k, ng, n, |row, col| {
            (wi8[row * k + col] as i8) as f32
        });
        let tol = 1e-3 * mag.max(1.0);
        worst = worst.max(ma);
        println!(
            "  oq8 M={m} K={k} N={n}: max_abs={ma:.5} (mag={mag:.2}) tol={tol:.5} -> {}",
            if ma <= tol { "PASS" } else { "FAIL" }
        );
        assert!(ma <= tol, "oq8 parity FAIL");
    }

    println!(
        "parity_gemv_oq_residual_sigmoid_scaled on {}: worst max_abs={worst:.5} -> PASS",
        gpu.arch
    );
}

fn fbytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[allow(clippy::too_many_arguments)]
fn oracle_check(
    got: &[f32],
    y0: &[f32],
    c: &[f32],
    scales: &[f32],
    x: &[f32],
    m: usize,
    k: usize,
    ng: usize,
    n: usize,
    qw: impl Fn(usize, usize) -> f32,
) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    let mut max_mag = 0.0f32;
    for t in 0..n {
        let gate = sigmoid(c[t]);
        for row in 0..m {
            // Group-wise accumulation, matching the kernel's Σ_g sc_g·(Σ qw·x).
            let mut acc = 0.0f32;
            for g in 0..ng {
                let mut gsum = 0.0f32;
                for j in 0..256 {
                    let col = g * 256 + j;
                    gsum += qw(row, col) * x[t * k + col];
                }
                acc += gsum * scales[row * ng + g];
            }
            let want = y0[t * m + row] + gate * acc;
            let g = got[t * m + row];
            max_abs = max_abs.max((g - want).abs());
            max_mag = max_mag.max(want.abs());
        }
    }
    (max_abs, max_mag)
}
