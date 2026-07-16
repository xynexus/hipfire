// SPDX-License-Identifier: Apache-2.0
//! Parity: mixed-precision Opus GEMM (unsigned codes + zero-point fold) vs a CPU
//! reference, for W{8,4,2,1}A8 on one kernel body.
//!
//! Validates `gemm_opus_tiled_wmma_u` — the unsigned-weight / offset-fold path
//! (`gemm_opus_tiled_wmma.hip`). Weights are stored as unsigned codes u∈[0,2^b-1],
//! the WMMA weight operand is flagged unsigned, and the symmetric zero-point
//! Z=2^(b-1) is folded out per group with the activation group sum. The math is
//! the `hipfire_quantize::opus_lowbit` reference (inlined here to avoid a dep
//! cycle; the CPU identity `Σ u·x − Z·Σx == Σ(u−Z)·x` is proven by that crate's
//! unit tests).
//!
//! PASS = GPU output matches the CPU fold reference to <1e-3 relative L2 for every
//! width (proves the kernel implements the fold correctly). SQNR vs f32 is
//! reported for context (poor at 1-bit, as expected).
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemm_wNa8u_fold [M K B]

use hipfire_rdna::{DType, Gpu};

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    let mut u = || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        (s as f32 + 0.5) / 2_147_483_648.0
    };
    (0..n)
        .map(|_| {
            let u1 = u().max(1e-7);
            let u2 = u();
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        })
        .collect()
}

const fn zero_point(bits: u32) -> i32 {
    1 << (bits - 1)
}

/// Per-group symmetric unsigned quant (mirrors opus_lowbit::quantize_symmetric).
fn quantize_weights(w: &[f32], m: usize, k: usize, group: usize, bits: u32) -> (Vec<u8>, Vec<f32>) {
    let z = zero_point(bits);
    let (qmin, qmax) = (-z, z - 1);
    let ng = k / group;
    let mut codes = vec![0u8; m * k];
    let mut scales = vec![0.0f32; m * ng];
    for row in 0..m {
        for g in 0..ng {
            let base = row * k + g * group;
            let amax = (0..group).map(|i| w[base + i].abs()).fold(0.0f32, f32::max);
            let s = if amax > 0.0 { amax / z as f32 } else { 1.0 };
            scales[row * ng + g] = s;
            for i in 0..group {
                let q = ((w[base + i] / s).round() as i32).clamp(qmin, qmax);
                codes[base + i] = (q + z) as u8;
            }
        }
    }
    (codes, scales)
}

/// Dense LSB-first pack of one row's codes → `k*bits/8` bytes.
fn pack_row_dense(row_codes: &[u8], bits: u32) -> Vec<u8> {
    let per_byte = (8 / bits) as usize;
    let mask = ((1u32 << bits) - 1) as u8;
    let mut out = vec![0u8; row_codes.len() / per_byte];
    for (i, &c) in row_codes.iter().enumerate() {
        out[i / per_byte] |= (c & mask) << ((i % per_byte) as u32 * bits);
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let m: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let k: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let b: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let group = 64usize;
    assert_eq!(k % group, 0, "K must be a multiple of group ({group})");

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!(
            "SKIP parity_gemm_wNa8u_fold: {} lacks wave32 WMMA",
            gpu.arch
        );
        return;
    }
    let ng = k / group;

    let w = lcg(1, m * k);
    let x = lcg(2, b * k);

    // ── Activations quantized ON GPU by quantize_act_oq8_sum (validates the Xsum
    // wiring): int8 + per-group scale + per-group signed sum. Cross-check vs CPU.
    let x_dev = gpu.upload_f32(&x, &[b, k]).unwrap();
    let xq_dev = gpu.upload_raw(&vec![0u8; b * k], &[b, k]).unwrap();
    let xs_dev = gpu.alloc_tensor(&[b * ng], DType::F32).unwrap();
    let xsum_dev = gpu.upload_raw(&vec![0u8; b * ng * 4], &[b * ng]).unwrap();
    gpu.quantize_act_oq8_sum(&x_dev, &xq_dev, &xs_dev, &xsum_dev, b, k, group)
        .unwrap();
    gpu.device_synchronize().unwrap();

    let xq: Vec<i8> = gpu
        .download_raw(&xq_dev, b * k)
        .unwrap()
        .iter()
        .map(|&v| v as i8)
        .collect();
    let xs = gpu.download_f32(&xs_dev).unwrap();
    let xsum: Vec<i32> = gpu
        .download_raw(&xsum_dev, b * ng * 4)
        .unwrap()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // Cross-check Xsum against a CPU recompute of Σ xq over each group.
    let mut xsum_mismatch = 0usize;
    for bi in 0..b {
        for g in 0..ng {
            let cpu: i32 = (0..group).map(|i| xq[bi * k + g * group + i] as i32).sum();
            if cpu != xsum[bi * ng + g] {
                xsum_mismatch += 1;
            }
        }
    }
    println!(
        "quantize_act_oq8_sum on {}: Xsum vs CPU recompute -> {} ({} mismatch)",
        gpu.arch,
        if xsum_mismatch == 0 {
            "EXACT"
        } else {
            "BROKEN"
        },
        xsum_mismatch
    );
    if xsum_mismatch != 0 {
        std::process::exit(1);
    }

    let mut all_pass = true;
    for &bits in &[8u32, 4, 2, 1] {
        let z = zero_point(bits);
        let (codes, wscale) = quantize_weights(&w, m, k, group, bits);
        // Pack each row densely, concatenate (row stride = k*bits/8).
        let stride = k * bits as usize / 8;
        let mut packed = vec![0u8; m * stride];
        for row in 0..m {
            let pr = pack_row_dense(&codes[row * k..(row + 1) * k], bits);
            packed[row * stride..(row + 1) * stride].copy_from_slice(&pr);
        }
        let wp_dev = gpu.upload_raw(&packed, &[m * stride]).unwrap();
        let ws_dev = gpu.upload_f32(&wscale, &[m * ng]).unwrap();
        let yf_dev = gpu.alloc_tensor(&[b * m], DType::F32).unwrap();

        gpu.gemm_opus_tiled_wmma_u(
            bits as usize,
            &wp_dev,
            &ws_dev,
            &xq_dev,
            &xs_dev,
            &xsum_dev,
            &yf_dev,
            m,
            k,
            b,
            group,
            2,
            2,
        )
        .unwrap();
        gpu.device_synchronize().unwrap();
        let y = gpu.download_f32(&yf_dev).unwrap();

        // ── CPU fold reference + f32 truth (same quantized inputs).
        let mut ref_sq = 0.0f64;
        let mut gpu_vs_ref_sq = 0.0f64;
        let (mut sig, mut err, mut dot, mut nr, mut ng2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for bi in 0..b {
            for mi in 0..m {
                // CPU fold: Σ_g sw·sx·(Σ u·x − Z·Σx)
                let mut fold = 0.0f32;
                for g in 0..ng {
                    let base = g * group;
                    let mut iacc = 0i32;
                    for i in 0..group {
                        iacc += codes[mi * k + base + i] as i32 * xq[bi * k + base + i] as i32;
                    }
                    let folded = iacc - z * xsum[bi * ng + g];
                    fold += folded as f32 * wscale[mi * ng + g] * xs[bi * ng + g];
                }
                let g_out = y[bi * m + mi];
                ref_sq += (fold as f64).powi(2);
                gpu_vs_ref_sq += ((fold - g_out) as f64).powi(2);
                // f32 truth from dequantized operands
                let r: f32 = (0..k)
                    .map(|ki| {
                        let g = ki / group;
                        let w_hat = (codes[mi * k + ki] as i32 - z) as f32 * wscale[mi * ng + g];
                        let x_hat = xq[bi * k + ki] as f32 * xs[bi * ng + g];
                        w_hat * x_hat
                    })
                    .sum();
                sig += (r as f64).powi(2);
                err += ((r - g_out) as f64).powi(2);
                dot += r as f64 * g_out as f64;
                nr += (r as f64).powi(2);
                ng2 += (g_out as f64).powi(2);
            }
        }
        let gpu_vs_ref = (gpu_vs_ref_sq / ref_sq.max(1e-30)).sqrt();
        let sqnr_db = 10.0 * (sig / err.max(1e-30)).log10();
        let cos = dot / (nr.sqrt() * ng2.sqrt() + 1e-30);
        let pass = gpu_vs_ref < 1e-3;
        all_pass &= pass;
        println!(
            "W{bits}A8u fold  M={m} K={k} B={b} g={group} on {}: GPU-vs-CPUfold relL2={gpu_vs_ref:.2e} | vs f32 cos={cos:.5} SQNR={sqnr_db:.1}dB -> {}",
            gpu.arch,
            if pass { "PASS" } else { "FAIL" }
        );
    }

    if !all_pass {
        std::process::exit(1);
    }
}
