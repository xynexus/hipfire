#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for `gemm_bf16_tiled_wmma_lds` (LDS-staged DiT bf16 GEMM) vs the
//! zero-LDS `gemm_bf16_tiled_wmma`. Both use `v_wmma_f32_16x16x16_bf16_w32` and
//! accumulate over K in the SAME order, so they must be **BIT-EXACT** (max_abs==0).
//! Small shapes also check a CPU bf16 reference (loose). Covers unaligned M/B.
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemm_bf16_tiled_wmma_lds

use hipfire_rdna::{DType, Gpu};

fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + lsb) >> 16) as u16
}
fn f32_to_bf16_value(x: f32) -> f32 {
    f32::from_bits((f32_to_bf16_bits(x) as u32) << 16)
}
fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|&v| f32_to_bf16_bits(v).to_le_bytes()).collect()
}
fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            ((s as f32 / 2_147_483_648.0) - 0.5) * 0.2
        })
        .collect()
}

fn run(gpu: &mut Gpu, m: usize, k: usize, b: usize) -> bool {
    let w = lcg(1, m * k);
    let x = lcg(2, b * k);
    let mut w_gpu = gpu.upload_raw(&bf16_bytes(&w), &[m, k]).unwrap();
    w_gpu.dtype = DType::BF16;
    let x_gpu = gpu.upload_f32(&x, &[b, k]).unwrap();
    let y_tiled = gpu.alloc_tensor(&[b, m], DType::F32).unwrap();
    let y_lds = gpu.alloc_tensor(&[b, m], DType::F32).unwrap();

    gpu.gemm_bf16_tiled_wmma(&w_gpu, &x_gpu, &y_tiled, m, k, b, 4, 4).unwrap();
    gpu.gemm_bf16_tiled_wmma_lds(&w_gpu, &x_gpu, &y_lds, m, k, b).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let yt = gpu.download_f32(&y_tiled).unwrap();
    let yl = gpu.download_f32(&y_lds).unwrap();

    // LDS vs tiled: BIT-EXACT.
    let mut exact = 0.0f32;
    let mut mag = 0.0f32;
    for i in 0..b * m {
        exact = exact.max((yl[i] - yt[i]).abs());
        mag = mag.max(yt[i].abs());
    }
    // CPU bf16 ref (only for small K — O(M·K·B)).
    let (mut cpu_max, mut cpu_tol) = (0.0f32, 0.0f32);
    if k <= 256 {
        for bb in 0..b {
            for mm in 0..m {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += f32_to_bf16_value(w[mm * k + kk]) * f32_to_bf16_value(x[bb * k + kk]);
                }
                cpu_max = cpu_max.max((yl[bb * m + mm] - acc).abs());
            }
        }
        cpu_tol = 3.0 * mag.max(1.0) * 2f32.powi(-8); // bf16 mantissa
    }
    let exact_pass = exact == 0.0;
    let cpu_pass = k > 256 || cpu_max <= cpu_tol;
    let pass = exact_pass && cpu_pass;
    println!(
        "  M={m:<6} K={k:<5} B={b:<5}: LDS-vs-tiled max_abs={exact:.6} [{}]  {}  -> {}",
        if exact_pass { "BIT-EXACT" } else { "DIFF!" },
        if k <= 256 {
            format!("cpu={cpu_max:.5}(tol {cpu_tol:.5})[{}]", if cpu_pass { "ok" } else { "FAIL" })
        } else {
            "cpu=skip(K>256)".to_string()
        },
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP gemm_bf16_tiled_wmma_lds parity: {} lacks wave32 WMMA", gpu.arch);
        return;
    }
    println!("gemm_bf16_tiled_wmma_lds parity on {}", gpu.arch);
    let shapes = [
        // aligned to BM=64 / BN=128
        (128usize, 64usize, 128usize),
        (256, 128, 256),
        (1536, 6144, 256), // DiT GQA-ish
        (6144, 6144, 128), // DiT attn (real shape, small B)
        // unaligned M (not %64) and B (not %128)
        (1000, 64, 100),
        (17, 128, 5),
        (64, 192, 129),
    ];
    let mut all = true;
    for (m, k, b) in shapes {
        all &= run(&mut gpu, m, k, b);
    }
    println!("{}", if all { "ALL PASS" } else { "SOME FAILED" });
    if !all {
        std::process::exit(1);
    }
}
