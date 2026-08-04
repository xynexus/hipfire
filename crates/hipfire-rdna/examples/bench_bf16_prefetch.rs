// SPDX-License-Identifier: Apache-2.0
//! Chases the ~15% dependent-load stall the free-ALU probe exposed (plan §12c):
//! benches software-pipelined (prefetched) variants of the production register-
//! tiled bf16 DiT GEMM against it, and parity-checks each for bit-exactness.
//!
//! All tilings accumulate over K in the same order with the same WMMA builtin, so
//! every variant must be BIT-EXACT to `gemm_bf16_tiled_wmma_4x4`.
//! Includes a 30-launch warmup and a baseline re-measure at the END, so a win
//! cannot be a clock-ramp artifact.

use hipfire_rdna::{DType, Gpu};
use std::time::Instant;

fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn bf16_bytes(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .flat_map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            f32_to_bf16_bits(((s as f32 / 2_147_483_648.0) - 0.5) * 0.2).to_le_bytes()
        })
        .collect()
}
fn f32v(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            ((s as f32 / 2_147_483_648.0) - 0.5) * 0.2
        })
        .collect()
}

const VARIANTS: [(&str, usize, usize); 5] = [
    ("gemm_bf16_pf_4x4_x", 4, 4),
    ("gemm_bf16_pf_4x4_a", 4, 4),
    ("gemm_bf16_pf_4x4_both", 4, 4),
    ("gemm_bf16_pf_4x2_both", 4, 2),
    ("gemm_bf16_pf_2x2_both", 2, 2),
];

fn main() {
    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP: no wmma");
        return;
    }
    let iters = 10usize;
    let n = 2048usize;

    // ---- parity (small shape, K%32==0) ----
    {
        let (m, k, b) = (256usize, 512usize, 256usize);
        let mut w = gpu.upload_raw(&bf16_bytes(3, m * k), &[m, k]).unwrap();
        w.dtype = DType::BF16;
        let x = gpu.upload_f32(&f32v(5, b * k), &[b, k]).unwrap();
        let y_ref = gpu.alloc_tensor(&[b * m], DType::F32).unwrap();
        gpu.gemm_bf16_tiled_wmma(&w, &x, &y_ref, m, k, b, 4, 4).unwrap();
        gpu.device_synchronize().unwrap();
        let yr = gpu.download_f32(&y_ref).unwrap();
        let mut all = true;
        for (entry, mb, nb) in VARIANTS {
            let y = gpu.alloc_tensor(&[b * m], DType::F32).unwrap();
            gpu.gemm_bf16_tiled_wmma_pf(entry, &w, &x, &y, m, k, b, mb, nb).unwrap();
            gpu.device_synchronize().unwrap();
            let yv = gpu.download_f32(&y).unwrap();
            let mut mx = 0.0f32;
            for i in 0..b * m {
                mx = mx.max((yv[i] - yr[i]).abs());
            }
            let ok = mx == 0.0;
            all &= ok;
            println!("parity {entry:<24} max_abs={mx:.6} [{}]", if ok { "BIT-EXACT" } else { "DIFF!" });
            gpu.free_tensor(y).ok();
        }
        gpu.free_tensor(y_ref).ok();
        if !all {
            println!("PARITY FAILED — not benching");
            std::process::exit(1);
        }
        println!();
    }

    // ---- throughput at the DiT attn shape ----
    let (m, k) = (6144usize, 6144usize);
    let mut w = gpu.upload_raw(&bf16_bytes(7, m * k), &[m, k]).unwrap();
    w.dtype = DType::BF16;
    let x = gpu.upload_f32(&f32v(9, n * k), &[n, k]).unwrap();
    let y = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();

    macro_rules! med {
        ($call:expr) => {{
            for _ in 0..3 {
                $call;
            }
            gpu.device_synchronize().unwrap();
            let mut ms = Vec::new();
            for _ in 0..iters {
                let t = Instant::now();
                $call;
                gpu.device_synchronize().unwrap();
                ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ms[ms.len() / 2]
        }};
    }

    // clock/thermal warmup before any timing
    for _ in 0..30 {
        gpu.gemm_bf16_tiled_wmma(&w, &x, &y, m, k, n, 4, 4).unwrap();
    }
    gpu.device_synchronize().unwrap();

    println!("bf16 DiT GEMM prefetch sweep  arch={}  M={m} K={k} N={n}  peak=16.6 TFLOP/s", gpu.arch);
    let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
    let base = med!(gpu.gemm_bf16_tiled_wmma(&w, &x, &y, m, k, n, 4, 4).unwrap());
    println!(
        "{:<26} {:>9.3} ms  {:>7.1}%pk   (baseline)",
        "tiled_4x4 (production)",
        base,
        gflop / base / 16.6 * 100.0
    );
    for (entry, mb, nb) in VARIANTS {
        let t = med!(gpu
            .gemm_bf16_tiled_wmma_pf(entry, &w, &x, &y, m, k, n, mb, nb)
            .unwrap());
        println!(
            "{:<26} {:>9.3} ms  {:>7.1}%pk   {:.2}x",
            entry.trim_start_matches("gemm_bf16_"),
            t,
            gflop / t / 16.6 * 100.0,
            base / t
        );
    }
    let ctrl = med!(gpu.gemm_bf16_tiled_wmma(&w, &x, &y, m, k, n, 4, 4).unwrap());
    println!(
        "\nCONTROL baseline re-measured LAST: {ctrl:.3} ms (first {base:.3}, {:+.1}%) -> {}",
        (ctrl / base - 1.0) * 100.0,
        if (ctrl / base - 1.0).abs() < 0.05 {
            "order-independent, speedups are REAL"
        } else {
            "ORDER-DEPENDENT — compare against this control"
        }
    );
    gpu.free_tensor(y).ok();

    // ---- the winning variant across ALL DiT shapes (a win on one shape is not
    // enough: the LDS kernel won on 3 shapes and regressed GQA) ----
    println!("\n--- pf_4x4_x across all Krea-2 DiT shapes (N={n}) ---");
    println!("{:<30} {:>10} {:>10} {:>8}", "shape", "tiled ms", "pf_x ms", "speedup");
    for (label, sm, sk) in [
        ("attn q/o/gate M6144 K6144", 6144usize, 6144usize),
        ("attn kv (GQA)  M1536 K6144", 1536, 6144),
        ("ffn gate/up    M16384 K6144", 16384, 6144),
        ("ffn down       M6144 K16384", 6144, 16384),
    ] {
        let mut sw = gpu.upload_raw(&bf16_bytes(7, sm * sk), &[sm, sk]).unwrap();
        sw.dtype = DType::BF16;
        let sx = gpu.upload_f32(&f32v(9, n * sk), &[n, sk]).unwrap();
        let sy = gpu.alloc_tensor(&[n * sm], DType::F32).unwrap();
        let tb = med!(gpu.gemm_bf16_tiled_wmma(&sw, &sx, &sy, sm, sk, n, 4, 4).unwrap());
        let tp = med!(gpu
            .gemm_bf16_tiled_wmma_pf("gemm_bf16_pf_4x4_x", &sw, &sx, &sy, sm, sk, n, 4, 4)
            .unwrap());
        println!("{label:<30} {tb:>10.3} {tp:>10.3} {:>7.2}x", tb / tp);
        gpu.free_tensor(sy).ok();
    }
}
