// SPDX-License-Identifier: Apache-2.0
//! Stream D1 throughput gate — the Krea-2 DiT bf16 GEMM. Times the current
//! zero-LDS register-tiled `gemm_bf16_tiled_wmma_4x4` (8.4% of peak) vs the new
//! LDS-staged `gemm_bf16_tiled_wmma_lds`, at real DiT shapes, and reports TFLOP/s
//! + % of the ~16.6 TFLOP/s bf16-WMMA peak. Both dispatches stage the f32
//! activation to bf16 (ensure_bf16_x), so the head-to-head includes staging —
//! matching the HIPFIRE_PROFILE GEMM_NS number.

use hipfire_rdna::{DType, Gpu};
use std::time::Instant;

const PEAK_TFLOPS: f64 = 16.6;

fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn bf16_bytes(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .flat_map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            let v = ((s as f32 / 2_147_483_648.0) - 0.5) * 0.2;
            f32_to_bf16_bits(v).to_le_bytes()
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

fn main() {
    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP: no wmma");
        return;
    }
    let iters = 20usize;
    let n = 2048usize; // DiT tokens (≈ 1024²/... representative compute-bound regime)
    // (label, M=out_features, K=in_features) — Krea-2 DiT (hidden 6144, FFN 16384).
    let shapes = [
        ("attn q/o/gate M6144 K6144", 6144usize, 6144usize),
        ("attn kv (GQA)  M1536 K6144", 1536usize, 6144usize),
        ("ffn gate/up    M16384 K6144", 16384usize, 6144usize),
        ("ffn down       M6144 K16384", 6144usize, 16384usize),
    ];
    println!("DiT bf16 GEMM  arch={}  N={n}  iters={iters}  peak={PEAK_TFLOPS} TFLOP/s", gpu.arch);
    println!("{:<30} {:>10} {:>10}   {:>10} {:>10}   speedup", "shape", "tiled ms", "tiled %pk", "lds ms", "lds %pk");

    for (label, m, k) in shapes {
        let mut w = gpu.upload_raw(&bf16_bytes(7, m * k), &[m, k]).unwrap();
        w.dtype = DType::BF16;
        let x = gpu.upload_f32(&f32v(9, n * k), &[n, k]).unwrap();
        let y = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();

        macro_rules! med_ms {
            ($call:expr) => {{
                for _ in 0..3 {
                    $call;
                }
                gpu.device_synchronize().unwrap();
                let mut ms = Vec::with_capacity(iters);
                for _ in 0..iters {
                    let t = Instant::now();
                    $call;
                    gpu.device_synchronize().unwrap();
                    ms.push(t.elapsed().as_secs_f64() * 1e3);
                }
                ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                ms[iters / 2]
            }};
        }

        let t_tiled = med_ms!(gpu.gemm_bf16_tiled_wmma(&w, &x, &y, m, k, n, 4, 4).unwrap());
        let t_lds = med_ms!(gpu.gemm_bf16_tiled_wmma_lds(&w, &x, &y, m, k, n).unwrap());
        let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
        let tf = |ms: f64| gflop / ms; // GFLOP / ms = TFLOP/s
        let pk = |ms: f64| tf(ms) / PEAK_TFLOPS * 100.0;
        println!(
            "{label:<30} {t_tiled:>10.3} {:>9.1}%   {t_lds:>10.3} {:>9.1}%   {:.2}x",
            pk(t_tiled),
            pk(t_lds),
            t_tiled / t_lds
        );
        gpu.free_tensor(y).ok();
    }

    // Free-ALU headroom sweep (attn 6144×6144): add `extra` throwaway VALU FMAs
    // per WMMA; the knee where time rises = the free-compute budget a QTIP/codebook
    // decode or correction branch hides in for ~zero warm-step cost.
    println!("\n--- free-ALU headroom (attn M6144 K6144, N={n}) ---");
    println!("{:>8}  {:>10}  {:>10}  {:>12}", "extra", "ms", "vs 0", "extra-FMA/WMMA");
    let (m, k) = (6144usize, 6144usize);
    let mut w = gpu.upload_raw(&bf16_bytes(7, m * k), &[m, k]).unwrap();
    w.dtype = DType::BF16;
    let x = gpu.upload_f32(&f32v(9, n * k), &[n, k]).unwrap();
    let y = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
    let mut base = 0.0f64;
    for &extra in &[0usize, 8, 16, 32, 48, 64, 96, 128, 192, 256] {
        macro_rules! med {
            ($call:expr) => {{
                for _ in 0..3 { $call; }
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
        let t = med!(gpu.bench_bf16_lds_freealu(&w, &x, &y, m, k, n, extra).unwrap());
        if extra == 0 { base = t; }
        println!("{extra:>8}  {t:>10.3}  {:>9.2}x  {extra:>12}", t / base);
    }
    gpu.free_tensor(y).ok();
}
