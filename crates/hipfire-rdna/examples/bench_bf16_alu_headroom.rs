// SPDX-License-Identifier: Apache-2.0
//! Free-ALU headroom of the **production** register-tiled 4x4 bf16 DiT GEMM
//! (`gemm_bf16_tiled_wmma_4x4`) — the companion to the LDS-kernel sweep in
//! `bench_dit_bf16_gemm`. Same question: how much unrelated VALU work does the
//! compute-bound DiT GEMM absorb for free? That budget is what a codebook /
//! trellis weight decode (QTIP, LO-BCQ) or a correction branch must fit inside.
//!
//! The two kernels differ in how many weight elements a lane materializes per
//! WMMA — 4x4 tiled: TILE_MB*16/(TILE_MB*TILE_NB) = 4; LDS 2x2: 8 — so the same
//! free-FMA budget buys 2x more decode ops/element on the tiled kernel. Reported
//! directly below.

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

fn main() {
    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP: no wmma");
        return;
    }
    let iters = 10usize;
    let n = 2048usize;
    let (m, k) = (6144usize, 6144usize); // DiT attn
    let mut w = gpu.upload_raw(&bf16_bytes(7, m * k), &[m, k]).unwrap();
    w.dtype = DType::BF16;
    let x = gpu.upload_f32(&f32v(9, n * k), &[n, k]).unwrap();
    let y = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();

    // Global clock/thermal warmup BEFORE any timing: the first-measured config is
    // otherwise inflated by the GPU ramping up, which would fake a "more work is
    // faster" result. 30 launches of the zero-ALU kernel.
    for _ in 0..30 {
        gpu.bench_bf16_alu_headroom("gemm_bf16_alu0", &w, &x, &y, m, k, n).unwrap();
    }
    gpu.device_synchronize().unwrap();

    println!("free-ALU headroom, PRODUCTION 4x4 tiled bf16 GEMM  arch={}  M={m} K={k} N={n}", gpu.arch);
    println!(
        "{:>10} {:>12} {:>10} {:>9}  {:>16}",
        "FMA/K-step", "FMA/WMMA", "ms", "vs 0", "free ops/weight-el"
    );
    // NALU = side FMAs per K-step; the 4x4 kernel issues 16 WMMAs per K-step and
    // a lane materializes 4 weight elements per WMMA (64 per K-step).
    let mut base = 0.0f64;
    for (entry, nalu) in [
        ("gemm_bf16_alu0", 0usize),
        ("gemm_bf16_alu64", 64),
        ("gemm_bf16_alu128", 128),
        ("gemm_bf16_alu256", 256),
        ("gemm_bf16_alu512", 512),
        ("gemm_bf16_alu1024", 1024),
        ("gemm_bf16_alu2048", 2048),
    ] {
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
        let t = med!(gpu
            .bench_bf16_alu_headroom(entry, &w, &x, &y, m, k, n)
            .unwrap());
        if nalu == 0 {
            base = t;
        }
        println!(
            "{nalu:>10} {:>12} {t:>10.3} {:>8.2}x  {:>16.2}",
            nalu / 16,
            t / base,
            nalu as f64 / 64.0
        );
    }

    // CONTROL: re-measure the zero-ALU kernel LAST. If it now matches its
    // first-measured time, the sweep is order-independent and any speedup from
    // added ALU is real; if it drops to the level of the ALU variants, the
    // "faster with more work" reading was a clock-ramp artifact.
    let t_ctrl = {
        for _ in 0..3 {
            gpu.bench_bf16_alu_headroom("gemm_bf16_alu0", &w, &x, &y, m, k, n).unwrap();
        }
        gpu.device_synchronize().unwrap();
        let mut ms = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            gpu.bench_bf16_alu_headroom("gemm_bf16_alu0", &w, &x, &y, m, k, n).unwrap();
            gpu.device_synchronize().unwrap();
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ms[ms.len() / 2]
    };
    println!(
        "\nCONTROL alu0 re-measured LAST: {t_ctrl:.3} ms (first: {base:.3} ms, {:+.1}%)",
        (t_ctrl / base - 1.0) * 100.0
    );
    println!(
        "  -> {}",
        if (t_ctrl / base - 1.0).abs() < 0.05 {
            "order-independent: an ALU speedup would be REAL"
        } else {
            "ORDER-DEPENDENT: the first-measured config was inflated (clock ramp) — compare vs this control"
        }
    );
}
