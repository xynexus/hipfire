// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Qwen3.8-27B projection shapes: compact w64 GEMM + separate n_ov=3 overlay pass
//! against the GEMM with the overlay summed in its epilogue (`_epi`).
//!
//!   cargo run --release -p hipfire-rdna --example bench_overlay_epi [B]
use hipfire_rdna::{DType, Gpu, OQ_OVERLAY_SLACK};
use std::time::Instant;

fn main() {
    let b: usize = std::env::args().nth(1).map_or(512, |v| v.parse().unwrap());
    let mut gpu = Gpu::init().unwrap();
    let mut s = 0x1234_5679u32;
    let mut rnd = move || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        s >> 8
    };
    println!("  proj        M      K    B  gemm+pass_ms  epi_ms  speedup");
    for &(name, m, k) in &[
        ("gate/up", 34816usize, 5120usize),
        ("down", 5120, 17408),
        ("qkv", 12288, 5120),
        ("o", 5120, 6144),
    ] {
        let ng = k / 256;
        let mut w: Vec<u8> = (0..m * ng * 136).map(|_| rnd() as u8).collect();
        for i in 0..m * ng {
            let o = m * ng * 128 + i * 8;
            w[o..o + 2].copy_from_slice(&0x2c00u16.to_le_bytes());
        }
        let x: Vec<u8> = (0..b * k).map(|_| rnd() as u8).collect();
        let xt: Vec<u8> = (0..k * b + OQ_OVERLAY_SLACK).map(|_| rnd() as u8).collect();
        let wd = gpu.upload_raw(&w, &[w.len()]).unwrap();
        let xd = gpu.upload_raw(&x, &[x.len()]).unwrap();
        let xsd = gpu.upload_f32(&vec![0.01; b * ng], &[b * ng]).unwrap();
        let xtd = gpu.upload_raw(&xt, &[xt.len()]).unwrap();
        let xstd = gpu
            .upload_f32(
                &vec![0.01; ng * b + OQ_OVERLAY_SLACK],
                &[ng * b + OQ_OVERLAY_SLACK],
            )
            .unwrap();
        let yd = gpu.alloc_tensor(&[b * m], DType::F32).unwrap();
        let time = |gpu: &mut Gpu, epi: bool| {
            let run = |gpu: &mut Gpu| {
                if epi {
                    gpu.gemm_oq_compact_iu4x2_w64_epi(
                        &wd, &xd, &xsd, &yd, m, k, b, 136, &xtd, &xstd,
                    )
                    .unwrap();
                } else {
                    gpu.gemm_oq_compact_iu4x2_w64(&wd, &xd, &xsd, &yd, m, k, b, 136)
                        .unwrap();
                    gpu.oq_compact_overlay_correct_t(&wd, &xtd, &xstd, &yd, m, k, b, 256, 136)
                        .unwrap();
                }
            };
            for _ in 0..3 {
                run(gpu);
            }
            gpu.device_synchronize().unwrap();
            let t = Instant::now();
            for _ in 0..10 {
                run(gpu);
            }
            gpu.device_synchronize().unwrap();
            t.elapsed().as_secs_f64() * 100.0
        };
        let sep = time(&mut gpu, false);
        let epi = time(&mut gpu, true);
        println!(
            "  {name:8} {m:6} {k:6} {b:4} {sep:12.3} {epi:7.3}  {:5.2}x",
            sep / epi
        );
    }
}
