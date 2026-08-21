// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! What would a compact-resident iu4 GEMM be worth? Prices the EXISTING W4A4
//! `gemm_oq4_grouped_wmma` (iu4 x iu4) at prefill shapes as the ceiling proxy
//! for a compact iu4 path, against `gemm_oq_compact_grouped_wmma` (iu8) on the
//! same shapes.
//!
//! Measured on this box: iu4 WMMA is 1.876x iu8 (99.2 vs 52.9 TOPS), and int4
//! activations also HALVE the activation traffic that the compact GEMM is
//! actually bottlenecked on.
//!
//!   cargo run --release -p hipfire-rdna --example bench_w4a4_vs_compact

use hipfire_rdna::{DType, Gpu};
use std::time::Instant;

const GROUP: usize = 256;

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut seed = 0x0BADC0DEu32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        (seed >> 16) as u32
    };
    let iters = 10usize;
    println!("W4A4 (iu4) vs compact (iu8) at prefill shapes");
    println!("iu4 peak 99.2 TOPS, iu8 peak 52.9 TOPS (measured by probe_gfx1151_iu4_wmma)\n");
    println!("  proj             M      K    B      ms     TOPS   % of 99.2");

    for &(name, m, k, b) in &[
        ("gate/up", 17408usize, 5120usize, 256usize),
        ("down", 5120, 17408, 256),
        ("qkv", 6144, 5120, 256),
        ("wo", 5120, 4096, 256),
    ] {
        let ng = k / GROUP;
        let w: Vec<u8> = (0..m * k / 2).map(|_| (rnd() & 0xff) as u8).collect();
        let ws: Vec<f32> = (0..m * ng)
            .map(|_| 0.01 + (rnd() % 97) as f32 * 1e-4)
            .collect();
        let x: Vec<u8> = (0..b * k / 2).map(|_| (rnd() & 0xff) as u8).collect();
        let xs: Vec<f32> = (0..b * ng)
            .map(|_| 0.01 + (rnd() % 89) as f32 * 1e-4)
            .collect();

        let wb = gpu.upload_raw(&w, &[w.len()]).expect("w");
        let wsb = gpu.upload_f32(&ws, &[ws.len()]).expect("ws");
        let xb = gpu.upload_raw(&x, &[x.len()]).expect("x");
        let xsb = gpu.upload_f32(&xs, &[xs.len()]).expect("xs");
        let yb = gpu.alloc_tensor(&[b * m], DType::F32).expect("y");

        gpu.gemm_oq4_grouped_wmma(&wb, &wsb, &xb, &xsb, &yb, m, k, b, GROUP)
            .expect("warm");
        gpu.device_synchronize().expect("sync");
        let t0 = Instant::now();
        for _ in 0..iters {
            gpu.gemm_oq4_grouped_wmma(&wb, &wsb, &xb, &xsb, &yb, m, k, b, GROUP)
                .expect("launch");
        }
        gpu.device_synchronize().expect("sync");
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let tops = 2.0 * (m as f64) * (k as f64) * (b as f64) / (ms * 1e-3) / 1e12;
        println!(
            "  {name:<12} {m:>6} {k:>6} {b:>4} {ms:>7.3} {tops:>8.2} {:>9.1}%",
            100.0 * tops / 99.2
        );
        let _ = gpu.free_tensor(wb);
        let _ = gpu.free_tensor(wsb);
        let _ = gpu.free_tensor(xb);
        let _ = gpu.free_tensor(xsb);
        let _ = gpu.free_tensor(yb);
    }
}
