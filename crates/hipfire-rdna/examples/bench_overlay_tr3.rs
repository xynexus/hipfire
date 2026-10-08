// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Time the compact overlay correction at Qwen3.8-27B projection shapes, B=512:
//! the n_ov=3 kernel (`_tr3`) against the generic `_tr` (HIPFIRE_OQ_OVERLAY_TR3=0).
//!
//!   cargo run --release -p hipfire-rdna --example bench_overlay_tr3 [B]
use hipfire_rdna::{Gpu, OQ_OVERLAY_SLACK};
use std::time::Instant;

fn main() {
    let b: usize = std::env::args().nth(1).map_or(512, |v| v.parse().unwrap());
    let mut gpu = Gpu::init().unwrap();
    let mut s = 0x2468_ace1u32;
    let mut rnd = move || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        s >> 8
    };
    println!("  proj        M      K    B   tr_ms  tr3_ms  speedup");
    for &(name, m, k) in &[
        ("gate/up", 34816usize, 5120usize),
        ("down", 5120, 17408),
        ("qkv", 12288, 5120),
        ("o", 5120, 6144),
    ] {
        let ng = k / 256;
        let block_stride = 136usize;
        let mut w = vec![0u8; m * ng * block_stride];
        let side_base = m * ng * 128;
        for i in 0..m * ng {
            let base = side_base + i * 8;
            w[base..base + 2].copy_from_slice(&0x2c00u16.to_le_bytes()); // 0.0625
            for e in 0..3 {
                w[base + 2 + 2 * e] = (rnd() % 256) as u8;
                w[base + 3 + 2 * e] = (rnd() % 255) as u8;
            }
        }
        let xt: Vec<u8> = (0..k * b + OQ_OVERLAY_SLACK).map(|_| rnd() as u8).collect();
        let xs: Vec<u8> = (0..(ng * b + OQ_OVERLAY_SLACK))
            .flat_map(|_| 0.01f32.to_le_bytes())
            .collect();
        let wd = gpu.upload_raw(&w, &[w.len()]).unwrap();
        let xtd = gpu.upload_raw(&xt, &[xt.len()]).unwrap();
        let xsd = gpu.upload_raw(&xs, &[ng * b + OQ_OVERLAY_SLACK]).unwrap();
        let yd = gpu.upload_raw(&vec![0u8; b * m * 4], &[b * m]).unwrap();
        let time = |gpu: &mut Gpu, tr3: bool| {
            std::env::set_var("HIPFIRE_OQ_OVERLAY_TR3", if tr3 { "1" } else { "0" });
            for _ in 0..3 {
                gpu.oq_compact_overlay_correct_t(&wd, &xtd, &xsd, &yd, m, k, b, 256, block_stride)
                    .unwrap();
            }
            gpu.device_synchronize().unwrap();
            let iters = 20;
            let t = Instant::now();
            for _ in 0..iters {
                gpu.oq_compact_overlay_correct_t(&wd, &xtd, &xsd, &yd, m, k, b, 256, block_stride)
                    .unwrap();
            }
            gpu.device_synchronize().unwrap();
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        };
        let tr = time(&mut gpu, false);
        let tr3 = time(&mut gpu, true);
        println!(
            "  {name:8} {m:6} {k:6} {b:4} {tr:7.3} {tr3:7.3}  {:5.2}x",
            tr / tr3
        );
    }
}
