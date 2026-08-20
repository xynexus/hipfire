// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! What does the Opus W8A8 batched GEMM cost at the batch sizes speculative
//! verify actually uses?
//!
//! A rocprofv3 trace of a DFlash2 run on Qwen3.8-27B put `gemm_oq8_grouped_wmma`
//! at 69 % of GPU time and 1881 us/call, against `gemv_oq8_grouped_v2` moving the
//! same weight bytes in 413 us — 47 GB/s versus 215 GB/s. Verify batches 9-17
//! positions, so this sweeps that range on the real layer shapes and reports
//! effective weight bandwidth, which is the number that matters: both kernels
//! read the weight matrix exactly once, so anything below the GEMV's figure is
//! the batched kernel leaving bandwidth on the floor.
//!
//! **Cache warning.** gfx1151 is LPDDR5X-8000 on a 256-bit bus (~256 GB/s
//! theoretical) behind a 32 MB MALL. Any shape whose weights fit in 32 MB, timed
//! in a loop over ONE buffer, reports cache bandwidth, not DRAM — and any figure
//! above ~256 GB/s is definitionally a cache hit. `--cold` round-robins over
//! enough distinct weight buffers to evict the MALL between touches, which is the
//! number that describes a real forward pass (every layer's weights are cold).
//! Run both: the gap between them IS the cache contribution.
//!
//! Reports GB/s over WEIGHT bytes only (M*K int8 + M*K/group f32 scales). The
//! activations are a rounding error at these batch sizes and are re-read from
//! cache, so counting them would flatter the kernel.
//!
//!   cargo run --release -p hipfire-rdna --example bench_oq8_gemm_small_n

use hipfire_rdna::{DType, Gpu};

const GROUP: usize = 256;

fn lcg_i8(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            ((s >> 7) as i8 as u8) ^ 0x11
        })
        .collect()
}

fn lcg_f32(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            0.002 + (s as f32 / 2_147_483_648.0) * 0.01
        })
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    // The three dense-27B projection shapes, by weight footprint.
    let sweep = std::env::args().any(|a| a == "--sweep-m");
    let owned: Vec<(String, usize, usize)> = if sweep {
        // Fixed K, rising M: isolates whether the collapse tracks the output
        // row count or the shape's K/stride.
        [5120usize, 8192, 10240, 12288, 16384, 17408, 20480]
            .iter()
            .map(|&m| (format!("M={m:<6} K=5120"), m, 5120usize))
            .chain(
                // Same M, K a clean power of two: separates M from the 5120-byte
                // row stride.
                [4096usize, 8192]
                    .iter()
                    .map(|&k| (format!("M=17408  K={k}"), 17408, k)),
            )
            .collect()
    } else {
        vec![
            ("gate/up  [17408, 5120]".to_string(), 17408, 5120),
            ("down     [5120, 17408]".to_string(), 5120, 17408),
            ("o_proj   [5120,  5120]".to_string(), 5120, 5120),
        ]
    };
    let shapes: Vec<(&str, usize, usize)> =
        owned.iter().map(|(n, m, k)| (n.as_str(), *m, *k)).collect();
    let shapes = &shapes[..];
    let batches: &[usize] = if sweep { &[9] } else { &[1, 8, 9, 16, 17, 32] };
    const ITERS: usize = 20;

    if std::env::args().any(|a| a == "--split-m") {
        // Does the collapse follow the LAUNCH's M, or the buffer? Same weights,
        // same total work, issued as `chunks` back-to-back calls over row slabs.
        // If chunking recovers the small-M bandwidth, the fix is a dispatch-level
        // loop and no kernel change at all.
        let (m, k, b) = (17408usize, 5120usize, 9usize);
        let ng = k / GROUP;
        let w_bytes = m * k + m * ng * 4;
        let w = gpu.upload_raw(&lcg_i8(1, m * k), &[m * k]).expect("w");
        let ws = gpu.upload_f32(&lcg_f32(2, m * ng), &[m * ng]).expect("ws");
        let x = gpu.upload_raw(&lcg_i8(3, b * k), &[b * k]).expect("x");
        let xs = gpu.upload_f32(&lcg_f32(4, b * ng), &[b * ng]).expect("xs");
        let y = gpu.alloc_tensor(&[b * m], DType::F32).expect("y");
        println!("split-M: [17408, 5120] B=9, one call vs N row-slab calls\n");
        for &rows in &[17408usize, 8704, 4352, 2176, 1088, 544] {
            let n_chunks = m / rows;
            let run = |g: &mut Gpu| {
                for c in 0..n_chunks {
                    let wc = w.sub_offset(c * rows * k, rows * k);
                    let wsc = ws.sub_offset(c * rows * ng, rows * ng);
                    // Y is [B, M] so a row slab is a strided view per column;
                    // write into a slab-sized scratch instead and ignore layout
                    // here — this measures the READ side, which is the question.
                    let yc = y.sub_offset(0, b * rows);
                    g.gemm_oq8_grouped_wmma(&wc, &wsc, &x, &xs, &yc, rows, k, b, GROUP)
                        .expect("gemm");
                }
            };
            for _ in 0..3 {
                run(&mut gpu);
            }
            gpu.hip.device_synchronize().expect("sync");
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                run(&mut gpu);
            }
            gpu.hip.device_synchronize().expect("sync");
            let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
            println!(
                "   rows/call={rows:<6} calls={n_chunks:<3} {us:>8.1} us   {:>6.1} GB/s",
                w_bytes as f64 / us / 1000.0
            );
        }
        return;
    }

    // gfx1151: 32 MB MALL. Enough distinct copies to blow through it twice over.
    const MALL_BYTES: usize = 32 * 1024 * 1024;
    let cold = std::env::args().any(|a| a == "--cold");
    if cold {
        println!("Opus W8A8 batched GEMM — COLD weights (round-robin, MALL evicted)");
        println!("  gfx1151 DRAM peak ~256 GB/s; anything near/above that is cache.\n");
        for &(name, m, k) in shapes {
            let ng = k / GROUP;
            let w_bytes = m * k + m * ng * 4;
            let copies = (2 * MALL_BYTES).div_ceil(w_bytes).max(2);
            let ws_t = gpu.upload_f32(&lcg_f32(2, m * ng), &[m * ng]).expect("ws");
            let wcopies: Vec<_> = (0..copies)
                .map(|i| {
                    gpu.upload_raw(&lcg_i8(10 + i as u32, m * k), &[m * k])
                        .expect("w")
                })
                .collect();
            let b = 9usize;
            let x = gpu.upload_raw(&lcg_i8(3, b * k), &[b * k]).expect("x");
            let xs = gpu.upload_f32(&lcg_f32(4, b * ng), &[b * ng]).expect("xs");
            let y = gpu.alloc_tensor(&[b * m], DType::F32).expect("y");
            for i in 0..copies {
                gpu.gemm_oq8_grouped_wmma(&wcopies[i], &ws_t, &x, &xs, &y, m, k, b, GROUP)
                    .expect("gemm");
            }
            gpu.hip.device_synchronize().expect("sync");
            // Several passes over the round-robin: with only `copies` (2-3) timed
            // calls the number is noise, which is how the first cold run made a
            // 12 % swing look like a slab regression.
            const PASSES: usize = 8;
            let mut best = f64::MAX;
            for _ in 0..PASSES {
                let t0 = std::time::Instant::now();
                for i in 0..copies {
                    gpu.gemm_oq8_grouped_wmma(&wcopies[i], &ws_t, &x, &xs, &y, m, k, b, GROUP)
                        .expect("gemm");
                }
                gpu.hip.device_synchronize().expect("sync");
                let per = t0.elapsed().as_secs_f64() * 1e6 / copies as f64;
                best = best.min(per);
            }
            let us = best;
            println!(
                "{name}  B=9  {us:>8.1} us   {:>6.1} GB/s   ({copies} distinct copies, \
                 {:.0} MiB working set)",
                w_bytes as f64 / us / 1000.0,
                (copies * w_bytes) as f64 / (1024.0 * 1024.0)
            );
            let _ = (
                gpu.free_tensor(x),
                gpu.free_tensor(xs),
                gpu.free_tensor(y),
                gpu.free_tensor(ws_t),
            );
            for w in wcopies {
                let _ = gpu.free_tensor(w);
            }
        }
        return;
    }

    println!("Opus W8A8 batched GEMM, weight-bandwidth view (gfx1151)");
    println!("  GB/s counts WEIGHT bytes only; every batch size reads the same matrix once.\n");

    for &(name, m, k) in shapes {
        let ng = k / GROUP;
        let w_bytes = m * k + m * ng * 4;
        let w = gpu.upload_raw(&lcg_i8(1, m * k), &[m * k]).expect("w");
        let ws = gpu.upload_f32(&lcg_f32(2, m * ng), &[m * ng]).expect("ws");
        println!(
            "{name}   weights {:.1} MiB",
            w_bytes as f64 / (1024.0 * 1024.0)
        );
        let mut base_us = 0f64;
        for &b in batches {
            let x = gpu.upload_raw(&lcg_i8(3, b * k), &[b * k]).expect("x");
            let xs = gpu.upload_f32(&lcg_f32(4, b * ng), &[b * ng]).expect("xs");
            let y = gpu.alloc_tensor(&[b * m], DType::F32).expect("y");

            // Warm up: first call JITs, and the L2 state should be the same for
            // every timed iteration.
            for _ in 0..3 {
                gpu.gemm_oq8_grouped_wmma(&w, &ws, &x, &xs, &y, m, k, b, GROUP)
                    .expect("gemm");
            }
            gpu.hip.device_synchronize().expect("sync");
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                gpu.gemm_oq8_grouped_wmma(&w, &ws, &x, &xs, &y, m, k, b, GROUP)
                    .expect("gemm");
            }
            gpu.hip.device_synchronize().expect("sync");
            let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
            if b == 1 {
                base_us = us;
            }
            let gbs = w_bytes as f64 / us / 1000.0;
            let per_row = us / b as f64;
            println!(
                "   B={b:<3} {us:>8.1} us   {gbs:>6.1} GB/s   {per_row:>7.1} us/row   \
                 {:.2}x the B=1 call",
                us / base_us
            );
            let _ = (gpu.free_tensor(x), gpu.free_tensor(xs), gpu.free_tensor(y));
        }
        let _ = (gpu.free_tensor(w), gpu.free_tensor(ws));
        println!();
    }
}
