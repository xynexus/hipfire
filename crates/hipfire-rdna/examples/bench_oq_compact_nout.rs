// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! What the sparse overlay costs the compact DECODE GEMV, as a function of N_out.
//!
//! The standing claim is "1 -> 8 corrections cost ~4x", which was measured on the
//! W8A8 variant and never re-checked on the A16 kernel decode actually runs. This
//! sweeps N_out on the live dispatch path and reports ACHIEVED BANDWIDTH.
//!
//! GB/s is the right lens, not milliseconds: raising N_out also raises the block
//! stride (130 + 2*N_out), so a purely bandwidth-bound kernel gets SLOWER in ms
//! while holding GB/s FLAT. Flat GB/s means the overlay rides along free and the
//! only fix is fewer bytes; falling GB/s means the overlay is costing time the
//! memory system was not asking for, and the kernel is what needs fixing.
//!
//!   cargo run --release -p hipfire-rdna --example bench_oq_compact_nout

use hipfire_rdna::{DType, Gpu};

fn lcg(seed: u32) -> impl FnMut() -> u32 {
    let mut s = seed.max(1);
    move || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        s
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    const GROUP: usize = 256;
    // Real Qwen3.8-27B OqPlusCompact projection classes, full M — this is a
    // bandwidth measurement, so the sizes have to be the ones decode really reads.
    let shapes: &[(&str, usize, usize)] = &[
        ("down     [5120, 17408]", 5120, 17408),
        ("gate/up  [17408, 5120]", 17408, 5120),
        ("lm_head [248320, 5120]", 248320, 5120),
    ];
    let iters = 30usize;
    // Profiling wants ONE configuration per process, so a counter run can be
    // attributed without matching dispatch ordering. Unset = sweep everything.
    let only_shape = std::env::var("BENCH_SHAPE").ok();
    let only_nout = std::env::var("BENCH_NOUT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());

    println!("compact decode GEMV — overlay cost vs N_out (group={GROUP})");
    println!("bytes = M * (K/256) * (130 + 2*N_out); GB/s flat => overlay is free\n");

    for &(name, m, k) in shapes {
        if let Some(f) = &only_shape {
            if !name.starts_with(f.as_str()) {
                continue;
            }
        }
        let ng = k / GROUP;
        println!("{name}");
        println!("  N_out  stride     MiB     ms      GB/s   vs N_out=1");
        let mut base_bw = 0f64;
        for n_out in [1usize, 2, 3, 4, 6, 8] {
            if let Some(f) = only_nout {
                if n_out != f {
                    continue;
                }
            }
            let block_stride = 2 + GROUP / 2 + 2 * n_out;
            let bytes = m * ng * block_stride;
            let mut rnd = lcg(0x00c0_de01 ^ (k as u32) ^ ((n_out as u32) << 16));
            let mut blocks = vec![0u8; bytes];
            for r in 0..m {
                for g in 0..ng {
                    let off = (r * ng + g) * block_stride;
                    let bits = (((10 + rnd() % 9) as u16) << 10) | (rnd() % 1024) as u16;
                    blocks[off..off + 2].copy_from_slice(&bits.to_le_bytes());
                    for i in 0..GROUP / 2 {
                        blocks[off + 2 + i] = (rnd() & 0xff) as u8;
                    }
                    let hdr = 2 + GROUP / 2;
                    // Distinct indices: duplicates would be normalized away at load
                    // anyway, and the kernel is allowed to assume that.
                    let mut used = [false; GROUP];
                    for s in 0..n_out {
                        let mut idx = (rnd() % GROUP as u32) as usize;
                        while used[idx] {
                            idx = (idx + 1) % GROUP;
                        }
                        used[idx] = true;
                        blocks[off + hdr + 2 * s] = idx as u8;
                        blocks[off + hdr + 2 * s + 1] = (rnd() & 0xff) as u8;
                    }
                }
            }
            let x: Vec<f32> = (0..k).map(|_| (rnd() % 2000) as f32 * 1e-3 - 1.0).collect();
            let wb = gpu.upload_raw(&blocks, &[blocks.len()]).expect("w");
            let xb = gpu.upload_f32(&x, &[k]).expect("x");
            let yb = gpu.alloc_tensor(&[m], DType::F32).expect("y");

            for _ in 0..5 {
                gpu.gemv_oq_compact_grouped_auto(&wb, &xb, &yb, m, k, GROUP, block_stride)
                    .expect("warm");
            }
            gpu.device_synchronize().expect("sync");
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                gpu.gemv_oq_compact_grouped_auto(&wb, &xb, &yb, m, k, GROUP, block_stride)
                    .expect("run");
            }
            gpu.device_synchronize().expect("sync");
            let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
            let bw = bytes as f64 / (ms * 1e-3) / 1e9;
            if n_out == 1 {
                base_bw = bw;
            }
            println!(
                "  {:>4}   {:>4}   {:>7.1}  {:>6.3}  {:>7.1}    {:>5.1}%",
                n_out,
                block_stride,
                bytes as f64 / (1024.0 * 1024.0),
                ms,
                bw,
                100.0 * bw / base_bw
            );
        }
        println!();
    }
}
