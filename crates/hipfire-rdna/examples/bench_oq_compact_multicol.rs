// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Microbenchmark for `gemv_oq_compact_multicol` at REAL Qwen3.8-27B shapes.
//!
//! This kernel is 80.7% of a spec-decode profile and sustains ~35 GB/s against a
//! 233 GB/s ceiling, so it needs an isolated bench: end-to-end tok/s cannot price
//! it, because any change that breaks its numerics also collapses the drafter's
//! acceptance and the throughput move is then acceptance, not kernel speed.
//!
//!   cargo run --release -p hipfire-rdna --example bench_oq_compact_multicol

use hipfire_rdna::{DType, Gpu};
use std::time::Instant;

const GROUP: usize = 256;

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut seed = 0x1357_9BDFu32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        (seed >> 16) as u32
    };
    // BENCH_NOUT=0 removes the sparse overlay entirely, isolating its scattered
    // per-(row,column) LDS gather from the dense dot4 path.
    let n_out = std::env::var("BENCH_NOUT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3usize);
    let stride = 2 + GROUP / 2 + 2 * n_out; // 136 at n_out=3 => 4.25 bits
    let iters = 30usize;

    println!("gemv_oq_compact_multicol at Qwen3.8-27B shapes (n_out={n_out}, stride={stride})");
    println!("weight bytes only; 233 GB/s is the measured achievable ceiling\n");
    println!("  proj             M      K   B    MiB      ms      GB/s   % of 233");

    for &(name, m, k, b) in &[
        ("gate/up", 17408usize, 5120usize, 8usize),
        ("down", 5120, 17408, 8),
        ("qkv", 6144, 5120, 8),
        ("wo", 5120, 4096, 8),
        ("gate/up B=1", 17408, 5120, 1),
        ("gate/up B=16", 17408, 5120, 16),
        // Past 16 the entry macro drops RW (narrow 4/2/1, wide 3/1) because the
        // N*RW accumulators stop fitting. Tree verify linearizes a budget-B tree
        // to B+1 tokens, so these are the widths a DDTree actually asks for, and
        // the cliff between 16 and 17 is what decides whether wide trees pay.
        ("gate/up B=12", 17408, 5120, 12),
        ("gate/up B=17", 17408, 5120, 17),
        ("gate/up B=24", 17408, 5120, 24),
        ("gate/up B=32", 17408, 5120, 32),
    ] {
        let ng = k / GROUP;
        let nblk = m * ng;
        let bytes = nblk * stride;
        let mut blocks = vec![0u8; bytes];
        for blk in 0..nblk {
            let off = blk * stride;
            let bits = (((14 + rnd() % 3) as u16) << 10) | (rnd() % 1024) as u16;
            blocks[off..off + 2].copy_from_slice(&bits.to_le_bytes());
            for i in 0..GROUP / 2 {
                blocks[off + 2 + i] = (rnd() & 0xff) as u8;
            }
            let hdr = 2 + GROUP / 2;
            let mut used = [false; GROUP];
            for s in 0..n_out {
                let mut idx = (rnd() % GROUP as u32) as usize;
                while used[idx] {
                    idx = (idx + 1) % GROUP;
                }
                used[idx] = true;
                blocks[off + hdr + 2 * s] = idx as u8;
                blocks[off + hdr + 2 * s + 1] = (rnd() & 0xff) as u8;
                let nb = &mut blocks[off + 2 + idx / 2];
                *nb &= if idx % 2 == 0 { 0xf0 } else { 0x0f };
            }
        }
        // Split planes: all nibble groups first, then all [f16 scale][table].
        let side = stride - GROUP / 2;
        let mut dev = vec![0u8; bytes];
        for blk in 0..nblk {
            let src = blk * stride;
            dev[blk * (GROUP / 2)..blk * (GROUP / 2) + GROUP / 2]
                .copy_from_slice(&blocks[src + 2..src + 2 + GROUP / 2]);
            let d = nblk * (GROUP / 2) + blk * side;
            dev[d..d + 2].copy_from_slice(&blocks[src..src + 2]);
            dev[d + 2..d + side].copy_from_slice(&blocks[src + 2 + GROUP / 2..src + stride]);
        }

        let xq: Vec<i8> = (0..b * k).map(|_| (rnd() % 255) as i8).collect();
        let xs: Vec<f32> = (0..b * ng)
            .map(|_| (rnd() % 1000) as f32 * 1e-5 + 1e-4)
            .collect();
        let wb = gpu.upload_raw(&dev, &[dev.len()]).expect("w");
        let xqb = gpu
            .upload_raw(
                unsafe { std::slice::from_raw_parts(xq.as_ptr() as *const u8, xq.len()) },
                &[xq.len()],
            )
            .expect("xq");
        let xsb = gpu.upload_f32(&xs, &[xs.len()]).expect("xs");
        let yb = gpu.alloc_tensor(&[b * m], DType::F32).expect("y");

        gpu.gemv_oq_compact_multicol(&wb, &xqb, &xsb, &yb, m, k, b, stride)
            .expect("warm");
        gpu.device_synchronize().expect("sync");
        let t0 = Instant::now();
        for _ in 0..iters {
            gpu.gemv_oq_compact_multicol(&wb, &xqb, &xsb, &yb, m, k, b, stride)
                .expect("launch");
        }
        gpu.device_synchronize().expect("sync");
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let gbs = bytes as f64 / (ms * 1e-3) / 1e9;
        println!(
            "  {name:<12} {m:>6} {k:>6} {b:>3} {:>6.1} {ms:>7.3} {gbs:>9.1} {:>9.1}%",
            bytes as f64 / (1024.0 * 1024.0),
            100.0 * gbs / 233.0
        );
        let _ = gpu.free_tensor(wb);
        let _ = gpu.free_tensor(xqb);
        let _ = gpu.free_tensor(xsb);
        let _ = gpu.free_tensor(yb);
    }
}
