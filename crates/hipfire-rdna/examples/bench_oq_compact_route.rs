// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! The SERVING entry point for Opus-compact weights, `gemm_oq_compact_act_batched`
//! (f32 activations in: quantize + route to multicol / IU4 / WMMA), swept over the
//! batch widths a decode step actually has (1..64 sessions, x drafts) at the
//! Qwen3.8-27B shapes. Reports ms, int8 TOPS (vs ~56 peak) and weight GB/s (vs the
//! ~233 GB/s measured ceiling): small B should sit near the bandwidth line, large B
//! should climb toward the compute line.
//!
//!   cargo run --release -p hipfire-rdna --example bench_oq_compact_route [B,B,...]

use hipfire_rdna::{DType, Gpu};
use std::time::Instant;

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    // The continuous-batching route (wide multicol / narrow-N tile); unset
    // HIPFIRE_BENCH_SERVING=0 to bench the single-request routing instead.
    gpu.set_oq_batch_serving(std::env::var("HIPFIRE_BENCH_SERVING").as_deref() != Ok("0"));
    let mut seed = 0x1357_9BDFu32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        (seed >> 16) as u32
    };
    let group = 256usize;
    let n_out = 3usize;
    let stride = 2 + group / 2 + 2 * n_out;
    let bs: Vec<usize> = std::env::args()
        .nth(1)
        .map(|s| s.split(',').map(|v| v.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![1, 8, 16, 32, 33, 48, 64, 96, 128, 256, 512]);
    let shapes = [
        ("gate/up", 17408usize, 5120usize),
        ("down", 5120, 17408),
        ("qkv", 6144, 5120),
        ("wo", 5120, 4096),
    ];
    println!("  proj        M      K     B       ms     TOPS  %56    wGB/s");
    for &(name, m, k) in &shapes {
        let ng = k / group;
        let nblk = m * ng;
        let bytes = nblk * stride;
        let mut blocks = vec![0u8; bytes];
        for blk in 0..nblk {
            let off = blk * stride;
            let bits = (((14 + rnd() % 3) as u16) << 10) | (rnd() % 1024) as u16;
            blocks[off..off + 2].copy_from_slice(&bits.to_le_bytes());
            for i in 0..group / 2 {
                blocks[off + 2 + i] = (rnd() & 0xff) as u8;
            }
            let hdr = 2 + group / 2;
            let mut used = vec![false; group];
            for s in 0..n_out {
                let mut idx = (rnd() % group as u32) as usize;
                while used[idx] {
                    idx = (idx + 1) % group;
                }
                used[idx] = true;
                blocks[off + hdr + 2 * s] = idx as u8;
                blocks[off + hdr + 2 * s + 1] = (rnd() & 0xff) as u8;
                let nb = &mut blocks[off + 2 + idx / 2];
                *nb &= if idx % 2 == 0 { 0xf0 } else { 0x0f };
            }
        }
        // Split planes, as loaded: all nibble groups, then all [f16 scale][table].
        let side = stride - group / 2;
        let mut dev = vec![0u8; bytes];
        for blk in 0..nblk {
            let src = blk * stride;
            dev[blk * (group / 2)..blk * (group / 2) + group / 2]
                .copy_from_slice(&blocks[src + 2..src + 2 + group / 2]);
            let d = nblk * (group / 2) + blk * side;
            dev[d..d + 2].copy_from_slice(&blocks[src..src + 2]);
            dev[d + 2..d + side].copy_from_slice(&blocks[src + 2 + group / 2..src + stride]);
        }
        let wb = gpu.upload_raw(&dev, &[dev.len()]).expect("w");
        for &b in &bs {
            let x: Vec<f32> = (0..b * k)
                .map(|_| (rnd() % 2000) as f32 * 1e-3 - 1.0)
                .collect();
            let xb = gpu.upload_f32(&x, &[x.len()]).expect("x");
            let yb = gpu.alloc_tensor(&[b * m], DType::F32).expect("y");
            let run = |gpu: &mut Gpu| {
                gpu.gemm_oq_compact_act_batched(&wb, &xb, &yb, m, k, b, stride)
                    .expect("gemm")
            };
            run(&mut gpu);
            gpu.device_synchronize().unwrap();
            let iters = if b <= 32 { 20 } else { 10 };
            let t0 = Instant::now();
            for _ in 0..iters {
                run(&mut gpu);
            }
            gpu.device_synchronize().unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
            let tops = 2.0 * (m * k * b) as f64 / (ms * 1e-3) / 1e12;
            let gbs = bytes as f64 / (ms * 1e-3) / 1e9;
            println!(
                "  {name:<8} {m:>6} {k:>6} {b:>5} {ms:>8.3} {tops:>8.2} {:>4.0}% {gbs:>8.1}",
                100.0 * tops / 56.0
            );
            let _ = gpu.free_tensor(xb);
            let _ = gpu.free_tensor(yb);
        }
        let _ = gpu.free_tensor(wb);
    }
}
