// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Head-to-head: compact iu8 (`gemm_oq_compact_grouped_wmma`) vs compact iu4
//! (`gemm_oq_compact_iu4_wmma`) at real prefill shapes. Same weight bytes; the
//! iu4 arm feeds the bulk nibbles to the matrix core raw and takes int4
//! activations, halving the activation traffic.
//!
//! This kernel is 85.1% of a 2048-token prefill profile and the whole prefill
//! runs at ~14% of the ~56 TOPS int8 peak, which is why prefill is FLAT at
//! ~160 tok/s and declining with length instead of amortizing upward.
//! Reports effective TOPS, since this path is compute-bound, not
//! bandwidth-bound like the B<=16 GEMV.
//!
//!   cargo run --release -p hipfire-rdna --example bench_oq_compact_iu4

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
    let n_out = 3usize;
    let stride = 2 + GROUP / 2 + 2 * n_out; // 136 at n_out=3 => 4.25 bits
    let iters = 10usize;

    println!("gemv_oq_compact_multicol at Qwen3.8-27B shapes (n_out={n_out}, stride={stride})");
    println!("weight bytes only; 233 GB/s is the measured achievable ceiling\n");
    println!("  proj             M      K    B     iu8ms  iu8TOPS    iu4ms  iu4TOPS  speedup   [8-bit via 2 iu4 passes]");

    for &(name, m, k, b) in &[
        ("gate/up", 17408usize, 5120usize, 256usize),
        ("down", 5120, 17408, 256),
        ("qkv", 6144, 5120, 256),
        ("wo", 5120, 4096, 256),
        ("gate/up B=128", 17408, 5120, 128),
        ("gate/up B=512", 17408, 5120, 512),
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
        // Packed signed int4 activations, [B, K/2], byte = k_even | k_odd<<4.
        let x4: Vec<u8> = (0..b * k / 2).map(|_| (rnd() & 0xff) as u8).collect();
        // Two digit planes for the 8-bit-via-2-passes arm (same total bytes as
        // one int8 activation, which is the point).
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

        let x4b = gpu.upload_raw(&x4, &[x4.len()]).expect("x4");
        gpu.gemm_oq_compact_grouped_wmma(&wb, &xqb, &xsb, &yb, m, k, b, GROUP, stride)
            .expect("warm");
        gpu.gemm_oq_compact_iu4_wmma(&wb, &x4b, &xsb, &yb, m, k, b, GROUP, stride)
            .expect("warm4");
        gpu.device_synchronize().expect("sync");
        let t0 = Instant::now();
        for _ in 0..iters {
            gpu.gemm_oq_compact_grouped_wmma(&wb, &xqb, &xsb, &yb, m, k, b, GROUP, stride)
                .expect("launch");
        }
        gpu.device_synchronize().expect("sync");
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        // wave64 EXACT 2-pass arm — the production math on the tuned structure.
        let w64x2ms = {
            gpu.gemm_oq_compact_iu4x2_w64(&wb, &xqb, &xsb, &yb, m, k, b, stride)
                .expect("warm w64x2");
            gpu.device_synchronize().expect("sync");
            let tw = Instant::now();
            for _ in 0..iters {
                gpu.gemm_oq_compact_iu4x2_w64(&wb, &xqb, &xsb, &yb, m, k, b, stride)
                    .expect("w64x2");
            }
            gpu.device_synchronize().expect("sync");
            tw.elapsed().as_secs_f64() * 1e3 / iters as f64
        };
        // wave64 1-pass arm: nothing in the runtime routes here yet, so this is
        // purely "what would the wave64 recipe buy at our shapes".
        let w64ms = {
            gpu.gemm_oq_compact_iu4_w64(&wb, &x4b, &xsb, &yb, m, k, b, stride)
                .expect("warm w64");
            gpu.device_synchronize().expect("sync");
            let tw = Instant::now();
            for _ in 0..iters {
                gpu.gemm_oq_compact_iu4_w64(&wb, &x4b, &xsb, &yb, m, k, b, stride)
                    .expect("w64");
            }
            gpu.device_synchronize().expect("sync");
            tw.elapsed().as_secs_f64() * 1e3 / iters as f64
        };
        let t1 = Instant::now();
        for _ in 0..iters {
            gpu.gemm_oq_compact_iu4_wmma(&wb, &x4b, &xsb, &yb, m, k, b, GROUP, stride)
                .expect("launch4");
        }
        gpu.device_synchronize().expect("sync");
        let ms4 = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        gpu.gemm_oq_compact_iu4x2_wmma(&wb, &xqb, &xsb, &yb, m, k, b, GROUP, stride)
            .expect("warm2p");
        gpu.device_synchronize().expect("sync");
        let t2p = Instant::now();
        for _ in 0..iters {
            gpu.gemm_oq_compact_iu4x2_wmma(&wb, &xqb, &xsb, &yb, m, k, b, GROUP, stride)
                .expect("run2p");
        }
        gpu.device_synchronize().expect("sync");
        let ms2p = t2p.elapsed().as_secs_f64() * 1e3 / iters as f64;
        // Cost of the sparse overlay correction that makes the iu4 arm complete.
        gpu.gemv_oq_compact_overlay_correct(&wb, &x4b, &xsb, &yb, m, k, b, GROUP, stride)
            .expect("warmc");
        gpu.device_synchronize().expect("sync");
        let t2 = Instant::now();
        for _ in 0..iters {
            gpu.gemv_oq_compact_overlay_correct(&wb, &x4b, &xsb, &yb, m, k, b, GROUP, stride)
                .expect("corr");
        }
        gpu.device_synchronize().expect("sync");
        let msc = t2.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let tops4 = 2.0 * (m as f64) * (k as f64) * (b as f64) / (ms4 * 1e-3) / 1e12;
        // 2 ops per MAC, as TOPS is conventionally quoted for int8.
        let tops = 2.0 * (m as f64) * (k as f64) * (b as f64) / (ms * 1e-3) / 1e12;
        let r2i8 = ms / ms2p;
        let r2i4 = ms2p / ms4;
        println!(
            "  {name:<14} {m:>6} {k:>6} {b:>4} {ms:>8.3} {tops:>8.2} {ms4:>8.3} {tops4:>8.2} {:>8.2}x  corr={msc:>6.3}ms ({:>4.1}%) net={:>5.2}x  2p={ms2p:>7.3}ms {r2i8:>5.2}x-iu8 {r2i4:>5.2}x-1p  w64={w64ms:>7.3}ms {:>5.2}x-vs-1p  W64x2={w64x2ms:>7.3}ms {:>5.2}x-vs-2p",
            ms / ms4,
            100.0 * msc / ms4,
            ms / (ms4 + msc),
            ms4 / w64ms,
            ms2p / w64x2ms
        );
        let _ = bytes;
        let _ = gpu.free_tensor(x4b);
        let _ = gpu.free_tensor(wb);
        let _ = gpu.free_tensor(xqb);
        let _ = gpu.free_tensor(xsb);
        let _ = gpu.free_tensor(yb);
    }
}
