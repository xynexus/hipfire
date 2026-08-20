// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! What access SHAPE sustains DRAM bandwidth under load?
//!
//! The compact decode GEMV moves the right bytes — FETCH_SIZE says 1.05x
//! overfetch — yet sustains ~213 GB/s against a 248.5 GB/s pure-read stream.
//! Since it is not reading too much, the shortfall has to be shape. Each arm
//! varies ONE property against the same weight payload:
//!
//!   stream          flat dwordx4 grid-stride                — the upper bound
//!   blocked/136+s   the real block: nibbles + f16 scale + overlay table
//!   blocked/136     same stride, nibble reads only          — prices side reads
//!   blocked/128     power-of-two stride, same instructions  — prices the
//!                   non-power-of-two block that straddles every 128 B line
//!   split           nibbles in a contiguous [M, K/2] plane + an 8 B/block side
//!                   plane — same total bytes, same bit budget, better shape
//!
//!   cargo run --release -p hipfire-rdna --example bench_oq_layout

use hipfire_rdna::{DType, Gpu};
use std::ffi::c_void;
use std::time::Instant;

const SRC: &str = include_str!("../../../kernels/src/bench_oq_layout.hip");

struct Blob(Vec<u8>);
impl Blob {
    fn new() -> Self {
        Blob(Vec::new())
    }
    fn align(&mut self, a: usize) {
        while self.0.len() % a != 0 {
            self.0.push(0);
        }
    }
    fn ptr(&mut self, p: *const c_void) -> &mut Self {
        self.align(8);
        self.0.extend_from_slice(&(p as u64).to_ne_bytes());
        self
    }
    fn i64v(&mut self, v: i64) -> &mut Self {
        self.align(8);
        self.0.extend_from_slice(&v.to_ne_bytes());
        self
    }
    fn i32v(&mut self, v: i32) -> &mut Self {
        self.align(4);
        self.0.extend_from_slice(&v.to_ne_bytes());
        self
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    // Real gate/up shape with M scaled up so the footprint is ~4x the 32 MiB
    // MALL. At the true 47 MB a chunk sits in cache and reports a bandwidth the
    // 13 GB/token sweep never actually sees.
    let (m, k) = (69632usize, 5120usize);
    let ng = k / 256;
    let iters = 20usize;
    for f in [
        "bwl_stream",
        "bwl_blocked",
        "bwl_split",
        "bwl_rowsplit",
        "bwl_tsplit",
    ] {
        gpu.ensure_kernel_public("bench_oq_layout", SRC, f)
            .expect("compile");
    }

    let alloc = |gpu: &mut Gpu, bytes: usize| {
        let t = gpu.alloc_tensor(&[bytes / 4], DType::F32).expect("alloc");
        gpu.fill_f32(&t, 1.0).expect("fill");
        t
    };
    let n136 = m * ng * 136;
    let w = alloc(&mut gpu, n136);
    let wn = alloc(&mut gpu, m * ng * 128);
    let ws = alloc(&mut gpu, m * ng * 8);
    let y = gpu.alloc_tensor(&[m], DType::F32).expect("y");

    println!("compact GEMV layout vs achieved bandwidth (M={m}, K={k}, ng={ng})");
    println!("every arm moves the same weight payload; only the SHAPE differs\n");
    println!("  arm              bytes MiB     ms      GB/s   vs stream");

    let mut base = 0f64;
    let mut bench = |gpu: &mut Gpu, name: &str, bytes: usize, run: &dyn Fn(&Gpu)| {
        run(gpu);
        gpu.device_synchronize().expect("sync");
        let t0 = Instant::now();
        for _ in 0..iters {
            run(gpu);
        }
        gpu.device_synchronize().expect("sync");
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let gbs = bytes as f64 / (ms * 1e-3) / 1e9;
        if base == 0.0 {
            base = gbs;
        }
        println!(
            "  {name:<15} {:>9.1}  {ms:>6.3}  {gbs:>8.1}    {:>5.1}%",
            bytes as f64 / (1024.0 * 1024.0),
            100.0 * gbs / base
        );
    };

    let (wp, ynp) = (w.buf.as_ptr(), y.buf.as_ptr());
    bench(&mut gpu, "stream", n136, &|g: &Gpu| {
        let mut kb = Blob::new();
        kb.ptr(wp as *const c_void)
            .ptr(ynp as *const c_void)
            .i64v(n136 as i64);
        g.launch_kernel_blob("bwl_stream", [4096, 1, 1], [256, 1, 1], 0, &mut kb.0)
            .expect("stream");
    });

    for (name, stride, side, bytes) in [
        ("blocked/136+s", 136i32, 1i32, n136),
        ("blocked/136", 136, 0, n136),
        ("blocked/128", 128, 0, m * ng * 128),
    ] {
        bench(&mut gpu, name, bytes, &|g: &Gpu| {
            let mut kb = Blob::new();
            kb.ptr(wp as *const c_void)
                .ptr(ynp as *const c_void)
                .i32v(m as i32)
                .i32v(ng as i32)
                .i32v(stride)
                .i32v(side);
            g.launch_kernel_blob("bwl_blocked", [m as u32, 1, 1], [32, 1, 1], 0, &mut kb.0)
                .expect("blocked");
        });
    }

    let (wnp, wsp) = (wn.buf.as_ptr(), ws.buf.as_ptr());
    bench(&mut gpu, "split", n136, &|g: &Gpu| {
        let mut kb = Blob::new();
        kb.ptr(wnp as *const c_void)
            .ptr(wsp as *const c_void)
            .ptr(ynp as *const c_void)
            .i32v(m as i32)
            .i32v(ng as i32);
        g.launch_kernel_blob("bwl_split", [m as u32, 1, 1], [32, 1, 1], 0, &mut kb.0)
            .expect("split");
    });

    bench(&mut gpu, "rowsplit", n136, &|g: &Gpu| {
        let mut kb = Blob::new();
        kb.ptr(wp as *const c_void)
            .ptr(ynp as *const c_void)
            .i32v(m as i32)
            .i32v(ng as i32)
            .i32v(136);
        g.launch_kernel_blob("bwl_rowsplit", [m as u32, 1, 1], [32, 1, 1], 0, &mut kb.0)
            .expect("rowsplit");
    });

    bench(&mut gpu, "tensorsplit", n136, &|g: &Gpu| {
        let mut kb = Blob::new();
        kb.ptr(wp as *const c_void)
            .ptr(ynp as *const c_void)
            .i32v(m as i32)
            .i32v(ng as i32)
            .i32v(8);
        g.launch_kernel_blob("bwl_tsplit", [m as u32, 1, 1], [32, 1, 1], 0, &mut kb.0)
            .expect("tsplit");
    });
}
