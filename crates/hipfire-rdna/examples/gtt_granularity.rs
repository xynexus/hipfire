// SPDX-License-Identifier: MIT
// Copyright (c) 2026 mad-lab-kbando
// hipfire — see LICENSE and NOTICE in the project root.

//! Measure the GTT cost of a device allocation of a given size.
//!
//! Answers one question: does `hipMalloc` consume more GTT than it is asked
//! for, and if so on what granularity? A routed-expert tensor is 1,064,960 B
//! (1.0156 MiB -- a hair over a power of two), and a 35B MoE load was observed
//! consuming 63.10 GiB of GTT for 33.58 GiB of requests.
//!
//! Usage: gtt_granularity <bytes> [count]

fn gtt_used() -> u64 {
    std::fs::read_to_string("/sys/class/drm/card1/device/mem_info_gtt_used")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let size: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(1_064_960);
    let count: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(2000);

    assert!(size % 4 == 0, "size must be a multiple of 4 (allocated as F32)");
    let mut gpu = hipfire_rdna::Gpu::init().expect("gpu");
    // Warm the context so its own allocations are not attributed to the test.
    let _warm = gpu.alloc_tensor(&[1024], hipfire_rdna::DType::F32).expect("warm");

    let before = gtt_used();
    let mut keep = Vec::with_capacity(count);
    for _ in 0..count {
        // F32 with size/4 elements: DType has no byte-sized variant, and every
        // size under test is a multiple of 4.
        keep.push(
            gpu.alloc_tensor(&[size / 4], hipfire_rdna::DType::F32)
                .expect("alloc"),
        );
    }
    let after = gtt_used();

    let requested = (size * count) as f64;
    let consumed = (after - before) as f64;
    println!(
        "size={size} B  count={count}\n  requested = {:.3} GiB\n  gtt delta = {:.3} GiB\n  ratio     = {:.3}x\n  per-alloc = {:.0} B (requested {size})",
        requested / (1u64 << 30) as f64,
        consumed / (1u64 << 30) as f64,
        consumed / requested,
        consumed / count as f64,
    );
    std::mem::drop(keep);
}
