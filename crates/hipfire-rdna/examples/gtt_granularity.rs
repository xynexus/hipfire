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
//! Usage: gtt_granularity <bytes>[,<bytes>...] [count]
//!
//! A COMMA-SEPARATED list is allocated round-robin, and that matters more than
//! it looks: the driver suballocates from 2 MiB blocks, so allocations smaller
//! than a block PACK, and the packing depends on what is next to them. Measured
//! one-size-at-a-time, a 1 114 112 B gate_up reads 1.881x (one per block) and a
//! 557 056 B down reads 1.254x (three per block). Measured as the loader
//! actually issues them — alternating — the pair sums to 1 671 168 B, fits ONE
//! block, and costs 1.255x. The isolated 1.881x was never available to win back.
//!
//! So: measure the interleaved stream, not one size in a loop. A single-size
//! run answers a question nobody is asking.

fn gtt_used() -> u64 {
    std::fs::read_to_string("/sys/class/drm/card1/device/mem_info_gtt_used")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let sizes: Vec<usize> = args
        .next()
        .map(|a| {
            a.split(',')
                .map(|t| t.trim().parse().expect("size must be an integer"))
                .collect()
        })
        .unwrap_or_else(|| vec![1_064_960]);
    let count: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(2000);

    assert!(!sizes.is_empty(), "need at least one size");
    for size in &sizes {
        assert!(
            size % 4 == 0,
            "size {size} must be a multiple of 4 (allocated as F32)"
        );
    }
    let mut gpu = hipfire_rdna::Gpu::init().expect("gpu");
    // Warm the context so its own allocations are not attributed to the test.
    let _warm = gpu
        .alloc_tensor(&[1024], hipfire_rdna::DType::F32)
        .expect("warm");

    let before = gtt_used();
    let mut keep = Vec::with_capacity(count * sizes.len());
    for _ in 0..count {
        // One pass over the list per round, so the stream INTERLEAVES the sizes
        // the way a loader issuing gate_up/down per expert does.
        for size in &sizes {
            // F32 with size/4 elements: DType has no byte-sized variant, and
            // every size under test is a multiple of 4.
            keep.push(
                gpu.alloc_tensor(&[size / 4], hipfire_rdna::DType::F32)
                    .expect("alloc"),
            );
        }
    }
    let after = gtt_used();

    let per_round: usize = sizes.iter().sum();
    let n_allocs = count * sizes.len();
    let requested = (per_round * count) as f64;
    let consumed = (after - before) as f64;
    let list = sizes
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "sizes=[{list}] B  rounds={count}  allocs={n_allocs}\n  \
         requested = {:.3} GiB\n  gtt delta = {:.3} GiB\n  ratio     = {:.3}x\n  \
         per-round = {:.0} B (requested {per_round})",
        requested / (1u64 << 30) as f64,
        consumed / (1u64 << 30) as f64,
        consumed / requested,
        consumed / count as f64,
    );
    std::mem::drop(keep);
}
