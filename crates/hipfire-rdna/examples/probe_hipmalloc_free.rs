// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Does hipFree give memory back to the system on this host? Allocates ~4 GiB as
//! many buffers of varied sizes (1..48 MiB, like per-request KV/state), frees them
//! all, and reports GTT (/sys/class/drm/card1/device/mem_info_gtt_used) at each
//! stage; then repeats with sizes shifted so a size-keyed cache could not reuse.
//!
//!   cargo run --release -p hipfire-rdna --example probe_hipmalloc_free

use hipfire_rdna::Gpu;

fn gtt_mib() -> u64 {
    std::fs::read_to_string("/sys/class/drm/card1/device/mem_info_gtt_used")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
        >> 20
}

fn main() {
    let gpu = Gpu::init().unwrap();
    for round in 0..3usize {
        let g0 = gtt_mib();
        let mut bufs = Vec::new();
        let mut total = 0usize;
        let mut i = 0usize;
        while total < (4usize << 30) {
            let mib = 1 + (i * 7 + round * 13) % 48;
            let size = mib << 20 | (i % 5) * 4096;
            bufs.push(gpu.hip.malloc(size).unwrap());
            total += size;
            i += 1;
        }
        let g1 = gtt_mib();
        for b in bufs {
            gpu.hip.free(b).unwrap();
        }
        let g2 = gtt_mib();
        println!(
            "round {round}: {i} buffers, {} MiB: GTT {g0} -> {g1} (+{}) -> after free {g2} (still +{})",
            total >> 20,
            g1.saturating_sub(g0),
            g2.saturating_sub(g0)
        );
    }
}
