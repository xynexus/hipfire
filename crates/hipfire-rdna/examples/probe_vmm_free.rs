// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Does `VmmRegion::free` give a multi-page region's memory back? Maps 256 MiB of
//! 128 KiB pages (each its own hipMemCreate; the KV page size), frees the region
//! with its single range unmap, and reports the result and GTT before/after (gfx11
//! APU: /sys/class/drm/card1/device/mem_info_gtt_used). Measured 2026-10-02: it
//! does -- +256 MiB mapped, back to the same GTT after free, Ok(()).
//!
//!   cargo run --release -p hipfire-rdna --example probe_vmm_free

use hip_bridge::vmm::VmmRegion;
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
    let g = gpu.hip.vmm_granularity().expect("VMM granularity");
    let page = (128 * 1024usize).div_ceil(g) * g; // kv_page_bytes default
    let pages = (256usize << 20) / page;
    println!("page {page} B, {pages} pages = 256 MiB");
    // Alias rounds: B maps A's pages (hipMemRetainAllocationHandle + hipMemMap),
    // the way a fork shares a checkpoint's prefix; free B, then A. Every page
    // should be gone once both regions are freed.
    for round in 0..4 {
        let g0 = gtt_mib();
        let mut a = VmmRegion::reserve(&gpu.hip, pages * page, page).unwrap();
        a.ensure_mapped(&gpu.hip, pages * page).unwrap();
        let mut b = VmmRegion::reserve(&gpu.hip, pages * page, page).unwrap();
        let shared = b.alias_prefix_from(&gpu.hip, &a, pages * page).unwrap();
        let g1 = gtt_mib();
        // Odd rounds free the SOURCE first (a checkpoint evicted while a fork
        // still maps its prefix), even rounds the alias first.
        let (fb, g2, fa) = if round % 2 == 1 {
            let fa = a.free(&gpu.hip);
            let g2 = gtt_mib();
            (b.free(&gpu.hip), g2, fa)
        } else {
            let fb = b.free(&gpu.hip);
            let g2 = gtt_mib();
            (fb, g2, a.free(&gpu.hip))
        };
        let g3 = gtt_mib();
        println!(
            "alias round {round} ({}): shared {} MiB; GTT {g0} -> mapped {g1} -> after first free {g2} -> after both {g3} (still +{} MiB); free = {fb:?}/{fa:?}",
            if round % 2 == 1 { "source first" } else { "alias first" },
            shared >> 20,
            g3.saturating_sub(g0)
        );
    }
    for round in 0..3 {
        let g0 = gtt_mib();
        let mut r = VmmRegion::reserve(&gpu.hip, pages * page, page).unwrap();
        r.ensure_mapped(&gpu.hip, pages * page).unwrap();
        let g1 = gtt_mib();
        let freed = r.free(&gpu.hip);
        let g2 = gtt_mib();
        println!(
            "round {round}: GTT {g0} -> mapped {g1} (+{}) -> freed {g2} (still +{} MiB); free() = {freed:?}",
            g1 - g0,
            g2.saturating_sub(g0)
        );
    }
}
