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
