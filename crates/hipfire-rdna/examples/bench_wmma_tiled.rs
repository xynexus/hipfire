// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Rung 5: what does REAL GEMM TILING cost, on top of everything rungs 1-4 pay?
//!
//! BM=64 BN=128 BK=64, 8 wave32 waves as 2x4, WMt=WNt=2. Double-buffered
//! staging from a source larger than MALL, per-group fold every 4 strips, and
//! fragment addresses derived from (wave, subtile, k-step) exactly as a GEMM
//! does. 114 VGPRs, 12 waves/SIMD, 12 KB LDS.
//!
//! Reference: rung 4 (synthetic addressing) reached 88.0 TOPS; the shipping
//! wave64 compact GEMM runs at 48.2 with 234 VGPRs and 3 waves/SIMD.

fn main() {
    let mut gpu = hipfire_rdna::Gpu::init().expect("gpu init");
    let blocks = 512u32;
    let kstrips = 512i32;
    const SRC_W: usize = 16 * 1024 * 1024; // 64 MB, past the 32 MB MALL
    let out = gpu
        .upload_raw(&vec![0u8; blocks as usize * 4], &[blocks as usize])
        .expect("out");
    let src = gpu
        .upload_raw(&vec![0u8; SRC_W * 4], &[SRC_W])
        .expect("src");

    // per workgroup per strip: 8 waves x (BK/16=4) x (WMt*WNt=4) = 128 WMMA
    let wmma = blocks as f64 * kstrips as f64 * 128.0;
    let ops = wmma * 8192.0;

    gpu.wmma_iu4_tiled(&out, &src, blocks, kstrips, SRC_W)
        .expect("warm");
    gpu.device_synchronize().expect("sync");
    let t = std::time::Instant::now();
    for _ in 0..5 {
        gpu.wmma_iu4_tiled(&out, &src, blocks, kstrips, SRC_W)
            .expect("run");
    }
    gpu.device_synchronize().expect("sync");
    let s = t.elapsed().as_secs_f64() / 5.0;
    let tops = ops / s / 1e12;
    println!(
        "rung 5, real GEMM tiling: {:.3} ms, {tops:.1} TOPS",
        s * 1e3
    );
    println!("  rung 4 (synthetic addressing) 88.0 TOPS");
    println!(
        "  shipping wave64 compact GEMM  48.2 TOPS  -> this is {:.2}x it",
        tops / 48.2
    );
}
