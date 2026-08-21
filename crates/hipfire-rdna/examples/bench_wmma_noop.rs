// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure iu4 WMMA issue rate, no memory, sweeping INDEPENDENT accumulator chains.
//!
//! One WMMA depends on the previous one writing the same accumulator, so C=1
//! exposes the full instruction latency and larger C gives the scheduler
//! independent work to interleave. Where the curve flattens is the minimum
//! accumulator count a real kernel needs; its height is the honest ceiling to
//! design a kernel against.
//!
//!   cargo run --release -p hipfire-rdna --example bench_wmma_noop

use hipfire_rdna::Gpu;
use std::time::Instant;

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    let blocks = 512u32; // 512 x 256 thr; plenty to fill 40 CUs
    let iters = 2048i32;
    let reps = 5usize;
    println!("iu4 WMMA issue rate, no memory (blocks={blocks}, iters={iters})");
    println!("  one 16x16x16 WMMA = 4096 MAC = 8192 ops\n");
    println!(
        "  {:>6}  {:>10} {:>10}   {:>10} {:>10}",
        "chains", "w64 ms", "w64 TOPS", "w32 ms", "w32 TOPS"
    );

    let out = gpu
        .upload_raw(&vec![0u8; blocks as usize * 4], &[blocks as usize])
        .expect("out");

    for &c in &[1u32, 2, 4, 8, 16, 32] {
        let mut row = format!("  {c:>6}");
        for wave64 in [true, false] {
            // waves per block: 256 threads = 4 wave64 or 8 wave32
            let waves_per_block = if wave64 { 4.0 } else { 8.0 };
            let ops = blocks as f64 * waves_per_block * iters as f64 * c as f64 * 8192.0;
            if let Err(e) = gpu.wmma_iu4_noop(&out, blocks, iters, c, wave64) {
                if c == 1 && !wave64 {
                    eprintln!("w32 error: {e:?}");
                }
                row.push_str(&format!("  {:>10} {:>10}", "-", "-"));
                continue;
            }
            gpu.device_synchronize().expect("sync");
            let mut best = f64::MAX;
            for _ in 0..reps {
                let t = Instant::now();
                gpu.wmma_iu4_noop(&out, blocks, iters, c, wave64)
                    .expect("run");
                gpu.device_synchronize().expect("sync");
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            row.push_str(&format!(
                "  {:>10.3} {:>10.1}",
                best,
                ops / (best * 1e-3) / 1e12
            ));
        }
        println!("{row}");
    }
}
