// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Step 1 of a first-principles GEMM ladder: what does OPERAND SUPPLY cost?
//!
//! `bench_wmma_noop` measures the iu4 matrix core with loop-invariant operands —
//! pure issue rate. A real GEMM cannot do that: both operands come from LDS
//! every instruction. This runs the identical chain sweep with LDS-resident
//! operands, so the gap between the two curves is operand supply and nothing
//! else.
//!
//! Reference points on gfx1151: issue peak is 2048 ops/WGP/cycle x 20 WGP x
//! ~2.9 GHz = 119 TOPS; our shipping compact GEMM runs at 48.2 TOPS of iu4
//! issue, which is 41% of that. This bench says how much of the missing 59% is
//! already spent just feeding the instruction.

fn main() {
    let mut gpu = hipfire_rdna::Gpu::init().expect("gpu init");
    let blocks = 512u32;
    let iters = 2048i32;
    let waves_per_block = 8.0; // 256 threads / wave32
    println!("iu4 WMMA: issue rate vs LDS-fed operands (blocks={blocks}, iters={iters})");
    println!("gfx1151 iu4 issue peak = 119 TOPS; shipping compact GEMM = 48.2 TOPS\n");
    println!(
        "  {:>7} {:>11} {:>10} {:>11} {:>10} {:>11} {:>8}",
        "chains", "issue TOPS", "LDS TOPS", "supply", "+fold TOPS", "fold cost", "vs GEMM"
    );

    let out = gpu
        .upload_raw(&vec![0u8; blocks as usize * 4], &[blocks as usize])
        .expect("out");

    for c in [1u32, 2, 4, 8, 16] {
        let ops = blocks as f64 * waves_per_block * iters as f64 * c as f64 * 8192.0;
        // mode: 0 = pure issue, 1 = LDS-fed, 2 = LDS-fed + per-group fold
        let time = |gpu: &mut hipfire_rdna::Gpu, mode: u8| -> Option<f64> {
            let run = |g: &mut hipfire_rdna::Gpu| match mode {
                0 => g.wmma_iu4_noop(&out, blocks, iters, c, false),
                1 => g.wmma_iu4_lds_probe(&out, blocks, iters, c, false),
                _ => g.wmma_iu4_lds_probe(&out, blocks, iters, c, true),
            };
            run(gpu).ok()?;
            gpu.device_synchronize().ok()?;
            let t = std::time::Instant::now();
            for _ in 0..5 {
                run(gpu).ok()?;
            }
            gpu.device_synchronize().ok()?;
            Some(t.elapsed().as_secs_f64() / 5.0)
        };
        let (Some(si), Some(sl), Some(sf)) =
            (time(&mut gpu, 0), time(&mut gpu, 1), time(&mut gpu, 2))
        else {
            println!("  {c:>7}  (launch failed)");
            continue;
        };
        let (ti, tl, tf) = (ops / si / 1e12, ops / sl / 1e12, ops / sf / 1e12);
        println!(
            "  {:>7} {:>11.1} {:>10.1} {:>10.2}x {:>10.1} {:>10.2}x {:>7.2}x",
            c,
            ti,
            tl,
            sl / si,
            tf,
            sf / sl,
            tf / 48.2
        );
    }
}
