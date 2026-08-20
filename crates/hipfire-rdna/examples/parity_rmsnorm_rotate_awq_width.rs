// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! `fused_rmsnorm_mq_rotate_awq` must produce the same result at every workgroup
//! width.
//!
//! The kernel runs as a SINGLE workgroup — the RMS reduction and the FWHT both
//! span the whole row — so blockDim is its only parallelism, and decode widened
//! it from 256 to 1024 threads. Widening changes the order of the phase-1b
//! sum-of-squares reduction, so the result is not bit-identical and must be
//! checked as a numeric equivalence rather than assumed.
//!
//! This is the ONLY coverage that change has: the tiny-quant fixtures are
//! hidden=256 and never reach the `k >= 2048` branch that selects 1024.
//!
//!   cargo run --release -p hipfire-rdna --example parity_rmsnorm_rotate_awq_width

use hipfire_rdna::{DType, Gpu};

fn lcg(seed: u32) -> impl FnMut() -> f32 {
    let mut s = seed.max(1);
    move || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        (s % 20000) as f32 * 1e-4 - 1.0
    }
}

fn run(gpu: &mut Gpu, k: usize, block: u32) -> Vec<f32> {
    std::env::set_var("HIPFIRE_RMSNORM_ROTATE_BLOCK", block.to_string());
    let mut rnd = lcg(0x5eed_1234 ^ k as u32);
    let x: Vec<f32> = (0..k).map(|_| rnd()).collect();
    let w: Vec<f32> = (0..k).map(|_| 1.0 + 0.25 * rnd()).collect();
    // AWQ scales are positive by construction.
    let a: Vec<f32> = (0..k).map(|_| 0.5 + 0.4 * (rnd() + 1.0)).collect();
    let xb = gpu.upload_f32(&x, &[k]).expect("x");
    let wb = gpu.upload_f32(&w, &[k]).expect("w");
    let ab = gpu.upload_f32(&a, &[k]).expect("awq");
    let ob = gpu.alloc_tensor(&[k], DType::F32).expect("out");
    gpu.fused_rmsnorm_rotate_mq_awq(&xb, &wb, &ab, &ob, k, 1e-6)
        .expect("launch");
    gpu.device_synchronize().expect("sync");
    gpu.download_f32(&ob).expect("download")
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    // Real Qwen3.8-27B / 35B-A3B hidden sizes, plus one that is not a multiple
    // of the wide block so the strided phase-1 loops are exercised too.
    let widths: &[usize] = &[2048, 5120, 6144, 2560];
    let mut fail = 0usize;
    println!("fused_rmsnorm_mq_rotate_awq — workgroup-width equivalence\n");
    println!("      K    max|rel|   verdict");
    for &k in widths {
        let a = run(&mut gpu, k, 256);
        let b = run(&mut gpu, k, 1024);
        let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
        let rel = a
            .iter()
            .zip(&b)
            .fold(0f32, |m, (p, q)| m.max((p - q).abs() / scale));
        // Only the reduction ORDER differs, so this should sit at f32 epsilon.
        let ok = rel < 1e-5;
        if !ok {
            fail += 1;
        }
        println!(
            "  {k:>5}   {rel:>9.2e}   {}",
            if ok { "PASS" } else { "FAIL" }
        );
    }
    if fail == 0 {
        println!("\nparity_rmsnorm_rotate_awq_width: PASS");
    } else {
        println!("\nparity_rmsnorm_rotate_awq_width: FAIL ({fail} width(s))");
        std::process::exit(1);
    }
}
