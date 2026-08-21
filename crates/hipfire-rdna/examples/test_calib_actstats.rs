// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Verifies `calib_actstats_reduce_f32` (per-channel Σx, Σ|x|, Σx⁴, max|x|)
//! against a CPU reference, INCLUDING accumulate-in-place across two calls —
//! the corpus contract is that rows 0-2 add and row 3 takes a max.

use hipfire_rdna::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    let (n, k) = (37usize, 133usize);
    let mut seed = 0x9E3779B9u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        ((seed >> 8) as f32 / 8388608.0) - 1.0
    };
    let x1: Vec<f32> = (0..n * k).map(|_| rnd() * 3.0).collect();
    let x2: Vec<f32> = (0..n * k).map(|_| rnd() * 7.0).collect();

    let acc = gpu.alloc_tensor(&[4 * k], DType::F32).expect("acc");
    gpu.fill_f32(&acc, 0.0).expect("zero");
    for xs in [&x1, &x2] {
        let d = gpu.upload_f32(xs, &[n * k]).expect("x");
        gpu.calib_actstats_reduce_f32(&d, &acc, n, k).expect("run");
        let _ = gpu.free_tensor(d);
    }
    let got = gpu.download_f32(&acc).expect("dl");

    let mut worst = 0f32;
    for c in 0..k {
        let (mut s, mut sa, mut s4, mut mx) = (0f32, 0f32, 0f32, 0f32);
        for xs in [&x1, &x2] {
            for r in 0..n {
                let v = xs[r * k + c];
                s += v;
                sa += v.abs();
                s4 += (v * v) * (v * v);
                mx = mx.max(v.abs());
            }
        }
        for (i, want) in [s, sa, s4, mx].iter().enumerate() {
            let d = (got[i * k + c] - want).abs() / want.abs().max(1e-3);
            worst = worst.max(d);
        }
    }
    println!("calib_actstats_reduce_f32 max|rel| vs CPU = {worst:.3e}");
    if worst < 1e-4 {
        println!("[PASS] accumulate-in-place across 2 calls matches CPU");
    } else {
        println!("[FAIL]");
        std::process::exit(1);
    }
}
