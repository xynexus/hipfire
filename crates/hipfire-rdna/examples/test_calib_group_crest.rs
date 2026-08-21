// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Verifies `calib_group_crest_reduce_f32` against a CPU reference, including
//! accumulate-across-calls (row 0 sums, row 1 maxes) — the corpus contract.

use hipfire_rdna::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    let (n, k, group) = (53usize, 1024usize, 256usize);
    let ng = k / group;
    let mut seed = 0x51A5_1A5Du32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        ((seed >> 8) as f32 / 8388608.0) - 1.0
    };
    // Deliberately spiky: a few channels get a large value so the per-group
    // crest is not uniformly ~1 and the max row is actually exercised.
    let mk = |r: &mut dyn FnMut() -> f32| -> Vec<f32> {
        let mut v: Vec<f32> = (0..n * k).map(|_| r() * 0.5).collect();
        for i in (0..n * k).step_by(377) {
            v[i] *= 40.0;
        }
        v
    };
    let x1 = mk(&mut rnd);
    let x2 = mk(&mut rnd);

    let acc = gpu.alloc_tensor(&[2 * ng], DType::F32).expect("acc");
    gpu.fill_f32(&acc, 0.0).expect("zero");
    for xs in [&x1, &x2] {
        let d = gpu.upload_f32(xs, &[n * k]).expect("x");
        gpu.calib_group_crest_reduce_f32(&d, &acc, n, k, group)
            .expect("run");
        let _ = gpu.free_tensor(d);
    }
    let got = gpu.download_f32(&acc).expect("dl");

    let mut worst = 0f32;
    for g in 0..ng {
        let (mut sum, mut mx) = (0f32, 0f32);
        for xs in [&x1, &x2] {
            for r in 0..n {
                let base = r * k + g * group;
                let mut m = 0f32;
                let mut ss = 0f32;
                for i in 0..group {
                    let v = xs[base + i];
                    m = m.max(v.abs());
                    ss += v * v;
                }
                let rms = (ss / group as f32).sqrt();
                let c = if rms > 1e-20 { m / rms } else { 0.0 };
                sum += c;
                mx = mx.max(c);
            }
        }
        for (i, want) in [sum, mx].iter().enumerate() {
            let d = (got[i * ng + g] - want).abs() / want.abs().max(1e-3);
            worst = worst.max(d);
        }
    }
    println!("calib_group_crest_reduce_f32 max|rel| vs CPU = {worst:.3e}");
    println!(
        "  sample: group0 mean crest = {:.2}, max = {:.2}",
        got[0] / (2 * n) as f32,
        got[ng]
    );
    if worst < 1e-4 {
        println!("[PASS] sum + max accumulate across 2 calls, matches CPU");
    } else {
        println!("[FAIL]");
        std::process::exit(1);
    }
}
