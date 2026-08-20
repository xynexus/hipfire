// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU parity for the DFlash2 grouped dynamic causal conv against the CPU
//! reference in `hipfire_runtime::dflash2`, which is itself parity-checked
//! against z-lab/dflash on real checkpoint weights. Uses the real Qwen3.8-27B
//! DFlash2 geometry (hidden 5120, group 16, 2 taps).

use hipfire_rdna::{DType, Gpu};

fn cpu_ref(
    h: &[f32],
    d: &[f32],
    b: &[f32],
    len: usize,
    hid: usize,
    ks: usize,
    gs: usize,
) -> Vec<f32> {
    let groups = hid / gs;
    let mut out = vec![0f32; len * hid];
    for t in 0..len {
        for tap in 0..ks {
            if t < tap {
                continue;
            }
            for c in 0..hid {
                let v = h[(t - tap) * hid + c];
                let k = b[tap * hid + c] + d[(t * ks + tap) * groups + c / gs];
                out[t * hid + c] += k * v;
            }
        }
    }
    out
}

fn main() {
    let (len, hid, ks, gs) = (8usize, 5120usize, 2usize, 16usize);
    let groups = hid / gs;
    // Deterministic pseudo-random inputs; no RNG dependency.
    let mk = |n: usize, seed: u64| -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.1
            })
            .collect()
    };
    let h = mk(len * hid, 1);
    let d = mk(len * ks * groups, 2);
    let b = mk(ks * hid, 3);
    let want = cpu_ref(&h, &d, &b, len, hid, ks, gs);

    let mut gpu = Gpu::init().expect("gpu");
    let hg = gpu.upload_f32(&h, &[len * hid]).unwrap();
    let dg = gpu.upload_f32(&d, &[len * ks * groups]).unwrap();
    let bg = gpu.upload_f32(&b, &[ks * hid]).unwrap();
    let yg = gpu.alloc_tensor(&[len * hid], DType::F32).unwrap();
    gpu.dflash2_grouped_dynamic_conv(&hg, &dg, &bg, &yg, len, hid, ks, gs)
        .expect("dflash2 conv");
    let got = gpu.download_f32(&yg).expect("download");

    let mut max_abs = 0f32;
    let mut at = 0usize;
    for i in 0..want.len() {
        let e = (got[i] - want[i]).abs();
        if e > max_abs {
            max_abs = e;
            at = i;
        }
    }
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
    let ok = max_abs <= 1e-5 * scale;
    println!(
        "parity_dflash2_conv len={len} hidden={hid} taps={ks} group={gs}: \
         max|Δ|={max_abs:.3e} (ref|max|={scale:.3e}, at {at}) -> {}",
        if ok { "PASS" } else { "FAIL" }
    );
    if !ok {
        std::process::exit(1);
    }
}
