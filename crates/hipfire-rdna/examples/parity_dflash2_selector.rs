// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU parity for the DFlash2 candidate-selector scores against the CPU
//! reference in `hipfire_runtime::dflash2`, which is parity-checked against
//! z-lab/dflash on real checkpoint weights. Real geometry: rank 256, top_k 16.

use hipfire_rdna::{DType, Gpu};

fn main() {
    let (top_k, rank) = (16usize, 256usize);
    let mk = |n: usize, seed: u64| -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.5
            })
            .collect()
    };
    let unary = mk(top_k, 11);
    let pred = mk(rank, 12);
    let hp = mk(rank, 13);
    let suc = mk(top_k * rank, 14);

    // CPU reference: gate the predecessor row by the projected hidden, dot with
    // each candidate's successor row.
    let gated: Vec<f32> = (0..rank).map(|r| pred[r] * hp[r]).collect();
    let want: Vec<f32> = (0..top_k)
        .map(|k| {
            unary[k]
                + gated
                    .iter()
                    .zip(&suc[k * rank..(k + 1) * rank])
                    .map(|(g, s)| g * s)
                    .sum::<f32>()
        })
        .collect();

    let mut gpu = Gpu::init().expect("gpu");
    let ug = gpu.upload_f32(&unary, &[top_k]).unwrap();
    let pg = gpu.upload_f32(&pred, &[rank]).unwrap();
    let hg = gpu.upload_f32(&hp, &[rank]).unwrap();
    let sg = gpu.upload_f32(&suc, &[top_k * rank]).unwrap();
    let og = gpu.alloc_tensor(&[top_k], DType::F32).unwrap();
    gpu.dflash2_candidate_selector(&ug, &pg, &hg, &sg, &og, top_k, rank)
        .expect("dflash2 selector");
    let got = gpu.download_f32(&og).expect("download");

    let mut max_abs = 0f32;
    for i in 0..top_k {
        max_abs = max_abs.max((got[i] - want[i]).abs());
    }
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
    // The argmax is what actually drives the greedy trace, so check it too.
    let am = |v: &[f32]| (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b });
    let argmax_ok = am(&got) == am(&want);
    let ok = max_abs <= 1e-5 * scale && argmax_ok;
    println!(
        "parity_dflash2_selector top_k={top_k} rank={rank}: max|Δ|={max_abs:.3e} \
         (ref|max|={scale:.3e}) argmax {} -> {}",
        if argmax_ok { "match" } else { "MISMATCH" },
        if ok { "PASS" } else { "FAIL" }
    );
    if !ok {
        std::process::exit(1);
    }
}
