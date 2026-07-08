// SPDX-License-Identifier: Apache-2.0
//! Numerical parity check for the bf16-compute training forward GEMM.
//!
//! The finite-difference gradchecks can't validate the bf16 path (perturbations
//! fall below bf16 precision), so this compares `gemm_bf16c_train_nt` against the
//! scalar f32 `gemm_f32_train` reference directly, over shapes representative of
//! the DSpark body (ingest / ctx K·V / block projections / lm-head) plus a
//! non-multiple-of-16 K to exercise the tail masking. Reports cosine + max
//! rel-err per shape; a WMMA fragment/layout bug shows up as a low cosine.

use hipfire_rdna::{DType, Gpu, HipResult};

fn rand_vec(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / (1u64 << 31) as f32; // ~[0,2)
            (u - 1.0) * scale
        })
        .collect()
}

fn main() -> HipResult<()> {
    let mut gpu = Gpu::init().expect("GPU init");

    // (m tokens, k contract, n out) — body-representative, last one has K%16!=0.
    let shapes = [
        (128usize, 2560usize, 2560usize), // ctx K/V-ish
        (128, 12800, 2560),               // ingest fc (large K)
        (56, 2560, 262208),               // batched lm-head (wb=8)
        (7, 2560, 10240),                 // block MLP gate/up (tiny M)
        (56, 2560, 1024),                 // batched k/v proj
        (40, 2576, 2048),                 // K=2576 not mult of 16 (tail mask)
    ];

    let mut worst_cos = 1.0f64;
    for (i, &(m, k, n)) in shapes.iter().enumerate() {
        let xh = rand_vec(m * k, 0x1000 + i as u64, 1.0 / (k as f32).sqrt());
        let wh = rand_vec(n * k, 0x2000 + i as u64, 1.0 / (k as f32).sqrt());
        let x = gpu.upload_f32(&xh, &[m, k])?;
        let w = gpu.upload_f32(&wh, &[n, k])?;

        let y_ref = gpu.zeros(&[m * n], DType::F32)?;
        gpu.gemm_f32_train(&x, &w, &y_ref, m, n, k, k, k, false, true)?;
        let y_bf = gpu.zeros(&[m * n], DType::F32)?;
        gpu.gemm_bf16c_train_nt(&x, &w, &y_bf, m, k, n)?;
        let y_f16 = gpu.zeros(&[m * n], DType::F32)?;
        gpu.gemm_f16c_train_nt(&x, &w, &y_f16, m, k, n)?;

        let a = gpu.download_f32(&y_ref)?;
        let b = gpu.download_f32(&y_bf)?;
        let c = gpu.download_f32(&y_f16)?;

        let metrics = |cand: &[f32]| -> (f64, f64) {
            let (mut dot, mut na, mut nb, mut max_rel) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for j in 0..a.len() {
                let (av, bv) = (a[j] as f64, cand[j] as f64);
                dot += av * bv;
                na += av * av;
                nb += bv * bv;
                max_rel = max_rel.max((av - bv).abs() / av.abs().max(1e-6));
            }
            (dot / (na.sqrt() * nb.sqrt() + 1e-30), max_rel)
        };
        let (cos_bf, rel_bf) = metrics(&b);
        let (cos_f16, rel_f16) = metrics(&c);
        worst_cos = worst_cos.min(cos_bf).min(cos_f16);
        println!(
            "  m={:>4} k={:>5} n={:>6}  bf16 cos={:.6} rel={:.4}  |  f16 cos={:.6} rel={:.4}",
            m, k, n, cos_bf, rel_bf, cos_f16, rel_f16
        );

        for t in [x, w, y_ref, y_bf, y_f16] {
            gpu.free_tensor(t)?;
        }
    }

    // Random inputs + f32 accumulate ⇒ bf16 input rounding gives cos well above
    // this; a WMMA layout bug would tank it toward 0.
    if worst_cos > 0.999 {
        println!("PARITY OK — worst cos {worst_cos:.6} (> 0.999)");
        Ok(())
    } else {
        println!("PARITY FAIL — worst cos {worst_cos:.6} (<= 0.999)");
        std::process::exit(1);
    }
}
