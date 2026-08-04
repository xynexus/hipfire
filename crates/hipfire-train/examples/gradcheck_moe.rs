#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Finite-difference gradcheck for the routed-MoE MLP backward.
//!
//! `moe_backward` is hand-derived — the expert chain, the gate adjoint, and the
//! softmax-with-renormalisation router Jacobian — so nothing about it is
//! self-evidently right. This compares the analytic `d_x` against central
//! differences of a scalar loss `L = <d_out_seed, moe_forward(x)>`, which is
//! the only check that actually exercises the derivation.
//!
//! Two traps this is built to catch:
//!   * **Gate path dropped.** If `d_gate` or the router Jacobian is wrong, `d_x`
//!     is still *nearly* right, because most of the gradient flows through the
//!     experts. A loose tolerance passes a broken router.
//!   * **Routing flips.** Perturbing `x` can change which experts win top-k,
//!     and at that point the function is genuinely discontinuous and finite
//!     differences are meaningless. `eps` is kept small and the check reports
//!     how many coordinates were skipped for a flip rather than averaging them
//!     in silently.
//!
//! Run: cargo run --release -p hipfire-train --example gradcheck_moe

use hipfire_rdna::Gpu;
use hipfire_train::ops::moe::{
    free_moe_acts, moe_backward, moe_forward, ExpertWeights, MoeDims, MoeWeights,
};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init()?;
    let d = MoeDims {
        seq: 6,
        h: 16,
        inter: 24,
        n_experts: 4,
        top_k: 2,
    };
    let (seq, h, inter, ne) = (d.seq, d.h, d.inter, d.n_experts);

    let mut s = 0x5eed_1234u64;
    let rnd = |n: usize, s: &mut u64| (0..n).map(|_| 0.3 * lcg(s)).collect::<Vec<f32>>();

    let x_host = rnd(seq * h, &mut s);
    let router_h = rnd(ne * h, &mut s);
    let eg: Vec<Vec<f32>> = (0..ne).map(|_| rnd(inter * h, &mut s)).collect();
    let eu: Vec<Vec<f32>> = (0..ne).map(|_| rnd(inter * h, &mut s)).collect();
    let ed: Vec<Vec<f32>> = (0..ne).map(|_| rnd(h * inter, &mut s)).collect();
    let seed = rnd(seq * h, &mut s); // the loss covector

    let router = gpu.upload_f32(&router_h, &[ne * h])?;
    let gts: Vec<_> = eg
        .iter()
        .map(|v| gpu.upload_f32(v, &[inter * h]).unwrap())
        .collect();
    let uts: Vec<_> = eu
        .iter()
        .map(|v| gpu.upload_f32(v, &[inter * h]).unwrap())
        .collect();
    let dts: Vec<_> = ed
        .iter()
        .map(|v| gpu.upload_f32(v, &[h * inter]).unwrap())
        .collect();
    let w = MoeWeights {
        router: &router,
        experts: (0..ne)
            .map(|e| ExpertWeights {
                wgate: &gts[e],
                wup: &uts[e],
                wdown: &dts[e],
            })
            .collect(),
    };

    // Analytic gradient.
    let x = gpu.upload_f32(&x_host, &[seq * h])?;
    let (out, acts) = moe_forward(&mut gpu, &x, &w, &d)?;
    let routing: Vec<u32> = acts.idx.clone();
    let d_out = gpu.upload_f32(&seed, &[seq * h])?;
    let (d_x, adj) = moe_backward(&mut gpu, &d_out, &w, &acts, &d)?;
    let d_x_host = gpu.download_f32(&d_x)?;
    let base: f32 = gpu
        .download_f32(&out)?
        .iter()
        .zip(seed.iter())
        .map(|(a, b)| a * b)
        .sum();
    free_moe_acts(&mut gpu, acts)?;

    println!(
        "MoE gradcheck: seq={seq} h={h} inter={inter} experts={ne} top_k={}",
        d.top_k
    );
    println!("  loss {base:.6}");
    let served: Vec<usize> = (0..ne).map(|e| adj.d_expert_out[e].len() / h).collect();
    println!("  rows per expert: {served:?}");

    // Central differences, skipping any coordinate whose perturbation flips the
    // routing — there the function is genuinely discontinuous.
    let eps = 1e-3f32;
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    let mut flips = 0usize;
    let mut checked = 0usize;
    for i in (0..seq * h).step_by(7) {
        let probe =
            |delta: f32, gpu: &mut Gpu| -> Result<(f32, bool), Box<dyn std::error::Error>> {
                let mut xp = x_host.clone();
                xp[i] += delta;
                let xt = gpu.upload_f32(&xp, &[seq * h])?;
                let (o, a) = moe_forward(gpu, &xt, &w, &d)?;
                let flipped = a.idx != routing;
                let l: f32 = gpu
                    .download_f32(&o)?
                    .iter()
                    .zip(seed.iter())
                    .map(|(p, q)| p * q)
                    .sum();
                free_moe_acts(gpu, a)?;
                gpu.free_tensor(o)?;
                gpu.free_tensor(xt)?;
                Ok((l, flipped))
            };
        let (lp, f1) = probe(eps, &mut gpu)?;
        let (lm, f2) = probe(-eps, &mut gpu)?;
        if f1 || f2 {
            flips += 1;
            continue;
        }
        let num = (lp - lm) / (2.0 * eps);
        let ana = d_x_host[i];
        let rel = (num - ana).abs() / num.abs().max(ana.abs()).max(1e-4);
        if rel > worst {
            worst = rel;
            worst_i = i;
        }
        checked += 1;
    }

    println!("  checked {checked} coords, {flips} skipped (routing flip)");
    println!("  worst relative error {worst:.3e} at coord {worst_i}");
    if worst < 2e-2 && checked > 0 {
        println!("\nPASS");
        Ok(())
    } else {
        println!("\nFAIL");
        std::process::exit(1)
    }
}
