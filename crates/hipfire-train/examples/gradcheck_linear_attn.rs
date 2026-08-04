#![allow(clippy::needless_range_loop)]
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Finite-difference gradcheck for the FULL `linear_attn` layer, plus a
//! causality assertion.
//!
//! `gradcheck_deltanet` covers the recurrence core. This covers everything
//! wrapped around it — the qkv projection, the depthwise conv1d, the SiLU
//! split, the alpha/beta activations, the gated per-head RMSNorm and the output
//! projection — by checking `d_x`, which only exists if the whole chain
//! composed. Every branch of the fan-out from `x` (qkv, a, b, z) contributes,
//! so dropping any one of the four `matvec_seq_bwd` calls shows up here.
//!
//! The causality check is separate and NOT redundant: a conv1d whose taps point
//! the wrong way in time is perfectly differentiable, so its gradient matches
//! finite differences exactly. Only perturbing a future token and asserting the
//! past is unchanged catches it — and getting it wrong would leak the next
//! token into the current one, which trains a model that cannot generate.
//!
//! Run: cargo run --release -p hipfire-train --example gradcheck_linear_attn

use hipfire_train::ops::deltanet::{
    linear_attn_backward, linear_attn_forward, LinearAttnDims, LinearAttnWeights,
};

fn lcg(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn main() {
    // hd_k != hd_v and n_heads > 1, same as the core check: collapsing either
    // would hide an indexing bug in the [Q|K|V] split.
    let d = LinearAttnDims {
        seq: 6,
        h: 12,
        n_heads: 2,
        hd_k: 5,
        hd_v: 3,
        conv_k: 4,
        eps: 1e-6,
    };
    let (seq, h, nh) = (d.seq, d.h, d.n_heads);
    let qkv_dim = nh * (2 * d.hd_k + d.hd_v);

    let mut s = 0xbee71_u64;
    let rnd = |n: usize, s: &mut u64, k: f32| (0..n).map(|_| k * lcg(s)).collect::<Vec<f32>>();

    let x = rnd(seq * h, &mut s, 0.5);
    let in_proj_qkv = rnd(qkv_dim * h, &mut s, 0.3);
    // in_proj_a / in_proj_b are deliberately LARGER than the other projections.
    // At 0.3 the alpha and beta paths contribute so little to d_x that dropping
    // the softplus derivative entirely still lands at 2.9e-3 — under any sane
    // tolerance. The scale is set where the falsification actually bites.
    let in_proj_a = rnd(nh * h, &mut s, 1.5);
    let in_proj_b = rnd(nh * h, &mut s, 1.5);
    let in_proj_z = rnd(nh * d.hd_v * h, &mut s, 0.3);
    let conv1d = rnd(qkv_dim * d.conv_k, &mut s, 0.5);
    // A_log is stored as a log, so exp(A_log) is the real decay magnitude;
    // keeping it near 0 puts alpha in the regime the model runs in.
    let a_log = rnd(nh, &mut s, 0.3);
    let dt_bias = rnd(nh, &mut s, 0.3);
    let norm = (0..d.hd_v)
        .map(|_| 1.0 + 0.2 * lcg(&mut s))
        .collect::<Vec<_>>();
    let out_proj = rnd(h * nh * d.hd_v, &mut s, 0.3);
    let seed = rnd(seq * h, &mut s, 1.0);

    let w = LinearAttnWeights {
        in_proj_qkv: &in_proj_qkv,
        in_proj_a: &in_proj_a,
        in_proj_b: &in_proj_b,
        in_proj_z: &in_proj_z,
        conv1d: &conv1d,
        a_log: &a_log,
        dt_bias: &dt_bias,
        norm: &norm,
        out_proj: &out_proj,
    };

    let (_, acts) = linear_attn_forward(&x, &w, &d);
    let g = linear_attn_backward(&seed, &x, &w, &acts, &d);
    let dx = &g.d_x;

    let loss = |x: &[f32]| -> f64 {
        let (o, _) = linear_attn_forward(x, &w, &d);
        o.iter()
            .zip(seed.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum()
    };

    println!(
        "linear_attn gradcheck: seq={seq} h={h} heads={nh} hd_k={} hd_v={} conv_k={}",
        d.hd_k, d.hd_v, d.conv_k
    );

    let eps = 1e-3f32;
    let (mut worst, mut worst_i) = (0.0f64, 0usize);
    for i in (0..seq * h).step_by(3) {
        let (mut up, mut dn) = (x.clone(), x.clone());
        up[i] += eps;
        dn[i] -= eps;
        let num = (loss(&up) - loss(&dn)) / (2.0 * eps as f64);
        let rel = (num - dx[i] as f64).abs() / num.abs().max(dx[i].abs() as f64).max(1e-3);
        if rel > worst {
            worst = rel;
            worst_i = i;
        }
    }
    println!("  d_x worst rel {worst:.3e} at {worst_i}");

    // The alpha chain, checked where it cannot hide. dt_bias probes the
    // softplus derivative and a_log probes the -exp(A_log) factor; both are
    // per-head scalars, so their gradients are not diluted by the 26 qkv
    // channels the way d_x is.
    let probe = |name: &str, base: &[f32], ana: &[f32]| -> f64 {
        let mut worst_p = 0.0f64;
        for hh in 0..nh {
            let bump = |delta: f32| -> f64 {
                let mut v = base.to_vec();
                v[hh] += delta;
                let is_a_log = name == "d_a_log";
                let w2 = LinearAttnWeights {
                    a_log: if is_a_log { &v } else { &a_log },
                    dt_bias: if is_a_log { &dt_bias } else { &v },
                    ..w
                };
                let (o, _) = linear_attn_forward(&x, &w2, &d);
                o.iter()
                    .zip(seed.iter())
                    .map(|(p, q)| *p as f64 * *q as f64)
                    .sum()
            };
            // 3e-2, not the 1e-3 used for d_x. dgate is ~2 orders below dbeta,
            // so at 1e-3 the loss difference is buried in f32 forward noise and
            // reads 5.4e-2 — a "failure" that is entirely cancellation. Swept:
            // 5.4e-2 @1e-3, 5.3e-3 @3e-3, 2.4e-3 @1e-2, 1.7e-3 @3e-2,
            // 3.8e-3 @1e-1, 2.8e-2 @3e-1. The U-shape bottoming out is what a
            // correct derivative looks like; a wrong one has a floor.
            let ae = 3e-2f32;
            let num = (bump(ae) - bump(-ae)) / (2.0 * ae as f64);
            let rel = (num - ana[hh] as f64).abs() / num.abs().max(ana[hh].abs() as f64).max(1e-6);
            worst_p = worst_p.max(rel);
        }
        println!("  {name} worst rel {worst_p:.3e}");
        worst_p
    };
    let worst_a =
        probe("d_dt_bias", &dt_bias, &g.d_dt_bias).max(probe("d_a_log", &a_log, &g.d_a_log));

    // Causality: perturbing the LAST token must not move any earlier output.
    // Both the conv1d taps and the recurrence are directional; a flipped conv
    // gradchecks clean and only fails here.
    let (base, _) = linear_attn_forward(&x, &w, &d);
    let mut xf = x.clone();
    xf[(seq - 1) * h] += 1.0;
    let (pert, _) = linear_attn_forward(&xf, &w, &d);
    let leak = (0..(seq - 1) * h)
        .map(|i| (base[i] - pert[i]).abs())
        .fold(0.0f32, f32::max);
    println!("  causality: max leak into earlier tokens {leak:.3e}");

    if worst < 1e-2 && worst_a < 1e-2 && leak == 0.0 {
        println!("\nPASS — full layer matches finite differences and is causal");
    } else {
        println!("\nFAIL (worst {worst:.3e}, alpha {worst_a:.3e}, leak {leak:.3e})");
        std::process::exit(1);
    }
}
