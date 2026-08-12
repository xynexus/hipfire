#![allow(clippy::needless_range_loop)]
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Finite-difference gradcheck for the gated delta-rule recurrence.
//!
//! Checks ALL FIVE inputs independently. `k` is the one that matters most:
//! it receives gradient from three separate places — the `kv` dot product, the
//! outer product in the state update, and the previous state through `kv` — so
//! dropping one term still leaves `dq`, `dv`, `dgate` and `dbeta` correct and
//! only `dk` wrong. A check that only probed the output would miss it.
//!
//! `gate` is checked too because `α = exp(gate)` appears in three terms (the
//! state decay, the `kv` scaling inside delta, and hence `dkv`), and it is easy
//! to account for one and forget the others.
//!
//! Run: cargo run --release -p hipfire-train --example gradcheck_deltanet

use hipfire_train::ops::deltanet::{deltanet_backward, deltanet_forward};

fn lcg(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn main() {
    // Small but not degenerate: >1 head and hd_k != hd_v would both hide
    // indexing bugs if collapsed.
    let (seq, nh, hd_k, hd_v) = (6usize, 2usize, 8usize, 6usize);
    let mut s = 0xd317a_u64;
    let rnd = |n: usize, s: &mut u64, k: f32| (0..n).map(|_| k * lcg(s)).collect::<Vec<f32>>();

    let q = rnd(seq * nh * hd_k, &mut s, 0.5);
    let k = rnd(seq * nh * hd_k, &mut s, 0.5);
    let v = rnd(seq * nh * hd_v, &mut s, 0.5);
    // gate < 0 so α = exp(gate) ∈ (0,1): a decaying state, which is the regime
    // the model actually runs in. α > 1 would blow the recurrence up over seq.
    let gate: Vec<f32> = rnd(seq * nh, &mut s, 0.5).iter().map(|x| x - 0.8).collect();
    let beta = rnd(seq * nh, &mut s, 0.5);
    let seed = rnd(seq * nh * hd_v, &mut s, 1.0);

    let (_, acts) = deltanet_forward(&q, &k, &v, &gate, &beta, seq, nh, hd_k, hd_v);
    let (dq, dk, dv, dgate, dbeta) =
        deltanet_backward(&seed, &q, &k, &v, &beta, &acts, seq, nh, hd_k, hd_v);

    let loss = |q: &[f32], k: &[f32], v: &[f32], g: &[f32], b: &[f32]| -> f64 {
        let (o, _) = deltanet_forward(q, k, v, g, b, seq, nh, hd_k, hd_v);
        o.iter()
            .zip(seed.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum()
    };

    let eps = 1e-3f32;
    let mut worst_all = 0.0f64;
    println!("DeltaNet gradcheck: seq={seq} heads={nh} hd_k={hd_k} hd_v={hd_v}");

    for (name, base, ana) in [
        ("dq", &q, &dq),
        ("dk", &k, &dk),
        ("dv", &v, &dv),
        ("dgate", &gate, &dgate),
        ("dbeta", &beta, &dbeta),
    ] {
        let mut worst = 0.0f64;
        let mut worst_i = 0usize;
        for i in (0..base.len()).step_by(3) {
            let mut up = base.clone();
            let mut dn = base.clone();
            up[i] += eps;
            dn[i] -= eps;
            let (lp, lm) = match name {
                "dq" => (
                    loss(&up, &k, &v, &gate, &beta),
                    loss(&dn, &k, &v, &gate, &beta),
                ),
                "dk" => (
                    loss(&q, &up, &v, &gate, &beta),
                    loss(&q, &dn, &v, &gate, &beta),
                ),
                "dv" => (
                    loss(&q, &k, &up, &gate, &beta),
                    loss(&q, &k, &dn, &gate, &beta),
                ),
                "dgate" => (loss(&q, &k, &v, &up, &beta), loss(&q, &k, &v, &dn, &beta)),
                _ => (loss(&q, &k, &v, &gate, &up), loss(&q, &k, &v, &gate, &dn)),
            };
            let num = (lp - lm) / (2.0 * eps as f64);
            let a = ana[i] as f64;
            let rel = (num - a).abs() / num.abs().max(a.abs()).max(1e-3);
            if rel > worst {
                worst = rel;
                worst_i = i;
            }
        }
        println!("  {name:<6} worst rel {worst:.3e} at {worst_i}");
        worst_all = worst_all.max(worst);
    }

    if worst_all < 1e-2 {
        println!("\nPASS — all five inputs match finite differences");
    } else {
        println!("\nFAIL (worst {worst_all:.3e})");
        std::process::exit(1);
    }
}
