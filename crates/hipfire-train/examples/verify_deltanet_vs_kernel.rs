// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Cross-check the host delta-rule recurrence against the INFERENCE kernel.
//!
//! `gradcheck_deltanet` proves the backward is the derivative of the forward.
//! That is internal consistency, and it is silent on whether the forward is the
//! RIGHT recurrence: derive a backward from a wrong forward and the pair
//! gradchecks perfectly. Only hipfire's own `gated_delta_net_f32` — what the
//! 35B actually runs — settles that.
//!
//! (Measured, so the distinction is not hypothetical: dropping `alpha` in the
//! forward alone FAILS the gradcheck at 4.7e-1, because the backward still
//! carries the term. It is the consistent-pair mistake the gradcheck cannot
//! see, and this check can.)
//!
//! Two failure modes this catches, both verified by falsification:
//!
//!   * dropping the `alpha` inside `delta = (v - alpha*kv)*beta` — caught at
//!     rel 1.9e1
//!   * transposing S, `[hd_v, hd_k]` against `[hd_k, hd_v]` — caught at rel
//!     7.7e-1. This one is shape-VALID whenever the two dims are equal, and
//!     the real model has both at 128, so no dimension check would flag it.
//!
//! The kernel is hard-wired to head_dim 128 (`ensure_gdn_hd128`), so this runs
//! at the real width rather than a toy one.
//!
//! Run: cargo run --release -p hipfire-train --features deltanet \
//!        --example verify_deltanet_vs_kernel

use hipfire_rdna::{DType, Gpu};
use hipfire_train::ops::deltanet::deltanet_forward;

fn lcg(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init()?;
    const HD: usize = 128; // the kernel's compile-time width
    let (seq, nh) = (6usize, 2usize);

    let mut s = 0x0de17a_u64;
    let rnd = |n: usize, s: &mut u64, k: f32| (0..n).map(|_| k * lcg(s)).collect::<Vec<f32>>();

    let q = rnd(seq * nh * HD, &mut s, 0.5);
    let k = rnd(seq * nh * HD, &mut s, 0.5);
    let v = rnd(seq * nh * HD, &mut s, 0.5);
    // gate < 0 so alpha = exp(gate) decays, the regime the model runs in.
    let gate: Vec<f32> = rnd(seq * nh, &mut s, 0.5).iter().map(|x| x - 0.8).collect();
    let beta = rnd(seq * nh, &mut s, 0.5);

    let (host_out, _) = deltanet_forward(&q, &k, &v, &gate, &beta, seq, nh, HD, HD);

    let qt = gpu.upload_f32(&q, &[seq * nh * HD])?;
    let kt = gpu.upload_f32(&k, &[seq * nh * HD])?;
    let vt = gpu.upload_f32(&v, &[seq * nh * HD])?;
    let gt = gpu.upload_f32(&gate, &[seq * nh])?;
    let bt = gpu.upload_f32(&beta, &[seq * nh])?;
    // State starts at zero, matching the host's implicit S_{-1} = 0.
    let state = gpu.zeros(&[nh * HD * HD], DType::F32)?;
    let out = gpu.zeros(&[seq * nh * HD], DType::F32)?;

    gpu.gated_delta_net_f32(&qt, &kt, &vt, &gt, &bt, &state, &out, seq, nh, HD)?;
    let gpu_out = gpu.download_f32(&out)?;

    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    let mut mag = 0.0f32;
    for i in 0..seq * nh * HD {
        let d = (host_out[i] - gpu_out[i]).abs();
        mag = mag.max(gpu_out[i].abs());
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    let rel = worst / mag.max(1e-6);

    println!("delta-rule host vs gated_delta_net_f32: seq={seq} heads={nh} head_dim={HD}");
    println!("  output magnitude {mag:.4}");
    println!("  worst abs diff {worst:.3e} at {worst_i} (rel {rel:.3e})");

    for t in [qt, kt, vt, gt, bt, state, out] {
        gpu.free_tensor(t)?;
    }

    // f32 reduction order differs (the kernel does a 32-lane shuffle tree, the
    // host a serial sum over 128 terms), so exact equality is not the bar; a
    // wrong recurrence misses by O(1), not by 1e-5.
    if rel < 1e-4 && mag > 0.0 {
        println!("\nPASS — the host recurrence is the one the 35B runs");
        Ok(())
    } else {
        println!("\nFAIL — rel {rel:.3e}");
        std::process::exit(1)
    }
}
