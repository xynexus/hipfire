// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Validate the FP64-accumulate DeltaNet ORACLE against an independent f64 CPU
//! reference — and validate the f32 kernel against the same reference.
//!
//! The oracle was produced by widening the f32 kernel's tile and arithmetic to
//! `double`, partly by mechanical transformation. It is now being used to make a
//! load-bearing claim (that the f32 reference drifts ~7x more than FP16 storage
//! does), so it needs to be something we KNOW rather than something we assume.
//! An oracle nobody checked is just a second guess.
//!
//! What a pass looks like:
//!   * `f64acc` vs the CPU f64 reference: at the FP32 STORAGE floor, ~1e-8, NOT
//!     f64 epsilon. The oracle accumulates in double but still stores state and
//!     output as f32, so one narrowing at the end is unavoidable and bounds it
//!     near f32 eps (~6e-8). An earlier version of this test asserted 1e-15 and
//!     "failed" a correct kernel; the bound was wrong, not the kernel.
//!   * `f32` vs the same reference: ~3e-7 here — an order of magnitude worse.
//!     That gap IS the term the oracle exists to isolate. If the two matched,
//!     the oracle would be measuring nothing and the whole exercise would be
//!     circular.
//!
//! Measured on gfx1103, 24 tokens, 2 heads, head_dim 128:
//!     f32 kernel     state 2.997e-7   output 3.035e-7
//!     f64acc oracle  state 2.497e-8   output 2.582e-8
//!
//! The first version of the oracle got TILE_ROWS wrong (8 instead of 4, inferred
//! from a stale comment in the f32 kernel rather than read from its `#define`).
//! The dispatcher launches 128/TILE_ROWS blocks, so the blocks overran the row
//! range and the oracle returned garbage at ~1.0 relative error — while still
//! producing plausible-looking aggregate numbers in a serving run. That is the
//! entire reason this test exists.
//!
//!   cargo run --release -p hipfire-rdna --example parity_gated_delta_net_f64acc

use hipfire_rdna::{Gpu, GpuTensor};

const HD: usize = 128;
const N_HEADS: usize = 2;
const N_TOKENS: usize = 24;

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            ((s as f32 + 0.5) / 2_147_483_648.0) * 2.0 - 1.0
        })
        .collect()
}

/// The recurrence, in f64, straight from the kernel body:
///   kv    = <S[r,:], k_t>
///   delta = (v_t[r] - alpha*kv) * beta
///   S[r,c] = alpha*S[r,c] + k_t[c]*delta
///   out[r] = <S[r,:], q_t>            (using the UPDATED row)
fn cpu_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state0: &[f32],
) -> (Vec<f64>, Vec<f64>) {
    let stride = N_HEADS * HD;
    let mut s: Vec<f64> = state0.iter().map(|&x| x as f64).collect();
    let mut out = vec![0.0f64; N_TOKENS * stride];
    for t in 0..N_TOKENS {
        for h in 0..N_HEADS {
            let alpha = (gate[t * N_HEADS + h] as f64).exp();
            let beta_v = beta[t * N_HEADS + h] as f64;
            let base = t * stride + h * HD;
            for r in 0..HD {
                let row = h * HD * HD + r * HD;
                let mut kv = 0.0f64;
                for c in 0..HD {
                    kv += s[row + c] * k[base + c] as f64;
                }
                let delta = (v[base + r] as f64 - alpha * kv) * beta_v;
                let mut acc = 0.0f64;
                for c in 0..HD {
                    s[row + c] = alpha * s[row + c] + k[base + c] as f64 * delta;
                    acc += s[row + c] * q[base + c] as f64;
                }
                out[base + r] = acc;
            }
        }
    }
    (s, out)
}

fn up(gpu: &mut Gpu, v: &[f32]) -> GpuTensor {
    gpu.upload_raw(
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
        &[v.len()],
    )
    .unwrap()
}

fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        num += ((*g as f64) - w).powi(2);
        den += w * w;
    }
    (num / den.max(1e-300)).sqrt()
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    let stride = N_HEADS * HD;
    let q = lcg(1, N_TOKENS * stride);
    let k = lcg(2, N_TOKENS * stride);
    let v = lcg(3, N_TOKENS * stride);
    // Keep alpha near 1: exp(gate) multiplies the state every step, so a wide
    // gate either explodes or annihilates it and the comparison stops being
    // about precision.
    let gate: Vec<f32> = lcg(4, N_TOKENS * N_HEADS)
        .iter()
        .map(|x| x * 0.02)
        .collect();
    let beta: Vec<f32> = lcg(5, N_TOKENS * N_HEADS).iter().map(|x| x * 0.5).collect();
    let state0: Vec<f32> = lcg(6, N_HEADS * HD * HD).iter().map(|x| x * 0.1).collect();

    let (ref_state, ref_out) = cpu_reference(&q, &k, &v, &gate, &beta, &state0);

    let mut run = |oracle: bool| -> (Vec<f32>, Vec<f32>) {
        // The dispatcher reads this once via OnceLock, so each variant needs its
        // own process; this example is run twice by the harness below instead.
        let qd = up(&mut gpu, &q);
        let kd = up(&mut gpu, &k);
        let vd = up(&mut gpu, &v);
        let gd = up(&mut gpu, &gate);
        let bd = up(&mut gpu, &beta);
        let sd = up(&mut gpu, &state0);
        let od = up(&mut gpu, &vec![0.0f32; N_TOKENS * stride]);
        let _ = oracle;
        gpu.gated_delta_net_f32_batch_seq(&qd, &kd, &vd, &gd, &bd, &sd, &od, N_TOKENS, N_HEADS, HD)
            .unwrap();
        gpu.device_synchronize().unwrap();
        (
            gpu.download_f32(&sd).unwrap(),
            gpu.download_f32(&od).unwrap(),
        )
    };

    let oracle_on = std::env::var("HIPFIRE_DN_STATE_F64_ORACLE").ok().as_deref() == Some("1");
    let (gpu_state, gpu_out) = run(oracle_on);

    let s_err = rel_err(&gpu_state, &ref_state);
    let o_err = rel_err(&gpu_out, &ref_out);
    let label = if oracle_on {
        "f64acc ORACLE"
    } else {
        "f32 kernel"
    };
    println!("{label}: tokens={N_TOKENS} heads={N_HEADS} head_dim={HD}");
    println!("  state rel L2 err vs f64 CPU reference : {s_err:.6e}");
    println!("  output rel L2 err vs f64 CPU reference: {o_err:.6e}");

    // The oracle's floor is FP32 STORAGE rounding (~6e-8), not f64 epsilon: it
    // accumulates in double but stores f32. The f32 kernel's bound is loose
    // enough to catch a structural break without pinning a hardware-specific
    // digit.
    let bound = if oracle_on { 1e-7 } else { 1e-2 };
    if s_err < bound && o_err < bound {
        println!("PASS (bound {bound:.0e})");
    } else {
        println!("FAIL: exceeded {bound:.0e}");
        std::process::exit(1);
    }
}
