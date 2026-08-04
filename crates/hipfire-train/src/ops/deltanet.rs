// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Gated delta-rule linear attention (DeltaNet), forward and backward.
//!
//! Qwen3.5/3.6 are HYBRIDS: `Qwen3.6-35B-A3B` runs 30 of its 40 layers on this
//! and only 10 on softmax attention, so gamma for that model needs this path,
//! not just the GQA one.
//!
//! # The recurrence
//!
//! Transcribed from `kernels/src/gated_delta_net.hip` rather than a paper, so
//! the backward differentiates what actually runs. Per head, state
//! `S [HD_v, HD_k]`, per token:
//!
//! ```text
//! kv_t    = S_{t-1} · k_t                      (state BEFORE the update)
//! delta_t = (v_t − α_t·kv_t) · β_t             α_t = exp(gate_t)
//! S_t     = α_t·S_{t-1} + delta_t ⊗ k_t
//! out_t   = S_t · q_t                          (state AFTER the update)
//! ```
//!
//! The asymmetry is load-bearing: `kv` reads the pre-update state and `out`
//! reads the post-update one. Differentiating a version that used `S_{t-1}` for
//! both would be self-consistent and wrong.
//!
//! # Host-side, deliberately
//!
//! The recurrence is sequential in `t`, and a GPU backward would need its own
//! kernel with a reverse scan. This is calibration — one pass over a few short
//! sequences, once per model — so the cost is irrelevant and being obviously
//! correct is not. Every step is checked against finite differences in
//! `gradcheck_deltanet`.
//!
//! Cost at seq 256, 2 heads, HD 128: ~8M MAC per pass. Milliseconds.

/// Saved forward state for the backward: `S_t` after every step, plus the
/// pre-update `kv_t`.
///
/// Storing all states costs `seq · n_heads · HD_v · HD_k` floats — 33.5 MB at
/// seq 256 / 2 heads / HD 128. The alternative, running the recurrence
/// backwards as `S_{t-1} = (S_t − delta_t⊗k_t)/α_t`, divides by α and is
/// unstable exactly where the gate is doing its job.
pub struct DeltaNetActs {
    pub states: Vec<f32>, // [seq, n_heads, hd_v, hd_k], S AFTER step t
    pub kv: Vec<f32>,     // [seq, n_heads, hd_v], pre-update S·k
    pub alpha: Vec<f32>,  // [seq, n_heads]
}

/// Forward. `q`/`k` are `[seq, n_heads, hd_k]`, `v` is `[seq, n_heads, hd_v]`,
/// `gate`/`beta` are `[seq, n_heads]`. Returns `out [seq, n_heads, hd_v]`.
pub fn deltanet_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    seq: usize,
    n_heads: usize,
    hd_k: usize,
    hd_v: usize,
) -> (Vec<f32>, DeltaNetActs) {
    let mut s = vec![0.0f32; n_heads * hd_v * hd_k];
    let mut out = vec![0.0f32; seq * n_heads * hd_v];
    let mut acts = DeltaNetActs {
        states: vec![0.0f32; seq * n_heads * hd_v * hd_k],
        kv: vec![0.0f32; seq * n_heads * hd_v],
        alpha: vec![0.0f32; seq * n_heads],
    };

    for t in 0..seq {
        for h in 0..n_heads {
            let a = gate[t * n_heads + h].exp();
            let b = beta[t * n_heads + h];
            acts.alpha[t * n_heads + h] = a;
            let ko = (t * n_heads + h) * hd_k;
            let vo = (t * n_heads + h) * hd_v;
            let so = h * hd_v * hd_k;

            for r in 0..hd_v {
                // kv on the PRE-update state.
                let mut kv = 0.0f32;
                for c in 0..hd_k {
                    kv += s[so + r * hd_k + c] * k[ko + c];
                }
                acts.kv[vo + r] = kv;
                let delta = (v[vo + r] - a * kv) * b;
                for c in 0..hd_k {
                    s[so + r * hd_k + c] = a * s[so + r * hd_k + c] + k[ko + c] * delta;
                }
                // out on the POST-update state.
                let mut o = 0.0f32;
                for c in 0..hd_k {
                    o += s[so + r * hd_k + c] * q[ko + c];
                }
                out[vo + r] = o;
            }
            let dst = ((t * n_heads + h) * hd_v) * hd_k;
            acts.states[dst..dst + hd_v * hd_k].copy_from_slice(&s[so..so + hd_v * hd_k]);
        }
    }
    (out, acts)
}

/// Gradients of `q`, `k`, `v`, `gate`, `beta` given `d_out`.
///
/// Reverse scan carrying `dS`, the gradient w.r.t. the state entering step `t`.
/// Note `k_t` receives gradient from three places — the `kv` dot, the outer
/// product in the state update, and (via `kv`) the previous state — which is
/// the easiest term to drop and the reason the gradcheck perturbs `k` too.
#[allow(clippy::too_many_arguments)]
pub fn deltanet_backward(
    d_out: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    acts: &DeltaNetActs,
    seq: usize,
    n_heads: usize,
    hd_k: usize,
    hd_v: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dq = vec![0.0f32; seq * n_heads * hd_k];
    let mut dk = vec![0.0f32; seq * n_heads * hd_k];
    let mut dv = vec![0.0f32; seq * n_heads * hd_v];
    let mut dgate = vec![0.0f32; seq * n_heads];
    let mut dbeta = vec![0.0f32; seq * n_heads];
    // dS[h] = dL/dS_t, carried backwards.
    let mut ds = vec![0.0f32; n_heads * hd_v * hd_k];

    for t in (0..seq).rev() {
        for h in 0..n_heads {
            let a = acts.alpha[t * n_heads + h];
            let b = beta[t * n_heads + h];
            let ko = (t * n_heads + h) * hd_k;
            let vo = (t * n_heads + h) * hd_v;
            let so = h * hd_v * hd_k;
            let st = ((t * n_heads + h) * hd_v) * hd_k; // S_t
                                                        // S_{t-1}: the previous step's saved state, or zero at t = 0.
            let sp: Option<usize> = if t > 0 {
                Some((((t - 1) * n_heads + h) * hd_v) * hd_k)
            } else {
                None
            };

            let mut da = 0.0f32;
            for r in 0..hd_v {
                let g = d_out[vo + r];
                // out_t = S_t · q_t
                for c in 0..hd_k {
                    dq[ko + c] += acts.states[st + r * hd_k + c] * g;
                    ds[so + r * hd_k + c] += g * q[ko + c];
                }

                // S_t = α·S_{t-1} + delta ⊗ k
                let kv = acts.kv[vo + r];
                let delta = (v[vo + r] - a * kv) * b;
                let mut dd = 0.0f32; // dL/d delta_r
                for c in 0..hd_k {
                    let dsv = ds[so + r * hd_k + c];
                    dd += dsv * k[ko + c];
                    dk[ko + c] += dsv * delta;
                    let sprev = sp.map_or(0.0, |p| acts.states[p + r * hd_k + c]);
                    da += dsv * sprev;
                    ds[so + r * hd_k + c] = a * dsv;
                }

                // delta = (v − α·kv)·β
                dv[vo + r] += b * dd;
                dbeta[t * n_heads + h] += dd * (v[vo + r] - a * kv);
                da += dd * (-b * kv);
                let dkv = -b * a * dd;

                // kv = S_{t-1} · k
                for c in 0..hd_k {
                    let sprev = sp.map_or(0.0, |p| acts.states[p + r * hd_k + c]);
                    dk[ko + c] += dkv * sprev;
                    ds[so + r * hd_k + c] += dkv * k[ko + c];
                }
            }
            // α = exp(gate)
            dgate[t * n_heads + h] += da * a;
        }
    }
    (dq, dk, dv, dgate, dbeta)
}
