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

// ─── Full linear_attn layer ──────────────────────────────────────────────────
//
// Every formula below is transcribed from the inference path, not a paper:
//
//   q,k,v = split(silu(conv1d(Wqkv·x)))   conv1d.rs: "Fused conv1d+SiLU",
//                                          channel layout [Q | K | V]
//   beta  = sigmoid(Wb·x)                  fused.rs: "sigmoid(dn_beta)"
//   gate  = softplus(Wa·x + dt_bias)·(-exp(A_log))
//                                          activation.rs: alpha_gate_f32
//   out   = Wo · (rmsnorm(dn) * silu(Wz·x))
//                                          gated.rs: "rmsnorm(x) * silu(z)"
//
// The gate form is why alpha = exp(gate) lands in (0,1): softplus is positive
// and -exp(A_log) is negative, so the product is always negative and the state
// always decays.

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn dsilu(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s * (1.0 + x * (1.0 - s))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Saved state for [`linear_attn_backward`].
pub struct LinearAttnActs {
    pub qkv_pre: Vec<f32>, // conv1d output BEFORE silu, [seq, qkv_dim]
    /// q/k AFTER the per-head L2 norm (and q's 1/sqrt(hd) scale) — what the
    /// recurrence actually consumed.
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    /// q/k BEFORE that norm, which its backward needs.
    pub q_raw: Vec<f32>,
    pub k_raw: Vec<f32>,
    pub v: Vec<f32>,
    pub a_raw: Vec<f32>, // Wa·x, pre dt_bias/softplus
    pub b_raw: Vec<f32>, // Wb·x, pre sigmoid
    pub beta: Vec<f32>,
    pub gate: Vec<f32>,
    pub dn_out: Vec<f32>, // recurrence output
    pub z: Vec<f32>,      // Wz·x
    pub dn: DeltaNetActs,
}

pub struct LinearAttnDims {
    pub seq: usize,
    pub h: usize,
    pub n_heads: usize,
    pub hd_k: usize,
    pub hd_v: usize,
    pub conv_k: usize,
    pub eps: f32,
}

/// Row-major `[out, in]` matvec over a sequence: `y[t] = W · x[t]`.
fn matvec_seq(x: &[f32], w: &[f32], seq: usize, din: usize, dout: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; seq * dout];
    for t in 0..seq {
        for o in 0..dout {
            let mut acc = 0.0f32;
            for i in 0..din {
                acc += w[o * din + i] * x[t * din + i];
            }
            y[t * dout + o] = acc;
        }
    }
    y
}

/// `dx[t] += Wᵀ · dy[t]`, the transpose pass.
fn matvec_seq_bwd(dy: &[f32], w: &[f32], dx: &mut [f32], seq: usize, din: usize, dout: usize) {
    for t in 0..seq {
        for o in 0..dout {
            let g = dy[t * dout + o];
            if g == 0.0 {
                continue;
            }
            for i in 0..din {
                dx[t * din + i] += w[o * din + i] * g;
            }
        }
    }
}

/// Depthwise CAUSAL conv1d, kernel `conv_k`, per channel.
///
/// Causal means taps reach BACKWARD in time: position t reads t, t-1, ...
/// Getting this direction wrong leaks future tokens and is invisible in a
/// gradcheck — only a causality test catches it, which is why
/// `gradcheck_linear_attn` asserts that output t is independent of input t+1.
fn conv1d_causal(x: &[f32], w: &[f32], seq: usize, ch: usize, conv_k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; seq * ch];
    for t in 0..seq {
        for c in 0..ch {
            let mut acc = 0.0f32;
            for j in 0..conv_k {
                // Tap j is (conv_k-1-j) steps in the past.
                let src = t as isize - (conv_k - 1 - j) as isize;
                if src >= 0 {
                    acc += w[c * conv_k + j] * x[src as usize * ch + c];
                }
            }
            y[t * ch + c] = acc;
        }
    }
    y
}

fn conv1d_causal_bwd(dy: &[f32], w: &[f32], dx: &mut [f32], seq: usize, ch: usize, conv_k: usize) {
    for t in 0..seq {
        for c in 0..ch {
            let g = dy[t * ch + c];
            if g == 0.0 {
                continue;
            }
            for j in 0..conv_k {
                let src = t as isize - (conv_k - 1 - j) as isize;
                if src >= 0 {
                    dx[src as usize * ch + c] += w[c * conv_k + j] * g;
                }
            }
        }
    }
}

/// Weights for one `linear_attn` layer. All frozen; row-major `[out, in]`.
pub struct LinearAttnWeights<'a> {
    pub in_proj_qkv: &'a [f32],
    pub in_proj_a: &'a [f32],
    pub in_proj_b: &'a [f32],
    pub in_proj_z: &'a [f32],
    pub conv1d: &'a [f32],  // [qkv_dim, 1, conv_k]
    pub a_log: &'a [f32],   // [n_heads]
    pub dt_bias: &'a [f32], // [n_heads]
    pub norm: &'a [f32],    // [hd_v]
    pub out_proj: &'a [f32],
}

/// The non-projection half of the layer: conv1d, the activations, the
/// recurrence and the gated norm.
///
/// Split out so the GPU path and the host path share ONE implementation of the
/// math. The projections are big dense GEMMs that must run on device — at 35B
/// they are ~3 GMAC per layer per sequence and a host matvec would take
/// minutes — while the recurrence is inherently sequential and tiny. Anything
/// that duplicated the core to move the GEMMs would put the gradchecked math
/// on a path nothing checks.
pub struct LinearAttnCore<'a> {
    pub conv1d: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm: &'a [f32],
}

impl<'a> LinearAttnWeights<'a> {
    fn core(&self) -> LinearAttnCore<'a> {
        LinearAttnCore {
            conv1d: self.conv1d,
            a_log: self.a_log,
            dt_bias: self.dt_bias,
            norm: self.norm,
        }
    }
}

/// Core forward: projection OUTPUTS in, `normed [seq, n_heads*hd_v]` out.
pub fn linear_attn_core_forward(
    qkv: &[f32],
    a_raw: &[f32],
    b_raw: &[f32],
    z: &[f32],
    w: &LinearAttnCore,
    d: &LinearAttnDims,
) -> (Vec<f32>, LinearAttnActs) {
    let (seq, nh, hk, hv) = (d.seq, d.n_heads, d.hd_k, d.hd_v);
    let qkv_dim = nh * (hk + hk + hv);
    let qkv_pre = conv1d_causal(qkv, w.conv1d, seq, qkv_dim, d.conv_k);

    // Split [Q | K | V] after silu.
    let (mut q, mut k, mut v) = (
        vec![0.0f32; seq * nh * hk],
        vec![0.0f32; seq * nh * hk],
        vec![0.0f32; seq * nh * hv],
    );
    for t in 0..seq {
        let base = t * qkv_dim;
        for i in 0..nh * hk {
            q[t * nh * hk + i] = silu(qkv_pre[base + i]);
            k[t * nh * hk + i] = silu(qkv_pre[base + nh * hk + i]);
        }
        for i in 0..nh * hv {
            v[t * nh * hv + i] = silu(qkv_pre[base + 2 * nh * hk + i]);
        }
    }

    // Per-head L2 norm on q and k, plus q's 1/sqrt(hd) scale — the
    // `fused_qk_l2_norm_scale_f32` stage that sits between the conv split and
    // the recurrence. Omitting it leaves the delta-rule state unbounded: on a
    // real model at seq 64 the state grows until the backward overflows, and
    // the loss sits at ln(vocab). A short random fixture never shows it.
    let (q_raw, k_raw) = (q.clone(), k.clone());
    let q_scale = 1.0 / (hk as f32).sqrt();
    for t in 0..seq {
        for hh in 0..nh {
            let o = t * nh * hk + hh * hk;
            let qs: f32 = (0..hk).map(|i| q[o + i] * q[o + i]).sum();
            let ks: f32 = (0..hk).map(|i| k[o + i] * k[o + i]).sum();
            let qg = 1.0 / (qs + d.eps).sqrt();
            let kg = 1.0 / (ks + d.eps).sqrt();
            for i in 0..hk {
                q[o + i] *= qg * q_scale;
                k[o + i] *= kg;
            }
        }
    }

    let mut gate = vec![0.0f32; seq * nh];
    let mut beta = vec![0.0f32; seq * nh];
    for t in 0..seq {
        for hh in 0..nh {
            let i = t * nh + hh;
            // Numerically stable softplus. The naive ln(1+exp(x)) overflows
            // to inf for x > ~88 in f32, and then gate = -inf makes
            // d_a_log += dgate * gate evaluate 0 * -inf = NaN in the backward.
            // A small random fixture never reaches that; real dt_bias does.
            let x = a_raw[i] + w.dt_bias[hh];
            let sp = x.max(0.0) + (1.0 + (-x.abs()).exp()).ln();
            gate[i] = sp * -(w.a_log[hh].exp());
            beta[i] = sigmoid(b_raw[i]);
        }
    }

    let (dn_out, dn) = deltanet_forward(&q, &k, &v, &gate, &beta, seq, nh, hk, hv);

    let mut normed = vec![0.0f32; seq * nh * hv];
    for t in 0..seq {
        for hh in 0..nh {
            let o = (t * nh + hh) * hv;
            let ss: f32 = (0..hv).map(|i| dn_out[o + i] * dn_out[o + i]).sum();
            let inv = 1.0 / (ss / hv as f32 + d.eps).sqrt();
            for i in 0..hv {
                normed[o + i] = dn_out[o + i] * inv * w.norm[i] * silu(z[o + i]);
            }
        }
    }

    (
        normed,
        LinearAttnActs {
            qkv_pre,
            q,
            k,
            q_raw,
            k_raw,
            v,
            a_raw: a_raw.to_vec(),
            b_raw: b_raw.to_vec(),
            beta,
            gate,
            dn_out,
            z: z.to_vec(),
            dn,
        },
    )
}

/// Core backward: adjoint at `normed` in, adjoints at the projection OUTPUTS
/// out. Everything the caller needs to finish through its own GEMMs.
pub struct LinearAttnCoreGrads {
    pub d_qkv: Vec<f32>,
    pub d_a_raw: Vec<f32>,
    pub d_b_raw: Vec<f32>,
    pub d_z: Vec<f32>,
    pub d_dt_bias: Vec<f32>,
    pub d_a_log: Vec<f32>,
}

pub fn linear_attn_core_backward(
    d_normed: &[f32],
    w: &LinearAttnCore,
    a: &LinearAttnActs,
    d: &LinearAttnDims,
) -> LinearAttnCoreGrads {
    let (seq, nh, hk, hv) = (d.seq, d.n_heads, d.hd_k, d.hd_v);
    let qkv_dim = nh * (hk + hk + hv);

    // normed = rmsnorm(dn) * norm_w * silu(z)
    let mut d_dn = vec![0.0f32; seq * nh * hv];
    let mut d_z = vec![0.0f32; seq * nh * hv];
    for t in 0..seq {
        for hh in 0..nh {
            let o = (t * nh + hh) * hv;
            let ss: f32 = (0..hv).map(|i| a.dn_out[o + i] * a.dn_out[o + i]).sum();
            let ms = ss / hv as f32 + d.eps;
            let inv = 1.0 / ms.sqrt();
            // g_i is the gradient w.r.t. the NORMALISED value; the rmsnorm
            // Jacobian then couples the head.
            let mut gi = vec![0.0f32; hv];
            let mut dot = 0.0f32;
            for i in 0..hv {
                let sz = silu(a.z[o + i]);
                gi[i] = d_normed[o + i] * w.norm[i] * sz;
                d_z[o + i] =
                    d_normed[o + i] * w.norm[i] * a.dn_out[o + i] * inv * dsilu(a.z[o + i]);
                dot += gi[i] * a.dn_out[o + i];
            }
            for i in 0..hv {
                d_dn[o + i] = inv * (gi[i] - a.dn_out[o + i] * dot * inv * inv / hv as f32);
            }
        }
    }

    let (dq_n, dk_n, dv, dgate, dbeta) =
        deltanet_backward(&d_dn, &a.q, &a.k, &a.v, &a.beta, &a.dn, seq, nh, hk, hv);

    // L2-norm backward. y = c*x*g with g = (sum(x^2)+eps)^-1/2, so
    // dx = c*g*(dy - x * g^2 * <dy, x>). c is 1/sqrt(hd) for q, 1 for k.
    let mut dq = vec![0.0f32; seq * nh * hk];
    let mut dk = vec![0.0f32; seq * nh * hk];
    let q_scale = 1.0 / (hk as f32).sqrt();
    for t in 0..seq {
        for hh in 0..nh {
            let o = t * nh * hk + hh * hk;
            for (src, dst, xr, c) in [
                (&dq_n, &mut dq, &a.q_raw, q_scale),
                (&dk_n, &mut dk, &a.k_raw, 1.0),
            ] {
                let ss: f32 = (0..hk).map(|i| xr[o + i] * xr[o + i]).sum();
                let g = 1.0 / (ss + d.eps).sqrt();
                let dot: f32 = (0..hk).map(|i| src[o + i] * xr[o + i]).sum();
                for i in 0..hk {
                    dst[o + i] = c * g * (src[o + i] - xr[o + i] * g * g * dot);
                }
            }
        }
    }

    let mut d_a_raw = vec![0.0f32; seq * nh];
    let mut d_b_raw = vec![0.0f32; seq * nh];
    let mut d_dt_bias = vec![0.0f32; nh];
    let mut d_a_log = vec![0.0f32; nh];
    for t in 0..seq {
        for hh in 0..nh {
            let i = t * nh + hh;
            let sp_in = a.a_raw[i] + w.dt_bias[hh];
            // dt_bias enters sp_in exactly as a_raw does, so it collects the
            // same adjoint summed over time.
            d_a_raw[i] = dgate[i] * -(w.a_log[hh].exp()) * sigmoid(sp_in);
            d_dt_bias[hh] += d_a_raw[i];
            // gate = softplus * -exp(a_log), so d(gate)/d(a_log) = gate itself.
            d_a_log[hh] += dgate[i] * a.gate[i];
            let s = sigmoid(a.b_raw[i]);
            d_b_raw[i] = dbeta[i] * s * (1.0 - s);
        }
    }

    // Re-join q/k/v adjoints through silu into the conv output.
    let mut d_qkv_pre = vec![0.0f32; seq * qkv_dim];
    for t in 0..seq {
        let base = t * qkv_dim;
        for i in 0..nh * hk {
            d_qkv_pre[base + i] = dq[t * nh * hk + i] * dsilu(a.qkv_pre[base + i]);
            d_qkv_pre[base + nh * hk + i] =
                dk[t * nh * hk + i] * dsilu(a.qkv_pre[base + nh * hk + i]);
        }
        for i in 0..nh * hv {
            d_qkv_pre[base + 2 * nh * hk + i] =
                dv[t * nh * hv + i] * dsilu(a.qkv_pre[base + 2 * nh * hk + i]);
        }
    }
    let mut d_qkv = vec![0.0f32; seq * qkv_dim];
    conv1d_causal_bwd(&d_qkv_pre, w.conv1d, &mut d_qkv, seq, qkv_dim, d.conv_k);

    LinearAttnCoreGrads {
        d_qkv,
        d_a_raw,
        d_b_raw,
        d_z,
        d_dt_bias,
        d_a_log,
    }
}

/// Full `linear_attn` forward: `x [seq, h]` → `out [seq, h]`.
pub fn linear_attn_forward(
    x: &[f32],
    w: &LinearAttnWeights,
    d: &LinearAttnDims,
) -> (Vec<f32>, LinearAttnActs) {
    let (seq, h, nh, hk, hv) = (d.seq, d.h, d.n_heads, d.hd_k, d.hd_v);
    let qkv_dim = nh * (hk + hk + hv);

    let qkv = matvec_seq(x, w.in_proj_qkv, seq, h, qkv_dim);
    let a_raw = matvec_seq(x, w.in_proj_a, seq, h, nh);
    let b_raw = matvec_seq(x, w.in_proj_b, seq, h, nh);
    let z = matvec_seq(x, w.in_proj_z, seq, h, nh * hv);

    let (normed, acts) = linear_attn_core_forward(&qkv, &a_raw, &b_raw, &z, &w.core(), d);
    let out = matvec_seq(&normed, w.out_proj, seq, nh * hv, h);
    (out, acts)
}

/// Gradients out of [`linear_attn_backward`].
///
/// `d_dt_bias` and `d_a_log` are the only two weight gradients returned, and
/// they are here because nothing else can test the alpha activation. The
/// per-head RMSNorm downstream is scale-invariant, so it quotients out most of
/// alpha's uniform scaling of the state: measured, `dgate` runs two orders of
/// magnitude below `dbeta`, and dropping the softplus derivative entirely
/// perturbs `d_x` by only ~1e-3. These two vectors put the alpha chain on a
/// gradcheck that it cannot hide inside.
pub struct LinearAttnGrads {
    pub d_x: Vec<f32>,
    pub d_dt_bias: Vec<f32>,
    pub d_a_log: Vec<f32>,
}

/// Full `linear_attn` backward. Projection weights frozen; see
/// [`LinearAttnGrads`] for what comes back.
pub fn linear_attn_backward(
    d_out: &[f32],
    x: &[f32],
    w: &LinearAttnWeights,
    a: &LinearAttnActs,
    d: &LinearAttnDims,
) -> LinearAttnGrads {
    let (seq, h, nh, hk, hv) = (d.seq, d.h, d.n_heads, d.hd_k, d.hd_v);
    let qkv_dim = nh * (hk + hk + hv);
    let mut dx = vec![0.0f32; seq * h];

    let mut d_normed = vec![0.0f32; seq * nh * hv];
    matvec_seq_bwd(d_out, w.out_proj, &mut d_normed, seq, nh * hv, h);

    let g = linear_attn_core_backward(&d_normed, &w.core(), a, d);

    matvec_seq_bwd(&g.d_z, w.in_proj_z, &mut dx, seq, h, nh * hv);
    matvec_seq_bwd(&g.d_a_raw, w.in_proj_a, &mut dx, seq, h, nh);
    matvec_seq_bwd(&g.d_b_raw, w.in_proj_b, &mut dx, seq, h, nh);
    matvec_seq_bwd(&g.d_qkv, w.in_proj_qkv, &mut dx, seq, h, qkv_dim);

    let _ = (x, qkv_dim);
    LinearAttnGrads {
        d_x: dx,
        d_dt_bias: g.d_dt_bias,
        d_a_log: g.d_a_log,
    }
}
