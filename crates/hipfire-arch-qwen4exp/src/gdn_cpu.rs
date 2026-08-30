// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Gated DeltaNet — CPU reference, streaming (one token at a time).
//!
//! [`crate::gdn`] is the GPU dispatch path. This is the oracle it is differenced
//! against, and the only CPU gated-delta-rule in this tree — qwen35 has the same
//! mixer but implements it only on the GPU.
//!
//! Written in the RECURRENT form. The reference uses a chunked kernel for prefill
//! and the recurrent one for decode; the two are algebraically equal, so a
//! streaming implementation is comparable against either, and the recurrent form
//! is what decode actually runs.
//!
//! Details that are easy to get wrong, all of them load-bearing:
//!
//! * `g = -exp(A_log) * softplus(a + dt_bias)` is a DECAY: it must stay <= 0, and
//!   the state is multiplied by `exp(g)` each step.
//! * q and k are L2-normalised with an ADDITIVE epsilon inside the rsqrt
//!   (`rsqrt(sum(x*x) + eps)`), not a clamped norm, then q is scaled by
//!   `1/sqrt(head_k)`.
//! * With more value heads than key heads, q and k are `repeat_interleave`d, so
//!   value head `h` pairs with key head `h / (n_v / n_k)` — interleaved, NOT
//!   `h % n_k`. Those differ for every head once the ratio exceeds one.
//! * The output norm is gated by `z` through a SIGMOID here (`output_gate_type`),
//!   not the silu that the sibling Qwen families use.

/// Per-layer streaming state.
pub struct GdnCpuState {
    /// `[conv_dim * (kernel - 1)]`, oldest-first per channel.
    pub conv: Vec<f32>,
    /// `[n_v * head_k * head_v]`.
    pub recurrent: Vec<f32>,
}

pub struct GdnCpu<'a> {
    /// `[2 * key_dim + value_dim, hidden]`
    pub in_proj_qkv: &'a [f32],
    /// `[value_dim, hidden]`
    pub in_proj_z: &'a [f32],
    /// `[n_v, hidden]`
    pub in_proj_a: &'a [f32],
    /// `[n_v, hidden]`
    pub in_proj_b: &'a [f32],
    /// `[conv_dim, kernel]`, depthwise
    pub conv_weight: &'a [f32],
    /// `[n_v]`
    pub a_log: &'a [f32],
    /// `[n_v]`
    pub dt_bias: &'a [f32],
    /// `[head_v]`, plain (ones-init) weight
    pub norm_weight: &'a [f32],
    /// `[hidden, value_dim]`
    pub out_proj: &'a [f32],
    pub hidden: usize,
    pub n_k: usize,
    pub n_v: usize,
    pub head_k: usize,
    pub head_v: usize,
    pub kernel: usize,
    /// `output_gate_type == "sigmoid"`; false means silu (the `hidden_act` fallback).
    pub gate_sigmoid: bool,
    pub eps: f32,
}

fn mv(w: &[f32], x: &[f32], o: usize, i: usize) -> Vec<f32> {
    (0..o)
        .map(|r| (0..i).map(|c| w[r * i + c] * x[c]).sum())
        .collect()
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

impl GdnCpu<'_> {
    pub fn key_dim(&self) -> usize {
        self.n_k * self.head_k
    }
    pub fn value_dim(&self) -> usize {
        self.n_v * self.head_v
    }
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    pub fn zero_state(&self) -> GdnCpuState {
        GdnCpuState {
            conv: vec![0.0; self.conv_dim() * (self.kernel - 1)],
            recurrent: vec![0.0; self.n_v * self.head_k * self.head_v],
        }
    }

    /// One token in, one token out (`hidden`).
    pub fn step(&self, h: &[f32], st: &mut GdnCpuState) -> Vec<f32> {
        assert_eq!(h.len(), self.hidden);
        let (kd, vd, cd) = (self.key_dim(), self.value_dim(), self.conv_dim());

        let qkv = mv(self.in_proj_qkv, h, cd, self.hidden);
        let z = mv(self.in_proj_z, h, vd, self.hidden);
        let a = mv(self.in_proj_a, h, self.n_v, self.hidden);
        let b = mv(self.in_proj_b, h, self.n_v, self.hidden);

        // Causal depthwise conv over the ring, then silu. Tap j reads j positions
        // back; the newest tap is the current input.
        let ring = self.kernel - 1;
        let mut conv = vec![0.0f32; cd];
        for c in 0..cd {
            let mut acc = self.conv_weight[c * self.kernel + (self.kernel - 1)] * qkv[c];
            for j in 0..ring {
                acc += self.conv_weight[c * self.kernel + j] * st.conv[c * ring + j];
            }
            conv[c] = silu(acc);
        }
        for c in 0..cd {
            for j in 0..ring - 1 {
                st.conv[c * ring + j] = st.conv[c * ring + j + 1];
            }
            st.conv[c * ring + ring - 1] = qkv[c];
        }

        let (q, k, v) = (&conv[..kd], &conv[kd..2 * kd], &conv[2 * kd..]);
        let rep = self.n_v / self.n_k;
        let scale = 1.0 / (self.head_k as f32).sqrt();

        let mut core = vec![0.0f32; vd];
        for hv in 0..self.n_v {
            let hk = hv / rep; // repeat_interleave, not modulo
            let qh = &q[hk * self.head_k..(hk + 1) * self.head_k];
            let kh = &k[hk * self.head_k..(hk + 1) * self.head_k];
            let vh = &v[hv * self.head_v..(hv + 1) * self.head_v];

            let qn = 1.0 / (qh.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
            let kn = 1.0 / (kh.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();

            let beta = 1.0 / (1.0 + (-b[hv]).exp());
            let softplus = {
                let x = a[hv] + self.dt_bias[hv];
                // Numerically safe softplus; the reference relies on torch's.
                if x > 20.0 {
                    x
                } else {
                    (1.0 + x.exp()).ln()
                }
            };
            let g = (-self.a_log[hv].exp() * softplus).exp();

            let s = &mut st.recurrent
                [hv * self.head_k * self.head_v..(hv + 1) * self.head_k * self.head_v];
            for x in s.iter_mut() {
                *x *= g;
            }
            // kv_mem = S^T k ; delta = (v - kv_mem) * beta ; S += k (x) delta
            let mut kv_mem = vec![0.0f32; self.head_v];
            for ik in 0..self.head_k {
                let kk = kh[ik] * kn;
                for iv in 0..self.head_v {
                    kv_mem[iv] += s[ik * self.head_v + iv] * kk;
                }
            }
            for ik in 0..self.head_k {
                let kk = kh[ik] * kn;
                for iv in 0..self.head_v {
                    s[ik * self.head_v + iv] += kk * (vh[iv] - kv_mem[iv]) * beta;
                }
            }
            for ik in 0..self.head_k {
                let qq = qh[ik] * qn * scale;
                for iv in 0..self.head_v {
                    core[hv * self.head_v + iv] += s[ik * self.head_v + iv] * qq;
                }
            }
        }

        let normed = crate::hc::gated_rmsnorm(
            &core,
            &z,
            self.norm_weight,
            self.n_v,
            self.head_v,
            self.eps,
            self.gate_sigmoid,
        );
        mv(self.out_proj, &normed, self.hidden, vd)
    }
}
