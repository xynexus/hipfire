// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Routed mixture-of-experts MLP, forward and backward.
//!
//! The dense block's MLP is one SwiGLU over `[gate, up, down]`. A routed MoE
//! replaces it with `n_experts` such MLPs and a router that picks `top_k` of
//! them per token:
//!
//! ```text
//! logits[t,e] = x[t] · Wr[e]                      router, [seq, n_experts]
//! (idx, g)    = top_k(softmax(logits[t]))         per token, renormalised
//! y[t]        = Σ_j g[t,j] · SwiGLU_{idx[t,j]}(x[t])
//! ```
//!
//! This is written for CALIBRATION, not for training throughput: the loop is
//! over experts, each gathering its own tokens. That costs a pass per expert
//! but keeps every expert's input, output and adjoint separately addressable —
//! which is the entire point here, because per-expert `gamma` needs the adjoint
//! at each expert's output on its own tokens (see
//! docs/npu/scope-gamma-layer-streamed.md).
//!
//! Base weights are frozen, as everywhere in this crate: the backward returns
//! the input gradient and the per-expert output adjoints, not weight gradients.

use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

use crate::ops::linear::{linear_backward_x, linear_forward};
use crate::ops::swiglu::{swiglu_backward, swiglu_forward};

/// One expert's frozen SwiGLU weights.
pub struct ExpertWeights<'a> {
    pub wgate: &'a GpuTensor, // [inter, h]
    pub wup: &'a GpuTensor,   // [inter, h]
    pub wdown: &'a GpuTensor, // [h, inter]
}

/// The always-on shared expert qwen3.5/3.6 runs alongside the routed ones.
///
/// `w_scalar_gate` projects to a SINGLE scalar per token, not to `inter` —
/// `shared_expert_gate: [1, hidden]` in the layout. Its sigmoid scales the
/// whole shared branch, and it is a separate tensor from `wgate`, which is the
/// SwiGLU gate half. Conflating the two is the obvious way to get this wrong.
pub struct SharedExpert<'a> {
    pub w_scalar_gate: &'a GpuTensor, // [1, h]
    pub wgate: &'a GpuTensor,         // [shared_inter, h]
    pub wup: &'a GpuTensor,           // [shared_inter, h]
    pub wdown: &'a GpuTensor,         // [h, shared_inter]
    pub inter: usize,
}

/// Router plus experts for one MoE layer.
pub struct MoeWeights<'a> {
    pub router: &'a GpuTensor, // [n_experts, h]
    pub experts: Vec<ExpertWeights<'a>>,
    /// `None` for BLS-style layers with no shared branch.
    pub shared: Option<SharedExpert<'a>>,
}

pub struct MoeDims {
    pub seq: usize,
    pub h: usize,
    pub inter: usize,
    pub n_experts: usize,
    pub top_k: usize,
}

/// Saved forward state the backward consumes.
pub struct MoeActivations {
    /// Router logits `[seq, n_experts]`, pre-softmax.
    pub logits: Vec<f32>,
    /// Chosen expert index per (token, slot): `[seq, top_k]`.
    pub idx: Vec<u32>,
    /// Renormalised gate per (token, slot): `[seq, top_k]`.
    pub gate: Vec<f32>,
    /// Rows routed to each expert, and their slot within that token's top_k.
    pub rows: Vec<Vec<(usize, usize)>>,
    /// Per-expert saved SwiGLU operands, needed by its backward.
    pub e_gate: Vec<GpuTensor>,
    pub e_up: Vec<GpuTensor>,
    pub e_act: Vec<GpuTensor>,
    /// Per-expert gathered input rows `[n_rows, h]`.
    pub e_in: Vec<GpuTensor>,
    /// Per-expert output rows `[n_rows, h]`, BEFORE the gate scaling — this is
    /// the down_proj output, which is what a per-expert adjoint is taken at.
    pub e_out: Vec<GpuTensor>,
    /// Shared-expert saved state, when the layer has one.
    pub shared: Option<SharedActs>,
}

/// Shared-expert forward state. Every tensor is full-sequence: the branch is
/// always on, so there is no gather and no per-expert row bookkeeping.
pub struct SharedActs {
    pub s_gate: GpuTensor, // [seq, inter] SwiGLU gate half
    pub s_up: GpuTensor,   // [seq, inter]
    pub s_act: GpuTensor,  // [seq, inter] post-SwiGLU
    pub s_out: GpuTensor,  // [seq, h] down_proj output, BEFORE the scalar gate
    pub scalar: Vec<f32>,  // [seq] sigmoid(w_scalar_gate . x)
}

/// Softmax over the full router row, then top-k with renormalisation.
///
/// Done on the host: it is `seq * n_experts` floats once per layer, negligible
/// beside the expert GEMMs, and keeping the selection here means the backward
/// can walk the exact same routing without re-deriving it.
/// Softmax → top-k → renormalise over the kept set.
///
/// Public so `verify_moe_router` can cross-check it against the inference
/// kernel at the real expert count; nothing else outside this module uses it.
pub fn route(logits: &[f32], seq: usize, n_experts: usize, top_k: usize) -> (Vec<u32>, Vec<f32>) {
    let mut idx = vec![0u32; seq * top_k];
    let mut gate = vec![0.0f32; seq * top_k];
    for t in 0..seq {
        let row = &logits[t * n_experts..(t + 1) * n_experts];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = row.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let mut order: Vec<usize> = (0..n_experts).collect();
        order.sort_by(|&a, &b| {
            exp[b]
                .partial_cmp(&exp[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        // Renormalise over the kept set, which is what the inference path does
        // (`moe_softmax_topk_renorm_k8`) — the gates must sum to 1 per token.
        let kept: f32 = order[..top_k].iter().map(|&e| exp[e] / sum).sum();
        for (j, &e) in order[..top_k].iter().enumerate() {
            idx[t * top_k + j] = e as u32;
            gate[t * top_k + j] = (exp[e] / sum) / kept.max(1e-20);
        }
    }
    (idx, gate)
}

/// `y[t] = Σ_j gate[t,j] · SwiGLU_{idx[t,j]}(x[t])`.
pub fn moe_forward(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &MoeWeights,
    d: &MoeDims,
) -> HipResult<(GpuTensor, MoeActivations)> {
    let (seq, h, inter) = (d.seq, d.h, d.inter);

    let logits_t = gpu.zeros(&[seq * d.n_experts], DType::F32)?;
    linear_forward(gpu, x, w.router, &logits_t, seq, h, d.n_experts)?;
    let logits = gpu.download_f32(&logits_t)?;
    gpu.free_tensor(logits_t)?;
    let (idx, gate) = route(&logits, seq, d.n_experts, d.top_k);

    // Invert the routing: which (token, slot) pairs each expert must serve.
    let mut rows: Vec<Vec<(usize, usize)>> = vec![Vec::new(); d.n_experts];
    for t in 0..seq {
        for j in 0..d.top_k {
            rows[idx[t * d.top_k + j] as usize].push((t, j));
        }
    }

    let x_host = gpu.download_f32(x)?;
    let mut out_host = vec![0.0f32; seq * h];

    let (mut e_gate, mut e_up, mut e_act, mut e_in, mut e_out) =
        (vec![], vec![], vec![], vec![], vec![]);
    for e in 0..d.n_experts {
        let n = rows[e].len();
        // An expert with no routed tokens still needs placeholder entries so
        // every per-expert vector stays index-aligned with the expert id.
        let n1 = n.max(1);
        let xin = if n > 0 {
            let mut buf = vec![0.0f32; n * h];
            for (r, &(t, _)) in rows[e].iter().enumerate() {
                buf[r * h..(r + 1) * h].copy_from_slice(&x_host[t * h..(t + 1) * h]);
            }
            gpu.upload_f32(&buf, &[n * h])?
        } else {
            gpu.zeros(&[n1 * h], DType::F32)?
        };
        let g = gpu.zeros(&[n1 * inter], DType::F32)?;
        let u = gpu.zeros(&[n1 * inter], DType::F32)?;
        let a = gpu.zeros(&[n1 * inter], DType::F32)?;
        let o = gpu.zeros(&[n1 * h], DType::F32)?;
        if n > 0 {
            linear_forward(gpu, &xin, w.experts[e].wgate, &g, n, h, inter)?;
            linear_forward(gpu, &xin, w.experts[e].wup, &u, n, h, inter)?;
            swiglu_forward(gpu, &g, &u, &a, n * inter)?;
            linear_forward(gpu, &a, w.experts[e].wdown, &o, n, inter, h)?;
            let o_host = gpu.download_f32(&o)?;
            for (r, &(t, j)) in rows[e].iter().enumerate() {
                let gw = gate[t * d.top_k + j];
                for c in 0..h {
                    out_host[t * h + c] += gw * o_host[r * h + c];
                }
            }
        }
        e_gate.push(g);
        e_up.push(u);
        e_act.push(a);
        e_in.push(xin);
        e_out.push(o);
    }
    // Shared branch: y += sigmoid(w_sg . x) * SwiGLU_shared(x), every token.
    let shared = match &w.shared {
        None => None,
        Some(sh) => {
            let si = sh.inter;
            let sg = gpu.zeros(&[seq * si], DType::F32)?;
            let su = gpu.zeros(&[seq * si], DType::F32)?;
            let sa = gpu.zeros(&[seq * si], DType::F32)?;
            let so = gpu.zeros(&[seq * h], DType::F32)?;
            linear_forward(gpu, x, sh.wgate, &sg, seq, h, si)?;
            linear_forward(gpu, x, sh.wup, &su, seq, h, si)?;
            swiglu_forward(gpu, &sg, &su, &sa, seq * si)?;
            linear_forward(gpu, &sa, sh.wdown, &so, seq, si, h)?;

            let sgl = gpu.zeros(&[seq], DType::F32)?;
            linear_forward(gpu, x, sh.w_scalar_gate, &sgl, seq, h, 1)?;
            let scalar: Vec<f32> = gpu
                .download_f32(&sgl)?
                .iter()
                .map(|v| 1.0 / (1.0 + (-v).exp()))
                .collect();
            gpu.free_tensor(sgl)?;

            let so_host = gpu.download_f32(&so)?;
            for t in 0..seq {
                for c in 0..h {
                    out_host[t * h + c] += scalar[t] * so_host[t * h + c];
                }
            }
            Some(SharedActs {
                s_gate: sg,
                s_up: su,
                s_act: sa,
                s_out: so,
                scalar,
            })
        }
    };

    let out = gpu.upload_f32(&out_host, &[seq * h])?;

    Ok((
        out,
        MoeActivations {
            logits,
            idx,
            gate,
            rows,
            e_gate,
            e_up,
            e_act,
            e_in,
            e_out,
            shared,
        },
    ))
}

/// Per-expert output adjoints, `[n_rows, h]` per expert, in expert order.
///
/// Taken at the down_proj output BEFORE the gate scaling, matching where the
/// dense path takes `d_down`. Empty for an expert no token routed to.
pub struct MoeAdjoints {
    pub d_expert_out: Vec<Vec<f32>>,
    /// Router output adjoint `[seq, n_experts]`.
    pub d_router_out: Vec<f32>,
}

/// Backward. Returns `d_x` and the per-expert adjoints; base weights frozen.
pub fn moe_backward(
    gpu: &mut Gpu,
    d_out: &GpuTensor,
    w: &MoeWeights,
    a: &MoeActivations,
    d: &MoeDims,
) -> HipResult<(GpuTensor, MoeAdjoints)> {
    let (seq, h, inter) = (d.seq, d.h, d.inter);
    let d_out_host = gpu.download_f32(d_out)?;

    let mut d_x_host = vec![0.0f32; seq * h];
    // d(loss)/d(gate[t,j]) = <d_out[t], expert_out[t,j]>, collected per token
    // so the router softmax backward can be done once at the end.
    let mut d_gate = vec![0.0f32; seq * d.top_k];
    let mut d_expert_out: Vec<Vec<f32>> = Vec::with_capacity(d.n_experts);

    for e in 0..d.n_experts {
        let n = a.rows[e].len();
        if n == 0 {
            d_expert_out.push(Vec::new());
            continue;
        }
        // Adjoint at this expert's output: the upstream gradient of the rows it
        // served, scaled by that row's gate.
        let mut d_o = vec![0.0f32; n * h];
        let o_host = gpu.download_f32(&a.e_out[e])?;
        for (r, &(t, j)) in a.rows[e].iter().enumerate() {
            let gw = a.gate[t * d.top_k + j];
            let mut dot = 0.0f32;
            for c in 0..h {
                let g = d_out_host[t * h + c];
                d_o[r * h + c] = gw * g;
                dot += g * o_host[r * h + c];
            }
            d_gate[t * d.top_k + j] = dot;
        }
        d_expert_out.push(d_o.clone());

        let d_o_t = gpu.upload_f32(&d_o, &[n * h])?;
        let d_a = gpu.zeros(&[n * inter], DType::F32)?;
        linear_backward_x(gpu, &d_o_t, w.experts[e].wdown, &d_a, n, inter, h, false)?;
        let d_g = gpu.zeros(&[n * inter], DType::F32)?;
        let d_u = gpu.zeros(&[n * inter], DType::F32)?;
        swiglu_backward(gpu, &d_a, &a.e_gate[e], &a.e_up[e], &d_g, &d_u, n * inter)?;

        let d_xe = gpu.zeros(&[n * h], DType::F32)?;
        linear_backward_x(gpu, &d_g, w.experts[e].wgate, &d_xe, n, h, inter, false)?;
        let mut acc = gpu.download_f32(&d_xe)?;
        linear_backward_x(gpu, &d_u, w.experts[e].wup, &d_xe, n, h, inter, false)?;
        let up_part = gpu.download_f32(&d_xe)?;
        for (v, u) in acc.iter_mut().zip(up_part.iter()) {
            *v += *u;
        }
        for (r, &(t, _)) in a.rows[e].iter().enumerate() {
            for c in 0..h {
                d_x_host[t * h + c] += acc[r * h + c];
            }
        }

        for t in [d_o_t, d_a, d_g, d_u, d_xe] {
            gpu.free_tensor(t)?;
        }
    }

    // Shared branch backward. Two paths reach x: through the SwiGLU chain
    // (scaled by the scalar gate) and through the scalar gate itself, whose
    // adjoint is <d_out[t], s_out[t]> — the dot with the UNSCALED branch
    // output, exactly as the routed gate's adjoint is taken against e_out.
    if let (Some(sh), Some(sa)) = (&w.shared, &a.shared) {
        let si = sh.inter;
        let so_host = gpu.download_f32(&sa.s_out)?;
        let mut d_so = vec![0.0f32; seq * h];
        let mut d_scalar_pre = vec![0.0f32; seq];
        for t in 0..seq {
            let mut dot = 0.0f32;
            for c in 0..h {
                let g = d_out_host[t * h + c];
                d_so[t * h + c] = sa.scalar[t] * g;
                dot += g * so_host[t * h + c];
            }
            // sigmoid' = s(1-s)
            d_scalar_pre[t] = dot * sa.scalar[t] * (1.0 - sa.scalar[t]);
        }

        let d_so_t = gpu.upload_f32(&d_so, &[seq * h])?;
        let d_sa = gpu.zeros(&[seq * si], DType::F32)?;
        linear_backward_x(gpu, &d_so_t, sh.wdown, &d_sa, seq, si, h, false)?;
        let d_sg = gpu.zeros(&[seq * si], DType::F32)?;
        let d_su = gpu.zeros(&[seq * si], DType::F32)?;
        swiglu_backward(gpu, &d_sa, &sa.s_gate, &sa.s_up, &d_sg, &d_su, seq * si)?;

        let d_xs = gpu.zeros(&[seq * h], DType::F32)?;
        linear_backward_x(gpu, &d_sg, sh.wgate, &d_xs, seq, h, si, false)?;
        let mut acc = gpu.download_f32(&d_xs)?;
        linear_backward_x(gpu, &d_su, sh.wup, &d_xs, seq, h, si, false)?;
        for (v, u) in acc.iter_mut().zip(gpu.download_f32(&d_xs)?.iter()) {
            *v += *u;
        }
        let d_sp = gpu.upload_f32(&d_scalar_pre, &[seq])?;
        linear_backward_x(gpu, &d_sp, sh.w_scalar_gate, &d_xs, seq, h, 1, false)?;
        for (v, u) in acc.iter_mut().zip(gpu.download_f32(&d_xs)?.iter()) {
            *v += *u;
        }
        for (v, u) in d_x_host.iter_mut().zip(acc.iter()) {
            *v += *u;
        }
        for t in [d_so_t, d_sa, d_sg, d_su, d_xs, d_sp] {
            gpu.free_tensor(t)?;
        }
    }

    // Router backward. Softmax-then-renormalise over the kept set is, for the
    // kept entries, algebraically a softmax restricted to those entries — so
    // the standard softmax Jacobian applies with the renormalised gates, and
    // the unkept logits receive zero (top-k is piecewise constant in the
    // logits; it has no gradient of its own).
    let mut d_router_out = vec![0.0f32; seq * d.n_experts];
    for t in 0..seq {
        let dot: f32 = (0..d.top_k)
            .map(|j| d_gate[t * d.top_k + j] * a.gate[t * d.top_k + j])
            .sum();
        for j in 0..d.top_k {
            let e = a.idx[t * d.top_k + j] as usize;
            let g = a.gate[t * d.top_k + j];
            d_router_out[t * d.n_experts + e] = g * (d_gate[t * d.top_k + j] - dot);
        }
    }
    let d_logits = gpu.upload_f32(&d_router_out, &[seq * d.n_experts])?;
    let d_xr = gpu.zeros(&[seq * h], DType::F32)?;
    linear_backward_x(gpu, &d_logits, w.router, &d_xr, seq, h, d.n_experts, false)?;
    let xr = gpu.download_f32(&d_xr)?;
    for (v, r) in d_x_host.iter_mut().zip(xr.iter()) {
        *v += *r;
    }
    gpu.free_tensor(d_logits)?;
    gpu.free_tensor(d_xr)?;

    let d_x = gpu.upload_f32(&d_x_host, &[seq * h])?;
    Ok((
        d_x,
        MoeAdjoints {
            d_expert_out,
            d_router_out,
        },
    ))
}

/// Free one layer's saved MoE activations.
pub fn free_moe_acts(gpu: &mut Gpu, a: MoeActivations) -> HipResult<()> {
    for v in [a.e_gate, a.e_up, a.e_act, a.e_in, a.e_out] {
        for t in v {
            gpu.free_tensor(t)?;
        }
    }
    if let Some(s) = a.shared {
        for t in [s.s_gate, s.s_up, s.s_act, s.s_out] {
            gpu.free_tensor(t)?;
        }
    }
    Ok(())
}
