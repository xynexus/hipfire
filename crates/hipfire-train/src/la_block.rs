#![allow(clippy::too_many_arguments)]
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! A `linear_attn` block: the DeltaNet half of a qwen3.5/3.6 hybrid layer,
//! with the MLP left to the caller exactly as [`crate::block::
//! block_forward_attn_only`] does for self-attention.
//!
//! The projections run on device and the core (conv1d, activations,
//! recurrence, gated norm) runs on the host. That split is not a shortcut: at
//! 35B the projections are billions of MACs per layer per sequence and a host
//! matvec would take minutes, while the delta-rule recurrence is inherently
//! sequential and small — `seq * heads * hd_k * hd_v` — so it gains nothing
//! from the GPU and would need a custom kernel to run there at all. The core
//! is shared verbatim with the pure-host path, so the gradchecked math has
//! exactly one implementation.

use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

use crate::ops::deltanet::{
    linear_attn_core_backward, linear_attn_core_forward, LinearAttnActs, LinearAttnCore,
    LinearAttnDims,
};
use crate::ops::linear::{linear_backward_x, linear_forward};
use crate::ops::rmsnorm::{rmsnorm_backward, rmsnorm_forward};

/// One `linear_attn` layer's frozen weights.
///
/// The four small tensors stay on the HOST: `conv1d` is `[qkv_dim, 1, conv_k]`
/// (a few tens of KB even at 35B) and `a_log`/`dt_bias`/`norm` are per-head or
/// per-head-dim vectors. The core consumes them as slices, so uploading them
/// would only mean downloading them again.
pub struct LinearAttnBlockWeights<'a> {
    pub norm1: &'a GpuTensor,
    pub in_proj_qkv: &'a GpuTensor,
    pub in_proj_a: &'a GpuTensor,
    pub in_proj_b: &'a GpuTensor,
    pub in_proj_z: &'a GpuTensor,
    pub out_proj: &'a GpuTensor,
    pub norm2: &'a GpuTensor,
    pub conv1d: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm: &'a [f32],
}

impl<'a> LinearAttnBlockWeights<'a> {
    fn core(&self) -> LinearAttnCore<'a> {
        LinearAttnCore {
            conv1d: self.conv1d,
            a_log: self.a_log,
            dt_bias: self.dt_bias,
            norm: self.norm,
        }
    }
}

/// Saved forward state. `x_mid` is the post-attention residual the caller adds
/// its MLP output to; `xn2` is that residual normed, the MLP's input.
pub struct LaBlockActs {
    pub xn1: GpuTensor,
    pub rinv1: GpuTensor,
    pub x_mid: GpuTensor,
    pub xn2: GpuTensor,
    pub rinv2: GpuTensor,
    pub core: LinearAttnActs,
    /// `out_proj`'s input, `[seq, n_heads*hd_v]`, kept on the host.
    pub normed: Vec<f32>,
}

/// Adjoints at each projection's OUTPUT — what a gamma pass integrates.
pub struct LaBlockAdjoints {
    pub d_qkv: Vec<f32>,
    pub d_a_raw: Vec<f32>,
    pub d_b_raw: Vec<f32>,
    pub d_z: Vec<f32>,
    /// `out_proj`'s output adjoint, which is the block's attention-side
    /// residual gradient.
    pub d_out_proj: Vec<f32>,
    pub d_dt_bias: Vec<f32>,
    pub d_a_log: Vec<f32>,
}

/// Attention-half forward. Returns `xn2`'s owner in `acts`; the caller runs the
/// MLP on `acts.xn2` and adds the result to `acts.x_mid`.
pub fn la_block_forward(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &LinearAttnBlockWeights,
    d: &LinearAttnDims,
) -> HipResult<LaBlockActs> {
    let (seq, h, nh, hk, hv) = (d.seq, d.h, d.n_heads, d.hd_k, d.hd_v);
    // q/k are nk*hd_k wide, NOT nh*hd_k — see LinearAttnDims::n_k_heads. Using
    // the value-head count here overshoots the real projection width and the
    // GEMM walks off the end of the weight (memory fault in gemm_f32_train).
    let qkv_dim = 2 * d.nk() * hk + nh * hv;
    let vd = nh * hv;

    let xn1 = gpu.zeros(&[seq * h], DType::F32)?;
    let rinv1 = gpu.zeros(&[seq], DType::F32)?;
    rmsnorm_forward(gpu, x, w.norm1, &xn1, &rinv1, seq, h, d.eps)?;

    let mut proj = Vec::with_capacity(4);
    for (wt, out_dim) in [
        (w.in_proj_qkv, qkv_dim),
        (w.in_proj_a, nh),
        (w.in_proj_b, nh),
        (w.in_proj_z, vd),
    ] {
        let t = gpu.zeros(&[seq * out_dim], DType::F32)?;
        linear_forward(gpu, &xn1, wt, &t, seq, h, out_dim)?;
        proj.push(gpu.download_f32(&t)?);
        gpu.free_tensor(t)?;
    }

    let (normed, core) =
        linear_attn_core_forward(&proj[0], &proj[1], &proj[2], &proj[3], &w.core(), d);

    let normed_t = gpu.upload_f32(&normed, &[seq * vd])?;
    let attn = gpu.zeros(&[seq * h], DType::F32)?;
    linear_forward(gpu, &normed_t, w.out_proj, &attn, seq, vd, h)?;
    let x_mid = gpu.zeros(&[seq * h], DType::F32)?;
    gpu.add_f32(x, &attn, &x_mid)?;
    gpu.free_tensor(normed_t)?;
    gpu.free_tensor(attn)?;

    let xn2 = gpu.zeros(&[seq * h], DType::F32)?;
    let rinv2 = gpu.zeros(&[seq], DType::F32)?;
    rmsnorm_forward(gpu, &x_mid, w.norm2, &xn2, &rinv2, seq, h, d.eps)?;

    Ok(LaBlockActs {
        xn1,
        rinv1,
        x_mid,
        xn2,
        rinv2,
        core,
        normed,
    })
}

/// Backward, taking the MLP's input gradient as `d_xn2` — the same contract as
/// [`crate::block::block_backward_from_dxn2`].
///
/// `d_xn2` must be the real gradient: it is the ONLY route by which the MLP's
/// error reaches the attention half, so a placeholder here silently truncates
/// the gradient rather than failing.
pub fn la_block_backward(
    gpu: &mut Gpu,
    d_x_out: &GpuTensor,
    d_xn2: &GpuTensor,
    x: &GpuTensor,
    w: &LinearAttnBlockWeights,
    a: &LaBlockActs,
    d: &LinearAttnDims,
) -> HipResult<(GpuTensor, LaBlockAdjoints)> {
    let (seq, h, nh, hk, hv) = (d.seq, d.h, d.n_heads, d.hd_k, d.hd_v);
    // q/k are nk*hd_k wide, NOT nh*hd_k — see LinearAttnDims::n_k_heads. Using
    // the value-head count here overshoots the real projection width and the
    // GEMM walks off the end of the weight (memory fault in gemm_f32_train).
    let qkv_dim = 2 * d.nk() * hk + nh * hv;
    let vd = nh * hv;
    let dw_dummy = gpu.zeros(&[h], DType::F32)?;

    // x_out = x_mid + mlp(xn2), so d_x_mid = d_x_out + norm2_backward(d_xn2).
    let d_mid_norm = gpu.zeros(&[seq * h], DType::F32)?;
    rmsnorm_backward(
        gpu,
        d_xn2,
        &a.x_mid,
        w.norm2,
        &a.rinv2,
        &d_mid_norm,
        &dw_dummy,
        seq,
        h,
    )?;
    let d_x_mid = gpu.zeros(&[seq * h], DType::F32)?;
    gpu.add_f32(d_x_out, &d_mid_norm, &d_x_mid)?;
    gpu.free_tensor(d_mid_norm)?;

    // x_mid = x + out_proj(normed): the attention branch's output adjoint IS
    // d_x_mid, and x also receives it directly through the residual.
    let d_out_proj = gpu.download_f32(&d_x_mid)?;
    let normed_t = gpu.upload_f32(&a.normed, &[seq * vd])?;
    let d_normed_t = gpu.zeros(&[seq * vd], DType::F32)?;
    linear_backward_x(gpu, &d_x_mid, w.out_proj, &d_normed_t, seq, vd, h, false)?;
    let d_normed = gpu.download_f32(&d_normed_t)?;
    gpu.free_tensor(normed_t)?;
    gpu.free_tensor(d_normed_t)?;

    let cg = linear_attn_core_backward(&d_normed, &w.core(), &a.core, d);

    // Every projection reads xn1, so their input gradients sum there.
    let mut d_xn1_host = vec![0.0f32; seq * h];
    let scratch = gpu.zeros(&[seq * h], DType::F32)?;
    for (adj, wt, out_dim) in [
        (&cg.d_qkv, w.in_proj_qkv, qkv_dim),
        (&cg.d_a_raw, w.in_proj_a, nh),
        (&cg.d_b_raw, w.in_proj_b, nh),
        (&cg.d_z, w.in_proj_z, vd),
    ] {
        let t = gpu.upload_f32(adj, &[seq * out_dim])?;
        linear_backward_x(gpu, &t, wt, &scratch, seq, h, out_dim, false)?;
        for (v, u) in d_xn1_host
            .iter_mut()
            .zip(gpu.download_f32(&scratch)?.iter())
        {
            *v += *u;
        }
        gpu.free_tensor(t)?;
    }
    gpu.free_tensor(scratch)?;

    let d_xn1 = gpu.upload_f32(&d_xn1_host, &[seq * h])?;
    let d_x_norm = gpu.zeros(&[seq * h], DType::F32)?;
    rmsnorm_backward(
        gpu, &d_xn1, x, w.norm1, &a.rinv1, &d_x_norm, &dw_dummy, seq, h,
    )?;
    let d_x = gpu.zeros(&[seq * h], DType::F32)?;
    gpu.add_f32(&d_x_mid, &d_x_norm, &d_x)?;

    for t in [d_xn1, d_x_norm, d_x_mid, dw_dummy] {
        gpu.free_tensor(t)?;
    }

    Ok((
        d_x,
        LaBlockAdjoints {
            d_qkv: cg.d_qkv,
            d_a_raw: cg.d_a_raw,
            d_b_raw: cg.d_b_raw,
            d_z: cg.d_z,
            d_out_proj,
            d_dt_bias: cg.d_dt_bias,
            d_a_log: cg.d_a_log,
        },
    ))
}

pub fn free_la_block_acts(gpu: &mut Gpu, a: LaBlockActs) -> HipResult<()> {
    for t in [a.xn1, a.rinv1, a.x_mid, a.xn2, a.rinv2] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}
