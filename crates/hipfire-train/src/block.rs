// SPDX-License-Identifier: Apache-2.0
//! One pre-norm LLaMA transformer block, fp32, with LoRA on q_proj and v_proj.
//!
//! Forward saves every activation the backward needs into `BlockActivations`;
//! backward consumes them and produces LoRA gradients + the input gradient.
//! Base weights are FROZEN (no weight gradients except the LoRA adapters).
//!
//! fwd:  xn1 = rmsnorm(x); q = lora_q(xn1); k = lin(xn1); v = lora_v(xn1);
//!       q,k = rope(q),rope(k); ctx = gqa(q,k,v); attn = lin_o(ctx);
//!       x_mid = x + attn; xn2 = rmsnorm(x_mid);
//!       act = swiglu(lin_g(xn2), lin_u(xn2)); mlp = lin_d(act); x_out = x_mid + mlp

use crate::ops::attention::{gqa_backward, gqa_forward};
use crate::ops::linear::{linear_backward_w, linear_backward_x, linear_forward};
use crate::ops::lora::{lora_backward, lora_forward};
use crate::ops::rmsnorm::{rmsnorm_backward, rmsnorm_forward};
use crate::ops::rope::{rope_backward, rope_forward};
use crate::ops::swiglu::{swiglu_backward, swiglu_forward};
use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

#[derive(Clone, Copy)]
pub struct BlockDims {
    pub seq: usize,
    pub h: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub lora_scale: f32,
    pub lora_rank: usize,
}

impl BlockDims {
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv * self.head_dim
    }
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// Frozen base weights for one block (HF row-major `[out, in]`).
pub struct BlockWeights<'a> {
    pub norm1: &'a GpuTensor,
    /// Per-head RMSNorm on q and k, applied after the projection and BEFORE
    /// rope — the qwen3 family's QK-norm. `None` for llama-shaped models that
    /// do not have it. Both are present or neither: a checkpoint with one is
    /// malformed, and treating a missing one as identity would quietly change
    /// the attention scores.
    pub q_norm: Option<&'a GpuTensor>,
    pub k_norm: Option<&'a GpuTensor>,
    /// qwen3.5 `attn_output_gate`: `wq` emits `2*q_dim` with Q and a gate
    /// INTERLEAVED per head — `[Q_h0, G_h0, Q_h1, G_h1, ...]` — and the gate
    /// multiplies the attention output before `o_proj`
    /// (`sigmoid_mul_f32(fa_attn_out, fa_gate)` in the runtime).
    ///
    /// LoRA on q_proj is NOT applied on a gated layer: the adapter's `B` is
    /// `[q_dim, r]` while the base projection emits `2*q_dim`, so there is no
    /// single correct place to add it without deciding whether the adapter also
    /// steers the gate. The forward checks that `bq` is all-zero and errors
    /// otherwise rather than silently dropping a trained adapter.
    pub attn_out_gate: bool,
    pub wq: &'a GpuTensor,
    pub wk: &'a GpuTensor,
    pub wv: &'a GpuTensor,
    pub wo: &'a GpuTensor,
    pub norm2: &'a GpuTensor,
    pub wgate: &'a GpuTensor,
    pub wup: &'a GpuTensor,
    pub wdown: &'a GpuTensor,
}

/// Trainable LoRA adapters (q_proj, v_proj).
pub struct BlockLora<'a> {
    pub aq: &'a GpuTensor,
    pub bq: &'a GpuTensor,
    pub av: &'a GpuTensor,
    pub bv: &'a GpuTensor,
}

/// Gradients of the trainable block params: LoRA adapters + the two RMSNorm
/// weights (the layernorms QTIP recovery FT tunes). Base linears stay frozen.
pub struct BlockLoraGrad {
    pub daq: GpuTensor,
    pub dbq: GpuTensor,
    pub dav: GpuTensor,
    pub dbv: GpuTensor,
    pub dnorm1: GpuTensor, // [h]
    pub dnorm2: GpuTensor, // [h]
}

/// Saved forward activations needed by the backward pass.
pub struct BlockActivations {
    pub xn1: GpuTensor,
    pub rinv1: GpuTensor,
    /// Pre-QK-norm q and k with their row inverse-norms. Kept only when
    /// `q_norm`/`k_norm` are set, since the backward needs the norm's input.
    pub q_pre: Option<GpuTensor>,
    pub k_pre: Option<GpuTensor>,
    pub rinv_q: Option<GpuTensor>,
    pub rinv_k: Option<GpuTensor>,
    /// Pre-sigmoid attention gate `[seq, q_dim]`, and the context it scaled.
    pub attn_gate: Option<GpuTensor>,
    pub ctx_pre_gate: Option<GpuTensor>,
    pub hq: GpuTensor,
    pub hv: GpuTensor,
    pub q_r: GpuTensor,
    pub k_r: GpuTensor,
    pub v: GpuTensor,
    pub p_all: GpuTensor,
    pub ctx: GpuTensor,
    pub x_mid: GpuTensor,
    pub xn2: GpuTensor,
    pub rinv2: GpuTensor,
    pub gate: GpuTensor,
    pub up: GpuTensor,
    pub act: GpuTensor,
    pub pos: GpuTensor,
}

/// Forward. Returns `x_out` `[seq*h]` and the saved activations.
pub fn block_forward(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &BlockWeights,
    lora: &BlockLora,
    dims: &BlockDims,
    pos_host: &[f32],
    layer_idx: usize,
) -> HipResult<(GpuTensor, BlockActivations)> {
    block_forward_inner(gpu, x, w, lora, dims, pos_host, layer_idx, true)
}

/// Forward through ATTENTION ONLY, stopping after `norm2`.
///
/// Returns `x_out = x_mid` (no MLP added) and activations whose `xn2` is the
/// MLP's input. A routed-MoE block runs its own MLP on that `xn2` and adds the
/// result to `acts.x_mid` itself; pair with [`block_backward_from_dxn2`].
///
/// `acts.gate` / `up` / `act` are 1-element placeholders on this path — nothing
/// reads them, and the matching backward skips the dense MLP entirely.
pub fn block_forward_attn_only(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &BlockWeights,
    lora: &BlockLora,
    dims: &BlockDims,
    pos_host: &[f32],
    layer_idx: usize,
) -> HipResult<(GpuTensor, BlockActivations)> {
    block_forward_inner(gpu, x, w, lora, dims, pos_host, layer_idx, false)
}

#[allow(clippy::too_many_arguments)]
fn block_forward_inner(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &BlockWeights,
    lora: &BlockLora,
    dims: &BlockDims,
    pos_host: &[f32],
    layer_idx: usize,
    dense_mlp: bool,
) -> HipResult<(GpuTensor, BlockActivations)> {
    let (seq, h, inter) = (dims.seq, dims.h, dims.inter);
    let (qd, kvd, r) = (dims.q_dim(), dims.kv_dim(), dims.lora_rank);

    let xn1 = gpu.zeros(&[seq * h], DType::F32)?;
    let rinv1 = gpu.zeros(&[seq], DType::F32)?;
    rmsnorm_forward(gpu, x, w.norm1, &xn1, &rinv1, seq, h, dims.eps)?;

    // q = lora_q(xn1), k = lin(xn1), v = lora_v(xn1)
    let hq = gpu.zeros(&[seq * r], DType::F32)?;
    let loraq_s = gpu.zeros(&[seq * qd], DType::F32)?;
    let (q, attn_gate) = if w.attn_out_gate {
        // Refuse rather than silently ignore a trained q adapter — see the
        // note on BlockWeights::attn_out_gate.
        if gpu.download_f32(lora.bq)?.iter().any(|&v| v != 0.0) {
            return Err(hipfire_rdna::HipError::unsupported(
                "block: LoRA on q_proj with attn_out_gate (bq is nonzero)",
            ));
        }
        let qfull = gpu.zeros(&[seq * 2 * qd], DType::F32)?;
        linear_forward(gpu, &xn1, w.wq, &qfull, seq, h, 2 * qd)?;
        // Deinterleave [Q_h0, G_h0, Q_h1, G_h1, ...] on the host: it is
        // seq*2*q_dim floats beside a GEMM of seq*h*2*q_dim MACs, and the GPU
        // helper is shaped for decode rows, not a whole sequence.
        let f = gpu.download_f32(&qfull)?;
        gpu.free_tensor(qfull)?;
        let (mut qh, mut gh) = (vec![0.0f32; seq * qd], vec![0.0f32; seq * qd]);
        let hd = dims.head_dim;
        for t in 0..seq {
            for head in 0..dims.n_heads {
                for j in 0..hd {
                    let src = t * 2 * qd + head * 2 * hd + j;
                    qh[t * qd + head * hd + j] = f[src];
                    gh[t * qd + head * hd + j] = f[src + hd];
                }
            }
        }
        (
            gpu.upload_f32(&qh, &[seq * qd])?,
            Some(gpu.upload_f32(&gh, &[seq * qd])?),
        )
    } else {
        let q = gpu.zeros(&[seq * qd], DType::F32)?;
        lora_forward(
            gpu,
            &xn1,
            w.wq,
            lora.aq,
            lora.bq,
            &hq,
            &loraq_s,
            &q,
            seq,
            h,
            qd,
            r,
            dims.lora_scale,
        )?;
        (q, None)
    };
    let k = gpu.zeros(&[seq * kvd], DType::F32)?;
    linear_forward(gpu, &xn1, w.wk, &k, seq, h, kvd)?;
    let hv = gpu.zeros(&[seq * r], DType::F32)?;
    let lorav_s = gpu.zeros(&[seq * kvd], DType::F32)?;
    let v = gpu.zeros(&[seq * kvd], DType::F32)?;
    lora_forward(
        gpu,
        &xn1,
        w.wv,
        lora.av,
        lora.bv,
        &hv,
        &lorav_s,
        &v,
        seq,
        h,
        kvd,
        r,
        dims.lora_scale,
    )?;

    // QK-norm (qwen3): per-head rmsnorm over head_dim, before rope. Rows are
    // seq*n_heads because q is [seq, n_heads*head_dim] row-major, so each head
    // is one contiguous row.
    let (q, k, q_pre, k_pre, rinv_q, rinv_k) = match (w.q_norm, w.k_norm) {
        (Some(qn), Some(kn)) => {
            let qh = gpu.zeros(&[seq * qd], DType::F32)?;
            let rq = gpu.zeros(&[seq * dims.n_heads], DType::F32)?;
            rmsnorm_forward(
                gpu,
                &q,
                qn,
                &qh,
                &rq,
                seq * dims.n_heads,
                dims.head_dim,
                dims.eps,
            )?;
            let kh = gpu.zeros(&[seq * kvd], DType::F32)?;
            let rk = gpu.zeros(&[seq * dims.n_kv], DType::F32)?;
            rmsnorm_forward(
                gpu,
                &k,
                kn,
                &kh,
                &rk,
                seq * dims.n_kv,
                dims.head_dim,
                dims.eps,
            )?;
            (qh, kh, Some(q), Some(k), Some(rq), Some(rk))
        }
        (None, None) => (q, k, None, None, None, None),
        _ => {
            return Err(hipfire_rdna::HipError::unsupported(
                "block: q_norm and k_norm must both be set or both be absent",
            ))
        }
    };

    // rope(q), rope(k)
    let pos = gpu.upload_f32(pos_host, &[seq])?;
    let q_r = gpu.zeros(&[seq * qd], DType::F32)?;
    rope_forward(
        gpu,
        &q,
        &q_r,
        &pos,
        seq * dims.n_heads,
        dims.n_heads,
        dims.head_dim,
        dims.rope_base,
    )?;
    let k_r = gpu.zeros(&[seq * kvd], DType::F32)?;
    rope_forward(
        gpu,
        &k,
        &k_r,
        &pos,
        seq * dims.n_kv,
        dims.n_kv,
        dims.head_dim,
        dims.rope_base,
    )?;

    // KV-compression sim-noise (KVarN-4bit + CASK merge) on post-RoPE K and V,
    // forward-only (STE) — no-op unless HIPFIRE_KVNOISE=1. See crate::kv_noise.
    let (k_r, v) = crate::kv_noise::maybe_compress_kv(
        gpu,
        k_r,
        v,
        crate::kv_noise::cfg_from_env(),
        seq,
        kvd,
        dims.head_dim,
    )?;

    // Static rank-r latent-KV sim (forward-only STE) on post-RoPE K and V —
    // no-op unless calibrated projectors are installed. See crate::latent_kv.
    let (k_r, v) =
        crate::latent_kv::maybe_project(layer_idx, gpu, k_r, v, seq, kvd, dims.head_dim)?;

    // attention
    let p_all = gpu.zeros(&[dims.n_heads * seq * seq], DType::F32)?;
    let ctx = gpu.zeros(&[seq * qd], DType::F32)?;
    gqa_forward(
        gpu,
        &q_r,
        &k_r,
        &v,
        &p_all,
        &ctx,
        seq,
        dims.n_heads,
        dims.n_kv,
        dims.head_dim,
        dims.attn_scale(),
    )?;

    // attn_output_gate: ctx *= sigmoid(gate), before o_proj.
    let ctx_pre_gate = match &attn_gate {
        Some(g) => {
            let keep = gpu.zeros(&[seq * qd], DType::F32)?;
            gpu.memcpy_dtod_auto(&keep.buf, &ctx.buf, seq * qd * 4)?;
            gpu.sigmoid_mul_f32(&ctx, g)?;
            Some(keep)
        }
        None => None,
    };

    // o_proj + residual
    let attn = gpu.zeros(&[seq * h], DType::F32)?;
    linear_forward(gpu, &ctx, w.wo, &attn, seq, qd, h)?;
    let x_mid = gpu.zeros(&[seq * h], DType::F32)?;
    gpu.add_f32(x, &attn, &x_mid)?;

    // norm2 + mlp + residual
    let xn2 = gpu.zeros(&[seq * h], DType::F32)?;
    let rinv2 = gpu.zeros(&[seq], DType::F32)?;
    rmsnorm_forward(gpu, &x_mid, w.norm2, &xn2, &rinv2, seq, h, dims.eps)?;
    let n_mlp = if dense_mlp { seq * inter } else { 1 };
    let gate = gpu.zeros(&[n_mlp], DType::F32)?;
    let up = gpu.zeros(&[n_mlp], DType::F32)?;
    let act = gpu.zeros(&[n_mlp], DType::F32)?;
    let x_out = gpu.zeros(&[seq * h], DType::F32)?;
    let mlp = if dense_mlp {
        linear_forward(gpu, &xn2, w.wgate, &gate, seq, h, inter)?;
        linear_forward(gpu, &xn2, w.wup, &up, seq, h, inter)?;
        swiglu_forward(gpu, &gate, &up, &act, seq * inter)?;
        let mlp = gpu.zeros(&[seq * h], DType::F32)?;
        linear_forward(gpu, &act, w.wdown, &mlp, seq, inter, h)?;
        gpu.add_f32(&x_mid, &mlp, &x_out)?;
        Some(mlp)
    } else {
        // Attention only: the caller supplies the MLP. x_out is the residual so
        // far, and it adds its own MLP output to `acts.x_mid`.
        gpu.memcpy_dtod_auto(&x_out.buf, &x_mid.buf, seq * h * 4)?;
        None
    };

    // Return forward scratch the backward never reads (pre-rope q/k, lora
    // pre-scale, attn/mlp pre-residual) to the pool — GpuTensor has no Drop, so
    // without this each block_forward leaks ~5 MB and many-step training OOMs.
    for t in [loraq_s, q, k, lorav_s, attn] {
        gpu.free_tensor(t)?;
    }
    if let Some(m) = mlp {
        gpu.free_tensor(m)?;
    }

    Ok((
        x_out,
        BlockActivations {
            xn1,
            rinv1,
            q_pre,
            k_pre,
            rinv_q,
            rinv_k,
            attn_gate,
            ctx_pre_gate,
            hq,
            hv,
            q_r,
            k_r,
            v,
            p_all,
            ctx,
            x_mid,
            xn2,
            rinv2,
            gate,
            up,
            act,
            pos,
        },
    ))
}

/// Backward. `d_x_out` `[seq*h]` upstream; writes LoRA grads and returns the
/// input gradient `d_x` `[seq*h]`. `x` is the original block input (for norm1).
#[allow(clippy::too_many_arguments)]
/// Gradients of the FROZEN base block linears — only produced by
/// `block_backward_full` (for from-scratch training, e.g. the PFlash drafter).
/// Recovery FT leaves the base frozen and never asks for these.
pub struct BlockWeightGrad {
    pub dwq: GpuTensor,    // [q_dim, h]
    pub dwk: GpuTensor,    // [kv_dim, h]
    pub dwv: GpuTensor,    // [kv_dim, h]
    pub dwo: GpuTensor,    // [h, q_dim]
    pub dwgate: GpuTensor, // [inter, h]
    pub dwup: GpuTensor,   // [inter, h]
    pub dwdown: GpuTensor, // [h, inter]
}

/// Return one block's saved activations to the pool (GpuTensor has no Drop).
pub fn free_block_acts(gpu: &mut Gpu, b: BlockActivations) -> HipResult<()> {
    let BlockActivations {
        xn1,
        rinv1,
        q_pre,
        k_pre,
        rinv_q,
        rinv_k,
        attn_gate,
        ctx_pre_gate,
        hq,
        hv,
        q_r,
        k_r,
        v,
        p_all,
        ctx,
        x_mid,
        xn2,
        rinv2,
        gate,
        up,
        act,
        pos,
    } = b;
    for t in [
        xn1, rinv1, hq, hv, q_r, k_r, v, p_all, ctx, x_mid, xn2, rinv2, gate, up, act, pos,
    ] {
        gpu.free_tensor(t)?;
    }
    for t in [q_pre, k_pre, rinv_q, rinv_k, attn_gate, ctx_pre_gate]
        .into_iter()
        .flatten()
    {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

/// Recovery-FT backward: base frozen, returns LoRA + norm grads only.
/// Per-linear OUTPUT adjoints (∂ℓ/∂z) captured during backward, downloaded to host —
/// the input to a GuidedQuant Fisher weight `w[n] = mean_c adj[n,c]²`. Row-major, seq
/// rows: d_q [seq,qd], d_k/d_v [seq,kvd], d_attn (o_proj output) [seq,h], d_gate/d_up
/// [seq,inter].
pub struct BlockAdjoints {
    pub d_q: Vec<f32>,
    pub d_k: Vec<f32>,
    pub d_v: Vec<f32>,
    pub d_attn: Vec<f32>,
    pub d_gate: Vec<f32>,
    pub d_up: Vec<f32>,
    /// down_proj's output adjoint `[seq,h]`. NOT produced by the backward — it
    /// is the block's incoming `d_x`, because down_proj's output IS the block
    /// output pre-residual. Filled by the caller that owns `d_x` (see
    /// `model_guided_adjoints`); left empty by `block_backward_capture` itself,
    /// which never sees it.
    pub d_down: Vec<f32>,
}

pub fn block_backward(
    gpu: &mut Gpu,
    d_x_out: &GpuTensor,
    x: &GpuTensor,
    w: &BlockWeights,
    lora: &BlockLora,
    acts: &BlockActivations,
    dims: &BlockDims,
) -> HipResult<(GpuTensor, BlockLoraGrad)> {
    let (d_x, lora_g, _, _) =
        block_backward_inner(gpu, d_x_out, x, w, lora, acts, dims, false, false, None)?;
    Ok((d_x, lora_g))
}

/// From-scratch backward: also returns gradients for the base linears.
pub fn block_backward_full(
    gpu: &mut Gpu,
    d_x_out: &GpuTensor,
    x: &GpuTensor,
    w: &BlockWeights,
    lora: &BlockLora,
    acts: &BlockActivations,
    dims: &BlockDims,
) -> HipResult<(GpuTensor, BlockLoraGrad, BlockWeightGrad)> {
    let (d_x, lora_g, wg, _) =
        block_backward_inner(gpu, d_x_out, x, w, lora, acts, dims, true, false, None)?;
    Ok((d_x, lora_g, wg.expect("want_w=true ⇒ Some")))
}

/// Backward that additionally captures the per-linear output adjoints (for the
/// GuidedQuant Fisher-weighted Hessian). Opt-in; existing paths are unchanged.
pub fn block_backward_capture(
    gpu: &mut Gpu,
    d_x_out: &GpuTensor,
    x: &GpuTensor,
    w: &BlockWeights,
    lora: &BlockLora,
    acts: &BlockActivations,
    dims: &BlockDims,
) -> HipResult<(GpuTensor, BlockLoraGrad, BlockAdjoints)> {
    let (d_x, lora_g, _, adj) =
        block_backward_inner(gpu, d_x_out, x, w, lora, acts, dims, false, true, None)?;
    Ok((d_x, lora_g, adj.expect("want_capture=true ⇒ Some")))
}

/// One routed-MoE transformer block: the dense block's attention, with the
/// SwiGLU MLP replaced by a router and `top_k` experts.
///
/// Built on [`block_forward_attn_only`] rather than a second copy of the
/// attention sequence, so a fix to rope/GQA/norm reaches both block kinds. The
/// composition is just the residual identity `x_out = x_mid + moe(xn2)`.
pub fn moe_block_forward(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &BlockWeights,
    moe_w: &crate::ops::moe::MoeWeights,
    lora: &BlockLora,
    dims: &BlockDims,
    moe_dims: &crate::ops::moe::MoeDims,
    pos_host: &[f32],
    layer_idx: usize,
) -> HipResult<(GpuTensor, BlockActivations, crate::ops::moe::MoeActivations)> {
    let (x_mid_only, acts) = block_forward_attn_only(gpu, x, w, lora, dims, pos_host, layer_idx)?;
    let (moe_out, moe_acts) = crate::ops::moe::moe_forward(gpu, &acts.xn2, moe_w, moe_dims)?;
    let x_out = gpu.zeros(&[dims.seq * dims.h], DType::F32)?;
    gpu.add_f32(&acts.x_mid, &moe_out, &x_out)?;
    gpu.free_tensor(moe_out)?;
    gpu.free_tensor(x_mid_only)?;
    Ok((x_out, acts, moe_acts))
}

/// Backward for [`moe_block_forward`], capturing both the attention-side
/// adjoints and the per-expert ones.
///
/// `x_out = x_mid + moe(xn2)` gives `d_moe_out = d_x_out` directly, and
/// `moe_backward` returns the gradient w.r.t. its own input — which IS `d_xn2`.
#[allow(clippy::too_many_arguments)]
pub fn moe_block_backward_capture(
    gpu: &mut Gpu,
    d_x_out: &GpuTensor,
    x: &GpuTensor,
    w: &BlockWeights,
    moe_w: &crate::ops::moe::MoeWeights,
    lora: &BlockLora,
    acts: &BlockActivations,
    moe_acts: &crate::ops::moe::MoeActivations,
    dims: &BlockDims,
    moe_dims: &crate::ops::moe::MoeDims,
) -> HipResult<(GpuTensor, BlockAdjoints, crate::ops::moe::MoeAdjoints)> {
    let (d_xn2, moe_adj) = crate::ops::moe::moe_backward(gpu, d_x_out, moe_w, moe_acts, moe_dims)?;
    let (d_x, _lora_g, adj) =
        block_backward_from_dxn2(gpu, d_x_out, &d_xn2, x, w, lora, acts, dims)?;
    gpu.free_tensor(d_xn2)?;
    Ok((d_x, adj, moe_adj))
}

/// Backward for a block whose MLP was run by the CALLER, taking that MLP's
/// input gradient as `d_xn2`.
///
/// Pairs with [`block_forward_attn_only`]. The dense MLP chain is skipped
/// entirely; `d_xn2` is fed straight to `norm2`'s backward and on into the
/// attention half, which is why it must be the real gradient and not a
/// placeholder — see the note in `block_backward_inner`.
#[allow(clippy::too_many_arguments)]
pub fn block_backward_from_dxn2(
    gpu: &mut Gpu,
    d_x_out: &GpuTensor,
    d_xn2: &GpuTensor,
    x: &GpuTensor,
    w: &BlockWeights,
    lora: &BlockLora,
    acts: &BlockActivations,
    dims: &BlockDims,
) -> HipResult<(GpuTensor, BlockLoraGrad, BlockAdjoints)> {
    let (d_x, lora_g, _, adj) = block_backward_inner(
        gpu,
        d_x_out,
        x,
        w,
        lora,
        acts,
        dims,
        false,
        true,
        Some(d_xn2),
    )?;
    Ok((d_x, lora_g, adj.expect("want_capture=true ⇒ Some")))
}

#[allow(clippy::too_many_arguments)]
fn block_backward_inner(
    gpu: &mut Gpu,
    d_x_out: &GpuTensor,
    x: &GpuTensor,
    w: &BlockWeights,
    lora: &BlockLora,
    acts: &BlockActivations,
    dims: &BlockDims,
    want_w: bool,
    want_capture: bool,
    d_xn2_ext: Option<&GpuTensor>,
) -> HipResult<(
    GpuTensor,
    BlockLoraGrad,
    Option<BlockWeightGrad>,
    Option<BlockAdjoints>,
)> {
    let (seq, h, inter) = (dims.seq, dims.h, dims.inter);
    let (qd, kvd, r) = (dims.q_dim(), dims.kv_dim(), dims.lora_rank);
    let dnorm1 = gpu.zeros(&[h], DType::F32)?; // trainable RMSNorm grads
    let dnorm2 = gpu.zeros(&[h], DType::F32)?;

    // ── MLP branch ──────────────────────────────────────────────────────────
    // x_out = x_mid + mlp  ⇒ d_mlp = d_x_out, and d_x_mid starts = d_x_out.
    //
    // With `d_xn2_ext` the caller ran its own MLP (a routed MoE) and already
    // has d_xn2 — the dense chain is skipped and its value used directly. Note
    // the MLP is NOT a separable addend: d_xn2 feeds rmsnorm_backward below and
    // thence the attention half, so passing zero here would silently corrupt
    // the attention gradient rather than merely omit the MLP.
    let (d_xn2, d_act, d_gate, d_up) = match d_xn2_ext {
        Some(_) => (
            gpu.zeros(&[1], DType::F32)?,
            gpu.zeros(&[1], DType::F32)?,
            gpu.zeros(&[1], DType::F32)?,
            gpu.zeros(&[1], DType::F32)?,
        ),
        None => {
            let d_act = gpu.zeros(&[seq * inter], DType::F32)?;
            linear_backward_x(gpu, d_x_out, w.wdown, &d_act, seq, inter, h, false)?;
            let d_gate = gpu.zeros(&[seq * inter], DType::F32)?;
            let d_up = gpu.zeros(&[seq * inter], DType::F32)?;
            swiglu_backward(
                gpu,
                &d_act,
                &acts.gate,
                &acts.up,
                &d_gate,
                &d_up,
                seq * inter,
            )?;
            let d_xn2 = gpu.zeros(&[seq * h], DType::F32)?;
            linear_backward_x(gpu, &d_gate, w.wgate, &d_xn2, seq, h, inter, false)?;
            linear_backward_x(gpu, &d_up, w.wup, &d_xn2, seq, h, inter, true)?;
            (d_xn2, d_act, d_gate, d_up)
        }
    };
    let d_xn2_use = d_xn2_ext.unwrap_or(&d_xn2);
    // norm2 backward → adds into d_x_mid; dnorm2 is the trainable weight grad
    let d_x_mid = gpu.zeros(&[seq * h], DType::F32)?;
    gpu.memcpy_dtod_auto(&d_x_mid.buf, &d_x_out.buf, seq * h * 4)?; // residual
    let d_xmid_norm = gpu.zeros(&[seq * h], DType::F32)?;
    rmsnorm_backward(
        gpu,
        d_xn2_use,
        &acts.x_mid,
        w.norm2,
        &acts.rinv2,
        &d_xmid_norm,
        &dnorm2,
        seq,
        h,
    )?;
    gpu.add_inplace_f32(&d_x_mid, &d_xmid_norm)?;

    // ── Attention branch ──────────────────────────────────────────────────────
    // x_mid = x + attn ⇒ d_attn = d_x_mid, d_x starts = d_x_mid.
    let d_ctx = gpu.zeros(&[seq * qd], DType::F32)?;
    linear_backward_x(gpu, &d_x_mid, w.wo, &d_ctx, seq, qd, h, false)?;

    // attn_output_gate backward. ctx_gated = ctx * sigmoid(g), so the adjoint
    // splits: gqa sees d_ctx * sigmoid(g), and the gate collects
    // d_ctx * ctx * sigmoid'(g). Done on the host beside the deinterleave the
    // forward already does there.
    let d_gate_host = match (&acts.attn_gate, &acts.ctx_pre_gate) {
        (Some(g), Some(ctx_pre)) => {
            let (gh, ch) = (gpu.download_f32(g)?, gpu.download_f32(ctx_pre)?);
            let mut dc = gpu.download_f32(&d_ctx)?;
            let mut dg = vec![0.0f32; seq * qd];
            for i in 0..seq * qd {
                let sg = 1.0 / (1.0 + (-gh[i]).exp());
                dg[i] = dc[i] * ch[i] * sg * (1.0 - sg);
                dc[i] *= sg;
            }
            let bytes: Vec<u8> = dc.iter().flat_map(|v| v.to_ne_bytes()).collect();
            gpu.memcpy_htod_auto(&d_ctx.buf, &bytes)?;
            Some(dg)
        }
        _ => None,
    };
    let d_q_r = gpu.zeros(&[seq * qd], DType::F32)?;
    let d_k_r = gpu.zeros(&[seq * kvd], DType::F32)?;
    let d_v = gpu.zeros(&[seq * kvd], DType::F32)?;
    gqa_backward(
        gpu,
        &d_ctx,
        &acts.q_r,
        &acts.k_r,
        &acts.v,
        &acts.p_all,
        &d_q_r,
        &d_k_r,
        &d_v,
        seq,
        dims.n_heads,
        dims.n_kv,
        dims.head_dim,
        dims.attn_scale(),
    )?;
    // rope backward
    let d_q = gpu.zeros(&[seq * qd], DType::F32)?;
    rope_backward(
        gpu,
        &d_q_r,
        &d_q,
        &acts.pos,
        seq * dims.n_heads,
        dims.n_heads,
        dims.head_dim,
        dims.rope_base,
    )?;
    let d_k = gpu.zeros(&[seq * kvd], DType::F32)?;
    rope_backward(
        gpu,
        &d_k_r,
        &d_k,
        &acts.pos,
        seq * dims.n_kv,
        dims.n_kv,
        dims.head_dim,
        dims.rope_base,
    )?;

    // QK-norm backward: rope's input was the NORMED q/k, so its adjoint has to
    // pass back through the per-head rmsnorm before reaching the projections.
    // Skipping this leaves d_x wrong by the norm's Jacobian — a scale-ish error
    // that looks plausible and trains badly.
    let (d_q, d_k) = match (w.q_norm, w.k_norm, &acts.q_pre, &acts.k_pre) {
        (Some(qn), Some(kn), Some(q_pre), Some(k_pre)) => {
            let dwq = gpu.zeros(&[dims.head_dim], DType::F32)?;
            let d_q_pre = gpu.zeros(&[seq * qd], DType::F32)?;
            rmsnorm_backward(
                gpu,
                &d_q,
                q_pre,
                qn,
                acts.rinv_q.as_ref().expect("rinv_q with q_norm"),
                &d_q_pre,
                &dwq,
                seq * dims.n_heads,
                dims.head_dim,
            )?;
            let d_k_pre = gpu.zeros(&[seq * kvd], DType::F32)?;
            rmsnorm_backward(
                gpu,
                &d_k,
                k_pre,
                kn,
                acts.rinv_k.as_ref().expect("rinv_k with k_norm"),
                &d_k_pre,
                &dwq,
                seq * dims.n_kv,
                dims.head_dim,
            )?;
            gpu.free_tensor(dwq)?;
            gpu.free_tensor(d_q)?;
            gpu.free_tensor(d_k)?;
            (d_q_pre, d_k_pre)
        }
        _ => (d_q, d_k),
    };

    // q/k/v projection backward → accumulate into d_xn1, produce LoRA grads
    let d_xn1 = gpu.zeros(&[seq * h], DType::F32)?;
    let daq = gpu.zeros(&[r * h], DType::F32)?;
    let dbq = gpu.zeros(&[qd * r], DType::F32)?;
    let dav = gpu.zeros(&[r * h], DType::F32)?;
    let dbv = gpu.zeros(&[kvd * r], DType::F32)?;
    let dyl_q = gpu.zeros(&[seq * qd], DType::F32)?;
    let dh_q = gpu.zeros(&[seq * r], DType::F32)?;
    if let Some(dg) = &d_gate_host {
        // Re-interleave [dQ_h0, dG_h0, dQ_h1, ...] to match wq's output layout.
        // daq/dbq stay zero: LoRA is refused on a gated layer in the forward.
        let dqh = gpu.download_f32(&d_q)?;
        let hd = dims.head_dim;
        let mut full = vec![0.0f32; seq * 2 * qd];
        for t in 0..seq {
            for head in 0..dims.n_heads {
                for j in 0..hd {
                    let dst = t * 2 * qd + head * 2 * hd + j;
                    full[dst] = dqh[t * qd + head * hd + j];
                    full[dst + hd] = dg[t * qd + head * hd + j];
                }
            }
        }
        let d_full = gpu.upload_f32(&full, &[seq * 2 * qd])?;
        linear_backward_x(gpu, &d_full, w.wq, &d_xn1, seq, h, 2 * qd, true)?;
        gpu.free_tensor(d_full)?;
    } else {
        lora_backward(
            gpu,
            &d_q,
            &acts.xn1,
            w.wq,
            lora.aq,
            lora.bq,
            &acts.hq,
            &dyl_q,
            &dh_q,
            &daq,
            &dbq,
            &d_xn1,
            seq,
            h,
            qd,
            r,
            dims.lora_scale,
            true,
        )?;
    }
    linear_backward_x(gpu, &d_k, w.wk, &d_xn1, seq, h, kvd, true)?;
    let dyl_v = gpu.zeros(&[seq * kvd], DType::F32)?;
    let dh_v = gpu.zeros(&[seq * r], DType::F32)?;
    lora_backward(
        gpu,
        &d_v,
        &acts.xn1,
        w.wv,
        lora.av,
        lora.bv,
        &acts.hv,
        &dyl_v,
        &dh_v,
        &dav,
        &dbv,
        &d_xn1,
        seq,
        h,
        kvd,
        r,
        dims.lora_scale,
        true,
    )?;

    // norm1 backward → adds into d_x; dnorm1 is the trainable weight grad
    let d_x = gpu.zeros(&[seq * h], DType::F32)?;
    gpu.memcpy_dtod_auto(&d_x.buf, &d_x_mid.buf, seq * h * 4)?; // residual
    let d_x_norm = gpu.zeros(&[seq * h], DType::F32)?;
    rmsnorm_backward(
        gpu,
        &d_xn1,
        x,
        w.norm1,
        &acts.rinv1,
        &d_x_norm,
        &dnorm1,
        seq,
        h,
    )?;
    gpu.add_inplace_f32(&d_x, &d_x_norm)?;

    // Base-linear weight grads (from-scratch training only). dw = dyᵀ·x; every
    // (dy, input) pair below is already materialised above.
    let wg = if want_w {
        let dwq = gpu.zeros(&[qd * h], DType::F32)?;
        let dwk = gpu.zeros(&[kvd * h], DType::F32)?;
        let dwv = gpu.zeros(&[kvd * h], DType::F32)?;
        let dwo = gpu.zeros(&[h * qd], DType::F32)?;
        let dwgate = gpu.zeros(&[inter * h], DType::F32)?;
        let dwup = gpu.zeros(&[inter * h], DType::F32)?;
        let dwdown = gpu.zeros(&[h * inter], DType::F32)?;
        // attention projections (input = acts.xn1)
        linear_backward_w(gpu, &d_q, &acts.xn1, &dwq, seq, h, qd, false)?;
        linear_backward_w(gpu, &d_k, &acts.xn1, &dwk, seq, h, kvd, false)?;
        linear_backward_w(gpu, &d_v, &acts.xn1, &dwv, seq, h, kvd, false)?;
        // output projection (input = acts.ctx)
        linear_backward_w(gpu, &d_x_mid, &acts.ctx, &dwo, seq, qd, h, false)?;
        // MLP (gate/up input = acts.xn2; down input = acts.act)
        linear_backward_w(gpu, &d_gate, &acts.xn2, &dwgate, seq, h, inter, false)?;
        linear_backward_w(gpu, &d_up, &acts.xn2, &dwup, seq, h, inter, false)?;
        linear_backward_w(gpu, d_x_out, &acts.act, &dwdown, seq, inter, h, false)?;
        Some(BlockWeightGrad {
            dwq,
            dwk,
            dwv,
            dwo,
            dwgate,
            dwup,
            dwdown,
        })
    } else {
        None
    };

    // GuidedQuant: capture the six output adjoints (still alive) to host before free.
    let adjoints = if want_capture {
        Some(BlockAdjoints {
            // Filled by the caller that owns d_x; the backward never sees it.
            d_down: Vec::new(),
            d_q: gpu.download_f32(&d_q)?,
            d_k: gpu.download_f32(&d_k)?,
            d_v: gpu.download_f32(&d_v)?,
            d_attn: gpu.download_f32(&d_x_mid)?,
            d_gate: gpu.download_f32(&d_gate)?,
            d_up: gpu.download_f32(&d_up)?,
        })
    } else {
        None
    };

    // Return internal temporaries to the pool — GpuTensor has no Drop, so without
    // this the per-step training graph climbs ~50 MB/layer and OOMs. Only the
    // returned grads (d_x, BlockLoraGrad, BlockWeightGrad) survive.
    for t in [
        d_act,
        d_gate,
        d_up,
        d_xn2,
        d_x_mid,
        d_xmid_norm,
        d_ctx,
        d_q_r,
        d_k_r,
        d_v,
        d_q,
        d_k,
        d_xn1,
        d_x_norm,
        dyl_q,
        dh_q,
        dyl_v,
        dh_v,
    ] {
        gpu.free_tensor(t)?;
    }

    Ok((
        d_x,
        BlockLoraGrad {
            daq,
            dbq,
            dav,
            dbv,
            dnorm1,
            dnorm2,
        },
        wg,
        adjoints,
    ))
}
