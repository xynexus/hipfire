// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! The vision tower on the GPU. Mirrors [`crate::vision`], which
//! `tests/reference_oracle.rs` differences against the pinned upstream tower.
//!
//! Attention reuses `vit_attention_f32` unchanged: it already reads a fused
//! `[N, 3*hidden]` qkv with q/k/v at `head*head_dim`, `hidden + head*head_dim` and
//! `2*hidden + head*head_dim` — exactly this model's layout — and it is
//! bidirectional, which is what a ViT wants.

use crate::config::VisionConfig;
use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

pub struct VisionBlockWeights {
    pub norm1_w: GpuTensor,
    pub norm1_b: GpuTensor,
    pub norm2_w: GpuTensor,
    pub norm2_b: GpuTensor,
    pub qkv_w: GpuTensor,
    pub qkv_b: GpuTensor,
    pub proj_w: GpuTensor,
    pub proj_b: GpuTensor,
    pub fc1_w: GpuTensor,
    pub fc1_b: GpuTensor,
    pub fc2_w: GpuTensor,
    pub fc2_b: GpuTensor,
}

pub struct VisionScratch {
    normed: GpuTensor,
    qkv: GpuTensor,
    ctx: GpuTensor,
    proj: GpuTensor,
    inter: GpuTensor,
    mlp_out: GpuTensor,
}

impl VisionScratch {
    pub fn new(gpu: &mut Gpu, v: &VisionConfig, max_tokens: usize) -> HipResult<Self> {
        let z = |g: &mut Gpu, n: usize| g.zeros(&[n], DType::F32);
        Ok(Self {
            normed: z(gpu, max_tokens * v.hidden)?,
            qkv: z(gpu, max_tokens * 3 * v.hidden)?,
            ctx: z(gpu, max_tokens * v.hidden)?,
            proj: z(gpu, max_tokens * v.hidden)?,
            inter: z(gpu, max_tokens * v.intermediate)?,
            mlp_out: z(gpu, max_tokens * v.hidden)?,
        })
    }
}

/// One pre-norm block: `x + attn(norm1(x))`, then `x + mlp(norm2(x))`.
///
/// `x` is updated in place, which is what makes the two residual adds free.
pub fn vision_block(
    gpu: &mut Gpu,
    v: &VisionConfig,
    w: &VisionBlockWeights,
    s: &mut VisionScratch,
    x: &GpuTensor,
    n_tok: usize,
    cos: &GpuTensor,
    sin: &GpuTensor,
    eps: f32,
) -> HipResult<()> {
    let (h, nh, hd) = (v.hidden, v.n_heads, v.head_dim());

    gpu.layernorm_batched(x, &w.norm1_w, &w.norm1_b, &s.normed, n_tok, h, eps)?;
    gpu.vision_linear_bias(
        &s.normed,
        &w.qkv_w,
        Some(&w.qkv_b),
        &s.qkv,
        n_tok as i32,
        h as i32,
        (3 * h) as i32,
    )?;
    gpu.vision_rope_qkv(
        &s.qkv,
        cos,
        sin,
        n_tok as i32,
        h as i32,
        nh as i32,
        hd as i32,
    )?;
    gpu.vit_attention_f32(&s.qkv, &s.ctx, n_tok, h, nh, hd)?;
    gpu.vision_linear_bias(
        &s.ctx,
        &w.proj_w,
        Some(&w.proj_b),
        &s.proj,
        n_tok as i32,
        h as i32,
        h as i32,
    )?;
    gpu.add_inplace_f32(x, &s.proj)?;

    gpu.layernorm_batched(x, &w.norm2_w, &w.norm2_b, &s.normed, n_tok, h, eps)?;
    gpu.vision_linear_bias(
        &s.normed,
        &w.fc1_w,
        Some(&w.fc1_b),
        &s.inter,
        n_tok as i32,
        h as i32,
        v.intermediate as i32,
    )?;
    // The block MLP uses the TANH gelu; the merger uses the exact erf one.
    gpu.gelu_tanh_f32(&s.inter, &s.inter, n_tok * v.intermediate)?;
    gpu.vision_linear_bias(
        &s.inter,
        &w.fc2_w,
        Some(&w.fc2_b),
        &s.mlp_out,
        n_tok as i32,
        v.intermediate as i32,
        h as i32,
    )?;
    gpu.add_inplace_f32(x, &s.mlp_out)
}

pub struct MergerWeights {
    pub norm_w: GpuTensor,
    pub norm_b: GpuTensor,
    pub fc1_w: GpuTensor,
    pub fc1_b: GpuTensor,
    pub fc2_w: GpuTensor,
    pub fc2_b: GpuTensor,
}

/// Patch merger. Normalises at the UNMERGED width, then folds `merge_unit`
/// patches into one token — so the reshape is implicit in the row count handed to
/// the first linear, not a separate step.
pub fn vision_merger(
    gpu: &mut Gpu,
    v: &VisionConfig,
    w: &MergerWeights,
    x: &GpuTensor,
    n_tok: usize,
    out: &GpuTensor,
    eps: f32,
) -> HipResult<()> {
    let wide = v.hidden * v.merge_unit();
    let merged = n_tok / v.merge_unit();
    let normed = gpu.zeros(&[n_tok * v.hidden], DType::F32)?;
    gpu.layernorm_batched(x, &w.norm_w, &w.norm_b, &normed, n_tok, v.hidden, eps)?;
    let t = gpu.zeros(&[merged * wide], DType::F32)?;
    gpu.vision_linear_bias(
        &normed,
        &w.fc1_w,
        Some(&w.fc1_b),
        &t,
        merged as i32,
        wide as i32,
        wide as i32,
    )?;
    gpu.vision_gelu_erf(&t, &t, (merged * wide) as i32)?;
    gpu.vision_linear_bias(
        &t,
        &w.fc2_w,
        Some(&w.fc2_b),
        out,
        merged as i32,
        wide as i32,
        v.out_hidden as i32,
    )
}
