// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! The PLE block on the GPU. Mirrors [`crate::ple::PleLayer`].
//!
//! The n-gram LOOKUP is the caller's job: the table is 102 GB in the shipped model
//! and lives on disk behind [`crate::ngram_store`], so it is not something this
//! step can own. What arrives here is the already-gathered `[embed_dim]` row.

use crate::config::Qwen4ExpConfig;
use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

pub struct PleWeights {
    /// `[hc_count * hidden, embed_dim]`
    pub key_proj: GpuTensor,
    /// `[hidden, embed_dim]`
    pub value_proj: GpuTensor,
    pub norm_key: GpuTensor,
    pub norm_query: GpuTensor,
    pub norm_conv: GpuTensor,
    /// `[hc_count * hidden, kernel]`, depthwise
    pub conv_weight: GpuTensor,
}

pub struct PleScratch {
    key: GpuTensor,
    key_normed: GpuTensor,
    value: GpuTensor,
    query: GpuTensor,
    gated: GpuTensor,
    normed: GpuTensor,
    conv: GpuTensor,
    /// `[hc_count * hidden * (kernel - 1) * dilation]` — the dilated tap history.
    pub conv_state: GpuTensor,
}

impl PleScratch {
    pub fn new(gpu: &mut Gpu, cfg: &Qwen4ExpConfig) -> HipResult<Self> {
        let n = cfg
            .ngram
            .as_ref()
            .expect("PLE scratch needs an ngram config");
        let width = cfg.gated_residual.count * cfg.hidden;
        let state_len = (n.conv_kernel - 1) * n.ngram_size;
        let z = |g: &mut Gpu, k: usize| g.zeros(&[k], DType::F32);
        Ok(Self {
            key: z(gpu, width)?,
            key_normed: z(gpu, width)?,
            value: z(gpu, cfg.hidden)?,
            query: z(gpu, width)?,
            gated: z(gpu, width)?,
            normed: z(gpu, width)?,
            conv: z(gpu, width)?,
            conv_state: z(gpu, width * state_len)?,
        })
    }
}

/// One token. `hidden_wide` is `[hc_count * hidden]`, `ngram_embed` is
/// `[embed_dim]`, `out` is `[hc_count * hidden]`.
pub fn ple_step(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    w: &PleWeights,
    s: &mut PleScratch,
    hidden_wide: &GpuTensor,
    ngram_embed: &GpuTensor,
    out: &GpuTensor,
) -> HipResult<()> {
    let n = cfg.ngram.as_ref().expect("ple_step needs an ngram config");
    let hc = cfg.gated_residual.count;
    let width = hc * cfg.hidden;
    let (h32, hc32) = (cfg.hidden as i32, hc as i32);

    gpu.gemv_f32(&w.key_proj, ngram_embed, &s.key)?;
    gpu.hc_grouped_rmsnorm(
        &s.key,
        &w.norm_key,
        &s.key_normed,
        h32,
        hc32,
        cfg.rms_norm_eps,
    )?;
    gpu.gemv_f32(&w.value_proj, ngram_embed, &s.value)?;
    gpu.hc_grouped_rmsnorm(
        hidden_wide,
        &w.norm_query,
        &s.query,
        h32,
        hc32,
        cfg.rms_norm_eps,
    )?;
    gpu.ple_stream_gate(&s.key_normed, &s.query, &s.value, &s.gated, hc32, h32)?;

    // The conv reads the NORMED gated value; the residual adds the UN-normed one.
    gpu.hc_grouped_rmsnorm(
        &s.gated,
        &w.norm_conv,
        &s.normed,
        h32,
        hc32,
        cfg.rms_norm_eps,
    )?;
    gpu.ple_dilated_conv_silu(
        &s.conv_state,
        &s.normed,
        &w.conv_weight,
        &s.conv,
        width as i32,
        ((n.conv_kernel - 1) * n.ngram_size) as i32,
        n.conv_kernel as i32,
        n.ngram_size as i32,
    )?;
    gpu.add_f32(&s.gated, &s.conv, out)
}
