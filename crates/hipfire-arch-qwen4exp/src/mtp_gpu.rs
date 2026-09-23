// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! The multi-token-prediction head on the GPU — the drafter.
//!
//! Mirrors [`crate::mtp`] (CPU), which is the reference this is differenced
//! against. The head is structurally ONE trunk sparse-attention layer plus the
//! stream mixer, so every kernel it needs already exists: `hc_read`/`hc_write`
//! for the gated residual, `qsa_decode_step` for attention, `moe_forward` for
//! the routed experts. The only piece with no trunk equivalent is
//! [`fuse_inputs`], which builds the head's wide input from the trunk's wide
//! residual plus the next token's embedding.
//!
//! # Why this is worth running
//!
//! Measured on the shipped 180B (`examples/mtp_probe`, self-generated
//! continuation): **56.2% token acceptance** against a do-nothing baseline of
//! 0.0%, with the head quantised to `oq8` rather than training precision. See
//! `docs/todo/2026-09-24-qwen4exp-mtp-drafter-validated.md` — including why an
//! earlier synthetic-prompt run read −0.0226 and said the opposite.
//!
//! # The conventions, pinned by measurement not by shape
//!
//! * `fc_embedding`'s narrow output is BROADCAST to every stream
//!   ([`Fusion::BroadcastAllStreams`]), which beat stream-0-only by ~0.19.
//! * The head consumes `h_t` + `emb(x_{t+1})` and predicts `x_{t+2}` — the
//!   documented MTP convention, and the best of the six index variants tried.

use crate::attn_gpu::{qsa_decode_step, QsaCache, QsaScratch, QsaWeights};
use crate::config::Qwen4ExpConfig;
use crate::hc_gpu::{hc_read, hc_write, HcScratch, HcWeights};
use crate::moe_gpu::{moe_forward, MoeScratch, MoeWeights};
use crate::mtp::Fusion;
use crate::trunk_gpu::TensorReader;
use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};
use hipfire_runtime::weights::{weight_gemv, WeightTensor};

/// The head's weights: one decoder layer, its two fusion projections, and the
/// stream mixer. No embedding or `lm_head` — `mtp_use_dedicated_embeddings` is
/// false in the shipped model, so both are the trunk's.
pub struct MtpWeightsGpu {
    pub pre_fc_norm_hidden: GpuTensor,
    pub pre_fc_norm_embedding: GpuTensor,
    pub fc_hidden: WeightTensor,
    pub fc_embedding: WeightTensor,
    pub attn_hc: HcWeights,
    pub mlp_hc: HcWeights,
    pub qsa: QsaWeights,
    pub moe: MoeWeights,
    pub mixer: HcWeights,
}

/// Scratch for one head step. Separate from the trunk's so a draft never
/// scribbles on the state the verify pass still needs.
pub struct MtpScratchGpu {
    hc: HcScratch,
    qsa: QsaScratch,
    moe: MoeScratch,
    /// The head's own wide stream, `[hc_count * hidden]`.
    wide: GpuTensor,
    mixed: GpuTensor,
    block_out: GpuTensor,
    /// Per-stream normalised hidden, and the normalised embedding.
    normed_wide: GpuTensor,
    normed_emb: GpuTensor,
    proj_emb: GpuTensor,
    proj_hidden: GpuTensor,
    /// The collapsed output — same shape the trunk's mixer emits, so the
    /// trunk's `lm_head` consumes it unchanged.
    pub collapsed: GpuTensor,
}

impl MtpScratchGpu {
    pub fn new(gpu: &mut Gpu, cfg: &Qwen4ExpConfig, max_seq: usize) -> HipResult<Self> {
        let (hidden, hc) = (cfg.hidden, cfg.gated_residual.count);
        let width = hc * hidden;
        Ok(Self {
            hc: HcScratch::new(gpu, cfg)?,
            qsa: QsaScratch::new(gpu, cfg, max_seq)?,
            moe: MoeScratch::new(gpu, cfg)?,
            wide: gpu.zeros(&[width], DType::F32)?,
            mixed: gpu.zeros(&[hidden], DType::F32)?,
            block_out: gpu.zeros(&[hidden], DType::F32)?,
            normed_wide: gpu.zeros(&[width], DType::F32)?,
            normed_emb: gpu.zeros(&[hidden], DType::F32)?,
            proj_emb: gpu.zeros(&[hidden], DType::F32)?,
            proj_hidden: gpu.zeros(&[hidden], DType::F32)?,
            collapsed: gpu.zeros(&[hidden], DType::F32)?,
        })
    }
}

/// Build the head's wide input: `fc_hidden` per stream, `fc_embedding` once,
/// the narrow result broadcast across streams.
///
/// `fc_hidden` is `[hidden, hidden]` and cannot consume a wide vector in one go,
/// which is what pins it to a per-stream application; `pre_fc_norm_hidden` is
/// `[hc_count * hidden]`, which is what pins the input to the WIDE residual
/// rather than the collapsed output. See [`crate::mtp`] for the full argument.
fn fuse_inputs(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    w: &MtpWeightsGpu,
    s: &mut MtpScratchGpu,
    trunk_wide: &GpuTensor,
    embedding: &GpuTensor,
    fusion: Fusion,
) -> HipResult<()> {
    let (hidden, hc) = (cfg.hidden, cfg.gated_residual.count);

    // Grouped over streams for the hidden side; a single group for the narrow
    // embedding — the same kernel, hc=1.
    gpu.hc_grouped_rmsnorm(
        trunk_wide,
        &w.pre_fc_norm_hidden,
        &s.normed_wide,
        hidden as i32,
        hc as i32,
        cfg.rms_norm_eps,
    )?;
    gpu.hc_grouped_rmsnorm(
        embedding,
        &w.pre_fc_norm_embedding,
        &s.normed_emb,
        hidden as i32,
        1,
        cfg.rms_norm_eps,
    )?;
    weight_gemv(gpu, &w.fc_embedding, &s.normed_emb, &s.proj_emb)?;

    for st in 0..hc {
        let src = s.normed_wide.sub_offset(st * hidden, hidden);
        let dst = s.wide.sub_offset(st * hidden, hidden);
        weight_gemv(gpu, &w.fc_hidden, &src, &s.proj_hidden)?;
        let take_embed = match fusion {
            Fusion::BroadcastAllStreams => true,
            Fusion::Stream0Only => st == 0,
        };
        gpu.memcpy_dtod_auto(&dst.buf, &s.proj_hidden.buf, hidden * 4)?;
        if take_embed {
            gpu.add_inplace_f32(&dst, &s.proj_emb)?;
        }
    }
    Ok(())
}

/// One draft step: the trunk's wide residual at `t` plus `emb(x_{t+1})` in,
/// a collapsed hidden out. The caller applies the trunk's `lm_head`.
///
/// `pos` and `visible` describe the head's OWN attention timeline, which runs
/// alongside the trunk's rather than sharing it — the head has its own KV cache
/// for exactly this reason.
#[allow(clippy::too_many_arguments)]
pub fn mtp_decode_step(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    w: &MtpWeightsGpu,
    cache: &mut QsaCache,
    s: &mut MtpScratchGpu,
    trunk_wide: &GpuTensor,
    embedding: &GpuTensor,
    pos: usize,
    fusion: Fusion,
) -> HipResult<()> {
    fuse_inputs(gpu, cfg, w, s, trunk_wide, embedding, fusion)?;
    let visible: Vec<usize> = (0..=pos).collect();

    // Attention half, then the MoE half — the same residual shape both times,
    // exactly as a trunk layer.
    hc_read(gpu, cfg, &w.attn_hc, &mut s.hc, &s.wide, &s.mixed)?;
    qsa_decode_step(
        gpu,
        cfg,
        &w.qsa,
        &mut s.qsa,
        cache,
        &s.mixed,
        pos,
        &visible,
        &s.block_out,
    )?;
    hc_write(gpu, cfg, &s.hc, &s.wide, &s.block_out)?;

    hc_read(gpu, cfg, &w.mlp_hc, &mut s.hc, &s.wide, &s.mixed)?;
    moe_forward(gpu, cfg, &w.moe, &mut s.moe, &s.mixed, &s.block_out)?;
    hc_write(gpu, cfg, &s.hc, &s.wide, &s.block_out)?;

    // The mixer's own norm is the last normalisation, as in the trunk.
    hc_read(gpu, cfg, &w.mixer, &mut s.hc, &s.wide, &s.collapsed)
}

impl MtpWeightsGpu {
    /// Upload the head from a reader over an MTP sidecar.
    ///
    /// ⚠️ The sidecar stores routed experts SPLIT per expert
    /// (`experts.<e>.<proj>.weight`) because `hipfire-quantize` splits stacked
    /// MoE tensors on the way in, while the source ships them stacked. The
    /// trunk's `stack_experts` expects the stacked name, so a reader for this
    /// must present them stacked — the probe restacks, and so must any caller
    /// here. Getting it wrong is a missing-weight error, not a wrong answer.
    pub fn upload(gpu: &mut Gpu, cfg: &Qwen4ExpConfig, w: &dyn TensorReader) -> HipResult<Self> {
        crate::trunk_gpu::upload_mtp_head(gpu, cfg, w)
    }
}
