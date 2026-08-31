// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! The gated residual on the GPU — the spine every layer passes through twice.
//!
//! Mirrors [`crate::hc::GatedResidual`], which `tests/reference_oracle.rs` pins
//! against the upstream implementation. `examples/parity_hc_gpu_vs_cpu` differences
//! the two.
//!
//! The read side is six launches rather than one fused kernel. That is deliberate
//! for now: each step already has a differenced kernel behind it, and fusing before
//! the composition is verified would make a mismatch much harder to localise. The
//! shapes here are small (a `[hc*hidden]` norm and two skinny GEMVs), so the launch
//! count — not the arithmetic — is what a later fusion would be buying back.

use crate::config::Qwen4ExpConfig;
use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};
use hipfire_runtime::weights::{weight_gemv, WeightTensor};

pub struct HcWeights {
    /// `[hc_count * hidden]`
    pub hc_norm: GpuTensor,
    /// `[lowrank, hc_count * hidden]`
    pub mix_down: WeightTensor,
    /// `[hc_count * hidden, lowrank]`
    pub mix_up: WeightTensor,
    /// `[hc_count, hc_count * hidden]`, absent for the model-level mixer.
    pub block_inject: Option<WeightTensor>,
}

/// Scratch for one residual read. Sized off the config, reused across layers.
pub struct HcScratch {
    pub normed: GpuTensor,
    lowrank: GpuTensor,
    mix: GpuTensor,
    /// `[hc_count]` write gates; empty for the mixer.
    pub inject: GpuTensor,
}

impl HcScratch {
    pub fn new(gpu: &mut Gpu, cfg: &Qwen4ExpConfig) -> HipResult<Self> {
        let width = cfg.gated_residual.count * cfg.hidden;
        Ok(Self {
            normed: gpu.zeros(&[width], DType::F32)?,
            lowrank: gpu.zeros(&[cfg.gated_residual.lowrank], DType::F32)?,
            mix: gpu.zeros(&[width], DType::F32)?,
            inject: gpu.zeros(&[cfg.gated_residual.count], DType::F32)?,
        })
    }
}

/// Read: normalise the streams, build the per-channel mix, collapse to `[hidden]`.
///
/// Also leaves `scratch.normed` and (when the layer injects) `scratch.inject`
/// populated — the write side needs both, and recomputing the norm would double
/// the cost of the spine.
pub fn hc_read(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    w: &HcWeights,
    s: &mut HcScratch,
    streams: &GpuTensor,
    mixed_out: &GpuTensor,
) -> HipResult<()> {
    let hc = cfg.gated_residual.count;
    let width = hc * cfg.hidden;
    let inv = 1.0 / hc as f32;

    gpu.hc_grouped_rmsnorm(
        streams,
        &w.hc_norm,
        &s.normed,
        cfg.hidden as i32,
        hc as i32,
        cfg.rms_norm_eps,
    )?;
    // Low-rank gate: down -> /hc_count -> silu -> up -> sigmoid. The division is
    // INSIDE the silu, before the expand.
    weight_gemv(gpu, &w.mix_down, &s.normed, &s.lowrank)?;
    gpu.hc_scaled_silu(
        &s.lowrank,
        &s.lowrank,
        cfg.gated_residual.lowrank as i32,
        inv,
    )?;
    weight_gemv(gpu, &w.mix_up, &s.lowrank, &s.mix)?;
    gpu.hc_sigmoid(&s.mix, &s.mix, width as i32)?;
    // MEAN over streams of the per-channel-gated normed streams.
    gpu.hc_input_map_perchannel(&s.mix, &s.normed, mixed_out, cfg.hidden as i32, hc as i32)?;

    if let Some(bi) = w.block_inject.as_ref() {
        weight_gemv(gpu, bi, &s.normed, &s.inject)?;
        gpu.hc_inject_gate(&s.inject, &s.inject, hc as i32, inv)?;
    }
    Ok(())
}

/// Write: add the block output into every stream, scaled by that stream's gate.
///
/// Operates on the RAW streams, not the normalised ones — the normalisation exists
/// to compute the gates, not to replace the residual.
pub fn hc_write(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    s: &HcScratch,
    streams: &GpuTensor,
    block_out: &GpuTensor,
) -> HipResult<()> {
    gpu.hc_residual_inject(
        streams,
        block_out,
        &s.inject,
        cfg.hidden as i32,
        cfg.gated_residual.count as i32,
    )
}

// ── GPU teardown ────────────────────────────────────────────────────────────
//
// Every `free` below DESTRUCTURES its struct exhaustively rather than naming
// fields to free. That is deliberate: a field added later fails to compile until
// someone decides what happens to it, where a `self.a; self.b;` list would just
// silently leak the new tensor. `unload` on a 360 GB model has no test that would
// catch that.

impl HcWeights {
    pub fn free(self, gpu: &mut Gpu) {
        let Self {
            hc_norm,
            mix_down,
            mix_up,
            block_inject,
        } = self;
        let _ = gpu.free_tensor(hc_norm);
        for t in [Some(mix_down), Some(mix_up), block_inject]
            .into_iter()
            .flatten()
        {
            let _ = gpu.free_tensor(t.buf);
        }
    }
}

impl HcScratch {
    pub fn free(self, gpu: &mut Gpu) {
        let Self {
            normed,
            lowrank,
            mix,
            inject,
        } = self;
        for t in [normed, lowrank, mix, inject] {
            let _ = gpu.free_tensor(t);
        }
    }
}
