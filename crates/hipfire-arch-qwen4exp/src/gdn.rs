// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Gated DeltaNet layer for qwen4_exp — scratch, state, and the decode step.
//!
//! Every kernel this needs already exists and is exercised at exactly this
//! model's geometry (Qwen3.8-27B ships 16 QK / 48 V heads at head_dim 128, conv
//! kernel 4). The only compute delta is the output gate, which qwen4_exp sets to
//! sigmoid where Qwen3.5/3.8 use silu — supplied by `gated_norm_sigmoid_f32`.
//!
//! So this drives the rdna kernels directly rather than threading a qwen4_exp flag
//! through qwen35's driver. That driver calls `gated_norm_f32` at **8 sites across
//! 6 files**, every one on a shipping Qwen3.5/3.8 hot path, and widening it for a
//! second family is the tax the separate-crate decision exists to avoid. The cost
//! of this route is the scratch and state bookkeeping below; the benefit is that
//! nothing shipping changes.
//!
//! Deliberately independent of the serving tier: this is a layer step over GPU
//! tensors, so it does not wait on the `ServingFactory`-vs-ladder decision.

use crate::config::Qwen4ExpConfig;
use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

/// Per-layer weights, already resident.
pub struct GdnWeights {
    /// `[qkv_dim, hidden]` — Q and K at the key span, V at the value span.
    pub in_proj_qkv: GpuTensor,
    /// `[z_dim, hidden]` — the output gate, spanning V.
    pub in_proj_z: GpuTensor,
    /// `[value_heads, hidden]` each.
    pub in_proj_a: GpuTensor,
    pub in_proj_b: GpuTensor,
    /// `[qkv_dim, 1, conv_kernel]`, flattened.
    pub conv_weight: GpuTensor,
    /// `[value_heads]` each.
    pub a_log: GpuTensor,
    pub dt_bias: GpuTensor,
    /// `[value_head_dim]` — per-head RMSNorm, applied PLAIN (ones-init), unlike
    /// this family's other norm which carries a `+1`.
    pub norm_weight: GpuTensor,
    /// `[hidden, z_dim]`
    pub out_proj: GpuTensor,
}

/// Recurrent state for one sequence in one layer.
pub struct GdnState {
    /// `[value_heads, key_head_dim, value_head_dim]` f32.
    pub recurrent: GpuTensor,
    /// `[qkv_dim, conv_kernel - 1]` f32 ring.
    pub conv: GpuTensor,
}

impl GdnState {
    pub fn zeros(gpu: &mut Gpu, cfg: &Qwen4ExpConfig) -> HipResult<Self> {
        let d = &cfg.deltanet;
        Ok(Self {
            recurrent: gpu.zeros(
                &[d.value_heads * d.key_head_dim * d.value_head_dim],
                DType::F32,
            )?,
            conv: gpu.zeros(&[d.qkv_dim() * (d.conv_kernel - 1)], DType::F32)?,
        })
    }
}

/// Reusable per-step buffers. Allocated once per sequence, not per layer — the
/// layers run one at a time and none of this survives the step.
pub struct GdnScratch {
    qkv: GpuTensor,
    z: GpuTensor,
    a: GpuTensor,
    b: GpuTensor,
    alpha: GpuTensor,
    beta: GpuTensor,
    q_raw: GpuTensor,
    k_raw: GpuTensor,
    v: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    attn: GpuTensor,
    gated: GpuTensor,
}

impl GdnScratch {
    pub fn new(gpu: &mut Gpu, cfg: &Qwen4ExpConfig) -> HipResult<Self> {
        let d = &cfg.deltanet;
        let k_span = d.key_heads * d.key_head_dim;
        let v_span = d.z_dim();
        let f32z = |g: &mut Gpu, n: usize| g.zeros(&[n], DType::F32);
        Ok(Self {
            qkv: f32z(gpu, d.qkv_dim())?,
            z: f32z(gpu, v_span)?,
            a: f32z(gpu, d.value_heads)?,
            b: f32z(gpu, d.value_heads)?,
            alpha: f32z(gpu, d.value_heads)?,
            beta: f32z(gpu, d.value_heads)?,
            q_raw: f32z(gpu, k_span)?,
            k_raw: f32z(gpu, k_span)?,
            v: f32z(gpu, v_span)?,
            // Q and K are repeat-interleaved up to the VALUE head count before the
            // recurrence, so these are v_span wide, not k_span.
            q: f32z(gpu, v_span)?,
            k: f32z(gpu, v_span)?,
            attn: f32z(gpu, v_span)?,
            gated: f32z(gpu, v_span)?,
        })
    }
}

/// One decode step through a Gated DeltaNet layer.
///
/// `x` is `[hidden]` (the gated residual's collapsed read), `y` is `[hidden]`.
/// `state` is advanced in place.
///
/// Sequence, matching the reference and qwen35's driver:
/// project → gates → causal conv + SiLU + split → QK L2-norm & scale →
/// repeat-interleave Q/K to the V head count → recurrence → gated norm → out.
pub fn gdn_decode_step(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    w: &GdnWeights,
    s: &mut GdnScratch,
    st: &mut GdnState,
    x: &GpuTensor,
    y: &GpuTensor,
) -> HipResult<()> {
    let d = &cfg.deltanet;
    let hd = d.key_head_dim;

    gpu.gemv_f32(&w.in_proj_qkv, x, &s.qkv)?;
    gpu.gemv_f32(&w.in_proj_z, x, &s.z)?;
    gpu.gemv_f32(&w.in_proj_a, x, &s.a)?;
    gpu.gemv_f32(&w.in_proj_b, x, &s.b)?;

    // beta = sigmoid(b); alpha = softplus(a + dt_bias) * -exp(A_log).
    // The kernel reads `a`/`b` in place through the alpha/beta buffers.
    gpu.memcpy_dtod_auto(&s.beta.buf, &s.b.buf, d.value_heads * 4)?;
    gpu.memcpy_dtod_auto(&s.alpha.buf, &s.a.buf, d.value_heads * 4)?;
    gpu.fused_sigmoid_alpha_gate_f32(&s.beta, &s.alpha, &w.dt_bias, &w.a_log, d.value_heads)?;

    // Causal depthwise conv + SiLU, splitting the fused projection into Q, K at
    // the key span and V at the value span. UNDILATED — unlike the PLE conv in the
    // same model, which is dilated by ngram_size.
    gpu.conv1d_silu_split_f32(
        &s.q_raw,
        &s.k_raw,
        &s.v,
        &s.qkv,
        &w.conv_weight,
        &st.conv,
        d.key_heads * hd,
        d.z_dim(),
    )?;

    gpu.fused_qk_l2_norm_scale_f32(
        &s.q_raw,
        &s.k_raw,
        d.key_heads,
        hd,
        1.0 / (hd as f32).sqrt(),
        cfg.rms_norm_eps,
    )?;

    // 48 V heads over 16 QK heads: Q and K are materialised up to the V count
    // before the recurrence, which is what makes the kernel's uniform-head
    // assumption hold.
    let ratio = d.value_per_key();
    if ratio > 1 {
        gpu.repeat_interleave_qk_f32(&s.q_raw, &s.k_raw, &s.q, &s.k, d.key_heads, ratio, hd)?;
    } else {
        let bytes = d.key_heads * hd * 4;
        gpu.memcpy_dtod_auto(&s.q.buf, &s.q_raw.buf, bytes)?;
        gpu.memcpy_dtod_auto(&s.k.buf, &s.k_raw.buf, bytes)?;
    }

    gpu.gated_delta_net_f32(
        &s.q,
        &s.k,
        &s.v,
        &s.alpha,
        &s.beta,
        &st.recurrent,
        &s.attn,
        1,
        d.value_heads,
        d.value_head_dim,
    )?;

    // THE qwen4_exp delta: sigmoid, where Qwen3.5/3.8 gate with silu.
    if d.output_gate_sigmoid {
        gpu.gated_norm_sigmoid_f32(
            &s.attn,
            &s.z,
            &w.norm_weight,
            &s.gated,
            d.value_heads as i32,
            d.value_head_dim as i32,
            1,
            cfg.rms_norm_eps,
        )?;
    } else {
        // The Qwen3.5/3.8 sibling takes usize dims and no batch arg.
        gpu.gated_norm_f32(
            &s.attn,
            &s.z,
            &w.norm_weight,
            &s.gated,
            d.value_heads,
            d.value_head_dim,
            cfg.rms_norm_eps,
        )?;
    }

    gpu.gemv_f32(&w.out_proj, &s.gated, y)
}

// ── GPU teardown ────────────────────────────────────────────────────────────
//
// Every `free` below DESTRUCTURES its struct exhaustively rather than naming
// fields to free. That is deliberate: a field added later fails to compile until
// someone decides what happens to it, where a `self.a; self.b;` list would just
// silently leak the new tensor. `unload` on a 360 GB model has no test that would
// catch that.

impl GdnWeights {
    pub fn free(self, gpu: &mut Gpu) {
        let Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            conv_weight,
            a_log,
            dt_bias,
            norm_weight,
            out_proj,
        } = self;
        for t in [
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            conv_weight,
            a_log,
            dt_bias,
            norm_weight,
            out_proj,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

impl GdnState {
    /// Zero the recurrent state and the conv ring. A new sequence must not
    /// inherit the previous one's history — this is the whole state a GDN layer
    /// carries between tokens.
    pub fn reset(&self, gpu: &mut Gpu) -> hipfire_rdna::HipResult<()> {
        // No in-place zero primitive exists, so write host zeros over the
        // buffers. Both are small (one S matrix and one conv ring per layer) and
        // this runs once per sequence, not per token.
        for t in [&self.recurrent, &self.conv] {
            let zeros = vec![0u8; t.buf.size()];
            gpu.memcpy_htod_auto(&t.buf, &zeros)?;
        }
        Ok(())
    }

    pub fn free(self, gpu: &mut Gpu) {
        let Self { recurrent, conv } = self;
        for t in [recurrent, conv] {
            let _ = gpu.free_tensor(t);
        }
    }
}

impl GdnScratch {
    pub fn free(self, gpu: &mut Gpu) {
        let Self {
            qkv,
            z,
            a,
            b,
            alpha,
            beta,
            q_raw,
            k_raw,
            v,
            q,
            k,
            attn,
            gated,
        } = self;
        for t in [
            qkv, z, a, b, alpha, beta, q_raw, k_raw, v, q, k, attn, gated,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}
