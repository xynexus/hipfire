// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
// Qwen3.5 model-specific kernel extensions.
//
// These operations are unique to Qwen3.5's linear attention layers (DeltaNet):
// the gated linear recurrence with quantized/FP32 state, conv-state ring
// buffer management, and tree-batched speculative-decode variants. They
// don't fit into the standard dispatch families because the state is
// model-owned and the recurrence is an inherently sequential kernel.

use hipfire_rdna::{Gpu, GpuTensor};

// ── State quantization ─────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
/// Mirrors `hipfire_arch_qwen35::qwen35::StateQuant`; keep the two in step.
/// Q8/Q4 removed 2026-08-09 (silent corruption); FP32 default, FP16 opt-in.
pub enum StateQuant {
    FP32,
    FP16,
}

// ── Parameter structs ──────────────────────────────────

/// Parameters for a single-token DeltaNet state update.
///
/// The gated delta net recurrence:
///   S' = gate · S + beta · (k ⊗ v)
///   output = S · q
///
/// where S is the recurrent state (n_heads × head_dim × head_dim),
/// quantized per the `quant` field.
pub struct DeltaNetStepParams<'a> {
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    pub state: &'a GpuTensor,
    pub s_scales: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub n_heads: usize,
    pub head_dim: usize,
    pub quant: StateQuant,
    /// Absolute sequence position of this token. Seeds the Q8 state-requant
    /// stochastic rounding deterministically (issue #17).
    pub seq_pos: usize,
    /// DeltaNet (linear-attention) layer index; mixed into the same seed so
    /// layers do not share a dither sequence.
    pub delta_layer: usize,
}

/// Parameters for batched sequential DeltaNet updates (prefill path).
///
/// Q, K, V, gate, beta, and output are batched [n_tokens, n_heads, head_dim].
/// The state is updated in-place for all n_tokens.
pub struct DeltaNetBatchParams<'a> {
    pub q_batch: &'a GpuTensor,
    pub k_batch: &'a GpuTensor,
    pub v_batch: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub beta_batch: &'a GpuTensor,
    pub state: &'a GpuTensor,
    pub s_scales: &'a GpuTensor,
    pub output_batch: &'a GpuTensor,
    pub n_tokens: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub quant: StateQuant,
    /// Absolute sequence position of `q_batch[0]`; token `i` of the block is
    /// seeded at `seq_pos + i` (issue #17).
    pub seq_pos: usize,
    /// DeltaNet (linear-attention) layer index.
    pub delta_layer: usize,
}

/// Parameters for tree-batched DeltaNet (speculative-decode path).
///
/// Adds a tape buffer and parent-indices array so sibling tokens read
/// the correct parent's post-update state.
pub struct DeltaNetTreeParams<'a> {
    pub q_batch: &'a GpuTensor,
    pub k_batch: &'a GpuTensor,
    pub v_batch: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub beta_batch: &'a GpuTensor,
    /// Pre-block state snapshot (READ-ONLY). f32 or f16 per `quant`.
    pub s_init: &'a GpuTensor,
    /// Per-token tape, `[n_tokens x n_heads x HD x HD]`. No scales: neither f32
    /// nor f16 needs them — that pair of buffers existed only for Q8.
    pub s_tape: &'a GpuTensor,
    pub parent_indices: &'a GpuTensor,
    pub output_batch: &'a GpuTensor,
    pub n_tokens: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    /// Picks the tree kernel. The tape's element size follows it, so a mismatch
    /// reads every element at the wrong offset.
    pub quant: StateQuant,
}

/// Parameters for DeltaNet conv-state ring-buffer management.
pub struct ConvStateParams<'a> {
    pub state: &'a GpuTensor,
    pub input: &'a GpuTensor,
    pub conv_channels: usize,
    pub kernel_size: usize,
    pub position: usize,
}

// ── Trait ──────────────────────────────────────────────

pub trait Qwen35ModelExt {
    /// Run a single-token DeltaNet state update.
    ///
    /// Dispatches to the f32 or f16 kernel based on `params.quant`.
    fn run_delta_net_step(&self, gpu: &mut Gpu, params: &DeltaNetStepParams) -> Result<(), String>;

    /// Run batched sequential DeltaNet updates (prefill).
    ///
    /// Dispatches to the f32 or f16 batched kernel.
    fn run_delta_net_batch(
        &self,
        gpu: &mut Gpu,
        params: &DeltaNetBatchParams,
    ) -> Result<(), String>;

    /// Run tree-batched DeltaNet (speculative-decode path).
    ///
    /// f32 and f16 both supported; the Q8-only era ended with the Q8 tree kernel.
    fn run_delta_net_tree(&self, gpu: &mut Gpu, params: &DeltaNetTreeParams) -> Result<(), String>;

    /// Zero the conv-state ring buffer.
    fn reset_conv_state(
        &self,
        gpu: &mut Gpu,
        state: &GpuTensor,
        conv_state_size: usize,
    ) -> Result<(), String>;
}

// ── Default implementations ────────────────────────────

impl Qwen35ModelExt for () {
    fn run_delta_net_step(&self, gpu: &mut Gpu, params: &DeltaNetStepParams) -> Result<(), String> {
        match params.quant {
            StateQuant::FP32 => gpu.gated_delta_net_f32(
                params.q,
                params.k,
                params.v,
                params.gate,
                params.beta,
                params.state,
                params.output,
                1,
                params.n_heads,
                params.head_dim,
            ),
            StateQuant::FP16 => gpu.gated_delta_net_f16_batch_seq(
                params.q,
                params.k,
                params.v,
                params.gate,
                params.beta,
                params.state,
                params.output,
                1,
                params.n_heads,
                params.head_dim,
            ),
        }
        .map_err(|e| format!("delta_net_step: {e:?}"))
    }

    fn run_delta_net_batch(
        &self,
        gpu: &mut Gpu,
        params: &DeltaNetBatchParams,
    ) -> Result<(), String> {
        // Both precisions have a real batched kernel now, so the old
        // "loop the single-token kernel" fallback is gone. It existed because
        // Q4 had no batch variant.
        // Chunkwise-parallel when opted in: resolves the batch's tokens together
        // rather than one at a time, which is what lets a batched prefill (and
        // spec-decode verify) amortize on a DeltaNet-heavy stack. Same
        // recurrence, different summation order — see `hipfire_rdna::gdn_chunk`.
        if matches!(params.quant, StateQuant::FP32) && hipfire_rdna::gdn_chunk::chunk_enabled() {
            return gpu.gated_delta_net_f32_chunk(
                params.q_batch,
                params.k_batch,
                params.v_batch,
                params.gate_batch,
                params.beta_batch,
                params.state,
                params.output_batch,
                params.n_tokens,
                params.n_heads,
                params.head_dim,
            );
        }
        match params.quant {
            StateQuant::FP32 => gpu.gated_delta_net_f32_batch_seq(
                params.q_batch,
                params.k_batch,
                params.v_batch,
                params.gate_batch,
                params.beta_batch,
                params.state,
                params.output_batch,
                params.n_tokens,
                params.n_heads,
                params.head_dim,
            ),
            StateQuant::FP16 => gpu.gated_delta_net_f16_batch_seq(
                params.q_batch,
                params.k_batch,
                params.v_batch,
                params.gate_batch,
                params.beta_batch,
                params.state,
                params.output_batch,
                params.n_tokens,
                params.n_heads,
                params.head_dim,
            ),
        }
        .map_err(|e| format!("delta_net_batch: {e:?}"))
    }

    fn run_delta_net_tree(&self, gpu: &mut Gpu, params: &DeltaNetTreeParams) -> Result<(), String> {
        match params.quant {
            StateQuant::FP32 => gpu.gated_delta_net_f32_tree_batch_seq(
                params.q_batch,
                params.k_batch,
                params.v_batch,
                params.gate_batch,
                params.beta_batch,
                params.s_init,
                params.s_tape,
                params.parent_indices,
                params.output_batch,
                params.n_tokens,
                params.n_heads,
                params.head_dim,
            ),
            StateQuant::FP16 => gpu.gated_delta_net_f16_tree_batch_seq(
                params.q_batch,
                params.k_batch,
                params.v_batch,
                params.gate_batch,
                params.beta_batch,
                params.s_init,
                params.s_tape,
                params.parent_indices,
                params.output_batch,
                params.n_tokens,
                params.n_heads,
                params.head_dim,
            ),
        }
        .map_err(|e| format!("delta_net_tree: {e:?}"))
    }

    fn reset_conv_state(
        &self,
        gpu: &mut Gpu,
        state: &GpuTensor,
        _conv_state_size: usize,
    ) -> Result<(), String> {
        gpu.hip
            .memset(&state.buf, 0, state.buf.size())
            .map_err(|e| format!("reset_conv_state: {e:?}"))
    }
}
