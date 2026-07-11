// SPDX-License-Identifier: Apache-2.0
// hipfire — embeddinggemma bidirectional encoder forward. See LICENSE / NOTICE.

//! The embeddinggemma encode pass: one **bidirectional** batched prefill over all
//! `m` tokens, then host-side mean pooling, the sentence-transformers Dense heads,
//! and L2 normalization.
//!
//! The transformer layer body is identical to Gemma-3's batched prefill
//! (`hipfire_arch_gemma3::forward::forward_prefill_batch`) with two swaps:
//!
//! * **Attention** — the causal, KV-cache-backed `attention_f32_batched` is replaced
//!   by [`Gpu::attention_dflash_f32`], a genuine non-causal self-attention over the
//!   `m` present tokens (no KV cache; every token sees every token). For
//!   embeddinggemma `query_pre_attn_scalar == head_dim == 256`, so the kernel's
//!   built-in `1/√head_dim` scale is already Gemma's and no `q_prescale` bake is
//!   needed (it is `1.0`).
//! * **Output** — instead of a final-position lm_head, the final-normed hidden
//!   states `[m, hidden]` are pooled → projected → normalized on the host.
//!
//! ## Sliding window
//!
//! embeddinggemma interleaves local (`sliding_window = 512`) and global layers, but
//! the attention is *bidirectional within the window*. For `m ≤ sliding_window`
//! (the common retrieval-chunk case) the window never clips, so full bidirectional
//! attention is exact on every layer. For `m > sliding_window` the local layers
//! would need a banded mask that this bring-up path does not yet apply; we warn and
//! proceed (still exact on the global layers, an approximation on the local ones).

use hip_bridge::HipResult;
use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_runtime::weights::{weight_gemm, WeightTensor};

use crate::config::{EmbeddingGemmaConfig, PoolingMode};
use crate::weights::EmbeddingGemmaWeights;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Projection {
    Query,
    Key,
    Value,
    AttentionOutput,
    Gate,
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionBoundary {
    Fallback,
    AttentionOnly,
    OutputProjected,
}

pub trait LinearProjector {
    fn project(
        &mut self,
        gpu: &mut Gpu,
        layer_idx: usize,
        projection: Projection,
        weight: &WeightTensor,
        input: &GpuTensor,
        output: &GpuTensor,
        rows: usize,
    ) -> HipResult<()>;

    /// Execute one complete encoder layer when the backend owns the resident
    /// attention, FFN, residual, and normalization boundary. The normalized
    /// input and residual are separate because the admitted R34 ABI consumes
    /// both. Returning `false` selects the canonical operation sequence below.
    fn project_layer(
        &mut self,
        _gpu: &mut Gpu,
        _layer_idx: usize,
        _normalized_input: &GpuTensor,
        _residual_and_output: &GpuTensor,
        _rows: usize,
    ) -> HipResult<bool> {
        Ok(false)
    }

    fn take_layer_debug_hidden(&mut self) -> Option<Vec<f32>> {
        None
    }

    fn take_layer_debug_ffn(&mut self) -> Option<Vec<f32>> {
        None
    }

    /// Execute the complete QKV projection, Q/K normalization, RoPE, and
    /// bidirectional attention boundary when a resident backend owns it.
    /// Returning `false` selects the canonical per-operation fallback below.
    fn project_attention(
        &mut self,
        _gpu: &mut Gpu,
        _layer_idx: usize,
        _input: &GpuTensor,
        _attention_output: &GpuTensor,
        _projected_output: &GpuTensor,
        _rows: usize,
    ) -> HipResult<AttentionBoundary> {
        Ok(AttentionBoundary::Fallback)
    }

    #[allow(clippy::too_many_arguments)]
    fn project_qkv(
        &mut self,
        gpu: &mut Gpu,
        layer_idx: usize,
        wq: &WeightTensor,
        wk: &WeightTensor,
        wv: &WeightTensor,
        input: &GpuTensor,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        rows: usize,
    ) -> HipResult<()> {
        self.project(gpu, layer_idx, Projection::Query, wq, input, q, rows)?;
        self.project(gpu, layer_idx, Projection::Key, wk, input, k, rows)?;
        self.project(gpu, layer_idx, Projection::Value, wv, input, v, rows)
    }

    #[allow(clippy::too_many_arguments)]
    fn project_gate_up(
        &mut self,
        gpu: &mut Gpu,
        layer_idx: usize,
        gate_weight: &WeightTensor,
        up_weight: &WeightTensor,
        input: &GpuTensor,
        gate: &GpuTensor,
        up: &GpuTensor,
        rows: usize,
    ) -> HipResult<()> {
        self.project(
            gpu,
            layer_idx,
            Projection::Gate,
            gate_weight,
            input,
            gate,
            rows,
        )?;
        self.project(gpu, layer_idx, Projection::Up, up_weight, input, up, rows)
    }

    #[allow(clippy::too_many_arguments)]
    fn project_ffn(
        &mut self,
        gpu: &mut Gpu,
        layer_idx: usize,
        gate_weight: &WeightTensor,
        up_weight: &WeightTensor,
        down_weight: &WeightTensor,
        input: &GpuTensor,
        gate: &GpuTensor,
        up: &GpuTensor,
        activated: &GpuTensor,
        output: &GpuTensor,
        rows: usize,
    ) -> HipResult<()> {
        self.project_gate_up(
            gpu,
            layer_idx,
            gate_weight,
            up_weight,
            input,
            gate,
            up,
            rows,
        )?;
        gpu.gelu_mul_f32(gate, up, activated)?;
        self.project(
            gpu,
            layer_idx,
            Projection::Down,
            down_weight,
            activated,
            output,
            rows,
        )
    }
}

pub struct GpuLinearProjector;

impl LinearProjector for GpuLinearProjector {
    fn project(
        &mut self,
        gpu: &mut Gpu,
        _layer_idx: usize,
        _projection: Projection,
        weight: &WeightTensor,
        input: &GpuTensor,
        output: &GpuTensor,
        rows: usize,
    ) -> HipResult<()> {
        weight_gemm(gpu, weight, input, output, rows)
    }
}

/// Encode `tokens` into the native-dimension, L2-normalized sentence embedding.
/// The caller is responsible for prepending any task prompt and for Matryoshka
/// truncation (see [`EmbeddingGemmaConfig::resolve_dims`]).
pub fn embed_forward(
    gpu: &mut Gpu,
    weights: &EmbeddingGemmaWeights,
    cfg: &EmbeddingGemmaConfig,
    tokens: &[u32],
) -> Result<Vec<f32>, String> {
    if tokens.is_empty() {
        return Err("embeddinggemma: empty token sequence".to_string());
    }
    let mut projector = GpuLinearProjector;
    embed_forward_with_projector(gpu, weights, cfg, tokens, &mut projector)
}

pub fn embed_forward_with_projector<P: LinearProjector>(
    gpu: &mut Gpu,
    weights: &EmbeddingGemmaWeights,
    cfg: &EmbeddingGemmaConfig,
    tokens: &[u32],
    projector: &mut P,
) -> Result<Vec<f32>, String> {
    if tokens.is_empty() {
        return Err("embeddinggemma: empty token sequence".to_string());
    }
    let hidden = encode_pooled_hidden_with_projector(gpu, weights, cfg, tokens, projector)
        .map_err(|e| format!("embeddinggemma: encode failed: {e:?}"))?;
    project_dense_with_capture(&weights.dense_heads, hidden, |_, _| Ok(()))
}

/// Apply the ordered sentence-transformers Dense heads, exposing each head's
/// input before projection. Calibration uses this to capture host-resident
/// linear activations without changing the serving path.
pub(crate) fn project_dense_with_capture<F>(
    dense_heads: &[crate::weights::DenseHeadHost],
    mut hidden: Vec<f32>,
    mut capture: F,
) -> Result<Vec<f32>, String>
where
    F: FnMut(usize, &[f32]) -> Result<(), String>,
{
    for (head_idx, head) in dense_heads.iter().enumerate() {
        capture(head_idx, &hidden)?;
        hidden = head.apply(&hidden);
    }
    l2_normalize(&mut hidden);
    Ok(hidden)
}

/// Run the bidirectional transformer stack + final norm + mean pooling, returning
/// the pooled (but not yet projected) hidden vector `[hidden_size]`.
pub(crate) fn encode_pooled_hidden(
    gpu: &mut Gpu,
    weights: &EmbeddingGemmaWeights,
    cfg: &EmbeddingGemmaConfig,
    tokens: &[u32],
) -> HipResult<Vec<f32>> {
    let mut projector = GpuLinearProjector;
    encode_pooled_hidden_with_projector(gpu, weights, cfg, tokens, &mut projector)
}

fn encode_pooled_hidden_with_projector<P: LinearProjector>(
    gpu: &mut Gpu,
    weights: &EmbeddingGemmaWeights,
    cfg: &EmbeddingGemmaConfig,
    tokens: &[u32],
    projector: &mut P,
) -> HipResult<Vec<f32>> {
    let trace_phases = std::env::var("HIPFIRE_EMBED_TRACE_PHASES").is_ok_and(|value| value != "0");
    let compare_resident_layer =
        std::env::var("HIPFIRE_EMBED_COMPARE_RESIDENT_LAYER").is_ok_and(|value| value != "0");
    let mut qkv_ms = 0.0f64;
    let mut attention_core_ms = 0.0f64;
    let mut attention_output_ms = 0.0f64;
    let mut ffn_core_ms = 0.0f64;
    let mut ffn_output_ms = 0.0f64;
    let dim = cfg.hidden_size;
    let n_heads = cfg.num_attention_heads;
    let n_kv_heads = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let inter = cfg.intermediate_size;
    let eps = cfg.rms_norm_eps;
    let m = tokens.len();
    let g3 = crate::weights::gemma3_config(cfg);
    let backbone = &weights.backbone;

    if m > cfg.sliding_window {
        eprintln!(
            "embeddinggemma: sequence length {m} exceeds sliding_window {} — local \
             layers are approximated as full-bidirectional (banded mask not yet \
             implemented)",
            cfg.sliding_window
        );
    }

    // ── Embed all m tokens (×√hidden) into a [m, dim] residual batch ──
    let x_batch = gpu.alloc_owned(&[m * dim], DType::F32)?;
    let embed_tmp = gpu.alloc_owned(&[dim], DType::F32)?;
    for (i, &tok) in tokens.iter().enumerate() {
        if let Some(host_embedding) = &weights.host_embedding {
            let mut row = host_embedding
                .row(tok, dim)
                .map_err(|e| hip_bridge::HipError::new(0, &e))?;
            let scale = cfg.embed_scale();
            for value in &mut row {
                *value *= scale;
            }
            let uploaded = gpu.upload_f32(&row, &[dim])?;
            gpu.memcpy_dtod_at_auto(&x_batch.buf, i * dim * 4, &uploaded.buf, 0, dim * 4)?;
            gpu.free_tensor(uploaded)?;
        } else {
            hipfire_arch_gemma3::forward::embed_token(gpu, backbone, &g3, &embed_tmp, tok)?;
            gpu.memcpy_dtod_at_auto(&x_batch.buf, i * dim * 4, &embed_tmp.buf, 0, dim * 4)?;
        }
    }

    // Encoder positions 0..m as an i32 device table.
    let positions = gpu.alloc_owned(&[m], DType::F32)?;
    {
        let bytes: Vec<u8> = (0i32..m as i32).flat_map(|p| p.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&positions.buf, &bytes)?;
    }

    // Batched scratch (OwnedTensor frees on drop).
    let tmp = gpu.alloc_owned(&[m * dim], DType::F32)?;
    let q = gpu.alloc_owned(&[m * q_dim], DType::F32)?;
    let k = gpu.alloc_owned(&[m * kv_dim], DType::F32)?;
    let v = gpu.alloc_owned(&[m * kv_dim], DType::F32)?;
    let attn_out = gpu.alloc_owned(&[m * q_dim], DType::F32)?;
    let o = gpu.alloc_owned(&[m * dim], DType::F32)?;
    let gate = gpu.alloc_owned(&[m * inter], DType::F32)?;
    let up = gpu.alloc_owned(&[m * inter], DType::F32)?;
    let ffn = gpu.alloc_owned(&[m * inter], DType::F32)?;
    let compare_input = compare_resident_layer
        .then(|| gpu.alloc_owned(&[m * dim], DType::F32))
        .transpose()?;
    let compare_output = compare_resident_layer
        .then(|| gpu.alloc_owned(&[m * dim], DType::F32))
        .transpose()?;

    for layer_idx in 0..cfg.num_hidden_layers {
        let layer = &backbone.layers[layer_idx];
        let stage_started = std::time::Instant::now();

        // ── Attention block (bidirectional) ──
        gpu.rmsnorm_batched(&x_batch, &layer.input_norm, &tmp, m, dim, eps)?;
        let compare_this_layer = compare_resident_layer && layer_idx == 0;
        if compare_this_layer {
            gpu.memcpy_dtod_at_auto(
                &compare_input.as_ref().expect("comparison input").buf,
                0,
                &x_batch.buf,
                0,
                m * dim * size_of::<f32>(),
            )?;
        }
        let completed_layer = projector.project_layer(gpu, layer_idx, &tmp, &x_batch, m)?;
        if completed_layer && !compare_this_layer {
            if trace_phases {
                gpu.device_synchronize()?;
                attention_core_ms += stage_started.elapsed().as_secs_f64() * 1e3;
            }
            continue;
        }
        if completed_layer {
            gpu.memcpy_dtod_at_auto(
                &compare_output.as_ref().expect("comparison output").buf,
                0,
                &x_batch.buf,
                0,
                m * dim * size_of::<f32>(),
            )?;
            gpu.memcpy_dtod_at_auto(
                &x_batch.buf,
                0,
                &compare_input.as_ref().expect("comparison input").buf,
                0,
                m * dim * size_of::<f32>(),
            )?;
        }
        let attention_boundary =
            projector.project_attention(gpu, layer_idx, &tmp, &attn_out, &o, m)?;
        if attention_boundary != AttentionBoundary::Fallback {
            if trace_phases {
                gpu.device_synchronize()?;
                attention_core_ms += stage_started.elapsed().as_secs_f64() * 1e3;
            }
        } else {
            projector.project_qkv(
                gpu, layer_idx, &layer.wq, &layer.wk, &layer.wv, &tmp, &q, &k, &v, m,
            )?;
            if trace_phases {
                gpu.device_synchronize()?;
                qkv_ms += stage_started.elapsed().as_secs_f64() * 1e3;
            }

            // Per-head QK-norm (q_norm carries the baked Q pre-scale, 1.0 here).
            let stage_started = std::time::Instant::now();
            gpu.rmsnorm_batched(&q, &layer.q_norm, &q, m * n_heads, head_dim, eps)?;
            gpu.rmsnorm_batched(&k, &layer.k_norm, &k, m * n_kv_heads, head_dim, eps)?;

            gpu.rope_batched_f32(
                &q,
                &k,
                &positions,
                n_heads,
                n_kv_heads,
                head_dim,
                cfg.rope_base_for_layer(layer_idx),
                m,
            )?;

            // Bidirectional self-attention: B = L = m, no causal mask.
            gpu.attention_dflash_f32(&q, &k, &v, &attn_out, m, m, n_heads, n_kv_heads, head_dim)?;
            if trace_phases {
                gpu.device_synchronize()?;
                attention_core_ms += stage_started.elapsed().as_secs_f64() * 1e3;
            }
        }

        let stage_started = std::time::Instant::now();
        if attention_boundary != AttentionBoundary::OutputProjected {
            projector.project(
                gpu,
                layer_idx,
                Projection::AttentionOutput,
                &layer.wo,
                &attn_out,
                &o,
                m,
            )?;
        }
        gpu.rmsnorm_batched(&o, &layer.post_attn_norm, &tmp, m, dim, eps)?;
        gpu.add_f32(&x_batch, &tmp, &x_batch)?;
        if trace_phases {
            gpu.device_synchronize()?;
            attention_output_ms += stage_started.elapsed().as_secs_f64() * 1e3;
        }

        // ── FFN block (GeGLU) ──
        let stage_started = std::time::Instant::now();
        gpu.rmsnorm_batched(&x_batch, &layer.pre_ffn_norm, &tmp, m, dim, eps)?;
        if completed_layer && compare_this_layer {
            if let Some(resident_hidden) = projector.take_layer_debug_hidden() {
                gpu.device_synchronize()?;
                let fallback_hidden = gpu.download_f32(&tmp)?;
                let (cosine, max_abs) = tensor_metrics(&resident_hidden, &fallback_hidden);
                let min_row = min_row_cosine(&resident_hidden, &fallback_hidden, dim);
                eprintln!(
                    "embeddinggemma_resident_hidden_compare layer={layer_idx} cosine={cosine:.8} min_row_cosine={min_row:.8} max_abs={max_abs:.7}"
                );
            }
        }
        projector.project_ffn(
            gpu,
            layer_idx,
            &layer.w_gate,
            &layer.w_up,
            &layer.w_down,
            &tmp,
            &gate,
            &up,
            &ffn,
            &o,
            m,
        )?;
        if completed_layer && compare_this_layer {
            if let Some(resident_ffn) = projector.take_layer_debug_ffn() {
                gpu.device_synchronize()?;
                let fallback_ffn = gpu.download_f32(&o)?;
                let (cosine, max_abs) = tensor_metrics(&resident_ffn, &fallback_ffn);
                let min_row = min_row_cosine(&resident_ffn, &fallback_ffn, dim);
                eprintln!(
                    "embeddinggemma_resident_ffn_compare layer={layer_idx} cosine={cosine:.8} min_row_cosine={min_row:.8} max_abs={max_abs:.7}"
                );
            }
        }
        if trace_phases {
            gpu.device_synchronize()?;
            ffn_core_ms += stage_started.elapsed().as_secs_f64() * 1e3;
        }
        let stage_started = std::time::Instant::now();
        gpu.rmsnorm_batched(&o, &layer.post_ffn_norm, &tmp, m, dim, eps)?;
        gpu.add_f32(&x_batch, &tmp, &x_batch)?;
        if completed_layer && compare_this_layer {
            gpu.device_synchronize()?;
            let resident = gpu.download_f32(compare_output.as_ref().expect("comparison output"))?;
            let fallback = gpu.download_f32(&x_batch)?;
            let (cosine, max_abs) = tensor_metrics(&resident, &fallback);
            let min_row = min_row_cosine(&resident, &fallback, dim);
            eprintln!(
                "embeddinggemma_resident_layer_compare layer={layer_idx} cosine={cosine:.8} min_row_cosine={min_row:.8} max_abs={max_abs:.7}"
            );
        }
        if trace_phases {
            gpu.device_synchronize()?;
            ffn_output_ms += stage_started.elapsed().as_secs_f64() * 1e3;
        }
    }

    // Final norm (model.norm) → the ST Transformer's last_hidden_state.
    gpu.rmsnorm_batched(&x_batch, &backbone.output_norm, &tmp, m, dim, eps)?;
    let hidden_flat = gpu.download_f32(&tmp)?;
    if trace_phases {
        eprintln!(
            "embeddinggemma_phase_trace rows={m} layers={} qkv_ms={qkv_ms:.3} attention_core_ms={attention_core_ms:.3} attention_output_ms={attention_output_ms:.3} ffn_core_ms={ffn_core_ms:.3} ffn_output_ms={ffn_output_ms:.3}",
            cfg.num_hidden_layers,
        );
    }
    gpu.reclaim_pending();

    Ok(pool(&hidden_flat, m, dim, cfg.pooling_mode))
}

fn tensor_metrics(left: &[f32], right: &[f32]) -> (f64, f32) {
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&left, &right) in left.iter().zip(right) {
        dot += left as f64 * right as f64;
        left_norm += (left as f64).powi(2);
        right_norm += (right as f64).powi(2);
        max_abs = max_abs.max((left - right).abs());
    }
    (dot / (left_norm.sqrt() * right_norm.sqrt()), max_abs)
}

fn min_row_cosine(left: &[f32], right: &[f32], width: usize) -> f64 {
    left.chunks_exact(width)
        .zip(right.chunks_exact(width))
        .map(|(left, right)| tensor_metrics(left, right).0)
        .fold(f64::INFINITY, f64::min)
}

/// Reduce the `[m, dim]` final hidden states to one `[dim]` vector. embeddinggemma
/// uses mean pooling with `include_prompt = true`, so every position participates.
fn pool(hidden_flat: &[f32], m: usize, dim: usize, mode: PoolingMode) -> Vec<f32> {
    match mode {
        PoolingMode::Mean => {
            let mut acc = vec![0.0f32; dim];
            for row in 0..m {
                let base = row * dim;
                for (d, a) in acc.iter_mut().enumerate() {
                    *a += hidden_flat[base + d];
                }
            }
            let inv = 1.0 / m as f32;
            for a in &mut acc {
                *a *= inv;
            }
            acc
        }
        PoolingMode::LastToken => hidden_flat[(m - 1) * dim..m * dim].to_vec(),
        PoolingMode::Cls => hidden_flat[0..dim].to_vec(),
    }
}

/// L2-normalize in place (no-op on a zero vector).
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_averages_rows() {
        // 2 rows × 3 dims.
        let h = [1.0, 2.0, 3.0, 3.0, 4.0, 5.0];
        assert_eq!(pool(&h, 2, 3, PoolingMode::Mean), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn last_token_pool_takes_final_row() {
        let h = [1.0, 2.0, 3.0, 3.0, 4.0, 5.0];
        assert_eq!(pool(&h, 2, 3, PoolingMode::LastToken), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn l2_normalize_unit_length() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((n - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dense_capture_observes_inputs_in_projection_order() {
        use crate::weights::DenseHeadHost;

        let dense_heads = vec![
            DenseHeadHost {
                in_features: 2,
                out_features: 2,
                w: vec![1.0, 0.0, 0.0, 2.0],
                awq_scale: None,
            },
            DenseHeadHost {
                in_features: 2,
                out_features: 1,
                w: vec![1.0, 1.0],
                awq_scale: None,
            },
        ];
        let mut captured = Vec::new();
        let output = project_dense_with_capture(&dense_heads, vec![3.0, 4.0], |idx, input| {
            captured.push((idx, input.to_vec()));
            Ok(())
        })
        .unwrap();
        assert_eq!(captured, vec![(0, vec![3.0, 4.0]), (1, vec![3.0, 8.0])]);
        assert!((output[0] - 1.0).abs() < 1e-6);
    }
}
