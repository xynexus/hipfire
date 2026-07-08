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
use hipfire_rdna::{DType, Gpu};
use hipfire_runtime::weights::weight_gemm;

use crate::config::{EmbeddingGemmaConfig, PoolingMode};
use crate::weights::EmbeddingGemmaWeights;

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
    let hidden = encode_pooled_hidden(gpu, weights, cfg, tokens)
        .map_err(|e| format!("embeddinggemma: encode failed: {e:?}"))?;

    // Sentence-transformers head: Dense projections (Identity activation) then L2.
    let mut v = hidden;
    for head in &weights.dense_heads {
        v = head.apply(&v);
    }
    l2_normalize(&mut v);
    Ok(v)
}

/// Run the bidirectional transformer stack + final norm + mean pooling, returning
/// the pooled (but not yet projected) hidden vector `[hidden_size]`.
fn encode_pooled_hidden(
    gpu: &mut Gpu,
    weights: &EmbeddingGemmaWeights,
    cfg: &EmbeddingGemmaConfig,
    tokens: &[u32],
) -> HipResult<Vec<f32>> {
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
        hipfire_arch_gemma3::forward::embed_token(gpu, backbone, &g3, &embed_tmp, tok)?;
        gpu.memcpy_dtod_at_auto(&x_batch.buf, i * dim * 4, &embed_tmp.buf, 0, dim * 4)?;
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

    for layer_idx in 0..cfg.num_hidden_layers {
        let layer = &backbone.layers[layer_idx];

        // ── Attention block (bidirectional) ──
        gpu.rmsnorm_batched(&x_batch, &layer.input_norm, &tmp, m, dim, eps)?;
        weight_gemm(gpu, &layer.wq, &tmp, &q, m)?;
        weight_gemm(gpu, &layer.wk, &tmp, &k, m)?;
        weight_gemm(gpu, &layer.wv, &tmp, &v, m)?;

        // Per-head QK-norm (q_norm carries the baked Q pre-scale, 1.0 here).
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
        gpu.attention_dflash_f32(
            &q, &k, &v, &attn_out, m, m, n_heads, n_kv_heads, head_dim,
        )?;

        weight_gemm(gpu, &layer.wo, &attn_out, &o, m)?;
        gpu.rmsnorm_batched(&o, &layer.post_attn_norm, &tmp, m, dim, eps)?;
        gpu.add_f32(&x_batch, &tmp, &x_batch)?;

        // ── FFN block (GeGLU) ──
        gpu.rmsnorm_batched(&x_batch, &layer.pre_ffn_norm, &tmp, m, dim, eps)?;
        weight_gemm(gpu, &layer.w_gate, &tmp, &gate, m)?;
        weight_gemm(gpu, &layer.w_up, &tmp, &up, m)?;
        gpu.gelu_mul_f32(&gate, &up, &ffn)?;
        weight_gemm(gpu, &layer.w_down, &ffn, &o, m)?;
        gpu.rmsnorm_batched(&o, &layer.post_ffn_norm, &tmp, m, dim, eps)?;
        gpu.add_f32(&x_batch, &tmp, &x_batch)?;
    }

    // Final norm (model.norm) → the ST Transformer's last_hidden_state.
    gpu.rmsnorm_batched(&x_batch, &backbone.output_norm, &tmp, m, dim, eps)?;
    let hidden_flat = gpu.download_f32(&tmp)?;
    gpu.reclaim_pending();

    Ok(pool(&hidden_flat, m, dim, cfg.pooling_mode))
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
}
