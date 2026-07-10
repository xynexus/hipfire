// SPDX-License-Identifier: Apache-2.0
// hipfire — embeddinggemma weight loading. See LICENSE / NOTICE.

//! GPU-resident embeddinggemma weights.
//!
//! The transformer backbone is byte-for-byte a Gemma-3 text decoder, so we reuse
//! [`hipfire_arch_gemma3::load_weights`] verbatim (embedding table, 4-norm layers,
//! per-head QK-norm, GeGLU, the `(1+w)` norm-offset + `q_prescale` bakes). The only
//! embeddinggemma-specific tensors are the sentence-transformers **Dense** heads.
//!
//! The Dense heads are tiny (`768×3072` + `3072×768`) and are applied to a single
//! pooled vector, so we keep them **host-resident `Vec<f32>`** and run the two
//! projections on the CPU (see `forward::project_dense`). That avoids a GPU GEMM /
//! pooling kernel entirely — the only device work is the 24 transformer layers.
//!
//! We reuse Gemma-3's loader with `tie_word_embeddings = true`, which re-uploads the
//! embedding bytes as a (never-used) `lm_head`. That is a known ~embed-table memory
//! waste on an encoder that has no output projection; acceptable for bring-up and
//! flagged for a lean-loader follow-up.

use hipfire_arch_gemma3::config::Gemma3Config;
use hipfire_arch_gemma3::weights::Gemma3Weights;
use hipfire_rdna::Gpu;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::quant::f16_to_f32;

use crate::config::EmbeddingGemmaConfig;

/// One host-resident Dense projection head: `y = x · Wᵀ` (Identity activation,
/// no bias for embeddinggemma). `w` is row-major `[out_features, in_features]`.
pub struct DenseHeadHost {
    pub in_features: usize,
    pub out_features: usize,
    pub w: Vec<f32>,
    pub awq_scale: Option<Vec<f32>>,
}

impl DenseHeadHost {
    /// `y[o] = Σ_i x[i] · W[o, i]`. Panics if `x.len() != in_features`.
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_features, "dense head input dim mismatch");
        let mut y = vec![0.0f32; self.out_features];
        for (o, y_o) in y.iter_mut().enumerate() {
            let row = &self.w[o * self.in_features..(o + 1) * self.in_features];
            *y_o = row
                .iter()
                .zip(x)
                .enumerate()
                .map(|(i, (w, xi))| {
                    let input = self.awq_scale.as_ref().map_or(*xi, |scale| *xi / scale[i]);
                    w * input
                })
                .sum();
        }
        y
    }
}

/// GPU-resident embeddinggemma weights: the Gemma-3 backbone plus the host-side
/// Dense projection heads.
pub struct EmbeddingGemmaWeights {
    pub backbone: Gemma3Weights,
    pub dense_heads: Vec<DenseHeadHost>,
    pub(crate) host_embedding: Option<HostEmbedding>,
}

pub(crate) enum HostEmbedding {
    F16(Vec<u8>),
    Bf16(Vec<u8>),
}

impl HostEmbedding {
    pub(crate) fn row(&self, token: u32, dim: usize) -> Result<Vec<f32>, String> {
        let start = token as usize * dim * 2;
        let data = match self {
            Self::F16(data) | Self::Bf16(data) => data,
        };
        let end = start + dim * 2;
        if end > data.len() {
            return Err(format!(
                "embeddinggemma: token {token} exceeds host embedding table"
            ));
        }
        Ok(data[start..end]
            .chunks_exact(2)
            .map(|bytes| {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                match self {
                    Self::F16(_) => f16_to_f32(bits),
                    Self::Bf16(_) => f32::from_bits((bits as u32) << 16),
                }
            })
            .collect())
    }
}

impl EmbeddingGemmaWeights {
    pub fn load(
        hfq: &mut HfqFile,
        cfg: &EmbeddingGemmaConfig,
        gpu: &mut Gpu,
    ) -> Result<Self, String> {
        let g3 = gemma3_config(cfg);
        let backbone = Gemma3Weights::load(hfq, &g3, gpu)?;

        let mut dense_heads = Vec::with_capacity(cfg.dense_heads.len());
        for (i, h) in cfg.dense_heads.iter().enumerate() {
            // Importer stores each ST Dense head as `dense.{i}.weight`, row-major
            // `[out_features, in_features]`, in f16/f32/bf16 (kept near-lossless).
            let name = format!("dense.{i}.weight");
            let w = load_dense_head_f32(hfq, &name, h.out_features, h.in_features)?;
            if h.activation != "identity" {
                return Err(format!(
                    "embeddinggemma: Dense head {i} activation {:?} unsupported \
                     (only Identity is implemented)",
                    h.activation
                ));
            }
            dense_heads.push(DenseHeadHost {
                in_features: h.in_features,
                out_features: h.out_features,
                w,
                awq_scale: load_dense_awq_scale(hfq, &name, h.in_features)?,
            });
        }

        Ok(Self {
            backbone,
            dense_heads,
            host_embedding: None,
        })
    }

    pub fn load_for_calibration(
        hfq: &mut HfqFile,
        cfg: &EmbeddingGemmaConfig,
        gpu: &mut Gpu,
    ) -> Result<Self, String> {
        let (embedding_info, embedding_data) = hfq
            .tensor_data_vec("model.embed_tokens.weight")
            .ok_or_else(|| "embeddinggemma: model.embed_tokens.weight not found".to_string())?;
        let host_embedding = match embedding_info.quant_type {
            1 => HostEmbedding::F16(embedding_data),
            16 => HostEmbedding::Bf16(embedding_data),
            quant_type => {
                return Err(format!(
                    "embeddinggemma calibration: host embedding requires f16/bf16 source, got qt={quant_type}"
                ))
            }
        };
        let g3 = gemma3_config(cfg);
        let backbone =
            hipfire_arch_gemma3::weights::load_encoder_weights_prefixed(hfq, &g3, gpu, "")
                .map_err(|e| format!("embeddinggemma: load encoder weights failed: {e:?}"))?;

        let mut dense_heads = Vec::with_capacity(cfg.dense_heads.len());
        for (i, head) in cfg.dense_heads.iter().enumerate() {
            let name = format!("dense.{i}.weight");
            let weights = load_dense_head_f32(hfq, &name, head.out_features, head.in_features)?;
            if head.activation != "identity" {
                return Err(format!(
                    "embeddinggemma: Dense head {i} activation {:?} unsupported (only Identity is implemented)",
                    head.activation
                ));
            }
            dense_heads.push(DenseHeadHost {
                in_features: head.in_features,
                out_features: head.out_features,
                w: weights,
                awq_scale: load_dense_awq_scale(hfq, &name, head.in_features)?,
            });
        }

        Ok(Self {
            backbone,
            dense_heads,
            host_embedding: Some(host_embedding),
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.backbone.free_gpu(gpu);
    }
}

/// Project a Gemma-3-shaped [`EmbeddingGemmaConfig`] down to the backbone
/// [`Gemma3Config`] the reused loader/forward consume. `tie_word_embeddings=true`
/// makes the loader tolerate the absent `lm_head` (re-uploads the embed bytes).
pub fn gemma3_config(cfg: &EmbeddingGemmaConfig) -> Gemma3Config {
    Gemma3Config {
        hidden_size: cfg.hidden_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        intermediate_size: cfg.intermediate_size,
        vocab_size: cfg.vocab_size,
        max_position_embeddings: cfg.max_position_embeddings,
        rms_norm_eps: cfg.rms_norm_eps,
        rope_theta: cfg.rope_theta,
        rope_local_base_freq: cfg.rope_local_base_freq,
        sliding_window: cfg.sliding_window,
        sliding_window_pattern: cfg.sliding_window_pattern,
        query_pre_attn_scalar: cfg.query_pre_attn_scalar,
        hidden_activation: cfg.hidden_activation.clone(),
        tie_word_embeddings: true,
        gemma_norm_offset: cfg.gemma_norm_offset,
        eos_token_id: 1,
    }
}

/// Load a Dense head weight matrix to a host `Vec<f32>` (dequantizing f16/bf16).
/// The head is small and correctness-critical, so it never lives on GPU.
fn load_dense_head_f32(
    hfq: &HfqFile,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>, String> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("embeddinggemma: Dense tensor not found: {name}"))?;
    let expected = out_features * in_features;
    let w: Vec<f32> = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        16 => data
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        34 => hipfire_runtime::quant::dequant_oq4g256(&data, expected),
        35 => hipfire_runtime::quant::dequant_oq8g256(&data, expected),
        36 => hipfire_runtime::quant::dequant_oqplus_compact(&data, out_features, in_features),
        qt => {
            return Err(format!(
                "embeddinggemma: Dense head {name} must be f16/f32/bf16/OQ4/OQ4.25/OQ8, got qt={qt}"
            ))
        }
    };
    if w.len() != expected {
        return Err(format!(
            "embeddinggemma: Dense head {name} has {} elements, expected {expected} \
             ({out_features}×{in_features})",
            w.len()
        ));
    }
    Ok(w)
}

fn load_dense_awq_scale(
    hfq: &HfqFile,
    weight_name: &str,
    in_features: usize,
) -> Result<Option<Vec<f32>>, String> {
    let sidecar_name = weight_name.strip_suffix(".weight").map_or_else(
        || format!("{weight_name}.awq_scale.weight"),
        |stem| format!("{stem}.awq_scale.weight"),
    );
    let Some((info, data)) = hfq.tensor_data_vec(&sidecar_name) else {
        return Ok(None);
    };
    if info.quant_type != 1 || data.len() != in_features * 2 {
        return Err(format!(
            "embeddinggemma: Dense AWQ sidecar {sidecar_name} must be f16[{in_features}]"
        ));
    }
    Ok(Some(
        data.chunks_exact(2)
            .map(|bytes| f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_head_applies_identity_matmul() {
        // 2×3 matrix, row-major [out=2, in=3].
        let h = DenseHeadHost {
            in_features: 3,
            out_features: 2,
            w: vec![1.0, 0.0, 0.0, /* row0 */ 0.0, 1.0, 1.0 /* row1 */],
            awq_scale: None,
        };
        let y = h.apply(&[2.0, 3.0, 4.0]);
        assert_eq!(y, vec![2.0, 7.0]);
    }

    #[test]
    fn dense_head_applies_awq_input_scale() {
        let head = DenseHeadHost {
            in_features: 2,
            out_features: 1,
            w: vec![4.0, 9.0],
            awq_scale: Some(vec![2.0, 3.0]),
        };
        assert_eq!(head.apply(&[2.0, 3.0]), vec![13.0]);
    }
}
