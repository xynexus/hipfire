// SPDX-License-Identifier: Apache-2.0
// hipfire — embeddinggemma config parsing. See LICENSE / NOTICE.

//! [`EmbeddingGemmaConfig`] and the HFQ-metadata parser.
//!
//! embeddinggemma-300m shares the Gemma-3 text-decoder shape (GQA, per-head
//! QK-norm, 4 norms/layer, GeGLU, dual-θ sliding-window interleave, the `(1+w)`
//! norm-offset baked at ingest), with three deltas that make it an *encoder*:
//!
//! 1. **Bidirectional attention** — no causal mask; every token attends to every
//!    other token (within the sliding window for local layers).
//! 2. **Mean pooling** — the sentence vector is the (attention-mask-weighted) mean
//!    of the final-layer hidden states, not a per-position logit.
//! 3. **Dense projection head(s)** — the sentence-transformers Matryoshka bottleneck
//!    applied to the pooled vector, then L2 normalization.
//!
//! The quantizer embeds the original `config.json` under `metadata.config` (the
//! Gemma-3 shape) and the sentence-transformers post-processing under
//! `metadata.sentence_transformers` (pooling mode, Dense head shapes/activations,
//! Matryoshka dims, task prompts). This parser reads both.

/// One sentence-transformers `Dense` projection head: `y = act(x · Wᵀ + b)`.
/// `weight` is stored row-major `[out_features, in_features]` (torch `nn.Linear`).
#[derive(Debug, Clone, PartialEq)]
pub struct DenseHead {
    pub in_features: usize,
    pub out_features: usize,
    pub has_bias: bool,
    /// `"identity"` (embeddinggemma) or `"tanh"`/`"gelu"` — validated at load.
    pub activation: String,
}

/// How the per-token hidden states are reduced to one sentence vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingMode {
    /// Attention-mask-weighted mean over positions (embeddinggemma default).
    Mean,
    /// Hidden state of the final non-pad position.
    LastToken,
    /// Hidden state of the first position (CLS).
    Cls,
}

impl PoolingMode {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "mean" | "pooling_mode_mean_tokens" => Some(Self::Mean),
            "lasttoken" | "last_token" | "pooling_mode_lasttoken" => Some(Self::LastToken),
            "cls" | "pooling_mode_cls_token" => Some(Self::Cls),
            _ => None,
        }
    }
}

/// embeddinggemma shape + post-processing constants. Cheap to clone, `Send`.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingGemmaConfig {
    // ── Gemma-3 backbone shape (identical fields to Gemma3Config) ──
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_local_base_freq: f32,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub query_pre_attn_scalar: f32,
    pub hidden_activation: String,
    /// The `(1+w)` offset the quantizer baked into the norm weights at ingest.
    pub gemma_norm_offset: f32,

    // ── Encoder / sentence-transformers post-processing ──
    /// Bidirectional (non-causal) attention. Always true for embeddinggemma; kept
    /// explicit so a future causal sibling can reuse this crate.
    pub bidirectional: bool,
    pub pooling_mode: PoolingMode,
    /// Ordered Dense projection heads applied to the pooled vector.
    pub dense_heads: Vec<DenseHead>,
    /// Native embedding dimension (output of the last Dense head, e.g. 768).
    pub embedding_dim: usize,
    /// Valid Matryoshka truncation lengths, descending (e.g. [768,512,256,128]).
    pub matryoshka_dims: Vec<usize>,
    /// Instruction prefix prepended to a **query** before encoding. embeddinggemma
    /// was trained with `"task: search result | query: "`; omitting it degrades
    /// retrieval/rerank quality. Empty ⇒ no prefix.
    pub query_prompt: String,
    /// Instruction prefix prepended to a **document/passage** before encoding
    /// (embeddinggemma: `"title: none | text: "`).
    pub document_prompt: String,
}

impl EmbeddingGemmaConfig {
    /// Embedding normalizer — Gemma scales the looked-up embedding by √hidden_size.
    pub fn embed_scale(&self) -> f32 {
        (self.hidden_size as f32).sqrt()
    }

    /// Pre-scale applied to Q so the attention kernel's built-in `1/√head_dim`
    /// softmax scale equals Gemma's `1/√query_pre_attn_scalar`. Baked into q_norm
    /// at load, exactly as gemma3 does.
    pub fn q_prescale(&self) -> f32 {
        (self.head_dim as f32 / self.query_pre_attn_scalar).sqrt()
    }

    /// Per-layer RoPE base (global θ on every `sliding_window_pattern`-th layer,
    /// local θ otherwise). Matches gemma3's interleave.
    pub fn is_global_layer(&self, layer_idx: usize) -> bool {
        self.sliding_window_pattern > 0
            && (layer_idx + 1).is_multiple_of(self.sliding_window_pattern)
    }

    pub fn rope_base_for_layer(&self, layer_idx: usize) -> f32 {
        if self.is_global_layer(layer_idx) {
            self.rope_theta
        } else {
            self.rope_local_base_freq
        }
    }

    /// The largest embedding length this artifact can emit.
    pub fn max_output_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Clamp a requested Matryoshka dim to a supported length (the largest valid
    /// dim ≤ request, or the native dim if the request is absent/too large).
    pub fn resolve_dims(&self, requested: Option<usize>) -> usize {
        match requested {
            None => self.embedding_dim,
            Some(d) => {
                if d >= self.embedding_dim {
                    self.embedding_dim
                } else if self.matryoshka_dims.contains(&d) {
                    d
                } else {
                    // Snap down to the largest supported dim ≤ request.
                    self.matryoshka_dims
                        .iter()
                        .copied()
                        .filter(|&m| m <= d)
                        .max()
                        .unwrap_or(self.embedding_dim)
                }
            }
        }
    }
}

/// Parse an [`EmbeddingGemmaConfig`] from an HFQ file's embedded metadata JSON.
pub fn config_from_metadata_json(metadata_json: &str) -> Option<EmbeddingGemmaConfig> {
    let meta: serde_json::Value = serde_json::from_str(metadata_json).ok()?;
    let config = meta.get("config")?;
    // embeddinggemma is `gemma3_text`-shaped; tolerate a `text_config` nest too.
    let tc = config.get("text_config").unwrap_or(config);

    let hidden_size = tc.get("hidden_size")?.as_u64()? as usize;
    let num_hidden_layers = tc.get("num_hidden_layers")?.as_u64()? as usize;
    let num_attention_heads = tc.get("num_attention_heads")?.as_u64()? as usize;
    let num_key_value_heads = tc
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(num_attention_heads as u64) as usize;
    let head_dim = tc
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(hidden_size / num_attention_heads);
    let intermediate_size = tc.get("intermediate_size")?.as_u64()? as usize;
    let vocab_size = tc.get("vocab_size")?.as_u64()? as usize;
    let max_position_embeddings = tc
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(2048) as usize;
    let rms_norm_eps = tc
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;
    let rope_theta = tc
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(1_000_000.0) as f32;
    let rope_local_base_freq = tc
        .get("rope_local_base_freq")
        .and_then(|v| v.as_f64())
        .unwrap_or(10_000.0) as f32;
    let sliding_window = tc
        .get("sliding_window")
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as usize;
    let sliding_window_pattern = tc
        .get("sliding_window_pattern")
        .and_then(|v| v.as_u64())
        .unwrap_or(6) as usize;
    let query_pre_attn_scalar = tc
        .get("query_pre_attn_scalar")
        .and_then(|v| v.as_f64())
        .unwrap_or(head_dim as f64) as f32;
    let hidden_activation = tc
        .get("hidden_activation")
        .and_then(|v| v.as_str())
        .unwrap_or("gelu_pytorch_tanh")
        .to_string();
    let gemma_norm_offset = meta
        .get("gemma_norm_offset")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;

    // ── sentence-transformers post-processing block ──
    // Accept it as a top-level metadata sibling (`metadata.sentence_transformers`)
    // or nested inside the embedded config (`metadata.config.sentence_transformers`),
    // which is where the importer's prep step folds it.
    let st = meta
        .get("sentence_transformers")
        .or_else(|| config.get("sentence_transformers"));
    let pooling_mode = st
        .and_then(|s| s.get("pooling_mode"))
        .and_then(|v| v.as_str())
        .and_then(PoolingMode::from_str)
        .unwrap_or(PoolingMode::Mean);

    let dense_heads: Vec<DenseHead> = st
        .and_then(|s| s.get("dense"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|d| {
                    Some(DenseHead {
                        in_features: d.get("in_features")?.as_u64()? as usize,
                        out_features: d.get("out_features")?.as_u64()? as usize,
                        has_bias: d.get("bias").and_then(|v| v.as_bool()).unwrap_or(false),
                        activation: d
                            .get("activation")
                            .and_then(|v| v.as_str())
                            .unwrap_or("identity")
                            .to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    // Native embedding dim = output of the last Dense head, else the pooled hidden.
    let embedding_dim = dense_heads
        .last()
        .map(|d| d.out_features)
        .unwrap_or(hidden_size);

    let query_prompt = st
        .and_then(|s| s.get("query_prompt"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let document_prompt = st
        .and_then(|s| s.get("document_prompt"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let matryoshka_dims: Vec<usize> = st
        .and_then(|s| s.get("matryoshka_dims"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            let mut d: Vec<usize> = arr
                .iter()
                .filter_map(|x| x.as_u64().map(|v| v as usize))
                .collect();
            d.sort_unstable_by(|a, b| b.cmp(a)); // descending
            d
        })
        .filter(|d| !d.is_empty())
        .unwrap_or_else(|| vec![embedding_dim]);

    Some(EmbeddingGemmaConfig {
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        max_position_embeddings,
        rms_norm_eps,
        rope_theta,
        rope_local_base_freq,
        sliding_window,
        sliding_window_pattern,
        query_pre_attn_scalar,
        hidden_activation,
        gemma_norm_offset,
        bidirectional: true,
        pooling_mode,
        dense_heads,
        embedding_dim,
        matryoshka_dims,
        query_prompt,
        document_prompt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// google/embeddinggemma-300m shape (verified against the model card), wrapped
    /// in the HFQ metadata envelope the quantizer emits, including the ST block.
    fn embeddinggemma_300m_metadata() -> String {
        serde_json::json!({
            "architecture": "embeddinggemma",
            "gemma_norm_offset": 1.0,
            "config": {
                "model_type": "gemma3_text",
                "hidden_size": 768,
                "num_hidden_layers": 24,
                "num_attention_heads": 3,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "intermediate_size": 1152,
                "vocab_size": 262144,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1_000_000.0,
                "rope_local_base_freq": 10_000.0,
                "sliding_window": 512,
                "sliding_window_pattern": 6,
                "query_pre_attn_scalar": 256,
                "hidden_activation": "gelu_pytorch_tanh",
                "max_position_embeddings": 2048
            },
            "sentence_transformers": {
                "pooling_mode": "mean",
                "dense": [
                    {"in_features": 768, "out_features": 3072, "bias": false, "activation": "identity"},
                    {"in_features": 3072, "out_features": 768, "bias": false, "activation": "identity"}
                ],
                "matryoshka_dims": [768, 512, 256, 128]
            }
        })
        .to_string()
    }

    #[test]
    fn parses_embeddinggemma_300m() {
        let cfg = config_from_metadata_json(&embeddinggemma_300m_metadata()).unwrap();
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 3);
        assert_eq!(cfg.num_key_value_heads, 1);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.vocab_size, 262144);
        assert!(cfg.bidirectional);
        assert_eq!(cfg.pooling_mode, PoolingMode::Mean);
        assert_eq!(cfg.dense_heads.len(), 2);
        assert_eq!(cfg.embedding_dim, 768);
        assert_eq!(cfg.matryoshka_dims, vec![768, 512, 256, 128]);
        assert_eq!(cfg.gemma_norm_offset, 1.0);
    }

    #[test]
    fn q_prescale_is_one_when_scalar_equals_head_dim() {
        let cfg = config_from_metadata_json(&embeddinggemma_300m_metadata()).unwrap();
        // 300m: query_pre_attn_scalar == head_dim == 256 ⇒ prescale 1.0.
        assert!((cfg.q_prescale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn resolve_dims_snaps_to_supported_matryoshka() {
        let cfg = config_from_metadata_json(&embeddinggemma_300m_metadata()).unwrap();
        assert_eq!(cfg.resolve_dims(None), 768);
        assert_eq!(cfg.resolve_dims(Some(768)), 768);
        assert_eq!(cfg.resolve_dims(Some(256)), 256);
        assert_eq!(cfg.resolve_dims(Some(300)), 256); // snap down to nearest supported
        assert_eq!(cfg.resolve_dims(Some(4096)), 768); // clamp to native
    }

    #[test]
    fn defaults_to_mean_pool_no_dense_when_st_block_absent() {
        let meta = serde_json::json!({
            "architecture": "embeddinggemma",
            "config": { "hidden_size": 768, "num_hidden_layers": 2,
                "num_attention_heads": 3, "intermediate_size": 1152, "vocab_size": 262144 }
        })
        .to_string();
        let cfg = config_from_metadata_json(&meta).unwrap();
        assert_eq!(cfg.pooling_mode, PoolingMode::Mean);
        assert!(cfg.dense_heads.is_empty());
        assert_eq!(cfg.embedding_dim, 768); // falls back to pooled hidden size
    }
}
