// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.8-Flash-Next (`qwen4_exp`) config.
//!
//! Parses the nested `text_config` and derives the geometry the loader and the
//! forward pass need. Every default and every derived quantity here was checked
//! against the vendored reference (`third_party/transformers-qwen4_exp/`), because
//! four of them are silent-wrong if guessed:
//!
//! * **`ple_layer_ids` is ONE-BASED.** The reference matches on
//!   `ple_layer_ids.index(layer_idx + 1)`, which is why the shipped checkpoint's
//!   `[2]` names its tensors `layers.1.ple.*`. Off by one here injects the n-gram
//!   features into the wrong layer and still runs.
//! * **`norm_topk_prob` defaults TRUE** (`configuration_qwen4_exp.py:163`), and the
//!   shipped `config.json` omits the key — so the default is load-bearing.
//! * **`output_gate_type` names the GATED-RMSNORM activation** on the DeltaNet
//!   output gate, not the attention gate. The reference feeds
//!   `config.output_gate_type or config.hidden_act` into `Qwen4ExpTextRMSNormGated`.
//! * **The indexer budget is in TOKENS**; the selection is over BLOCKS, and
//!   `block_topk = indexer_budget / indexer_compress_ratio`.
//!
//! The mrope section is also validated rather than trusted: it must sum to half the
//! rotary dimension, which is the invariant that catches a `partial_rotary_factor`
//! and `mrope_section` that disagree.

use serde_json::Value;

/// Which mixer a layer uses. The stack is a repeating 3:1 pattern — three
/// `LinearAttention` (Gated DeltaNet) layers then one `SparseAttention`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Gated DeltaNet: a recurrent linear-attention mixer with a short conv.
    LinearAttention,
    /// Qwen Sparse Attention: full attention gated by the micro-block indexer.
    SparseAttention,
}

/// The hashed n-gram embedding block, present on exactly one layer.
#[derive(Debug, Clone, PartialEq)]
pub struct NgramConfig {
    /// ZERO-BASED model layer this block rides. `ple_layer_ids` in the file is
    /// one-based, so `[2]` means layer 1.
    pub layer_idx: usize,
    /// Position of this block WITHIN `ple_layer_ids` — the reference's
    /// `ple_layer_index`, from `ple_layer_ids.index(layer_idx + 1)`.
    ///
    /// Distinct from [`Self::layer_idx`] and easy to conflate: with
    /// `ple_layer_ids = [2]` the block sits on layer 1 but is PLE block 0. It is
    /// this ordinal — not the layer — that seeds the hash multipliers and offsets
    /// the prime ladder (`global_head_idx = ple_index * heads + head_idx`), so
    /// swapping the two silently produces a different, wrong addressing scheme.
    pub ple_index: usize,
    /// Width of the concatenated n-gram embedding (all heads).
    pub embed_dim: usize,
    /// Longest n-gram order. Orders `2..=ngram_size` each get their own heads.
    pub ngram_size: usize,
    /// Hash heads per order.
    pub heads_per_ngram: usize,
    /// Base for the per-head prime moduli.
    pub vocab_size_base: u64,
    /// The padded row total is rounded up to a multiple of this.
    pub divisible_by: u64,
    /// How many shards the flat table is split into on disk.
    pub shards: usize,
    /// Depthwise conv kernel width over the injected features.
    pub conv_kernel: usize,
    /// Seed for the derived hash multipliers.
    pub seed: u64,
}

impl NgramConfig {
    /// Total hash heads: one set per n-gram ORDER, and the orders run `2..=ngram_size`.
    pub fn heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }
    /// Dimensions each head contributes; the heads concatenate to `embed_dim`.
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.heads()
    }
    /// Tokens of history the hash needs, i.e. `ngram_size - 1` predecessors.
    pub fn context_len(&self) -> usize {
        self.ngram_size - 1
    }
}

/// The micro-block sparse-attention indexer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexerConfig {
    pub n_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Budget in TOKENS.
    pub budget: usize,
    /// Tokens per micro-block.
    pub compress_ratio: usize,
}

impl IndexerConfig {
    /// Blocks selected per query. The budget is expressed in tokens and the
    /// selection is over blocks, so the two differ by `compress_ratio`.
    pub fn block_topk(&self) -> usize {
        self.budget / self.compress_ratio
    }
    /// Output width of the single fused query+key projection.
    pub fn qk_proj_out(&self) -> usize {
        (self.n_heads + self.kv_heads) * self.head_dim
    }
}

/// The 4-wide gated residual (hyper-connections).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatedResidualConfig {
    /// Number of residual streams.
    pub count: usize,
    /// Rank of the low-rank read-side mix.
    pub lowrank: usize,
}

/// Gated DeltaNet geometry. Q and K share `key_heads`; V has its own, larger count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaNetConfig {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel: usize,
    /// Activation on the output gate's gated RMSNorm.
    pub output_gate_sigmoid: bool,
}

impl DeltaNetConfig {
    /// V heads per K head — Q and K are repeat-interleaved up to the V count
    /// before the recurrence.
    pub fn value_per_key(&self) -> usize {
        self.value_heads / self.key_heads
    }
    /// Fused `in_proj_qkv` output width: Q and K at the key span, V at the value span.
    pub fn qkv_dim(&self) -> usize {
        self.key_heads * self.key_head_dim * 2 + self.value_heads * self.value_head_dim
    }
    /// `in_proj_z` output width — the output gate spans V.
    pub fn z_dim(&self) -> usize {
        self.value_heads * self.value_head_dim
    }
}

/// Routed mixture-of-experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeConfig {
    pub num_experts: usize,
    pub experts_per_tok: usize,
    pub intermediate: usize,
    pub shared_intermediate: usize,
    /// Divide the selected top-k probabilities by their sum.
    pub norm_topk_prob: bool,
}

/// The vision tower's own geometry. `None` on a text-only checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub intermediate: usize,
    pub out_hidden: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub num_position_embeddings: usize,
}

impl VisionConfig {
    /// Values per patch fed to the patch embedding.
    pub fn per_patch(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }
    /// Side length of the square learned position grid.
    pub fn num_grid_per_side(&self) -> usize {
        (self.num_position_embeddings as f64).sqrt() as usize
    }
    pub fn merge_unit(&self) -> usize {
        self.spatial_merge_size * self.spatial_merge_size
    }
    pub fn head_dim(&self) -> usize {
        self.hidden / self.n_heads
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4ExpConfig {
    pub hidden: usize,
    pub vocab: usize,
    pub layers: usize,
    pub layer_types: Vec<LayerType>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Fraction of `head_dim` that carries rotary position.
    pub partial_rotary_factor: f32,
    pub rope_theta: f32,
    pub mrope_interleaved: bool,
    pub mrope_section: Vec<usize>,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub deltanet: DeltaNetConfig,
    pub indexer: IndexerConfig,
    pub moe: MoeConfig,
    pub gated_residual: GatedResidualConfig,
    pub ngram: Option<NgramConfig>,
    /// Layers in the embedded multi-token-prediction head, if present.
    /// `max_position_embeddings` from the checkpoint.
    pub max_position: usize,
    /// End-of-turn token. Lives under `text_config` in this family (the shipped
    /// checkpoint says 248044), NOT at the config root where a flat text model
    /// would put it. Defaults to `vocab - 1` only so a fixture without one still
    /// loads; a real artifact always carries it.
    pub eos_token_id: u32,
    pub mtp_layers: usize,
    /// Whether the checkpoint carries a vision tower.
    pub has_vision: bool,
    /// Present when the checkpoint carries a vision tower.
    pub vision: Option<VisionConfig>,
}

fn usize_at(v: &Value, k: &str) -> Option<usize> {
    v.get(k)?.as_u64().map(|n| n as usize)
}
fn f32_at(v: &Value, k: &str) -> Option<f32> {
    v.get(k)?.as_f64().map(|n| n as f32)
}

impl Qwen4ExpConfig {
    /// Rotary dimensions per head. The rest of the head is plain dot product.
    pub fn rotary_dim(&self) -> usize {
        ((self.head_dim as f32) * self.partial_rotary_factor) as usize
    }
    /// Width of the widened residual stream that every block reads and writes.
    pub fn hc_hidden(&self) -> usize {
        self.hidden * self.gated_residual.count
    }
    /// `q_proj` output width. It is DOUBLE the head span because the projection
    /// carries the attention output gate interleaved with the queries — the real
    /// checkpoint's `[12288, 2560]` against `24 heads * 256 = 6144`.
    pub fn q_proj_out(&self) -> usize {
        self.n_heads * self.head_dim * 2
    }
    pub fn kv_proj_out(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Parse from a top-level `config.json` body.
    /// Parse out of an HFQ artifact's `metadata_json`.
    ///
    /// The quantizer wraps the source config in an
    /// `{"architecture":..., "config":{...}, "tokenizer":...}` envelope (the
    /// Qwen3.5 / DeepSeek-V4 pattern), so the inner `config` is the original HF
    /// config and still carries the `text_config` nesting `from_json` expects.
    ///
    /// Split from `from_hfq` so the unwrapping is testable without an `HfqFile`.
    pub fn from_metadata_json(metadata: &str) -> Result<Self, String> {
        let wrapper: Value = serde_json::from_str(metadata)
            .map_err(|e| format!("qwen4_exp: metadata_json is not valid JSON: {e}"))?;
        let inner = wrapper
            .get("config")
            .ok_or_else(|| "qwen4_exp: metadata_json has no `config` envelope".to_string())?;
        Self::from_json(inner)
    }

    pub fn from_json(root: &Value) -> Result<Self, String> {
        let text = root
            .get("text_config")
            .ok_or("qwen4_exp config has no `text_config`")?;

        let hidden = usize_at(text, "hidden_size").ok_or("missing hidden_size")?;
        let layers = usize_at(text, "num_hidden_layers").ok_or("missing num_hidden_layers")?;

        // `layer_types` is explicit in this family rather than derived from the
        // interval, so trust the list and only cross-check its length.
        let layer_types: Vec<LayerType> = text
            .get("layer_types")
            .and_then(|v| v.as_array())
            .ok_or("missing layer_types")?
            .iter()
            .map(|v| match v.as_str() {
                Some("linear_attention") => Ok(LayerType::LinearAttention),
                // Both spellings mean the same layer. The reference normalises
                // `full_attention` -> `qwen_sparse_attention` on load, commenting
                // "the real checkpoint contains `full_attention` entries for layers
                // that are actually using an indexer" — so the shipped file's
                // spelling is the legacy one and the canonical name is the other.
                // Accept both or a re-exported checkpoint stops loading.
                Some("full_attention") | Some("qwen_sparse_attention") => {
                    Ok(LayerType::SparseAttention)
                }
                other => Err(format!("unknown layer_type {other:?}")),
            })
            .collect::<Result<_, _>>()?;
        if layer_types.len() != layers {
            return Err(format!(
                "layer_types has {} entries but num_hidden_layers is {layers}",
                layer_types.len()
            ));
        }

        let head_dim = usize_at(text, "head_dim").ok_or("missing head_dim")?;
        let partial_rotary_factor = f32_at(text, "partial_rotary_factor").unwrap_or(1.0);

        // rope lives in a nested object, with `partial_rotary_factor` repeated there.
        let rope = text.get("rope_parameters");
        let rope_theta = rope
            .and_then(|r| f32_at(r, "rope_theta"))
            .or_else(|| f32_at(text, "rope_theta"))
            .unwrap_or(10_000.0);
        let mrope_interleaved = rope
            .and_then(|r| r.get("mrope_interleaved"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let mrope_section: Vec<usize> = rope
            .and_then(|r| r.get("mrope_section"))
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64())
                    .map(|n| n as usize)
                    .collect()
            })
            .unwrap_or_default();

        let deltanet = DeltaNetConfig {
            key_heads: usize_at(text, "linear_num_key_heads")
                .ok_or("missing linear_num_key_heads")?,
            value_heads: usize_at(text, "linear_num_value_heads")
                .ok_or("missing linear_num_value_heads")?,
            key_head_dim: usize_at(text, "linear_key_head_dim")
                .ok_or("missing linear_key_head_dim")?,
            value_head_dim: usize_at(text, "linear_value_head_dim")
                .ok_or("missing linear_value_head_dim")?,
            conv_kernel: usize_at(text, "linear_conv_kernel_dim").unwrap_or(4),
            // Names the gated-RMSNorm activation on the DeltaNet output gate.
            // Absent means fall back to `hidden_act`, which is silu for this family.
            output_gate_sigmoid: text
                .get("output_gate_type")
                .and_then(|v| v.as_str())
                .map(|s| s == "sigmoid")
                .unwrap_or(false),
        };
        if deltanet.value_heads % deltanet.key_heads != 0 {
            return Err(format!(
                "linear_num_value_heads ({}) must be a multiple of linear_num_key_heads ({})",
                deltanet.value_heads, deltanet.key_heads
            ));
        }

        let indexer = IndexerConfig {
            n_heads: usize_at(text, "indexer_n_heads").ok_or("missing indexer_n_heads")?,
            kv_heads: usize_at(text, "indexer_kv_heads").unwrap_or(1),
            head_dim: usize_at(text, "indexer_head_dim").ok_or("missing indexer_head_dim")?,
            budget: usize_at(text, "indexer_budget").ok_or("missing indexer_budget")?,
            compress_ratio: usize_at(text, "indexer_compress_ratio").unwrap_or(1),
        };
        if indexer.compress_ratio == 0 || indexer.budget % indexer.compress_ratio != 0 {
            return Err(format!(
                "indexer_budget ({}) must be a positive multiple of indexer_compress_ratio ({})",
                indexer.budget, indexer.compress_ratio
            ));
        }

        let moe = MoeConfig {
            num_experts: usize_at(text, "num_experts").ok_or("missing num_experts")?,
            experts_per_tok: usize_at(text, "num_experts_per_tok")
                .ok_or("missing num_experts_per_tok")?,
            intermediate: usize_at(text, "moe_intermediate_size")
                .ok_or("missing moe_intermediate_size")?,
            shared_intermediate: usize_at(text, "shared_expert_intermediate_size").unwrap_or(0),
            // Defaults TRUE, and the shipped config omits it.
            norm_topk_prob: text
                .get("norm_topk_prob")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
        };
        if moe.experts_per_tok == 0 || moe.experts_per_tok > moe.num_experts {
            return Err(format!(
                "num_experts_per_tok ({}) must be in 1..={}",
                moe.experts_per_tok, moe.num_experts
            ));
        }

        let gated_residual = GatedResidualConfig {
            count: usize_at(text, "hc_count").unwrap_or(1),
            lowrank: usize_at(text, "hc_lowrank").unwrap_or(0),
        };

        // ONE-BASED in the file; stored zero-based here.
        let ngram = match text.get("ple_layer_ids").and_then(|v| v.as_array()) {
            Some(ids) if !ids.is_empty() => {
                let one_based = ids[0]
                    .as_u64()
                    .ok_or("ple_layer_ids[0] is not an integer")?
                    as usize;
                if one_based == 0 || one_based > layers {
                    return Err(format!(
                        "ple_layer_ids[0] = {one_based} is out of range for {layers} layers \
                         (the field is ONE-BASED)"
                    ));
                }
                let ngram_size = usize_at(text, "ngram_size").ok_or("missing ngram_size")?;
                if ngram_size < 2 {
                    return Err(format!("ngram_size must be >= 2, got {ngram_size}"));
                }
                let cfg = NgramConfig {
                    layer_idx: one_based - 1,
                    // First (and, in the shipped checkpoint, only) PLE block.
                    ple_index: 0,
                    embed_dim: usize_at(text, "ple_embed_dim").unwrap_or(hidden),
                    ngram_size,
                    heads_per_ngram: usize_at(text, "heads_per_ngram")
                        .ok_or("missing heads_per_ngram")?,
                    vocab_size_base: text
                        .get("ngram_vocab_size_base")
                        .and_then(|v| v.as_u64())
                        .ok_or("missing ngram_vocab_size_base")?,
                    divisible_by: text
                        .get("make_ngram_vocab_size_divisible_by")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(1),
                    shards: usize_at(text, "split_ngram_parts").unwrap_or(1),
                    conv_kernel: usize_at(text, "ple_conv_kernel_size").unwrap_or(4),
                    // Defaults 1234 (`configuration_qwen4_exp.py:156`), and the
                    // shipped config OMITS the key — so this default decides the
                    // hash multipliers. Verified: 1234 reproduces the checkpoint's
                    // stored `layer_multipliers` exactly; 0 does not.
                    seed: text.get("seed").and_then(|v| v.as_u64()).unwrap_or(1234),
                };
                // `split_ngram_parts` shards the table on disk; loading concatenates
                // them. The plan divides the padded row count by the shard count, so
                // a non-divisible pair would silently under-count rows and reject an
                // otherwise valid checkpoint. The shipped file divides exactly
                // (320,001,536 / 128 = 2,500,012); nothing guarantees a re-export does.
                let (_, _, padded) =
                    crate::ngram_head_layout(cfg.vocab_size_base, cfg.heads(), cfg.divisible_by);
                if cfg.shards == 0 || padded as usize % cfg.shards != 0 {
                    return Err(format!(
                        "split_ngram_parts ({}) must divide the padded n-gram vocab ({padded})",
                        cfg.shards
                    ));
                }
                if cfg.heads() == 0 || cfg.embed_dim % cfg.heads() != 0 {
                    return Err(format!(
                        "ple_embed_dim ({}) must divide evenly across {} n-gram heads",
                        cfg.embed_dim,
                        cfg.heads()
                    ));
                }
                Some(cfg)
            }
            _ => None,
        };

        let cfg = Self {
            hidden,
            vocab: usize_at(text, "vocab_size").ok_or("missing vocab_size")?,
            layers,
            layer_types,
            n_heads: usize_at(text, "num_attention_heads").ok_or("missing num_attention_heads")?,
            n_kv_heads: usize_at(text, "num_key_value_heads")
                .ok_or("missing num_key_value_heads")?,
            head_dim,
            partial_rotary_factor,
            rope_theta,
            mrope_interleaved,
            mrope_section,
            rms_norm_eps: f32_at(text, "rms_norm_eps").unwrap_or(1e-6),
            tie_word_embeddings: text
                .get("tie_word_embeddings")
                .or_else(|| root.get("tie_word_embeddings"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            deltanet,
            indexer,
            moe,
            gated_residual,
            ngram,
            max_position: usize_at(text, "max_position_embeddings").unwrap_or(4096),
            eos_token_id: usize_at(text, "eos_token_id")
                .map(|v| v as u32)
                .unwrap_or_else(|| {
                    usize_at(text, "vocab_size").unwrap_or(1).saturating_sub(1) as u32
                }),
            mtp_layers: usize_at(text, "mtp_num_hidden_layers").unwrap_or(0),
            vision: root.get("vision_config").map(|v| VisionConfig {
                depth: usize_at(v, "depth").unwrap_or(0),
                hidden: usize_at(v, "hidden_size").unwrap_or(0),
                n_heads: usize_at(v, "num_heads").unwrap_or(1),
                intermediate: usize_at(v, "intermediate_size").unwrap_or(0),
                out_hidden: usize_at(v, "out_hidden_size").unwrap_or(hidden),
                in_channels: usize_at(v, "in_channels").unwrap_or(3),
                patch_size: usize_at(v, "patch_size").unwrap_or(16),
                temporal_patch_size: usize_at(v, "temporal_patch_size").unwrap_or(2),
                spatial_merge_size: usize_at(v, "spatial_merge_size").unwrap_or(1),
                num_position_embeddings: usize_at(v, "num_position_embeddings").unwrap_or(0),
            }),
            has_vision: root.get("vision_config").is_some(),
        };

        // The mrope sections partition the rotary HALF-dimension (one entry per
        // position axis: temporal, height, width). A section list that disagrees
        // with `partial_rotary_factor` is the shape of bug that still produces
        // coherent text on text-only prompts, so refuse it here.
        if !cfg.mrope_section.is_empty() {
            let sum: usize = cfg.mrope_section.iter().sum();
            if sum * 2 != cfg.rotary_dim() {
                return Err(format!(
                    "mrope_section sums to {sum} (x2 = {}) but rotary_dim is {} \
                     (head_dim {} * partial_rotary_factor {})",
                    sum * 2,
                    cfg.rotary_dim(),
                    cfg.head_dim,
                    cfg.partial_rotary_factor
                ));
            }
        }
        // The vision tower's output is scattered DIRECTLY into the text embedding
        // stream at the image-placeholder positions, so its width must match. The
        // reference enforces this too, but only at the moment of the scatter, where
        // it surfaces as a confusing "features do not match" count error.
        if let Some(v) = cfg.vision.as_ref() {
            if v.depth > 0 && v.out_hidden != cfg.hidden {
                return Err(format!(
                    "vision out_hidden_size ({}) must equal the text hidden_size ({}) — \
                     merged vision tokens are spliced into the text embedding stream",
                    v.out_hidden, cfg.hidden
                ));
            }
        }
        Ok(cfg)
    }

    /// Parse from raw `config.json` bytes.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, String> {
        let v: Value = serde_json::from_slice(bytes).map_err(|e| format!("config.json: {e}"))?;
        Self::from_json(&v)
    }

    /// Layers using the sparse-attention mixer, in order.
    /// A conservative KV budget for state allocation.
    ///
    /// The shipped model declares a 262144 context, and allocating that up front
    /// per sparse-attention layer is not what a caller usually wants. The daemon
    /// knows its own budget and should call `TrunkState::new` with it; this is the
    /// fallback for callers that do not, and it is bounded on purpose.
    pub fn max_seq_hint(&self) -> usize {
        self.max_position.min(4096)
    }

    pub fn sparse_attention_layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.layer_types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == LayerType::SparseAttention)
            .map(|(i, _)| i)
    }
}
