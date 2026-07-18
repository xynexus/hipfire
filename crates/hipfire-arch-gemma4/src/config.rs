// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Strict Gemma 4 text-config parsing and load-time per-layer lowering.

use hipfire_runtime::layered_kv::{KvStorageKind, LayerKvSpec, LayeredKvPlan};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionGeometry {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopePlan {
    FullHalfSplit {
        theta: f32,
        dim: usize,
    },
    ProportionalHalfSplit {
        theta: f32,
        rotary_dim: usize,
        basis_dim: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueProjection {
    Separate,
    FromPreNormKey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvProducer {
    Own,
    SharedFrom { producer_layer: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnPlan {
    Dense {
        intermediate: usize,
    },
    DensePlusMoe {
        dense_intermediate: usize,
        expert_intermediate: usize,
        experts: usize,
        top_k: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKind {
    Sliding,
    Full,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4LayerPlan {
    pub kind: AttentionKind,
    pub attention: AttentionGeometry,
    pub rope: RopePlan,
    pub value_projection: ValueProjection,
    pub kv_producer: KvProducer,
    pub ffn: FfnPlan,
}

#[derive(Clone, Debug)]
pub struct Gemma4Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub final_logit_softcapping: f32,
    pub tie_word_embeddings: bool,
    pub hidden_size_per_layer_input: usize,
    pub use_double_wide_mlp: bool,
    pub layers: Vec<Gemma4LayerPlan>,
}

fn required<'a>(value: &'a Value, key: &str) -> Result<&'a Value, String> {
    value
        .get(key)
        .ok_or_else(|| format!("Gemma 4 text config is missing required `{key}`"))
}

fn req_usize(value: &Value, key: &str) -> Result<usize, String> {
    required(value, key)?
        .as_u64()
        .map(|v| v as usize)
        .ok_or_else(|| format!("Gemma 4 `{key}` must be a non-negative integer"))
}

fn req_f32(value: &Value, key: &str) -> Result<f32, String> {
    required(value, key)?
        .as_f64()
        .map(|v| v as f32)
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("Gemma 4 `{key}` must be finite"))
}

fn req_bool(value: &Value, key: &str) -> Result<bool, String> {
    required(value, key)?
        .as_bool()
        .ok_or_else(|| format!("Gemma 4 `{key}` must be boolean"))
}

fn optional_usize(value: &Value, key: &str) -> Result<Option<usize>, String> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .map(|v| Some(v as usize))
            .ok_or_else(|| format!("Gemma 4 `{key}` must be an integer or null")),
    }
}

fn rope_entry<'a>(text: &'a Value, kind: &str) -> Result<&'a Value, String> {
    required(required(text, "rope_parameters")?, kind)
}

impl Gemma4Config {
    pub fn from_json_str(json: &str) -> Result<Self, String> {
        let root: Value = serde_json::from_str(json)
            .map_err(|error| format!("invalid Gemma 4 config JSON: {error}"))?;
        Self::from_value(&root)
    }

    pub fn from_value(root: &Value) -> Result<Self, String> {
        let text = root.get("text_config").unwrap_or(root);
        let model_type = required(text, "model_type")?
            .as_str()
            .ok_or_else(|| "Gemma 4 `model_type` must be a string".to_string())?
            .to_string();
        if ![
            "gemma4",
            "gemma4_text",
            "gemma4_unified",
            "gemma4_unified_text",
        ]
        .contains(&model_type.as_str())
        {
            return Err(format!(
                "unsupported Gemma 4 text model_type `{model_type}`"
            ));
        }

        let hidden_size = req_usize(text, "hidden_size")?;
        let vocab_size = req_usize(text, "vocab_size")?;
        let num_hidden_layers = req_usize(text, "num_hidden_layers")?;
        let intermediate_size = req_usize(text, "intermediate_size")?;
        let q_heads = req_usize(text, "num_attention_heads")?;
        let local_kv_heads = req_usize(text, "num_key_value_heads")?;
        // Upstream's stable default is the local KV count when no distinct
        // global count is serialized (E2B/E4B).
        let global_kv_heads =
            optional_usize(text, "num_global_key_value_heads")?.unwrap_or(local_kv_heads);
        let local_head_dim = req_usize(text, "head_dim")?;
        let global_head_dim = req_usize(text, "global_head_dim")?;
        let sliding_window = req_usize(text, "sliding_window")?;
        let max_position_embeddings = req_usize(text, "max_position_embeddings")?;
        let rms_norm_eps = req_f32(text, "rms_norm_eps")?;
        let final_logit_softcapping = req_f32(text, "final_logit_softcapping")?;
        let tie_word_embeddings = req_bool(text, "tie_word_embeddings")?;
        let attention_k_eq_v = req_bool(text, "attention_k_eq_v")?;
        let enable_moe = req_bool(text, "enable_moe_block")?;
        let use_double_wide_mlp = req_bool(text, "use_double_wide_mlp")?;
        let hidden_size_per_layer_input = req_usize(text, "hidden_size_per_layer_input")?;
        let shared_layers = req_usize(text, "num_kv_shared_layers")?;

        for (name, value) in [
            ("hidden_size", hidden_size),
            ("vocab_size", vocab_size),
            ("num_hidden_layers", num_hidden_layers),
            ("intermediate_size", intermediate_size),
            ("num_attention_heads", q_heads),
            ("num_key_value_heads", local_kv_heads),
            ("num_global_key_value_heads", global_kv_heads),
            ("head_dim", local_head_dim),
            ("global_head_dim", global_head_dim),
            ("sliding_window", sliding_window),
        ] {
            if value == 0 {
                return Err(format!("Gemma 4 `{name}` must be nonzero"));
            }
        }
        if q_heads % local_kv_heads != 0 || q_heads % global_kv_heads != 0 {
            return Err(format!(
                "Gemma 4 Q heads {q_heads} must divide evenly across local/global KV heads {local_kv_heads}/{global_kv_heads}"
            ));
        }
        if shared_layers > num_hidden_layers {
            return Err(format!(
                "Gemma 4 shared tail {shared_layers} exceeds {num_hidden_layers} layers"
            ));
        }
        if final_logit_softcapping <= 0.0 || rms_norm_eps <= 0.0 {
            return Err("Gemma 4 softcap and RMS epsilon must be positive".to_string());
        }

        let raw_types = required(text, "layer_types")?
            .as_array()
            .ok_or_else(|| "Gemma 4 `layer_types` must be an array".to_string())?;
        if raw_types.len() != num_hidden_layers {
            return Err(format!(
                "Gemma 4 layer_types length {} != num_hidden_layers {num_hidden_layers}",
                raw_types.len()
            ));
        }
        let kinds = raw_types
            .iter()
            .enumerate()
            .map(|(layer, value)| match value.as_str() {
                Some("sliding_attention") => Ok(AttentionKind::Sliding),
                Some("full_attention") => Ok(AttentionKind::Full),
                other => Err(format!(
                    "Gemma 4 layer {layer} has unsupported layer type {other:?}"
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let local_rope = rope_entry(text, "sliding_attention")?;
        if required(local_rope, "rope_type")?.as_str() != Some("default") {
            return Err("Gemma 4 sliding attention requires default RoPE".to_string());
        }
        let local_theta = req_f32(local_rope, "rope_theta")?;
        let global_rope = rope_entry(text, "full_attention")?;
        if required(global_rope, "rope_type")?.as_str() != Some("proportional") {
            return Err("Gemma 4 full attention requires proportional RoPE".to_string());
        }
        let global_theta = req_f32(global_rope, "rope_theta")?;
        let partial = req_f32(global_rope, "partial_rotary_factor")?;
        let rotary_dim_f = global_head_dim as f32 * partial;
        let rotary_dim = rotary_dim_f.round() as usize;
        if partial <= 0.0
            || partial > 1.0
            || (rotary_dim_f - rotary_dim as f32).abs() > 1e-4
            || rotary_dim == 0
            || rotary_dim % 2 != 0
            || local_head_dim % 2 != 0
        {
            return Err(format!(
                "Gemma 4 invalid rotary geometry: local={local_head_dim}, global={global_head_dim}, factor={partial}"
            ));
        }

        let ffn = if enable_moe {
            let expert_intermediate = optional_usize(text, "moe_intermediate_size")?
                .ok_or_else(|| "Gemma 4 MoE requires `moe_intermediate_size`".to_string())?;
            let experts = optional_usize(text, "num_experts")?
                .ok_or_else(|| "Gemma 4 MoE requires `num_experts`".to_string())?;
            let top_k = optional_usize(text, "top_k_experts")?
                .ok_or_else(|| "Gemma 4 MoE requires `top_k_experts`".to_string())?;
            if expert_intermediate == 0 || experts == 0 || top_k == 0 || top_k > experts {
                return Err("Gemma 4 MoE dimensions/top-k are invalid".to_string());
            }
            FfnPlan::DensePlusMoe {
                dense_intermediate: intermediate_size,
                expert_intermediate,
                experts,
                top_k,
            }
        } else {
            FfnPlan::Dense {
                intermediate: intermediate_size,
            }
        };

        let shared_start = num_hidden_layers - shared_layers;
        let mut last_owned: [Option<usize>; 2] = [None, None];
        let mut layers = Vec::with_capacity(num_hidden_layers);
        for (layer_idx, kind) in kinds.into_iter().enumerate() {
            let slot = usize::from(kind == AttentionKind::Full);
            let (attention, rope, value_projection) = match kind {
                AttentionKind::Sliding => (
                    AttentionGeometry {
                        q_heads,
                        kv_heads: local_kv_heads,
                        head_dim: local_head_dim,
                    },
                    RopePlan::FullHalfSplit {
                        theta: local_theta,
                        dim: local_head_dim,
                    },
                    ValueProjection::Separate,
                ),
                AttentionKind::Full => (
                    AttentionGeometry {
                        q_heads,
                        kv_heads: global_kv_heads,
                        head_dim: global_head_dim,
                    },
                    RopePlan::ProportionalHalfSplit {
                        theta: global_theta,
                        rotary_dim,
                        basis_dim: global_head_dim,
                    },
                    if attention_k_eq_v {
                        ValueProjection::FromPreNormKey
                    } else {
                        ValueProjection::Separate
                    },
                ),
            };
            let is_shared = layer_idx >= shared_start;
            let kv_producer = if is_shared {
                KvProducer::SharedFrom {
                    producer_layer: last_owned[slot].ok_or_else(|| {
                        format!(
                            "Gemma 4 shared layer {layer_idx} has no earlier owned {kind:?} producer"
                        )
                    })?,
                }
            } else {
                last_owned[slot] = Some(layer_idx);
                KvProducer::Own
            };
            // `use_double_wide_mlp`: the KV-shared tail layers double their dense MLP
            // intermediate (E2B/E4B: 6144 → 12288) to offset the shared KV.
            let layer_ffn = match ffn {
                FfnPlan::Dense { intermediate } if use_double_wide_mlp && is_shared => {
                    FfnPlan::Dense {
                        intermediate: intermediate * 2,
                    }
                }
                other => other,
            };
            layers.push(Gemma4LayerPlan {
                kind,
                attention,
                rope,
                value_projection,
                kv_producer,
                ffn: layer_ffn,
            });
        }

        Ok(Self {
            model_type,
            hidden_size,
            vocab_size,
            num_hidden_layers,
            intermediate_size,
            rms_norm_eps,
            max_position_embeddings,
            sliding_window,
            final_logit_softcapping,
            tie_word_embeddings,
            hidden_size_per_layer_input,
            use_double_wide_mlp,
            layers,
        })
    }

    pub fn layered_kv_plan(&self, max_seq: usize) -> Result<LayeredKvPlan, String> {
        if max_seq == 0 || max_seq > self.max_position_embeddings {
            return Err(format!(
                "Gemma 4 context {max_seq} must be in 1..={} ",
                self.max_position_embeddings
            ));
        }
        let specs = self
            .layers
            .iter()
            .map(|layer| {
                let storage = match layer.kind {
                    AttentionKind::Sliding => KvStorageKind::SlidingWindow {
                        window: self.sliding_window.min(max_seq),
                    },
                    AttentionKind::Full => KvStorageKind::Full,
                };
                let owned = LayerKvSpec::owned(
                    layer.attention.q_heads,
                    layer.attention.kv_heads,
                    layer.attention.head_dim,
                    storage,
                );
                match layer.kv_producer {
                    KvProducer::Own => owned,
                    KvProducer::SharedFrom { producer_layer } => owned.shared(producer_layer),
                }
            })
            .collect();
        LayeredKvPlan::build(max_seq, specs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG_31B: &str =
        include_str!("../../../benchmarks/gemma4/fixtures/configs/gemma-4-31B.json");
    const CONFIG_E4B: &str =
        include_str!("../../../benchmarks/gemma4/fixtures/configs/gemma-4-E4B.json");
    const CONFIG_MOE: &str =
        include_str!("../../../benchmarks/gemma4/fixtures/configs/gemma-4-26B-A4B.json");

    #[test]
    fn lowers_released_31b_mixed_geometry_and_proportional_rope() {
        let cfg = Gemma4Config::from_json_str(CONFIG_31B).unwrap();
        assert_eq!(cfg.layers.len(), 60);
        assert_eq!(cfg.layers[0].attention.head_dim, 256);
        assert_eq!(cfg.layers[5].attention.head_dim, 512);
        assert_eq!(cfg.layers[5].attention.kv_heads, 4);
        assert_eq!(
            cfg.layers[5].rope,
            RopePlan::ProportionalHalfSplit {
                theta: 1_000_000.0,
                rotary_dim: 128,
                basis_dim: 512,
            }
        );
    }

    #[test]
    fn released_e4b_resolves_shared_tail_by_attention_type() {
        let cfg = Gemma4Config::from_json_str(CONFIG_E4B).unwrap();
        assert_eq!(cfg.hidden_size_per_layer_input, 256);
        for layer in 24..42 {
            assert!(matches!(
                cfg.layers[layer].kv_producer,
                KvProducer::SharedFrom { producer_layer } if producer_layer < 24
            ));
        }
        let plan = cfg.layered_kv_plan(128).unwrap();
        assert_eq!(plan.physical_owned_layers(), 24);
    }

    #[test]
    fn released_moe_keeps_dense_and_routed_dimensions_distinct() {
        let cfg = Gemma4Config::from_json_str(CONFIG_MOE).unwrap();
        assert!(matches!(
            cfg.layers[0].ffn,
            FfnPlan::DensePlusMoe {
                experts: 128,
                top_k: 8,
                ..
            }
        ));
    }

    #[test]
    fn rejects_layer_roster_length_drift() {
        let mut value: Value = serde_json::from_str(CONFIG_31B).unwrap();
        value["text_config"]["layer_types"] = Value::Array(Vec::new());
        let error = Gemma4Config::from_value(&value).unwrap_err();
        assert!(error.contains("layer_types length"));
    }
}
