// SPDX-License-Identifier: Apache-2.0

use hipfire_runtime::layered_kv::{KvStorageKind, LayerKvSpec, LayeredKvPlan};
use serde::Deserialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cohere2LayerKind {
    Full,
    Sliding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cohere2MlpKind {
    Dense,
    Sparse,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cohere2Config {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub expert_intermediate: usize,
    pub dense_intermediate: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub logit_scale: f32,
    pub tie_word_embeddings: bool,
    pub layer_kinds: Vec<Cohere2LayerKind>,
    pub mlp_kinds: Vec<Cohere2MlpKind>,
    /// The upstream Cohere2-MoE attention applies RoPE to sliding layers and
    /// to dense-prefix layers when `prefix_dense_sliding_window_pattern == 1`.
    pub force_rope: Vec<bool>,
}

#[derive(Deserialize)]
struct RawConfig {
    model_type: String,
    architectures: Vec<String>,
    hidden_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    prefix_dense_intermediate_size: usize,
    first_k_dense_replace: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    num_shared_experts: usize,
    expert_selection_fn: String,
    norm_topk_prob: bool,
    use_parallel_block: bool,
    use_gated_activation: bool,
    hidden_act: String,
    attention_bias: bool,
    layer_types: Vec<String>,
    sliding_window: usize,
    max_position_embeddings: usize,
    rope_theta: f32,
    rms_norm_eps: Option<f32>,
    logit_scale: f32,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    prefix_dense_sliding_window_pattern: usize,
}

const fn default_true() -> bool {
    true
}

impl Cohere2Config {
    pub fn from_json_str(json: &str) -> Result<Self, String> {
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|error| format!("invalid Cohere2 config JSON: {error}"))?;
        let config = value
            .get("config")
            .and_then(|config| config.get("text_config").or(Some(config)))
            .or_else(|| value.get("text_config"))
            .unwrap_or(&value);
        let raw: RawConfig = serde_json::from_value(config.clone())
            .map_err(|error| format!("invalid Cohere2 config JSON: {error}"))?;
        if raw.model_type != "cohere2_moe"
            || raw.architectures.as_slice() != ["Cohere2MoeForCausalLM"]
            || raw.hidden_size == 0
            || raw.vocab_size == 0
            || raw.num_hidden_layers == 0
            || raw.q_heads_invalid()
            || raw.head_dim == 0
            || raw.head_dim % 2 != 0
            || raw.intermediate_size == 0
            || raw.prefix_dense_intermediate_size == 0
            || raw.first_k_dense_replace > raw.num_hidden_layers
            || raw.num_experts == 0
            || raw.num_experts_per_tok == 0
            || raw.num_experts_per_tok > raw.num_experts
            || raw.num_shared_experts != 0
            || raw.expert_selection_fn != "sigmoid"
            || raw.norm_topk_prob
            || !raw.use_parallel_block
            || !raw.use_gated_activation
            || raw.hidden_act != "silu"
            || raw.attention_bias
            || raw.layer_types.len() != raw.num_hidden_layers
            || raw.sliding_window == 0
            || raw.max_position_embeddings < raw.sliding_window
            || !raw.rope_theta.is_finite()
            || raw.rope_theta <= 0.0
            || raw
                .rms_norm_eps
                .is_none_or(|eps| !eps.is_finite() || eps <= 0.0)
            || !raw.logit_scale.is_finite()
            || raw.logit_scale <= 0.0
            || (raw.logit_scale - 1.0).abs() > f32::EPSILON
            || raw.prefix_dense_sliding_window_pattern == 0
        {
            return Err("unsupported or invalid Cohere2-MoE configuration".into());
        }
        let layer_kinds = raw
            .layer_types
            .iter()
            .enumerate()
            .map(|(layer, kind)| match kind.as_str() {
                "full_attention" => Ok(Cohere2LayerKind::Full),
                "sliding_attention" => Ok(Cohere2LayerKind::Sliding),
                other => Err(format!(
                    "Cohere2 layer {layer} has unsupported kind {other:?}"
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mlp_kinds = (0..raw.num_hidden_layers)
            .map(|layer| {
                if layer < raw.first_k_dense_replace {
                    Cohere2MlpKind::Dense
                } else {
                    Cohere2MlpKind::Sparse
                }
            })
            .collect::<Vec<_>>();
        let force_rope = layer_kinds
            .iter()
            .zip(&mlp_kinds)
            .map(|(attention, mlp)| {
                *attention == Cohere2LayerKind::Sliding
                    || (*mlp == Cohere2MlpKind::Dense
                        && raw.prefix_dense_sliding_window_pattern == 1)
            })
            .collect();
        Ok(Self {
            hidden_size: raw.hidden_size,
            vocab_size: raw.vocab_size,
            num_hidden_layers: raw.num_hidden_layers,
            q_heads: raw.num_attention_heads,
            kv_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            expert_intermediate: raw.intermediate_size,
            dense_intermediate: raw.prefix_dense_intermediate_size,
            num_experts: raw.num_experts,
            top_k: raw.num_experts_per_tok,
            sliding_window: raw.sliding_window,
            max_position_embeddings: raw.max_position_embeddings,
            rope_theta: raw.rope_theta,
            norm_eps: raw.rms_norm_eps.unwrap(),
            logit_scale: raw.logit_scale,
            tie_word_embeddings: raw.tie_word_embeddings,
            layer_kinds,
            mlp_kinds,
            force_rope,
        })
    }

    pub fn layered_kv_plan(&self, max_seq: usize) -> Result<LayeredKvPlan, String> {
        let layers = self
            .layer_kinds
            .iter()
            .map(|kind| {
                let storage = match kind {
                    Cohere2LayerKind::Full => KvStorageKind::Full,
                    Cohere2LayerKind::Sliding => KvStorageKind::SlidingWindow {
                        window: self.sliding_window.min(max_seq),
                    },
                };
                LayerKvSpec::owned(self.q_heads, self.kv_heads, self.head_dim, storage)
            })
            .collect();
        LayeredKvPlan::build(max_seq, layers)
    }
}

impl RawConfig {
    fn q_heads_invalid(&self) -> bool {
        self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.num_attention_heads % self.num_key_value_heads != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"{
      "model_type":"cohere2_moe","architectures":["Cohere2MoeForCausalLM"],
      "hidden_size":2048,"vocab_size":262144,"num_hidden_layers":5,
      "num_attention_heads":32,"num_key_value_heads":4,"head_dim":128,
      "intermediate_size":768,"prefix_dense_intermediate_size":3072,
      "first_k_dense_replace":1,"num_experts":128,"num_experts_per_tok":8,
      "num_shared_experts":0,"expert_selection_fn":"sigmoid","norm_topk_prob":false,
      "use_parallel_block":true,"use_gated_activation":true,"hidden_act":"silu","attention_bias":false,
      "layer_types":["full_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],
      "sliding_window":4096,"max_position_embeddings":500000,"rope_theta":50000.0,
      "rms_norm_eps":0.000001,"logit_scale":1.0,"tie_word_embeddings":true,
      "prefix_dense_sliding_window_pattern":1
    }"#;

    #[test]
    fn parses_bls_parallel_moe_and_rope_policy() {
        let config = Cohere2Config::from_json_str(FIXTURE).unwrap();
        assert_eq!(config.mlp_kinds[0], Cohere2MlpKind::Dense);
        assert_eq!(config.mlp_kinds[1], Cohere2MlpKind::Sparse);
        assert_eq!(config.force_rope, vec![true, true, true, true, false]);
        assert_eq!(config.top_k, 8);
    }

    #[test]
    fn rejects_softmax_or_normalized_router() {
        assert!(Cohere2Config::from_json_str(&FIXTURE.replace("sigmoid", "softmax")).is_err());
        assert!(Cohere2Config::from_json_str(
            &FIXTURE.replace("\"norm_topk_prob\":false", "\"norm_topk_prob\":true")
        )
        .is_err());
    }

    #[test]
    fn layered_kv_plan_preserves_full_and_bounds_sliding_storage() {
        let config = Cohere2Config::from_json_str(FIXTURE).unwrap();
        let plan = config.layered_kv_plan(1024).unwrap();
        assert_eq!(plan.layers().len(), 5);
        assert_eq!(plan.layers()[0].storage, KvStorageKind::Full);
        assert_eq!(
            plan.layers()[1].storage,
            KvStorageKind::SlidingWindow { window: 1024 }
        );
    }
}
