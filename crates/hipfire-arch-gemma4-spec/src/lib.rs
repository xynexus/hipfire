// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Gemma 4 text core (arch id 24).
//!
//! The standard/text/unified wrapper names resolve to one base identity. Runtime
//! math remains in `hipfire-arch-gemma4`; this crate owns only detection, source
//! precision policy, stacked-expert ingest layout, and deterministic tiny model
//! descriptions for offline conversion tests.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    ExpertLayout, Ingest, Init, PrecisionClass, TensorRole, TensorSpec, ToyFixture, ToyModel,
    ARCH_ID_GEMMA4,
};

pub const GEMMA4_ARCH_ID: ArchId = ArchId(ARCH_ID_GEMMA4 as u16);

pub struct Gemma4Spec;

impl Arch for Gemma4Spec {
    fn id(&self) -> ArchId {
        GEMMA4_ARCH_ID
    }

    fn family(&self) -> &'static str {
        "gemma4"
    }

    fn model_types(&self) -> &'static [&'static str] {
        &[
            "gemma4",
            "gemma4_text",
            "gemma4_unified",
            "gemma4_unified_text",
        ]
    }
}

fn is_source_precision(tensor: &str) -> bool {
    tensor.contains("norm")
        || tensor.ends_with("layer_scalar")
        || tensor.contains("router.scale")
        || tensor.contains("per_expert_scale")
        || tensor.contains("embed_tokens_per_layer")
        || tensor.contains("per_layer_input")
        || tensor.contains("per_layer_projection")
}

impl Ingest for Gemma4Spec {
    fn role(&self, tensor: &str) -> TensorRole {
        if tensor.contains("router.proj")
            || tensor.contains("router.scale")
            || tensor.contains("per_expert_scale")
        {
            TensorRole::Router
        } else {
            transformer_role(tensor)
        }
    }

    fn importance(&self, tensor: &str) -> u8 {
        if is_source_precision(tensor) {
            255
        } else {
            default_importance(self.role(tensor))
        }
    }

    fn requires(&self, tensor: &str) -> CapReq {
        default_requires(self.role(tensor))
    }

    fn precision_class(&self, tensor: &str) -> PrecisionClass {
        if is_source_precision(tensor) {
            PrecisionClass::SourcePrecision
        } else {
            hipfire_arch_api::default_precision_class(self.role(tensor))
        }
    }

    fn expert_layout(&self) -> ExpertLayout {
        ExpertLayout::StackedGateUpDown
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gemma4ToyVariant {
    Dense,
    PleSharing,
    DenseMoe,
}

impl Gemma4ToyVariant {
    pub fn name(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::PleSharing => "ple-sharing",
            Self::DenseMoe => "dense-moe",
        }
    }
}

struct Tiny {
    variant: Gemma4ToyVariant,
    hidden: usize,
    dense_intermediate: usize,
    expert_intermediate: usize,
    ple_dim: usize,
    vocab: usize,
    layers: usize,
    q_heads: usize,
    local_kv_heads: usize,
    global_kv_heads: usize,
    local_head_dim: usize,
    global_head_dim: usize,
    experts: usize,
    top_k: usize,
}

impl Tiny {
    fn for_variant(variant: Gemma4ToyVariant) -> Self {
        match variant {
            Gemma4ToyVariant::Dense => Self {
                variant,
                hidden: 256,
                dense_intermediate: 384,
                expert_intermediate: 0,
                ple_dim: 0,
                vocab: 512,
                layers: 2,
                q_heads: 2,
                local_kv_heads: 1,
                global_kv_heads: 1,
                local_head_dim: 64,
                global_head_dim: 128,
                experts: 0,
                top_k: 0,
            },
            Gemma4ToyVariant::PleSharing => Self {
                variant,
                hidden: 256,
                dense_intermediate: 384,
                expert_intermediate: 0,
                ple_dim: 32,
                vocab: 512,
                layers: 4,
                q_heads: 2,
                local_kv_heads: 1,
                global_kv_heads: 1,
                local_head_dim: 64,
                global_head_dim: 128,
                experts: 0,
                top_k: 0,
            },
            Gemma4ToyVariant::DenseMoe => Self {
                variant,
                hidden: 256,
                dense_intermediate: 256,
                expert_intermediate: 128,
                ple_dim: 0,
                vocab: 512,
                layers: 2,
                q_heads: 2,
                local_kv_heads: 1,
                global_kv_heads: 1,
                local_head_dim: 64,
                global_head_dim: 128,
                experts: 8,
                top_k: 8,
            },
        }
    }

    fn layer_types(&self) -> Vec<&'static str> {
        (0..self.layers)
            .map(|layer| {
                if layer % 2 == 0 {
                    "sliding_attention"
                } else {
                    "full_attention"
                }
            })
            .collect()
    }

    fn config(&self) -> serde_json::Value {
        let shared = usize::from(self.variant == Gemma4ToyVariant::PleSharing) * 2;
        let moe = self.variant == Gemma4ToyVariant::DenseMoe;
        serde_json::json!({
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "text_config": {
                "attention_bias": false,
                "attention_dropout": 0.0,
                "attention_k_eq_v": self.variant != Gemma4ToyVariant::PleSharing,
                "bos_token_id": 2,
                "dtype": "bfloat16",
                "enable_moe_block": moe,
                "eos_token_id": 1,
                "final_logit_softcapping": 30.0,
                "global_head_dim": self.global_head_dim,
                "head_dim": self.local_head_dim,
                "hidden_activation": "gelu_pytorch_tanh",
                "hidden_size": self.hidden,
                "hidden_size_per_layer_input": self.ple_dim,
                "intermediate_size": self.dense_intermediate,
                "layer_types": self.layer_types(),
                "max_position_embeddings": 256,
                "model_type": "gemma4_text",
                "moe_intermediate_size": if moe { Some(self.expert_intermediate) } else { None },
                "num_attention_heads": self.q_heads,
                "num_experts": if moe { Some(self.experts) } else { None },
                "num_global_key_value_heads": self.global_kv_heads,
                "num_hidden_layers": self.layers,
                "num_key_value_heads": self.local_kv_heads,
                "num_kv_shared_layers": shared,
                "pad_token_id": 0,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1_000_000.0,
                        "rope_type": "proportional"
                    },
                    "sliding_attention": {
                        "rope_theta": 10_000.0,
                        "rope_type": "default"
                    }
                },
                "sliding_window": 32,
                "tie_word_embeddings": true,
                "top_k_experts": if moe { Some(self.top_k) } else { None },
                "use_cache": true,
                "use_double_wide_mlp": false,
                "vocab_size": self.vocab,
                "vocab_size_per_layer_input": self.vocab
            },
            "_comment": format!("hipfire Gemma 4 {} deterministic gating fixture", self.variant.name())
        })
    }

    fn norm(name: String, dim: usize) -> TensorSpec {
        TensorSpec::f16(name, vec![dim], Init::NormOnes)
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let mut tensors = vec![
            TensorSpec::new(
                "model.language_model.embed_tokens.weight",
                vec![self.vocab, self.hidden],
                Init::Uniform(0.05),
            ),
            Self::norm("model.language_model.norm.weight".into(), self.hidden),
        ];
        if self.ple_dim > 0 {
            tensors.push(TensorSpec::new(
                "model.language_model.embed_tokens_per_layer.weight",
                vec![self.vocab, self.layers * self.ple_dim],
                Init::Uniform(0.05),
            ));
            tensors.push(TensorSpec::new(
                "model.language_model.per_layer_model_projection.weight",
                vec![self.layers * self.ple_dim, self.hidden],
                Init::Uniform(0.05),
            ));
            tensors.push(Self::norm(
                "model.language_model.per_layer_projection_norm.weight".into(),
                self.ple_dim,
            ));
        }

        let shared_start = self
            .layers
            .saturating_sub(usize::from(self.variant == Gemma4ToyVariant::PleSharing) * 2);
        for (layer, layer_type) in self.layer_types().into_iter().enumerate() {
            let prefix = format!("model.language_model.layers.{layer}");
            let attn = format!("{prefix}.self_attn");
            let head_dim = if layer_type == "full_attention" {
                self.global_head_dim
            } else {
                self.local_head_dim
            };
            let kv_heads = if layer_type == "full_attention" {
                self.global_kv_heads
            } else {
                self.local_kv_heads
            };
            let q_dim = self.q_heads * head_dim;
            let kv_dim = kv_heads * head_dim;
            let shared = layer >= shared_start;

            tensors.push(Self::norm(
                format!("{prefix}.input_layernorm.weight"),
                self.hidden,
            ));
            tensors.push(Self::norm(format!("{attn}.q_norm.weight"), head_dim));
            tensors.push(TensorSpec::new(
                format!("{attn}.q_proj.weight"),
                vec![q_dim, self.hidden],
                Init::Uniform(0.05),
            ));
            if !shared {
                tensors.push(Self::norm(format!("{attn}.k_norm.weight"), head_dim));
                tensors.push(TensorSpec::new(
                    format!("{attn}.k_proj.weight"),
                    vec![kv_dim, self.hidden],
                    Init::Uniform(0.05),
                ));
                if self.variant == Gemma4ToyVariant::PleSharing || layer_type != "full_attention" {
                    tensors.push(TensorSpec::new(
                        format!("{attn}.v_proj.weight"),
                        vec![kv_dim, self.hidden],
                        Init::Uniform(0.05),
                    ));
                }
            }
            tensors.push(TensorSpec::new(
                format!("{attn}.o_proj.weight"),
                vec![self.hidden, q_dim],
                Init::Uniform(0.05),
            ));
            tensors.push(Self::norm(
                format!("{prefix}.post_attention_layernorm.weight"),
                self.hidden,
            ));
            tensors.push(Self::norm(
                format!("{prefix}.pre_feedforward_layernorm.weight"),
                self.hidden,
            ));
            tensors.push(Self::norm(
                format!("{prefix}.post_feedforward_layernorm.weight"),
                self.hidden,
            ));
            tensors.push(TensorSpec::new(
                format!("{prefix}.mlp.gate_proj.weight"),
                vec![self.dense_intermediate, self.hidden],
                Init::Uniform(0.05),
            ));
            tensors.push(TensorSpec::new(
                format!("{prefix}.mlp.up_proj.weight"),
                vec![self.dense_intermediate, self.hidden],
                Init::Uniform(0.05),
            ));
            tensors.push(TensorSpec::new(
                format!("{prefix}.mlp.down_proj.weight"),
                vec![self.hidden, self.dense_intermediate],
                Init::Uniform(0.05),
            ));
            tensors.push(TensorSpec::new(
                format!("{prefix}.layer_scalar"),
                vec![1],
                Init::NormOnes,
            ));

            if self.ple_dim > 0 {
                tensors.push(TensorSpec::new(
                    format!("{prefix}.per_layer_input_gate.weight"),
                    vec![self.ple_dim, self.hidden],
                    Init::Uniform(0.05),
                ));
                tensors.push(TensorSpec::new(
                    format!("{prefix}.per_layer_projection.weight"),
                    vec![self.hidden, self.ple_dim],
                    Init::Uniform(0.05),
                ));
                tensors.push(Self::norm(
                    format!("{prefix}.post_per_layer_input_norm.weight"),
                    self.hidden,
                ));
            }

            if self.experts > 0 {
                tensors.push(TensorSpec::new(
                    format!("{prefix}.experts.gate_up_proj"),
                    vec![self.experts, 2 * self.expert_intermediate, self.hidden],
                    Init::Uniform(0.05),
                ));
                tensors.push(TensorSpec::new(
                    format!("{prefix}.experts.down_proj"),
                    vec![self.experts, self.hidden, self.expert_intermediate],
                    Init::Uniform(0.05),
                ));
                tensors.push(Self::norm(format!("{prefix}.router.scale"), self.hidden));
                tensors.push(TensorSpec::f16(
                    format!("{prefix}.router.per_expert_scale"),
                    vec![self.experts],
                    Init::NormOnes,
                ));
                tensors.push(TensorSpec::new(
                    format!("{prefix}.router.proj.weight"),
                    vec![self.experts, self.hidden],
                    Init::Uniform(0.05),
                ));
                for expert in 0..self.experts {
                    let ep = format!("{prefix}.experts.{expert}");
                    tensors.push(TensorSpec::new(
                        format!("{ep}.gate_proj.weight"),
                        vec![self.expert_intermediate, self.hidden],
                        Init::Uniform(0.05),
                    ));
                    tensors.push(TensorSpec::new(
                        format!("{ep}.up_proj.weight"),
                        vec![self.expert_intermediate, self.hidden],
                        Init::Uniform(0.05),
                    ));
                    tensors.push(TensorSpec::new(
                        format!("{ep}.down_proj.weight"),
                        vec![self.hidden, self.expert_intermediate],
                        Init::Uniform(0.05),
                    ));
                }
            }
        }
        tensors
    }

    fn fixture(&self) -> ToyFixture {
        ToyFixture {
            config_json: serde_json::to_string_pretty(&self.config())
                .expect("serialize Gemma 4 toy config"),
            tensors: self.manifest(),
        }
    }
}

pub fn fixture_for(variant: Gemma4ToyVariant, _seed: u64) -> ToyFixture {
    Tiny::for_variant(variant).fixture()
}

impl ToyModel for Gemma4Spec {
    fn fixture(&self, seed: u64) -> ToyFixture {
        fixture_for(Gemma4ToyVariant::Dense, seed)
    }

    fn fixture_names(&self) -> &'static [&'static str] {
        &["dense", "ple-sharing", "dense-moe"]
    }

    fn fixture_named(&self, name: &str, seed: u64) -> Option<ToyFixture> {
        let variant = match name {
            "default" | "dense" => Gemma4ToyVariant::Dense,
            "ple-sharing" => Gemma4ToyVariant::PleSharing,
            "dense-moe" => Gemma4ToyVariant::DenseMoe,
            _ => return None,
        };
        Some(fixture_for(variant, seed))
    }
}

static GEMMA4_SPEC: Gemma4Spec = Gemma4Spec;
register_arch!(GEMMA4_SPEC, Ingest, ToyModel);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    fn has(fixture: &ToyFixture, suffix: &str) -> bool {
        fixture
            .tensors
            .iter()
            .any(|tensor| tensor.name.ends_with(suffix))
    }

    #[test]
    fn registry_exposes_identity_ingest_and_toys() {
        let registry = ArchRegistry::build();
        let arch = registry.get(GEMMA4_ARCH_ID).expect("Gemma 4 registered");
        assert_eq!(arch.family, "gemma4");
        for model_type in [
            "gemma4",
            "gemma4_text",
            "gemma4_unified",
            "gemma4_unified_text",
        ] {
            assert_eq!(
                registry.find_by_model_type(model_type).unwrap().id,
                GEMMA4_ARCH_ID
            );
        }
        assert!(arch.caps.ingest.is_some());
        assert_eq!(
            arch.caps.ingest.unwrap().expert_layout(),
            ExpertLayout::StackedGateUpDown
        );
        assert_eq!(arch.caps.toy_model.unwrap().fixture_names().len(), 3);
    }

    #[test]
    fn sensitive_tensors_remain_at_source_precision() {
        for tensor in [
            "model.language_model.norm.weight",
            "model.language_model.layers.0.layer_scalar",
            "model.language_model.embed_tokens_per_layer.weight",
            "model.language_model.layers.0.router.scale",
            "model.language_model.layers.0.router.per_expert_scale",
        ] {
            assert_eq!(
                Gemma4Spec.precision_class(tensor),
                PrecisionClass::SourcePrecision
            );
        }
        assert_eq!(
            Gemma4Spec.precision_class("model.language_model.layers.0.mlp.up_proj.weight"),
            PrecisionClass::Compressed
        );
    }

    #[test]
    fn dense_fixture_has_mixed_geometry_and_k_equals_v_global() {
        let fixture = fixture_for(Gemma4ToyVariant::Dense, 7);
        let config: serde_json::Value = serde_json::from_str(&fixture.config_json).unwrap();
        assert_eq!(config["text_config"]["head_dim"], 64);
        assert_eq!(config["text_config"]["global_head_dim"], 128);
        assert!(has(&fixture, "layers.1.self_attn.k_proj.weight"));
        assert!(!has(&fixture, "layers.1.self_attn.v_proj.weight"));
    }

    #[test]
    fn ple_fixture_has_real_shared_layers_without_kv_storage_weights() {
        let fixture = fixture_for(Gemma4ToyVariant::PleSharing, 7);
        assert!(has(&fixture, "embed_tokens_per_layer.weight"));
        for layer in 2..4 {
            assert!(!has(
                &fixture,
                &format!("layers.{layer}.self_attn.k_proj.weight")
            ));
            assert!(!has(
                &fixture,
                &format!("layers.{layer}.self_attn.v_proj.weight")
            ));
        }
        assert!(has(&fixture, "layers.0.self_attn.k_proj.weight"));
        assert!(has(&fixture, "layers.1.self_attn.k_proj.weight"));
    }

    #[test]
    fn moe_fixture_has_dense_and_rank3_top8_experts() {
        let fixture = fixture_for(Gemma4ToyVariant::DenseMoe, 7);
        let config: serde_json::Value = serde_json::from_str(&fixture.config_json).unwrap();
        assert_eq!(config["text_config"]["top_k_experts"], 8);
        assert!(has(&fixture, "layers.0.mlp.gate_proj.weight"));
        let experts = fixture
            .tensors
            .iter()
            .find(|tensor| tensor.name.ends_with("layers.0.experts.gate_up_proj"))
            .unwrap();
        assert_eq!(experts.shape, vec![8, 256, 256]);
        assert!(has(&fixture, "layers.0.router.per_expert_scale"));
    }
}
