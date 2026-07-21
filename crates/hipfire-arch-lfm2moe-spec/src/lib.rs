// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the LFM2.5 family (arch_id 11; dense + MoE variants):
//! identity + the `Ingest` quant-policy (shared transformer prior, which covers the
//! short-conv mixer). Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, Init, TensorRole, TensorSpec, ToyFixture, ToyModel,
};

/// LFM2.5 family header id.
pub const LFM2_ARCH_ID: ArchId = ArchId(11);

/// Lean identity marker for the LFM2.5 offline spec.
pub struct Lfm2Spec;

impl Arch for Lfm2Spec {
    fn id(&self) -> ArchId {
        LFM2_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "lfm2"
    }

    fn model_types(&self) -> &'static [&'static str] {
        &["lfm2", "lfm2_moe"]
    }
}

impl Ingest for Lfm2Spec {
    fn role(&self, tensor: &str) -> TensorRole {
        transformer_role(tensor)
    }
    fn importance(&self, tensor: &str) -> u8 {
        default_importance(self.role(tensor))
    }
    fn requires(&self, tensor: &str) -> CapReq {
        default_requires(self.role(tensor))
    }
}

/// Tiny LFM2.5-MoE fixture with one short-conv+dense layer and one
/// full-attention+MoE layer. Expert dimensions stay 256-aligned so the grouped
/// MQ/HFQ expert kernels exercised by the tiny gates can load the fixture.
struct Lfm2Tiny {
    hidden: usize,
    dense_inter: usize,
    moe_inter: usize,
    vocab: usize,
    layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    conv_kernel: usize,
    experts: usize,
    experts_per_tok: usize,
}

impl Lfm2Tiny {
    fn preset() -> Self {
        Self {
            hidden: 256,
            dense_inter: 256,
            moe_inter: 256,
            vocab: 4096,
            layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 128,
            conv_kernel: 3,
            experts: 8,
            experts_per_tok: 2,
        }
    }

    fn layer_types(&self) -> Vec<&'static str> {
        vec!["conv", "full_attention"]
    }

    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Lfm2MoeForCausalLM"],
            "model_type": "lfm2_moe",
            "vocab_size": self.vocab,
            "hidden_size": self.hidden,
            "num_hidden_layers": self.layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "conv_L_cache": self.conv_kernel,
            "intermediate_size": self.dense_inter,
            "moe_intermediate_size": self.moe_inter,
            "num_experts": self.experts,
            "num_experts_per_tok": self.experts_per_tok,
            "num_dense_layers": 1,
            "rope_parameters": { "rope_theta": 5_000_000.0 },
            "norm_eps": 1e-5,
            "max_position_embeddings": 4096,
            "norm_topk_prob": true,
            "use_expert_bias": true,
            "routed_scaling_factor": 1.0,
            "tie_word_embeddings": true,
            "layer_types": self.layer_types(),
            "dtype": "bfloat16",
            "_comment": "hipfire tiny random-init gating fixture - not a real model",
        })
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let mut t = Vec::new();
        t.push(TensorSpec::new(
            "model.embed_tokens.weight",
            vec![self.vocab, h],
            Init::Uniform(0.04),
        ));
        t.push(TensorSpec::f16(
            "model.embedding_norm.weight",
            vec![h],
            Init::NormOnes,
        ));

        for i in 0..self.layers {
            let p = format!("model.layers.{i}");
            t.push(TensorSpec::f16(
                format!("{p}.operator_norm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.ffn_norm.weight"),
                vec![h],
                Init::NormOnes,
            ));

            if i == 0 {
                t.push(TensorSpec::new(
                    format!("{p}.conv.in_proj.weight"),
                    vec![3 * h, h],
                    Init::Uniform(0.04),
                ));
                t.push(TensorSpec::f16(
                    format!("{p}.conv.conv.weight"),
                    vec![h, 1, self.conv_kernel],
                    Init::Uniform(0.04),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.conv.out_proj.weight"),
                    vec![h, h],
                    Init::Uniform(0.04),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.feed_forward.w1.weight"),
                    vec![self.dense_inter, h],
                    Init::Uniform(0.04),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.feed_forward.w3.weight"),
                    vec![self.dense_inter, h],
                    Init::Uniform(0.04),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.feed_forward.w2.weight"),
                    vec![h, self.dense_inter],
                    Init::Uniform(0.04),
                ));
            } else {
                let sa = format!("{p}.self_attn");
                t.push(TensorSpec::new(
                    format!("{sa}.q_proj.weight"),
                    vec![q_dim, h],
                    Init::Uniform(0.04),
                ));
                t.push(TensorSpec::new(
                    format!("{sa}.k_proj.weight"),
                    vec![kv_dim, h],
                    Init::Uniform(0.04),
                ));
                t.push(TensorSpec::new(
                    format!("{sa}.v_proj.weight"),
                    vec![kv_dim, h],
                    Init::Uniform(0.04),
                ));
                t.push(TensorSpec::new(
                    format!("{sa}.out_proj.weight"),
                    vec![h, q_dim],
                    Init::Uniform(0.04),
                ));
                t.push(TensorSpec::f16(
                    format!("{sa}.q_layernorm.weight"),
                    vec![self.head_dim],
                    Init::NormOnes,
                ));
                t.push(TensorSpec::f16(
                    format!("{sa}.k_layernorm.weight"),
                    vec![self.head_dim],
                    Init::NormOnes,
                ));

                let ff = format!("{p}.feed_forward");
                t.push(TensorSpec::new(
                    format!("{ff}.gate.weight"),
                    vec![self.experts, h],
                    Init::Uniform(0.04),
                ));
                t.push(TensorSpec::f16(
                    format!("{ff}.expert_bias"),
                    vec![self.experts],
                    Init::Uniform(0.02),
                ));
                for e in 0..self.experts {
                    let ep = format!("{ff}.experts.{e}");
                    t.push(TensorSpec::new(
                        format!("{ep}.w1.weight"),
                        vec![self.moe_inter, h],
                        Init::Uniform(0.04),
                    ));
                    t.push(TensorSpec::new(
                        format!("{ep}.w3.weight"),
                        vec![self.moe_inter, h],
                        Init::Uniform(0.04),
                    ));
                    t.push(TensorSpec::new(
                        format!("{ep}.w2.weight"),
                        vec![h, self.moe_inter],
                        Init::Uniform(0.04),
                    ));
                }
            }
        }
        t
    }
}

impl ToyModel for Lfm2Spec {
    fn fixture(&self, _seed: u64) -> ToyFixture {
        let m = Lfm2Tiny::preset();
        ToyFixture {
            config_json: serde_json::to_string_pretty(&m.config_json())
                .expect("serialize lfm2 toy config"),
            tensors: m.manifest(),
        }
    }
}

static LFM2_SPEC: Lfm2Spec = Lfm2Spec;
register_arch!(LFM2_SPEC, Ingest, ToyModel);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_ingest() {
        let reg = ArchRegistry::build();
        let a = reg.get(LFM2_ARCH_ID).expect("lfm2 spec registered");
        assert_eq!(a.family, "lfm2");
        assert!(a.caps.ingest.is_some());
        assert!(a.caps.toy_model.is_some());
    }

    #[test]
    fn toy_fixture_declared() {
        let f = Lfm2Spec.fixture(0);
        let has = |sub: &str| f.tensors.iter().any(|s| s.name.contains(sub));
        assert!(f.config_json.contains("\"model_type\": \"lfm2_moe\""));
        assert!(has(".conv.in_proj.weight"), "has short-conv mixer");
        assert!(has(".self_attn.q_proj.weight"), "has attention mixer");
        assert!(has(".feed_forward.w1.weight"), "has dense FFN");
        assert!(has(".feed_forward.experts.0.w1.weight"), "has MoE experts");
        let n_params: usize = f
            .tensors
            .iter()
            .map(|s| s.shape.iter().product::<usize>())
            .sum();
        assert!(n_params < 10_000_000, "fixture must stay <10M params");
    }
}
