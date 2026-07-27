// SPDX-License-Identifier: Apache-2.0
//! Lean offline identity, quant-policy, and deterministic fixture for the
//! Cohere2-MoE family used by BLS Mini Code 1.0.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, Init, TensorRole, TensorSpec, ToyFixture, ToyModel, ARCH_ID_COHERE2_MOE,
};

pub const COHERE2_MOE_ARCH_ID: ArchId = ArchId(ARCH_ID_COHERE2_MOE as u16);

pub struct Cohere2MoeSpec;

impl Arch for Cohere2MoeSpec {
    fn id(&self) -> ArchId {
        COHERE2_MOE_ARCH_ID
    }

    fn family(&self) -> &'static str {
        "cohere2-moe"
    }

    fn model_types(&self) -> &'static [&'static str] {
        &["cohere2_moe"]
    }
}

impl Ingest for Cohere2MoeSpec {
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

fn tiny_fixture() -> ToyFixture {
    let hidden = 256usize;
    let dense_inter = 512usize;
    let expert_inter = 256usize;
    let vocab = 512usize;
    let layers = 2usize;
    let q_heads = 4usize;
    let kv_heads = 2usize;
    let head_dim = 64usize;
    let experts = 4usize;
    let mut tensors = vec![
        TensorSpec::new(
            "model.embed_tokens.weight",
            vec![vocab, hidden],
            Init::Uniform(0.05),
        ),
        TensorSpec::f16("model.norm.weight", vec![hidden], Init::NormOnes),
    ];
    for layer in 0..layers {
        let prefix = format!("model.layers.{layer}");
        let attn = format!("{prefix}.self_attn");
        tensors.push(TensorSpec::f16(
            format!("{prefix}.input_layernorm.weight"),
            vec![hidden],
            Init::NormOnes,
        ));
        for (name, shape) in [
            (
                format!("{attn}.q_proj.weight"),
                vec![q_heads * head_dim, hidden],
            ),
            (
                format!("{attn}.k_proj.weight"),
                vec![kv_heads * head_dim, hidden],
            ),
            (
                format!("{attn}.v_proj.weight"),
                vec![kv_heads * head_dim, hidden],
            ),
            (
                format!("{attn}.o_proj.weight"),
                vec![hidden, q_heads * head_dim],
            ),
        ] {
            tensors.push(TensorSpec::new(name, shape, Init::Uniform(0.05)));
        }
        if layer == 0 {
            for (name, shape) in [
                (
                    format!("{prefix}.mlp.gate_proj.weight"),
                    vec![dense_inter, hidden],
                ),
                (
                    format!("{prefix}.mlp.up_proj.weight"),
                    vec![dense_inter, hidden],
                ),
                (
                    format!("{prefix}.mlp.down_proj.weight"),
                    vec![hidden, dense_inter],
                ),
            ] {
                tensors.push(TensorSpec::new(name, shape, Init::Uniform(0.05)));
            }
        } else {
            tensors.push(TensorSpec::new(
                format!("{prefix}.mlp.gate.weight"),
                vec![experts, hidden],
                Init::Uniform(0.05),
            ));
            for expert in 0..experts {
                let ep = format!("{prefix}.mlp.experts.{expert}");
                for (name, shape) in [
                    (format!("{ep}.gate_proj.weight"), vec![expert_inter, hidden]),
                    (format!("{ep}.up_proj.weight"), vec![expert_inter, hidden]),
                    (format!("{ep}.down_proj.weight"), vec![hidden, expert_inter]),
                ] {
                    tensors.push(TensorSpec::new(name, shape, Init::Uniform(0.05)));
                }
            }
        }
    }
    let config = serde_json::json!({
        "architectures": ["Cohere2MoeForCausalLM"],
        "model_type": "cohere2_moe",
        "dtype": "bfloat16",
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": q_heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": vocab,
        "intermediate_size": expert_inter,
        "prefix_dense_intermediate_size": dense_inter,
        "first_k_dense_replace": 1,
        "prefix_dense_sliding_window_pattern": 1,
        "num_experts": experts,
        "num_experts_per_tok": 2,
        "num_shared_experts": 0,
        "expert_selection_fn": "sigmoid",
        "norm_topk_prob": false,
        "use_parallel_block": true,
        "use_gated_activation": true,
        "hidden_act": "silu",
        "attention_bias": false,
        "layer_types": ["full_attention", "sliding_attention"],
        "sliding_window": 128,
        "max_position_embeddings": 512,
        "rope_theta": 50000.0,
        "rms_norm_eps": 1e-6,
        "logit_scale": 1.0,
        "tie_word_embeddings": true,
    });
    ToyFixture {
        config_json: serde_json::to_string_pretty(&config).expect("serialize Cohere2 fixture"),
        tensors,
    }
}

impl ToyModel for Cohere2MoeSpec {
    fn fixture(&self, _seed: u64) -> ToyFixture {
        tiny_fixture()
    }
}

static COHERE2_MOE_SPEC: Cohere2MoeSpec = Cohere2MoeSpec;
register_arch!(COHERE2_MOE_SPEC, Ingest, ToyModel);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::{ArchRegistry, ExpertLayout};

    #[test]
    fn registers_identity_and_split_expert_ingest() {
        let registry = ArchRegistry::build();
        let arch = registry
            .find_by_model_type("cohere2_moe")
            .expect("Cohere2 identity");
        assert_eq!(arch.id, COHERE2_MOE_ARCH_ID);
        assert_eq!(arch.family, "cohere2-moe");
        let ingest = arch.caps.ingest.expect("Cohere2 ingest");
        assert_eq!(ingest.expert_layout(), ExpertLayout::None);
        assert_eq!(
            ingest.role("model.layers.1.mlp.gate.weight"),
            TensorRole::Router
        );
        assert_eq!(
            ingest.role("model.layers.1.mlp.experts.3.down_proj.weight"),
            TensorRole::Expert
        );
    }

    #[test]
    fn fixture_models_dense_first_then_parallel_moe() {
        let fixture = tiny_fixture();
        assert!(fixture
            .tensors
            .iter()
            .any(|tensor| tensor.name == "model.layers.0.mlp.gate_proj.weight"));
        assert!(fixture
            .tensors
            .iter()
            .any(|tensor| tensor.name == "model.layers.1.mlp.experts.3.down_proj.weight"));
        assert!(fixture.config_json.contains("\"use_parallel_block\": true"));
    }
}
