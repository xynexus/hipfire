// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Zyphra ZAYA1 MoE family (arch_id 16): identity + the
//! `Ingest` quant-policy (shared transformer prior). Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    ExpertLayout, Ingest, Init, TensorRole, TensorSpec, ToyFixture, ToyModel,
};

/// ZAYA1 family header id.
pub const ZAYA_ARCH_ID: ArchId = ArchId(16);

/// Lean identity marker for the ZAYA1 offline spec.
pub struct ZayaSpec;

impl Arch for ZayaSpec {
    fn id(&self) -> ArchId {
        ZAYA_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "zaya"
    }
    fn model_types(&self) -> &'static [&'static str] {
        &["zaya"]
    }
}

impl Ingest for ZayaSpec {
    fn role(&self, tensor: &str) -> TensorRole {
        transformer_role(tensor)
    }
    fn importance(&self, tensor: &str) -> u8 {
        default_importance(self.role(tensor))
    }
    fn requires(&self, tensor: &str) -> CapReq {
        default_requires(self.role(tensor))
    }
    fn expert_layout(&self) -> ExpertLayout {
        ExpertLayout::StackedGateUpDown
    }
}

struct ZayaTiny {
    hidden: usize,
    vocab: usize,
    blocks: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    moe_inter: usize,
    router_hidden: usize,
    experts: usize,
    cca_time0: usize,
    cca_time1: usize,
}

impl ZayaTiny {
    fn preset() -> Self {
        Self {
            hidden: 256,
            vocab: 4096,
            blocks: 2,
            heads: 2,
            kv_heads: 1,
            head_dim: 128,
            moe_inter: 256,
            router_hidden: 256,
            experts: 4,
            cca_time0: 2,
            cca_time1: 2,
        }
    }

    fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    fn conv_channels(&self) -> usize {
        self.q_dim() + self.kv_dim()
    }

    fn router_experts(&self) -> usize {
        self.experts + 1
    }

    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["ZayaForCausalLM"],
            "attention_bias": false,
            "bos_token_id": 2,
            "cca_time0": self.cca_time0,
            "cca_time1": self.cca_time1,
            "dtype": "bfloat16",
            "eos_token_id": 106,
            "head_dim": self.head_dim,
            "hidden_act": "silu",
            "hidden_size": self.hidden,
            "lm_head_bias": false,
            "max_position_embeddings": 4096,
            "model_type": "zaya",
            "moe_intermediate_size": self.moe_inter,
            "moe_router_topk": 1,
            "num_attention_heads": self.heads,
            "num_experts": self.experts,
            "num_hidden_layers": self.blocks,
            "num_key_value_heads": self.kv_heads,
            "pad_token_id": 0,
            "partial_rotary_factor": 0.5,
            "rms_norm_eps": 1e-5,
            "router_hidden_size": self.router_hidden,
            "rope_theta": 5_000_000.0,
            "sliding_window": null,
            "tie_word_embeddings": true,
            "vocab_size": self.vocab,
            "zaya_use_eda": true,
            "zaya_use_mod": true,
            "_comment": "hipfire tiny random-init gating fixture - not a real model",
        })
    }

    fn residual_scale(t: &mut Vec<TensorSpec>, base: &str, h: usize) {
        t.push(TensorSpec::f16(
            format!("{base}.hidden_states_scale"),
            vec![h],
            Init::NormOnes,
        ));
        t.push(TensorSpec::f16(
            format!("{base}.hidden_states_bias"),
            vec![h],
            Init::Zeros,
        ));
        t.push(TensorSpec::f16(
            format!("{base}.residual_scale"),
            vec![h],
            Init::NormOnes,
        ));
        t.push(TensorSpec::f16(
            format!("{base}.residual_bias"),
            vec![h],
            Init::Zeros,
        ));
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let q_dim = self.q_dim();
        let kv_dim = self.kv_dim();
        let v_half = kv_dim / 2;
        let conv_ch = self.conv_channels();
        let rh = self.router_hidden;
        let mut t = Vec::new();

        t.push(TensorSpec::new(
            "model.embed_tokens.weight",
            vec![self.vocab, h],
            Init::Uniform(0.04),
        ));
        t.push(TensorSpec::f16(
            "model.input_hidden_states_scale",
            vec![h],
            Init::NormOnes,
        ));
        t.push(TensorSpec::f16(
            "model.input_hidden_states_bias",
            vec![h],
            Init::Zeros,
        ));
        t.push(TensorSpec::f16(
            "model.norm.weight",
            vec![h],
            Init::NormOnes,
        ));

        for l in 0..self.blocks {
            let p = format!("model.layers.{l}");
            let attn = format!("{p}.self_attn");
            let qkv = format!("{attn}.qkv_proj");
            let gate = format!("{p}.mlp.gate");
            let rmlp = format!("{gate}.router_mlp");

            t.push(TensorSpec::f16(
                format!("{p}.input_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{qkv}.q_proj.weight"),
                vec![q_dim, h],
                Init::Uniform(0.035),
            ));
            t.push(TensorSpec::new(
                format!("{qkv}.k_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.035),
            ));
            t.push(TensorSpec::new(
                format!("{qkv}.v_proj_current.weight"),
                vec![v_half, h],
                Init::Uniform(0.035),
            ));
            t.push(TensorSpec::new(
                format!("{qkv}.v_proj_delayed.weight"),
                vec![v_half, h],
                Init::Uniform(0.035),
            ));
            t.push(TensorSpec::f16(
                format!("{qkv}.conv_qk_depthwise.weight"),
                vec![conv_ch, self.cca_time0],
                Init::Uniform(0.02),
            ));
            t.push(TensorSpec::f16(
                format!("{qkv}.conv_qk_depthwise.bias"),
                vec![conv_ch],
                Init::Zeros,
            ));
            t.push(TensorSpec::f16(
                format!("{qkv}.conv_qk_grouped.weight"),
                vec![conv_ch, self.head_dim, self.cca_time1],
                Init::Uniform(0.02),
            ));
            t.push(TensorSpec::f16(
                format!("{qkv}.conv_qk_grouped.bias"),
                vec![conv_ch],
                Init::Zeros,
            ));
            t.push(TensorSpec::f16(
                format!("{attn}.qk_norm.temp"),
                vec![self.kv_heads],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{attn}.o_proj.weight"),
                vec![h, q_dim],
                Init::Uniform(0.035),
            ));

            t.push(TensorSpec::new(
                format!("{gate}.down_proj.weight"),
                vec![rh, h],
                Init::Uniform(0.035),
            ));
            t.push(TensorSpec::f16(
                format!("{gate}.down_proj.bias"),
                vec![rh],
                Init::Zeros,
            ));
            if l != 0 {
                t.push(TensorSpec::f16(
                    format!("{gate}.router_states_scale"),
                    vec![rh],
                    Init::NormOnes,
                ));
            }
            t.push(TensorSpec::f16(
                format!("{rmlp}.norm.weight"),
                vec![rh],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{rmlp}.fc1.weight"),
                vec![rh, rh],
                Init::Uniform(0.035),
            ));
            t.push(TensorSpec::f16(
                format!("{rmlp}.fc1.bias"),
                vec![rh],
                Init::Zeros,
            ));
            t.push(TensorSpec::new(
                format!("{rmlp}.fc2.weight"),
                vec![rh, rh],
                Init::Uniform(0.035),
            ));
            t.push(TensorSpec::f16(
                format!("{rmlp}.fc2.bias"),
                vec![rh],
                Init::Zeros,
            ));
            t.push(TensorSpec::new(
                format!("{rmlp}.out_proj.weight"),
                vec![self.router_experts(), rh],
                Init::Uniform(0.035),
            ));
            t.push(TensorSpec::f16(
                format!("{gate}.balancing_biases"),
                vec![self.router_experts()],
                Init::Zeros,
            ));

            for e in 0..self.experts {
                t.push(TensorSpec::new(
                    format!("{p}.mlp.experts.{e}.gate_up_proj.weight"),
                    vec![2 * self.moe_inter, h],
                    Init::Uniform(0.035),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.experts.{e}.down_proj.weight"),
                    vec![h, self.moe_inter],
                    Init::Uniform(0.035),
                ));
            }

            Self::residual_scale(&mut t, &format!("{p}.post_attention_residual_scale"), h);
            Self::residual_scale(&mut t, &format!("{p}.post_mlp_residual_scale"), h);
        }

        t
    }
}

impl ToyModel for ZayaSpec {
    fn fixture(&self, _seed: u64) -> ToyFixture {
        let m = ZayaTiny::preset();
        ToyFixture {
            config_json: serde_json::to_string_pretty(&m.config_json())
                .expect("serialize zaya toy config"),
            tensors: m.manifest(),
        }
    }
}

static ZAYA_SPEC: ZayaSpec = ZayaSpec;
register_arch!(ZAYA_SPEC, Ingest, ToyModel);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_ingest() {
        let reg = ArchRegistry::build();
        let a = reg.get(ZAYA_ARCH_ID).expect("zaya spec registered");
        assert_eq!(a.family, "zaya");
        assert!(a.caps.ingest.is_some());
        assert!(a.caps.toy_model.is_some());
    }

    #[test]
    fn tiny_fixture_has_split_experts_and_eda_block() {
        let f = ZAYA_SPEC.fixture(42);
        let has = |name: &str| f.tensors.iter().any(|t| t.name == name);
        assert!(has("model.layers.0.mlp.experts.0.gate_up_proj.weight"));
        assert!(has("model.layers.0.mlp.experts.0.down_proj.weight"));
        assert!(!has("model.layers.0.mlp.gate.router_states_scale"));
        assert!(has("model.layers.1.mlp.gate.router_states_scale"));
        assert!(has("model.layers.1.post_mlp_residual_scale.residual_bias"));
        let params: usize = f
            .tensors
            .iter()
            .map(|t| t.shape.iter().product::<usize>())
            .sum();
        assert!(params < 10_000_000, "zaya tiny fixture must stay small");
    }
}
