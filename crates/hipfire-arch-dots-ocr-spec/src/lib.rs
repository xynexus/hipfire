// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the dots.ocr (Qwen2-VL) family (arch_id 8): identity + the
//! `Ingest` quant-policy (shared transformer prior). Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, Init, TensorRole, TensorSpec, ToyFixture, ToyModel,
};

/// dots.ocr family header id.
pub const DOTS_OCR_ARCH_ID: ArchId = ArchId(8);

/// Lean identity marker for the dots.ocr offline spec.
pub struct DotsOcrSpec;

impl Arch for DotsOcrSpec {
    fn id(&self) -> ArchId {
        DOTS_OCR_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "dots-ocr"
    }
}

impl Ingest for DotsOcrSpec {
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

/// Tiny dots.ocr fixture. The text decoder mirrors Qwen2's tiny shape and the
/// vision tower is reduced to a one-block 2x2-patch encoder. Keep the patch and
/// merge dimensions aligned with the production Rust image preprocessing
/// constants so the tiny gates can exercise the image-to-patch path.
struct DotsOcrTiny {
    hidden: usize,
    inter: usize,
    vocab: usize,
    text_layers: usize,
    text_heads: usize,
    text_kv_heads: usize,
    head_dim: usize,
    vision_hidden: usize,
    vision_inter: usize,
    vision_layers: usize,
    vision_heads: usize,
    patch_size: usize,
    spatial_merge_size: usize,
}

impl DotsOcrTiny {
    fn preset() -> Self {
        Self {
            hidden: 256,
            inter: 512,
            vocab: 4096,
            text_layers: 2,
            text_heads: 2,
            text_kv_heads: 1,
            head_dim: 128,
            vision_hidden: 256,
            vision_inter: 512,
            vision_layers: 1,
            vision_heads: 4,
            patch_size: 14,
            spatial_merge_size: 2,
        }
    }

    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["DotsOCRForCausalLM"],
            "model_type": "dots_ocr",
            "text_config": {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": self.hidden,
                "intermediate_size": self.inter,
                "vocab_size": self.vocab,
                "num_hidden_layers": self.text_layers,
                "num_attention_heads": self.text_heads,
                "num_key_value_heads": self.text_kv_heads,
                "head_dim": self.head_dim,
                "attention_bias": true,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "rope_theta": 1_000_000.0,
                "max_position_embeddings": 4096,
                "tie_word_embeddings": true,
                "dtype": "bfloat16"
            },
            "vision_config": {
                "model_type": "dots_vision",
                "embed_dim": self.vision_hidden,
                "hidden_size": self.vision_hidden,
                "out_hidden_size": self.hidden,
                "num_hidden_layers": self.vision_layers,
                "num_attention_heads": self.vision_heads,
                "head_dim": self.vision_hidden / self.vision_heads,
                "intermediate_size": self.vision_inter,
                "patch_size": self.patch_size,
                "spatial_merge_size": self.spatial_merge_size,
                "temporal_patch_size": 1,
                "num_channels": 3,
                "use_bias": false,
                "post_norm": true,
                "rms_norm_eps": 1e-5
            },
            "dtype": "bfloat16",
            "_comment": "hipfire tiny random-init dots.ocr gating fixture — not a real model",
        })
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let q_dim = self.text_heads * self.head_dim;
        let kv_dim = self.text_kv_heads * self.head_dim;
        let mut t = Vec::new();
        t.push(TensorSpec::new(
            "model.embed_tokens.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        t.push(TensorSpec::f16(
            "model.norm.weight",
            vec![h],
            Init::NormOnes,
        ));
        for i in 0..self.text_layers {
            let p = format!("model.layers.{i}");
            t.push(TensorSpec::f16(
                format!("{p}.input_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{p}.self_attn.q_proj.weight"),
                vec![q_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.self_attn.q_proj.bias"),
                vec![q_dim],
                Init::Uniform(0.02),
            ));
            t.push(TensorSpec::new(
                format!("{p}.self_attn.k_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.self_attn.k_proj.bias"),
                vec![kv_dim],
                Init::Uniform(0.02),
            ));
            t.push(TensorSpec::new(
                format!("{p}.self_attn.v_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.self_attn.v_proj.bias"),
                vec![kv_dim],
                Init::Uniform(0.02),
            ));
            t.push(TensorSpec::new(
                format!("{p}.self_attn.o_proj.weight"),
                vec![h, q_dim],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.gate_proj.weight"),
                vec![self.inter, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.up_proj.weight"),
                vec![self.inter, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.down_proj.weight"),
                vec![h, self.inter],
                Init::Uniform(0.05),
            ));
        }

        let vh = self.vision_hidden;
        let vi = self.vision_inter;
        let patch_dim = 3 * self.patch_size * self.patch_size;
        let merge_dim = vh * self.spatial_merge_size * self.spatial_merge_size;
        t.push(TensorSpec::new(
            "vision_tower.patch_embed.patchifier.proj.weight",
            vec![vh, 3, self.patch_size, self.patch_size],
            Init::Uniform(0.03),
        ));
        t.push(TensorSpec::f16(
            "vision_tower.patch_embed.patchifier.proj.bias",
            vec![vh],
            Init::Zeros,
        ));
        t.push(TensorSpec::f16(
            "vision_tower.patch_embed.patchifier.norm.weight",
            vec![vh],
            Init::NormOnes,
        ));
        debug_assert_eq!(patch_dim, 3 * self.patch_size * self.patch_size);
        for i in 0..self.vision_layers {
            let p = format!("vision_tower.blocks.{i}");
            t.push(TensorSpec::f16(
                format!("{p}.norm1.weight"),
                vec![vh],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{p}.attn.qkv.weight"),
                vec![3 * vh, vh],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.attn.proj.weight"),
                vec![vh, vh],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.norm2.weight"),
                vec![vh],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.fc1.weight"),
                vec![vi, vh],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.fc2.weight"),
                vec![vh, vi],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.fc3.weight"),
                vec![vi, vh],
                Init::Uniform(0.03),
            ));
        }
        t.push(TensorSpec::f16(
            "vision_tower.post_trunk_norm.weight",
            vec![vh],
            Init::NormOnes,
        ));
        t.push(TensorSpec::f16(
            "vision_tower.merger.ln_q.weight",
            vec![vh],
            Init::NormOnes,
        ));
        t.push(TensorSpec::f16(
            "vision_tower.merger.ln_q.bias",
            vec![vh],
            Init::Zeros,
        ));
        t.push(TensorSpec::new(
            "vision_tower.merger.mlp.0.weight",
            vec![merge_dim, merge_dim],
            Init::Uniform(0.03),
        ));
        t.push(TensorSpec::f16(
            "vision_tower.merger.mlp.0.bias",
            vec![merge_dim],
            Init::Zeros,
        ));
        t.push(TensorSpec::new(
            "vision_tower.merger.mlp.2.weight",
            vec![h, merge_dim],
            Init::Uniform(0.03),
        ));
        t.push(TensorSpec::f16(
            "vision_tower.merger.mlp.2.bias",
            vec![h],
            Init::Zeros,
        ));
        t
    }
}

impl ToyModel for DotsOcrSpec {
    fn fixture(&self, _seed: u64) -> ToyFixture {
        let m = DotsOcrTiny::preset();
        ToyFixture {
            config_json: serde_json::to_string_pretty(&m.config_json())
                .expect("serialize dots-ocr toy config"),
            tensors: m.manifest(),
        }
    }
}

static DOTS_OCR_SPEC: DotsOcrSpec = DotsOcrSpec;
register_arch!(DOTS_OCR_SPEC, Ingest, ToyModel);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_ingest() {
        let reg = ArchRegistry::build();
        let a = reg.get(DOTS_OCR_ARCH_ID).expect("dots-ocr spec registered");
        assert_eq!(a.family, "dots-ocr");
        assert!(a.caps.ingest.is_some());
        assert!(a.caps.toy_model.is_some());
    }

    #[test]
    fn toy_fixture_has_text_and_vision() {
        let f = DotsOcrSpec.fixture(0);
        let config: serde_json::Value = serde_json::from_str(&f.config_json).unwrap();
        assert_eq!(
            config.get("model_type").and_then(|v| v.as_str()),
            Some("dots_ocr")
        );
        assert!(config.get("text_config").is_some(), "nested text config");
        assert!(config.get("vision_config").is_some(), "vision config");
        let has = |suf: &str| f.tensors.iter().any(|s| s.name.ends_with(suf));
        assert!(has("model.embed_tokens.weight"), "Qwen2 text weights");
        assert!(
            has("vision_tower.patch_embed.patchifier.proj.weight"),
            "vision patch embed"
        );
        assert!(has("vision_tower.merger.mlp.2.weight"), "vision merger");
        let params: usize = f
            .tensors
            .iter()
            .map(|s| s.shape.iter().product::<usize>())
            .sum();
        assert!(params < 10_000_000, "dots-ocr fixture must stay tiny");
    }
}
