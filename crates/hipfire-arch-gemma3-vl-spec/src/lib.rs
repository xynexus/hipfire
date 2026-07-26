// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Gemma-3 multimodal family (arch_id 13): identity + the
//! `Ingest` quant-policy (shared transformer prior; the SigLIP vision tensors fall
//! through to the generic roles). Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, Init, TensorRole, TensorSpec, ToyFixture, ToyModel,
};

/// Gemma-3 multimodal family header id.
pub const GEMMA3_VL_ARCH_ID: ArchId = ArchId(13);

/// Lean identity marker for the Gemma-3 multimodal offline spec.
pub struct Gemma3VlSpec;

impl Arch for Gemma3VlSpec {
    fn id(&self) -> ArchId {
        GEMMA3_VL_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "gemma3-vl"
    }
}

impl Ingest for Gemma3VlSpec {
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

/// Tiny Gemma3-VL fixture. The decoder mirrors the Gemma3 text tiny fixture under
/// the multimodal `language_model.` prefix, while the SigLIP/projector side is
/// reduced to a one-layer, 2x2-patch tower. The tiny gate uses the backend's
/// text-only path, but the artifact still has to load the complete multimodal
/// bundle so vision/projector tensor naming stays covered.
struct Gemma3VlTiny {
    text_hidden: usize,
    text_inter: usize,
    vocab: usize,
    text_layers: usize,
    text_heads: usize,
    text_kv_heads: usize,
    text_head_dim: usize,
    vision_hidden: usize,
    vision_inter: usize,
    vision_layers: usize,
    vision_heads: usize,
    patch_size: usize,
    image_size: usize,
    mm_tokens: usize,
}

impl Gemma3VlTiny {
    fn preset() -> Self {
        Self {
            text_hidden: 256,
            text_inter: 512,
            vocab: 4096,
            text_layers: 4,
            text_heads: 2,
            text_kv_heads: 1,
            text_head_dim: 128,
            vision_hidden: 256,
            vision_inter: 512,
            vision_layers: 1,
            vision_heads: 4,
            patch_size: 16,
            image_size: 32,
            mm_tokens: 1,
        }
    }

    fn config_json(&self) -> String {
        format!(
            r#"{{
  "architectures": [
    "Gemma3ForConditionalGeneration"
  ],
  "model_type": "gemma3",
  "image_token_index": 4095,
  "boi_token_index": 4093,
  "eoi_token_index": 4094,
  "mm_tokens_per_image": {mm_tokens},
  "text_config": {{
    "model_type": "gemma3_text",
    "hidden_size": {text_hidden},
    "intermediate_size": {text_inter},
    "vocab_size": {vocab},
    "num_hidden_layers": {text_layers},
    "num_attention_heads": {text_heads},
    "num_key_value_heads": {text_kv_heads},
    "head_dim": {text_head_dim},
    "query_pre_attn_scalar": {text_head_dim},
    "sliding_window": 64,
    "sliding_window_pattern": 2,
    "rope_theta": 1000000.0,
    "rope_local_base_freq": 10000.0,
    "hidden_activation": "gelu_pytorch_tanh",
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 4096,
    "tie_word_embeddings": true,
    "dtype": "bfloat16"
  }},
  "vision_config": {{
    "model_type": "siglip_vision_model",
    "hidden_size": {vision_hidden},
    "num_hidden_layers": {vision_layers},
    "num_attention_heads": {vision_heads},
    "intermediate_size": {vision_inter},
    "image_size": {image_size},
    "patch_size": {patch_size},
    "num_channels": 3,
    "layer_norm_eps": 1e-6
  }},
  "dtype": "bfloat16",
  "_comment": "hipfire tiny random-init Gemma3-VL gating fixture — not a real model"
}}"#,
            mm_tokens = self.mm_tokens,
            text_hidden = self.text_hidden,
            text_inter = self.text_inter,
            vocab = self.vocab,
            text_layers = self.text_layers,
            text_heads = self.text_heads,
            text_kv_heads = self.text_kv_heads,
            text_head_dim = self.text_head_dim,
            vision_hidden = self.vision_hidden,
            vision_layers = self.vision_layers,
            vision_heads = self.vision_heads,
            vision_inter = self.vision_inter,
            image_size = self.image_size,
            patch_size = self.patch_size,
        )
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.text_hidden;
        let q_dim = self.text_heads * self.text_head_dim;
        let kv_dim = self.text_kv_heads * self.text_head_dim;
        let mut t = Vec::new();
        t.push(TensorSpec::new(
            "language_model.model.embed_tokens.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        t.push(TensorSpec::f16(
            "language_model.model.norm.weight",
            vec![h],
            Init::NormOnes,
        ));
        for i in 0..self.text_layers {
            let p = format!("language_model.model.layers.{i}");
            let sa = format!("{p}.self_attn");
            t.push(TensorSpec::f16(
                format!("{p}.input_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{sa}.q_norm.weight"),
                vec![self.text_head_dim],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{sa}.k_norm.weight"),
                vec![self.text_head_dim],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{sa}.q_proj.weight"),
                vec![q_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.k_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.v_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.o_proj.weight"),
                vec![h, q_dim],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.pre_feedforward_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.post_feedforward_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.gate_proj.weight"),
                vec![self.text_inter, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.up_proj.weight"),
                vec![self.text_inter, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.down_proj.weight"),
                vec![h, self.text_inter],
                Init::Uniform(0.05),
            ));
        }

        let vh = self.vision_hidden;
        let vi = self.vision_inter;
        let vp = "vision_tower.vision_model";
        let patch_dim = 3 * self.patch_size * self.patch_size;
        let patches_per_side = self.image_size / self.patch_size;
        let n_patches = patches_per_side * patches_per_side;
        t.push(TensorSpec::new(
            format!("{vp}.embeddings.patch_embedding.weight"),
            vec![vh, 3, self.patch_size, self.patch_size],
            Init::Uniform(0.03),
        ));
        t.push(TensorSpec::f16(
            format!("{vp}.embeddings.patch_embedding.bias"),
            vec![vh],
            Init::Zeros,
        ));
        t.push(TensorSpec::new(
            format!("{vp}.embeddings.position_embedding.weight"),
            vec![n_patches, vh],
            Init::Uniform(0.01),
        ));
        debug_assert_eq!(patch_dim, 3 * self.patch_size * self.patch_size);
        for i in 0..self.vision_layers {
            let p = format!("{vp}.encoder.layers.{i}");
            t.push(TensorSpec::f16(
                format!("{p}.layer_norm1.weight"),
                vec![vh],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.layer_norm1.bias"),
                vec![vh],
                Init::Zeros,
            ));
            for proj in ["q_proj", "k_proj", "v_proj"] {
                t.push(TensorSpec::new(
                    format!("{p}.self_attn.{proj}.weight"),
                    vec![vh, vh],
                    Init::Uniform(0.03),
                ));
                t.push(TensorSpec::f16(
                    format!("{p}.self_attn.{proj}.bias"),
                    vec![vh],
                    Init::Zeros,
                ));
            }
            t.push(TensorSpec::new(
                format!("{p}.self_attn.out_proj.weight"),
                vec![vh, vh],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.self_attn.out_proj.bias"),
                vec![vh],
                Init::Zeros,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.layer_norm2.weight"),
                vec![vh],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.layer_norm2.bias"),
                vec![vh],
                Init::Zeros,
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.fc1.weight"),
                vec![vi, vh],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.mlp.fc1.bias"),
                vec![vi],
                Init::Zeros,
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.fc2.weight"),
                vec![vh, vi],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.mlp.fc2.bias"),
                vec![vh],
                Init::Zeros,
            ));
        }
        t.push(TensorSpec::f16(
            format!("{vp}.post_layernorm.weight"),
            vec![vh],
            Init::NormOnes,
        ));
        t.push(TensorSpec::f16(
            format!("{vp}.post_layernorm.bias"),
            vec![vh],
            Init::Zeros,
        ));
        t.push(TensorSpec::f16(
            "multi_modal_projector.mm_soft_emb_norm.weight",
            vec![vh],
            Init::NormOnes,
        ));
        t.push(TensorSpec::new(
            "multi_modal_projector.mm_input_projection_weight",
            vec![vh, h],
            Init::Uniform(0.03),
        ));
        t
    }
}

impl ToyModel for Gemma3VlSpec {
    fn fixture(&self, _seed: u64) -> ToyFixture {
        let m = Gemma3VlTiny::preset();
        ToyFixture {
            config_json: m.config_json(),
            tensors: m.manifest(),
        }
    }
}

static GEMMA3_VL_SPEC: Gemma3VlSpec = Gemma3VlSpec;
register_arch!(GEMMA3_VL_SPEC, Ingest, ToyModel);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_ingest() {
        let reg = ArchRegistry::build();
        let a = reg
            .get(GEMMA3_VL_ARCH_ID)
            .expect("gemma3-vl spec registered");
        assert_eq!(a.family, "gemma3-vl");
        assert!(a.caps.ingest.is_some());
        assert!(a.caps.toy_model.is_some());
    }

    #[test]
    fn toy_fixture_is_complete_and_tiny() {
        let f = Gemma3VlSpec.fixture(0);
        assert!(f.config_json.contains("\"model_type\": \"gemma3\""));
        assert!(
            f.config_json.contains("\"vision_config\""),
            "multimodal fixture must auto-detect as arch 13"
        );
        let names: Vec<_> = f.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"language_model.model.embed_tokens.weight"));
        assert!(names.contains(&"vision_tower.vision_model.embeddings.patch_embedding.weight"));
        assert!(names.contains(&"multi_modal_projector.mm_input_projection_weight"));
        let params: usize = f
            .tensors
            .iter()
            .map(|t| t.shape.iter().product::<usize>())
            .sum();
        assert!(params < 10_000_000, "fixture has {params} params");
    }
}
