// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the DeepSeek-V4 family (arch_id 9): identity + the
//! `Ingest` quant-policy. Deps only `hipfire-arch-api`.
//!
//! Uses the shared transformer prior plus a family override: the MLA compressor and
//! indexer projections are numerically critical (they generate the compressed-KV
//! and index streams), so they sit at max importance — the arch-neutral replacement
//! for the old `is_deepseek4_keep_f16` name-match. No format is named here; the
//! deployment maps max importance to its highest-precision codec.

use hipfire_arch_api::{
    default_importance, default_precision_class, default_requires, register_arch, transformer_role,
    Arch, ArchId, CapReq, Ingest, Init, PrecisionClass, TensorRole, TensorSpec, ToyFixture,
    ToyModel,
};

/// DeepSeek-V4 family header id.
pub const DEEPSEEK4_ARCH_ID: ArchId = ArchId(9);

/// Lean identity marker for the DeepSeek-V4 offline spec.
pub struct Deepseek4Spec;

impl Deepseek4Spec {
    /// MLA compressor / indexer projections — precision-critical stream generators.
    ///
    /// The antirez DS4 reference keeps these at source precision because compression measurably
    /// regresses PPL on DeepSeek-V4: (1) attn compressor `wkv`+`wgate`, (2) indexer
    /// `wq_b`+`weights_proj`, (3) indexer compressor `wkv`+`wgate` (matched by the same
    /// `.compressor.wkv.weight` suffix). All small (≤32 MiB combined across 43 layers).
    /// The router `.ffn.gate.weight` is deliberately NOT here — antirez ships it as a
    /// 4-bit codec and the known-good quant matches; it takes the role default.
    fn is_critical_stream(name: &str) -> bool {
        name.ends_with(".compressor.wkv.weight")
            || name.ends_with(".compressor.wgate.weight")
            || name.ends_with(".indexer.wq_b.weight")
            || name.ends_with(".indexer.weights_proj.weight")
    }
}

impl Arch for Deepseek4Spec {
    fn id(&self) -> ArchId {
        DEEPSEEK4_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "deepseek4"
    }

    fn model_types(&self) -> &'static [&'static str] {
        &["deepseek_v4", "deepseek4", "deepseek_v4_flash"]
    }
}

impl Ingest for Deepseek4Spec {
    fn role(&self, tensor: &str) -> TensorRole {
        transformer_role(tensor)
    }
    fn importance(&self, tensor: &str) -> u8 {
        if Self::is_critical_stream(tensor) {
            255
        } else {
            default_importance(self.role(tensor))
        }
    }
    fn requires(&self, tensor: &str) -> CapReq {
        default_requires(self.role(tensor))
    }
    fn precision_class(&self, tensor: &str) -> PrecisionClass {
        // The MLA compressor/indexer streams are kept at source fidelity (the old
        // `is_deepseek4_keep_f16`); everything else takes the role default. This is
        // the model-definition the quantizer's deepseek source-precision path reads
        // instead of a name-match — no format named here.
        if Self::is_critical_stream(tensor) {
            PrecisionClass::SourcePrecision
        } else {
            default_precision_class(self.role(tensor))
        }
    }
}

/// Tiny DeepSeek4 fixture: two score-routed, uncompressed layers. This exercises
/// the production Q/O-LoRA, Hyper-Connections, shared expert, router, and routed
/// MQ2-Lloyd expert paths without requiring compressed-KV/indexer tensors or MTP.
struct Deepseek4Tiny {
    hidden: usize,
    vocab: usize,
    layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_lora_rank: usize,
    o_lora_rank: usize,
    o_groups: usize,
    experts: usize,
    experts_per_tok: usize,
    moe_inter: usize,
    hc_mult: usize,
    index_n_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    compress_ratios: Vec<u32>,
    mtp_layers: usize,
}

impl Deepseek4Tiny {
    fn preset() -> Self {
        Self {
            hidden: 256,
            vocab: 4096,
            layers: 2,
            n_heads: 8,
            n_kv_heads: 1,
            head_dim: 128,
            q_lora_rank: 256,
            o_lora_rank: 256,
            o_groups: 1,
            experts: 8,
            experts_per_tok: 2,
            moe_inter: 256,
            hc_mult: 4,
            index_n_heads: 1,
            index_head_dim: 128,
            index_topk: 16,
            compress_ratios: vec![0, 0],
            mtp_layers: 0,
        }
    }

    fn compressed_kv() -> Self {
        Self {
            compress_ratios: vec![0, 4],
            ..Self::preset()
        }
    }

    fn mtp() -> Self {
        Self {
            layers: 1,
            compress_ratios: vec![0],
            mtp_layers: 1,
            ..Self::preset()
        }
    }

    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "vocab_size": self.vocab,
            "hidden_size": self.hidden,
            "num_hidden_layers": self.layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "q_lora_rank": self.q_lora_rank,
            "o_lora_rank": self.o_lora_rank,
            "qk_rope_head_dim": 32,
            "o_groups": self.o_groups,
            "n_routed_experts": self.experts,
            "n_shared_experts": 1,
            "num_experts_per_tok": self.experts_per_tok,
            "moe_intermediate_size": self.moe_inter,
            "routed_scaling_factor": 2.2,
            "topk_method": "noaux_tc",
            "scoring_func": "sqrtsoftplus",
            "norm_topk_prob": true,
            "swiglu_limit": 10.0,
            "hc_mult": self.hc_mult,
            "hc_sinkhorn_iters": 2,
            "hc_eps": 1e-6,
            "index_n_heads": self.index_n_heads,
            "index_head_dim": self.index_head_dim,
            "index_topk": self.index_topk,
            "compress_ratios": self.compress_ratios.clone(),
            "compress_rope_theta": 160000.0,
            "rope_theta": 10000.0,
            "rope_scaling": {
                "type": "yarn",
                "factor": 1.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32,
                "beta_slow": 1,
            },
            "sliding_window": 16,
            "num_nextn_predict_layers": self.mtp_layers,
            "num_hash_layers": 0,
            "dtype": "bfloat16",
            "_comment": "hipfire tiny random-init gating fixture - not a real model",
        })
    }

    fn hc_tensors(prefix: &str, t: &mut Vec<TensorSpec>, h: usize, hc: usize) {
        t.push(TensorSpec::f16(
            format!("{prefix}_base"),
            vec![24],
            Init::Zeros,
        ));
        t.push(TensorSpec::f16(
            format!("{prefix}_fn"),
            vec![24, hc * h],
            Init::Uniform(0.01),
        ));
        t.push(TensorSpec::f16(
            format!("{prefix}_scale"),
            vec![3],
            Init::Uniform(0.01),
        ));
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let per_group_in = (self.n_heads / self.o_groups) * self.head_dim;
        let mut t = Vec::new();

        t.push(TensorSpec::new(
            "embed.weight",
            vec![self.vocab, h],
            Init::Uniform(0.03),
        ));
        t.push(TensorSpec::new(
            "head.weight",
            vec![self.vocab, h],
            Init::Uniform(0.03),
        ));
        t.push(TensorSpec::f16("norm.weight", vec![h], Init::NormOnes));
        t.push(TensorSpec::f16(
            "hc_head_base",
            vec![self.hc_mult],
            Init::Zeros,
        ));
        t.push(TensorSpec::f16(
            "hc_head_fn",
            vec![self.hc_mult, self.hc_mult * h],
            Init::Uniform(0.01),
        ));
        t.push(TensorSpec::f16("hc_head_scale", vec![1], Init::Zeros));

        for l in 0..self.layers {
            let p = format!("layers.{l}");
            t.push(TensorSpec::f16(
                format!("{p}.attn_norm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.ffn_norm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.attn.q_norm.weight"),
                vec![self.q_lora_rank],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.attn.kv_norm.weight"),
                vec![kv_dim],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.attn.attn_sink"),
                vec![self.n_heads],
                Init::Zeros,
            ));
            t.push(TensorSpec::new(
                format!("{p}.attn.wq_a.weight"),
                vec![self.q_lora_rank, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.attn.wq_b.weight"),
                vec![q_dim, self.q_lora_rank],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.attn.wkv.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.attn.wo_a.weight"),
                vec![self.o_groups * self.o_lora_rank, per_group_in],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.attn.wo_b.weight"),
                vec![h, self.o_groups * self.o_lora_rank],
                Init::Uniform(0.03),
            ));
            let ratio = self.compress_ratios.get(l).copied().unwrap_or(0);
            if ratio > 0 {
                let coff = if ratio == 4 { 2 } else { 1 };
                let main_dim = coff * self.head_dim;
                t.push(TensorSpec::new(
                    format!("{p}.attn.compressor.wkv.weight"),
                    vec![main_dim, h],
                    Init::Uniform(0.03),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.attn.compressor.wgate.weight"),
                    vec![main_dim, h],
                    Init::Uniform(0.03),
                ));
                t.push(TensorSpec::f16(
                    format!("{p}.attn.compressor.norm.weight"),
                    vec![self.head_dim],
                    Init::NormOnes,
                ));
                t.push(TensorSpec::f16(
                    format!("{p}.attn.compressor.ape"),
                    vec![ratio as usize, main_dim],
                    Init::Zeros,
                ));
                if ratio == 4 {
                    let idx_dim = coff * self.index_head_dim;
                    t.push(TensorSpec::new(
                        format!("{p}.attn.indexer.wq_b.weight"),
                        vec![self.index_n_heads * self.index_head_dim, self.q_lora_rank],
                        Init::Uniform(0.03),
                    ));
                    t.push(TensorSpec::new(
                        format!("{p}.attn.indexer.weights_proj.weight"),
                        vec![self.index_n_heads, h],
                        Init::Uniform(0.03),
                    ));
                    t.push(TensorSpec::new(
                        format!("{p}.attn.indexer.compressor.wkv.weight"),
                        vec![idx_dim, h],
                        Init::Uniform(0.03),
                    ));
                    t.push(TensorSpec::new(
                        format!("{p}.attn.indexer.compressor.wgate.weight"),
                        vec![idx_dim, h],
                        Init::Uniform(0.03),
                    ));
                    t.push(TensorSpec::f16(
                        format!("{p}.attn.indexer.compressor.norm.weight"),
                        vec![self.index_head_dim],
                        Init::NormOnes,
                    ));
                    t.push(TensorSpec::f16(
                        format!("{p}.attn.indexer.compressor.ape"),
                        vec![ratio as usize, idx_dim],
                        Init::Zeros,
                    ));
                }
            }

            Self::hc_tensors(&format!("{p}.hc_attn"), &mut t, h, self.hc_mult);
            Self::hc_tensors(&format!("{p}.hc_ffn"), &mut t, h, self.hc_mult);

            t.push(TensorSpec::new(
                format!("{p}.ffn.gate.weight"),
                vec![self.experts, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.ffn.gate.bias"),
                vec![self.experts],
                Init::Uniform(0.01),
            ));
            t.push(TensorSpec::new(
                format!("{p}.ffn.shared_experts.w1.weight"),
                vec![self.moe_inter, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.ffn.shared_experts.w2.weight"),
                vec![h, self.moe_inter],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.ffn.shared_experts.w3.weight"),
                vec![self.moe_inter, h],
                Init::Uniform(0.03),
            ));
            for e in 0..self.experts {
                let ep = format!("{p}.ffn.experts.{e}");
                t.push(TensorSpec::new(
                    format!("{ep}.w1.weight"),
                    vec![self.moe_inter, h],
                    Init::Uniform(0.03),
                ));
                t.push(TensorSpec::new(
                    format!("{ep}.w2.weight"),
                    vec![h, self.moe_inter],
                    Init::Uniform(0.03),
                ));
                t.push(TensorSpec::new(
                    format!("{ep}.w3.weight"),
                    vec![self.moe_inter, h],
                    Init::Uniform(0.03),
                ));
            }
        }

        for l in 0..self.mtp_layers {
            let p = format!("mtp.{l}");
            t.push(TensorSpec::f16(
                format!("{p}.attn_norm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.ffn_norm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.attn.q_norm.weight"),
                vec![self.q_lora_rank],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.attn.kv_norm.weight"),
                vec![kv_dim],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.attn.attn_sink"),
                vec![self.n_heads],
                Init::Zeros,
            ));
            t.push(TensorSpec::new(
                format!("{p}.attn.wq_a.weight"),
                vec![self.q_lora_rank, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.attn.wq_b.weight"),
                vec![q_dim, self.q_lora_rank],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.attn.wkv.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.attn.wo_a.weight"),
                vec![self.o_groups * self.o_lora_rank, per_group_in],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.attn.wo_b.weight"),
                vec![h, self.o_groups * self.o_lora_rank],
                Init::Uniform(0.03),
            ));
            Self::hc_tensors(&format!("{p}.hc_attn"), &mut t, h, self.hc_mult);
            Self::hc_tensors(&format!("{p}.hc_ffn"), &mut t, h, self.hc_mult);
            t.push(TensorSpec::new(
                format!("{p}.ffn.gate.weight"),
                vec![self.experts, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.ffn.gate.bias"),
                vec![self.experts],
                Init::Uniform(0.01),
            ));
            t.push(TensorSpec::new(
                format!("{p}.ffn.shared_experts.w1.weight"),
                vec![self.moe_inter, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.ffn.shared_experts.w2.weight"),
                vec![h, self.moe_inter],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.ffn.shared_experts.w3.weight"),
                vec![self.moe_inter, h],
                Init::Uniform(0.03),
            ));
            for e in 0..self.experts {
                let ep = format!("{p}.ffn.experts.{e}");
                t.push(TensorSpec::new(
                    format!("{ep}.w1.weight"),
                    vec![self.moe_inter, h],
                    Init::Uniform(0.03),
                ));
                t.push(TensorSpec::new(
                    format!("{ep}.w2.weight"),
                    vec![h, self.moe_inter],
                    Init::Uniform(0.03),
                ));
                t.push(TensorSpec::new(
                    format!("{ep}.w3.weight"),
                    vec![self.moe_inter, h],
                    Init::Uniform(0.03),
                ));
            }
            t.push(TensorSpec::f16(
                format!("{p}.enorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.hnorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{p}.e_proj.weight"),
                vec![h, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.h_proj.weight"),
                vec![h, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.norm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.hc_head_base"),
                vec![self.hc_mult],
                Init::Zeros,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.hc_head_fn"),
                vec![self.hc_mult, self.hc_mult * h],
                Init::Uniform(0.01),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.hc_head_scale"),
                vec![1],
                Init::Zeros,
            ));
        }

        t
    }
}

impl ToyModel for Deepseek4Spec {
    fn fixture(&self, _seed: u64) -> ToyFixture {
        let m = Deepseek4Tiny::preset();
        ToyFixture {
            config_json: serde_json::to_string_pretty(&m.config_json())
                .expect("serialize deepseek4 toy config"),
            tensors: m.manifest(),
        }
    }

    fn fixture_names(&self) -> &'static [&'static str] {
        &["text-core", "compressed-kv", "mtp"]
    }

    fn fixture_named(&self, name: &str, _seed: u64) -> Option<ToyFixture> {
        let m = match name {
            "default" | "text-core" => Deepseek4Tiny::preset(),
            "compressed-kv" => Deepseek4Tiny::compressed_kv(),
            "mtp" | "mtp-draft" => Deepseek4Tiny::mtp(),
            _ => return None,
        };
        Some(ToyFixture {
            config_json: serde_json::to_string_pretty(&m.config_json())
                .expect("serialize deepseek4 toy config"),
            tensors: m.manifest(),
        })
    }
}

static DEEPSEEK4_SPEC: Deepseek4Spec = Deepseek4Spec;
register_arch!(DEEPSEEK4_SPEC, Ingest, ToyModel);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_ingest_and_protects_mla_streams() {
        let reg = ArchRegistry::build();
        let a = reg
            .get(DEEPSEEK4_ARCH_ID)
            .expect("deepseek4 spec registered");
        assert_eq!(a.family, "deepseek4");
        assert!(a.caps.toy_model.is_some());
        let ing = a.caps.ingest.expect("Ingest declared");
        // The MLA compressor/indexer are max-importance (was is_deepseek4_keep_f16).
        assert_eq!(
            ing.importance("model.layers.0.self_attn.compressor.wkv.weight"),
            255
        );
        // …and, in the finer model-def, sit at SourcePrecision — distinct from the
        // rest of the importance-255 protected set (attention is only High). This is
        // the split the coarse importance scalar could not express.
        for t in [
            ".compressor.wkv.weight",
            ".compressor.wgate.weight",
            ".indexer.wq_b.weight",
            ".indexer.weights_proj.weight",
        ] {
            assert_eq!(
                ing.precision_class(&format!("model.layers.0.self_attn{t}")),
                PrecisionClass::SourcePrecision
            );
        }
        assert!(
            ing.precision_class("model.layers.0.self_attn.q_proj.weight")
                < PrecisionClass::SourcePrecision
        );
    }

    #[test]
    fn toy_fixture_declared() {
        let f = Deepseek4Spec.fixture(0);
        let has = |suf: &str| f.tensors.iter().any(|s| s.name.ends_with(suf));
        assert!(f.config_json.contains("\"model_type\": \"deepseek_v4\""));
        assert!(has("attn.wq_a.weight"), "Q-LoRA A");
        assert!(has("attn.wq_b.weight"), "Q-LoRA B");
        assert!(has("attn.wo_a.weight"), "O-LoRA A");
        assert!(has("attn.wo_b.weight"), "O-LoRA B");
        assert!(has("hc_attn_fn"), "attention HC");
        assert!(has("ffn.gate.bias"), "score-routed gate bias");
        assert!(has("ffn.shared_experts.w1.weight"), "shared expert");
        assert!(has("ffn.experts.0.w1.weight"), "routed expert");
        assert!(
            !f.tensors.iter().any(|s| s.name.contains("compressor")),
            "uncompressed tiny should not require compressor tensors"
        );
        let n_params: usize = f
            .tensors
            .iter()
            .map(|s| s.shape.iter().product::<usize>())
            .sum();
        assert!(n_params < 10_000_000, "fixture must stay <10M params");
    }

    #[test]
    fn toy_fixture_mtp_declared() {
        let f = Deepseek4Spec.fixture_named("mtp", 0).expect("mtp fixture");
        let has = |suf: &str| f.tensors.iter().any(|s| s.name.ends_with(suf));
        assert!(f.config_json.contains("\"num_nextn_predict_layers\": 1"));
        assert!(has("mtp.0.enorm.weight"), "MTP embed norm");
        assert!(has("mtp.0.hnorm.weight"), "MTP hidden norm");
        assert!(has("mtp.0.e_proj.weight"), "MTP embed projection");
        assert!(has("mtp.0.h_proj.weight"), "MTP hidden projection");
        assert!(has("mtp.0.attn.wq_a.weight"), "MTP attention");
        assert!(has("mtp.0.ffn.experts.0.w1.weight"), "MTP routed expert");
        assert!(has("mtp.0.hc_head_fn"), "MTP HC head");
        let n_params: usize = f
            .tensors
            .iter()
            .map(|s| s.shape.iter().product::<usize>())
            .sum();
        assert!(n_params < 10_000_000, "MTP fixture must stay <10M params");
    }
}
