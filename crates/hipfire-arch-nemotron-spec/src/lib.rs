// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Mamba-2 hybrid families: Nemotron-H (arch_id 14) and
//! pure Mamba-2 (arch_id 15). Identity + the `Ingest` quant-policy (shared
//! transformer prior, which classifies the SSM `in_proj`/`out_proj`/`conv1d` mixer
//! tensors). Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_precision_class, default_requires, register_arch, transformer_role,
    Arch, ArchId, CapReq, Ingest, Init, PrecisionClass, TensorRole, TensorSpec, ToyFixture,
    ToyModel,
};

/// Mamba-2 block tensors that corrupt SSM state when lossy: the mixer ingress
/// (`in_proj`, generating the gate/x/B/C/dt streams) and the residual writers
/// (`out_proj`/`down_proj`/`o_proj`). Pinned above a tight low-bit budget — this is the
/// model-definition the quantizer reads instead of the old
/// `is_nemotron_h_mq4_q8_protected` name-match (shared by Nemotron-H + pure Mamba-2).
fn is_state_critical(tensor: &str) -> bool {
    tensor.starts_with("backbone.layers.")
        && (tensor.ends_with(".mixer.in_proj.weight")
            || tensor.ends_with(".mixer.out_proj.weight")
            || tensor.ends_with(".mixer.down_proj.weight")
            || tensor.ends_with(".mixer.o_proj.weight"))
}

/// Shared `precision_class`: pin the state-critical mixer tensors, else role default.
fn state_aware_precision_class(role: TensorRole, tensor: &str) -> PrecisionClass {
    if is_state_critical(tensor) {
        PrecisionClass::Pinned
    } else {
        default_precision_class(role)
    }
}

/// Nemotron-H header id.
pub const NEMOTRON_H_ARCH_ID: ArchId = ArchId(14);
/// Pure Mamba-2 header id.
pub const MAMBA2_ARCH_ID: ArchId = ArchId(15);

/// Lean identity marker for the Nemotron-H offline spec.
pub struct NemotronHSpec;
impl Arch for NemotronHSpec {
    fn id(&self) -> ArchId {
        NEMOTRON_H_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "nemotron-h"
    }
    /// The stack is dense-FFN or routed-MoE, decided by whether
    /// `hybrid_override_pattern` contains an `E` block. The loader already
    /// branches on it (`has_moe`, `hipfire-arch-nemotron/src/lib.rs`), so the
    /// two need different loading and earn distinct identities.
    fn variants(&self) -> &'static [&'static str] {
        &["dense", "moe"]
    }
}
impl Ingest for NemotronHSpec {
    fn role(&self, tensor: &str) -> TensorRole {
        transformer_role(tensor)
    }
    fn importance(&self, tensor: &str) -> u8 {
        default_importance(self.role(tensor))
    }
    fn requires(&self, tensor: &str) -> CapReq {
        default_requires(self.role(tensor))
    }
    fn precision_class(&self, tensor: &str) -> PrecisionClass {
        state_aware_precision_class(self.role(tensor), tensor)
    }
}

/// Tiny hybrid Nemotron-H (arch 14) fixture. This is intentionally separate from
/// the pure Mamba-2 fixture: it includes Mamba, dense MLP, and attention blocks
/// so OQ rows exercise the hybrid block dispatcher.
struct NemotronHTiny {
    hidden: usize,
    vocab: usize,
    layers: usize,
    mamba_heads: usize,
    mamba_head_dim: usize,
    d_state: usize,
    ngroups: usize,
    conv_kernel: usize,
    chunk_size: usize,
    attn_heads: usize,
    kv_heads: usize,
    attn_head_dim: usize,
    mlp_intermediate: usize,
}

impl NemotronHTiny {
    fn preset() -> Self {
        Self {
            hidden: 256,
            vocab: 4096,
            layers: 4,
            mamba_heads: 8,
            mamba_head_dim: 64,
            d_state: 128,
            ngroups: 1,
            conv_kernel: 4,
            chunk_size: 64,
            attn_heads: 2,
            kv_heads: 1,
            attn_head_dim: 128,
            mlp_intermediate: 512,
        }
    }

    fn d_inner(&self) -> usize {
        self.mamba_heads * self.mamba_head_dim
    }

    fn conv_dim(&self) -> usize {
        self.d_inner() + 2 * self.ngroups * self.d_state
    }

    fn projection_size(&self) -> usize {
        self.d_inner() + self.conv_dim() + self.mamba_heads
    }

    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["NemotronHForCausalLM"],
            "model_type": "nemotron_h",
            "hidden_size": self.hidden,
            "vocab_size": self.vocab,
            "num_hidden_layers": self.layers,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": false,
            "hybrid_override_pattern": "M-*-",
            "mamba_num_heads": self.mamba_heads,
            "mamba_head_dim": self.mamba_head_dim,
            "ssm_state_size": self.d_state,
            "n_groups": self.ngroups,
            "conv_kernel": self.conv_kernel,
            "chunk_size": self.chunk_size,
            "use_conv_bias": true,
            "mamba_proj_bias": false,
            "num_attention_heads": self.attn_heads,
            "num_key_value_heads": self.kv_heads,
            "head_dim": self.attn_head_dim,
            "attention_bias": false,
            "intermediate_size": self.mlp_intermediate,
            "mlp_hidden_act": "relu2",
            "time_step_min": 0.001,
            "time_step_max": 0.1,
            "_comment": "hipfire tiny random-init gating fixture — not a real model",
        })
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let d_inner = self.d_inner();
        let conv_dim = self.conv_dim();
        let projection_size = self.projection_size();
        let q_dim = self.attn_heads * self.attn_head_dim;
        let kv_dim = self.kv_heads * self.attn_head_dim;
        let mut t = Vec::new();
        t.push(TensorSpec::new(
            "backbone.embeddings.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        t.push(TensorSpec::f16(
            "backbone.norm_f.weight",
            vec![h],
            Init::NormOnes,
        ));
        t.push(TensorSpec::new(
            "lm_head.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        for i in 0..self.layers {
            let p = format!("backbone.layers.{i}");
            let m = format!("{p}.mixer");
            t.push(TensorSpec::f16(
                format!("{p}.norm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            match i {
                0 => {
                    t.push(TensorSpec::new(
                        format!("{m}.in_proj.weight"),
                        vec![projection_size, h],
                        Init::Uniform(0.04),
                    ));
                    t.push(TensorSpec::f16(
                        format!("{m}.conv1d.weight"),
                        vec![conv_dim, 1, self.conv_kernel],
                        Init::Uniform(0.03),
                    ));
                    t.push(TensorSpec::f16(
                        format!("{m}.conv1d.bias"),
                        vec![conv_dim],
                        Init::Zeros,
                    ));
                    t.push(TensorSpec::f16(
                        format!("{m}.A_log"),
                        vec![self.mamba_heads],
                        Init::ALog,
                    ));
                    t.push(TensorSpec::f16(
                        format!("{m}.D"),
                        vec![self.mamba_heads],
                        Init::NormOnes,
                    ));
                    t.push(TensorSpec::f16(
                        format!("{m}.dt_bias"),
                        vec![self.mamba_heads],
                        Init::Zeros,
                    ));
                    t.push(TensorSpec::f16(
                        format!("{m}.norm.weight"),
                        vec![d_inner],
                        Init::NormOnes,
                    ));
                    t.push(TensorSpec::new(
                        format!("{m}.out_proj.weight"),
                        vec![h, d_inner],
                        Init::Uniform(0.04),
                    ));
                }
                2 => {
                    t.push(TensorSpec::new(
                        format!("{m}.q_proj.weight"),
                        vec![q_dim, h],
                        Init::Uniform(0.04),
                    ));
                    t.push(TensorSpec::new(
                        format!("{m}.k_proj.weight"),
                        vec![kv_dim, h],
                        Init::Uniform(0.04),
                    ));
                    t.push(TensorSpec::new(
                        format!("{m}.v_proj.weight"),
                        vec![kv_dim, h],
                        Init::Uniform(0.04),
                    ));
                    t.push(TensorSpec::new(
                        format!("{m}.o_proj.weight"),
                        vec![h, q_dim],
                        Init::Uniform(0.04),
                    ));
                }
                _ => {
                    t.push(TensorSpec::new(
                        format!("{m}.up_proj.weight"),
                        vec![self.mlp_intermediate, h],
                        Init::Uniform(0.04),
                    ));
                    t.push(TensorSpec::new(
                        format!("{m}.down_proj.weight"),
                        vec![h, self.mlp_intermediate],
                        Init::Uniform(0.04),
                    ));
                }
            }
        }
        t
    }
}

impl ToyModel for NemotronHSpec {
    fn fixture(&self, _seed: u64) -> ToyFixture {
        let n = NemotronHTiny::preset();
        ToyFixture {
            config_json: serde_json::to_string_pretty(&n.config_json())
                .expect("serialize nemotron-h toy config"),
            tensors: n.manifest(),
        }
    }
}

/// Lean identity marker for the pure Mamba-2 offline spec.
pub struct Mamba2Spec;
impl Arch for Mamba2Spec {
    fn id(&self) -> ArchId {
        MAMBA2_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "mamba2"
    }
}
impl Ingest for Mamba2Spec {
    fn role(&self, tensor: &str) -> TensorRole {
        transformer_role(tensor)
    }
    fn importance(&self, tensor: &str) -> u8 {
        default_importance(self.role(tensor))
    }
    fn requires(&self, tensor: &str) -> CapReq {
        default_requires(self.role(tensor))
    }
    fn precision_class(&self, tensor: &str) -> PrecisionClass {
        state_aware_precision_class(self.role(tensor), tensor)
    }
}

/// Tiny pure Mamba-2 (arch 15) config. Mirrors state-spaces tensor names:
/// `backbone.embedding.weight`, `backbone.layers.L.mixer.*`, `backbone.norm_f`.
/// Ported verbatim from the quantizer's old `fixture.rs` so the emitted bytes stay
/// identical (the tiny-quant golden baselines depend on them).
struct Mamba2Tiny {
    hidden: usize,
    vocab: usize,
    layers: usize,
    expand: usize,
    head_dim: usize,
    d_state: usize,
    ngroups: usize,
    conv_kernel: usize,
    chunk_size: usize,
}

impl Mamba2Tiny {
    fn preset() -> Self {
        Self {
            hidden: 256,
            vocab: 4096,
            layers: 2,
            expand: 2,
            head_dim: 64,
            d_state: 128,
            ngroups: 1,
            conv_kernel: 4,
            chunk_size: 64,
        }
    }

    fn d_inner(&self) -> usize {
        self.hidden * self.expand
    }

    fn num_heads(&self) -> usize {
        self.d_inner() / self.head_dim
    }

    fn conv_dim(&self) -> usize {
        self.d_inner() + 2 * self.ngroups * self.d_state
    }

    fn projection_size(&self) -> usize {
        self.d_inner() + self.conv_dim() + self.num_heads()
    }

    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Mamba2ForCausalLM"],
            "d_model": self.hidden,
            "d_intermediate": 0,
            "n_layer": self.layers,
            "vocab_size": self.vocab,
            "ssm_cfg": {
                "layer": "Mamba2",
                "d_state": self.d_state,
                "d_conv": self.conv_kernel,
                "expand": self.expand,
                "headdim": self.head_dim,
                "ngroups": self.ngroups,
                "chunk_size": self.chunk_size,
            },
            "attn_layer_idx": [],
            "attn_cfg": {},
            "rms_norm": true,
            "residual_in_fp32": true,
            "fused_add_norm": true,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": true,
            "rms_norm_eps": 1e-5,
            "_comment": "hipfire tiny random-init gating fixture — not a real model",
        })
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let d_inner = self.d_inner();
        let conv_dim = self.conv_dim();
        let projection_size = self.projection_size();
        let heads = self.num_heads();
        let mut t = Vec::new();
        t.push(TensorSpec::new(
            "backbone.embedding.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        t.push(TensorSpec::f16(
            "backbone.norm_f.weight",
            vec![h],
            Init::NormOnes,
        ));
        t.push(TensorSpec::new(
            "lm_head.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        for i in 0..self.layers {
            let p = format!("backbone.layers.{i}");
            let m = format!("{p}.mixer");
            t.push(TensorSpec::f16(
                format!("{p}.norm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{m}.in_proj.weight"),
                vec![projection_size, h],
                Init::Uniform(0.04),
            ));
            t.push(TensorSpec::f16(
                format!("{m}.conv1d.weight"),
                vec![conv_dim, 1, self.conv_kernel],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::f16(
                format!("{m}.conv1d.bias"),
                vec![conv_dim],
                Init::Zeros,
            ));
            t.push(TensorSpec::f16(
                format!("{m}.A_log"),
                vec![heads],
                Init::ALog,
            ));
            t.push(TensorSpec::f16(
                format!("{m}.D"),
                vec![heads],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{m}.dt_bias"),
                vec![heads],
                Init::Zeros,
            ));
            t.push(TensorSpec::f16(
                format!("{m}.norm.weight"),
                vec![d_inner],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{m}.out_proj.weight"),
                vec![h, d_inner],
                Init::Uniform(0.04),
            ));
        }
        t
    }
}

impl ToyModel for Mamba2Spec {
    // Tiny random-init gating fixture, declared arch-side. Ported verbatim from the
    // quantizer's old fixture so the emitted bytes stay identical (the tiny-quant
    // golden baselines depend on them). Only the pure Mamba-2 arch emits a fixture;
    // the Nemotron-H spec does not.
    fn fixture(&self, _seed: u64) -> ToyFixture {
        let m = Mamba2Tiny::preset();
        ToyFixture {
            config_json: serde_json::to_string_pretty(&m.config_json())
                .expect("serialize mamba2 toy config"),
            tensors: m.manifest(),
        }
    }
}

static NEMOTRON_H_SPEC: NemotronHSpec = NemotronHSpec;
static MAMBA2_SPEC: Mamba2Spec = Mamba2Spec;
register_arch!(NEMOTRON_H_SPEC, Ingest, ToyModel);
register_arch!(MAMBA2_SPEC, Ingest, ToyModel);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_nemotron_and_mamba2() {
        let reg = ArchRegistry::build();
        assert_eq!(reg.get(NEMOTRON_H_ARCH_ID).unwrap().family, "nemotron-h");
        assert!(reg.get(NEMOTRON_H_ARCH_ID).unwrap().caps.ingest.is_some());
        assert_eq!(reg.get(MAMBA2_ARCH_ID).unwrap().family, "mamba2");
        assert!(reg.get(MAMBA2_ARCH_ID).unwrap().caps.ingest.is_some());
    }

    #[test]
    fn state_critical_mixer_tensors_are_pinned_faithfully() {
        // Exactly reproduces the old quantizer `is_nemotron_h_mq4_q8_protected`
        // truth table, now declared arch-side and shared by both Mamba-2 archs.
        let reg = ArchRegistry::build();
        for id in [NEMOTRON_H_ARCH_ID, MAMBA2_ARCH_ID] {
            let ing = reg.get(id).unwrap().caps.ingest.unwrap();
            for t in [
                "backbone.layers.0.mixer.in_proj.weight",
                "backbone.layers.0.mixer.out_proj.weight",
                "backbone.layers.1.mixer.down_proj.weight",
                "backbone.layers.12.mixer.o_proj.weight",
            ] {
                assert_eq!(ing.precision_class(t), PrecisionClass::Pinned, "{t}");
            }
            // up_proj is NOT protected (was `!is_nemotron_h_mq4_q8_protected`).
            assert!(
                ing.precision_class("backbone.layers.0.mixer.up_proj.weight")
                    < PrecisionClass::Pinned
            );
            // The router (structurally protected) is High, not Pinned — so the
            // low-bit pinned path won't over-reach it.
            assert!(
                ing.precision_class("backbone.layers.1.mixer.gate.weight") < PrecisionClass::Pinned
            );
        }
    }

    #[test]
    fn mamba2_toy_fixture_is_declared() {
        let f = Mamba2Spec.fixture(0);
        assert!(!f.tensors.is_empty(), "mamba2 fixture emits tensors");
        assert!(
            f.config_json.contains("Mamba2ForCausalLM"),
            "config declares the Mamba-2 model type: {}",
            f.config_json
        );

        let reg = ArchRegistry::build();
        assert!(reg.get(MAMBA2_ARCH_ID).unwrap().caps.toy_model.is_some());
    }

    #[test]
    fn nemotron_h_toy_fixture_is_hybrid_and_aligned() {
        let f = NemotronHSpec.fixture(0);
        assert!(
            f.config_json
                .contains("\"hybrid_override_pattern\": \"M-*-\""),
            "config declares hybrid Nemotron-H blocks: {}",
            f.config_json
        );
        let has = |name: &str| f.tensors.iter().any(|t| t.name == name);
        assert!(has("backbone.layers.0.mixer.in_proj.weight"));
        assert!(has("backbone.layers.1.mixer.up_proj.weight"));
        assert!(has("backbone.layers.2.mixer.q_proj.weight"));
        assert!(has("backbone.layers.3.mixer.down_proj.weight"));
        assert!(f.tensors.iter().any(|t| {
            t.name.ends_with(".weight")
                && t.shape.last() == Some(&256)
                && t.name.contains(".mixer.")
        }));
        let reg = ArchRegistry::build();
        assert!(reg
            .get(NEMOTRON_H_ARCH_ID)
            .unwrap()
            .caps
            .toy_model
            .is_some());
    }
}
