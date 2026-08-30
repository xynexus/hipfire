// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! What a zaya artifact must contain, declared once.
//!
//! These are the names `ZayaGpuWeights::load` (`gpu.rs`) constructs inline, and
//! that `weights.rs` and `hipfire-arch-zaya-spec`'s toy fixture each re-spell.
//! Declaring them here is what lets a mismatch be reported as a set difference
//! instead of surfacing as whichever `format!` happened to be looked up first.

use hipfire_arch_api::tensor_manifest::{ManifestBounds, TensorManifest, TensorPattern};

/// Per-layer and per-expert tensors, in the hipfire-normalised vocabulary.
///
/// NOTE this is one of TWO conventions in circulation for zaya. The upstream
/// Zyphra checkpoints spell the same tensors
/// `self_attn.qkv.linear_q.weight` / `zaya_block.experts.local_experts.N.linear_fc1.weight`
/// and are translated on import; `calibration_stream.rs` still speaks that
/// dialect. A manifest per convention is how the two stop being confusable.
const PATTERNS: &[TensorPattern] = &[
    // ── global ──
    TensorPattern::required("model.embed_tokens.weight"),
    TensorPattern::required("model.norm.weight"),
    TensorPattern::required("model.input_hidden_states_scale"),
    TensorPattern::required("model.input_hidden_states_bias"),
    // ── per layer: attention ──
    TensorPattern::required("model.layers.{layer}.input_layernorm.weight"),
    TensorPattern::required("model.layers.{layer}.post_attention_layernorm.weight"),
    TensorPattern::required("model.layers.{layer}.self_attn.o_proj.weight"),
    TensorPattern::required("model.layers.{layer}.self_attn.qk_norm.temp"),
    TensorPattern::required("model.layers.{layer}.self_attn.qkv_proj.q_proj.weight"),
    TensorPattern::required("model.layers.{layer}.self_attn.qkv_proj.k_proj.weight"),
    TensorPattern::required("model.layers.{layer}.self_attn.qkv_proj.v_proj_current.weight"),
    TensorPattern::required("model.layers.{layer}.self_attn.qkv_proj.v_proj_delayed.weight"),
    TensorPattern::required("model.layers.{layer}.self_attn.qkv_proj.conv_qk_depthwise.weight"),
    TensorPattern::required("model.layers.{layer}.self_attn.qkv_proj.conv_qk_depthwise.bias"),
    TensorPattern::required("model.layers.{layer}.self_attn.qkv_proj.conv_qk_grouped.weight"),
    TensorPattern::required("model.layers.{layer}.self_attn.qkv_proj.conv_qk_grouped.bias"),
    // ── per layer: residual affines ──
    TensorPattern::required(
        "model.layers.{layer}.post_attention_residual_scale.hidden_states_scale",
    ),
    TensorPattern::required(
        "model.layers.{layer}.post_attention_residual_scale.hidden_states_bias",
    ),
    TensorPattern::required("model.layers.{layer}.post_attention_residual_scale.residual_scale"),
    TensorPattern::required("model.layers.{layer}.post_attention_residual_scale.residual_bias"),
    TensorPattern::required("model.layers.{layer}.post_mlp_residual_scale.hidden_states_scale"),
    TensorPattern::required("model.layers.{layer}.post_mlp_residual_scale.hidden_states_bias"),
    TensorPattern::required("model.layers.{layer}.post_mlp_residual_scale.residual_scale"),
    TensorPattern::required("model.layers.{layer}.post_mlp_residual_scale.residual_bias"),
    // ── per layer: EDA router ──
    TensorPattern::required("model.layers.{layer}.mlp.gate.balancing_biases"),
    // Layer 0 has none: the EDA router scales state carried from the PREVIOUS
    // block. Confirmed on ZAYA1-8B (40 layers, 39 of these, first at layer 1).
    TensorPattern::required_from_layer(1, "model.layers.{layer}.mlp.gate.router_states_scale"),
    TensorPattern::required("model.layers.{layer}.mlp.gate.down_proj.weight"),
    TensorPattern::required("model.layers.{layer}.mlp.gate.down_proj.bias"),
    TensorPattern::required("model.layers.{layer}.mlp.gate.router_mlp.norm.weight"),
    TensorPattern::required("model.layers.{layer}.mlp.gate.router_mlp.fc1.weight"),
    TensorPattern::required("model.layers.{layer}.mlp.gate.router_mlp.fc1.bias"),
    TensorPattern::required("model.layers.{layer}.mlp.gate.router_mlp.fc2.weight"),
    TensorPattern::required("model.layers.{layer}.mlp.gate.router_mlp.fc2.bias"),
    TensorPattern::required("model.layers.{layer}.mlp.gate.router_mlp.out_proj.weight"),
    // ── per (layer, expert) ──
    TensorPattern::required("model.layers.{layer}.mlp.experts.{expert}.gate_up_proj.weight"),
    TensorPattern::required("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"),
    // ── companions a CALIBRATED artifact adds ──
    // Optional, not unexpected: an activation-aware quant writes a per-tensor
    // AWQ scale beside the weight it scales. Declaring them keeps a calibrated
    // artifact from reporting three unclaimed shapes on every load.
    TensorPattern::optional("model.layers.{layer}.mlp.gate.down_proj.awq_scale.weight"),
    TensorPattern::optional(
        "model.layers.{layer}.mlp.experts.{expert}.gate_up_proj.awq_scale.weight",
    ),
    TensorPattern::optional("model.layers.{layer}.mlp.experts.{expert}.down_proj.awq_scale.weight"),
];

/// Build the manifest for a given geometry.
pub fn zaya_manifest(layers: usize, experts: usize) -> TensorManifest {
    TensorManifest {
        arch: "zaya",
        bounds: ManifestBounds::new(layers, experts),
        patterns: PATTERNS.to_vec(),
        // zaya has one layer kind; every pattern is All or From.
        layer_classes: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The toy fixture and the manifest are two independent spellings of the
    /// same tensor set, so checking one against the other is exactly the drift
    /// this is meant to catch -- and it is free, with no artifact on disk.
    #[test]
    fn the_toy_fixture_satisfies_the_manifest() {
        use hipfire_arch_api::ToyModel;
        let toy = crate::ZayaSpec;
        let fixture = toy.fixture(42);
        let names: Vec<&str> = fixture.tensors.iter().map(|t| t.name.as_str()).collect();
        let cfg: serde_json::Value = serde_json::from_str(&fixture.config_json).unwrap();
        let layers = cfg["num_hidden_layers"].as_u64().unwrap() as usize;
        let experts = cfg["num_experts"].as_u64().unwrap() as usize;

        let report = zaya_manifest(layers, experts).validate(names.iter().copied());
        assert!(report.is_ok(), "\n{}", report.render("zaya"));
    }

    /// Validate against a REAL artifact's tensor list when one is pointed at by
    /// `HIPFIRE_ZAYA_NAMES` (one name per line, e.g. from `hipfire hfq list`).
    /// Skipped when unset, so the suite stays hermetic; the point is that the
    /// manifest is checkable against a 2483-tensor production artifact and not
    /// only against a 2-layer toy.
    #[test]
    fn real_artifact_names_validate_when_provided() {
        let Ok(path) = std::env::var("HIPFIRE_ZAYA_NAMES") else {
            return;
        };
        let text = std::fs::read_to_string(&path).expect("read name list");
        let names: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect();
        let layers: usize = std::env::var("HIPFIRE_ZAYA_LAYERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(40);
        let experts: usize = std::env::var("HIPFIRE_ZAYA_EXPERTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32);
        let report = zaya_manifest(layers, experts).validate(names.iter().copied());
        assert!(report.is_ok(), "\n{}", report.render("zaya"));
    }

    /// An artifact in the UPSTREAM Zyphra convention must report as a convention
    /// mismatch -- both halves populated -- not as a single missing tensor.
    /// These are the real shapes in `ZAYA1-8B--bf16.hfq`.
    #[test]
    fn upstream_named_artifact_reports_a_convention_mismatch() {
        let mut names = vec!["model.embed_tokens.weight".to_string()];
        for l in 0..2 {
            names.push(format!("model.layers.{l}.self_attn.qkv.linear_q.weight"));
            names.push(format!("model.layers.{l}.self_attn.qkv.val_proj1.weight"));
            names.push(format!("model.layers.{l}.input_norm.weight"));
            for e in 0..4 {
                names.push(format!(
                    "model.layers.{l}.zaya_block.experts.local_experts.{e}.linear_fc1.weight"
                ));
            }
        }
        let report = zaya_manifest(2, 4).validate(names.iter().map(String::as_str));
        assert!(!report.missing.is_empty(), "should miss the hipfire names");
        assert!(
            !report.unclaimed.is_empty(),
            "should not claim upstream names"
        );
        let text = report.render("zaya");
        assert!(text.contains("DIFFERENT"), "{text}");
        // Collapsed: 8 upstream expert tensors become ONE shape line.
        assert!(report
            .unclaimed
            .iter()
            .any(|(shape, n)| shape.contains("local_experts.{n}.linear_fc1") && *n == 8));
    }
}
