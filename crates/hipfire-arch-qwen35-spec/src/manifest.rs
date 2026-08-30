// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! What a Qwen3.5 text artifact must contain, declared once.
//!
//! Qwen3.5 is the case that motivates [`LayerScope::Class`]: it INTERLEAVES two
//! layer kinds, listed in the config's `layer_types`, so `linear_attn.*` and
//! `self_attn.*` live on disjoint layer sets. A manifest that assumed every
//! per-layer tensor exists on every layer would report a healthy model as
//! missing three quarters of its attention weights.
//!
//! It is also where a name disagreement has already cost something. The same
//! tensors are spelled with a `model.language_model.` prefix in one place and a
//! bare `model.` prefix in another, and `tiny_harness.rs` rewrites between them
//! at runtime because otherwise the calibrated quant path matches zero tensors
//! and quietly degrades. `PREFIX_ALIASES` makes both spellings explicit here
//! instead of leaving them to be discovered.

use hipfire_arch_api::tensor_manifest::{ManifestBounds, TensorManifest, TensorPattern};
use std::collections::BTreeMap;

/// Layer-class names, matching the config's `layer_types` values.
pub const LINEAR_ATTENTION: &str = "linear_attention";
pub const FULL_ATTENTION: &str = "full_attention";

const PATTERNS: &[TensorPattern] = &[
    // ── global ──
    TensorPattern::required("model.embed_tokens.weight"),
    TensorPattern::required("model.norm.weight"),
    // Untied heads carry their own lm_head; tied ones reuse the embedding.
    TensorPattern::optional("lm_head.weight"),
    // ── every layer ──
    TensorPattern::required("model.layers.{layer}.input_layernorm.weight"),
    TensorPattern::required("model.layers.{layer}.post_attention_layernorm.weight"),
    // ── dense MLP layers ──
    TensorPattern::optional("model.layers.{layer}.mlp.gate_proj.weight"),
    TensorPattern::optional("model.layers.{layer}.mlp.up_proj.weight"),
    TensorPattern::optional("model.layers.{layer}.mlp.down_proj.weight"),
    // ── MoE layers (optional: the dense variant has none) ──
    TensorPattern::optional("model.layers.{layer}.mlp.gate.weight"),
    TensorPattern::optional("model.layers.{layer}.mlp.experts.{expert}.gate_up_proj.weight"),
    TensorPattern::optional("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"),
    // ── full-attention layers only ──
    TensorPattern::required_on_class(
        FULL_ATTENTION,
        "model.layers.{layer}.self_attn.q_proj.weight",
    ),
    TensorPattern::required_on_class(
        FULL_ATTENTION,
        "model.layers.{layer}.self_attn.k_proj.weight",
    ),
    TensorPattern::required_on_class(
        FULL_ATTENTION,
        "model.layers.{layer}.self_attn.v_proj.weight",
    ),
    TensorPattern::required_on_class(
        FULL_ATTENTION,
        "model.layers.{layer}.self_attn.o_proj.weight",
    ),
    TensorPattern::required_on_class(
        FULL_ATTENTION,
        "model.layers.{layer}.self_attn.q_norm.weight",
    ),
    TensorPattern::required_on_class(
        FULL_ATTENTION,
        "model.layers.{layer}.self_attn.k_norm.weight",
    ),
    // ── linear-attention (gated DeltaNet) layers only ──
    TensorPattern::required_on_class(LINEAR_ATTENTION, "model.layers.{layer}.linear_attn.A_log"),
    TensorPattern::required_on_class(LINEAR_ATTENTION, "model.layers.{layer}.linear_attn.dt_bias"),
    TensorPattern::required_on_class(
        LINEAR_ATTENTION,
        "model.layers.{layer}.linear_attn.conv1d.weight",
    ),
    TensorPattern::required_on_class(
        LINEAR_ATTENTION,
        "model.layers.{layer}.linear_attn.norm.weight",
    ),
    TensorPattern::required_on_class(
        LINEAR_ATTENTION,
        "model.layers.{layer}.linear_attn.in_proj_qkv.weight",
    ),
    TensorPattern::required_on_class(
        LINEAR_ATTENTION,
        "model.layers.{layer}.linear_attn.in_proj_a.weight",
    ),
    TensorPattern::required_on_class(
        LINEAR_ATTENTION,
        "model.layers.{layer}.linear_attn.in_proj_b.weight",
    ),
    TensorPattern::required_on_class(
        LINEAR_ATTENTION,
        "model.layers.{layer}.linear_attn.in_proj_z.weight",
    ),
    TensorPattern::required_on_class(
        LINEAR_ATTENTION,
        "model.layers.{layer}.linear_attn.out_proj.weight",
    ),
    // ── companions a CALIBRATED artifact adds ──
    // An activation-aware quant writes a per-tensor AWQ scale beside the weight
    // it scales. Which linears get one depends on the quant plan, so these are
    // optional rather than required: a calibrated 0.8B dense artifact carries
    // exactly one shape (24x `mlp.down_proj.awq_scale.weight`), while a MoE
    // build carries the expert ones instead.
    TensorPattern::optional("model.layers.{layer}.mlp.down_proj.awq_scale.weight"),
    TensorPattern::optional("model.layers.{layer}.mlp.gate_proj.awq_scale.weight"),
    TensorPattern::optional("model.layers.{layer}.mlp.up_proj.awq_scale.weight"),
    TensorPattern::optional(
        "model.layers.{layer}.mlp.experts.{expert}.gate_up_proj.awq_scale.weight",
    ),
    TensorPattern::optional("model.layers.{layer}.mlp.experts.{expert}.down_proj.awq_scale.weight"),
];

/// Split `layer_types` into the two named classes.
///
/// An empty or absent list yields empty classes, and a class-scoped pattern then
/// covers no layers — so a config this code has never seen degrades to "checks
/// less", never to "reports a healthy model broken".
pub fn layer_classes(layer_types: &[String]) -> BTreeMap<&'static str, Vec<usize>> {
    let mut m: BTreeMap<&'static str, Vec<usize>> = BTreeMap::new();
    for (i, t) in layer_types.iter().enumerate() {
        let class = match t.as_str() {
            "full_attention" => FULL_ATTENTION,
            "linear_attention" => LINEAR_ATTENTION,
            _ => continue,
        };
        m.entry(class).or_default().push(i);
    }
    m
}

/// Build the manifest for a given geometry and layer-type list.
pub fn qwen35_manifest(layers: usize, experts: usize, layer_types: &[String]) -> TensorManifest {
    TensorManifest {
        arch: "qwen3_5",
        bounds: ManifestBounds { layers, experts },
        patterns: PATTERNS.to_vec(),
        layer_classes: layer_classes(layer_types),
    }
}

/// Prefix spellings the same tensor set appears under.
///
/// Real checkpoints use `model.language_model.`; the tiny fixtures and the `.hfq`
/// weight names use the short `model.`. `HfqFile::resolve_idx` already aliases
/// between them at lookup time, so a manifest check has to normalise too or it
/// reports every tensor missing on whichever spelling it was not written in.
pub const PREFIX_ALIASES: &[(&str, &str)] = &[("model.language_model.", "model.")];

/// Normalise a tensor name to the short-prefix spelling the manifest uses.
pub fn normalize_prefix(name: &str) -> String {
    for (long, short) in PREFIX_ALIASES {
        if let Some(rest) = name.strip_prefix(long) {
            return format!("{short}{rest}");
        }
    }
    name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ToyModel;

    fn fixture_names_and_cfg() -> (Vec<String>, serde_json::Value) {
        let f = crate::Qwen35Spec.fixture(42);
        let names = f
            .tensors
            .iter()
            .map(|t| normalize_prefix(&t.name))
            .collect();
        (names, serde_json::from_str(&f.config_json).unwrap())
    }

    #[test]
    fn the_toy_fixture_satisfies_the_manifest() {
        let (names, cfg) = fixture_names_and_cfg();
        let layers = cfg["num_hidden_layers"].as_u64().unwrap() as usize;
        let experts = cfg["num_experts"].as_u64().unwrap_or(0) as usize;
        let types: Vec<String> = cfg["layer_types"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|v| v.as_str().unwrap_or("").to_string())
                    .collect()
            })
            .unwrap_or_default();
        let report =
            qwen35_manifest(layers, experts, &types).validate(names.iter().map(String::as_str));
        assert!(report.is_ok(), "\n{}", report.render("qwen3_5"));
    }

    /// The interleave is the point: attention tensors must be expected ONLY on
    /// full-attention layers. Scoping them to every layer would report a healthy
    /// model as missing them on the linear-attention majority.
    #[test]
    fn layer_classes_split_the_interleave() {
        let types: Vec<String> = [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let classes = layer_classes(&types);
        assert_eq!(classes[FULL_ATTENTION], vec![3]);
        assert_eq!(classes[LINEAR_ATTENTION], vec![0, 1, 2]);

        let m = qwen35_manifest(4, 0, &types);
        let expected = m.expected();
        assert!(expected.contains(&"model.layers.3.self_attn.q_proj.weight".to_string()));
        assert!(!expected.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(expected.contains(&"model.layers.0.linear_attn.A_log".to_string()));
        assert!(!expected.contains(&"model.layers.3.linear_attn.A_log".to_string()));
    }

    /// Validate against a REAL artifact when pointed at one:
    /// `HIPFIRE_Q35_NAMES` (one tensor name per line) plus `HIPFIRE_Q35_LAYERS`,
    /// `HIPFIRE_Q35_EXPERTS`, `HIPFIRE_Q35_TYPES` (JSON array). Skipped when
    /// unset so the suite stays hermetic.
    #[test]
    fn real_artifact_names_validate_when_provided() {
        let Ok(path) = std::env::var("HIPFIRE_Q35_NAMES") else {
            return;
        };
        let text = std::fs::read_to_string(&path).expect("read name list");
        let names: Vec<String> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(normalize_prefix)
            .collect();
        let layers: usize = std::env::var("HIPFIRE_Q35_LAYERS")
            .unwrap()
            .parse()
            .unwrap();
        let experts: usize = std::env::var("HIPFIRE_Q35_EXPERTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let types: Vec<String> = std::env::var("HIPFIRE_Q35_TYPES")
            .ok()
            .and_then(|v| serde_json::from_str(&v).ok())
            .unwrap_or_default();
        let report =
            qwen35_manifest(layers, experts, &types).validate(names.iter().map(String::as_str));
        assert!(report.is_ok(), "\n{}", report.render("qwen3_5"));
    }

    #[test]
    fn the_long_prefix_normalises_to_the_short_one() {
        assert_eq!(
            normalize_prefix("model.language_model.layers.0.self_attn.q_proj.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(normalize_prefix("model.norm.weight"), "model.norm.weight");
    }

    /// Unknown layer-type strings must degrade to "checks less", never to
    /// "reports a healthy model broken".
    #[test]
    fn an_unknown_layer_type_covers_no_layers() {
        let types = vec!["something_new".to_string(); 4];
        let m = qwen35_manifest(4, 0, &types);
        assert!(m.layer_classes.is_empty());
        assert!(!m.expected().iter().any(|n| n.contains("self_attn")));
    }
}
