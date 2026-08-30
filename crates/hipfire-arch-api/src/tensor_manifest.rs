// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! What an architecture REQUIRES an artifact to contain, declared once.
//!
//! Today every loader spells its tensor names inline at the point of use --
//! `format!("{p}.self_attn.q_proj.weight")` handed straight to a lookup that
//! `ok_or_else`es. Three things follow from that, all of them observed in this
//! tree:
//!
//! 1. The same names are re-spelled per consumer. Zaya's ~25 leaf names appear
//!    in its GPU loader, its CPU loader, and its toy fixture; qwen35's appear in
//!    four places.
//! 2. A disagreement between those spellings is silent. `tiny_harness.rs`
//!    rewrites a `model.language_model.` prefix to `model.` at runtime, with a
//!    comment recording that otherwise the calibrated path "matches 0 tensors"
//!    and quietly degrades to the uncalibrated one -- a quality regression
//!    caused by nothing but a name.
//! 3. A wholly wrong artifact reports as one missing tensor. `ZAYA1-8B--bf16.hfq`
//!    in the local store carries the UPSTREAM Zyphra names
//!    (`self_attn.qkv.linear_q.weight`, `zaya_block.experts.local_experts.N.*`)
//!    and shares just 2 of ~35 name shapes with what the loader asks for. The
//!    loader would stop at the first lookup and say "missing tensor", naming one
//!    string, rather than "this artifact uses a different naming convention".
//!
//! A manifest is the independent expectation that makes those detectable. It is
//! deliberately NOT read from the artifact: an artifact that declares its own
//! layout can only ever validate against itself. The arch says what it needs,
//! the artifact's index says what it has, and the check is the comparison.

use std::collections::BTreeMap;

/// Index placeholders a template may carry.
pub const LAYER: &str = "{layer}";
pub const EXPERT: &str = "{expert}";
/// On-disk shard index. A table too large for one tensor is split at rest and
/// concatenated at load; qwen4_exp's n-gram embedding is split into
/// `split_ngram_parts` shards, so the count comes from config like the others.
pub const SHARD: &str = "{shard}";

/// One tensor-name template plus whether the artifact must carry it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorPattern {
    /// Dotted name, optionally containing [`LAYER`] and/or [`EXPERT`], e.g.
    /// `"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"`.
    pub template: &'static str,
    /// `false` for tensors an artifact may legitimately omit (an untied head, an
    /// optional bias). Absence of an optional pattern is not an error; presence
    /// still counts as claimed, so it does not show up as unclaimed either.
    pub required: bool,
    /// Which layers this pattern applies to. Not every per-layer tensor exists
    /// on every layer, and a manifest that assumes it does reports a real model
    /// as broken -- twice over in this tree:
    ///
    /// - zaya's `mlp.gate.router_states_scale` starts at layer 1 (the EDA router
    ///   scales state carried from the PREVIOUS block; block 0 has none). The
    ///   40-layer ZAYA1-8B artifact carries 39, from layer 1. [`LayerScope::From`]
    /// - qwen3.5 interleaves layer TYPES, listed in config `layer_types`, so
    ///   `linear_attn.*` and `self_attn.*` live on disjoint layer sets.
    ///   [`LayerScope::Class`]
    pub layers: LayerScope,
}

/// Which layers a pattern covers. Resolved against the manifest's bounds and,
/// for [`LayerScope::Class`], its `layer_classes` map -- so the const pattern
/// table stays const while the actual indices come from config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerScope {
    /// Every layer in `0..bounds.layers`.
    All,
    /// `from..bounds.layers`.
    From(usize),
    /// The layer indices the arch registered under this class name.
    Class(&'static str),
}

impl TensorPattern {
    pub const fn required(template: &'static str) -> Self {
        Self {
            template,
            required: true,
            layers: LayerScope::All,
        }
    }
    pub const fn optional(template: &'static str) -> Self {
        Self {
            template,
            required: false,
            layers: LayerScope::All,
        }
    }
    /// Required, but only from `layer_from` onward.
    pub const fn required_from_layer(layer_from: usize, template: &'static str) -> Self {
        Self {
            template,
            required: true,
            layers: LayerScope::From(layer_from),
        }
    }
    /// Required, but only on the layers the arch registered under `class`.
    pub const fn required_on_class(class: &'static str, template: &'static str) -> Self {
        Self {
            template,
            required: true,
            layers: LayerScope::Class(class),
        }
    }
    /// Optional, on the layers the arch registered under `class`.
    ///
    /// The combination is what an OPTIONAL COMPONENT needs: qwen4_exp's MTP head
    /// and vision tower may be absent entirely (so not required), and when
    /// present they range over their own index set rather than the text layers
    /// (so class-scoped). Without this, such a component is either falsely
    /// demanded or its tensors report as unclaimed.
    pub const fn optional_on_class(class: &'static str, template: &'static str) -> Self {
        Self {
            template,
            required: false,
            layers: LayerScope::Class(class),
        }
    }
}

/// The extents the placeholders range over, taken from the model config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManifestBounds {
    pub layers: usize,
    pub experts: usize,
    /// How many on-disk shards a [`SHARD`] template ranges over. `1` for an arch
    /// that never shards a table, which is also what [`ManifestBounds::new`]
    /// gives, so a pattern without `{shard}` is unaffected either way.
    pub shards: usize,
}

impl ManifestBounds {
    /// Bounds for an arch that shards nothing.
    pub const fn new(layers: usize, experts: usize) -> Self {
        Self {
            layers,
            experts,
            shards: 1,
        }
    }
}

/// An architecture's full tensor expectation.
#[derive(Debug, Clone)]
pub struct TensorManifest {
    pub arch: &'static str,
    pub bounds: ManifestBounds,
    pub patterns: Vec<TensorPattern>,
    /// Named layer sets a [`LayerScope::Class`] pattern resolves against, filled
    /// from the model config (e.g. qwen3.5's `layer_types`). A class with no
    /// entry covers no layers, so a pattern scoped to an absent class simply
    /// expands to nothing rather than falsely reporting every layer missing.
    pub layer_classes: BTreeMap<&'static str, Vec<usize>>,
}

/// Which pattern a name matched, and with what indices.
fn expand(
    pattern: &TensorPattern,
    bounds: ManifestBounds,
    classes: &BTreeMap<&'static str, Vec<usize>>,
) -> Vec<String> {
    let template = pattern.template;
    let layer_indices: Vec<usize> = if template.contains(LAYER) {
        match pattern.layers {
            LayerScope::All => (0..bounds.layers).collect(),
            LayerScope::From(f) => (f..bounds.layers).collect(),
            LayerScope::Class(c) => classes.get(c).cloned().unwrap_or_default(),
        }
    } else {
        vec![0]
    };
    let experts = if template.contains(EXPERT) {
        bounds.experts
    } else {
        1
    };
    let shards = if template.contains(SHARD) {
        bounds.shards
    } else {
        1
    };
    let mut out = Vec::with_capacity(layer_indices.len() * experts * shards);
    for l in layer_indices {
        let with_layer = template.replace(LAYER, &l.to_string());
        for e in 0..experts {
            let with_expert = with_layer.replace(EXPERT, &e.to_string());
            for sh in 0..shards {
                out.push(with_expert.replace(SHARD, &sh.to_string()));
            }
        }
    }
    out
}

/// A two-way comparison of an arch's manifest against an artifact's index,
/// COLLAPSED BY PATTERN.
///
/// Collapsing is the point. A 40-layer, 32-expert zaya artifact in the wrong
/// naming convention is ~2400 individually missing tensors; printed one per line
/// that is noise a reader skips. Printed as "40x missing
/// `model.layers.{layer}.self_attn.qkv_proj.q_proj.weight`" beside "1280x
/// unclaimed `...zaya_block.experts.local_experts.{expert}...`", the diagnosis
/// reads itself.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ManifestReport {
    /// Required patterns with at least one absent expansion: (template, count).
    pub missing: Vec<(String, usize)>,
    /// Artifact tensors no pattern claimed, collapsed back to a shape by
    /// replacing digit runs in layer/expert position: (shape, count).
    pub unclaimed: Vec<(String, usize)>,
}

impl ManifestReport {
    pub fn is_ok(&self) -> bool {
        self.missing.is_empty() && self.unclaimed.is_empty()
    }

    /// Operator-facing summary. Empty string when the report is clean.
    pub fn render(&self, arch: &str) -> String {
        if self.is_ok() {
            return String::new();
        }
        let mut s = format!("tensor manifest mismatch for arch `{arch}`:\n");
        for (t, n) in &self.missing {
            s.push_str(&format!("  missing   {n:>5}x  {t}\n"));
        }
        for (t, n) in &self.unclaimed {
            s.push_str(&format!("  unclaimed {n:>5}x  {t}\n"));
        }
        if !self.missing.is_empty() && !self.unclaimed.is_empty() {
            s.push_str(
                "\n  Both halves non-empty usually means the artifact is in a DIFFERENT\n  \
                 NAMING CONVENTION, not that tensors are absent -- compare the shapes.\n",
            );
        }
        s
    }
}

/// Collapse a concrete tensor name back to a shape by replacing every
/// `.<digits>.` run with the placeholder, so unclaimed names group.
fn shape_of(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for (i, seg) in name.split('.').enumerate() {
        if i > 0 {
            out.push('.');
        }
        if !seg.is_empty() && seg.bytes().all(|b| b.is_ascii_digit()) {
            out.push_str("{n}");
        } else {
            out.push_str(seg);
        }
    }
    out
}

impl TensorManifest {
    /// Every name this manifest expects, expanded.
    pub fn expected(&self) -> Vec<String> {
        self.patterns
            .iter()
            .flat_map(|p| expand(p, self.bounds, &self.layer_classes))
            .collect()
    }

    /// Compare against the names an artifact actually carries.
    pub fn validate<'a, I: IntoIterator<Item = &'a str>>(&self, present: I) -> ManifestReport {
        let present: std::collections::HashSet<&str> = present.into_iter().collect();
        let mut claimed: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut missing: Vec<(String, usize)> = Vec::new();

        for p in &self.patterns {
            let mut absent = 0usize;
            for name in expand(p, self.bounds, &self.layer_classes) {
                if present.contains(name.as_str()) {
                    claimed.insert(name);
                } else {
                    absent += 1;
                }
            }
            if p.required && absent > 0 {
                missing.push((p.template.to_string(), absent));
            }
        }

        let mut unclaimed_counts: BTreeMap<String, usize> = BTreeMap::new();
        for name in &present {
            if !claimed.contains(*name) {
                *unclaimed_counts.entry(shape_of(name)).or_default() += 1;
            }
        }
        ManifestReport {
            missing,
            unclaimed: unclaimed_counts.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> TensorManifest {
        TensorManifest {
            arch: "test",
            bounds: ManifestBounds::new(2, 2),
            patterns: vec![
                TensorPattern::required("model.embed_tokens.weight"),
                TensorPattern::required("model.layers.{layer}.self_attn.q_proj.weight"),
                TensorPattern::required(
                    "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                ),
                TensorPattern::optional("lm_head.weight"),
            ],
            layer_classes: Default::default(),
        }
    }

    #[test]
    fn expansion_covers_the_index_cross_product() {
        let names = manifest().expected();
        assert!(names.contains(&"model.embed_tokens.weight".to_string()));
        assert!(names.contains(&"model.layers.1.self_attn.q_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.1.mlp.experts.1.down_proj.weight".to_string()));
        // 1 global + 2 layers + 2*2 experts + 1 optional head
        assert_eq!(names.len(), 1 + 2 + 4 + 1);
    }

    #[test]
    fn a_complete_artifact_validates_clean() {
        let m = manifest();
        let mut names = m.expected();
        names.retain(|n| n != "lm_head.weight"); // optional, legitimately absent
        let report = m.validate(names.iter().map(String::as_str));
        assert!(report.is_ok(), "{}", report.render("test"));
    }

    #[test]
    fn missing_required_and_extra_tensors_are_both_reported() {
        let m = manifest();
        let mut names = m.expected();
        names.retain(|n| !n.contains("q_proj")); // drop a required pattern
        names.push("model.layers.0.something_new.weight".to_string());
        let report = m.validate(names.iter().map(String::as_str));
        assert_eq!(
            report.missing,
            vec![(
                "model.layers.{layer}.self_attn.q_proj.weight".to_string(),
                2
            )]
        );
        assert_eq!(
            report.unclaimed,
            vec![("model.layers.{n}.something_new.weight".to_string(), 1)]
        );
    }

    /// The case this exists for: an artifact in a different naming convention
    /// reports as a convention mismatch, not as one missing tensor.
    #[test]
    fn a_wrong_convention_artifact_reports_both_halves() {
        let m = manifest();
        let foreign: Vec<String> = vec![
            "model.embed_tokens.weight".to_string(),
            "model.layers.0.self_attn.qkv.linear_q.weight".to_string(),
            "model.layers.1.self_attn.qkv.linear_q.weight".to_string(),
        ];
        let report = m.validate(foreign.iter().map(String::as_str));
        assert!(!report.missing.is_empty() && !report.unclaimed.is_empty());
        let text = report.render("test");
        assert!(text.contains("DIFFERENT"), "{text}");
        // The foreign names collapse to ONE shape line, not one line per layer.
        assert_eq!(
            report.unclaimed,
            vec![(
                "model.layers.{n}.self_attn.qkv.linear_q.weight".to_string(),
                2
            )]
        );
    }
}
