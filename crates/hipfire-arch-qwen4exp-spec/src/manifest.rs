// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! What a Qwen3.8-Flash-Next artifact must contain, declared once.
//!
//! This family is the densest case the manifest has met: **1658 tensor names** in
//! the shipped checkpoint, across four components that a given artifact may carry
//! in any combination — text trunk, the PLE / n-gram block, an MTP head, and a
//! vision tower. Three separate index dimensions run through those names, and
//! only one of them is the text layer index.
//!
//! Three things here are easy to get wrong, and each has already been settled
//! elsewhere in this crate at some cost:
//!
//! 1. **Layer kinds interleave.** `layer_types` alternates
//!    `linear_attention` (Gated DeltaNet) with the sparse-attention layer, so
//!    `linear_attn.*` and `self_attn.*` cover disjoint layer sets. A manifest
//!    that assumed every per-layer tensor exists on every layer would report a
//!    healthy model as missing three quarters of its attention weights.
//! 2. **PLE rides ONE layer, and the index is off by one in the file.**
//!    `ple_layer_ids` is one-based, so the shipped `[2]` means layer **1** and
//!    the tensors are named `layers.1.ple.*`. `config.rs` carries the same
//!    warning: getting this wrong injects n-gram features into the wrong layer
//!    and still runs.
//! 3. **`{shard}` is not `{expert}`.** The n-gram table is split into
//!    `split_ngram_parts` shards on disk. It is tempting to reuse the expert
//!    placeholder — this family's routed experts are STACKED, so `{expert}`
//!    happens to be free — but that would leave the next reader decoding why a
//!    shard is called an expert.
//!
//! MTP and vision are optional as *components*, so their patterns are optional
//! too: a text-only artifact carrying neither is not broken. Their index sets
//! ride [`LayerScope::Class`] rather than new placeholders, because a class is
//! just a named list of indices and nothing requires those indices to be text
//! layers.

use hipfire_arch_api::tensor_manifest::{ManifestBounds, TensorManifest, TensorPattern};
use std::collections::BTreeMap;

/// Layer classes. The first two mirror `layer_types` values; the last three are
/// index sets for components that are not text layers at all.
pub const LINEAR_ATTENTION: &str = "linear_attention";
pub const SPARSE_ATTENTION: &str = "sparse_attention";
/// The single layer the PLE / n-gram block rides (see the off-by-one note above).
pub const PLE_LAYER: &str = "ple_layer";
/// MTP head layer indices.
pub const MTP_LAYER: &str = "mtp_layer";
/// Vision tower block indices.
pub const VISION_BLOCK: &str = "vision_block";

const PATTERNS: &[TensorPattern] = &[
    // ── global text trunk ───────────────────────────────────────────────────
    TensorPattern::required("model.language_model.embed_tokens.weight"),
    // The head is NOT tied on this family — the shipped checkpoint carries its
    // own `[248320, 2560]`. Optional anyway, so a tied derivative still passes.
    TensorPattern::optional("lm_head.weight"),
    // The 4-wide gated residual ("hyper-connections") has one global mixer...
    TensorPattern::required("model.language_model.hyper_connection_mixer.hc_norm.weight"),
    TensorPattern::required(
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
    ),
    TensorPattern::required(
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
    ),
    // ...and TWO per layer, one guarding each sub-block. Both exist on every
    // layer regardless of its kind.
    TensorPattern::required("model.language_model.layers.{layer}.attn_hyper_connection.hc_norm.weight"),
    TensorPattern::required(
        "model.language_model.layers.{layer}.attn_hyper_connection.input_mix_weight_down.weight",
    ),
    TensorPattern::required(
        "model.language_model.layers.{layer}.attn_hyper_connection.input_mix_weight_up.weight",
    ),
    TensorPattern::required(
        "model.language_model.layers.{layer}.attn_hyper_connection.block_inject_weight.weight",
    ),
    TensorPattern::required("model.language_model.layers.{layer}.mlp_hyper_connection.hc_norm.weight"),
    TensorPattern::required(
        "model.language_model.layers.{layer}.mlp_hyper_connection.input_mix_weight_down.weight",
    ),
    TensorPattern::required(
        "model.language_model.layers.{layer}.mlp_hyper_connection.input_mix_weight_up.weight",
    ),
    TensorPattern::required(
        "model.language_model.layers.{layer}.mlp_hyper_connection.block_inject_weight.weight",
    ),
    // ── Gated DeltaNet layers ───────────────────────────────────────────────
    // The split `in_proj_qkv`/`_z`/`_a`/`_b` spelling is hipfire's native form;
    // HF's fused `in_proj_qkvz` is the outlier, and an artifact carrying THAT
    // shows up here as four missing patterns beside one unclaimed shape.
    TensorPattern::required_on_class(
        LINEAR_ATTENTION,
        "model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight",
    ),
    TensorPattern::required_on_class(
        LINEAR_ATTENTION,
        "model.language_model.layers.{layer}.linear_attn.in_proj_z.weight",
    ),
    TensorPattern::required_on_class(
        LINEAR_ATTENTION,
        "model.language_model.layers.{layer}.linear_attn.in_proj_a.weight",
    ),
    TensorPattern::required_on_class(
        LINEAR_ATTENTION,
        "model.language_model.layers.{layer}.linear_attn.in_proj_b.weight",
    ),
    TensorPattern::required_on_class(
        LINEAR_ATTENTION,
        "model.language_model.layers.{layer}.linear_attn.conv1d.weight",
    ),
    TensorPattern::required_on_class(
        LINEAR_ATTENTION,
        "model.language_model.layers.{layer}.linear_attn.A_log",
    ),
    TensorPattern::required_on_class(
        LINEAR_ATTENTION,
        "model.language_model.layers.{layer}.linear_attn.dt_bias",
    ),
    TensorPattern::required_on_class(
        LINEAR_ATTENTION,
        "model.language_model.layers.{layer}.linear_attn.norm.weight",
    ),
    TensorPattern::required_on_class(
        LINEAR_ATTENTION,
        "model.language_model.layers.{layer}.linear_attn.out_proj.weight",
    ),
    // ── sparse-attention (QSA) layers ───────────────────────────────────────
    TensorPattern::required_on_class(
        SPARSE_ATTENTION,
        "model.language_model.layers.{layer}.self_attn.q_proj.weight",
    ),
    TensorPattern::required_on_class(
        SPARSE_ATTENTION,
        "model.language_model.layers.{layer}.self_attn.k_proj.weight",
    ),
    TensorPattern::required_on_class(
        SPARSE_ATTENTION,
        "model.language_model.layers.{layer}.self_attn.v_proj.weight",
    ),
    TensorPattern::required_on_class(
        SPARSE_ATTENTION,
        "model.language_model.layers.{layer}.self_attn.o_proj.weight",
    ),
    TensorPattern::required_on_class(
        SPARSE_ATTENTION,
        "model.language_model.layers.{layer}.self_attn.q_norm.weight",
    ),
    TensorPattern::required_on_class(
        SPARSE_ATTENTION,
        "model.language_model.layers.{layer}.self_attn.k_norm.weight",
    ),
    // The QSA indexer: what selects the blocks attention is allowed to see.
    TensorPattern::required_on_class(
        SPARSE_ATTENTION,
        "model.language_model.layers.{layer}.self_attn.indexer.index_qk_proj.weight",
    ),
    TensorPattern::required_on_class(
        SPARSE_ATTENTION,
        "model.language_model.layers.{layer}.self_attn.indexer.q_layernorm.weight",
    ),
    TensorPattern::required_on_class(
        SPARSE_ATTENTION,
        "model.language_model.layers.{layer}.self_attn.indexer.k_layernorm.weight",
    ),
    // ── MoE, on every layer ─────────────────────────────────────────────────
    // Routed experts are STACKED into one tensor per projection, so there is no
    // `{expert}` here and no per-expert name to expand. That is why `{expert}`
    // is free on this family — and why it must still not be borrowed for shards.
    TensorPattern::required("model.language_model.layers.{layer}.mlp.gate.weight"),
    TensorPattern::required("model.language_model.layers.{layer}.mlp.experts.gate_up_proj"),
    TensorPattern::required("model.language_model.layers.{layer}.mlp.experts.down_proj"),
    TensorPattern::required("model.language_model.layers.{layer}.mlp.shared_expert.gate_proj.weight"),
    TensorPattern::required("model.language_model.layers.{layer}.mlp.shared_expert.up_proj.weight"),
    TensorPattern::required("model.language_model.layers.{layer}.mlp.shared_expert.down_proj.weight"),
    TensorPattern::required("model.language_model.layers.{layer}.mlp.shared_expert_gate.weight"),
    // ── PLE / n-gram, on its ONE layer ──────────────────────────────────────
    TensorPattern::required_on_class(PLE_LAYER, "model.language_model.layers.{layer}.ple.conv1d.weight"),
    TensorPattern::required_on_class(PLE_LAYER, "model.language_model.layers.{layer}.ple.key_proj.weight"),
    TensorPattern::required_on_class(
        PLE_LAYER,
        "model.language_model.layers.{layer}.ple.value_proj.weight",
    ),
    TensorPattern::required_on_class(PLE_LAYER, "model.language_model.layers.{layer}.ple.norm_query.weight"),
    TensorPattern::required_on_class(PLE_LAYER, "model.language_model.layers.{layer}.ple.norm_key.weight"),
    TensorPattern::required_on_class(PLE_LAYER, "model.language_model.layers.{layer}.ple.norm_conv.weight"),
    // The table itself — 102 GB at source width, 41% of the model's parameters,
    // split across `split_ngram_parts` shards on disk and concatenated at load.
    TensorPattern::required_on_class(
        PLE_LAYER,
        "model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding.shard_{shard}.weight",
    ),
    // The n-gram block also stores three DERIVED tables beside the shards: the
    // per-head offsets and vocab sizes, and the hash multipliers. OPTIONAL, and
    // the two-way check is what settled which: the shipped checkpoint carries all
    // three (so a manifest omitting them reports a healthy artifact as carrying
    // junk), while the fixture does not (the loader marks them `Expect::derived`
    // — reproducible from config, so an artifact may legitimately leave them out).
    // Required would fail the fixture; absent would fail the checkpoint.
    TensorPattern::optional_on_class(
        PLE_LAYER,
        "model.language_model.layers.{layer}.ple.ple_embedding.ngram_heads_offsets",
    ),
    TensorPattern::optional_on_class(
        PLE_LAYER,
        "model.language_model.layers.{layer}.ple.ple_embedding.ngram_heads_vocab_sizes",
    ),
    TensorPattern::optional_on_class(
        PLE_LAYER,
        "model.language_model.layers.{layer}.ple.ple_embedding.layer_multipliers",
    ),
    // ── MTP head (optional component) ───────────────────────────────────────
    // The head is a FULL trunk layer, not a thin projection: it carries its own
    // gated-residual mixer, sparse attention with its indexer, and a complete MoE
    // block. Declaring only a couple of its tensors is what the two-way check
    // caught against the shipped checkpoint — 34 shapes reported unclaimed.
    TensorPattern::optional("mtp.fc_embedding.weight"),
    TensorPattern::optional("mtp.fc_hidden.weight"),
    TensorPattern::optional("mtp.pre_fc_norm_embedding.weight"),
    TensorPattern::optional("mtp.pre_fc_norm_hidden.weight"),
    TensorPattern::optional("mtp.hyper_connection_mixer.hc_norm.weight"),
    TensorPattern::optional("mtp.hyper_connection_mixer.input_mix_weight_down.weight"),
    TensorPattern::optional("mtp.hyper_connection_mixer.input_mix_weight_up.weight"),
    TensorPattern::optional_on_class(MTP_LAYER, "mtp.layers.{layer}.attn_hyper_connection.hc_norm.weight"),
    TensorPattern::optional_on_class(
        MTP_LAYER,
        "mtp.layers.{layer}.attn_hyper_connection.input_mix_weight_down.weight",
    ),
    TensorPattern::optional_on_class(
        MTP_LAYER,
        "mtp.layers.{layer}.attn_hyper_connection.input_mix_weight_up.weight",
    ),
    TensorPattern::optional_on_class(
        MTP_LAYER,
        "mtp.layers.{layer}.attn_hyper_connection.block_inject_weight.weight",
    ),
    TensorPattern::optional_on_class(MTP_LAYER, "mtp.layers.{layer}.mlp_hyper_connection.hc_norm.weight"),
    TensorPattern::optional_on_class(
        MTP_LAYER,
        "mtp.layers.{layer}.mlp_hyper_connection.input_mix_weight_down.weight",
    ),
    TensorPattern::optional_on_class(
        MTP_LAYER,
        "mtp.layers.{layer}.mlp_hyper_connection.input_mix_weight_up.weight",
    ),
    TensorPattern::optional_on_class(
        MTP_LAYER,
        "mtp.layers.{layer}.mlp_hyper_connection.block_inject_weight.weight",
    ),
    // MTP attention is the SPARSE kind (it carries an indexer), never GDN.
    TensorPattern::optional_on_class(MTP_LAYER, "mtp.layers.{layer}.self_attn.q_proj.weight"),
    TensorPattern::optional_on_class(MTP_LAYER, "mtp.layers.{layer}.self_attn.k_proj.weight"),
    TensorPattern::optional_on_class(MTP_LAYER, "mtp.layers.{layer}.self_attn.v_proj.weight"),
    TensorPattern::optional_on_class(MTP_LAYER, "mtp.layers.{layer}.self_attn.o_proj.weight"),
    TensorPattern::optional_on_class(MTP_LAYER, "mtp.layers.{layer}.self_attn.q_norm.weight"),
    TensorPattern::optional_on_class(MTP_LAYER, "mtp.layers.{layer}.self_attn.k_norm.weight"),
    TensorPattern::optional_on_class(
        MTP_LAYER,
        "mtp.layers.{layer}.self_attn.indexer.index_qk_proj.weight",
    ),
    TensorPattern::optional_on_class(
        MTP_LAYER,
        "mtp.layers.{layer}.self_attn.indexer.q_layernorm.weight",
    ),
    TensorPattern::optional_on_class(
        MTP_LAYER,
        "mtp.layers.{layer}.self_attn.indexer.k_layernorm.weight",
    ),
    TensorPattern::optional_on_class(MTP_LAYER, "mtp.layers.{layer}.mlp.gate.weight"),
    TensorPattern::optional_on_class(MTP_LAYER, "mtp.layers.{layer}.mlp.experts.gate_up_proj"),
    TensorPattern::optional_on_class(MTP_LAYER, "mtp.layers.{layer}.mlp.experts.down_proj"),
    TensorPattern::optional_on_class(MTP_LAYER, "mtp.layers.{layer}.mlp.shared_expert.gate_proj.weight"),
    TensorPattern::optional_on_class(MTP_LAYER, "mtp.layers.{layer}.mlp.shared_expert.up_proj.weight"),
    TensorPattern::optional_on_class(MTP_LAYER, "mtp.layers.{layer}.mlp.shared_expert.down_proj.weight"),
    TensorPattern::optional_on_class(MTP_LAYER, "mtp.layers.{layer}.mlp.shared_expert_gate.weight"),
    // ── vision tower (optional component) ───────────────────────────────────
    // `patch_embed.proj` and `pos_embed` both contain the substring "embed" but
    // are NOT gathered tables; the spec's role tests pin that separately.
    TensorPattern::optional("model.visual.patch_embed.proj.weight"),
    TensorPattern::optional("model.visual.patch_embed.proj.bias"),
    TensorPattern::optional("model.visual.pos_embed.weight"),
    // The tower's blocks are pre-norm with BIASES — LayerNorm, not the trunk's
    // bias-free RMSNorm. Missing these was the other half of what the checkpoint
    // check caught.
    TensorPattern::optional_on_class(VISION_BLOCK, "model.visual.blocks.{layer}.norm1.weight"),
    TensorPattern::optional_on_class(VISION_BLOCK, "model.visual.blocks.{layer}.norm1.bias"),
    TensorPattern::optional_on_class(VISION_BLOCK, "model.visual.blocks.{layer}.norm2.weight"),
    TensorPattern::optional_on_class(VISION_BLOCK, "model.visual.blocks.{layer}.norm2.bias"),
    TensorPattern::optional_on_class(VISION_BLOCK, "model.visual.blocks.{layer}.attn.qkv.weight"),
    TensorPattern::optional_on_class(VISION_BLOCK, "model.visual.blocks.{layer}.attn.qkv.bias"),
    TensorPattern::optional_on_class(VISION_BLOCK, "model.visual.blocks.{layer}.attn.proj.weight"),
    TensorPattern::optional_on_class(VISION_BLOCK, "model.visual.blocks.{layer}.attn.proj.bias"),
    TensorPattern::optional_on_class(VISION_BLOCK, "model.visual.blocks.{layer}.mlp.linear_fc1.weight"),
    TensorPattern::optional_on_class(VISION_BLOCK, "model.visual.blocks.{layer}.mlp.linear_fc1.bias"),
    TensorPattern::optional_on_class(VISION_BLOCK, "model.visual.blocks.{layer}.mlp.linear_fc2.weight"),
    TensorPattern::optional_on_class(VISION_BLOCK, "model.visual.blocks.{layer}.mlp.linear_fc2.bias"),
    TensorPattern::optional("model.visual.merger.norm.weight"),
    TensorPattern::optional("model.visual.merger.norm.bias"),
    TensorPattern::optional("model.visual.merger.linear_fc1.weight"),
    TensorPattern::optional("model.visual.merger.linear_fc1.bias"),
    TensorPattern::optional("model.visual.merger.linear_fc2.weight"),
    TensorPattern::optional("model.visual.merger.linear_fc2.bias"),
    // ── calibrated-artifact companions ──────────────────────────────────────
    // AWQ writes a per-input-channel scale beside the tensor it scales. Which
    // linears get one depends on the quant plan, so these are optional; without
    // them a calibrated artifact reports its scales as unclaimed.
    TensorPattern::optional("model.language_model.layers.{layer}.mlp.experts.down_proj.awq_scale.weight"),
    TensorPattern::optional(
        "model.language_model.layers.{layer}.mlp.experts.gate_up_proj.awq_scale.weight",
    ),
    TensorPattern::optional(
        "model.language_model.layers.{layer}.mlp.shared_expert.down_proj.awq_scale.weight",
    ),
];

/// Geometry the placeholders range over. `ngram_layer` is the ZERO-BASED layer
/// (`ple_layer_ids` in the file is one-based — see the module note).
#[derive(Debug, Clone, Copy)]
pub struct Qwen4ExpGeometry {
    pub layers: usize,
    pub experts: usize,
    pub ngram_layer: Option<usize>,
    pub ngram_shards: usize,
    pub mtp_layers: usize,
    pub vision_blocks: usize,
}

/// Split `layer_types` and the component sizes into named index sets.
///
/// An unrecognised layer kind is skipped rather than guessed, and a component
/// with no layers yields an empty class. Both degrade to "checks less", never to
/// "reports a healthy model broken" — the property that makes it safe to point
/// this at a config shape nobody has seen yet.
pub fn layer_classes(
    layer_types: &[String],
    geom: Qwen4ExpGeometry,
) -> BTreeMap<&'static str, Vec<usize>> {
    let mut m: BTreeMap<&'static str, Vec<usize>> = BTreeMap::new();
    for (i, t) in layer_types.iter().enumerate() {
        let class = match t.as_str() {
            "linear_attention" => LINEAR_ATTENTION,
            // The reference normalises to `qwen_sparse_attention`; older configs
            // and the fixture spell it `full_attention`. Both mean this layer.
            "qwen_sparse_attention" | "full_attention" | "sparse_attention" => SPARSE_ATTENTION,
            _ => continue,
        };
        m.entry(class).or_default().push(i);
    }
    if let Some(l) = geom.ngram_layer {
        m.insert(PLE_LAYER, vec![l]);
    }
    if geom.mtp_layers > 0 {
        m.insert(MTP_LAYER, (0..geom.mtp_layers).collect());
    }
    if geom.vision_blocks > 0 {
        m.insert(VISION_BLOCK, (0..geom.vision_blocks).collect());
    }
    m
}

/// Build the manifest for a given geometry and layer-type list.
pub fn qwen4exp_manifest(layer_types: &[String], geom: Qwen4ExpGeometry) -> TensorManifest {
    TensorManifest {
        arch: "qwen4_exp",
        bounds: ManifestBounds {
            layers: geom.layers,
            experts: geom.experts,
            shards: geom.ngram_shards.max(1),
        },
        patterns: PATTERNS.to_vec(),
        layer_classes: layer_classes(layer_types, geom),
    }
}

/// Prefix spellings the same tensor set appears under.
///
/// The text trunk lives under `model.language_model.` in both the checkpoint and
/// the fixture here, unlike qwen3.5 where the two disagree. Kept explicit so that
/// if a short-prefix derivative ever appears the check normalises rather than
/// reporting every text tensor missing.
pub const PREFIX_ALIASES: &[(&str, &str)] = &[("model.", "model.language_model.")];

/// Normalise a bare `model.layers.` spelling to the long prefix the manifest uses.
/// Leaves `model.visual.` and an already-long name alone.
pub fn normalize_prefix(name: &str) -> String {
    if name.starts_with("model.language_model.") || name.starts_with("model.visual.") {
        return name.to_string();
    }
    if let Some(rest) = name.strip_prefix("model.") {
        return format!("model.language_model.{rest}");
    }
    name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ToyModel;

    /// The fixture's geometry, read from the config it ships rather than
    /// hardcoded — so a fixture change moves this test with it instead of
    /// silently invalidating it.
    fn fixture_names_and_geometry() -> (Vec<String>, Vec<String>, Qwen4ExpGeometry) {
        let f = crate::Qwen4ExpSpec.fixture(42);
        let names: Vec<String> = f.tensors.iter().map(|t| t.name.clone()).collect();
        let root: serde_json::Value = serde_json::from_str(&f.config_json).expect("fixture config");
        // The config is VL-nested: the trunk's geometry lives under `text_config`,
        // with the vision tower beside it. Fall back to the root so a future
        // text-only config shape still resolves.
        let cfg = if root.get("text_config").is_some() {
            root["text_config"].clone()
        } else {
            root.clone()
        };
        let layer_types: Vec<String> = cfg["layer_types"]
            .as_array()
            .expect("layer_types")
            .iter()
            .map(|v| v.as_str().unwrap_or_default().to_string())
            .collect();
        // `ple_layer_ids` is ONE-BASED in the file; the manifest wants zero-based.
        let ngram_layer = cfg["ple_layer_ids"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|v| v.as_u64())
            .map(|v| v as usize - 1);
        let geom = Qwen4ExpGeometry {
            layers: layer_types.len(),
            experts: cfg["num_experts"].as_u64().unwrap_or(0) as usize,
            ngram_layer,
            ngram_shards: cfg["split_ngram_parts"].as_u64().unwrap_or(1) as usize,
            // The fixture is text-only; both components are absent, which the
            // optional patterns must tolerate silently.
            mtp_layers: 0,
            vision_blocks: 0,
        };
        (names, layer_types, geom)
    }

    /// The manifest must accept the fixture with NOTHING missing and NOTHING
    /// unclaimed. Both halves matter: missing catches a manifest that asks for
    /// what the family does not have, unclaimed catches one that has not been
    /// told about a tensor the family really carries.
    #[test]
    fn manifest_matches_the_fixture_exactly() {
        let (names, layer_types, geom) = fixture_names_and_geometry();
        let m = qwen4exp_manifest(&layer_types, geom);
        let report = m.validate(names.iter().map(|s| s.as_str()));
        assert!(
            report.is_ok(),
            "fixture should satisfy the manifest exactly:\n{}",
            report.render("qwen4_exp")
        );
    }

    /// The interleave is the whole reason for class scoping: GDN and sparse
    /// attention must cover DISJOINT layers that together tile the trunk.
    #[test]
    fn layer_classes_split_the_interleave() {
        let (_, layer_types, geom) = fixture_names_and_geometry();
        let c = layer_classes(&layer_types, geom);
        let gdn = c.get(LINEAR_ATTENTION).cloned().unwrap_or_default();
        let qsa = c.get(SPARSE_ATTENTION).cloned().unwrap_or_default();
        assert!(!gdn.is_empty() && !qsa.is_empty(), "both kinds present");
        assert!(
            gdn.iter().all(|l| !qsa.contains(l)),
            "a layer cannot be both kinds: gdn={gdn:?} qsa={qsa:?}"
        );
        assert_eq!(
            gdn.len() + qsa.len(),
            layer_types.len(),
            "classes tile the trunk"
        );
    }

    /// PLE rides ONE layer. A manifest that scoped it to every layer would demand
    /// the 102 GB table on all 48 of them.
    #[test]
    fn ple_is_scoped_to_a_single_layer() {
        let (names, layer_types, geom) = fixture_names_and_geometry();
        let ple: Vec<&String> = names.iter().filter(|n| n.contains(".ple.")).collect();
        assert!(!ple.is_empty(), "fixture carries a PLE block");
        let c = layer_classes(&layer_types, geom);
        assert_eq!(c.get(PLE_LAYER).map(|v| v.len()), Some(1));
        // ...and it is the layer the tensors are actually named for.
        let l = c[PLE_LAYER][0];
        assert!(
            ple.iter().all(|n| n.contains(&format!(".layers.{l}.ple."))),
            "PLE class layer {l} must match the tensor names: {:?}",
            ple.first()
        );
    }

    /// Every shard is demanded, not just shard 0 — the failure that would ship a
    /// model with three quarters of its n-gram table missing.
    #[test]
    fn every_ngram_shard_is_expected() {
        let (_, layer_types, mut geom) = fixture_names_and_geometry();
        geom.ngram_shards = 4;
        let expected = qwen4exp_manifest(&layer_types, geom).expected();
        let l = geom.ngram_layer.expect("fixture has a PLE layer");
        for s in 0..4 {
            let want = format!(
                "model.language_model.layers.{l}.ple.ple_embedding.ngram_embedding.shard_{s}.weight"
            );
            assert!(expected.contains(&want), "missing {want}");
        }
    }

    /// A wrong-convention artifact must report as BOTH halves non-empty — that is
    /// the signal the renderer keys its "different naming convention" hint off.
    /// Here: HF's fused `in_proj_qkvz` in place of the split spelling.
    #[test]
    fn fused_upstream_spelling_reports_as_a_convention_mismatch() {
        let (names, layer_types, geom) = fixture_names_and_geometry();
        let renamed: Vec<String> = names
            .iter()
            .map(|n| n.replace("linear_attn.in_proj_qkv.", "linear_attn.in_proj_qkvz."))
            .collect();
        let report =
            qwen4exp_manifest(&layer_types, geom).validate(renamed.iter().map(|s| s.as_str()));
        assert!(!report.missing.is_empty(), "the split name is missing");
        assert!(!report.unclaimed.is_empty(), "the fused name is unclaimed");
        assert!(report.render("qwen4_exp").contains("DIFFERENT"));
    }

    /// Absent optional components must be silent. A text-only artifact carries no
    /// vision tower and no MTP head, and that is not a defect.
    #[test]
    fn absent_optional_components_are_not_errors() {
        let (names, layer_types, geom) = fixture_names_and_geometry();
        assert!(
            !names.iter().any(|n| n.starts_with("model.visual.")),
            "fixture is text-only"
        );
        let report =
            qwen4exp_manifest(&layer_types, geom).validate(names.iter().map(|s| s.as_str()));
        assert!(report.missing.is_empty(), "{:?}", report.missing);
    }
}
