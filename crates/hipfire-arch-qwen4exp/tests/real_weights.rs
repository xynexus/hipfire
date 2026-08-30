// SPDX-License-Identifier: Apache-2.0
// hipfire — diff the loader's expected tensor manifest against the REAL checkpoint.
//
// `qwen3_8_flash_next.shapes.json` is the name -> shape map of all 1658 tensors in
// `models--Qwen--Qwen3.8-Flash-Next.hfa`, read out of the archive's 131 verbatim
// safetensors headers. If `weights::plan` disagrees with it anywhere — a name, a
// transposition, an off-by-one on the n-gram layer — that is a load failure on a
// 238 GB conversion, and this catches it in milliseconds instead.

use hipfire_arch_qwen4exp::{plan, LayerType, Mismatch, Qwen4ExpConfig};
use std::collections::BTreeMap;

const CONFIG: &[u8] = include_bytes!("qwen3_8_flash_next.config.json");
const SHAPES: &[u8] = include_bytes!("qwen3_8_flash_next.shapes.json");

fn real_shapes() -> BTreeMap<String, Vec<usize>> {
    let v: serde_json::Value = serde_json::from_slice(SHAPES).unwrap();
    v.as_object()
        .unwrap()
        .iter()
        .map(|(k, s)| {
            (
                k.clone(),
                s.as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect(),
            )
        })
        .collect()
}

fn cfg() -> Qwen4ExpConfig {
    Qwen4ExpConfig::from_slice(CONFIG).unwrap()
}

/// THE test. Every tensor the loader expects must exist in the shipped checkpoint
/// at exactly the shape the config implies.
#[test]
fn plan_matches_the_shipped_checkpoint_exactly() {
    let available = real_shapes();
    assert_eq!(available.len(), 1658, "fixture is the full tensor list");
    let p = plan(&cfg());
    if let Err(bad) = p.validate_against(&available) {
        let shown: Vec<String> = bad.iter().take(20).map(|m| m.to_string()).collect();
        panic!(
            "{} mismatches between the plan and the real checkpoint:\n  {}",
            bad.len(),
            shown.join("\n  ")
        );
    }
}

/// The trunk plan must not silently under-claim. Count what it covers against the
/// checkpoint's own totals, so a plan that "passes" by expecting almost nothing
/// fails here instead.
#[test]
fn plan_covers_the_whole_text_trunk() {
    let available = real_shapes();
    let c = cfg();
    let p = plan(&c);

    // Vision is a separate subsystem and out of scope here; MTP is planned but
    // counted separately (see `mtp_head_matches_the_shipped_checkpoint`), so it is
    // filtered from BOTH sides rather than only from the checkpoint.
    let trunk: Vec<&String> = available
        .keys()
        .filter(|k| !k.starts_with("model.visual.") && !k.starts_with("mtp."))
        .collect();
    let planned: std::collections::BTreeSet<&str> =
        p.tensors.iter().map(|e| e.name.as_str()).collect();
    let unplanned: Vec<&&String> = trunk
        .iter()
        .filter(|k| !planned.contains(k.as_str()))
        .collect();
    assert!(
        unplanned.is_empty(),
        "{} trunk tensors are not in the plan, e.g. {:?}",
        unplanned.len(),
        unplanned.iter().take(8).collect::<Vec<_>>()
    );
    // Assert the real totals rather than a formula, so a change on either side
    // has to be looked at. 1294 trunk + 333 vision + 31 mtp = the 1658 tensors the
    // archive holds; the plan covers the trunk exactly, with nothing spare.
    assert_eq!(trunk.len(), 1294, "text trunk of the shipped checkpoint");
    let planned_trunk = p
        .tensors
        .iter()
        .filter(|e| !e.name.starts_with("mtp.") && !e.name.starts_with("model.visual."))
        .count();
    assert_eq!(
        planned_trunk, 1294,
        "the plan covers it exactly — no extras, no gaps"
    );
    assert_eq!(
        p.len(),
        1658,
        "the whole checkpoint: 1294 trunk + 31 mtp + 333 vision"
    );
}

/// The n-gram block must land on layer 1, not layer 2. `ple_layer_ids` is
/// one-based, and getting it wrong plans 128 shard names under the wrong layer —
/// which the checkpoint does not have, so it would surface as 128 missing tensors.
#[test]
fn ngram_shards_are_planned_under_layer_one() {
    let p = plan(&cfg());
    let shards: Vec<&str> = p
        .tensors
        .iter()
        .map(|e| e.name.as_str())
        .filter(|n| n.contains("ngram_embedding.shard_"))
        .collect();
    assert_eq!(shards.len(), 128);
    assert!(
        shards
            .iter()
            .all(|n| n.starts_with("model.language_model.layers.1.ple.")),
        "n-gram shards must be planned under layer 1 (ple_layer_ids [2] is one-based)"
    );
    // And the derived row count tiles the shards at the checkpoint's own size.
    let available = real_shapes();
    let one = available
        .get("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight")
        .expect("shard 0 exists");
    assert_eq!(*one, vec![2_500_012, 160]);
}

/// A deliberately wrong config must be REJECTED by the diff, otherwise the test
/// above proves nothing. This is the fault injection for the whole file: move the
/// n-gram block one layer over and the plan should stop matching.
#[test]
fn diff_detects_a_misplaced_ngram_layer() {
    let mut v: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    v["text_config"]["ple_layer_ids"] = serde_json::json!([3]); // one-based -> layer 2
    let wrong = Qwen4ExpConfig::from_json(&v).unwrap();
    assert_eq!(wrong.ngram.as_ref().unwrap().layer_idx, 2);
    let bad = plan(&wrong)
        .validate_against(&real_shapes())
        .expect_err("a misplaced n-gram layer must not validate");
    // 128 shards + the 6 ple projections/norms move; the 3 derivable index buffers
    // do not count, since absence is legal for those.
    assert!(
        bad.len() >= 128,
        "expected the whole n-gram block to be missing, got {} mismatches",
        bad.len()
    );
    assert!(bad.iter().all(|m| matches!(m, Mismatch::Missing { .. })));
}

/// Fault injection on shapes rather than names: a transposed projection must be
/// caught, not silently accepted.
#[test]
fn diff_detects_a_transposed_projection() {
    let mut available = real_shapes();
    let k = "model.language_model.layers.3.self_attn.q_proj.weight";
    let orig = available[k].clone();
    available.insert(k.into(), vec![orig[1], orig[0]]);
    let bad = plan(&cfg())
        .validate_against(&available)
        .expect_err("a transposed q_proj must not validate");
    assert!(matches!(&bad[0], Mismatch::Shape { name, .. } if name == k));
}

/// Layer kind decides which mixer is planned; a DeltaNet layer must not expect
/// attention tensors and vice versa.
#[test]
fn layer_kind_selects_the_mixer_exclusively() {
    let c = cfg();
    let p = plan(&c);
    let names: Vec<&str> = p.tensors.iter().map(|e| e.name.as_str()).collect();
    for (l, kind) in c.layer_types.iter().enumerate() {
        let lp = format!("model.language_model.layers.{l}.");
        let has_gdn = names
            .iter()
            .any(|n| n.starts_with(&lp) && n.contains("linear_attn."));
        let has_attn = names
            .iter()
            .any(|n| n.starts_with(&lp) && n.contains("self_attn."));
        match kind {
            LayerType::LinearAttention => assert!(has_gdn && !has_attn, "layer {l}"),
            LayerType::SparseAttention => assert!(has_attn && !has_gdn, "layer {l}"),
        }
    }
}

/// This family has none of the three norms a transformer loader reflexively
/// expects. Planning any of them would be an immediate load failure.
#[test]
fn plans_no_pre_norms_and_no_final_norm() {
    let p = plan(&cfg());
    for n in p.tensors.iter().map(|e| e.name.as_str()) {
        assert!(!n.contains("input_layernorm"), "{n}");
        assert!(!n.contains("post_attention_layernorm"), "{n}");
        assert!(!n.ends_with("language_model.norm.weight"), "{n}");
    }
    // The replacement is present instead.
    assert!(p
        .tensors
        .iter()
        .any(|e| e.name.ends_with("hyper_connection_mixer.hc_norm.weight")));
}

/// The embedded MTP head, against the shipped checkpoint's own tensor list.
///
/// ⚠️ The pinned reference implementation does NOT implement MTP — it drops those
/// weights on load (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`). So unlike
/// every other part of this port there is no reference forward to difference
/// against, and the plan is checked against the checkpoint's SHAPES alone. The
/// composition (how the two `fc_*` projections combine into the layer input) is
/// therefore NOT pinned by anything and must not be guessed at in a serving path.
#[test]
fn mtp_head_matches_the_shipped_checkpoint() {
    let cfg = cfg();
    let available = real_shapes();
    let plan = plan(&cfg);

    let mtp: BTreeMap<String, Vec<usize>> = available
        .iter()
        .filter(|(k, _)| k.starts_with("mtp."))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    assert_eq!(mtp.len(), 31, "shipped checkpoint's MTP head");

    let planned: Vec<_> = plan
        .tensors
        .iter()
        .filter(|e| e.name.starts_with("mtp."))
        .collect();
    assert_eq!(
        planned.len(),
        31,
        "plan covers {} MTP tensors, checkpoint has 31",
        planned.len()
    );

    for e in &planned {
        match mtp.get(&e.name) {
            None => panic!("plan expects `{}`, not in the checkpoint", e.name),
            Some(shape) => assert_eq!(*shape, e.shape, "shape mismatch for `{}`", e.name),
        }
    }
    for name in mtp.keys() {
        assert!(
            planned.iter().any(|e| &e.name == name),
            "checkpoint has `{name}`, the plan does not"
        );
    }
}

/// The MTP head reuses the trunk's embedding and `lm_head` rather than carrying
/// its own — `mtp_use_dedicated_embeddings` is false in the shipped model, and the
/// absence of those tensors is what confirms it.
#[test]
fn mtp_head_has_no_embedding_or_head_of_its_own() {
    let available = real_shapes();
    for suffix in ["embed_tokens.weight", "lm_head.weight"] {
        let found: Vec<_> = available
            .keys()
            .filter(|k| k.starts_with("mtp.") && k.ends_with(suffix))
            .collect();
        assert!(
            found.is_empty(),
            "MTP carries its own `{suffix}`: {found:?} — it should share the trunk's"
        );
    }
}

/// The vision tower, against the shipped checkpoint's own tensor list.
///
/// Unlike the MTP head this one HAS a reference forward (see
/// `reference_oracle.rs`), so the plan is not the only thing pinning it — but the
/// plan still has to match the real file exactly, in both directions.
#[test]
fn vision_tower_matches_the_shipped_checkpoint() {
    let cfg = cfg();
    let available = real_shapes();
    let plan = plan(&cfg);

    let vis: BTreeMap<String, Vec<usize>> = available
        .iter()
        .filter(|(k, _)| k.starts_with("model.visual."))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    assert_eq!(vis.len(), 333, "shipped checkpoint's vision tower");

    let planned: Vec<_> = plan
        .tensors
        .iter()
        .filter(|e| e.name.starts_with("model.visual."))
        .collect();
    assert_eq!(
        planned.len(),
        333,
        "plan covers {} vision tensors",
        planned.len()
    );

    for e in &planned {
        match vis.get(&e.name) {
            None => panic!("plan expects `{}`, not in the checkpoint", e.name),
            Some(shape) => assert_eq!(*shape, e.shape, "shape mismatch for `{}`", e.name),
        }
    }
    for name in vis.keys() {
        assert!(
            planned.iter().any(|e| &e.name == name),
            "checkpoint has `{name}`, the plan does not"
        );
    }
}

/// The MTP composition is inferred from shapes, so the inference itself is tested.
///
/// `src/mtp.rs` argues that one composition is the only one consistent with the
/// checkpoint. That argument is only worth anything if its premises are checked
/// against the real file rather than asserted in a comment — if a future revision
/// adds a wide→narrow projection under `mtp.`, the inference collapses and this
/// test is what says so.
#[test]
fn the_mtp_composition_argument_holds_against_the_checkpoint() {
    let cfg = cfg();
    let available = real_shapes();
    let (hidden, wide) = (cfg.hidden, cfg.hc_hidden());

    // 1. The incoming hidden state is WIDE: its norm spans all streams.
    assert_eq!(
        available["mtp.pre_fc_norm_hidden.weight"],
        vec![wide],
        "pre_fc_norm_hidden should span the wide residual"
    );
    // ...while the embedding side is narrow.
    assert_eq!(available["mtp.pre_fc_norm_embedding.weight"], vec![hidden]);

    // 2. The head's layer also expects a wide input.
    assert_eq!(
        available["mtp.layers.0.attn_hyper_connection.hc_norm.weight"],
        vec![wide]
    );

    // 3. Neither `fc_*` can consume a wide vector, and NO general wide->narrow
    //    projection exists under `mtp.` — the only `[*, wide]` matrices are
    //    hyper-connection internals. This is what forces `fc_hidden` to run per
    //    stream.
    assert_eq!(available["mtp.fc_hidden.weight"], vec![hidden, hidden]);
    assert_eq!(available["mtp.fc_embedding.weight"], vec![hidden, hidden]);
    let wide_inputs: Vec<&String> = available
        .iter()
        .filter(|(k, v)| k.starts_with("mtp.") && v.len() == 2 && v[1] == wide)
        .map(|(k, _)| k)
        .collect();
    for k in &wide_inputs {
        assert!(
            k.contains("input_mix_weight_down") || k.contains("block_inject_weight"),
            "`{k}` takes a wide input but is not a hyper-connection internal — \
             the MTP composition argument in src/mtp.rs no longer holds"
        );
    }

    // 4. The mixer is a `use_combine = false` collapse — it has no
    //    `block_inject_weight`, so it is the FINAL collapse, not an input stage.
    assert!(
        !available.contains_key("mtp.hyper_connection_mixer.block_inject_weight.weight"),
        "the MTP mixer injects a block output, so it is not a plain final collapse"
    );

    // 5. No embedding or head of its own — those are the trunk's.
    assert!(
        !available
            .keys()
            .any(|k| k.starts_with("mtp.") && k.contains("embed_tokens")),
        "MTP has its own embedding; it should share the trunk's"
    );
}
