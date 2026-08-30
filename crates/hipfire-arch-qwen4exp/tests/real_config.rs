// SPDX-License-Identifier: Apache-2.0
// hipfire — parse the SHIPPED Qwen3.8-Flash-Next config, not a synthetic one.
//
// `qwen3_8_flash_next.config.json` is the verbatim `config.json` from
// `models--Qwen--Qwen3.8-Flash-Next.hfa`. Testing against a hand-written config
// only proves the parser agrees with itself; these assertions are pinned to the
// real geometry, so a parser that drifts from the checkpoint fails here.

use hipfire_arch_qwen4exp::{ngram_head_layout, LayerType, Qwen4ExpConfig};

const REAL: &[u8] = include_bytes!("qwen3_8_flash_next.config.json");

fn cfg() -> Qwen4ExpConfig {
    Qwen4ExpConfig::from_slice(REAL).expect("shipped config must parse")
}

#[test]
fn parses_the_shipped_geometry() {
    let c = cfg();
    assert_eq!((c.hidden, c.vocab, c.layers), (2560, 248320, 48));
    assert_eq!((c.n_heads, c.n_kv_heads, c.head_dim), (24, 2, 256));
    assert!(!c.tie_word_embeddings, "this checkpoint has an UNTIED head");
    assert!(c.has_vision, "the shipped artifact carries a vision tower");
    assert_eq!(c.mtp_layers, 1);
}

/// The stack is `12 x (3 x DeltaNet -> 1 x sparse attention)`.
#[test]
fn layer_stack_is_three_to_one() {
    let c = cfg();
    let sparse: Vec<usize> = c.sparse_attention_layers().collect();
    assert_eq!(sparse.len(), 12, "48 layers at interval 4");
    assert_eq!(
        c.layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count(),
        36
    );
    // Every 4th layer, and the first sparse layer is index 3 — not 0.
    assert_eq!(sparse[0], 3);
    assert!(sparse.windows(2).all(|w| w[1] - w[0] == 4));
}

/// Q and K share 16 heads while V has 48 — a 3:1 grouping the loader must split
/// correctly, and the fused projection packs them at different spans.
#[test]
fn deltanet_grouping_and_projection_widths() {
    let d = cfg().deltanet;
    assert_eq!((d.key_heads, d.value_heads), (16, 48));
    assert_eq!(d.value_per_key(), 3);
    // 16*128*2 + 48*128 = 10240, matching `in_proj_qkv [10240, 2560]`.
    assert_eq!(d.qkv_dim(), 10240);
    // 48*128 = 6144, matching `in_proj_z [6144, 2560]`.
    assert_eq!(d.z_dim(), 6144);
    assert_eq!(d.conv_kernel, 4);
    assert!(
        d.output_gate_sigmoid,
        "output_gate_type is sigmoid here, where Qwen3.5 is silu"
    );
}

/// The budget is in TOKENS and the selection is over BLOCKS.
#[test]
fn indexer_budget_is_tokens_and_selection_is_blocks() {
    let i = cfg().indexer;
    assert_eq!((i.n_heads, i.kv_heads, i.head_dim), (4, 1, 128));
    assert_eq!((i.budget, i.compress_ratio), (2048, 4));
    assert_eq!(i.block_topk(), 512, "2048 tokens / 4 per block");
    // (4 + 1) * 128 = 640, matching `index_qk_proj [640, 2560]`.
    assert_eq!(i.qk_proj_out(), 640);
}

/// `norm_topk_prob` is ABSENT from the shipped config, so the default decides
/// whether the routed weights are renormalised. The reference defaults it true.
#[test]
fn moe_and_the_absent_norm_topk_default() {
    let c = cfg();
    let raw: serde_json::Value = serde_json::from_slice(REAL).unwrap();
    assert!(
        raw["text_config"].get("norm_topk_prob").is_none(),
        "precondition: the shipped config omits norm_topk_prob"
    );
    assert!(
        c.moe.norm_topk_prob,
        "must default TRUE, matching configuration_qwen4_exp.py"
    );
    assert_eq!(
        (c.moe.num_experts, c.moe.experts_per_tok, c.moe.intermediate),
        (512, 10, 640)
    );
    assert_eq!(c.moe.shared_intermediate, 640);
    assert_ne!(
        c.moe.intermediate % 256,
        0,
        "the geometry the indexed path refuses"
    );
}

/// The residual is four streams wide; every block reads and writes it.
#[test]
fn gated_residual_widens_the_stream_fourfold() {
    let c = cfg();
    assert_eq!((c.gated_residual.count, c.gated_residual.lowrank), (4, 320));
    assert_eq!(c.hc_hidden(), 10240, "matches hc_norm [10240]");
}

/// `ple_layer_ids` is ONE-BASED — `[2]` means layer index 1, which is why the
/// checkpoint names those tensors `layers.1.ple.*`. Off by one here injects the
/// n-gram features into the wrong layer and still produces output.
#[test]
fn ple_layer_id_is_one_based() {
    let raw: serde_json::Value = serde_json::from_slice(REAL).unwrap();
    assert_eq!(raw["text_config"]["ple_layer_ids"][0], 2, "precondition");
    let n = cfg().ngram.expect("shipped config has an n-gram block");
    assert_eq!(n.layer_idx, 1, "one-based [2] converts to zero-based 1");
}

/// 16 heads over a 2560-wide embedding gives the checkpoint's 160-dim shards, and
/// the derived prime ladder must reproduce the stored per-head vocab sizes.
#[test]
fn ngram_head_layout_matches_the_checkpoint() {
    let n = cfg().ngram.unwrap();
    assert_eq!((n.ngram_size, n.heads_per_ngram), (3, 8));
    assert_eq!(n.heads(), 16, "8 heads per order x 2 orders");
    assert_eq!(n.head_dim(), 160, "matches shard [2500012, 160]");
    assert_eq!(n.context_len(), 2, "two predecessor tokens");
    assert_eq!(n.shards, 128);

    let (sizes, offsets, padded) = ngram_head_layout(n.vocab_size_base, n.heads(), n.divisible_by);
    // Verbatim from the checkpoint's own `ngram_heads_vocab_sizes` buffer.
    assert_eq!(
        sizes,
        vec![
            20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
            20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171
        ]
    );
    // ... and its `ngram_heads_offsets`.
    assert_eq!(offsets[0], 0);
    assert_eq!(offsets[1], 20000003);
    assert_eq!(offsets[15], 300001275);
    // The padded total tiles the 128 shards exactly, at the checkpoint's row count.
    assert_eq!(padded, 320_001_536);
    assert_eq!(padded as usize / n.shards, 2_500_012);
}

/// Partial rotary: only 64 of each 256-wide head carries position, and the mrope
/// sections must partition that half-dimension. A section list that disagrees is
/// the bug that still yields coherent text on text-only prompts.
#[test]
fn partial_rotary_and_mrope_sections_agree() {
    let c = cfg();
    assert_eq!(c.rotary_dim(), 64, "256 * 0.25");
    assert!(c.mrope_interleaved);
    assert_eq!(c.mrope_section, vec![11, 11, 10]);
    assert_eq!(c.mrope_section.iter().sum::<usize>() * 2, c.rotary_dim());
}

/// Q carries the output gate interleaved with the queries, so its projection is
/// twice the head span — the checkpoint's `q_proj [12288, 2560]`.
#[test]
fn q_projection_is_doubled() {
    let c = cfg();
    assert_eq!(c.q_proj_out(), 12288);
    assert_eq!(c.kv_proj_out(), 512, "2 kv heads x 256");
}

/// Malformed configs must be refused rather than silently mis-parsed. Each of
/// these is a real failure mode: an off-by-one in the one-based layer id, a
/// section list that disagrees with the rotary width, and a top-k that cannot be
/// selected.
#[test]
fn invalid_configs_are_refused() {
    let mut v: serde_json::Value = serde_json::from_slice(REAL).unwrap();
    v["text_config"]["ple_layer_ids"] = serde_json::json!([0]);
    assert!(
        Qwen4ExpConfig::from_json(&v).is_err(),
        "a zero ple_layer_id is invalid under one-based indexing"
    );

    let mut v: serde_json::Value = serde_json::from_slice(REAL).unwrap();
    v["text_config"]["rope_parameters"]["mrope_section"] = serde_json::json!([11, 11, 11]);
    assert!(
        Qwen4ExpConfig::from_json(&v).is_err(),
        "mrope_section must sum to half the rotary dim"
    );

    let mut v: serde_json::Value = serde_json::from_slice(REAL).unwrap();
    v["text_config"]["num_experts_per_tok"] = serde_json::json!(999);
    assert!(
        Qwen4ExpConfig::from_json(&v).is_err(),
        "top-k exceeds expert count"
    );

    let mut v: serde_json::Value = serde_json::from_slice(REAL).unwrap();
    v["text_config"]["linear_num_value_heads"] = serde_json::json!(47);
    assert!(
        Qwen4ExpConfig::from_json(&v).is_err(),
        "V heads must be a multiple of K heads"
    );
}

// ─── n-gram addressing, pinned against the checkpoint's own buffers ───────────

use hipfire_arch_qwen4exp::{build_layer_multipliers, NgramHasher};

/// The multipliers are DERIVED from `seed`, and the shipped config omits `seed`
/// — so the default decides them. Verified against the checkpoint's stored
/// `layer_multipliers` buffer, read from the archive.
#[test]
fn derived_multipliers_reproduce_the_checkpoint() {
    let c = cfg();
    let n = c.ngram.as_ref().unwrap();
    assert_eq!(
        n.seed, 1234,
        "default seed; the shipped config omits the key"
    );
    let m = build_layer_multipliers(c.vocab as u64, n.ngram_size, n.ple_index, n.seed);
    assert_eq!(
        m,
        vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071],
        "must reproduce the stored layer_multipliers exactly"
    );
    // Seed 0 does NOT — proving the default is load-bearing, not incidental.
    assert_ne!(
        build_layer_multipliers(c.vocab as u64, n.ngram_size, n.ple_index, 0),
        m
    );
}

/// Multipliers are forced odd and bounded so `token * multiplier` stays inside
/// the signed range the reference's integer path assumes.
#[test]
fn multipliers_are_odd_and_bounded() {
    let c = cfg();
    let n = c.ngram.as_ref().unwrap();
    for m in build_layer_multipliers(c.vocab as u64, n.ngram_size, n.ple_index, n.seed) {
        assert_eq!(m % 2, 1, "multipliers must be odd");
        assert!(
            m.checked_mul(c.vocab as u64).is_some() && m * (c.vocab as u64) <= (1u64 << 63) - 1,
            "token * multiplier must not exceed i64::MAX"
        );
    }
}

fn hasher() -> NgramHasher {
    let c = cfg();
    NgramHasher::from_config(c.ngram.as_ref().unwrap(), c.vocab as u64, 248_044)
}

/// Every head must land inside its own slice of the flat table — an index that
/// escapes into a neighbouring head's rows is the silent failure this guards.
#[test]
fn every_row_lands_in_its_own_head_slice() {
    let h = hasher();
    let preds = [Some(1234u32), Some(5678)];
    for cur in [0u32, 1, 999, 248_319] {
        let rows = h.rows(cur, &preds);
        assert_eq!(rows.len(), 16);
        for (i, r) in rows.iter().enumerate() {
            let lo = h.offsets()[i];
            let hi = lo + h.vocab_sizes()[i];
            assert!(
                *r >= lo && *r < hi,
                "head {i}: row {r} outside [{lo}, {hi})"
            );
        }
    }
}

/// The two orders (bigram, trigram) own disjoint head ranges, and the bigram must
/// ignore the second predecessor while the trigram uses it.
#[test]
fn bigram_ignores_the_older_token_and_trigram_does_not() {
    let h = hasher();
    let a = h.rows(7, &[Some(11), Some(22)]);
    let b = h.rows(7, &[Some(99), Some(22)]); // changed only the OLDER token
    assert_eq!(a[..8], b[..8], "bigram heads 0-7 must not see t-2");
    assert_ne!(a[8..], b[8..], "trigram heads 8-15 must see t-2");
}

/// An n-gram must never span a document boundary: reaching back across EOS fills
/// with EOS. A token's own EOS does not cut its own context.
#[test]
fn eos_cuts_the_window_but_not_the_current_token() {
    let h = hasher();
    let eos = 248_044u32;
    // t-2 is EOS, so the trigram sees EOS there — same as if it were missing.
    let across = h.rows(7, &[Some(eos), Some(22)]);
    let missing = h.rows(7, &[None, Some(22)]);
    assert_eq!(
        across, missing,
        "EOS and absent predecessors are equivalent"
    );
    // The cut LATCHES: once t-1 is EOS, t-2 is EOS regardless of what it was.
    let latch_a = h.rows(7, &[Some(11), Some(eos)]);
    let latch_b = h.rows(7, &[Some(99), Some(eos)]);
    assert_eq!(latch_a, latch_b, "a cut at t-1 must latch through t-2");
    // The current token being EOS does NOT blank its own context.
    assert_ne!(
        h.rows(eos, &[Some(11), Some(22)]),
        h.rows(eos, &[Some(11), Some(99)]),
        "the token's own EOS must not cut the window behind it"
    );
}

/// Flat rows map onto the 128 shards by plain division, because the shards tile a
/// uniform slice of one padded table.
#[test]
fn rows_locate_into_the_right_shard() {
    let h = hasher();
    assert_eq!((h.shards(), h.shard_rows()), (128, 2_500_012));
    assert_eq!(
        h.locate(0),
        hipfire_arch_qwen4exp::RowLocation {
            shard: 0,
            row_in_shard: 0
        }
    );
    let l = h.locate(2_500_012);
    assert_eq!(l.shard, 1);
    assert_eq!(l.row_in_shard, 0);
    // Every row a real lookup can produce stays inside the 128 shards.
    for r in h.rows(1234, &[Some(1), Some(2)]) {
        assert!(h.locate(r).shard < 128, "row {r} escaped the shard range");
    }
}

/// Stored buffers are authoritative and must override the derivation; a
/// wrong-length buffer must be refused rather than silently truncated.
#[test]
fn stored_buffers_override_and_are_length_checked() {
    let c = cfg();
    let n = c.ngram.as_ref().unwrap();
    let base = NgramHasher::from_config(n, c.vocab as u64, 248_044);
    let overridden = base
        .clone()
        .with_stored(Some(vec![3, 5, 7]), None, None)
        .unwrap();
    assert_eq!(overridden.multipliers(), &[3, 5, 7]);
    assert_ne!(
        overridden.rows(7, &[Some(1), Some(2)]),
        base.rows(7, &[Some(1), Some(2)])
    );
    assert!(base
        .clone()
        .with_stored(Some(vec![1, 2]), None, None)
        .is_err());
    assert!(base.with_stored(None, Some(vec![0; 3]), None).is_err());
}

/// The HFQ envelope: the quantizer nests the source config under `config`, so a
/// served artifact must parse to exactly what the source directory parses to.
///
/// A missing envelope is an error rather than a silent fallback — reading the
/// wrapper as if it were the config yields "no text_config", which is a confusing
/// way to report a packaging problem.
#[test]
fn hfq_metadata_envelope_round_trips() {
    let raw = include_str!("qwen3_8_flash_next.config.json");
    let direct = Qwen4ExpConfig::from_json(&serde_json::from_str(raw).unwrap()).unwrap();

    let wrapped = serde_json::json!({
        "architecture": "qwen4_exp",
        "config": serde_json::from_str::<serde_json::Value>(raw).unwrap(),
        "tokenizer": {},
    });
    let via_hfq = Qwen4ExpConfig::from_metadata_json(&wrapped.to_string()).unwrap();
    assert_eq!(via_hfq.layers, direct.layers);
    assert_eq!(via_hfq.hidden, direct.hidden);
    assert_eq!(via_hfq.vocab, direct.vocab);
    assert_eq!(via_hfq.moe.num_experts, direct.moe.num_experts);
    assert_eq!(via_hfq.indexer.budget, direct.indexer.budget);
    assert!(via_hfq.ngram.is_some());

    let err = Qwen4ExpConfig::from_metadata_json(r#"{"architecture":"qwen4_exp"}"#).unwrap_err();
    assert!(err.contains("envelope"), "unhelpful error: {err}");
}
