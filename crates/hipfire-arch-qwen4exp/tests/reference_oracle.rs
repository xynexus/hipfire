// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Differences the port against the PINNED upstream `qwen4_exp` implementation.
//!
//! `tests/real_config.rs` and `tests/real_weights.rs` pin the offline half against
//! the shipped 48-layer checkpoint's own config and tensor list. Those catch a
//! wrong SHAPE. They cannot catch a wrong COMPOSITION — a residual routed to the
//! wrong stream, a norm applied on the wrong side — because there is no reference
//! activation to compare with.
//!
//! This one runs against a real forward from transformers @5f8ab9bb, on a tiny
//! config that keeps the real structure (3 GatedDeltaNet layers, then one sparse
//! attention layer, one PLE layer, 4-wide hyper-connections, routed MoE).
//!
//! The artifact is generated, not committed — see `scripts/qwen4exp_oracle.py`.
//! Absent, these tests SKIP with a message rather than fail: the generator needs a
//! pinned transformers that is not part of a normal checkout.

use std::collections::BTreeMap;
use std::path::PathBuf;

use hipfire_arch_qwen4exp::config::Qwen4ExpConfig;
use hipfire_arch_qwen4exp::weights::{plan, Plan};

/// Where the trunk sits inside the shipped checkpoint's name space.
const TEXT_PREFIX: &str = "model.language_model.";

/// Oracle (`Qwen4ExpForCausalLM`) name -> shipped-checkpoint name.
fn rename(name: &str) -> String {
    match name.strip_prefix("model.") {
        Some(rest) => format!("{TEXT_PREFIX}{rest}"),
        None => name.to_string(), // `lm_head.*`
    }
}

fn oracle_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/oracle")
}

/// `None` (with a printed reason) when the artifact has not been generated.
fn load_manifest() -> Option<serde_json::Value> {
    let p = oracle_dir().join("oracle.json");
    if !p.exists() {
        eprintln!(
            "SKIP: {} absent — regenerate with scripts/qwen4exp_oracle.py",
            p.display()
        );
        return None;
    }
    Some(serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap())
}

/// The reference's own `state_dict` must satisfy the port's weight plan.
///
/// This is a stronger statement than the checkpoint test: the checkpoint is one
/// config, while the plan is a FUNCTION of config, and this exercises it at a
/// completely different point (4 layers, 8 experts, hc_lowrank 16).
#[test]
fn plan_matches_the_reference_state_dict() {
    let Some(m) = load_manifest() else { return };
    let cfg = Qwen4ExpConfig::from_json(&m["config"]).expect("oracle config parses");
    let p = plan(&cfg);
    let cfg_shards = cfg.ngram.as_ref().expect("oracle has a PLE layer").shards;

    let mut present: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (name, meta) in m["tensors"].as_object().unwrap() {
        let Some(stripped) = name.strip_prefix("w.").or_else(|| name.strip_prefix("b.")) else {
            continue; // activations, not weights
        };
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        // The oracle dumps `Qwen4ExpForCausalLM`, whose trunk sits at `model.*`.
        // In the shipped checkpoint that same trunk is one level deeper, under the
        // conditional-generation wrapper — `model.language_model.*` — which is what
        // `plan()` targets (pinned against the real file by `real_weights.rs`).
        // `lm_head` is a sibling of the trunk in both and keeps its bare name.
        // `split_ngram_parts` shards the n-gram table in the CHECKPOINT; the
        // reference concatenates on load and holds one tensor. Expand it back into
        // the shard names the plan expects. This also cross-checks the port's
        // prime-layout math: the shard rows only sum to the reference's table if
        // `ngram_head_layout` derived the same padded vocab the reference did.
        if stripped.ends_with("ngram_embedding.weight") {
            let shards = cfg_shards;
            assert_eq!(
                shape[0] % shards,
                0,
                "reference n-gram table {shape:?} does not split into {shards} shards"
            );
            let base = stripped.trim_end_matches(".weight");
            for i in 0..shards {
                present.insert(
                    rename(&format!("{base}.shard_{i}.weight")),
                    vec![shape[0] / shards, shape[1]],
                );
            }
            continue;
        }
        present.insert(rename(stripped), shape);
    }
    assert!(
        present.len() > 40,
        "manifest looks empty ({} weights) — the generator or the prefix convention drifted",
        present.len()
    );

    // The oracle builds the text model and the vision tower SEPARATELY, so its
    // text state dict carries neither `model.visual.*` nor the MTP head. Compare
    // the trunk only; those two have their own checkpoint-pinned tests in
    // `real_weights.rs`.
    let p = Plan {
        tensors: p
            .tensors
            .into_iter()
            .filter(|e| !e.name.starts_with("model.visual.") && !e.name.starts_with("mtp."))
            .collect(),
    };
    if let Err(mism) = p.validate_against(&present) {
        panic!(
            "plan disagrees with the reference state_dict ({} mismatches):\n{}",
            mism.len(),
            mism.iter()
                .take(25)
                .map(|x| format!("  {x:?}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

/// The tiny config must exercise the structure, not degenerate past it. A config
/// where every layer is linear, or where QSA cannot exclude anything, would make
/// every downstream comparison vacuously easy.
#[test]
fn the_oracle_config_actually_exercises_the_hard_paths() {
    let Some(m) = load_manifest() else { return };
    let cfg = Qwen4ExpConfig::from_json(&m["config"]).expect("oracle config parses");
    let n_sparse = cfg.sparse_attention_layers().count();
    assert!(n_sparse > 0, "no sparse-attention layer in the oracle");
    assert!(
        n_sparse < cfg.layers,
        "every layer is sparse-attention — the GatedDeltaNet path is untested"
    );
    assert!(cfg.ngram.is_some(), "no PLE layer in the oracle");
    assert!(
        cfg.moe.experts_per_tok < cfg.moe.num_experts,
        "routing is degenerate: top-{} of {}",
        cfg.moe.experts_per_tok,
        cfg.moe.num_experts
    );
    assert!(
        cfg.gated_residual.count > 1,
        "hyper-connections collapsed to a single stream"
    );
    // QSA must actually exclude at the oracle's sequence length, or the sparse
    // path degenerates to dense and proves nothing.
    let n_tok = 16;
    let dense_below = cfg.indexer.budget + cfg.indexer.compress_ratio - 1;
    assert!(
        n_tok > dense_below,
        "at {n_tok} tokens QSA cannot exclude (dense below {dense_below}) — \
         lower indexer_budget in the generator"
    );
}

// ── activation differencing ─────────────────────────────────────────────────

/// The oracle blob plus its manifest, for reading reference tensors by name.
struct Oracle {
    m: serde_json::Value,
    blob: Vec<u8>,
}

impl Oracle {
    fn open() -> Option<Self> {
        let m = load_manifest()?;
        let blob = std::fs::read(oracle_dir().join("oracle.bin")).unwrap();
        Some(Self { m, blob })
    }

    fn cfg(&self) -> Qwen4ExpConfig {
        Qwen4ExpConfig::from_json(&self.m["config"]).expect("oracle config parses")
    }

    fn shape(&self, name: &str) -> Vec<usize> {
        self.m["tensors"][name]["shape"]
            .as_array()
            .unwrap_or_else(|| panic!("oracle has no tensor `{name}`"))
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    }

    fn get(&self, name: &str) -> Vec<f32> {
        let t = &self.m["tensors"][name];
        assert!(!t.is_null(), "oracle has no tensor `{name}`");
        let off = t["offset"].as_u64().unwrap() as usize;
        let n = t["numel"].as_u64().unwrap() as usize;
        self.blob[off..off + n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// Row `t` of a `[1, T, D]` activation.
    fn row(&self, name: &str, t: usize) -> Vec<f32> {
        let s = self.shape(name);
        let d = *s.last().unwrap();
        self.get(name)[t * d..(t + 1) * d].to_vec()
    }

    /// `input_ids` is stored as exact integers, not in the float blob.
    fn tokens(&self) -> Vec<usize> {
        self.ints("input_ids")
            .into_iter()
            .map(|v| v as usize)
            .collect()
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn check(label: &str, got: &[f32], want: &[f32], tol: f32) {
    let d = max_abs_diff(got, want);
    let mag = want.iter().map(|v| v.abs()).fold(0.0, f32::max);
    assert!(
        d <= tol,
        "{label}: max|Δ| = {d:.3e} > {tol:.1e} (reference magnitude {mag:.3})"
    );
    println!("  {label:34} max|Δ| = {d:.3e}  (mag {mag:.3})");
}

/// `hidden_states = inputs_embeds.repeat(1, 1, hc_count)`.
///
/// The wide stream is seeded by REPEATING the embedding, not by zero-padding the
/// other streams. Zeros would look plausible and train-time-equivalent; they are
/// not what the reference does, and every later layer would inherit the error.
#[test]
fn embedding_seeds_every_hyper_connection_stream() {
    let Some(o) = Oracle::open() else { return };
    let cfg = o.cfg();
    let embed = o.get("w.model.embed_tokens.weight");
    let hidden = cfg.hidden;
    let hc = cfg.gated_residual.count;

    for (t, &tok) in o.tokens().iter().enumerate() {
        let e = &embed[tok * hidden..(tok + 1) * hidden];
        let want = o.row("a.layer0.in", t);
        let got: Vec<f32> = (0..hc).flat_map(|_| e.iter().copied()).collect();
        assert_eq!(got.len(), want.len());
        check(&format!("embed+repeat t={t}"), &got, &want, 1e-6);
    }
}

/// The model-level mixer collapses `hc_count * hidden` -> `hidden`, and is the
/// ONLY final norm: there is no separate `model.norm` before the head.
#[test]
fn mixer_collapses_the_streams() {
    let Some(o) = Oracle::open() else { return };
    let cfg = o.cfg();
    let (hn, md, mu) = (
        o.get("w.model.hyper_connection_mixer.hc_norm.weight"),
        o.get("w.model.hyper_connection_mixer.input_mix_weight_down.weight"),
        o.get("w.model.hyper_connection_mixer.input_mix_weight_up.weight"),
    );
    let gr = hipfire_arch_qwen4exp::hc::GatedResidual {
        hc_norm: &hn,
        mix_down: &md,
        mix_up: &mu,
        block_inject: None, // use_combine=False
        hc_count: cfg.gated_residual.count,
        hidden: cfg.hidden,
        lowrank: cfg.gated_residual.lowrank,
        eps: cfg.rms_norm_eps,
    };
    for t in [0usize, 7, 15] {
        let got = gr.read(&o.row("a.mixer.in", t)).mixed_input;
        check(&format!("mixer t={t}"), &got, &o.row("a.mixer", t), 2e-6);
    }
}

/// `logits = lm_head(mixer_out)`, with the head untied from the embedding.
#[test]
fn lm_head_produces_the_logits() {
    let Some(o) = Oracle::open() else { return };
    let cfg = o.cfg();
    let w = o.get("w.lm_head.weight");
    let vocab = o.shape("w.lm_head.weight")[0];
    assert_eq!(vocab, cfg.vocab);
    for t in [0usize, 15] {
        let h = o.row("a.mixer", t);
        let got: Vec<f32> = (0..vocab)
            .map(|v| (0..cfg.hidden).map(|d| w[v * cfg.hidden + d] * h[d]).sum())
            .collect();
        check(
            &format!("lm_head t={t}"),
            &got,
            &o.row("out.logits", t),
            2e-5,
        );
    }
}

/// The whole routed-MoE block: softmax over ALL experts, top-k, renorm, expert
/// SwiGLU, plus the always-on sigmoid-gated shared expert.
///
/// Run on every layer, because the routing depends on the activations and a bug
/// that happens to select the same experts on one layer will not on another.
#[test]
fn moe_block_matches_the_reference() {
    use hipfire_arch_qwen4exp::moe::{Expert, MoeLayer};
    let Some(o) = Oracle::open() else { return };
    let cfg = o.cfg();
    let (hidden, mi) = (cfg.hidden, cfg.moe.intermediate);
    let smi = cfg.moe.shared_intermediate;
    let n_exp = cfg.moe.num_experts;

    for l in 0..cfg.layers {
        let p = format!("w.model.layers.{l}.mlp");
        let (router, gu, dn) = (
            o.get(&format!("{p}.gate.weight")),
            o.get(&format!("{p}.experts.gate_up_proj")),
            o.get(&format!("{p}.experts.down_proj")),
        );
        let (sg, su, sd, seg) = (
            o.get(&format!("{p}.shared_expert.gate_proj.weight")),
            o.get(&format!("{p}.shared_expert.up_proj.weight")),
            o.get(&format!("{p}.shared_expert.down_proj.weight")),
            o.get(&format!("{p}.shared_expert_gate.weight")),
        );
        let (gu_sz, dn_sz) = (2 * mi * hidden, hidden * mi);
        let experts: Vec<Expert> = (0..n_exp)
            .map(|e| Expert {
                gate_up: &gu[e * gu_sz..(e + 1) * gu_sz],
                down: &dn[e * dn_sz..(e + 1) * dn_sz],
            })
            .collect();
        let moe = MoeLayer {
            router: &router,
            experts,
            shared_gate: &sg,
            shared_up: &su,
            shared_down: &sd,
            shared_expert_gate: &seg,
            hidden,
            mi,
            shared_mi: smi,
            top_k: cfg.moe.experts_per_tok,
            norm_topk_prob: cfg.moe.norm_topk_prob,
        };
        for t in [0usize, 9, 15] {
            let got = moe.forward(&o.row(&format!("a.layer{l}.mlp.in"), t));
            check(
                &format!("moe L{l} t={t}"),
                &got,
                &o.row(&format!("a.layer{l}.mlp"), t),
                5e-6,
            );
        }
    }
}

/// The layer's WIRING, isolated from its block math.
///
/// Substitutes the reference's own recorded block outputs for the token mixer and
/// the MoE, so anything that fails here is composition: the PLE add, the residual
/// read, or the write-back. Both halves are checked independently —
///   * read:  the collapsed input must equal what the reference fed its block
///   * write: the layer output must equal the reference's
/// which localises a failure to one side instead of just "the layer is wrong".
///
/// Covers the GatedDeltaNet layers, the PLE layer, and the sparse-attention layer.
#[test]
fn layer_composition_matches_the_reference() {
    use hipfire_arch_qwen4exp::hc::GatedResidual;
    let Some(o) = Oracle::open() else { return };
    let cfg = o.cfg();
    let (hc, hidden) = (cfg.gated_residual.count, cfg.hidden);
    let ple_layer = cfg.ngram.as_ref().map(|n| n.layer_idx);

    for l in 0..cfg.layers {
        let lp = format!("w.model.layers.{l}");
        let ap = format!("a.layer{l}");
        let mixer_tag = if o.m["tensors"][format!("{ap}.linear_attn")].is_null() {
            "self_attn"
        } else {
            "linear_attn"
        };

        // Both hyper-connections for this layer.
        let mut grw = Vec::new();
        for which in ["attn_hyper_connection", "mlp_hyper_connection"] {
            grw.push((
                o.get(&format!("{lp}.{which}.hc_norm.weight")),
                o.get(&format!("{lp}.{which}.input_mix_weight_down.weight")),
                o.get(&format!("{lp}.{which}.input_mix_weight_up.weight")),
                o.get(&format!("{lp}.{which}.block_inject_weight.weight")),
            ));
        }
        let gr = |i: usize| GatedResidual {
            hc_norm: &grw[i].0,
            mix_down: &grw[i].1,
            mix_up: &grw[i].2,
            block_inject: Some(&grw[i].3),
            hc_count: hc,
            hidden,
            lowrank: cfg.gated_residual.lowrank,
            eps: cfg.rms_norm_eps,
        };

        for t in [0usize, 9, 15] {
            let mut h = o.row(&format!("{ap}.in"), t);
            // PLE is additive on the WIDE stream, BEFORE the residual read.
            if ple_layer == Some(l) {
                for (v, p) in h.iter_mut().zip(o.row(&format!("{ap}.ple"), t)) {
                    *v += p;
                }
            }

            let r0 = gr(0).read(&h);
            check(
                &format!("L{l} t={t} read -> {mixer_tag}"),
                &r0.mixed_input,
                &o.row(&format!("{ap}.{mixer_tag}.in"), t),
                5e-6,
            );
            gr(0).write(
                &mut h,
                &o.row(&format!("{ap}.{mixer_tag}"), t),
                r0.inject.as_ref().unwrap(),
            );

            let r1 = gr(1).read(&h);
            check(
                &format!("L{l} t={t} read -> mlp"),
                &r1.mixed_input,
                &o.row(&format!("{ap}.mlp.in"), t),
                5e-6,
            );
            gr(1).write(
                &mut h,
                &o.row(&format!("{ap}.mlp"), t),
                r1.inject.as_ref().unwrap(),
            );

            check(&format!("L{l} t={t} layer out"), &h, &o.row(&ap, t), 5e-6);
        }
    }
}

impl Oracle {
    /// Exact integer tensor from the manifest (hash multipliers, row ids).
    fn ints(&self, name: &str) -> Vec<i64> {
        let t = &self.m["tensors"][name];
        assert!(!t.is_null(), "oracle has no tensor `{name}`");
        t["ints"]
            .as_array()
            .unwrap_or_else(|| panic!("`{name}` is not an integer tensor"))
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect()
    }
}

/// The derived n-gram addressing: hash multipliers, per-head primes, offsets.
///
/// All three are stored in the checkpoint as buffers, so a wrong derivation is
/// silently overridden at load — until something recomputes them, which the
/// on-disk n-gram format does. The multipliers run to ~4e16 and are compared as
/// exact integers.
#[test]
fn ngram_addressing_matches_the_reference() {
    let Some(o) = Oracle::open() else { return };
    let cfg = o.cfg();
    let n = cfg.ngram.as_ref().expect("oracle has a PLE layer");
    let bp = format!("b.model.layers.{}.ple.ple_embedding", n.layer_idx);

    let got_mul = hipfire_arch_qwen4exp::build_layer_multipliers(
        cfg.vocab as u64,
        n.ngram_size,
        n.ple_index,
        n.seed,
    );
    let want_mul: Vec<u64> = o
        .ints(&format!("{bp}.layer_multipliers"))
        .into_iter()
        .map(|v| v as u64)
        .collect();
    assert_eq!(got_mul, want_mul, "hash multipliers differ");

    let (sizes, offsets, _padded) = hipfire_arch_qwen4exp::ngram_head_layout_at(
        n.vocab_size_base,
        n.heads(),
        n.divisible_by,
        n.ple_index,
    );
    let want_sizes: Vec<u64> = o
        .ints(&format!("{bp}.ngram_heads_vocab_sizes"))
        .into_iter()
        .map(|v| v as u64)
        .collect();
    let want_offs: Vec<u64> = o
        .ints(&format!("{bp}.ngram_heads_offsets"))
        .into_iter()
        .map(|v| v as u64)
        .collect();
    assert_eq!(sizes, want_sizes, "per-head prime vocab sizes differ");
    assert_eq!(offsets, want_offs, "per-head table offsets differ");
}

/// The hashed row ids themselves, per token.
///
/// The oracle's token stream contains the EOS id partway through, so the
/// segment-aware context window is exercised rather than assumed: after an EOS the
/// window must fall back to EOS instead of reading across the boundary.
#[test]
fn ngram_row_ids_match_the_reference() {
    use hipfire_arch_qwen4exp::ngram::NgramHasher;
    let Some(o) = Oracle::open() else { return };
    let cfg = o.cfg();
    let n = cfg.ngram.as_ref().expect("oracle has a PLE layer");
    let eos = 2u32; // the generator's eos_token_id
    let h = NgramHasher::from_config(n, cfg.vocab as u64, eos);

    let toks: Vec<u32> = o.ints("input_ids").into_iter().map(|v| v as u32).collect();
    assert!(
        toks.contains(&eos),
        "the oracle token stream has no EOS — the segment-aware window is untested"
    );

    // The reference prepends `context_len` EOS cells, then slices the tail.
    let ctx_len = n.ngram_size - 1;
    let mut hist: Vec<u32> = vec![eos; ctx_len];
    hist.extend_from_slice(&toks);

    let want = o.ints(&format!("a.layer{}.ngram_rows.in", n.layer_idx));
    let heads = n.heads();
    assert_eq!(want.len(), toks.len() * heads);

    for t in 0..toks.len() {
        let i = t + ctx_len;
        let preds: Vec<Option<u32>> = hist[..i].iter().map(|&v| Some(v)).collect();
        let got = h.rows(hist[i], &preds);
        let exp: Vec<u64> = want[t * heads..(t + 1) * heads]
            .iter()
            .map(|&v| v as u64)
            .collect();
        assert_eq!(
            got,
            exp,
            "n-gram rows differ at token {t} (id {}, {} positions after the EOS)",
            hist[i],
            toks[..t]
                .iter()
                .rposition(|&v| v == eos)
                .map_or(t, |p| t - p)
        );
    }
}

/// The PLE block end to end: n-gram lookup, per-stream gate, dilated conv.
///
/// Run as a streaming loop over the sequence with a fresh conv state, which is the
/// decode shape — so this also proves the streaming form reproduces the
/// reference's batched convolution, not just the algebra.
#[test]
fn ple_block_matches_the_reference() {
    use hipfire_arch_qwen4exp::ngram::NgramHasher;
    use hipfire_arch_qwen4exp::ple::PleLayer;
    let Some(o) = Oracle::open() else { return };
    let cfg = o.cfg();
    let n = cfg.ngram.as_ref().expect("oracle has a PLE layer").clone();
    let l = n.layer_idx;
    let lp = format!("w.model.layers.{l}.ple");

    // The n-gram table, then the concatenated per-head lookup.
    let table = o.get(&format!("{lp}.ple_embedding.ngram_embedding.weight"));
    let head_dim = n.head_dim();
    let eos = 2u32;
    let h = NgramHasher::from_config(&n, cfg.vocab as u64, eos);
    let toks: Vec<u32> = o.ints("input_ids").into_iter().map(|v| v as u32).collect();
    let ctx_len = n.ngram_size - 1;
    let mut hist: Vec<u32> = vec![eos; ctx_len];
    hist.extend_from_slice(&toks);

    let (kp, vp) = (
        o.get(&format!("{lp}.key_proj.weight")),
        o.get(&format!("{lp}.value_proj.weight")),
    );
    let (nk, nq, nc, cw) = (
        o.get(&format!("{lp}.norm_key.weight")),
        o.get(&format!("{lp}.norm_query.weight")),
        o.get(&format!("{lp}.norm_conv.weight")),
        o.get(&format!("{lp}.conv1d.weight")),
    );
    let ple = PleLayer {
        key_proj: &kp,
        value_proj: &vp,
        norm_key: &nk,
        norm_query: &nq,
        norm_conv: &nc,
        conv_weight: &cw,
        hc_count: cfg.gated_residual.count,
        hidden: cfg.hidden,
        embed_dim: n.embed_dim,
        kernel: n.conv_kernel,
        dilation: n.ngram_size,
        eps: cfg.rms_norm_eps,
    };
    let mut state = vec![0.0f32; ple.width() * ple.state_len()];

    for t in 0..toks.len() {
        // n-gram lookup: concatenate this token's per-head rows.
        let rows = h.rows(
            hist[t + ctx_len],
            &hist[..t + ctx_len]
                .iter()
                .map(|&v| Some(v))
                .collect::<Vec<_>>(),
        );
        let embed: Vec<f32> = rows
            .iter()
            .flat_map(|&r| table[r as usize * head_dim..(r as usize + 1) * head_dim].to_vec())
            .collect();
        check(
            &format!("ple embed t={t}"),
            &embed,
            &o.row(&format!("a.layer{l}.ple_embedding"), t),
            1e-6,
        );

        let got = ple.step(&o.row(&format!("a.layer{l}.ple.in"), t), &embed, &mut state);
        check(
            &format!("ple out t={t}"),
            &got,
            &o.row(&format!("a.layer{l}.ple"), t),
            5e-6,
        );
    }
}

/// The Gated DeltaNet mixer, streamed token by token.
///
/// The reference runs its CHUNKED kernel for this 16-token prefill while this is
/// the RECURRENT form, so agreement here is a statement about the algebra, not
/// just about a shared code path — the two are independent implementations of the
/// same rule. Tolerance is looser than the elementwise blocks for that reason.
#[test]
fn gdn_block_matches_the_reference() {
    use hipfire_arch_qwen4exp::gdn_cpu::GdnCpu;
    let Some(o) = Oracle::open() else { return };
    let cfg = o.cfg();
    let d = &cfg.deltanet;

    for l in 0..cfg.layers {
        let lp = format!("w.model.layers.{l}.linear_attn");
        if o.m["tensors"][format!("{lp}.A_log")].is_null() {
            continue; // sparse-attention layer
        }
        let (qkv, zz, aa, bb) = (
            o.get(&format!("{lp}.in_proj_qkv.weight")),
            o.get(&format!("{lp}.in_proj_z.weight")),
            o.get(&format!("{lp}.in_proj_a.weight")),
            o.get(&format!("{lp}.in_proj_b.weight")),
        );
        let (cw, al, dt, nw, op) = (
            o.get(&format!("{lp}.conv1d.weight")),
            o.get(&format!("{lp}.A_log")),
            o.get(&format!("{lp}.dt_bias")),
            o.get(&format!("{lp}.norm.weight")),
            o.get(&format!("{lp}.out_proj.weight")),
        );
        let g = GdnCpu {
            in_proj_qkv: &qkv,
            in_proj_z: &zz,
            in_proj_a: &aa,
            in_proj_b: &bb,
            conv_weight: &cw,
            a_log: &al,
            dt_bias: &dt,
            norm_weight: &nw,
            out_proj: &op,
            hidden: cfg.hidden,
            n_k: d.key_heads,
            n_v: d.value_heads,
            head_k: d.key_head_dim,
            head_v: d.value_head_dim,
            kernel: d.conv_kernel,
            gate_sigmoid: d.output_gate_sigmoid,
            eps: cfg.rms_norm_eps,
        };
        let mut st = g.zero_state();
        let n_tok = o.shape(&format!("a.layer{l}.linear_attn.in"))[1];
        for t in 0..n_tok {
            let got = g.step(&o.row(&format!("a.layer{l}.linear_attn.in"), t), &mut st);
            check(
                &format!("gdn L{l} t={t}"),
                &got,
                &o.row(&format!("a.layer{l}.linear_attn"), t),
                2e-4,
            );
        }
    }
}

/// The sparse-attention block: doubled `q_proj`, per-head norms, RoPE, masked
/// attention, sigmoid output gate, `o_proj`.
///
/// Uses the reference's own selection mask and rotary tables, so a failure here is
/// the attention block itself rather than the indexer or mrope. The selection is
/// verified separately (`parity_qwen4exp_kernels`, and `qsa::select`'s own tests).
#[test]
fn qsa_attention_block_matches_the_reference() {
    use hipfire_arch_qwen4exp::attn::QsaAttention;
    let Some(o) = Oracle::open() else { return };
    let cfg = o.cfg();

    for l in cfg.sparse_attention_layers().collect::<Vec<_>>() {
        let lp = format!("w.model.layers.{l}.self_attn");
        let (qp, kp, vp, op) = (
            o.get(&format!("{lp}.q_proj.weight")),
            o.get(&format!("{lp}.k_proj.weight")),
            o.get(&format!("{lp}.v_proj.weight")),
            o.get(&format!("{lp}.o_proj.weight")),
        );
        let (qn, kn) = (
            o.get(&format!("{lp}.q_norm.weight")),
            o.get(&format!("{lp}.k_norm.weight")),
        );
        let a = QsaAttention {
            q_proj: &qp,
            k_proj: &kp,
            v_proj: &vp,
            o_proj: &op,
            q_norm: &qn,
            k_norm: &kn,
            hidden: cfg.hidden,
            n_heads: cfg.n_heads,
            n_kv: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            eps: cfg.rms_norm_eps,
        };

        let hs = o.get(&format!("a.layer{l}.self_attn.in"));
        let n_tok = o.shape(&format!("a.layer{l}.self_attn.in"))[1];
        let (cos, sin) = (o.get("a.rotary"), o.get("a.rotary.1"));

        // Combined mask: causal AND the indexer's selection.
        let sel = o.ints(&format!("a.layer{l}.sa.indexer"));
        let visible: Vec<bool> = (0..n_tok)
            .flat_map(|i| (0..n_tok).map(move |j| (i, j)))
            .map(|(i, j)| j <= i && sel[i * n_tok + j] != 0)
            .collect();
        let kept = visible.iter().filter(|v| **v).count();
        assert!(
            kept < n_tok * (n_tok + 1) / 2,
            "the selection mask excludes nothing — this layer is testing dense attention"
        );

        let got = a.forward(&hs, n_tok, &cos, &sin, &visible);
        for t in 0..n_tok {
            check(
                &format!("qsa attn L{l} t={t}"),
                &got[t * cfg.hidden..(t + 1) * cfg.hidden],
                &o.row(&format!("a.layer{l}.self_attn"), t),
                5e-5,
            );
        }
    }
}

/// The rotary tables, and the claim that mRoPE collapses to plain RoPE on text.
///
/// Checks the derived `inv_freq` against the checkpoint's own buffer first: if
/// that drifts, cos/sin would still "match" a wrong table computed the same wrong
/// way on both sides of a self-consistent test.
#[test]
fn rope_tables_match_the_reference() {
    use hipfire_arch_qwen4exp::rope::{cos_sin, inv_freq};
    let Some(o) = Oracle::open() else { return };
    let cfg = o.cfg();
    let rd = cfg.rotary_dim();

    let got = inv_freq(rd, cfg.rope_theta);
    check(
        "inv_freq",
        &got,
        &o.get("b.model.rotary_emb.inv_freq"),
        1e-6,
    );

    let n_tok = o.shape("a.rotary")[1];
    let pos: Vec<usize> = (0..n_tok).collect();
    let (cos, sin) = cos_sin(&pos, &got);
    check("rope cos", &cos, &o.get("a.rotary"), 1e-6);
    check("rope sin", &sin, &o.get("a.rotary.1"), 1e-6);
}

/// The indexer's selection mask, exactly — it is a set of booleans, so there is no
/// tolerance to hide behind.
#[test]
fn indexer_mask_matches_the_reference() {
    use hipfire_arch_qwen4exp::attn::Indexer;
    let Some(o) = Oracle::open() else { return };
    let cfg = o.cfg();
    let ix = &cfg.indexer;

    for l in cfg.sparse_attention_layers().collect::<Vec<_>>() {
        let lp = format!("w.model.layers.{l}.self_attn.indexer");
        let (qk, qn, kn) = (
            o.get(&format!("{lp}.index_qk_proj.weight")),
            o.get(&format!("{lp}.q_layernorm.weight")),
            o.get(&format!("{lp}.k_layernorm.weight")),
        );
        let idx = Indexer {
            qk_proj: &qk,
            q_norm: &qn,
            k_norm: &kn,
            hidden: cfg.hidden,
            n_heads: ix.n_heads,
            kv_heads: ix.kv_heads,
            head_dim: ix.head_dim,
            budget: ix.budget,
            compress_ratio: ix.compress_ratio,
            eps: cfg.rms_norm_eps,
        };
        let hs = o.get(&format!("a.layer{l}.self_attn.in"));
        let n_tok = o.shape(&format!("a.layer{l}.self_attn.in"))[1];
        let (cos, sin) = (o.get("a.rotary"), o.get("a.rotary.1"));
        let causal: Vec<bool> = (0..n_tok)
            .flat_map(|i| (0..n_tok).map(move |j| j <= i))
            .collect();

        let got = idx.select_mask(&hs, n_tok, &cos, &sin, &causal);
        let want: Vec<bool> = o
            .ints(&format!("a.layer{l}.sa.indexer"))
            .into_iter()
            .map(|v| v != 0)
            .collect();
        for t in 0..n_tok {
            let (g, w) = (
                &got[t * n_tok..(t + 1) * n_tok],
                &want[t * n_tok..(t + 1) * n_tok],
            );
            assert_eq!(
                g,
                w,
                "indexer mask differs at L{l} query {t}\n  got  {:?}\n  want {:?}",
                g.iter()
                    .enumerate()
                    .filter(|(_, v)| **v)
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>(),
                w.iter()
                    .enumerate()
                    .filter(|(_, v)| **v)
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>(),
            );
        }
        println!("  indexer L{l}: {n_tok} queries exact");
    }
}

/// THE END-TO-END TEST: tokens in, logits out, against the reference.
///
/// Nothing is substituted from the reference here — not the rotary tables, not the
/// selection mask, not a single block output. Every earlier test in this file
/// isolates one piece so a failure here can be localised; this one is the claim
/// that the pieces compose.
#[test]
fn full_trunk_matches_the_reference_logits() {
    use hipfire_arch_qwen4exp::trunk::{forward, WeightSource};
    let Some(o) = Oracle::open() else { return };
    let cfg = o.cfg();

    /// Serves canonical checkpoint names out of the oracle's own naming.
    struct Src {
        cache: std::collections::HashMap<String, Vec<f32>>,
    }
    impl WeightSource for Src {
        fn get(&self, name: &str) -> &[f32] {
            self.cache
                .get(name)
                .unwrap_or_else(|| panic!("trunk asked for missing weight `{name}`"))
        }
    }

    let mut cache = std::collections::HashMap::new();
    for (k, _) in o.m["tensors"].as_object().unwrap() {
        if let Some(stripped) = k.strip_prefix("w.") {
            cache.insert(rename(stripped), o.get(k));
        }
    }
    let src = Src { cache };

    let tokens: Vec<u32> = o.ints("input_ids").into_iter().map(|v| v as u32).collect();
    let got = forward(&cfg, &src, &tokens, 2);
    let want = o.get("out.logits");
    assert_eq!(got.len(), want.len());

    // Report per position, so a divergence that only appears deep in the sequence
    // is visible rather than hidden behind a single max.
    for t in 0..tokens.len() {
        let (g, wv) = (
            &got[t * cfg.vocab..(t + 1) * cfg.vocab],
            &want[t * cfg.vocab..(t + 1) * cfg.vocab],
        );
        check(&format!("logits t={t}"), g, wv, 3e-4);
        // The argmax is what generation actually consumes.
        let am = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };
        assert_eq!(am(g), am(wv), "argmax differs at position {t}");
    }
}

// ── vision tower ────────────────────────────────────────────────────────────

/// The vision tower, stage by stage against the reference.
///
/// Unlike the MTP head, upstream DOES implement this, so it gets a real oracle.
/// The rotary tables are taken from the reference so a failure here is the tower
/// rather than the grid-position derivation, which is checked separately.
#[test]
fn vision_tower_matches_the_reference() {
    use hipfire_arch_qwen4exp::vision::{layer_norm, merger, patch_embed, VisionBlock};
    let Some(o) = Oracle::open() else { return };
    let v = &o.m["config"]["vision_config"];
    let hidden = v["hidden_size"].as_u64().unwrap() as usize;
    let n_heads = v["num_heads"].as_u64().unwrap() as usize;
    let inter = v["intermediate_size"].as_u64().unwrap() as usize;
    let depth = v["depth"].as_u64().unwrap() as usize;
    let merge = v["spatial_merge_size"].as_u64().unwrap() as usize;
    let out_hidden = v["out_hidden_size"].as_u64().unwrap() as usize;
    let (ch, tps, ps) = (
        v["in_channels"].as_u64().unwrap() as usize,
        v["temporal_patch_size"].as_u64().unwrap() as usize,
        v["patch_size"].as_u64().unwrap() as usize,
    );
    let eps = 1e-6f32;
    let n_tok = o.shape("va.patch_embed")[0];
    let hd = hidden / n_heads;

    // Patch embed: a Conv3d whose stride equals its kernel is a per-patch linear.
    let got = patch_embed(
        &o.get("v.pixels"),
        &o.get("vw.patch_embed.proj.weight"),
        &o.get("vw.patch_embed.proj.bias"),
        ch * tps * ps * ps,
        hidden,
    );
    check("vision patch_embed", &got, &o.get("va.patch_embed"), 5e-6);

    // cos/sin: the reference concatenates the half-width table with itself.
    let rot = o.get("va.rotary");
    let half = rot.len() / n_tok;
    assert_eq!(half * 2, hd, "rotary half-width {half} vs head_dim {hd}");
    let mut cos = vec![0.0f32; n_tok * hd];
    let mut sin = vec![0.0f32; n_tok * hd];
    for t in 0..n_tok {
        for i in 0..hd {
            let f = rot[t * half + i % half];
            cos[t * hd + i] = f.cos();
            sin[t * hd + i] = f.sin();
        }
    }

    for l in 0..depth {
        let bp = format!("vw.blocks.{l}");
        let g = |n: &str| o.get(&format!("{bp}.{n}"));
        let (n1w, n1b, n2w, n2b) = (
            g("norm1.weight"),
            g("norm1.bias"),
            g("norm2.weight"),
            g("norm2.bias"),
        );
        let (qw, qb, pw, pb) = (
            g("attn.qkv.weight"),
            g("attn.qkv.bias"),
            g("attn.proj.weight"),
            g("attn.proj.bias"),
        );
        let (f1w, f1b, f2w, f2b) = (
            g("mlp.linear_fc1.weight"),
            g("mlp.linear_fc1.bias"),
            g("mlp.linear_fc2.weight"),
            g("mlp.linear_fc2.bias"),
        );
        let blk = VisionBlock {
            norm1_w: &n1w,
            norm1_b: &n1b,
            norm2_w: &n2w,
            norm2_b: &n2b,
            qkv_w: &qw,
            qkv_b: &qb,
            proj_w: &pw,
            proj_b: &pb,
            fc1_w: &f1w,
            fc1_b: &f1b,
            fc2_w: &f2w,
            fc2_b: &f2b,
            hidden,
            n_heads,
            intermediate: inter,
            eps,
        };
        let x = o.get(&format!("va.block{l}.in"));

        // Attention and MLP separately first, so a failure localises.
        let n1 = layer_norm(&x, &n1w, &n1b, hidden, eps);
        check(
            &format!("vision L{l} norm1"),
            &n1,
            &o.get(&format!("va.block{l}.attn.in")),
            5e-6,
        );
        check(
            &format!("vision L{l} attn"),
            &blk.attention(&n1, n_tok, &cos, &sin),
            &o.get(&format!("va.block{l}.attn")),
            2e-5,
        );
        check(
            &format!("vision L{l} mlp"),
            &blk.mlp(&o.get(&format!("va.block{l}.mlp.in"))),
            &o.get(&format!("va.block{l}.mlp")),
            2e-5,
        );
        check(
            &format!("vision L{l} block"),
            &blk.forward(&x, n_tok, &cos, &sin),
            &o.get(&format!("va.block{l}")),
            3e-5,
        );
    }

    // Merger: normalise at the UNMERGED width, then fold merge^2 patches.
    let got = merger(
        &o.get("va.merger.in"),
        &o.get("vw.merger.norm.weight"),
        &o.get("vw.merger.norm.bias"),
        &o.get("vw.merger.linear_fc1.weight"),
        &o.get("vw.merger.linear_fc1.bias"),
        &o.get("vw.merger.linear_fc2.weight"),
        &o.get("vw.merger.linear_fc2.bias"),
        hidden,
        merge * merge,
        out_hidden,
        eps,
    );
    check("vision merger", &got, &o.get("v.pooler_output"), 2e-5);
}

/// The two GELUs in this tower are NOT interchangeable.
///
/// The block MLPs use the tanh approximation, the merger the exact erf form. They
/// agree to ~1e-3, which is close enough that swapping them looks like numerical
/// noise in an end-to-end comparison and passes a loose tolerance.
#[test]
fn the_two_vision_gelus_are_distinguishable() {
    use hipfire_arch_qwen4exp::vision::{gelu_erf, gelu_tanh};
    let mut worst = 0.0f32;
    for i in -40..40 {
        let x = i as f32 * 0.1;
        worst = worst.max((gelu_tanh(x) - gelu_erf(x)).abs());
        // Both must still be a GELU: exact at 0, and asymptotic to x and 0.
        assert!((gelu_tanh(0.0)).abs() < 1e-9 && (gelu_erf(0.0)).abs() < 1e-6);
    }
    assert!(
        worst > 1e-4,
        "the two GELUs differ by only {worst:.2e} — this test cannot tell them apart"
    );
    assert!(
        (gelu_erf(6.0) - 6.0).abs() < 1e-4,
        "erf gelu should approach x"
    );
    assert!(gelu_erf(-6.0).abs() < 1e-4, "erf gelu should approach 0");
}

/// Patch ordering and the position-grid resample — the two derivations the tower
/// test substituted from the reference.
///
/// The interpolation weights are not captured directly (they are a local in the
/// reference's forward), so the check is on the RESULT: the position embedding is
/// exactly what the tower adds to the patch embedding, i.e.
/// `block0.in - patch_embed`.
#[test]
fn vision_position_grid_matches_the_reference() {
    use hipfire_arch_qwen4exp::vision::{pos_embed_interpolation, pos_embeds, position_ids};
    let Some(o) = Oracle::open() else { return };
    let v = &o.m["config"]["vision_config"];
    let hidden = v["hidden_size"].as_u64().unwrap() as usize;
    let merge = v["spatial_merge_size"].as_u64().unwrap() as usize;
    let num_grid = (v["num_position_embeddings"].as_u64().unwrap() as f64).sqrt() as usize;

    let grid = o.ints("v.grid_thw");
    let (t, h, w) = (grid[0] as usize, grid[1] as usize, grid[2] as usize);

    // Patch order, against the reference's own position_ids.
    let got = position_ids(t, h, w, merge);
    let want = o.ints("va.rotary.in");
    assert_eq!(got.len() * 2, want.len());
    for (i, &(r, c)) in got.iter().enumerate() {
        assert_eq!(
            (r as i64, c as i64),
            (want[i * 2], want[i * 2 + 1]),
            "patch {i} position differs — check the merge-block ordering"
        );
    }

    // Corner indices, against the reference's gather.
    let interp = pos_embed_interpolation(&got, h, w, num_grid);
    let want_idx = o.ints("va.pos_embed.in");
    for (i, (idx, _)) in interp.iter().enumerate() {
        for j in 0..4 {
            assert_eq!(
                idx[j] as i64,
                want_idx[i * 4 + j],
                "patch {i} corner {j} differs"
            );
        }
    }

    // The rotary frequencies, so nothing in the tower test is substituted any more.
    let hd = hidden / (v["num_heads"].as_u64().unwrap() as usize);
    let mine_rot = hipfire_arch_qwen4exp::vision::rotary_frequencies(&got, hd / 2, 10000.0);
    check("vision rotary", &mine_rot, &o.get("va.rotary"), 1e-6);

    // The weighted sum, via what the tower actually added.
    let table = o.get("vw.pos_embed.weight");
    let mine = pos_embeds(&table, &interp, hidden);
    let added: Vec<f32> = o
        .get("va.block0.in")
        .iter()
        .zip(o.get("va.patch_embed"))
        .map(|(a, b)| a - b)
        .collect();
    check("vision pos_embeds", &mine, &added, 5e-6);
}

/// Text/vision fusion: merged vision tokens scattered into the text embedding.
///
/// Captured off the FULL `Qwen4ExpModel`, which is where the splice happens — the
/// text model alone never sees it.
#[test]
fn image_embeds_splice_into_the_text_stream() {
    use hipfire_arch_qwen4exp::vision::splice_image_embeds;
    let Some(o) = Oracle::open() else { return };
    let ids: Vec<u32> = o
        .ints("f.input_ids")
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let img_tok = o.ints("f.image_token_id")[0] as u32;
    let hidden = o.shape("f.fused_embeds")[2];
    let embed = o.get("f.embed_tokens");
    let image_embeds = o.get("f.image_embeds");

    let n_slots = ids.iter().filter(|&&t| t == img_tok).count();
    assert!(n_slots > 0, "the fusion prompt has no image placeholders");
    assert!(
        n_slots < ids.len(),
        "the prompt is ALL placeholders — text positions are untested"
    );

    // Start from a plain embedding lookup, then splice.
    let mut mine: Vec<f32> = ids
        .iter()
        .flat_map(|&t| embed[t as usize * hidden..(t as usize + 1) * hidden].to_vec())
        .collect();
    let n = splice_image_embeds(&mut mine, &ids, &image_embeds, img_tok, hidden).unwrap();
    assert_eq!(n, n_slots);
    check("fused embeds", &mine, &o.get("f.fused_embeds"), 1e-6);

    // A count mismatch must be an error, not a truncated splice.
    let mut short = mine.clone();
    let err = splice_image_embeds(
        &mut short,
        &ids,
        &image_embeds[..(n_slots - 1) * hidden],
        img_tok,
        hidden,
    )
    .unwrap_err();
    assert!(err.contains("placeholders"), "unhelpful error: {err}");
}

/// The vision tower's output width must equal the text hidden size, since merged
/// tokens are spliced straight into the text embedding stream.
///
/// The reference only catches this at the scatter, as a confusing feature-count
/// error; the config parser rejects it up front.
#[test]
fn vision_width_must_match_the_text_hidden() {
    let Some(o) = Oracle::open() else { return };
    let mut bad = o.m["config"].clone();
    let h = bad["text_config"]["hidden_size"].as_u64().unwrap();
    bad["vision_config"]["out_hidden_size"] = serde_json::json!(h * 2);
    let err = Qwen4ExpConfig::from_json(&bad).unwrap_err();
    assert!(
        err.contains("out_hidden_size") && err.contains("hidden_size"),
        "unhelpful error: {err}"
    );
}
