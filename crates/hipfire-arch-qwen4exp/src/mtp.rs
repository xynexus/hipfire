// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! The embedded multi-token-prediction head — CPU.
//!
//! # ⚠️ This is SHAPE-INFERRED, not reference-verified
//!
//! Every other part of this port is differenced against the pinned upstream
//! implementation. This one cannot be: upstream sets
//! `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` and DROPS these weights on
//! load, so there is no reference forward to compare against. Treat the numbers
//! this produces as unverified until an implementation or a trace exists.
//!
//! What the checkpoint's shapes DO pin, which is more than it first appears:
//!
//! * `pre_fc_norm_hidden` is `[hc_count * hidden]`, so the hidden state arriving
//!   from the trunk is the **wide** residual, not the collapsed output.
//! * The head's layer expects a wide input too — its `hc_norm` is
//!   `[hc_count * hidden]`.
//! * `fc_hidden` is `[hidden, hidden]`, which cannot consume a wide vector in one
//!   go. The ONLY `[*, hc_count * hidden]` matrices anywhere under `mtp.` are
//!   hyper-connection internals (`input_mix_weight_down`, `block_inject_weight`) —
//!   there is no general wide→narrow projection. So `fc_hidden` must be applied
//!   per stream, keeping the stream count intact.
//! * `mtp.hyper_connection_mixer` is a `use_combine = false` gated residual, i.e.
//!   the same final collapse the trunk uses before `lm_head`. Spending it at the
//!   INPUT instead would leave nothing to collapse the head's output.
//!
//! That leaves one composition consistent with all four facts, which is what this
//! implements. `mtp_use_dedicated_embeddings` is false in the shipped model, so the
//! embedding table and `lm_head` are the trunk's.
//!
//! The genuinely unpinned part is how `fc_embedding`'s narrow output reaches the
//! wide stream — broadcast-added to every stream is the natural reading and the one
//! used here, but nothing in the shapes rules out, say, adding it to stream 0 only.

use crate::attn::{Indexer, QsaAttention};
use crate::config::Qwen4ExpConfig;
use crate::hc::{grouped_rmsnorm, GatedResidual};
use crate::moe::{Expert, MoeLayer};
use crate::trunk::WeightSource;

/// How `fc_embedding`'s narrow output reaches the WIDE stream.
///
/// The module docs call this the genuinely unpinned part: "broadcast-added to
/// every stream is the natural reading ... but nothing in the shapes rules out,
/// say, adding it to stream 0 only." Both are expressible so the question can be
/// MEASURED rather than assumed — `examples/mtp_probe` scores them against a
/// do-nothing baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fusion {
    /// Add the projected embedding to every stream (the original reading).
    BroadcastAllStreams,
    /// Add it to stream 0 only, leaving the others carrying hidden alone.
    Stream0Only,
}

/// Build the head's wide input from the trunk's wide hidden state and the
/// embedding of the token being predicted from.
///
/// See the module docs: `fc_hidden` runs per stream, `fc_embedding` once, and the
/// narrow result is broadcast across streams.
pub fn fuse_inputs(
    cfg: &Qwen4ExpConfig,
    wide_hidden: &[f32],
    embedding: &[f32],
    pre_fc_norm_hidden: &[f32],
    pre_fc_norm_embedding: &[f32],
    fc_hidden: &[f32],
    fc_embedding: &[f32],
    fusion: Fusion,
) -> Vec<f32> {
    let (h, hc) = (cfg.hidden, cfg.gated_residual.count);
    assert_eq!(wide_hidden.len(), hc * h);
    assert_eq!(embedding.len(), h);

    let hn = grouped_rmsnorm(wide_hidden, pre_fc_norm_hidden, h, cfg.rms_norm_eps);
    let en = grouped_rmsnorm(embedding, pre_fc_norm_embedding, h, cfg.rms_norm_eps);
    let mv = |w: &[f32], x: &[f32]| -> Vec<f32> {
        (0..h)
            .map(|r| (0..h).map(|c| w[r * h + c] * x[c]).sum())
            .collect()
    };
    let pe = mv(fc_embedding, &en);

    let mut out = vec![0.0f32; hc * h];
    for s in 0..hc {
        let ph = mv(fc_hidden, &hn[s * h..(s + 1) * h]);
        let take_embed = match fusion {
            Fusion::BroadcastAllStreams => true,
            Fusion::Stream0Only => s == 0,
        };
        for d in 0..h {
            out[s * h + d] = ph[d] + if take_embed { pe[d] } else { 0.0 };
        }
    }
    out
}

/// Assemble [`MtpWeights`] from a weight source.
///
/// The head is structurally one trunk sparse-attention layer plus a stream
/// mixer, so this mirrors `trunk::forward`'s per-layer assembly with the `mtp.`
/// prefix. Kept beside the forward it feeds: the tensor NAMES are the contract
/// between `weights.rs` (which declares the expectations) and this, and a
/// mismatch is a missing-weight panic rather than a wrong answer.
///
/// `mtp_use_dedicated_embeddings` is false in the shipped model, so there is no
/// embed or `lm_head` here — the caller supplies the trunk's.
pub fn weights_from<'a>(cfg: &Qwen4ExpConfig, w: &'a dyn WeightSource) -> MtpWeights<'a> {
    let (hidden, hc) = (cfg.hidden, cfg.gated_residual.count);
    let lp = "mtp.layers.0";
    let gated = |which: &str, prefix: &str, inject: bool| GatedResidual {
        hc_norm: w.get(&format!("{prefix}.{which}hc_norm.weight")),
        mix_down: w.get(&format!("{prefix}.{which}input_mix_weight_down.weight")),
        mix_up: w.get(&format!("{prefix}.{which}input_mix_weight_up.weight")),
        block_inject: inject.then(|| w.get(&format!("{prefix}.{which}block_inject_weight.weight"))),
        hc_count: hc,
        hidden,
        lowrank: cfg.gated_residual.lowrank,
        eps: cfg.rms_norm_eps,
    };
    let (mi, smi) = (cfg.moe.intermediate, cfg.moe.shared_intermediate);
    let mp = format!("{lp}.mlp");
    let gu = w.get(&format!("{mp}.experts.gate_up_proj"));
    let dn = w.get(&format!("{mp}.experts.down_proj"));
    let (gu_sz, dn_sz) = (2 * mi * hidden, hidden * mi);
    let sa = format!("{lp}.self_attn");
    let ix = &cfg.indexer;
    MtpWeights {
        pre_fc_norm_hidden: w.get("mtp.pre_fc_norm_hidden.weight"),
        pre_fc_norm_embedding: w.get("mtp.pre_fc_norm_embedding.weight"),
        fc_hidden: w.get("mtp.fc_hidden.weight"),
        fc_embedding: w.get("mtp.fc_embedding.weight"),
        attn_hc: gated("", &format!("{lp}.attn_hyper_connection"), true),
        mlp_hc: gated("", &format!("{lp}.mlp_hyper_connection"), true),
        attn: QsaAttention {
            q_proj: w.get(&format!("{sa}.q_proj.weight")),
            k_proj: w.get(&format!("{sa}.k_proj.weight")),
            v_proj: w.get(&format!("{sa}.v_proj.weight")),
            o_proj: w.get(&format!("{sa}.o_proj.weight")),
            q_norm: w.get(&format!("{sa}.q_norm.weight")),
            k_norm: w.get(&format!("{sa}.k_norm.weight")),
            hidden,
            n_heads: cfg.n_heads,
            n_kv: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            eps: cfg.rms_norm_eps,
        },
        indexer: Indexer {
            qk_proj: w.get(&format!("{sa}.indexer.index_qk_proj.weight")),
            q_norm: w.get(&format!("{sa}.indexer.q_layernorm.weight")),
            k_norm: w.get(&format!("{sa}.indexer.k_layernorm.weight")),
            hidden,
            n_heads: ix.n_heads,
            kv_heads: ix.kv_heads,
            head_dim: ix.head_dim,
            budget: ix.budget,
            compress_ratio: ix.compress_ratio,
            eps: cfg.rms_norm_eps,
        },
        moe: MoeLayer {
            router: w.get(&format!("{mp}.gate.weight")),
            experts: (0..cfg.moe.num_experts)
                .map(|e| Expert {
                    gate_up: &gu[e * gu_sz..(e + 1) * gu_sz],
                    down: &dn[e * dn_sz..(e + 1) * dn_sz],
                })
                .collect(),
            shared_gate: w.get(&format!("{mp}.shared_expert.gate_proj.weight")),
            shared_up: w.get(&format!("{mp}.shared_expert.up_proj.weight")),
            shared_down: w.get(&format!("{mp}.shared_expert.down_proj.weight")),
            shared_expert_gate: w.get(&format!("{mp}.shared_expert_gate.weight")),
            hidden,
            mi,
            shared_mi: smi,
            top_k: cfg.moe.experts_per_tok,
            norm_topk_prob: cfg.moe.norm_topk_prob,
        },
        mixer: gated("", "mtp.hyper_connection_mixer", false),
    }
}

/// Weights for the head's single decoder layer, plus its stream mixer.
///
/// Structurally identical to a trunk sparse-attention layer, which is why the
/// weight plan builds it with the same helpers.
pub struct MtpWeights<'a> {
    pub pre_fc_norm_hidden: &'a [f32],
    pub pre_fc_norm_embedding: &'a [f32],
    pub fc_hidden: &'a [f32],
    pub fc_embedding: &'a [f32],
    pub attn_hc: GatedResidual<'a>,
    pub mlp_hc: GatedResidual<'a>,
    pub attn: QsaAttention<'a>,
    pub indexer: Indexer<'a>,
    pub moe: MoeLayer<'a>,
    pub mixer: GatedResidual<'a>,
}

/// One MTP step over a whole sequence, returning the collapsed `[n_tok, hidden]`
/// state. The caller applies the trunk's `lm_head`, which the head shares.
pub fn forward(
    cfg: &Qwen4ExpConfig,
    w: &MtpWeights<'_>,
    wide_hidden: &[f32],
    embeddings: &[f32],
    n_tok: usize,
    cos: &[f32],
    sin: &[f32],
    fusion: Fusion,
) -> Vec<f32> {
    let (h, hc) = (cfg.hidden, cfg.gated_residual.count);
    let width = hc * h;

    let mut wide: Vec<Vec<f32>> = (0..n_tok)
        .map(|t| {
            fuse_inputs(
                cfg,
                &wide_hidden[t * width..(t + 1) * width],
                &embeddings[t * h..(t + 1) * h],
                w.pre_fc_norm_hidden,
                w.pre_fc_norm_embedding,
                w.fc_hidden,
                w.fc_embedding,
                fusion,
            )
        })
        .collect();

    let causal: Vec<bool> = (0..n_tok)
        .flat_map(|i| (0..n_tok).map(move |j| j <= i))
        .collect();

    // Attention half.
    let reads: Vec<_> = (0..n_tok).map(|t| w.attn_hc.read(&wide[t])).collect();
    let mixed: Vec<f32> = reads.iter().flat_map(|r| r.mixed_input.clone()).collect();
    let sel = w.indexer.select_mask(&mixed, n_tok, cos, sin, &causal);
    let visible: Vec<bool> = causal.iter().zip(&sel).map(|(c, s)| *c && *s).collect();
    let attn_out = w.attn.forward(&mixed, n_tok, cos, sin, &visible);
    for t in 0..n_tok {
        let inj = reads[t].inject.as_ref().expect("mtp layer injects");
        w.attn_hc
            .write(&mut wide[t], &attn_out[t * h..(t + 1) * h], inj);
    }

    // MoE half.
    let reads: Vec<_> = (0..n_tok).map(|t| w.mlp_hc.read(&wide[t])).collect();
    for t in 0..n_tok {
        let out = w.moe.forward(&reads[t].mixed_input);
        let inj = reads[t].inject.as_ref().expect("mtp layer injects");
        w.mlp_hc.write(&mut wide[t], &out, inj);
    }

    // The mixer's own norm is the last normalisation, exactly as in the trunk.
    (0..n_tok)
        .flat_map(|t| w.mixer.read(&wide[t]).mixed_input)
        .collect()
}
