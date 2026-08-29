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
use crate::moe::MoeLayer;

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
        for d in 0..h {
            out[s * h + d] = ph[d] + pe[d];
        }
    }
    out
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
