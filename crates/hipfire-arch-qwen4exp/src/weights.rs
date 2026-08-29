// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! What tensors a Qwen3.8-Flash-Next checkpoint must contain, and their shapes.
//!
//! This is deliberately a PURE function of the config: `plan(&cfg)` returns the
//! complete expected manifest, and [`Plan::validate_against`] diffs it against what
//! a checkpoint actually holds. Keeping it free of GPU types is what lets the whole
//! weight-mapping layer be tested against the real 1658-tensor checkpoint without
//! allocating anything — a loader bug that would otherwise surface as a failed
//! 238 GB conversion becomes a unit test.
//!
//! Two structural facts drive most of the manifest, and both are easy to get wrong:
//!
//! * This family has **no** `input_layernorm`, **no** `post_attention_layernorm`,
//!   and **no** final `model.norm`. The gated residual's `hc_norm` replaces all
//!   three, and a loader that expects the usual pre-norms finds nothing.
//! * A layer is EITHER Gated DeltaNet OR sparse attention, never both, and the
//!   n-gram block rides exactly one layer.

use crate::config::{LayerType, Qwen4ExpConfig};
use std::collections::BTreeMap;

/// Prefix every text-model tensor carries in this checkpoint.
pub const TEXT_PREFIX: &str = "model.language_model";

/// One expected tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expect {
    pub name: String,
    pub shape: Vec<usize>,
    /// Integer addressing metadata rather than weights. These are DERIVABLE from
    /// config (see `ngram_head_layout`), so a checkpoint may omit them — but when
    /// present they are authoritative and must be preferred.
    pub derivable: bool,
}

impl Expect {
    fn w(name: String, shape: Vec<usize>) -> Self {
        Self {
            name,
            shape,
            derivable: false,
        }
    }
    fn derived(name: String, shape: Vec<usize>) -> Self {
        Self {
            name,
            shape,
            derivable: true,
        }
    }
}

/// How a checkpoint differs from what the config says it should hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mismatch {
    Missing {
        name: String,
        expected: Vec<usize>,
    },
    Shape {
        name: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mismatch::Missing { name, expected } => {
                write!(f, "missing {name} (expected {expected:?})")
            }
            Mismatch::Shape {
                name,
                expected,
                found,
            } => {
                write!(f, "{name}: expected {expected:?}, found {found:?}")
            }
        }
    }
}

/// The complete expected tensor manifest for one checkpoint.
#[derive(Debug, Clone)]
pub struct Plan {
    pub tensors: Vec<Expect>,
}

impl Plan {
    /// Total tensors the text model expects, derivable metadata included.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Diff against a checkpoint's actual `name -> shape` map.
    ///
    /// Only reports tensors the plan expects: a checkpoint carrying EXTRA tensors
    /// (a vision tower, an MTP head) is not an error, because this plans the text
    /// trunk and those are separate subsystems.
    pub fn validate_against(
        &self,
        available: &BTreeMap<String, Vec<usize>>,
    ) -> Result<(), Vec<Mismatch>> {
        let mut bad = Vec::new();
        for e in &self.tensors {
            match available.get(&e.name) {
                None if e.derivable => {}
                None => bad.push(Mismatch::Missing {
                    name: e.name.clone(),
                    expected: e.shape.clone(),
                }),
                Some(found) if *found != e.shape => bad.push(Mismatch::Shape {
                    name: e.name.clone(),
                    expected: e.shape.clone(),
                    found: found.clone(),
                }),
                Some(_) => {}
            }
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(bad)
        }
    }
}

/// Build the expected manifest for the text trunk.
///
/// Vision and MTP are deliberately excluded: they are separate subsystems with
/// their own loaders, and including them here would make a text-only checkpoint
/// look broken.

/// One sparse-attention layer's tensors. Shared by the trunk and the MTP head,
/// whose layer is structurally identical.
fn push_sparse_attention(t: &mut Vec<Expect>, cfg: &Qwen4ExpConfig, lp: &str) {
    let hidden = cfg.hidden;
    t.push(Expect::w(
        format!("{lp}.self_attn.q_proj.weight"),
        vec![cfg.q_proj_out(), hidden],
    ));
    for kv in ["k_proj", "v_proj"] {
        t.push(Expect::w(
            format!("{lp}.self_attn.{kv}.weight"),
            vec![cfg.kv_proj_out(), hidden],
        ));
    }
    t.push(Expect::w(
        format!("{lp}.self_attn.o_proj.weight"),
        vec![hidden, cfg.n_heads * cfg.head_dim],
    ));
    for n in ["q_norm", "k_norm"] {
        t.push(Expect::w(
            format!("{lp}.self_attn.{n}.weight"),
            vec![cfg.head_dim],
        ));
    }
    // One fused projection feeds both indexer queries and its single shared key head.
    t.push(Expect::w(
        format!("{lp}.self_attn.indexer.index_qk_proj.weight"),
        vec![cfg.indexer.qk_proj_out(), hidden],
    ));
    for n in ["q_layernorm", "k_layernorm"] {
        t.push(Expect::w(
            format!("{lp}.self_attn.indexer.{n}.weight"),
            vec![cfg.indexer.head_dim],
        ));
    }
}

/// One MoE block's tensors. Routed experts ship STACKED across the expert axis in
/// a safetensors source; the quantizer splits them per expert (see
/// `trunk_gpu::stack_experts`), which is a storage difference, not a plan one.
fn push_moe(t: &mut Vec<Expect>, cfg: &Qwen4ExpConfig, lp: &str) {
    let (hidden, m) = (cfg.hidden, &cfg.moe);
    t.push(Expect::w(
        format!("{lp}.mlp.gate.weight"),
        vec![m.num_experts, hidden],
    ));
    t.push(Expect::w(
        format!("{lp}.mlp.experts.gate_up_proj"),
        vec![m.num_experts, m.intermediate * 2, hidden],
    ));
    t.push(Expect::w(
        format!("{lp}.mlp.experts.down_proj"),
        vec![m.num_experts, hidden, m.intermediate],
    ));
    for proj in ["gate_proj", "up_proj"] {
        t.push(Expect::w(
            format!("{lp}.mlp.shared_expert.{proj}.weight"),
            vec![m.shared_intermediate, hidden],
        ));
    }
    t.push(Expect::w(
        format!("{lp}.mlp.shared_expert.down_proj.weight"),
        vec![hidden, m.shared_intermediate],
    ));
    t.push(Expect::w(
        format!("{lp}.mlp.shared_expert_gate.weight"),
        vec![1, hidden],
    ));
}

pub fn plan(cfg: &Qwen4ExpConfig) -> Plan {
    let p = TEXT_PREFIX;
    let hidden = cfg.hidden;
    let hcw = cfg.hc_hidden();
    let mut t: Vec<Expect> = Vec::new();

    t.push(Expect::w(
        format!("{p}.embed_tokens.weight"),
        vec![cfg.vocab, hidden],
    ));
    if !cfg.tie_word_embeddings {
        // The head is untied in this family, so it is its own tensor and does NOT
        // live under the text prefix.
        t.push(Expect::w("lm_head.weight".into(), vec![cfg.vocab, hidden]));
    }
    // The model-level stream mixer. Its `hc_norm` is the ONLY pre-head norm — this
    // family has no `model.norm`.
    t.push(Expect::w(
        format!("{p}.hyper_connection_mixer.hc_norm.weight"),
        vec![hcw],
    ));
    t.push(Expect::w(
        format!("{p}.hyper_connection_mixer.input_mix_weight_down.weight"),
        vec![cfg.gated_residual.lowrank, hcw],
    ));
    t.push(Expect::w(
        format!("{p}.hyper_connection_mixer.input_mix_weight_up.weight"),
        vec![hcw, cfg.gated_residual.lowrank],
    ));

    for (l, kind) in cfg.layer_types.iter().enumerate() {
        let lp = format!("{p}.layers.{l}");

        // Two gated-residual blocks per layer, one guarding each sub-block.
        for which in ["attn_hyper_connection", "mlp_hyper_connection"] {
            t.push(Expect::w(format!("{lp}.{which}.hc_norm.weight"), vec![hcw]));
            t.push(Expect::w(
                format!("{lp}.{which}.input_mix_weight_down.weight"),
                vec![cfg.gated_residual.lowrank, hcw],
            ));
            t.push(Expect::w(
                format!("{lp}.{which}.input_mix_weight_up.weight"),
                vec![hcw, cfg.gated_residual.lowrank],
            ));
            t.push(Expect::w(
                format!("{lp}.{which}.block_inject_weight.weight"),
                vec![cfg.gated_residual.count, hcw],
            ));
        }

        match kind {
            LayerType::LinearAttention => {
                let d = &cfg.deltanet;
                t.push(Expect::w(
                    format!("{lp}.linear_attn.in_proj_qkv.weight"),
                    vec![d.qkv_dim(), hidden],
                ));
                t.push(Expect::w(
                    format!("{lp}.linear_attn.in_proj_z.weight"),
                    vec![d.z_dim(), hidden],
                ));
                for ab in ["in_proj_a", "in_proj_b"] {
                    t.push(Expect::w(
                        format!("{lp}.linear_attn.{ab}.weight"),
                        vec![d.value_heads, hidden],
                    ));
                }
                t.push(Expect::w(
                    format!("{lp}.linear_attn.conv1d.weight"),
                    vec![d.qkv_dim(), 1, d.conv_kernel],
                ));
                t.push(Expect::w(
                    format!("{lp}.linear_attn.A_log"),
                    vec![d.value_heads],
                ));
                t.push(Expect::w(
                    format!("{lp}.linear_attn.dt_bias"),
                    vec![d.value_heads],
                ));
                // Per-head RMSNorm over the VALUE head dim, not the model hidden.
                t.push(Expect::w(
                    format!("{lp}.linear_attn.norm.weight"),
                    vec![d.value_head_dim],
                ));
                t.push(Expect::w(
                    format!("{lp}.linear_attn.out_proj.weight"),
                    vec![hidden, d.z_dim()],
                ));
            }
            LayerType::SparseAttention => {
                push_sparse_attention(&mut t, cfg, &lp);
            }
        }

        // MoE on every layer.
        push_moe(&mut t, cfg, &lp);

        // The n-gram block rides exactly one layer.
        if let Some(n) = cfg.ngram.as_ref().filter(|n| n.layer_idx == l) {
            t.push(Expect::w(
                format!("{lp}.ple.conv1d.weight"),
                vec![hcw, 1, n.conv_kernel],
            ));
            // Keys are scored against the WIDENED residual, so key_proj is hc-wide
            // while value_proj is hidden-wide — an asymmetry worth pinning.
            t.push(Expect::w(
                format!("{lp}.ple.key_proj.weight"),
                vec![hcw, n.embed_dim],
            ));
            t.push(Expect::w(
                format!("{lp}.ple.value_proj.weight"),
                vec![hidden, n.embed_dim],
            ));
            for nn in ["norm_query", "norm_key", "norm_conv"] {
                t.push(Expect::w(format!("{lp}.ple.{nn}.weight"), vec![hcw]));
            }
            let (_, _, padded) =
                crate::ngram_head_layout(n.vocab_size_base, n.heads(), n.divisible_by);
            let rows = padded as usize / n.shards;
            for s in 0..n.shards {
                t.push(Expect::w(
                    format!("{lp}.ple.ple_embedding.ngram_embedding.shard_{s}.weight"),
                    vec![rows, n.head_dim()],
                ));
            }
            // Integer addressing buffers — derivable, so absence is not an error.
            t.push(Expect::derived(
                format!("{lp}.ple.ple_embedding.ngram_heads_offsets"),
                vec![n.heads()],
            ));
            t.push(Expect::derived(
                format!("{lp}.ple.ple_embedding.ngram_heads_vocab_sizes"),
                vec![n.heads()],
            ));
            t.push(Expect::derived(
                format!("{lp}.ple.ple_embedding.layer_multipliers"),
                vec![n.ngram_size],
            ));
        }
    }

    // ── the embedded MTP head ───────────────────────────────────────────────
    //
    // One extra decoder layer, structurally identical to a trunk sparse-attention
    // layer (its own hyper-connections, MoE, indexer, and stream mixer), fed by two
    // projections: the next token's EMBEDDING and the trunk's final hidden state,
    // each pre-normalised.
    //
    // Note the asymmetry, which is what pins the widths: `pre_fc_norm_embedding` is
    // `[hidden]` (the embedding is narrow) while `pre_fc_norm_hidden` is
    // `[hc_count * hidden]` (the trunk hidden state is the WIDE residual). Both
    // `fc_*` matrices are `[hidden, hidden]`.
    //
    // `mtp_use_dedicated_embeddings` is false in the shipped model, so the head
    // shares the trunk's embedding table and `lm_head` — there are no embed or head
    // tensors here.
    for l in 0..cfg.mtp_layers {
        let lp = format!("mtp.layers.{l}");
        for which in ["attn_hyper_connection", "mlp_hyper_connection"] {
            t.push(Expect::w(format!("{lp}.{which}.hc_norm.weight"), vec![hcw]));
            t.push(Expect::w(
                format!("{lp}.{which}.input_mix_weight_down.weight"),
                vec![cfg.gated_residual.lowrank, hcw],
            ));
            t.push(Expect::w(
                format!("{lp}.{which}.input_mix_weight_up.weight"),
                vec![hcw, cfg.gated_residual.lowrank],
            ));
            t.push(Expect::w(
                format!("{lp}.{which}.block_inject_weight.weight"),
                vec![cfg.gated_residual.count, hcw],
            ));
        }
        push_sparse_attention(&mut t, cfg, &lp);
        push_moe(&mut t, cfg, &lp);
    }
    if cfg.mtp_layers > 0 {
        t.push(Expect::w(
            "mtp.pre_fc_norm_embedding.weight".into(),
            vec![hidden],
        ));
        t.push(Expect::w("mtp.pre_fc_norm_hidden.weight".into(), vec![hcw]));
        t.push(Expect::w(
            "mtp.fc_embedding.weight".into(),
            vec![hidden, hidden],
        ));
        t.push(Expect::w(
            "mtp.fc_hidden.weight".into(),
            vec![hidden, hidden],
        ));
        t.push(Expect::w(
            "mtp.hyper_connection_mixer.hc_norm.weight".into(),
            vec![hcw],
        ));
        t.push(Expect::w(
            "mtp.hyper_connection_mixer.input_mix_weight_down.weight".into(),
            vec![cfg.gated_residual.lowrank, hcw],
        ));
        t.push(Expect::w(
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight".into(),
            vec![hcw, cfg.gated_residual.lowrank],
        ));
    }

    // ── vision tower ────────────────────────────────────────────────────────
    //
    // A conventional pre-norm ViT: LayerNorm (weight AND bias, unlike the text
    // trunk's RMSNorm) and a bias on every projection. `patch_embed.proj` is a
    // Conv3d, so its weight keeps the 5-D kernel shape on disk even though a
    // stride-equals-kernel conv is a per-patch linear.
    if let Some(v) = cfg.vision.as_ref() {
        let vp = "model.visual";
        t.push(Expect::w(
            format!("{vp}.patch_embed.proj.weight"),
            vec![
                v.hidden,
                v.in_channels,
                v.temporal_patch_size,
                v.patch_size,
                v.patch_size,
            ],
        ));
        t.push(Expect::w(
            format!("{vp}.patch_embed.proj.bias"),
            vec![v.hidden],
        ));
        t.push(Expect::w(
            format!("{vp}.pos_embed.weight"),
            vec![v.num_position_embeddings, v.hidden],
        ));
        for b in 0..v.depth {
            let bp = format!("{vp}.blocks.{b}");
            for n in ["norm1", "norm2"] {
                t.push(Expect::w(format!("{bp}.{n}.weight"), vec![v.hidden]));
                t.push(Expect::w(format!("{bp}.{n}.bias"), vec![v.hidden]));
            }
            t.push(Expect::w(
                format!("{bp}.attn.qkv.weight"),
                vec![3 * v.hidden, v.hidden],
            ));
            t.push(Expect::w(format!("{bp}.attn.qkv.bias"), vec![3 * v.hidden]));
            t.push(Expect::w(
                format!("{bp}.attn.proj.weight"),
                vec![v.hidden, v.hidden],
            ));
            t.push(Expect::w(format!("{bp}.attn.proj.bias"), vec![v.hidden]));
            t.push(Expect::w(
                format!("{bp}.mlp.linear_fc1.weight"),
                vec![v.intermediate, v.hidden],
            ));
            t.push(Expect::w(
                format!("{bp}.mlp.linear_fc1.bias"),
                vec![v.intermediate],
            ));
            t.push(Expect::w(
                format!("{bp}.mlp.linear_fc2.weight"),
                vec![v.hidden, v.intermediate],
            ));
            t.push(Expect::w(
                format!("{bp}.mlp.linear_fc2.bias"),
                vec![v.hidden],
            ));
        }
        // The merger normalises at the UNMERGED width, then folds merge^2 patches.
        let wide = v.hidden * v.merge_unit();
        t.push(Expect::w(
            format!("{vp}.merger.norm.weight"),
            vec![v.hidden],
        ));
        t.push(Expect::w(format!("{vp}.merger.norm.bias"), vec![v.hidden]));
        t.push(Expect::w(
            format!("{vp}.merger.linear_fc1.weight"),
            vec![wide, wide],
        ));
        t.push(Expect::w(
            format!("{vp}.merger.linear_fc1.bias"),
            vec![wide],
        ));
        t.push(Expect::w(
            format!("{vp}.merger.linear_fc2.weight"),
            vec![v.out_hidden, wide],
        ));
        t.push(Expect::w(
            format!("{vp}.merger.linear_fc2.bias"),
            vec![v.out_hidden],
        ));
    }

    Plan { tensors: t }
}
