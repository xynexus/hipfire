// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! The text trunk, composed from the verified blocks — CPU reference.
//!
//! Prefill over a whole token sequence: embed, seed the hyper-connection streams,
//! run every layer, collapse, project to logits. The token mixers are streamed
//! (Gated DeltaNet) or sequence-wide (sparse attention) as each requires.
//!
//! The composition itself carries three things worth stating, because each is a
//! plausible-wrong-reading away from silently different output:
//!
//! * The wide stream is seeded by REPEATING the embedding across all `hc_count`
//!   streams — not by zero-padding the streams after the first.
//! * PLE is ADDITIVE on the wide stream and runs BEFORE the residual read.
//! * There is no final `model.norm`. The mixer's own `hc_norm` is the last
//!   normalisation before `lm_head`.

use crate::attn::{Indexer, QsaAttention};
use crate::config::{LayerType, Qwen4ExpConfig};
use crate::gdn_cpu::GdnCpu;
use crate::hc::GatedResidual;
use crate::moe::{Expert, MoeLayer};
use crate::ngram::NgramHasher;
use crate::ple::PleLayer;

/// Somewhere to look weights up by their checkpoint name.
pub trait WeightSource {
    fn get(&self, name: &str) -> &[f32];
}

/// Logits for every position: `[n_tok, vocab]`.
pub fn forward(cfg: &Qwen4ExpConfig, w: &dyn WeightSource, tokens: &[u32], eos: u32) -> Vec<f32> {
    let (hidden, hc) = (cfg.hidden, cfg.gated_residual.count);
    let width = hc * hidden;
    let n_tok = tokens.len();
    let p = "model.language_model";

    // Embed, then seed every stream with a copy of it.
    let embed = w.get(&format!("{p}.embed_tokens.weight"));
    let mut wide: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| {
            let e = &embed[t as usize * hidden..(t as usize + 1) * hidden];
            (0..hc).flat_map(|_| e.iter().copied()).collect()
        })
        .collect();

    let ifreq = crate::rope::inv_freq(cfg.rotary_dim(), cfg.rope_theta);
    let (cos, sin) = crate::rope::cos_sin(&(0..n_tok).collect::<Vec<_>>(), &ifreq);
    let causal: Vec<bool> = (0..n_tok)
        .flat_map(|i| (0..n_tok).map(move |j| j <= i))
        .collect();

    for l in 0..cfg.layers {
        let lp = format!("{p}.layers.{l}");

        // PLE, additive on the wide stream, before the residual read.
        if let Some(n) = cfg.ngram.as_ref().filter(|n| n.layer_idx == l) {
            let pl = format!("{lp}.ple");
            let ple = PleLayer {
                key_proj: w.get(&format!("{pl}.key_proj.weight")),
                value_proj: w.get(&format!("{pl}.value_proj.weight")),
                norm_key: w.get(&format!("{pl}.norm_key.weight")),
                norm_query: w.get(&format!("{pl}.norm_query.weight")),
                norm_conv: w.get(&format!("{pl}.norm_conv.weight")),
                conv_weight: w.get(&format!("{pl}.conv1d.weight")),
                hc_count: hc,
                hidden,
                embed_dim: n.embed_dim,
                kernel: n.conv_kernel,
                dilation: n.ngram_size,
                eps: cfg.rms_norm_eps,
            };
            let table = w.get(&format!("{pl}.ple_embedding.ngram_embedding.weight"));
            let hasher = NgramHasher::from_config(n, cfg.vocab as u64, eos);
            let hd = n.head_dim();
            let ctx_len = n.ngram_size - 1;
            let mut hist: Vec<u32> = vec![eos; ctx_len];
            hist.extend_from_slice(tokens);
            let mut state = vec![0.0f32; ple.width() * ple.state_len()];
            for t in 0..n_tok {
                let preds: Vec<Option<u32>> =
                    hist[..t + ctx_len].iter().map(|&v| Some(v)).collect();
                let rows = hasher.rows(hist[t + ctx_len], &preds);
                let emb: Vec<f32> = rows
                    .iter()
                    .flat_map(|&r| table[r as usize * hd..(r as usize + 1) * hd].to_vec())
                    .collect();
                let out = ple.step(&wide[t], &emb, &mut state);
                for (v, o) in wide[t].iter_mut().zip(&out) {
                    *v += o;
                }
            }
        }

        // Two hyper-connections per layer: one around the token mixer, one around
        // the MoE. Same shape both times.
        for half in 0..2 {
            let which = if half == 0 {
                "attn_hyper_connection"
            } else {
                "mlp_hyper_connection"
            };
            let gr = GatedResidual {
                hc_norm: w.get(&format!("{lp}.{which}.hc_norm.weight")),
                mix_down: w.get(&format!("{lp}.{which}.input_mix_weight_down.weight")),
                mix_up: w.get(&format!("{lp}.{which}.input_mix_weight_up.weight")),
                block_inject: Some(w.get(&format!("{lp}.{which}.block_inject_weight.weight"))),
                hc_count: hc,
                hidden,
                lowrank: cfg.gated_residual.lowrank,
                eps: cfg.rms_norm_eps,
            };
            let reads: Vec<_> = (0..n_tok).map(|t| gr.read(&wide[t])).collect();
            let mixed: Vec<f32> = reads.iter().flat_map(|r| r.mixed_input.clone()).collect();

            let block_out: Vec<f32> = if half == 1 {
                let mp = format!("{lp}.mlp");
                let (mi, smi) = (cfg.moe.intermediate, cfg.moe.shared_intermediate);
                let (gu, dn) = (
                    w.get(&format!("{mp}.experts.gate_up_proj")),
                    w.get(&format!("{mp}.experts.down_proj")),
                );
                let (gu_sz, dn_sz) = (2 * mi * hidden, hidden * mi);
                let moe = MoeLayer {
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
                };
                (0..n_tok)
                    .flat_map(|t| moe.forward(&mixed[t * hidden..(t + 1) * hidden]))
                    .collect()
            } else if cfg.layer_types[l] == LayerType::LinearAttention {
                let la = format!("{lp}.linear_attn");
                let d = &cfg.deltanet;
                let g = GdnCpu {
                    in_proj_qkv: w.get(&format!("{la}.in_proj_qkv.weight")),
                    in_proj_z: w.get(&format!("{la}.in_proj_z.weight")),
                    in_proj_a: w.get(&format!("{la}.in_proj_a.weight")),
                    in_proj_b: w.get(&format!("{la}.in_proj_b.weight")),
                    conv_weight: w.get(&format!("{la}.conv1d.weight")),
                    a_log: w.get(&format!("{la}.A_log")),
                    dt_bias: w.get(&format!("{la}.dt_bias")),
                    norm_weight: w.get(&format!("{la}.norm.weight")),
                    out_proj: w.get(&format!("{la}.out_proj.weight")),
                    hidden,
                    n_k: d.key_heads,
                    n_v: d.value_heads,
                    head_k: d.key_head_dim,
                    head_v: d.value_head_dim,
                    kernel: d.conv_kernel,
                    gate_sigmoid: d.output_gate_sigmoid,
                    eps: cfg.rms_norm_eps,
                };
                let mut st = g.zero_state();
                (0..n_tok)
                    .flat_map(|t| g.step(&mixed[t * hidden..(t + 1) * hidden], &mut st))
                    .collect()
            } else {
                let sa = format!("{lp}.self_attn");
                let ix = &cfg.indexer;
                let indexer = Indexer {
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
                };
                let sel = indexer.select_mask(&mixed, n_tok, &cos, &sin, &causal);
                let visible: Vec<bool> = causal.iter().zip(&sel).map(|(c, s)| *c && *s).collect();
                let attn = QsaAttention {
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
                };
                attn.forward(&mixed, n_tok, &cos, &sin, &visible)
            };

            for t in 0..n_tok {
                let inj = reads[t].inject.as_ref().expect("layer residual injects");
                gr.write(&mut wide[t], &block_out[t * hidden..(t + 1) * hidden], inj);
            }
        }
    }

    // Collapse (this is also the final norm), then the untied head.
    let mixer = GatedResidual {
        hc_norm: w.get(&format!("{p}.hyper_connection_mixer.hc_norm.weight")),
        mix_down: w.get(&format!(
            "{p}.hyper_connection_mixer.input_mix_weight_down.weight"
        )),
        mix_up: w.get(&format!(
            "{p}.hyper_connection_mixer.input_mix_weight_up.weight"
        )),
        block_inject: None,
        hc_count: hc,
        hidden,
        lowrank: cfg.gated_residual.lowrank,
        eps: cfg.rms_norm_eps,
    };
    let head = w.get("lm_head.weight");
    let mut logits = vec![0.0f32; n_tok * cfg.vocab];
    for t in 0..n_tok {
        let h = mixer.read(&wide[t]).mixed_input;
        for v in 0..cfg.vocab {
            logits[t * cfg.vocab + v] = (0..hidden).map(|d| head[v * hidden + d] * h[d]).sum();
        }
    }
    let _ = width;
    logits
}
