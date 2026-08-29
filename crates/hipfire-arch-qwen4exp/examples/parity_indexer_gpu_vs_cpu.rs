// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU QSA indexer vs the CPU reference — the selected token set, exactly.
//!
//! The selection is a set of booleans, so there is no tolerance to hide in: the
//! GPU either picks the same blocks or it does not.
//!
//! The pipeline under test is pool -> norm -> rotate -> score -> top-k ->
//! slot list. The rotate step is why `qsa_block_score` could not be used as
//! written: it pools and scores in one launch, leaving nowhere to put the
//! normalisation and the rotation the reference applies in between.
//!
//! Sequence lengths straddle `dense_below` so both regimes are exercised: at or
//! under the budget the selection must be everything, past it it must exclude.

use hipfire_arch_qwen4exp::attn::Indexer;
use hipfire_arch_qwen4exp::config::Qwen4ExpConfig;
use hipfire_arch_qwen4exp::rope::{cos_sin, inv_freq};
use hipfire_rdna::{DType, Gpu};

fn seeded(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

fn main() {
    let cfg = Qwen4ExpConfig::from_json(&serde_json::json!({
        "text_config": {
            "vocab_size": 128, "hidden_size": 128, "intermediate_size": 64,
            "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 64, "layer_types": ["linear_attention", "linear_attention",
                "linear_attention", "full_attention"],
            "linear_num_key_heads": 2, "linear_num_value_heads": 6,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "hc_count": 4, "hc_lowrank": 16,
            "ple_layer_ids": [2], "ple_embed_dim": 128, "ple_conv_kernel_size": 4,
            "ngram_size": 3, "heads_per_ngram": 2,
            "ngram_vocab_size_base": 2000, "make_ngram_vocab_size_divisible_by": 8,
            "split_ngram_parts": 128, "seed": 1234,
            "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 64,
            "indexer_budget": 8, "indexer_compress_ratio": 4,
            "output_gate_type": "sigmoid", "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6, "max_position_embeddings": 512, "eos_token_id": 2,
        }
    }))
    .expect("config");
    let ix = cfg.indexer.clone();
    let (hidden, ihd, inh) = (cfg.hidden, ix.head_dim, ix.n_heads);

    let qk = seeded((inh + ix.kv_heads) * ihd * hidden, 3);
    let qn = seeded(ihd, 5);
    let kn = seeded(ihd, 7);
    let cpu = Indexer {
        qk_proj: &qk,
        q_norm: &qn,
        k_norm: &kn,
        hidden,
        n_heads: inh,
        kv_heads: ix.kv_heads,
        head_dim: ihd,
        budget: ix.budget,
        compress_ratio: ix.compress_ratio,
        eps: cfg.rms_norm_eps,
    };

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("parity_indexer_gpu_vs_cpu: no GPU ({e}) — skipped");
            return;
        }
    };
    let g_qk = gpu
        .upload_f32(&qk, &[(inh + ix.kv_heads) * ihd, hidden])
        .unwrap();
    let g_qn = gpu.upload_f32(&qn, &[ihd]).unwrap();
    let g_kn = gpu.upload_f32(&kn, &[ihd]).unwrap();
    let ifreq = inv_freq(ihd, cfg.rope_theta);

    let dense_below = ix.budget + ix.compress_ratio - 1;
    let mut all_ok = true;
    for &n_tok in &[8usize, dense_below + 1, 16, 32, 64] {
        let hs = seeded(n_tok * hidden, 100 + n_tok as u32);
        let (cos, sin) = cos_sin(&(0..n_tok).collect::<Vec<_>>(), &ifreq);
        let causal: Vec<bool> = (0..n_tok)
            .flat_map(|i| (0..n_tok).map(move |j| j <= i))
            .collect();
        let want = cpu.select_mask(&hs, n_tok, &cos, &sin, &causal);

        // Per-position projections, then per-query selection. Only the last query
        // is checked on GPU: that is the decode shape, and it is the query with
        // the most blocks to choose between.
        let t = n_tok - 1;
        let mut raw_keys = Vec::with_capacity(n_tok * ihd);
        for p in 0..n_tok {
            let g_h = gpu
                .upload_f32(&hs[p * hidden..(p + 1) * hidden], &[hidden])
                .unwrap();
            let g_qkv = gpu.zeros(&[(inh + ix.kv_heads) * ihd], DType::F32).unwrap();
            gpu.gemv_f32(&g_qk, &g_h, &g_qkv).unwrap();
            let all = gpu.download_f32(&g_qkv).unwrap();
            raw_keys.extend_from_slice(&all[inh * ihd..]);
            if p == t {
                // Query: per-head norm, then rotate at this position.
                let g_q = gpu.upload_f32(&all[..inh * ihd], &[inh, ihd]).unwrap();
                let g_qn_out = gpu.zeros(&[inh * ihd], DType::F32).unwrap();
                gpu.rms_norm_heads_shared_w(
                    &g_q,
                    &g_qn,
                    &g_qn_out,
                    ihd as i32,
                    inh as i32,
                    cfg.rms_norm_eps,
                )
                .unwrap();
                let pos = gpu.upload_f32(&[f32::from_bits(p as u32)], &[1]).unwrap();
                gpu.rope_partial_interleaved_f32_batched(
                    &g_qn_out,
                    &g_qn_out,
                    &pos,
                    inh,
                    0,
                    ihd,
                    ihd,
                    ihd,
                    cfg.rope_theta,
                    1,
                    0,
                )
                .unwrap();
                // Block keys: pool + norm, rotate at each block's first position.
                let visible: Vec<usize> = (0..=t).collect();
                let n_blocks = visible.len() / ix.compress_ratio;
                let vis_f: Vec<f32> = visible.iter().map(|v| f32::from_bits(*v as u32)).collect();
                let g_vis = gpu.upload_f32(&vis_f, &[visible.len()]).unwrap();
                let g_keys = gpu.upload_f32(&raw_keys, &[n_tok, ihd]).unwrap();
                let g_bk = gpu.zeros(&[n_blocks.max(1) * ihd], DType::F32).unwrap();
                let g_st = gpu.zeros(&[n_blocks.max(1)], DType::F32).unwrap();
                gpu.qsa_pool_norm_blocks(
                    &g_keys,
                    &g_vis,
                    &g_kn,
                    &g_bk,
                    &g_st,
                    n_blocks as i32,
                    ix.compress_ratio as i32,
                    ihd as i32,
                    cfg.rms_norm_eps,
                )
                .unwrap();
                if n_blocks > 0 {
                    gpu.rope_partial_interleaved_f32_batched(
                        &g_bk,
                        &g_bk,
                        &g_st,
                        1,
                        0,
                        ihd,
                        ihd,
                        ihd,
                        cfg.rope_theta,
                        n_blocks,
                        0,
                    )
                    .unwrap();
                }
                let g_sc = gpu.zeros(&[n_blocks.max(1)], DType::F32).unwrap();
                gpu.qsa_score_prepared(
                    &g_qn_out,
                    &g_bk,
                    &g_sc,
                    inh as i32,
                    ihd as i32,
                    n_blocks as i32,
                )
                .unwrap();
                let g_bm = gpu.zeros(&[n_blocks.max(1)], DType::F32).unwrap();
                if n_blocks > 0 {
                    gpu.qsa_topk_mask(
                        &g_sc,
                        &g_bm,
                        n_blocks as i32,
                        (ix.budget / ix.compress_ratio).min(n_blocks) as i32,
                    )
                    .unwrap();
                }
                let g_idx = gpu.zeros(&[visible.len()], DType::F32).unwrap();
                let g_cnt = gpu.zeros(&[1], DType::F32).unwrap();
                gpu.qsa_select_indices(
                    &g_bm,
                    &g_vis,
                    &g_idx,
                    &g_cnt,
                    visible.len() as i32,
                    n_blocks as i32,
                    ix.compress_ratio as i32,
                )
                .unwrap();
                let n_sel = gpu.download_f32(&g_cnt).unwrap()[0].to_bits() as usize;
                let got: Vec<usize> = gpu.download_f32(&g_idx).unwrap()[..n_sel]
                    .iter()
                    .map(|v| v.to_bits() as usize)
                    .collect();
                let expect: Vec<usize> = (0..n_tok).filter(|&j| want[t * n_tok + j]).collect();
                let ok = got == expect;
                all_ok &= ok;
                let dense = n_tok <= dense_below;
                println!(
                    "  n={n_tok:<3} query {t:<3} kept {:<3} of {:<3} dense={dense:<5} {}",
                    got.len(),
                    t + 1,
                    if ok { "ok" } else { "FAIL" }
                );
                if !ok {
                    println!("    got  {got:?}\n    want {expect:?}");
                }
            }
        }
    }
    println!(
        "parity_indexer_gpu_vs_cpu: {}",
        if all_ok { "OK" } else { "FAILED" }
    );
    if !all_ok {
        std::process::exit(1);
    }
}
