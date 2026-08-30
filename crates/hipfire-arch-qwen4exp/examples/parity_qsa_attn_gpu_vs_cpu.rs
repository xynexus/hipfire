// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU QSA attention block (decode, with a KV cache) vs the CPU reference.
//!
//! The CPU side runs the whole sequence at once; the GPU side streams it one token
//! at a time through a cache, selecting per query. Agreement therefore also says
//! the cache write and the per-step selection reproduce the whole-sequence form —
//! which a single-token check could not.
//!
//! Sequence length is taken past `dense_below` so the selection genuinely excludes.

use hipfire_arch_qwen4exp::attn::{Indexer, QsaAttention};
use hipfire_arch_qwen4exp::attn_gpu::{qsa_decode_step, QsaCache, QsaScratch, QsaWeights};
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
    // head_dim 128: `attention_cold_slots` is compiled per head width, and 128 is
    // the variant this family uses.
    let cfg = Qwen4ExpConfig::from_json(&serde_json::json!({
        "text_config": {
            "vocab_size": 128, "hidden_size": 128, "intermediate_size": 64,
            "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 128, "layer_types": ["linear_attention", "linear_attention",
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
            "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 128,
            "indexer_budget": 8, "indexer_compress_ratio": 4,
            "output_gate_type": "sigmoid", "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6, "max_position_embeddings": 512, "eos_token_id": 2,
        }
    }))
    .expect("config");

    let (hidden, nh, nkv, hd) = (cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let ix = cfg.indexer.clone();
    let n_tok = 24;
    let max_seq = 64;

    let qp = seeded(nh * hd * 2 * hidden, 3);
    let kp = seeded(nkv * hd * hidden, 5);
    let vp = seeded(nkv * hd * hidden, 7);
    let op = seeded(hidden * nh * hd, 9);
    let qn = seeded(hd, 11);
    let kn = seeded(hd, 13);
    let iqk = seeded((ix.n_heads + ix.kv_heads) * ix.head_dim * hidden, 15);
    let iqn = seeded(ix.head_dim, 17);
    let ikn = seeded(ix.head_dim, 19);
    let hs = seeded(n_tok * hidden, 21);

    // CPU: indexer -> combined mask -> attention, whole sequence.
    let ifreq = inv_freq(cfg.rotary_dim(), cfg.rope_theta);
    let (cos, sin) = cos_sin(&(0..n_tok).collect::<Vec<_>>(), &ifreq);
    let causal: Vec<bool> = (0..n_tok)
        .flat_map(|i| (0..n_tok).map(move |j| j <= i))
        .collect();
    let idx = Indexer {
        qk_proj: &iqk,
        q_norm: &iqn,
        k_norm: &ikn,
        hidden,
        n_heads: ix.n_heads,
        kv_heads: ix.kv_heads,
        head_dim: ix.head_dim,
        budget: ix.budget,
        compress_ratio: ix.compress_ratio,
        eps: cfg.rms_norm_eps,
    };
    let sel = idx.select_mask(&hs, n_tok, &cos, &sin, &causal);
    let visible: Vec<bool> = causal.iter().zip(&sel).map(|(c, s)| *c && *s).collect();
    let attn = QsaAttention {
        q_proj: &qp,
        k_proj: &kp,
        v_proj: &vp,
        o_proj: &op,
        q_norm: &qn,
        k_norm: &kn,
        hidden,
        n_heads: nh,
        n_kv: nkv,
        head_dim: hd,
        eps: cfg.rms_norm_eps,
    };
    let want = attn.forward(&hs, n_tok, &cos, &sin, &visible);

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("parity_qsa_attn_gpu_vs_cpu: no GPU ({e}) — skipped");
            return;
        }
    };
    let w = QsaWeights {
        q_proj: gpu.upload_f32(&qp, &[nh * hd * 2, hidden]).unwrap(),
        k_proj: gpu.upload_f32(&kp, &[nkv * hd, hidden]).unwrap(),
        v_proj: gpu.upload_f32(&vp, &[nkv * hd, hidden]).unwrap(),
        o_proj: gpu.upload_f32(&op, &[hidden, nh * hd]).unwrap(),
        q_norm: gpu.upload_f32(&qn, &[hd]).unwrap(),
        k_norm: gpu.upload_f32(&kn, &[hd]).unwrap(),
        ix_qk_proj: gpu
            .upload_f32(&iqk, &[(ix.n_heads + ix.kv_heads) * ix.head_dim, hidden])
            .unwrap(),
        ix_q_norm: gpu.upload_f32(&iqn, &[ix.head_dim]).unwrap(),
        ix_k_norm: gpu.upload_f32(&ikn, &[ix.head_dim]).unwrap(),
    };
    let mut cache = QsaCache::new(&mut gpu, &cfg, max_seq).unwrap();
    let mut s = QsaScratch::new(&mut gpu, &cfg, max_seq).unwrap();

    let (mut worst, mut worst_t) = (0.0f32, 0usize);
    for t in 0..n_tok {
        let gx = gpu
            .upload_f32(&hs[t * hidden..(t + 1) * hidden], &[hidden])
            .unwrap();
        let gout = gpu.zeros(&[hidden], DType::F32).unwrap();
        let vis: Vec<usize> = (0..=t).collect();
        qsa_decode_step(&mut gpu, &cfg, &w, &mut s, &mut cache, &gx, t, &vis, &gout).unwrap();
        let got = gpu.download_f32(&gout).unwrap();
        let d = got
            .iter()
            .zip(&want[t * hidden..(t + 1) * hidden])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if d > worst {
            worst = d;
            worst_t = t;
        }
    }

    // The selection must actually have excluded something by the end, or this
    // reduces to a dense-attention test.
    let dense_below = ix.budget + ix.compress_ratio - 1;
    let excluded = n_tok > dense_below;
    let tol = 5e-4;
    let ok = worst <= tol && excluded;
    println!(
        "parity_qsa_attn_gpu_vs_cpu: {n_tok} tokens, worst max|Δ| = {worst:.3e} at t={worst_t} \
         (tol {tol:.0e}), selection excludes: {excluded} -> {}",
        if ok { "OK" } else { "FAILED" }
    );
    if !ok {
        std::process::exit(1);
    }
}
