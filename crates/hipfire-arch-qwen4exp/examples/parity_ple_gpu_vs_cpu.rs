// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU PLE block vs the CPU reference, streamed over many tokens.
//!
//! Streaming matters here: the dilated conv carries a 9-deep state, and with
//! `dilation = 3` only three of those slots are ever read. A state-advance bug that
//! shifts by the wrong stride is invisible until enough tokens have passed for the
//! wrong tap to hold real data, so a short run would pass.

use hipfire_arch_qwen4exp::config::Qwen4ExpConfig;
use hipfire_arch_qwen4exp::ple::PleLayer;
use hipfire_arch_qwen4exp::ple_gpu::{ple_step, PleScratch, PleWeights};
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
            "head_dim": 16, "layer_types": ["linear_attention", "linear_attention",
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
            "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 16,
            "indexer_budget": 8, "indexer_compress_ratio": 4,
            "output_gate_type": "sigmoid",
            "rms_norm_eps": 1e-6, "max_position_embeddings": 256, "eos_token_id": 2,
        }
    }))
    .expect("config");
    let n = cfg.ngram.clone().expect("ngram config");
    let (hidden, hc, ed) = (cfg.hidden, cfg.gated_residual.count, n.embed_dim);
    let width = hc * hidden;

    let (kp, vp) = (seeded(width * ed, 3), seeded(hidden * ed, 5));
    let (nk, nq, nc) = (seeded(width, 7), seeded(width, 9), seeded(width, 11));
    let cw = seeded(width * n.conv_kernel, 13);

    let cpu = PleLayer {
        key_proj: &kp,
        value_proj: &vp,
        norm_key: &nk,
        norm_query: &nq,
        norm_conv: &nc,
        conv_weight: &cw,
        hc_count: hc,
        hidden,
        embed_dim: ed,
        kernel: n.conv_kernel,
        dilation: n.ngram_size,
        eps: cfg.rms_norm_eps,
    };

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("parity_ple_gpu_vs_cpu: no GPU ({e}) — skipped");
            return;
        }
    };
    let w = PleWeights {
        key_proj: gpu.upload_f32(&kp, &[width, ed]).unwrap(),
        value_proj: gpu.upload_f32(&vp, &[hidden, ed]).unwrap(),
        norm_key: gpu.upload_f32(&nk, &[width]).unwrap(),
        norm_query: gpu.upload_f32(&nq, &[width]).unwrap(),
        norm_conv: gpu.upload_f32(&nc, &[width]).unwrap(),
        conv_weight: gpu.upload_f32(&cw, &[width, n.conv_kernel]).unwrap(),
    };
    let mut gs = PleScratch::new(&mut gpu, &cfg).unwrap();
    let mut cs = vec![0.0f32; cpu.width() * cpu.state_len()];

    // Long enough that the deepest dilated tap (t-9) is reading real history.
    let n_steps = 20;
    let (mut worst, mut worst_t) = (0.0f32, 0usize);
    for t in 0..n_steps {
        let hw = seeded(width, 100 + t as u32);
        let emb = seeded(ed, 500 + t as u32);
        let want = cpu.step(&hw, &emb, &mut cs);

        let ghw = gpu.upload_f32(&hw, &[width]).unwrap();
        let gemb = gpu.upload_f32(&emb, &[ed]).unwrap();
        let gout = gpu.zeros(&[width], DType::F32).unwrap();
        ple_step(&mut gpu, &cfg, &w, &mut gs, &ghw, &gemb, &gout).unwrap();
        let got = gpu.download_f32(&gout).unwrap();

        let d = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if d > worst {
            worst = d;
            worst_t = t;
        }
    }

    let state = gpu.download_f32(&gs.conv_state).unwrap();
    let live = state.iter().any(|v| v.abs() > 1e-6);
    let tol = 2e-5;
    let ok = worst <= tol && live;
    println!(
        "parity_ple_gpu_vs_cpu: {n_steps} steps, worst max|Δ| = {worst:.3e} at t={worst_t} \
         (tol {tol:.0e}), conv state live: {live} -> {}",
        if ok { "OK" } else { "FAILED" }
    );
    if !ok {
        std::process::exit(1);
    }
}
