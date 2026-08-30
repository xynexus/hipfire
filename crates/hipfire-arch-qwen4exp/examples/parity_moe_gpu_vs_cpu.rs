// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU MoE block vs the CPU reference, at k=8 AND k=10.
//!
//! k=10 is the point: the router kernel used to bake `#define K_TOP 8`, which
//! silently dropped two of this family's ten routed experts per token — a
//! quality loss with no error anywhere. k=8 is kept as the regression arm, since
//! that is what every other family in the tree routes.
//!
//! The selected expert SET is checked as well as the output. Two different expert
//! sets can produce similar-looking activations, so an output-only comparison can
//! pass while the routing is wrong.

use hipfire_arch_qwen4exp::config::Qwen4ExpConfig;
use hipfire_arch_qwen4exp::moe::{Expert, MoeLayer};
use hipfire_arch_qwen4exp::moe_gpu::{moe_forward, MoeScratch, MoeWeights};
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

fn cfg_with_k(k: usize, n_exp: usize) -> Qwen4ExpConfig {
    Qwen4ExpConfig::from_json(&serde_json::json!({
        "text_config": {
            "vocab_size": 128, "hidden_size": 128, "intermediate_size": 64,
            "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 128, "layer_types": ["linear_attention", "linear_attention",
                "linear_attention", "full_attention"],
            "linear_num_key_heads": 2, "linear_num_value_heads": 6,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": n_exp, "num_experts_per_tok": k, "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32, "norm_topk_prob": true,
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
    .expect("config")
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("parity_moe_gpu_vs_cpu: no GPU ({e}) — skipped");
            return;
        }
    };
    let mut all_ok = true;
    let mut worst_all = 0.0f32;

    // k=8 is the shape every other family routes; k=10 is this one's.
    for &k in &[8usize, 10] {
        let n_exp = 16;
        let cfg = cfg_with_k(k, n_exp);
        let (hidden, mi, smi) = (
            cfg.hidden,
            cfg.moe.intermediate,
            cfg.moe.shared_intermediate,
        );
        let (gu_sz, dn_sz) = (2 * mi * hidden, hidden * mi);

        let router = seeded(n_exp * hidden, 3);
        let gu = seeded(n_exp * gu_sz, 5);
        let dn = seeded(n_exp * dn_sz, 7);
        let (sg, su, sd) = (
            seeded(smi * hidden, 9),
            seeded(smi * hidden, 11),
            seeded(hidden * smi, 13),
        );
        let seg = seeded(hidden, 15);

        let cpu = MoeLayer {
            router: &router,
            experts: (0..n_exp)
                .map(|e| Expert {
                    gate_up: &gu[e * gu_sz..(e + 1) * gu_sz],
                    down: &dn[e * dn_sz..(e + 1) * dn_sz],
                })
                .collect(),
            shared_gate: &sg,
            shared_up: &su,
            shared_down: &sd,
            shared_expert_gate: &seg,
            hidden,
            mi,
            shared_mi: smi,
            top_k: k,
            norm_topk_prob: cfg.moe.norm_topk_prob,
        };

        let w = MoeWeights {
            router: gpu.upload_f32(&router, &[n_exp, hidden]).unwrap(),
            gate_up: hipfire_arch_qwen4exp::moe_gpu::ExpertStack {
                buf: gpu.upload_f32(&gu, &[n_exp, 2 * mi, hidden]).unwrap(),
                dtype: hipfire_rdna::DType::F32,
                rows: 2 * mi,
                cols: hidden,
                stride: 2 * mi * hidden,
            },
            down: hipfire_arch_qwen4exp::moe_gpu::ExpertStack {
                buf: gpu.upload_f32(&dn, &[n_exp, hidden, mi]).unwrap(),
                dtype: hipfire_rdna::DType::F32,
                rows: hidden,
                cols: mi,
                stride: hidden * mi,
            },
            shared_gate: gpu.upload_f32(&sg, &[smi, hidden]).unwrap(),
            shared_up: gpu.upload_f32(&su, &[smi, hidden]).unwrap(),
            shared_down: gpu.upload_f32(&sd, &[hidden, smi]).unwrap(),
            shared_expert_gate: gpu.upload_f32(&seg, &[1, hidden]).unwrap(),
        };
        let mut s = MoeScratch::new(&mut gpu, &cfg).unwrap();

        let mut worst = 0.0f32;
        let mut mag = 0.0f32;
        let mut routing_ok = true;
        for t in 0..12u32 {
            let x = seeded(hidden, 200 + t);
            let want = cpu.forward(&x);
            let want_r = cpu.route(&x);

            let gx = gpu.upload_f32(&x, &[hidden]).unwrap();
            let gout = gpu.zeros(&[hidden], DType::F32).unwrap();
            moe_forward(&mut gpu, &cfg, &w, &mut s, &gx, &gout).unwrap();
            let got = gpu.download_f32(&gout).unwrap();
            worst = worst.max(
                got.iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max),
            );
            mag = mag.max(want.iter().map(|v| v.abs()).fold(0.0f32, f32::max));

            // The expert SET must match, not just the output.
            let idx = gpu.download_f32(&s.topk_idx_view()).unwrap();
            let mut got_set: Vec<usize> = idx.iter().map(|v| v.to_bits() as usize).collect();
            let mut want_set = want_r.experts.clone();
            got_set.sort_unstable();
            want_set.sort_unstable();
            if got_set != want_set {
                routing_ok = false;
                println!("    k={k} t={t}: routed {got_set:?}, want {want_set:?}");
            }
        }
        worst_all = worst_all.max(worst);
        // 2e-4 matches the other composed parities in this crate. The block sums
        // k expert outputs, each from a two-GEMV chain, so CPU and GPU differ in
        // reduction order; what must be EXACT is the routing, and that is asserted
        // separately. The magnitude is printed so the absolute number is readable.
        let ok = worst <= 2e-4 && routing_ok;
        all_ok &= ok;
        println!(
            "  k={k:<3} of {n_exp} experts: worst max|Δ| = {worst:.3e} (mag {mag:.3}), \
             routing exact: {routing_ok} -> {}",
            if ok { "ok" } else { "FAIL" }
        );
    }

    println!(
        "parity_moe_gpu_vs_cpu: worst {worst_all:.3e} -> {}",
        if all_ok { "OK" } else { "FAILED" }
    );
    if !all_ok {
        std::process::exit(1);
    }
}
