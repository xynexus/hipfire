// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! THE GPU TRUNK, END TO END: tokens in, logits out, vs the CPU trunk.
//!
//! The CPU trunk is itself differenced against the pinned upstream implementation
//! (`tests/reference_oracle.rs`), so agreement here chains all the way back to the
//! reference: upstream -> CPU -> GPU, for the whole model rather than a block.
//!
//! The GPU side runs DECODE — one token at a time, every layer's state carried
//! across steps — while the CPU side runs the whole sequence at once. Matching
//! therefore also says the streamed form reproduces the batched one.
//!
//! The argmax is checked as well as the values: that is what generation consumes,
//! and it is the thing a small numeric drift can still flip.

use hipfire_arch_qwen4exp::config::Qwen4ExpConfig;

use hipfire_arch_qwen4exp::trunk::{forward, WeightSource};
use hipfire_arch_qwen4exp::trunk_gpu::{
    decode_step, TensorReader, TrunkScratch, TrunkState, TrunkWeights,
};
use hipfire_rdna::Gpu;
use std::collections::HashMap;

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

struct Src(HashMap<String, Vec<f32>>);
impl TensorReader for Src {
    fn read(&self, name: &str) -> Result<Vec<f32>, String> {
        self.0
            .get(name)
            .cloned()
            .ok_or_else(|| format!("missing weight `{name}`"))
    }
}
impl WeightSource for Src {
    fn get(&self, name: &str) -> &[f32] {
        self.0
            .get(name)
            .unwrap_or_else(|| panic!("missing weight `{name}`"))
    }
}

fn main() {
    let cfg = Qwen4ExpConfig::from_json(&serde_json::json!({
        "text_config": {
            "vocab_size": 64, "hidden_size": 128, "intermediate_size": 64,
            "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 128, "layer_types": ["linear_attention", "linear_attention",
                "linear_attention", "full_attention"],
            "linear_num_key_heads": 2, "linear_num_value_heads": 6,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32, "norm_topk_prob": true,
            "hc_count": 4, "hc_lowrank": 16,
            "ple_layer_ids": [2], "ple_embed_dim": 128, "ple_conv_kernel_size": 4,
            "ngram_size": 3, "heads_per_ngram": 2,
            "ngram_vocab_size_base": 2000, "make_ngram_vocab_size_divisible_by": 8,
            "split_ngram_parts": 128, "seed": 1234,
            "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 128,
            "indexer_budget": 8, "indexer_compress_ratio": 4,
            "output_gate_type": "sigmoid", "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6, "max_position_embeddings": 256, "eos_token_id": 2,
        }
    }))
    .expect("config");

    let (hidden, hc, vocab) = (cfg.hidden, cfg.gated_residual.count, cfg.vocab);
    let width = hc * hidden;
    let lr = cfg.gated_residual.lowrank;
    let m = &cfg.moe;
    let d = &cfg.deltanet;
    let ix = &cfg.indexer;
    let n = cfg.ngram.clone().expect("ngram");
    let p = "model.language_model";

    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    // Seed from the NAME, so every tensor differs and there is no shared counter
    // whose ordering could silently change what the test covers.
    fn name_seed(name: &str) -> u32 {
        name.bytes().fold(2_166_136_261u32, |h, b| {
            (h ^ b as u32).wrapping_mul(16_777_619)
        })
    }
    let put = |w: &mut HashMap<String, Vec<f32>>, name: String, len: usize| {
        let sd = name_seed(&name);
        w.insert(name, seeded(len, sd));
    };
    put(&mut w, format!("{p}.embed_tokens.weight"), vocab * hidden);
    put(&mut w, "lm_head.weight".into(), vocab * hidden);
    for which in ["hyper_connection_mixer"] {
        put(&mut w, format!("{p}.{which}.hc_norm.weight"), width);
        put(
            &mut w,
            format!("{p}.{which}.input_mix_weight_down.weight"),
            lr * width,
        );
        put(
            &mut w,
            format!("{p}.{which}.input_mix_weight_up.weight"),
            width * lr,
        );
    }
    for l in 0..cfg.layers {
        let lp = format!("{p}.layers.{l}");
        for which in ["attn_hyper_connection", "mlp_hyper_connection"] {
            put(&mut w, format!("{lp}.{which}.hc_norm.weight"), width);
            put(
                &mut w,
                format!("{lp}.{which}.input_mix_weight_down.weight"),
                lr * width,
            );
            put(
                &mut w,
                format!("{lp}.{which}.input_mix_weight_up.weight"),
                width * lr,
            );
            put(
                &mut w,
                format!("{lp}.{which}.block_inject_weight.weight"),
                hc * width,
            );
        }
        if l == 3 {
            let sa = format!("{lp}.self_attn");
            put(
                &mut w,
                format!("{sa}.q_proj.weight"),
                cfg.n_heads * cfg.head_dim * 2 * hidden,
            );
            put(
                &mut w,
                format!("{sa}.k_proj.weight"),
                cfg.n_kv_heads * cfg.head_dim * hidden,
            );
            put(
                &mut w,
                format!("{sa}.v_proj.weight"),
                cfg.n_kv_heads * cfg.head_dim * hidden,
            );
            put(
                &mut w,
                format!("{sa}.o_proj.weight"),
                hidden * cfg.n_heads * cfg.head_dim,
            );
            put(&mut w, format!("{sa}.q_norm.weight"), cfg.head_dim);
            put(&mut w, format!("{sa}.k_norm.weight"), cfg.head_dim);
            put(
                &mut w,
                format!("{sa}.indexer.index_qk_proj.weight"),
                (ix.n_heads + ix.kv_heads) * ix.head_dim * hidden,
            );
            put(
                &mut w,
                format!("{sa}.indexer.q_layernorm.weight"),
                ix.head_dim,
            );
            put(
                &mut w,
                format!("{sa}.indexer.k_layernorm.weight"),
                ix.head_dim,
            );
        } else {
            let la = format!("{lp}.linear_attn");
            put(
                &mut w,
                format!("{la}.in_proj_qkv.weight"),
                d.qkv_dim() * hidden,
            );
            put(&mut w, format!("{la}.in_proj_z.weight"), d.z_dim() * hidden);
            put(
                &mut w,
                format!("{la}.in_proj_a.weight"),
                d.value_heads * hidden,
            );
            put(
                &mut w,
                format!("{la}.in_proj_b.weight"),
                d.value_heads * hidden,
            );
            put(
                &mut w,
                format!("{la}.conv1d.weight"),
                d.qkv_dim() * d.conv_kernel,
            );
            // A_log must stay in log((0.01, 16]) or the decay is meaningless.
            let alog_name = format!("{la}.A_log");
            let alog = seeded(d.value_heads, name_seed(&alog_name))
                .iter()
                .map(|v| (v * 2.0).exp().ln())
                .collect();
            w.insert(alog_name, alog);
            put(&mut w, format!("{la}.dt_bias"), d.value_heads);
            put(&mut w, format!("{la}.norm.weight"), d.value_head_dim);
            put(&mut w, format!("{la}.out_proj.weight"), hidden * d.z_dim());
        }
        let mp = format!("{lp}.mlp");
        put(&mut w, format!("{mp}.gate.weight"), m.num_experts * hidden);
        put(
            &mut w,
            format!("{mp}.experts.gate_up_proj"),
            m.num_experts * 2 * m.intermediate * hidden,
        );
        put(
            &mut w,
            format!("{mp}.experts.down_proj"),
            m.num_experts * hidden * m.intermediate,
        );
        put(
            &mut w,
            format!("{mp}.shared_expert.gate_proj.weight"),
            m.shared_intermediate * hidden,
        );
        put(
            &mut w,
            format!("{mp}.shared_expert.up_proj.weight"),
            m.shared_intermediate * hidden,
        );
        put(
            &mut w,
            format!("{mp}.shared_expert.down_proj.weight"),
            hidden * m.shared_intermediate,
        );
        put(&mut w, format!("{mp}.shared_expert_gate.weight"), hidden);
        if l == n.layer_idx {
            let pl = format!("{lp}.ple");
            put(&mut w, format!("{pl}.key_proj.weight"), width * n.embed_dim);
            put(
                &mut w,
                format!("{pl}.value_proj.weight"),
                hidden * n.embed_dim,
            );
            put(&mut w, format!("{pl}.norm_key.weight"), width);
            put(&mut w, format!("{pl}.norm_query.weight"), width);
            put(&mut w, format!("{pl}.norm_conv.weight"), width);
            put(&mut w, format!("{pl}.conv1d.weight"), width * n.conv_kernel);
            let (_, _, padded) = hipfire_arch_qwen4exp::ngram_head_layout_at(
                n.vocab_size_base,
                n.heads(),
                n.divisible_by,
                n.ple_index,
            );
            put(
                &mut w,
                format!("{pl}.ple_embedding.ngram_embedding.weight"),
                padded as usize * n.head_dim(),
            );
        }
    }

    let eos = 2u32;
    let tokens: Vec<u32> = vec![3, 17, 42, 5, 9, 7, 61, 23, 11, 2, 8, 34, 6, 41, 19, 55];
    let src = Src(w);
    let want = forward(&cfg, &src, &tokens, eos);

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("parity_trunk_gpu_vs_cpu: no GPU ({e}) — skipped");
            return;
        }
    };
    let gw = TrunkWeights::upload(&mut gpu, &cfg, &src).unwrap();
    assert_experts_are_one_slab(&gw, &cfg);
    let mut st = TrunkState::new(&mut gpu, &cfg, 64).unwrap();
    let mut sc = TrunkScratch::new(&mut gpu, &cfg, 64).unwrap();
    let embed = src.get(&format!("{p}.embed_tokens.weight")).to_vec();
    let ngram = src
        .get(&format!(
            "{p}.layers.{}.ple.ple_embedding.ngram_embedding.weight",
            n.layer_idx
        ))
        .to_vec();

    let (mut worst, mut worst_t, mut argmax_ok) = (0.0f32, 0usize, true);
    for t in 0..tokens.len() {
        let got = decode_step(
            &mut gpu,
            &cfg,
            &gw,
            &mut st,
            &mut sc,
            &embed,
            Some(&hipfire_arch_qwen4exp::ngram_rows::ResidentRows { table: &ngram }),
            &tokens[..=t],
            t,
            eos,
        )
        .unwrap();
        let wv = &want[t * vocab..(t + 1) * vocab];
        let d = got
            .iter()
            .zip(wv)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if d > worst {
            worst = d;
            worst_t = t;
        }
        let am = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };
        if am(&got) != am(wv) {
            argmax_ok = false;
            println!("    argmax differs at t={t}: {} vs {}", am(&got), am(wv));
        }
    }

    let mag = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let tol = 1e-3;
    let ok = worst <= tol && argmax_ok;
    println!(
        "parity_trunk_gpu_vs_cpu: {} tokens, {} layers, worst max|Δ| = {worst:.3e} at t={worst_t} \
         (mag {mag:.2}, tol {tol:.0e}), argmax identical: {argmax_ok} -> {}",
        tokens.len(),
        cfg.layers,
        if ok { "OK" } else { "FAILED" }
    );
    if !ok {
        std::process::exit(1);
    }
}

/// Experts must be ONE allocation per layer, not one per expert.
///
/// Why it matters: on gfx1151 a GTT allocation above 2 MiB rounds up to a multiple
/// of 2 MiB, so the waste is a flat 2 MiB PER ALLOCATION. At the shipped quantised
/// expert size a module sits a hair over the line and pays 1.638x; grouping brings
/// that to 1.024x. Across 512 experts x 49 MoE layers it is 105 GB versus 66 GB —
/// the difference between the experts fitting resident and not. That ratio is
/// measured by `hipfire-rdna/examples/gtt_slab_vs_permodule`; what is checked here
/// is the STRUCTURAL property that makes it apply.
///
/// `stack_experts` gets this right by construction (one `upload_f32` for the whole
/// stack), so this guards against a refactor quietly reverting to per-expert
/// uploads — which would cost ~39 GB with nothing failing.
fn assert_experts_are_one_slab(w: &TrunkWeights, cfg: &Qwen4ExpConfig) {
    let m = &cfg.moe;
    for (i, l) in w.layers.iter().enumerate() {
        let want_gu = m.num_experts * 2 * m.intermediate * cfg.hidden;
        let want_dn = m.num_experts * cfg.hidden * m.intermediate;
        assert_eq!(
            l.moe.gate_up.resident_elems(),
            want_gu,
            "layer {i}: gate_up is {} elements, not the whole {}-expert stack — \
             experts are being allocated separately",
            l.moe.gate_up.resident_elems(),
            m.num_experts
        );
        assert_eq!(
            l.moe.down.resident_elems(),
            want_dn,
            "layer {i}: down is not one stack"
        );
    }
    println!(
        "  experts: 1 slab per projection per layer ({} experts each) — grouped",
        m.num_experts
    );
}
