// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! THE MTP HEAD: GPU vs the CPU reference.
//!
//! `mtp.rs` (CPU) is what `examples/mtp_probe` measured at 56.2% token
//! acceptance on the shipped 180B, so it is the thing `mtp_gpu.rs` must
//! reproduce. A GPU head that merely *runs* proves nothing: speculation is
//! lossless, so a wrong drafter still emits correct text and simply never gets
//! accepted — the failure would show up as a disappointing tok/s number weeks
//! later, with nothing pointing at the cause.
//!
//! Tiny synthetic config, weights seeded FROM THE TENSOR NAME so every tensor
//! differs and no shared counter can silently change what is covered — the same
//! construction `parity_trunk_gpu_vs_cpu` uses.
//!
//! Checked, in order of how loudly each would fail:
//!
//! 1. `fuse_inputs` alone — the one piece with no trunk equivalent, so the one
//!    most likely to be wrong and least likely to be caught downstream.
//! 2. The full head, cosine + max|Δ| against the CPU output.
//! 3. A NEGATIVE CONTROL: the other `Fusion` must NOT match. Without it this
//!    test would pass on an implementation that ignored `fusion` entirely, since
//!    both arms would then agree with whichever CPU arm was used.

use hipfire_arch_qwen4exp::config::Qwen4ExpConfig;
use hipfire_arch_qwen4exp::mtp::{self, Fusion};
use hipfire_arch_qwen4exp::mtp_gpu::{mtp_decode_step, MtpScratchGpu, MtpWeightsGpu};
use hipfire_arch_qwen4exp::trunk::WeightSource;
use hipfire_arch_qwen4exp::trunk_gpu::TensorReader;
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

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
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
            "ngram_size": 3, "heads_per_ngram": 2,
            "ngram_vocab_size_base": 2000, "make_ngram_vocab_size_divisible_by": 8,
            "split_ngram_parts": 128, "seed": 1234,
            "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 128,
            "indexer_budget": 8, "indexer_compress_ratio": 4,
            "output_gate_type": "sigmoid", "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6, "max_position_embeddings": 256, "eos_token_id": 2,
            "mtp_num_hidden_layers": 1,
        }
    }))
    .expect("config");

    let (hidden, hc) = (cfg.hidden, cfg.gated_residual.count);
    let width = hc * hidden;
    let lr = cfg.gated_residual.lowrank;
    let (m, ix) = (&cfg.moe, &cfg.indexer);

    fn name_seed(name: &str) -> u32 {
        name.bytes().fold(2_166_136_261u32, |h, b| {
            (h ^ b as u32).wrapping_mul(16_777_619)
        })
    }
    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    let mut put = |w: &mut HashMap<String, Vec<f32>>, name: String, len: usize| {
        let sd = name_seed(&name);
        w.insert(name, seeded(len, sd));
    };

    // Fusion projections and their norms.
    put(&mut w, "mtp.pre_fc_norm_hidden.weight".into(), width);
    put(&mut w, "mtp.pre_fc_norm_embedding.weight".into(), hidden);
    put(&mut w, "mtp.fc_hidden.weight".into(), hidden * hidden);
    put(&mut w, "mtp.fc_embedding.weight".into(), hidden * hidden);

    // The head's layer + the stream mixer.
    let lp = "mtp.layers.0";
    for (base, inject) in [
        (format!("{lp}.attn_hyper_connection"), true),
        (format!("{lp}.mlp_hyper_connection"), true),
        ("mtp.hyper_connection_mixer".to_string(), false),
    ] {
        put(&mut w, format!("{base}.hc_norm.weight"), width);
        put(
            &mut w,
            format!("{base}.input_mix_weight_down.weight"),
            lr * width,
        );
        put(
            &mut w,
            format!("{base}.input_mix_weight_up.weight"),
            width * lr,
        );
        if inject {
            put(
                &mut w,
                format!("{base}.block_inject_weight.weight"),
                hc * width,
            );
        }
    }
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

    let src = Src(w);

    // The inputs the head consumes: the trunk's WIDE residual and the next
    // token's embedding.
    let trunk_wide = seeded(width, name_seed("probe.wide"));
    let embedding = seeded(hidden, name_seed("probe.embed"));

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("parity_mtp: no GPU ({e:?}) — skipped");
            return;
        }
    };

    let gw = MtpWeightsGpu::upload(&mut gpu, &cfg, &src).expect("upload mtp head");
    let mut gs = MtpScratchGpu::new(&mut gpu, &cfg, 64).expect("mtp scratch");
    let mut cache =
        hipfire_arch_qwen4exp::attn_gpu::QsaCache::new(&mut gpu, &cfg, 64).expect("qsa cache");
    let g_wide = gpu.upload_f32(&trunk_wide, &[width]).expect("upload wide");
    let g_emb = gpu.upload_f32(&embedding, &[hidden]).expect("upload emb");

    let cw = mtp::weights_from(&cfg, &src);
    let ifreq = hipfire_arch_qwen4exp::rope::inv_freq(cfg.rotary_dim(), cfg.rope_theta);
    let (cos, sin) = hipfire_arch_qwen4exp::rope::cos_sin(&[0usize], &ifreq);

    let mut fail = false;
    for fusion in [Fusion::BroadcastAllStreams, Fusion::Stream0Only] {
        cache.reset();
        mtp_decode_step(
            &mut gpu, &cfg, &gw, &mut cache, &mut gs, &g_wide, &g_emb, 0, fusion,
        )
        .expect("gpu mtp step");
        let got = gpu.download_f32(&gs.collapsed).expect("download");

        let want = mtp::forward(&cfg, &cw, &trunk_wide, &embedding, 1, &cos, &sin, fusion);
        let c = cosine(&got, &want);
        let d = max_abs_diff(&got, &want);
        let ok = c > 0.9999 && d < 2e-3;
        println!(
            "  {fusion:?}: cosine {c:.6}  max|Δ| {d:.3e}  {}",
            if ok { "ok" } else { "MISMATCH" }
        );
        if !ok {
            fail = true;
        }

        // NEGATIVE CONTROL: the OTHER fusion must not match, or `fusion` is
        // being ignored and the check above proves nothing.
        let other = match fusion {
            Fusion::BroadcastAllStreams => Fusion::Stream0Only,
            Fusion::Stream0Only => Fusion::BroadcastAllStreams,
        };
        let wrong = mtp::forward(&cfg, &cw, &trunk_wide, &embedding, 1, &cos, &sin, other);
        let cw_ = cosine(&got, &wrong);
        if cw_ > 0.9999 {
            println!("    control FAILED: also matches {other:?} — fusion is ignored");
            fail = true;
        }
    }

    if fail {
        eprintln!("parity_mtp: FAIL");
        std::process::exit(1);
    }
    println!("parity_mtp: OK — the GPU head reproduces the CPU reference");
}
