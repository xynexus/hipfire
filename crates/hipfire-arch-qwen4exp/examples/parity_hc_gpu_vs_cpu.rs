// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU gated residual vs the CPU reference, both halves.
//!
//! The hyper-connection is the spine: every layer reads through it twice and
//! writes through it twice, so an error here corrupts everything downstream while
//! each individual kernel still passes its own parity. Differencing the COMPOSITION
//! is the point.
//!
//! Read and write are checked separately so a failure localises, and the mixer
//! variant (no block injection) is checked too — it is the model's final norm.

use hipfire_arch_qwen4exp::config::Qwen4ExpConfig;
use hipfire_arch_qwen4exp::hc::GatedResidual;
use hipfire_arch_qwen4exp::hc_gpu::{hc_read, hc_write, HcScratch, HcWeights};
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

fn maxd(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn main() {
    let cfg = Qwen4ExpConfig::from_json(&serde_json::json!({
        "text_config": {
            "vocab_size": 128, "hidden_size": 256, "intermediate_size": 64,
            "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "layer_types": ["linear_attention", "linear_attention",
                "linear_attention", "full_attention"],
            "linear_num_key_heads": 2, "linear_num_value_heads": 6,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "hc_count": 4, "hc_lowrank": 32,
            "ple_layer_ids": [2], "ple_embed_dim": 256, "ple_conv_kernel_size": 4,
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

    let (hidden, hc, lr) = (
        cfg.hidden,
        cfg.gated_residual.count,
        cfg.gated_residual.lowrank,
    );
    let width = hc * hidden;
    let (hn, md, mu, bi) = (
        seeded(width, 3),
        seeded(lr * width, 5),
        seeded(width * lr, 7),
        seeded(hc * width, 9),
    );

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("parity_hc_gpu_vs_cpu: no GPU ({e}) — skipped");
            return;
        }
    };
    let mut worst_read = 0.0f32;
    let mut worst_write = 0.0f32;
    let mut worst_mixer = 0.0f32;

    for trial in 0..8u32 {
        let streams = seeded(width, 100 + trial);
        let block_out = seeded(hidden, 200 + trial);

        for &with_inject in &[true, false] {
            let cpu = GatedResidual {
                hc_norm: &hn,
                mix_down: &md,
                mix_up: &mu,
                block_inject: if with_inject { Some(&bi) } else { None },
                hc_count: hc,
                hidden,
                lowrank: lr,
                eps: cfg.rms_norm_eps,
            };
            let r = cpu.read(&streams);

            let w = HcWeights {
                hc_norm: gpu.upload_f32(&hn, &[width]).unwrap(),
                mix_down: gpu.upload_f32(&md, &[lr, width]).unwrap(),
                mix_up: gpu.upload_f32(&mu, &[width, lr]).unwrap(),
                block_inject: with_inject.then(|| gpu.upload_f32(&bi, &[hc, width]).unwrap()),
            };
            let mut s = HcScratch::new(&mut gpu, &cfg).unwrap();
            let gs = gpu.upload_f32(&streams, &[width]).unwrap();
            let gmix = gpu.zeros(&[hidden], DType::F32).unwrap();
            hc_read(&mut gpu, &cfg, &w, &mut s, &gs, &gmix).unwrap();
            let got = gpu.download_f32(&gmix).unwrap();
            let d = maxd(&got, &r.mixed_input);
            if with_inject {
                worst_read = worst_read.max(d);
            } else {
                worst_mixer = worst_mixer.max(d);
            }

            if with_inject {
                // Write back through the same scratch the read populated.
                let inj = gpu.download_f32(&s.inject).unwrap();
                let dg = maxd(&inj, r.inject.as_ref().unwrap());
                worst_read = worst_read.max(dg);
                // The gate must span (0, 2): a missing factor of two reads as a
                // plausible (0, 1) gate and halves every residual write.
                assert!(
                    inj.iter().all(|v| *v > 0.0 && *v < 2.0),
                    "inject gate outside (0, 2): {inj:?}"
                );

                let mut want = streams.clone();
                cpu.write(&mut want, &block_out, r.inject.as_ref().unwrap());
                let gb = gpu.upload_f32(&block_out, &[hidden]).unwrap();
                hc_write(&mut gpu, &cfg, &s, &gs, &gb).unwrap();
                worst_write = worst_write.max(maxd(&gpu.download_f32(&gs).unwrap(), &want));
            }
        }
    }

    let tol = 2e-5;
    let ok = worst_read <= tol && worst_write <= tol && worst_mixer <= tol;
    println!(
        "parity_hc_gpu_vs_cpu: read {worst_read:.3e} write {worst_write:.3e} \
         mixer {worst_mixer:.3e} (tol {tol:.0e}) -> {}",
        if ok { "OK" } else { "FAILED" }
    );
    if !ok {
        std::process::exit(1);
    }
}
