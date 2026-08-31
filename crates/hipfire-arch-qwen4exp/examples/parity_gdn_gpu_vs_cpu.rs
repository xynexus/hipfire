// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU `gdn_decode_step` vs the CPU `GdnCpu` reference, streamed.
//!
//! This closes the chain. `tests/reference_oracle.rs` pins `GdnCpu` against the
//! pinned upstream implementation; this pins the GPU path against `GdnCpu`. Neither
//! test alone says the GPU serving path computes Gated DeltaNet correctly — together
//! they do.
//!
//! Run over many steps rather than one: the recurrent state and the conv ring both
//! carry across steps, so a state-update bug is invisible at t=0 and compounds
//! afterwards. A single-step check would pass on a decayless recurrence.

use hipfire_arch_qwen4exp::config::Qwen4ExpConfig;
use hipfire_arch_qwen4exp::gdn::{gdn_decode_step, GdnScratch, GdnState, GdnWeights};
use hipfire_arch_qwen4exp::gdn_cpu::GdnCpu;
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
    let cfg_json = serde_json::json!({
        "text_config": {
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 64,
            "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "layer_types": ["linear_attention", "linear_attention",
                "linear_attention", "full_attention"],
            // head_dim 128 and a 3:1 value:key ratio, because that is what the
            // shipped model uses (16 key / 48 value heads) and the Gated DeltaNet
            // kernels are HD-specialised — a 16-wide head does not dispatch at all.
            "linear_num_key_heads": 2, "linear_num_value_heads": 6,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "hc_count": 4, "hc_lowrank": 16,
            "ple_layer_ids": [2], "ple_embed_dim": 64, "ple_conv_kernel_size": 4,
            "ngram_size": 3, "heads_per_ngram": 2,
            "ngram_vocab_size_base": 2000, "make_ngram_vocab_size_divisible_by": 8,
            "split_ngram_parts": 128, "seed": 1234,
            "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 16,
            "indexer_budget": 8, "indexer_compress_ratio": 4,
            "output_gate_type": "sigmoid",
            "rms_norm_eps": 1e-6, "max_position_embeddings": 256, "eos_token_id": 2,
        }
    });
    let cfg = Qwen4ExpConfig::from_json(&cfg_json).expect("config");
    let d = &cfg.deltanet;
    let (hidden, nv, hk, hv) = (cfg.hidden, d.value_heads, d.key_head_dim, d.value_head_dim);
    let (qkv_dim, z_dim) = (d.qkv_dim(), d.z_dim());

    let w_qkv = seeded(qkv_dim * hidden, 11);
    let w_z = seeded(z_dim * hidden, 13);
    let w_a = seeded(nv * hidden, 17);
    let w_b = seeded(nv * hidden, 19);
    let w_conv = seeded(qkv_dim * d.conv_kernel, 23);
    // A_log is log(A) with A in (0.01, 16]; keep it in that range or the decay is
    // meaningless and the test says nothing about the recurrence.
    let a_log: Vec<f32> = seeded(nv, 29)
        .iter()
        .map(|v| (v * 2.0).exp().ln())
        .collect();
    let dt_bias = seeded(nv, 31);
    let w_norm = seeded(hv, 37);
    let w_out = seeded(hidden * z_dim, 41);

    let cpu = GdnCpu {
        in_proj_qkv: &w_qkv,
        in_proj_z: &w_z,
        in_proj_a: &w_a,
        in_proj_b: &w_b,
        conv_weight: &w_conv,
        a_log: &a_log,
        dt_bias: &dt_bias,
        norm_weight: &w_norm,
        out_proj: &w_out,
        hidden,
        n_k: d.key_heads,
        n_v: nv,
        head_k: hk,
        head_v: hv,
        kernel: d.conv_kernel,
        gate_sigmoid: d.output_gate_sigmoid,
        eps: cfg.rms_norm_eps,
    };

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("parity_gdn_gpu_vs_cpu: no GPU ({e}) — skipped");
            return;
        }
    };
    let up = |g: &mut Gpu, v: &[f32], shape: &[usize]| g.upload_f32(v, shape).unwrap();
    let gw = GdnWeights {
        in_proj_qkv: hipfire_arch_qwen4exp::trunk_gpu::f32_weight(
            &mut gpu,
            &w_qkv,
            &[qkv_dim, hidden],
        )
        .unwrap(),
        in_proj_z: hipfire_arch_qwen4exp::trunk_gpu::f32_weight(&mut gpu, &w_z, &[z_dim, hidden])
            .unwrap(),
        in_proj_a: hipfire_arch_qwen4exp::trunk_gpu::f32_weight(&mut gpu, &w_a, &[nv, hidden])
            .unwrap(),
        in_proj_b: hipfire_arch_qwen4exp::trunk_gpu::f32_weight(&mut gpu, &w_b, &[nv, hidden])
            .unwrap(),
        conv_weight: up(&mut gpu, &w_conv, &[qkv_dim, d.conv_kernel]),
        a_log: up(&mut gpu, &a_log, &[nv]),
        dt_bias: up(&mut gpu, &dt_bias, &[nv]),
        norm_weight: up(&mut gpu, &w_norm, &[hv]),
        out_proj: hipfire_arch_qwen4exp::trunk_gpu::f32_weight(&mut gpu, &w_out, &[hidden, z_dim])
            .unwrap(),
    };
    let mut scratch = GdnScratch::new(&mut gpu, &cfg).unwrap();
    let mut gst = GdnState::zeros(&mut gpu, &cfg).unwrap();
    let mut cst = cpu.zero_state();

    let n_steps = 24;
    let mut worst = 0.0f32;
    let mut worst_t = 0;
    for t in 0..n_steps {
        let x = seeded(hidden, 101 + t as u32);
        let want = cpu.step(&x, &mut cst);

        let gx = up(&mut gpu, &x, &[hidden]);
        let gy = gpu.zeros(&[hidden], DType::F32).unwrap();
        gdn_decode_step(&mut gpu, &cfg, &gw, &mut scratch, &mut gst, &gx, &gy).unwrap();
        let got = gpu.download_f32(&gy).unwrap();

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

    // The state must actually have moved: a recurrence stuck at zero would agree
    // with a CPU reference that is also stuck at zero.
    let rec = gpu.download_f32(&gst.recurrent).unwrap();
    let live = rec.iter().any(|v| v.abs() > 1e-6);
    let tol = 2e-4;
    let ok = worst <= tol && live;
    println!(
        "parity_gdn_gpu_vs_cpu: {n_steps} steps, worst max|Δ| = {worst:.3e} at t={worst_t} \
         (tol {tol:.0e}), recurrent state live: {live} -> {}",
        if ok { "OK" } else { "FAILED" }
    );
    if !live {
        println!("  FAILED: the recurrent state is all zeros — the test proves nothing");
    }
    if !ok {
        std::process::exit(1);
    }
}
