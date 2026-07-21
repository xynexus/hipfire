// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Locked tiny-dense dual-run parity at every layer boundary and final logits.

use hipfire_arch_gemma4::{
    forward_step_lowered, forward_step_reference, lower_dense_forward, Gemma4Config,
    Gemma4CoreWeights, Gemma4DenseLayerWeights, Gemma4DenseState, Gemma4DenseWeights,
    Gemma4ForwardCapture,
};
use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_runtime::weights::{EmbeddingFormat, WeightTensor};

const OPERATOR_LIMIT: f32 = 1e-6;

fn values(len: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let phase = ((index * 37 + seed * 101) % 997) as f32 * 0.017;
            phase.sin() * scale
        })
        .collect()
}

fn norm(gpu: &mut Gpu, len: usize, seed: usize) -> GpuTensor {
    let data = values(len, seed, 0.08)
        .into_iter()
        .map(|value| 0.95 + value)
        .collect::<Vec<_>>();
    gpu.upload_f32(&data, &[len]).unwrap()
}

fn weight(gpu: &mut Gpu, m: usize, k: usize, seed: usize) -> WeightTensor {
    let data = values(m * k, seed, 0.025 / (k as f32).sqrt());
    WeightTensor {
        buf: gpu.upload_f32(&data, &[m, k]).unwrap(),
        gpu_dtype: DType::F32,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    }
}

fn config() -> Gemma4Config {
    Gemma4Config::from_json_str(
        r#"{
          "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 64,
            "vocab_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 96,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 32,
            "global_head_dim": 64,
            "sliding_window": 8,
            "max_position_embeddings": 16,
            "rms_norm_eps": 0.000001,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "attention_k_eq_v": true,
            "enable_moe_block": false,
            "use_double_wide_mlp": false,
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
            "layer_types": ["sliding_attention", "full_attention"],
            "rope_parameters": {
              "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
              "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25
              }
            }
          }
        }"#,
    )
    .unwrap()
}

fn weights(gpu: &mut Gpu, config: &Gemma4Config) -> Gemma4DenseWeights {
    let mut layers = Vec::new();
    for (layer_idx, plan) in config.layers.iter().enumerate() {
        let q_dim = plan.attention.q_heads * plan.attention.head_dim;
        let kv_dim = plan.attention.kv_heads * plan.attention.head_dim;
        let seed = 10 + layer_idx * 20;
        layers.push(Gemma4DenseLayerWeights {
            input_norm: norm(gpu, config.hidden_size, seed),
            q_norm: norm(gpu, plan.attention.head_dim, seed + 1),
            k_norm: norm(gpu, plan.attention.head_dim, seed + 2),
            wq: weight(gpu, q_dim, config.hidden_size, seed + 3),
            wk: weight(gpu, kv_dim, config.hidden_size, seed + 4),
            wv: (layer_idx == 0).then(|| weight(gpu, kv_dim, config.hidden_size, seed + 5)),
            wo: weight(gpu, config.hidden_size, q_dim, seed + 6),
            post_attn_norm: norm(gpu, config.hidden_size, seed + 7),
            pre_ffn_norm: norm(gpu, config.hidden_size, seed + 8),
            post_ffn_norm: norm(gpu, config.hidden_size, seed + 9),
            w_gate: weight(gpu, config.intermediate_size, config.hidden_size, seed + 10),
            w_up: weight(gpu, config.intermediate_size, config.hidden_size, seed + 11),
            w_down: weight(gpu, config.hidden_size, config.intermediate_size, seed + 12),
            layer_scalar: if layer_idx == 0 { 0.97 } else { 1.03 },
            ple: None,
            moe: None,
        });
    }
    Gemma4DenseWeights {
        core: Gemma4CoreWeights {
            token_embd: gpu
                .upload_f32(
                    &values(config.vocab_size * config.hidden_size, 1, 0.04),
                    &[config.vocab_size, config.hidden_size],
                )
                .unwrap(),
            embd_format: EmbeddingFormat::F32,
            embedding_source_bf16: false,
            output_norm: norm(gpu, config.hidden_size, 2),
            output: weight(gpu, config.vocab_size, config.hidden_size, 3),
            tied_lm_head: true,
        },
        layers,
        ple: None,
    }
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn main() {
    let config = config();
    let mut gpu = Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);
    let weights = weights(&mut gpu, &config);
    let mut reference_state = Gemma4DenseState::new(&mut gpu, &config, 16).unwrap();
    let mut lowered_state = Gemma4DenseState::new(&mut gpu, &config, 16).unwrap();
    let lowered = lower_dense_forward(&config, &lowered_state);
    assert_eq!(lowered.layers.len(), 2);
    assert_eq!(lowered.layers[0].len(), 5);
    assert_eq!(lowered.final_program.len(), 3);

    let mut layer_max = [0.0f32; 2];
    let mut logits_max = 0.0f32;
    // Cross the SWA boundary at 7/8/9 while the global layer grows normally.
    for position in 0..10 {
        let token = (7 + position) as u32;
        let mut reference = Gemma4ForwardCapture::default();
        let mut actual = Gemma4ForwardCapture::default();
        forward_step_reference(
            &mut gpu,
            &weights,
            &config,
            &mut reference_state,
            token,
            Some(&mut reference),
        )
        .unwrap();
        forward_step_lowered(
            &mut gpu,
            &weights,
            &config,
            &mut lowered_state,
            &lowered,
            token,
            Some(&mut actual),
        )
        .unwrap();
        assert_eq!(reference.layer_boundaries.len(), 2);
        assert_eq!(actual.layer_boundaries.len(), 2);
        for layer in 0..2 {
            let error = max_abs(
                &reference.layer_boundaries[layer],
                &actual.layer_boundaries[layer],
            );
            layer_max[layer] = layer_max[layer].max(error);
            assert!(
                error <= OPERATOR_LIMIT,
                "position {position} layer {layer} reference/lowered max abs {error}"
            );
        }
        let error = max_abs(&reference.logits, &actual.logits);
        logits_max = logits_max.max(error);
        assert!(error <= OPERATOR_LIMIT);
    }
    assert_eq!(reference_state.next_pos(), 10);
    assert_eq!(lowered_state.next_pos(), 10);
    for (layer, error) in layer_max.into_iter().enumerate() {
        println!("layer {layer}: max_abs={error:.8}");
    }
    println!("logits: max_abs={logits_max:.8}");

    println!("dense_reference_lowered_parity: PASS");

    reference_state.free_gpu(&mut gpu);
    lowered_state.free_gpu(&mut gpu);
    weights.free_gpu(&mut gpu);
    gpu.drain_pool();
}
