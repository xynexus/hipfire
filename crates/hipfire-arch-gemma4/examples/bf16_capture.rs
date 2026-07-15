// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Full-model Gemma 4 BF16 capture for offline Transformers comparison.
//!
//! Usage: `bf16_capture MODEL.hfq PROMPT.txt OUT_DIR LAYERS [MAX_NEW]`, where
//! `LAYERS` is a comma-separated zero-based decoder-layer list. The prompt is
//! already rendered (raw for base models, official Jinja bytes for IT models).

use hipfire_arch_gemma4::{
    forward_step_lowered, forward_step_reference, load_dense_weights, lower_dense_forward, Gemma4,
    Gemma4DenseState, Gemma4ForwardCapture,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

fn write_f32(path: &Path, values: &[f32]) -> Result<(), String> {
    let mut writer = BufWriter::new(File::create(path).map_err(|error| error.to_string())?);
    for value in values {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

fn parse_layers(raw: &str, count: usize) -> Result<Vec<usize>, String> {
    let mut layers = raw
        .split(',')
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| format!("invalid capture layer `{value}`: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    layers.sort_unstable();
    layers.dedup();
    if layers.iter().any(|&layer| layer >= count) {
        return Err(format!(
            "capture layers {layers:?} exceed layer count {count}"
        ));
    }
    Ok(layers)
}

fn main() -> Result<(), String> {
    let args = std::env::args().collect::<Vec<_>>();
    if !(5..=7).contains(&args.len()) {
        return Err(
            "usage: bf16_capture MODEL.hfq PROMPT.txt OUT_DIR LAYERS [MAX_NEW [OPERATOR_LAYER]]"
                .to_string(),
        );
    }
    let model = PathBuf::from(&args[1]);
    let prompt = fs::read_to_string(&args[2]).map_err(|error| error.to_string())?;
    let out = PathBuf::from(&args[3]);
    let max_new = args
        .get(5)
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| format!("invalid MAX_NEW: {error}"))?
        .unwrap_or(8);
    let operator_layer = args
        .get(6)
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| format!("invalid OPERATOR_LAYER: {error}"))?;

    let mut hfq = HfqFile::open(&model).map_err(|error| error.to_string())?;
    if hfq.arch_id != Gemma4::arch_id() {
        return Err(format!(
            "capture expected Gemma 4 arch {}, got {}",
            Gemma4::arch_id(),
            hfq.arch_id
        ));
    }
    let tokenizer =
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|error| error.to_string())?;
    let tokens = tokenizer.encode(&prompt);
    if tokens.is_empty() {
        return Err("rendered prompt tokenized to zero ids".to_string());
    }
    let config = Gemma4::config_from_hfq(&hfq)?;
    let layers = parse_layers(&args[4], config.num_hidden_layers)?;
    if operator_layer.is_some_and(|layer| layer >= config.num_hidden_layers) {
        return Err(format!(
            "operator capture layer {operator_layer:?} exceeds layer count {}",
            config.num_hidden_layers
        ));
    }
    let max_seq = tokens
        .len()
        .checked_add(max_new)
        .ok_or_else(|| "capture context overflow".to_string())?;
    if max_seq > config.max_position_embeddings {
        return Err(format!(
            "capture context {max_seq} exceeds trained context {}",
            config.max_position_embeddings
        ));
    }

    let mut gpu =
        hipfire_rdna::Gpu::init_with_device(0).map_err(|error| format!("GPU init: {error:?}"))?;
    let weights = load_dense_weights(&mut hfq, &mut gpu, &config)
        .map_err(|error| format!("Gemma 4 weights: {error:?}"))?;
    let mut state = Gemma4DenseState::new(&mut gpu, &config, max_seq)
        .map_err(|error| format!("Gemma 4 state: {error:?}"))?;
    let lowered = lower_dense_forward(&config, &state);

    let mut argmax_per_position = Vec::with_capacity(tokens.len());
    let mut operator_history = Vec::new();
    let mut final_capture = Gemma4ForwardCapture {
        operator_layer,
        ..Gemma4ForwardCapture::default()
    };
    for (position, &token) in tokens.iter().enumerate() {
        let is_final = position + 1 == tokens.len();
        if operator_layer.is_some() {
            let mut step_capture = Gemma4ForwardCapture {
                operator_layer,
                ..Gemma4ForwardCapture::default()
            };
            forward_step_reference(
                &mut gpu,
                &weights,
                &config,
                &mut state,
                token,
                Some(&mut step_capture),
            )
            .map_err(|error| format!("Gemma 4 prompt position {position}: {error:?}"))?;
            operator_history.push(step_capture.operator_boundaries.clone());
            if is_final {
                final_capture = step_capture;
            }
        } else {
            let capture = is_final.then_some(&mut final_capture);
            forward_step_lowered(
                &mut gpu, &weights, &config, &mut state, &lowered, token, capture,
            )
            .map_err(|error| format!("Gemma 4 prompt position {position}: {error:?}"))?;
        }
        argmax_per_position.push(
            gpu.argmax_f32(state.logits_tensor(), config.vocab_size)
                .map_err(|error| format!("Gemma 4 argmax position {position}: {error:?}"))?,
        );
    }

    let mut generated_ids = Vec::with_capacity(max_new);
    let mut next = *argmax_per_position.last().expect("nonempty prompt");
    let end_of_turn = tokenizer.special_token_id("<end_of_turn>");
    for step in 0..max_new {
        generated_ids.push(next);
        if tokenizer.is_terminator(next) || end_of_turn == Some(next) {
            break;
        }
        forward_step_lowered(
            &mut gpu, &weights, &config, &mut state, &lowered, next, None,
        )
        .map_err(|error| format!("Gemma 4 generated step {step}: {error:?}"))?;
        next = gpu
            .argmax_f32(state.logits_tensor(), config.vocab_size)
            .map_err(|error| format!("Gemma 4 generated argmax {step}: {error:?}"))?;
    }

    fs::create_dir_all(&out).map_err(|error| error.to_string())?;
    write_f32(&out.join("final_hidden.f32"), &final_capture.final_hidden)?;
    write_f32(&out.join("final_logits.f32"), &final_capture.logits)?;
    for &layer in &layers {
        write_f32(
            &out.join(format!("hidden_layer_{layer}.f32")),
            &final_capture.layer_boundaries[layer],
        )?;
    }
    for (name, values) in &final_capture.operator_boundaries {
        write_f32(&out.join(format!("operator_{name}.f32")), values)?;
    }
    for (position, boundaries) in operator_history.iter().enumerate() {
        for (name, values) in boundaries {
            write_f32(
                &out.join(format!("operator_position_{position}_{name}.f32")),
                values,
            )?;
        }
    }
    let metadata = serde_json::json!({
        "model": model,
        "arch_id": hfq.arch_id,
        "input_ids": tokens,
        "input_token_count": tokens.len(),
        "argmax_per_position": argmax_per_position,
        "capture_mode": "sequential",
        "generated_ids": generated_ids,
        "captured_layers": layers,
        "operator_layer": operator_layer,
        "operator_boundaries": final_capture.operator_boundaries.keys().collect::<Vec<_>>(),
        "operator_positions": operator_history.len(),
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "max_new_tokens": max_new,
        "gpu_arch": gpu.arch,
    });
    fs::write(
        out.join("capture.json"),
        serde_json::to_vec_pretty(&metadata).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;

    state.free_gpu(&mut gpu);
    weights.free_gpu(&mut gpu);
    gpu.drain_pool();
    println!(
        "bf16_capture: PASS (tokens={} generated={} layers={:?})",
        metadata["input_token_count"], metadata["max_new_tokens"], metadata["captured_layers"]
    );
    Ok(())
}
