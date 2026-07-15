// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Full-model Gemma 4 candidate capture for offline Transformers comparison.
//
// Despite the historical example name, this accepts every weight encoding the
// Gemma 4 loader supports, including OQ8 and OQ8++. `INPUT` is either rendered
// UTF-8 prompt bytes or a JSON array/object containing exact `input_ids`.
// Set `HIPFIRE_GEMMA4_CAPTURE_LIFECYCLE=1` to additionally require a reset
// rerun and a full unload/reload rerun to reproduce the first capture exactly.

use hipfire_arch_gemma4::{
    forward_step_lowered, forward_step_reference, generation_eos_ids_from_hfq, load_dense_weights,
    lower_dense_forward, Gemma4, Gemma4Config, Gemma4DenseState, Gemma4DenseWeights,
    Gemma4ForwardCapture,
};
use hipfire_dispatch::pipeline::superop::LoweredForward;
use hipfire_rdna::Gpu;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

struct CaptureRun {
    final_capture: Gemma4ForwardCapture,
    argmax_per_position: Vec<u32>,
    operator_history: Vec<BTreeMap<String, Vec<f32>>>,
    generated_ids: Vec<u32>,
}

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

fn stable_input_hash(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv64:{hash:016x}")
}

fn input_tokens(
    path: &Path,
    tokenizer: &Tokenizer,
) -> Result<(Vec<u32>, &'static str, String), String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let hash = stable_input_hash(&bytes);
    if path.extension().and_then(|value| value.to_str()) != Some("json") {
        let prompt = std::str::from_utf8(&bytes)
            .map_err(|error| format!("{} is not UTF-8: {error}", path.display()))?;
        return Ok((tokenizer.encode(prompt), "rendered_prompt", hash));
    }

    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let exact_ids = value
        .as_array()
        .or_else(|| value.get("input_ids").and_then(serde_json::Value::as_array));
    let (ids, repeated_length) = if let Some(ids) = exact_ids {
        (ids, None)
    } else if let Some(ids) = value
        .get("input_ids_pattern")
        .and_then(serde_json::Value::as_array)
    {
        let length = value
            .get("length")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| "input_ids_pattern requires a positive u64 length".to_string())?;
        (ids, Some(length))
    } else {
        return Err(format!(
            "{} must contain input_ids or input_ids_pattern plus length",
            path.display()
        ));
    };
    let pattern = ids
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| format!("input token {index} is not a u32"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let tokens = match repeated_length {
        Some(length) if !pattern.is_empty() && length > 0 => {
            pattern.iter().copied().cycle().take(length).collect()
        }
        Some(_) => return Err("input_ids_pattern and length must both be nonzero".to_string()),
        None => pattern,
    };
    Ok((tokens, "exact_token_ids", hash))
}

#[allow(clippy::too_many_arguments)]
fn run_capture(
    gpu: &mut Gpu,
    weights: &Gemma4DenseWeights,
    config: &Gemma4Config,
    state: &mut Gemma4DenseState,
    lowered: &LoweredForward,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    tokens: &[u32],
    max_new: usize,
    operator_layer: Option<usize>,
) -> Result<CaptureRun, String> {
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
            forward_step_reference(gpu, weights, config, state, token, Some(&mut step_capture))
                .map_err(|error| format!("Gemma 4 prompt position {position}: {error:?}"))?;
            operator_history.push(step_capture.operator_boundaries.clone());
            if is_final {
                final_capture = step_capture;
            }
        } else {
            let capture = is_final.then_some(&mut final_capture);
            forward_step_lowered(gpu, weights, config, state, lowered, token, capture)
                .map_err(|error| format!("Gemma 4 prompt position {position}: {error:?}"))?;
        }
        argmax_per_position.push(
            gpu.argmax_f32(state.logits_tensor(), config.vocab_size)
                .map_err(|error| format!("Gemma 4 argmax position {position}: {error:?}"))?,
        );
    }

    let mut generated_ids = Vec::with_capacity(max_new);
    let mut next = *argmax_per_position.last().expect("nonempty prompt");
    for step in 0..max_new {
        generated_ids.push(next);
        if tokenizer.is_terminator(next) || eos_token_ids.contains(&next) {
            break;
        }
        forward_step_lowered(gpu, weights, config, state, lowered, next, None)
            .map_err(|error| format!("Gemma 4 generated step {step}: {error:?}"))?;
        next = gpu
            .argmax_f32(state.logits_tensor(), config.vocab_size)
            .map_err(|error| format!("Gemma 4 generated argmax {step}: {error:?}"))?;
    }

    Ok(CaptureRun {
        final_capture,
        argmax_per_position,
        operator_history,
        generated_ids,
    })
}

fn exact_lifecycle_match(reference: &CaptureRun, candidate: &CaptureRun) -> bool {
    reference.argmax_per_position == candidate.argmax_per_position
        && reference.generated_ids == candidate.generated_ids
        && reference.final_capture.layer_boundaries == candidate.final_capture.layer_boundaries
        && reference.final_capture.final_hidden == candidate.final_capture.final_hidden
        && reference.final_capture.logits == candidate.final_capture.logits
        && reference.final_capture.operator_boundaries
            == candidate.final_capture.operator_boundaries
        && reference.operator_history == candidate.operator_history
}

fn main() -> Result<(), String> {
    let args = std::env::args().collect::<Vec<_>>();
    if !(5..=7).contains(&args.len()) {
        return Err(
            "usage: bf16_capture MODEL.hfq INPUT OUT_DIR LAYERS [MAX_NEW [OPERATOR_LAYER]]"
                .to_string(),
        );
    }
    let model = PathBuf::from(&args[1]);
    let input = PathBuf::from(&args[2]);
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
    let (tokens, input_kind, input_hash) = input_tokens(&input, &tokenizer)?;
    if tokens.is_empty() {
        return Err("capture input contains zero token ids".to_string());
    }
    let config = Gemma4::config_from_hfq(&hfq)?;
    if let Some((index, token)) = tokens
        .iter()
        .enumerate()
        .find(|(_, token)| **token as usize >= config.vocab_size)
    {
        return Err(format!(
            "input token {token} at index {index} exceeds vocabulary {}",
            config.vocab_size
        ));
    }
    let eos_token_ids = generation_eos_ids_from_hfq(&hfq);
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
    let first = run_capture(
        &mut gpu,
        &weights,
        &config,
        &mut state,
        &lowered,
        &tokenizer,
        &eos_token_ids,
        &tokens,
        max_new,
        operator_layer,
    )?;
    let lifecycle = std::env::var("HIPFIRE_GEMMA4_CAPTURE_LIFECYCLE")
        .ok()
        .as_deref()
        == Some("1");
    let mut reset_exact_match = None;
    let mut reload_exact_match = None;
    if lifecycle {
        state.reset();
        let second = run_capture(
            &mut gpu,
            &weights,
            &config,
            &mut state,
            &lowered,
            &tokenizer,
            &eos_token_ids,
            &tokens,
            max_new,
            operator_layer,
        )?;
        let matched = exact_lifecycle_match(&first, &second);
        reset_exact_match = Some(matched);
        if !matched {
            state.free_gpu(&mut gpu);
            weights.free_gpu(&mut gpu);
            return Err("Gemma 4 reset rerun differs from the first capture".to_string());
        }
    }

    state.free_gpu(&mut gpu);
    weights.free_gpu(&mut gpu);
    gpu.drain_pool();

    if lifecycle {
        let mut reload_hfq = HfqFile::open(&model).map_err(|error| error.to_string())?;
        let reload_config = Gemma4::config_from_hfq(&reload_hfq)?;
        let reload_weights = load_dense_weights(&mut reload_hfq, &mut gpu, &reload_config)
            .map_err(|error| format!("Gemma 4 reload weights: {error:?}"))?;
        let mut reload_state = Gemma4DenseState::new(&mut gpu, &reload_config, max_seq)
            .map_err(|error| format!("Gemma 4 reload state: {error:?}"))?;
        let reload_lowered = lower_dense_forward(&reload_config, &reload_state);
        let reloaded = run_capture(
            &mut gpu,
            &reload_weights,
            &reload_config,
            &mut reload_state,
            &reload_lowered,
            &tokenizer,
            &eos_token_ids,
            &tokens,
            max_new,
            operator_layer,
        )?;
        let matched = exact_lifecycle_match(&first, &reloaded);
        reload_exact_match = Some(matched);
        reload_state.free_gpu(&mut gpu);
        reload_weights.free_gpu(&mut gpu);
        gpu.drain_pool();
        if !matched {
            return Err("Gemma 4 unload/reload rerun differs from the first capture".to_string());
        }
    }

    fs::create_dir_all(&out).map_err(|error| error.to_string())?;
    write_f32(
        &out.join("final_hidden.f32"),
        &first.final_capture.final_hidden,
    )?;
    write_f32(&out.join("final_logits.f32"), &first.final_capture.logits)?;
    for &layer in &layers {
        write_f32(
            &out.join(format!("hidden_layer_{layer}.f32")),
            &first.final_capture.layer_boundaries[layer],
        )?;
    }
    for (name, values) in &first.final_capture.operator_boundaries {
        write_f32(&out.join(format!("operator_{name}.f32")), values)?;
    }
    for (position, boundaries) in first.operator_history.iter().enumerate() {
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
        "input_path": input,
        "input_kind": input_kind,
        "input_hash": input_hash,
        "tokenizer_source": "hfq_metadata",
        "input_ids": tokens,
        "input_token_count": tokens.len(),
        "argmax_per_position": first.argmax_per_position,
        "capture_mode": "sequential",
        "generated_ids": first.generated_ids,
        "captured_layers": layers,
        "operator_layer": operator_layer,
        "operator_boundaries": first.final_capture.operator_boundaries.keys().collect::<Vec<_>>(),
        "operator_positions": first.operator_history.len(),
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "max_new_tokens": max_new,
        "gpu_arch": gpu.arch,
        "lifecycle": {
            "requested": lifecycle,
            "reset_exact_match": reset_exact_match,
            "unload_reload_exact_match": reload_exact_match,
        },
    });
    fs::write(
        out.join("capture.json"),
        serde_json::to_vec_pretty(&metadata).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;

    println!(
        "gemma4_capture: PASS (tokens={} generated={} layers={:?} lifecycle={})",
        metadata["input_token_count"],
        metadata["max_new_tokens"],
        metadata["captured_layers"],
        lifecycle
    );
    Ok(())
}
