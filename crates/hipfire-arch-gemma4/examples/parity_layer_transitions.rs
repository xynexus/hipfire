// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Compare every dense Gemma 4 decoder layer from exact frozen-oracle inputs.
//!
//! This diagnostic bypasses embedding, cross-layer propagation, final norm,
//! and the LM head. It still executes each selected layer position by position
//! so its own full/SWA KV history is real. Coordinate it with `hipfire gpu-lock`.
//!
//! Usage: `parity_layer_transitions MODEL.hfq INPUT_DIR OUTPUT.json`

use hipfire_arch_gemma4::config::AttentionKind;
use hipfire_arch_gemma4::{
    diagnostic_forward_layer_from_hidden, load_dense_weights, Gemma4, Gemma4DenseState,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

fn read_f32(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if !bytes.len().is_multiple_of(4) {
        return Err(format!("{} has a partial F32 value", path.display()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn round_bf16(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
}

fn metrics(expected: &[f32], actual: &[f32]) -> Value {
    let mut dot = 0.0f64;
    let mut expected_sq = 0.0f64;
    let mut actual_sq = 0.0f64;
    let mut error_sq = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut bf16_exact = 0usize;
    let mut non_finite = 0usize;
    for (&reference, &candidate) in expected.iter().zip(actual) {
        if !reference.is_finite() || !candidate.is_finite() {
            non_finite += 1;
            continue;
        }
        let r = f64::from(reference);
        let c = f64::from(candidate);
        dot += r * c;
        expected_sq += r * r;
        actual_sq += c * c;
        error_sq += (r - c).powi(2);
        max_abs = max_abs.max((reference - candidate).abs());
        bf16_exact += usize::from(reference.to_bits() == round_bf16(candidate).to_bits());
    }
    json!({
        "maximum_absolute_error": max_abs,
        "normalized_rmse": (error_sq / expected_sq).sqrt(),
        "cosine": dot / (expected_sq.sqrt() * actual_sq.sqrt()),
        "bf16_exact": bf16_exact,
        "values": expected.len(),
        "non_finite_values": non_finite,
    })
}

fn required_usize(manifest: &Value, name: &str) -> Result<usize, String> {
    manifest
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("manifest `{name}` must be a non-negative integer"))
}

fn main() -> Result<(), String> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 4 {
        return Err("usage: parity_layer_transitions MODEL.hfq INPUT_DIR OUTPUT.json".into());
    }
    let model = PathBuf::from(&args[1]);
    let input_dir = PathBuf::from(&args[2]);
    let output = PathBuf::from(&args[3]);
    let manifest: Value = serde_json::from_slice(
        &fs::read(input_dir.join("manifest.json")).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("input manifest: {error}"))?;
    if manifest.get("schema").and_then(Value::as_str)
        != Some("hipfire.gemma4.layer-transition-inputs.v1")
    {
        return Err("input manifest has the wrong schema".into());
    }
    let positions = required_usize(&manifest, "positions")?;
    let hidden_size = required_usize(&manifest, "hidden_size")?;
    let layer_count = required_usize(&manifest, "layers")?;
    if positions == 0 || layer_count == 0 {
        return Err("transition input set must contain positions and layers".into());
    }

    let mut hfq = HfqFile::open(&model).map_err(|error| error.to_string())?;
    if hfq.arch_id != Gemma4::arch_id() {
        return Err(format!(
            "transition parity expected Gemma 4 arch {}, got {}",
            Gemma4::arch_id(),
            hfq.arch_id
        ));
    }
    let config = Gemma4::config_from_hfq(&hfq)?;
    if hidden_size != config.hidden_size || layer_count != config.num_hidden_layers {
        return Err(format!(
            "input geometry hidden/layers={hidden_size}/{layer_count} does not match model {}/{}",
            config.hidden_size, config.num_hidden_layers
        ));
    }

    let mut gpu =
        hipfire_rdna::Gpu::init_with_device(0).map_err(|error| format!("GPU init: {error:?}"))?;
    let weights = load_dense_weights(&mut hfq, &mut gpu, &config)
        .map_err(|error| format!("Gemma 4 weights: {error:?}"))?;
    let mut state = Gemma4DenseState::new(&mut gpu, &config, positions)
        .map_err(|error| format!("Gemma 4 state: {error:?}"))?;

    let mut layer_results = Vec::with_capacity(layer_count);
    for layer in 0..layer_count {
        let inputs = read_f32(&input_dir.join(format!("input_layer_{layer}.f32")))?;
        let expected = read_f32(&input_dir.join(format!("expected_layer_{layer}.f32")))?;
        let expected_values = positions * hidden_size;
        if inputs.len() != expected_values || expected.len() != expected_values {
            return Err(format!(
                "layer {layer} input/expected lengths {}/{} do not match {expected_values}",
                inputs.len(),
                expected.len()
            ));
        }

        state.reset();
        let mut actual = Vec::with_capacity(expected_values);
        for position in 0..positions {
            let start = position * hidden_size;
            actual.extend(
                diagnostic_forward_layer_from_hidden(
                    &mut gpu,
                    &weights,
                    &config,
                    &mut state,
                    layer,
                    position,
                    &inputs[start..start + hidden_size],
                )
                .map_err(|error| format!("Gemma 4 layer {layer} position {position}: {error:?}"))?,
            );
        }
        let result = metrics(&expected, &actual);
        let per_position = (0..positions)
            .map(|position| {
                let start = position * hidden_size;
                metrics(
                    &expected[start..start + hidden_size],
                    &actual[start..start + hidden_size],
                )
            })
            .collect::<Vec<_>>();
        let final_position = per_position.last().expect("nonempty positions");
        let kind = match config.layers[layer].kind {
            AttentionKind::Sliding => "sliding",
            AttentionKind::Full => "full",
        };
        println!(
            "layer={layer:02} kind={kind:<7} all_nrmse={:.9} final_max_abs={:.9} \
             final_nrmse={:.9} final_cosine={:.9} final_bf16_exact={}/{} non_finite={}",
            result["normalized_rmse"].as_f64().unwrap(),
            final_position["maximum_absolute_error"].as_f64().unwrap(),
            final_position["normalized_rmse"].as_f64().unwrap(),
            final_position["cosine"].as_f64().unwrap(),
            final_position["bf16_exact"].as_u64().unwrap(),
            final_position["values"].as_u64().unwrap(),
            final_position["non_finite_values"].as_u64().unwrap(),
        );
        layer_results.push(json!({
            "layer": layer,
            "attention_kind": kind,
            "all_positions": result,
            "final_position": final_position,
            "per_position": per_position,
        }));
    }

    let report = json!({
        "schema": "hipfire.gemma4.layer-transition-parity.v1",
        "model": model,
        "input_manifest": input_dir.join("manifest.json"),
        "gpu_arch": gpu.arch,
        "positions": positions,
        "hidden_size": hidden_size,
        "layers": layer_results,
    });
    fs::write(
        &output,
        serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("{}: {error}", output.display()))?;

    state.free_gpu(&mut gpu);
    weights.free_gpu(&mut gpu);
    gpu.drain_pool();
    println!("wrote {}", output.display());
    Ok(())
}
