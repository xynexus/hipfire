// SPDX-License-Identifier: Apache-2.0
// hipfire — EmbeddingGemma calibration collection. See LICENSE / NOTICE.

use std::collections::HashMap;

use hipfire_rdna::Gpu;
use hipfire_runtime::calibration::{
    capture_host_activations, collect_embedding_grouped, CalibSummary,
};
use hipfire_runtime::weights::WeightTensor;

use crate::config::{EmbeddingGemmaConfig, PoolingMode};
use crate::forward::{encode_pooled_hidden, project_dense_with_capture};
use crate::weights::EmbeddingGemmaWeights;

fn put(map: &mut HashMap<usize, String>, weight: &WeightTensor, name: String) {
    map.insert(weight.buf.buf.as_ptr() as usize, name);
}

fn layer_tensor_names(layer_idx: usize) -> [String; 7] {
    let prefix = format!("model.layers.{layer_idx}");
    [
        format!("{prefix}.self_attn.q_proj"),
        format!("{prefix}.self_attn.k_proj"),
        format!("{prefix}.self_attn.v_proj"),
        format!("{prefix}.self_attn.o_proj"),
        format!("{prefix}.mlp.gate_proj"),
        format!("{prefix}.mlp.up_proj"),
        format!("{prefix}.mlp.down_proj"),
    ]
}

#[cfg(test)]
fn backbone_tensor_names(start_layer: usize, end_layer: usize) -> Vec<String> {
    (start_layer..end_layer)
        .flat_map(layer_tensor_names)
        .collect()
}

fn build_capture_names_for_layers(
    weights: &EmbeddingGemmaWeights,
    start_layer: usize,
    end_layer: usize,
) -> HashMap<usize, String> {
    let mut map = HashMap::new();
    for (layer_idx, layer) in weights
        .backbone
        .layers
        .iter()
        .enumerate()
        .skip(start_layer)
        .take(end_layer.saturating_sub(start_layer))
    {
        let layer_weights = [
            &layer.wq,
            &layer.wk,
            &layer.wv,
            &layer.wo,
            &layer.w_gate,
            &layer.w_up,
            &layer.w_down,
        ];
        for (weight, name) in layer_weights.into_iter().zip(layer_tensor_names(layer_idx)) {
            put(&mut map, weight, name);
        }
    }
    map
}

fn validate_samples(samples: &[Vec<u32>], sliding_window: usize) -> Result<(), String> {
    if samples.is_empty() {
        return Err("embeddinggemma calibration: sample set is empty".to_string());
    }
    if let Some((sample_idx, sample)) = samples
        .iter()
        .enumerate()
        .find(|(_, sample)| sample.is_empty() || sample.len() > sliding_window)
    {
        if sample.is_empty() {
            return Err(format!(
                "embeddinggemma calibration: sample {sample_idx} is empty"
            ));
        }
        return Err(format!(
            "embeddinggemma calibration: sample {sample_idx} length {} exceeds sliding_window {sliding_window}; exact calibration does not approximate local attention",
            sample.len()
        ));
    }
    Ok(())
}

fn layers_per_pass() -> usize {
    std::env::var("HIPFIRE_EMBEDDING_CALIB_LAYERS_PER_PASS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(4)
}

fn pooling_name(mode: PoolingMode) -> &'static str {
    match mode {
        PoolingMode::Mean => "mean",
        PoolingMode::LastToken => "last_token",
        PoolingMode::Cls => "cls",
    }
}

/// Collect exact bidirectional embedding-forward activations into an arch-19
/// calibration sidecar. Every sample is forwarded independently so pooling and
/// Dense-head inputs match serving behavior.
pub fn collect_calibration_artifacts(
    gpu: &mut Gpu,
    weights: &EmbeddingGemmaWeights,
    config: &EmbeddingGemmaConfig,
    samples: &[Vec<u32>],
    output: &std::path::Path,
    provenance: &[(&str, serde_json::Value)],
) -> Result<CalibSummary, String> {
    validate_samples(samples, config.sliding_window)?;

    let mut metadata = provenance.to_vec();
    metadata.push(("task", serde_json::json!("embedding")));
    metadata.push(("bidirectional", serde_json::json!(true)));
    metadata.push((
        "pooling_mode",
        serde_json::json!(pooling_name(config.pooling_mode)),
    ));
    metadata.push(("dense_heads_captured", serde_json::json!(true)));

    collect_embedding_grouped(
        gpu,
        19,
        config.num_hidden_layers,
        layers_per_pass(),
        output,
        samples,
        &metadata,
        |start, end| build_capture_names_for_layers(weights, start, end),
        |gpu, group_idx, _sample_idx, tokens| {
            let hidden = encode_pooled_hidden(gpu, weights, config, tokens)
                .map_err(|e| format!("embeddinggemma calibration forward: {e:?}"))?;
            if group_idx == 0 {
                project_dense_with_capture(&weights.dense_heads, hidden, |head_idx, input| {
                    capture_host_activations(
                        gpu,
                        &format!("dense.{head_idx}"),
                        input,
                        1,
                        input.len(),
                    )
                })?;
            } else {
                project_dense_with_capture(&weights.dense_heads, hidden, |_, _| Ok(()))?;
            }
            Ok(())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_names_cover_backbone_and_dense_heads() {
        let mut names = backbone_tensor_names(0, 2);
        names.extend((0..2).map(|head_idx| format!("dense.{head_idx}")));

        assert!(names.contains(&"model.layers.0.self_attn.q_proj".to_string()));
        assert!(names.contains(&"model.layers.1.self_attn.o_proj".to_string()));
        assert!(names.contains(&"model.layers.0.mlp.gate_proj".to_string()));
        assert!(names.contains(&"model.layers.1.mlp.up_proj".to_string()));
        assert!(names.contains(&"model.layers.1.mlp.down_proj".to_string()));
        assert!(names.contains(&"dense.0".to_string()));
        assert!(names.contains(&"dense.1".to_string()));
        assert_eq!(names.len(), 16);
    }

    #[test]
    fn exact_calibration_rejects_empty_and_over_window_samples() {
        assert!(validate_samples(&[], 4).is_err());
        assert!(validate_samples(&[Vec::new()], 4).is_err());
        let error = validate_samples(&[vec![1, 2, 3, 4, 5]], 4).unwrap_err();
        assert!(error.contains("sample 0 length 5 exceeds sliding_window 4"));
        assert!(validate_samples(&[vec![1, 2], vec![3]], 4).is_ok());
    }
}
