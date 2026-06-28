// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ZAYA1 calibration: generate the LDLQ full-[K,K] Hessian (+ imatrix) sidecar
//! for `oq4++`/`oq8++` quantization. Mirrors qwen35/gemma3: set the shared
//! `CalibCollector` as `gpu.active_capture`, register each dense linear's buffer
//! → HFQ name in `gpu.capture_names`, run the capturing forward
//! ([`crate::gpu::gpu_forward_calib`]) over the calibration tokens, then stream
//! the HFQM `.calib.hfq` the quantizer reads via `--hessian`.
//!
//! First cut: dense projections only (q/k/v/o, router fc/down/out, tied lm_head)
//! get a full Hessian — exactly the LDLQ-eligible set. Routed experts are not
//! captured (they'd be imatrix-only and need a per-expert pass); they fall back
//! to RTN/AWQ at quantize time.

use crate::gpu::{build_capture_names, gpu_forward_calib, ZayaGpuWeights};
use crate::{ZayaConfig, ARCH_ID_ZAYA};
use hipfire_runtime::calibration::CalibCollector;
use rdna_compute::Gpu;
use std::path::Path;

/// Calibration knobs (KLDREF reserved for parity with the other arches; unused
/// in this first cut).
pub struct CalibOpts {
    pub kldref: bool,
    pub kldref_topk: usize,
}

/// Result counts reported by [`collect_calibration_artifacts`].
pub struct CalibSummary {
    pub n_hessian: usize,
    pub n_imatrix: usize,
    pub max_consistency: f32,
}

/// Run the calibration forward over `tokens` and write the HFQM Hessian sidecar
/// to `output`. `provenance` is folded into the sidecar metadata.
pub fn collect_calibration_artifacts(
    gpu: &mut Gpu,
    weights: &ZayaGpuWeights,
    config: &ZayaConfig,
    tokens: &[u32],
    _opts: &CalibOpts,
    output: &Path,
    provenance: &[(&str, serde_json::Value)],
) -> Result<CalibSummary, String> {
    if tokens.is_empty() {
        return Err("zaya calib: empty calibration corpus".to_string());
    }
    // Dense-only capture → every captured tensor wants a full Hessian.
    let collector = std::sync::Arc::new(CalibCollector::new());
    gpu.capture_names = build_capture_names(weights);
    gpu.active_capture = Some(collector.clone());

    let run = gpu_forward_calib(gpu, weights, config, tokens);

    gpu.active_capture = None;
    gpu.capture_names = std::collections::HashMap::new();
    run?;

    let descriptors = collector.tensor_descriptors();
    if descriptors.is_empty() {
        return Err("zaya calib: no tensors captured (check capture_names wiring)".to_string());
    }
    let n_hessian = descriptors.iter().filter(|d| d.has_hessian).count();
    let n_imatrix = descriptors.len();
    let mut per_tensor_tokens = serde_json::Map::new();
    for d in &descriptors {
        per_tensor_tokens.insert(d.name.clone(), serde_json::json!(d.n_tokens));
    }
    let mut meta = serde_json::json!({
        "artifact_kind": "calibration",
        "text_only": true,
        "n_hessian": n_hessian,
        "n_imatrix": n_imatrix,
        "per_tensor_tokens": serde_json::Value::Object(per_tensor_tokens),
        "artifacts": ["hessian", "imatrix"],
    });
    if let Some(obj) = meta.as_object_mut() {
        for (k, v) in provenance {
            obj.insert((*k).to_string(), v.clone());
        }
    }
    let metadata_json = serde_json::to_string(&meta).unwrap();
    let max_consistency = collector
        .write_streaming(gpu, output, ARCH_ID_ZAYA, &metadata_json, &[])
        .map_err(|e| format!("zaya calib write {}: {e}", output.display()))?;
    collector.free_gpu(gpu);

    Ok(CalibSummary {
        n_hessian,
        n_imatrix,
        max_consistency,
    })
}
