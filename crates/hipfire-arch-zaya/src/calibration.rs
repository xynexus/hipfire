// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ZAYA1 calibration: generate the LDLQ full-[K,K] Hessian (+ imatrix) sidecar
//! for `oq4++`/`oq8++` quantization. Thin arch driver over the shared
//! [`hipfire_runtime::calibration::collect`]: register each linear's buffer →
//! HFQ name (`build_capture_names`), run the capturing forward
//! ([`crate::gpu::gpu_forward_calib`]), and the general driver streams the HFQM
//! `.calib.hfq`.
//!
//! Dense projections (q/k/v/o, router fc/down/out, tied lm_head) get a full
//! Hessian — exactly the LDLQ-eligible set. Routed experts (matched by
//! `.experts.`) are imatrix-only (Σx², for the AWQ `+` scale — no [K,K]).
//! Experts not selected by the corpus under top-1 routing get no imatrix and
//! fall back to RTN.

use crate::gpu::{build_capture_names, gpu_forward_calib, ZayaGpuWeights};
use crate::{ZayaConfig, ARCH_ID_ZAYA};
use hipfire_runtime::calibration::{collect, CalibForward};
use rdna_compute::Gpu;
use std::path::Path;

pub use hipfire_runtime::calibration::CalibSummary;

/// Calibration knobs (KLDREF reserved for parity with the other arches; unused
/// in this first cut — ZAYA captures Hessian + imatrix only).
pub struct CalibOpts {
    pub kldref: bool,
    pub kldref_topk: usize,
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
    let mut static_meta: Vec<(&str, serde_json::Value)> = provenance.to_vec();
    static_meta.push(("text_only", serde_json::json!(true)));
    collect(
        gpu,
        ARCH_ID_ZAYA,
        build_capture_names(weights),
        vec![".experts.".to_string()],
        output,
        &static_meta,
        |gpu| {
            gpu_forward_calib(gpu, weights, config, tokens)?;
            Ok(CalibForward::default())
        },
    )
}
