// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ZAYA1 calibration: generate the LDLQ full-[K,K] Hessian (+ imatrix) sidecar
//! for `oq4++`/`oq8++` quantization, plus an optional KLDREF reference. Thin
//! arch driver over the shared [`hipfire_runtime::calibration::collect`]:
//! register each linear's buffer → HFQ name (`build_capture_names`), run the
//! capturing forward ([`crate::gpu::gpu_forward_calib`]), and the general driver
//! streams the HFQM `.calib.hfq`.
//!
//! Dense projections (q/k/v/o, router fc/down/out, tied lm_head) get a full
//! Hessian — exactly the LDLQ-eligible set. Routed experts (matched by
//! `.experts.`) are imatrix-only (Σx², for the AWQ `+` scale — no [K,K]).
//! Experts not selected by the corpus under top-1 routing get no imatrix and
//! fall back to RTN. With `--kldref`, the capturing forward also taps the tied
//! lm-head logits and bakes the per-position top-k reference into the sidecar.

use crate::gpu::{build_capture_names, gpu_forward_calib, ZayaGpuWeights};
use crate::{ZayaConfig, ARCH_ID_ZAYA};
use hipfire_rdna::Gpu;
use hipfire_runtime::calibration::contracts::legacy_kldref_tensors;
use hipfire_runtime::calibration::{collect, CalibForward};
use hipfire_runtime::hfq::HfqMemTensor;
use std::path::Path;

pub use hipfire_runtime::calibration::contracts::KldRefOptions as CalibOpts;
pub use hipfire_runtime::calibration::CalibSummary;

/// Pack the per-position `(logZ, top-k)` KLDREF reference into HFQ tensors
/// (`lm_head.kldref_{idx,logit,logz}`), matching the other arches' layout.
fn kldref_extra(kldref: &[(f32, Vec<(u32, f32)>)]) -> Vec<HfqMemTensor> {
    legacy_kldref_tensors(kldref).expect("internally generated Zaya KLDREF rows are valid")
}

/// Run the calibration forward over `tokens` and write the HFQM Hessian sidecar
/// to `output`. `provenance` is folded into the sidecar metadata.
pub fn collect_calibration_artifacts(
    gpu: &mut Gpu,
    weights: &ZayaGpuWeights,
    config: &ZayaConfig,
    tokens: &[u32],
    opts: &CalibOpts,
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
            // Attention is O(seq²), so ONE long calibration sequence makes the
            // capture superlinear in the token budget: 8192 -> 32768 tokens
            // measured 747s -> 10659s, 14.3x for 4x the tokens. Splitting the
            // stream into independent sequences makes it linear, and the
            // Hessian is a sum of per-row outer products, so it does not care
            // whether the rows came from one context or many. The KLD
            // reference is built at n_ctx=2048 anyway, so shorter calibration
            // sequences match the evaluation distribution rather than diverge
            // from it. Unset keeps the historical single-sequence behaviour.
            let seq_len = hipfire_env::CALIB_SEQ_LEN
                .parse::<usize>()
                .filter(|n| *n >= 2)
                .unwrap_or(tokens.len().max(2));
            let mut kldref: Vec<(f32, Vec<(u32, f32)>)> = Vec::new();
            let n_seq = tokens.len().div_ceil(seq_len);
            for (i, chunk) in tokens.chunks(seq_len).enumerate() {
                // A 1-token sequence has no next-token target and would only
                // contribute a degenerate row.
                if chunk.len() < 2 {
                    continue;
                }
                if n_seq > 1 {
                    eprintln!(
                        "  calib sequence {}/{n_seq} ({} tokens)",
                        i + 1,
                        chunk.len()
                    );
                }
                kldref.extend(gpu_forward_calib(
                    gpu,
                    weights,
                    config,
                    chunk,
                    opts.kldref.then_some(opts.kldref_topk),
                )?);
            }
            let extra_tensors = kldref_extra(&kldref);
            let mut extra_meta: Vec<(String, serde_json::Value)> = Vec::new();
            if !kldref.is_empty() {
                let np = kldref.len();
                let kk = kldref[0].1.len();
                extra_meta.push((
                    "kldref".to_string(),
                    serde_json::json!({ "n_positions": np, "top_k": kk }),
                ));
                extra_meta.push((
                    "artifacts".to_string(),
                    serde_json::json!(["hessian", "imatrix", "kldref"]),
                ));
            }
            Ok(CalibForward {
                extra_tensors,
                extra_meta,
            })
        },
    )
}

/// Daemon calibration seam: collect from the already-resident [`ZayaModel`] by
/// delegating to [`collect_calibration_artifacts`] over its GPU-resident weights.
impl hipfire_runtime::calibration::CalibratableBackend for crate::arch::ZayaModel {
    fn collect_calibration(
        &self,
        gpu: &mut Gpu,
        _tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
        tokens: &[u32],
        kldref: bool,
        output: &Path,
        provenance: &[(&str, serde_json::Value)],
    ) -> Result<CalibSummary, String> {
        let opts = CalibOpts {
            kldref,
            kldref_topk: 64,
        };
        collect_calibration_artifacts(
            gpu,
            self.weights(),
            self.config(),
            tokens,
            &opts,
            output,
            provenance,
        )
    }
}
