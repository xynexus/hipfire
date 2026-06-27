// SPDX-License-Identifier: Apache-2.0
// hipfire — LFM2/LFM2-MoE text calibration collection.

use crate::config::Lfm2MoeConfig;
use crate::forward::decode_step;
use crate::lfm2moe::{Ffn, Lfm2MoeState, Lfm2MoeWeights, Mixer};
use hip_bridge::{HipError, HipResult};
use hipfire_runtime::calibration::{
    finalize_calibration, logsumexp, topk_logits, CalibCollector, CalibSummary,
};
use hipfire_runtime::weights::WeightTensor;
use rdna_compute::Gpu;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Options for [`collect_calibration_artifacts`].
pub struct CalibOpts {
    /// Capture the lm-head top-K logits + logZ per position (KLDREF reference).
    pub kldref: bool,
    pub kldref_topk: usize,
}

impl Default for CalibOpts {
    fn default() -> Self {
        Self {
            kldref: false,
            kldref_topk: 64,
        }
    }
}

fn put(m: &mut HashMap<usize, String>, wt: &WeightTensor, name: impl Into<String>) {
    m.insert(wt.buf.buf.as_ptr() as usize, name.into());
}

/// Build the capture map for LFM2 dense/router projections.
///
/// Names match checkpoint tensor keys without the `.weight` suffix so the
/// quantizer can join `<name>.imatrix`/`<name>.hessian` to source weights.
/// Routed expert weights are captured explicitly in `forward.rs` because the
/// fused indexed kernels do not have one weight pointer per source tensor: gate
/// and up are byte-fused into `gate_up`, but the calibration package needs
/// separate checkpoint-style names for `w1` and `w3`.
pub fn build_capture_names(weights: &Lfm2MoeWeights) -> HashMap<usize, String> {
    let mut m = HashMap::new();
    for (i, layer) in weights.layers.iter().enumerate() {
        let p = format!("model.layers.{i}");
        match &layer.mixer {
            Mixer::Conv(c) => {
                put(&mut m, &c.in_proj, format!("{p}.conv.in_proj"));
                put(&mut m, &c.out_proj, format!("{p}.conv.out_proj"));
            }
            Mixer::Attention(a) => {
                put(&mut m, &a.wq, format!("{p}.self_attn.q_proj"));
                put(&mut m, &a.wk, format!("{p}.self_attn.k_proj"));
                put(&mut m, &a.wv, format!("{p}.self_attn.v_proj"));
                put(&mut m, &a.wo, format!("{p}.self_attn.out_proj"));
            }
        }
        match &layer.ffn {
            Ffn::Dense(d) => {
                put(&mut m, &d.w1, format!("{p}.feed_forward.w1"));
                put(&mut m, &d.w3, format!("{p}.feed_forward.w3"));
                put(&mut m, &d.w2, format!("{p}.feed_forward.w2"));
            }
            Ffn::Moe(moe) => {
                put(&mut m, &moe.router, format!("{p}.feed_forward.gate"));
            }
        }
    }
    m
}

/// Collect calibration Hessians/imatrices from the LFM2 text decoder.
///
/// This covers dense projection calls that route through `weight_gemv` plus
/// calibration-only taps around the indexed routed-expert kernels. Dense/router
/// tensors get full Hessians; routed expert tensors are imatrix-only because
/// full per-expert Hessians do not fit for the 8B-A1B model.
pub fn collect_calibration_artifacts(
    gpu: &mut Gpu,
    weights: &Lfm2MoeWeights,
    config: &Lfm2MoeConfig,
    tokens: &[u32],
    opts: &CalibOpts,
    output: &Path,
    provenance: &[(&str, serde_json::Value)],
) -> HipResult<CalibSummary> {
    let collector = Arc::new(CalibCollector::with_imatrix_only(vec![
        ".feed_forward.experts.".to_string(),
    ]));
    gpu.capture_names = build_capture_names(weights);
    gpu.active_capture = Some(collector.clone());

    let mut state = Lfm2MoeState::new(gpu, config)
        .map_err(|e| HipError::new(0, &format!("lfm2 calib state: {e}")))?;
    let mut kldref: Vec<(f32, Vec<(u32, f32)>)> = Vec::new();
    let mut run_result: HipResult<()> = Ok(());
    for (pos, &tok) in tokens.iter().enumerate() {
        match decode_step(config, weights, &mut state, gpu, tok, pos as u32) {
            Ok(logits) => {
                if opts.kldref {
                    kldref.push((logsumexp(&logits), topk_logits(&logits, opts.kldref_topk)));
                }
            }
            Err(e) => {
                run_result = Err(HipError::new(0, &format!("lfm2 calib decode: {e}")));
                break;
            }
        }
    }
    gpu.active_capture = None;
    gpu.capture_names = HashMap::new();
    run_result?;

    // The arch-agnostic descriptor counts, KLDREF block, provenance merge, and
    // streaming write are shared with every other arch collector.
    let base_meta = serde_json::json!({
        "arch": "lfm2",
        "text_only": true,
        "captures": "decode_step_weight_gemv+routed_expert_indexed_tap",
        "routed_expert_capture": "imatrix-only-selected-experts",
    });
    finalize_calibration(
        &collector,
        gpu,
        output,
        crate::ARCH_ID,
        &kldref,
        base_meta,
        provenance,
    )
    .map_err(|e| HipError::new(0, &format!("lfm2 calib: {e}")))
}

#[cfg(test)]
mod tests {
    #[test]
    fn lfm2_calibration_name_examples_match_checkpoint_keys() {
        assert_eq!(
            format!("model.layers.7.conv.in_proj"),
            "model.layers.7.conv.in_proj"
        );
        assert_eq!(
            format!("model.layers.7.feed_forward.gate"),
            "model.layers.7.feed_forward.gate"
        );
    }
}
