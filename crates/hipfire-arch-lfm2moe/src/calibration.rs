// SPDX-License-Identifier: Apache-2.0
// hipfire — LFM2/LFM2-MoE text calibration collection.

use crate::config::Lfm2MoeConfig;
use crate::forward::decode_step;
use crate::lfm2moe::{Ffn, Lfm2MoeState, Lfm2MoeWeights, Mixer};
use hip_bridge::{HipError, HipResult};
use hipfire_rdna::Gpu;
use hipfire_runtime::calibration::contracts::legacy_kldref_tensors;
use hipfire_runtime::calibration::{collect, logsumexp, topk_logits, CalibForward};
use hipfire_runtime::hfq::HfqMemTensor;
use hipfire_runtime::weights::WeightTensor;
use std::collections::HashMap;
use std::path::Path;

pub use hipfire_runtime::calibration::contracts::KldRefOptions as CalibOpts;
pub use hipfire_runtime::calibration::CalibSummary;

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

fn kldref_extra(kldref: &[(f32, Vec<(u32, f32)>)]) -> Vec<HfqMemTensor> {
    legacy_kldref_tensors(kldref).expect("internally generated LFM2 KLDREF rows are valid")
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
    let mut static_meta: Vec<(&str, serde_json::Value)> = provenance.to_vec();
    static_meta.push(("arch", serde_json::json!("lfm2")));
    static_meta.push(("text_only", serde_json::json!(true)));
    static_meta.push((
        "captures",
        serde_json::json!("decode_step_weight_gemv+routed_expert_indexed_tap"),
    ));
    static_meta.push((
        "routed_expert_capture",
        serde_json::json!("imatrix-only-selected-experts"),
    ));

    // Routed experts are imatrix-only (full per-expert Hessians do not fit for
    // the 8B-A1B); the decode loop also taps the lm-head logits for KLDREF.
    collect(
        gpu,
        crate::ARCH_ID,
        build_capture_names(weights),
        vec![".feed_forward.experts.".to_string()],
        output,
        &static_meta,
        |gpu| {
            let mut state =
                Lfm2MoeState::new(gpu, config).map_err(|e| format!("lfm2 calib state: {e}"))?;
            let mut kldref: Vec<(f32, Vec<(u32, f32)>)> = Vec::new();
            for (pos, &tok) in tokens.iter().enumerate() {
                let logits = decode_step(config, weights, &mut state, gpu, tok, pos as u32)
                    .map_err(|e| format!("lfm2 calib decode: {e}"))?;
                if opts.kldref {
                    kldref.push((logsumexp(&logits), topk_logits(&logits, opts.kldref_topk)));
                }
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
    .map_err(|e| HipError::new(0, &e))
}

/// Thin `&weights`/`&config` adapter so the daemon's loose LFM2 resident slots
/// (`lfm2moe_weights`/`lfm2moe_config`) satisfy the calibration seam without a
/// bundled backend type — mirrors `qwen35::Qwen35CalibBackend`.
pub struct Lfm2MoeCalibBackend<'a> {
    pub weights: &'a Lfm2MoeWeights,
    pub config: &'a Lfm2MoeConfig,
}

impl hipfire_runtime::calibration::CalibratableBackend for Lfm2MoeCalibBackend<'_> {
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
            self.weights,
            self.config,
            tokens,
            &opts,
            output,
            provenance,
        )
        .map_err(|e| e.to_string())
    }
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
