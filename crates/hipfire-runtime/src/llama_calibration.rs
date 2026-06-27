// SPDX-License-Identifier: Apache-2.0
// hipfire — LLaMA / Mistral / plain-Qwen text calibration collection.
//
//! Calibration collector for the dense LLaMA family (`arch_id = 0` for
//! LLaMA/Mistral, `arch_id = 1` for plain Qwen3/Qwen2). It drives the
//! single-token [`crate::llama::forward`] decode loop with the generic
//! [`crate::calibration::CalibCollector`] armed, so every dense projection's
//! input activation is captured at the `weight_gemv` chokepoint, then streams a
//! `<model>.calib.hfq` bundling the per-tensor Hessian + imatrix (+ KLDREF).
//!
//! Unlike the qwen35/gemma3/lfm2 collectors — which live in their arch crates
//! because those forward paths physically moved out of the runtime — the LLaMA
//! forward body *is* the shared transformer infrastructure in
//! `crates/hipfire-runtime/src/llama.rs`, so its collector is hosted here next
//! to it and reuses [`crate::calibration::finalize_calibration`] for the
//! arch-agnostic metadata + streaming-write boilerplate.

use crate::calibration::{
    finalize_calibration, logsumexp, topk_logits, CalibCollector, CalibSummary,
};
use crate::llama::{self, LlamaConfig, LlamaWeights};
use crate::weights::WeightTensor;
use hip_bridge::HipResult;
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

/// Build the capture map (weight buffer ptr -> checkpoint tensor name) for the
/// dense LLaMA projections.
///
/// Names match the HFQ tensor keys without the `.weight` suffix so the
/// quantizer joins `<name>.imatrix`/`<name>.hessian` to the source weights:
/// `model.layers.{i}.self_attn.{q,k,v,o}_proj` and
/// `model.layers.{i}.mlp.{gate,up,down}_proj`. The lm-head (`output`) is not
/// captured for a Hessian — like the other arch collectors it is KLDREF-only.
pub fn build_capture_names(weights: &LlamaWeights) -> HashMap<usize, String> {
    let mut m = HashMap::new();
    let mut put = |wt: &WeightTensor, name: String| {
        m.insert(wt.buf.buf.as_ptr() as usize, name);
    };
    for (i, layer) in weights.layers.iter().enumerate() {
        let p = format!("model.layers.{i}");
        put(&layer.wq, format!("{p}.self_attn.q_proj"));
        put(&layer.wk, format!("{p}.self_attn.k_proj"));
        put(&layer.wv, format!("{p}.self_attn.v_proj"));
        put(&layer.wo, format!("{p}.self_attn.o_proj"));
        put(&layer.w_gate, format!("{p}.mlp.gate_proj"));
        put(&layer.w_up, format!("{p}.mlp.up_proj"));
        put(&layer.w_down, format!("{p}.mlp.down_proj"));
    }
    m
}

/// Collect calibration Hessians/imatrices from the dense LLaMA-family decoder.
///
/// Every dense projection routes through `weight_gemv`, so all seven per-layer
/// projections get full Hessians (this family has no routed MoE experts). The
/// pass runs on the bf16/q8 reference model passed in `weights`.
pub fn collect_calibration_artifacts(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tokens: &[u32],
    opts: &CalibOpts,
    output: &Path,
    provenance: &[(&str, serde_json::Value)],
) -> HipResult<CalibSummary> {
    let collector = Arc::new(CalibCollector::new());
    gpu.capture_names = build_capture_names(weights);
    gpu.active_capture = Some(collector.clone());

    let mut kv = crate::kv::KvCache::new_gpu(
        gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        tokens.len() + 16,
    )?;

    let mut kldref: Vec<(f32, Vec<(u32, f32)>)> = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        let logits = llama::forward(gpu, weights, config, tok, pos, &mut kv)?;
        if opts.kldref {
            kldref.push((logsumexp(&logits), topk_logits(&logits, opts.kldref_topk)));
        }
    }
    gpu.active_capture = None;
    gpu.capture_names = HashMap::new();
    kv.free_gpu(gpu);

    let base_meta = serde_json::json!({
        "arch": "llama",
        "text_only": true,
        "captures": "forward_weight_gemv",
    });
    finalize_calibration(
        &collector,
        gpu,
        output,
        hipfire_model::ARCH_ID_LLAMA_MISTRAL,
        &kldref,
        base_meta,
        provenance,
    )
    .map_err(|e| hip_bridge::HipError::new(0, &format!("llama calib: {e}")))
}
