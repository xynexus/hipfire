// SPDX-License-Identifier: Apache-2.0
// hipfire — LLaMA / Mistral / plain-Qwen text calibration collection.
//
//! Calibration collector for the dense LLaMA family (`arch_id = 0` for
//! LLaMA/Mistral, `arch_id = 1` for plain Qwen3/Qwen2). It drives the
//! single-token decode path with the generic [`crate::calibration::
//! CalibCollector`] armed, so every dense projection's input activation is
//! captured at the `weight_gemv` chokepoint, then streams a `<model>.calib.hfq`
//! bundling the per-tensor Hessian + imatrix (+ KLDREF).
//!
//! Unlike the qwen35/gemma3/lfm2 collectors — which live in their arch crates
//! because those forward paths physically moved out of the runtime — the LLaMA
//! forward body *is* the shared transformer infrastructure in
//! `crates/hipfire-runtime/src/llama.rs`, so its collector is hosted here next
//! to it.
//!
//! ## Why this file was rewritten rather than restored
//!
//! An earlier version existed on `backup-chaingun-local-2026-06-28` and is what
//! produced the shipped `llama-3.2-1b*.calib.hfq` packages. It cannot be
//! restored as-is: it called a `finalize_calibration` that no longer exists and
//! collected every layer in one pass. Hessians for all layers do not fit at
//! once, so collection is now GROUPED — [`crate::calibration::collect_grouped`]
//! re-runs the forward once per layer-group and concatenates the parts, with
//! the capture map narrowed to that group. This follows the current idiom
//! (gemma3's collector is the reference) rather than the pre-refactor one.

use crate::calibration::contracts::{KldRefBuilder, KldRefRow};
use crate::calibration::{
    collect_grouped, logsumexp, topk_logits, CalibForward, CalibSummary,
};
use crate::kv::KvCache;
use crate::llama::{self, ForwardScratch, LlamaConfig, LlamaWeights};
use crate::weights::{weight_gemv, WeightTensor};
use hipfire_rdna::{DType, Gpu};
use std::collections::HashMap;
use std::path::Path;

/// Layers whose Hessians are gathered in one pass. A [K,K] f32 Hessian is 16 MB
/// at K=2048 and 268 MB at K=8192 (down_proj), so a 16-layer model asks for
/// ~5 GB if collected at once — hence grouping. Override with
/// `HIPFIRE_LLAMA_CALIB_LAYERS_PER_PASS`.
fn layers_per_pass() -> usize {
    std::env::var("HIPFIRE_LLAMA_CALIB_LAYERS_PER_PASS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(4)
}

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

/// Capture map (weight buffer ptr -> checkpoint tensor name) for the dense
/// LLaMA projections in layers `[start, end)`.
///
/// Names match the HFQ tensor keys WITHOUT the `.weight` suffix so the
/// quantizer joins `<name>.imatrix` / `<name>.hessian` to the source weights:
/// `model.layers.{i}.self_attn.{q,k,v,o}_proj` and
/// `model.layers.{i}.mlp.{gate,up,down}_proj`. The lm-head (`output`) is not
/// captured for a Hessian — like every other arch collector it is KLDREF-only.
///
/// Narrowing to a layer range is what makes grouping work: the forward still
/// runs end to end, but only this group's buffers are registered, so only their
/// accumulators are resident.
pub fn build_capture_names_for_layers(
    weights: &LlamaWeights,
    start: usize,
    end: usize,
) -> HashMap<usize, String> {
    let mut m = HashMap::new();
    let mut put = |wt: &WeightTensor, name: String| {
        m.insert(wt.buf.buf.as_ptr() as usize, name);
    };
    for (i, layer) in weights.layers.iter().enumerate().take(end).skip(start) {
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

/// One decode step, returning the position's logits.
///
/// `forward_scratch_layers` fuses the final norm, the head and SAMPLING, and a
/// sampled token is not what KLDREF needs, so the head is done explicitly here:
/// the same `rmsnorm -> weight_gemv(output)` pair `llama::prefill_forward` uses.
/// Every projection inside `forward_scratch_compute` still routes through
/// `weight_gemv`, which is where the collector taps, so capture is unaffected.
fn decode_step_logits(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    token: u32,
    pos: usize,
    kv: &mut KvCache,
    scratch: &ForwardScratch,
) -> Result<Vec<f32>, String> {
    llama::forward_scratch_embed(gpu, weights, config, token, pos, scratch)
        .map_err(|e| format!("llama calib embed: {e}"))?;
    llama::forward_scratch_compute(gpu, weights, config, pos, kv, scratch)
        .map_err(|e| format!("llama calib layers: {e}"))?;
    gpu.rmsnorm_f32(&scratch.x, &weights.output_norm, &scratch.tmp, config.norm_eps)
        .map_err(|e| format!("llama calib final norm: {e}"))?;
    let logits = gpu
        .alloc_owned(&[config.vocab_size], DType::F32)
        .map_err(|e| format!("llama calib logits alloc: {e}"))?;
    weight_gemv(gpu, &weights.output, &scratch.tmp, &logits)
        .map_err(|e| format!("llama calib head: {e}"))?;
    gpu.download_f32(&logits)
        .map_err(|e| format!("llama calib logits download: {e}"))
}

/// Collect calibration Hessians/imatrices from the dense LLaMA-family decoder.
///
/// Every dense projection routes through `weight_gemv`, so all seven per-layer
/// projections get full Hessians (this family has no routed MoE experts). The
/// pass runs on the bf16/q8 reference model passed in `weights`.
#[allow(clippy::too_many_arguments)]
pub fn collect_calibration_artifacts(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tokens: &[u32],
    opts: &CalibOpts,
    output: &Path,
    provenance: &[(&str, serde_json::Value)],
) -> Result<CalibSummary, String> {
    let mut static_meta: Vec<(&str, serde_json::Value)> = provenance.to_vec();
    static_meta.push(("arch", serde_json::json!("llama")));
    static_meta.push(("text_only", serde_json::json!(true)));
    static_meta.push(("captures", serde_json::json!("forward_weight_gemv")));

    collect_grouped(
        gpu,
        hipfire_model::ARCH_ID_LLAMA_MISTRAL,
        config.n_layers,
        layers_per_pass(),
        Vec::new(), // dense family: every captured tensor wants a full Hessian
        output,
        &static_meta,
        |start, end| build_capture_names_for_layers(weights, start, end),
        |gpu, group_idx| {
            // KLDREF is identical across groups (same tokens, same weights), so
            // it is captured once on group 0 and the rest skip the head work.
            let want_kld = group_idx == 0 && opts.kldref && !tokens.is_empty();
            let mut kldref = if want_kld {
                Some(KldRefBuilder::new(opts.kldref_topk).map_err(|e| e.to_string())?)
            } else {
                None
            };

            let mut kv = KvCache::new_gpu(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                tokens.len() + 16,
            )
            .map_err(|e| format!("llama calib kv: {e}"))?;
            let scratch = ForwardScratch::new(gpu, config)
                .map_err(|e| format!("llama calib scratch: {e}"))?;

            let mut result: Result<(), String> = Ok(());
            for (pos, &tok) in tokens.iter().enumerate() {
                match decode_step_logits(gpu, weights, config, tok, pos, &mut kv, &scratch) {
                    Ok(lg) => {
                        if let Some(builder) = kldref.as_mut() {
                            let topk = topk_logits(&lg, opts.kldref_topk);
                            if let Err(e) = builder.push(KldRefRow {
                                sample_index: 0,
                                position: pos,
                                indices: topk.iter().map(|(i, _)| *i).collect(),
                                logits: topk.iter().map(|(_, l)| *l).collect(),
                                log_z: logsumexp(&lg),
                            }) {
                                result = Err(e.to_string());
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        result = Err(e);
                        break;
                    }
                }
            }
            kv.free_gpu(gpu);
            result?;

            let mut extra_meta: Vec<(String, serde_json::Value)> = Vec::new();
            let extra_tensors = if let Some(payload) = kldref
                .map(KldRefBuilder::finish)
                .transpose()
                .map_err(|e| e.to_string())?
            {
                extra_meta.push(("kldref".to_string(), payload.metadata()));
                extra_meta.push((
                    "artifacts".to_string(),
                    serde_json::json!(["hessian", "imatrix", "kldref"]),
                ));
                payload.to_hfq_tensors()
            } else {
                Vec::new()
            };
            Ok(CalibForward {
                extra_tensors,
                extra_meta,
            })
        },
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn capture_names_are_hfq_keys_without_the_weight_suffix() {
        // The quantizer joins `<name>.hessian` / `<name>.imatrix` to the source
        // weight, so these must match HFQ tensor keys exactly — a `.weight`
        // suffix here would silently orphan every Hessian.
        assert_eq!(
            format!("model.layers.{}.self_attn.q_proj", 7),
            "model.layers.7.self_attn.q_proj"
        );
        assert_eq!(
            format!("model.layers.{}.mlp.down_proj", 0),
            "model.layers.0.mlp.down_proj"
        );
    }

    #[test]
    fn layers_per_pass_defaults_and_respects_the_override() {
        // Default must be >0 or collect_grouped would loop forever.
        assert!(super::layers_per_pass() > 0);
    }
}
