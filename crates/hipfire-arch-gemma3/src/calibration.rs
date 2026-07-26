// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3 text-decoder calibration collection. See LICENSE / NOTICE.

use hip_bridge::{HipError, HipResult};
use hipfire_rdna::Gpu;
use hipfire_runtime::calibration::contracts::{
    CalibError, CalibrationJob, KldRefBuilder, KldRefRow, SampleSet,
};
use hipfire_runtime::calibration::residual_probe::ResidualProbe;
use hipfire_runtime::calibration::{collect_grouped, logsumexp, topk_logits, CalibForward};
use hipfire_runtime::llama::HiddenCaptureSink;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weights::WeightTensor;

pub use hipfire_runtime::calibration::CalibSummary;

use crate::config::Gemma3Config;
use crate::forward::{
    embed_token, forward_prefill_batch, forward_step, forward_step_capture, Gemma3State,
};
use crate::weights::Gemma3Weights;

pub use hipfire_runtime::calibration::contracts::KldRefOptions as CalibOpts;

fn put(
    m: &mut std::collections::HashMap<usize, String>,
    wt: &WeightTensor,
    name: impl Into<String>,
) {
    m.insert(wt.buf.buf.as_ptr() as usize, name.into());
}

/// Build the calibration capture map for the Gemma3 text decoder.
///
/// Names match the source HFQ tensor keys without the `.weight` suffix. Pure
/// text Gemma3 uses `prefix=""`; Gemma3-VL text-only collection uses
/// `prefix="language_model."`.
pub fn build_capture_names(
    weights: &Gemma3Weights,
    prefix: &str,
) -> std::collections::HashMap<usize, String> {
    build_capture_names_for_layers(weights, prefix, 0, weights.layers.len())
}

fn build_capture_names_for_layers(
    weights: &Gemma3Weights,
    prefix: &str,
    start_layer: usize,
    end_layer: usize,
) -> std::collections::HashMap<usize, String> {
    let mut m = std::collections::HashMap::new();
    for (i, layer) in weights
        .layers
        .iter()
        .enumerate()
        .skip(start_layer)
        .take(end_layer.saturating_sub(start_layer))
    {
        let p = format!("{prefix}model.layers.{i}");
        put(&mut m, &layer.wq, format!("{p}.self_attn.q_proj"));
        put(&mut m, &layer.wk, format!("{p}.self_attn.k_proj"));
        put(&mut m, &layer.wv, format!("{p}.self_attn.v_proj"));
        put(&mut m, &layer.wo, format!("{p}.self_attn.o_proj"));
        put(&mut m, &layer.w_gate, format!("{p}.mlp.gate_proj"));
        put(&mut m, &layer.w_up, format!("{p}.mlp.up_proj"));
        put(&mut m, &layer.w_down, format!("{p}.mlp.down_proj"));
    }
    m
}

fn layers_per_pass() -> usize {
    std::env::var("HIPFIRE_GEMMA3_CALIB_LAYERS_PER_PASS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(4)
}

fn run_text_forward_for_capture(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    config: &Gemma3Config,
    tokens: &[u32],
    opts: &CalibOpts,
    sample_index: usize,
    include_terminal_kld: bool,
    kldref: Option<&mut KldRefBuilder>,
) -> HipResult<()> {
    // Per-position KLDREF needs lm-head logits at EVERY position, but batched
    // prefill only emits last-position logits — so that path stays per-token.
    // The common case (no KLDREF) runs WIDE batched prefill: each `weight_gemm`
    // linear captures the whole microbatch in a single launch (the `weight_gemm`
    // calibration tap), instead of one N=1 GEMV per token. `HIPFIRE_GEMMA3_CALIB_NO_BATCH=1`
    // forces the legacy per-token path (for bit-for-bit comparison / debugging).
    let want_kldref = kldref.is_some() && opts.kldref;
    let force_per_token = std::env::var("HIPFIRE_GEMMA3_CALIB_NO_BATCH")
        .ok()
        .as_deref()
        == Some("1");
    if want_kldref || force_per_token {
        return run_text_forward_per_token(
            gpu,
            weights,
            config,
            tokens,
            opts,
            sample_index,
            include_terminal_kld,
            kldref,
        );
    }

    let dim = config.hidden_size;
    let microbatch = std::env::var("HIPFIRE_GEMMA3_CALIB_MICROBATCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(128);
    let mut state = Gemma3State::new(gpu, config)
        .map_err(|e| HipError::new(0, &format!("gemma3 calib state: {e}")))?;
    let mut start_pos = 0usize;
    let mut result = Ok(());
    for chunk in tokens.chunks(microbatch) {
        let x_batch = match gpu.alloc_tensor(&[chunk.len() * dim], hipfire_rdna::DType::F32) {
            Ok(t) => t,
            Err(e) => {
                result = Err(e);
                break;
            }
        };
        // Embed each token of the chunk into a contiguous [m, dim] activation,
        // then run the batched prefill forward (taps fire inside `weight_gemm`).
        let r = (|| {
            for (i, &t) in chunk.iter().enumerate() {
                embed_token(gpu, weights, config, &state.x, t)?;
                gpu.memcpy_dtod_at_auto(&x_batch.buf, i * dim * 4, &state.x.buf, 0, dim * 4)?;
            }
            forward_prefill_batch(
                gpu,
                weights,
                config,
                &mut state,
                &x_batch,
                chunk.len(),
                start_pos,
            )
        })();
        let _ = gpu.free_tensor(x_batch);
        if let Err(e) = r {
            result = Err(e);
            break;
        }
        start_pos += chunk.len();
    }
    state.free_gpu(gpu);
    result
}

/// Legacy single-token calibration forward. Retained for the KLDREF path (needs
/// per-position logits) and as a `HIPFIRE_GEMMA3_CALIB_NO_BATCH=1` reference for
/// validating the batched path produces equivalent Hessians/imatrices.
fn run_text_forward_per_token(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    config: &Gemma3Config,
    tokens: &[u32],
    opts: &CalibOpts,
    sample_index: usize,
    include_terminal_kld: bool,
    mut kldref: Option<&mut KldRefBuilder>,
) -> HipResult<()> {
    let mut state = Gemma3State::new(gpu, config)
        .map_err(|e| HipError::new(0, &format!("gemma3 calib state: {e}")))?;
    let mut result = Ok(());
    for (position, &tok) in tokens.iter().enumerate() {
        if let Err(e) = forward_step(gpu, weights, config, &mut state, tok) {
            result = Err(e);
            break;
        }
        if let Some(builder) = kldref
            .as_deref_mut()
            .filter(|_| include_terminal_kld || position + 1 < tokens.len())
        {
            match gpu.download_f32(&state.logits) {
                Ok(lg) => {
                    let topk = topk_logits(&lg, opts.kldref_topk);
                    if let Err(error) = builder.push(KldRefRow {
                        sample_index,
                        position,
                        indices: topk.iter().map(|(index, _)| *index).collect(),
                        logits: topk.iter().map(|(_, logit)| *logit).collect(),
                        log_z: logsumexp(&lg),
                    }) {
                        result = Err(HipError::new(0, &error.to_string()));
                        break;
                    }
                }
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
    }
    state.free_gpu(gpu);
    result
}

/// Collect calibration Hessians/imatrices from the Gemma3 text decoder only.
///
/// For a Gemma3-VL artifact, pass `prefix="language_model."`; vision/projector
/// tensors are not loaded and cannot be captured.
pub fn collect_calibration_artifacts_text_only(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    config: &Gemma3Config,
    _tokenizer: &Tokenizer,
    tokens: &[u32],
    opts: &CalibOpts,
    output: &std::path::Path,
    prefix: &str,
    provenance: &[(&str, serde_json::Value)],
) -> HipResult<CalibSummary> {
    collect_calibration_artifacts_sequences_text_only(
        gpu,
        weights,
        config,
        &[tokens],
        true,
        opts,
        output,
        prefix,
        provenance,
    )
}

/// Resident-oracle Gemma3 collection over the exact independent sample set
/// used by the layer-streamed engine. Every sequence gets fresh KV/state;
/// KLDREF rows exclude terminal positions and retain their sample map.
#[allow(clippy::too_many_arguments)]
pub fn collect_calibration_artifacts_samples_text_only(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    config: &Gemma3Config,
    _tokenizer: &Tokenizer,
    samples: &SampleSet,
    opts: &CalibOpts,
    output: &std::path::Path,
    prefix: &str,
    provenance: &[(&str, serde_json::Value)],
) -> HipResult<CalibSummary> {
    let sequences = samples
        .samples()
        .iter()
        .map(|sample| sample.tokens.as_slice())
        .collect::<Vec<_>>();
    collect_calibration_artifacts_sequences_text_only(
        gpu, weights, config, &sequences, false, opts, output, prefix, provenance,
    )
}

/// Resident Gemma3 parity oracle for a frozen native calibration job.
///
/// The optional residual probe reuses the already-loaded resident weights and
/// captures the same settled post-FFN block boundary as the layer-streamed
/// engine. It is a bounded debug/parity pass, not a second model load.
#[allow(clippy::too_many_arguments)]
pub fn collect_calibration_artifacts_job_text_only_with_residual_probe(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    config: &Gemma3Config,
    tokenizer: &Tokenizer,
    job: &CalibrationJob,
    arch_id: u32,
    output: &std::path::Path,
    prefix: &str,
    provenance: &[(&str, serde_json::Value)],
    residual_probe: Option<(&std::path::Path, usize)>,
) -> HipResult<CalibSummary> {
    let opts = CalibOpts {
        kldref: job.options.kldref,
        kldref_topk: job.options.kldref_top_k,
    };
    let summary = collect_calibration_artifacts_samples_text_only(
        gpu,
        weights,
        config,
        tokenizer,
        &job.samples,
        &opts,
        output,
        prefix,
        provenance,
    )?;
    if let Some((path, max_rows)) = residual_probe {
        collect_residual_probe_job_text_only(gpu, weights, config, job, arch_id, max_rows, path)?;
    }
    Ok(summary)
}

/// Emit a bounded resident full-stack post-layer residual probe for Gemma3.
/// Rows follow the job's canonical sample-major order; each sample receives a
/// fresh state so the resident oracle preserves the streamed engine's sequence
/// independence.
pub fn collect_residual_probe_job_text_only(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    config: &Gemma3Config,
    job: &CalibrationJob,
    arch_id: u32,
    max_rows: usize,
    output: &std::path::Path,
) -> HipResult<()> {
    let family = match arch_id {
        hipfire_model::ARCH_ID_GEMMA3_TEXT => "gemma3",
        hipfire_model::ARCH_ID_GEMMA3_VL => "gemma3-vl",
        _ => {
            return Err(HipError::new(
                0,
                &format!("gemma3 residual probe received unsupported architecture {arch_id}"),
            ));
        }
    };
    let mut probe = ResidualProbe::new(
        arch_id,
        family,
        "resident-full-stack",
        job,
        config.hidden_size,
        config.num_hidden_layers,
        max_rows,
    )
    .map_err(|error| HipError::new(0, &error.to_string()))?;
    let probe_rows = probe.row_count();
    let extract_layers = (0..config.num_hidden_layers).collect::<Vec<_>>();
    let capture_values = probe_rows
        .checked_mul(config.num_hidden_layers)
        .and_then(|values| values.checked_mul(config.hidden_size))
        .ok_or_else(|| HipError::new(0, "gemma3 residual probe shape overflows usize"))?;
    let mut position_major = Vec::with_capacity(capture_values);
    let mut remaining = probe_rows;

    for sample in job.samples.samples() {
        if remaining == 0 {
            break;
        }
        let take = sample.tokens.len().min(remaining);
        let mut state = Gemma3State::new(gpu, config)
            .map_err(|error| HipError::new(0, &format!("gemma3 residual probe state: {error}")))?;
        let result = (|| {
            let mut sink = HiddenCaptureSink {
                extract_layers: &extract_layers,
                hidden: &mut position_major,
                hidden_gpu: None,
            };
            for &token in &sample.tokens[..take] {
                forward_step_capture(gpu, weights, config, &mut state, token, Some(&mut sink))?;
            }
            Ok::<(), HipError>(())
        })();
        state.free_gpu(gpu);
        result?;
        remaining -= take;
    }
    if remaining != 0 {
        return Err(HipError::new(
            0,
            &format!("gemma3 residual probe is missing {remaining} canonical rows"),
        ));
    }

    let layer_values = transpose_position_major_hidden(
        &position_major,
        probe_rows,
        config.num_hidden_layers,
        config.hidden_size,
    )
    .map_err(|error| HipError::new(0, &error.to_string()))?;
    for (layer, values) in layer_values.into_iter().enumerate() {
        probe
            .push_layer(layer, values)
            .map_err(|error| HipError::new(0, &error.to_string()))?;
    }
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|error| HipError::new(0, &error.to_string()))?;
    }
    probe
        .write(output)
        .map_err(|error| HipError::new(0, &error.to_string()))
}

fn transpose_position_major_hidden(
    captured: &[f32],
    rows: usize,
    layers: usize,
    hidden_width: usize,
) -> Result<Vec<Vec<f32>>, CalibError> {
    let row_stride = layers
        .checked_mul(hidden_width)
        .ok_or_else(|| CalibError::InvalidOptions("residual probe shape overflow".into()))?;
    let expected = rows
        .checked_mul(row_stride)
        .ok_or_else(|| CalibError::InvalidOptions("residual probe shape overflow".into()))?;
    if captured.len() != expected {
        return Err(CalibError::Runtime(format!(
            "position-major residual capture has {} values, expected {expected}",
            captured.len()
        )));
    }
    let layer_values = rows
        .checked_mul(hidden_width)
        .ok_or_else(|| CalibError::InvalidOptions("residual probe shape overflow".into()))?;
    let mut result = (0..layers)
        .map(|_| vec![0.0; layer_values])
        .collect::<Vec<_>>();
    for row in 0..rows {
        for (layer, values) in result.iter_mut().enumerate() {
            let src = row * row_stride + layer * hidden_width;
            let dst = row * hidden_width;
            values[dst..dst + hidden_width].copy_from_slice(&captured[src..src + hidden_width]);
        }
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn collect_calibration_artifacts_sequences_text_only(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    config: &Gemma3Config,
    sequences: &[&[u32]],
    include_terminal_kld: bool,
    opts: &CalibOpts,
    output: &std::path::Path,
    prefix: &str,
    provenance: &[(&str, serde_json::Value)],
) -> HipResult<CalibSummary> {
    let mut static_meta: Vec<(&str, serde_json::Value)> = provenance.to_vec();
    static_meta.push(("text_only", serde_json::json!(true)));
    static_meta.push(("text_prefix", serde_json::json!(prefix)));

    // Gemma3 is dense (no MoE) so every captured tensor wants a full Hessian, but
    // all layers at once do not fit — the grouped driver re-runs the forward per
    // layer-group and concatenates the parts. KLDREF is captured once (group 0).
    //
    // `HIPFIRE_GEMMA3_CALIB_IMATRIX_ONLY=1` skips the [K,K] Hessians (every name
    // contains "", so all tensors fall to imatrix-only). This is the fast
    // AWQ-style calibration mode: with no K×K outer-product or multi-GB Hessian
    // write, the forward becomes the bottleneck — which is where the batched
    // prefill tap pays off.
    let imatrix_only = if std::env::var("HIPFIRE_GEMMA3_CALIB_IMATRIX_ONLY")
        .ok()
        .as_deref()
        == Some("1")
    {
        vec![String::new()]
    } else {
        Vec::new()
    };
    collect_grouped(
        gpu,
        0,
        config.num_hidden_layers,
        layers_per_pass(),
        imatrix_only,
        output,
        &static_meta,
        |start, end| build_capture_names_for_layers(weights, prefix, start, end),
        |gpu, group_idx| {
            let n_kld = sequences
                .iter()
                .map(|tokens| {
                    if include_terminal_kld {
                        tokens.len()
                    } else {
                        tokens.len().saturating_sub(1)
                    }
                })
                .sum::<usize>();
            let mut kldref = if group_idx == 0 && opts.kldref && n_kld > 0 {
                Some(KldRefBuilder::new(opts.kldref_topk).map_err(|e| e.to_string())?)
            } else {
                None
            };
            for (sample_index, tokens) in sequences.iter().enumerate() {
                run_text_forward_for_capture(
                    gpu,
                    weights,
                    config,
                    tokens,
                    opts,
                    sample_index,
                    include_terminal_kld,
                    kldref.as_mut(),
                )
                .map_err(|e| format!("gemma3 calib forward: {e}"))?;
            }
            let mut extra_meta: Vec<(String, serde_json::Value)> = Vec::new();
            let extra_tensors = if let Some(payload) = kldref
                .map(KldRefBuilder::finish)
                .transpose()
                .map_err(|e| e.to_string())?
            {
                extra_meta.push(("kldref".to_string(), payload.metadata()));
                if group_idx == 0 {
                    extra_meta.push((
                        "artifacts".to_string(),
                        serde_json::json!(["hessian", "imatrix", "kldref"]),
                    ));
                }
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
    .map_err(|e| HipError::new(0, &e))
}

/// Daemon calibration seam: collect from the already-resident [`Gemma3Backend`]
/// (arch_id 12, dense text). Delegates to the text-only collector with an empty
/// prefix; the resident backend already holds bf16 weights + config. Gemma3-VL
/// (arch_id 13, `Gemma3VlBackend`, `language_model.` prefix) is not yet wired.
impl hipfire_runtime::calibration::CalibratableBackend for crate::arch::Gemma3Backend {
    fn collect_calibration(
        &self,
        gpu: &mut Gpu,
        tokenizer: &Tokenizer,
        tokens: &[u32],
        kldref: bool,
        output: &std::path::Path,
        provenance: &[(&str, serde_json::Value)],
    ) -> Result<CalibSummary, String> {
        let opts = CalibOpts {
            kldref,
            kldref_topk: 64,
        };
        collect_calibration_artifacts_text_only(
            gpu,
            &self.weights,
            &self.config,
            tokenizer,
            tokens,
            &opts,
            output,
            "",
            provenance,
        )
        .map_err(|e| e.to_string())
    }

    fn collect_calibration_job(
        &self,
        gpu: &mut Gpu,
        tokenizer: &Tokenizer,
        job: &hipfire_runtime::calibration::contracts::CalibrationJob,
        output: &std::path::Path,
        provenance: &[(&str, serde_json::Value)],
    ) -> Result<CalibSummary, String> {
        let provenance = hipfire_runtime::calibration::calibration_job_provenance(job, provenance)?;
        let opts = CalibOpts {
            kldref: job.options.kldref,
            kldref_topk: job.options.kldref_top_k,
        };
        collect_calibration_artifacts_samples_text_only(
            gpu,
            &self.weights,
            &self.config,
            tokenizer,
            &job.samples,
            &opts,
            output,
            "",
            &provenance,
        )
        .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::transpose_position_major_hidden;

    #[test]
    fn capture_names_keep_expected_prefixes() {
        assert_eq!(
            format!("{}model.layers.7.self_attn.q_proj", ""),
            "model.layers.7.self_attn.q_proj"
        );
        assert_eq!(
            format!("{}model.layers.7.mlp.down_proj", "language_model."),
            "language_model.model.layers.7.mlp.down_proj"
        );
    }

    #[test]
    fn resident_probe_transposes_position_major_capture_to_layer_major_rows() {
        // Two rows, three layers, two hidden values. `HiddenCaptureSink`
        // appends [row][layer][hidden], while ResidualProbe stores one
        // [row][hidden] tensor per layer.
        let captured = vec![
            0.0, 1.0, 10.0, 11.0, 20.0, 21.0, // row 0
            2.0, 3.0, 12.0, 13.0, 22.0, 23.0, // row 1
        ];

        let layers = transpose_position_major_hidden(&captured, 2, 3, 2).unwrap();

        assert_eq!(layers[0], vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(layers[1], vec![10.0, 11.0, 12.0, 13.0]);
        assert_eq!(layers[2], vec![20.0, 21.0, 22.0, 23.0]);
    }
}
