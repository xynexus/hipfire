#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// hipfire — Tier-1 native single-load artifact collector (thin CLI).
//
//! Loads a bf16 `.hfq` once and runs the matching arch collector, writing a
//! unified `<model>.calib.hfq` bundling the per-tensor Hessian + imatrix (+
//! MoE router histogram for MoE models, + KLDREF with `--kldref`). Gemma3-VL
//! (`arch_id=13`) is collected text-only through the `language_model.` prefix.
//! EmbeddingGemma (`arch_id=19`) preserves independent corpus samples and
//! captures its bidirectional backbone plus host-side Dense projection heads.
//!
//! Run:
//!   cargo run --release -p hipfire-runtime --example collect_artifacts -- \
//!     --model ~/.hipfire/models/qwen3.5-0.8b-bf16.hfq \
//!     --corpus benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt \
//!     --output /tmp/qwen3.5-0.8b.calib.hfq --max-tokens 256 [--kldref] \
//!     [--job-from /tmp/streamed.calib.hfq \
//!      --residual-probe-output /tmp/resident.residuals.hfq \
//!      --residual-probe-rows 16]

use hipfire_arch_embeddinggemma::{self as embeddinggemma, calibration as embeddinggemma_calib};
use hipfire_arch_gemma3::calibration as gemma3_calib;
use hipfire_arch_gemma3::weights as gemma3_weights;
use hipfire_arch_gemma3::{self as gemma3};
use hipfire_arch_lfm2moe::calibration as lfm2_calib;
use hipfire_arch_lfm2moe::{Lfm2MoeConfig, Lfm2MoeWeights};
use hipfire_arch_minimax::calibration as minimax_calib;
use hipfire_arch_minimax::{MiniMaxConfig, MiniMaxWeights};
use hipfire_arch_nemotron::{calibration as nemotron_calib, model::NemotronModel, NemotronHConfig};
use hipfire_arch_qwen35::qwen35::{self, CalibOpts as QwenCalibOpts};
use hipfire_arch_zaya::{calibration as zaya_calib, ZayaConfig};
use hipfire_rdna::Gpu;
use hipfire_runtime::calibration::contracts::CalibrationJob;
use hipfire_runtime::calibration::{collect_qwen3_embedding_artifacts, tokenize_embedding_samples};
use std::path::Path;

fn arg(flag: &str, default: Option<String>) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == flag)
        .and_then(|i| a.get(i + 1).cloned())
        .or(default)
}

struct ResidentParityJob {
    job: CalibrationJob,
    arch_id: u32,
    family: String,
    artifact: String,
}

fn load_resident_parity_job(path: &str) -> ResidentParityJob {
    let hfq = hipfire_runtime::hfq::HfqFile::open_index_only(Path::new(path))
        .expect("open --job-from calibration artifact");
    let metadata: serde_json::Value =
        serde_json::from_str(&hfq.metadata_json).expect("parse --job-from metadata");
    assert_eq!(
        metadata
            .get("artifact_kind")
            .and_then(|value| value.as_str()),
        Some("calibration"),
        "--job-from must name a completed calibration artifact"
    );
    let job = serde_json::from_value(
        metadata
            .get("job")
            .cloned()
            .expect("--job-from calibration artifact has no native job contract"),
    )
    .expect("parse --job-from native calibration job");
    let family = metadata
        .get("family")
        .and_then(|value| value.as_str())
        .expect("--job-from calibration artifact has no family")
        .to_string();
    ResidentParityJob {
        job,
        arch_id: hfq.arch_id,
        family,
        artifact: path.to_string(),
    }
}

fn main() {
    let model = arg("--model", None).expect("--model required");
    // Optional under `--synthetic-tokens` (no corpus is read in that mode).
    let corpus = arg("--corpus", Some(String::new())).unwrap();
    let output = arg("--output", Some("/tmp/native.calib.hfq".into())).unwrap();
    let max_tokens: usize = arg("--max-tokens", Some("512".into()))
        .unwrap()
        .parse()
        .unwrap();
    let parity_job = arg("--job-from", None).map(|path| load_resident_parity_job(&path));
    let residual_probe_output = arg("--residual-probe-output", None);
    let residual_probe_rows: usize = arg("--residual-probe-rows", Some("16".into()))
        .unwrap()
        .parse()
        .expect("--residual-probe-rows must be an integer");
    assert!(
        residual_probe_rows > 0,
        "--residual-probe-rows must be nonzero"
    );
    assert!(
        residual_probe_output.is_none() || parity_job.is_some(),
        "--residual-probe-output requires --job-from so row provenance is exact"
    );
    let cli_kldref = std::env::args().any(|a| a == "--kldref");
    let want_kldref = parity_job
        .as_ref()
        .map(|parity| parity.job.options.kldref)
        .unwrap_or(cli_kldref);
    // Tiny seeded fixtures (`hipfire-quantize --emit-fixture <family>`) carry a
    // synthetic tokenizer with no real `model`, so the corpus-encode path can't
    // run. `--synthetic-tokens` skips the tokenizer and feeds seeded random ids
    // in `[0, vocab)` — enough to exercise the capturing forward + streamed
    // collector for pipeline validation. Real models still use `--corpus`.
    let synthetic = std::env::args().any(|a| a == "--synthetic-tokens");
    assert!(
        !(synthetic && parity_job.is_some()),
        "--job-from conflicts with --synthetic-tokens"
    );
    if cli_kldref {
        if let Some(parity) = &parity_job {
            assert!(
                parity.job.options.kldref,
                "--kldref conflicts with the no-KLD native job in --job-from"
            );
        }
    }
    let seed: u64 = arg("--seed", Some("0".into())).unwrap().parse().unwrap();

    let mut hfq = hipfire_runtime::hfq::HfqFile::open(Path::new(&model)).expect("open model");
    // `--arch <id>` overrides the hfq's stored arch_id. Needed for hfqs that
    // predate proper arch tagging (e.g. some qwen3 MoE bf16 hfqs are stamped
    // arch_id=0/llama but load fine through the qwen35 backend at 5/6).
    let source_arch_id = arg("--arch", None)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(hfq.arch_id);
    if let Some(parity) = &parity_job {
        assert_eq!(
            source_arch_id, parity.arch_id,
            "resident model architecture differs from --job-from artifact"
        );
        assert!(
            parity.job.samples.samples().len() == 1
                || matches!(source_arch_id, 5 | 6 | 12 | 13),
            "--job-from has multiple independent samples, but this resident family has no state-reset oracle"
        );
    }
    assert!(
        residual_probe_output.is_none() || matches!(source_arch_id, 5 | 6),
        "resident residual probes are currently implemented for Qwen3.5 arch 5/6"
    );

    let embedding_metadata =
        hipfire_model::embedding::EmbeddingMetadata::from_hfq_metadata_json(&hfq.metadata_json)
            .expect("embedding metadata");
    let is_embedding_workload =
        source_arch_id == 19 || (source_arch_id == 1 && embedding_metadata.is_some());

    if is_embedding_workload && synthetic {
        panic!("embedding calibration requires --corpus sample boundaries; --synthetic-tokens is unsupported");
    }
    if is_embedding_workload && want_kldref {
        panic!("embedding calibration does not produce autoregressive KLDREF artifacts");
    }

    // Loaded lazily — synthetic mode has no usable tokenizer; only the gemma3
    // text-only arm actually consumes it (asserts Some below).
    let tokenizer: Option<hipfire_runtime::tokenizer::Tokenizer> = if synthetic {
        None
    } else {
        Some(
            hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                .expect("tok"),
        )
    };

    // Embedding calibration prepends the SAME document_prompt that `embed_forward`
    // applies at inference (config document_prompt, e.g. "title: none | text: "),
    // so the tapped activations match the served distribution instead of bare
    // text. Opt out with `--no-calib-prompt` for a legacy unprompted calibration.
    let calib_apply_prompt =
        is_embedding_workload && !std::env::args().any(|a| a == "--no-calib-prompt");
    let calib_doc_prompt: String = if calib_apply_prompt {
        embedding_metadata
            .as_ref()
            .map(|metadata| {
                metadata
                    .prompt(hipfire_model::embedding::EmbeddingInputType::Document)
                    .to_string()
            })
            .unwrap_or_else(|| {
                embeddinggemma::config_from_metadata_json(&hfq.metadata_json)
                    .expect("embeddinggemma config for calibration prompt")
                    .document_prompt
            })
    } else {
        String::new()
    };
    let embedding_samples: Vec<Vec<u32>> = if is_embedding_workload {
        let text = std::fs::read_to_string(&corpus).expect("read embedding corpus as UTF-8");
        if calib_apply_prompt {
            eprintln!("embedding calibration: prepending document_prompt {calib_doc_prompt:?}");
        } else {
            eprintln!("embedding calibration: --no-calib-prompt (unprompted samples)");
        }
        let samples = tokenize_embedding_samples(&text, max_tokens, |sample| {
            let prompted;
            let s: &str = if calib_apply_prompt {
                prompted = format!("{calib_doc_prompt}{sample}");
                &prompted
            } else {
                sample
            };
            tokenizer.as_ref().unwrap().encode(s)
        });
        if samples.is_empty() {
            panic!(
                "EmbeddingGemma corpus produced no complete non-empty samples within --max-tokens"
            );
        }
        samples
    } else {
        Vec::new()
    };

    let tokens_owned: Vec<u32> = if is_embedding_workload {
        Vec::new()
    } else if let Some(parity) = &parity_job {
        if parity.job.samples.samples().len() == 1 {
            parity.job.samples.samples()[0].tokens.clone()
        } else {
            Vec::new()
        }
    } else if synthetic {
        // Parse vocab_size from the hfq metadata (flat or under `config`).
        let meta: serde_json::Value =
            serde_json::from_str(&hfq.metadata_json).expect("metadata json");
        let vocab = meta
            .get("vocab_size")
            .or_else(|| meta.get("config").and_then(|c| c.get("vocab_size")))
            .and_then(|v| v.as_u64())
            .expect("vocab_size in metadata") as u32;
        // xorshift64 — matches the tiny-gate token generator's intent.
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15) | 1;
        (0..max_tokens)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % vocab as u64) as u32
            })
            .collect()
    } else {
        let raw = std::fs::read(&corpus).expect("read corpus");
        let take = (max_tokens * 8).min(raw.len());
        let text = String::from_utf8_lossy(&raw[..take]).to_string();
        tokenizer.as_ref().unwrap().encode(&text)
    };
    let n_tok = if is_embedding_workload {
        embedding_samples.iter().map(Vec::len).sum()
    } else if let Some(parity) = &parity_job {
        parity.job.samples.total_rows()
    } else {
        tokens_owned.len().min(max_tokens)
    };
    let tokens = if is_embedding_workload
        || parity_job
            .as_ref()
            .is_some_and(|parity| parity.job.samples.samples().len() > 1)
    {
        &[][..]
    } else {
        &tokens_owned[..n_tok]
    };
    let kldref_topk = parity_job
        .as_ref()
        .map(|parity| parity.job.options.kldref_top_k)
        .unwrap_or(64);
    eprintln!(
        "calibrating on {n_tok} tokens across {} sample(s) (kldref={want_kldref}, synthetic={synthetic})",
        if is_embedding_workload {
            embedding_samples.len()
        } else if let Some(parity) = &parity_job {
            parity.job.samples.samples().len()
        } else {
            1
        }
    );

    let mut gpu = Gpu::init().expect("gpu");
    eprintln!("GPU: {}", gpu.arch);

    // Provenance keys (caller-known) layered onto the driver's technical metadata.
    let mut provenance = vec![
        ("source_model", serde_json::json!(model)),
        ("corpus", serde_json::json!(corpus)),
        ("n_calib_tokens", serde_json::json!(n_tok)),
        ("source_arch_id", serde_json::json!(source_arch_id)),
        ("calib_document_prompt", serde_json::json!(calib_doc_prompt)),
    ];
    if let Some(parity) = &parity_job {
        provenance.extend([
            ("family", serde_json::json!(&parity.family)),
            ("job", serde_json::to_value(&parity.job).unwrap()),
            ("resident_oracle", serde_json::json!(true)),
            (
                "oracle_streamed_artifact",
                serde_json::json!(&parity.artifact),
            ),
        ]);
    }
    let t0 = std::time::Instant::now();
    let (n_hessian, n_imatrix, max_consistency, mode) = match source_arch_id {
        1 if is_embedding_workload => {
            let config =
                hipfire_runtime::hfq::config_from_hfq(&hfq).expect("qwen3 embedding config");
            let weights = hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, &mut gpu)
                .expect("qwen3 embedding weights");
            let summary = collect_qwen3_embedding_artifacts(
                &mut gpu,
                &weights,
                &config,
                &embedding_samples,
                Path::new(&output),
                &provenance,
            )
            .expect("collect");
            (
                summary.n_hessian,
                summary.n_imatrix,
                summary.max_consistency,
                "qwen3-embedding",
            )
        }
        5 | 6 => {
            let config = qwen35::config_from_hfq(&hfq).expect("qwen35 config");
            let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");
            let opts = QwenCalibOpts {
                kldref: want_kldref,
                kldref_topk: kldref_topk,
            };
            let summary = if let Some(parity) = &parity_job {
                qwen35::collect_calibration_artifacts_job_with_residual_probe(
                    &mut gpu,
                    &weights,
                    &config,
                    &parity.job,
                    Path::new(&output),
                    &provenance,
                    residual_probe_output
                        .as_ref()
                        .map(|path| (Path::new(path), residual_probe_rows)),
                )
            } else {
                qwen35::collect_calibration_artifacts(
                    &mut gpu,
                    &weights,
                    &config,
                    tokens,
                    &opts,
                    Path::new(&output),
                    &provenance,
                )
            }
            .expect("collect");
            (
                summary.n_hessian,
                summary.n_imatrix,
                summary.max_consistency,
                "qwen35",
            )
        }
        12 | 13 => {
            let prefix = if source_arch_id == 13 {
                "language_model."
            } else {
                ""
            };
            let config = gemma3::config_from_hfq(&hfq).expect("gemma3 config");
            let weights =
                gemma3_weights::load_weights_prefixed(&mut hfq, &config, &mut gpu, prefix)
                    .expect("load_weights");
            let opts = gemma3_calib::CalibOpts {
                kldref: want_kldref,
                kldref_topk: kldref_topk,
            };
            let tokenizer = tokenizer
                .as_ref()
                .expect("gemma3 text-only collect needs a tokenizer (not --synthetic-tokens)");
            let summary = if let Some(parity) = &parity_job {
                gemma3_calib::collect_calibration_artifacts_samples_text_only(
                    &mut gpu,
                    &weights,
                    &config,
                    tokenizer,
                    &parity.job.samples,
                    &opts,
                    Path::new(&output),
                    prefix,
                    &provenance,
                )
            } else {
                gemma3_calib::collect_calibration_artifacts_text_only(
                    &mut gpu,
                    &weights,
                    &config,
                    tokenizer,
                    tokens,
                    &opts,
                    Path::new(&output),
                    prefix,
                    &provenance,
                )
            }
            .expect("collect");
            (
                summary.n_hessian,
                summary.n_imatrix,
                summary.max_consistency,
                if source_arch_id == 13 {
                    "gemma3-vl-text-only"
                } else {
                    "gemma3-text"
                },
            )
        }
        19 => {
            let config = embeddinggemma::config_from_metadata_json(&hfq.metadata_json)
                .expect("embeddinggemma config");
            let weights = embeddinggemma::EmbeddingGemmaWeights::load_for_calibration(
                &mut hfq, &config, &mut gpu,
            )
            .expect("load_weights");
            let summary = embeddinggemma_calib::collect_calibration_artifacts(
                &mut gpu,
                &weights,
                &config,
                &embedding_samples,
                Path::new(&output),
                &provenance,
            )
            .expect("collect");
            (
                summary.n_hessian,
                summary.n_imatrix,
                summary.max_consistency,
                "embeddinggemma",
            )
        }
        11 => {
            let config = Lfm2MoeConfig::from_hfq(&hfq).expect("lfm2 config");
            let weights = Lfm2MoeWeights::load(&mut hfq, &config, &mut gpu).expect("lfm2 weights");
            let opts = lfm2_calib::CalibOpts {
                kldref: want_kldref,
                kldref_topk: kldref_topk,
            };
            let summary = lfm2_calib::collect_calibration_artifacts(
                &mut gpu,
                &weights,
                &config,
                tokens,
                &opts,
                Path::new(&output),
                &provenance,
            )
            .expect("collect");
            (
                summary.n_hessian,
                summary.n_imatrix,
                summary.max_consistency,
                "lfm2-text",
            )
        }
        16 => {
            let meta: serde_json::Value =
                serde_json::from_str(&hfq.metadata_json).expect("zaya metadata");
            let cfg_json = meta.get("config").unwrap_or(&meta);
            let config = ZayaConfig::from_json(cfg_json).expect("zaya config");
            let weights = hipfire_arch_zaya::gpu::ZayaGpuWeights::load(&hfq, &mut gpu, &config)
                .expect("zaya weights");
            let opts = zaya_calib::CalibOpts {
                kldref: want_kldref,
                kldref_topk: kldref_topk,
            };
            let summary = zaya_calib::collect_calibration_artifacts(
                &mut gpu,
                &weights,
                &config,
                tokens,
                &opts,
                Path::new(&output),
                &provenance,
            )
            .expect("collect");
            (
                summary.n_hessian,
                summary.n_imatrix,
                summary.max_consistency,
                "zaya",
            )
        }
        10 => {
            // MiniMax-M2 (MoE: GQA attention + indexed-expert MoE). Attention
            // q/k/v/o, the MoE router, and lm_head route through `weight_gemv`
            // and get full Hessians; routed experts are not captured yet
            // (indexed MoE kernels need explicit taps — documented follow-on).
            let config = MiniMaxConfig::from_hfq(&hfq).expect("minimax config");
            let weights =
                MiniMaxWeights::load(&mut hfq, &config, &mut gpu, None).expect("minimax weights");
            let opts = minimax_calib::CalibOpts {
                kldref: want_kldref,
                kldref_topk: kldref_topk,
            };
            let summary = minimax_calib::collect_calibration_artifacts(
                &mut gpu,
                &weights,
                &config,
                tokens,
                &opts,
                Path::new(&output),
                &provenance,
            )
            .expect("collect");
            (
                summary.n_hessian,
                summary.n_imatrix,
                summary.max_consistency,
                "minimax",
            )
        }
        14 => {
            // Dense nemotron_h (Nano-4B). Config lives in the hfq metadata's
            // `config` key (same as serving). MoE Nano-30B experts are
            // imatrix-only — a follow-on (build_capture_names skips them).
            let meta: serde_json::Value =
                serde_json::from_str(&hfq.metadata_json).expect("nemotron metadata parse");
            let cfg_json = meta
                .get("config")
                .expect("nemotron metadata_json missing 'config'");
            let config = NemotronHConfig::from_json(cfg_json).expect("nemotron config");
            let mut model = NemotronModel::from_hfq(&mut gpu, &hfq, config, n_tok + 16)
                .expect("nemotron from_hfq");
            let opts = nemotron_calib::CalibOpts {
                kldref: want_kldref,
                kldref_topk: kldref_topk,
            };
            let summary = nemotron_calib::collect_calibration_artifacts(
                &mut gpu,
                &mut model,
                tokens,
                &opts,
                Path::new(&output),
                &provenance,
            )
            .expect("collect");
            (
                summary.n_hessian,
                summary.n_imatrix,
                summary.max_consistency,
                "nemotron-h",
            )
        }
        other => {
            panic!(
                "collect_artifacts: unsupported arch_id {other}; handled 5/6/10/11/12/13/14/16/19"
            )
        }
    };
    eprintln!(
        "collected {n_hessian} hessian + {n_imatrix} imatrix tensors in {:.1}s; mode={mode}; max diag(H)-vs-Σx² rel-err = {:.3e} {}",
        t0.elapsed().as_secs_f64(),
        max_consistency,
        if max_consistency < 1e-4 {
            "[CONSISTENT]"
        } else {
            "[MISMATCH]"
        }
    );
    eprintln!("wrote calib HFQ: {output}");
    if max_consistency >= 1e-4 {
        std::process::exit(1);
    }
}
