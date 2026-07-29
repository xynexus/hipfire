// SPDX-License-Identifier: Apache-2.0
//! The pass-1 (streamed calibration) → pass-2 (quantize) orchestration, ported
//! from `two_pass_quantize.main`.
//!
//! The process-quantum `run_calibration_pass` respawn scheduler is REPLACED, per
//! M6: pass 1 drives the in-process layer-stream engine
//! (`crate::calibrate::run_from_command`) in a single process — the byte-identical
//! `single_process` mode — instead of respawning the binary with
//! `--pause-after-layers`. A daemon-resident caller would instead drive the
//! `DaemonCalibration` one-layer-per-turn session; either way the surrounding
//! recipe → manifest → inspect/audit → preflight → quantize orchestration below
//! is unchanged. Pass 2 stays a scoped `hipfire lock run -- hipfire-quantize`
//! subprocess, exactly as the Python does (the quantizer is standalone per
//! AGENTS.md).

use super::manifest::{update_manifest, ManifestUpdate};
use super::preflight::{pass_two_storage_preflight, require_pass_two_storage};
use super::recipe::{Recipe, RecipeInputs};
use super::{dig, python_resolve};
use crate::calibrate::run_from_command;
use hipfire_runtime::calibration::layer_stream::CalibrateCommand;
use serde_json::{json, Value};
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

pub struct TwoPassConfig {
    pub model: PathBuf,
    pub calib: PathBuf,
    pub output: PathBuf,
    pub manifest: PathBuf,
    pub quant_format: String,
    pub corpus: PathBuf,
    pub n_sequences: u64,
    pub ctx_len: u64,
    pub batch_size: u64,
    pub time_tile: u64,
    pub max_rows: u64,
    pub layer_prefetch_bytes: u64,
    pub kldref_topk: u64,
    pub min_expert_activations: u64,
    pub expert_capture_target: u64,
    pub expert_capture_tile_rows: u64,
    pub required_expert_fraction: f64,
    pub sampling_seed: u64,
    pub expert_coverage_policy: String,
    pub quant_args: Vec<String>,
    /// `hipfire-quantize` binary (pass two).
    pub quantizer: String,
    /// `hipfire` CLI, used for the scoped GPU lock around pass two.
    pub hipfire: String,
    pub skip_calib: bool,
    pub dry_run: bool,
}

impl TwoPassConfig {
    fn recipe_inputs(&self) -> RecipeInputs {
        RecipeInputs {
            model: self.model.clone(),
            calib: self.calib.clone(),
            output: self.output.clone(),
            quant_format: self.quant_format.clone(),
            corpus: self.corpus.clone(),
            n_sequences: self.n_sequences,
            ctx_len: self.ctx_len,
            batch_size: self.batch_size,
            time_tile: self.time_tile,
            max_rows: self.max_rows,
            layer_prefetch_bytes: self.layer_prefetch_bytes,
            kldref_topk: self.kldref_topk,
            min_expert_activations: self.min_expert_activations,
            expert_capture_target: self.expert_capture_target,
            expert_capture_tile_rows: self.expert_capture_tile_rows,
            required_expert_fraction: self.required_expert_fraction,
            sampling_seed: self.sampling_seed,
            expert_coverage_policy: self.expert_coverage_policy.clone(),
            quant_args: self.quant_args.clone(),
        }
    }

    /// The `hipfire-coexistence calibrate` argument vector, matching
    /// `two_pass_quantize.build_commands` (minus the binary — we parse it into a
    /// `CalibrateCommand` and drive the engine in-process).
    fn collect_args(&self) -> Vec<String> {
        vec![
            "--model".into(),
            self.model.to_string_lossy().into_owned(),
            "--output".into(),
            self.calib.to_string_lossy().into_owned(),
            "--corpus".into(),
            self.corpus.to_string_lossy().into_owned(),
            "--sequences".into(),
            self.n_sequences.to_string(),
            "--context".into(),
            self.ctx_len.to_string(),
            "--sequence-batch".into(),
            self.batch_size.to_string(),
            "--time-tile".into(),
            self.time_tile.to_string(),
            "--max-rows".into(),
            self.max_rows.to_string(),
            "--layer-prefetch-bytes".into(),
            self.layer_prefetch_bytes.to_string(),
            "--kldref".into(),
            "--kldref-topk".into(),
            self.kldref_topk.to_string(),
            "--min-expert-activations".into(),
            self.min_expert_activations.to_string(),
            "--expert-capture-target".into(),
            self.expert_capture_target.to_string(),
            "--expert-capture-tile-rows".into(),
            self.expert_capture_tile_rows.to_string(),
            "--required-expert-fraction".into(),
            self.required_expert_fraction.to_string(),
            "--sampling-seed".into(),
            self.sampling_seed.to_string(),
            "--expert-coverage-policy".into(),
            self.expert_coverage_policy.clone(),
            "--resume".into(),
        ]
    }

    /// The scoped quantizer command: `hipfire lock run two-pass-quantization --
    /// hipfire-quantize --input <model> --output <out> --format <fmt> --hessian
    /// <calib> <quant_args...>`.
    fn quant_command(&self) -> Vec<String> {
        let mut cmd = vec![
            self.hipfire.clone(),
            "lock".into(),
            "run".into(),
            "two-pass-quantization".into(),
            "--".into(),
            self.quantizer.clone(),
            "--input".into(),
            self.model.to_string_lossy().into_owned(),
            "--output".into(),
            self.output.to_string_lossy().into_owned(),
            "--format".into(),
            self.quant_format.clone(),
            "--hessian".into(),
            self.calib.to_string_lossy().into_owned(),
        ];
        cmd.extend(self.quant_args.iter().cloned());
        cmd
    }
}

fn inspect_artifact(path: &std::path::Path) -> Result<Value, Box<dyn Error>> {
    crate::artifact::inspect_artifact(path)
}

fn audit_calibration(path: &std::path::Path) -> Result<Value, Box<dyn Error>> {
    let report = crate::calibration_audit::audit_calibration_artifact(path)?;
    Ok(serde_json::to_value(report)?)
}

fn validate_calibration_inspection(inspection: &Value) -> Result<(), Box<dyn Error>> {
    let metadata = inspection.get("metadata");
    if dig(Some(inspection), &["metadata", "artifact_kind"]).and_then(|v| v.as_str())
        != Some("calibration")
    {
        return Err("native pass produced an artifact without artifact_kind=calibration".into());
    }
    let ledger = metadata.and_then(|m| m.get("read_ledger"));
    let ledger = ledger.and_then(|v| v.as_object()).ok_or("native calibration artifact has no read_ledger")?;
    if ledger.get("missing_logical").and_then(|v| v.as_array()).map(|a| !a.is_empty()).unwrap_or(false) {
        return Err("native calibration read ledger has missing tensors".into());
    }
    if ledger.get("duplicate_logical").and_then(|v| v.as_array()).map(|a| !a.is_empty()).unwrap_or(false) {
        return Err("native calibration read ledger has duplicate reads".into());
    }
    Ok(())
}

fn validate_calibration_audit(audit: &Value, inspection: &Value) -> Result<(), Box<dyn Error>> {
    if audit.get("schema").and_then(|v| v.as_str()) != Some("hipfire.calibration_audit.v1")
        || audit.get("valid").and_then(|v| v.as_bool()) != Some(true)
    {
        return Err("native calibration artifact did not pass the structural audit".into());
    }
    if audit.get("errors").and_then(|v| v.as_array()).map(|a| !a.is_empty()).unwrap_or(false) {
        return Err("native calibration structural audit reports errors".into());
    }
    if audit.get("artifact_fingerprint") != inspection.get("artifact_fingerprint") {
        return Err("native calibration structural audit fingerprint differs from inspection".into());
    }
    if audit.get("index_only").and_then(|v| v.as_bool()) != Some(true)
        || audit.get("payload_values_checked").and_then(|v| v.as_bool()) != Some(false)
    {
        return Err("native calibration structural audit has an unknown evidence scope".into());
    }
    Ok(())
}

fn validate_quantized_inspection(inspection: &Value) -> Result<(), Box<dyn Error>> {
    if dig(Some(inspection), &["metadata", "quantization_hash", "value"])
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .is_none()
    {
        return Err("quantized artifact has no embedded quantization_hash".into());
    }
    if dig(Some(inspection), &["metadata", "calibration"]).and_then(|v| v.as_object()).is_none() {
        return Err("quantized artifact has no embedded calibration provenance".into());
    }
    Ok(())
}

/// Run the calibration *dry plan* (`calibrate --dry-run`) to get the `expected`
/// recipe JSON — the planned geometry / samples / experts / fingerprints WITHOUT
/// running the GPU forward. Twin of `two_pass_quantize.inspect_calibration_plan`
/// (+ `calibration_validation_command`): where the Python appends `--dry-run` to
/// the collect command and parses the child's stdout, the Rust reuses the exact
/// in-process dry-run path the CLI uses — `run_from_command` returns the same
/// `dry_run_report` Value before ever touching the GPU or its lock.
pub fn inspect_calibration_plan(command: &CalibrateCommand) -> Result<Value, Box<dyn Error>> {
    let mut plan_command = command.clone();
    plan_command.dry_run = true;
    run_from_command(&plan_command)
}

/// Python `repr()` for the scalar JSON types these fields hold, so a rejection
/// message is byte-identical to `_require_equal`: `'str'`, `4`, `1.0`, `True`,
/// `None`. Objects/arrays (only the `sampling` field) fall back to JSON, which
/// only appears in a message on an actual mismatch.
fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::String(s) => format!("'{s}'"),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                format!("{:?}", n.as_f64().unwrap_or(f64::NAN))
            }
        }
        other => other.to_string(),
    }
}

/// `dig` that returns an owned `Value` (Null when missing), so both sides of a
/// comparison behave like Python's `dict.get(...)` returning `None`.
fn at(value: &Value, path: &[&str]) -> Value {
    dig(Some(value), path).cloned().unwrap_or(Value::Null)
}

/// Twin of `two_pass_quantize._require_equal`: reject reuse on inequality with the
/// same `reused calibration {label} mismatch: artifact=..., requested=...` shape.
fn require_equal(label: &str, actual: &Value, expected: &Value) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "reused calibration {label} mismatch: artifact={}, requested={}",
            py_repr(actual),
            py_repr(expected)
        ));
    }
    Ok(())
}

/// Bind a reused calibration artifact to the producer's exact semantic recipe —
/// the ~30-field check ported faithfully from
/// `two_pass_quantize.validate_reusable_calibration`. `inspection` is the artifact
/// `inspect` JSON; `expected` is the dry-plan from [`inspect_calibration_plan`].
/// Any mismatch rejects reuse (rather than silently reusing a stale calibration).
pub fn validate_reusable_calibration(inspection: &Value, expected: &Value) -> Result<(), String> {
    let md = &["metadata"];
    let job = &["metadata", "job"];
    let options = &["metadata", "job", "options"];
    let samples = &["metadata", "job", "samples"];
    let source_manifest = &["metadata", "source_manifest"];
    let geometry = &["metadata", "microbatch_geometry"];
    let quota = &["metadata", "job", "options", "expert_quota"];

    require_equal(
        "run fingerprint",
        &at(inspection, &[md[0], "run_fingerprint"]),
        &at(expected, &["run_fingerprint"]),
    )?;
    require_equal("family", &at(inspection, &[md[0], "family"]), &at(expected, &["model", "family"]))?;
    require_equal(
        "adapter_version",
        &at(inspection, &[md[0], "adapter_version"]),
        &at(expected, &["model", "adapter_version"]),
    )?;
    require_equal("arch_id", &at(inspection, &[md[0], "arch_id"]), &at(expected, &["model", "arch_id"]))?;
    require_equal(
        "source fingerprint",
        &at(inspection, &[source_manifest[0], source_manifest[1], "fingerprint"]),
        &at(expected, &["source_plan", "source_fingerprint"]),
    )?;
    require_equal(
        "source job fingerprint",
        &at(inspection, &[job[0], job[1], "source_fingerprint"]),
        &at(expected, &["source_plan", "source_fingerprint"]),
    )?;
    require_equal(
        "source shards",
        &at(inspection, &[source_manifest[0], source_manifest[1], "shards"]),
        &at(expected, &["source_plan", "shards"]),
    )?;
    require_equal(
        "tokenizer fingerprint",
        &at(inspection, &[job[0], job[1], "tokenizer_fingerprint"]),
        &at(expected, &["source_plan", "tokenizer_fingerprint"]),
    )?;
    require_equal(
        "corpus fingerprint",
        &at(inspection, &[job[0], job[1], "corpus_fingerprint"]),
        &at(expected, &["corpus", "corpus_fingerprint"]),
    )?;
    require_equal(
        "sample fingerprint",
        &at(inspection, &[samples[0], samples[1], samples[2], "fingerprint"]),
        &at(expected, &["corpus", "sample_fingerprint"]),
    )?;
    // Computed sample stats: count of samples, and the summed per-sample token count.
    let sample_list = dig(Some(inspection), &[samples[0], samples[1], samples[2], "samples"])
        .and_then(|v| v.as_array());
    let sample_count = sample_list.map(|a| a.len()).unwrap_or(0);
    let sample_rows: usize = sample_list
        .map(|a| {
            a.iter()
                .map(|s| s.get("tokens").and_then(|t| t.as_array()).map(|t| t.len()).unwrap_or(0))
                .sum()
        })
        .unwrap_or(0);
    require_equal("sample count", &json!(sample_count), &at(expected, &["corpus", "sequences"]))?;
    require_equal(
        "sample context",
        &at(inspection, &[samples[0], samples[1], samples[2], "context_len"]),
        &at(expected, &["corpus", "context"]),
    )?;
    require_equal("sample rows", &json!(sample_rows), &at(expected, &["corpus", "rows"]))?;

    require_equal(
        "geometry sequence_batch",
        &at(inspection, &[geometry[0], geometry[1], "sequence_batch"]),
        &at(expected, &["microbatch", "sequence_batch"]),
    )?;
    require_equal(
        "geometry time_tile",
        &at(inspection, &[geometry[0], geometry[1], "time_tile"]),
        &at(expected, &["microbatch", "time_tile"]),
    )?;
    require_equal(
        "geometry row_budget",
        &at(inspection, &[geometry[0], geometry[1], "row_budget"]),
        &at(expected, &["microbatch", "max_rows"]),
    )?;
    require_equal(
        "job sequence_batch",
        &at(inspection, &[options[0], options[1], options[2], "sequence_batch"]),
        &at(expected, &["microbatch", "sequence_batch"]),
    )?;
    require_equal(
        "job time_tile",
        &at(inspection, &[options[0], options[1], options[2], "time_tile"]),
        &at(expected, &["microbatch", "time_tile"]),
    )?;
    require_equal(
        "job max_rows",
        &at(inspection, &[options[0], options[1], options[2], "max_rows"]),
        &at(expected, &["microbatch", "max_rows"]),
    )?;
    require_equal(
        "boundary precision",
        &at(inspection, &[options[0], options[1], options[2], "boundary_precision"]),
        &json!("f32"),
    )?;

    require_equal(
        "minimum_rows",
        &at(inspection, &[quota[0], quota[1], quota[2], quota[3], "min_rows"]),
        &at(expected, &["expert_capture", "minimum_rows"]),
    )?;
    require_equal(
        "target_rows",
        &at(inspection, &[quota[0], quota[1], quota[2], quota[3], "target_rows"]),
        &at(expected, &["expert_capture", "target_rows"]),
    )?;
    require_equal(
        "tile_rows",
        &at(inspection, &[quota[0], quota[1], quota[2], quota[3], "tile_rows"]),
        &at(expected, &["expert_capture", "tile_rows"]),
    )?;
    require_equal(
        "sampling",
        &at(inspection, &[quota[0], quota[1], quota[2], quota[3], "sampling"]),
        &at(expected, &["expert_capture", "sampling"]),
    )?;
    require_equal(
        "required_fraction",
        &at(inspection, &[options[0], options[1], options[2], "required_expert_fraction"]),
        &at(expected, &["expert_capture", "required_fraction"]),
    )?;
    require_equal(
        "coverage_policy",
        &at(inspection, &[options[0], options[1], options[2], "expert_coverage_policy"]),
        &at(expected, &["expert_capture", "coverage_policy"]),
    )?;
    require_equal(
        "KLDREF enabled",
        &at(inspection, &[options[0], options[1], options[2], "kldref"]),
        &at(expected, &["kldref", "enabled"]),
    )?;
    require_equal(
        "KLDREF top_k",
        &at(inspection, &[options[0], options[1], options[2], "kldref_top_k"]),
        &at(expected, &["kldref", "top_k"]),
    )?;
    Ok(())
}

fn run_subprocess(command: &[String]) -> Result<(), Box<dyn Error>> {
    let status = std::process::Command::new(&command[0])
        .args(&command[1..])
        .status()
        .map_err(|e| format!("spawn {:?}: {e}", command[0]))?;
    if !status.success() {
        return Err(format!("command failed ({status}): {}", command.join(" ")).into());
    }
    Ok(())
}

/// Run the two-pass workflow. Returns the final manifest.
pub fn run(cfg: &TwoPassConfig) -> Result<Value, Box<dyn Error>> {
    let recipe = Recipe::build(&cfg.recipe_inputs())?;
    let collect_args = cfg.collect_args();
    let quant_command = cfg.quant_command();

    if cfg.dry_run {
        println!("pass 1/2: hipfire-coexistence calibrate {}", collect_args.join(" "));
        println!("pass 2/2: {}", quant_command.join(" "));
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "manifest": cfg.manifest.to_string_lossy(),
                "recipe_fingerprint": recipe.recipe_fingerprint,
            }))?
        );
        return Ok(json!({"dry_run": true, "recipe_fingerprint": recipe.recipe_fingerprint}));
    }

    // Recipe-fingerprint gate: an existing manifest with a different recipe means
    // the geometry changed and resume would be unsafe.
    if let Ok(text) = std::fs::read_to_string(&cfg.manifest) {
        if let Ok(previous) = serde_json::from_str::<Value>(&text) {
            if let Some(prior) = previous.get("recipe_fingerprint").and_then(|v| v.as_str()) {
                if prior != recipe.recipe_fingerprint {
                    return Err(format!(
                        "two-pass manifest recipe mismatch: existing {prior}, requested {}",
                        recipe.recipe_fingerprint
                    )
                    .into());
                }
            }
        }
    }

    // Pass 1: calibration (in-process engine), unless reusing an existing artifact.
    let calibration_command = CalibrateCommand::parse(&collect_args).map_err(|e| format!("calibrate: {e}"))?;

    // Dry-plan the calibration (CPU-only, before any GPU work) to get the
    // `expected` recipe the reuse rebind checks against. Python computes this in
    // both the run and the skip-calib branch.
    if cfg.skip_calib && !cfg.calib.is_file() {
        return Err(format!(
            "--skip-calib requires an existing artifact: {}",
            cfg.calib.display()
        )
        .into());
    }
    let expected_calibration =
        inspect_calibration_plan(&calibration_command).map_err(|e| format!("calibrate dry-plan: {e}"))?;

    if !cfg.skip_calib {
        update_manifest(
            &cfg.manifest,
            &recipe,
            "calibration_running",
            ManifestUpdate {
                calibration_execution: Some(json!({
                    "mode": "single_process",
                    "process_segment_layers": 0,
                    "release_seconds": 0,
                })),
                ..Default::default()
            },
        )?;
        let started = Instant::now();
        let report = match run_from_command(&calibration_command) {
            Ok(report) => report,
            Err(error) => {
                let elapsed = started.elapsed().as_secs_f64();
                let manifest = std::fs::read_to_string(&cfg.manifest)
                    .ok()
                    .and_then(|t| serde_json::from_str(&t).ok())
                    .unwrap_or(json!({}));
                let _ = update_manifest(
                    &cfg.manifest,
                    &recipe,
                    "calibration_failed",
                    ManifestUpdate {
                        failure: Some(json!({
                            "recorded_at": super::utc_now(),
                            "kind": "exception",
                            "message": error.to_string(),
                        })),
                        phase_timings: Some(super::manifest::accumulate_attempt_timing(
                            &manifest, "calibration", elapsed,
                        )),
                        ..Default::default()
                    },
                );
                return Err(error);
            }
        };
        let elapsed = started.elapsed().as_secs_f64();
        if report.get("status").and_then(|v| v.as_str()) == Some("paused") {
            return Err("calibration paused unexpectedly (no pause boundary was requested)".into());
        }
        let manifest = std::fs::read_to_string(&cfg.manifest)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(json!({}));
        update_manifest(
            &cfg.manifest,
            &recipe,
            "calibration_validating",
            ManifestUpdate {
                calibration_execution: Some(json!({
                    "mode": "single_process",
                    "process_segment_layers": 0,
                    "release_seconds": 0,
                    "completed_layers": report.get("layers").cloned().unwrap_or(Value::Null),
                    "total_layers": report.get("layers").cloned().unwrap_or(Value::Null),
                    "artifact_complete": true,
                })),
                phase_timings: Some(super::manifest::accumulate_attempt_timing(
                    &manifest, "calibration", elapsed,
                )),
                ..Default::default()
            },
        )?;
    }

    // Inspect + structurally audit the calibration artifact.
    let calibration = inspect_artifact(&cfg.calib)?;
    validate_calibration_inspection(&calibration)?;
    let calibration_audit = audit_calibration(&cfg.calib)?;
    validate_calibration_audit(&calibration_audit, &calibration)?;
    // Bind the artifact (reused or freshly produced) to its exact semantic
    // recipe. On the skip-calib path this is what refuses a stale calibration.
    validate_reusable_calibration(&calibration, &expected_calibration)?;
    update_manifest(
        &cfg.manifest,
        &recipe,
        "calibration_complete",
        ManifestUpdate {
            calibration: Some(calibration.clone()),
            calibration_audit: Some(calibration_audit.clone()),
            ..Default::default()
        },
    )?;

    // Pass-two storage preflight (index-only byte math).
    let storage_preflight = pass_two_storage_preflight(
        &cfg.model,
        &cfg.output,
        &cfg.quant_format,
        &calibration,
        None,
    )?;
    let sufficient = dig(Some(&storage_preflight), &["filesystem", "sufficient"])
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    update_manifest(
        &cfg.manifest,
        &recipe,
        if sufficient { "quantization_ready" } else { "quantization_refused_storage" },
        ManifestUpdate {
            calibration: Some(calibration.clone()),
            calibration_audit: Some(calibration_audit.clone()),
            storage_preflight: Some(storage_preflight.clone()),
            ..Default::default()
        },
    )?;
    require_pass_two_storage(&storage_preflight)?;

    // Pass 2: quantize under the scoped GPU lock.
    update_manifest(
        &cfg.manifest,
        &recipe,
        "quantization_running",
        ManifestUpdate {
            calibration: Some(calibration.clone()),
            calibration_audit: Some(calibration_audit.clone()),
            storage_preflight: Some(storage_preflight.clone()),
            ..Default::default()
        },
    )?;
    let started = Instant::now();
    if let Err(error) = run_subprocess(&quant_command) {
        let elapsed = started.elapsed().as_secs_f64();
        let manifest = std::fs::read_to_string(&cfg.manifest)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(json!({}));
        let _ = update_manifest(
            &cfg.manifest,
            &recipe,
            "quantization_failed",
            ManifestUpdate {
                calibration: Some(calibration.clone()),
                calibration_audit: Some(calibration_audit.clone()),
                storage_preflight: Some(storage_preflight.clone()),
                failure: Some(json!({
                    "recorded_at": super::utc_now(),
                    "kind": "process_error",
                    "message": error.to_string(),
                })),
                phase_timings: Some(super::manifest::accumulate_attempt_timing(
                    &manifest, "quantization", elapsed,
                )),
                ..Default::default()
            },
        );
        return Err(error);
    }
    let elapsed = started.elapsed().as_secs_f64();

    let quantized = inspect_artifact(&cfg.output)?;
    validate_quantized_inspection(&quantized)?;
    let manifest = std::fs::read_to_string(&cfg.manifest)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(json!({}));
    let final_manifest = update_manifest(
        &cfg.manifest,
        &recipe,
        "complete",
        ManifestUpdate {
            calibration: Some(calibration),
            calibration_audit: Some(calibration_audit),
            storage_preflight: Some(storage_preflight),
            quantized: Some(quantized),
            phase_timings: Some(super::manifest::accumulate_attempt_timing(
                &manifest, "quantization", elapsed,
            )),
            ..Default::default()
        },
    )?;
    Ok(final_manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal but faithful (inspection, expected) pair reproducing the real
    // field values captured from the /tmp/e2e Qwen3.5-0.8B calibration artifact
    // and its `calibrate --dry-run` plan, which Python
    // `validate_reusable_calibration` ACCEPTS.
    fn golden_pair() -> (Value, Value) {
        let sampling = json!({"kind": "deterministic_first", "seed": 1});
        let shards = json!([{
            "file": "model.safetensors-00001-of-00001.safetensors",
            "bytes": 1746942600u64,
            "identity_kind": "huggingface_blob_digest",
            "identity": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
        }]);
        let samples: Vec<Value> = (0..4).map(|_| json!({"tokens": vec![0u32; 64]})).collect();
        let inspection = json!({
            "metadata": {
                "run_fingerprint": "fnv64:d4ff9fa71a504e8f",
                "family": "qwen3.5",
                "adapter_version": "qwen3.5-stream-v2",
                "arch_id": 5,
                "source_manifest": {"fingerprint": "fnv64:921e21656fa33904", "shards": shards.clone()},
                "microbatch_geometry": {"sequence_batch": 4, "time_tile": 16, "row_budget": 256},
                "job": {
                    "source_fingerprint": "fnv64:921e21656fa33904",
                    "tokenizer_fingerprint": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
                    "corpus_fingerprint": "b1b72c4c35eebd31d23a515f349b66e3176a242496499384ba0e93713ad39503",
                    "options": {
                        "sequence_batch": 4,
                        "time_tile": 16,
                        "max_rows": 256,
                        "boundary_precision": "f32",
                        "required_expert_fraction": 1.0,
                        "expert_coverage_policy": "preserve-undercovered",
                        "kldref": true,
                        "kldref_top_k": 64,
                        "expert_quota": {
                            "min_rows": 2048,
                            "target_rows": 4096,
                            "tile_rows": 256,
                            "sampling": sampling.clone()
                        }
                    },
                    "samples": {
                        "fingerprint": "fnv1a64:defd7352474c01ab",
                        "context_len": 64,
                        "samples": samples
                    }
                }
            }
        });
        let expected = json!({
            "run_fingerprint": "fnv64:d4ff9fa71a504e8f",
            "model": {"family": "qwen3.5", "adapter_version": "qwen3.5-stream-v2", "arch_id": 5},
            "source_plan": {
                "source_fingerprint": "fnv64:921e21656fa33904",
                "shards": shards,
                "tokenizer_fingerprint": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
            },
            "corpus": {
                "corpus_fingerprint": "b1b72c4c35eebd31d23a515f349b66e3176a242496499384ba0e93713ad39503",
                "sample_fingerprint": "fnv1a64:defd7352474c01ab",
                "sequences": 4,
                "context": 64,
                "rows": 256
            },
            "microbatch": {"sequence_batch": 4, "time_tile": 16, "max_rows": 256},
            "expert_capture": {
                "minimum_rows": 2048,
                "target_rows": 4096,
                "tile_rows": 256,
                "sampling": sampling,
                "required_fraction": 1.0,
                "coverage_policy": "preserve-undercovered"
            },
            "kldref": {"enabled": true, "top_k": 64}
        });
        (inspection, expected)
    }

    #[test]
    fn reuse_accepts_matching_recipe_like_python() {
        let (inspection, expected) = golden_pair();
        assert!(validate_reusable_calibration(&inspection, &expected).is_ok());
    }

    #[test]
    fn reuse_rejects_with_python_field_labels() {
        // (perturb expected, exact Python golden message)
        let cases: &[(fn(&mut Value), &str)] = &[
            (
                |e| e["microbatch"]["sequence_batch"] = json!(8),
                "reused calibration geometry sequence_batch mismatch: artifact=4, requested=8",
            ),
            (
                |e| e["corpus"]["sequences"] = json!(99),
                "reused calibration sample count mismatch: artifact=4, requested=99",
            ),
            (
                |e| e["expert_capture"]["minimum_rows"] = json!(111),
                "reused calibration minimum_rows mismatch: artifact=2048, requested=111",
            ),
            (
                |e| e["run_fingerprint"] = json!("run:BOGUS"),
                "reused calibration run fingerprint mismatch: artifact='fnv64:d4ff9fa71a504e8f', requested='run:BOGUS'",
            ),
            (
                |e| e["kldref"]["top_k"] = json!(7),
                "reused calibration KLDREF top_k mismatch: artifact=64, requested=7",
            ),
        ];
        for (perturb, expect_msg) in cases {
            let (inspection, mut expected) = golden_pair();
            perturb(&mut expected);
            let err = validate_reusable_calibration(&inspection, &expected)
                .expect_err("perturbation must reject reuse");
            assert_eq!(&err, expect_msg);
        }
    }
}

/// Resolve the manifest path default (`<output>.two-pass.json`) the way Python
/// does when `--manifest` is omitted.
pub fn default_manifest_path(output: &std::path::Path) -> PathBuf {
    // Python: args.output.with_suffix(".two-pass.json") — replaces the final
    // ".hfq" with ".two-pass.json".
    python_resolve(&output.with_extension("two-pass.json"))
}
