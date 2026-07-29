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

/// Resolve the manifest path default (`<output>.two-pass.json`) the way Python
/// does when `--manifest` is omitted.
pub fn default_manifest_path(output: &std::path::Path) -> PathBuf {
    // Python: args.output.with_suffix(".two-pass.json") — replaces the final
    // ".hfq" with ".two-pass.json".
    python_resolve(&output.with_extension("two-pass.json"))
}
