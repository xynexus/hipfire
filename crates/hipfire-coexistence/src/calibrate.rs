// SPDX-License-Identifier: Apache-2.0
//! CLI orchestration for native layer-stream calibration.
//!
//! The GPU calibration/induction engine (the forward-pass evidence producer)
//! now lives in `hipfire_runtime::calibration::layer_stream`; this file keeps
//! the daemon-free CLI surface: argument parsing entry, the GPU self-lock,
//! dry-run planning, and the storage byte-math preflight. See the runtime
//! module for `LayerStreamEngine`, `CalibrationSession`, and the
//! `DaemonCalibration` daemon wrapper.

// The coexistence CLI's aggregation edge: force-link the family crates so their
// native calibration adapters register into the runtime inventory that
// `resolve_adapter` reads. Adapter factories and architecture ownership stay in
// the family crates rather than a generic table here.
#[allow(unused_imports)]
use hipfire_arch_gemma3 as _;
#[allow(unused_imports)]
use hipfire_arch_qwen35 as _;

use hipfire_model::ModelSource;
use hipfire_rdna::Gpu;
use hipfire_runtime::calibration::boundary::BoundaryBackend;
use hipfire_runtime::calibration::contracts::{
    CalibError, CalibrationJob, CapturePolicy, CaptureRegistry,
};
use hipfire_runtime::calibration::schedule::MicrobatchGeometry;
use hipfire_runtime::calibration::source::{
    ReadLedgerSnapshot, TensorLoadPlan, TensorOwner, LAYER_PREFETCH_WORKER_CHUNK_BYTES,
};
use hipfire_runtime::calibration::stream::ModelInspection;
use hipfire_runtime::calibration::layer_stream::{
    build_calibration_run_inputs, calibration_engine_build_identity, calibration_run_fingerprint,
    default_boundary_directory, host_memory_snapshot, kldref_row_stride, layer_prefetch_decision,
    resolve_geometry, CalibrateCommand, CalibrationRunInputs, CalibrationRunOutcome,
    LayerStreamEngine, SourceManifestIdentity, CALIBRATION_PREFETCH_HOST_RESERVE_BYTES,
    CALIBRATION_PREFETCH_MIN_SWAP_FREE_DENOMINATOR,
};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[cfg(target_os = "linux")]
use std::ffi::CString;
#[cfg(target_os = "linux")]
use std::os::unix::ffi::OsStrExt;

const ARTIFACT_ESTIMATE_FIXED_OVERHEAD_BYTES: u64 = 64 * 1024 * 1024;
const ARTIFACT_ESTIMATE_MIN_SAFETY_BYTES: u64 = 4 * 1024 * 1024 * 1024;

const CALIBRATE_USAGE: &str = "usage: hipfire-coexistence calibrate \
--model <safetensors-dir-or-cache-root> --corpus <text> --output <model.calib.hfq> \
[--sequences N] [--context N] [--sequence-batch auto|N] [--time-tile auto|N] \
[--max-rows N (default: 2048)] [--min-expert-activations N] [--expert-capture-target N] \
[--expert-capture-tile-rows N] [--required-expert-fraction F] \
[--expert-coverage-policy strict|preserve-undercovered] [--kldref|--no-kldref] \
[--kldref-topk N] [--kldref-rows N] [--layer-prefetch-bytes N (default: 17179869184; 0 disables)] \
[--boundary-dir DIR|--boundary-ram] [--no-resume] \
[--pause-after-layers N] [--residual-probe-output PATH --residual-probe-rows N] \
[--dry-run]";

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum HessianEstimateStorage {
    DenseF32,
    Bf16TrilDiagF32,
}

impl HessianEstimateStorage {
    fn from_env() -> Self {
        match std::env::var("HIPFIRE_CALIB_HESSIAN_STORAGE")
            .ok()
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("f32" | "dense-f32" | "full-f32" | "legacy") => Self::DenseF32,
            _ => Self::Bf16TrilDiagF32,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
struct ArtifactStorageEstimate {
    hessian_storage: HessianEstimateStorage,
    capture_payload_bytes: u64,
    kldref_payload_bytes: u64,
    boundary_spool_bytes: u64,
    assembling_peak_payload_bytes: u64,
    fixed_container_overhead_bytes: u64,
    safety_margin_bytes: u64,
    completed_artifact_estimate_bytes: u64,
    required_free_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
struct FilesystemSpace {
    probe_path: PathBuf,
    available_bytes: u64,
    required_free_bytes: u64,
    sufficient: bool,
}

fn nearest_existing_path(path: &Path) -> Option<PathBuf> {
    let mut candidate = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()?.to_path_buf()
    };
    while !candidate.exists() {
        if !candidate.pop() {
            return None;
        }
    }
    Some(candidate)
}

#[cfg(target_os = "linux")]
fn filesystem_space(path: &Path, required_free_bytes: u64) -> Option<FilesystemSpace> {
    let probe_path = nearest_existing_path(path)?;
    let c_path = CString::new(probe_path.as_os_str().as_bytes()).ok()?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: c_path is NUL-terminated and statvfs initializes stats on success.
    if unsafe { libc::statvfs(c_path.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: the successful statvfs call initialized stats.
    let stats = unsafe { stats.assume_init() };
    let available_bytes = (stats.f_bavail as u64).saturating_mul(stats.f_frsize as u64);
    Some(FilesystemSpace {
        probe_path,
        available_bytes,
        required_free_bytes,
        sufficient: available_bytes >= required_free_bytes,
    })
}

#[cfg(not(target_os = "linux"))]
fn filesystem_space(_path: &Path, _required_free_bytes: u64) -> Option<FilesystemSpace> {
    None
}

fn checked_add_bytes(left: u64, right: u64, label: &str) -> Result<u64, CalibError> {
    left.checked_add(right)
        .ok_or_else(|| CalibError::InvalidOptions(format!("{label} byte estimate overflow")))
}

fn checked_mul_bytes(left: u64, right: u64, label: &str) -> Result<u64, CalibError> {
    left.checked_mul(right)
        .ok_or_else(|| CalibError::InvalidOptions(format!("{label} byte estimate overflow")))
}

fn artifact_storage_estimate(
    capture: &CaptureRegistry,
    hessian_storage: HessianEstimateStorage,
    kldref_positions: usize,
    kldref_top_k: usize,
    boundary_bytes: usize,
    boundary_on_disk: bool,
) -> Result<ArtifactStorageEstimate, CalibError> {
    let mut capture_payload_bytes = 0u64;
    for descriptor in capture.descriptors() {
        if descriptor.policy == CapturePolicy::Skip {
            continue;
        }
        let width = u64::try_from(descriptor.input_width)
            .map_err(|_| CalibError::InvalidOptions("capture width does not fit u64".into()))?;
        let aliases = u64::try_from(descriptor.output_names.len()).map_err(|_| {
            CalibError::InvalidOptions("capture alias count does not fit u64".into())
        })?;
        let imatrix = checked_mul_bytes(width, 4, "imatrix")?;
        let per_alias = if descriptor.policy == CapturePolicy::HessianAndImatrix {
            let hessian = match hessian_storage {
                HessianEstimateStorage::DenseF32 => checked_mul_bytes(
                    checked_mul_bytes(width, width, "dense Hessian")?,
                    4,
                    "dense Hessian",
                )?,
                HessianEstimateStorage::Bf16TrilDiagF32 => {
                    // Exact F32 diagonal plus BF16 lower strict triangle.
                    checked_add_bytes(
                        checked_mul_bytes(width, 4, "compact Hessian diagonal")?,
                        checked_mul_bytes(
                            width,
                            width.saturating_sub(1),
                            "compact Hessian triangle",
                        )?,
                        "compact Hessian",
                    )?
                }
            };
            checked_add_bytes(hessian, imatrix, "capture alias")?
        } else {
            imatrix
        };
        capture_payload_bytes = checked_add_bytes(
            capture_payload_bytes,
            checked_mul_bytes(per_alias, aliases, "capture aliases")?,
            "capture payload",
        )?;
    }

    let positions = u64::try_from(kldref_positions)
        .map_err(|_| CalibError::InvalidOptions("KLD row count does not fit u64".into()))?;
    let top_k = u64::try_from(kldref_top_k)
        .map_err(|_| CalibError::InvalidOptions("KLD top-k does not fit u64".into()))?;
    let kldref_payload_bytes = if positions == 0 || top_k == 0 {
        0
    } else {
        let row_bytes = checked_add_bytes(checked_mul_bytes(top_k, 8, "KLD top-k")?, 4, "KLD row")?;
        checked_mul_bytes(positions, row_bytes, "KLD payload")?
    };
    let boundary_spool_bytes = if boundary_on_disk {
        u64::try_from(boundary_bytes)
            .map_err(|_| CalibError::InvalidOptions("boundary bytes do not fit u64".into()))?
    } else {
        0
    };
    let assembling_peak_payload_bytes = checked_add_bytes(
        checked_mul_bytes(capture_payload_bytes, 2, "part plus assembly payload")?,
        kldref_payload_bytes,
        "assembly payload",
    )?;
    let completed_artifact_estimate_bytes = checked_add_bytes(
        checked_add_bytes(
            capture_payload_bytes,
            kldref_payload_bytes,
            "completed artifact",
        )?,
        ARTIFACT_ESTIMATE_FIXED_OVERHEAD_BYTES,
        "completed artifact",
    )?;
    let before_safety = checked_add_bytes(
        checked_add_bytes(
            assembling_peak_payload_bytes,
            boundary_spool_bytes,
            "calibration peak disk",
        )?,
        ARTIFACT_ESTIMATE_FIXED_OVERHEAD_BYTES,
        "calibration peak disk",
    )?;
    let safety_margin_bytes = (before_safety / 10).max(ARTIFACT_ESTIMATE_MIN_SAFETY_BYTES);
    let required_free_bytes =
        checked_add_bytes(before_safety, safety_margin_bytes, "required free")?;
    Ok(ArtifactStorageEstimate {
        hessian_storage,
        capture_payload_bytes,
        kldref_payload_bytes,
        boundary_spool_bytes,
        assembling_peak_payload_bytes,
        fixed_container_overhead_bytes: ARTIFACT_ESTIMATE_FIXED_OVERHEAD_BYTES,
        safety_margin_bytes,
        completed_artifact_estimate_bytes,
        required_free_bytes,
    })
}


pub fn run_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help") {
        println!("{CALIBRATE_USAGE}");
        return Ok(());
    }
    let command = CalibrateCommand::parse(args)?;
    let report = run_from_command(&command)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// Run one calibration pass in-process (planning, GPU self-lock, layer stream)
/// and return the same JSON report the CLI prints. This is the in-process
/// engine entry point the Rust induction driver drives instead of respawning
/// the calibrate binary (the `two_pass_quantize.py` `run_calibration_pass`
/// process-quantum scheduler it replaces). A daemon-resident caller drives the
/// `DaemonCalibration` one-layer-per-turn session instead and never reaches
/// here. On `--dry-run` this returns the dry-run plan without touching the GPU.
pub fn run_from_command(command: &CalibrateCommand) -> Result<serde_json::Value, Box<dyn Error>> {
    // Source/adapter/job construction is factored into the runtime engine so the
    // CLI and the daemon op feed byte-identical inputs to `begin`.
    let CalibrationRunInputs {
        snapshot,
        source,
        mut adapter,
        adapter_family,
        adapter_version,
        job,
        source_manifest,
    } = build_calibration_run_inputs(&command)?;
    let inspection = adapter.inspect(&source)?;
    inspection.validate()?;
    let capture = adapter.capture_plan(&inspection, &job)?;
    let tensor_plan = TensorLoadPlan::build(&source, inspection.tensor_requests.clone())?;
    let geometry = resolve_geometry(&job)?;
    let resource_estimate = adapter.resource_estimate(&inspection, &job, geometry)?;
    let engine_build = calibration_engine_build_identity()?;
    let run_fingerprint = calibration_run_fingerprint(
        adapter.as_ref(),
        &inspection,
        &tensor_plan,
        &job,
        geometry,
    )?;
    let dry_run = dry_run_report(
        &command,
        &snapshot,
        &source,
        adapter_family,
        adapter_version,
        &inspection,
        &tensor_plan,
        geometry,
        &job,
        &engine_build,
        &run_fingerprint,
        &source_manifest,
        resource_estimate.as_ref(),
        &capture,
    )?;
    if command.dry_run {
        return Ok(dry_run);
    }

    let storage_estimate = calibration_storage_estimate(&command, &inspection, &job, &capture)?;
    if !command.resume {
        if let Some(space) = filesystem_space(&command.output, storage_estimate.required_free_bytes)
        {
            if !space.sufficient {
                return Err(format!(
                    "calibrate: output filesystem at {} has {} available bytes, below the conservative {}-byte calibration requirement; choose another output/boundary filesystem or free space",
                    space.probe_path.display(),
                    space.available_bytes,
                    space.required_free_bytes,
                )
                .into());
            }
        }
    }

    if let Some(parent) = command
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let boundary_backend = if command.boundary_ram {
        BoundaryBackend::Ram
    } else {
        BoundaryBackend::Mmap {
            directory: command
                .boundary_directory
                .clone()
                .unwrap_or_else(|| default_boundary_directory(&command.output)),
        }
    };
    let _gpu_lock = acquire_gpu_lock()?;
    let mut gpu = Gpu::init()?;
    let result = LayerStreamEngine::new(boundary_backend, &command.output)
        .with_resume(command.resume)
        .with_pause_after_layers(command.pause_after_layers)
        .with_layer_prefetch_bytes(command.layer_prefetch_bytes)
        .with_residual_probe(
            command.residual_probe_output.clone(),
            command.residual_probe_rows,
        )
        .run(adapter.as_mut(), &source, &mut gpu, &job)?;
    let report = match result {
        CalibrationRunOutcome::Complete(result) => serde_json::json!({
            "status": "complete",
            "artifact": result.artifact_path,
            "residual_probe": command.residual_probe_output,
            "family": result.model.family,
            "layers": result.model.num_layers,
            "hessian_tensors": result.artifact.n_hessian,
            "imatrix_tensors": result.artifact.n_imatrix,
            "max_consistency": result.artifact.max_consistency,
            "kldref_positions": result.kldref_positions,
            "microbatch_geometry": result.geometry,
            "geometry_tuning": result.geometry_tuning,
            "layer_timings": result.layer_timings,
            "boundary_checkpoint": result.boundary_checkpoint,
            "read_ledger": read_ledger_cli_summary(&result.read_ledger),
        }),
        CalibrationRunOutcome::Paused(result) => serde_json::json!({
            "status": "paused",
            "artifact": null,
            "intended_artifact": result.artifact_path,
            "family": result.model.family,
            "layers": result.model.num_layers,
            "completed_layers": result.boundary_checkpoint.completed_layers,
            "resume_required": true,
            "microbatch_geometry": result.geometry,
            "geometry_tuning": result.geometry_tuning,
            "latest_layer_timing": result.layer_timings.last(),
            "boundary_checkpoint": result.boundary_checkpoint,
            "read_ledger": read_ledger_cli_summary(&result.read_ledger),
        }),
    };
    Ok(report)
}

fn read_ledger_cli_summary(snapshot: &ReadLedgerSnapshot) -> serde_json::Value {
    serde_json::json!({
        "planned_logical_count": snapshot.planned_logical.len(),
        "consumed_logical_count": snapshot.consumed_logical.len(),
        "read_canonical_count": snapshot.read_canonical.len(),
        "logical_bytes_read": snapshot.logical_bytes_read,
        "duplicate_logical": snapshot.duplicate_logical,
        "missing_logical_count": snapshot.missing_logical.len(),
        "full_ledger_persisted_in_checkpoint_or_artifact": true,
    })
}

fn acquire_gpu_lock() -> Result<hipfire_lock::FlockGuard, Box<dyn Error>> {
    let path = hipfire_lock::gpu_resource_lock_path();
    let mut guard = hipfire_lock::FlockGuard::open(&path)?;
    let mut waited = 0u64;
    guard.lock_blocking(Duration::from_secs(2), None, |holder| {
        waited += 2;
        let holder = if holder.is_empty() {
            "unknown holder"
        } else {
            holder
        };
        eprintln!("calibrate: GPU busy ({holder}); waited {waited}s");
    })?;
    guard.write_holder(&format!(
        "{} hipfire-coexistence calibrate",
        std::process::id()
    ))?;
    Ok(guard)
}

fn dry_run_report(
    command: &CalibrateCommand,
    snapshot: &Path,
    source: &dyn ModelSource,
    family: &str,
    adapter_version: &str,
    inspection: &ModelInspection,
    tensor_plan: &TensorLoadPlan,
    geometry: MicrobatchGeometry,
    job: &CalibrationJob,
    engine_build: &str,
    run_fingerprint: &str,
    source_manifest: &SourceManifestIdentity,
    resource_estimate: Option<&hipfire_runtime::calibration::stream::CalibrationResourceEstimate>,
    capture: &CaptureRegistry,
) -> Result<serde_json::Value, CalibError> {
    let mut source_dtypes = inspection
        .tensor_requests
        .iter()
        .filter_map(|request| source.tensor_info(&request.source_name))
        .map(|info| info.dtype.clone())
        .collect::<Vec<_>>();
    source_dtypes.sort();
    source_dtypes.dedup();
    let boundary_bytes = calibration_boundary_bytes(job, inspection)?;
    let max_layer_source_bytes = (0..inspection.num_layers)
        .map(|layer| tensor_plan.bytes_for(TensorOwner::Layer(layer)))
        .max()
        .unwrap_or(0);
    let kldref_positions = calibration_kldref_positions(job);
    let storage_estimate = calibration_storage_estimate(command, inspection, job, capture)?;
    let storage_filesystem =
        filesystem_space(&command.output, storage_estimate.required_free_bytes);
    let host_memory = host_memory_snapshot();
    let prefetch_decision = layer_prefetch_decision(
        command.layer_prefetch_bytes,
        max_layer_source_bytes,
        host_memory,
    );
    let kld_row_stride = kldref_row_stride(job.samples.total_rows(), job.options.kldref_rows);
    Ok(serde_json::json!({
        "command": "calibrate",
        "dry_run": command.dry_run,
        "engine_build": engine_build,
        "run_fingerprint": run_fingerprint,
        "model": {
            "requested_path": command.model,
            "snapshot_path": snapshot,
            "family": family,
            "adapter_version": adapter_version,
            "arch_id": inspection.arch_id,
            "layers": inspection.num_layers,
            "hidden_width": inspection.hidden_width,
            "vocab_size": inspection.vocab_size,
            "source_dtypes": source_dtypes,
        },
        "corpus": {
            "path": command.corpus,
            "sequences": job.samples.samples().len(),
            "context": job.samples.context_len(),
            "rows": job.samples.total_rows(),
            "sample_fingerprint": job.samples.fingerprint(),
            "corpus_fingerprint": job.corpus_fingerprint,
            "kldref_positions": if job.options.kldref { Some(kldref_positions) } else { None },
        },
        "microbatch": {
            "sequence_batch": geometry.sequence_batch,
            "time_tile": geometry.time_tile,
            "max_rows": geometry.row_budget,
        },
        "memory": {
            "boundary_bytes": boundary_bytes,
            "persistent_source_bytes": tensor_plan.bytes_for(TensorOwner::Persistent),
            "max_layer_source_bytes": max_layer_source_bytes,
            "layer_prefetch": {
                "mode": "resident-staging",
                "worker_chunk_bytes": LAYER_PREFETCH_WORKER_CHUNK_BYTES,
                "configured_bytes": command.layer_prefetch_bytes,
                "host_reserve_bytes": CALIBRATION_PREFETCH_HOST_RESERVE_BYTES,
                "next_layer_upload_reserve_bytes": max_layer_source_bytes,
                "minimum_swap_free_fraction": 1.0 / CALIBRATION_PREFETCH_MIN_SWAP_FREE_DENOMINATOR as f64,
                "requires_zero_full_pressure_avg10": true,
                "effective_max_layer_bytes": prefetch_decision.bytes,
                "disabled_reason": prefetch_decision.disabled_reason,
                "host_memory": host_memory,
            },
            "adapter_estimate": resource_estimate,
        },
        "storage": {
            "estimate": storage_estimate,
            "filesystem": storage_filesystem,
            "fresh_run_is_refused_when_insufficient": true,
            "resume_uses_checkpoint_specific_existing_allocations": true,
        },
        "source_plan": {
            "logical_tensors": tensor_plan.entries().len(),
            "unique_source_bytes": tensor_plan.unique_source_bytes(),
            "source_fingerprint": job.source_fingerprint,
            "shards": source_manifest.shards,
            "tokenizer_fingerprint": job.tokenizer_fingerprint,
        },
        "expert_capture": {
            "minimum_rows": job.options.expert_quota.min_rows,
            "target_rows": job.options.expert_quota.target_rows,
            "limit_rows": job.options.expert_quota.limit_rows()?,
            "tile_rows": job.options.expert_quota.tile_rows,
            "sampling": job.options.expert_quota.sampling,
            "maximum_batch_slack_rows": job.options.expert_quota.tile_rows - 1,
            "required_fraction": job.options.required_expert_fraction,
            "coverage_policy": job.options.expert_coverage_policy,
        },
        "kldref": {
            "enabled": job.options.kldref,
            "top_k": job.options.kldref_top_k,
            "row_cap": job.options.kldref_rows,
            "row_stride": kld_row_stride,
            "projected_rows": job.samples.total_rows().div_ceil(kld_row_stride),
        },
        "output": command.output,
        "expected_artifacts": {
            "calibration": command.output,
            "resume": command.resume,
            "pause_after_layers": command.pause_after_layers,
            "residual_probe": command.residual_probe_output.as_ref().map(|path| serde_json::json!({
                "output": path,
                "rows": command.residual_probe_rows,
                "requires_fresh_uninterrupted_run": true,
            })),
            "boundary_checkpoint": if command.boundary_ram {
                None
            } else {
                Some(command.boundary_directory.clone().unwrap_or_else(|| default_boundary_directory(&command.output)))
            },
            "contains_kldref": job.options.kldref,
        },
        "read_ledger_rules": {
            "each_logical_tensor_consumed_once": true,
            "physical_alias_read_once": true,
            "owner_scoped": true,
            "artifact_finalization_requires_complete_ledger": true,
        },
    }))
}

fn calibration_boundary_bytes(
    job: &CalibrationJob,
    inspection: &ModelInspection,
) -> Result<usize, CalibError> {
    job.samples
        .total_rows()
        .checked_mul(inspection.hidden_width)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>() * 2))
        .ok_or_else(|| CalibError::InvalidOptions("boundary byte estimate overflow".into()))
}

fn calibration_kldref_positions(job: &CalibrationJob) -> usize {
    job.samples
        .samples()
        .iter()
        .map(|sample| sample.tokens.len().saturating_sub(1))
        .sum()
}

fn calibration_storage_estimate(
    command: &CalibrateCommand,
    inspection: &ModelInspection,
    job: &CalibrationJob,
    capture: &CaptureRegistry,
) -> Result<ArtifactStorageEstimate, CalibError> {
    artifact_storage_estimate(
        capture,
        HessianEstimateStorage::from_env(),
        if job.options.kldref {
            calibration_kldref_positions(job)
        } else {
            0
        },
        if job.options.kldref {
            job.options.kldref_top_k
        } else {
            0
        },
        calibration_boundary_bytes(job, inspection)?,
        !command.boundary_ram,
    )
}


#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::calibration::contracts::{CaptureDescriptor, CaptureId, ProjectionRole};

    // The adapter-registry linkage assertion belongs here, not in the runtime
    // lib: registration is a property of a binary that links the family crates
    // as real dependencies (this crate does, via the crate-root `as _` edge),
    // whereas runtime links them only as test-only dev-deps.
    #[test]
    fn linked_calibration_adapters_are_unique_and_resolve_by_architecture() {
        use hipfire_runtime::calibration::stream::{
            registered_calibration_adapter, validate_calibration_adapter_registry,
        };

        validate_calibration_adapter_registry().unwrap();
        for (arch_id, family, version) in [
            (5, "qwen3.5", "qwen3.5-stream-v2"),
            (6, "qwen3.5", "qwen3.5-stream-v2"),
            (12, "gemma3", "gemma3-stream-v1"),
        ] {
            let adapter = registered_calibration_adapter(arch_id)
                .unwrap()
                .expect("linked calibration adapter");
            assert_eq!(adapter.family(), family);
            assert_eq!(adapter.adapter_version(), version);
        }
        assert!(registered_calibration_adapter(u32::MAX).unwrap().is_none());
    }

    #[test]
    fn artifact_storage_estimate_accounts_for_aliases_kld_boundary_and_assembly() {
        let mut capture = CaptureRegistry::default();
        capture
            .register(CaptureDescriptor {
                id: CaptureId::new(0, ProjectionRole::QueryInput, None),
                output_names: vec!["q_proj".into(), "k_proj".into()],
                input_width: 4,
                policy: CapturePolicy::HessianAndImatrix,
                layer: 0,
                role: ProjectionRole::QueryInput,
                expert: None,
                expert_quota: None,
            })
            .unwrap();
        capture
            .register(CaptureDescriptor {
                id: CaptureId::new(0, ProjectionRole::GateUpInput, Some(0)),
                output_names: vec!["expert.gate".into(), "expert.up".into()],
                input_width: 3,
                policy: CapturePolicy::ImatrixOnly,
                layer: 0,
                role: ProjectionRole::GateUpInput,
                expert: Some(0),
                expert_quota: Some(Default::default()),
            })
            .unwrap();

        let estimate = artifact_storage_estimate(
            &capture,
            HessianEstimateStorage::Bf16TrilDiagF32,
            10,
            2,
            1_000,
            true,
        )
        .unwrap();

        // Compact K=4 Hessian is 28 bytes, plus a 16-byte imatrix, emitted for
        // both aliases. Expert aliases each contribute a 12-byte imatrix.
        assert_eq!(estimate.capture_payload_bytes, 112);
        // Per KLD row: top-k u32 indices + f32 logits + one f32 logZ.
        assert_eq!(estimate.kldref_payload_bytes, 10 * (2 * 8 + 4));
        assert_eq!(estimate.boundary_spool_bytes, 1_000);
        assert_eq!(estimate.assembling_peak_payload_bytes, 112 * 2 + 200);
        assert!(estimate.required_free_bytes > estimate.assembling_peak_payload_bytes);

        let ram = artifact_storage_estimate(
            &capture,
            HessianEstimateStorage::DenseF32,
            0,
            0,
            1_000,
            false,
        )
        .unwrap();
        assert_eq!(ram.boundary_spool_bytes, 0);
        assert_eq!(ram.capture_payload_bytes, 184);
    }
}

