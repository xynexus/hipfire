// SPDX-License-Identifier: Apache-2.0
//! Index-only structural audit for native calibration artifacts.

use crate::artifact::{index_fingerprint, index_identity};
use hipfire_runtime::calibration::contracts::{
    CalibrationJob, ExpertCoveragePolicy, ExpertLayerTelemetry, LayerExpert, SamplePosition,
    SampleSet, CALIBRATION_JOB_SCHEMA_VERSION,
};
use hipfire_runtime::calibration::schedule::MicrobatchGeometry;
use hipfire_runtime::calibration::source::ReadLedgerSnapshot;
use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::error::Error;
use std::path::{Path, PathBuf};

const AUDIT_SCHEMA: &str = "hipfire.calibration_audit.v1";
const F32_QUANT_TYPE: u8 = 2;
const COMPACT_HESSIAN_QUANT_TYPE: u8 = 130;

#[derive(Debug, Clone, Default, Serialize)]
pub struct ReadLedgerAuditSummary {
    pub planned_logical: usize,
    pub consumed_logical: usize,
    pub read_canonical: usize,
    pub duplicate_logical: usize,
    pub missing_logical: usize,
    pub logical_bytes_read: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct KldRefAuditSummary {
    pub enabled: bool,
    pub n_positions: usize,
    pub top_k: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ExpertAuditSummary {
    pub layers: usize,
    pub declared_experts: usize,
    pub capture_points: usize,
    pub deficit_capture_points: usize,
    pub preserved_experts: usize,
    pub layers_meeting_required_fraction: usize,
    pub legacy_admitted_capture_points: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CalibrationAuditSummary {
    pub family: Option<String>,
    pub adapter_version: Option<String>,
    pub job_schema_version: Option<u32>,
    pub samples: usize,
    pub sample_rows: usize,
    pub context_length: usize,
    pub n_hessian: usize,
    pub n_imatrix: usize,
    pub read_ledger: Option<ReadLedgerAuditSummary>,
    pub kldref: KldRefAuditSummary,
    pub experts: ExpertAuditSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationAuditReport {
    pub schema: &'static str,
    pub artifact: PathBuf,
    pub artifact_fingerprint: String,
    pub fingerprint_scope: &'static str,
    pub bytes: u64,
    pub version: u32,
    pub arch_id: u32,
    pub tensor_count: usize,
    pub index_only: bool,
    pub payload_values_checked: bool,
    pub valid: bool,
    pub summary: CalibrationAuditSummary,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

fn nonempty_string(metadata: &Value, key: &str, errors: &mut Vec<String>) -> Option<String> {
    match metadata.get(key).and_then(Value::as_str) {
        Some(value) if !value.is_empty() => Some(value.to_string()),
        _ => {
            errors.push(format!("metadata {key} must be a nonempty string"));
            None
        }
    }
}

fn unsigned_metadata(metadata: &Value, key: &str, errors: &mut Vec<String>) -> Option<usize> {
    match metadata
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
    {
        Some(value) => Some(value),
        None => {
            errors.push(format!("metadata {key} must be an unsigned integer"));
            None
        }
    }
}

fn checked_dense_f32_bytes(shape: &[u32]) -> Option<usize> {
    shape.iter().try_fold(4usize, |bytes, dim| {
        bytes.checked_mul(usize::try_from(*dim).ok()?)
    })
}

fn validate_f32_tensor(tensor: &HfqTensorInfo, expected_shape: &[u32], errors: &mut Vec<String>) {
    if tensor.quant_type != F32_QUANT_TYPE || tensor.shape != expected_shape {
        errors.push(format!(
            "{} has dtype/shape {}/{:?}, expected F32/{expected_shape:?}",
            tensor.name, tensor.quant_type, tensor.shape
        ));
    }
    if let Some(expected_bytes) = checked_dense_f32_bytes(expected_shape) {
        if tensor.data_size != expected_bytes {
            errors.push(format!(
                "{} has {} payload bytes, expected {expected_bytes}",
                tensor.name, tensor.data_size
            ));
        }
    } else {
        errors.push(format!("{} shape byte count overflows usize", tensor.name));
    }
}

fn validate_calibration_tensors(
    hfq: &HfqFile,
    metadata: &Value,
    summary: &mut CalibrationAuditSummary,
    errors: &mut Vec<String>,
) {
    let mut names = BTreeSet::new();
    for tensor in hfq.tensors() {
        if !names.insert(tensor.name.as_str()) {
            errors.push(format!("duplicate tensor index entry {}", tensor.name));
        }
    }

    let hessian = hfq
        .tensors()
        .iter()
        .filter_map(|tensor| {
            tensor
                .name
                .strip_suffix(".hessian")
                .map(|name| (name, tensor))
        })
        .collect::<Vec<_>>();
    let imatrix = hfq
        .tensors()
        .iter()
        .filter_map(|tensor| {
            tensor
                .name
                .strip_suffix(".imatrix")
                .map(|name| (name, tensor))
        })
        .collect::<Vec<_>>();
    let hessian_names = hessian
        .iter()
        .map(|(name, _)| *name)
        .collect::<BTreeSet<_>>();
    let imatrix_names = imatrix
        .iter()
        .map(|(name, _)| *name)
        .collect::<BTreeSet<_>>();
    summary.n_hessian = hessian.len();
    summary.n_imatrix = imatrix.len();

    if unsigned_metadata(metadata, "n_hessian", errors) != Some(hessian.len()) {
        errors.push(format!(
            "metadata n_hessian does not match {} indexed Hessian tensors",
            hessian.len()
        ));
    }
    if unsigned_metadata(metadata, "n_imatrix", errors) != Some(imatrix.len()) {
        errors.push(format!(
            "metadata n_imatrix does not match {} indexed imatrix tensors",
            imatrix.len()
        ));
    }
    for missing in hessian_names.difference(&imatrix_names) {
        errors.push(format!("Hessian {missing} has no matching imatrix"));
    }

    for (name, tensor) in &imatrix {
        if tensor.quant_type != F32_QUANT_TYPE || tensor.shape.len() != 1 || tensor.shape[0] == 0 {
            errors.push(format!(
                "{name}.imatrix must be a nonempty rank-1 F32 tensor, got dtype/shape {}/{:?}",
                tensor.quant_type, tensor.shape
            ));
        } else if checked_dense_f32_bytes(&tensor.shape) != Some(tensor.data_size) {
            errors.push(format!(
                "{name}.imatrix payload size does not match its shape"
            ));
        }
    }
    for (name, tensor) in &hessian {
        let square =
            tensor.shape.len() == 2 && tensor.shape[0] > 0 && tensor.shape[0] == tensor.shape[1];
        if !square {
            errors.push(format!("{name}.hessian must have a nonempty square shape"));
            continue;
        }
        let k = tensor.shape[0] as usize;
        let expected_bytes = match tensor.quant_type {
            F32_QUANT_TYPE => k.checked_mul(k).and_then(|values| values.checked_mul(4)),
            COMPACT_HESSIAN_QUANT_TYPE => k.checked_mul(4).and_then(|diagonal| {
                k.checked_mul(k.saturating_sub(1))
                    .and_then(|lower| diagonal.checked_add(lower))
            }),
            other => {
                errors.push(format!("{name}.hessian has unsupported quant type {other}"));
                None
            }
        };
        if expected_bytes != Some(tensor.data_size) {
            errors.push(format!(
                "{name}.hessian payload size does not match its encoding"
            ));
        }
        if let Some((_, diagonal)) = imatrix.iter().find(|(candidate, _)| candidate == name) {
            if diagonal.shape.first() != tensor.shape.first() {
                errors.push(format!(
                    "{name} Hessian width {:?} differs from imatrix width {:?}",
                    tensor.shape.first(),
                    diagonal.shape.first()
                ));
            }
        }
    }

    match metadata.get("per_tensor_tokens").and_then(Value::as_object) {
        Some(per_tensor_tokens) => {
            let recorded = per_tensor_tokens
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            if recorded != imatrix_names {
                errors.push("per_tensor_tokens keys do not exactly match imatrix tensors".into());
            }
            for (name, value) in per_tensor_tokens {
                if value.as_u64().is_none() {
                    errors.push(format!("per_tensor_tokens[{name}] must be unsigned"));
                }
            }
        }
        None => errors.push("metadata per_tensor_tokens must be an object".into()),
    }
}

fn validate_ledger(
    ledger: &ReadLedgerSnapshot,
    errors: &mut Vec<String>,
) -> ReadLedgerAuditSummary {
    if ledger.planned_logical.is_empty() {
        errors.push("read ledger has no planned logical tensors".into());
    }
    if !ledger.consumed_logical.is_subset(&ledger.planned_logical) {
        errors.push("read ledger consumed set contains unplanned tensors".into());
    }
    if !ledger.read_canonical.is_subset(&ledger.consumed_logical) {
        errors.push("read ledger canonical-read set is not a subset of consumed tensors".into());
    }
    if !ledger.duplicate_logical.is_empty() {
        errors.push(format!(
            "read ledger records {} duplicate logical reads",
            ledger.duplicate_logical.len()
        ));
    }
    let expected_missing = ledger
        .planned_logical
        .difference(&ledger.consumed_logical)
        .cloned()
        .collect::<BTreeSet<_>>();
    if ledger.missing_logical != expected_missing {
        errors.push("read ledger missing set does not equal planned minus consumed".into());
    }
    if !expected_missing.is_empty() {
        errors.push(format!(
            "read ledger is incomplete: {} logical tensors are missing",
            expected_missing.len()
        ));
    }
    if !ledger.read_canonical.is_empty() && ledger.logical_bytes_read == 0 {
        errors.push("read ledger records canonical reads but zero source bytes".into());
    }
    ReadLedgerAuditSummary {
        planned_logical: ledger.planned_logical.len(),
        consumed_logical: ledger.consumed_logical.len(),
        read_canonical: ledger.read_canonical.len(),
        duplicate_logical: ledger.duplicate_logical.len(),
        missing_logical: ledger.missing_logical.len(),
        logical_bytes_read: ledger.logical_bytes_read,
    }
}

fn validate_geometry(metadata: &Value, job: &CalibrationJob, errors: &mut Vec<String>) {
    let geometry = match metadata
        .get("microbatch_geometry")
        .cloned()
        .map(serde_json::from_value::<MicrobatchGeometry>)
    {
        Some(Ok(geometry)) => geometry,
        Some(Err(error)) => {
            errors.push(format!("invalid microbatch_geometry: {error}"));
            return;
        }
        None => {
            errors.push("metadata lacks microbatch_geometry".into());
            return;
        }
    };
    if let Err(error) = geometry.validate() {
        errors.push(error.to_string());
    }
    if job
        .options
        .sequence_batch
        .is_some_and(|value| value != geometry.sequence_batch)
    {
        errors.push("microbatch sequence_batch differs from the explicit job option".into());
    }
    if job
        .options
        .time_tile
        .is_some_and(|value| value != geometry.time_tile)
    {
        errors.push("microbatch time_tile differs from the explicit job option".into());
    }
    if geometry.row_budget != job.options.max_rows {
        errors.push("microbatch row budget differs from the job maximum rows".into());
    }
    if let Some(selected) = metadata
        .get("geometry_tuning")
        .and_then(|value| value.get("selected"))
    {
        if selected != &serde_json::to_value(geometry).unwrap_or(Value::Null) {
            errors.push("geometry_tuning.selected differs from microbatch_geometry".into());
        }
    } else {
        errors.push("metadata lacks geometry_tuning.selected".into());
    }
}

fn validate_job_recording(metadata: &Value, job: &CalibrationJob, errors: &mut Vec<String>) {
    match SampleSet::new(
        job.samples.samples().to_vec(),
        job.samples.context_len(),
        job.samples.sampling_seed(),
    ) {
        Ok(rebuilt) if rebuilt == job.samples => {}
        Ok(_) => errors.push(
            "serialized sample ordering or fingerprint does not match its tokens/geometry".into(),
        ),
        Err(error) => errors.push(format!("invalid serialized sample set: {error}")),
    }

    let quota = &job.options.expert_quota;
    let expected_limit = quota.limit_rows().ok();
    let recorded_quota = metadata.get("expert_capture_quota");
    let expected_sampling = serde_json::to_value(quota.sampling).unwrap_or(Value::Null);
    for (key, expected) in [
        ("minimum_rows", Some(quota.min_rows)),
        ("target_rows", Some(quota.target_rows)),
        ("limit_rows", expected_limit),
        ("tile_rows", u64::try_from(quota.tile_rows).ok()),
        (
            "maximum_batch_slack_rows",
            u64::try_from(quota.tile_rows.saturating_sub(1)).ok(),
        ),
    ] {
        if recorded_quota
            .and_then(|value| value.get(key))
            .and_then(Value::as_u64)
            != expected
        {
            errors.push(format!(
                "expert_capture_quota.{key} differs from the serialized job"
            ));
        }
    }
    if recorded_quota.and_then(|value| value.get("sampling")) != Some(&expected_sampling) {
        errors.push("expert_capture_quota.sampling differs from the serialized job".into());
    }

    let expected_boundary =
        serde_json::to_value(job.options.boundary_precision).unwrap_or(Value::Null);
    if metadata
        .get("effective_precision")
        .and_then(|value| value.get("boundary"))
        != Some(&expected_boundary)
    {
        errors.push("effective_precision.boundary differs from the serialized job".into());
    }
}

fn validate_experts(
    metadata: &Value,
    job: &CalibrationJob,
    errors: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> ExpertAuditSummary {
    let telemetry: Vec<ExpertLayerTelemetry> = match metadata
        .get("expert_telemetry")
        .cloned()
        .map(serde_json::from_value)
    {
        Some(Ok(value)) => value,
        Some(Err(error)) => {
            errors.push(format!("invalid expert_telemetry: {error}"));
            return ExpertAuditSummary::default();
        }
        None => {
            errors.push("metadata lacks expert_telemetry".into());
            return ExpertAuditSummary::default();
        }
    };
    let preserved: Vec<LayerExpert> = match metadata
        .get("preserve_high_precision")
        .cloned()
        .map(serde_json::from_value)
    {
        Some(Ok(value)) => value,
        Some(Err(error)) => {
            errors.push(format!("invalid preserve_high_precision: {error}"));
            Vec::new()
        }
        None => {
            errors.push("metadata lacks preserve_high_precision".into());
            Vec::new()
        }
    };

    let telemetry_layers = telemetry
        .iter()
        .map(|layer| layer.layer)
        .collect::<Vec<_>>();
    if !telemetry_layers.windows(2).all(|pair| pair[0] < pair[1]) {
        errors.push("expert telemetry layers must be strictly increasing and unique".into());
    }
    let preserved_set = preserved.iter().copied().collect::<BTreeSet<_>>();
    if preserved_set.len() != preserved.len()
        || preserved_set.iter().copied().collect::<Vec<_>>() != preserved
    {
        errors.push("preserve_high_precision must be sorted and duplicate-free".into());
    }

    let mut expected_preserve = BTreeSet::new();
    let mut summary = ExpertAuditSummary {
        layers: telemetry.len(),
        preserved_experts: preserved_set.len(),
        ..ExpertAuditSummary::default()
    };
    for layer in &telemetry {
        summary.declared_experts = summary.declared_experts.saturating_add(layer.num_experts);
        summary.capture_points = summary
            .capture_points
            .saturating_add(layer.num_experts.saturating_mul(2));
        if layer.quota != job.options.expert_quota {
            errors.push(format!(
                "layer {} expert quota differs from the serialized job",
                layer.layer
            ));
        }
        if let Err(error) = layer.reconcile() {
            errors.push(error.to_string());
            continue;
        }
        let coverage = layer.coverage_report(
            job.options.expert_coverage_policy,
            job.options.required_expert_fraction,
        );
        summary.deficit_capture_points = summary
            .deficit_capture_points
            .saturating_add(coverage.deficits.len());
        if coverage.complete {
            summary.layers_meeting_required_fraction += 1;
        }
        if job.options.expert_coverage_policy == ExpertCoveragePolicy::Strict && !coverage.complete
        {
            errors.push(format!(
                "layer {} fails strict expert coverage: {:.6} < {:.6}",
                layer.layer, coverage.covered_fraction, coverage.required_fraction
            ));
        }
        if job.options.expert_coverage_policy == ExpertCoveragePolicy::PreserveUndercovered {
            expected_preserve.extend(
                coverage
                    .deficits
                    .iter()
                    .map(|deficit| LayerExpert::new(deficit.layer, deficit.expert)),
            );
        }
        summary.legacy_admitted_capture_points += layer
            .gate_up
            .iter()
            .chain(&layer.down)
            .filter(|stats| stats.admitted_rows > 0 && !stats.launch_telemetry_recorded)
            .count();
    }

    if job.options.expert_coverage_policy == ExpertCoveragePolicy::Strict
        && !preserved_set.is_empty()
    {
        errors.push("strict coverage artifact unexpectedly preserves experts".into());
    }
    if job.options.expert_coverage_policy == ExpertCoveragePolicy::PreserveUndercovered
        && preserved_set != expected_preserve
    {
        let missing = expected_preserve.difference(&preserved_set).count();
        let extra = preserved_set.difference(&expected_preserve).count();
        errors.push(format!(
            "preserve_high_precision differs from telemetry deficits: missing={missing}, extra={extra}"
        ));
    }
    if summary.legacy_admitted_capture_points > 0 {
        warnings.push(format!(
            "{} admitted expert capture points lack reduction-launch telemetry",
            summary.legacy_admitted_capture_points
        ));
    }
    summary
}

fn validate_kldref(
    hfq: &HfqFile,
    metadata: &Value,
    job: &CalibrationJob,
    errors: &mut Vec<String>,
) -> KldRefAuditSummary {
    let names = [
        "lm_head.kldref_idx",
        "lm_head.kldref_logit",
        "lm_head.kldref_logz",
    ];
    if !job.options.kldref {
        if metadata.get("kldref").is_some()
            || names
                .iter()
                .any(|name| hfq.find_tensor_info(name).is_some())
        {
            errors.push("no-KLD job contains KLDREF metadata or tensors".into());
        }
        return KldRefAuditSummary::default();
    }

    let Some(kldref) = metadata.get("kldref").and_then(Value::as_object) else {
        errors.push("KLD-enabled job lacks KLDREF metadata".into());
        return KldRefAuditSummary {
            enabled: true,
            ..KldRefAuditSummary::default()
        };
    };
    let n_positions = kldref
        .get("n_positions")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(0);
    let top_k = kldref
        .get("top_k")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(0);
    if n_positions == 0 {
        errors.push("KLDREF n_positions must be greater than zero".into());
    }
    if top_k != job.options.kldref_top_k {
        errors.push(format!(
            "KLDREF top-k {top_k} differs from job top-k {}",
            job.options.kldref_top_k
        ));
    }

    match kldref.get("position_map").and_then(Value::as_array) {
        Some(position_map) => {
            if position_map.len() != n_positions {
                errors.push(format!(
                    "KLDREF position_map has {} rows, expected {n_positions}",
                    position_map.len()
                ));
            }
            let mut unique = BTreeSet::new();
            for value in position_map {
                match serde_json::from_value::<SamplePosition>(value.clone()) {
                    Ok(position) => {
                        let valid = job
                            .samples
                            .samples()
                            .get(position.sample_index)
                            .is_some_and(|sample| position.position < sample.tokens.len());
                        if !valid {
                            errors.push(format!(
                                "KLDREF position ({},{}) is outside the serialized sample set",
                                position.sample_index, position.position
                            ));
                        }
                        if !unique.insert((position.sample_index, position.position)) {
                            errors.push(format!(
                                "KLDREF position ({},{}) is duplicated",
                                position.sample_index, position.position
                            ));
                        }
                    }
                    Err(error) => errors.push(format!("invalid KLDREF position_map row: {error}")),
                }
            }
        }
        None => errors.push("KLDREF metadata lacks position_map".into()),
    }

    let matrix_shape = [n_positions as u32, top_k as u32];
    let vector_shape = [n_positions as u32];
    for (name, expected_shape) in [
        (names[0], matrix_shape.as_slice()),
        (names[1], matrix_shape.as_slice()),
        (names[2], vector_shape.as_slice()),
    ] {
        match hfq.find_tensor_info(name) {
            Some(tensor) => validate_f32_tensor(tensor, expected_shape, errors),
            None => errors.push(format!("artifact lacks {name}")),
        }
    }
    KldRefAuditSummary {
        enabled: true,
        n_positions,
        top_k,
    }
}

/// Audit metadata and tensor-index invariants without reading payload pages.
/// Numerical finiteness remains the responsibility of the full comparison and
/// quality gates; the report states that limitation explicitly.
pub fn audit_calibration_artifact(path: &Path) -> Result<CalibrationAuditReport, Box<dyn Error>> {
    let hfq = HfqFile::open_index_only(path)?;
    let metadata: Value = serde_json::from_str(&hfq.metadata_json)?;
    let identity = index_identity(&hfq, metadata.clone());
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut summary = CalibrationAuditSummary::default();

    if metadata.get("artifact_kind").and_then(Value::as_str) != Some("calibration") {
        errors.push("artifact_kind must be calibration".into());
    }
    if metadata.get("arch_id").and_then(Value::as_u64) != Some(u64::from(hfq.arch_id)) {
        errors.push("metadata arch_id differs from the HFQ header".into());
    }
    summary.family = nonempty_string(&metadata, "family", &mut errors);
    summary.adapter_version = nonempty_string(&metadata, "adapter_version", &mut errors);
    nonempty_string(&metadata, "run_fingerprint", &mut errors);
    if metadata
        .get("engine_build")
        .and_then(Value::as_str)
        .is_none()
    {
        warnings.push("artifact predates engine_build provenance metadata".into());
    }

    validate_calibration_tensors(&hfq, &metadata, &mut summary, &mut errors);
    let job: Option<CalibrationJob> = match metadata.get("job").cloned().map(serde_json::from_value)
    {
        Some(Ok(job)) => Some(job),
        Some(Err(error)) => {
            errors.push(format!("invalid calibration job: {error}"));
            None
        }
        None => {
            errors.push("metadata lacks calibration job".into());
            None
        }
    };
    if let Some(job) = job.as_ref() {
        summary.job_schema_version = Some(job.schema_version);
        summary.samples = job.samples.samples().len();
        summary.sample_rows = job.samples.total_rows();
        summary.context_length = job.samples.context_len();
        if job.schema_version == 0 || job.schema_version > CALIBRATION_JOB_SCHEMA_VERSION {
            errors.push(format!(
                "unsupported calibration job schema {}, current maximum is {CALIBRATION_JOB_SCHEMA_VERSION}",
                job.schema_version
            ));
        } else if job.schema_version < CALIBRATION_JOB_SCHEMA_VERSION {
            warnings.push(format!(
                "calibration job schema {} predates current schema {CALIBRATION_JOB_SCHEMA_VERSION}",
                job.schema_version
            ));
        }
        if let Err(error) = job.options.validate() {
            errors.push(error.to_string());
        }
        if job.source_fingerprint.is_empty()
            || job.tokenizer_fingerprint.is_empty()
            || job.corpus_fingerprint.is_empty()
        {
            errors.push("calibration job provenance fingerprints must be nonempty".into());
        }
        if metadata
            .get("source_manifest")
            .and_then(|value| value.get("fingerprint"))
            .and_then(Value::as_str)
            != Some(job.source_fingerprint.as_str())
        {
            errors.push("source_manifest fingerprint differs from the calibration job".into());
        }
        validate_job_recording(&metadata, job, &mut errors);
        validate_geometry(&metadata, job, &mut errors);
        summary.experts = validate_experts(&metadata, job, &mut errors, &mut warnings);
        summary.kldref = validate_kldref(&hfq, &metadata, job, &mut errors);
    }

    match metadata
        .get("read_ledger")
        .cloned()
        .map(serde_json::from_value::<ReadLedgerSnapshot>)
    {
        Some(Ok(ledger)) => summary.read_ledger = Some(validate_ledger(&ledger, &mut errors)),
        Some(Err(error)) => errors.push(format!("invalid read_ledger: {error}")),
        None => errors.push("metadata lacks read_ledger".into()),
    }
    let max_consistency = metadata.get("max_consistency").and_then(Value::as_f64);
    if !max_consistency.is_some_and(|value| value.is_finite() && value >= 0.0) {
        errors.push("max_consistency must be finite and nonnegative".into());
    }

    Ok(CalibrationAuditReport {
        schema: AUDIT_SCHEMA,
        artifact: path.to_path_buf(),
        artifact_fingerprint: index_fingerprint(&identity)?,
        fingerprint_scope: "hfq_metadata_and_tensor_index_v1",
        bytes: std::fs::metadata(path)?.len(),
        version: hfq.version,
        arch_id: hfq.arch_id,
        tensor_count: hfq.tensors().len(),
        index_only: true,
        payload_values_checked: false,
        valid: errors.is_empty(),
        summary,
        errors,
        warnings,
    })
}

pub fn run_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    let input = args
        .windows(2)
        .find(|pair| pair[0] == "--input")
        .map(|pair| PathBuf::from(&pair[1]))
        .ok_or("artifact audit-calibration requires --input <artifact.calib.hfq>")?;
    if args.len() != 2 {
        return Err("artifact audit-calibration accepts only --input <artifact.calib.hfq>".into());
    }
    let report = audit_calibration_artifact(&input)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !report.valid {
        return Err(format!(
            "calibration artifact audit failed with {} error(s)",
            report.errors.len()
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::calibration::contracts::{
        BoundaryPrecision, CalibrationOptions, CalibrationSample, ExpertCaptureQuota,
        ExpertCaptureStats, ExpertSamplingPolicy, LayerRouterStats, SampleSet, WeightStats,
    };
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
    use serde_json::json;

    fn weight_stats(count: u64) -> WeightStats {
        WeightStats {
            count,
            sum: count as f64,
            sum_squared: count as f64,
        }
    }

    fn capture_stats(rows: u64, tile_rows: usize) -> ExpertCaptureStats {
        ExpertCaptureStats {
            seen_rows: rows,
            admitted_rows: rows,
            launch_telemetry_recorded: true,
            capture_gather_launches: 1,
            full_reduction_tiles: rows / tile_rows as u64,
            partial_reduction_tiles: u64::from(rows % tile_rows as u64 != 0),
            full_weight: weight_stats(rows),
            admitted_weight: weight_stats(rows),
            ..ExpertCaptureStats::default()
        }
    }

    fn job(min_rows: u64, policy: ExpertCoveragePolicy) -> CalibrationJob {
        let samples = SampleSet::new(
            vec![CalibrationSample::new("sample-0", vec![1, 2], "test")],
            2,
            1,
        )
        .unwrap();
        CalibrationJob::new(
            "source",
            "tokenizer",
            samples,
            CalibrationOptions {
                sequence_batch: Some(1),
                time_tile: Some(2),
                max_rows: 2,
                boundary_precision: BoundaryPrecision::F32,
                expert_quota: ExpertCaptureQuota {
                    min_rows,
                    target_rows: min_rows.max(2),
                    tile_rows: 2,
                    sampling: ExpertSamplingPolicy::DeterministicFirst { seed: 1 },
                },
                required_expert_fraction: 1.0,
                expert_coverage_policy: policy,
                kldref: true,
                kldref_top_k: 2,
            },
        )
        .unwrap()
        .with_corpus_fingerprint("corpus")
        .unwrap()
    }

    fn telemetry(job: &CalibrationJob) -> ExpertLayerTelemetry {
        ExpertLayerTelemetry {
            layer: 0,
            num_experts: 1,
            k_top: 1,
            quota: job.options.expert_quota,
            router: LayerRouterStats {
                routed_tokens: 2,
                routed_slots: 2,
                microbatches: 1,
                active_expert_sum: 1,
                max_active_experts: 1,
                top1_hits: vec![2],
                topk_hits: vec![2],
                route_weights: vec![weight_stats(2)],
                ..LayerRouterStats::default()
            },
            gate_up: vec![capture_stats(2, 2)],
            down: vec![capture_stats(2, 2)],
        }
    }

    fn tensors(include_kld_logz: bool) -> Vec<HfqMemTensor> {
        let mut tensors = vec![
            HfqMemTensor {
                name: "dense.0.hessian".into(),
                quant_type: F32_QUANT_TYPE,
                shape: vec![2, 2],
                group_size: 0,
                data: vec![0; 16],
            },
            HfqMemTensor {
                name: "dense.0.imatrix".into(),
                quant_type: F32_QUANT_TYPE,
                shape: vec![2],
                group_size: 0,
                data: vec![0; 8],
            },
            HfqMemTensor {
                name: "lm_head.kldref_idx".into(),
                quant_type: F32_QUANT_TYPE,
                shape: vec![1, 2],
                group_size: 0,
                data: vec![0; 8],
            },
            HfqMemTensor {
                name: "lm_head.kldref_logit".into(),
                quant_type: F32_QUANT_TYPE,
                shape: vec![1, 2],
                group_size: 0,
                data: vec![0; 8],
            },
        ];
        if include_kld_logz {
            tensors.push(HfqMemTensor {
                name: "lm_head.kldref_logz".into(),
                quant_type: F32_QUANT_TYPE,
                shape: vec![1],
                group_size: 0,
                data: vec![0; 4],
            });
        }
        tensors
    }

    fn metadata(job: &CalibrationJob, telemetry: ExpertLayerTelemetry) -> Value {
        json!({
            "artifact_kind": "calibration",
            "engine_build": "fixture-engine",
            "family": "fixture",
            "adapter_version": "fixture.v1",
            "arch_id": 77,
            "source_manifest": {"fingerprint": "source", "shards": []},
            "effective_precision": {"boundary": "f32", "source_dtypes": ["F32"]},
            "run_fingerprint": "run",
            "job": job,
            "microbatch_geometry": {"sequence_batch": 1, "time_tile": 2, "row_budget": 2},
            "geometry_tuning": {"selected": {"sequence_batch": 1, "time_tile": 2, "row_budget": 2}},
            "expert_telemetry": [telemetry],
            "expert_capture_quota": {
                "minimum_rows": job.options.expert_quota.min_rows,
                "target_rows": job.options.expert_quota.target_rows,
                "limit_rows": job.options.expert_quota.limit_rows().unwrap(),
                "tile_rows": job.options.expert_quota.tile_rows,
                "maximum_batch_slack_rows": job.options.expert_quota.tile_rows - 1,
                "sampling": job.options.expert_quota.sampling
            },
            "preserve_high_precision": [],
            "read_ledger": {
                "planned_logical": ["dense.0.weight"],
                "consumed_logical": ["dense.0.weight"],
                "read_canonical": ["dense.0.weight"],
                "logical_bytes_read": 16,
                "duplicate_logical": [],
                "missing_logical": []
            },
            "n_hessian": 1,
            "n_imatrix": 1,
            "per_tensor_tokens": {"dense.0": 2},
            "max_consistency": 0.0,
            "kldref": {
                "n_positions": 1,
                "top_k": 2,
                "position_map": [{"sample_index": 0, "position": 0}]
            }
        })
    }

    fn write_fixture(name: &str, metadata: &Value, tensors: &[HfqMemTensor]) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "hipfire-calibration-audit-{}-{name}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("fixture.calib.hfq");
        write_hfqm_package_mem(&path, 77, &metadata.to_string(), tensors).unwrap();
        path
    }

    #[test]
    fn valid_artifact_reconciles_ledger_kldref_and_experts_index_only() {
        let job = job(2, ExpertCoveragePolicy::PreserveUndercovered);
        let path = write_fixture("valid", &metadata(&job, telemetry(&job)), &tensors(true));
        let report = audit_calibration_artifact(&path).unwrap();
        assert!(report.valid, "{:?}", report.errors);
        assert!(report.index_only);
        assert!(!report.payload_values_checked);
        assert_eq!(report.summary.experts.deficit_capture_points, 0);
        assert_eq!(report.summary.kldref.n_positions, 1);
        assert_eq!(report.summary.read_ledger.unwrap().missing_logical, 0);
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn missing_kld_tensor_is_rejected() {
        let job = job(2, ExpertCoveragePolicy::PreserveUndercovered);
        let path = write_fixture(
            "missing-kld",
            &metadata(&job, telemetry(&job)),
            &tensors(false),
        );
        let report = audit_calibration_artifact(&path).unwrap();
        assert!(!report.valid);
        assert!(report
            .errors
            .iter()
            .any(|error| error.contains("lacks lm_head.kldref_logz")));
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn preserve_fallback_must_exactly_match_undercovered_experts() {
        let job = job(3, ExpertCoveragePolicy::PreserveUndercovered);
        let layer = telemetry(&job);
        let mut valid_metadata = metadata(&job, layer.clone());
        valid_metadata["preserve_high_precision"] = json!([{"layer": 0, "expert": 0}]);
        let valid_path = write_fixture("preserve-valid", &valid_metadata, &tensors(true));
        let valid = audit_calibration_artifact(&valid_path).unwrap();
        assert!(valid.valid, "{:?}", valid.errors);
        assert_eq!(valid.summary.experts.deficit_capture_points, 2);

        let invalid_path =
            write_fixture("preserve-invalid", &metadata(&job, layer), &tensors(true));
        let invalid = audit_calibration_artifact(&invalid_path).unwrap();
        assert!(!invalid.valid);
        assert!(invalid
            .errors
            .iter()
            .any(|error| error.contains("differs from telemetry deficits")));
        std::fs::remove_dir_all(valid_path.parent().unwrap()).unwrap();
        std::fs::remove_dir_all(invalid_path.parent().unwrap()).unwrap();
    }

    #[test]
    fn malformed_expert_stream_accounting_is_rejected() {
        let job = job(2, ExpertCoveragePolicy::PreserveUndercovered);
        let mut layer = telemetry(&job);
        layer.down[0].admitted_weight.count = 1;
        let path = write_fixture("bad-expert", &metadata(&job, layer), &tensors(true));
        let report = audit_calibration_artifact(&path).unwrap();
        assert!(!report.valid);
        assert!(report
            .errors
            .iter()
            .any(|error| error.contains("admitted-stream weight stats")));
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn forged_sample_fingerprint_and_quota_recording_are_rejected() {
        let job = job(2, ExpertCoveragePolicy::PreserveUndercovered);
        let mut metadata = metadata(&job, telemetry(&job));
        metadata["job"]["samples"]["fingerprint"] = json!("forged");
        metadata["expert_capture_quota"]["target_rows"] = json!(99);
        let path = write_fixture("bad-job-recording", &metadata, &tensors(true));
        let report = audit_calibration_artifact(&path).unwrap();
        assert!(!report.valid);
        assert!(report
            .errors
            .iter()
            .any(|error| error.contains("sample ordering or fingerprint")));
        assert!(report
            .errors
            .iter()
            .any(|error| error.contains("target_rows differs")));
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }
}
