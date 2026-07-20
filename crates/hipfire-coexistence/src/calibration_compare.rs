// SPDX-License-Identifier: Apache-2.0
//! Family-neutral numerical comparison for resident and layer-streamed
//! calibration artifacts.
//!
//! This belongs in coexistence tooling rather than an inference binary: it
//! reads two completed HFQM packages, verifies their matched-corpus contract,
//! and compares every logical Hessian, imatrix, and KLDREF value.

use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::path::{Path, PathBuf};

const F32_QUANT_TYPE: u8 = 2;
const COMPACT_HESSIAN_QUANT_TYPE: u8 = 130;

#[derive(Debug, Clone, Copy)]
pub struct CalibrationCompareOptions {
    pub atol: f32,
    pub rtol: f32,
    pub max_reports: usize,
    pub require_provenance: bool,
}

impl Default for CalibrationCompareOptions {
    fn default() -> Self {
        Self {
            atol: 1.0e-5,
            rtol: 5.0e-3,
            max_reports: 50,
            require_provenance: true,
        }
    }
}

impl CalibrationCompareOptions {
    fn validate(self) -> Result<Self, Box<dyn Error>> {
        if !self.atol.is_finite() || self.atol < 0.0 {
            return Err("calibration comparison --atol must be finite and non-negative".into());
        }
        if !self.rtol.is_finite() || self.rtol < 0.0 {
            return Err("calibration comparison --rtol must be finite and non-negative".into());
        }
        if self.max_reports == 0 {
            return Err("calibration comparison --max-reports must be nonzero".into());
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetadataCheckStatus {
    Match,
    Mismatch,
    Unavailable,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetadataCheck {
    pub field: String,
    pub status: MetadataCheckStatus,
    pub required_for_matched_provenance: bool,
    pub reference: Option<Value>,
    pub candidate: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorMismatch {
    pub name: String,
    pub values_compared: u64,
    pub mismatched_values: u64,
    pub non_finite_values: u64,
    pub max_abs_error: f32,
    pub max_rel_error: f32,
    pub reference_quant_type: u8,
    pub candidate_quant_type: u8,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouterParityReport {
    pub reference_format: Option<String>,
    pub candidate_format: Option<String>,
    pub compared_layers: usize,
    pub compared_weight_sums: u64,
    pub mismatched_layers: usize,
    pub mismatched_weight_sums: u64,
    pub max_weight_abs_error: f64,
    pub max_weight_rel_error: f64,
    pub errors: Vec<String>,
    pub ok: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationComparisonReport {
    pub schema_version: u32,
    pub reference: String,
    pub candidate: String,
    pub atol: f32,
    pub rtol: f32,
    pub provenance_required: bool,
    pub provenance_complete: bool,
    pub metadata_checks: Vec<MetadataCheck>,
    pub router_parity: Option<RouterParityReport>,
    pub reference_tensor_count: usize,
    pub candidate_tensor_count: usize,
    pub compared_tensors: usize,
    pub compared_values: u64,
    pub mismatched_tensors: usize,
    pub mismatched_values: u64,
    pub non_finite_values: u64,
    pub max_abs_error: f32,
    pub max_rel_error: f32,
    pub worst_tensor: Option<String>,
    pub missing_from_reference: Vec<String>,
    pub missing_from_candidate: Vec<String>,
    pub structural_errors: Vec<String>,
    pub tensor_mismatches: Vec<TensorMismatch>,
    pub reports_truncated: bool,
    pub ok: bool,
}

/// Compare completed calibration artifacts without loading a model or using a
/// GPU. Dense F32 and compact BF16-triangle Hessians share one logical view, so
/// a resident oracle and streamed collector may use different storage encodings
/// without turning storage precision into a false structural failure.
pub fn compare_calibration_artifacts(
    reference_path: &Path,
    candidate_path: &Path,
    options: CalibrationCompareOptions,
) -> Result<CalibrationComparisonReport, Box<dyn Error>> {
    let options = options.validate()?;
    let reference = HfqFile::open_index_only(reference_path)?;
    let candidate = HfqFile::open_index_only(candidate_path)?;
    let reference_metadata: Value = serde_json::from_str(&reference.metadata_json)?;
    let candidate_metadata: Value = serde_json::from_str(&candidate.metadata_json)?;

    validate_calibration_kind("reference", &reference_metadata)?;
    validate_calibration_kind("candidate", &candidate_metadata)?;

    let mut metadata_checks = vec![metadata_check(
        "arch_id",
        Some(Value::from(reference.arch_id)),
        Some(Value::from(candidate.arch_id)),
        false,
    )];
    for (field, pointer, required) in [
        ("family", "/family", false),
        ("n_hessian", "/n_hessian", false),
        ("n_imatrix", "/n_imatrix", false),
        ("per_tensor_tokens", "/per_tensor_tokens", false),
        ("kldref", "/kldref", false),
        ("corpus_fingerprint", "/job/corpus_fingerprint", true),
        ("sample_fingerprint", "/job/samples/fingerprint", true),
    ] {
        metadata_checks.push(metadata_check(
            field,
            reference_metadata.pointer(pointer).cloned(),
            candidate_metadata.pointer(pointer).cloned(),
            required,
        ));
    }
    let provenance_complete = metadata_checks
        .iter()
        .filter(|check| check.required_for_matched_provenance)
        .all(|check| matches!(check.status, MetadataCheckStatus::Match));
    let router_parity = compare_router_telemetry(
        &reference_metadata,
        &candidate_metadata,
        options.atol as f64,
        options.rtol as f64,
    );

    let reference_tensors = tensor_map(reference.tensors());
    let candidate_tensors = tensor_map(candidate.tensors());
    let reference_names = reference_tensors.keys().cloned().collect::<BTreeSet<_>>();
    let candidate_names = candidate_tensors.keys().cloned().collect::<BTreeSet<_>>();
    let missing_from_reference = candidate_names
        .difference(&reference_names)
        .cloned()
        .collect::<Vec<_>>();
    let missing_from_candidate = reference_names
        .difference(&candidate_names)
        .cloned()
        .collect::<Vec<_>>();

    let mut structural_errors = Vec::new();
    let mut mismatches = Vec::new();
    let mut compared_tensors = 0usize;
    let mut compared_values = 0u64;
    let mut mismatched_values = 0u64;
    let mut non_finite_values = 0u64;
    let mut max_abs_error = 0.0f32;
    let mut max_rel_error = 0.0f32;
    let mut worst_tensor = None;

    for name in reference_names.intersection(&candidate_names) {
        let reference_info = reference_tensors[name];
        let candidate_info = candidate_tensors[name];
        if reference_info.shape != candidate_info.shape {
            structural_errors.push(format!(
                "{name}: shape {:?} != {:?}",
                reference_info.shape, candidate_info.shape
            ));
            continue;
        }
        if reference_info.group_size != candidate_info.group_size {
            structural_errors.push(format!(
                "{name}: group_size {} != {}",
                reference_info.group_size, candidate_info.group_size
            ));
            continue;
        }
        let Some((_, reference_data)) = reference.tensor_data_vec(name) else {
            structural_errors.push(format!("{name}: could not read reference payload"));
            continue;
        };
        let Some((_, candidate_data)) = candidate.tensor_data_vec(name) else {
            structural_errors.push(format!("{name}: could not read candidate payload"));
            continue;
        };
        let reference_values = match NumericValues::new(reference_info, &reference_data) {
            Ok(values) => values,
            Err(error) => {
                structural_errors.push(format!("{name}: reference {error}"));
                continue;
            }
        };
        let candidate_values = match NumericValues::new(candidate_info, &candidate_data) {
            Ok(values) => values,
            Err(error) => {
                structural_errors.push(format!("{name}: candidate {error}"));
                continue;
            }
        };
        if reference_values.len() != candidate_values.len() {
            structural_errors.push(format!(
                "{name}: logical value count {} != {}",
                reference_values.len(),
                candidate_values.len()
            ));
            continue;
        }

        let exact = name == "lm_head.kldref_idx";
        let (tensor_atol, tensor_rtol) = if exact {
            (0.0, 0.0)
        } else {
            (options.atol, options.rtol)
        };
        let mut tensor = TensorMismatch {
            name: name.clone(),
            values_compared: 0,
            mismatched_values: 0,
            non_finite_values: 0,
            max_abs_error: 0.0,
            max_rel_error: 0.0,
            reference_quant_type: reference_info.quant_type,
            candidate_quant_type: candidate_info.quant_type,
        };
        for (left, right) in reference_values.zip(candidate_values) {
            tensor.values_compared += 1;
            if !left.is_finite() || !right.is_finite() {
                tensor.non_finite_values += 1;
                tensor.mismatched_values += 1;
                continue;
            }
            let abs = (left - right).abs();
            let scale = left.abs().max(right.abs());
            let rel = abs / scale.max(f32::MIN_POSITIVE);
            tensor.max_abs_error = tensor.max_abs_error.max(abs);
            tensor.max_rel_error = tensor.max_rel_error.max(rel);
            if abs > tensor_atol + tensor_rtol * scale {
                tensor.mismatched_values += 1;
            }
        }
        compared_tensors += 1;
        compared_values += tensor.values_compared;
        mismatched_values += tensor.mismatched_values;
        non_finite_values += tensor.non_finite_values;
        max_abs_error = max_abs_error.max(tensor.max_abs_error);
        if tensor.max_rel_error > max_rel_error {
            max_rel_error = tensor.max_rel_error;
            worst_tensor = Some(name.clone());
        }
        if tensor.mismatched_values > 0 {
            mismatches.push(tensor);
        }
    }

    mismatches.sort_by(|left, right| {
        right
            .mismatched_values
            .cmp(&left.mismatched_values)
            .then_with(|| right.max_rel_error.total_cmp(&left.max_rel_error))
            .then_with(|| left.name.cmp(&right.name))
    });
    let mismatched_tensors = mismatches.len();
    let reports_truncated = mismatches.len() > options.max_reports;
    mismatches.truncate(options.max_reports);
    let metadata_mismatch = metadata_checks
        .iter()
        .any(|check| matches!(check.status, MetadataCheckStatus::Mismatch));
    let ok = missing_from_reference.is_empty()
        && missing_from_candidate.is_empty()
        && structural_errors.is_empty()
        && mismatched_values == 0
        && !metadata_mismatch
        && router_parity.as_ref().is_none_or(|report| report.ok)
        && (!options.require_provenance || provenance_complete);

    Ok(CalibrationComparisonReport {
        schema_version: 2,
        reference: reference_path.display().to_string(),
        candidate: candidate_path.display().to_string(),
        atol: options.atol,
        rtol: options.rtol,
        provenance_required: options.require_provenance,
        provenance_complete,
        metadata_checks,
        router_parity,
        reference_tensor_count: reference_tensors.len(),
        candidate_tensor_count: candidate_tensors.len(),
        compared_tensors,
        compared_values,
        mismatched_tensors,
        mismatched_values,
        non_finite_values,
        max_abs_error,
        max_rel_error,
        worst_tensor,
        missing_from_reference,
        missing_from_candidate,
        structural_errors,
        tensor_mismatches: mismatches,
        reports_truncated,
        ok,
    })
}

#[derive(Debug)]
struct RouterLayer {
    routed_tokens: u64,
    routed_slots: u64,
    dropped_indices: u64,
    top1_hits: Vec<u64>,
    topk_hits: Vec<u64>,
    weight_sums: Vec<f64>,
}

fn compare_router_telemetry(
    reference: &Value,
    candidate: &Value,
    atol: f64,
    rtol: f64,
) -> Option<RouterParityReport> {
    let reference = normalize_router_layers(reference);
    let candidate = normalize_router_layers(candidate);
    if reference.is_none() && candidate.is_none() {
        return None;
    }
    let mut report = RouterParityReport {
        reference_format: reference.as_ref().map(|(format, _)| (*format).into()),
        candidate_format: candidate.as_ref().map(|(format, _)| (*format).into()),
        compared_layers: 0,
        compared_weight_sums: 0,
        mismatched_layers: 0,
        mismatched_weight_sums: 0,
        max_weight_abs_error: 0.0,
        max_weight_rel_error: 0.0,
        errors: Vec::new(),
        ok: false,
    };
    let Some((_, reference)) = reference else {
        report
            .errors
            .push("reference artifact has no per-layer router telemetry".into());
        return Some(report);
    };
    let Some((_, candidate)) = candidate else {
        report
            .errors
            .push("candidate artifact has no per-layer router telemetry".into());
        return Some(report);
    };

    for layer in reference
        .keys()
        .chain(candidate.keys())
        .copied()
        .collect::<BTreeSet<_>>()
    {
        let Some(left) = reference.get(&layer) else {
            report
                .errors
                .push(format!("router layer {layer} exists only in candidate"));
            report.mismatched_layers += 1;
            continue;
        };
        let Some(right) = candidate.get(&layer) else {
            report
                .errors
                .push(format!("router layer {layer} exists only in reference"));
            report.mismatched_layers += 1;
            continue;
        };
        report.compared_layers += 1;
        let mut layer_mismatch = false;
        for (field, a, b) in [
            ("routed_tokens", left.routed_tokens, right.routed_tokens),
            ("routed_slots", left.routed_slots, right.routed_slots),
            (
                "dropped_indices",
                left.dropped_indices,
                right.dropped_indices,
            ),
        ] {
            if a != b {
                report
                    .errors
                    .push(format!("router layer {layer} {field} {a} != {b}"));
                layer_mismatch = true;
            }
        }
        for (field, a, b) in [
            ("top1_hits", &left.top1_hits, &right.top1_hits),
            ("topk_hits", &left.topk_hits, &right.topk_hits),
        ] {
            if a != b {
                report.errors.push(format!(
                    "router layer {layer} {field} differs (reference_len={} candidate_len={})",
                    a.len(),
                    b.len()
                ));
                layer_mismatch = true;
            }
        }
        if left.weight_sums.len() != right.weight_sums.len() {
            report.errors.push(format!(
                "router layer {layer} weight_sums length {} != {}",
                left.weight_sums.len(),
                right.weight_sums.len()
            ));
            layer_mismatch = true;
        } else {
            for (&a, &b) in left.weight_sums.iter().zip(&right.weight_sums) {
                report.compared_weight_sums += 1;
                if !a.is_finite() || !b.is_finite() {
                    report.mismatched_weight_sums += 1;
                    layer_mismatch = true;
                    continue;
                }
                let abs = (a - b).abs();
                let scale = a.abs().max(b.abs());
                let rel = abs / scale.max(f64::MIN_POSITIVE);
                report.max_weight_abs_error = report.max_weight_abs_error.max(abs);
                report.max_weight_rel_error = report.max_weight_rel_error.max(rel);
                if abs > atol + rtol * scale {
                    report.mismatched_weight_sums += 1;
                    layer_mismatch = true;
                }
            }
        }
        if layer_mismatch {
            report.mismatched_layers += 1;
        }
    }
    report.ok = report.errors.is_empty()
        && report.mismatched_layers == 0
        && report.mismatched_weight_sums == 0;
    Some(report)
}

fn normalize_router_layers(
    metadata: &Value,
) -> Option<(&'static str, BTreeMap<usize, RouterLayer>)> {
    if metadata.get("moe_router_histogram").is_some() {
        let layers = metadata
            .pointer("/moe_router_histogram/per_layer")?
            .as_array()?;
        let mut normalized = BTreeMap::new();
        for layer in layers {
            let index = json_usize(layer, "layer")?;
            normalized.insert(
                index,
                RouterLayer {
                    routed_tokens: json_u64(layer, "routed_tokens")?,
                    routed_slots: json_u64(layer, "routed_slots")?,
                    dropped_indices: json_u64(layer, "dropped_indices")?,
                    top1_hits: json_u64_vec(layer, "top1_hits")?,
                    topk_hits: json_u64_vec(layer, "topk_hits")?,
                    weight_sums: json_f64_vec(layer, "weight_sums")?,
                },
            );
        }
        return Some(("resident_histogram", normalized));
    }
    let telemetry = metadata.get("expert_telemetry")?.as_array()?;
    let mut normalized = BTreeMap::new();
    for layer in telemetry {
        let index = json_usize(layer, "layer")?;
        let router = layer.get("router")?;
        let weight_sums = router
            .get("route_weights")?
            .as_array()?
            .iter()
            .map(|weight| weight.get("sum")?.as_f64())
            .collect::<Option<Vec<_>>>()?;
        normalized.insert(
            index,
            RouterLayer {
                routed_tokens: json_u64(router, "routed_tokens")?,
                routed_slots: json_u64(router, "routed_slots")?,
                dropped_indices: json_u64(router, "dropped_indices")?,
                top1_hits: json_u64_vec(router, "top1_hits")?,
                topk_hits: json_u64_vec(router, "topk_hits")?,
                weight_sums,
            },
        );
    }
    Some(("streamed_expert_telemetry", normalized))
}

fn json_usize(value: &Value, key: &str) -> Option<usize> {
    usize::try_from(json_u64(value, key)?).ok()
}

fn json_u64(value: &Value, key: &str) -> Option<u64> {
    value.get(key)?.as_u64()
}

fn json_u64_vec(value: &Value, key: &str) -> Option<Vec<u64>> {
    value
        .get(key)?
        .as_array()?
        .iter()
        .map(Value::as_u64)
        .collect()
}

fn json_f64_vec(value: &Value, key: &str) -> Option<Vec<f64>> {
    value
        .get(key)?
        .as_array()?
        .iter()
        .map(Value::as_f64)
        .collect()
}

pub fn run_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut reference = None;
    let mut candidate = None;
    let mut options = CalibrationCompareOptions::default();
    let mut index = 0usize;
    while index < args.len() {
        let flag = &args[index];
        match flag.as_str() {
            "--allow-unproven-provenance" => {
                options.require_provenance = false;
                index += 1;
            }
            "--reference" | "--candidate" | "--atol" | "--rtol" | "--max-reports" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| format!("artifact compare-calibration: {flag} needs a value"))?;
                match flag.as_str() {
                    "--reference" => reference = Some(PathBuf::from(value)),
                    "--candidate" => candidate = Some(PathBuf::from(value)),
                    "--atol" => options.atol = value.parse()?,
                    "--rtol" => options.rtol = value.parse()?,
                    "--max-reports" => options.max_reports = value.parse()?,
                    _ => unreachable!(),
                }
                index += 2;
            }
            _ => {
                return Err(format!("artifact compare-calibration: unknown argument {flag}").into())
            }
        }
    }
    let reference = reference
        .ok_or("artifact compare-calibration requires --reference <resident.calib.hfq>")?;
    let candidate = candidate
        .ok_or("artifact compare-calibration requires --candidate <streamed.calib.hfq>")?;
    let report = compare_calibration_artifacts(&reference, &candidate, options)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !report.ok {
        return Err("calibration artifacts are not within the requested parity contract".into());
    }
    Ok(())
}

fn validate_calibration_kind(label: &str, metadata: &Value) -> Result<(), Box<dyn Error>> {
    if metadata.get("artifact_kind").and_then(Value::as_str) != Some("calibration") {
        return Err(format!("{label} artifact is not a completed calibration package").into());
    }
    Ok(())
}

fn metadata_check(
    field: &str,
    reference: Option<Value>,
    candidate: Option<Value>,
    required_for_matched_provenance: bool,
) -> MetadataCheck {
    let status = match (&reference, &candidate) {
        (Some(left), Some(right)) if left == right => MetadataCheckStatus::Match,
        (Some(_), Some(_)) => MetadataCheckStatus::Mismatch,
        _ => MetadataCheckStatus::Unavailable,
    };
    MetadataCheck {
        field: field.into(),
        status,
        required_for_matched_provenance,
        reference,
        candidate,
    }
}

fn tensor_map(tensors: &[HfqTensorInfo]) -> BTreeMap<String, &HfqTensorInfo> {
    tensors
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor))
        .collect()
}

enum NumericValues<'a> {
    LinearF32 {
        data: &'a [u8],
        index: usize,
        count: usize,
    },
    DenseHessian {
        data: &'a [u8],
        k: usize,
        diag: usize,
        row: usize,
        column: usize,
        remaining: usize,
    },
    CompactHessian {
        data: &'a [u8],
        k: usize,
        diag: usize,
        triangle: usize,
        remaining: usize,
    },
}

impl<'a> NumericValues<'a> {
    fn new(info: &HfqTensorInfo, data: &'a [u8]) -> Result<Self, String> {
        let hessian = info.name.ends_with(".hessian");
        if hessian {
            if info.shape.len() != 2 || info.shape[0] != info.shape[1] {
                return Err(format!("Hessian has non-square shape {:?}", info.shape));
            }
            let k = info.shape[0] as usize;
            let logical = k
                .checked_mul(k + 1)
                .and_then(|values| values.checked_div(2))
                .ok_or_else(|| "Hessian logical size overflow".to_string())?;
            return match info.quant_type {
                F32_QUANT_TYPE => {
                    let expected = k
                        .checked_mul(k)
                        .and_then(|values| values.checked_mul(4))
                        .ok_or_else(|| "dense Hessian byte size overflow".to_string())?;
                    if data.len() != expected {
                        return Err(format!(
                            "dense F32 Hessian has {} bytes, expected {expected}",
                            data.len()
                        ));
                    }
                    Ok(Self::DenseHessian {
                        data,
                        k,
                        diag: 0,
                        row: 1,
                        column: 0,
                        remaining: logical,
                    })
                }
                COMPACT_HESSIAN_QUANT_TYPE => {
                    let expected = k
                        .checked_mul(4)
                        .and_then(|diag| {
                            k.checked_mul(k.saturating_sub(1))
                                .and_then(|triangle| diag.checked_add(triangle))
                        })
                        .ok_or_else(|| "compact Hessian byte size overflow".to_string())?;
                    if data.len() != expected {
                        return Err(format!(
                            "compact Hessian has {} bytes, expected {expected}",
                            data.len()
                        ));
                    }
                    Ok(Self::CompactHessian {
                        data,
                        k,
                        diag: 0,
                        triangle: 0,
                        remaining: logical,
                    })
                }
                quant_type => Err(format!("unsupported Hessian quant type {quant_type}")),
            };
        }
        if info.quant_type != F32_QUANT_TYPE {
            return Err(format!(
                "unsupported calibration tensor quant type {}",
                info.quant_type
            ));
        }
        let count = checked_shape_elements(&info.shape)?;
        let expected = count
            .checked_mul(4)
            .ok_or_else(|| "F32 tensor byte size overflow".to_string())?;
        if data.len() != expected {
            return Err(format!(
                "F32 tensor has {} bytes, expected {expected}",
                data.len()
            ));
        }
        Ok(Self::LinearF32 {
            data,
            index: 0,
            count,
        })
    }

    fn len(&self) -> usize {
        match self {
            Self::LinearF32 { index, count, .. } => count - index,
            Self::DenseHessian { remaining, .. } | Self::CompactHessian { remaining, .. } => {
                *remaining
            }
        }
    }
}

impl Iterator for NumericValues<'_> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::LinearF32 { data, index, count } => {
                if *index == *count {
                    return None;
                }
                let value = read_f32(data, *index * 4);
                *index += 1;
                Some(value)
            }
            Self::DenseHessian {
                data,
                k,
                diag,
                row,
                column,
                remaining,
            } => {
                if *remaining == 0 {
                    return None;
                }
                *remaining -= 1;
                if *diag < *k {
                    let value = read_f32(data, (*diag * *k + *diag) * 4);
                    *diag += 1;
                    return Some(value);
                }
                let value = read_f32(data, (*row * *k + *column) * 4);
                *column += 1;
                if *column == *row {
                    *row += 1;
                    *column = 0;
                }
                Some(value)
            }
            Self::CompactHessian {
                data,
                k,
                diag,
                triangle,
                remaining,
            } => {
                if *remaining == 0 {
                    return None;
                }
                *remaining -= 1;
                if *diag < *k {
                    let value = read_f32(data, *diag * 4);
                    *diag += 1;
                    return Some(value);
                }
                let offset = *k * 4 + *triangle * 2;
                *triangle += 1;
                let bits = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());
                Some(f32::from_bits(u32::from(bits) << 16))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for NumericValues<'_> {}

fn checked_shape_elements(shape: &[u32]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |product, dimension| {
        product
            .checked_mul(*dimension as usize)
            .ok_or_else(|| "tensor element count overflow".to_string())
    })
}

fn read_f32(data: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn metadata(corpus: &str, sample: &str, tokens: u64) -> String {
        serde_json::json!({
            "artifact_kind": "calibration",
            "family": "fixture",
            "n_hessian": 1,
            "n_imatrix": 1,
            "per_tensor_tokens": {"layer.0.proj": tokens},
            "job": {
                "corpus_fingerprint": corpus,
                "samples": {"fingerprint": sample}
            }
        })
        .to_string()
    }

    fn write_fixture(path: &Path, meta: &str, imatrix: &[f32], hessian: &[f32]) {
        write_hfqm_package_mem(
            path,
            5,
            meta,
            &[
                HfqMemTensor {
                    name: "layer.0.proj.hessian".into(),
                    quant_type: 2,
                    shape: vec![2, 2],
                    group_size: 0,
                    data: f32_bytes(hessian),
                },
                HfqMemTensor {
                    name: "layer.0.proj.imatrix".into(),
                    quant_type: 2,
                    shape: vec![2],
                    group_size: 0,
                    data: f32_bytes(imatrix),
                },
            ],
        )
        .unwrap();
    }

    fn fixture_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "hipfire-calibration-compare-{label}-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn identical_matched_artifacts_pass() {
        let root = fixture_root("identical");
        let reference = root.join("resident.calib.hfq");
        let candidate = root.join("streamed.calib.hfq");
        let meta = metadata("corpus-a", "samples-a", 8);
        write_fixture(&reference, &meta, &[1.0, 2.0], &[1.0, 0.25, 0.25, 2.0]);
        write_fixture(&candidate, &meta, &[1.0, 2.0], &[1.0, 0.25, 0.25, 2.0]);

        let report = compare_calibration_artifacts(
            &reference,
            &candidate,
            CalibrationCompareOptions::default(),
        )
        .unwrap();
        assert!(report.ok);
        assert!(report.provenance_complete);
        assert_eq!(report.compared_tensors, 2);
        assert_eq!(report.mismatched_values, 0);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn numerical_and_provenance_drift_fail() {
        let root = fixture_root("drift");
        let reference = root.join("resident.calib.hfq");
        let candidate = root.join("streamed.calib.hfq");
        write_fixture(
            &reference,
            &metadata("corpus-a", "samples-a", 8),
            &[1.0, 2.0],
            &[1.0, 0.25, 0.25, 2.0],
        );
        write_fixture(
            &candidate,
            &metadata("corpus-b", "samples-b", 8),
            &[1.0, 3.0],
            &[1.0, 0.25, 0.25, 2.0],
        );

        let report = compare_calibration_artifacts(
            &reference,
            &candidate,
            CalibrationCompareOptions::default(),
        )
        .unwrap();
        assert!(!report.ok);
        assert!(!report.provenance_complete);
        assert_eq!(report.mismatched_tensors, 1);
        assert_eq!(report.mismatched_values, 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn compact_and_dense_hessians_share_a_logical_view() {
        let dense_info = HfqTensorInfo {
            name: "x.hessian".into(),
            quant_type: 2,
            shape: vec![3, 3],
            group_size: 0,
            data_offset: 0,
            data_size: 36,
        };
        let compact_info = HfqTensorInfo {
            name: "x.hessian".into(),
            quant_type: 130,
            shape: vec![3, 3],
            group_size: 0,
            data_offset: 0,
            data_size: 18,
        };
        let dense = f32_bytes(&[1.0, 0.5, 0.25, 0.5, 2.0, 0.75, 0.25, 0.75, 3.0]);
        let mut compact = f32_bytes(&[1.0, 2.0, 3.0]);
        for value in [0.5f32, 0.25, 0.75] {
            compact.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        let dense_values = NumericValues::new(&dense_info, &dense)
            .unwrap()
            .collect::<Vec<_>>();
        let compact_values = NumericValues::new(&compact_info, &compact)
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(dense_values, compact_values);
        assert_eq!(dense_values, vec![1.0, 2.0, 3.0, 0.5, 0.25, 0.75]);
    }

    #[test]
    fn missing_provenance_is_only_allowed_explicitly() {
        let root = fixture_root("legacy");
        let reference = root.join("resident.calib.hfq");
        let candidate = root.join("streamed.calib.hfq");
        let meta = r#"{"artifact_kind":"calibration","family":"fixture","n_hessian":1,"n_imatrix":1,"per_tensor_tokens":{"layer.0.proj":8}}"#;
        write_fixture(&reference, meta, &[1.0, 2.0], &[1.0, 0.25, 0.25, 2.0]);
        write_fixture(&candidate, meta, &[1.0, 2.0], &[1.0, 0.25, 0.25, 2.0]);

        let strict = compare_calibration_artifacts(
            &reference,
            &candidate,
            CalibrationCompareOptions::default(),
        )
        .unwrap();
        assert!(!strict.ok);
        let relaxed = compare_calibration_artifacts(
            &reference,
            &candidate,
            CalibrationCompareOptions {
                require_provenance: false,
                ..CalibrationCompareOptions::default()
            },
        )
        .unwrap();
        assert!(relaxed.ok);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn router_parity_normalizes_resident_and_streamed_telemetry() {
        let resident = serde_json::json!({
            "moe_router_histogram": {
                "per_layer": [{
                    "layer": 3,
                    "routed_tokens": 2,
                    "routed_slots": 4,
                    "dropped_indices": 0,
                    "top1_hits": [1, 1],
                    "topk_hits": [2, 2],
                    "weight_sums": [1.25, 0.75]
                }]
            }
        });
        let streamed = serde_json::json!({
            "expert_telemetry": [{
                "layer": 3,
                "router": {
                    "routed_tokens": 2,
                    "routed_slots": 4,
                    "dropped_indices": 0,
                    "top1_hits": [1, 1],
                    "topk_hits": [2, 2],
                    "route_weights": [
                        {"count": 2, "sum": 1.25, "sum_squared": 0.8},
                        {"count": 2, "sum": 0.75, "sum_squared": 0.4}
                    ]
                }
            }]
        });
        let matching = compare_router_telemetry(&resident, &streamed, 1e-6, 1e-6).unwrap();
        assert!(matching.ok);
        assert_eq!(matching.compared_layers, 1);
        assert_eq!(matching.compared_weight_sums, 2);

        let mut drifted = streamed;
        drifted["expert_telemetry"][0]["router"]["topk_hits"] = serde_json::json!([3, 1]);
        let mismatch = compare_router_telemetry(&resident, &drifted, 1e-6, 1e-6).unwrap();
        assert!(!mismatch.ok);
        assert_eq!(mismatch.mismatched_layers, 1);
    }
}
