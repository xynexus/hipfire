// SPDX-License-Identifier: Apache-2.0
//! Numerical comparison for bounded resident/layer-streamed residual probes.

use hipfire_runtime::calibration::residual_probe::ResidualProbe;
use serde::Serialize;
use std::error::Error;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
pub struct ResidualCompareOptions {
    pub atol: f32,
    pub rtol: f32,
    pub max_reports: usize,
}

impl Default for ResidualCompareOptions {
    fn default() -> Self {
        Self {
            atol: 1.0e-5,
            rtol: 5.0e-3,
            max_reports: 50,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResidualMismatch {
    pub layer: usize,
    pub row: usize,
    pub column: usize,
    pub reference: f32,
    pub candidate: f32,
    pub abs_error: f32,
    pub rel_error: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResidualComparisonReport {
    pub schema_version: u32,
    pub reference: String,
    pub candidate: String,
    pub atol: f32,
    pub rtol: f32,
    pub family: String,
    pub sample_fingerprint: String,
    pub rows: usize,
    pub hidden_width: usize,
    pub compared_layers: usize,
    pub compared_values: u64,
    pub mismatched_values: u64,
    pub max_abs_error: f32,
    pub max_rel_error: f32,
    pub worst_layer: Option<usize>,
    pub structural_errors: Vec<String>,
    pub mismatches: Vec<ResidualMismatch>,
    pub reports_truncated: bool,
    pub ok: bool,
}

pub fn compare_residual_probes(
    reference_name: impl Into<String>,
    reference: &ResidualProbe,
    candidate_name: impl Into<String>,
    candidate: &ResidualProbe,
    options: ResidualCompareOptions,
) -> Result<ResidualComparisonReport, Box<dyn Error>> {
    if !options.atol.is_finite() || options.atol < 0.0 {
        return Err("residual comparison --atol must be finite and non-negative".into());
    }
    if !options.rtol.is_finite() || options.rtol < 0.0 {
        return Err("residual comparison --rtol must be finite and non-negative".into());
    }
    if options.max_reports == 0 {
        return Err("residual comparison --max-reports must be nonzero".into());
    }

    let mut structural_errors = Vec::new();
    let metadata_pairs = [
        (
            "architecture",
            reference.arch_id.to_string(),
            candidate.arch_id.to_string(),
        ),
        (
            "family",
            reference.metadata.family.clone(),
            candidate.metadata.family.clone(),
        ),
        (
            "sample_fingerprint",
            reference.metadata.sample_fingerprint.clone(),
            candidate.metadata.sample_fingerprint.clone(),
        ),
        (
            "source_fingerprint",
            reference.metadata.source_fingerprint.clone(),
            candidate.metadata.source_fingerprint.clone(),
        ),
        (
            "tokenizer_fingerprint",
            reference.metadata.tokenizer_fingerprint.clone(),
            candidate.metadata.tokenizer_fingerprint.clone(),
        ),
        (
            "corpus_fingerprint",
            reference.metadata.corpus_fingerprint.clone(),
            candidate.metadata.corpus_fingerprint.clone(),
        ),
        (
            "hidden_width",
            reference.metadata.hidden_width.to_string(),
            candidate.metadata.hidden_width.to_string(),
        ),
        (
            "total_layers",
            reference.metadata.total_layers.to_string(),
            candidate.metadata.total_layers.to_string(),
        ),
    ];
    for (field, left, right) in metadata_pairs {
        if left != right {
            structural_errors.push(format!("{field} {left:?} != {right:?}"));
        }
    }
    if reference.metadata.rows != candidate.metadata.rows {
        structural_errors.push("canonical residual probe rows differ".into());
    }
    if reference.layers.len() != candidate.layers.len() {
        structural_errors.push(format!(
            "layer payload count {} != {}",
            reference.layers.len(),
            candidate.layers.len()
        ));
    }

    let rows = reference.metadata.rows.len();
    let width = reference.metadata.hidden_width;
    let mut compared_values = 0u64;
    let mut mismatched_values = 0u64;
    let mut max_abs_error = 0.0f32;
    let mut max_rel_error = 0.0f32;
    let mut worst_layer = None;
    let mut mismatches = Vec::new();
    let compared_layers = reference.layers.len().min(candidate.layers.len());
    if structural_errors.is_empty() {
        for layer in 0..compared_layers {
            let left = &reference.layers[layer];
            let right = &candidate.layers[layer];
            if left.len() != rows * width || right.len() != rows * width {
                structural_errors.push(format!(
                    "layer {layer} payload lengths {}/{} do not match shape [{rows}, {width}]",
                    left.len(),
                    right.len()
                ));
                continue;
            }
            for (index, (&a, &b)) in left.iter().zip(right).enumerate() {
                compared_values += 1;
                let abs = (a - b).abs();
                let scale = a.abs().max(b.abs());
                let rel = abs / scale.max(f32::MIN_POSITIVE);
                if abs > max_abs_error {
                    max_abs_error = abs;
                    worst_layer = Some(layer);
                }
                max_rel_error = max_rel_error.max(rel);
                if !a.is_finite() || !b.is_finite() || abs > options.atol + options.rtol * scale {
                    mismatched_values += 1;
                    if mismatches.len() < options.max_reports {
                        mismatches.push(ResidualMismatch {
                            layer,
                            row: index / width,
                            column: index % width,
                            reference: a,
                            candidate: b,
                            abs_error: abs,
                            rel_error: rel,
                        });
                    }
                }
            }
        }
    }
    Ok(ResidualComparisonReport {
        schema_version: 1,
        reference: reference_name.into(),
        candidate: candidate_name.into(),
        atol: options.atol,
        rtol: options.rtol,
        family: reference.metadata.family.clone(),
        sample_fingerprint: reference.metadata.sample_fingerprint.clone(),
        rows,
        hidden_width: width,
        compared_layers,
        compared_values,
        mismatched_values,
        max_abs_error,
        max_rel_error,
        worst_layer,
        structural_errors: structural_errors.clone(),
        reports_truncated: mismatched_values as usize > mismatches.len(),
        mismatches,
        ok: structural_errors.is_empty() && mismatched_values == 0,
    })
}

pub fn run_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut reference = None;
    let mut candidate = None;
    let mut options = ResidualCompareOptions::default();
    let mut index = 0usize;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("compare-residuals: {flag} requires a value"))?;
        match flag {
            "--reference" => reference = Some(PathBuf::from(value)),
            "--candidate" => candidate = Some(PathBuf::from(value)),
            "--atol" => options.atol = value.parse()?,
            "--rtol" => options.rtol = value.parse()?,
            "--max-reports" => options.max_reports = value.parse()?,
            _ => return Err(format!("compare-residuals: unknown flag {flag}").into()),
        }
        index += 2;
    }
    let reference = reference.ok_or("compare-residuals: --reference <probe.hfq> is required")?;
    let candidate = candidate.ok_or("compare-residuals: --candidate <probe.hfq> is required")?;
    let left = ResidualProbe::read(Path::new(&reference))?;
    let right = ResidualProbe::read(Path::new(&candidate))?;
    let report = compare_residual_probes(
        reference.display().to_string(),
        &left,
        candidate.display().to_string(),
        &right,
        options,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !report.ok {
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::calibration::contracts::{
        BoundaryPrecision, CalibrationJob, CalibrationOptions, CalibrationSample,
        ExpertCaptureQuota, ExpertCoveragePolicy, ExpertSamplingPolicy, SampleSet,
    };

    fn probe(producer: &str, drift: f32) -> ResidualProbe {
        let samples =
            SampleSet::new(vec![CalibrationSample::new("a", vec![1, 2], "test")], 2, 1).unwrap();
        let job = CalibrationJob::new(
            "source",
            "tokenizer",
            samples,
            CalibrationOptions {
                sequence_batch: Some(1),
                time_tile: Some(2),
                max_rows: 2,
                boundary_precision: BoundaryPrecision::F32,
                expert_quota: ExpertCaptureQuota {
                    min_rows: 1,
                    target_rows: 1,
                    tile_rows: 1,
                    sampling: ExpertSamplingPolicy::DeterministicFirst { seed: 1 },
                },
                required_expert_fraction: 1.0,
                expert_coverage_policy: ExpertCoveragePolicy::Strict,
                kldref: false,
                kldref_top_k: 1,
                kldref_rows: None,
            },
        )
        .unwrap();
        let mut probe = ResidualProbe::new(6, "qwen3.5", producer, &job, 2, 1, 2).unwrap();
        probe
            .push_layer(0, vec![1.0 + drift, 2.0, 3.0, 4.0])
            .unwrap();
        probe
    }

    #[test]
    fn residual_comparison_accepts_tolerance_and_reports_drift() {
        let reference = probe("resident", 0.0);
        let near = probe("streamed", 1.0e-6);
        assert!(
            compare_residual_probes(
                "resident",
                &reference,
                "streamed",
                &near,
                ResidualCompareOptions::default()
            )
            .unwrap()
            .ok
        );
        let far = probe("streamed", 0.1);
        let report = compare_residual_probes(
            "resident",
            &reference,
            "streamed",
            &far,
            ResidualCompareOptions::default(),
        )
        .unwrap();
        assert!(!report.ok);
        assert_eq!(report.mismatched_values, 1);
    }
}
