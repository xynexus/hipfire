// SPDX-License-Identifier: Apache-2.0
//! Bounded, family-neutral post-layer residual probes for collector parity.

use super::contracts::{CalibError, CalibrationJob};
use crate::hfq::{write_hfqm_package_mem, HfqFile, HfqMemTensor};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const RESIDUAL_PROBE_SCHEMA_VERSION: u32 = 1;
const F32_QUANT_TYPE: u8 = 2;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidualProbeRow {
    pub sample_index: usize,
    pub position: usize,
    pub token: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidualProbeMetadata {
    pub schema_version: u32,
    pub artifact_kind: String,
    pub producer: String,
    pub family: String,
    pub source_fingerprint: String,
    pub tokenizer_fingerprint: String,
    pub sample_fingerprint: String,
    pub corpus_fingerprint: String,
    pub hidden_width: usize,
    pub total_layers: usize,
    pub rows: Vec<ResidualProbeRow>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResidualProbe {
    pub arch_id: u32,
    pub metadata: ResidualProbeMetadata,
    /// Sample-major rows for each post-layer boundary, in logical layer order.
    pub layers: Vec<Vec<f32>>,
}

impl ResidualProbe {
    pub fn new(
        arch_id: u32,
        family: impl Into<String>,
        producer: impl Into<String>,
        job: &CalibrationJob,
        hidden_width: usize,
        total_layers: usize,
        max_rows: usize,
    ) -> Result<Self, CalibError> {
        if hidden_width == 0 || total_layers == 0 || max_rows == 0 {
            return Err(CalibError::InvalidOptions(
                "residual probe width, layer count, and row limit must be nonzero".into(),
            ));
        }
        let rows = job
            .samples
            .samples()
            .iter()
            .enumerate()
            .flat_map(|(sample_index, sample)| {
                sample
                    .tokens
                    .iter()
                    .copied()
                    .enumerate()
                    .map(move |(position, token)| ResidualProbeRow {
                        sample_index,
                        position,
                        token,
                    })
            })
            .take(max_rows)
            .collect::<Vec<_>>();
        if rows.is_empty() {
            return Err(CalibError::InvalidSamples(
                "residual probe job contains no sample rows".into(),
            ));
        }
        Ok(Self {
            arch_id,
            metadata: ResidualProbeMetadata {
                schema_version: RESIDUAL_PROBE_SCHEMA_VERSION,
                artifact_kind: "calibration-residual-probe".into(),
                producer: producer.into(),
                family: family.into(),
                source_fingerprint: job.source_fingerprint.clone(),
                tokenizer_fingerprint: job.tokenizer_fingerprint.clone(),
                sample_fingerprint: job.samples.fingerprint().to_string(),
                corpus_fingerprint: job.corpus_fingerprint.clone(),
                hidden_width,
                total_layers,
                rows,
            },
            layers: Vec::with_capacity(total_layers),
        })
    }

    pub fn row_count(&self) -> usize {
        self.metadata.rows.len()
    }

    pub fn push_layer(&mut self, layer: usize, values: Vec<f32>) -> Result<(), CalibError> {
        if layer != self.layers.len() || layer >= self.metadata.total_layers {
            return Err(CalibError::InvalidOptions(format!(
                "residual probe layer {layer} is not the next logical layer {}",
                self.layers.len()
            )));
        }
        let expected = self
            .row_count()
            .checked_mul(self.metadata.hidden_width)
            .ok_or_else(|| CalibError::InvalidOptions("residual probe shape overflow".into()))?;
        if values.len() != expected {
            return Err(CalibError::InvalidOptions(format!(
                "residual probe layer {layer} has {} values, expected {expected}",
                values.len()
            )));
        }
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(CalibError::Runtime(format!(
                "residual probe layer {layer} contains a non-finite value at index {index}"
            )));
        }
        self.layers.push(values);
        Ok(())
    }

    pub fn write(&self, path: &Path) -> Result<(), CalibError> {
        if self.layers.len() != self.metadata.total_layers {
            return Err(CalibError::InvalidOptions(format!(
                "residual probe has {} layers, expected {}",
                self.layers.len(),
                self.metadata.total_layers
            )));
        }
        let rows = u32::try_from(self.row_count()).map_err(|_| {
            CalibError::InvalidOptions("residual probe row count exceeds u32".into())
        })?;
        let width = u32::try_from(self.metadata.hidden_width)
            .map_err(|_| CalibError::InvalidOptions("residual probe width exceeds u32".into()))?;
        let tensors = self
            .layers
            .iter()
            .enumerate()
            .map(|(layer, values)| HfqMemTensor {
                name: format!("layer.{layer}.post_residual"),
                quant_type: F32_QUANT_TYPE,
                shape: vec![rows, width],
                group_size: 0,
                data: values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            })
            .collect::<Vec<_>>();
        let metadata = serde_json::to_string(&self.metadata)
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        write_hfqm_package_mem(path, self.arch_id, &metadata, &tensors)
            .map_err(|error| CalibError::Runtime(error.to_string()))
    }

    pub fn read(path: &Path) -> Result<Self, CalibError> {
        let package = HfqFile::open_index_only(path)
            .map_err(|error| CalibError::Runtime(format!("open {}: {error}", path.display())))?;
        let metadata: ResidualProbeMetadata = serde_json::from_str(&package.metadata_json)
            .map_err(|error| CalibError::Runtime(format!("parse {}: {error}", path.display())))?;
        if metadata.schema_version != RESIDUAL_PROBE_SCHEMA_VERSION
            || metadata.artifact_kind != "calibration-residual-probe"
        {
            return Err(CalibError::InvalidOptions(format!(
                "{} is not a supported calibration residual probe",
                path.display()
            )));
        }
        let expected_shape = vec![
            u32::try_from(metadata.rows.len()).map_err(|_| {
                CalibError::InvalidOptions("residual probe row count exceeds u32".into())
            })?,
            u32::try_from(metadata.hidden_width).map_err(|_| {
                CalibError::InvalidOptions("residual probe width exceeds u32".into())
            })?,
        ];
        let mut layers = Vec::with_capacity(metadata.total_layers);
        for layer in 0..metadata.total_layers {
            let name = format!("layer.{layer}.post_residual");
            let (info, bytes) = package.tensor_data_vec(&name).ok_or_else(|| {
                CalibError::InvalidOptions(format!("residual probe is missing {name}"))
            })?;
            if info.quant_type != F32_QUANT_TYPE || info.shape != expected_shape {
                return Err(CalibError::InvalidOptions(format!(
                    "residual probe tensor {name} has quant type {} shape {:?}, expected F32 {:?}",
                    info.quant_type, info.shape, expected_shape
                )));
            }
            let expected_bytes = metadata
                .rows
                .len()
                .checked_mul(metadata.hidden_width)
                .and_then(|values| values.checked_mul(4))
                .ok_or_else(|| CalibError::InvalidOptions("residual probe size overflow".into()))?;
            if bytes.len() != expected_bytes {
                return Err(CalibError::InvalidOptions(format!(
                    "residual probe tensor {name} has {} bytes, expected {expected_bytes}",
                    bytes.len()
                )));
            }
            let values = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            if let Some(index) = values.iter().position(|value| !value.is_finite()) {
                return Err(CalibError::Runtime(format!(
                    "residual probe tensor {name} contains a non-finite value at index {index}"
                )));
            }
            layers.push(values);
        }
        Ok(Self {
            arch_id: package.arch_id,
            metadata,
            layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::contracts::{
        BoundaryPrecision, CalibrationOptions, CalibrationSample, ExpertCaptureQuota,
        ExpertCoveragePolicy, ExpertSamplingPolicy, SampleSet,
    };

    fn job() -> CalibrationJob {
        let samples = SampleSet::new(
            vec![
                CalibrationSample {
                    id: "a".into(),
                    stratum: "test".into(),
                    tokens: vec![1, 2],
                },
                CalibrationSample {
                    id: "b".into(),
                    stratum: "test".into(),
                    tokens: vec![3, 4],
                },
            ],
            2,
            7,
        )
        .unwrap();
        CalibrationJob::new(
            "source",
            "tokenizer",
            samples,
            CalibrationOptions {
                sequence_batch: Some(2),
                time_tile: Some(2),
                max_rows: 4,
                boundary_precision: BoundaryPrecision::F32,
                expert_quota: ExpertCaptureQuota {
                    min_rows: 1,
                    target_rows: 1,
                    tile_rows: 1,
                    sampling: ExpertSamplingPolicy::DeterministicFirst { seed: 7 },
                },
                required_expert_fraction: 1.0,
                expert_coverage_policy: ExpertCoveragePolicy::Strict,
                kldref: false,
                kldref_top_k: 1,
                kldref_rows: None,
            },
        )
        .unwrap()
    }

    #[test]
    fn residual_probe_round_trips_sample_major_rows() {
        let mut probe = ResidualProbe::new(6, "qwen3.5", "resident", &job(), 2, 2, 3).unwrap();
        assert_eq!(probe.row_count(), 3);
        for layer in 0..2 {
            probe
                .push_layer(layer, vec![layer as f32; probe.row_count() * 2])
                .unwrap();
        }
        let path = std::env::temp_dir().join(format!(
            "hipfire-residual-probe-{}-{}.hfq",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        probe.write(&path).unwrap();
        let loaded = ResidualProbe::read(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(loaded, probe);
    }
}
