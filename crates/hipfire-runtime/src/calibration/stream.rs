// SPDX-License-Identifier: Apache-2.0
// hipfire — family-neutral layer-stream adapter contracts.

use super::contracts::{
    CalibError, CalibrationJob, CaptureRegistry, ExpertLayerTelemetry, KldRefRow, SampleRow,
};
use super::schedule::{LayerMicrobatch, MicrobatchGeometry};
use super::source::{PlannedTensorReader, TensorLoadRequest};
use crate::weights::WeightTensor;
use hipfire_model::ModelSource;
use hipfire_rdna::{DType, Gpu, GpuTensor};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CalibrationResourceEstimate {
    pub scratch_bytes: u64,
    pub state_bytes_per_sequence: u64,
    pub active_state_bytes: u64,
    pub details: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct LayerCapturePartSummary {
    pub descriptors: Vec<super::CalibTensorDesc>,
    pub max_consistency: f32,
    pub expert_telemetry: Option<ExpertLayerTelemetry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInspection {
    pub family: String,
    pub arch_id: u32,
    pub hidden_width: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub tensor_requests: Vec<TensorLoadRequest>,
}

impl ModelInspection {
    pub fn validate(&self) -> Result<(), CalibError> {
        if self.family.is_empty() {
            return Err(CalibError::InvalidSourcePlan(
                "calibration family name must not be empty".into(),
            ));
        }
        if self.hidden_width == 0 || self.vocab_size == 0 || self.num_layers == 0 {
            return Err(CalibError::InvalidSourcePlan(
                "hidden width, vocabulary size, and layer count must be nonzero".into(),
            ));
        }
        if self.tensor_requests.is_empty() {
            return Err(CalibError::InvalidSourcePlan(
                "family adapter produced an empty tensor plan".into(),
            ));
        }
        Ok(())
    }
}

/// Embeds scheduler rows into F32 residual rows in the same order.
pub trait CalibrationEmbedding {
    fn execute(
        &mut self,
        gpu: &mut Gpu,
        rows: &[SampleRow],
        output_f32: &mut [f32],
    ) -> Result<(), CalibError>;

    fn finish(&mut self, _gpu: &mut Gpu) -> Result<(), CalibError> {
        Ok(())
    }
}

/// One resident transformer layer. Adapter state must keep independent
/// sequence state across time tiles and reset only rows marked `reset_state`.
pub trait CalibrationLayer {
    fn execute(
        &mut self,
        gpu: &mut Gpu,
        batch: &LayerMicrobatch,
        input_f32: &[f32],
        output_f32: &mut [f32],
        capture: &CaptureRegistry,
    ) -> Result<(), CalibError>;

    fn write_capture_part(
        &mut self,
        gpu: &mut Gpu,
        path: &Path,
        arch_id: u32,
        metadata_json: &str,
    ) -> Result<LayerCapturePartSummary, CalibError>;

    fn finish(&mut self, _gpu: &mut Gpu) -> Result<(), CalibError> {
        Ok(())
    }
}

/// Final norm + language head. Returns already-reduced top-k/log-normalizer
/// rows, so the generic engine never downloads full vocabulary logits.
pub trait CalibrationFinalizer {
    fn execute_kld(
        &mut self,
        gpu: &mut Gpu,
        batch: &LayerMicrobatch,
        residual_f32: &[f32],
        top_k: usize,
        output: &mut Vec<KldRefRow>,
    ) -> Result<(), CalibError>;

    fn finish(&mut self, _gpu: &mut Gpu) -> Result<(), CalibError> {
        Ok(())
    }
}

const KLD_VOCAB_TILE: usize = 2048;
const KLD_REDUCER_TOP_K: usize = 256;

/// Family-neutral final norm + raw lm-head reducer used by streamed adapters.
/// It keeps logits vocabulary-tiled on GPU and returns only top-k/logZ rows.
pub struct RmsNormLmHeadFinalizer {
    norm: Option<GpuTensor>,
    lm_head: Option<WeightTensor>,
    residual: Option<GpuTensor>,
    normed: Option<GpuTensor>,
    logits_tile: Option<GpuTensor>,
    top_values: Option<GpuTensor>,
    top_indices: Option<GpuTensor>,
    chunk_max: Option<GpuTensor>,
    chunk_sum: Option<GpuTensor>,
    dim: usize,
    vocab_size: usize,
    norm_eps: f32,
    max_rows: usize,
}

impl RmsNormLmHeadFinalizer {
    pub fn new(
        gpu: &mut Gpu,
        norm: GpuTensor,
        lm_head: WeightTensor,
        dim: usize,
        vocab_size: usize,
        norm_eps: f32,
        max_rows: usize,
    ) -> Result<Self, CalibError> {
        if max_rows == 0 || dim == 0 || vocab_size == 0 {
            let _ = gpu.free_tensor(norm);
            let _ = gpu.free_tensor(lm_head.buf);
            return Err(CalibError::InvalidOptions(
                "KLD finalizer geometry must be nonzero".into(),
            ));
        }
        let mut finalizer = Self {
            norm: Some(norm),
            lm_head: Some(lm_head),
            residual: None,
            normed: None,
            logits_tile: None,
            top_values: None,
            top_indices: None,
            chunk_max: None,
            chunk_sum: None,
            dim,
            vocab_size,
            norm_eps,
            max_rows,
        };
        let result = (|| {
            finalizer.residual = Some(
                gpu.alloc_tensor(&[max_rows * dim], DType::F32)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            );
            finalizer.normed = Some(
                gpu.alloc_tensor(&[max_rows * dim], DType::F32)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            );
            finalizer.logits_tile = Some(
                gpu.alloc_tensor(&[max_rows * KLD_VOCAB_TILE], DType::F32)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            );
            finalizer.top_values = Some(
                gpu.alloc_tensor(&[max_rows * KLD_REDUCER_TOP_K], DType::F32)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            );
            finalizer.top_indices = Some(
                gpu.alloc_tensor(&[max_rows * KLD_REDUCER_TOP_K], DType::F32)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            );
            finalizer.chunk_max = Some(
                gpu.alloc_tensor(&[max_rows], DType::F32)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            );
            finalizer.chunk_sum = Some(
                gpu.alloc_tensor(&[max_rows], DType::F32)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            );
            Ok::<(), CalibError>(())
        })();
        if let Err(error) = result {
            let _ = finalizer.finish(gpu);
            return Err(error);
        }
        Ok(finalizer)
    }
}

impl CalibrationFinalizer for RmsNormLmHeadFinalizer {
    fn execute_kld(
        &mut self,
        gpu: &mut Gpu,
        batch: &LayerMicrobatch,
        residual_f32: &[f32],
        top_k: usize,
        output: &mut Vec<KldRefRow>,
    ) -> Result<(), CalibError> {
        let rows = batch.rows.len();
        if rows == 0 || rows > self.max_rows || residual_f32.len() != rows * self.dim {
            return Err(CalibError::InvalidOptions(
                "finalizer residual shape does not match its scheduler batch".into(),
            ));
        }
        if top_k == 0 || top_k > KLD_REDUCER_TOP_K || top_k > self.vocab_size {
            return Err(CalibError::InvalidOptions(format!(
                "KLD top-k must be in 1..={}, got {top_k}",
                KLD_REDUCER_TOP_K.min(self.vocab_size)
            )));
        }
        gpu.hip
            .memcpy_htod(
                &self.residual.as_ref().unwrap().buf,
                f32_slice_as_bytes(residual_f32),
            )
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        let residual = self
            .residual
            .as_ref()
            .unwrap()
            .sub_offset(0, rows * self.dim);
        let normed = self.normed.as_ref().unwrap().sub_offset(0, rows * self.dim);
        gpu.rmsnorm_batched(
            &residual,
            self.norm.as_ref().unwrap(),
            &normed,
            rows,
            self.dim,
            self.norm_eps,
        )
        .map_err(|error| CalibError::Runtime(error.to_string()))?;

        let mut candidates = vec![Vec::<(u32, f32)>::new(); rows];
        let mut log_max = vec![f32::NEG_INFINITY; rows];
        let mut scaled_sum = vec![0.0f64; rows];
        let lm_head = self.lm_head.as_ref().unwrap();
        for global_start in (0..self.vocab_size).step_by(KLD_VOCAB_TILE) {
            let tile_rows = (self.vocab_size - global_start).min(KLD_VOCAB_TILE);
            let weight = lm_head
                .buf
                .sub_offset(global_start * self.dim, tile_rows * self.dim);
            let logits = self
                .logits_tile
                .as_ref()
                .unwrap()
                .sub_offset(0, rows * tile_rows);
            match lm_head.gpu_dtype {
                DType::F32 => gpu
                    .gemm_f32_register_tiled(&weight, &normed, &logits, tile_rows, self.dim, rows),
                DType::F16 => {
                    gpu.gemm_f16_x_f32_wmma(&weight, &normed, &logits, tile_rows, self.dim, rows)
                }
                DType::BF16 => {
                    gpu.gemm_bf16_x_bf16_wmma(&weight, &normed, &logits, tile_rows, self.dim, rows)
                }
                dtype => {
                    return Err(CalibError::InvalidSourcePlan(format!(
                        "KLD finalizer does not support lm_head dtype {dtype:?}"
                    )))
                }
            }
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
            gpu.kld_tile_topk_lse_f32(
                &logits,
                self.top_values.as_ref().unwrap(),
                self.top_indices.as_ref().unwrap(),
                self.chunk_max.as_ref().unwrap(),
                self.chunk_sum.as_ref().unwrap(),
                rows,
                tile_rows,
                global_start,
                1,
            )
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
            let values = gpu
                .download_f32(
                    &self
                        .top_values
                        .as_ref()
                        .unwrap()
                        .sub_offset(0, rows * KLD_REDUCER_TOP_K),
                )
                .map_err(|error| CalibError::Runtime(error.to_string()))?;
            let indices = download_i32_prefix(
                gpu,
                self.top_indices.as_ref().unwrap(),
                rows * KLD_REDUCER_TOP_K,
            )?;
            let maxima = gpu
                .download_f32(&self.chunk_max.as_ref().unwrap().sub_offset(0, rows))
                .map_err(|error| CalibError::Runtime(error.to_string()))?;
            let sums = gpu
                .download_f32(&self.chunk_sum.as_ref().unwrap().sub_offset(0, rows))
                .map_err(|error| CalibError::Runtime(error.to_string()))?;
            for row in 0..rows {
                merge_logsumexp(
                    &mut log_max[row],
                    &mut scaled_sum[row],
                    maxima[row],
                    sums[row],
                );
                let offset = row * KLD_REDUCER_TOP_K;
                for rank in 0..KLD_REDUCER_TOP_K {
                    let index = indices[offset + rank];
                    let value = values[offset + rank];
                    if index >= 0 && (index as usize) < self.vocab_size && value.is_finite() {
                        candidates[row].push((index as u32, value));
                    }
                }
            }
        }
        output.reserve(rows);
        for (row_index, row) in batch.rows.iter().enumerate() {
            candidates[row_index].sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            candidates[row_index].truncate(top_k);
            if candidates[row_index].len() != top_k || scaled_sum[row_index] <= 0.0 {
                return Err(CalibError::Runtime(
                    "KLD reducer produced incomplete candidates or a non-positive sum".into(),
                ));
            }
            output.push(KldRefRow {
                sample_index: row.sample_index,
                position: row.position,
                indices: candidates[row_index]
                    .iter()
                    .map(|(index, _)| *index)
                    .collect(),
                logits: candidates[row_index]
                    .iter()
                    .map(|(_, value)| *value)
                    .collect(),
                log_z: log_max[row_index] + scaled_sum[row_index].ln() as f32,
            });
        }
        Ok(())
    }

    fn finish(&mut self, gpu: &mut Gpu) -> Result<(), CalibError> {
        for tensor in [
            self.chunk_sum.take(),
            self.chunk_max.take(),
            self.top_indices.take(),
            self.top_values.take(),
            self.logits_tile.take(),
            self.normed.take(),
            self.residual.take(),
            self.norm.take(),
        ]
        .into_iter()
        .flatten()
        {
            gpu.free_tensor(tensor)
                .map_err(|error| CalibError::Runtime(error.to_string()))?;
        }
        if let Some(weight) = self.lm_head.take() {
            gpu.free_tensor(weight.buf)
                .map_err(|error| CalibError::Runtime(error.to_string()))?;
        }
        Ok(())
    }
}

fn f32_slice_as_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
}

fn download_i32_prefix(gpu: &Gpu, tensor: &GpuTensor, len: usize) -> Result<Vec<i32>, CalibError> {
    let mut bytes = vec![0u8; len * 4];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &tensor.buf)
        .map_err(|error| CalibError::Runtime(error.to_string()))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn merge_logsumexp(global_max: &mut f32, global_scaled_sum: &mut f64, max: f32, sum: f32) {
    let merged_max = (*global_max).max(max);
    *global_scaled_sum = *global_scaled_sum * ((*global_max - merged_max) as f64).exp()
        + sum as f64 * ((max - merged_max) as f64).exp();
    *global_max = merged_max;
}

/// Family math and tensor naming live behind this seam. Corpus order,
/// microbatching, boundary storage, read accounting, checkpoints, and artifact
/// completion remain generic engine responsibilities.
pub trait CalibrationFamilyAdapter {
    fn family(&self) -> &'static str;

    fn adapter_version(&self) -> &'static str;

    fn resource_estimate(
        &self,
        _model: &ModelInspection,
        _job: &CalibrationJob,
        _geometry: MicrobatchGeometry,
    ) -> Result<Option<CalibrationResourceEstimate>, CalibError> {
        Ok(None)
    }

    fn effective_precision(&self, gpu: &Gpu) -> serde_json::Value;

    fn inspect(&mut self, source: &dyn ModelSource) -> Result<ModelInspection, CalibError>;

    fn capture_plan(
        &self,
        model: &ModelInspection,
        job: &CalibrationJob,
    ) -> Result<CaptureRegistry, CalibError>;

    fn load_embedding(
        &mut self,
        reader: &mut PlannedTensorReader<'_, '_, '_>,
        gpu: &mut Gpu,
        model: &ModelInspection,
        job: &CalibrationJob,
    ) -> Result<Box<dyn CalibrationEmbedding>, CalibError>;

    fn load_layer(
        &mut self,
        reader: &mut PlannedTensorReader<'_, '_, '_>,
        gpu: &mut Gpu,
        model: &ModelInspection,
        layer: usize,
        job: &CalibrationJob,
    ) -> Result<Box<dyn CalibrationLayer>, CalibError>;

    fn load_finalizer(
        &mut self,
        reader: &mut PlannedTensorReader<'_, '_, '_>,
        gpu: &mut Gpu,
        model: &ModelInspection,
        job: &CalibrationJob,
    ) -> Result<Box<dyn CalibrationFinalizer>, CalibError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_inspection_rejects_zero_geometry_and_empty_plan() {
        let empty = ModelInspection {
            family: "qwen3.5".into(),
            arch_id: 6,
            hidden_width: 0,
            vocab_size: 1,
            num_layers: 1,
            tensor_requests: Vec::new(),
        };
        assert!(empty.validate().is_err());
    }

    #[test]
    fn tiled_kld_host_merge_is_exact_and_tie_stable() {
        let tiles = [vec![1.0f32, 3.0, 2.0], vec![3.0f32, -2.0, 0.5]];
        let mut global_max = f32::NEG_INFINITY;
        let mut global_sum = 0.0f64;
        let mut candidates = Vec::new();
        let mut offset = 0u32;
        for tile in &tiles {
            let tile_max = tile.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let tile_sum = tile.iter().map(|value| (*value - tile_max).exp()).sum();
            merge_logsumexp(&mut global_max, &mut global_sum, tile_max, tile_sum);
            candidates.extend(
                tile.iter()
                    .copied()
                    .enumerate()
                    .map(|(index, value)| (offset + index as u32, value)),
            );
            offset += tile.len() as u32;
        }
        candidates.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        candidates.truncate(3);
        assert_eq!(candidates, vec![(1, 3.0), (3, 3.0), (2, 2.0)]);
        let all = tiles.concat();
        let cpu_max = all.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let cpu_logz = cpu_max
            + all
                .iter()
                .map(|value| (*value - cpu_max).exp())
                .sum::<f32>()
                .ln();
        let merged_logz = global_max + global_sum.ln() as f32;
        assert!((merged_logz - cpu_logz).abs() < 1.0e-6);
    }
}
