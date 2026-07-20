// SPDX-License-Identifier: Apache-2.0
// hipfire — deterministic sequence/time calibration microbatches.

use super::contracts::{CalibError, SampleRow, SampleSet};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MicrobatchGeometry {
    pub sequence_batch: usize,
    pub time_tile: usize,
    pub row_budget: usize,
}

impl MicrobatchGeometry {
    pub fn validate(self) -> Result<Self, CalibError> {
        if self.sequence_batch == 0 || self.time_tile == 0 || self.row_budget == 0 {
            return Err(CalibError::InvalidOptions(
                "sequence batch, time tile, and row budget must be nonzero".into(),
            ));
        }
        let rectangular_rows = self
            .sequence_batch
            .checked_mul(self.time_tile)
            .ok_or_else(|| CalibError::InvalidOptions("microbatch row count overflow".into()))?;
        if rectangular_rows > self.row_budget {
            return Err(CalibError::InvalidOptions(format!(
                "sequence_batch*time_tile is {rectangular_rows}, above row budget {}",
                self.row_budget
            )));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerMicrobatch {
    pub sequence_start: usize,
    pub sequence_end: usize,
    pub time_start: usize,
    pub time_end: usize,
    pub rows: Vec<SampleRow>,
    /// Sample-major rows in the persistent boundary store, in `rows` order.
    pub boundary_rows: Vec<usize>,
}

pub struct MicrobatchPlanner {
    geometry: MicrobatchGeometry,
}

impl MicrobatchPlanner {
    pub fn new(geometry: MicrobatchGeometry) -> Result<Self, CalibError> {
        Ok(Self {
            geometry: geometry.validate()?,
        })
    }

    pub const fn geometry(&self) -> MicrobatchGeometry {
        self.geometry
    }

    /// Stable sequence-group/time-tile traversal. Ragged tails omit absent
    /// positions; no sample is concatenated with another sample.
    pub fn plan(&self, samples: &SampleSet) -> Vec<LayerMicrobatch> {
        let mut batches = Vec::new();
        let mut sample_offsets = Vec::with_capacity(samples.samples().len());
        let mut offset = 0usize;
        for sample in samples.samples() {
            sample_offsets.push(offset);
            offset += sample.tokens.len();
        }
        for sequence_start in (0..samples.samples().len()).step_by(self.geometry.sequence_batch) {
            let sequence_end =
                (sequence_start + self.geometry.sequence_batch).min(samples.samples().len());
            let max_len = samples.samples()[sequence_start..sequence_end]
                .iter()
                .map(|sample| sample.tokens.len())
                .max()
                .unwrap_or(0);
            for time_start in (0..max_len).step_by(self.geometry.time_tile) {
                let time_end = (time_start + self.geometry.time_tile).min(max_len);
                let mut rows =
                    Vec::with_capacity((sequence_end - sequence_start) * (time_end - time_start));
                let mut boundary_rows = Vec::with_capacity(rows.capacity());
                for position in time_start..time_end {
                    for (sample_index, sample) in samples.samples()[sequence_start..sequence_end]
                        .iter()
                        .enumerate()
                    {
                        let sample_index = sequence_start + sample_index;
                        if let Some(&token) = sample.tokens.get(position) {
                            rows.push(SampleRow {
                                sample_index,
                                position,
                                token,
                                reset_state: position == 0,
                            });
                            boundary_rows.push(sample_offsets[sample_index] + position);
                        }
                    }
                }
                if !rows.is_empty() {
                    debug_assert!(rows.len() <= self.geometry.row_budget);
                    batches.push(LayerMicrobatch {
                        sequence_start,
                        sequence_end,
                        time_start,
                        time_end,
                        rows,
                        boundary_rows,
                    });
                }
            }
        }
        batches
    }

    /// Largest rectangular geometry from the supplied deterministic candidates
    /// that fits the row budget. Allocation probing can further reduce it
    /// without changing sample order.
    pub fn choose_largest(
        row_budget: usize,
        sequence_candidates: &[usize],
        time_candidates: &[usize],
    ) -> Result<MicrobatchGeometry, CalibError> {
        let mut candidates = Vec::new();
        for &sequence_batch in sequence_candidates {
            for &time_tile in time_candidates {
                let Some(rows) = sequence_batch.checked_mul(time_tile) else {
                    continue;
                };
                if rows <= row_budget && rows > 0 {
                    candidates.push((rows, sequence_batch, time_tile));
                }
            }
        }
        candidates.sort();
        let (_, sequence_batch, time_tile) = candidates.pop().ok_or_else(|| {
            CalibError::InvalidOptions("no microbatch candidate fits the row budget".into())
        })?;
        MicrobatchGeometry {
            sequence_batch,
            time_tile,
            row_budget,
        }
        .validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationMemoryEstimate {
    pub boundary_bytes: u64,
    pub model_state_bytes: u64,
    pub gpu_scratch_bytes: u64,
    pub resident_weight_bytes: u64,
    pub peak_bytes: u64,
}

impl CalibrationMemoryEstimate {
    pub fn checked(
        rows: usize,
        width: usize,
        model_state_bytes: u64,
        gpu_scratch_bytes: u64,
        resident_weight_bytes: u64,
    ) -> Result<Self, CalibError> {
        let boundary_bytes = (rows as u64)
            .checked_mul(width as u64)
            .and_then(|values| values.checked_mul(4))
            .and_then(|one| one.checked_mul(2))
            .ok_or_else(|| CalibError::InvalidOptions("memory estimate overflow".into()))?;
        let peak_bytes = boundary_bytes
            .checked_add(model_state_bytes)
            .and_then(|bytes| bytes.checked_add(gpu_scratch_bytes))
            .and_then(|bytes| bytes.checked_add(resident_weight_bytes))
            .ok_or_else(|| CalibError::InvalidOptions("memory estimate overflow".into()))?;
        Ok(Self {
            boundary_bytes,
            model_state_bytes,
            gpu_scratch_bytes,
            resident_weight_bytes,
            peak_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::boundary::{BoundaryBackend, BoundaryStore};
    use crate::calibration::contracts::CalibrationSample;
    use std::collections::BTreeMap;

    fn samples() -> SampleSet {
        SampleSet::new(
            vec![
                CalibrationSample::new("a", vec![1, 2, 3, 4, 5], "x"),
                CalibrationSample::new("b", vec![6, 7], "x"),
                CalibrationSample::new("c", vec![8, 9, 10], "x"),
            ],
            8,
            7,
        )
        .unwrap()
    }

    #[test]
    fn ragged_batches_preserve_each_samples_positions_and_resets() {
        let samples = samples();
        let planner = MicrobatchPlanner::new(MicrobatchGeometry {
            sequence_batch: 2,
            time_tile: 2,
            row_budget: 4,
        })
        .unwrap();
        let batches = planner.plan(&samples);
        assert!(batches.iter().all(|batch| batch.rows.len() <= 4));
        let mut positions: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut boundary_rows = Vec::new();
        let mut reset_count = 0;
        for row in batches.iter().flat_map(|batch| &batch.rows) {
            positions
                .entry(row.sample_index)
                .or_default()
                .push(row.position);
            reset_count += usize::from(row.reset_state);
        }
        for batch in &batches {
            assert_eq!(batch.rows.len(), batch.boundary_rows.len());
            boundary_rows.extend_from_slice(&batch.boundary_rows);
        }
        for (sample_index, positions) in positions {
            assert_eq!(
                positions,
                (0..samples.samples()[sample_index].tokens.len()).collect::<Vec<_>>()
            );
        }
        assert_eq!(reset_count, samples.samples().len());
        assert_eq!(
            batches.iter().map(|batch| batch.rows.len()).sum::<usize>(),
            samples.total_rows()
        );
        boundary_rows.sort_unstable();
        assert_eq!(boundary_rows, (0..samples.total_rows()).collect::<Vec<_>>());
    }

    #[test]
    fn geometry_choice_is_deterministic_and_budgeted() {
        let geometry = MicrobatchPlanner::choose_largest(128, &[1, 2, 4, 8], &[8, 16, 32]).unwrap();
        assert_eq!(geometry.sequence_batch * geometry.time_tile, 128);
        assert_eq!(
            geometry,
            MicrobatchPlanner::choose_largest(128, &[1, 2, 4, 8], &[8, 16, 32]).unwrap()
        );
        assert!(MicrobatchPlanner::choose_largest(0, &[1], &[1]).is_err());
    }

    #[test]
    fn memory_estimate_matches_tiny_boundary_allocation() {
        let store = BoundaryStore::create(BoundaryBackend::Ram, 7, 5, 1, "fp", "engine").unwrap();
        let estimate = CalibrationMemoryEstimate::checked(7, 5, 11, 13, 17).unwrap();
        assert_eq!(estimate.boundary_bytes as usize, store.allocated_bytes());
        assert_eq!(estimate.peak_bytes, estimate.boundary_bytes + 11 + 13 + 17);
    }
}
