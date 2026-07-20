// SPDX-License-Identifier: Apache-2.0
// hipfire — family-neutral grouped-MoE routing and shape contracts.

//! CPU-side contracts shared by arch adapters and the calibration engine.
//!
//! Kernel wrappers remain in `hipfire-rdna`; this module owns the routing
//! permutation, tile padding, inverse mapping, and source-row geometry that do
//! not depend on Qwen or any other model family.

use serde::{Deserialize, Serialize};
use std::fmt;

use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

pub const GROUPED_MOE_BLOCK_ROWS: usize = 16;

/// Owned scatter/grouped-GEMM scratch shared by model-family adapters.
pub struct GroupedMoeScratch {
    pub max_rows: usize,
    pub k_top: usize,
    pub num_experts: usize,
    pub m_total_max: usize,
    pub expert_token_counts: GpuTensor,
    pub expert_offsets: GpuTensor,
    pub sorted_slot_index: GpuTensor,
    pub inverse_perm: GpuTensor,
    pub expert_tile_ids: GpuTensor,
    pub y_gate_up_grouped: GpuTensor,
    pub y_down_grouped: GpuTensor,
}

impl GroupedMoeScratch {
    pub fn new(
        gpu: &mut Gpu,
        max_rows: usize,
        k_top: usize,
        num_experts: usize,
        gate_up_output_width: usize,
        down_output_width: usize,
    ) -> HipResult<Self> {
        let m_total_max = grouped_m_total_max(max_rows, k_top, num_experts)
            .map_err(|error| hipfire_rdna::HipError::new(0, &error.to_string()))?;
        let total_slots_max = max_rows.checked_mul(k_top).ok_or_else(|| {
            hipfire_rdna::HipError::new(0, "grouped-MoE scratch routed-slot overflow")
        })?;
        Ok(Self {
            max_rows,
            k_top,
            num_experts,
            m_total_max,
            // i32 kernel buffers are byte-addressed Raw tensors.
            expert_token_counts: gpu.alloc_tensor(&[num_experts * 4], DType::Raw)?,
            expert_offsets: gpu.alloc_tensor(&[(num_experts + 1) * 4], DType::Raw)?,
            sorted_slot_index: gpu.alloc_tensor(&[m_total_max * 4], DType::Raw)?,
            inverse_perm: gpu.alloc_tensor(&[total_slots_max * 4], DType::Raw)?,
            expert_tile_ids: gpu
                .alloc_tensor(&[(m_total_max / GROUPED_MOE_BLOCK_ROWS) * 4], DType::Raw)?,
            y_gate_up_grouped: gpu
                .alloc_tensor(&[m_total_max * gate_up_output_width], DType::F32)?,
            y_down_grouped: gpu.alloc_tensor(&[m_total_max * down_output_width], DType::F32)?,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for tensor in [
            self.expert_token_counts,
            self.expert_offsets,
            self.sorted_slot_index,
            self.inverse_perm,
            self.expert_tile_ids,
            self.y_gate_up_grouped,
            self.y_down_grouped,
        ] {
            let _ = gpu.free_tensor(tensor);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupedMoeError {
    InvalidShape(String),
    InvalidRouting(String),
    Overflow(String),
}

impl fmt::Display for GroupedMoeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(message) => write!(f, "invalid grouped-MoE shape: {message}"),
            Self::InvalidRouting(message) => write!(f, "invalid grouped-MoE routing: {message}"),
            Self::Overflow(message) => write!(f, "grouped-MoE size overflow: {message}"),
        }
    }
}

impl std::error::Error for GroupedMoeError {}

#[inline]
pub fn align_up(value: usize, alignment: usize) -> usize {
    assert!(
        alignment.is_power_of_two(),
        "alignment must be a power of two"
    );
    (value + alignment - 1) & !(alignment - 1)
}

#[inline]
pub fn grouped_m_total_max(
    max_batch: usize,
    k_top: usize,
    num_experts: usize,
) -> Result<usize, GroupedMoeError> {
    validate_dimensions(max_batch, k_top, num_experts)?;
    let routed_slots = max_batch
        .checked_mul(k_top)
        .ok_or_else(|| GroupedMoeError::Overflow("max_batch * K-top".into()))?;
    let padding = num_experts
        .checked_mul(GROUPED_MOE_BLOCK_ROWS - 1)
        .ok_or_else(|| GroupedMoeError::Overflow("expert padding".into()))?;
    let upper = routed_slots
        .checked_add(padding)
        .ok_or_else(|| GroupedMoeError::Overflow("routed slots + padding".into()))?;
    Ok(align_up(upper, GROUPED_MOE_BLOCK_ROWS))
}

#[inline]
pub fn grouped_m_total_bound(
    total_slots: usize,
    num_experts: usize,
) -> Result<usize, GroupedMoeError> {
    if total_slots == 0 || num_experts == 0 {
        return Err(GroupedMoeError::InvalidShape(
            "total slots and expert count must be non-zero".into(),
        ));
    }
    let live_expert_bound = total_slots.min(num_experts);
    let padding = live_expert_bound
        .checked_mul(GROUPED_MOE_BLOCK_ROWS - 1)
        .ok_or_else(|| GroupedMoeError::Overflow("live-expert padding".into()))?;
    let upper = total_slots
        .checked_add(padding)
        .ok_or_else(|| GroupedMoeError::Overflow("routed slots + live padding".into()))?;
    Ok(align_up(upper, GROUPED_MOE_BLOCK_ROWS))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupedMoeShape {
    pub total_slots: usize,
    pub m_total_bound: usize,
    pub gate_up_x_row_div: usize,
    pub gate_up_source_rows: usize,
    pub down_x_row_div: usize,
    pub down_source_rows: usize,
}

pub fn grouped_moe_shape(
    rows: usize,
    k_top: usize,
    num_experts: usize,
) -> Result<GroupedMoeShape, GroupedMoeError> {
    validate_dimensions(rows, k_top, num_experts)?;
    let total_slots = rows
        .checked_mul(k_top)
        .ok_or_else(|| GroupedMoeError::Overflow("rows * K-top".into()))?;
    Ok(GroupedMoeShape {
        total_slots,
        m_total_bound: grouped_m_total_bound(total_slots, num_experts)?,
        // A sorted routed slot encodes token*K_TOP + rank. Gate/up reads the
        // original token row, while down reads the flattened routed row.
        gate_up_x_row_div: k_top,
        gate_up_source_rows: rows,
        down_x_row_div: 1,
        down_source_rows: total_slots,
    })
}

fn validate_dimensions(
    rows: usize,
    k_top: usize,
    num_experts: usize,
) -> Result<(), GroupedMoeError> {
    if rows == 0 || k_top == 0 || num_experts == 0 {
        return Err(GroupedMoeError::InvalidShape(
            "rows, K-top, and expert count must be non-zero".into(),
        ));
    }
    if k_top > num_experts {
        return Err(GroupedMoeError::InvalidShape(format!(
            "K-top {k_top} exceeds expert count {num_experts}"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupedMoeRoutingPlan {
    pub rows: usize,
    pub k_top: usize,
    pub num_experts: usize,
    pub total_slots: usize,
    pub m_total: usize,
    pub counts: Vec<usize>,
    /// Physical padded start for each expert plus the final `m_total` sentinel.
    pub offsets: Vec<usize>,
    /// Flat `token*K_TOP + rank` indices; `-1` denotes tile padding.
    pub sorted_slot_index: Vec<i32>,
    /// Flat routed slot to physical sorted position.
    pub inverse_perm: Vec<i32>,
    /// One expert id per `GROUPED_MOE_BLOCK_ROWS` tile.
    pub expert_tile_ids: Vec<i32>,
}

impl GroupedMoeRoutingPlan {
    pub fn build(
        topk_indices: &[usize],
        rows: usize,
        k_top: usize,
        num_experts: usize,
    ) -> Result<Self, GroupedMoeError> {
        validate_dimensions(rows, k_top, num_experts)?;
        let total_slots = rows
            .checked_mul(k_top)
            .ok_or_else(|| GroupedMoeError::Overflow("rows * K-top".into()))?;
        if topk_indices.len() != total_slots {
            return Err(GroupedMoeError::InvalidRouting(format!(
                "received {} route indices, expected {total_slots}",
                topk_indices.len()
            )));
        }
        if total_slots > i32::MAX as usize || num_experts > i32::MAX as usize {
            return Err(GroupedMoeError::Overflow(
                "routing indices exceed i32 kernel contract".into(),
            ));
        }

        let mut slots_by_expert = vec![Vec::<i32>::new(); num_experts];
        for (flat_slot, &expert) in topk_indices.iter().enumerate() {
            if expert >= num_experts {
                return Err(GroupedMoeError::InvalidRouting(format!(
                    "slot {flat_slot} selected expert {expert}, but expert count is {num_experts}"
                )));
            }
            slots_by_expert[expert].push(flat_slot as i32);
        }

        let counts = slots_by_expert.iter().map(Vec::len).collect::<Vec<_>>();
        let m_total = slots_by_expert.iter().try_fold(0usize, |total, slots| {
            total
                .checked_add(if slots.is_empty() {
                    0
                } else {
                    align_up(slots.len(), GROUPED_MOE_BLOCK_ROWS)
                })
                .ok_or_else(|| GroupedMoeError::Overflow("padded routed rows".into()))
        })?;
        let mut offsets = Vec::with_capacity(num_experts + 1);
        let mut sorted_slot_index = Vec::with_capacity(m_total);
        let mut inverse_perm = vec![-1i32; total_slots];
        let mut expert_tile_ids = Vec::with_capacity(m_total / GROUPED_MOE_BLOCK_ROWS);

        for (expert, slots) in slots_by_expert.iter().enumerate() {
            offsets.push(sorted_slot_index.len());
            if slots.is_empty() {
                continue;
            }
            let padded = align_up(slots.len(), GROUPED_MOE_BLOCK_ROWS);
            for &flat_slot in slots {
                inverse_perm[flat_slot as usize] = sorted_slot_index.len() as i32;
                sorted_slot_index.push(flat_slot);
            }
            sorted_slot_index.resize(sorted_slot_index.len() + padded - slots.len(), -1);
            expert_tile_ids.extend(std::iter::repeat_n(
                expert as i32,
                padded / GROUPED_MOE_BLOCK_ROWS,
            ));
        }
        offsets.push(sorted_slot_index.len());

        Ok(Self {
            rows,
            k_top,
            num_experts,
            total_slots,
            m_total,
            counts,
            offsets,
            sorted_slot_index,
            inverse_perm,
            expert_tile_ids,
        })
    }

    pub fn live_experts(&self) -> usize {
        self.counts.iter().filter(|&&count| count != 0).count()
    }

    pub fn real_slots_for_expert(&self, expert: usize) -> Option<&[i32]> {
        let start = *self.offsets.get(expert)?;
        let count = *self.counts.get(expert)?;
        Some(&self.sorted_slot_index[start..start + count])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PagedMoeExpertBucket {
    pub expert: u16,
    pub m_total: usize,
    pub sorted_slot_index: Vec<i32>,
    pub inverse_perm: Vec<i32>,
    pub expert_tile_ids: Vec<i32>,
}

pub fn build_paged_expert_buckets(
    topk_indices: &[usize],
    rows: usize,
    k_top: usize,
    num_experts: usize,
) -> Result<Vec<PagedMoeExpertBucket>, GroupedMoeError> {
    if num_experts > u16::MAX as usize {
        return Err(GroupedMoeError::Overflow(format!(
            "paged expert count {num_experts} exceeds u16 metadata"
        )));
    }
    let plan = GroupedMoeRoutingPlan::build(topk_indices, rows, k_top, num_experts)?;
    let mut buckets = Vec::with_capacity(plan.live_experts());
    for expert in 0..num_experts {
        let slots = plan.real_slots_for_expert(expert).unwrap();
        if slots.is_empty() {
            continue;
        }
        let m_total = align_up(slots.len(), GROUPED_MOE_BLOCK_ROWS);
        let mut sorted_slot_index = vec![-1; m_total];
        sorted_slot_index[..slots.len()].copy_from_slice(slots);
        let mut inverse_perm = vec![-1; plan.total_slots];
        for (sorted_position, &flat_slot) in slots.iter().enumerate() {
            inverse_perm[flat_slot as usize] = sorted_position as i32;
        }
        buckets.push(PagedMoeExpertBucket {
            expert: expert as u16,
            m_total,
            sorted_slot_index,
            inverse_perm,
            expert_tile_ids: vec![expert as i32; m_total / GROUPED_MOE_BLOCK_ROWS],
        });
    }
    Ok(buckets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_a3b_and_a17b_shapes_are_tile_aligned() {
        let k8 = grouped_moe_shape(256, 8, 256).unwrap();
        assert_eq!(k8.total_slots, 2048);
        assert_eq!(k8.m_total_bound, 5888);
        assert_eq!(k8.gate_up_x_row_div, 8);
        assert_eq!(k8.down_source_rows, 2048);
        assert_eq!(k8.m_total_bound % GROUPED_MOE_BLOCK_ROWS, 0);

        let k10 = grouped_moe_shape(256, 10, 512).unwrap();
        assert_eq!(k10.total_slots, 2560);
        assert_eq!(k10.gate_up_x_row_div, 10);
        assert_eq!(k10.down_source_rows, 2560);
        assert_eq!(k10.m_total_bound % GROUPED_MOE_BLOCK_ROWS, 0);
    }

    #[test]
    fn routing_plan_round_trips_slots_with_padding_and_empty_experts() {
        let routes = [2, 0, 2, 0, 2, 3, 0, 3];
        let plan = GroupedMoeRoutingPlan::build(&routes, 2, 4, 5).unwrap();
        assert_eq!(plan.counts, vec![3, 0, 3, 2, 0]);
        assert_eq!(plan.live_experts(), 3);
        assert_eq!(plan.m_total, 3 * GROUPED_MOE_BLOCK_ROWS);
        assert_eq!(plan.expert_tile_ids, vec![0, 2, 3]);
        for (flat_slot, &sorted_position) in plan.inverse_perm.iter().enumerate() {
            assert!(sorted_position >= 0);
            assert_eq!(
                plan.sorted_slot_index[sorted_position as usize],
                flat_slot as i32
            );
        }
        assert!(plan.real_slots_for_expert(1).unwrap().is_empty());
    }

    #[test]
    fn skewed_k10_routing_uses_one_tile_and_keeps_every_slot() {
        let routes = vec![7usize; 32 * 10];
        let plan = GroupedMoeRoutingPlan::build(&routes, 32, 10, 512).unwrap();
        assert_eq!(plan.live_experts(), 1);
        assert_eq!(plan.counts[7], 320);
        assert_eq!(plan.m_total, 320);
        assert!(plan.expert_tile_ids.iter().all(|&expert| expert == 7));
    }

    #[test]
    fn paged_buckets_match_full_plan_for_k10() {
        let routes = (0..40).map(|slot| slot % 5).collect::<Vec<_>>();
        let plan = GroupedMoeRoutingPlan::build(&routes, 4, 10, 16).unwrap();
        let buckets = build_paged_expert_buckets(&routes, 4, 10, 16).unwrap();
        assert_eq!(buckets.len(), 5);
        for bucket in buckets {
            assert_eq!(
                &bucket.sorted_slot_index[..plan.counts[bucket.expert as usize]],
                plan.real_slots_for_expert(bucket.expert as usize).unwrap()
            );
        }
    }

    #[test]
    fn routing_plan_rejects_bad_lengths_and_experts() {
        assert!(GroupedMoeRoutingPlan::build(&[0, 1], 1, 3, 4).is_err());
        assert!(GroupedMoeRoutingPlan::build(&[0, 4, 1], 1, 3, 4).is_err());
        assert!(grouped_moe_shape(1, 5, 4).is_err());
    }
}
