// SPDX-License-Identifier: Apache-2.0
// hipfire — family-neutral routed-expert calibration admission.

//! Converts a grouped-MoE routing permutation into quota-capped calibration
//! staging work. Teacher execution owns the complete routing plan; this module
//! filters only the capture stream and carries partial reduction tiles across
//! model microbatches.

use super::contracts::{
    CalibError, CaptureId, CapturePolicy, CaptureRegistry, ExpertCaptureRole, ExpertTelemetry,
    ProjectionRole,
};
use super::CalibCollector;
use crate::moe::grouped::GroupedMoeRoutingPlan;
use hipfire_dispatch::families::moe::{
    MoePrefillCapture, MoePrefillCaptureBatch, MoePrefillCapturePoint,
};
use hipfire_dispatch::types::DispatchError;
use hipfire_rdna::{DType, Gpu, GpuTensor};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertBatchAdmission {
    pub expert: usize,
    pub seen_rows: usize,
    pub admitted_rows: usize,
    pub batch_slack_rows: usize,
    pub quota_skipped_rows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureStageAction {
    pub layer: usize,
    pub expert: usize,
    pub role: ExpertCaptureRole,
    pub sorted_start: usize,
    pub rows: usize,
    pub destination_row: usize,
    pub source_row_div: usize,
    pub tile_rows: usize,
    pub flush_full_tile: bool,
}

impl CaptureStageAction {
    pub fn source_rows(&self, routing: &GroupedMoeRoutingPlan) -> Result<Vec<usize>, CalibError> {
        let end = self
            .sorted_start
            .checked_add(self.rows)
            .ok_or_else(|| CalibError::InvalidCapture("capture action range overflow".into()))?;
        let slots = routing
            .sorted_slot_index
            .get(self.sorted_start..end)
            .ok_or_else(|| {
                CalibError::InvalidCapture(format!(
                    "capture action {}..{end} exceeds sorted routing length {}",
                    self.sorted_start,
                    routing.sorted_slot_index.len()
                ))
            })?;
        slots
            .iter()
            .map(|&slot| {
                if slot < 0 {
                    Err(CalibError::InvalidCapture(
                        "capture action includes a padded routed slot".into(),
                    ))
                } else {
                    Ok(slot as usize / self.source_row_div)
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedExpertCapturePlan {
    pub layer: usize,
    pub role: ExpertCaptureRole,
    pub seen_rows: usize,
    pub admitted_rows: usize,
    pub batch_slack_rows: usize,
    pub quota_skipped_rows: usize,
    pub admissions: Vec<ExpertBatchAdmission>,
    pub actions: Vec<CaptureStageAction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialCaptureTile {
    pub layer: usize,
    pub expert: usize,
    pub role: ExpertCaptureRole,
    pub rows: usize,
}

#[derive(Debug, Clone)]
pub struct ExpertCaptureStaging {
    num_layers: usize,
    num_experts: usize,
    tile_rows: usize,
    staged_rows: Vec<usize>,
}

#[derive(Debug, Clone)]
struct PendingCaptureBatch {
    routing: GroupedMoeRoutingPlan,
    route_weights: Vec<f32>,
}

#[derive(Debug, Clone)]
struct GroupedCaptureState {
    telemetry: ExpertTelemetry,
    staging: ExpertCaptureStaging,
    pending: BTreeMap<(i32, usize), PendingCaptureBatch>,
}

/// Runtime implementation of the dispatch crate's family-neutral routed MoE
/// capture callback. Router indices/weights are downloaded once at the gate/up
/// seam, then reused at the down seam; activation rows and reductions remain on
/// GPU. The callback never modifies the teacher's routing tensors.
pub struct GroupedMoeCalibrationCapture {
    collector: Arc<CalibCollector>,
    registry: Arc<CaptureRegistry>,
    state: Mutex<GroupedCaptureState>,
}

impl GroupedMoeCalibrationCapture {
    pub fn new(
        registry: Arc<CaptureRegistry>,
        telemetry: ExpertTelemetry,
    ) -> Result<Self, CalibError> {
        Self::with_collector(registry, telemetry, Arc::new(CalibCollector::new()))
    }

    pub fn with_collector(
        registry: Arc<CaptureRegistry>,
        telemetry: ExpertTelemetry,
        collector: Arc<CalibCollector>,
    ) -> Result<Self, CalibError> {
        validate_expert_registry(&registry, &telemetry)?;
        let staging = ExpertCaptureStaging::new(
            telemetry.num_layers,
            telemetry.num_experts,
            telemetry.quota.tile_rows,
        )?;
        Ok(Self {
            collector,
            registry,
            state: Mutex::new(GroupedCaptureState {
                telemetry,
                staging,
                pending: BTreeMap::new(),
            }),
        })
    }

    pub fn collector(&self) -> Arc<CalibCollector> {
        Arc::clone(&self.collector)
    }

    pub fn telemetry_snapshot(&self) -> ExpertTelemetry {
        self.state.lock().unwrap().telemetry.clone()
    }

    /// Finish capture accounting at corpus exhaustion. Returned partial tiles
    /// are diagnostic launch records; the collector retains their real rows and
    /// folds them during its streaming write.
    pub fn finalize(&self) -> Result<Vec<PartialCaptureTile>, CalibError> {
        let mut state = self.state.lock().unwrap();
        if !state.pending.is_empty() {
            return Err(CalibError::InvalidCapture(format!(
                "{} routed batches reached gate/up capture without matching down capture",
                state.pending.len()
            )));
        }
        let GroupedCaptureState {
            telemetry, staging, ..
        } = &mut *state;
        let partials = finalize_capture_staging(telemetry, staging)?;
        telemetry.reconcile()?;
        Ok(partials)
    }
}

impl MoePrefillCapture for GroupedMoeCalibrationCapture {
    fn capture(
        &self,
        gpu: &mut Gpu,
        batch: &MoePrefillCaptureBatch<'_>,
    ) -> Result<(), DispatchError> {
        self.capture_impl(gpu, batch)
            .map_err(|error| DispatchError::Capture(error.to_string()))
    }
}

impl GroupedMoeCalibrationCapture {
    fn capture_impl(
        &self,
        gpu: &mut Gpu,
        batch: &MoePrefillCaptureBatch<'_>,
    ) -> Result<(), CalibError> {
        validate_dispatch_batch(batch)?;
        let key = (gpu.device_id, batch.layer);
        match batch.point {
            MoePrefillCapturePoint::GateUpInput => {
                let total_slots = batch
                    .batch_size
                    .checked_mul(batch.k_top)
                    .ok_or_else(|| CalibError::InvalidRouting("routed slot overflow".into()))?;
                let indices = download_i32_prefix(gpu, batch.topk_indices, total_slots)?
                    .into_iter()
                    .enumerate()
                    .map(|(slot, expert)| {
                        usize::try_from(expert).map_err(|_| {
                            CalibError::InvalidRouting(format!(
                                "flat slot {slot} selected negative expert {expert}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let route_weights = download_f32_prefix(gpu, batch.topk_weights, total_slots)?;
                let routing = GroupedMoeRoutingPlan::build(
                    &indices,
                    batch.batch_size,
                    batch.k_top,
                    batch.num_experts,
                )
                .map_err(|error| CalibError::InvalidRouting(error.to_string()))?;

                let plan = {
                    let mut state = self.state.lock().unwrap();
                    if state.pending.contains_key(&key) {
                        return Err(CalibError::InvalidCapture(format!(
                            "device {} layer {} started a second routed batch before down capture",
                            gpu.device_id, batch.layer
                        )));
                    }
                    let GroupedCaptureState {
                        telemetry,
                        staging,
                        pending,
                    } = &mut *state;
                    record_grouped_router_batch(telemetry, batch.layer, &routing, &route_weights)?;
                    let plan = plan_grouped_expert_capture(
                        telemetry,
                        staging,
                        batch.layer,
                        ExpertCaptureRole::GateUpInput,
                        &routing,
                        &route_weights,
                    )?;
                    pending.insert(
                        key,
                        PendingCaptureBatch {
                            routing,
                            route_weights,
                        },
                    );
                    plan
                };
                self.collector.capture_grouped_plan(
                    gpu,
                    &self.registry,
                    batch.source,
                    batch.sorted_slot_index,
                    &plan,
                )
            }
            MoePrefillCapturePoint::DownInput => {
                let plan = {
                    let mut state = self.state.lock().unwrap();
                    let pending = state.pending.remove(&key).ok_or_else(|| {
                        CalibError::InvalidCapture(format!(
                            "device {} layer {} reached down capture without gate/up capture",
                            gpu.device_id, batch.layer
                        ))
                    })?;
                    if pending.routing.rows != batch.batch_size
                        || pending.routing.k_top != batch.k_top
                        || pending.routing.num_experts != batch.num_experts
                    {
                        return Err(CalibError::InvalidCapture(
                            "down capture geometry differs from gate/up capture".into(),
                        ));
                    }
                    let GroupedCaptureState {
                        telemetry, staging, ..
                    } = &mut *state;
                    let plan = plan_grouped_expert_capture(
                        telemetry,
                        staging,
                        batch.layer,
                        ExpertCaptureRole::DownInput,
                        &pending.routing,
                        &pending.route_weights,
                    )?;
                    plan
                };
                self.collector.capture_grouped_plan(
                    gpu,
                    &self.registry,
                    batch.source,
                    batch.sorted_slot_index,
                    &plan,
                )
            }
        }
    }
}

fn validate_expert_registry(
    registry: &CaptureRegistry,
    telemetry: &ExpertTelemetry,
) -> Result<(), CalibError> {
    for layer in 0..telemetry.num_layers {
        for expert in 0..telemetry.num_experts {
            for (role, projection) in [
                (ExpertCaptureRole::GateUpInput, ProjectionRole::GateUpInput),
                (ExpertCaptureRole::DownInput, ProjectionRole::DownInput),
            ] {
                let id = CaptureId::new(layer, projection, Some(expert));
                let descriptor = registry.get(id).ok_or_else(|| {
                    CalibError::InvalidCapture(format!(
                        "missing {role} descriptor for layer {layer} expert {expert}"
                    ))
                })?;
                if descriptor.policy != CapturePolicy::ImatrixOnly
                    || descriptor.expert_quota != Some(telemetry.quota)
                {
                    return Err(CalibError::InvalidCapture(format!(
                        "descriptor {} must be imatrix-only and use the job expert quota",
                        id.0
                    )));
                }
            }
        }
    }
    Ok(())
}

fn validate_dispatch_batch(batch: &MoePrefillCaptureBatch<'_>) -> Result<(), CalibError> {
    if batch.batch_size == 0
        || batch.k_top == 0
        || batch.num_experts == 0
        || batch.source_width == 0
    {
        return Err(CalibError::InvalidCapture(
            "routed capture batch has a zero dimension".into(),
        ));
    }
    let expected_divisor = match batch.point {
        MoePrefillCapturePoint::GateUpInput => batch.k_top,
        MoePrefillCapturePoint::DownInput => 1,
    };
    if batch.source_row_div != expected_divisor {
        return Err(CalibError::InvalidCapture(format!(
            "routed capture source divisor {} does not match expected {expected_divisor}",
            batch.source_row_div
        )));
    }
    if batch.topk_indices.dtype != DType::Raw
        || batch.topk_weights.dtype != DType::F32
        || batch.sorted_slot_index.dtype != DType::Raw
        || batch.source.dtype != DType::F32
    {
        return Err(CalibError::InvalidCapture(
            "routed capture tensor dtypes do not match the i32/F32 contract".into(),
        ));
    }
    Ok(())
}

fn download_i32_prefix(gpu: &Gpu, tensor: &GpuTensor, len: usize) -> Result<Vec<i32>, CalibError> {
    let bytes = len
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| CalibError::InvalidRouting("router index byte-size overflow".into()))?;
    if tensor.buf.size() < bytes {
        return Err(CalibError::InvalidRouting(format!(
            "router index buffer has {} bytes, expected at least {bytes}",
            tensor.buf.size()
        )));
    }
    let raw = gpu
        .download_raw(tensor, bytes)
        .map_err(|error| CalibError::Runtime(error.to_string()))?;
    Ok(raw
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn download_f32_prefix(gpu: &Gpu, tensor: &GpuTensor, len: usize) -> Result<Vec<f32>, CalibError> {
    let bytes = len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| CalibError::InvalidRouting("router weight byte-size overflow".into()))?;
    if tensor.buf.size() < bytes {
        return Err(CalibError::InvalidRouting(format!(
            "router weight buffer has {} bytes, expected at least {bytes}",
            tensor.buf.size()
        )));
    }
    let raw = gpu
        .download_raw(tensor, bytes)
        .map_err(|error| CalibError::Runtime(error.to_string()))?;
    Ok(raw
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

impl ExpertCaptureStaging {
    pub fn new(
        num_layers: usize,
        num_experts: usize,
        tile_rows: usize,
    ) -> Result<Self, CalibError> {
        if num_layers == 0 || num_experts == 0 || tile_rows == 0 {
            return Err(CalibError::InvalidOptions(
                "expert capture staging requires non-zero layers, experts, and tile rows".into(),
            ));
        }
        let entries = num_layers
            .checked_mul(num_experts)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CalibError::InvalidOptions("expert capture staging size overflow".into())
            })?;
        Ok(Self {
            num_layers,
            num_experts,
            tile_rows,
            staged_rows: vec![0; entries],
        })
    }

    fn index(
        &self,
        layer: usize,
        expert: usize,
        role: ExpertCaptureRole,
    ) -> Result<usize, CalibError> {
        if layer >= self.num_layers {
            return Err(CalibError::InvalidCapture(format!(
                "capture staging layer {layer} is outside 0..{}",
                self.num_layers
            )));
        }
        if expert >= self.num_experts {
            return Err(CalibError::InvalidCapture(format!(
                "capture staging expert {expert} is outside 0..{}",
                self.num_experts
            )));
        }
        let role_index = match role {
            ExpertCaptureRole::GateUpInput => 0,
            ExpertCaptureRole::DownInput => 1,
        };
        Ok((layer * self.num_experts + expert) * 2 + role_index)
    }

    pub fn staged_rows(
        &self,
        layer: usize,
        expert: usize,
        role: ExpertCaptureRole,
    ) -> Result<usize, CalibError> {
        Ok(self.staged_rows[self.index(layer, expert, role)?])
    }

    fn stage(
        &mut self,
        layer: usize,
        expert: usize,
        role: ExpertCaptureRole,
        sorted_start: usize,
        rows: usize,
        source_row_div: usize,
    ) -> Result<Vec<CaptureStageAction>, CalibError> {
        let index = self.index(layer, expert, role)?;
        let mut source_offset = 0usize;
        let mut remaining = rows;
        let mut actions = Vec::new();
        while remaining != 0 {
            let destination_row = self.staged_rows[index];
            let available = self.tile_rows - destination_row;
            let take = remaining.min(available);
            let staged = destination_row + take;
            let flush_full_tile = staged == self.tile_rows;
            actions.push(CaptureStageAction {
                layer,
                expert,
                role,
                sorted_start: sorted_start + source_offset,
                rows: take,
                destination_row,
                source_row_div,
                tile_rows: self.tile_rows,
                flush_full_tile,
            });
            self.staged_rows[index] = if flush_full_tile { 0 } else { staged };
            source_offset += take;
            remaining -= take;
        }
        Ok(actions)
    }

    pub fn drain_partials(&mut self) -> Vec<PartialCaptureTile> {
        let mut partials = Vec::new();
        for layer in 0..self.num_layers {
            for expert in 0..self.num_experts {
                for role in [ExpertCaptureRole::GateUpInput, ExpertCaptureRole::DownInput] {
                    let index = self
                        .index(layer, expert, role)
                        .expect("loop bounds match capture staging dimensions");
                    let rows = std::mem::take(&mut self.staged_rows[index]);
                    if rows != 0 {
                        partials.push(PartialCaptureTile {
                            layer,
                            expert,
                            role,
                            rows,
                        });
                    }
                }
            }
        }
        partials
    }
}

pub fn record_grouped_router_batch(
    telemetry: &mut ExpertTelemetry,
    layer: usize,
    routing: &GroupedMoeRoutingPlan,
    route_weights: &[f32],
) -> Result<(), CalibError> {
    validate_batch_contract(telemetry, None, layer, routing, route_weights)?;
    telemetry.record_grouped_batch_shape(
        layer,
        routing.total_slots,
        routing.m_total,
        routing.live_experts(),
    )?;
    let indices = flat_expert_indices(routing)?;
    for row in 0..routing.rows {
        let start = row * routing.k_top;
        telemetry.record_router_selection(
            layer,
            &indices[start..start + routing.k_top],
            &route_weights[start..start + routing.k_top],
        )?;
    }
    Ok(())
}

pub fn plan_grouped_expert_capture(
    telemetry: &mut ExpertTelemetry,
    staging: &mut ExpertCaptureStaging,
    layer: usize,
    role: ExpertCaptureRole,
    routing: &GroupedMoeRoutingPlan,
    route_weights: &[f32],
) -> Result<GroupedExpertCapturePlan, CalibError> {
    validate_batch_contract(telemetry, Some(staging), layer, routing, route_weights)?;
    let source_row_div = match role {
        ExpertCaptureRole::GateUpInput => routing.k_top,
        ExpertCaptureRole::DownInput => 1,
    };
    let mut admissions = Vec::with_capacity(routing.live_experts());
    let mut actions = Vec::new();
    let mut seen_rows = 0usize;
    let mut admitted_rows = 0usize;
    let mut batch_slack_rows = 0usize;

    for expert in 0..routing.num_experts {
        let slots = routing
            .real_slots_for_expert(expert)
            .expect("expert is inside routing plan bounds");
        if slots.is_empty() {
            continue;
        }
        let weights = slots
            .iter()
            .map(|&slot| route_weights[slot as usize])
            .collect::<Vec<_>>();
        let slack_before = telemetry
            .capture_stats(layer, expert, role)
            .batch_slack_rows;
        let admitted = telemetry.record_capture_batch(layer, expert, role, &weights)?;
        let slack_after = telemetry
            .capture_stats(layer, expert, role)
            .batch_slack_rows;
        let slack = usize::try_from(slack_after - slack_before)
            .map_err(|_| CalibError::InvalidCapture("expert batch slack exceeds usize".into()))?;
        let skipped = slots.len() - admitted;
        admissions.push(ExpertBatchAdmission {
            expert,
            seen_rows: slots.len(),
            admitted_rows: admitted,
            batch_slack_rows: slack,
            quota_skipped_rows: skipped,
        });
        seen_rows += slots.len();
        admitted_rows += admitted;
        batch_slack_rows += slack;
        if admitted != 0 {
            let expert_actions = staging.stage(
                layer,
                expert,
                role,
                routing.offsets[expert],
                admitted,
                source_row_div,
            )?;
            telemetry.record_capture_launches(
                layer,
                expert,
                role,
                expert_actions.len(),
                expert_actions
                    .iter()
                    .filter(|action| action.flush_full_tile)
                    .count(),
            )?;
            actions.extend(expert_actions);
        }
    }

    telemetry.mark_layer_saturated_if_complete(layer)?;

    Ok(GroupedExpertCapturePlan {
        layer,
        role,
        seen_rows,
        admitted_rows,
        batch_slack_rows,
        quota_skipped_rows: seen_rows - admitted_rows,
        admissions,
        actions,
    })
}

fn finalize_capture_staging(
    telemetry: &mut ExpertTelemetry,
    staging: &mut ExpertCaptureStaging,
) -> Result<Vec<PartialCaptureTile>, CalibError> {
    let partials = staging.drain_partials();
    for partial in &partials {
        telemetry.record_partial_reduction_tile(partial.layer, partial.expert, partial.role)?;
    }
    Ok(partials)
}

fn validate_batch_contract(
    telemetry: &ExpertTelemetry,
    staging: Option<&ExpertCaptureStaging>,
    layer: usize,
    routing: &GroupedMoeRoutingPlan,
    route_weights: &[f32],
) -> Result<(), CalibError> {
    if layer >= telemetry.num_layers {
        return Err(CalibError::InvalidRouting(format!(
            "layer {layer} is outside telemetry layer count {}",
            telemetry.num_layers
        )));
    }
    if routing.num_experts != telemetry.num_experts || routing.k_top != telemetry.k_top {
        return Err(CalibError::InvalidRouting(format!(
            "routing shape E{}/K{} does not match telemetry E{}/K{}",
            routing.num_experts, routing.k_top, telemetry.num_experts, telemetry.k_top
        )));
    }
    if route_weights.len() != routing.total_slots {
        return Err(CalibError::InvalidRouting(format!(
            "received {} route weights, expected {}",
            route_weights.len(),
            routing.total_slots
        )));
    }
    if let Some(slot) = route_weights.iter().position(|weight| !weight.is_finite()) {
        return Err(CalibError::InvalidRouting(format!(
            "route weight at flat slot {slot} is non-finite"
        )));
    }
    if let Some(staging) = staging {
        if staging.num_layers != telemetry.num_layers
            || staging.num_experts != telemetry.num_experts
            || staging.tile_rows != telemetry.quota.tile_rows
        {
            return Err(CalibError::InvalidCapture(format!(
                "capture staging {}/{}/{} does not match telemetry {}/{}/{}",
                staging.num_layers,
                staging.num_experts,
                staging.tile_rows,
                telemetry.num_layers,
                telemetry.num_experts,
                telemetry.quota.tile_rows
            )));
        }
    }
    Ok(())
}

fn flat_expert_indices(routing: &GroupedMoeRoutingPlan) -> Result<Vec<usize>, CalibError> {
    let mut indices = vec![usize::MAX; routing.total_slots];
    for expert in 0..routing.num_experts {
        for &slot in routing
            .real_slots_for_expert(expert)
            .expect("expert is inside routing plan bounds")
        {
            let slot = slot as usize;
            if std::mem::replace(&mut indices[slot], expert) != usize::MAX {
                return Err(CalibError::InvalidRouting(format!(
                    "flat routed slot {slot} is assigned more than once"
                )));
            }
        }
    }
    if let Some(slot) = indices.iter().position(|&expert| expert == usize::MAX) {
        return Err(CalibError::InvalidRouting(format!(
            "flat routed slot {slot} has no expert assignment"
        )));
    }
    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::contracts::{CaptureDescriptor, ExpertCaptureQuota};

    fn quota(min_rows: u64, target_rows: u64, tile_rows: usize) -> ExpertCaptureQuota {
        ExpertCaptureQuota {
            min_rows,
            target_rows,
            tile_rows,
            ..ExpertCaptureQuota::default()
        }
    }

    fn registry(
        num_layers: usize,
        num_experts: usize,
        quota: ExpertCaptureQuota,
    ) -> CaptureRegistry {
        let mut registry = CaptureRegistry::default();
        for layer in 0..num_layers {
            for expert in 0..num_experts {
                for (role, name) in [
                    (ProjectionRole::GateUpInput, "gate_up_proj"),
                    (ProjectionRole::DownInput, "down_proj"),
                ] {
                    registry
                        .register(CaptureDescriptor {
                            id: CaptureId::new(layer, role, Some(expert)),
                            output_names: vec![format!(
                                "model.layers.{layer}.mlp.experts.{expert}.{name}"
                            )],
                            input_width: if role == ProjectionRole::GateUpInput {
                                8
                            } else {
                                16
                            },
                            policy: CapturePolicy::ImatrixOnly,
                            layer,
                            role,
                            expert: Some(expert),
                            expert_quota: Some(quota),
                        })
                        .unwrap();
                }
            }
        }
        registry
    }

    #[test]
    fn runtime_callback_requires_complete_family_neutral_expert_registry() {
        let quota = quota(2, 4, 2);
        let telemetry = ExpertTelemetry::new(2, 3, 2, quota, 8).unwrap();
        let callback =
            GroupedMoeCalibrationCapture::new(Arc::new(registry(2, 3, quota)), telemetry.clone())
                .unwrap();
        assert!(callback.collector().is_empty());
        assert_eq!(callback.telemetry_snapshot(), telemetry);
        assert!(callback.finalize().unwrap().is_empty());

        let incomplete =
            GroupedMoeCalibrationCapture::new(Arc::new(CaptureRegistry::default()), telemetry);
        assert!(matches!(incomplete, Err(CalibError::InvalidCapture(_))));
    }

    #[test]
    fn carries_partial_tile_and_stops_at_exact_quota() {
        let mut telemetry = ExpertTelemetry::new(1, 1, 1, quota(2, 4, 2), 8).unwrap();
        let mut staging = ExpertCaptureStaging::new(1, 1, 2).unwrap();

        let first = GroupedMoeRoutingPlan::build(&[0, 0, 0], 3, 1, 1).unwrap();
        record_grouped_router_batch(&mut telemetry, 0, &first, &[1.0; 3]).unwrap();
        let first_capture = plan_grouped_expert_capture(
            &mut telemetry,
            &mut staging,
            0,
            ExpertCaptureRole::GateUpInput,
            &first,
            &[1.0; 3],
        )
        .unwrap();
        assert_eq!(first_capture.admitted_rows, 3);
        assert_eq!(first_capture.actions.len(), 2);
        assert_eq!(first_capture.actions[0].rows, 2);
        assert!(first_capture.actions[0].flush_full_tile);
        assert_eq!(first_capture.actions[1].destination_row, 0);
        assert_eq!(first_capture.actions[1].rows, 1);
        assert!(!first_capture.actions[1].flush_full_tile);
        assert_eq!(
            staging
                .staged_rows(0, 0, ExpertCaptureRole::GateUpInput)
                .unwrap(),
            1
        );

        let second = GroupedMoeRoutingPlan::build(&[0, 0, 0], 3, 1, 1).unwrap();
        record_grouped_router_batch(&mut telemetry, 0, &second, &[0.5; 3]).unwrap();
        let second_capture = plan_grouped_expert_capture(
            &mut telemetry,
            &mut staging,
            0,
            ExpertCaptureRole::GateUpInput,
            &second,
            &[0.5; 3],
        )
        .unwrap();
        assert_eq!(second.total_slots, 3, "teacher routing remains intact");
        assert_eq!(second_capture.admitted_rows, 1);
        assert_eq!(second_capture.quota_skipped_rows, 2);
        assert_eq!(second_capture.actions.len(), 1);
        assert_eq!(second_capture.actions[0].destination_row, 1);
        assert_eq!(second_capture.actions[0].rows, 1);
        assert!(second_capture.actions[0].flush_full_tile);
        assert_eq!(
            staging
                .staged_rows(0, 0, ExpertCaptureRole::GateUpInput)
                .unwrap(),
            0
        );
        let stats = telemetry.capture_stats(0, 0, ExpertCaptureRole::GateUpInput);
        assert_eq!(
            (
                stats.seen_rows,
                stats.admitted_rows,
                stats.quota_skipped_rows
            ),
            (6, 4, 2)
        );
        assert_eq!(stats.capture_gather_launches, 3);
        assert_eq!(stats.full_reduction_tiles, 2);
        assert_eq!(stats.partial_reduction_tiles, 0);
        let router = &telemetry.layer_snapshot(0).unwrap().router;
        assert_eq!(router.microbatches, 2);
        assert_eq!(router.active_expert_sum, 2);
        assert_eq!(router.max_active_experts, 1);
        assert_eq!(router.padded_routed_rows, 26);
    }

    #[test]
    fn unaligned_target_fills_open_tile_without_scheduling_another() {
        let mut telemetry = ExpertTelemetry::new(1, 1, 1, quota(2, 3, 2), 8).unwrap();
        let mut staging = ExpertCaptureStaging::new(1, 1, 2).unwrap();
        let routing = GroupedMoeRoutingPlan::build(&[0, 0, 0, 0, 0, 0], 6, 1, 1).unwrap();
        record_grouped_router_batch(&mut telemetry, 0, &routing, &[1.0; 6]).unwrap();

        let capture = plan_grouped_expert_capture(
            &mut telemetry,
            &mut staging,
            0,
            ExpertCaptureRole::GateUpInput,
            &routing,
            &[1.0; 6],
        )
        .unwrap();

        assert_eq!(capture.admitted_rows, 4);
        assert_eq!(capture.batch_slack_rows, 1);
        assert_eq!(capture.quota_skipped_rows, 2);
        assert_eq!(capture.actions.len(), 2);
        assert!(capture.actions.iter().all(|action| action.flush_full_tile));
        assert_eq!(capture.admissions[0].batch_slack_rows, 1);
        let down = plan_grouped_expert_capture(
            &mut telemetry,
            &mut staging,
            0,
            ExpertCaptureRole::DownInput,
            &routing,
            &[1.0; 6],
        )
        .unwrap();
        assert_eq!(down.batch_slack_rows, 1);
        assert_eq!(
            telemetry
                .layer_snapshot(0)
                .unwrap()
                .router
                .saturated_after_routed_tokens,
            Some(6)
        );
        telemetry.reconcile().unwrap();
    }

    #[test]
    fn finalization_records_one_partial_reduction_per_nonempty_role() {
        let mut telemetry = ExpertTelemetry::new(1, 1, 1, quota(2, 4, 2), 8).unwrap();
        let mut staging = ExpertCaptureStaging::new(1, 1, 2).unwrap();
        let routing = GroupedMoeRoutingPlan::build(&[0, 0, 0], 3, 1, 1).unwrap();
        record_grouped_router_batch(&mut telemetry, 0, &routing, &[1.0; 3]).unwrap();
        for role in [ExpertCaptureRole::GateUpInput, ExpertCaptureRole::DownInput] {
            plan_grouped_expert_capture(&mut telemetry, &mut staging, 0, role, &routing, &[1.0; 3])
                .unwrap();
        }

        let partials = finalize_capture_staging(&mut telemetry, &mut staging).unwrap();
        assert_eq!(partials.len(), 2);
        for role in [ExpertCaptureRole::GateUpInput, ExpertCaptureRole::DownInput] {
            let stats = telemetry.capture_stats(0, 0, role);
            assert_eq!(stats.capture_gather_launches, 2);
            assert_eq!(stats.full_reduction_tiles, 1);
            assert_eq!(stats.partial_reduction_tiles, 1);
        }
        telemetry.reconcile().unwrap();
    }

    #[test]
    fn saturated_and_undercovered_routes_are_filtered_independently() {
        let mut telemetry = ExpertTelemetry::new(1, 2, 2, quota(1, 2, 2), 8).unwrap();
        let mut staging = ExpertCaptureStaging::new(1, 2, 2).unwrap();

        let saturation = GroupedMoeRoutingPlan::build(&[0, 0], 1, 2, 2).unwrap();
        plan_grouped_expert_capture(
            &mut telemetry,
            &mut staging,
            0,
            ExpertCaptureRole::GateUpInput,
            &saturation,
            &[0.7, 0.3],
        )
        .unwrap();

        let mixed = GroupedMoeRoutingPlan::build(&[0, 1], 1, 2, 2).unwrap();
        let capture = plan_grouped_expert_capture(
            &mut telemetry,
            &mut staging,
            0,
            ExpertCaptureRole::GateUpInput,
            &mixed,
            &[0.6, 0.4],
        )
        .unwrap();
        assert_eq!(mixed.total_slots, 2);
        assert_eq!(
            capture.admissions[0],
            ExpertBatchAdmission {
                expert: 0,
                seen_rows: 1,
                admitted_rows: 0,
                batch_slack_rows: 0,
                quota_skipped_rows: 1,
            }
        );
        assert_eq!(capture.admissions[1].expert, 1);
        assert_eq!(capture.admissions[1].admitted_rows, 1);
        assert!(capture.actions.iter().all(|action| action.expert == 1));
    }

    #[test]
    fn k10_source_rows_match_gate_up_and_down_layouts() {
        let routes = (0..20).map(|slot| slot % 10).collect::<Vec<_>>();
        let routing = GroupedMoeRoutingPlan::build(&routes, 2, 10, 16).unwrap();
        let mut telemetry = ExpertTelemetry::new(1, 16, 10, quota(1, 4, 2), 16).unwrap();
        let mut staging = ExpertCaptureStaging::new(1, 16, 2).unwrap();
        record_grouped_router_batch(&mut telemetry, 0, &routing, &[0.1; 20]).unwrap();

        let gate = plan_grouped_expert_capture(
            &mut telemetry,
            &mut staging,
            0,
            ExpertCaptureRole::GateUpInput,
            &routing,
            &[0.1; 20],
        )
        .unwrap();
        let gate_nine = gate
            .actions
            .iter()
            .find(|action| action.expert == 9)
            .unwrap();
        assert_eq!(gate_nine.source_rows(&routing).unwrap(), vec![0, 1]);

        let down = plan_grouped_expert_capture(
            &mut telemetry,
            &mut staging,
            0,
            ExpertCaptureRole::DownInput,
            &routing,
            &[0.1; 20],
        )
        .unwrap();
        let down_nine = down
            .actions
            .iter()
            .find(|action| action.expert == 9)
            .unwrap();
        assert_eq!(down_nine.source_rows(&routing).unwrap(), vec![9, 19]);
        telemetry.reconcile().unwrap();
    }

    #[test]
    fn finalization_reports_partial_tiles_without_padding_counts() {
        let mut telemetry = ExpertTelemetry::new(1, 2, 1, quota(1, 4, 4), 8).unwrap();
        let mut staging = ExpertCaptureStaging::new(1, 2, 4).unwrap();
        let routing = GroupedMoeRoutingPlan::build(&[1], 1, 1, 2).unwrap();
        plan_grouped_expert_capture(
            &mut telemetry,
            &mut staging,
            0,
            ExpertCaptureRole::DownInput,
            &routing,
            &[1.0],
        )
        .unwrap();
        assert_eq!(
            staging.drain_partials(),
            vec![PartialCaptureTile {
                layer: 0,
                expert: 1,
                role: ExpertCaptureRole::DownInput,
                rows: 1,
            }]
        );
        assert!(staging.drain_partials().is_empty());
    }
}
