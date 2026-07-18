// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-layer logical KV plan over grouped physical [`KvCache`] allocations.
//!
//! Existing homogeneous cache constructors remain the physical primitive. This
//! module groups compatible owned layers, maps logical layers to group slots,
//! represents SWA rings and shared producers explicitly, and sizes mixed-width
//! attention scratch once at the maximum required geometry.

use crate::kv::KvCache;
use hip_bridge::HipResult;
use hipfire_rdna::{DType, Gpu, GpuTensor};
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvStorageKind {
    Full,
    SlidingWindow { window: usize },
}

impl KvStorageKind {
    pub fn physical_cap(self, max_seq: usize) -> usize {
        match self {
            Self::Full => max_seq,
            Self::SlidingWindow { window } => window,
        }
    }

    pub fn visible(self, absolute_pos: usize) -> Range<usize> {
        let end = absolute_pos + 1;
        match self {
            Self::Full => 0..end,
            Self::SlidingWindow { window } => end.saturating_sub(window)..end,
        }
    }

    pub fn physical_position(self, absolute_pos: usize) -> usize {
        match self {
            Self::Full => absolute_pos,
            Self::SlidingWindow { window } => absolute_pos % window,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerKvSpec {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub storage: KvStorageKind,
    pub shared_from: Option<usize>,
}

impl LayerKvSpec {
    pub fn owned(q_heads: usize, kv_heads: usize, head_dim: usize, storage: KvStorageKind) -> Self {
        Self {
            q_heads,
            kv_heads,
            head_dim,
            storage,
            shared_from: None,
        }
    }

    pub fn shared(mut self, producer_layer: usize) -> Self {
        self.shared_from = Some(producer_layer);
        self
    }

    pub fn q_width(&self) -> usize {
        self.q_heads * self.head_dim
    }

    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicalKvBinding {
    Owned { group: usize, slot: usize },
    SharedFrom { producer_layer: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalKvGroupPlan {
    pub storage: KvStorageKind,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub owned_layers: usize,
}

impl PhysicalKvGroupPlan {
    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    pub fn allocation_bytes(&self, max_seq: usize) -> usize {
        self.owned_layers
            * self.storage.physical_cap(max_seq)
            * self.kv_width()
            * 2
            * std::mem::size_of::<f32>()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionScratchRequirements {
    pub q_width: usize,
    pub kv_width: usize,
    pub attention_width: usize,
}

#[derive(Clone, Debug)]
pub struct LayeredKvPlan {
    max_seq: usize,
    layers: Vec<LayerKvSpec>,
    bindings: Vec<LogicalKvBinding>,
    groups: Vec<PhysicalKvGroupPlan>,
    scratch: AttentionScratchRequirements,
}

impl LayeredKvPlan {
    pub fn build(max_seq: usize, layers: Vec<LayerKvSpec>) -> Result<Self, String> {
        if max_seq == 0 {
            return Err("layered KV max_seq must be nonzero".to_string());
        }
        if layers.is_empty() {
            return Err("layered KV plan must contain at least one layer".to_string());
        }

        let mut groups: Vec<PhysicalKvGroupPlan> = Vec::new();
        let mut bindings = Vec::with_capacity(layers.len());
        let mut max_q = 0usize;
        let mut max_kv = 0usize;

        for (layer_idx, layer) in layers.iter().enumerate() {
            if layer.q_heads == 0 || layer.kv_heads == 0 || layer.head_dim == 0 {
                return Err(format!(
                    "layer {layer_idx}: q_heads, kv_heads, and head_dim must be nonzero"
                ));
            }
            if !layer.q_heads.is_multiple_of(layer.kv_heads) {
                return Err(format!(
                    "layer {layer_idx}: q_heads {} is not divisible by kv_heads {}",
                    layer.q_heads, layer.kv_heads
                ));
            }
            if let KvStorageKind::SlidingWindow { window } = layer.storage {
                if window == 0 || window > max_seq {
                    return Err(format!(
                        "layer {layer_idx}: sliding window {window} must be in 1..={max_seq}"
                    ));
                }
            }
            max_q = max_q.max(layer.q_width());
            max_kv = max_kv.max(layer.kv_width());

            if let Some(producer_layer) = layer.shared_from {
                if producer_layer >= layer_idx {
                    return Err(format!(
                        "layer {layer_idx}: shared producer {producer_layer} must precede the consumer"
                    ));
                }
                let producer = &layers[producer_layer];
                if producer.shared_from.is_some() {
                    return Err(format!(
                        "layer {layer_idx}: producer {producer_layer} is itself shared; producers must own storage"
                    ));
                }
                if producer.kv_heads != layer.kv_heads
                    || producer.head_dim != layer.head_dim
                    || producer.storage != layer.storage
                {
                    return Err(format!(
                        "layer {layer_idx}: shared geometry/storage does not match producer {producer_layer}"
                    ));
                }
                bindings.push(LogicalKvBinding::SharedFrom { producer_layer });
                continue;
            }

            let group = groups
                .iter()
                .position(|group| {
                    group.storage == layer.storage
                        && group.kv_heads == layer.kv_heads
                        && group.head_dim == layer.head_dim
                })
                .unwrap_or_else(|| {
                    groups.push(PhysicalKvGroupPlan {
                        storage: layer.storage,
                        kv_heads: layer.kv_heads,
                        head_dim: layer.head_dim,
                        owned_layers: 0,
                    });
                    groups.len() - 1
                });
            let slot = groups[group].owned_layers;
            groups[group].owned_layers += 1;
            bindings.push(LogicalKvBinding::Owned { group, slot });
        }

        Ok(Self {
            max_seq,
            layers,
            bindings,
            groups,
            scratch: AttentionScratchRequirements {
                q_width: max_q,
                kv_width: max_kv,
                attention_width: max_q,
            },
        })
    }

    pub fn homogeneous(
        n_layers: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<Self, String> {
        Self::build(
            max_seq,
            vec![LayerKvSpec::owned(q_heads, kv_heads, head_dim, KvStorageKind::Full); n_layers],
        )
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn layers(&self) -> &[LayerKvSpec] {
        &self.layers
    }

    pub fn bindings(&self) -> &[LogicalKvBinding] {
        &self.bindings
    }

    pub fn groups(&self) -> &[PhysicalKvGroupPlan] {
        &self.groups
    }

    pub fn scratch_requirements(&self) -> AttentionScratchRequirements {
        self.scratch
    }

    pub fn physical_owned_layers(&self) -> usize {
        self.groups.iter().map(|group| group.owned_layers).sum()
    }

    pub fn allocation_bytes(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.allocation_bytes(self.max_seq))
            .sum()
    }

    pub fn binding(&self, layer: usize) -> Result<LogicalKvBinding, String> {
        self.bindings
            .get(layer)
            .copied()
            .ok_or_else(|| format!("logical layer {layer} is out of range"))
    }

    pub fn resolved_binding(&self, layer: usize) -> Result<(usize, usize, usize), String> {
        match self.binding(layer)? {
            LogicalKvBinding::Owned { group, slot } => Ok((layer, group, slot)),
            LogicalKvBinding::SharedFrom { producer_layer } => {
                match self.binding(producer_layer)? {
                    LogicalKvBinding::Owned { group, slot } => Ok((producer_layer, group, slot)),
                    LogicalKvBinding::SharedFrom { .. } => {
                        Err("validated producer unexpectedly lacks storage".to_string())
                    }
                }
            }
        }
    }

    pub fn physical_position(&self, layer: usize, absolute_pos: usize) -> Result<usize, String> {
        if absolute_pos >= self.max_seq {
            return Err(format!(
                "absolute position {absolute_pos} exceeds max_seq {}",
                self.max_seq
            ));
        }
        let (producer, _, _) = self.resolved_binding(layer)?;
        Ok(self.layers[producer]
            .storage
            .physical_position(absolute_pos))
    }

    pub fn visible_positions(
        &self,
        layer: usize,
        absolute_pos: usize,
    ) -> Result<Range<usize>, String> {
        if absolute_pos >= self.max_seq {
            return Err(format!(
                "absolute position {absolute_pos} exceeds max_seq {}",
                self.max_seq
            ));
        }
        let (producer, _, _) = self.resolved_binding(layer)?;
        Ok(self.layers[producer].storage.visible(absolute_pos))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvSequenceCursor {
    next_pos: usize,
}

impl KvSequenceCursor {
    pub fn next_pos(&self) -> usize {
        self.next_pos
    }

    pub fn advance(&mut self, written_pos: usize, max_seq: usize) -> Result<(), String> {
        if written_pos != self.next_pos {
            return Err(format!(
                "KV growth is non-contiguous: wrote {written_pos}, expected {}",
                self.next_pos
            ));
        }
        if written_pos >= max_seq {
            return Err(format!(
                "KV position {written_pos} exceeds max_seq {max_seq}"
            ));
        }
        self.next_pos += 1;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.next_pos = 0;
    }
}

pub struct LayeredKvArena {
    plan: LayeredKvPlan,
    groups: Vec<KvCache>,
    cursor: KvSequenceCursor,
}

pub struct LayerCacheView<'a> {
    pub producer_layer: usize,
    pub group: usize,
    pub slot: usize,
    pub physical_position: usize,
    pub visible_positions: Range<usize>,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    /// KVarN metadata: `true` when this group's `KvCache` stores variance-normalized
    /// K + Q8 V (only Full-storage groups under [`LayeredKvArena::new_kvarn`]). The
    /// consumer then routes attention through `kvarn_attend` instead of `attention_f32`.
    pub quant_kvarn: bool,
    /// K recent-window ring for the KVarN path (`Some` iff `quant_kvarn`).
    pub k_window: Option<&'a GpuTensor>,
    /// Per-slot physical capacity (KVarN block/window sizing).
    pub physical_cap: usize,
    /// KVarN K-code bit width (2/4/8); meaningless unless `quant_kvarn`.
    pub kvarn_bits: usize,
}

impl LayeredKvArena {
    pub fn new_fp32(gpu: &mut Gpu, plan: LayeredKvPlan) -> HipResult<Self> {
        let mut groups = Vec::with_capacity(plan.groups.len());
        for group in &plan.groups {
            let owned = vec![true; group.owned_layers];
            groups.push(KvCache::new_gpu_capped_filtered(
                gpu,
                &owned,
                group.kv_heads,
                group.head_dim,
                plan.max_seq,
                group.storage.physical_cap(plan.max_seq),
            )?);
        }
        Ok(Self {
            plan,
            groups,
            cursor: KvSequenceCursor::default(),
        })
    }

    /// KVarN variant of [`Self::new_fp32`]: Full-storage groups whose
    /// `head_dim ∈ {128, 256}` hold variance-normalized `bits`-bit K + Q8 V;
    /// SlidingWindow (local) groups and any incompatible geometry stay F32 (the
    /// local rings never carry the long-context KV, matching gemma3's choice).
    /// Only gemma4 rides this arena's quant path, so no other family is affected.
    pub fn new_kvarn(gpu: &mut Gpu, plan: LayeredKvPlan, bits: usize) -> HipResult<Self> {
        let mut groups = Vec::with_capacity(plan.groups.len());
        for group in &plan.groups {
            let owned = vec![true; group.owned_layers];
            let cap = group.storage.physical_cap(plan.max_seq);
            let kvarn_ok = matches!(group.storage, KvStorageKind::Full)
                && (group.head_dim == 128 || group.head_dim == 256);
            let cache = if kvarn_ok {
                KvCache::new_gpu_kvarn_capped_filtered(
                    gpu,
                    &owned,
                    group.kv_heads,
                    group.head_dim,
                    plan.max_seq,
                    cap,
                    bits,
                )?
            } else {
                KvCache::new_gpu_capped_filtered(
                    gpu,
                    &owned,
                    group.kv_heads,
                    group.head_dim,
                    plan.max_seq,
                    cap,
                )?
            };
            groups.push(cache);
        }
        Ok(Self {
            plan,
            groups,
            cursor: KvSequenceCursor::default(),
        })
    }

    /// Compatibility adapter for an existing homogeneous F32 consumer. It
    /// validates the same logical plan used by mixed caches, then returns the
    /// unchanged physical `KvCache` type and constructor behavior.
    pub fn homogeneous_fp32_cache(
        gpu: &mut Gpu,
        n_layers: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<KvCache> {
        LayeredKvPlan::homogeneous(n_layers, q_heads, kv_heads, head_dim, max_seq)
            .expect("valid homogeneous KV geometry");
        KvCache::new_gpu(gpu, n_layers, kv_heads, head_dim, max_seq)
    }

    pub fn plan(&self) -> &LayeredKvPlan {
        &self.plan
    }

    pub fn next_pos(&self) -> usize {
        self.cursor.next_pos()
    }

    pub fn advance(&mut self, written_pos: usize) -> Result<(), String> {
        self.cursor.advance(written_pos, self.plan.max_seq)
    }

    pub fn reset(&mut self) {
        self.cursor.reset();
        for group in &mut self.groups {
            group.compact_offset = 0;
        }
    }

    pub fn view(&self, layer: usize, absolute_pos: usize) -> Result<LayerCacheView<'_>, String> {
        let (producer_layer, group, slot) = self.plan.resolved_binding(layer)?;
        let cache = &self.groups[group];
        Ok(LayerCacheView {
            producer_layer,
            group,
            slot,
            physical_position: self.plan.physical_position(layer, absolute_pos)?,
            visible_positions: self.plan.visible_positions(layer, absolute_pos)?,
            kv_heads: cache.n_kv_heads,
            head_dim: cache.head_dim,
            k: &cache.k_gpu[slot],
            v: &cache.v_gpu[slot],
            quant_kvarn: cache.quant_kvarn,
            k_window: if cache.quant_kvarn {
                Some(&cache.k_window[slot])
            } else {
                None
            },
            physical_cap: cache.physical_cap,
            kvarn_bits: cache.kvarn_bits,
        })
    }

    pub fn store_f32(
        &mut self,
        gpu: &Gpu,
        layer: usize,
        absolute_pos: usize,
        k: &[f32],
        v: &[f32],
    ) -> HipResult<()> {
        let LogicalKvBinding::Owned { group, slot } = self
            .plan
            .binding(layer)
            .unwrap_or_else(|error| panic!("layered KV store: {error}"))
        else {
            panic!("layered KV store: shared layer {layer} does not own K/V storage");
        };
        let spec = &self.plan.layers[layer];
        assert_eq!(k.len(), spec.kv_width(), "layer {layer}: K width mismatch");
        assert_eq!(v.len(), spec.kv_width(), "layer {layer}: V width mismatch");
        let physical_pos = self
            .plan
            .physical_position(layer, absolute_pos)
            .unwrap_or_else(|error| panic!("layered KV store: {error}"));
        match spec.storage {
            KvStorageKind::Full => self.groups[group].store_kv_pub(gpu, slot, physical_pos, k, v),
            KvStorageKind::SlidingWindow { window } => {
                // The existing SWA stage/attention kernels consume a head-major
                // `[kv_head, head_dim, window]` ring. `KvCache` is the flat
                // physical allocation owner; this diagnostic host-store path
                // writes that established ring layout explicitly.
                for head in 0..spec.kv_heads {
                    for dim in 0..spec.head_dim {
                        let source = head * spec.head_dim + dim;
                        let ring = (head * spec.head_dim + dim) * window + physical_pos;
                        gpu.hip.memcpy_htod_offset(
                            &self.groups[group].k_gpu[slot].buf,
                            ring * std::mem::size_of::<f32>(),
                            &k[source].to_ne_bytes(),
                        )?;
                        gpu.hip.memcpy_htod_offset(
                            &self.groups[group].v_gpu[slot].buf,
                            ring * std::mem::size_of::<f32>(),
                            &v[source].to_ne_bytes(),
                        )?;
                    }
                }
                Ok(())
            }
        }
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for group in self.groups {
            group.free_gpu(gpu);
        }
    }
}

pub struct LayeredAttentionScratch {
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    attention: GpuTensor,
    requirements: AttentionScratchRequirements,
}

/// Non-owning per-layer views. Do not free these tensors; the parent scratch
/// owns the four maximum-size allocations.
pub struct LayerAttentionScratchView {
    pub q: GpuTensor,
    pub k: GpuTensor,
    pub v: GpuTensor,
    pub attention: GpuTensor,
}

impl LayeredAttentionScratch {
    pub fn new(gpu: &mut Gpu, plan: &LayeredKvPlan) -> HipResult<Self> {
        let requirements = plan.scratch_requirements();
        Ok(Self {
            q: gpu.alloc_tensor(&[requirements.q_width], DType::F32)?,
            k: gpu.alloc_tensor(&[requirements.kv_width], DType::F32)?,
            v: gpu.alloc_tensor(&[requirements.kv_width], DType::F32)?,
            attention: gpu.alloc_tensor(&[requirements.attention_width], DType::F32)?,
            requirements,
        })
    }

    pub fn requirements(&self) -> AttentionScratchRequirements {
        self.requirements
    }

    pub fn view(
        &self,
        plan: &LayeredKvPlan,
        layer: usize,
    ) -> Result<LayerAttentionScratchView, String> {
        let spec = plan
            .layers
            .get(layer)
            .ok_or_else(|| format!("scratch layer {layer} is out of range"))?;
        if spec.q_width() > self.requirements.q_width
            || spec.kv_width() > self.requirements.kv_width
        {
            return Err(format!("scratch requirements do not cover layer {layer}"));
        }
        Ok(LayerAttentionScratchView {
            q: self.q.sub_offset(0, spec.q_width()),
            k: self.k.sub_offset(0, spec.kv_width()),
            v: self.v.sub_offset(0, spec.kv_width()),
            attention: self.attention.sub_offset(0, spec.q_width()),
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for tensor in [self.q, self.k, self.v, self.attention] {
            let _ = gpu.free_tensor(tensor);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mixed_plan() -> LayeredKvPlan {
        LayeredKvPlan::build(
            32,
            vec![
                LayerKvSpec::owned(4, 2, 256, KvStorageKind::SlidingWindow { window: 8 }),
                LayerKvSpec::owned(8, 1, 512, KvStorageKind::Full),
                LayerKvSpec::owned(4, 2, 256, KvStorageKind::SlidingWindow { window: 8 }).shared(0),
                LayerKvSpec::owned(8, 1, 512, KvStorageKind::Full).shared(1),
            ],
        )
        .unwrap()
    }

    #[test]
    fn mixed_geometry_groups_only_owned_physical_layers_and_sizes_scratch_once() {
        let plan = mixed_plan();
        assert_eq!(plan.layers().len(), 4);
        assert_eq!(plan.physical_owned_layers(), 2);
        assert_eq!(plan.groups().len(), 2);
        let local = 8 * (2 * 256) * 2 * 4;
        let global = 32 * (1 * 512) * 2 * 4;
        assert_eq!(plan.allocation_bytes(), local + global);
        assert_eq!(
            plan.scratch_requirements(),
            AttentionScratchRequirements {
                q_width: 8 * 512,
                kv_width: 2 * 256,
                attention_width: 8 * 512,
            }
        );
    }

    #[test]
    fn shared_layers_allocate_zero_and_resolve_the_right_producer() {
        let plan = mixed_plan();
        assert_eq!(
            plan.binding(2).unwrap(),
            LogicalKvBinding::SharedFrom { producer_layer: 0 }
        );
        assert_eq!(plan.resolved_binding(2).unwrap(), (0, 0, 0));
        assert_eq!(plan.resolved_binding(3).unwrap(), (1, 1, 0));
    }

    #[test]
    fn local_and_global_boundaries_map_window_minus_one_window_plus_one() {
        let plan = mixed_plan();
        assert_eq!(plan.physical_position(0, 7).unwrap(), 7);
        assert_eq!(plan.visible_positions(0, 7).unwrap(), 0..8);
        assert_eq!(plan.physical_position(0, 8).unwrap(), 0);
        assert_eq!(plan.visible_positions(0, 8).unwrap(), 1..9);
        assert_eq!(plan.physical_position(2, 9).unwrap(), 1);
        assert_eq!(plan.visible_positions(2, 9).unwrap(), 2..10);
        assert_eq!(plan.physical_position(1, 9).unwrap(), 9);
        assert_eq!(plan.visible_positions(1, 9).unwrap(), 0..10);
    }

    #[test]
    fn cursor_growth_reset_and_second_request_are_clean() {
        let mut cursor = KvSequenceCursor::default();
        for pos in 0..4 {
            cursor.advance(pos, 4).unwrap();
        }
        assert_eq!(cursor.next_pos(), 4);
        assert!(cursor.advance(4, 4).is_err());
        cursor.reset();
        assert_eq!(cursor.next_pos(), 0);
        cursor.advance(0, 4).unwrap();
        assert_eq!(cursor.next_pos(), 1);
        assert!(cursor.advance(0, 4).is_err());
    }

    #[test]
    fn homogeneous_adapter_plan_is_one_group_with_one_slot_per_layer() {
        let plan = LayeredKvPlan::homogeneous(3, 8, 2, 128, 16).unwrap();
        assert_eq!(plan.groups().len(), 1);
        assert_eq!(plan.groups()[0].owned_layers, 3);
        assert_eq!(plan.physical_owned_layers(), 3);
        for layer in 0..3 {
            assert_eq!(
                plan.binding(layer).unwrap(),
                LogicalKvBinding::Owned {
                    group: 0,
                    slot: layer
                }
            );
        }
    }

    #[test]
    fn invalid_sharing_and_geometry_are_rejected_at_plan_time() {
        let bad_forward = vec![
            LayerKvSpec::owned(4, 2, 256, KvStorageKind::Full).shared(1),
            LayerKvSpec::owned(4, 2, 256, KvStorageKind::Full),
        ];
        assert!(LayeredKvPlan::build(16, bad_forward).is_err());
        let bad_geometry = vec![
            LayerKvSpec::owned(4, 2, 256, KvStorageKind::Full),
            LayerKvSpec::owned(8, 1, 512, KvStorageKind::Full).shared(0),
        ];
        assert!(LayeredKvPlan::build(16, bad_geometry).is_err());
    }
}
