// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Priority scheduling and session batching policy shared by control planes.

#[cfg(test)]
use hipfire_model::model_worker_key_id;
use hipfire_model::{
    model_arch_family_from_str, normalize_model_worker_key, same_model_worker_key,
    AcceleratorDeviceInfo, AcceleratorInventory, ModelArchFamily, ModelWorkerKey,
};
use hipfire_state::{generate_state_kind_sets_match_exactly, normalize_generate_state_kind_set};
use std::collections::{BTreeMap, HashSet};

pub const SCHED_PRIORITY_REALTIME: u8 = 0;
pub const SCHED_PRIORITY_DEFAULT: u8 = 64;
pub const SCHED_PRIORITY_OPPORTUNISTIC: u8 = 255;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceBudget {
    pub system_memory_budget_bytes: u64,
    pub system_memory_headroom_bytes: u64,
    pub vram_budget_bytes: u64,
    pub vram_headroom_bytes: u64,
}

impl ResourceBudget {
    pub fn disabled() -> Self {
        Self {
            system_memory_budget_bytes: 0,
            system_memory_headroom_bytes: 0,
            vram_budget_bytes: 0,
            vram_headroom_bytes: 0,
        }
    }

    fn effective_system_limit(self) -> Option<u64> {
        effective_limit(
            self.system_memory_budget_bytes,
            self.system_memory_headroom_bytes,
        )
    }

    fn effective_vram_limit(self) -> Option<u64> {
        effective_limit(self.vram_budget_bytes, self.vram_headroom_bytes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceUsage {
    pub system_memory_bytes: u64,
    pub vram_bytes: u64,
}

impl ResourceUsage {
    pub fn zero() -> Self {
        Self {
            system_memory_bytes: 0,
            vram_bytes: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidencyMode {
    Auto,
    Full,
    QwenMoeModules,
}

impl ResidencyMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Full => "full",
            Self::QwenMoeModules => "qwen_moe_modules",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "full" => Some(Self::Full),
            "qwen_moe_modules" | "qwen35_moe_modules" | "qwen3.5_moe_modules" => {
                Some(Self::QwenMoeModules)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWorkerLedgerEntry {
    pub worker_key_id: String,
    pub model_path: String,
    pub residency_mode: ResidencyMode,
    pub resource_usage: ResourceUsage,
    pub last_used_seq: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelResidencyRequest {
    pub worker_key_id: String,
    pub model_path: String,
    pub requested_mode: ResidencyMode,
    pub estimated_full: ResourceUsage,
    pub estimated_qwen_moe_modules: Option<ResourceUsage>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelResidencyPlan {
    pub worker_key_id: String,
    pub residency_mode: ResidencyMode,
    pub module_vram_budget_bytes: Option<u64>,
    pub resource_usage: ResourceUsage,
    pub unload_worker_key_ids: Vec<String>,
    pub reason: String,
}

fn effective_limit(budget: u64, headroom: u64) -> Option<u64> {
    (budget > 0).then_some(budget.saturating_sub(headroom))
}

fn usage_fits(budget: ResourceBudget, usage: ResourceUsage) -> bool {
    budget
        .effective_system_limit()
        .is_none_or(|limit| usage.system_memory_bytes <= limit)
        && budget
            .effective_vram_limit()
            .is_none_or(|limit| usage.vram_bytes <= limit)
}

fn add_usage(a: ResourceUsage, b: ResourceUsage) -> ResourceUsage {
    ResourceUsage {
        system_memory_bytes: a.system_memory_bytes.saturating_add(b.system_memory_bytes),
        vram_bytes: a.vram_bytes.saturating_add(b.vram_bytes),
    }
}

fn subtract_usage(a: ResourceUsage, b: ResourceUsage) -> ResourceUsage {
    ResourceUsage {
        system_memory_bytes: a.system_memory_bytes.saturating_sub(b.system_memory_bytes),
        vram_bytes: a.vram_bytes.saturating_sub(b.vram_bytes),
    }
}

fn ledger_usage(workers: &[ResidentWorkerLedgerEntry]) -> ResourceUsage {
    workers.iter().fold(ResourceUsage::zero(), |sum, worker| {
        add_usage(sum, worker.resource_usage)
    })
}

pub fn plan_model_residency(
    budget: ResourceBudget,
    request: ModelResidencyRequest,
    resident_workers: &[ResidentWorkerLedgerEntry],
) -> Result<ModelResidencyPlan, String> {
    if resident_workers
        .iter()
        .any(|worker| worker.worker_key_id == request.worker_key_id)
    {
        return Ok(ModelResidencyPlan {
            worker_key_id: request.worker_key_id,
            residency_mode: ResidencyMode::Full,
            module_vram_budget_bytes: None,
            resource_usage: ResourceUsage::zero(),
            unload_worker_key_ids: Vec::new(),
            reason: "worker_already_resident".to_string(),
        });
    }

    let (mode, usage) = match request.requested_mode {
        ResidencyMode::Full => (ResidencyMode::Full, request.estimated_full),
        ResidencyMode::QwenMoeModules => {
            let Some(usage) = request.estimated_qwen_moe_modules else {
                return Err(
                    "qwen_moe_modules residency requested but module metadata is unavailable"
                        .to_string(),
                );
            };
            (ResidencyMode::QwenMoeModules, usage)
        }
        ResidencyMode::Auto => {
            if usage_fits(
                budget,
                add_usage(ledger_usage(resident_workers), request.estimated_full),
            ) {
                (ResidencyMode::Full, request.estimated_full)
            } else if let Some(module_usage) = request.estimated_qwen_moe_modules {
                (ResidencyMode::QwenMoeModules, module_usage)
            } else {
                (ResidencyMode::Full, request.estimated_full)
            }
        }
    };

    if !usage_fits(budget, usage) {
        return Err(format!(
            "requested {} residency exceeds configured budget/headroom",
            mode.as_str()
        ));
    }

    let mut current = ledger_usage(resident_workers);
    let mut unload = Vec::new();
    if !usage_fits(budget, add_usage(current, usage)) {
        let mut victims = resident_workers.to_vec();
        victims.sort_by_key(|worker| worker.last_used_seq);
        for victim in victims {
            current = subtract_usage(current, victim.resource_usage);
            unload.push(victim.worker_key_id);
            if usage_fits(budget, add_usage(current, usage)) {
                break;
            }
        }
    }

    if !usage_fits(budget, add_usage(current, usage)) {
        return Err("insufficient budget after evicting all eligible resident workers".to_string());
    }

    Ok(ModelResidencyPlan {
        worker_key_id: request.worker_key_id,
        residency_mode: mode,
        module_vram_budget_bytes: (mode == ResidencyMode::QwenMoeModules)
            .then_some(usage.vram_bytes),
        resource_usage: usage,
        unload_worker_key_ids: unload,
        reason: "admitted".to_string(),
    })
}

/// Work classes coordinated by the long-lived accelerator orchestrator.
///
/// Token and image work may be microbatched when callers provide the same
/// compatibility key. Training and maintenance remain singleton operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkloadClass {
    TokenPrefill,
    TokenDecode,
    ImageGeneration,
    Training,
    Maintenance,
}

/// Opaque scheduler attribution. Identity influences fairness and queued
/// cancellation, never runtime compatibility or model inputs.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct WorkloadOwner {
    pub user_id: Option<String>,
    pub token_id: Option<String>,
}

impl WorkloadOwner {
    pub fn authenticated(user_id: impl Into<String>, token_id: Option<String>) -> Self {
        Self {
            user_id: Some(user_id.into()),
            token_id,
        }
    }

    pub fn fairness_key(&self) -> &str {
        self.user_id.as_deref().unwrap_or("anonymous-local")
    }
}

impl WorkloadClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TokenPrefill => "token_prefill",
            Self::TokenDecode => "token_decode",
            Self::ImageGeneration => "image_generation",
            Self::Training => "training",
            Self::Maintenance => "maintenance",
        }
    }

    fn supports_microbatching(self) -> bool {
        matches!(
            self,
            Self::TokenPrefill | Self::TokenDecode | Self::ImageGeneration
        )
    }

    /// The coarse billing / rate-limit class this scheduling class rolls up to.
    /// This is the single source of truth that unifies the two taxonomies: a
    /// request classified once (as a scheduler `WorkloadClass`) derives its
    /// `hipfire_auth::WorkloadClass` here rather than being classified twice and
    /// risking drift.
    pub fn billing_class(self) -> hipfire_auth::WorkloadClass {
        match self {
            Self::TokenPrefill | Self::TokenDecode => hipfire_auth::WorkloadClass::Text,
            Self::ImageGeneration => hipfire_auth::WorkloadClass::Image,
            Self::Training => hipfire_auth::WorkloadClass::Training,
            Self::Maintenance => hipfire_auth::WorkloadClass::Other,
        }
    }
}

/// Conservatively additive resources used for admission and active leases.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorkloadResources {
    pub system_memory_bytes: u64,
    pub vram_bytes: u64,
    pub gpu_slots: u32,
    pub npu_slots: u32,
    pub cpu_threads: u32,
}

impl WorkloadResources {
    fn add(self, other: Self) -> Self {
        Self {
            system_memory_bytes: self
                .system_memory_bytes
                .saturating_add(other.system_memory_bytes),
            vram_bytes: self.vram_bytes.saturating_add(other.vram_bytes),
            gpu_slots: self.gpu_slots.saturating_add(other.gpu_slots),
            npu_slots: self.npu_slots.saturating_add(other.npu_slots),
            cpu_threads: self.cpu_threads.saturating_add(other.cpu_threads),
        }
    }

    fn subtract(self, other: Self) -> Self {
        Self {
            system_memory_bytes: self
                .system_memory_bytes
                .saturating_sub(other.system_memory_bytes),
            vram_bytes: self.vram_bytes.saturating_sub(other.vram_bytes),
            gpu_slots: self.gpu_slots.saturating_sub(other.gpu_slots),
            npu_slots: self.npu_slots.saturating_sub(other.npu_slots),
            cpu_threads: self.cpu_threads.saturating_sub(other.cpu_threads),
        }
    }

    fn fits_within(self, capacity: Self) -> bool {
        self.system_memory_bytes <= capacity.system_memory_bytes
            && self.vram_bytes <= capacity.vram_bytes
            && self.gpu_slots <= capacity.gpu_slots
            && self.npu_slots <= capacity.npu_slots
            && self.cpu_threads <= capacity.cpu_threads
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkloadSpec {
    pub id: String,
    pub class: WorkloadClass,
    pub priority: u8,
    pub enqueued_at_ms: u64,
    pub resources: WorkloadResources,
    pub owner: WorkloadOwner,
    /// Stable caller-defined compatibility key. It must include every property
    /// that affects a shared runtime invocation, such as worker, model, shape,
    /// quant/state mode, sampler, and precision.
    pub microbatch_key: Option<String>,
    pub max_microbatch_size: usize,
    pub exclusive: bool,
}

impl WorkloadSpec {
    pub fn singleton(
        id: impl Into<String>,
        class: WorkloadClass,
        priority: u8,
        enqueued_at_ms: u64,
        resources: WorkloadResources,
    ) -> Self {
        Self {
            id: id.into(),
            class,
            priority,
            enqueued_at_ms,
            resources,
            owner: WorkloadOwner::default(),
            microbatch_key: None,
            max_microbatch_size: 1,
            exclusive: class == WorkloadClass::Training,
        }
    }

    pub fn microbatchable(
        id: impl Into<String>,
        class: WorkloadClass,
        priority: u8,
        enqueued_at_ms: u64,
        resources: WorkloadResources,
        microbatch_key: impl Into<String>,
        max_microbatch_size: usize,
    ) -> Self {
        Self {
            id: id.into(),
            class,
            priority,
            enqueued_at_ms,
            resources,
            owner: WorkloadOwner::default(),
            microbatch_key: Some(microbatch_key.into()),
            max_microbatch_size: max_microbatch_size.max(1),
            exclusive: false,
        }
    }

    pub fn with_owner(mut self, owner: WorkloadOwner) -> Self {
        self.owner = owner;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkloadBatchLease {
    pub lease_id: u64,
    pub class: WorkloadClass,
    pub workloads: Vec<WorkloadSpec>,
    pub resources: WorkloadResources,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ContinuousSchedulerSnapshot {
    pub queued: usize,
    pub active_batches: usize,
    pub active_workloads: usize,
    pub active_resources: WorkloadResources,
    pub exclusive_active: bool,
}

/// Priority scheduler for continuous heterogeneous accelerator work.
///
/// The scheduler owns admission and leases, not execution. A server-owned
/// orchestrator repeatedly calls [`Self::next_batch`], dispatches the returned
/// work through the appropriate runtime, and completes the lease afterward.
#[derive(Debug)]
pub struct ContinuousWorkScheduler {
    capacity: WorkloadResources,
    max_queued: usize,
    aging_ms: u64,
    buckets: Vec<Vec<WorkloadSpec>>,
    queued_ids: HashSet<String>,
    active: BTreeMap<u64, WorkloadBatchLease>,
    next_lease_id: u64,
    last_scheduled_owner: Vec<Option<String>>,
}

impl ContinuousWorkScheduler {
    pub fn new(capacity: WorkloadResources, max_queued: usize, aging_ms: u64) -> Self {
        Self {
            capacity,
            max_queued,
            aging_ms,
            buckets: (0..=255).map(|_| Vec::new()).collect(),
            queued_ids: HashSet::new(),
            active: BTreeMap::new(),
            next_lease_id: 1,
            last_scheduled_owner: vec![None; 256],
        }
    }

    pub fn enqueue(&mut self, mut workload: WorkloadSpec) -> Result<(), String> {
        if workload.id.trim().is_empty() {
            return Err("workload id must not be empty".to_string());
        }
        if self.queued_ids.contains(&workload.id)
            || self
                .active
                .values()
                .any(|lease| lease.workloads.iter().any(|item| item.id == workload.id))
        {
            return Err(format!(
                "workload is already queued or active: {}",
                workload.id
            ));
        }
        if self.max_queued > 0 && self.queued_ids.len() >= self.max_queued {
            return Err(format!(
                "continuous scheduler backpressure: queued={} max={}",
                self.queued_ids.len(),
                self.max_queued
            ));
        }
        if !workload.resources.fits_within(self.capacity) {
            return Err(format!(
                "workload {} resource request exceeds scheduler capacity",
                workload.id
            ));
        }
        if workload.class == WorkloadClass::Training {
            workload.exclusive = true;
            workload.microbatch_key = None;
            workload.max_microbatch_size = 1;
        }
        workload.max_microbatch_size = workload.max_microbatch_size.max(1);
        let id = workload.id.clone();
        self.buckets[workload.priority as usize].push(workload);
        self.queued_ids.insert(id);
        Ok(())
    }

    pub fn cancel_pending(&mut self, id: &str) -> bool {
        if !self.queued_ids.contains(id) {
            return false;
        }
        for bucket in &mut self.buckets {
            if let Some(index) = bucket.iter().position(|workload| workload.id == id) {
                bucket.remove(index);
                self.queued_ids.remove(id);
                return true;
            }
        }
        self.queued_ids.remove(id);
        false
    }

    pub fn cancel_pending_by_user(&mut self, user_id: &str) -> Vec<WorkloadSpec> {
        self.cancel_pending_where(|owner| owner.user_id.as_deref() == Some(user_id))
    }

    pub fn cancel_pending_by_token(&mut self, token_id: &str) -> Vec<WorkloadSpec> {
        self.cancel_pending_where(|owner| owner.token_id.as_deref() == Some(token_id))
    }

    fn cancel_pending_where(
        &mut self,
        predicate: impl Fn(&WorkloadOwner) -> bool,
    ) -> Vec<WorkloadSpec> {
        let mut removed = Vec::new();
        for bucket in &mut self.buckets {
            let mut index = 0;
            while index < bucket.len() {
                if predicate(&bucket[index].owner) {
                    let workload = bucket.remove(index);
                    self.queued_ids.remove(&workload.id);
                    removed.push(workload);
                } else {
                    index += 1;
                }
            }
        }
        removed
    }

    pub fn next_batch(&mut self, now_ms: u64) -> Option<WorkloadBatchLease> {
        if self
            .active
            .values()
            .any(|lease| lease.workloads.iter().any(|workload| workload.exclusive))
        {
            return None;
        }

        let (priority, seed_index) = self.next_seed(now_ms)?;
        let seed = self.buckets[priority].get(seed_index)?.clone();
        if seed.exclusive && !self.active.is_empty() {
            return None;
        }

        let available = self.capacity.subtract(self.active_resources());
        if !seed.resources.fits_within(available) {
            return None;
        }

        let mut selected_indices = vec![seed_index];
        let mut resources = seed.resources;
        let mut microbatch_limit = seed.max_microbatch_size;
        if seed.class.supports_microbatching() && seed.microbatch_key.is_some() {
            for (index, candidate) in self.buckets[priority].iter().enumerate() {
                if selected_indices.len() >= microbatch_limit || index == seed_index {
                    continue;
                }
                if !workloads_microbatch_compatible(&seed, candidate) {
                    continue;
                }
                let candidate_limit = microbatch_limit.min(candidate.max_microbatch_size);
                if selected_indices.len() >= candidate_limit {
                    continue;
                }
                let combined = resources.add(candidate.resources);
                if combined.fits_within(available) {
                    resources = combined;
                    microbatch_limit = candidate_limit;
                    selected_indices.push(index);
                }
            }
        }

        selected_indices.sort_unstable();
        let mut workloads = Vec::with_capacity(selected_indices.len());
        for index in selected_indices.into_iter().rev() {
            workloads.push(self.buckets[priority].remove(index));
        }
        workloads.reverse();
        for workload in &workloads {
            self.queued_ids.remove(&workload.id);
        }

        let lease = WorkloadBatchLease {
            lease_id: self.next_lease_id,
            class: seed.class,
            workloads,
            resources,
        };
        self.next_lease_id = self.next_lease_id.saturating_add(1);
        self.active.insert(lease.lease_id, lease.clone());
        self.last_scheduled_owner[priority] = Some(seed.owner.fairness_key().to_string());
        Some(lease)
    }

    pub fn complete(&mut self, lease_id: u64) -> Option<WorkloadBatchLease> {
        self.active.remove(&lease_id)
    }

    /// The priority bucket the next `next_batch` call would draw from (honouring
    /// aging), without removing anything. Lower = served sooner. A running batch
    /// polls this to decide whether a higher-priority workload is waiting: if the
    /// peeked priority is strictly less than the running batch's own priority, a
    /// more-urgent workload would be granted next, so the batch should yield.
    pub fn peek_next_priority(&self, now_ms: u64) -> Option<u8> {
        self.next_seed(now_ms).map(|(priority, _)| priority as u8)
    }

    pub fn snapshot(&self) -> ContinuousSchedulerSnapshot {
        ContinuousSchedulerSnapshot {
            queued: self.queued_ids.len(),
            active_batches: self.active.len(),
            active_workloads: self
                .active
                .values()
                .map(|lease| lease.workloads.len())
                .sum(),
            active_resources: self.active_resources(),
            exclusive_active: self
                .active
                .values()
                .any(|lease| lease.workloads.iter().any(|workload| workload.exclusive)),
        }
    }

    fn active_resources(&self) -> WorkloadResources {
        self.active
            .values()
            .fold(WorkloadResources::default(), |total, lease| {
                total.add(lease.resources)
            })
    }

    fn next_seed(&self, now_ms: u64) -> Option<(usize, usize)> {
        if self.aging_ms > 0 {
            let mut oldest: Option<(u64, usize, usize)> = None;
            for (priority, bucket) in self.buckets.iter().enumerate() {
                for (index, workload) in bucket.iter().enumerate() {
                    if now_ms.saturating_sub(workload.enqueued_at_ms) < self.aging_ms {
                        continue;
                    }
                    let candidate = (workload.enqueued_at_ms, priority, index);
                    if oldest.is_none_or(|current| candidate < current) {
                        oldest = Some(candidate);
                    }
                }
            }
            if let Some((_, priority, index)) = oldest {
                return Some((priority, index));
            }
        }
        self.buckets
            .iter()
            .enumerate()
            .find_map(|(priority, bucket)| {
                if bucket.is_empty() {
                    return None;
                }
                let owners = distinct_owner_keys(bucket);
                let selected_index = self.last_scheduled_owner[priority]
                    .as_ref()
                    .and_then(|last| owners.iter().position(|owner| owner == last))
                    .map(|index| (index + 1) % owners.len())
                    .unwrap_or(0);
                let selected = &owners[selected_index];
                bucket
                    .iter()
                    .position(|workload| workload.owner.fairness_key() == selected)
                    .map(|index| (priority, index))
            })
    }
}

fn distinct_owner_keys(bucket: &[WorkloadSpec]) -> Vec<String> {
    let mut seen = HashSet::new();
    bucket
        .iter()
        .filter_map(|workload| {
            let owner = workload.owner.fairness_key().to_string();
            seen.insert(owner.clone()).then_some(owner)
        })
        .collect()
}

fn workloads_microbatch_compatible(a: &WorkloadSpec, b: &WorkloadSpec) -> bool {
    !a.exclusive
        && !b.exclusive
        && a.class == b.class
        && a.class.supports_microbatching()
        && a.microbatch_key.is_some()
        && a.microbatch_key == b.microbatch_key
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchedulerPriorityClass {
    Realtime,
    High,
    Interactive,
    Background,
    Bulk,
    Opportunistic,
}

impl SchedulerPriorityClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Realtime => "realtime",
            Self::High => "high",
            Self::Interactive => "interactive",
            Self::Background => "background",
            Self::Bulk => "bulk",
            Self::Opportunistic => "opportunistic",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchedulerPriorityPolicy {
    pub priority: u8,
    pub priority_class: SchedulerPriorityClass,
    pub coalesce_wait_ms: u64,
    pub max_batch_size: usize,
    pub resident_state_max: usize,
    pub spillable_batch_max: usize,
    pub disk_spill_allowed: bool,
    pub disk_spill_min_priority: u8,
    pub target_pair_tokens: usize,
    pub max_processing_ms: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SchedulerPolicyEnv {
    values: BTreeMap<String, String>,
}

impl SchedulerPolicyEnv {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn from_pairs<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self {
            values: pairs
                .into_iter()
                .map(|(key, value)| (key.into(), value.into()))
                .collect(),
        }
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpportunisticDispatchInput {
    pub compatible_queued_tokens: usize,
    pub schedule_clear: bool,
    pub target_pair_tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerPrefillPolicyControls {
    pub resident_state_cache: bool,
    pub resident_checkpoint_max: usize,
    pub state_cache_disk: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionStateHandle {
    pub worker_key: ModelWorkerKey,
    pub state_kinds: Vec<String>,
    pub logical_position: usize,
    pub cached_prefix_tokens: usize,
    pub runtime_state_handle: Option<String>,
    pub daemon_prefix_hash: Option<String>,
    pub daemon_prefix_len: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestSessionDraft {
    pub id: String,
    pub owner: WorkloadOwner,
    pub worker_key: ModelWorkerKey,
    pub priority: u8,
    pub prompt_tokens: Vec<u32>,
    pub suffix_tokens: Vec<u32>,
    pub cached_prefix_tokens: usize,
    pub state_handle: SessionStateHandle,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CreateRequestSessionInput {
    pub id: String,
    pub owner: WorkloadOwner,
    pub worker_key: ModelWorkerKey,
    pub prompt_tokens: Vec<u32>,
    pub cached_prefix_tokens: Option<usize>,
    pub priority: Option<i64>,
    pub state_kinds: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueuedPrefillRequest {
    pub session: RequestSessionDraft,
    pub enqueued_at_ms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NextBatchInput {
    pub now_ms: u64,
}

fn fair_ordered_prefill_bucket(
    bucket: &[QueuedPrefillRequest],
    last_owner: Option<&str>,
) -> Vec<QueuedPrefillRequest> {
    let mut owners = Vec::<String>::new();
    for entry in bucket {
        let owner = entry.session.owner.fairness_key().to_string();
        if !owners.contains(&owner) {
            owners.push(owner);
        }
    }
    if owners.len() <= 1 {
        return bucket.to_vec();
    }
    let start = last_owner
        .and_then(|last| owners.iter().position(|owner| owner == last))
        .map(|index| (index + 1) % owners.len())
        .unwrap_or(0);
    owners.rotate_left(start);

    let mut offsets = vec![0usize; owners.len()];
    let mut ordered = Vec::with_capacity(bucket.len());
    while ordered.len() < bucket.len() {
        let mut progressed = false;
        for (owner_index, owner) in owners.iter().enumerate() {
            let Some(entry) = bucket
                .iter()
                .filter(|entry| entry.session.owner.fairness_key() == owner)
                .nth(offsets[owner_index])
            else {
                continue;
            };
            offsets[owner_index] += 1;
            ordered.push(entry.clone());
            progressed = true;
        }
        if !progressed {
            break;
        }
    }
    ordered
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillBatchSelection {
    pub sessions: Vec<RequestSessionDraft>,
    pub policy: SchedulerPriorityPolicy,
    pub total_prompt_tokens: usize,
    pub total_suffix_tokens: usize,
    pub max_prompt_tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkerDeviceCapabilityStatus {
    pub supported: bool,
    pub capability: &'static str,
    pub reason: &'static str,
}

impl WorkerDeviceCapabilityStatus {
    pub fn supported() -> Self {
        Self {
            supported: true,
            capability: "supported",
            reason: "worker_device_available",
        }
    }

    pub fn unprobed() -> Self {
        Self {
            supported: true,
            capability: "unknown",
            reason: "accelerator_inventory_not_probed",
        }
    }

    pub fn unsupported(reason: &'static str) -> Self {
        Self {
            supported: false,
            capability: "unsupported",
            reason,
        }
    }
}

fn parse_integer(value: Option<&str>, fallback: i64) -> i64 {
    let Some(value) = value else {
        return fallback;
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return fallback;
    }
    trimmed
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .map(|v| v.floor() as i64)
        .unwrap_or(fallback)
}

fn parse_boolean(value: Option<&str>, fallback: bool) -> bool {
    let Some(value) = value else {
        return fallback;
    };
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "on" | "yes"
    )
}

pub fn clamp_scheduler_priority(value: i64) -> u8 {
    value.clamp(0, 255) as u8
}

pub fn parse_scheduler_priority(value: Option<&str>, fallback: u8) -> u8 {
    clamp_scheduler_priority(parse_integer(value, i64::from(fallback)))
}

pub fn parse_default_scheduler_priority(env: &SchedulerPolicyEnv) -> u8 {
    parse_scheduler_priority(
        env.get("HIPFIRE_SCHED_PRIORITY_DEFAULT"),
        SCHED_PRIORITY_DEFAULT,
    )
}

pub fn scheduler_priority_class(priority: u8) -> SchedulerPriorityClass {
    match priority {
        0 => SchedulerPriorityClass::Realtime,
        1..=63 => SchedulerPriorityClass::High,
        64..=127 => SchedulerPriorityClass::Interactive,
        128..=191 => SchedulerPriorityClass::Background,
        192..=254 => SchedulerPriorityClass::Bulk,
        255 => SchedulerPriorityClass::Opportunistic,
    }
}

pub fn parse_server_prefill_policy_controls(
    env: &SchedulerPolicyEnv,
) -> ServerPrefillPolicyControls {
    // Default ON: resident shared-prefix KV reuse — the batching/swarm design relies
    // on byte-identical prefixes reusing prefill KV. Env vars still override.
    let resident_state_cache = parse_boolean(
        env.get("HIPFIRE_SERVER_PREFILL_STATE_CACHE"),
        parse_boolean(env.get("HIPFIRE_SCHED_STATE_CACHE_RESIDENT"), true),
    );
    let resident_checkpoint_max = parse_integer(
        env.get("HIPFIRE_STATE_CACHE_MAX_CHECKPOINTS")
            .or_else(|| env.get("HIPFIRE_SERVER_PREFILL_STATE_CACHE_MAX")),
        4,
    )
    .clamp(0, 64) as usize;
    // Default ON: persist prefix/state checkpoints to disk too.
    let state_cache_disk = parse_boolean(
        env.get("HIPFIRE_SCHED_STATE_CACHE_DISK"),
        parse_boolean(
            env.get("HIPFIRE_SERVER_PREFILL_BATCH_STATE_CACHE_DISK"),
            true,
        ),
    );
    let legacy_state_cache_disk = parse_boolean(
        env.get("HIPFIRE_SERVER_PREFILL_BATCH_STATE_CACHE_DISK"),
        false,
    );
    ServerPrefillPolicyControls {
        resident_state_cache,
        resident_checkpoint_max,
        state_cache_disk: state_cache_disk || legacy_state_cache_disk,
    }
}

pub fn server_prefill_batch_enabled(env: &SchedulerPolicyEnv) -> bool {
    // On by default (Phase 2). The continuous-batching runner only takes effect
    // for batch-eligible requests (see `batch_runner::batch_eligible`); everything
    // else falls back to the legacy per-request path. `HIPFIRE_SERVER_PREFILL_BATCH=0`
    // (or `off`/`false`/`no`) is the kill switch back to the legacy path for all.
    env.get("HIPFIRE_SERVER_PREFILL_BATCH")
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "off" | "false" | "no"
            )
        })
        .unwrap_or(true)
}

pub fn server_prefill_batch_health_json(env: &SchedulerPolicyEnv) -> serde_json::Value {
    if !server_prefill_batch_enabled(env) {
        return serde_json::json!({ "enabled": false });
    }

    let priority = parse_default_scheduler_priority(env);
    let policy = scheduler_policy_for_priority(priority, env);
    let controls = parse_server_prefill_policy_controls(env);

    let mut payload = serde_json::json!({
        "enabled": true,
        "queued": 0,
        "eligible": 0,
        "selected": 0,
        "skipped": 0,
        "total_batches": 0,
        "fused_batches": 0,
        "fallback_batches": 0,
        "batch_size_histogram": {},
        "cache_hits": 0,
        "cache_misses": 0,
        "metadata_cache_hits": 0,
        "runtime_cache_hits": 0,
        "queue_size": 0,
        "pending_requests": 0,
        "resident_runtime_sessions": 0,
        "resident_decode_sessions": 0,
        "resident_checkpoints": 0,
        "resident_checkpoint_max": controls.resident_checkpoint_max,
        "resident_state_cache": controls.resident_state_cache,
        "resident_state_limit": policy.resident_state_max,
        "spillable_batch_max": policy.spillable_batch_max,
        "spillable_sessions": 0,
        "state_cache_disk": controls.state_cache_disk,
        "state_cache_disk_min_priority": policy.disk_spill_min_priority,
        "disk_spill_allowed": policy.disk_spill_allowed,
        "state_cache_evictions_total": 0,
        "state_cache_recompute_required_total": 0,
        "generate_batch_prefill_capability": "unknown",
        "generate_batch_prefill_capability_reason": "rust_server_daemon_capability_not_probed",
        "queue_wait_reason": "disabled",
        "fallback_reason": "rust_server_scheduler_metadata_only",
        "runtime_dispatch_skipped_reason": "rust_server_prefill_queue_not_enabled",
        "selected_batch_size": 0,
        "last_prefill_tokens": 0,
        "last_prefill_ms": 0,
        "last_prefill_tok_s": 0,
    });
    payload["policy"] = serde_json::json!({
        "priority": policy.priority,
        "priority_class": policy.priority_class.as_str(),
        "max_batch": policy.max_batch_size,
        "wait_ms": policy.coalesce_wait_ms,
        "target_pair_tokens": policy.target_pair_tokens,
        "max_processing_ms": policy.max_processing_ms,
    });
    payload
}

pub fn server_decode_batch_health_json(env: &SchedulerPolicyEnv) -> serde_json::Value {
    if !server_prefill_batch_enabled(env) {
        return serde_json::json!({ "enabled": false });
    }
    serde_json::json!({
        "enabled": true,
        "eligible": 0,
        "selected": 0,
        "skipped": 0,
        "active_sessions": 0,
        "selected_batch_size": 0,
        "total_batches": 0,
        "serial_batches": 0,
        "fused_batches": 0,
        "last_skipped_reason": "rust_server_decode_scheduler_not_enabled",
        "fallback_reason": "rust_server_scheduler_metadata_only",
    })
}

pub fn server_state_cache_health_json(env: &SchedulerPolicyEnv) -> serde_json::Value {
    if !server_prefill_batch_enabled(env) {
        return serde_json::json!({ "enabled": false });
    }
    let controls = parse_server_prefill_policy_controls(env);
    serde_json::json!({
        "enabled": controls.resident_state_cache || controls.state_cache_disk,
        "resident_enabled": controls.resident_state_cache,
        "resident_checkpoints": 0,
        "resident_checkpoint_max": controls.resident_checkpoint_max,
        "disk_enabled": controls.state_cache_disk,
        "daemon_prefix_hash": false,
        "daemon_prefix_hash_entries": 0,
        "semantic_boundary_checkpoints": false,
        "semantic_boundary_checkpoint_entries": 0,
        "prefix_hash_preflight_requests": 0,
        "prefix_hash_preflight_candidates": 0,
        "prefix_hash_preflight_matches": 0,
        "prefix_hash_preflight_boundary_matches": 0,
        "shared_prefix_fanout_groups": 0,
        "shared_prefix_fanout_followers": 0,
        "responses_previous_response_hits": 0,
        "responses_previous_response_misses": 0,
        "responses_stored_contexts": 0,
        "entries": 0,
        "bytes": 0,
        "metadata_hits": 0,
        "runtime_hits": 0,
        "evictions_total": 0,
        "recompute_required_total": 0,
    })
}

pub fn scheduler_policy_for_priority(
    priority: u8,
    env: &SchedulerPolicyEnv,
) -> SchedulerPriorityPolicy {
    let priority_class = scheduler_priority_class(priority);
    let max_batch_size = parse_integer(
        env.get("HIPFIRE_SCHED_PREFILL_BATCH_MAX")
            .or_else(|| env.get("HIPFIRE_SERVER_PREFILL_BATCH_MAX")),
        8,
    )
    .clamp(1, 64) as usize;
    let disk_spill_min_priority =
        parse_scheduler_priority(env.get("HIPFIRE_SCHED_STATE_CACHE_DISK_MIN_PRIORITY"), 128);
    let disk_spill_allowed = parse_server_prefill_policy_controls(env).state_cache_disk
        && priority >= disk_spill_min_priority;

    let state_policy_for_max = |effective_max_batch_size: usize| -> (usize, usize) {
        let resident_state_max = parse_integer(
            env.get("HIPFIRE_SCHED_RESIDENT_STATE_MAX"),
            effective_max_batch_size as i64,
        )
        .clamp(1, 64) as usize;
        let spillable_batch_max = parse_integer(
            env.get("HIPFIRE_SCHED_SPILLABLE_BATCH_MAX"),
            effective_max_batch_size as i64,
        )
        .clamp(resident_state_max as i64, 64) as usize;
        (resident_state_max, spillable_batch_max)
    };

    let legacy_interactive_wait = env.get("HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS");
    let realtime_wait =
        parse_integer(env.get("HIPFIRE_SCHED_PREFILL_WAIT_MS_REALTIME"), 0).max(0) as u64;
    let interactive_default = legacy_interactive_wait
        .map(|_| parse_integer(legacy_interactive_wait, 5))
        .unwrap_or(5);
    let interactive_wait = parse_integer(
        env.get("HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE"),
        interactive_default,
    )
    .max(0) as u64;
    let legacy_background_wait = legacy_interactive_wait
        .map(|_| parse_integer(legacy_interactive_wait, 25).max(0) * 2)
        .unwrap_or(25);
    let background_wait = parse_integer(
        env.get("HIPFIRE_SCHED_PREFILL_WAIT_MS_BACKGROUND"),
        legacy_background_wait,
    )
    .max(0) as u64;
    let opportunistic_background_wait =
        parse_integer(env.get("HIPFIRE_SCHED_PREFILL_WAIT_MS_BACKGROUND"), 25).max(0) as u64;
    let opportunistic_pair_tokens =
        parse_integer(env.get("HIPFIRE_SCHED_OPPORTUNISTIC_MIN_PAIR_TOKENS"), 256).max(1) as usize;

    let build = |coalesce_wait_ms, max_batch_size, target_pair_tokens, max_processing_ms| {
        let (resident_state_max, spillable_batch_max) = state_policy_for_max(max_batch_size);
        SchedulerPriorityPolicy {
            priority,
            priority_class,
            coalesce_wait_ms,
            max_batch_size,
            resident_state_max,
            spillable_batch_max,
            disk_spill_allowed,
            disk_spill_min_priority,
            target_pair_tokens,
            max_processing_ms,
        }
    };

    match priority_class {
        SchedulerPriorityClass::Realtime => build(realtime_wait, 1, 1, 25),
        SchedulerPriorityClass::High => {
            build(interactive_wait.min(2), max_batch_size.min(4), 32, 50)
        }
        SchedulerPriorityClass::Interactive => build(interactive_wait, max_batch_size, 64, 100),
        SchedulerPriorityClass::Background => build(background_wait, max_batch_size, 128, 250),
        SchedulerPriorityClass::Bulk => build(
            background_wait.saturating_mul(2),
            max_batch_size,
            opportunistic_pair_tokens,
            500,
        ),
        SchedulerPriorityClass::Opportunistic => build(
            opportunistic_background_wait.saturating_mul(4),
            max_batch_size,
            opportunistic_pair_tokens,
            1000,
        ),
    }
}

pub fn should_dispatch_opportunistic(input: OpportunisticDispatchInput) -> bool {
    input.schedule_clear || input.compatible_queued_tokens >= input.target_pair_tokens.max(1)
}

pub fn create_request_session_draft(input: CreateRequestSessionInput) -> RequestSessionDraft {
    let cached_prefix_tokens = input
        .cached_prefix_tokens
        .unwrap_or(0)
        .min(input.prompt_tokens.len());
    let worker_key = normalize_model_worker_key(&input.worker_key);
    let suffix_tokens = input.prompt_tokens[cached_prefix_tokens..].to_vec();
    let priority = input
        .priority
        .map(clamp_scheduler_priority)
        .unwrap_or(SCHED_PRIORITY_DEFAULT);
    RequestSessionDraft {
        id: input.id,
        owner: input.owner,
        worker_key: worker_key.clone(),
        priority,
        prompt_tokens: input.prompt_tokens,
        suffix_tokens,
        cached_prefix_tokens,
        state_handle: SessionStateHandle {
            worker_key,
            state_kinds: input.state_kinds,
            logical_position: cached_prefix_tokens,
            cached_prefix_tokens,
            runtime_state_handle: None,
            daemon_prefix_hash: None,
            daemon_prefix_len: None,
        },
    }
}

fn worker_key_has_feature(worker_key: &ModelWorkerKey, feature: &str) -> bool {
    worker_key
        .feature_flags
        .iter()
        .any(|flag| flag.eq_ignore_ascii_case(feature))
}

fn worker_key_family_contains(worker_key: &ModelWorkerKey, needle: &str) -> bool {
    let needle = needle.to_ascii_lowercase();
    worker_key.arch_id.to_ascii_lowercase().contains(&needle)
        || worker_key
            .artifact_path
            .to_ascii_lowercase()
            .contains(&needle)
        || worker_key
            .feature_flags
            .iter()
            .any(|flag| flag.to_ascii_lowercase().contains(&needle))
}

/// Canonical arch family for a worker key, resolved from the numeric arch_id
/// via the single-source [`model_arch_family_from_str`] table. Returns
/// [`ModelArchFamily::Unknown`] for non-numeric (legacy-name) arch_ids, which
/// the classifiers below still cover with a name substring fallback.
fn worker_key_arch_family(worker_key: &ModelWorkerKey) -> ModelArchFamily {
    model_arch_family_from_str(&worker_key.arch_id)
}

fn worker_key_is_qwen35(worker_key: &ModelWorkerKey) -> bool {
    // Source of truth is the canonical arch-family table (arch_id 5/6). The
    // name fallbacks remain for legacy string arch_ids / path-encoded families.
    matches!(
        worker_key_arch_family(worker_key),
        ModelArchFamily::Qwen35Dense | ModelArchFamily::Qwen35Moe
    ) || worker_key_family_contains(worker_key, "qwen35")
        || worker_key_family_contains(worker_key, "qwen3.5")
}

fn worker_key_is_state_arena_conservative(worker_key: &ModelWorkerKey) -> bool {
    // Canonical arch-family table (arch_id 10/11/14) + legacy name fallbacks.
    matches!(
        worker_key_arch_family(worker_key),
        ModelArchFamily::MiniMaxM2 | ModelArchFamily::Lfm2Moe | ModelArchFamily::NemotronH
    ) || worker_key_family_contains(worker_key, "minimax")
        || worker_key_family_contains(worker_key, "lfm2")
        || worker_key_family_contains(worker_key, "nemotron")
}

fn worker_key_has_hierarchical_kv(worker_key: &ModelWorkerKey) -> bool {
    let state_mode = worker_key.state_mode.to_ascii_lowercase();
    state_mode.contains("hier")
        || state_mode.contains("two_tier")
        || worker_key_has_feature(worker_key, "hierarchical_kv")
}

fn state_kinds_have_private_or_mamba(kinds: &[String]) -> bool {
    let normalized = normalize_generate_state_kind_set(kinds);
    normalized.iter().any(|kind| {
        matches!(
            kind.as_str(),
            "mamba_ssm" | "mamba_conv" | "backend_private" | "architecture_specific"
        )
    })
}

fn worker_key_requires_token_ordered_recurrent(worker_key: &ModelWorkerKey) -> bool {
    let state_mode = worker_key.state_mode.to_ascii_lowercase();
    state_mode.contains("mamba") || state_mode.contains("lfm2") || state_mode.contains("short_conv")
}

fn prefill_session_multi_session_batchable(session: &RequestSessionDraft) -> bool {
    if worker_key_has_feature(&session.worker_key, "multi_session_state_batch")
        || worker_key_has_feature(&session.worker_key, "fused_state_batch")
    {
        return true;
    }
    if worker_key_is_qwen35(&session.worker_key) {
        return true;
    }
    if worker_key_is_state_arena_conservative(&session.worker_key)
        || worker_key_requires_token_ordered_recurrent(&session.worker_key)
        || state_kinds_have_private_or_mamba(&session.state_handle.state_kinds)
    {
        return false;
    }
    true
}

pub fn sessions_compatible_for_prefill(a: &RequestSessionDraft, b: &RequestSessionDraft) -> bool {
    if !same_model_worker_key(&a.worker_key, &b.worker_key) {
        return false;
    }
    if worker_key_has_hierarchical_kv(&a.worker_key)
        != worker_key_has_hierarchical_kv(&b.worker_key)
    {
        return false;
    }
    if !generate_state_kind_sets_match_exactly(
        &a.state_handle.state_kinds,
        &b.state_handle.state_kinds,
    ) {
        return false;
    }
    if !prefill_session_multi_session_batchable(a) || !prefill_session_multi_session_batchable(b) {
        return a.id == b.id;
    }
    true
}

pub fn worker_device_capability_status(
    worker_key: &ModelWorkerKey,
    inventory: &AcceleratorInventory,
) -> WorkerDeviceCapabilityStatus {
    if inventory.source == "not_probed" {
        return WorkerDeviceCapabilityStatus::unprobed();
    }
    let worker_key = normalize_model_worker_key(worker_key);
    let accelerator_kind = worker_key.accelerator_kind.as_deref().unwrap_or("hip");
    let device_id = worker_key.device_id.as_deref().unwrap_or("0");
    let Some(device) = inventory
        .devices
        .iter()
        .find(|device| device.kind == accelerator_kind && device.device_id == device_id)
    else {
        return WorkerDeviceCapabilityStatus::unsupported("worker_device_not_found");
    };
    worker_device_capability_status_for_device(device)
}

fn worker_device_capability_status_for_device(
    device: &AcceleratorDeviceInfo,
) -> WorkerDeviceCapabilityStatus {
    if device.available {
        WorkerDeviceCapabilityStatus::supported()
    } else {
        WorkerDeviceCapabilityStatus::unsupported("worker_device_unavailable")
    }
}

#[derive(Clone, Debug)]
pub struct PriorityPrefillScheduler {
    env: SchedulerPolicyEnv,
    accelerator_inventory: Option<AcceleratorInventory>,
    buckets: Vec<Vec<QueuedPrefillRequest>>,
    queued_ids: HashSet<String>,
    queued_count: usize,
    last_scheduled_owner: Vec<Option<String>>,
}

impl Default for PriorityPrefillScheduler {
    fn default() -> Self {
        Self::new(SchedulerPolicyEnv::empty())
    }
}

impl PriorityPrefillScheduler {
    pub fn new(env: SchedulerPolicyEnv) -> Self {
        Self {
            env,
            accelerator_inventory: None,
            buckets: (0..=255).map(|_| Vec::new()).collect(),
            queued_ids: HashSet::new(),
            queued_count: 0,
            last_scheduled_owner: vec![None; 256],
        }
    }

    pub fn with_accelerator_inventory(
        env: SchedulerPolicyEnv,
        accelerator_inventory: AcceleratorInventory,
    ) -> Self {
        Self {
            accelerator_inventory: Some(accelerator_inventory),
            ..Self::new(env)
        }
    }

    pub fn size(&self) -> usize {
        self.queued_count
    }

    pub fn has_queued(&self, id: &str) -> bool {
        self.queued_ids.contains(id)
    }

    pub fn enqueue(
        &mut self,
        session: RequestSessionDraft,
        enqueued_at_ms: u64,
    ) -> Result<(), String> {
        self.require_worker_device_capability(&session)?;
        let max_queued = self.max_queued_requests();
        if max_queued > 0 && self.queued_count >= max_queued {
            return Err(format!(
                "prefill scheduler backpressure: queued={} max={max_queued}",
                self.queued_count
            ));
        }
        if self.queued_ids.contains(&session.id) {
            return Err(format!("request session is already queued: {}", session.id));
        }
        let priority = session.priority as usize;
        let id = session.id.clone();
        self.buckets[priority].push(QueuedPrefillRequest {
            session,
            enqueued_at_ms,
        });
        self.queued_ids.insert(id);
        self.queued_count += 1;
        Ok(())
    }

    pub fn enqueue_if_absent(
        &mut self,
        session: RequestSessionDraft,
        enqueued_at_ms: u64,
    ) -> Result<bool, String> {
        if self.has_queued(&session.id) {
            return Ok(false);
        }
        self.enqueue(session, enqueued_at_ms)?;
        Ok(true)
    }

    fn require_worker_device_capability(
        &self,
        session: &RequestSessionDraft,
    ) -> Result<(), String> {
        let Some(inventory) = self.accelerator_inventory.as_ref() else {
            return Ok(());
        };
        let status = worker_device_capability_status(&session.worker_key, inventory);
        if status.supported {
            Ok(())
        } else {
            Err(format!(
                "prefill scheduler worker device unsupported: request={} reason={}",
                session.id, status.reason
            ))
        }
    }

    pub fn cancel(&mut self, id: &str) -> bool {
        if !self.queued_ids.contains(id) {
            return false;
        }
        for bucket in &mut self.buckets {
            if let Some(index) = bucket.iter().position(|entry| entry.session.id == id) {
                bucket.remove(index);
                self.queued_ids.remove(id);
                self.queued_count = self.queued_count.saturating_sub(1);
                return true;
            }
        }
        self.queued_ids.remove(id);
        false
    }

    pub fn cancel_by_user(&mut self, user_id: &str) -> Vec<RequestSessionDraft> {
        self.cancel_where(|owner| owner.user_id.as_deref() == Some(user_id))
    }

    pub fn cancel_by_token(&mut self, token_id: &str) -> Vec<RequestSessionDraft> {
        self.cancel_where(|owner| owner.token_id.as_deref() == Some(token_id))
    }

    fn cancel_where(
        &mut self,
        predicate: impl Fn(&WorkloadOwner) -> bool,
    ) -> Vec<RequestSessionDraft> {
        let mut removed = Vec::new();
        for bucket in &mut self.buckets {
            let mut index = 0;
            while index < bucket.len() {
                if predicate(&bucket[index].session.owner) {
                    let entry = bucket.remove(index);
                    self.queued_ids.remove(&entry.session.id);
                    self.queued_count = self.queued_count.saturating_sub(1);
                    removed.push(entry.session);
                } else {
                    index += 1;
                }
            }
        }
        removed
    }

    pub fn next_prefill_batch(&mut self, input: NextBatchInput) -> Option<PrefillBatchSelection> {
        if let Some(aged) = self.select_aged_candidate(input.now_ms) {
            if let Some(first) = aged.sessions.first() {
                self.last_scheduled_owner[first.priority as usize] =
                    Some(first.owner.fairness_key().to_string());
            }
            self.remove_selected(&aged.sessions);
            return Some(aged);
        }

        for priority in 0..self.buckets.len() {
            if self.buckets[priority].is_empty() {
                continue;
            }
            let ordered = fair_ordered_prefill_bucket(
                &self.buckets[priority],
                self.last_scheduled_owner[priority].as_deref(),
            );
            let candidate = self.select_from_bucket(priority as u8, &ordered, input.now_ms)?;
            self.remove_selected(&candidate.sessions);
            self.last_scheduled_owner[priority] = candidate
                .sessions
                .first()
                .map(|session| session.owner.fairness_key().to_string());
            return Some(candidate);
        }
        None
    }

    fn select_from_bucket(
        &self,
        priority: u8,
        bucket: &[QueuedPrefillRequest],
        now_ms: u64,
    ) -> Option<PrefillBatchSelection> {
        let first = bucket.first()?;
        let policy = scheduler_policy_for_priority(first.session.priority, &self.env);
        let selection_limit = self.selection_limit(&policy);
        let compatible = bucket
            .iter()
            .filter(|entry| sessions_compatible_for_prefill(&first.session, &entry.session))
            .take(selection_limit)
            .cloned()
            .collect::<Vec<_>>();
        let total_suffix_tokens = compatible
            .iter()
            .map(|entry| entry.session.suffix_tokens.len())
            .sum();

        if policy.priority_class == SchedulerPriorityClass::Opportunistic {
            let dispatch = should_dispatch_opportunistic(OpportunisticDispatchInput {
                compatible_queued_tokens: total_suffix_tokens,
                schedule_clear: !self.has_queued_higher_priority(priority),
                target_pair_tokens: policy.target_pair_tokens,
            });
            return dispatch.then(|| self.selection(&compatible, policy));
        }

        let waited_ms = now_ms.saturating_sub(first.enqueued_at_ms);
        if compatible.len() >= selection_limit || waited_ms >= policy.coalesce_wait_ms {
            Some(self.selection(&compatible, policy))
        } else {
            None
        }
    }

    fn max_queued_requests(&self) -> usize {
        parse_integer(self.env.get("HIPFIRE_SCHED_PREFILL_MAX_QUEUED"), 256).max(0) as usize
    }

    fn aging_ms(&self) -> u64 {
        parse_integer(self.env.get("HIPFIRE_SCHED_DEADLINE_AGING_MS"), 0).max(0) as u64
    }

    fn select_aged_candidate(&self, now_ms: u64) -> Option<PrefillBatchSelection> {
        let aging_ms = self.aging_ms();
        if aging_ms == 0 {
            return None;
        }
        for bucket in &self.buckets {
            if bucket.is_empty() {
                continue;
            }
            let Some(first_aged) = bucket
                .iter()
                .find(|entry| now_ms.saturating_sub(entry.enqueued_at_ms) >= aging_ms)
            else {
                continue;
            };
            let policy = scheduler_policy_for_priority(first_aged.session.priority, &self.env);
            let selection_limit = self.selection_limit(&policy);
            let compatible = bucket
                .iter()
                .filter(|entry| {
                    sessions_compatible_for_prefill(&first_aged.session, &entry.session)
                })
                .take(selection_limit)
                .cloned()
                .collect::<Vec<_>>();
            return Some(self.selection(&compatible, policy));
        }
        None
    }

    fn has_queued_higher_priority(&self, priority: u8) -> bool {
        self.buckets[..priority as usize]
            .iter()
            .any(|bucket| !bucket.is_empty())
    }

    fn selection_limit(&self, policy: &SchedulerPriorityPolicy) -> usize {
        if policy.disk_spill_allowed {
            policy.max_batch_size.max(policy.spillable_batch_max)
        } else {
            policy.max_batch_size
        }
    }

    fn selection(
        &self,
        entries: &[QueuedPrefillRequest],
        policy: SchedulerPriorityPolicy,
    ) -> PrefillBatchSelection {
        let sessions = entries
            .iter()
            .map(|entry| entry.session.clone())
            .collect::<Vec<_>>();
        PrefillBatchSelection {
            total_prompt_tokens: sessions
                .iter()
                .map(|session| session.prompt_tokens.len())
                .sum(),
            total_suffix_tokens: sessions
                .iter()
                .map(|session| session.suffix_tokens.len())
                .sum(),
            max_prompt_tokens: sessions
                .iter()
                .map(|session| session.prompt_tokens.len())
                .max()
                .unwrap_or(0),
            sessions,
            policy,
        }
    }

    fn remove_selected(&mut self, sessions: &[RequestSessionDraft]) {
        for session in sessions {
            self.cancel(&session.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> SchedulerPolicyEnv {
        SchedulerPolicyEnv::from_pairs(pairs.iter().copied())
    }

    fn qwen_worker() -> ModelWorkerKey {
        ModelWorkerKey {
            artifact_path: "/models/qwen3.6-35b-a3b-mq4.hfq".to_string(),
            artifact_digest: Some("sha256:qwen-a3b".to_string()),
            arch_id: "6".to_string(),
            quant_family: "mq4".to_string(),
            state_mode: "q8+deltanet".to_string(),
            max_seq_bucket: 4096,
            accelerator_kind: None,
            device_id: None,
            feature_flags: vec!["prefill_batch".to_string(), "qwen35".to_string()],
        }
    }

    fn nemotron_worker() -> ModelWorkerKey {
        ModelWorkerKey {
            artifact_path: "/models/nemotron-3-ultra-550b-a55b-bf16.hfq".to_string(),
            artifact_digest: Some("sha256:nemotron".to_string()),
            arch_id: "nemotron3".to_string(),
            quant_family: "bf16".to_string(),
            state_mode: "q8+mamba".to_string(),
            max_seq_bucket: 8192,
            accelerator_kind: None,
            device_id: None,
            feature_flags: vec!["mamba".to_string(), "prefill_batch".to_string()],
        }
    }

    fn session(id: &str, priority: u8, tokens: usize) -> RequestSessionDraft {
        session_with(
            id,
            priority,
            tokens,
            qwen_worker(),
            &["attention_kv", "deltanet_recurrent"],
            0,
        )
    }

    fn owned_session(id: &str, user: &str, token: &str, priority: u8) -> RequestSessionDraft {
        let mut session = session(id, priority, 8);
        session.owner = WorkloadOwner::authenticated(user, Some(token.to_string()));
        session
    }

    fn session_with(
        id: &str,
        priority: u8,
        tokens: usize,
        worker_key: ModelWorkerKey,
        state_kinds: &[&str],
        cached_prefix_tokens: usize,
    ) -> RequestSessionDraft {
        create_request_session_draft(CreateRequestSessionInput {
            id: id.to_string(),
            owner: WorkloadOwner::default(),
            worker_key,
            prompt_tokens: (1..=tokens as u32).collect(),
            cached_prefix_tokens: Some(cached_prefix_tokens),
            priority: Some(i64::from(priority)),
            state_kinds: state_kinds.iter().map(|kind| kind.to_string()).collect(),
        })
    }

    fn ids(batch: Option<PrefillBatchSelection>) -> Vec<String> {
        batch
            .map(|batch| {
                batch
                    .sessions
                    .into_iter()
                    .map(|session| session.id)
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn priority_parsing_and_classes_match_bun_policy() {
        assert_eq!(clamp_scheduler_priority(-1), 0);
        assert_eq!(clamp_scheduler_priority(999), 255);
        assert_eq!(
            parse_default_scheduler_priority(&SchedulerPolicyEnv::empty()),
            64
        );
        assert_eq!(
            parse_default_scheduler_priority(&env(&[("HIPFIRE_SCHED_PRIORITY_DEFAULT", "192")])),
            192
        );
        assert_eq!(parse_scheduler_priority(Some("not-a-number"), 64), 64);
        assert_eq!(scheduler_priority_class(0).as_str(), "realtime");
        assert_eq!(scheduler_priority_class(63).as_str(), "high");
        assert_eq!(scheduler_priority_class(64).as_str(), "interactive");
        assert_eq!(scheduler_priority_class(191).as_str(), "background");
        assert_eq!(scheduler_priority_class(254).as_str(), "bulk");
        assert_eq!(scheduler_priority_class(255).as_str(), "opportunistic");
    }

    #[test]
    fn scheduler_policy_respects_waits_batch_limits_and_spill() {
        let policy = scheduler_policy_for_priority(
            64,
            &env(&[
                ("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "8"),
                ("HIPFIRE_SCHED_RESIDENT_STATE_MAX", "3"),
                ("HIPFIRE_SCHED_SPILLABLE_BATCH_MAX", "12"),
                ("HIPFIRE_SCHED_STATE_CACHE_DISK", "1"),
                ("HIPFIRE_SCHED_STATE_CACHE_DISK_MIN_PRIORITY", "64"),
                ("HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE", "7"),
            ]),
        );
        assert_eq!(policy.priority_class, SchedulerPriorityClass::Interactive);
        assert_eq!(policy.max_batch_size, 8);
        assert_eq!(policy.coalesce_wait_ms, 7);
        assert_eq!(policy.resident_state_max, 3);
        assert_eq!(policy.spillable_batch_max, 12);
        assert!(policy.disk_spill_allowed);
        assert_eq!(policy.disk_spill_min_priority, 64);

        let high =
            scheduler_policy_for_priority(1, &env(&[("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "16")]));
        assert_eq!(high.max_batch_size, 4);
        assert_eq!(high.resident_state_max, 4);
    }

    #[test]
    fn server_prefill_batch_enabled_by_default_with_kill_switch() {
        // Phase 2: on by default (unset -> enabled).
        assert!(server_prefill_batch_enabled(&SchedulerPolicyEnv::empty()));

        // The kill switch disables it and collapses the health JSON.
        let off = env(&[("HIPFIRE_SERVER_PREFILL_BATCH", "0")]);
        assert!(!server_prefill_batch_enabled(&off));
        assert_eq!(
            server_prefill_batch_health_json(&off),
            serde_json::json!({ "enabled": false })
        );
    }

    #[test]
    fn server_prefill_batch_health_uses_shared_scheduler_policy() {
        let payload = server_prefill_batch_health_json(&env(&[
            ("HIPFIRE_SERVER_PREFILL_BATCH", "1"),
            ("HIPFIRE_SCHED_PRIORITY_DEFAULT", "128"),
            ("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "8"),
            ("HIPFIRE_SCHED_RESIDENT_STATE_MAX", "3"),
            ("HIPFIRE_SCHED_SPILLABLE_BATCH_MAX", "12"),
            ("HIPFIRE_SCHED_STATE_CACHE_DISK", "1"),
            ("HIPFIRE_SCHED_STATE_CACHE_DISK_MIN_PRIORITY", "64"),
            ("HIPFIRE_SCHED_PREFILL_WAIT_MS_BACKGROUND", "11"),
        ]));

        assert_eq!(payload["enabled"], true);
        assert_eq!(payload["policy"]["priority"], 128);
        assert_eq!(payload["policy"]["priority_class"], "background");
        assert_eq!(payload["policy"]["max_batch"], 8);
        assert_eq!(payload["policy"]["wait_ms"], 11);
        assert_eq!(payload["resident_state_limit"], 3);
        assert_eq!(payload["spillable_batch_max"], 12);
        assert_eq!(payload["state_cache_disk"], true);
        assert_eq!(payload["disk_spill_allowed"], true);
        assert_eq!(
            payload["generate_batch_prefill_capability_reason"],
            "rust_server_daemon_capability_not_probed"
        );
    }

    #[test]
    fn server_health_state_cache_uses_shared_scheduler_controls() {
        let payload = server_state_cache_health_json(&env(&[
            ("HIPFIRE_SERVER_PREFILL_BATCH", "true"),
            ("HIPFIRE_SERVER_PREFILL_STATE_CACHE", "1"),
            ("HIPFIRE_STATE_CACHE_MAX_CHECKPOINTS", "5"),
            ("HIPFIRE_SERVER_PREFILL_BATCH_STATE_CACHE_DISK", "1"),
        ]));

        assert_eq!(payload["enabled"], true);
        assert_eq!(payload["resident_enabled"], true);
        assert_eq!(payload["resident_checkpoint_max"], 5);
        assert_eq!(payload["disk_enabled"], true);
    }

    #[test]
    fn scheduler_policy_matches_realtime_and_interactive_bun_parity() {
        let realtime = scheduler_policy_for_priority(0, &SchedulerPolicyEnv::empty());
        let interactive = scheduler_policy_for_priority(64, &SchedulerPolicyEnv::empty());
        assert_eq!(realtime.priority_class, SchedulerPriorityClass::Realtime);
        assert_eq!(realtime.coalesce_wait_ms, 0);
        assert_eq!(realtime.max_batch_size, 1);
        assert_eq!(realtime.resident_state_max, 1);
        assert_eq!(realtime.spillable_batch_max, 1);
        assert!(realtime.max_processing_ms < interactive.max_processing_ms);

        let configured = scheduler_policy_for_priority(
            64,
            &env(&[
                ("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "16"),
                ("HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE", "7"),
            ]),
        );
        assert_eq!(
            configured.priority_class,
            SchedulerPriorityClass::Interactive
        );
        assert_eq!(configured.coalesce_wait_ms, 7);
        assert_eq!(configured.max_batch_size, 16);
        assert_eq!(configured.target_pair_tokens, 64);
    }

    #[test]
    fn scheduler_policy_matches_legacy_wait_and_opportunistic_bun_parity() {
        let legacy = env(&[("HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS", "9")]);
        let interactive = scheduler_policy_for_priority(64, &legacy);
        let background = scheduler_policy_for_priority(128, &legacy);
        assert_eq!(interactive.coalesce_wait_ms, 9);
        assert_eq!(background.coalesce_wait_ms, 18);

        let opportunistic = scheduler_policy_for_priority(
            255,
            &env(&[
                ("HIPFIRE_SCHED_PREFILL_WAIT_MS_BACKGROUND", "20"),
                ("HIPFIRE_SCHED_OPPORTUNISTIC_MIN_PAIR_TOKENS", "512"),
            ]),
        );
        let default_background = scheduler_policy_for_priority(128, &SchedulerPolicyEnv::empty());
        assert_eq!(
            opportunistic.priority_class,
            SchedulerPriorityClass::Opportunistic
        );
        assert_eq!(opportunistic.coalesce_wait_ms, 80);
        assert_eq!(opportunistic.target_pair_tokens, 512);
        assert!(opportunistic.max_processing_ms > default_background.max_processing_ms);
    }

    #[test]
    fn scheduler_policy_matches_state_residency_and_spill_bun_parity() {
        let batch_env = env(&[("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "16")]);
        let realtime = scheduler_policy_for_priority(0, &batch_env);
        let high = scheduler_policy_for_priority(1, &batch_env);
        assert_eq!(realtime.max_batch_size, 1);
        assert_eq!(realtime.resident_state_max, 1);
        assert_eq!(realtime.spillable_batch_max, 1);
        assert_eq!(high.max_batch_size, 4);
        assert_eq!(high.resident_state_max, 4);
        assert_eq!(high.spillable_batch_max, 4);

        let disk_spill = env(&[("HIPFIRE_SCHED_STATE_CACHE_DISK", "1")]);
        assert!(!scheduler_policy_for_priority(64, &disk_spill).disk_spill_allowed);
        assert!(scheduler_policy_for_priority(128, &disk_spill).disk_spill_allowed);

        let legacy_disk_spill = env(&[("HIPFIRE_SERVER_PREFILL_BATCH_STATE_CACHE_DISK", "true")]);
        assert!(scheduler_policy_for_priority(255, &legacy_disk_spill).disk_spill_allowed);

        let clamped = scheduler_policy_for_priority(
            64,
            &env(&[
                ("HIPFIRE_SCHED_RESIDENT_STATE_MAX", "80"),
                ("HIPFIRE_SCHED_SPILLABLE_BATCH_MAX", "2"),
            ]),
        );
        assert_eq!(clamped.resident_state_max, 64);
        assert_eq!(clamped.spillable_batch_max, 64);
    }

    #[test]
    fn opportunistic_dispatch_waits_for_pairing_unless_clear() {
        assert!(!should_dispatch_opportunistic(OpportunisticDispatchInput {
            compatible_queued_tokens: 255,
            schedule_clear: false,
            target_pair_tokens: 256,
        }));
        assert!(should_dispatch_opportunistic(OpportunisticDispatchInput {
            compatible_queued_tokens: 256,
            schedule_clear: false,
            target_pair_tokens: 256,
        }));
        assert!(should_dispatch_opportunistic(OpportunisticDispatchInput {
            compatible_queued_tokens: 0,
            schedule_clear: true,
            target_pair_tokens: 256,
        }));
    }

    #[test]
    fn worker_keys_and_prefill_compatibility_match_session_policy() {
        let base = qwen_worker();
        let shuffled = ModelWorkerKey {
            feature_flags: vec!["qwen35".to_string(), "prefill_batch".to_string()],
            ..base.clone()
        };
        assert_eq!(model_worker_key_id(&base), model_worker_key_id(&shuffled));
        assert!(same_model_worker_key(&base, &shuffled));

        let a = session("a", 64, 3);
        let b = session_with(
            "b",
            64,
            2,
            shuffled,
            &["deltanet_recurrent", "attention_kv"],
            0,
        );
        let c = session_with(
            "c",
            64,
            1,
            nemotron_worker(),
            &["attention_kv", "mamba_ssm", "mamba_conv"],
            0,
        );
        let d = session_with(
            "d",
            64,
            1,
            nemotron_worker(),
            &["mamba_conv", "attention_kv", "mamba_ssm"],
            0,
        );
        let minimax = ModelWorkerKey {
            artifact_path: "/models/minimax-m2-mq4.hfq".to_string(),
            artifact_digest: Some("sha256:minimax".to_string()),
            arch_id: "10".to_string(),
            quant_family: "mq4".to_string(),
            state_mode: "attention_kv".to_string(),
            max_seq_bucket: 4096,
            accelerator_kind: None,
            device_id: None,
            feature_flags: vec!["prefill_batch".to_string(), "minimax".to_string()],
        };
        let e = session_with(
            "e",
            64,
            1,
            minimax.clone(),
            &["attention_kv", "backend_private"],
            0,
        );
        let f = session_with("f", 64, 1, minimax, &["backend_private", "attention_kv"], 0);
        assert!(sessions_compatible_for_prefill(&a, &b));
        assert!(!sessions_compatible_for_prefill(&a, &c));
        assert!(!sessions_compatible_for_prefill(&c, &d));
        assert!(!sessions_compatible_for_prefill(&e, &f));
    }

    #[test]
    fn worker_device_capability_uses_inventory_when_available() {
        let worker = qwen_worker();
        let inventory = AcceleratorInventory::from_devices(
            "daemon",
            vec![AcceleratorDeviceInfo::hip(
                "0",
                0,
                Some("gfx1201".to_string()),
                Some(24_000_000_000),
                Some(false),
                Some("HIP 6.4".to_string()),
            )],
        );

        let status = worker_device_capability_status(&worker, &inventory);
        assert!(status.supported);
        assert_eq!(status.capability, "supported");
        assert_eq!(status.reason, "worker_device_available");
    }

    #[test]
    fn worker_device_capability_preserves_unprobed_compatibility() {
        let status =
            worker_device_capability_status(&qwen_worker(), &AcceleratorInventory::not_probed());

        assert!(status.supported);
        assert_eq!(status.capability, "unknown");
        assert_eq!(status.reason, "accelerator_inventory_not_probed");
    }

    #[test]
    fn worker_device_capability_rejects_missing_or_unavailable_device() {
        let missing = worker_device_capability_status(
            &qwen_worker(),
            &AcceleratorInventory::from_devices(
                "daemon",
                vec![AcceleratorDeviceInfo::hip(
                    "1",
                    1,
                    Some("gfx1151".to_string()),
                    Some(96_000_000_000),
                    Some(true),
                    Some("HIP 7.2".to_string()),
                )],
            ),
        );
        assert!(!missing.supported);
        assert_eq!(missing.reason, "worker_device_not_found");

        let mut unavailable = AcceleratorDeviceInfo::hip(
            "0",
            0,
            Some("gfx1201".to_string()),
            Some(24_000_000_000),
            Some(false),
            Some("HIP 6.4".to_string()),
        );
        unavailable.available = false;
        unavailable.reason = Some("set_device failed".to_string());
        let status = worker_device_capability_status(
            &qwen_worker(),
            &AcceleratorInventory::from_devices("daemon", vec![unavailable]),
        );
        assert!(!status.supported);
        assert_eq!(status.reason, "worker_device_unavailable");
    }

    #[test]
    fn prefill_scheduler_rejects_sessions_for_unavailable_worker_devices() {
        let mut scheduler = PriorityPrefillScheduler::with_accelerator_inventory(
            SchedulerPolicyEnv::empty(),
            AcceleratorInventory::from_devices(
                "daemon",
                vec![AcceleratorDeviceInfo::hip(
                    "1",
                    1,
                    Some("gfx1151".to_string()),
                    Some(96_000_000_000),
                    Some(true),
                    Some("HIP 7.2".to_string()),
                )],
            ),
        );

        let err = scheduler.enqueue(session("a", 64, 3), 0).unwrap_err();
        assert!(err.contains("worker_device_not_found"));
        assert_eq!(scheduler.size(), 0);
    }

    #[test]
    fn prefill_scheduler_without_inventory_preserves_existing_admission() {
        let mut scheduler = PriorityPrefillScheduler::new(SchedulerPolicyEnv::empty());

        scheduler.enqueue(session("a", 64, 3), 0).unwrap();
        assert_eq!(scheduler.size(), 1);
    }

    #[test]
    fn prefill_scheduler_dispatches_priority_and_coalesces() {
        let mut scheduler = PriorityPrefillScheduler::new(env(&[
            ("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "3"),
            ("HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE", "5"),
        ]));
        scheduler
            .enqueue(session("interactive", 64, 16), 0)
            .unwrap();
        scheduler.enqueue(session("high", 1, 16), 0).unwrap();
        assert_eq!(
            ids(scheduler.next_prefill_batch(NextBatchInput { now_ms: 5 })),
            vec!["high"]
        );

        scheduler.enqueue(session("b", 64, 16), 2).unwrap();
        assert_eq!(
            ids(scheduler.next_prefill_batch(NextBatchInput { now_ms: 5 })),
            vec!["interactive", "b"]
        );
    }

    #[test]
    fn prefill_scheduler_respects_compatibility_spill_and_opportunistic_pairing() {
        let mut scheduler = PriorityPrefillScheduler::new(env(&[
            ("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "2"),
            ("HIPFIRE_SCHED_RESIDENT_STATE_MAX", "1"),
            ("HIPFIRE_SCHED_SPILLABLE_BATCH_MAX", "4"),
            ("HIPFIRE_SCHED_STATE_CACHE_DISK", "1"),
            ("HIPFIRE_SCHED_STATE_CACHE_DISK_MIN_PRIORITY", "128"),
            ("HIPFIRE_SCHED_PREFILL_WAIT_MS_BACKGROUND", "0"),
        ]));
        for id in ["a", "b", "c", "d"] {
            scheduler.enqueue(session(id, 128, 16), 0).unwrap();
        }
        let batch = scheduler
            .next_prefill_batch(NextBatchInput { now_ms: 0 })
            .unwrap();
        assert_eq!(
            batch
                .sessions
                .iter()
                .map(|s| s.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b", "c", "d"]
        );
        assert_eq!(batch.policy.resident_state_max, 1);
        assert_eq!(batch.policy.spillable_batch_max, 4);

        let mut blocked = PriorityPrefillScheduler::new(env(&[
            ("HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE", "5"),
            ("HIPFIRE_SCHED_OPPORTUNISTIC_MIN_PAIR_TOKENS", "32"),
        ]));
        blocked.enqueue(session("interactive", 64, 8), 0).unwrap();
        blocked
            .enqueue(
                session_with(
                    "op-a",
                    255,
                    64,
                    qwen_worker(),
                    &["attention_kv", "deltanet_recurrent"],
                    56,
                ),
                0,
            )
            .unwrap();
        assert!(blocked
            .next_prefill_batch(NextBatchInput { now_ms: 1 })
            .is_none());
        blocked
            .enqueue(
                session_with(
                    "op-b",
                    255,
                    64,
                    qwen_worker(),
                    &["attention_kv", "deltanet_recurrent"],
                    40,
                ),
                1,
            )
            .unwrap();
        assert_eq!(
            ids(blocked.next_prefill_batch(NextBatchInput { now_ms: 5 })),
            vec!["interactive"]
        );
        let paired = blocked
            .next_prefill_batch(NextBatchInput { now_ms: 5 })
            .unwrap();
        assert_eq!(
            paired
                .sessions
                .iter()
                .map(|s| s.id.as_str())
                .collect::<Vec<_>>(),
            vec!["op-a", "op-b"]
        );
        assert_eq!(paired.total_suffix_tokens, 32);
    }

    #[test]
    fn prefill_scheduler_cancel_aging_and_backpressure() {
        let mut scheduler = PriorityPrefillScheduler::new(env(&[
            ("HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE", "0"),
            ("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "2"),
        ]));
        scheduler.enqueue(session("a", 64, 16), 0).unwrap();
        let incoming = session("incoming", 64, 16);
        assert_eq!(
            ids(scheduler.next_prefill_batch(NextBatchInput { now_ms: 30 })),
            vec!["a"]
        );

        assert!(scheduler.enqueue_if_absent(incoming.clone(), 10).unwrap());
        assert!(!scheduler.enqueue_if_absent(incoming, 20).unwrap());
        assert_eq!(scheduler.size(), 1);
        assert!(scheduler.cancel("incoming"));
        assert_eq!(scheduler.size(), 0);

        let mut aged = PriorityPrefillScheduler::new(env(&[
            ("HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE", "1000"),
            ("HIPFIRE_SCHED_DEADLINE_AGING_MS", "50"),
        ]));
        aged.enqueue(session("high-waiting", 32, 16), 100).unwrap();
        aged.enqueue(session("aged-low", 128, 16), 0).unwrap();
        assert_eq!(
            ids(aged.next_prefill_batch(NextBatchInput { now_ms: 60 })),
            vec!["aged-low"]
        );

        let mut capped =
            PriorityPrefillScheduler::new(env(&[("HIPFIRE_SCHED_PREFILL_MAX_QUEUED", "1")]));
        capped.enqueue(session("first", 64, 16), 0).unwrap();
        assert!(capped
            .enqueue(session("second", 64, 16), 0)
            .unwrap_err()
            .contains("backpressure"));
    }

    #[test]
    fn residency_planner_selects_modules_when_full_does_not_fit_auto() {
        let budget = ResourceBudget {
            system_memory_budget_bytes: 0,
            system_memory_headroom_bytes: 0,
            vram_budget_bytes: 1_000,
            vram_headroom_bytes: 100,
        };
        let request = ModelResidencyRequest {
            worker_key_id: "worker-new".to_string(),
            model_path: "qwen.hfq".to_string(),
            requested_mode: ResidencyMode::Auto,
            estimated_full: ResourceUsage {
                system_memory_bytes: 0,
                vram_bytes: 1_200,
            },
            estimated_qwen_moe_modules: Some(ResourceUsage {
                system_memory_bytes: 0,
                vram_bytes: 700,
            }),
        };

        let plan = plan_model_residency(budget, request, &[]).unwrap();
        assert_eq!(plan.residency_mode, ResidencyMode::QwenMoeModules);
        assert_eq!(plan.module_vram_budget_bytes, Some(700));
    }

    #[test]
    fn residency_planner_evicts_oldest_workers_for_budget() {
        let budget = ResourceBudget {
            system_memory_budget_bytes: 0,
            system_memory_headroom_bytes: 0,
            vram_budget_bytes: 1_000,
            vram_headroom_bytes: 0,
        };
        let resident_workers = vec![
            ResidentWorkerLedgerEntry {
                worker_key_id: "old".to_string(),
                model_path: "old.hfq".to_string(),
                residency_mode: ResidencyMode::Full,
                resource_usage: ResourceUsage {
                    system_memory_bytes: 0,
                    vram_bytes: 500,
                },
                last_used_seq: 1,
            },
            ResidentWorkerLedgerEntry {
                worker_key_id: "newer".to_string(),
                model_path: "newer.hfq".to_string(),
                residency_mode: ResidencyMode::Full,
                resource_usage: ResourceUsage {
                    system_memory_bytes: 0,
                    vram_bytes: 300,
                },
                last_used_seq: 2,
            },
        ];
        let request = ModelResidencyRequest {
            worker_key_id: "incoming".to_string(),
            model_path: "incoming.hfq".to_string(),
            requested_mode: ResidencyMode::Full,
            estimated_full: ResourceUsage {
                system_memory_bytes: 0,
                vram_bytes: 600,
            },
            estimated_qwen_moe_modules: None,
        };

        let plan = plan_model_residency(budget, request, &resident_workers).unwrap();
        assert_eq!(plan.unload_worker_key_ids, vec!["old"]);
    }

    #[test]
    fn prefill_scheduler_round_robins_users_and_microbatches_across_them() {
        let mut fair =
            PriorityPrefillScheduler::new(env(&[("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "1")]));
        fair.enqueue(owned_session("a1", "alice", "ta", 0), 0)
            .unwrap();
        fair.enqueue(owned_session("a2", "alice", "ta", 0), 1)
            .unwrap();
        fair.enqueue(owned_session("b1", "bob", "tb", 0), 2)
            .unwrap();
        let first = fair
            .next_prefill_batch(NextBatchInput { now_ms: 10 })
            .unwrap();
        let second = fair
            .next_prefill_batch(NextBatchInput { now_ms: 11 })
            .unwrap();
        assert_eq!(first.sessions[0].owner.user_id.as_deref(), Some("alice"));
        assert_eq!(second.sessions[0].owner.user_id.as_deref(), Some("bob"));

        let mut batched =
            PriorityPrefillScheduler::new(env(&[("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "2")]));
        batched
            .enqueue(owned_session("a", "alice", "ta", 64), 0)
            .unwrap();
        batched
            .enqueue(owned_session("b", "bob", "tb", 64), 0)
            .unwrap();
        let batch = batched
            .next_prefill_batch(NextBatchInput { now_ms: 10 })
            .unwrap();
        assert_eq!(batch.sessions.len(), 2);
        assert_ne!(
            batch.sessions[0].owner.user_id,
            batch.sessions[1].owner.user_id
        );
    }

    #[test]
    fn prefill_scheduler_cancels_pending_credential_owners() {
        let mut scheduler = PriorityPrefillScheduler::default();
        scheduler
            .enqueue(owned_session("a1", "alice", "ta", 64), 0)
            .unwrap();
        scheduler
            .enqueue(owned_session("a2", "alice", "other", 64), 0)
            .unwrap();
        scheduler
            .enqueue(owned_session("b", "bob", "tb", 64), 0)
            .unwrap();
        assert_eq!(scheduler.cancel_by_token("ta").len(), 1);
        assert_eq!(scheduler.cancel_by_user("alice").len(), 1);
        assert_eq!(scheduler.size(), 1);
    }

    fn continuous_capacity() -> WorkloadResources {
        WorkloadResources {
            system_memory_bytes: 64_000,
            vram_bytes: 24_000,
            gpu_slots: 4,
            npu_slots: 1,
            cpu_threads: 16,
        }
    }

    fn token_workload(id: &str, priority: u8, enqueued_at_ms: u64) -> WorkloadSpec {
        WorkloadSpec::microbatchable(
            id,
            WorkloadClass::TokenDecode,
            priority,
            enqueued_at_ms,
            WorkloadResources {
                vram_bytes: 1_000,
                gpu_slots: 1,
                ..WorkloadResources::default()
            },
            "worker:qwen|state:q8+deltanet|decode",
            4,
        )
    }

    fn owned_token_workload(
        id: &str,
        user: &str,
        token: &str,
        priority: u8,
        enqueued_at_ms: u64,
    ) -> WorkloadSpec {
        token_workload(id, priority, enqueued_at_ms)
            .with_owner(WorkloadOwner::authenticated(user, Some(token.to_string())))
    }

    #[test]
    fn workload_class_billing_class_rolls_up_to_auth_taxonomy() {
        use hipfire_auth::WorkloadClass as Auth;
        assert_eq!(WorkloadClass::TokenPrefill.billing_class(), Auth::Text);
        assert_eq!(WorkloadClass::TokenDecode.billing_class(), Auth::Text);
        assert_eq!(WorkloadClass::ImageGeneration.billing_class(), Auth::Image);
        assert_eq!(WorkloadClass::Training.billing_class(), Auth::Training);
        assert_eq!(WorkloadClass::Maintenance.billing_class(), Auth::Other);
    }

    #[test]
    fn peek_next_priority_reports_most_urgent_queued_without_dequeuing() {
        let mut scheduler = ContinuousWorkScheduler::new(continuous_capacity(), 32, 0);
        assert_eq!(scheduler.peek_next_priority(0), None);
        scheduler.enqueue(token_workload("low", 128, 0)).unwrap();
        assert_eq!(scheduler.peek_next_priority(1), Some(128));
        // A more urgent (lower-number) workload takes over the peek.
        scheduler.enqueue(token_workload("high", 8, 1)).unwrap();
        assert_eq!(scheduler.peek_next_priority(2), Some(8));
        // Peek is non-destructive: the queue is untouched.
        assert_eq!(scheduler.snapshot().queued, 2);
    }

    #[test]
    fn continuous_scheduler_microbatches_compatible_token_work() {
        let mut scheduler = ContinuousWorkScheduler::new(continuous_capacity(), 32, 0);
        scheduler.enqueue(token_workload("a", 64, 0)).unwrap();
        scheduler.enqueue(token_workload("b", 64, 1)).unwrap();
        let mut incompatible = token_workload("c", 64, 2);
        incompatible.microbatch_key = Some("worker:other|decode".to_string());
        scheduler.enqueue(incompatible).unwrap();
        let mut singleton_limit = token_workload("d", 64, 3);
        singleton_limit.max_microbatch_size = 1;
        scheduler.enqueue(singleton_limit).unwrap();

        let lease = scheduler.next_batch(10).unwrap();

        assert_eq!(
            lease
                .workloads
                .iter()
                .map(|workload| workload.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        assert_eq!(lease.resources.gpu_slots, 2);
        assert_eq!(scheduler.snapshot().queued, 2);
        scheduler.complete(lease.lease_id).unwrap();
        assert_eq!(scheduler.snapshot().active_batches, 0);
    }

    #[test]
    fn continuous_scheduler_microbatches_compatible_image_work() {
        let mut scheduler = ContinuousWorkScheduler::new(continuous_capacity(), 32, 0);
        for id in ["image-a", "image-b"] {
            scheduler
                .enqueue(WorkloadSpec::microbatchable(
                    id,
                    WorkloadClass::ImageGeneration,
                    128,
                    0,
                    WorkloadResources {
                        vram_bytes: 4_000,
                        gpu_slots: 1,
                        ..WorkloadResources::default()
                    },
                    "krea2|1024x1024|flow_match|20|bf16",
                    2,
                ))
                .unwrap();
        }

        let lease = scheduler.next_batch(0).unwrap();

        assert_eq!(lease.class, WorkloadClass::ImageGeneration);
        assert_eq!(lease.workloads.len(), 2);
    }

    #[test]
    fn continuous_scheduler_drains_before_exclusive_training() {
        let mut scheduler = ContinuousWorkScheduler::new(continuous_capacity(), 32, 0);
        scheduler.enqueue(token_workload("decode", 0, 0)).unwrap();
        let decode = scheduler.next_batch(0).unwrap();
        scheduler
            .enqueue(WorkloadSpec::singleton(
                "train",
                WorkloadClass::Training,
                1,
                1,
                WorkloadResources {
                    vram_bytes: 20_000,
                    gpu_slots: 4,
                    cpu_threads: 8,
                    ..WorkloadResources::default()
                },
            ))
            .unwrap();

        assert!(scheduler.next_batch(2).is_none());
        scheduler.complete(decode.lease_id).unwrap();
        let training = scheduler.next_batch(3).unwrap();
        assert_eq!(training.class, WorkloadClass::Training);
        assert!(scheduler.snapshot().exclusive_active);
        assert!(scheduler.next_batch(4).is_none());
    }

    #[test]
    fn continuous_scheduler_ages_waiting_background_work() {
        let mut scheduler = ContinuousWorkScheduler::new(continuous_capacity(), 32, 100);
        scheduler
            .enqueue(token_workload("background", 192, 0))
            .unwrap();
        scheduler
            .enqueue(token_workload("interactive", 64, 150))
            .unwrap();

        let lease = scheduler.next_batch(200).unwrap();

        assert_eq!(lease.workloads[0].id, "background");
    }

    #[test]
    fn continuous_scheduler_rejects_duplicate_and_over_capacity_work() {
        let mut scheduler = ContinuousWorkScheduler::new(continuous_capacity(), 1, 0);
        scheduler.enqueue(token_workload("a", 64, 0)).unwrap();
        assert!(scheduler.enqueue(token_workload("a", 64, 0)).is_err());
        assert!(scheduler.enqueue(token_workload("b", 64, 0)).is_err());

        let mut scheduler = ContinuousWorkScheduler::new(continuous_capacity(), 0, 0);
        let oversized = WorkloadSpec::singleton(
            "oversized",
            WorkloadClass::ImageGeneration,
            64,
            0,
            WorkloadResources {
                vram_bytes: 25_000,
                gpu_slots: 1,
                ..WorkloadResources::default()
            },
        );
        assert!(scheduler.enqueue(oversized).is_err());
    }

    #[test]
    fn continuous_scheduler_round_robins_users_within_priority() {
        let mut scheduler = ContinuousWorkScheduler::new(continuous_capacity(), 32, 0);
        for workload in [
            owned_token_workload("a1", "alice", "ta", 64, 0),
            owned_token_workload("a2", "alice", "ta", 64, 1),
            owned_token_workload("b1", "bob", "tb", 64, 2),
        ] {
            let mut workload = workload;
            workload.max_microbatch_size = 1;
            scheduler.enqueue(workload).unwrap();
        }
        let first = scheduler.next_batch(10).unwrap();
        scheduler.complete(first.lease_id).unwrap();
        let second = scheduler.next_batch(11).unwrap();
        assert_eq!(first.workloads[0].owner.user_id.as_deref(), Some("alice"));
        assert_eq!(second.workloads[0].owner.user_id.as_deref(), Some("bob"));
    }

    #[test]
    fn continuous_scheduler_microbatches_across_users_and_cancels_by_owner() {
        let mut scheduler = ContinuousWorkScheduler::new(continuous_capacity(), 32, 0);
        scheduler
            .enqueue(owned_token_workload("a", "alice", "ta", 64, 0))
            .unwrap();
        scheduler
            .enqueue(owned_token_workload("b", "bob", "tb", 64, 1))
            .unwrap();
        let lease = scheduler.next_batch(10).unwrap();
        assert_eq!(lease.workloads.len(), 2);
        assert_ne!(
            lease.workloads[0].owner.user_id,
            lease.workloads[1].owner.user_id
        );
        scheduler.complete(lease.lease_id).unwrap();

        scheduler
            .enqueue(owned_token_workload("a2", "alice", "ta", 64, 2))
            .unwrap();
        scheduler
            .enqueue(owned_token_workload("a3", "alice", "other", 64, 3))
            .unwrap();
        scheduler
            .enqueue(owned_token_workload("b2", "bob", "tb", 64, 4))
            .unwrap();
        assert_eq!(scheduler.cancel_pending_by_token("ta").len(), 1);
        assert_eq!(scheduler.cancel_pending_by_user("alice").len(), 1);
        assert_eq!(scheduler.snapshot().queued, 1);
    }
}
