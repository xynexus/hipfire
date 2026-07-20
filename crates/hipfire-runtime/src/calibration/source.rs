// SPDX-License-Identifier: Apache-2.0
// hipfire — source planning and logical tensor-read accounting.

use super::contracts::CalibError;
use crate::quant::{f16_to_f32, f32_to_f16};
use crate::weights::WeightTensor;
use hipfire_model::{ModelSource, TensorInfo, TensorStorageLocation};
use hipfire_rdna::{DType, Gpu, GpuTensor};
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::thread::{self, JoinHandle};
use std::time::Instant;

#[cfg(unix)]
use std::os::fd::AsRawFd;

pub const LAYER_PREFETCH_WORKER_CHUNK_BYTES: usize = 8 * 1024 * 1024;
pub const SOURCE_UPLOAD_CHUNK_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TensorOwner {
    Persistent,
    Layer(usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorLoadRequest {
    pub logical_name: String,
    pub source_name: String,
    pub owner: TensorOwner,
    /// Another logical name whose already-loaded bytes are intentionally reused.
    pub alias_of: Option<String>,
}

impl TensorLoadRequest {
    pub fn tensor(
        logical_name: impl Into<String>,
        source_name: impl Into<String>,
        owner: TensorOwner,
    ) -> Self {
        Self {
            logical_name: logical_name.into(),
            source_name: source_name.into(),
            owner,
            alias_of: None,
        }
    }

    pub fn alias(
        logical_name: impl Into<String>,
        source_name: impl Into<String>,
        owner: TensorOwner,
        alias_of: impl Into<String>,
    ) -> Self {
        Self {
            logical_name: logical_name.into(),
            source_name: source_name.into(),
            owner,
            alias_of: Some(alias_of.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorLoadEntry {
    pub logical_name: String,
    pub source_name: String,
    pub owner: TensorOwner,
    pub alias_of: Option<String>,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub storage: TensorStorageLocation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorLoadPlan {
    entries: Vec<TensorLoadEntry>,
    unique_source_bytes: u64,
    bytes_by_owner: BTreeMap<TensorOwner, u64>,
}

impl TensorLoadPlan {
    pub fn build(
        source: &dyn ModelSource,
        requests: impl IntoIterator<Item = TensorLoadRequest>,
    ) -> Result<Self, CalibError> {
        let mut requests: Vec<_> = requests.into_iter().collect();
        requests.sort_by(|a, b| {
            a.alias_of
                .is_some()
                .cmp(&b.alias_of.is_some())
                .then_with(|| a.owner.cmp(&b.owner))
                .then_with(|| a.logical_name.cmp(&b.logical_name))
        });

        let mut logical_names = BTreeSet::new();
        for request in &requests {
            if request.logical_name.is_empty() || request.source_name.is_empty() {
                return Err(CalibError::InvalidSourcePlan(
                    "logical and source tensor names must not be empty".into(),
                ));
            }
            if !logical_names.insert(request.logical_name.clone()) {
                return Err(CalibError::InvalidSourcePlan(format!(
                    "duplicate logical tensor {}",
                    request.logical_name
                )));
            }
        }

        let request_by_name: BTreeMap<_, _> = requests
            .iter()
            .map(|request| (request.logical_name.as_str(), request))
            .collect();
        for request in &requests {
            if let Some(alias) = &request.alias_of {
                let canonical = request_by_name.get(alias.as_str()).ok_or_else(|| {
                    CalibError::InvalidSourcePlan(format!(
                        "alias {} targets missing logical tensor {alias}",
                        request.logical_name
                    ))
                })?;
                if canonical.alias_of.is_some() {
                    return Err(CalibError::InvalidSourcePlan(format!(
                        "alias chain {} -> {alias} is not allowed",
                        request.logical_name
                    )));
                }
                if canonical.source_name != request.source_name {
                    return Err(CalibError::InvalidSourcePlan(format!(
                        "alias {} source {} differs from canonical {} source {}",
                        request.logical_name, request.source_name, alias, canonical.source_name
                    )));
                }
            }
        }

        let mut entries = Vec::with_capacity(requests.len());
        let mut physical_owners: BTreeMap<(String, u64, u64), (String, TensorOwner)> =
            BTreeMap::new();
        let mut unique_source_bytes = 0u64;
        let mut bytes_by_owner = BTreeMap::new();
        for request in requests {
            let info = source.tensor_info(&request.source_name).ok_or_else(|| {
                CalibError::InvalidSourcePlan(format!(
                    "source tensor {} for {} is missing",
                    request.source_name, request.logical_name
                ))
            })?;
            let storage = source.tensor_storage(&request.source_name).ok_or_else(|| {
                CalibError::InvalidSourcePlan(format!(
                    "source tensor {} has no physical storage location",
                    request.source_name
                ))
            })?;
            if storage.byte_len != info.data_size as u64 {
                return Err(CalibError::InvalidSourcePlan(format!(
                    "source tensor {} metadata is {} bytes but storage is {} bytes",
                    request.source_name, info.data_size, storage.byte_len
                )));
            }
            let key = physical_key(&storage);
            if let Some((first_logical, _)) = physical_owners.get(&key) {
                if request.alias_of.as_deref() != Some(first_logical.as_str()) {
                    return Err(CalibError::InvalidSourcePlan(format!(
                        "logical tensors {first_logical} and {} share storage without an explicit alias",
                        request.logical_name
                    )));
                }
            } else {
                if request.alias_of.is_some() {
                    return Err(CalibError::InvalidSourcePlan(format!(
                        "alias {} was ordered before its canonical storage owner",
                        request.logical_name
                    )));
                }
                physical_owners.insert(key, (request.logical_name.clone(), request.owner));
                unique_source_bytes = unique_source_bytes
                    .checked_add(storage.byte_len)
                    .ok_or_else(|| {
                        CalibError::InvalidSourcePlan("source byte total overflow".into())
                    })?;
                *bytes_by_owner.entry(request.owner).or_insert(0) += storage.byte_len;
            }
            entries.push(TensorLoadEntry {
                logical_name: request.logical_name,
                source_name: request.source_name,
                owner: request.owner,
                alias_of: request.alias_of,
                dtype: info.dtype.clone(),
                shape: info.shape.clone(),
                storage,
            });
        }
        Ok(Self {
            entries,
            unique_source_bytes,
            bytes_by_owner,
        })
    }

    pub fn entries(&self) -> &[TensorLoadEntry] {
        &self.entries
    }

    pub const fn unique_source_bytes(&self) -> u64 {
        self.unique_source_bytes
    }

    pub fn bytes_for(&self, owner: TensorOwner) -> u64 {
        self.bytes_by_owner.get(&owner).copied().unwrap_or(0)
    }

    pub fn entry(&self, logical_name: &str) -> Option<&TensorLoadEntry> {
        self.entries
            .iter()
            .find(|entry| entry.logical_name == logical_name)
    }

    /// Return complete physical source ranges for one owner without consuming
    /// the read ledger. Ranges are sorted into backing-file order. A tensor that
    /// does not fit in the remaining lookahead budget is not staged because a
    /// partial range cannot satisfy a direct tensor view and would only consume
    /// host memory before the source mmap fallback runs.
    pub fn prefetch_ranges_for(
        &self,
        owner: TensorOwner,
        byte_budget: u64,
    ) -> Vec<TensorStorageLocation> {
        if byte_budget == 0 {
            return Vec::new();
        }
        let mut ranges = self
            .entries
            .iter()
            .filter(|entry| entry.owner == owner && entry.alias_of.is_none())
            .map(|entry| entry.storage.clone())
            .collect::<Vec<_>>();
        ranges.sort_by(|left, right| {
            left.path
                .cmp(&right.path)
                .then_with(|| left.byte_offset.cmp(&right.byte_offset))
                .then_with(|| left.byte_len.cmp(&right.byte_len))
        });
        ranges.dedup();

        let mut remaining = byte_budget;
        let mut bounded = Vec::new();
        for range in ranges {
            if remaining == 0 {
                break;
            }
            if range.byte_len > remaining {
                continue;
            }
            remaining -= range.byte_len;
            bounded.push(range);
        }
        bounded
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedSourceRange {
    pub location: TensorStorageLocation,
    pub bytes: Vec<u8>,
}

/// Bounded resident bytes produced by the lookahead worker. A tensor view is
/// served only when one staged range covers its complete physical payload;
/// clipped/failed ranges safely fall back to the source mmap.
#[derive(Debug, Default)]
pub struct LayerStagingBuffer {
    ranges: Vec<StagedSourceRange>,
    byte_len: u64,
}

impl LayerStagingBuffer {
    pub fn from_ranges(ranges: Vec<StagedSourceRange>) -> Self {
        let byte_len = ranges.iter().fold(0u64, |total, range| {
            total.saturating_add(u64::try_from(range.bytes.len()).unwrap_or(u64::MAX))
        });
        Self { ranges, byte_len }
    }

    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub fn is_empty(&self) -> bool {
        self.byte_len == 0
    }

    pub fn view(&self, tensor: &TensorStorageLocation) -> Option<&[u8]> {
        self.ranges.iter().find_map(|range| {
            if range.location.path != tensor.path || tensor.byte_offset < range.location.byte_offset
            {
                return None;
            }
            let start = usize::try_from(tensor.byte_offset - range.location.byte_offset).ok()?;
            let len = usize::try_from(tensor.byte_len).ok()?;
            let end = start.checked_add(len)?;
            range.bytes.get(start..end)
        })
    }
}

#[derive(Debug, Default)]
pub struct LayerPrefetchReport {
    pub requested_bytes: u64,
    pub completed_bytes: u64,
    pub ranges: usize,
    pub elapsed_us: u64,
    pub errors: Vec<String>,
    pub staging: LayerStagingBuffer,
}

/// A bounded background read that retains the next layer in resident host
/// memory. The worker reads through one fixed-size scratch chunk, while the
/// completed staging buffer stays within the engine's configured byte budget
/// and never touches tensor semantics or the logical read ledger.
pub struct LayerPrefetch {
    requested_bytes: u64,
    ranges: usize,
    handle: Option<JoinHandle<LayerPrefetchReport>>,
}

impl LayerPrefetch {
    pub fn spawn(ranges: Vec<TensorStorageLocation>) -> Result<Self, CalibError> {
        let requested_bytes = ranges
            .iter()
            .fold(0u64, |total, range| total.saturating_add(range.byte_len));
        let range_count = ranges.len();
        let handle = thread::Builder::new()
            .name("hipfire-layer-prefetch".into())
            .spawn(move || prefetch_ranges(ranges))
            .map_err(|error| {
                CalibError::Runtime(format!("failed to spawn layer-prefetch worker: {error}"))
            })?;
        Ok(Self {
            requested_bytes,
            ranges: range_count,
            handle: Some(handle),
        })
    }

    pub fn wait(mut self) -> LayerPrefetchReport {
        self.join()
    }

    fn join(&mut self) -> LayerPrefetchReport {
        let Some(handle) = self.handle.take() else {
            return LayerPrefetchReport {
                requested_bytes: self.requested_bytes,
                ranges: self.ranges,
                errors: vec!["layer-prefetch worker was already joined".into()],
                ..LayerPrefetchReport::default()
            };
        };
        handle.join().unwrap_or_else(|_| LayerPrefetchReport {
            requested_bytes: self.requested_bytes,
            ranges: self.ranges,
            errors: vec!["layer-prefetch worker panicked".into()],
            ..LayerPrefetchReport::default()
        })
    }
}

impl Drop for LayerPrefetch {
    fn drop(&mut self) {
        if self.handle.is_some() {
            let _ = self.join();
        }
    }
}

fn prefetch_ranges(ranges: Vec<TensorStorageLocation>) -> LayerPrefetchReport {
    let started = Instant::now();
    let requested_bytes = ranges
        .iter()
        .fold(0u64, |total, range| total.saturating_add(range.byte_len));
    let range_count = ranges.len();
    let mut completed_bytes = 0u64;
    let mut errors = Vec::new();
    let mut scratch = vec![0u8; LAYER_PREFETCH_WORKER_CHUNK_BYTES];
    let mut staged_ranges = Vec::with_capacity(range_count);
    for range in ranges {
        let mut file = match File::open(&range.path) {
            Ok(file) => file,
            Err(error) => {
                errors.push(format!("{}: {error}", range.path.display()));
                continue;
            }
        };
        if let Err(error) = file.seek(SeekFrom::Start(range.byte_offset)) {
            errors.push(format!(
                "{}@{}: {error}",
                range.path.display(),
                range.byte_offset
            ));
            continue;
        }
        let requested_len = match usize::try_from(range.byte_len) {
            Ok(len) => len,
            Err(_) => {
                errors.push(format!(
                    "{}@{}: range length {} exceeds host address space",
                    range.path.display(),
                    range.byte_offset,
                    range.byte_len
                ));
                continue;
            }
        };
        let mut bytes = Vec::new();
        if let Err(error) = bytes.try_reserve_exact(requested_len) {
            errors.push(format!(
                "{}@{}: could not reserve {} staged bytes: {error}",
                range.path.display(),
                range.byte_offset,
                range.byte_len
            ));
            continue;
        }
        let mut remaining = range.byte_len;
        while remaining > 0 {
            let chunk = usize::try_from(remaining.min(scratch.len() as u64)).unwrap();
            match file.read(&mut scratch[..chunk]) {
                Ok(0) => {
                    errors.push(format!(
                        "{}@{}: unexpected EOF with {remaining} bytes remaining",
                        range.path.display(),
                        range.byte_offset
                    ));
                    break;
                }
                Ok(read) => {
                    bytes.extend_from_slice(&scratch[..read]);
                    completed_bytes = completed_bytes.saturating_add(read as u64);
                    remaining -= read as u64;
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    errors.push(format!(
                        "{}@{}: {error}",
                        range.path.display(),
                        range.byte_offset
                    ));
                    break;
                }
            }
        }
        let staged_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        if staged_len > 0 {
            staged_ranges.push(StagedSourceRange {
                location: TensorStorageLocation {
                    path: range.path.clone(),
                    byte_offset: range.byte_offset,
                    byte_len: staged_len,
                },
                bytes,
            });
        }
        #[cfg(unix)]
        unsafe {
            // The anonymous staging bytes are now authoritative for this
            // lookahead. Avoid retaining a second copy in the page cache.
            libc::posix_fadvise(
                file.as_raw_fd(),
                range.byte_offset as libc::off_t,
                staged_len as libc::off_t,
                libc::POSIX_FADV_DONTNEED,
            );
        }
    }
    LayerPrefetchReport {
        requested_bytes,
        completed_bytes,
        ranges: range_count,
        elapsed_us: u64::try_from(started.elapsed().as_micros())
            .unwrap_or(u64::MAX)
            .max(1),
        errors,
        staging: LayerStagingBuffer::from_ranges(staged_ranges),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadAction {
    Read(TensorStorageLocation),
    Reuse { canonical_logical: String },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadLedgerSnapshot {
    #[serde(default)]
    pub planned_logical: BTreeSet<String>,
    pub consumed_logical: BTreeSet<String>,
    pub read_canonical: BTreeSet<String>,
    pub logical_bytes_read: u64,
    #[serde(default)]
    pub duplicate_logical: BTreeSet<String>,
    #[serde(default)]
    pub missing_logical: BTreeSet<String>,
}

pub struct ReadLedger<'a> {
    plan: &'a TensorLoadPlan,
    snapshot: ReadLedgerSnapshot,
}

impl<'a> ReadLedger<'a> {
    pub fn new(plan: &'a TensorLoadPlan) -> Self {
        Self {
            plan,
            snapshot: ReadLedgerSnapshot {
                planned_logical: plan
                    .entries
                    .iter()
                    .map(|entry| entry.logical_name.clone())
                    .collect(),
                consumed_logical: BTreeSet::new(),
                read_canonical: BTreeSet::new(),
                logical_bytes_read: 0,
                duplicate_logical: BTreeSet::new(),
                missing_logical: BTreeSet::new(),
            },
        }
    }

    pub fn resume(
        plan: &'a TensorLoadPlan,
        snapshot: ReadLedgerSnapshot,
    ) -> Result<Self, CalibError> {
        let mut snapshot = snapshot;
        let planned = plan
            .entries
            .iter()
            .map(|entry| entry.logical_name.clone())
            .collect::<BTreeSet<_>>();
        if snapshot.planned_logical.is_empty() {
            snapshot.planned_logical = planned;
        }
        let ledger = Self { plan, snapshot };
        ledger.validate_snapshot()?;
        Ok(ledger)
    }

    pub fn consume(&mut self, logical_name: &str) -> Result<ReadAction, CalibError> {
        let entry = self.plan.entry(logical_name).ok_or_else(|| {
            CalibError::ReadLedger(format!("unplanned logical tensor {logical_name}"))
        })?;
        if !self
            .snapshot
            .consumed_logical
            .insert(logical_name.to_string())
        {
            self.snapshot
                .duplicate_logical
                .insert(logical_name.to_string());
            return Err(CalibError::ReadLedger(format!(
                "logical tensor {logical_name} was consumed more than once"
            )));
        }
        if let Some(canonical) = &entry.alias_of {
            if !self.snapshot.consumed_logical.contains(canonical) {
                self.snapshot.consumed_logical.remove(logical_name);
                return Err(CalibError::ReadLedger(format!(
                    "alias {logical_name} was consumed before canonical tensor {canonical}"
                )));
            }
            return Ok(ReadAction::Reuse {
                canonical_logical: canonical.clone(),
            });
        }
        self.snapshot
            .read_canonical
            .insert(logical_name.to_string());
        self.snapshot.logical_bytes_read = self
            .snapshot
            .logical_bytes_read
            .checked_add(entry.storage.byte_len)
            .ok_or_else(|| CalibError::ReadLedger("logical byte count overflow".into()))?;
        Ok(ReadAction::Read(entry.storage.clone()))
    }

    pub fn assert_complete(&self) -> Result<(), CalibError> {
        let snapshot = self.snapshot();
        let missing = snapshot.missing_logical.iter().cloned().collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(CalibError::ReadLedger(format!(
                "planned tensors were not consumed: {}",
                missing.join(", ")
            )));
        }
        if self.snapshot.logical_bytes_read != self.plan.unique_source_bytes {
            return Err(CalibError::ReadLedger(format!(
                "read {} unique bytes but plan requires {}",
                self.snapshot.logical_bytes_read, self.plan.unique_source_bytes
            )));
        }
        Ok(())
    }

    pub fn snapshot(&self) -> ReadLedgerSnapshot {
        let mut snapshot = self.snapshot.clone();
        snapshot.missing_logical = snapshot
            .planned_logical
            .difference(&snapshot.consumed_logical)
            .cloned()
            .collect();
        snapshot
    }

    fn validate_snapshot(&self) -> Result<(), CalibError> {
        let planned = self
            .plan
            .entries
            .iter()
            .map(|entry| entry.logical_name.clone())
            .collect::<BTreeSet<_>>();
        if self.snapshot.planned_logical != planned {
            return Err(CalibError::ReadLedger(
                "checkpoint planned tensor set does not match the current source plan".into(),
            ));
        }
        if !self.snapshot.duplicate_logical.is_empty() {
            return Err(CalibError::ReadLedger(
                "checkpoint records duplicate logical tensor reads".into(),
            ));
        }
        for logical in &self.snapshot.consumed_logical {
            if self.plan.entry(logical).is_none() {
                return Err(CalibError::ReadLedger(format!(
                    "checkpoint contains unplanned logical tensor {logical}"
                )));
            }
        }
        let expected: u64 = self
            .snapshot
            .read_canonical
            .iter()
            .map(|name| {
                self.plan
                    .entry(name)
                    .map(|entry| entry.storage.byte_len)
                    .ok_or_else(|| {
                        CalibError::ReadLedger(format!(
                            "checkpoint contains unplanned canonical tensor {name}"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum();
        if expected != self.snapshot.logical_bytes_read {
            return Err(CalibError::ReadLedger(format!(
                "checkpoint byte count {} does not match canonical reads {expected}",
                self.snapshot.logical_bytes_read
            )));
        }
        Ok(())
    }
}

fn physical_key(location: &TensorStorageLocation) -> (String, u64, u64) {
    (
        location.path.to_string_lossy().into_owned(),
        location.byte_offset,
        location.byte_len,
    )
}

pub struct PlannedTensorView<'a> {
    pub info: &'a TensorInfo,
    pub bytes: &'a [u8],
    pub action: ReadAction,
    pub from_prefetch: bool,
    source: Option<&'a dyn ModelSource>,
    source_name: String,
    release_pages: bool,
    released_prefix: Cell<usize>,
}

impl PlannedTensorView<'_> {
    fn release_copied_range(&self, byte_offset: usize, byte_len: usize) -> bool {
        if !self.release_pages || byte_len == 0 || byte_offset != self.released_prefix.get() {
            return false;
        }
        let byte_len = byte_len.min(self.bytes.len().saturating_sub(byte_offset));
        let Some(source) = self.source else {
            return false;
        };
        if source.release_tensor_range_pages(&self.source_name, byte_offset, byte_len) {
            self.released_prefix.set(byte_offset + byte_len);
            true
        } else {
            false
        }
    }
}

/// Aggregate source-materialization phases for one owner-scoped load. The
/// layer-stream engine persists this beside its wall-clock `load_upload_us` so
/// a warm-source slowdown can be attributed before changing the buffering
/// strategy. Times are intentionally inclusive of successful tensors only;
/// failed loads abort the layer and never produce a checkpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SourceLoadTimings {
    pub tensor_count: u64,
    pub source_bytes: u64,
    pub gpu_upload_bytes: u64,
    pub staged_tensor_count: u64,
    pub staged_source_bytes: u64,
    /// Tensor lookup, logical-ledger consumption, and view construction.
    pub view_us: u64,
    /// Host dtype conversion or adjustment before upload.
    pub decode_us: u64,
    /// HIP allocation and synchronous host-to-device copy. Any mmap refaults
    /// incurred while HIP reads the source slice are included here; interleaved
    /// range-release advice is subtracted into `release_us`.
    pub upload_us: u64,
    /// Mapping/page-cache release after each completed upload chunk and at the
    /// end of the tensor view.
    pub release_us: u64,
}

impl SourceLoadTimings {
    fn record(
        &mut self,
        source_bytes: usize,
        gpu_upload_bytes: usize,
        from_prefetch: bool,
        view_us: u64,
        decode_us: u64,
        upload_us: u64,
        release_us: u64,
    ) {
        self.tensor_count = self.tensor_count.saturating_add(1);
        self.source_bytes = self
            .source_bytes
            .saturating_add(u64::try_from(source_bytes).unwrap_or(u64::MAX));
        self.gpu_upload_bytes = self
            .gpu_upload_bytes
            .saturating_add(u64::try_from(gpu_upload_bytes).unwrap_or(u64::MAX));
        if from_prefetch {
            self.staged_tensor_count = self.staged_tensor_count.saturating_add(1);
            self.staged_source_bytes = self
                .staged_source_bytes
                .saturating_add(u64::try_from(source_bytes).unwrap_or(u64::MAX));
        }
        self.view_us = self.view_us.saturating_add(view_us);
        self.decode_us = self.decode_us.saturating_add(decode_us);
        self.upload_us = self.upload_us.saturating_add(upload_us);
        self.release_us = self.release_us.saturating_add(release_us);
    }
}

fn elapsed_micros(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

impl Drop for PlannedTensorView<'_> {
    fn drop(&mut self) {
        if self.release_pages {
            if let Some(source) = self.source {
                let released_prefix = self.released_prefix.get().min(self.bytes.len());
                if released_prefix == 0 {
                    source.release_tensor_pages(&self.source_name);
                } else if released_prefix < self.bytes.len() {
                    let _ = source.release_tensor_range_pages(
                        &self.source_name,
                        released_prefix,
                        self.bytes.len() - released_prefix,
                    );
                }
            }
        }
    }
}

/// Owner-scoped view handed to a family adapter while one persistent group or
/// transformer layer is resident. Every successful view updates the logical
/// ledger before control returns to the adapter.
pub struct PlannedTensorReader<'source, 'ledger, 'plan> {
    source: &'source dyn ModelSource,
    ledger: &'ledger mut ReadLedger<'plan>,
    owner: TensorOwner,
    staging: Option<&'source LayerStagingBuffer>,
    timings: SourceLoadTimings,
}

impl<'source, 'ledger, 'plan> PlannedTensorReader<'source, 'ledger, 'plan> {
    pub fn new(
        source: &'source dyn ModelSource,
        ledger: &'ledger mut ReadLedger<'plan>,
        owner: TensorOwner,
    ) -> Self {
        Self {
            source,
            ledger,
            owner,
            staging: None,
            timings: SourceLoadTimings::default(),
        }
    }

    pub fn new_with_staging(
        source: &'source dyn ModelSource,
        ledger: &'ledger mut ReadLedger<'plan>,
        owner: TensorOwner,
        staging: &'source LayerStagingBuffer,
    ) -> Self {
        Self {
            source,
            ledger,
            owner,
            staging: Some(staging),
            timings: SourceLoadTimings::default(),
        }
    }

    pub const fn timings(&self) -> SourceLoadTimings {
        self.timings
    }

    fn record_load(
        &mut self,
        source_bytes: usize,
        gpu_upload_bytes: usize,
        from_prefetch: bool,
        view_us: u64,
        decode_us: u64,
        upload_us: u64,
        release_us: u64,
    ) {
        self.timings.record(
            source_bytes,
            gpu_upload_bytes,
            from_prefetch,
            view_us,
            decode_us,
            upload_us,
            release_us,
        );
    }

    pub fn read(&mut self, logical_name: &str) -> Result<PlannedTensorView<'_>, CalibError> {
        let entry = self.ledger.plan.entry(logical_name).ok_or_else(|| {
            CalibError::ReadLedger(format!("unplanned logical tensor {logical_name}"))
        })?;
        if entry.owner != self.owner {
            return Err(CalibError::ReadLedger(format!(
                "logical tensor {logical_name} belongs to {:?}, not active owner {:?}",
                entry.owner, self.owner
            )));
        }
        let staged_bytes = self
            .staging
            .and_then(|staging| staging.view(&entry.storage));
        let (info, bytes, from_prefetch) = if let Some(bytes) = staged_bytes {
            let info = self.source.tensor_info(&entry.source_name).ok_or_else(|| {
                CalibError::ReadLedger(format!(
                    "source tensor {} has planned storage but no metadata",
                    entry.source_name
                ))
            })?;
            (info, bytes, true)
        } else {
            let (info, bytes) = self.source.tensor_data(&entry.source_name).ok_or_else(|| {
                CalibError::ReadLedger(format!(
                    "source tensor {} has metadata but no readable payload",
                    entry.source_name
                ))
            })?;
            (info, bytes, false)
        };
        // A canonical tensor with a declared future alias (for example tied
        // embedding/lm-head storage) stays resident until that alias consumes
        // the same pages. Ordinary layer tensors and alias views release their
        // file-backed cache as soon as the upload/conversion scope ends.
        let release_pages = !from_prefetch
            && (entry.alias_of.is_some()
                || !self
                    .ledger
                    .plan
                    .entries()
                    .iter()
                    .any(|candidate| candidate.alias_of.as_deref() == Some(logical_name)));
        let source_name = entry.source_name.clone();
        let action = self.ledger.consume(logical_name)?;
        Ok(PlannedTensorView {
            info,
            bytes,
            action,
            from_prefetch,
            source: (!from_prefetch).then_some(self.source),
            source_name,
            release_pages,
            released_prefix: Cell::new(0),
        })
    }
}

/// Decode a raw BF16/F16/F32 source view to host F32. This is an offline
/// calibration/import helper, shared by family adapters so source dtype
/// handling cannot drift between Qwen, Gemma, and later families.
pub fn source_payload_f32(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, CalibError> {
    let element_bytes = match dtype {
        "F16" | "BF16" => 2,
        "F32" => 4,
        other => {
            return Err(CalibError::InvalidSourcePlan(format!(
                "unsupported source dtype {other}"
            )))
        }
    };
    if bytes.len() % element_bytes != 0 {
        return Err(CalibError::InvalidSourcePlan(format!(
            "{dtype} tensor byte count {} is not aligned to {element_bytes} bytes",
            bytes.len()
        )));
    }
    match dtype {
        "F16" => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect()),
        "BF16" => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| f32::from_bits((u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16))
            .collect()),
        "F32" => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()),
        _ => unreachable!("source dtype validated above"),
    }
}

pub fn validate_source_shape(
    info: &TensorInfo,
    expected: &[usize],
    logical_name: &str,
) -> Result<(), CalibError> {
    if info.shape != expected {
        return Err(CalibError::InvalidSourcePlan(format!(
            "tensor {logical_name} has shape {:?}; expected {expected:?}",
            info.shape
        )));
    }
    Ok(())
}

fn validate_source_bytes(
    bytes: &[u8],
    elements: usize,
    element_bytes: usize,
    dtype: &str,
) -> Result<(), CalibError> {
    let expected = elements
        .checked_mul(element_bytes)
        .ok_or_else(|| CalibError::InvalidSourcePlan("source tensor byte count overflow".into()))?;
    if bytes.len() != expected {
        return Err(CalibError::InvalidSourcePlan(format!(
            "{dtype} tensor is {} bytes; expected {expected}",
            bytes.len()
        )));
    }
    Ok(())
}

/// Upload a raw source tensor using native BF16 on gfx11/gfx12 and a portable
/// F16 conversion on older cards. F16 and F32 source payloads remain typed.
pub fn upload_source_payload(
    gpu: &Gpu,
    source_dtype: &str,
    bytes: &[u8],
    shape: &[usize],
) -> Result<GpuTensor, CalibError> {
    let expected_elements = shape.iter().try_fold(1usize, |total, &dimension| {
        total.checked_mul(dimension).ok_or_else(|| {
            CalibError::InvalidSourcePlan("source tensor element count overflow".into())
        })
    })?;
    match source_dtype {
        "F16" => {
            validate_source_bytes(bytes, expected_elements, 2, source_dtype)?;
            let mut tensor = gpu
                .upload_raw(bytes, shape)
                .map_err(|error| CalibError::Runtime(error.to_string()))?;
            tensor.dtype = DType::F16;
            Ok(tensor)
        }
        "BF16" if gpu.arch.starts_with("gfx11") || gpu.arch.starts_with("gfx12") => {
            validate_source_bytes(bytes, expected_elements, 2, source_dtype)?;
            let mut tensor = gpu
                .upload_raw(bytes, shape)
                .map_err(|error| CalibError::Runtime(error.to_string()))?;
            tensor.dtype = DType::BF16;
            Ok(tensor)
        }
        "BF16" => {
            validate_source_bytes(bytes, expected_elements, 2, source_dtype)?;
            let mut f16 = Vec::with_capacity(bytes.len());
            for chunk in bytes.chunks_exact(2) {
                let value = f32::from_bits((u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16);
                f16.extend_from_slice(&f32_to_f16(value).to_le_bytes());
            }
            let mut tensor = gpu
                .upload_raw(&f16, shape)
                .map_err(|error| CalibError::Runtime(error.to_string()))?;
            tensor.dtype = DType::F16;
            Ok(tensor)
        }
        "F32" => {
            validate_source_bytes(bytes, expected_elements, 4, source_dtype)?;
            let mut tensor = gpu
                .upload_raw(bytes, shape)
                .map_err(|error| CalibError::Runtime(error.to_string()))?;
            tensor.dtype = DType::F32;
            Ok(tensor)
        }
        other => Err(CalibError::InvalidSourcePlan(format!(
            "unsupported source dtype {other}"
        ))),
    }
}

fn native_source_gpu_dtype(gpu: &Gpu, source_dtype: &str) -> Option<DType> {
    match source_dtype {
        "F16" => Some(DType::F16),
        "BF16" if gpu.arch.starts_with("gfx11") || gpu.arch.starts_with("gfx12") => {
            Some(DType::BF16)
        }
        "F32" => Some(DType::F32),
        _ => None,
    }
}

pub fn load_source_matrix(
    reader: &mut PlannedTensorReader<'_, '_, '_>,
    gpu: &Gpu,
    logical_name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, CalibError> {
    let buf = load_source_tensor(reader, gpu, logical_name, &[m, k])?;
    let gpu_dtype = buf.dtype;
    if gpu_dtype == DType::Raw {
        return Err(CalibError::InvalidSourcePlan(format!(
            "matrix {logical_name} did not resolve a typed GPU dtype"
        )));
    }
    Ok(WeightTensor {
        buf,
        gpu_dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    })
}

/// Load one typed source tensor while accounting for view, upload/refault, and
/// release phases. Family adapters use this for stacked/fused tensors as well
/// as ordinary matrices so the persisted split covers the full layer payload.
pub fn load_source_tensor(
    reader: &mut PlannedTensorReader<'_, '_, '_>,
    gpu: &Gpu,
    logical_name: &str,
    shape: &[usize],
) -> Result<GpuTensor, CalibError> {
    let view_started = Instant::now();
    let view = reader.read(logical_name)?;
    let view_us = elapsed_micros(view_started);
    validate_source_shape(view.info, shape, logical_name)?;
    let source_bytes = view.bytes.len();
    let from_prefetch = view.from_prefetch;
    let mut chunk_release_us = 0u64;
    let upload_started = Instant::now();
    let upload = if let Some(dtype) = native_source_gpu_dtype(gpu, view.info.dtype.as_str()) {
        validate_source_bytes(
            view.bytes,
            shape.iter().product(),
            dtype.size(),
            &view.info.dtype,
        )
        .and_then(|()| {
            let mut tensor = gpu
                .upload_raw_chunked(
                    view.bytes,
                    shape,
                    SOURCE_UPLOAD_CHUNK_BYTES,
                    |byte_offset, byte_len| {
                        let release_started = Instant::now();
                        if view.release_copied_range(byte_offset, byte_len) {
                            chunk_release_us =
                                chunk_release_us.saturating_add(elapsed_micros(release_started));
                        }
                    },
                )
                .map_err(|error| CalibError::Runtime(error.to_string()))?;
            tensor.dtype = dtype;
            Ok(tensor)
        })
    } else {
        upload_source_payload(gpu, view.info.dtype.as_str(), view.bytes, shape)
    };
    let upload_us = elapsed_micros(upload_started).saturating_sub(chunk_release_us);
    let gpu_upload_bytes = shape
        .iter()
        .try_fold(1usize, |total, &dimension| total.checked_mul(dimension))
        .and_then(|elements| {
            upload
                .as_ref()
                .ok()
                .and_then(|tensor| elements.checked_mul(tensor.dtype.size()))
        })
        .unwrap_or(source_bytes);
    let release_started = Instant::now();
    drop(view);
    let release_us = chunk_release_us.saturating_add(elapsed_micros(release_started));
    reader.record_load(
        source_bytes,
        gpu_upload_bytes,
        from_prefetch,
        view_us,
        0,
        upload_us,
        release_us,
    );
    upload
}

pub fn load_source_f32_tensor(
    reader: &mut PlannedTensorReader<'_, '_, '_>,
    gpu: &mut Gpu,
    logical_name: &str,
    elements: usize,
    add_one: bool,
) -> Result<GpuTensor, CalibError> {
    let view_started = Instant::now();
    let view = reader.read(logical_name)?;
    let view_us = elapsed_micros(view_started);
    if view.info.shape.iter().product::<usize>() != elements {
        return Err(CalibError::InvalidSourcePlan(format!(
            "tensor {logical_name} has shape {:?}; expected {elements} elements",
            view.info.shape
        )));
    }
    let source_bytes = view.bytes.len();
    let from_prefetch = view.from_prefetch;
    let decode_started = Instant::now();
    let mut values = source_payload_f32(view.info.dtype.as_str(), view.bytes)?;
    if values.len() != elements {
        return Err(CalibError::InvalidSourcePlan(format!(
            "tensor {logical_name} decoded {} elements; expected {elements}",
            values.len()
        )));
    }
    if add_one {
        values.iter_mut().for_each(|value| *value += 1.0);
    }
    let decode_us = elapsed_micros(decode_started);
    let upload_started = Instant::now();
    let upload = gpu
        .upload_f32(&values, &[elements])
        .map_err(|error| CalibError::Runtime(error.to_string()));
    let upload_us = elapsed_micros(upload_started);
    let release_started = Instant::now();
    drop(view);
    let release_us = elapsed_micros(release_started);
    reader.record_load(
        source_bytes,
        elements.saturating_mul(std::mem::size_of::<f32>()),
        from_prefetch,
        view_us,
        decode_us,
        upload_us,
        release_us,
    );
    upload
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_model::{QuantConfig, TensorInfo};
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    struct FakeSource {
        root: PathBuf,
        infos: BTreeMap<String, TensorInfo>,
        locations: BTreeMap<String, TensorStorageLocation>,
        payload: Vec<u8>,
        released: RefCell<Vec<String>>,
        released_ranges: RefCell<Vec<(String, usize, usize)>>,
    }

    impl FakeSource {
        fn new() -> Self {
            let root = PathBuf::from("/fixture");
            let mut infos = BTreeMap::new();
            let mut locations = BTreeMap::new();
            for (name, shard, offset, len) in [
                ("embed", "model-00001.safetensors", 100, 16),
                ("l0.w", "model-00002.safetensors", 200, 24),
                ("l1.a", "model-00003.safetensors", 300, 10),
                ("l1.b", "model-00003.safetensors", 400, 20),
            ] {
                infos.insert(
                    name.into(),
                    TensorInfo {
                        name: name.into(),
                        dtype: "BF16".into(),
                        shape: vec![len / 2],
                        quant_type: 0xff,
                        data_offset: offset,
                        data_size: len,
                    },
                );
                locations.insert(
                    name.into(),
                    TensorStorageLocation {
                        path: root.join(shard),
                        byte_offset: offset as u64,
                        byte_len: len as u64,
                    },
                );
            }
            Self {
                root,
                infos,
                locations,
                payload: vec![0; 24],
                released: RefCell::new(Vec::new()),
                released_ranges: RefCell::new(Vec::new()),
            }
        }
    }

    impl ModelSource for FakeSource {
        fn metadata_json(&self) -> &str {
            "{}"
        }
        fn arch_id(&self) -> u32 {
            0
        }
        fn quant_config(&self) -> Option<&QuantConfig> {
            None
        }
        fn tensor_data(&self, name: &str) -> Option<(&TensorInfo, &[u8])> {
            let info = self.infos.get(name)?;
            Some((info, &self.payload[..info.data_size]))
        }
        fn release_tensor_pages(&self, name: &str) {
            self.released.borrow_mut().push(name.to_string());
        }
        fn release_tensor_range_pages(
            &self,
            name: &str,
            byte_offset: usize,
            byte_len: usize,
        ) -> bool {
            self.released_ranges
                .borrow_mut()
                .push((name.to_string(), byte_offset, byte_len));
            true
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.infos.get(name)
        }
        fn tensor_names(&self) -> Vec<&str> {
            self.infos.keys().map(String::as_str).collect()
        }
        fn path(&self) -> &Path {
            &self.root
        }
        fn tensor_storage(&self, name: &str) -> Option<TensorStorageLocation> {
            self.locations.get(name).cloned()
        }
    }

    fn fixture_plan() -> TensorLoadPlan {
        TensorLoadPlan::build(
            &FakeSource::new(),
            [
                TensorLoadRequest::tensor("embedding", "embed", TensorOwner::Persistent),
                TensorLoadRequest::alias("lm_head", "embed", TensorOwner::Persistent, "embedding"),
                TensorLoadRequest::tensor("layer0.weight", "l0.w", TensorOwner::Layer(0)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn plan_tracks_shards_unique_bytes_and_explicit_aliases() {
        let plan = fixture_plan();
        assert_eq!(plan.unique_source_bytes(), 40);
        assert_eq!(plan.bytes_for(TensorOwner::Persistent), 16);
        assert_eq!(plan.bytes_for(TensorOwner::Layer(0)), 24);
        assert!(plan
            .entry("embedding")
            .unwrap()
            .storage
            .path
            .ends_with("model-00001.safetensors"));
        assert_eq!(
            plan.entry("lm_head").unwrap().alias_of.as_deref(),
            Some("embedding")
        );
    }

    #[test]
    fn prefetch_ranges_are_owner_scoped_alias_free_and_budget_bounded() {
        let plan = TensorLoadPlan::build(
            &FakeSource::new(),
            [
                TensorLoadRequest::tensor("layer1.a", "l1.a", TensorOwner::Layer(1)),
                TensorLoadRequest::alias(
                    "layer1.a_alias",
                    "l1.a",
                    TensorOwner::Layer(1),
                    "layer1.a",
                ),
                TensorLoadRequest::tensor("layer1.b", "l1.b", TensorOwner::Layer(1)),
                TensorLoadRequest::tensor("layer0.weight", "l0.w", TensorOwner::Layer(0)),
            ],
        )
        .unwrap();

        let ranges = plan.prefetch_ranges_for(TensorOwner::Layer(1), 25);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].byte_offset, 300);
        assert_eq!(ranges[0].byte_len, 10);
        assert_eq!(ranges.iter().map(|range| range.byte_len).sum::<u64>(), 10);
        assert!(plan
            .prefetch_ranges_for(TensorOwner::Layer(1), 0)
            .is_empty());
    }

    #[test]
    fn layer_prefetch_retains_ranges_for_direct_tensor_views() {
        use std::io::Write;

        let path = std::env::temp_dir().join(format!(
            "hipfire-layer-prefetch-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&vec![0x5au8; 3 * 1024 * 1024]).unwrap();
        drop(file);

        let report = LayerPrefetch::spawn(vec![TensorStorageLocation {
            path: path.clone(),
            byte_offset: 512 * 1024,
            byte_len: 2 * 1024 * 1024,
        }])
        .unwrap()
        .wait();
        assert_eq!(report.requested_bytes, 2 * 1024 * 1024);
        assert_eq!(report.completed_bytes, report.requested_bytes);
        assert_eq!(report.ranges, 1);
        assert!(report.errors.is_empty());
        assert!(report.elapsed_us > 0);
        assert_eq!(report.staging.byte_len(), 2 * 1024 * 1024);
        let tensor = TensorStorageLocation {
            path: path.clone(),
            byte_offset: 768 * 1024,
            byte_len: 1024 * 1024,
        };
        assert_eq!(
            report.staging.view(&tensor).unwrap(),
            vec![0x5a; 1024 * 1024]
        );

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn owner_reader_prefers_staged_bytes_without_releasing_source_pages() {
        let source = FakeSource::new();
        let plan = TensorLoadPlan::build(
            &source,
            [TensorLoadRequest::tensor(
                "layer0.weight",
                "l0.w",
                TensorOwner::Layer(0),
            )],
        )
        .unwrap();
        let storage = plan.entry("layer0.weight").unwrap().storage.clone();
        let staging = LayerStagingBuffer::from_ranges(vec![StagedSourceRange {
            location: storage,
            bytes: vec![0x7b; 24],
        }]);
        let mut ledger = ReadLedger::new(&plan);
        let mut reader = PlannedTensorReader::new_with_staging(
            &source,
            &mut ledger,
            TensorOwner::Layer(0),
            &staging,
        );
        let view = reader.read("layer0.weight").unwrap();
        assert!(view.from_prefetch);
        assert_eq!(view.bytes, [0x7b; 24]);
        drop(view);
        assert!(source.released.borrow().is_empty());
    }

    #[test]
    fn alias_order_does_not_depend_on_lexical_logical_name() {
        let plan = TensorLoadPlan::build(
            &FakeSource::new(),
            [
                TensorLoadRequest::alias(
                    "a_lm_head",
                    "embed",
                    TensorOwner::Persistent,
                    "z_embedding",
                ),
                TensorLoadRequest::tensor("z_embedding", "embed", TensorOwner::Persistent),
            ],
        )
        .unwrap();
        assert_eq!(plan.unique_source_bytes(), 16);
        assert_eq!(
            plan.entry("a_lm_head").unwrap().alias_of.as_deref(),
            Some("z_embedding")
        );
    }

    #[test]
    fn plan_rejects_missing_duplicate_and_undeclared_shared_storage() {
        let source = FakeSource::new();
        let duplicate = TensorLoadPlan::build(
            &source,
            [
                TensorLoadRequest::tensor("x", "embed", TensorOwner::Persistent),
                TensorLoadRequest::tensor("x", "l0.w", TensorOwner::Layer(0)),
            ],
        );
        assert!(duplicate.is_err());
        assert!(TensorLoadPlan::build(
            &source,
            [TensorLoadRequest::tensor(
                "missing",
                "nope",
                TensorOwner::Persistent
            )],
        )
        .is_err());
        assert!(TensorLoadPlan::build(
            &source,
            [
                TensorLoadRequest::tensor("a", "embed", TensorOwner::Persistent),
                TensorLoadRequest::tensor("b", "embed", TensorOwner::Persistent),
            ],
        )
        .is_err());
    }

    #[test]
    fn read_ledger_reads_canonical_once_and_requires_every_logical_use() {
        let plan = fixture_plan();
        let mut ledger = ReadLedger::new(&plan);
        assert!(matches!(
            ledger.consume("embedding").unwrap(),
            ReadAction::Read(_)
        ));
        assert_eq!(
            ledger.consume("lm_head").unwrap(),
            ReadAction::Reuse {
                canonical_logical: "embedding".into()
            }
        );
        assert!(ledger.assert_complete().is_err());
        assert!(matches!(
            ledger.consume("layer0.weight").unwrap(),
            ReadAction::Read(_)
        ));
        ledger.assert_complete().unwrap();
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.logical_bytes_read, 40);
        assert_eq!(snapshot.planned_logical, snapshot.consumed_logical);
        assert!(snapshot.missing_logical.is_empty());
        assert!(snapshot.duplicate_logical.is_empty());
        assert!(ledger.consume("embedding").is_err());
    }

    #[test]
    fn read_ledger_snapshot_resumes_and_rejects_alias_first() {
        let plan = fixture_plan();
        let mut alias_first = ReadLedger::new(&plan);
        assert!(alias_first.consume("lm_head").is_err());

        let mut first = ReadLedger::new(&plan);
        first.consume("embedding").unwrap();
        let mut resumed = ReadLedger::resume(&plan, first.snapshot()).unwrap();
        resumed.consume("lm_head").unwrap();
        resumed.consume("layer0.weight").unwrap();
        resumed.assert_complete().unwrap();
    }

    #[test]
    fn owner_scoped_reader_rejects_cross_layer_consumption() {
        let source = FakeSource::new();
        let plan = fixture_plan();
        let mut ledger = ReadLedger::new(&plan);
        let mut reader = PlannedTensorReader::new(&source, &mut ledger, TensorOwner::Persistent);
        assert!(reader.read("layer0.weight").is_err());
    }

    #[test]
    fn planned_views_release_layer_pages_but_hold_tied_canonical_until_alias() {
        let source = FakeSource::new();
        let plan = fixture_plan();
        let mut ledger = ReadLedger::new(&plan);
        {
            let mut reader =
                PlannedTensorReader::new(&source, &mut ledger, TensorOwner::Persistent);
            let embedding = reader.read("embedding").unwrap();
            drop(embedding);
            assert!(source.released.borrow().is_empty());
            let lm_head = reader.read("lm_head").unwrap();
            drop(lm_head);
            assert_eq!(source.released.borrow().as_slice(), ["embed"]);
        }
        {
            let mut reader = PlannedTensorReader::new(&source, &mut ledger, TensorOwner::Layer(0));
            let layer = reader.read("layer0.weight").unwrap();
            drop(layer);
        }
        assert_eq!(source.released.borrow().as_slice(), ["embed", "l0.w"]);
        ledger.assert_complete().unwrap();
    }

    #[test]
    fn planned_view_releases_completed_upload_chunks_and_not_tied_canonical_ranges() {
        let source = FakeSource::new();
        let plan = fixture_plan();
        let mut ledger = ReadLedger::new(&plan);
        {
            let mut reader =
                PlannedTensorReader::new(&source, &mut ledger, TensorOwner::Persistent);
            let embedding = reader.read("embedding").unwrap();
            assert!(!embedding.release_copied_range(0, 8));
            assert!(source.released_ranges.borrow().is_empty());
        }
        {
            let mut reader = PlannedTensorReader::new(&source, &mut ledger, TensorOwner::Layer(0));
            let layer = reader.read("layer0.weight").unwrap();
            assert!(layer.release_copied_range(0, 8));
            assert!(layer.release_copied_range(8, 16));
            assert_eq!(
                source.released_ranges.borrow().as_slice(),
                [("l0.w".to_string(), 0, 8), ("l0.w".to_string(), 8, 16)]
            );
            drop(layer);
        }
        assert!(source.released.borrow().is_empty());
    }
}
