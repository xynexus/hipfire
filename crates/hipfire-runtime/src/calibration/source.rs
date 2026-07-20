// SPDX-License-Identifier: Apache-2.0
// hipfire — source planning and logical tensor-read accounting.

use super::contracts::CalibError;
use crate::quant::{f16_to_f32, f32_to_f16};
use crate::weights::WeightTensor;
use hipfire_model::{ModelSource, TensorInfo, TensorStorageLocation};
use hipfire_rdna::{DType, Gpu, GpuTensor};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

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
    source: &'a dyn ModelSource,
    source_name: String,
    release_pages: bool,
}

impl Drop for PlannedTensorView<'_> {
    fn drop(&mut self) {
        if self.release_pages {
            self.source.release_tensor_pages(&self.source_name);
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
        }
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
        let (info, bytes) = self.source.tensor_data(&entry.source_name).ok_or_else(|| {
            CalibError::ReadLedger(format!(
                "source tensor {} has metadata but no readable payload",
                entry.source_name
            ))
        })?;
        // A canonical tensor with a declared future alias (for example tied
        // embedding/lm-head storage) stays resident until that alias consumes
        // the same pages. Ordinary layer tensors and alias views release their
        // file-backed cache as soon as the upload/conversion scope ends.
        let release_pages = entry.alias_of.is_some()
            || !self
                .ledger
                .plan
                .entries()
                .iter()
                .any(|candidate| candidate.alias_of.as_deref() == Some(logical_name));
        let source_name = entry.source_name.clone();
        let action = self.ledger.consume(logical_name)?;
        Ok(PlannedTensorView {
            info,
            bytes,
            action,
            source: self.source,
            source_name,
            release_pages,
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

pub fn load_source_matrix(
    reader: &mut PlannedTensorReader<'_, '_, '_>,
    gpu: &Gpu,
    logical_name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, CalibError> {
    let view = reader.read(logical_name)?;
    validate_source_shape(view.info, &[m, k], logical_name)?;
    let buf = upload_source_payload(gpu, view.info.dtype.as_str(), view.bytes, &[m, k])?;
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

pub fn load_source_f32_tensor(
    reader: &mut PlannedTensorReader<'_, '_, '_>,
    gpu: &mut Gpu,
    logical_name: &str,
    elements: usize,
    add_one: bool,
) -> Result<GpuTensor, CalibError> {
    let view = reader.read(logical_name)?;
    if view.info.shape.iter().product::<usize>() != elements {
        return Err(CalibError::InvalidSourcePlan(format!(
            "tensor {logical_name} has shape {:?}; expected {elements} elements",
            view.info.shape
        )));
    }
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
    gpu.upload_f32(&values, &[elements])
        .map_err(|error| CalibError::Runtime(error.to_string()))
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
    }

    impl FakeSource {
        fn new() -> Self {
            let root = PathBuf::from("/fixture");
            let mut infos = BTreeMap::new();
            let mut locations = BTreeMap::new();
            for (name, shard, offset, len) in [
                ("embed", "model-00001.safetensors", 100, 16),
                ("l0.w", "model-00002.safetensors", 200, 24),
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
}
