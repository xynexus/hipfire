// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Offline compose/decompose tooling for HFQM role/feature sidecars.
//!
//! An `.hfq` model can be shipped either as a base container plus separate
//! sibling sidecar files (`<base>.mtp.hfq`, `.dflash.hfq`, `.triattn.hfq`,
//! `.calib.hfq`, discovered by `hipfire_model::detect_sidecars`) or as a single
//! bundled container carrying every feature's tensors (canonical name shape
//! `Family-Size.mtp.vl.mq4.hfq`).
//!
//! [`compose_hfq`] merges a base container and its sidecars into one bundle;
//! [`decompose_hfq`] splits a bundle back into its component files. They are a
//! lossless inverse pair: compose records a provenance manifest
//! ([`HFQM_COMPOSE_KEY`]) in the bundle metadata that stores, per component, the
//! original filename, `arch_id`, tensor name list, and verbatim metadata JSON —
//! so decompose reproduces each source file byte-for-byte without any per-arch
//! tensor-name inference. Neither operation transforms tensor payload bytes;
//! this is packaging granularity only, orthogonal to `hipfire optimize` (which
//! re-tiles weights into an arch-optimal layout).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use hipfire_runtime::hfq::{
    write_hfqm_package_streaming, HfqFile, HfqPackage, HfqStreamEntry, HFQM_ARCH_NON_WEIGHT_PACKAGE,
};
pub use hipfire_runtime::hfq_compose::{
    ComposeComponent, ComposeManifest, ComposeStoredEntry, ComposeStoredSegment,
    HFQM_COMPONENT_PREFIX, HFQM_COMPOSE_FORMAT, HFQM_COMPOSE_FORMAT_V1, HFQM_COMPOSE_KEY,
};
use hipfire_runtime::hfq_modules::{
    module_table_json, parse_module_table, validate_modules, HfqModuleRecord, HFQM_MODULE_TABLE_KEY,
};

/// Injected `role -> owned config-key list` map. Supplied by a caller that can
/// see the arch registry (`Arch::sidecar_config_keys`); this crate stays
/// arch-agnostic. Empty (the default) reproduces the pre-partition behavior:
/// no config keys move on compose/decompose.
pub type RoleConfigKeys = std::collections::BTreeMap<String, Vec<String>>;

/// Known role/feature tokens used to label a sidecar component. Purely
/// cosmetic (the exact reconstruction uses `filename`/`metadata_json`); this
/// only produces a friendly `tag` in the manifest.
pub const KNOWN_ROLES: &[&str] = &[
    "mtp", "dflash", "triattn", "vl", "calib", "hessian", "jinja",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ComposeCheckComponent {
    pub role: String,
    pub filename: String,
    pub source_format: String,
    pub arch_id: Option<u32>,
    pub entries: usize,
    pub byte_len: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ComposeCheckReport {
    pub compatible: bool,
    pub bundle_arch_id: u32,
    pub manifest_format: String,
    pub components: Vec<ComposeCheckComponent>,
}

/// Parse the optional composition manifest from a package. Both v1 and v2 are
/// accepted; callers that require namespaced/opaque component access should
/// use [`component_view`], which validates the referenced entries.
pub fn compose_manifest(pkg: &HfqPackage) -> io::Result<Option<ComposeManifest>> {
    compose_manifest_from_metadata(&pkg.metadata_json)
}

/// Parse a compose manifest directly from metadata/index content.
pub fn compose_manifest_from_metadata(metadata_json: &str) -> io::Result<Option<ComposeManifest>> {
    let metadata: serde_json::Value = serde_json::from_str(metadata_json).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("HFQM metadata is not valid JSON: {error}"),
        )
    })?;
    let Some(value) = metadata.get(HFQM_COMPOSE_KEY) else {
        return Ok(None);
    };
    let manifest: ComposeManifest = serde_json::from_value(value.clone()).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid {HFQM_COMPOSE_KEY} manifest: {error}"),
        )
    })?;
    if manifest.format != HFQM_COMPOSE_FORMAT && manifest.format != HFQM_COMPOSE_FORMAT_V1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported compose manifest format {:?}", manifest.format),
        ));
    }
    Ok(Some(manifest))
}

/// Borrowed, zero-copy view of one role component inside a composed package.
/// Tensor lookups use original source names while returning byte slices from
/// the bundle mmap. No component is materialized or written to a temporary
/// file.
pub struct HfqComponentView<'a> {
    package: &'a HfqPackage,
    component: &'a ComposeComponent,
    v2: bool,
}

impl<'a> HfqComponentView<'a> {
    pub fn role(&self) -> &str {
        &self.component.tag
    }

    pub fn source_format(&self) -> &str {
        &self.component.source_format
    }

    pub fn arch_id(&self) -> u32 {
        self.component.arch_id
    }

    pub fn metadata_json(&self) -> &str {
        &self.component.metadata_json
    }

    pub fn sha256(&self) -> &str {
        &self.component.sha256
    }

    pub fn original_byte_len(&self) -> u64 {
        self.component.byte_len
    }

    pub fn original_tensor_names(&self) -> impl Iterator<Item = &str> {
        self.component.tensors.iter().map(String::as_str)
    }

    pub fn entry(&self, original_name: &str) -> Option<&'a hipfire_runtime::hfq::HfqPackageEntry> {
        let stored_name = if self.v2 {
            self.component
                .stored_entries
                .iter()
                .find(|entry| entry.original_name == original_name)?
                .stored_name
                .as_str()
        } else if self
            .component
            .tensors
            .iter()
            .any(|name| name == original_name)
        {
            original_name
        } else {
            return None;
        };
        self.package.entry(stored_name)
    }

    pub fn blob_data(&self, original_name: &str) -> Option<&'a [u8]> {
        let entry = self.entry(original_name)?;
        self.package.blob_data(&entry.name)
    }

    pub fn opaque_bytes(&self) -> io::Result<Option<&'a [u8]>> {
        let Some(stored_name) = self.component.opaque_entry.as_deref() else {
            return Ok(None);
        };
        let entry = self.package.entry(stored_name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("component opaque entry {stored_name:?} is absent"),
            )
        })?;
        if entry.quant_type != hipfire_quant_format::QuantType::OpaqueBytes.code()
            || entry.data_size as u64 != self.component.byte_len
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("component opaque entry {stored_name:?} has invalid encoding or length"),
            ));
        }
        Ok(self.package.blob_data(stored_name))
    }
}

/// Borrowed component view over the serving [`hipfire_runtime::hfq::HfqFile`] reader.
/// This is directly consumable by runtime loaders and never extracts a file.
pub struct HfqFileComponentView<'a> {
    file: &'a hipfire_runtime::hfq::HfqFile,
    component: &'a ComposeComponent,
}

impl<'a> HfqFileComponentView<'a> {
    pub fn role(&self) -> &str {
        &self.component.tag
    }

    pub fn source_format(&self) -> &str {
        &self.component.source_format
    }

    pub fn arch_id(&self) -> u32 {
        self.component.arch_id
    }

    pub fn metadata_json(&self) -> &str {
        &self.component.metadata_json
    }

    pub fn sha256(&self) -> &str {
        &self.component.sha256
    }

    pub fn original_byte_len(&self) -> u64 {
        self.component.byte_len
    }

    pub fn entry(&self, original_name: &str) -> Option<&'a hipfire_runtime::hfq::HfqTensorInfo> {
        self.tensor_data(original_name).map(|(entry, _)| entry)
    }

    pub fn tensor_data(
        &self,
        original_name: &str,
    ) -> Option<(&'a hipfire_runtime::hfq::HfqTensorInfo, &'a [u8])> {
        let stored_name = self
            .component
            .stored_entries
            .iter()
            .find(|entry| entry.original_name == original_name)?
            .stored_name
            .as_str();
        self.file.tensor_data(stored_name)
    }

    pub fn tensor_data_vec(
        &self,
        original_name: &str,
    ) -> Option<(&'a hipfire_runtime::hfq::HfqTensorInfo, Vec<u8>)> {
        let stored_name = self
            .component
            .stored_entries
            .iter()
            .find(|entry| entry.original_name == original_name)?
            .stored_name
            .as_str();
        self.file.tensor_data_vec(stored_name)
    }

    pub fn opaque_bytes(&self) -> io::Result<Option<&'a [u8]>> {
        let Some(stored_name) = self.component.opaque_entry.as_deref() else {
            return Ok(None);
        };
        let (entry, bytes) = self.file.tensor_data(stored_name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("component opaque entry {stored_name:?} is absent"),
            )
        })?;
        if entry.quant_type != hipfire_quant_format::QuantType::OpaqueBytes.code()
            || entry.data_size as u64 != self.component.byte_len
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("component opaque entry {stored_name:?} has invalid encoding or length"),
            ));
        }
        Ok(Some(bytes))
    }

    pub fn opaque_bytes_vec(&self) -> io::Result<Option<Vec<u8>>> {
        let Some(stored_name) = self.component.opaque_entry.as_deref() else {
            return Ok(None);
        };
        let (entry, bytes) = self.file.tensor_data_vec(stored_name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("component opaque entry {stored_name:?} is absent"),
            )
        })?;
        if entry.quant_type != hipfire_quant_format::QuantType::OpaqueBytes.code()
            || entry.data_size as u64 != self.component.byte_len
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("component opaque entry {stored_name:?} has invalid encoding or length"),
            ));
        }
        Ok(Some(bytes))
    }

    /// Verify the original-artifact SHA-256 by streaming mapped ranges in
    /// source-byte order. A legacy manifest without a digest fails closed.
    pub fn verify_digest(&self) -> io::Result<()> {
        if self.component.sha256.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "embedded component has no strong source digest",
            ));
        }
        if self.component.source_format == "tria-v1" {
            let bytes = self.opaque_bytes_vec()?.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "TRIA component has no payload")
            })?;
            let actual = format!("{:x}", Sha256::digest(&bytes));
            return digest_result(&self.component.sha256, &actual);
        }

        struct Range<'a> {
            offset: u64,
            stored_name: &'a str,
            length: u64,
        }
        let mut ranges = Vec::with_capacity(
            self.component.stored_entries.len() + self.component.stored_segments.len(),
        );
        for stored in &self.component.stored_entries {
            let entry = self
                .file
                .find_tensor_info(&stored.stored_name)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("component entry {:?} is absent", stored.stored_name),
                    )
                })?;
            ranges.push(Range {
                offset: stored.original_offset,
                stored_name: &stored.stored_name,
                length: entry.data_size as u64,
            });
        }
        for segment in &self.component.stored_segments {
            let entry = self
                .file
                .find_tensor_info(&segment.stored_name)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("component segment {:?} is absent", segment.stored_name),
                    )
                })?;
            if entry.quant_type != hipfire_quant_format::QuantType::OpaqueBytes.code()
                || entry.data_size as u64 != segment.length
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "component segment {:?} encoding mismatch",
                        segment.stored_name
                    ),
                ));
            }
            ranges.push(Range {
                offset: segment.original_offset,
                stored_name: &segment.stored_name,
                length: segment.length,
            });
        }
        ranges.sort_by_key(|range| range.offset);
        let mut cursor = 0u64;
        let mut hasher = Sha256::new();
        for range in ranges {
            if range.offset != cursor {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "component byte coverage jumps from {cursor} to {}",
                        range.offset
                    ),
                ));
            }
            let (_, bytes) = self
                .file
                .tensor_data_vec(range.stored_name)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("component range {:?} is unreadable", range.stored_name),
                    )
                })?;
            if bytes.len() as u64 != range.length {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("component range {:?} length mismatch", range.stored_name),
                ));
            }
            hasher.update(&bytes);
            cursor = cursor.checked_add(bytes.len() as u64).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "component range overflow")
            })?;
        }
        if cursor != self.component.byte_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "component byte coverage {cursor} != declared {}",
                    self.component.byte_len
                ),
            ));
        }
        let actual = format!("{:x}", hasher.finalize());
        digest_result(&self.component.sha256, &actual)
    }
}

fn digest_result(expected: &str, actual: &str) -> io::Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("component SHA-256 mismatch: expected {expected}, got {actual}"),
        ))
    }
}

/// Resolve one role from a parsed manifest into a validated borrowed view.
/// Duplicate roles, missing entries, name-map inconsistencies, and entries
/// outside the reserved namespace are rejected before any consumer sees data.
pub fn component_view<'a>(
    package: &'a HfqPackage,
    manifest: &'a ComposeManifest,
    role: &str,
) -> io::Result<Option<HfqComponentView<'a>>> {
    let mut matches = manifest
        .components
        .iter()
        .filter(|component| component.tag == role);
    let Some(component) = matches.next() else {
        return Ok(None);
    };
    if matches.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bundle contains duplicate {role:?} components"),
        ));
    }
    let v2 = manifest.format == HFQM_COMPOSE_FORMAT;
    if v2 {
        if component.source_format == "hfqm" {
            if component.stored_entries.len() != component.tensors.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("component {role:?} tensor map length mismatch"),
                ));
            }
            for stored in &component.stored_entries {
                if role != "base" && !stored.stored_name.starts_with(HFQM_COMPONENT_PREFIX) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("component {role:?} entry is outside the reserved namespace"),
                    ));
                }
                if package.entry(&stored.stored_name).is_none() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("component entry {:?} is absent", stored.stored_name),
                    ));
                }
            }
        } else if component.source_format == "tria-v1" {
            let Some(stored_name) = component.opaque_entry.as_deref() else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "TRIA component has no opaque entry",
                ));
            };
            if !stored_name.starts_with(HFQM_COMPONENT_PREFIX) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "TRIA opaque entry is outside the reserved namespace",
                ));
            }
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported component source format {:?}",
                    component.source_format
                ),
            ));
        }
    }
    let view = HfqComponentView {
        package,
        component,
        v2,
    };
    if component.source_format == "tria-v1" {
        view.opaque_bytes()?;
    }
    Ok(Some(view))
}

/// Resolve and validate a v2 role component against a serving [`HfqFile`].
/// Runtime consumers must call [`HfqFileComponentView::verify_digest`] before
/// treating manifest role claims as authoritative.
pub fn file_component_view<'a>(
    file: &'a hipfire_runtime::hfq::HfqFile,
    manifest: &'a ComposeManifest,
    role: &str,
) -> io::Result<Option<HfqFileComponentView<'a>>> {
    if manifest.format != HFQM_COMPOSE_FORMAT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "embedded runtime components require a compose.v2 manifest",
        ));
    }
    let mut matches = manifest
        .components
        .iter()
        .filter(|component| component.tag == role);
    let Some(component) = matches.next() else {
        return Ok(None);
    };
    if matches.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bundle contains duplicate {role:?} components"),
        ));
    }
    match component.source_format.as_str() {
        "hfqm" => {
            if component.stored_entries.len() != component.tensors.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("component {role:?} tensor map length mismatch"),
                ));
            }
            let mut original_names = BTreeSet::new();
            for stored in &component.stored_entries {
                if !original_names.insert(stored.original_name.as_str()) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("component {role:?} has duplicate original tensor names"),
                    ));
                }
                if role != "base" && !stored.stored_name.starts_with(HFQM_COMPONENT_PREFIX) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("component {role:?} entry is outside the reserved namespace"),
                    ));
                }
                if file.find_tensor_info(&stored.stored_name).is_none() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("component entry {:?} is absent", stored.stored_name),
                    ));
                }
            }
        }
        "tria-v1" => {
            let Some(stored_name) = component.opaque_entry.as_deref() else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "TRIA component has no opaque entry",
                ));
            };
            if !stored_name.starts_with(HFQM_COMPONENT_PREFIX) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "TRIA opaque entry is outside the reserved namespace",
                ));
            }
        }
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported component source format {other:?}"),
            ));
        }
    }
    Ok(Some(HfqFileComponentView { file, component }))
}

/// The first known role token in a filename's dot-groups (e.g.
/// `Model.mtp.hfq` -> `mtp`), if any. Shared with the CLI so composed bundle
/// names are derived from the same role table.
pub fn sidecar_tag_from_filename(path: &Path) -> Option<String> {
    let fname = path
        .file_name()
        .map(|s| s.to_string_lossy().to_ascii_lowercase())?;
    let stem = fname.strip_suffix(".hfq").unwrap_or(&fname).to_string();
    stem.split('.')
        .find(|seg| KNOWN_ROLES.contains(seg))
        .map(|s| s.to_string())
}

/// Derive a friendly role tag for a sidecar from its filename dot-groups, then
/// its metadata, falling back to `"sidecar"`.
fn derive_tag(path: &Path, metadata_json: &str) -> String {
    if let Some(tag) = sidecar_tag_from_filename(path) {
        return tag;
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(metadata_json) {
        for key in ["role", "artifact_kind", "package_schema", "architecture"] {
            if let Some(s) = v.get(key).and_then(|x| x.as_str()) {
                return s.to_string();
            }
        }
    }
    "sidecar".to_string()
}

fn file_name_string(path: &Path) -> io::Result<String> {
    path.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("input path has no file name: {}", path.display()),
            )
        })
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn planned_hfqm_offsets(metadata_len: usize, entries: &[HfqStreamEntry]) -> io::Result<Vec<usize>> {
    let mut index_len = 4u64;
    for entry in entries {
        let name_len = u64::try_from(entry.name.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "HFQM entry name length overflow",
            )
        })?;
        let dims = u64::try_from(entry.shape.len()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "HFQM entry dimension overflow")
        })?;
        index_len = index_len
            .checked_add(2 + name_len + 1 + 1 + dims * 4 + 4 + 8 + 8)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "HFQM index overflow"))?;
    }
    let metadata_len = u64::try_from(metadata_len)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "HFQM metadata overflow"))?;
    let data_start = 32u64
        .checked_add(metadata_len)
        .and_then(|value| value.checked_add(index_len))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "HFQM layout overflow"))?;
    let mut cursor = data_start
        .checked_add(4095)
        .map(|value| value & !4095)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "HFQM layout overflow"))?;
    let mut offsets = Vec::with_capacity(entries.len());
    for entry in entries {
        cursor = cursor
            .checked_add(31)
            .map(|value| value & !31)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "HFQM layout overflow"))?;
        offsets.push(usize::try_from(cursor).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "HFQM offset exceeds usize")
        })?);
        cursor = cursor.checked_add(entry.data_len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "HFQM payload layout overflow")
        })?;
    }
    Ok(offsets)
}

fn rebase_module_table(
    modules: &[HfqModuleRecord],
    base_entry_by_name: &HashMap<String, usize>,
    entries: &[HfqStreamEntry],
    offsets: &[usize],
) -> io::Result<Vec<HfqModuleRecord>> {
    let mut rebased = modules.to_vec();
    for module in &mut rebased {
        let mut tensor_offsets = Vec::with_capacity(module.tensors.len());
        for tensor in &module.tensors {
            let entry_index = *base_entry_by_name.get(&tensor.name).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "HFQM module {} references absent tensor {}",
                        module.module_id, tensor.name
                    ),
                )
            })?;
            let entry = &entries[entry_index];
            let data_size = usize::try_from(entry.data_len).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "module tensor size exceeds usize",
                )
            })?;
            if entry.quant_type != tensor.quant_type
                || entry.shape != tensor.shape
                || entry.group_size != tensor.group_size
                || data_size != tensor.data_size
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "HFQM module tensor {} metadata differs from the base index",
                        tensor.name
                    ),
                ));
            }
            tensor_offsets.push(offsets[entry_index]);
        }
        let module_start = tensor_offsets.iter().copied().min().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("HFQM module {} contains no tensors", module.module_id),
            )
        })?;
        let mut module_end = module_start;
        for (tensor, offset) in module.tensors.iter_mut().zip(tensor_offsets) {
            tensor.rel_offset = offset.checked_sub(module_start).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "module tensor offset underflow")
            })?;
            module_end = module_end.max(offset.checked_add(tensor.data_size).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "module tensor range overflow")
            })?);
        }
        module.data_offset = module_start;
        module.data_size = module_end - module_start;
    }
    validate_modules(&rebased, usize::MAX)?;
    Ok(rebased)
}

fn compose_bundle_metadata(
    mut metadata: serde_json::Value,
    base_modules: Option<Vec<HfqModuleRecord>>,
    base_entry_by_name: &HashMap<String, usize>,
    entries: &[HfqStreamEntry],
) -> io::Result<String> {
    let Some(base_modules) = base_modules else {
        return serde_json::to_string(&metadata).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("serializing bundle metadata: {error}"),
            )
        });
    };
    for _ in 0..8 {
        let encoded = serde_json::to_string(&metadata).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("serializing bundle metadata: {error}"),
            )
        })?;
        let offsets = planned_hfqm_offsets(encoded.len(), entries)?;
        let table = serde_json::to_value(module_table_json(rebase_module_table(
            &base_modules,
            base_entry_by_name,
            entries,
            &offsets,
        )?))
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        if metadata.get(HFQM_MODULE_TABLE_KEY) == Some(&table) {
            return Ok(encoded);
        }
        metadata[HFQM_MODULE_TABLE_KEY] = table;
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "HFQM module offsets did not converge while composing bundle metadata",
    ))
}

fn source_magic(path: &Path) -> io::Result<[u8; 4]> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    Ok(magic)
}

fn component_name(role: &str, component_index: usize, suffix: &str) -> String {
    format!("{HFQM_COMPONENT_PREFIX}{role}/{component_index}/{suffix}")
}

fn validate_role(role: &str) -> io::Result<()> {
    if role.is_empty()
        || !role
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid component role {role:?}"),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct TriaV1Geometry {
    n_layers: u32,
    n_heads: u32,
    head_dim: u32,
    rope_theta: f32,
    partial_rotary_factor: f32,
}

fn validate_tria_v1(bytes: &[u8]) -> io::Result<TriaV1Geometry> {
    const HEADER: usize = 28;
    if bytes.len() < HEADER || &bytes[..4] != b"TRIA" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a TRIA sidecar",
        ));
    }
    let word = |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let version = word(4);
    if version != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported TRIA version {version}"),
        ));
    }
    let geometry = TriaV1Geometry {
        n_layers: word(8),
        n_heads: word(12),
        head_dim: word(16),
        rope_theta: f32::from_bits(word(20)),
        partial_rotary_factor: f32::from_bits(word(24)),
    };
    if geometry.n_layers == 0
        || geometry.n_heads == 0
        || geometry.head_dim == 0
        || geometry.head_dim % 2 != 0
        || !geometry.rope_theta.is_finite()
        || geometry.rope_theta <= 0.0
        || !geometry.partial_rotary_factor.is_finite()
        || !(0.0..=1.0).contains(&geometry.partial_rotary_factor)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid TRIA v1 geometry",
        ));
    }
    let centers = (geometry.n_layers as usize)
        .checked_mul(geometry.n_heads as usize)
        .and_then(|n| n.checked_mul(geometry.head_dim as usize / 2))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "TRIA geometry overflow"))?;
    let expected = HEADER
        .checked_add(centers.checked_mul(12).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "TRIA payload length overflow")
        })?)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "TRIA length overflow"))?;
    if bytes.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("TRIA v1 length {} != expected {expected}", bytes.len()),
        ));
    }
    for chunk in bytes[HEADER..].chunks_exact(4) {
        if !f32::from_le_bytes(chunk.try_into().unwrap()).is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "TRIA v1 contains a non-finite center value",
            ));
        }
    }
    Ok(geometry)
}

fn metadata_number<'a>(value: &'a serde_json::Value, key: &str) -> Option<f64> {
    value
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .or_else(|| {
            ["config", "text_config", "model_config", "dflash"]
                .iter()
                .filter_map(|scope| value.get(*scope))
                .find_map(|child| metadata_number(child, key))
        })
}

fn metadata_string<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            ["config", "text_config", "model_config", "dflash"]
                .iter()
                .filter_map(|scope| value.get(*scope))
                .find_map(|child| metadata_string(child, key))
        })
}

fn compatible_number(
    base: &serde_json::Value,
    component: &serde_json::Value,
    key: &str,
) -> io::Result<()> {
    let (Some(left), Some(right)) = (metadata_number(base, key), metadata_number(component, key))
    else {
        return Ok(());
    };
    let tolerance = if left.abs().max(right.abs()) > 0.0 {
        left.abs().max(right.abs()) * 1e-6
    } else {
        0.0
    };
    if (left - right).abs() > tolerance {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("component {key} {right} is incompatible with base {key} {left}"),
        ));
    }
    Ok(())
}

fn validate_dflash_metadata(base: &serde_json::Value, pkg: &HfqFile) -> io::Result<()> {
    if pkg.arch_id != 20 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("DFLASH component arch_id {} must be 20", pkg.arch_id),
        ));
    }
    let metadata: serde_json::Value =
        serde_json::from_str(&pkg.metadata_json).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("DFLASH metadata is not valid JSON: {error}"),
            )
        })?;
    if metadata
        .get("architecture")
        .and_then(|value| value.as_str())
        != Some("dflash")
        || !metadata
            .get("dflash")
            .is_some_and(|value| value.is_object())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "arch-20 component lacks content-backed DFLASH metadata",
        ));
    }
    // The drafter is its own transformer and may intentionally use different
    // attention-head geometry from the target. Bind only dimensions that cross
    // the target/draft interface; comparing draft heads/head_dim rejects known
    // good DFLASH pairs such as Qwen3.5-9B (target 16x256, draft 32x128).
    for key in ["hidden_size", "vocab_size", "rope_theta"] {
        compatible_number(base, &metadata, key)?;
    }
    if let (Some(target_layers), Some(bound_layers)) = (
        metadata_number(base, "num_hidden_layers"),
        metadata_number(&metadata, "num_target_layers"),
    ) {
        if target_layers != bound_layers {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "DFLASH num_target_layers {bound_layers} is incompatible with base num_hidden_layers {target_layers}"
                ),
            ));
        }
    }
    if let Some(target_layer_ids) = metadata
        .get("dflash")
        .and_then(|dflash| dflash.get("target_layer_ids"))
        .and_then(serde_json::Value::as_array)
    {
        let target_layers = metadata_number(base, "num_hidden_layers").map(|value| value as u64);
        for value in target_layer_ids {
            let layer = value.as_u64().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "DFLASH target_layer_ids must contain unsigned layer indices",
                )
            })?;
            if target_layers.is_some_and(|count| layer >= count) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("DFLASH target layer {layer} is outside the base model"),
                ));
            }
        }
        if let Some(draft_layers) = metadata_number(&metadata, "num_hidden_layers") {
            if draft_layers as usize != target_layer_ids.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "DFLASH target_layer_ids length {} does not match draft num_hidden_layers {draft_layers}",
                        target_layer_ids.len()
                    ),
                ));
            }
        }
    }
    for key in ["tokenizer_fingerprint", "tokenizer_hash", "tokenizer_id"] {
        if let (Some(left), Some(right)) =
            (metadata_string(base, key), metadata_string(&metadata, key))
        {
            if left != right {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("DFLASH {key} {right:?} is incompatible with base {key} {left:?}"),
                ));
            }
        }
    }
    Ok(())
}

fn validate_triattn_hfqm_metadata(
    base_arch: u32,
    base: &serde_json::Value,
    pkg: &HfqFile,
) -> io::Result<()> {
    use hipfire_runtime::triattn::{
        TriAttnContextPolicy, TriAttnPackageMetadata, TriAttnRopeConvention, TRIATTN_ARTIFACT_KIND,
        TRIATTN_HFQM_SCHEMA,
    };

    let metadata: TriAttnPackageMetadata =
        serde_json::from_str(&pkg.metadata_json).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("TriAttention metadata is invalid: {error}"),
            )
        })?;
    if metadata.artifact_kind != TRIATTN_ARTIFACT_KIND
        || metadata.package_schema != TRIATTN_HFQM_SCHEMA
        || metadata.model_arch_id != base_arch
        || pkg.arch_id != base_arch
        || metadata.model_layers == 0
        || metadata.model_fingerprint.is_empty()
        || metadata.corpus_fingerprint.is_empty()
        || metadata.adapter.is_empty()
        || metadata.engine.is_empty()
        || metadata.layers.is_empty()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "TriAttention HFQM has incomplete or incompatible identity metadata",
        ));
    }
    if metadata_number(base, "num_hidden_layers")
        .is_some_and(|layers| layers as u32 != metadata.model_layers)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "TriAttention model_layers {} does not match base num_hidden_layers",
                metadata.model_layers
            ),
        ));
    }
    let mut physical_layers = BTreeSet::new();
    let mut tensors = BTreeSet::new();
    for layer in &metadata.layers {
        let center_count = (layer.q_heads as u64)
            .checked_mul((layer.head_dim / 2) as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "center count overflow"))?;
        let expected_bytes = center_count.checked_mul(12).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "center tensor length overflow")
        })?;
        let entry = pkg.find_tensor_info(&layer.center_tensor).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("TriAttention tensor {:?} is absent", layer.center_tensor),
            )
        })?;
        if layer.physical_layer >= metadata.model_layers
            || !physical_layers.insert(layer.physical_layer)
            || !tensors.insert(layer.center_tensor.as_str())
            || layer.q_heads == 0
            || layer.kv_heads == 0
            || layer.q_heads % layer.kv_heads != 0
            || layer.head_dim == 0
            || layer.head_dim % 2 != 0
            || layer.rotary_dim > layer.head_dim
            || layer.rotary_dim % 2 != 0
            || (layer.rotary_dim == 0 && layer.rope_convention != TriAttnRopeConvention::None)
            || (layer.rotary_dim != 0 && layer.rope_convention == TriAttnRopeConvention::None)
            || !layer.rope_theta.is_finite()
            || layer.rope_theta <= 0.0
            || layer.center_offset != 0
            || layer.center_count != center_count
            || layer.sample_count == 0
            || (layer.context_policy == TriAttnContextPolicy::Sliding
                && layer.sliding_window.is_none())
            || (layer.context_policy == TriAttnContextPolicy::Full
                && layer.sliding_window.is_some())
            || entry.quant_type != hipfire_quant_format::QuantType::F32.code()
            || entry.shape != [layer.q_heads, layer.head_dim / 2, 3]
            || entry.data_size as u64 != expected_bytes
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid TriAttention record or tensor for physical layer {}",
                    layer.physical_layer
                ),
            ));
        }
    }
    if pkg.tensors().len() != metadata.layers.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "TriAttention HFQM contains undeclared tensors",
        ));
    }
    Ok(())
}

fn validate_tria_geometry(base: &serde_json::Value, geometry: TriaV1Geometry) -> io::Result<()> {
    let tria = serde_json::json!({
        "num_hidden_layers": geometry.n_layers,
        "num_attention_heads": geometry.n_heads,
        "head_dim": geometry.head_dim,
        "rope_theta": geometry.rope_theta,
        "partial_rotary_factor": geometry.partial_rotary_factor,
    });
    for key in [
        "num_hidden_layers",
        "num_attention_heads",
        "head_dim",
        "rope_theta",
        "partial_rotary_factor",
    ] {
        compatible_number(base, &tria, key)?;
    }
    Ok(())
}

/// Read-only compatibility check used by `hipfire model compose --check`.
/// Performs the same content-backed role, architecture, geometry, reserved
/// namespace, duplicate-role, length, and digest checks without creating an
/// output file.
pub fn check_compose_inputs(inputs: &[PathBuf]) -> io::Result<ComposeCheckReport> {
    if inputs.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "compose needs a base container plus at least one sidecar (>= 2 inputs)",
        ));
    }
    if source_magic(&inputs[0])? != *b"HFQM" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the first compose input must be an HFQM base model",
        ));
    }
    let base = HfqFile::open_index_only(&inputs[0])?;
    let base_metadata: serde_json::Value = serde_json::from_str(&base.metadata_json)
        .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
    let mut roles = BTreeSet::new();
    let mut components = Vec::with_capacity(inputs.len());

    for (index, path) in inputs.iter().enumerate() {
        let magic = source_magic(path)?;
        let byte_len = std::fs::metadata(path)?.len();
        let sha256 = sha256_file(path)?;
        if magic == *b"HFQM" {
            let pkg = HfqFile::open_index_only(path)?;
            let role = if index == 0 {
                "base".to_string()
            } else {
                derive_tag(path, &pkg.metadata_json).to_ascii_lowercase()
            };
            validate_role(&role)?;
            if index > 0 && !roles.insert(role.clone()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("duplicate component role {role:?}"),
                ));
            }
            if pkg
                .tensors()
                .iter()
                .any(|entry| entry.name.starts_with(HFQM_COMPONENT_PREFIX))
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{} uses the reserved component namespace", path.display()),
                ));
            }
            if role == "dflash" || pkg.arch_id == 20 {
                if role != "dflash" {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "arch-20 HFQM component must have the dflash role",
                    ));
                }
                validate_dflash_metadata(&base_metadata, &pkg)?;
            } else if role == "triattn" {
                validate_triattn_hfqm_metadata(base.arch_id, &base_metadata, &pkg)?;
            } else if index > 0
                && pkg.arch_id != base.arch_id
                && pkg.arch_id != HFQM_ARCH_NON_WEIGHT_PACKAGE
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "sidecar {} arch_id {} is incompatible with base arch_id {}",
                        path.display(),
                        pkg.arch_id,
                        base.arch_id
                    ),
                ));
            }
            components.push(ComposeCheckComponent {
                role,
                filename: file_name_string(path)?,
                source_format: "hfqm".to_string(),
                arch_id: Some(pkg.arch_id),
                entries: pkg.tensors().len(),
                byte_len,
                sha256,
            });
        } else if magic == *b"TRIA" && index > 0 {
            if !roles.insert("triattn".to_string()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "duplicate triattn component",
                ));
            }
            let bytes = std::fs::read(path)?;
            let geometry = validate_tria_v1(&bytes)?;
            validate_tria_geometry(&base_metadata, geometry)?;
            components.push(ComposeCheckComponent {
                role: "triattn".to_string(),
                filename: file_name_string(path)?,
                source_format: "tria-v1".to_string(),
                arch_id: None,
                entries: 1,
                byte_len,
                sha256,
            });
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{} is neither HFQM nor a supported TRIA v1 sidecar",
                    path.display()
                ),
            ));
        }
    }
    Ok(ComposeCheckReport {
        compatible: true,
        bundle_arch_id: base.arch_id,
        manifest_format: HFQM_COMPOSE_FORMAT.to_string(),
        components,
    })
}

/// Merge a base container (first input) and its role/feature sidecars into a
/// single bundled `.hfq` written to `out`. See [`compose_hfq_with_config_keys`];
/// this passes an empty [`RoleConfigKeys`] (no config-key merge).
pub fn compose_hfq(inputs: &[PathBuf], out: &Path) -> io::Result<PathBuf> {
    compose_hfq_with_config_keys_options(inputs, out, &RoleConfigKeys::new(), false)
}

/// As [`compose_hfq`], but additionally merges each role sidecar's owned config
/// keys (per `role_keys`, keyed by the component's role tag) UP into the
/// bundle's top-level config — so the composed whole bundle advertises every
/// feature whose tensors it contains (e.g. `vision_config` travels up from the
/// `vl` sidecar). This is the inverse of the decompose-time move; together they
/// keep config claims and tensor presence consistent across a round trip.
///
/// The base's `arch_id` becomes the bundle's. Ordinary sidecars must share
/// that `arch_id` or use [`HFQM_ARCH_NON_WEIGHT_PACKAGE`]; content-backed
/// DFLASH arch-20 and raw TRIA v1 components follow their role-specific
/// compatibility checks. Non-base HFQM entries are namespaced, so their
/// original tensor names may overlap the base. Returns the written bundle
/// path.
pub fn compose_hfq_with_config_keys(
    inputs: &[PathBuf],
    out: &Path,
    role_keys: &RoleConfigKeys,
) -> io::Result<PathBuf> {
    compose_hfq_with_config_keys_options(inputs, out, role_keys, false)
}

/// Component-aware composition with an explicit overwrite policy. Output is
/// written to a sibling temporary file and renamed only after the complete
/// bundle has been flushed.
pub fn compose_hfq_with_config_keys_options(
    inputs: &[PathBuf],
    out: &Path,
    role_keys: &RoleConfigKeys,
    overwrite: bool,
) -> io::Result<PathBuf> {
    if inputs.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "compose needs a base container plus at least one sidecar (>= 2 inputs)",
        ));
    }

    if out.exists() && !overwrite {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("refusing to overwrite {}", out.display()),
        ));
    }

    enum InputPayload {
        Hfqm(HfqFile),
        TriaV1(Vec<u8>),
    }
    struct InputComponent {
        path: PathBuf,
        payload: InputPayload,
        byte_len: u64,
        sha256: String,
    }
    enum StreamSource {
        HfqmTensor {
            component: usize,
            name: String,
        },
        FileRange {
            path: PathBuf,
            offset: u64,
            length: u64,
        },
        TriaV1 {
            component: usize,
        },
    }

    let mut source_components = Vec::with_capacity(inputs.len());
    for (index, path) in inputs.iter().enumerate() {
        let magic = source_magic(path).map_err(|error| {
            io::Error::new(error.kind(), format!("opening {}: {error}", path.display()))
        })?;
        let payload = if &magic == b"HFQM" {
            InputPayload::Hfqm(HfqFile::open_index_only(path).map_err(|error| {
                io::Error::new(error.kind(), format!("opening {}: {error}", path.display()))
            })?)
        } else if &magic == b"TRIA" && index > 0 {
            let bytes = std::fs::read(path)?;
            validate_tria_v1(&bytes)?;
            InputPayload::TriaV1(bytes)
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{} is neither HFQM nor a supported TRIA v1 sidecar",
                    path.display()
                ),
            ));
        };
        source_components.push(InputComponent {
            path: path.clone(),
            byte_len: std::fs::metadata(path)?.len(),
            sha256: sha256_file(path)?,
            payload,
        });
    }

    let InputPayload::Hfqm(base_pkg) = &source_components[0].payload else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the first compose input must be an HFQM base model",
        ));
    };
    let base_arch = base_pkg.arch_id;
    let base_metadata: serde_json::Value = serde_json::from_str(&base_pkg.metadata_json)
        .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));

    let mut roles = BTreeSet::new();
    let mut seen_stored_names = HashSet::new();
    let mut stream_entries = Vec::new();
    let mut stream_sources = Vec::new();
    let mut components = Vec::with_capacity(source_components.len());
    let mut base_stream_entry_by_name = HashMap::new();

    for (component_index, source) in source_components.iter().enumerate() {
        match &source.payload {
            InputPayload::Hfqm(pkg) => {
                let tag = if component_index == 0 {
                    "base".to_string()
                } else {
                    derive_tag(&source.path, &pkg.metadata_json).to_ascii_lowercase()
                };
                validate_role(&tag)?;
                if component_index > 0 && !roles.insert(tag.clone()) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("duplicate component role {tag:?}"),
                    ));
                }
                if tag == "dflash" || pkg.arch_id == 20 {
                    if tag != "dflash" {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "arch-20 HFQM component must have the dflash role",
                        ));
                    }
                    validate_dflash_metadata(&base_metadata, pkg)?;
                } else if tag == "triattn" {
                    validate_triattn_hfqm_metadata(base_arch, &base_metadata, pkg)?;
                } else if component_index > 0
                    && pkg.arch_id != base_arch
                    && pkg.arch_id != HFQM_ARCH_NON_WEIGHT_PACKAGE
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "sidecar {} arch_id {} is incompatible with base arch_id {}",
                            source.path.display(),
                            pkg.arch_id,
                            base_arch
                        ),
                    ));
                }

                let mut original_names = Vec::with_capacity(pkg.tensors().len());
                let mut stored_entries = Vec::with_capacity(pkg.tensors().len());
                for entry in pkg.tensors() {
                    if entry.name.starts_with(HFQM_COMPONENT_PREFIX) {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "source tensor {:?} uses reserved component namespace",
                                entry.name
                            ),
                        ));
                    }
                    let stored_name = if component_index == 0 {
                        entry.name.clone()
                    } else {
                        component_name(&tag, component_index, &entry.name)
                    };
                    if !seen_stored_names.insert(stored_name.clone()) {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("duplicate stored entry {stored_name:?}"),
                        ));
                    }
                    let stream_index = stream_entries.len();
                    stream_entries.push(HfqStreamEntry {
                        name: stored_name.clone(),
                        quant_type: entry.quant_type,
                        shape: entry.shape.clone(),
                        group_size: entry.group_size,
                        data_len: entry.data_size as u64,
                    });
                    stream_sources.push(StreamSource::HfqmTensor {
                        component: component_index,
                        name: entry.name.clone(),
                    });
                    if component_index == 0 {
                        base_stream_entry_by_name.insert(entry.name.clone(), stream_index);
                    }
                    original_names.push(entry.name.clone());
                    stored_entries.push(ComposeStoredEntry {
                        stored_name,
                        original_name: entry.name.clone(),
                        original_offset: entry.data_offset as u64,
                    });
                }

                let mut order: Vec<_> = pkg.tensors().iter().collect();
                order.sort_by_key(|entry| entry.data_offset);
                let mut cursor = 0u64;
                let mut stored_segments = Vec::new();
                for entry in order {
                    let offset = entry.data_offset as u64;
                    if offset < cursor {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "{} contains overlapping tensor ranges",
                                source.path.display()
                            ),
                        ));
                    }
                    if offset > cursor {
                        let length = offset - cursor;
                        let stored_name = component_name(
                            &tag,
                            component_index,
                            &format!("__segment/{}", stored_segments.len()),
                        );
                        if length > u32::MAX as u64 {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "HFQM non-tensor segment exceeds opaque entry shape limit",
                            ));
                        }
                        seen_stored_names.insert(stored_name.clone());
                        stream_entries.push(HfqStreamEntry {
                            name: stored_name.clone(),
                            quant_type: hipfire_quant_format::QuantType::OpaqueBytes.code(),
                            shape: vec![length as u32],
                            group_size: 0,
                            data_len: length,
                        });
                        stream_sources.push(StreamSource::FileRange {
                            path: source.path.clone(),
                            offset: cursor,
                            length,
                        });
                        stored_segments.push(ComposeStoredSegment {
                            stored_name,
                            original_offset: cursor,
                            length,
                        });
                    }
                    cursor = offset.checked_add(entry.data_size as u64).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "HFQM tensor range overflow")
                    })?;
                }
                if cursor < source.byte_len {
                    let length = source.byte_len - cursor;
                    if length > u32::MAX as u64 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "HFQM tail segment exceeds opaque entry shape limit",
                        ));
                    }
                    let stored_name = component_name(
                        &tag,
                        component_index,
                        &format!("__segment/{}", stored_segments.len()),
                    );
                    seen_stored_names.insert(stored_name.clone());
                    stream_entries.push(HfqStreamEntry {
                        name: stored_name.clone(),
                        quant_type: hipfire_quant_format::QuantType::OpaqueBytes.code(),
                        shape: vec![length as u32],
                        group_size: 0,
                        data_len: length,
                    });
                    stream_sources.push(StreamSource::FileRange {
                        path: source.path.clone(),
                        offset: cursor,
                        length,
                    });
                    stored_segments.push(ComposeStoredSegment {
                        stored_name,
                        original_offset: cursor,
                        length,
                    });
                } else if cursor > source.byte_len {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{} tensor data exceeds file length", source.path.display()),
                    ));
                }

                components.push(ComposeComponent {
                    tag,
                    filename: file_name_string(&source.path)?,
                    arch_id: pkg.arch_id,
                    tensors: original_names,
                    metadata_json: pkg.metadata_json.clone(),
                    source_format: "hfqm".to_string(),
                    hfqm_version: Some(pkg.version),
                    byte_len: source.byte_len,
                    sha256: source.sha256.clone(),
                    stored_entries,
                    stored_segments,
                    opaque_entry: None,
                });
            }
            InputPayload::TriaV1(bytes) => {
                let tag = "triattn".to_string();
                if !roles.insert(tag.clone()) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "duplicate triattn component",
                    ));
                }
                let geometry = validate_tria_v1(bytes)?;
                validate_tria_geometry(&base_metadata, geometry)?;
                if bytes.len() > u32::MAX as usize {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "TRIA component exceeds opaque entry shape limit",
                    ));
                }
                let stored_name = component_name(&tag, component_index, "payload");
                if !seen_stored_names.insert(stored_name.clone()) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("duplicate stored entry {stored_name:?}"),
                    ));
                }
                stream_entries.push(HfqStreamEntry {
                    name: stored_name.clone(),
                    quant_type: hipfire_quant_format::QuantType::OpaqueBytes.code(),
                    shape: vec![bytes.len() as u32],
                    group_size: 0,
                    data_len: bytes.len() as u64,
                });
                stream_sources.push(StreamSource::TriaV1 {
                    component: component_index,
                });
                components.push(ComposeComponent {
                    tag,
                    filename: file_name_string(&source.path)?,
                    arch_id: HFQM_ARCH_NON_WEIGHT_PACKAGE,
                    tensors: Vec::new(),
                    metadata_json: String::new(),
                    source_format: "tria-v1".to_string(),
                    hfqm_version: None,
                    byte_len: source.byte_len,
                    sha256: source.sha256.clone(),
                    stored_entries: Vec::new(),
                    stored_segments: Vec::new(),
                    opaque_entry: Some(stored_name),
                });
            }
        }
    }

    // Bundle metadata = base metadata object + the provenance manifest.
    let base_modules = parse_module_table(&base_pkg.metadata_json)?;
    let mut bundle_meta = base_metadata;
    // Lift each sidecar's owned config keys into the bundle's top-level config
    // so the whole bundle advertises the features it actually contains.
    if let serde_json::Value::Object(bundle_obj) = &mut bundle_meta {
        // `HfqPackage::open` has already merged any tail metadata into this
        // object. Its locator points into the original base file and must not
        // survive repacking under new offsets.
        bundle_obj.remove("tail_metadata");
        // Absolute module offsets are rebuilt below after the compose manifest
        // is present and the final index size is known.
        bundle_obj.remove(HFQM_MODULE_TABLE_KEY);
        for comp in components.iter().skip(1) {
            let Some(keys) = role_keys.get(&comp.tag) else {
                continue;
            };
            let Ok(serde_json::Value::Object(comp_obj)) =
                serde_json::from_str::<serde_json::Value>(&comp.metadata_json)
            else {
                continue;
            };
            for k in keys {
                if let Some(v) = comp_obj.get(k) {
                    bundle_obj.insert(k.clone(), v.clone());
                }
            }
        }
    }
    let manifest = ComposeManifest {
        format: HFQM_COMPOSE_FORMAT.to_string(),
        components,
    };
    bundle_meta[HFQM_COMPOSE_KEY] = serde_json::to_value(&manifest).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("serializing manifest: {e}"),
        )
    })?;
    let bundle_meta = compose_bundle_metadata(
        bundle_meta,
        base_modules,
        &base_stream_entry_by_name,
        &stream_entries,
    )?;

    let temp_name = format!(
        ".{}.hipfire-tmp-{}",
        out.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("bundle"),
        std::process::id()
    );
    let temp = out.with_file_name(temp_name);
    if temp.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("temporary output {} already exists", temp.display()),
        ));
    }
    let write_result =
        write_hfqm_package_streaming(&temp, base_arch, &bundle_meta, &stream_entries, |i, w| {
            match &stream_sources[i] {
                StreamSource::HfqmTensor { component, name } => {
                    let InputPayload::Hfqm(pkg) = &source_components[*component].payload else {
                        unreachable!("HFQM stream source must reference an HFQM component")
                    };
                    let info = pkg
                        .find_tensor_info(name)
                        .expect("manifest tensor must exist in source package");
                    let mut file = File::open(pkg.path())?;
                    file.seek(SeekFrom::Start(info.data_offset as u64))?;
                    let copied = io::copy(&mut file.take(info.data_size as u64), w)?;
                    pkg.drop_pages_range(info.data_offset, info.data_size);
                    if copied != info.data_size as u64 {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            format!(
                                "read {copied} of {} bytes for tensor {name:?} from {}",
                                info.data_size,
                                pkg.path().display()
                            ),
                        ));
                    }
                    Ok(())
                }
                StreamSource::FileRange {
                    path,
                    offset,
                    length,
                } => {
                    let mut file = File::open(path)?;
                    file.seek(SeekFrom::Start(*offset))?;
                    let copied = io::copy(&mut file.take(*length), w)?;
                    if copied != *length {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            format!("read {copied} of {length} bytes from {}", path.display()),
                        ));
                    }
                    Ok(())
                }
                StreamSource::TriaV1 { component } => {
                    let InputPayload::TriaV1(bytes) = &source_components[*component].payload else {
                        unreachable!("TRIA stream source must reference TRIA bytes")
                    };
                    w.write_all(bytes)
                }
            }
        });
    if let Err(error) = write_result {
        let _ = std::fs::remove_file(&temp);
        return Err(error);
    }
    if let Err(error) = std::fs::rename(&temp, out) {
        let _ = std::fs::remove_file(&temp);
        return Err(error);
    }

    Ok(out.to_path_buf())
}

/// Split a composed bundle back into its component files under `out_dir`,
/// reproducing each source file (base + sidecars) byte-for-byte from the
/// embedded provenance manifest. Errors if the container has no
/// [`HFQM_COMPOSE_KEY`] manifest. Returns the written file paths.
pub fn decompose_hfq(bundle: &Path, out_dir: &Path) -> io::Result<Vec<PathBuf>> {
    decompose_hfq_with_options(bundle, out_dir, false)
}

/// Decompose with an explicit overwrite policy. The default public wrapper is
/// fail-closed; callers must opt in to replacing existing component files.
pub fn decompose_hfq_with_options(
    bundle: &Path,
    out_dir: &Path,
    overwrite: bool,
) -> io::Result<Vec<PathBuf>> {
    let pkg = HfqPackage::open(bundle)
        .map_err(|e| io::Error::new(e.kind(), format!("opening {}: {e}", bundle.display())))?;
    let meta: serde_json::Value = serde_json::from_str(&pkg.metadata_json).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bundle metadata is not valid JSON: {e}"),
        )
    })?;
    let Some(manifest_value) = meta.get(HFQM_COMPOSE_KEY) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} has no {HFQM_COMPOSE_KEY} manifest; decompose only supports containers produced by `hipfire model compose`",
                bundle.display()
            ),
        ));
    };
    let manifest: ComposeManifest =
        serde_json::from_value(manifest_value.clone()).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid {HFQM_COMPOSE_KEY} manifest: {e}"),
            )
        })?;
    if manifest.format != HFQM_COMPOSE_FORMAT && manifest.format != HFQM_COMPOSE_FORMAT_V1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported compose manifest format {:?}", manifest.format),
        ));
    }

    std::fs::create_dir_all(out_dir)?;
    for comp in &manifest.components {
        if Path::new(&comp.filename)
            .file_name()
            .and_then(|name| name.to_str())
            != Some(comp.filename.as_str())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("component filename {:?} is not a basename", comp.filename),
            ));
        }
        let destination = out_dir.join(&comp.filename);
        if destination.exists() && !overwrite {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("refusing to overwrite {}", destination.display()),
            ));
        }
    }
    let mut written = Vec::with_capacity(manifest.components.len());
    for comp in &manifest.components {
        let destination = out_dir.join(&comp.filename);
        written.push(if manifest.format == HFQM_COMPOSE_FORMAT_V1 {
            write_component_with_options(
                &pkg,
                &destination,
                comp.arch_id,
                &comp.metadata_json,
                &comp.tensors,
                overwrite,
            )?
        } else {
            write_component_v2(&pkg, &destination, comp)?
        });
    }
    Ok(written)
}

fn write_component_v2(
    pkg: &HfqPackage,
    destination: &Path,
    component: &ComposeComponent,
) -> io::Result<PathBuf> {
    let temp_name = format!(
        ".{}.hipfire-tmp-{}",
        destination
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("component"),
        std::process::id()
    );
    let temp = destination.with_file_name(temp_name);
    let result = (|| -> io::Result<()> {
        let mut out = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        match component.source_format.as_str() {
            "tria-v1" => {
                let stored_name = component.opaque_entry.as_deref().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "TRIA component has no opaque entry",
                    )
                })?;
                let entry = pkg.entry(stored_name).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("TRIA payload {stored_name:?} is absent from bundle"),
                    )
                })?;
                if entry.quant_type != hipfire_quant_format::QuantType::OpaqueBytes.code()
                    || entry.data_size as u64 != component.byte_len
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "TRIA opaque entry encoding or length does not match manifest",
                    ));
                }
                let bytes = pkg
                    .blob_data(stored_name)
                    .expect("entry was validated above");
                validate_tria_v1(bytes)?;
                out.write_all(bytes)?;
            }
            "hfqm" => {
                if component.stored_entries.len() != component.tensors.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "HFQM component tensor map length does not match tensor list",
                    ));
                }
                #[derive(Clone, Copy)]
                struct Range<'a> {
                    offset: u64,
                    bytes: &'a [u8],
                }
                let mut ranges = Vec::with_capacity(
                    component.stored_entries.len() + component.stored_segments.len(),
                );
                let mut original_names = BTreeSet::new();
                for stored in &component.stored_entries {
                    if !original_names.insert(stored.original_name.as_str()) {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("duplicate original tensor name {:?}", stored.original_name),
                        ));
                    }
                    let bytes = pkg.blob_data(&stored.stored_name).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "stored tensor {:?} is absent from bundle",
                                stored.stored_name
                            ),
                        )
                    })?;
                    ranges.push(Range {
                        offset: stored.original_offset,
                        bytes,
                    });
                }
                for segment in &component.stored_segments {
                    let entry = pkg.entry(&segment.stored_name).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "stored segment {:?} is absent from bundle",
                                segment.stored_name
                            ),
                        )
                    })?;
                    if entry.quant_type != hipfire_quant_format::QuantType::OpaqueBytes.code()
                        || entry.data_size as u64 != segment.length
                    {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "stored segment {:?} encoding/length mismatch",
                                segment.stored_name
                            ),
                        ));
                    }
                    ranges.push(Range {
                        offset: segment.original_offset,
                        bytes: pkg
                            .blob_data(&segment.stored_name)
                            .expect("segment entry was validated above"),
                    });
                }
                ranges.sort_by_key(|range| range.offset);
                let mut cursor = 0u64;
                for range in &ranges {
                    if range.offset != cursor {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "component byte coverage jumps from {cursor} to {}",
                                range.offset
                            ),
                        ));
                    }
                    cursor = cursor
                        .checked_add(range.bytes.len() as u64)
                        .ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidData, "component range overflow")
                        })?;
                }
                if cursor != component.byte_len {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "component byte coverage {cursor} != declared {}",
                            component.byte_len
                        ),
                    ));
                }
                out.set_len(component.byte_len)?;
                for range in ranges {
                    out.seek(SeekFrom::Start(range.offset))?;
                    out.write_all(range.bytes)?;
                }
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported component source format {other:?}"),
                ));
            }
        }
        out.flush()?;
        drop(out);
        let actual = sha256_file(&temp)?;
        if component.sha256.is_empty() || actual != component.sha256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "component SHA-256 mismatch: expected {}, got {actual}",
                    component.sha256
                ),
            ));
        }
        std::fs::rename(&temp, destination)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result?;
    Ok(destination.to_path_buf())
}

/// Write one component `.hfq` (`tensor_names` pulled verbatim from `pkg`) with
/// the given `arch_id` and metadata. Shared by manifest-based and heuristic
/// decompose. Streams one tensor at a time out of the source mmap.
fn write_component_with_options(
    pkg: &HfqPackage,
    out_path: &Path,
    arch_id: u32,
    metadata_json: &str,
    tensor_names: &[String],
    overwrite: bool,
) -> io::Result<PathBuf> {
    if out_path.exists() && !overwrite {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("refusing to overwrite {}", out_path.display()),
        ));
    }
    let mut stream_entries = Vec::with_capacity(tensor_names.len());
    for name in tensor_names {
        let e = pkg.entry(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor {name:?} absent from bundle"),
            )
        })?;
        stream_entries.push(HfqStreamEntry {
            name: e.name.clone(),
            quant_type: e.quant_type,
            shape: e.shape.clone(),
            group_size: e.group_size,
            data_len: e.data_size as u64,
        });
    }
    let temp_name = format!(
        ".{}.hipfire-tmp-{}",
        out_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("component"),
        std::process::id()
    );
    let temp = out_path.with_file_name(temp_name);
    let write_result =
        write_hfqm_package_streaming(&temp, arch_id, metadata_json, &stream_entries, |i, w| {
            let data = pkg
                .blob_data(&tensor_names[i])
                .expect("tensor validated present above");
            w.write_all(data)
        });
    if let Err(error) = write_result {
        let _ = std::fs::remove_file(&temp);
        return Err(error);
    }
    if let Err(error) = std::fs::rename(&temp, out_path) {
        let _ = std::fs::remove_file(&temp);
        return Err(error);
    }
    Ok(out_path.to_path_buf())
}

/// True if `tensor_name` looks like it belongs to `role` (best-effort prefix
/// match used only by [`decompose_hfq_infer`]).
fn role_matches(role: &str, tensor_name: &str) -> bool {
    let n = tensor_name.to_ascii_lowercase();
    match role {
        "mtp" => n.contains("mtp"),
        "dflash" => n.contains("dflash") || n.contains("draft"),
        "triattn" => n.contains("triattn"),
        "vl" => [
            "vision",
            "visual",
            "siglip",
            "mm_projector",
            "multi_modal_projector",
        ]
        .iter()
        .any(|p| n.contains(p)),
        "calib" | "hessian" => {
            n.contains("calib") || n.contains("hessian") || n.contains("imatrix")
        }
        _ => false,
    }
}

/// All known role tokens present in a bundle filename's dot-groups, in order
/// (e.g. `Model.mtp.vl.mq4.hfq` -> `["mtp", "vl"]`).
fn role_tags_from_filename(path: &Path) -> Vec<String> {
    let fname = path
        .file_name()
        .map(|s| s.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let stem = fname.strip_suffix(".hfq").unwrap_or(&fname).to_string();
    stem.split('.')
        .filter(|seg| KNOWN_ROLES.contains(seg))
        .map(|s| s.to_string())
        .collect()
}

/// Roles inferred from the bundle's *tensor names* alone, for legacy bundles
/// whose filename carries no role dot-groups. Restricted to the model-feature
/// roles that leave an unambiguous tensor-name fingerprint; the calibration
/// roles (`calib`/`hessian`/`imatrix`) are mutually indistinguishable by tensor
/// name (see [`role_matches`]) and so are only inferable from the filename.
/// Returns the matching roles in a stable order (matching partition precedence).
fn roles_from_tensor_names(pkg: &HfqPackage) -> Vec<String> {
    const TENSOR_INFERABLE_ROLES: &[&str] = &["mtp", "dflash", "triattn", "vl"];
    TENSOR_INFERABLE_ROLES
        .iter()
        .filter(|role| pkg.entries().iter().any(|e| role_matches(role, &e.name)))
        .map(|s| s.to_string())
        .collect()
}

/// Bundle filename with the given role dot-groups removed (case-insensitive):
/// `Model.mtp.vl.mq4.hfq` + `[mtp, vl]` -> `Model.mq4.hfq`.
fn strip_role_groups(fname: &str, roles: &[String]) -> String {
    let stem = fname.strip_suffix(".hfq").unwrap_or(fname);
    let kept: Vec<&str> = stem
        .split('.')
        .filter(|seg| !roles.iter().any(|r| r.eq_ignore_ascii_case(seg)))
        .collect();
    format!("{}.hfq", kept.join("."))
}

/// Best-effort split of a bundle that has NO [`HFQM_COMPOSE_KEY`] manifest,
/// driven by the role dot-groups in the bundle filename plus tensor-name prefix
/// matching ([`role_matches`]). Each declared role claims its matching tensors
/// (first role wins); the remainder become the base. This is LOSSY — output
/// files are not guaranteed byte-identical to any original sidecars (metadata
/// and per-sidecar `arch_id` are synthesized), unlike manifest-based decompose.
///
/// Legacy bundles whose filename carries no role dot-groups fall back to
/// inferring roles from tensor names alone ([`roles_from_tensor_names`]).
/// Errors only if neither the filename nor the tensor names reveal any role.
pub fn decompose_hfq_infer(bundle: &Path, out_dir: &Path) -> io::Result<Vec<PathBuf>> {
    decompose_hfq_infer_with_config_keys(bundle, out_dir, &RoleConfigKeys::new())
}

/// As [`decompose_hfq_infer`], but moves each split-off role's owned config
/// keys (per `role_keys`) OUT of the base metadata and INTO that role's
/// sidecar, so the reconstructed base never advertises a feature whose tensors
/// were carved away (e.g. a `vision_config` left behind with no vision tensors).
pub fn decompose_hfq_infer_with_config_keys(
    bundle: &Path,
    out_dir: &Path,
    role_keys: &RoleConfigKeys,
) -> io::Result<Vec<PathBuf>> {
    decompose_hfq_infer_with_config_keys_options(bundle, out_dir, role_keys, false)
}

/// Heuristic decompose with an explicit overwrite policy.
pub fn decompose_hfq_infer_with_config_keys_options(
    bundle: &Path,
    out_dir: &Path,
    role_keys: &RoleConfigKeys,
    overwrite: bool,
) -> io::Result<Vec<PathBuf>> {
    let pkg = HfqPackage::open(bundle)
        .map_err(|e| io::Error::new(e.kind(), format!("opening {}: {e}", bundle.display())))?;
    let mut roles = role_tags_from_filename(bundle);
    if roles.is_empty() {
        // Legacy bundle: no role dot-groups in the filename. Recover the split
        // from tensor-name fingerprints instead.
        roles = roles_from_tensor_names(&pkg);
    }
    if roles.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} has no role features in its filename and no role-tagged tensors (mtp/dflash/triattn/vl); heuristic decompose needs role dot-groups (e.g. .mtp.vl), tensor-name role fingerprints, or a composed bundle with a {HFQM_COMPOSE_KEY} manifest",
                bundle.display()
            ),
        ));
    }

    // Partition tensors: each declared role claims its matching, still-unclaimed
    // tensors (first role wins); everything left is the base.
    let mut claimed = vec![false; pkg.entries().len()];
    let mut role_tensors: Vec<(String, Vec<String>)> = Vec::new();
    for role in &roles {
        let mut names = Vec::new();
        for (i, e) in pkg.entries().iter().enumerate() {
            if !claimed[i] && role_matches(role, &e.name) {
                claimed[i] = true;
                names.push(e.name.clone());
            }
        }
        if !names.is_empty() {
            role_tensors.push((role.clone(), names));
        }
    }
    if role_tensors.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "no tensors in {} matched any declared role (mtp/dflash/triattn/vl/calib); cannot infer a split",
                bundle.display()
            ),
        ));
    }
    let base_names: Vec<String> = pkg
        .entries()
        .iter()
        .enumerate()
        .filter(|(i, _)| !claimed[*i])
        .map(|(_, e)| e.name.clone())
        .collect();

    let bundle_fname = bundle
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "bundle.hfq".to_string());
    let base_fname = strip_role_groups(&bundle_fname, &roles);
    let base_stem = base_fname.strip_suffix(".hfq").unwrap_or(&base_fname);
    // Sidecars are `<family>.<role>.hfq`, where family drops the quant token (the
    // base stem's last dot-group) — matching the compose naming (base
    // `Model.mq4.hfq` + `Model.mtp.hfq` <-> `Model.mtp.mq4.hfq`).
    let family_stem = base_stem
        .rsplit_once('.')
        .map(|(head, _)| head)
        .unwrap_or(base_stem);

    let mut destinations =
        Vec::with_capacity(role_tensors.len() + usize::from(!base_names.is_empty()));
    if !base_names.is_empty() {
        destinations.push(out_dir.join(&base_fname));
    }
    destinations.extend(
        role_tensors
            .iter()
            .map(|(role, _)| out_dir.join(format!("{family_stem}.{role}.hfq"))),
    );
    if !overwrite {
        if let Some(existing) = destinations.iter().find(|path| path.exists()) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("refusing to overwrite {}", existing.display()),
            ));
        }
    }
    std::fs::create_dir_all(out_dir)?;

    // Move each split-off role's owned config keys out of the base metadata and
    // stash them per role, so the base no longer advertises carved-away features
    // and each sidecar carries its own config.
    let mut base_obj = match serde_json::from_str::<serde_json::Value>(&pkg.metadata_json) {
        Ok(serde_json::Value::Object(m)) => m,
        _ => serde_json::Map::new(),
    };
    let mut moved: std::collections::BTreeMap<String, serde_json::Map<String, serde_json::Value>> =
        std::collections::BTreeMap::new();
    for (role, _) in &role_tensors {
        if let Some(keys) = role_keys.get(role) {
            let dst = moved.entry(role.clone()).or_default();
            for k in keys {
                if let Some(v) = base_obj.remove(k) {
                    dst.insert(k.clone(), v);
                }
            }
        }
    }
    let base_meta_json = serde_json::to_string(&serde_json::Value::Object(base_obj))
        .unwrap_or_else(|_| pkg.metadata_json.clone());

    let mut written = Vec::new();
    if !base_names.is_empty() {
        written.push(write_component_with_options(
            &pkg,
            &out_dir.join(&base_fname),
            pkg.arch_id,
            &base_meta_json,
            &base_names,
            overwrite,
        )?);
    }
    for (role, names) in &role_tensors {
        let mut side_obj = serde_json::Map::new();
        side_obj.insert("role".to_string(), serde_json::Value::String(role.clone()));
        side_obj.insert("arch_id".to_string(), serde_json::json!(pkg.arch_id));
        side_obj.insert(
            "hipfire_infer".to_string(),
            serde_json::Value::String("heuristic.v1".to_string()),
        );
        if let Some(mv) = moved.get(role) {
            for (k, v) in mv {
                side_obj.insert(k.clone(), v.clone());
            }
        }
        let side_meta = serde_json::Value::Object(side_obj).to_string();
        written.push(write_component_with_options(
            &pkg,
            &out_dir.join(format!("{family_stem}.{role}.hfq")),
            pkg.arch_id,
            &side_meta,
            names,
            overwrite,
        )?);
    }
    Ok(written)
}

/// Decompose a bundle, preferring the lossless manifest path. See
/// [`decompose_hfq_auto_with_config_keys`]; this passes an empty
/// [`RoleConfigKeys`] (no config-key move on the heuristic path).
pub fn decompose_hfq_auto(bundle: &Path, out_dir: &Path, infer: bool) -> io::Result<Vec<PathBuf>> {
    decompose_hfq_auto_with_config_keys(bundle, out_dir, infer, &RoleConfigKeys::new())
}

/// As [`decompose_hfq_auto`], threading `role_keys` into the heuristic path so a
/// carved base drops the config keys its split-off sidecars now own. The lossless
/// manifest path is unaffected: it reproduces each component's stored metadata
/// verbatim, which is already role-consistent by construction.
pub fn decompose_hfq_auto_with_config_keys(
    bundle: &Path,
    out_dir: &Path,
    infer: bool,
    role_keys: &RoleConfigKeys,
) -> io::Result<Vec<PathBuf>> {
    decompose_hfq_auto_with_config_keys_options(bundle, out_dir, infer, role_keys, false)
}

/// Automatic manifest/heuristic decompose with an explicit overwrite policy.
pub fn decompose_hfq_auto_with_config_keys_options(
    bundle: &Path,
    out_dir: &Path,
    infer: bool,
    role_keys: &RoleConfigKeys,
    overwrite: bool,
) -> io::Result<Vec<PathBuf>> {
    let has_manifest = HfqPackage::open(bundle)
        .ok()
        .and_then(|pkg| serde_json::from_str::<serde_json::Value>(&pkg.metadata_json).ok())
        .map(|v| v.get(HFQM_COMPOSE_KEY).is_some())
        .unwrap_or(false);
    if has_manifest {
        decompose_hfq_with_options(bundle, out_dir, overwrite)
    } else if infer {
        decompose_hfq_infer_with_config_keys_options(bundle, out_dir, role_keys, overwrite)
    } else {
        decompose_hfq_with_options(bundle, out_dir, overwrite) // reuses the clear "no manifest" error
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn scratch_dir() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("hfq_compose_{}_{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn mem_tensor(name: &str, data: Vec<u8>) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type: 1,
            shape: vec![1, data.len() as u32],
            group_size: 0,
            data,
        }
    }

    fn tria_v1_bytes(
        n_layers: u32,
        n_heads: u32,
        head_dim: u32,
        rope_theta: f32,
        partial_rotary_factor: f32,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"TRIA");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&n_layers.to_le_bytes());
        bytes.extend_from_slice(&n_heads.to_le_bytes());
        bytes.extend_from_slice(&head_dim.to_le_bytes());
        bytes.extend_from_slice(&rope_theta.to_le_bytes());
        bytes.extend_from_slice(&partial_rotary_factor.to_le_bytes());
        let centers = n_layers as usize * n_heads as usize * head_dim as usize / 2;
        for index in 0..centers {
            for value in [index as f32 + 0.25, index as f32 - 0.5, 1.0] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn infer_splits_manifestless_bundle_by_filename_roles() {
        let dir = scratch_dir();
        // A bundle with NO hipfire_compose manifest, name declaring `.mtp`.
        let bundle = dir.join("Model.mtp.mq4.hfq");
        write_hfqm_package_mem(
            &bundle,
            5,
            r#"{"arch_id":5}"#,
            &[
                mem_tensor("model.embed.weight", vec![1, 2, 3, 4]),
                mem_tensor("model.mtp.head.weight", vec![9, 8, 7]),
            ],
        )
        .unwrap();

        // Without --infer, a manifest-less bundle is a hard error.
        assert!(decompose_hfq(&bundle, &dir.join("no")).is_err());

        // --infer splits on the `.mtp` filename role + tensor-name prefix.
        let out = dir.join("out");
        let written = decompose_hfq_infer(&bundle, &out).unwrap();
        assert_eq!(written.len(), 2);
        let base = HfqPackage::open(&out.join("Model.mq4.hfq")).unwrap();
        assert!(base.entry("model.embed.weight").is_some());
        assert!(base.entry("model.mtp.head.weight").is_none());
        let mtp = HfqPackage::open(&out.join("Model.mtp.hfq")).unwrap();
        assert!(mtp.entry("model.mtp.head.weight").is_some());
        assert!(mtp.entry("model.embed.weight").is_none());
        assert!(mtp.metadata_json.contains("heuristic.v1"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn infer_errors_when_no_roles_in_filename_or_tensors() {
        let dir = scratch_dir();
        let bundle = dir.join("Model.mq4.hfq"); // no role dot-groups
                                                // Tensor names carry no role fingerprint either, so nothing to split.
        write_hfqm_package_mem(&bundle, 5, "{}", &[mem_tensor("a", vec![1])]).unwrap();
        let err = decompose_hfq_infer(&bundle, &dir.join("out")).unwrap_err();
        assert!(err.to_string().contains("no role-tagged tensors"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn infer_splits_legacy_bundle_by_tensor_names() {
        let dir = scratch_dir();
        // Legacy bundle: plain filename, NO role dot-groups and NO manifest,
        // but a vision tensor betrays a `vl` sidecar hiding inside.
        let bundle = dir.join("Model.mq4.hfq");
        write_hfqm_package_mem(
            &bundle,
            5,
            r#"{"arch_id":5}"#,
            &[
                mem_tensor("model.embed.weight", vec![1, 2, 3, 4]),
                mem_tensor("model.vision.patch_embed.weight", vec![9, 8, 7]),
            ],
        )
        .unwrap();

        // Filename declares no roles, so the split is recovered from tensor names.
        let out = dir.join("out");
        let written = decompose_hfq_infer(&bundle, &out).unwrap();
        assert_eq!(written.len(), 2);
        // Base keeps the original (unstripped) filename; the vl tensor is carved
        // out into `<family>.vl.hfq`.
        let base = HfqPackage::open(&out.join("Model.mq4.hfq")).unwrap();
        assert!(base.entry("model.embed.weight").is_some());
        assert!(base.entry("model.vision.patch_embed.weight").is_none());
        let vl = HfqPackage::open(&out.join("Model.vl.hfq")).unwrap();
        assert!(vl.entry("model.vision.patch_embed.weight").is_some());
        assert!(vl.entry("model.embed.weight").is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compose_then_decompose_round_trips_byte_identical() {
        let dir = scratch_dir();
        let base = dir.join("Model.mq4.hfq");
        let mtp = dir.join("Model.mtp.hfq");
        let bundle = dir.join("Model.mtp.mq4.hfq");

        let base_meta = r#"{"arch_id":5,"role":"base"}"#;
        let mtp_meta = r#"{"arch_id":5,"role":"mtp"}"#;
        write_hfqm_package_mem(
            &base,
            5,
            base_meta,
            &[mem_tensor("model.embed.weight", vec![1, 2, 3, 4])],
        )
        .unwrap();
        write_hfqm_package_mem(
            &mtp,
            5,
            mtp_meta,
            &[mem_tensor("mtp.head.weight", vec![9, 8, 7])],
        )
        .unwrap();

        compose_hfq(&[base.clone(), mtp.clone()], &bundle).unwrap();

        // Bundle holds the union of tensors + a valid manifest.
        let pkg = HfqPackage::open(&bundle).unwrap();
        assert_eq!(pkg.arch_id, 5);
        assert!(pkg.entry("model.embed.weight").is_some());
        assert!(pkg
            .entry("__hipfire_component/mtp/1/mtp.head.weight")
            .is_some());
        let meta: serde_json::Value = serde_json::from_str(&pkg.metadata_json).unwrap();
        assert_eq!(meta["role"], "base");
        let manifest: ComposeManifest =
            serde_json::from_value(meta[HFQM_COMPOSE_KEY].clone()).unwrap();
        assert_eq!(manifest.format, HFQM_COMPOSE_FORMAT);
        assert_eq!(manifest.components.len(), 2);
        assert_eq!(manifest.components[0].tag, "base");
        assert_eq!(manifest.components[1].tag, "mtp");
        assert_eq!(manifest.components[1].source_format, "hfqm");
        assert_eq!(
            manifest.components[1].stored_entries[0].original_name,
            "mtp.head.weight"
        );

        // Decompose reproduces both source files byte-for-byte.
        let out = dir.join("out");
        let written = decompose_hfq(&bundle, &out).unwrap();
        assert_eq!(written.len(), 2);
        assert_eq!(
            std::fs::read(out.join("Model.mq4.hfq")).unwrap(),
            std::fs::read(&base).unwrap()
        );
        assert_eq!(
            std::fs::read(out.join("Model.mtp.hfq")).unwrap(),
            std::fs::read(&mtp).unwrap()
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compose_rejects_arch_mismatch() {
        let dir = scratch_dir();
        let base = dir.join("base.hfq");
        let side = dir.join("side.hfq");
        write_hfqm_package_mem(&base, 5, "{}", &[mem_tensor("a", vec![1])]).unwrap();
        write_hfqm_package_mem(&side, 7, "{}", &[mem_tensor("b", vec![2])]).unwrap();
        let err = compose_hfq(&[base, side], &dir.join("bundle.hfq")).unwrap_err();
        assert!(err.to_string().contains("incompatible"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compose_allows_non_weight_arch_zero_sidecar() {
        let dir = scratch_dir();
        let base = dir.join("base.hfq");
        let side = dir.join("side.jinja.hfq");
        write_hfqm_package_mem(&base, 5, "{}", &[mem_tensor("a", vec![1])]).unwrap();
        write_hfqm_package_mem(
            &side,
            HFQM_ARCH_NON_WEIGHT_PACKAGE,
            "{}",
            &[mem_tensor("b", vec![2])],
        )
        .unwrap();
        compose_hfq(&[base, side], &dir.join("bundle.hfq")).unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compose_accepts_arch20_dflash_with_colliding_names_and_round_trips() {
        let dir = scratch_dir();
        let base = dir.join("Qwen3.5-9B.oq4.hfq");
        let dflash = dir.join("Qwen3.5-9B.dflash.oq4+.hfq");
        let bundle = dir.join("Qwen3.5-9B.dflash.oq4.hfq");
        let base_meta = r#"{"config":{"hidden_size":4,"vocab_size":32,"num_hidden_layers":4,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":4,"rope_theta":10000.0}}"#;
        let draft_meta = r#"{"architecture":"dflash","config":{"hidden_size":4,"vocab_size":32,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":2,"rope_theta":10000.0},"dflash":{"block_size":4,"num_hidden_layers":2,"num_target_layers":4,"target_layer_ids":[0,3]}}"#;
        write_hfqm_package_mem(
            &base,
            5,
            base_meta,
            &[mem_tensor("layers.0.self_attn.q_proj.weight", vec![1, 2])],
        )
        .unwrap();
        write_hfqm_package_mem(
            &dflash,
            20,
            draft_meta,
            &[mem_tensor("layers.0.self_attn.q_proj.weight", vec![9, 8])],
        )
        .unwrap();

        compose_hfq(&[base.clone(), dflash.clone()], &bundle).unwrap();
        let pkg = HfqPackage::open(&bundle).unwrap();
        assert_eq!(
            pkg.blob_data("layers.0.self_attn.q_proj.weight"),
            Some(&[1, 2][..])
        );
        assert_eq!(
            pkg.blob_data("__hipfire_component/dflash/1/layers.0.self_attn.q_proj.weight"),
            Some(&[9, 8][..])
        );
        let manifest = compose_manifest(&pkg).unwrap().unwrap();
        let view = component_view(&pkg, &manifest, "dflash").unwrap().unwrap();
        assert_eq!(view.arch_id(), 20);
        assert_eq!(
            view.blob_data("layers.0.self_attn.q_proj.weight"),
            Some(&[9, 8][..])
        );

        let out = dir.join("out");
        decompose_hfq(&bundle, &out).unwrap();
        assert_eq!(
            std::fs::read(out.join(base.file_name().unwrap())).unwrap(),
            std::fs::read(&base).unwrap()
        );
        assert_eq!(
            std::fs::read(out.join(dflash.file_name().unwrap())).unwrap(),
            std::fs::read(&dflash).unwrap()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compose_accepts_raw_tria_and_round_trips_exact_bytes() {
        let dir = scratch_dir();
        let base = dir.join("Qwen3.5-2B.oq4.hfq");
        let tria = dir.join("Qwen3.5-2B.triattn.hfq");
        let bundle = dir.join("Qwen3.5-2B.triattn.oq4.hfq");
        write_hfqm_package_mem(
            &base,
            5,
            r#"{"config":{"num_hidden_layers":1,"num_attention_heads":1,"head_dim":2,"rope_theta":10000.0,"partial_rotary_factor":1.0}}"#,
            &[mem_tensor("model.embed_tokens.weight", vec![1, 2, 3])],
        )
        .unwrap();
        let tria_bytes = tria_v1_bytes(1, 1, 2, 10_000.0, 1.0);
        std::fs::write(&tria, &tria_bytes).unwrap();

        compose_hfq(&[base.clone(), tria.clone()], &bundle).unwrap();
        let pkg = HfqPackage::open(&bundle).unwrap();
        let metadata: serde_json::Value = serde_json::from_str(&pkg.metadata_json).unwrap();
        let manifest: ComposeManifest =
            serde_json::from_value(metadata[HFQM_COMPOSE_KEY].clone()).unwrap();
        assert_eq!(manifest.components[1].source_format, "tria-v1");
        assert_eq!(
            pkg.blob_data(manifest.components[1].opaque_entry.as_deref().unwrap()),
            Some(tria_bytes.as_slice())
        );
        let view = component_view(&pkg, &manifest, "triattn").unwrap().unwrap();
        assert_eq!(view.source_format(), "tria-v1");
        assert_eq!(view.opaque_bytes().unwrap(), Some(tria_bytes.as_slice()));
        assert_eq!(
            hipfire_runtime::triattn::TriAttnCenters::from_bytes(
                view.opaque_bytes().unwrap().unwrap()
            )
            .unwrap()
            .n_layers,
            1
        );

        let out = dir.join("out");
        decompose_hfq(&bundle, &out).unwrap();
        assert_eq!(
            std::fs::read(out.join(tria.file_name().unwrap())).unwrap(),
            tria_bytes
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compose_accepts_heterogeneous_triattn_hfqm_and_runtime_reads_embedded() {
        use hipfire_runtime::triattn::{
            BandCenter, TriAttnArtifact, TriAttnAttentionKind, TriAttnContextPolicy,
            TriAttnLayerRecord, TriAttnPackageMetadata, TriAttnRopeConvention,
            TRIATTN_ARTIFACT_KIND, TRIATTN_HFQM_SCHEMA,
        };

        let dir = scratch_dir();
        let base = dir.join("Gemma-4-test.oq4.hfq");
        let tria = dir.join("Gemma-4-test.triattn.hfq");
        let bundle = dir.join("Gemma-4-test.triattn.oq4.hfq");
        write_hfqm_package_mem(
            &base,
            24,
            r#"{"config":{"num_hidden_layers":3}}"#,
            &[mem_tensor("model.embed_tokens.weight", vec![1, 2, 3])],
        )
        .unwrap();
        let records = vec![
            TriAttnLayerRecord {
                physical_layer: 0,
                attention_kind: TriAttnAttentionKind::Full,
                q_heads: 2,
                kv_heads: 1,
                head_dim: 4,
                rotary_dim: 4,
                rope_theta: 10_000.0,
                rope_convention: TriAttnRopeConvention::Interleaved,
                context_policy: TriAttnContextPolicy::Full,
                sliding_window: None,
                kv_producer: None,
                center_tensor: "triattn.layers.0.centers".to_string(),
                center_offset: 0,
                center_count: 4,
                sample_count: 32,
            },
            TriAttnLayerRecord {
                physical_layer: 2,
                attention_kind: TriAttnAttentionKind::Sliding,
                q_heads: 4,
                kv_heads: 2,
                head_dim: 8,
                rotary_dim: 4,
                rope_theta: 1_000_000.0,
                rope_convention: TriAttnRopeConvention::Interleaved,
                context_policy: TriAttnContextPolicy::Sliding,
                sliding_window: Some(4096),
                kv_producer: Some(0),
                center_tensor: "triattn.layers.2.centers".to_string(),
                center_offset: 0,
                center_count: 16,
                sample_count: 32,
            },
        ];
        let artifact = TriAttnArtifact {
            metadata: TriAttnPackageMetadata {
                artifact_kind: TRIATTN_ARTIFACT_KIND.to_string(),
                package_schema: TRIATTN_HFQM_SCHEMA.to_string(),
                model_arch_id: 24,
                model_layers: 3,
                model_fingerprint: "sha256:model".to_string(),
                corpus_fingerprint: "sha256:corpus".to_string(),
                adapter: "gemma4-cask-v1".to_string(),
                engine: "hipfire-cask-v1".to_string(),
                layers: records,
            },
            centers: vec![
                vec![BandCenter::default(); 4],
                vec![BandCenter::default(); 16],
            ],
        };
        artifact.save_hfqm(&tria).unwrap();

        check_compose_inputs(&[base.clone(), tria.clone()]).unwrap();
        compose_hfq(&[base, tria.clone()], &bundle).unwrap();
        let runtime_file = hipfire_runtime::hfq::HfqFile::open(&bundle).unwrap();
        let manifest = hipfire_runtime::hfq_compose::compose_manifest_from_metadata(
            &runtime_file.metadata_json,
        )
        .unwrap()
        .unwrap();
        let view =
            hipfire_runtime::hfq_compose::file_component_view(&runtime_file, &manifest, "triattn")
                .unwrap()
                .unwrap();
        view.verify_digest().unwrap();
        assert_eq!(TriAttnArtifact::from_source(&view).unwrap(), artifact);

        let out = dir.join("out");
        decompose_hfq(&bundle, &out).unwrap();
        assert_eq!(
            std::fs::read(out.join(tria.file_name().unwrap())).unwrap(),
            std::fs::read(&tria).unwrap()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compose_namespaces_duplicate_tensor_names() {
        let dir = scratch_dir();
        let base = dir.join("base.hfq");
        let side = dir.join("side.hfq");
        write_hfqm_package_mem(&base, 5, "{}", &[mem_tensor("dup", vec![1])]).unwrap();
        write_hfqm_package_mem(&side, 5, "{}", &[mem_tensor("dup", vec![2])]).unwrap();
        let bundle = dir.join("bundle.hfq");
        compose_hfq(&[base.clone(), side.clone()], &bundle).unwrap();
        let pkg = HfqPackage::open(&bundle).unwrap();
        assert_eq!(pkg.blob_data("dup"), Some(&[1][..]));
        assert_eq!(
            pkg.blob_data("__hipfire_component/sidecar/1/dup"),
            Some(&[2][..])
        );
        let out = dir.join("out");
        decompose_hfq(&bundle, &out).unwrap();
        assert_eq!(
            std::fs::read(out.join("base.hfq")).unwrap(),
            std::fs::read(base).unwrap()
        );
        assert_eq!(
            std::fs::read(out.join("side.hfq")).unwrap(),
            std::fs::read(side).unwrap()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compose_check_accepts_three_way_gemma_bundle_and_round_trips() {
        let dir = scratch_dir();
        let base = dir.join("Gemma-4-26B-A4B-it.oq4.25++.hfq");
        let dflash = dir.join("Gemma-4-26B-A4B-it.dflash.oq4+.hfq");
        let tria = dir.join("Gemma-4-26B-A4B-it.triattn.hfq");
        let bundle = dir.join("Gemma-4-26B-A4B-it.dflash.triattn.oq4.25++.hfq");
        let base_meta = r#"{"config":{"hidden_size":4,"vocab_size":32,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,"rope_theta":10000.0,"partial_rotary_factor":1.0}}"#;
        let draft_meta = r#"{"architecture":"dflash","config":{"hidden_size":4,"vocab_size":32,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,"rope_theta":10000.0},"dflash":{"num_hidden_layers":1,"hidden_size":4,"intermediate_size":8,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,"vocab_size":32,"rms_norm_eps":0.000001,"rope_theta":10000.0,"block_size":4,"mask_token_id":0,"target_layer_ids":[0],"num_target_layers":1}}"#;
        write_hfqm_package_mem(
            &base,
            24,
            base_meta,
            &[mem_tensor("layers.0.self_attn.q_proj.weight", vec![1, 2])],
        )
        .unwrap();
        write_hfqm_package_mem(
            &dflash,
            20,
            draft_meta,
            &[mem_tensor("layers.0.self_attn.q_proj.weight", vec![9, 8])],
        )
        .unwrap();
        std::fs::write(&tria, tria_v1_bytes(1, 1, 2, 10_000.0, 1.0)).unwrap();

        let report = check_compose_inputs(&[base.clone(), dflash.clone(), tria.clone()]).unwrap();
        assert!(report.compatible);
        assert_eq!(report.bundle_arch_id, 24);
        assert_eq!(
            report
                .components
                .iter()
                .map(|component| component.role.as_str())
                .collect::<Vec<_>>(),
            ["base", "dflash", "triattn"]
        );

        compose_hfq(&[base.clone(), dflash.clone(), tria.clone()], &bundle).unwrap();
        let runtime_file = hipfire_runtime::hfq::HfqFile::open(&bundle).unwrap();
        let runtime_manifest = hipfire_runtime::hfq_compose::compose_manifest_from_metadata(
            &runtime_file.metadata_json,
        )
        .unwrap()
        .unwrap();
        let runtime_dflash = hipfire_runtime::hfq_compose::file_component_view(
            &runtime_file,
            &runtime_manifest,
            "dflash",
        )
        .unwrap()
        .unwrap();
        runtime_dflash.verify_digest().unwrap();
        let embedded_config =
            hipfire_runtime::dflash::DflashConfig::from_source(&runtime_dflash).unwrap();
        let standalone_file = hipfire_runtime::hfq::HfqFile::open(&dflash).unwrap();
        let standalone_config =
            hipfire_runtime::dflash::DflashConfig::from_hfq(&standalone_file).unwrap();
        assert_eq!(embedded_config.block_size, standalone_config.block_size);
        assert_eq!(
            embedded_config.target_layer_ids,
            standalone_config.target_layer_ids
        );
        assert_eq!(
            runtime_dflash
                .tensor_data("layers.0.self_attn.q_proj.weight")
                .unwrap()
                .1,
            &[9, 8]
        );
        let runtime_tria = hipfire_runtime::hfq_compose::file_component_view(
            &runtime_file,
            &runtime_manifest,
            "triattn",
        )
        .unwrap()
        .unwrap();
        runtime_tria.verify_digest().unwrap();
        assert_eq!(
            hipfire_runtime::triattn::TriAttnCenters::from_bytes(
                runtime_tria.opaque_bytes().unwrap().unwrap()
            )
            .unwrap()
            .n_layers,
            1
        );
        drop(runtime_file);
        let out = dir.join("out");
        decompose_hfq(&bundle, &out).unwrap();
        for source in [&base, &dflash, &tria] {
            assert_eq!(
                std::fs::read(out.join(source.file_name().unwrap())).unwrap(),
                std::fs::read(source).unwrap()
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compose_rejects_duplicate_roles_reserved_namespace_and_geometry_mismatch() {
        let dir = scratch_dir();
        let base = dir.join("Model.oq4.hfq");
        write_hfqm_package_mem(
            &base,
            5,
            r#"{"config":{"num_hidden_layers":1,"num_attention_heads":1,"head_dim":2}}"#,
            &[mem_tensor("base.weight", vec![1])],
        )
        .unwrap();
        let mtp_a = dir.join("Model-a.mtp.hfq");
        let mtp_b = dir.join("Model-b.mtp.hfq");
        write_hfqm_package_mem(&mtp_a, 5, r#"{"role":"mtp"}"#, &[mem_tensor("a", vec![2])])
            .unwrap();
        write_hfqm_package_mem(&mtp_b, 5, r#"{"role":"mtp"}"#, &[mem_tensor("b", vec![3])])
            .unwrap();
        let error = check_compose_inputs(&[base.clone(), mtp_a, mtp_b]).unwrap_err();
        assert!(error.to_string().contains("duplicate component role"));

        let reserved = dir.join("Model.vl.hfq");
        write_hfqm_package_mem(
            &reserved,
            5,
            r#"{"role":"vl"}"#,
            &[mem_tensor("__hipfire_component/hostile", vec![4])],
        )
        .unwrap();
        let error = compose_hfq(&[base.clone(), reserved], &dir.join("reserved.hfq")).unwrap_err();
        assert!(error.to_string().contains("reserved component namespace"));

        let tria = dir.join("Model.triattn.hfq");
        std::fs::write(&tria, tria_v1_bytes(2, 1, 2, 10_000.0, 1.0)).unwrap();
        let error = compose_hfq(&[base, tria], &dir.join("geometry.hfq")).unwrap_err();
        assert!(error.to_string().contains("num_hidden_layers"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn decompose_v2_rejects_digest_corruption() {
        let dir = scratch_dir();
        let base = dir.join("Model.oq4.hfq");
        let mtp = dir.join("Model.mtp.hfq");
        let bundle = dir.join("Model.mtp.oq4.hfq");
        write_hfqm_package_mem(&base, 5, "{}", &[mem_tensor("base", vec![1, 2])]).unwrap();
        write_hfqm_package_mem(
            &mtp,
            5,
            r#"{"role":"mtp"}"#,
            &[mem_tensor("mtp.weight", vec![7, 8])],
        )
        .unwrap();
        compose_hfq(&[base, mtp], &bundle).unwrap();
        let pkg = HfqPackage::open(&bundle).unwrap();
        let entry = pkg
            .entry("__hipfire_component/mtp/1/mtp.weight")
            .unwrap()
            .clone();
        drop(pkg);
        let mut file = OpenOptions::new().write(true).open(&bundle).unwrap();
        file.seek(SeekFrom::Start(entry.data_offset as u64))
            .unwrap();
        file.write_all(&[0xff]).unwrap();
        drop(file);
        let error = decompose_hfq(&bundle, &dir.join("out")).unwrap_err();
        assert!(error.to_string().contains("SHA-256 mismatch"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn decompose_v1_manifest_remains_byte_identical() {
        let dir = scratch_dir();
        let base = dir.join("Legacy.mq4.hfq");
        let mtp = dir.join("Legacy.mtp.hfq");
        write_hfqm_package_mem(
            &base,
            5,
            r#"{"role":"base"}"#,
            &[mem_tensor("base", vec![1, 2])],
        )
        .unwrap();
        write_hfqm_package_mem(
            &mtp,
            5,
            r#"{"role":"mtp"}"#,
            &[mem_tensor("mtp", vec![3, 4])],
        )
        .unwrap();
        let manifest = ComposeManifest {
            format: HFQM_COMPOSE_FORMAT_V1.to_string(),
            components: vec![
                ComposeComponent {
                    tag: "base".to_string(),
                    filename: "Legacy.mq4.hfq".to_string(),
                    arch_id: 5,
                    tensors: vec!["base".to_string()],
                    metadata_json: r#"{"role":"base"}"#.to_string(),
                    source_format: "hfqm".to_string(),
                    hfqm_version: None,
                    byte_len: 0,
                    sha256: String::new(),
                    stored_entries: Vec::new(),
                    stored_segments: Vec::new(),
                    opaque_entry: None,
                },
                ComposeComponent {
                    tag: "mtp".to_string(),
                    filename: "Legacy.mtp.hfq".to_string(),
                    arch_id: 5,
                    tensors: vec!["mtp".to_string()],
                    metadata_json: r#"{"role":"mtp"}"#.to_string(),
                    source_format: "hfqm".to_string(),
                    hfqm_version: None,
                    byte_len: 0,
                    sha256: String::new(),
                    stored_entries: Vec::new(),
                    stored_segments: Vec::new(),
                    opaque_entry: None,
                },
            ],
        };
        let bundle_meta = serde_json::json!({HFQM_COMPOSE_KEY: manifest}).to_string();
        let bundle = dir.join("Legacy.mtp.mq4.hfq");
        write_hfqm_package_mem(
            &bundle,
            5,
            &bundle_meta,
            &[
                mem_tensor("base", vec![1, 2]),
                mem_tensor("mtp", vec![3, 4]),
            ],
        )
        .unwrap();
        let out = dir.join("out");
        decompose_hfq(&bundle, &out).unwrap();
        assert_eq!(
            std::fs::read(out.join("Legacy.mq4.hfq")).unwrap(),
            std::fs::read(base).unwrap()
        );
        assert_eq!(
            std::fs::read(out.join("Legacy.mtp.hfq")).unwrap(),
            std::fs::read(mtp).unwrap()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn decompose_rejects_uncomposed_container() {
        let dir = scratch_dir();
        let plain = dir.join("plain.hfq");
        write_hfqm_package_mem(&plain, 5, "{}", &[mem_tensor("a", vec![1])]).unwrap();
        let err = decompose_hfq(&plain, &dir.join("out")).unwrap_err();
        assert!(err.to_string().contains("no hipfire_compose manifest"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn role_config_keys_move_out_of_base_and_compose_restores() {
        let dir = scratch_dir();
        // A VL monolith: metadata advertises `vision_config`, and it holds a
        // vision tensor (so infer carves a `vl` sidecar).
        let bundle = dir.join("Model.mq4.hfq");
        write_hfqm_package_mem(
            &bundle,
            5,
            r#"{"arch_id":5,"vision_config":{"depth":2},"text_only_field":true}"#,
            &[
                mem_tensor("model.embed.weight", vec![1, 2, 3, 4]),
                mem_tensor("model.vision.patch_embed.weight", vec![9, 8, 7]),
            ],
        )
        .unwrap();

        let mut keys = RoleConfigKeys::new();
        keys.insert("vl".to_string(), vec!["vision_config".to_string()]);

        let out = dir.join("out");
        let written = decompose_hfq_infer_with_config_keys(&bundle, &out, &keys).unwrap();
        assert_eq!(written.len(), 2);

        // Base no longer advertises vision_config, but keeps its other config.
        let base = HfqPackage::open(&out.join("Model.mq4.hfq")).unwrap();
        assert!(!base.metadata_json.contains("vision_config"));
        assert!(base.metadata_json.contains("text_only_field"));
        // The vl sidecar now owns vision_config.
        let vl = HfqPackage::open(&out.join("Model.vl.hfq")).unwrap();
        assert!(vl.metadata_json.contains("vision_config"));

        // Recompose base + vl → vision_config travels back to the bundle top level.
        let rebundled = dir.join("Rebundled.vl.mq4.hfq");
        compose_hfq_with_config_keys(
            &[out.join("Model.mq4.hfq"), out.join("Model.vl.hfq")],
            &rebundled,
            &keys,
        )
        .unwrap();
        assert!(HfqPackage::open(&rebundled)
            .unwrap()
            .metadata_json
            .contains("vision_config"));

        // But swap the vl sidecar out (compose base + a non-vl sidecar): the
        // bundle must NOT regain vision_config — no vision tensors, no claim.
        let mtp = dir.join("Model.mtp.hfq");
        write_hfqm_package_mem(
            &mtp,
            5,
            r#"{"role":"mtp"}"#,
            &[mem_tensor("model.mtp.w", vec![4])],
        )
        .unwrap();
        let swapped = dir.join("Swapped.mtp.mq4.hfq");
        compose_hfq_with_config_keys(&[out.join("Model.mq4.hfq"), mtp], &swapped, &keys).unwrap();
        assert!(!HfqPackage::open(&swapped)
            .unwrap()
            .metadata_json
            .contains("vision_config"));

        std::fs::remove_dir_all(&dir).ok();
    }
}
