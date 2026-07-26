// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Read-only access to components embedded by the offline HFQ tooling.
//!
//! Composition, decomposition, validation, and container writes intentionally
//! live in `hipfire-hfq-tooling`.  The inference crate owns only the stable
//! manifest schema and zero-copy views needed to consume embedded DFLASH and
//! TRIA components.

use std::collections::BTreeSet;
use std::io;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const HFQM_COMPOSE_KEY: &str = "hipfire_compose";
pub const HFQM_COMPOSE_FORMAT_V1: &str = "hipfire.hfqm.compose.v1";
pub const HFQM_COMPOSE_FORMAT: &str = "hipfire.hfqm.compose.v2";
pub const HFQM_COMPONENT_PREFIX: &str = "__hipfire_component/";

fn default_source_format() -> String {
    "hfqm".to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposeStoredEntry {
    pub stored_name: String,
    pub original_name: String,
    pub original_offset: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposeStoredSegment {
    pub stored_name: String,
    pub original_offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposeComponent {
    pub tag: String,
    pub filename: String,
    pub arch_id: u32,
    pub tensors: Vec<String>,
    pub metadata_json: String,
    #[serde(default = "default_source_format")]
    pub source_format: String,
    #[serde(default)]
    pub hfqm_version: Option<u32>,
    #[serde(default)]
    pub byte_len: u64,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub stored_entries: Vec<ComposeStoredEntry>,
    #[serde(default)]
    pub stored_segments: Vec<ComposeStoredSegment>,
    #[serde(default)]
    pub opaque_entry: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposeManifest {
    pub format: String,
    pub components: Vec<ComposeComponent>,
}

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

/// Borrowed component view over the serving HFQ reader. Payloads remain in the
/// bundle mmap; no temporary sidecar is extracted.
pub struct HfqFileComponentView<'a> {
    file: &'a crate::hfq::HfqFile,
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

    pub fn entry(&self, original_name: &str) -> Option<&'a crate::hfq::HfqTensorInfo> {
        self.tensor_data(original_name).map(|(entry, _)| entry)
    }

    pub fn tensor_data(
        &self,
        original_name: &str,
    ) -> Option<(&'a crate::hfq::HfqTensorInfo, &'a [u8])> {
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
    ) -> Option<(&'a crate::hfq::HfqTensorInfo, Vec<u8>)> {
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

    /// Verify the original artifact digest by streaming stored ranges in
    /// original-byte order. Legacy manifests without a digest fail closed.
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
            return digest_result(
                &self.component.sha256,
                &format!("{:x}", Sha256::digest(&bytes)),
            );
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
        digest_result(&self.component.sha256, &format!("{:x}", hasher.finalize()))
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

pub fn file_component_view<'a>(
    file: &'a crate::hfq::HfqFile,
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
