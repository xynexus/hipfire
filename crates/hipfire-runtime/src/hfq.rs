// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! HFQ (.hfq) file loader for hipfire-native Q4_F16 quantized models.

use crate::hfq_modules::{parse_module_table, validate_modules, HfqModuleRecord};
use crate::llama::{LlamaConfig, LlamaWeights, ModelArch};
use crate::quant::f16_to_f32;
use crate::safetensors_source::SafetensorsSource;
use crate::weights::{EmbeddingFormat, LayerWeights, WeightTensor};
use hip_bridge::{HipError, HipResult};
use hipfire_model::{ModelSource, QuantConfig, TensorInfo};
use hipfire_quant_format::{storage, QuantType};
use hipfire_rdna::{DType, Gpu, GpuTensor};
use std::collections::HashMap;
use std::fs::File;
use std::hash::Hasher;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use twox_hash::XxHash64;

// The OQ4 arch-packing transform now lives in `crate::oq4_arch` (single source
// of truth). Re-exported here so the historical `hipfire_runtime::hfq::{...}`
// paths (qwen35 re-export, nemotron, the optimize tool) keep resolving.
pub use crate::oq4_arch::{
    oq4_arch_combined_len, oq4_arch_load, oq4_pack_arch_combined, OQ4_ARCH_PACKED_QT,
    OQ4_CANONICAL_QT,
};
pub use crate::oq8_arch::{
    oq4_to_oq8_combined, oq8_arch_load, oq8_arch_load_allow_compact, oq8_combined,
    oqplus_compact_to_oq8_combined,
};

pub const HFQM_MAGIC: &[u8; 4] = b"HFQM";
pub const HFQM_VERSION: u32 = 2;
const HFQM_V2_OFFSET_ALIGN: usize = 32;
/// Reserved `arch_id` for HFQM containers that are intentionally shareable
/// across model families. Role sidecars tied to one family should use the
/// parent model's `arch_id`; use this value only when metadata explicitly
/// defines a family-independent compatibility contract.
pub const HFQM_ARCH_NON_WEIGHT_PACKAGE: u32 = 0;

#[derive(Debug, Clone)]
pub struct HfqPackageEntry {
    pub name: String,
    pub quant_type: u8,
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub data_offset: usize,
    pub data_size: usize,
}

pub struct HfqPackage {
    file: File,
    file_len: usize,
    pub version: u32,
    pub arch_id: u32,
    pub metadata_json: String,
    entries: Vec<HfqPackageEntry>,
    entry_map: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
pub struct HfqPackageWriteEntry {
    pub name: String,
    pub quant_type: u8,
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub source_path: std::path::PathBuf,
    pub data_size: u64,
}

fn json_blob_end(bytes: &[u8]) -> Option<usize> {
    let mut brace_depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if b == b'\\' && in_string {
            escape = true;
            continue;
        }
        if b == b'"' {
            in_string = !in_string;
            continue;
        }
        if !in_string {
            if b == b'{' {
                brace_depth += 1;
            } else if b == b'}' {
                brace_depth -= 1;
                if brace_depth == 0 {
                    return Some(i + 1);
                }
            }
        }
    }
    None
}

fn xxh64_hex(bytes: &[u8]) -> String {
    let mut h = XxHash64::with_seed(0);
    h.write(bytes);
    format!("{:016x}", h.finish())
}

fn merge_tail_metadata(
    front_json: String,
    mut read_tail: impl FnMut(usize, usize) -> std::io::Result<Vec<u8>>,
) -> std::io::Result<String> {
    let mut front: serde_json::Value = serde_json::from_str(&front_json).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("HFQM v2 front metadata is not valid JSON: {e}"),
        )
    })?;
    let Some(tail_meta) = front.get("tail_metadata").cloned() else {
        return Ok(front_json);
    };
    let offset = tail_meta
        .get("offset")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "HFQM v2 tail offset missing",
            )
        })? as usize;
    let size = tail_meta
        .get("size")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "HFQM v2 tail size missing")
        })? as usize;
    let bytes = read_tail(offset, size)?;
    if let Some(expected) = tail_meta
        .get("hash")
        .and_then(|h| h.get("value"))
        .and_then(|v| v.as_str())
    {
        let actual = xxh64_hex(&bytes);
        if actual != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("HFQM v2 tail hash mismatch: expected {expected}, got {actual}"),
            ));
        }
    }
    let tail: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("HFQM v2 tail metadata is not valid JSON: {e}"),
        )
    })?;
    if let Some(full) = tail.get("metadata").cloned() {
        if let (Some(front_map), Some(full_map)) = (front.as_object_mut(), full.as_object()) {
            for (k, v) in full_map {
                front_map.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
    }
    serde_json::to_string(&front)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// `index` covers the header + metadata index only (bytes `0..data_offset`);
/// `file_len` is the WHOLE file. The two were one value back when the caller
/// mapped the entire file, and conflating them now would reject every entry
/// whose payload lives past the index — `read_tail` exists for the same reason.
fn parse_hfqm_index(
    mmap: &[u8],
    base: usize,
    metadata_offset: usize,
    data_offset: usize,
    n_entries: usize,
    version: u32,
    file_len: usize,
    read_tail: impl Fn(usize, usize) -> std::io::Result<Vec<u8>>,
) -> std::io::Result<(String, Vec<HfqPackageEntry>, HashMap<String, usize>)> {
    if metadata_offset > data_offset || data_offset > file_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid HFQM offsets metadata={metadata_offset} data={data_offset}"),
        ));
    }
    let meta_bytes = &mmap[metadata_offset..data_offset];
    let json_end = json_blob_end(meta_bytes).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "HFQM metadata JSON did not end",
        )
    })?;
    let mut metadata_json = String::from_utf8_lossy(&meta_bytes[..json_end]).to_string();
    if version >= 2 {
        metadata_json = merge_tail_metadata(metadata_json, |offset, size| {
            let end = offset.checked_add(size).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "HFQM tail range overflow")
            })?;
            if end > file_len {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("HFQM tail range {offset}..{end} exceeds file size {file_len}"),
                ));
            }
            read_tail(offset, size)
        })?;
    }
    let mut pos = metadata_offset + json_end;
    if pos + 4 > data_offset {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "HFQM index missing tensor count",
        ));
    }
    let idx_n = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()) as usize;
    if idx_n != n_entries {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("HFQM index count {idx_n} != header count {n_entries}"),
        ));
    }
    pos += 4;

    let mut entries = Vec::with_capacity(n_entries);
    let mut entry_map = HashMap::new();
    for i in 0..n_entries {
        if pos + 2 > data_offset {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "HFQM index truncated at name length",
            ));
        }
        let name_len = u16::from_le_bytes(mmap[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        if pos + name_len + 2 > data_offset {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "HFQM index truncated at name/shape header",
            ));
        }
        let name = String::from_utf8_lossy(&mmap[pos..pos + name_len]).to_string();
        pos += name_len;
        let quant_type = mmap[pos];
        pos += 1;
        let n_dims = mmap[pos] as usize;
        pos += 1;
        let per_entry_tail = if version >= 2 { 20 } else { 12 };
        if pos + n_dims * 4 + per_entry_tail > data_offset {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "HFQM index truncated at shape/data_size",
            ));
        }
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }
        let group_size = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let data_size = u64::from_le_bytes(mmap[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let data_offset = if version >= 2 {
            let offset_div32 = u64::from_le_bytes(mmap[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            offset_div32
                .checked_mul(HFQM_V2_OFFSET_ALIGN)
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "HFQM v2 offset overflow")
                })?
        } else {
            data_offset
                + entries
                    .iter()
                    .map(|e: &HfqPackageEntry| e.data_size)
                    .sum::<usize>()
        };
        if data_offset % HFQM_V2_OFFSET_ALIGN != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("HFQM entry {name} offset {data_offset} is not 32-byte aligned"),
            ));
        }
        if data_offset + data_size > file_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "HFQM entry {name} data range {}..{} exceeds file size {}",
                    data_offset,
                    data_offset + data_size,
                    mmap.len()
                ),
            ));
        }
        entry_map.insert(name.clone(), i);
        entries.push(HfqPackageEntry {
            name,
            quant_type,
            shape,
            group_size,
            data_offset: data_offset - base,
            data_size,
        });
    }
    Ok((metadata_json, entries, entry_map))
}

impl HfqPackage {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        // Header + index by bounded reads; blobs by pread on demand. Never
        // mmap'd — see the rule at the top of AGENTS.md.
        let file = File::open(path)?;
        let file_len = file.metadata()?.len() as usize;
        if file_len < 32 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not an HFQM package",
            ));
        }
        let mut header = [0u8; 32];
        read_exact_at_portable(&file, &mut header, 0)?;
        if &header[0..4] != HFQM_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not an HFQM package",
            ));
        }
        let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
        let arch_id = u32::from_le_bytes(header[8..12].try_into().unwrap());
        let n_entries = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
        let metadata_offset = u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize;
        let data_offset = u64::from_le_bytes(header[24..32].try_into().unwrap()) as usize;
        if metadata_offset > data_offset || data_offset > file_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid HFQM offsets metadata={metadata_offset} data={data_offset}"),
            ));
        }
        // `parse_hfqm_index` indexes from byte 0, so give it a prefix buffer
        // covering the header + metadata index rather than the whole file.
        let mut index = vec![0u8; data_offset];
        read_exact_at_portable(&file, &mut index, 0)?;
        let (metadata_json, entries, entry_map) = parse_hfqm_index(
            &index,
            0,
            metadata_offset,
            data_offset,
            n_entries,
            version,
            file_len,
            |offset, size| {
                let mut tail = vec![0u8; size];
                read_exact_at_portable(&file, &mut tail, offset as u64)?;
                Ok(tail)
            },
        )?;
        Ok(Self {
            file,
            file_len,
            version,
            arch_id,
            metadata_json,
            entries,
            entry_map,
        })
    }

    pub fn entries(&self) -> &[HfqPackageEntry] {
        &self.entries
    }

    pub fn entry(&self, name: &str) -> Option<&HfqPackageEntry> {
        self.entry_map.get(name).map(|&idx| &self.entries[idx])
    }

    /// Returns OWNED bytes — previously a borrow into the mapping.
    pub fn blob_data(&self, name: &str) -> Option<Vec<u8>> {
        let entry = self.entry(name)?;
        let end = entry.data_offset.checked_add(entry.data_size)?;
        if end > self.file_len {
            return None;
        }
        let mut buf = vec![0u8; entry.data_size];
        read_exact_at_portable(&self.file, &mut buf, entry.data_offset as u64).ok()?;
        Some(buf)
    }
}

pub fn write_hfqm_package_from_files(
    path: &Path,
    arch_id: u32,
    metadata_json: &str,
    entries: &[HfqPackageWriteEntry],
) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    let metadata_bytes = metadata_json.as_bytes();
    let metadata_offset = 32u64;
    let index_offset = metadata_offset + metadata_bytes.len() as u64;
    let mut index_len = 4u64;
    for entry in entries {
        let name_bytes = entry.name.as_bytes();
        if name_bytes.len() > u16::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("HFQM entry name too long: {}", entry.name),
            ));
        }
        if entry.shape.len() > u8::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("HFQM entry has too many dims: {}", entry.name),
            ));
        }
        index_len += 2 + name_bytes.len() as u64 + 1 + 1 + entry.shape.len() as u64 * 4 + 4 + 8 + 8;
    }
    let data_start_unaligned = index_offset + index_len;
    let data_offset = (data_start_unaligned + 4095) & !4095;
    let mut planned_offsets = Vec::with_capacity(entries.len());
    let mut cursor = data_offset;
    for entry in entries {
        cursor = (cursor + 31) & !31;
        planned_offsets.push(cursor);
        cursor = cursor.saturating_add(entry.data_size);
    }
    let mut index = Vec::new();
    index.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (entry, offset) in entries.iter().zip(&planned_offsets) {
        let name_bytes = entry.name.as_bytes();
        index.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        index.extend_from_slice(name_bytes);
        index.push(entry.quant_type);
        index.push(entry.shape.len() as u8);
        for &dim in &entry.shape {
            index.extend_from_slice(&dim.to_le_bytes());
        }
        index.extend_from_slice(&entry.group_size.to_le_bytes());
        index.extend_from_slice(&entry.data_size.to_le_bytes());
        index.extend_from_slice(&(offset / 32).to_le_bytes());
    }

    f.write_all(HFQM_MAGIC)?;
    f.write_all(&HFQM_VERSION.to_le_bytes())?;
    f.write_all(&arch_id.to_le_bytes())?;
    f.write_all(&(entries.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;
    f.write_all(metadata_bytes)?;
    f.write_all(&index)?;
    let pad_size = (data_offset - (index_offset + index.len() as u64)) as usize;
    if pad_size > 0 {
        f.write_all(&vec![0u8; pad_size])?;
    }
    let mut buf = vec![0u8; 16 * 1024 * 1024];
    let mut pos = data_offset;
    for (entry, offset) in entries.iter().zip(&planned_offsets) {
        while pos < *offset {
            let pad = ((*offset - pos) as usize).min(8192);
            f.write_all(&vec![0u8; pad])?;
            pos += pad as u64;
        }
        let expected = std::fs::metadata(&entry.source_path)?.len();
        if expected != entry.data_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "HFQM entry {} source size {} != declared {}",
                    entry.name, expected, entry.data_size
                ),
            ));
        }
        let mut src = File::open(&entry.source_path)?;
        src.seek(SeekFrom::Start(0))?;
        loop {
            let n = src.read(&mut buf)?;
            if n == 0 {
                break;
            }
            f.write_all(&buf[..n])?;
            pos += n as u64;
        }
    }
    f.flush()?;
    Ok(())
}

/// One in-memory tensor for [`write_hfqm_package_mem`].
pub struct HfqMemTensor {
    pub name: String,
    pub quant_type: u8,
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub data: Vec<u8>,
}

/// Descriptor for one tensor in a streaming HFQM write. `data_len` is the exact
/// payload byte count (deterministic from `shape` + `quant_type`), so the index
/// can be written before any payload is materialized — that is what lets the
/// collector stream multi-GB Hessians one tensor at a time instead of holding
/// them all in RAM.
pub struct HfqStreamEntry {
    pub name: String,
    pub quant_type: u8,
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub data_len: u64,
}

/// Where every part of an HFQM package will land on disk.
pub struct HfqmLayout {
    pub metadata_offset: u64,
    pub index_offset: u64,
    pub index_len: u64,
    /// First byte of the payload region (4 KiB-aligned).
    pub data_offset: u64,
    /// Absolute file offset of each entry's payload, in index order. Each is
    /// 32-byte aligned, because the index stores `offset / 32`.
    pub tensor_offsets: Vec<u64>,
}

/// Plan an HFQM package's byte layout without writing it.
///
/// Exposed because a tool that rewrites tensor payloads has to know where each
/// one WILL land before the file exists — `hfqm_modules` records embed absolute
/// `data_offset`s, so a rewritten artifact needs its module table recomputed
/// against the new layout. Re-deriving this rule inside such a tool would let
/// the two drift silently; [`write_hfqm_package_streaming`] calls this same
/// function, on the same principle that keeps `optimize` and the loader sharing
/// one `oq4_pack_arch_combined`.
///
/// Note the layout depends on `metadata_len`, while a module-bearing metadata
/// blob depends on the offsets this returns. That circularity is resolved by
/// iterating to a fixed point: plan, rebuild the metadata, and re-plan until the
/// metadata length stops changing (only decimal digit widths move, so it
/// converges in a couple of rounds).
pub fn plan_hfqm_layout(
    metadata_len: usize,
    entries: &[HfqStreamEntry],
) -> std::io::Result<HfqmLayout> {
    let metadata_offset = 32u64;
    let index_offset = metadata_offset + metadata_len as u64;
    let mut index_len = 4u64;
    for e in entries {
        let nb = e.name.as_bytes();
        if nb.len() > u16::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("HFQM entry name too long: {}", e.name),
            ));
        }
        index_len += 2 + nb.len() as u64 + 1 + 1 + e.shape.len() as u64 * 4 + 4 + 8 + 8;
    }
    let data_start = index_offset + index_len;
    let data_offset = (data_start + 4095) & !4095;
    let mut tensor_offsets = Vec::with_capacity(entries.len());
    let mut cursor = data_offset;
    for e in entries {
        cursor = (cursor + 31) & !31;
        tensor_offsets.push(cursor);
        cursor = cursor.saturating_add(e.data_len);
    }
    Ok(HfqmLayout {
        metadata_offset,
        index_offset,
        index_len,
        data_offset,
        tensor_offsets,
    })
}

/// Streaming HFQM writer: write the header + metadata + index up front (payload
/// sizes come from `entries`), then call `write_nth(i, w)` once per entry, in
/// index order, to stream that tensor's `data_len` bytes directly to the file.
/// Only one tensor's payload need exist in memory at a time. `write_nth` MUST
/// write exactly `entries[i].data_len` bytes. This is the canonical HFQM layout
/// impl (the in-memory [`write_hfqm_package_mem`] is a thin wrapper over it).
pub fn write_hfqm_package_streaming(
    path: &Path,
    arch_id: u32,
    metadata_json: &str,
    entries: &[HfqStreamEntry],
    mut write_nth: impl FnMut(usize, &mut dyn Write) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let meta = metadata_json.as_bytes();
    let layout = plan_hfqm_layout(meta.len(), entries)?;
    let metadata_offset = layout.metadata_offset;
    let index_offset = layout.index_offset;
    let data_offset = layout.data_offset;
    let planned_offsets = layout.tensor_offsets;
    let mut index = Vec::new();
    index.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (e, offset) in entries.iter().zip(&planned_offsets) {
        let nb = e.name.as_bytes();
        index.extend_from_slice(&(nb.len() as u16).to_le_bytes());
        index.extend_from_slice(nb);
        index.push(e.quant_type);
        index.push(e.shape.len() as u8);
        for &d in &e.shape {
            index.extend_from_slice(&d.to_le_bytes());
        }
        index.extend_from_slice(&e.group_size.to_le_bytes());
        index.extend_from_slice(&e.data_len.to_le_bytes());
        index.extend_from_slice(&(offset / 32).to_le_bytes());
    }
    let mut f = std::io::BufWriter::new(File::create(path)?);
    f.write_all(HFQM_MAGIC)?;
    f.write_all(&HFQM_VERSION.to_le_bytes())?;
    f.write_all(&arch_id.to_le_bytes())?;
    f.write_all(&(entries.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;
    f.write_all(meta)?;
    f.write_all(&index)?;
    f.write_all(&vec![
        0u8;
        (data_offset - (index_offset + index.len() as u64))
            as usize
    ])?;
    // Stream each payload through a counting writer that enforces the declared
    // data_len, so a producer bug can't silently desync the index from the data.
    let mut pos = data_offset;
    for (i, e) in entries.iter().enumerate() {
        while pos < planned_offsets[i] {
            let pad = ((planned_offsets[i] - pos) as usize).min(8192);
            f.write_all(&vec![0u8; pad])?;
            pos += pad as u64;
        }
        let mut counter = CountingWriter {
            inner: &mut f,
            written: 0,
        };
        write_nth(i, &mut counter)?;
        if counter.written != e.data_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "HFQM entry {}: wrote {} bytes, index declared {}",
                    e.name, counter.written, e.data_len
                ),
            ));
        }
        pos += counter.written;
    }
    f.flush()?;
    Ok(())
}

/// Wraps a writer and counts bytes written, to verify a streaming producer
/// emitted exactly the declared payload length.
struct CountingWriter<'a, W: Write> {
    inner: &'a mut W,
    written: u64,
}

impl<W: Write> Write for CountingWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Write an HFQM container from in-memory tensors. Thin wrapper over
/// [`write_hfqm_package_streaming`] for callers that already hold every payload
/// in RAM (e.g. small sidecars, tests). Large producers (the calibration
/// collector) should stream instead.
pub fn write_hfqm_package_mem(
    path: &Path,
    arch_id: u32,
    metadata_json: &str,
    tensors: &[HfqMemTensor],
) -> std::io::Result<()> {
    let entries: Vec<HfqStreamEntry> = tensors
        .iter()
        .map(|t| HfqStreamEntry {
            name: t.name.clone(),
            quant_type: t.quant_type,
            shape: t.shape.clone(),
            group_size: t.group_size,
            data_len: t.data.len() as u64,
        })
        .collect();
    write_hfqm_package_streaming(path, arch_id, metadata_json, &entries, |i, w| {
        w.write_all(&tensors[i].data)
    })
}

/// Drop page cache for a file byte range via posix_fadvise(FADV_DONTNEED).
/// On unified-memory APUs (e.g. Strix Halo), mmap'd model data and
/// hipMalloc'd GPU copies share physical RAM — without this, loading
/// a 65 GB model consumes ~130 GB (mmap cache + GPU copy).
/// Note: madvise(MADV_DONTNEED) does NOT work on MAP_SHARED file-backed
/// mappings (memmap2 default). posix_fadvise on the fd does.
#[cfg(unix)]
fn fadvise_dontneed(fd: std::os::unix::io::RawFd, offset: usize, len: usize) {
    unsafe {
        libc::posix_fadvise(
            fd,
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        );
    }
}

#[cfg(not(unix))]
fn fadvise_dontneed(_fd: i32, _offset: usize, _len: usize) {}

#[derive(Debug, Clone)]
pub struct HfqTensorInfo {
    pub name: String,
    pub quant_type: u8, // 0=Q4F16G64, 1=F16, 2=F32
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub data_offset: usize,
    pub data_size: usize,
}

/// Map a safetensors dtype string to the HFQ `QuantType` byte used for
/// unquantized source tensors (bf16=16, f16=1, f32=2), matching what
/// `hipfire-quantize --format bf16` stamps. Returns `None` for dtypes
/// `HfqFile::from_safetensors` does not pass through (pre-quantized / fp8).
fn map_safetensors_dtype(dtype: &str) -> Option<u8> {
    match dtype {
        "BF16" | "bfloat16" => Some(16),
        "F16" | "float16" => Some(1),
        "F32" | "float32" => Some(2),
        _ => None,
    }
}

/// Byte width of a supported safetensors float dtype.
fn dtype_byte_width(dtype: &str) -> Option<usize> {
    match dtype {
        "BF16" | "bfloat16" | "F16" | "float16" => Some(2),
        "F32" | "float32" => Some(4),
        _ => None,
    }
}

/// Per-file ceiling for a captured sidecar. `vocab.json` runs ~7 MB on a large
/// vocabulary, so the cap has to clear that comfortably; anything above this is
/// a weight-shaped file we do not want inlined into the header.
const MAX_SIDECAR_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// Ceiling on the whole captured set. `metadata_json` is parsed in full every
/// time a model is opened, so the sidecar blob is a load-time cost on every
/// run, not just at conversion.
const MAX_SIDECAR_TOTAL_BYTES: u64 = 64 * 1024 * 1024;

/// Files that are never sidecars: weight shards, and the shard index, which is
/// meaningless once export re-shards at a different size.
fn is_weight_like(name: &str) -> bool {
    if name == "model.safetensors.index.json" {
        return true;
    }
    matches!(
        Path::new(name).extension().and_then(|e| e.to_str()),
        Some("safetensors" | "bin" | "pt" | "pth" | "gguf" | "h5" | "msgpack" | "onnx")
    )
}

/// Recursively capture the non-weight files of an HF snapshot so an export can
/// reproduce the directory byte-for-byte.
///
/// Keys are `/`-separated paths relative to `dir` (`assets/logo.png`). Values
/// are `{"text": ...}` for valid UTF-8 and `{"b64": ...}` otherwise, so JSON
/// sidecars stay readable in the header while PNGs still survive.
///
/// `tokenizer.json` is deliberately excluded: it is already stored verbatim
/// under `"tokenizer"`, and duplicating it costs ~20 MB in a blob that is
/// parsed on every model open. Dot-entries (`.git/`, `.cache/`) are skipped.
fn collect_hf_sidecars(dir: &Path) -> serde_json::Map<String, serde_json::Value> {
    let mut out = serde_json::Map::new();
    let mut total: u64 = 0;
    collect_hf_sidecars_into(dir, dir, 0, &mut total, &mut out);
    out
}

fn collect_hf_sidecars_into(
    root: &Path,
    dir: &Path,
    depth: u32,
    total: &mut u64,
    out: &mut serde_json::Map<String, serde_json::Value>,
) {
    if depth > 4 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    // Sort so the captured set is deterministic across filesystems.
    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();

    for path in paths {
        let Some(base) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if base.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect_hf_sidecars_into(root, &path, depth + 1, total, out);
            continue;
        }
        let Ok(rel) = path.strip_prefix(root) else {
            continue;
        };
        let Some(key) = rel.to_str().map(|s| s.replace('\\', "/")) else {
            continue;
        };
        if is_weight_like(&key) || key == "tokenizer.json" {
            continue;
        }
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        if meta.len() > MAX_SIDECAR_FILE_BYTES || *total + meta.len() > MAX_SIDECAR_TOTAL_BYTES {
            eprintln!(
                "hfq: skipping sidecar {key} ({} bytes) — over the capture budget",
                meta.len()
            );
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        *total += bytes.len() as u64;
        let value = match String::from_utf8(bytes) {
            Ok(text) => serde_json::json!({ "text": text }),
            Err(e) => {
                use base64::Engine;
                let b64 = base64::engine::general_purpose::STANDARD.encode(e.as_bytes());
                serde_json::json!({ "b64": b64 })
            }
        };
        out.insert(key, value);
    }
}

/// Decode the `"hf_sidecars"` blob written by [`collect_hf_sidecars`] back into
/// `(relative path, contents)` pairs, ready to be written into an export
/// directory. Entries that are malformed are dropped rather than failing the
/// whole export.
pub fn hf_sidecars_from_metadata(metadata_json: &str) -> Vec<(String, Vec<u8>)> {
    let Ok(meta) = serde_json::from_str::<serde_json::Value>(metadata_json) else {
        return Vec::new();
    };
    let Some(obj) = meta.get("hf_sidecars").and_then(|v| v.as_object()) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(obj.len());
    for (key, value) in obj {
        if let Some(text) = value.get("text").and_then(|t| t.as_str()) {
            out.push((key.clone(), text.as_bytes().to_vec()));
        } else if let Some(b64) = value.get("b64").and_then(|b| b.as_str()) {
            use base64::Engine;
            if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) {
                out.push((key.clone(), bytes));
            }
        }
    }
    out
}

/// Fold a safetensors directory's tokenizer sidecars into HFQ-style metadata so
/// an `HfqFile::from_safetensors` model is self-describing exactly like a real
/// `.hfq` (whose quantizer embeds these at convert time). A real `.hfq` carries
/// the tokenizer inline; a raw safetensors `metadata_json` (just
/// `{architecture, config}`) does not, so the tokenizer/chat/eos consumers
/// (`hfq_tokenizer_metadata`, `from_hfq_metadata`, `hfq_chat_template`) would
/// otherwise fail. Embeds, when present and not already set:
/// - `"tokenizer"`: raw `tokenizer.json` text (as a JSON string);
/// - `"tokenizer_config"`: parsed `tokenizer_config.json` (chat_template);
/// - `"generation_config"`: parsed `generation_config.json` (authoritative
///   bos/eos ids). Missing sidecars are simply skipped.
/// - `"hf_sidecars"`: every other non-weight file in the snapshot, verbatim.
///   The parsed keys above lose byte-level formatting and cover only what the
///   runtime consumes; a multimodal checkpoint also needs
///   `preprocessor_config.json`, `processor_config.json`, `vocab.json` and the
///   like, and none of it can be recovered later from an `.hfq` that never
///   stored it. See [`collect_hf_sidecars`].
fn embed_tokenizer_metadata(base: &str, dir: &Path) -> String {
    let mut meta: serde_json::Value =
        serde_json::from_str(base).unwrap_or_else(|_| serde_json::json!({}));
    let Some(obj) = meta.as_object_mut() else {
        return base.to_string();
    };
    if !obj.contains_key("tokenizer") {
        if let Ok(s) = std::fs::read_to_string(dir.join("tokenizer.json")) {
            obj.insert("tokenizer".to_string(), serde_json::Value::String(s));
        }
    }
    for (file, key) in [
        ("tokenizer_config.json", "tokenizer_config"),
        ("generation_config.json", "generation_config"),
    ] {
        if !obj.contains_key(key) {
            if let Ok(s) = std::fs::read_to_string(dir.join(file)) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                    obj.insert(key.to_string(), v);
                }
            }
        }
    }
    if !obj.contains_key("hf_sidecars") {
        let sidecars = collect_hf_sidecars(dir);
        if !sidecars.is_empty() {
            obj.insert(
                "hf_sidecars".to_string(),
                serde_json::Value::Object(sidecars),
            );
        }
    }
    serde_json::to_string(&meta).unwrap_or_else(|_| base.to_string())
}

pub struct HfqFile {
    _file: File,
    /// Path used to open the file. Exposed via [`Self::path`] so the
    /// weight pager can open its own file handle for paged reads without
    /// going through this struct (cleanly separates HfqFile's mmap-based
    /// tensor lookup from the pager's pread/io_uring transport).
    path: std::path::PathBuf,
    /// mmap for tensor data access on discrete-GPU systems where GPU VRAM
    /// is separate from system RAM (no double-buffering cost).
    /// `None` on unified-memory APUs (Strix Halo etc.) where mmap pages
    /// and hipMalloc share physical RAM — keeping the mmap alive doubles
    /// memory consumption. Dropped after header/index parsing via
    /// `drop_mmap()`. When `None`, all tensor reads go through `pread`.
    pub arch_id: u32,
    pub metadata_json: String,
    tensors: Vec<HfqTensorInfo>,
    tensor_map: HashMap<String, usize>,
    /// Reusable read buffer for pread-based tensor reads.
    /// Avoids page cache buildup on unified-memory APUs where mmap pages
    /// can't be evicted while the mapping exists (FADV_DONTNEED is ignored
    /// for mmap'd regions per Linux kernel docs).
    pread_buf: std::cell::RefCell<Vec<u8>>,
    pub version: u32,
    modules: Vec<HfqModuleRecord>,
    /// Present only for [`Self::from_safetensors`]: owns the mmapped safetensors
    /// shards that back tensor reads. When set, the byte accessors serve from
    /// these shard mmaps (indexed via `st_locs`) instead of `self.mmap`/
    /// `self._file`, and `drop_mmap` is a no-op — the shard mmaps are the only
    /// resident copy of the weights.
    st_source: Option<SafetensorsSource>,
    /// Parallel to `tensors`: the physical `(shard, offset, len)` of each
    /// tensor's bytes within `st_source`. Empty for file-backed HFQM opens.
    st_locs: Vec<StLoc>,
    /// Parallel to `tensors`: for a compressed-BF16 tensor being transparently
    /// expanded, the `(stored quant_type, offset, len)` of its packed bytes in
    /// the file. `None` for every normally-stored tensor.
    ///
    /// Lives here rather than on [`HfqTensorInfo`] so the index entry can present
    /// the *logical* view — `quant_type == 16`, `data_size == n * 2` — and every
    /// existing BF16 consumer keeps working untouched, while file-range users
    /// (layer paging, pread) still get the real compressed extent.
    bf16_packed: Vec<Option<(u8, usize, usize)>>,
    /// `ModelSource`-shaped mirror of `tensors`, built on first use.
    ///
    /// The trait hands out `&TensorInfo`, so a source has to STORE one — and
    /// HFQ stores `HfqTensorInfo`. That single borrow is what left
    /// `impl ModelSource for HfqFile` a stub returning `None` for years. The
    /// mirror is index metadata only (no payloads), a few hundred entries even
    /// on the largest artifacts, so materialising it is far cheaper than
    /// reshaping the trait around the difference.
    ///
    /// Lazy rather than built in each constructor: `HfqFile` has four, and a
    /// consumer that never touches the trait should not pay for it.
    ms_infos: std::cell::OnceCell<Vec<TensorInfo>>,
}

/// Keep GPU-decodable recodings packed in RAM instead of expanding them at load.
///
/// Off by default: expanding costs more RAM but needs no kernel support, so every
/// existing consumer keeps working. Turn it on only for models served through
/// kernels that decode the packed form natively (`gemv_bf16l3`), where it also
/// buys ~1.18x weight bandwidth — but only once the working set exceeds the GPU's
/// last-level cache; below that it is a measured slowdown.
///
/// Only [`QuantType::Bf16Lut3`] can stay resident. Huffman codes are bit-serial,
/// so `Bf16Huff` is always expanded regardless of this flag.
/// `HIPFIRE_BF16L3_RESIDENT=0` — an explicit opt-OUT, distinguishable from
/// unset because residency is on by default for heads.
fn bf16l3_resident_disabled() -> bool {
    hipfire_env::BF16L3_RESIDENT.get().as_deref() == Some("0")
}

fn bf16l3_resident() -> bool {
    // Present and not literally "0" — deliberately NOT `flag()`: this predates
    // the 1/true/on/yes spelling and any other value counts as on.
    hipfire_env::BF16L3_RESIDENT.is_set()
        && hipfire_env::BF16L3_RESIDENT.get().as_deref() != Some("0")
}

/// A tensor consumed by the output projection (and, when tied, the embedding
/// gather). Matches `hipfire_quantize::hfq_out::is_gather_shaped`, which is what
/// steers these to LUT3 in the first place.
///
fn is_head_tensor(name: &str) -> bool {
    name.contains("lm_head") || name.contains("embed_tokens")
}

/// Rewrite losslessly-recoded index entries to their logical view, returning the
/// physical extents. Which types are recodings, what they expand to, and how long
/// the expansion is all come from `hipfire_quant_format::storage` — the one place
/// that knowledge lives, so a new codec cannot be invisible here.
fn expand_bf16_index(tensors: &mut [HfqTensorInfo]) -> Vec<Option<(u8, usize, usize)>> {
    let resident = bf16l3_resident();
    tensors
        .iter_mut()
        .map(|t| {
            let stored = QuantType::from_code(t.quant_type)?;
            if !stored.is_lossless_recoding() {
                return None;
            }
            // Residency opts a GPU-decodable coding out of expansion.
            //
            // A LUT3 HEAD stays packed by DEFAULT. It is the only large tensor
            // that is a pure GEMV consumer, so it is the only one with a kernel
            // that reads the packed form — `gemv_bf16l3_xf32`, measured at
            // 1.917 ms against plain bf16's 3.241 at 128256 x 2048, taking
            // tg128 from 90.74 to 102.53 with byte-identical output.
            //
            // Deliberately NOT extended to every Bf16Lut3 tensor. Layer weights
            // must serve prefill too, `gemv_bf16l3_xf32` is batch-1, and there
            // is no BF16L3 GEMM — so they would be decoded at load anyway, for
            // no benefit and a second decode path to keep correct. The env var
            // still forces that global behaviour when set.
            //
            // When the model ties its head, this same entry also backs the
            // embedding gather. That is fine: the gather decodes it explicitly
            // at `token_embd` load, because a lookup reads one arbitrary row and
            // the escape plane is only addressable by walking a block.
            //
            // `HIPFIRE_BF16L3_RESIDENT=0` opts out entirely, head included.
            // A LUT3 HEAD stays packed by DEFAULT, so `gemv_bf16l3_xf32` serves
            // it: 1.917 ms against plain bf16's 3.241 at 128256 x 2048, worth
            // tg128 90.05 -> 101.45 with byte-identical output.
            //
            // This was attempted once before and reverted: `expand_bf16_index`
            // is arch-agnostic but the loaders were not, and the tiny-quant gate
            // went from 8 failures to 58 with `got qt=49` panics across qwen35,
            // gemma4, zaya, qwen2 and dots-ocr. Every one of those now decodes a
            // packed tensor, and forcing residency globally reproduces the
            // baseline 8 exactly — which is what makes the default safe.
            //
            // NOT extended to every Bf16Lut3 tensor. Layer weights must serve
            // prefill, `gemv_bf16l3_xf32` is batch-1, and there is no BF16L3
            // GEMM, so they are decoded at load regardless. The env var still
            // forces that global behaviour.
            //
            // On a tied model this entry also backs the embedding gather, which
            // decodes it explicitly at `token_embd` load — a lookup reads one
            // arbitrary row and the escape plane needs a block walk.
            //
            // `HIPFIRE_BF16L3_RESIDENT=0` opts out entirely, head included.
            let head_default = !bf16l3_resident_disabled() && is_head_tensor(&t.name);
            if (resident || head_default) && stored == QuantType::Bf16Lut3 {
                return None;
            }
            let physical = (t.quant_type, t.data_offset, t.data_size);
            let n: usize = t.shape.iter().map(|&d| d as usize).product();
            t.data_size = stored.logical_byte_len(n)?;
            t.quant_type = stored.logical() as u8;
            Some(physical)
        })
        .collect()
}

/// Decode a recoded payload back to its logical bytes.
///
/// Public because arch loaders outside this crate need it: residency leaves
/// `Bf16Lut3` packed, and any loader that then reads the tensor must decode
/// rather than assume plain bf16. Returns `None` if `stored_qt` is not a
/// lossless recoding.
///
/// Huffman decode is bit-serial (~600 MB/s/core), so a full artifact would take
/// minutes on one thread; it is spread across cores using the format's chunk
/// table. Byte-aligned codings ignore the thread count.
pub fn decode_bf16_packed(stored_qt: u8, packed: &[u8], n: usize) -> Option<Vec<u8>> {
    let stored = QuantType::from_code(stored_qt)?;
    storage::expand(stored, packed, n, decode_threads()).map(|b| b.into_owned())
}

/// True for the lossless BF16 recodings: `Bf16Lut3` (49) and `Bf16Huff` (50).
///
/// A tensor carrying one of these reaches an arch loader with a quant code the
/// loader's dtype match arms do not know — `--format bf16` applies `Bf16Huff`
/// by DEFAULT, and `expand_bf16_index` deliberately leaves `Bf16Lut3` head
/// tensors packed. Named rather than open-coded as `matches!(qt, 49 | 50)`,
/// which is how the same three-line check ended up in five arch crates.
#[inline]
pub fn is_packed_bf16(qt: u8) -> bool {
    matches!(qt, 49 | 50)
}

/// Decode a possibly-recoded payload to its logical form.
///
/// Returns the LOGICAL quant code with the bytes: `(16, expanded)` for a
/// recoding, otherwise the input unchanged and borrowed. `n` is the ELEMENT
/// count, which callers must take from the shape rather than `data_size` — a
/// tensor reaching here is one `expand_bf16_index` declined to expand, so its
/// `data_size` is still the packed physical length.
pub fn decode_recoded_bf16<'a>(
    qt: u8,
    data: &'a [u8],
    n: usize,
) -> Option<(u8, std::borrow::Cow<'a, [u8]>)> {
    if !is_packed_bf16(qt) {
        return Some((qt, std::borrow::Cow::Borrowed(data)));
    }
    decode_bf16_packed(qt, data, n).map(|v| (16u8, std::borrow::Cow::Owned(v)))
}

/// Why [`HfqFile::tensor_data_logical`] could not produce logical bytes.
///
/// Distinguishes absent from present-but-undecodable, because reporting a
/// present-but-compressed tensor as MISSING sends the reader after a broken
/// artifact instead of an unsupported storage coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalReadError {
    /// No tensor by that name.
    Missing,
    /// Present, but the recoding at this quant code would not expand.
    Decode(u8),
}

/// Threads used to expand a bit-serial payload. Every core, always — decode is
/// pure compute over independent chunks and reaches ~21 GB/s here, well past any
/// disk it overlaps with, so there is nothing worth tuning.
fn decode_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |p| p.get())
}

#[derive(Debug, Clone, Copy)]
struct StLoc {
    shard: usize,
    offset: usize,
    len: usize,
}

impl HfqFile {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        Self::open_at_offset(path, 0)
    }

    /// Open only the HFQM header, metadata, and tensor/module indexes.
    ///
    /// Unlike [`Self::open`], this does not mmap the full file. It is intended
    /// for large-artifact probes and repack/catalog tools that must prove they
    /// can inspect a 100GiB+ model without touching tensor payload pages.
    pub fn open_index_only(path: &Path) -> std::io::Result<Self> {
        Self::open_index_only_at_offset(path, 0)
    }

    pub fn open_index_only_at_offset(path: &Path, base_offset: u64) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len() as usize;
        let mut header = [0u8; 32];
        read_exact_at_portable(&file, &mut header, base_offset)?;
        if &header[0..4] != HFQM_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("not an HFQM container at offset {base_offset}"),
            ));
        }
        let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
        let arch_id = u32::from_le_bytes(header[8..12].try_into().unwrap());
        let n_tensors = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
        let base = base_offset as usize;
        let metadata_offset =
            u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize + base;
        let data_offset = u64::from_le_bytes(header[24..32].try_into().unwrap()) as usize + base;
        if metadata_offset > data_offset || data_offset > file_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid HFQM offsets metadata={metadata_offset} data={data_offset}"),
            ));
        }
        let mut meta_index = vec![0u8; data_offset - metadata_offset];
        read_exact_at_portable(&file, &mut meta_index, metadata_offset as u64)?;
        let (metadata_json, tensors, tensor_map) = parse_hfqm_meta_index(
            &meta_index,
            metadata_offset,
            data_offset,
            base,
            n_tensors,
            file_len,
            version,
            |offset, size| {
                let mut tail = vec![0u8; size];
                read_exact_at_portable(&file, &mut tail, offset as u64)?;
                Ok(tail)
            },
        )?;
        let modules = parse_module_table(&metadata_json)?.unwrap_or_default();
        if !modules.is_empty() {
            validate_modules(&modules, file_len)?;
        }
        let mut tensors = tensors;
        let bf16_packed = expand_bf16_index(&mut tensors);
        Ok(Self {
            ms_infos: std::cell::OnceCell::new(),
            _file: file,
            path: path.to_path_buf(),
            arch_id,
            metadata_json,
            tensors,
            tensor_map,
            pread_buf: std::cell::RefCell::new(Vec::new()),
            version,
            modules,
            st_source: None,
            st_locs: Vec::new(),
            bf16_packed,
        })
    }

    /// Open an HFQM container that lives inside a larger file, starting at
    /// `base_offset`. Used by the bundled `-mq4+mtp.hfq` loader to parse the
    /// MTP section embedded after the trunk's tensor data.
    ///
    /// The whole file is mmap'd, and the HFQM header is read starting at
    /// `base_offset`. All stored offsets inside the HFQM container are
    /// rebased to absolute file offsets (`base_offset + stored_offset`).
    ///
    /// Callers passing `base_offset = 0` go through the canonical [`Self::open`]
    /// entry point.
    pub fn open_at_offset(path: &Path, base_offset: u64) -> std::io::Result<Self> {
        // Weights are never mmap'd — see the rule at the top of AGENTS.md. This
        // used to map the whole file and only fall back to the pread loader past
        // 64 GiB; the two produce the same `HfqFile`, so there is one path now.
        //
        // The 64 GiB escape hatch was itself an admission: `mmap.advise(Sequential)`
        // raced the slab loader's O_DIRECT fd on the same inode into a kworker
        // deadlock at 291 GiB. The failure is not size-specific, only easier to
        // hit when large — a 9 MiB expert page-in failed on the paged 122B with
        // 118 GiB sitting in unreclaimable page cache.
        Self::open_index_only_at_offset(path, base_offset)
    }

    /// Build an in-memory `HfqFile` directly over a HuggingFace safetensors
    /// directory — no intermediate bf16 `.hfq` written to disk. This lets the
    /// resident calibration path (`collect_artifacts`) and any `&HfqFile`
    /// arch loader consume a raw HF checkpoint exactly as it consumes a bf16
    /// `.hfq`.
    ///
    /// The presentation matches what `hipfire-quantize --format bf16` writes,
    /// so the arch weight loaders need no changes:
    /// - dense tensor names are the raw HF safetensors keys (no ggml rename);
    /// - dtype is passed through, tagged with the matching `QuantType` byte
    ///   (bf16→16, f16→1, f32→2), bytes verbatim (no numeric conversion);
    /// - stacked routed-expert tensors (`...experts.gate_up_proj` /
    ///   `...experts.down_proj`, stored 3D as `[num_experts, …]`) are split
    ///   into the per-expert 2D `...experts.{x}.{proj}.weight` tensors the MoE
    ///   loaders request; the fused gate_up projection stays fused.
    ///
    /// Only float source dtypes (BF16/F16/F32) are handled; a pre-quantized HF
    /// checkpoint (GPTQ/AWQ/FP8) errors rather than mis-tagging its bytes.
    pub fn from_safetensors(dir: &Path) -> std::io::Result<Self> {
        let source = SafetensorsSource::open(dir)?;

        let mut tensors: Vec<HfqTensorInfo> = Vec::new();
        let mut tensor_map: HashMap<String, usize> = HashMap::new();
        let mut st_locs: Vec<StLoc> = Vec::new();

        // Deterministic order (HashMap iteration is not) for reproducible logs
        // and to avoid any latent index-order dependence in scan-style callers.
        let mut layout = source.tensor_layout();
        layout.sort_by(|a, b| a.0.cmp(&b.0));

        for (name, shard, offset, len, dtype, shape) in layout {
            let qt = map_safetensors_dtype(&dtype).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "{}: tensor {name} has unsupported dtype {dtype:?} \
                         (from_safetensors handles bf16/f16/f32 source only)",
                        dir.display()
                    ),
                )
            })?;
            let esz = dtype_byte_width(&dtype).expect("dtype width known when qt mapped");

            // Stacked routed-expert weights: 3D `[E, …]` parents named
            // `....experts.<proj>` with no `.weight` suffix. Split into the
            // per-expert 2D tensors the MoE loaders request. Each expert slice
            // is a contiguous sub-range of the parent's row-major bytes.
            let is_stacked_expert =
                shape.len() == 3 && name.contains(".experts.") && !name.ends_with(".weight");
            if is_stacked_expert {
                let (head, tail) = name.split_once(".experts.").expect("`.experts.` present");
                let n_experts = shape[0];
                let per_elems: usize = shape[1..].iter().product();
                let per_len = per_elems * esz;
                if per_len == 0 || per_len.checked_mul(n_experts) != Some(len) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "{}: stacked expert {name} shape {shape:?} dtype {dtype:?} \
                             does not match stored byte length {len}",
                            dir.display()
                        ),
                    ));
                }
                let child_shape: Vec<u32> = shape[1..].iter().map(|&d| d as u32).collect();
                for x in 0..n_experts {
                    let child = format!("{head}.experts.{x}.{tail}.weight");
                    let idx = tensors.len();
                    tensors.push(HfqTensorInfo {
                        name: child.clone(),
                        quant_type: qt,
                        shape: child_shape.clone(),
                        group_size: 0,
                        data_offset: 0, // unused: st-backed reads come from st_locs
                        data_size: per_len,
                    });
                    tensor_map.insert(child, idx);
                    st_locs.push(StLoc {
                        shard,
                        offset: offset + x * per_len,
                        len: per_len,
                    });
                }
            } else {
                let idx = tensors.len();
                tensors.push(HfqTensorInfo {
                    name: name.clone(),
                    quant_type: qt,
                    shape: shape.iter().map(|&d| d as u32).collect(),
                    group_size: 0,
                    data_offset: 0, // unused: st-backed reads come from st_locs
                    data_size: len,
                });
                tensor_map.insert(name, idx);
                st_locs.push(StLoc { shard, offset, len });
            }
        }

        let arch_id = source.arch_id();
        // Embed tokenizer/chat/generation sidecars so the model is
        // self-describing like a real `.hfq` (config stays under "config").
        let metadata_json = embed_tokenizer_metadata(source.metadata_json(), dir);
        // A real fd for the `_file` field. The safetensors byte path never
        // reads it (bytes come from the shard mmaps), and drop_mmap()
        // short-circuits for st-backed files, so config.json is a safe handle.
        let file = File::open(dir.join("config.json"))?;
        Ok(Self {
            ms_infos: std::cell::OnceCell::new(),
            _file: file,
            path: dir.to_path_buf(),
            arch_id,
            metadata_json,
            tensor_map,
            pread_buf: std::cell::RefCell::new(Vec::new()),
            version: HFQM_VERSION,
            modules: Vec::new(),
            st_source: Some(source),
            // Safetensors-backed tensors are never compressed: the dtype comes
            // straight from the shard headers.
            bf16_packed: vec![None; tensors.len()],
            tensors,
            st_locs,
        })
    }

    /// Drop the mmap to free the virtual address mapping. After this call,
    /// `tensor_data()` returns `None` and all reads go through `tensor_data_pread()`.
    ///
    /// On unified-memory APUs (Strix Halo, Steam Deck, etc.), GPU and CPU
    /// share physical RAM. Keeping the mmap alive while hipMalloc copies
    /// tensor data into GPU buffers doubles memory consumption (mmap pages
    /// + GPU copy both resident). Dropping the mmap after header/index
    /// parsing lets the kernel reclaim those pages.
    ///
    /// On discrete-GPU systems this is unnecessary (GPU VRAM is separate),
    /// so callers should only invoke this when UMA is detected.
    pub fn drop_mmap(&mut self) {
        if self.st_source.is_some() {
            // Safetensors-backed: the shard mmaps ARE the only resident copy of
            // the weights; there is no separate HFQM mmap to drop.
            return;
        }
        // Nothing to unmap any more — weights are read with pread. Kept as a
        // no-op rather than deleted so the ~10 call sites that ask for it still
        // express the intent, and because the FADV_DONTNEED below is still worth
        // doing: it releases whatever page cache the index reads left behind
        // before the slab loader opens its O_DIRECT fd on the same inode.
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            unsafe {
                libc::posix_fadvise(self._file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
            }
        }
    }

    /// Release the pread reuse buffer back to the allocator. After a
    /// large tensor is read (e.g. DeepSeek V4's `head.weight` at ~560 MB on a Q8F16
    /// build), `pread_buf` keeps that capacity for the rest of the
    /// `HfqFile`'s lifetime. On UMA systems where GPU and CPU share
    /// physical RAM, that competes with the routed-expert upload pass —
    /// the difference between fitting and OOM-at-layer-42 on the 88 GB
    /// deepseek4-q8-mtp build.
    ///
    /// Call between load phases when you know subsequent reads will be
    /// much smaller than the peak. Safe at any time; the next pread
    /// auto-grows the buffer as needed.
    pub fn shrink_pread_buf(&self) {
        let mut buf = self.pread_buf.borrow_mut();
        buf.clear();
        buf.shrink_to_fit();
    }

    /// Path the HFQ file was opened from. The weight pager uses this to
    /// open its own file handle for paged reads — keeping the pager's
    /// transport independent of this struct's lifetime / mmap.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The upstream HuggingFace Jinja `chat_template` baked into this
    /// .hfq's `tokenizer_config` metadata. `None` when the source model
    /// did not ship a chat_template (rare for instruct models, common
    /// for base models). The runtime renders this when present so prompt
    /// framing matches the model's training-time expectation; absent or
    /// failing renders fall back to the hand-rolled `ChatFrame` path.
    pub fn chat_template(&self) -> Option<String> {
        hipfire_model::hfq_chat_template(&self.metadata_json)
    }

    /// Resolve a tensor name, trying common prefix variants.
    ///
    /// Qwen3.5 safetensors-converted files store tensors under
    /// `model.language_model.layers.N.` while the canonical GGUF-derived
    /// hipfire-quantize path produces `model.layers.N.`. Callers consistently
    /// pass one prefix style; this helper tries the exact name first, then
    /// strips or adds the `model.language_model.` prefix so a model file
    /// from either pipeline loads cleanly. SentenceTransformers exports may
    /// also omit the outer `model.` prefix entirely, so that variant is tried
    /// as well. Returns `None` only when no variant matches — the per-callsite
    /// `?` early-return is preserved.
    /// Read a tensor's bytes with any lossless BF16 recoding decoded to plain
    /// bf16, returning the LOGICAL quant code alongside.
    ///
    /// This is what an arch loader wants: decoding once at the read boundary
    /// means the loader's dtype match arms only ever see logical codes, so a
    /// new caller cannot forget the case. Five arch crates each grew a private
    /// copy of exactly this (`16cb54d56`); they now share one.
    ///
    /// Note this is deliberately NOT what [`ModelSource::tensor`] does — that
    /// returns the payload for the coding the artifact DECLARES, so a
    /// `Bf16Lut3` head stays packed for the kernel that decodes it natively.
    /// Use this when you need bf16 in hand; use `tensor()` when you want the
    /// tensor as stored.
    pub fn tensor_data_logical(&self, name: &str) -> Result<(u8, Vec<u8>), LogicalReadError> {
        let (info, data) = self
            .tensor_data_vec(name)
            .ok_or(LogicalReadError::Missing)?;
        if !is_packed_bf16(info.quant_type) {
            return Ok((info.quant_type, data));
        }
        let n: usize = info.shape.iter().map(|&d| d as usize).product();
        let qt = info.quant_type;
        let logical = decode_bf16_packed(qt, &data, n).ok_or(LogicalReadError::Decode(qt))?;
        Ok((16, logical))
    }

    /// `ModelSource`-shaped view of the tensor index, built once on demand.
    ///
    /// Parallel to `self.tensors`, so a `resolve_idx` result indexes both.
    /// Sizes and offsets are the LOGICAL ones the index already advertises —
    /// for a transparently-expanded BF16 recoding that is the expanded length,
    /// matching what `tensor()` hands back. Anything wanting the physical
    /// extent (layer paging, pread) must keep using `physical_range`.
    fn ms_infos(&self) -> &[TensorInfo] {
        self.ms_infos.get_or_init(|| {
            self.tensors
                .iter()
                .map(|t| {
                    // LOGICAL view. A lossless BF16 recoding (Bf16Lut3 /
                    // Bf16Huff) IS bf16 — the coding is a storage detail — so
                    // the trait reports it as bf16 and `tensor()` hands back
                    // expanded bytes to match. Reporting the stored code here
                    // instead makes every consumer that switches on dtype
                    // reject a perfectly ordinary bf16 tensor, which is exactly
                    // how the qwen3.5 calibration adapter first met an `.hfq`
                    // ("unsupported dtype HFQ_QT49").
                    //
                    // Callers that specifically want the STORED form — serving,
                    // where `gemv_bf16l3` decodes natively and expanding would
                    // waste the whole point — use HfqFile's inherent accessors,
                    // which are unaffected.
                    let logical_qt = if is_packed_bf16(t.quant_type) {
                        16
                    } else {
                        t.quant_type
                    };
                    let n: usize = t.shape.iter().map(|&d| d as usize).product();
                    TensorInfo {
                        name: t.name.clone(),
                        // HFQ's authority is `quant_type`; `dtype` is the
                        // safetensors-shaped field and only three codes have a
                        // faithful spelling. The rest get an explicit marker
                        // rather than a plausible-looking lie.
                        dtype: match logical_qt {
                            16 => "BF16".to_string(),
                            1 => "F16".to_string(),
                            2 => "F32".to_string(),
                            other => format!("HFQ_QT{other}"),
                        },
                        shape: t.shape.iter().map(|&d| d as usize).collect(),
                        quant_type: logical_qt,
                        data_offset: t.data_offset,
                        // A packed tensor's `data_size` is its PHYSICAL length;
                        // the logical view is two bytes per element.
                        data_size: if is_packed_bf16(t.quant_type) {
                            n * 2
                        } else {
                            t.data_size
                        },
                    }
                })
                .collect()
        })
    }

    fn resolve_idx(&self, name: &str) -> Option<usize> {
        if let Some(&idx) = self.tensor_map.get(name) {
            return Some(idx);
        }
        // Strip "model.language_model." → "model."
        if let Some(rest) = name.strip_prefix("model.language_model.") {
            let short = format!("model.{rest}");
            if let Some(&idx) = self.tensor_map.get(&short) {
                return Some(idx);
            }
        }
        // Add "model.language_model." prefix: "model.X" → "model.language_model.X"
        if let Some(rest) = name.strip_prefix("model.") {
            // SentenceTransformers Qwen exports the transformer directly, so
            // its safetensors keys are `embed_tokens.*`, `layers.*`, `norm.*`
            // rather than the causal-LM wrapper's `model.*` keys.
            if let Some(&idx) = self.tensor_map.get(rest) {
                return Some(idx);
            }
            let long = format!("model.language_model.{rest}");
            if let Some(&idx) = self.tensor_map.get(&long) {
                return Some(idx);
            }
        }
        // Try with `model.` / `model.language_model.` added when name has no
        // `model.` prefix at all (e.g. `lm_head.weight`).
        if !name.starts_with("model.") {
            let with_model = format!("model.{name}");
            if let Some(&idx) = self.tensor_map.get(&with_model) {
                return Some(idx);
            }
            let with_lm = format!("model.language_model.{name}");
            if let Some(&idx) = self.tensor_map.get(&with_lm) {
                return Some(idx);
            }
        }
        None
    }

    /// Look up a tensor's metadata (name, quant_type, shape, byte offset/size)
    /// without copying its data. The weight pager calls this at load time to
    /// register byte ranges without forcing eager VRAM allocation.
    pub fn find_tensor_info(&self, name: &str) -> Option<&HfqTensorInfo> {
        let idx = self.resolve_idx(name)?;
        Some(&self.tensors[idx])
    }

    /// Borrow a tensor's bytes. Returns `None` for a losslessly recoded tensor —
    /// prefer [`Self::tensor_data_cow`] unless you know the tensor is never one.
    ///
    /// `--bf16-codec` DEFAULTS to `huff`, so any BF16 tensor in any recent
    /// artifact may be stored compressed, and `is_gather_shaped` steers
    /// `embed_tokens` / `lm_head` to LUT3 specifically — the big tensors are the
    /// likeliest to be recoded, not the least.
    ///
    /// The `None` is the trap: callers wrote `.expect("<name> not found")` and
    /// reported a PRESENT tensor as missing, or tested `.is_some()` and silently
    /// took a fallback path. Both shapes shipped (see the sweep in commits
    /// 4e6166cca / 226bb66b2 / 9a6a5bbd3), and one loaded an embedding table as
    /// an untied model's output weights with no error at all.
    ///
    /// For metadata alone use [`Self::find_tensor_info`] — it needs no bytes and
    /// so does not care how the tensor is stored.
    /// Returns OWNED bytes. It used to borrow from the file mapping; weights are
    /// no longer mmap'd (see the rule at the top of AGENTS.md), so there is
    /// nothing to borrow and this is `tensor_data_vec` under its old name.
    ///
    /// Prefer [`Self::tensor_data_pread`] on a hot path — it reuses one buffer
    /// instead of allocating per call. It is NOT a drop-in here: that buffer is
    /// shared, so holding two tensors at once (loading.rs reads qweight, qzeros
    /// and scales together) would have the second overwrite the first.
    pub fn tensor_data(&self, name: &str) -> Option<(&HfqTensorInfo, Vec<u8>)> {
        self.tensor_data_vec(name)
    }

    /// Read tensor data via pread into a reusable buffer, then FADV_DONTNEED
    /// the file range. On unified-memory APUs (Strix Halo etc.), mmap pages
    /// can't be evicted while the mapping exists, so pread + fadvise is the
    /// only way to prevent page cache from starving hipMalloc.
    ///
    /// Returns (info, guard) where guard derefs to `&[u8]`. The buffer is
    /// reused across calls — the previous data is overwritten.
    #[cfg(unix)]
    pub fn tensor_data_pread(
        &self,
        name: &str,
    ) -> Option<(&HfqTensorInfo, std::cell::Ref<'_, Vec<u8>>)> {
        use std::os::unix::io::AsRawFd;
        let idx = self.resolve_idx(name)?;
        let info = &self.tensors[idx];
        if let Some(source) = &self.st_source {
            let loc = self.st_locs[idx];
            let bytes = source.shard_bytes(loc.shard, loc.offset, loc.len)?;
            {
                let mut buf = self.pread_buf.borrow_mut();
                buf.clear();
                buf.extend_from_slice(bytes);
            }
            return Some((info, self.pread_buf.borrow()));
        }
        let fd = self._file.as_raw_fd();
        // Read the physical extent — compressed for a BF16L3 tensor, whose
        // `info.data_size` already advertises the expanded length.
        let (phys_off, phys_len) = self.physical_range(idx);
        {
            let mut buf = self.pread_buf.borrow_mut();
            buf.resize(phys_len, 0);
            let mut total_read = 0usize;
            while total_read < phys_len {
                let n = unsafe {
                    libc::pread(
                        fd,
                        buf[total_read..].as_mut_ptr() as *mut libc::c_void,
                        phys_len - total_read,
                        (phys_off + total_read) as libc::off_t,
                    )
                };
                if n <= 0 {
                    break;
                }
                total_read += n as usize;
            }
            // Evict these pages from cache — works because pread doesn't hold a mapping.
            fadvise_dontneed(fd, phys_off, phys_len);
            if let Some((qt, _, _)) = self.bf16_packed[idx] {
                *buf = decode_bf16_packed(qt, &buf, info.data_size / 2)?;
            }
        }
        Some((info, self.pread_buf.borrow()))
    }

    /// Non-unix fallback: just delegates to mmap-based tensor_data.
    #[cfg(not(unix))]
    pub fn tensor_data_pread(&self, name: &str) -> Option<(&HfqTensorInfo, &[u8])> {
        self.tensor_data(name)
    }

    /// Borrow the tensor's logical bytes, decoding only when they are stored
    /// compressed. Use this wherever a caller would reach for [`Self::tensor_data`]
    /// on a tensor that MIGHT be a lossless BF16 recoding (`Bf16Lut3` /
    /// `Bf16Huff`) — most importantly the embedding table, which is the one
    /// tensor big enough that recoding it is worth ~12% of an artifact.
    ///
    /// [`Self::tensor_data`] deliberately returns `None` for those (it can only
    /// hand back the mmap'd PACKED bytes, which tagged as BF16 would be
    /// garbage). Callers that then `.expect("<name> not found")` report a
    /// present-but-compressed tensor as MISSING, which is a genuinely
    /// misleading trail to follow — it reads as a broken artifact rather than
    /// an unsupported storage coding.
    ///
    /// Borrowed in the common uncompressed case, so this does not cost the
    /// embedding table's several hundred MB unless the decode is real.
    pub fn tensor_data_cow(
        &self,
        name: &str,
    ) -> Option<(&HfqTensorInfo, std::borrow::Cow<'_, [u8]>)> {
        let idx = self.resolve_idx(name)?;
        if self.bf16_packed[idx].is_some() {
            let (info, bytes) = self.tensor_data_vec(name)?;
            return Some((info, std::borrow::Cow::Owned(bytes)));
        }
        let (info, bytes) = self.tensor_data(name)?;
        Some((info, std::borrow::Cow::Owned(bytes)))
    }

    /// Read tensor data using the best available path:
    /// - Unix with pread support: pread + fadvise_dontneed (avoids page cache buildup)
    /// - Fallback: mmap slice (returns None if mmap was dropped)
    ///
    /// Returns owned Vec<u8> to avoid lifetime issues with the pread RefCell.
    pub fn tensor_data_vec(&self, name: &str) -> Option<(&HfqTensorInfo, Vec<u8>)> {
        let idx = self.resolve_idx(name)?;
        let info = &self.tensors[idx];

        if let Some(source) = &self.st_source {
            let loc = self.st_locs[idx];
            let bytes = source.shard_bytes(loc.shard, loc.offset, loc.len)?;
            return Some((info, bytes.to_vec()));
        }

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = self._file.as_raw_fd();
            // Physical extent: compressed for BF16L3, whose `info.data_size`
            // already advertises the expanded length.
            let (phys_off, phys_len) = self.physical_range(idx);
            let mut buf = vec![0u8; phys_len];
            let mut total_read = 0usize;
            while total_read < phys_len {
                let n = unsafe {
                    libc::pread(
                        fd,
                        buf[total_read..].as_mut_ptr() as *mut libc::c_void,
                        phys_len - total_read,
                        (phys_off + total_read) as libc::off_t,
                    )
                };
                if n <= 0 {
                    break;
                }
                total_read += n as usize;
            }
            fadvise_dontneed(fd, phys_off, phys_len);
            if let Some((qt, _, _)) = self.bf16_packed[idx] {
                buf = decode_bf16_packed(qt, &buf, info.data_size / 2)?;
            }
            return Some((info, buf));
        }

        #[cfg(not(unix))]
        {
            let mmap = self.mmap.as_ref()?;
            Some((
                info,
                mmap[info.data_offset..info.data_offset + info.data_size].to_vec(),
            ))
        }
    }

    /// Release page cache for a byte range. Only works if the range is NOT mmap'd.
    #[allow(dead_code)]
    pub fn drop_pages_range(&self, offset: usize, len: usize) {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            fadvise_dontneed(self._file.as_raw_fd(), offset, len);
        }
        #[cfg(not(unix))]
        {
            let _ = (offset, len);
        }
    }

    /// Return the (start_offset, end_offset) byte range covering all tensors
    /// whose name contains `prefix.` (e.g. "layers.5.").
    #[allow(dead_code)]
    pub fn layer_data_range(&self, prefix: &str) -> Option<(usize, usize)> {
        if self.st_source.is_some() {
            // st-backed tensors have no single-file byte range (data_offset is
            // unused); the shard fadvise hint this feeds does not apply.
            return None;
        }
        let needle = format!("{prefix}.");
        let mut lo = usize::MAX;
        let mut hi = 0usize;
        for (i, t) in self.tensors.iter().enumerate() {
            if t.name.contains(&needle) {
                // Physical extent: a BF16L3 tensor's file range is its
                // compressed length, not the expanded `data_size`. Using the
                // latter would over-cover and evict the next tensor's pages.
                let (off, len) = self.physical_range(i);
                lo = lo.min(off);
                hi = hi.max(off + len);
            }
        }
        if lo < hi {
            Some((lo, hi))
        } else {
            None
        }
    }

    /// The tensor's real byte extent in the file. For a transparently expanded
    /// BF16L3 tensor this is the compressed range, which is what page-eviction
    /// hints and raw reads need — `info.data_size` is the expanded length.
    fn physical_range(&self, idx: usize) -> (usize, usize) {
        match self.bf16_packed[idx] {
            Some((_, off, len)) => (off, len),
            None => (self.tensors[idx].data_offset, self.tensors[idx].data_size),
        }
    }

    /// Whether this tensor is stored BF16L3-compressed on disk and expanded on
    /// read. Such a tensor cannot be slab-loaded or mmap-borrowed: its file
    /// bytes are not the BF16 buffer the index advertises.
    /// The encoding actually stored on disk for `name`: its `quant_type` byte
    /// and its on-disk byte length.
    ///
    /// For a losslessly-recoded tensor both differ from the index's reported
    /// `quant_type`/`data_size`, which this reader rewrites to the expanded
    /// view so consumers need no per-codec branch. Inspection tooling wants the
    /// stored truth instead — without this it cannot tell a compressed artifact
    /// from a plain one.
    pub fn stored_encoding(&self, name: &str) -> Option<(u8, usize)> {
        let idx = self.resolve_idx(name)?;
        Some(match self.bf16_packed[idx] {
            Some((qt, _, len)) => (qt, len),
            None => (self.tensors[idx].quant_type, self.tensors[idx].data_size),
        })
    }

    pub fn is_bf16_expanded(&self, name: &str) -> bool {
        self.resolve_idx(name)
            .is_some_and(|i| self.bf16_packed[i].is_some())
    }

    /// True if any tensor is a transparently expanded compressed-BF16 tensor.
    pub fn has_bf16_expanded(&self) -> bool {
        self.bf16_packed.iter().any(|p| p.is_some())
    }

    fn find_tensor(&self, name: &str) -> Option<&HfqTensorInfo> {
        self.resolve_idx(name).map(|i| &self.tensors[i])
    }

    /// Returns the name of the first tensor whose `quant_type` matches `qt`,
    /// or `None` if none match. Used by the daemon's DFlash-refusal guard to
    /// detect MQ3/MQ2 body weights without iterating the index outside this
    /// module.
    pub fn first_tensor_with_quant_type(&self, qt: u8) -> Option<&str> {
        self.tensors
            .iter()
            .find(|t| t.quant_type == qt)
            .map(|t| t.name.as_str())
    }

    /// All tensors in index order. For tools that scan the file (e.g.
    /// dump_norms, quant_quality_mse, compare_hfq) — the engine itself
    /// looks tensors up by name via `find_tensor_info` /
    /// `tensor_data_vec`.
    pub fn tensors(&self) -> &[HfqTensorInfo] {
        &self.tensors
    }

    /// Stable content fingerprint over the metadata and tensor index — NOT the
    /// payload, so it is index-only cheap even on a 40 GB artefact.
    ///
    /// Recorded by calibration writers as `source_fingerprint` so a `.calib.hfq`
    /// can be tied back to the exact artefact it was captured from. A calib is
    /// only valid for the weights it saw; matching the source PATH proves
    /// nothing once that path is rebuilt. Scope matches the
    /// `hfq_metadata_and_tensor_index_v1` fingerprint `hipfire-coexistence`
    /// reports, but is computed independently — that one is `pub(crate)` there
    /// and the runtime must not depend on the coexistence crate.
    pub fn index_fingerprint(&self) -> String {
        // FNV-1a/64. Chosen over a cryptographic hash because this detects
        // accidental mismatch, not tampering, and must stay cheap.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |bytes: &[u8]| {
            for b in bytes {
                h ^= *b as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        };
        eat(self.metadata_json.as_bytes());
        eat(&self.arch_id.to_le_bytes());
        for t in &self.tensors {
            eat(t.name.as_bytes());
            eat(&[t.quant_type]);
            eat(&t.group_size.to_le_bytes());
            eat(&t.data_size.to_le_bytes());
            for d in &t.shape {
                eat(&d.to_le_bytes());
            }
        }
        format!("fnv64:{h:016x}")
    }

    /// The lossless recoding a tensor is STORED as, if any: `(quant_type,
    /// packed byte length)`.
    ///
    /// `tensors()` reports the LOGICAL view — `expand_bf16_index` rewrites a
    /// recoded entry's dtype and length at open, so a BF16L3-compressed tensor
    /// reads as plain `BF16` there. That is right for anyone consuming values
    /// and wrong for anyone asking what is on disk, which is what this answers.
    /// `None` means the tensor is stored exactly as `tensors()` describes it.
    ///
    /// Indices match `tensors()`.
    pub fn stored_recoding(&self, idx: usize) -> Option<(u8, usize)> {
        self.bf16_packed
            .get(idx)
            .copied()
            .flatten()
            .map(|(qt, _off, size)| (qt, size))
    }

    /// Whether this file supports the `O_DIRECT` GPU slab loader. True for a
    /// real file-backed HFQM; false for a safetensors-backed in-memory file
    /// (`from_safetensors`), whose bytes are spread across shards with no
    /// single `O_DIRECT`-openable path and unused `data_offset`s. Such callers
    /// fall back to the per-tensor `tensor_data_vec` path, which serves from
    /// the shard mmaps.
    pub fn supports_slab_load(&self) -> bool {
        // A BF16L3 tensor needs a CPU decode between file and VRAM, which the
        // slab loader's raw file→GPU copy cannot do. Fall back to the
        // per-tensor path for the whole file rather than silently uploading
        // compressed bytes tagged as BF16.
        self.st_source.is_none() && !self.has_bf16_expanded()
    }

    pub fn modules(&self) -> &[HfqModuleRecord] {
        &self.modules
    }
}

/// Parse the metadata+index block of one HFQM container.
///
/// `base` is the container's start within the file — nonzero for a container
/// embedded in another file (a bundled LoRA or MTP head). `metadata_offset` and
/// `data_offset` arrive already rebased by it; `base` is needed here because a
/// v2 index stores each tensor's offset relative to the container, so it has to
/// be rebased the same way. `base` is 0 for a standalone file.
fn parse_hfqm_meta_index(
    meta_index: &[u8],
    metadata_offset: usize,
    data_offset: usize,
    base: usize,
    n_tensors: usize,
    file_len: usize,
    version: u32,
    mut read_tail: impl FnMut(usize, usize) -> std::io::Result<Vec<u8>>,
) -> std::io::Result<(String, Vec<HfqTensorInfo>, HashMap<String, usize>)> {
    let json_end = json_blob_end(meta_index).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "HFQM metadata JSON did not end",
        )
    })?;
    let mut metadata_json = String::from_utf8_lossy(&meta_index[..json_end]).to_string();
    if version >= 2 {
        metadata_json = merge_tail_metadata(metadata_json, &mut read_tail)?;
    }
    let mut pos = json_end;
    if pos + 4 > meta_index.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "HFQM index missing tensor count",
        ));
    }
    let idx_n = u32::from_le_bytes(meta_index[pos..pos + 4].try_into().unwrap()) as usize;
    if idx_n != n_tensors {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("HFQM index count {idx_n} != header count {n_tensors}"),
        ));
    }
    pos += 4;

    let mut tensors = Vec::with_capacity(n_tensors);
    let mut tensor_map = HashMap::new();
    let mut cumulative_offset = data_offset;
    for i in 0..n_tensors {
        if pos + 2 > meta_index.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "HFQM index truncated at name length",
            ));
        }
        let name_len = u16::from_le_bytes(meta_index[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        if pos + name_len + 2 > meta_index.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "HFQM index truncated at name/shape header",
            ));
        }
        let name = String::from_utf8_lossy(&meta_index[pos..pos + name_len]).to_string();
        pos += name_len;
        let quant_type = meta_index[pos];
        pos += 1;
        let n_dims = meta_index[pos] as usize;
        pos += 1;
        let per_entry_tail = if version >= 2 { 20 } else { 12 };
        if pos + n_dims * 4 + per_entry_tail > meta_index.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "HFQM index truncated at shape/data_size",
            ));
        }
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(u32::from_le_bytes(
                meta_index[pos..pos + 4].try_into().unwrap(),
            ));
            pos += 4;
        }
        let group_size = u32::from_le_bytes(meta_index[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let data_size = u64::from_le_bytes(meta_index[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let tensor_offset = if version >= 2 {
            let offset_div32 =
                u64::from_le_bytes(meta_index[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            // Stored relative to the container, so rebase like the header
            // offsets. Without this an embedded container reads every tensor
            // `base` bytes early — out of the host file's own data — returning
            // plausible-looking garbage instead of failing.
            //
            // Alignment is a property of the container's own layout, so it is
            // checked before rebasing: `base` is where the host file happened to
            // end and is under no obligation to be 32-byte aligned.
            let relative = offset_div32
                .checked_mul(HFQM_V2_OFFSET_ALIGN)
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "HFQM v2 offset overflow")
                })?;
            if relative % HFQM_V2_OFFSET_ALIGN != 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("HFQM tensor {name} offset {relative} is not 32-byte aligned"),
                ));
            }
            relative.checked_add(base).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "HFQM v2 offset overflow")
            })?
        } else {
            let offset = cumulative_offset;
            cumulative_offset = cumulative_offset.saturating_add(data_size);
            offset
        };
        if tensor_offset + data_size > file_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "HFQM tensor {name} range {}..{} exceeds file size {}",
                    tensor_offset,
                    tensor_offset + data_size,
                    file_len
                ),
            ));
        }
        tensor_map.insert(name.clone(), i);
        tensors.push(HfqTensorInfo {
            name,
            quant_type,
            shape,
            group_size,
            data_offset: tensor_offset,
            data_size,
        });
    }
    let _ = metadata_offset;
    Ok((metadata_json, tensors, tensor_map))
}

fn read_exact_at_portable(file: &File, dst: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(dst, offset)?;
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let mut f = file.try_clone()?;
        f.seek(SeekFrom::Start(offset))?;
        f.read_exact(&mut dst)?;
        Ok(())
    }
}

// ─── ModelSource impl for HfqFile ───────────────────────────────────────────

impl ModelSource for HfqFile {
    fn metadata_json(&self) -> &str {
        &self.metadata_json
    }

    fn arch_id(&self) -> u32 {
        self.arch_id
    }

    fn quant_config(&self) -> Option<&QuantConfig> {
        None // HFQ files encode quant_type per-tensor, not via a global config
    }

    fn tensor_data(&self, name: &str) -> Option<(&TensorInfo, &[u8])> {
        // Borrowed-only, per the trait contract: a tensor whose stored bytes
        // are not its logical bytes has no logical buffer to borrow, so this
        // must decline rather than pair packed bytes with the logical
        // `TensorInfo`. Doing that pairing is a genuinely nasty failure — the
        // metadata says n*2 bytes of bf16 and the slice is the packed length,
        // so the caller reads a plausible-looking tensor of the wrong size
        // ("embedding payload length does not match its declared shape").
        //
        // Two distinct cases must both decline: `bf16_packed` (transparently
        // expanded, which the inherent accessor already refuses) and a
        // deliberately-resident coding the index still reports as packed.
        let idx = self.resolve_idx(name)?;
        if is_packed_bf16(self.tensors.get(idx)?.quant_type) {
            return None;
        }
        // Nothing to borrow: weights are read with pread into owned buffers, so
        // there is no mapping to hand a slice into. The trait spells this case
        // out — a source with nowhere to put the decoded buffer MUST return
        // `None` here and let consumers use `tensor()`, which this impl
        // overrides with the Cow path. Declining is the contract, not a gap.
        let _ = idx;
        None
    }

    fn tensor(&self, name: &str) -> Option<(&TensorInfo, std::borrow::Cow<'_, [u8]>)> {
        let idx = self.resolve_idx(name)?;
        // LOGICAL bytes, matching the logical `TensorInfo` from `ms_infos`.
        // Borrowed when stored verbatim, owned when a recoding — including a
        // deliberately-resident Bf16Lut3 head — has to be expanded. That
        // expansion is the whole reason the trait needed a Cow.
        let stored_qt = self.tensors.get(idx)?.quant_type;
        let bytes = if is_packed_bf16(stored_qt) {
            // Still packed in the index — `expand_bf16_index` declined to
            // expand it (a resident Bf16Lut3 head). Expand here so the payload
            // matches the logical `TensorInfo`.
            std::borrow::Cow::Owned(self.tensor_data_logical(name).ok()?.1)
        } else {
            // Verbatim, or already transparently expanded: keep the borrow.
            self.tensor_data_cow(name)?.1
        };
        Some((self.ms_infos().get(idx)?, bytes))
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        let idx = self.resolve_idx(name)?;
        self.ms_infos().get(idx)
    }

    fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    fn path(&self) -> &std::path::Path {
        &self.path
    }

    fn chat_template(&self) -> Option<String> {
        // Delegate to HfqFile's own chat_template method
        HfqFile::chat_template(self)
    }
}

// ─── Config from HFQ metadata ───────────────────────────────────────────────

pub fn config_from_hfq(hfq: &HfqFile) -> Option<LlamaConfig> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config")?;

    let arch_str = config.get("model_type")?.as_str()?;
    let arch = match arch_str {
        "llama" => ModelArch::Llama,
        "qwen3" | "qwen2" => ModelArch::Qwen3,
        _ => ModelArch::Llama,
    };

    let dim = config.get("hidden_size")?.as_u64()? as usize;
    let n_layers = config.get("num_hidden_layers")?.as_u64()? as usize;
    let n_heads = config.get("num_attention_heads")?.as_u64()? as usize;
    let n_kv_heads = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(n_heads as u64) as usize;
    let hidden_dim = config.get("intermediate_size")?.as_u64()? as usize;
    let vocab_size = config.get("vocab_size")?.as_u64()? as usize;
    let norm_eps = config
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5) as f32;
    let max_seq_len = config
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(2048) as usize;
    let rope_freq_base = config
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;

    let has_qk_norm = hfq
        .find_tensor("model.layers.0.self_attn.q_norm.weight")
        .is_some();

    let head_dim = config
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(dim / n_heads);

    let bos_token = config
        .get("bos_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as u32;
    let eos_token = config
        .get("eos_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as u32;

    Some(LlamaConfig {
        arch,
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        head_dim,
        norm_eps,
        max_seq_len,
        rope_freq_base,
        bos_token,
        eos_token,
        has_qk_norm,
    })
}

// ─── Weight Loading ─────────────────────────────────────────────────────────

/// Load a tensor as F32 on GPU (for norms, embeddings).
fn load_f16_tensor(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    st_name: &str,
    shape: &[usize],
) -> HipResult<GpuTensor> {
    // `tensor_data_cow`: a losslessly recoded (LUT3/Huffman) tensor is stored
    // compressed and the borrowing accessor refuses it, which would report a
    // present tensor as "not found". Borrows when stored plainly.
    let (info, data) = hfq
        .tensor_data_cow(st_name)
        .unwrap_or_else(|| panic!("tensor not found: {st_name}"));
    let data = data.as_ref();

    let f32_data: Vec<f32> = match info.quant_type {
        1 => {
            // F16
            data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
        2 => {
            // F32
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        16 => {
            // BF16 (QuantType::BF16). The qtip3 path stores 1-D norms (and other
            // non-256-divisible tensors) as BF16; the llama loader must accept
            // them for norm/embedding tensors. bf16→f32 = high 16 bits.
            data.chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect()
        }
        49 | 50 => {
            // A lossless recoding that residency (BF16L3) or the codec itself
            // (Huffman) left packed. Decode rather than panic: this is reached
            // for norms and other small tensors, which have no packed-consumer
            // kernel and whose decode cost is negligible.
            //
            // HIPFIRE_BF16L3_RESIDENT is global, so turning it on to keep the
            // lm_head packed also leaves every other bf16 tensor packed — norms
            // included. Before this arm that panicked with
            // `got quant_type=49` on model.norm.weight.
            let n: usize = info.shape.iter().map(|&d| d as usize).product();
            let logical = decode_bf16_packed(info.quant_type, data, n).unwrap_or_else(|| {
                panic!(
                    "failed to decode recoded tensor {st_name} (quant_type={})",
                    info.quant_type
                )
            });
            logical
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect()
        }
        _ => panic!(
            "expected F16/F32/BF16 tensor for {st_name}, got quant_type={}",
            info.quant_type
        ),
    };

    gpu.upload_f32(&f32_data, shape)
}

/// Load an AWQ scale sidecar tensor from an HFQ file onto GPU.
///
/// Phase A Stage A — AWQ sidecar lookup. The quantizer emits per-tensor
/// sidecars named `<weight_name>.awq_scale.weight` (1D F16, length K)
/// alongside MQ4-quantized weights. The forward path uses these to apply
/// `x /= awq_scale` before the rotation kernel, completing the AWQ
/// math `(W·s) · (x/s) = W·x`. Backward-compatible: when no sidecar
/// exists (the common case for pre-Stage-A .hfq files), this returns
/// None and the runtime behaves identically to before.
///
/// Naming convention: replace trailing `.weight` with `.awq_scale.weight`.
/// Matches hipfire-quantize's emit pattern.
///
/// Internally uses `tensor_data_vec`, which on Unix routes through pread
/// + fadvise_dontneed (avoids page cache buildup on unified-memory APUs)
/// and on non-Unix falls back to mmap. Sidecars are small (K ≤ ~12288
/// elements, ~48 KB peak), so the owned-Vec copy is negligible.
pub fn load_awq_scale(hfq: &HfqFile, gpu: &Gpu, weight_name: &str, k: usize) -> Option<GpuTensor> {
    let sidecar_name = match weight_name.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.awq_scale.weight"),
        None => format!("{weight_name}.awq_scale.weight"),
    };
    let (sc_info, sc_data) = hfq.tensor_data_vec(&sidecar_name)?;
    // Must be 1D F16, length K. quant_type 1 = F16 per the existing
    // load_f16_tensor path.
    if sc_info.quant_type != 1 {
        eprintln!(
            "warning: AWQ sidecar {sidecar_name} has quant_type={} (expected 1=F16); skipping",
            sc_info.quant_type
        );
        return None;
    }
    if sc_info.shape.len() != 1 || sc_info.shape[0] as usize != k {
        eprintln!(
            "warning: AWQ sidecar {sidecar_name} shape mismatch ({:?} vs expected [{}]); skipping",
            sc_info.shape, k
        );
        return None;
    }
    // Convert F16 → F32 on host before upload, so the kernel receives
    // a `const float*` and doesn't need <hip/hip_fp16.h>. The 2× VRAM
    // cost vs raw F16 is negligible at these sizes.
    let f32_data: Vec<f32> = sc_data
        .chunks_exact(2)
        .map(|c| crate::quant::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|&v| v.to_le_bytes()).collect();
    gpu.upload_raw(&f32_bytes, &[f32_bytes.len()]).ok()
}

/// Load a weight tensor (quantized or F16) onto GPU.
fn load_weight_tensor(
    hfq: &HfqFile,
    gpu: &Gpu,
    st_name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    // `tensor_data_cow`: a losslessly recoded (LUT3/Huffman) tensor is stored
    // compressed and the borrowing accessor refuses it, which would report a
    // present tensor as "not found". Borrows when stored plainly.
    let (info, data) = hfq
        .tensor_data_cow(st_name)
        .unwrap_or_else(|| panic!("tensor not found: {st_name}"));
    let data = data.as_ref();

    let wt_result: HipResult<WeightTensor> = match info.quant_type {
        0 => {
            // Q4F16G64
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok::<WeightTensor, HipError>(WeightTensor {
                buf,
                gpu_dtype: DType::Q4F16G64,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        3 => {
            // Q8F16 — same block format as GGML Q8_0 (34 bytes per 32 elements)
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::Q8_0,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        4 => {
            // Q4_K — GGML-compatible Q4_K blocks (144 bytes per 256 elements)
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::Q4K,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        5 => {
            // Q8HFQ — split-metadata layout (scales then values, 128B-aligned rows)
            let n_groups = k / 32;
            let raw_row = n_groups * 2 + k;
            let row_stride = (raw_row + 127) & !127;
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::Q8HFQ,
                m,
                k,
                row_stride,
                paro: None,
                awq_scale: None,
            })
        }
        6 => {
            // HFQ4-G256 — flat 4-bit, 136 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ4G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        7 => {
            // HFQ4-G128 — flat 4-bit, 72 bytes per 128 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ4G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        28 => {
            // PARO4-G128 — ParoQuant rotated-activation W4 probe format
            assert!(
                k % 128 == 0,
                "PARO4G128 weight {st_name} has K={k} but kernel requires K%128==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::ParoQ4G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        29 => {
            // PARO4-G128T — same metadata, qweight retiled as [M/8, K] for GEMV
            assert!(
                k % 128 == 0,
                "PARO4G128T weight {st_name} has K={k} but kernel requires K%128==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::ParoQ4G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        8 => {
            // HFQ6-G256 — 6-bit, 200 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ6G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        9 => {
            // HFQ2-G256 — flat 2-bit, 72 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ2G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        10 => {
            // HFQ2-G128 — flat 2-bit, 40 bytes per 128 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ2G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        11 => {
            // HFQ3-G256 — flat 3-bit, 104 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ3G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        12 => {
            // HFQ3-G128 — flat 3-bit, 56 bytes per 128 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ3G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        13 => {
            // MQ4-G256 — MagnumQuant FWHT-rotated 4-bit
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        14 => {
            // MQ8-G256 — MagnumQuant FWHT-rotated symmetric INT8, dp4a
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ8G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        17 => {
            // MQ3-G256 — MagnumQuant FWHT-rotated 3-bit, 104 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        18 => {
            // MQ2-G256 — MagnumQuant FWHT-rotated 2-bit, 72 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ2G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        19 => {
            // MQ2-G256-Lloyd — 2-bit + 4-entry fp16 codebook, 72 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ2G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        20 => {
            // MQ3-G256-Lloyd — 3-bit + 8-entry fp16 codebook, 112 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        21 => {
            // HFP4G32 — E2M1 + UE8M0 g32 + FP16 row scale.
            // Per-row hdr 16 B + (K/32) blocks × 17 B. See docs/quant-formats/hfp4.md.
            // K%256 — kernel constraint (gemv_hfp4g32 in dispatch.rs);
            // refuse here so a stale or externally-quantized file fails at
            // load instead of panicking on first dispatch.
            assert!(
                k % 256 == 0,
                "HFP4G32 v1 weight {st_name} has K={k} but kernel requires K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFP4G32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        24 => {
            // MFP4G32 — HFP4G32 + offline FWHT rotation (drop-in MQ4 replacement).
            // Same byte layout as qtype 21; format_flags=0x05 in row hdr.
            // See docs/quant-formats/hfp4.md.
            assert!(
                k % 256 == 0,
                "MFP4G32 weight {st_name} has K={k} but kernel + FWHT both require K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP4G32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        30 => {
            // MQ4-G256-Lloyd — 4-bit + 16-entry fp16 codebook, 160 bytes per 256 elements.
            // Renumbered from qtype 21 → 30 in mq4-lloyd merge to avoid HFP4G32=21 collision.
            // Models quantized pre-renumber MUST be re-quantized.
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        1 => {
            // F16 — keep native so `weight_gemm` takes the batched
            // `gemm_f16_x_f32_wmma` path instead of upcasting to F32 and falling
            // back to a per-token GEMV. Needs K % 16 == 0 (16-wide WMMA K
            // fragments); non-aligned linears stay on the F32 upcast path.
            // `HIPFIRE_BF16_WEIGHTS=f32` forces the old upcast (rollback).
            let force_f32 = hipfire_env::BF16_WEIGHTS.get().as_deref() == Some("f32");
            if k % 16 == 0 && !force_f32 {
                let mut buf = gpu.upload_raw(data, &[data.len()])?;
                buf.dtype = DType::F16;
                Ok(WeightTensor {
                    buf,
                    gpu_dtype: DType::F16,
                    m,
                    k,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                })
            } else {
                let f32_data: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
                };
                let buf = gpu.upload_raw(bytes, &[m, k])?;
                Ok(WeightTensor {
                    buf,
                    gpu_dtype: DType::F32,
                    m,
                    k,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                })
            }
        }
        16 => {
            // BF16 — keep native so `weight_gemm` takes the batched
            // `gemm_bf16_x_bf16_wmma` path (+ the m128 overlay) and decode uses
            // the bf16 `weight_gemv`, instead of upcasting to F32 and falling
            // back to a per-token GEMV loop (the arch-0 prefill bottleneck).
            // The bf16 WMMA kernel reads 16-wide K fragments, so it needs
            // K % 16 == 0; the rare non-aligned linear stays on the F32 upcast
            // path for correctness. `HIPFIRE_BF16_WEIGHTS=f32` forces the old
            // upcast everywhere (rollback / debugging).
            let force_f32 = hipfire_env::BF16_WEIGHTS.get().as_deref() == Some("f32");
            if k % 16 == 0 && !force_f32 {
                let mut buf = gpu.upload_raw(data, &[data.len()])?;
                buf.dtype = DType::BF16;
                Ok(WeightTensor {
                    buf,
                    gpu_dtype: DType::BF16,
                    m,
                    k,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                })
            } else {
                let f32_data: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                    .collect();
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
                };
                let buf = gpu.upload_raw(bytes, &[m, k])?;
                Ok(WeightTensor {
                    buf,
                    gpu_dtype: DType::F32,
                    m,
                    k,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                })
            }
        }
        qt @ (31 | 51) => {
            // Qtip3G256 (31) / Qtip3G256I3 (51) — packed bitshift-trellis
            // (100 B/group), served by gemv_qtip3g256 and gemv_qtip3g256i3
            // respectively. IDENTICAL bytes; the code selects which computed
            // codebook the kernel decodes with, which is exactly why they are two
            // codes: cross-decoding produces noise that no structural check sees.
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: if qt == 51 {
                    DType::Qtip3G256I3
                } else {
                    DType::Qtip3G256
                },
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        42 => {
            // Qtip4G256 — 4-bit bitshift-trellis (132 B/group), served by the
            // gemv_qtip4g256 kernel. Same plain-llama path as qtip3; the on-disk
            // bytes are uploaded raw and decoded on-the-fly (nibble unpack).
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::Qtip4G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        // Opus int8-activation family (qt 35 = OQ8 W8A8, 33 = OQ+ W4A8, 36 = OQ+
        // compact) — the shared `oq8_arch_load` repacks into the arch-combined
        // Oq8G256 device layout the iu8 GEMV/GEMM consume. Single source of truth
        // for these codes across every per-arch loader (qwen2 / nemotron mirror
        // this call); only OQ4 W4A4 (below) was wired here before, so families
        // loading through `load_weights_hfq` panicked on 35 despite the generic
        // dtype-dispatched kernels already supporting them.
        qt @ (33 | 35 | 36 | 38 | 52) => {
            let (bytes, gpu_dtype) = oq8_arch_load(qt, data, m, k)
                .expect("oq8_arch_load resolves the OQ8-family codes 33/35/36/38/52");
            let buf = gpu.upload_raw(&bytes, &[bytes.len()])?;
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
        // Oq4G256 (34, canonical) / Oq4G256ArchPacked (37, pre-packed) — Opus Quant
        // symmetric W4 (int4 per-256-group, FWHT-256 rotated). The shared
        // `oq4_arch_load` repacks the canonical form into the arch combined device
        // layout the Oq4 GEMV (`weight_gemv` Oq4G256 arm) and batched GEMM consume,
        // or borrows the pre-packed layout verbatim (zero-copy). K must be % 256
        // (non-divisible linears, e.g. down_proj k=1408, stay BF16). This wires oq4
        // into the dense-llama path — incl. SpinQuant-R1 `.hfq` (`--rotate`), whose
        // per-group FWHT is exactly the activation rotation this GEMV applies.
        OQ4_CANONICAL_QT | OQ4_ARCH_PACKED_QT => {
            let (bytes, gpu_dtype) = oq4_arch_load(info.quant_type, data, m, k)
                .expect("oq4_arch_load resolves the OQ4 canonical/arch-packed codes");
            let buf = gpu.upload_raw(&bytes, &[bytes.len()])?;
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
        49 | 50 => {
            // A lossless recoding left packed — BF16L3 under residency, or
            // Huffman. Decode to plain bf16 rather than keep it packed.
            //
            // Only a GEMV consumer can read the packed form: `gemv_bf16l3_xf32`
            // is batch-1, and there is no BF16L3 GEMM, so a layer weight kept
            // packed would work at decode and break at prefill. The lm_head is
            // different — it is GEMV-only — and stays packed via its own branch
            // in `load_weights_hfq`, which runs before this.
            //
            // HIPFIRE_BF16L3_RESIDENT is global, so enabling it to pack the head
            // also leaves every layer weight packed on an all-bf16 model. Before
            // this arm that panicked here with `unsupported quant_type 49`.
            let n: usize = info.shape.iter().map(|&d| d as usize).product();
            let logical = decode_bf16_packed(info.quant_type, data, n).unwrap_or_else(|| {
                panic!(
                    "failed to decode recoded weight {st_name} (quant_type={})",
                    info.quant_type
                )
            });
            let mut buf = gpu.upload_raw(&logical, &[m, k])?;
            buf.dtype = DType::BF16;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::BF16,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        _ => panic!(
            "unsupported quant_type {} for weight {st_name}",
            info.quant_type
        ),
    };
    let mut wt = wt_result?;
    // Centralized AWQ sidecar attachment. Replaces the prior per-arm
    // inline `load_awq_scale()` calls at the qt=13 / qt=17 arms — those
    // were the only loaders touching `awq_scale` and missing arms (qt=15
    // MQ6, qt=18 MQ2, qt=19/20 Lloyd, qt=24 MFP4) would silently drop
    // sidecars if added later. Routed through `DType::supports_awq_sidecar`
    // so future widening is a single helper edit, not a scattered
    // per-loader hunt. See dispatch.rs for the allow-list rationale.
    if wt.gpu_dtype.supports_awq_sidecar() {
        wt.awq_scale = load_awq_scale(hfq, gpu, st_name, k);
    }
    Ok(wt)
}

/// Load LLaMA weights from an HFQ file onto GPU.
pub fn load_weights_hfq(
    hfq: &HfqFile,
    config: &LlamaConfig,
    gpu: &mut Gpu,
) -> HipResult<LlamaWeights> {
    // R2 guard: the LLaMA-family loader does NOT read Q/K/V proj bias —
    // `LayerWeights` has no `wq_bias` / `wk_bias` / `wv_bias` fields and
    // the per-layer load below only names `*.q_proj.weight`. Qwen2
    // requires those biases (`attention_bias=true` is the modeling
    // default). The quantiser used to auto-tag every Qwen2 model as
    // `arch_id=1`, which the daemon dispatches to this loader; the
    // result was silently-wrong outputs with no warning. As of the
    // `--arch-id` flag (see `hipfire-quantize`), Qwen2 models should be
    // tagged `arch_id=7` and dispatched to `hipfire-arch-qwen2`.
    //
    // If we see `q_proj.bias` while loading as the LLaMA family, the
    // input is a mis-tagged Qwen2 HFQ. Refuse hard with a pointer at
    // the correct path. (Detection by manifest is robust to either the
    // model_type tag or the model family — both LLaMA and Qwen3 lack
    // these bias tensors, so any HFQ with `model.layers.0.self_attn.q_proj.bias`
    // is by definition a Qwen2-family input.)
    if hfq
        .find_tensor_info("model.layers.0.self_attn.q_proj.bias")
        .is_some()
    {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "refusing to load Qwen2 HFQ through the LLaMA family path: \
                 tensor `model.layers.0.self_attn.q_proj.bias` is present, \
                 which means this is a Qwen2 (attention_bias=true) model. \
                 The LLaMA loader drops Q/K/V proj bias and would produce \
                 wrong outputs. \
                 Current HFQ arch_id = {}. Re-quantise with \
                 `hipfire-quantize --arch-id 7 ...` so the daemon \
                 dispatches arch_id=7 to hipfire-arch-qwen2 (once that \
                 crate is wired in), or — for inspection only — load \
                 directly via `cargo run --example inspect_hfq -p \
                 hipfire-arch-qwen2 -- <path>`. \
                 See docs/plans/dots-ocr-devlog.md §7.",
                hfq.arch_id
            ),
        ));
    }

    eprintln!("  loading token_embd...");
    // `tensor_data_cow`, not `tensor_data`: a LUT3/Huffman-recoded embed table
    // is stored compressed, and the borrowing accessor refuses those. Reaching
    // for it here reported the tensor as MISSING when it was merely coded.
    let embd_info = hfq
        .tensor_data_cow("model.embed_tokens.weight")
        .expect("embed_tokens not found");
    let embd_info = (embd_info.0, embd_info.1.as_ref());
    let (token_embd, embd_fmt) = if embd_info.0.quant_type == 4 {
        // Q4_K: upload raw, use Q4K embedding lookup at inference
        eprintln!("    (Q4K raw, {} MB)", embd_info.1.len() / 1_000_000);
        (
            gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?,
            EmbeddingFormat::Q4K,
        )
    } else if embd_info.0.quant_type == 6 {
        eprintln!("    (HFQ4-G256 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (
            gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?,
            EmbeddingFormat::HFQ4G256,
        )
    } else if embd_info.0.quant_type == 7 {
        eprintln!("    (HFQ4-G128 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (
            gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?,
            EmbeddingFormat::HFQ4G128,
        )
    } else if embd_info.0.quant_type == 3 {
        // Q8F16: upload raw, use Q8 embedding lookup at inference
        eprintln!("    (Q8 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (
            gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?,
            EmbeddingFormat::Q8_0,
        )
    } else if embd_info.0.quant_type == 49 {
        // BF16L3 that residency kept packed. The GATHER cannot use it: a lookup
        // reads one arbitrary row, and BF16L3's escape plane is only addressable
        // by walking a block, so there is no gather kernel for it. Decode here.
        //
        // This is what HIPFIRE_BF16L3_RESIDENT's "a gather-read table ... will
        // fail to load" refers to; before this arm it panicked in
        // `load_f16_tensor` with `got quant_type=49`.
        //
        // The tied lm_head is a separate buffer and DOES stay packed — it is a
        // pure GEMV consumer, which `gemv_bf16l3_xf32` serves. So residency buys
        // the head's bandwidth without the gather losing its random access.
        let n = embd_info.0.shape.iter().map(|&d| d as usize).product();
        let logical = decode_bf16_packed(49, embd_info.1, n)
            .ok_or_else(|| HipError::new(0, "token_embd: BF16L3 decode failed"))?;
        eprintln!(
            "    (bf16l3 -> bf16 for gather, {} MB packed -> {} MB)",
            embd_info.1.len() / 1_000_000,
            logical.len() / 1_000_000
        );
        (
            gpu.upload_raw(&logical, &[logical.len()])?,
            EmbeddingFormat::BF16,
        )
    } else if embd_info.0.quant_type == 16 {
        // Native bf16 table: upload raw 2 B/elem; gather converts to f32 inline
        // (no F32 promotion). Keeps the largest tensor at half the memory.
        eprintln!("    (bf16 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (
            gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?,
            EmbeddingFormat::BF16,
        )
    } else if embd_info.0.quant_type == 1 {
        // Native f16 table: upload raw; gather converts f16->f32 inline.
        eprintln!("    (f16 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (
            gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?,
            EmbeddingFormat::F16,
        )
    } else {
        (
            load_f16_tensor(
                hfq,
                gpu,
                "model.embed_tokens.weight",
                &[config.vocab_size, config.dim],
            )?,
            EmbeddingFormat::F32,
        )
    };

    eprintln!("  loading output_norm...");
    let output_norm = load_f16_tensor(hfq, gpu, "model.norm.weight", &[config.dim])?;

    eprintln!("  loading output...");
    let mut return_packed_head: Option<WeightTensor> = None;
    let output = if hfq.find_tensor("lm_head.weight").is_some() {
        load_weight_tensor(hfq, gpu, "lm_head.weight", config.vocab_size, config.dim)?
    } else {
        // Tied embeddings — reuse token_embd as F32 output weights for the
        // logit GEMV. Dequant by the embed's actual format (qtip3 models store
        // embed as Q8F16; the old code assumed F16 and would garble Q8 bytes).
        // Same reason as the load above: a recoded table must be decoded here,
        // or the tied-output dequant reads packed bytes as if they were BF16.
        let (embd_t, data) = hfq.tensor_data_cow("model.embed_tokens.weight").unwrap();
        let data = data.as_ref();
        let n = config.vocab_size * config.dim;
        // A BF16 embedding is uploaded AS bf16 rather than widened.
        //
        // Widening recovers nothing — the stored value is already bf16, so f32
        // is exact re-encoding — while doubling the resident head and the
        // per-token read: 1050.7 MB against 525.3 MB at 128256 x 2048. It also
        // costs a ~1 GB host conversion at load, skipped entirely below.
        //
        // It used to be the right trade regardless, because BF16 had no GEMV
        // entry and `weight_gemv` fell back to a batch-1 WMMA GEMM: 14.6 ms
        // against `gemv_f32`'s 5.2 ms. `gemv_bf16_xf32` and its dispatch-family
        // registration remove that — bf16 weight against an f32 activation runs
        // in 3.2 ms and matches the widened path to 6.7e-8, f32 accumulation
        // noise. The widening is now pure cost.
        //
        // Only BF16 changes. Q8F16 / F16 / F32 embeddings still decode to f32,
        // their stored form not being one the GEMV path takes directly. The
        // embedding GATHER is untouched: `token_embd` is its own buffer, loaded
        // above.
        // BF16L3 stays PACKED when residency kept it so, and dispatches to
        // `gemv_bf16l3_xf32`: 1.917 ms against the plain-bf16 GEMV's 3.241 at
        // 128256 x 2048, reading 376.3 MB instead of 525.3. Lossless, and
        // verified at 2.570e-7 worst deviation with an identical argmax.
        //
        // Only reachable with HIPFIRE_BF16L3_RESIDENT set; without it
        // `expand_bf16_index` has already rewritten this entry to logical BF16
        // and decoded the bytes, so the arm below handles it.
        //
        // The kernel requires K % 256 == 0. Rather than assume a well-behaved
        // hidden size, decode explicitly when it does not hold — the bytes in
        // `data` are packed, so falling through to the f32 arm would read them
        // as raw bf16 and garble every logit.
        if embd_t.quant_type == 49 {
            if config.dim % 256 == 0 {
                let mut buf = gpu.upload_raw(data, &[data.len()])?;
                buf.dtype = DType::Bf16L3;
                return_packed_head = Some(WeightTensor {
                    buf,
                    gpu_dtype: DType::Bf16L3,
                    m: config.vocab_size,
                    k: config.dim,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                });
            } else {
                let logical = decode_bf16_packed(embd_t.quant_type, data, n)
                    .ok_or_else(|| HipError::new(0, "tied lm_head: BF16L3 decode failed"))?;
                let mut buf = gpu.upload_raw(&logical, &[config.vocab_size, config.dim])?;
                buf.dtype = DType::BF16;
                return_packed_head = Some(WeightTensor {
                    buf,
                    gpu_dtype: DType::BF16,
                    m: config.vocab_size,
                    k: config.dim,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                });
            }
        }
        if let Some(w) = return_packed_head.take() {
            w
        } else if embd_t.quant_type == 16 {
            let mut buf = gpu.upload_raw(data, &[config.vocab_size, config.dim])?;
            buf.dtype = DType::BF16;
            WeightTensor {
                buf,
                gpu_dtype: DType::BF16,
                m: config.vocab_size,
                k: config.dim,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            }
        } else {
            let f32_data: Vec<f32> = match embd_t.quant_type {
                3 => crate::quant::dequant_q8f16(data, n), // Q8F16: int8 + f16 scale, 34 B/block
                2 => data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                _ => data
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
            };
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[config.vocab_size, config.dim])?;
            WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m: config.vocab_size,
                k: config.dim,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            }
        }
    };

    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        crate::load_progress::report(i as u32 + 1, config.n_layers as u32, "weights");
        let p = format!("model.layers.{i}");
        let kv_dim = config.n_kv_heads * config.head_dim;
        let q_out_dim = config.n_heads * config.head_dim;

        let layer = LayerWeights {
            attn_norm: load_f16_tensor(
                hfq,
                gpu,
                &format!("{p}.input_layernorm.weight"),
                &[config.dim],
            )?,
            wq: load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.self_attn.q_proj.weight"),
                q_out_dim,
                config.dim,
            )?,
            wk: load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.self_attn.k_proj.weight"),
                kv_dim,
                config.dim,
            )?,
            wv: load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.self_attn.v_proj.weight"),
                kv_dim,
                config.dim,
            )?,
            wo: load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.self_attn.o_proj.weight"),
                config.dim,
                q_out_dim,
            )?,
            q_norm: if config.has_qk_norm {
                Some(load_f16_tensor(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.q_norm.weight"),
                    &[config.head_dim],
                )?)
            } else {
                None
            },
            k_norm: if config.has_qk_norm {
                Some(load_f16_tensor(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.k_norm.weight"),
                    &[config.head_dim],
                )?)
            } else {
                None
            },
            ffn_norm: load_f16_tensor(
                hfq,
                gpu,
                &format!("{p}.post_attention_layernorm.weight"),
                &[config.dim],
            )?,
            w_gate: load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.mlp.gate_proj.weight"),
                config.hidden_dim,
                config.dim,
            )?,
            w_up: load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.mlp.up_proj.weight"),
                config.hidden_dim,
                config.dim,
            )?,
            w_down: load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.mlp.down_proj.weight"),
                config.dim,
                config.hidden_dim,
            )?,
        };
        layers.push(layer);
    }

    Ok(LlamaWeights {
        token_embd,
        embd_format: embd_fmt,
        output_norm,
        output,
        layers,
    })
}

// ─── ParoQuant safetensors loading (LLaMA / Qwen3 arch) ────────────────────

/// Parse a LlamaConfig from a SafetensorsSource's metadata JSON.
/// The metadata JSON has structure: `{ "config": { ...config.json... } }`.
pub fn config_from_safetensors_llama(source: &dyn ModelSource) -> Option<LlamaConfig> {
    let meta: serde_json::Value = serde_json::from_str(source.metadata_json()).ok()?;
    let config = meta.get("config")?;

    let arch_str = config.get("model_type")?.as_str()?;
    let arch = match arch_str {
        "llama" | "mistral" => ModelArch::Llama,
        "qwen3" | "qwen2" => ModelArch::Qwen3,
        _ => ModelArch::Llama,
    };

    let dim = config.get("hidden_size")?.as_u64()? as usize;
    let n_layers = config.get("num_hidden_layers")?.as_u64()? as usize;
    let n_heads = config.get("num_attention_heads")?.as_u64()? as usize;
    let n_kv_heads = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(n_heads as u64) as usize;
    let hidden_dim = config.get("intermediate_size")?.as_u64()? as usize;
    let vocab_size = config.get("vocab_size")?.as_u64()? as usize;
    let norm_eps = config
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5) as f32;
    let max_seq_len = config
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(2048) as usize;
    let rope_freq_base = config
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;

    let head_dim = config
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(dim / n_heads);

    let bos_token = config
        .get("bos_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as u32;
    let eos_token = config
        .get("eos_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as u32;

    // Detect QK norm from tensor names
    let has_qk_norm = source
        .tensor_info("model.layers.0.self_attn.q_norm.weight")
        .is_some();

    Some(LlamaConfig {
        arch,
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        head_dim,
        norm_eps,
        max_seq_len,
        rope_freq_base,
        bos_token,
        eos_token,
        has_qk_norm,
    })
}

/// Repack AWQ-format INT4 weights into HFQ4G128 layout (ParoQuant uses AWQ packing).
///
/// SYNC: must match `repack_awq_to_hfq4g128` in
/// `crates/hipfire-arch-qwen35/src/qwen35.rs`. Duplicated to avoid a
/// cross-crate dependency cycle (the qwen35 crate already depends on this
/// one); keep the two bodies byte-identical when editing.
fn repack_awq_to_hfq4g128(
    qweight: &[u8],    // I32 raw bytes
    qzeros: &[u8],     // I32 raw bytes
    scales: &[u8],     // F16 raw bytes
    out_dim: usize,    // M (output features)
    in_dim: usize,     // K (input features)
    group_size: usize, // 128
) -> Vec<u8> {
    let groups_per_row = in_dim / group_size;
    let bytes_per_row = groups_per_row * 72;
    let mut out = vec![0u8; out_dim * bytes_per_row];

    debug_assert_eq!(
        qweight.as_ptr() as usize % 4,
        0,
        "AWQ qweight not 4-byte aligned"
    );
    let qw: &[u32] =
        unsafe { std::slice::from_raw_parts(qweight.as_ptr() as *const u32, qweight.len() / 4) };
    let qw_cols = out_dim / 8;

    debug_assert_eq!(
        qzeros.as_ptr() as usize % 4,
        0,
        "AWQ qzeros not 4-byte aligned"
    );
    let qz: &[u32] =
        unsafe { std::slice::from_raw_parts(qzeros.as_ptr() as *const u32, qzeros.len() / 4) };
    let qz_cols = out_dim / 8;

    debug_assert_eq!(
        scales.as_ptr() as usize % 2,
        0,
        "AWQ scales not 2-byte aligned"
    );
    let sc: &[u16] =
        unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u16, scales.len() / 2) };

    for m in 0..out_dim {
        for g in 0..groups_per_row {
            let row_off = m * bytes_per_row + g * 72;

            let scale_f16 = sc[g * out_dim + m];
            let scale_f32 = f16_to_f32(scale_f16);

            let zero_i32 = qz[g * qz_cols + m / 8];
            let zero_nibble = ((zero_i32 >> (AWQ_DEQUANT[m % 8] * 4)) & 0xF) as f32;
            let zero_f32 = -scale_f32 * zero_nibble;

            out[row_off..row_off + 4].copy_from_slice(&scale_f32.to_le_bytes());
            out[row_off + 4..row_off + 8].copy_from_slice(&zero_f32.to_le_bytes());

            const AWQ_DEQUANT: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
            let nibble_shift = AWQ_DEQUANT[m % 8] * 4;
            let qw_col = m / 8;
            for i in 0..64 {
                let in_idx0 = g * group_size + i * 2;
                let in_idx1 = in_idx0 + 1;

                let nib0 = ((qw[in_idx0 * qw_cols + qw_col] >> nibble_shift) & 0xF) as u8;
                let nib1 = ((qw[in_idx1 * qw_cols + qw_col] >> nibble_shift) & 0xF) as u8;

                out[row_off + 8 + i] = nib0 | (nib1 << 4);
            }
        }
    }

    out
}

/// Load a ParoQuant-quantized weight tensor from a safetensors source.
/// Repacks AWQ INT4 data to HFQ4G128 and uploads ParoQuant rotation metadata.
fn load_paroquant_weight_from_source(
    source: &dyn ModelSource,
    gpu: &Gpu,
    tensor_prefix: &str, // e.g. "model.layers.0.mlp.gate_proj"
    out_dim: usize,      // M
    in_dim: usize,       // K
    group_size: u32,
    krot: u8,
) -> HipResult<WeightTensor> {
    use crate::weights::ParoRotation;

    let qw_name = format!("{tensor_prefix}.qweight");
    let qz_name = format!("{tensor_prefix}.qzeros");
    let sc_name = format!("{tensor_prefix}.scales");
    let pairs_name = format!("{tensor_prefix}.pairs");
    let theta_name = format!("{tensor_prefix}.theta");
    let cs_name = format!("{tensor_prefix}.channel_scales");

    let (_, qw_data) = source
        .tensor_data(&qw_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {qw_name}")))?;
    let (_, qz_data) = source
        .tensor_data(&qz_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {qz_name}")))?;
    let (_, sc_data) = source
        .tensor_data(&sc_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {sc_name}")))?;

    let hfq_data = repack_awq_to_hfq4g128(
        qw_data,
        qz_data,
        sc_data,
        out_dim,
        in_dim,
        group_size as usize,
    );
    let buf = gpu.upload_raw(&hfq_data, &[hfq_data.len()])?;

    let (_, pairs_data) = source
        .tensor_data(&pairs_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {pairs_name}")))?;
    let (_, theta_data) = source
        .tensor_data(&theta_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {theta_name}")))?;
    let (_, cs_data) = source
        .tensor_data(&cs_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {cs_name}")))?;

    let pairs = gpu.upload_raw(pairs_data, &[pairs_data.len()])?;
    let theta = gpu.upload_raw(theta_data, &[theta_data.len()])?;
    let channel_scales = gpu.upload_raw(cs_data, &[cs_data.len()])?;

    Ok(WeightTensor {
        buf,
        gpu_dtype: DType::ParoQ4G128,
        m: out_dim,
        k: in_dim,
        row_stride: 0,
        paro: Some(ParoRotation {
            pairs,
            theta,
            channel_scales,
            krot: krot as u32,
            group_size,
            is_alias: false,
        }),
        awq_scale: None,
    })
}

/// Load an FP16 weight tensor from safetensors as F32 on GPU.
fn load_fp16_weight_tensor_from_source(
    source: &dyn ModelSource,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    let (_, data) = source
        .tensor_data(name)
        .ok_or_else(|| HipError::new(0, &format!("PARO tensor not found: {name}")))?;
    let f32_data: Vec<f32> = data
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4) };
    let buf = gpu.upload_raw(bytes, &[m, k])?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: DType::F32,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    })
}

/// Load a ParoQuant weight (quantized or FP16 fallback) using `model.` tensor prefix.
fn paro_load_llama_wt(
    source: &dyn ModelSource,
    gpu: &Gpu,
    prefix: &str, // e.g. "layers.0.self_attn.q_proj"
    m: usize,
    k: usize,
    gs: u32,
    kr: u8,
) -> HipResult<WeightTensor> {
    let fp = format!("model.{prefix}");
    if source.tensor_info(&format!("{fp}.qweight")).is_some() {
        load_paroquant_weight_from_source(source, gpu, &fp, m, k, gs, kr)
    } else {
        load_fp16_weight_tensor_from_source(source, gpu, &format!("{fp}.weight"), m, k)
    }
}

/// Load an F16 norm weight as F32 on GPU (raw, no +1.0 bias — HF convention).
fn paro_load_llama_norm_raw(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str, // e.g. "layers.0.input_layernorm.weight"
    shape: &[usize],
) -> HipResult<GpuTensor> {
    let full = format!("model.{name}");
    let (info, data) = source
        .tensor_data(&full)
        .ok_or_else(|| HipError::new(0, &format!("PARO tensor not found: {full}")))?;
    let v: Vec<f32> = if info.dtype == "F16" {
        data.chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    } else {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    gpu.upload_f32(&v, shape)
}

/// Load LLaMA/Qwen3 weights from a ParoQuant safetensors model.
///
/// Tensor naming convention: `model.layers.{i}.self_attn.q_proj.{qweight,...}`
/// (no `model.language_model.` prefix — that's Qwen3.5-specific).
pub fn load_weights_paroquant_llama(
    source: &dyn ModelSource,
    config: &LlamaConfig,
    gpu: &mut Gpu,
) -> HipResult<LlamaWeights> {
    let qc = source
        .quant_config()
        .ok_or_else(|| HipError::new(0, "ParoQuant model must have quantization_config"))?;
    let gs = qc.group_size;
    let kr = qc.krot;

    // Embedding
    eprintln!("  loading token_embd (ParoQuant LLaMA/Qwen3)...");
    let embd_name = "model.embed_tokens.weight";
    let (_, embd_data) = source
        .tensor_data(embd_name)
        .ok_or_else(|| HipError::new(0, "PARO tensor not found: embed_tokens not found"))?;
    let f32_embd: Vec<f32> = embd_data
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let token_embd = gpu.upload_f32(&f32_embd, &[config.vocab_size, config.dim])?;
    let embd_fmt = EmbeddingFormat::F32;

    // Output norm
    eprintln!("  loading output_norm...");
    let output_norm = paro_load_llama_norm_raw(source, gpu, "norm.weight", &[config.dim])?;

    // Output / lm_head (tied or separate)
    let output = if source.tensor_info("lm_head.weight").is_some() {
        eprintln!("  loading output (separate lm_head)...");
        let lm_prefix = "lm_head";
        if source
            .tensor_info(&format!("{lm_prefix}.qweight"))
            .is_some()
        {
            load_paroquant_weight_from_source(
                source,
                gpu,
                lm_prefix,
                config.vocab_size,
                config.dim,
                gs,
                kr,
            )?
        } else {
            load_fp16_weight_tensor_from_source(
                source,
                gpu,
                &format!("{lm_prefix}.weight"),
                config.vocab_size,
                config.dim,
            )?
        }
    } else {
        eprintln!("  loading output (tied embeddings)...");
        let (_, td) = source
            .tensor_data(embd_name)
            .ok_or_else(|| HipError::new(0, "PARO tensor not found: embed_tokens for lm_head"))?;
        let f: Vec<f32> = td
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(f.as_ptr() as *const u8, f.len() * 4) };
        let buf = gpu.upload_raw(bytes, &[config.vocab_size, config.dim])?;
        WeightTensor {
            buf,
            gpu_dtype: DType::F32,
            m: config.vocab_size,
            k: config.dim,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        }
    };

    // Layers
    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        eprintln!(
            "  loading layer {i}/{} (ParoQuant LLaMA/Qwen3)...",
            config.n_layers
        );
        crate::load_progress::report(i as u32 + 1, config.n_layers as u32, "weights");
        let p = format!("layers.{i}");
        let q_out_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;

        let q_norm = if config.has_qk_norm {
            Some(paro_load_llama_norm_raw(
                source,
                gpu,
                &format!("{p}.self_attn.q_norm.weight"),
                &[config.head_dim],
            )?)
        } else {
            None
        };
        let k_norm = if config.has_qk_norm {
            Some(paro_load_llama_norm_raw(
                source,
                gpu,
                &format!("{p}.self_attn.k_norm.weight"),
                &[config.head_dim],
            )?)
        } else {
            None
        };

        let layer = LayerWeights {
            attn_norm: paro_load_llama_norm_raw(
                source,
                gpu,
                &format!("{p}.input_layernorm.weight"),
                &[config.dim],
            )?,
            wq: paro_load_llama_wt(
                source,
                gpu,
                &format!("{p}.self_attn.q_proj"),
                q_out_dim,
                config.dim,
                gs,
                kr,
            )?,
            wk: paro_load_llama_wt(
                source,
                gpu,
                &format!("{p}.self_attn.k_proj"),
                kv_dim,
                config.dim,
                gs,
                kr,
            )?,
            wv: paro_load_llama_wt(
                source,
                gpu,
                &format!("{p}.self_attn.v_proj"),
                kv_dim,
                config.dim,
                gs,
                kr,
            )?,
            wo: paro_load_llama_wt(
                source,
                gpu,
                &format!("{p}.self_attn.o_proj"),
                config.dim,
                q_out_dim,
                gs,
                kr,
            )?,
            q_norm,
            k_norm,
            ffn_norm: paro_load_llama_norm_raw(
                source,
                gpu,
                &format!("{p}.post_attention_layernorm.weight"),
                &[config.dim],
            )?,
            w_gate: paro_load_llama_wt(
                source,
                gpu,
                &format!("{p}.mlp.gate_proj"),
                config.hidden_dim,
                config.dim,
                gs,
                kr,
            )?,
            w_up: paro_load_llama_wt(
                source,
                gpu,
                &format!("{p}.mlp.up_proj"),
                config.hidden_dim,
                config.dim,
                gs,
                kr,
            )?,
            w_down: paro_load_llama_wt(
                source,
                gpu,
                &format!("{p}.mlp.down_proj"),
                config.dim,
                config.hidden_dim,
                gs,
                kr,
            )?,
        };
        layers.push(layer);
    }

    Ok(LlamaWeights {
        token_embd,
        embd_format: embd_fmt,
        output_norm,
        output,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `impl ModelSource for HfqFile` returned `None` from `tensor_data` and
    /// `tensor_info` for years, with a comment saying the trait was "primarily
    /// for safetensors". Anything reaching an HFQ through the trait therefore
    /// saw an EMPTY model rather than an error — which is how streamed
    /// calibration ended up hard-wired to the concrete safetensors type.
    ///
    /// Guard the fix: through the trait, an HFQ must report its tensors, hand
    /// back their metadata, and yield bytes identical to the safetensors source
    /// it was built from.
    #[test]
    fn model_source_trait_sees_hfq_tensors() {
        use hipfire_model::ModelSource;

        let dir = temp_path("modelsource-trait");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama"}"#,
        )
        .unwrap();

        let payload: Vec<u8> = (0..512u32).map(|i| (i % 251) as u8).collect();
        let mut header = serde_json::Map::new();
        header.insert(
            "model.embed_tokens.weight".to_string(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": [16, 16],
                "data_offsets": [0, payload.len()],
            }),
        );
        let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&payload);
        std::fs::write(dir.join("model.safetensors"), &bytes).unwrap();

        let hfq = HfqFile::from_safetensors(&dir).unwrap();

        // The stub reported an empty model here.
        let names = ModelSource::tensor_names(&hfq);
        assert!(
            names.contains(&"model.embed_tokens.weight"),
            "trait must list HFQ tensors, got {names:?}"
        );

        let info = ModelSource::tensor_info(&hfq, "model.embed_tokens.weight")
            .expect("trait must expose tensor metadata for an HFQ");
        assert_eq!(info.shape, vec![16, 16]);
        assert_eq!(info.quant_type, 16, "BF16 source keeps its quant type");

        let (_, data) = ModelSource::tensor(&hfq, "model.embed_tokens.weight")
            .expect("trait must yield tensor bytes for an HFQ");
        assert_eq!(&*data, payload.as_slice());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "hipfire-hfqm-package-test-{}-{name}",
            std::process::id()
        ))
    }

    /// The sidecars a multimodal snapshot needs must survive the capture →
    /// metadata → restore path byte-for-byte, including a binary asset and a
    /// nested directory. This is the property that makes an `.hfq` → HF export
    /// reproduce a loadable checkpoint rather than a text-model skeleton.
    #[test]
    fn captures_and_restores_hf_sidecars_verbatim() {
        let dir = temp_path("sidecar-snapshot");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("assets")).unwrap();
        std::fs::create_dir_all(dir.join(".cache/huggingface")).unwrap();

        // Formatting that a parse/re-serialize round trip would destroy.
        let preproc = "{\n  \"image_size\":  384,\n  \"do_rescale\": true\n}\n";
        std::fs::write(dir.join("preprocessor_config.json"), preproc).unwrap();
        std::fs::write(dir.join("vocab.json"), r#"{"a":0,"b":1}"#).unwrap();
        std::fs::write(dir.join("chat_template.jinja"), "{{ bos_token }}").unwrap();
        // Invalid UTF-8 — must take the base64 arm.
        let png = [0x89u8, b'P', b'N', b'G', 0x0d, 0x0a, 0xff, 0xfe, 0x00];
        std::fs::write(dir.join("assets/logo.png"), png).unwrap();
        // Excluded: weight shard, shard index, dot-dir, and the tokenizer that
        // is already stored verbatim under its own key.
        std::fs::write(dir.join("model-00001-of-00002.safetensors"), [0u8; 8]).unwrap();
        std::fs::write(dir.join("model.safetensors.index.json"), "{}").unwrap();
        std::fs::write(dir.join("tokenizer.json"), r#"{"big":true}"#).unwrap();
        std::fs::write(dir.join(".cache/huggingface/junk"), "no").unwrap();

        let metadata = embed_tokenizer_metadata(r#"{"architecture":"test"}"#, &dir);
        let restored: std::collections::HashMap<String, Vec<u8>> =
            hf_sidecars_from_metadata(&metadata).into_iter().collect();

        assert_eq!(
            restored.get("preprocessor_config.json").map(|v| &v[..]),
            Some(preproc.as_bytes()),
            "byte-exact formatting must survive"
        );
        assert_eq!(
            restored.get("assets/logo.png").map(|v| &v[..]),
            Some(&png[..]),
            "binary assets must round-trip through base64"
        );
        assert!(restored.contains_key("vocab.json"));
        assert!(restored.contains_key("chat_template.jinja"));

        for excluded in [
            "model-00001-of-00002.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            ".cache/huggingface/junk",
        ] {
            assert!(
                !restored.contains_key(excluded),
                "{excluded} must not be captured"
            );
        }

        // The pre-existing tokenizer embedding is untouched by the sweep.
        let meta: serde_json::Value = serde_json::from_str(&metadata).unwrap();
        assert_eq!(
            meta.get("tokenizer").and_then(|t| t.as_str()),
            Some(r#"{"big":true}"#)
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn writes_and_reads_sidecar_hfqm_package() {
        let payload_a = temp_path("tokens.bin");
        let payload_b = temp_path("values.bin");
        let package_path = temp_path("ref.kldref.hfq");
        std::fs::write(&payload_a, [1u8, 2, 3, 4]).unwrap();
        std::fs::write(&payload_b, [5u8, 6, 7, 8, 9, 10, 11, 12]).unwrap();

        let metadata = serde_json::json!({
            "artifact_kind": "hipfire.kldref",
            "package_schema": "hipfire.kldref.v1",
            "arch_id": 5,
            "n_ctx": 2,
            "n_chunk": 1,
            "top_k": 1
        })
        .to_string();
        let entries = vec![
            HfqPackageWriteEntry {
                name: "kldref.tokens".to_string(),
                quant_type: 0,
                shape: vec![1, 2],
                group_size: 0,
                source_path: payload_a.clone(),
                data_size: 4,
            },
            HfqPackageWriteEntry {
                name: "kldref.top_log_probs".to_string(),
                quant_type: 0,
                shape: vec![1, 1, 2],
                group_size: 0,
                source_path: payload_b.clone(),
                data_size: 8,
            },
        ];
        write_hfqm_package_from_files(&package_path, 5, &metadata, &entries).unwrap();

        let package = HfqPackage::open(&package_path).unwrap();
        assert_eq!(package.version, HFQM_VERSION);
        assert_eq!(package.arch_id, 5);
        assert!(package
            .metadata_json
            .contains("\"artifact_kind\":\"hipfire.kldref\""));
        assert_eq!(package.entries().len(), 2);
        assert_eq!(package.entry("kldref.tokens").unwrap().shape, vec![1, 2]);
        assert_eq!(package.blob_data("kldref.tokens").unwrap(), &[1, 2, 3, 4]);
        assert_eq!(
            package.blob_data("kldref.top_log_probs").unwrap(),
            &[5, 6, 7, 8, 9, 10, 11, 12]
        );

        let _ = std::fs::remove_file(payload_a);
        let _ = std::fs::remove_file(payload_b);
        let _ = std::fs::remove_file(package_path);
    }

    #[test]
    fn canonical_model_prefix_resolves_stripped_sentence_transformer_tensor() {
        let payload = temp_path("stripped-tensor.bin");
        let package_path = temp_path("stripped-tensor.hfq");
        std::fs::write(&payload, 1.0f32.to_le_bytes()).unwrap();
        let entries = vec![HfqPackageWriteEntry {
            name: "embed_tokens.weight".to_string(),
            quant_type: 2,
            shape: vec![1],
            group_size: 0,
            source_path: payload.clone(),
            data_size: 4,
        }];
        write_hfqm_package_from_files(&package_path, 1, "{}", &entries).unwrap();

        let hfq = HfqFile::open(&package_path).unwrap();
        assert!(hfq.find_tensor_info("model.embed_tokens.weight").is_some());

        let _ = std::fs::remove_file(payload);
        let _ = std::fs::remove_file(package_path);
    }

    /// Write one safetensors shard from `(name, dtype, shape, bytes)` tuples,
    /// laying the data blob out in tuple order.
    fn write_safetensors_shard(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut blob: Vec<u8> = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = blob.len();
            blob.extend_from_slice(bytes);
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, blob.len()],
                }),
            );
        }
        let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&blob);
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn from_safetensors_passes_dense_and_splits_stacked_experts() {
        let dir = temp_path("st-from-dir");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["Qwen3_5_MoeForCausalLM"],"model_type":"qwen3_5_moe","num_experts":2}"#,
        )
        .unwrap();
        // Tokenizer sidecar embed: chat_template must surface through the
        // self-describing metadata just like a real `.hfq`.
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template":"SMOKE-TEMPLATE"}"#,
        )
        .unwrap();

        // Dense embed: bf16 [4] = 8 bytes. Stacked experts gate_up_proj:
        // bf16 [E=2, R=2, C=2] = 16 bytes; expert 0 = first 8, expert 1 = last 8.
        let embed: Vec<u8> = (1u8..=8).collect();
        let experts: Vec<u8> = (10u8..26).collect();
        write_safetensors_shard(
            &dir.join("model.safetensors"),
            &[
                ("model.embed_tokens.weight", "BF16", vec![4], embed.clone()),
                (
                    "model.layers.0.mlp.experts.gate_up_proj",
                    "BF16",
                    vec![2, 2, 2],
                    experts.clone(),
                ),
            ],
        );

        let hfq = HfqFile::from_safetensors(&dir).unwrap();
        assert_eq!(hfq.arch_id, 6, "qwen3.5-moe + num_experts>0 → arch 6");

        // Dense: raw HF name, bf16 tag (16), shape/bytes verbatim.
        let embed_info = hfq
            .find_tensor_info("model.embed_tokens.weight")
            .expect("dense embed present");
        assert_eq!(embed_info.quant_type, 16);
        assert_eq!(embed_info.shape, vec![4u32]);
        assert_eq!(
            hfq.tensor_data_vec("model.embed_tokens.weight").unwrap().1,
            embed
        );

        // Stacked parent is NOT exposed; per-expert 2D tensors are, with the
        // `.weight` suffix, fused gate_up kept, and correct byte sub-ranges.
        assert!(hfq
            .find_tensor_info("model.layers.0.mlp.experts.gate_up_proj")
            .is_none());
        let e0 = hfq
            .find_tensor_info("model.layers.0.mlp.experts.0.gate_up_proj.weight")
            .expect("expert 0 present");
        assert_eq!(e0.quant_type, 16);
        assert_eq!(e0.shape, vec![2u32, 2]);
        assert_eq!(
            hfq.tensor_data_vec("model.layers.0.mlp.experts.0.gate_up_proj.weight")
                .unwrap()
                .1,
            experts[0..8]
        );
        assert_eq!(
            hfq.tensor_data_vec("model.layers.0.mlp.experts.1.gate_up_proj.weight")
                .unwrap()
                .1,
            experts[8..16]
        );

        // Exactly three tensors: 1 dense + 2 experts (gate_up not split further).
        assert_eq!(hfq.tensors().len(), 3);

        // Sidecar tokenizer_config folded into metadata → chat_template surfaces.
        assert_eq!(hfq.chat_template().as_deref(), Some("SMOKE-TEMPLATE"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_safetensors_rejects_prequantized_dtype() {
        let dir = temp_path("st-from-dir-bad-dtype");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama"}"#,
        )
        .unwrap();
        write_safetensors_shard(
            &dir.join("model.safetensors"),
            &[("w", "F8_E4M3", vec![4], vec![0u8; 4])],
        );
        assert!(HfqFile::from_safetensors(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
