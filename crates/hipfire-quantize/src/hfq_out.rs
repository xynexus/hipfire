// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! HFQ output serialization + provenance.
//!
//! The `.hfq` writer (`write_hfq`), its streaming tensor spill, the
//! parameter-count / quantization-hash / git-provenance metadata builders, and
//! the small XXH64 helpers that back the quantization hash. Extracted from the
//! `hipfire-quantize` binary's `main.rs` so the GGUF import pipeline (now owned
//! by `hipfire-coexistence`) can produce byte-identical `.hfq` artifacts
//! through the same code path the native quantizer uses. See AGENTS.md: import
//! tooling lives outside the inference-adjacent quantize binary.

use hipfire_arch_api::{transformer_role, TensorRole};
use hipfire_quant_format::QuantType;
use std::fs::{File, OpenOptions};
use std::hash::Hasher;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use twox_hash::XxHash64;

/// HFQ container magic + format version.
pub const HFQ_MAGIC: &[u8; 4] = b"HFQM";
pub const HFQ_VERSION: u32 = 2;
const INVALID_HFQ_MAGIC: &[u8; 4] = b"HFQ!";
const HEADER_SIZE: u64 = 32;
const PAYLOAD_ALIGN: u64 = 4096;
const MODULE_ALIGN: u64 = 4096;
const TENSOR_ALIGN: u64 = 32;

pub struct HfqTensor {
    pub name: String,
    pub quant_type: QuantType,
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub data: Vec<u8>,
    /// When data is spilled to disk, this holds the byte count.
    /// `data` is empty and the bytes live in the spill file.
    pub spilled_len: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct HfqWriteProgress {
    pub written_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct HfqStreamTensor {
    pub name: String,
    pub quant_type: QuantType,
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub data_len: u64,
}

pub fn tensor_param_count(t: &HfqTensor) -> u64 {
    t.shape
        .iter()
        .fold(1u64, |acc, &dim| acc.saturating_mul(dim as u64))
}

pub fn config_u64_any(config: &serde_json::Value, keys: &[&str]) -> Option<u64> {
    fn get_from_scope(scope: &serde_json::Value, keys: &[&str]) -> Option<u64> {
        keys.iter().find_map(|key| scope.get(*key)?.as_u64())
    }

    get_from_scope(config, keys)
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|scope| get_from_scope(scope, keys))
        })
        .or_else(|| {
            config
                .get("moe")
                .and_then(|scope| get_from_scope(scope, keys))
        })
        .or_else(|| {
            config
                .get("ffn_config")
                .and_then(|scope| get_from_scope(scope, keys))
        })
}

pub fn model_config_from_metadata(metadata: &serde_json::Value) -> &serde_json::Value {
    metadata.get("config").unwrap_or(metadata)
}

pub fn routed_moe_config(metadata: &serde_json::Value) -> Option<(u64, u64)> {
    let config = model_config_from_metadata(metadata);
    let num_experts = config_u64_any(
        config,
        &[
            "num_experts",
            "n_routed_experts",
            "num_local_experts",
            "n_experts",
        ],
    )?;
    let top_k = config_u64_any(
        config,
        &[
            "num_experts_per_tok",
            "num_experts_per_token",
            "n_experts_per_tok",
            "moe_top_k",
            "top_k",
            "num_selected_experts",
        ],
    )?;
    if num_experts == 0 || top_k == 0 {
        None
    } else {
        Some((num_experts, top_k))
    }
}

pub fn parameter_counts_metadata(
    metadata: &serde_json::Value,
    tensors: &[HfqTensor],
    total_params: u64,
    quantized_params: u64,
    skipped_params: u64,
) -> serde_json::Value {
    let mut routed_expert_params = 0u64;
    for t in tensors {
        if transformer_role(&t.name) == TensorRole::Expert {
            routed_expert_params = routed_expert_params.saturating_add(tensor_param_count(t));
        }
    }

    let (active_params, effective_params, moe) = if routed_expert_params > 0 {
        if let Some((num_experts, top_k)) = routed_moe_config(metadata) {
            let numerator = routed_expert_params.saturating_mul(top_k);
            let routed_active = numerator / num_experts;
            let active = total_params
                .saturating_sub(routed_expert_params)
                .saturating_add(routed_active);
            (
                active,
                active,
                Some(serde_json::json!({
                    "num_experts": num_experts,
                    "num_experts_per_tok": top_k,
                    "routed_expert_params": routed_expert_params,
                    "routed_expert_active_params": routed_active,
                    "active_rule": "dense_and_shared_full_plus_routed_top_k_over_num_experts",
                    "routed_active_fraction": {
                        "numerator": numerator,
                        "denominator": num_experts,
                    },
                })),
            )
        } else {
            (
                total_params,
                total_params,
                Some(serde_json::json!({
                    "routed_expert_params": routed_expert_params,
                    "active_rule": "unknown_top_k_or_num_experts",
                })),
            )
        }
    } else {
        (total_params, total_params, None)
    };

    let source_total_params = total_params.saturating_add(skipped_params);
    let mut counts = serde_json::json!({
        "schema": "hipfire.parameter_counts.v1",
        "total_params": total_params,
        "source_total_params": source_total_params,
        "active_params": active_params,
        "effective_params": effective_params,
        "quantized_params": quantized_params,
        "skipped_params": skipped_params,
    });
    if let Some(moe) = moe {
        if let serde_json::Value::Object(ref mut map) = counts {
            map.insert("moe".to_string(), moe);
        }
    }
    counts
}

pub fn insert_parameter_counts_metadata(
    metadata: &mut serde_json::Value,
    tensors: &[HfqTensor],
    total_params: u64,
    quantized_params: u64,
    skipped_params: u64,
) {
    let counts = parameter_counts_metadata(
        metadata,
        tensors,
        total_params,
        quantized_params,
        skipped_params,
    );
    if let serde_json::Value::Object(ref mut map) = metadata {
        map.insert("parameter_counts".to_string(), counts);
    }
}

pub struct Xxh64 {
    inner: XxHash64,
}

impl Xxh64 {
    pub fn new(seed: u64) -> Self {
        Self {
            inner: XxHash64::with_seed(seed),
        }
    }

    pub fn update(&mut self, input: &[u8]) {
        self.inner.write(input);
    }

    pub fn digest(&self) -> u64 {
        self.inner.finish()
    }
}

pub fn xxh64_update_u8(h: &mut Xxh64, v: u8) {
    h.update(&[v]);
}

pub fn xxh64_update_u32(h: &mut Xxh64, v: u32) {
    h.update(&v.to_le_bytes());
}

pub fn xxh64_update_u64(h: &mut Xxh64, v: u64) {
    h.update(&v.to_le_bytes());
}

pub fn xxh64_hex_bytes(bytes: &[u8]) -> String {
    let mut h = Xxh64::new(0);
    h.update(bytes);
    format!("{:016x}", h.digest())
}

fn align_up(v: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (v + align - 1) & !(align - 1)
}

fn tensor_data_len(t: &HfqTensor) -> u64 {
    if t.spilled_len > 0 {
        t.spilled_len
    } else {
        t.data.len() as u64
    }
}

fn stream_tensor_data_len(t: &HfqStreamTensor) -> u64 {
    t.data_len
}

fn expert_key(name: &str) -> Option<(u16, u16)> {
    let parts: Vec<&str> = name.split('.').collect();
    let layer_pos = parts.iter().position(|p| *p == "layers")?;
    let layer = parts.get(layer_pos + 1)?.parse::<u16>().ok()?;
    let expert_pos = parts.iter().position(|p| *p == "experts")?;
    let expert = parts.get(expert_pos + 1)?.parse::<u16>().ok()?;
    Some((layer, expert))
}

fn split_front_tail_metadata(metadata_json: &str) -> std::io::Result<(serde_json::Value, Vec<u8>)> {
    let full: serde_json::Value = serde_json::from_str(metadata_json)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut front = match full.clone() {
        serde_json::Value::Object(map) => serde_json::Value::Object(map),
        _ => serde_json::json!({ "metadata": full.clone() }),
    };
    if let Some(map) = front.as_object_mut() {
        for key in [
            "config",
            "tokenizer",
            "tokenizer_config",
            "generation_config",
            "gguf_meta",
        ] {
            map.remove(key);
        }
        map.insert(
            "hfq_format".to_string(),
            serde_json::json!("hipfire.hfq.v2"),
        );
    }
    let tail = serde_json::json!({
        "format": "hipfire.hfq.tail.v1",
        "metadata": full,
    });
    let tail_bytes = serde_json::to_vec(&tail)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok((front, tail_bytes))
}

#[derive(Debug, Clone)]
struct PlannedTensor {
    offset: u64,
    data_len: u64,
}

fn build_v2_index(tensors: &[HfqTensor], planned: &[PlannedTensor]) -> std::io::Result<Vec<u8>> {
    let mut index_bytes = Vec::new();
    index_bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for (t, p) in tensors.iter().zip(planned) {
        if p.offset % TENSOR_ALIGN != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "tensor {} offset {} is not 32-byte aligned",
                    t.name, p.offset
                ),
            ));
        }
        let name_bytes = t.name.as_bytes();
        if name_bytes.len() > u16::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("HFQ tensor name too long: {}", t.name),
            ));
        }
        index_bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        index_bytes.extend_from_slice(name_bytes);
        index_bytes.push(t.quant_type as u8);
        index_bytes.push(t.shape.len() as u8);
        for &d in &t.shape {
            index_bytes.extend_from_slice(&d.to_le_bytes());
        }
        index_bytes.extend_from_slice(&t.group_size.to_le_bytes());
        index_bytes.extend_from_slice(&p.data_len.to_le_bytes());
        index_bytes.extend_from_slice(&(p.offset / TENSOR_ALIGN).to_le_bytes());
    }
    Ok(index_bytes)
}

fn build_v2_stream_index(
    tensors: &[HfqStreamTensor],
    planned: &[PlannedTensor],
) -> std::io::Result<Vec<u8>> {
    let mut index_bytes = Vec::new();
    index_bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for (t, p) in tensors.iter().zip(planned) {
        if p.offset % TENSOR_ALIGN != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "tensor {} offset {} is not 32-byte aligned",
                    t.name, p.offset
                ),
            ));
        }
        let name_bytes = t.name.as_bytes();
        if name_bytes.len() > u16::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("HFQ tensor name too long: {}", t.name),
            ));
        }
        index_bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        index_bytes.extend_from_slice(name_bytes);
        index_bytes.push(t.quant_type as u8);
        index_bytes.push(t.shape.len() as u8);
        for &d in &t.shape {
            index_bytes.extend_from_slice(&d.to_le_bytes());
        }
        index_bytes.extend_from_slice(&t.group_size.to_le_bytes());
        index_bytes.extend_from_slice(&p.data_len.to_le_bytes());
        index_bytes.extend_from_slice(&(p.offset / TENSOR_ALIGN).to_le_bytes());
    }
    Ok(index_bytes)
}

fn plan_offsets(tensors: &[HfqTensor], payload_start: u64) -> Vec<PlannedTensor> {
    let mut out = Vec::with_capacity(tensors.len());
    let mut cursor = align_up(payload_start, PAYLOAD_ALIGN);
    let mut active_expert: Option<(u16, u16)> = None;
    for t in tensors {
        let key = expert_key(&t.name);
        if key.is_some() && key != active_expert {
            cursor = align_up(cursor, MODULE_ALIGN);
        } else {
            cursor = align_up(cursor, TENSOR_ALIGN);
        }
        let data_len = tensor_data_len(t);
        out.push(PlannedTensor {
            offset: cursor,
            data_len,
        });
        cursor = cursor.saturating_add(data_len);
        active_expert = key;
    }
    out
}

fn plan_stream_offsets(tensors: &[HfqStreamTensor], payload_start: u64) -> Vec<PlannedTensor> {
    let mut out = Vec::with_capacity(tensors.len());
    let mut cursor = align_up(payload_start, PAYLOAD_ALIGN);
    let mut active_expert: Option<(u16, u16)> = None;
    for t in tensors {
        let key = expert_key(&t.name);
        if key.is_some() && key != active_expert {
            cursor = align_up(cursor, MODULE_ALIGN);
        } else {
            cursor = align_up(cursor, TENSOR_ALIGN);
        }
        let data_len = stream_tensor_data_len(t);
        out.push(PlannedTensor {
            offset: cursor,
            data_len,
        });
        cursor = cursor.saturating_add(data_len);
        active_expert = key;
    }
    out
}

fn build_front_metadata(
    mut front: serde_json::Value,
    tail_offset: u64,
    tail_bytes: &[u8],
    quant_hash: serde_json::Value,
) -> std::io::Result<String> {
    if let serde_json::Value::Object(ref mut map) = front {
        map.insert("quantization_hash".to_string(), quant_hash);
        map.insert(
            "tail_metadata".to_string(),
            serde_json::json!({
                "format": "hipfire.hfq.tail.v1",
                "offset": tail_offset,
                "size": tail_bytes.len() as u64,
                "hash": {
                    "algorithm": "xxh64",
                    "seed": 0,
                    "scope": "hfq_tail_metadata_v1",
                    "value": xxh64_hex_bytes(tail_bytes),
                }
            }),
        );
    }
    serde_json::to_string(&front)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

struct PayloadHashState {
    h: Xxh64,
    tensor_count: usize,
    payload_bytes: u64,
}

impl PayloadHashState {
    fn new() -> Self {
        let mut h = Xxh64::new(0);
        h.update(b"hipfire-hfq-v2-critical-index-and-payload");
        Self {
            h,
            tensor_count: 0,
            payload_bytes: 0,
        }
    }

    fn start_tensor(&mut self, t: &HfqTensor, p: &PlannedTensor) {
        let name_bytes = t.name.as_bytes();
        xxh64_update_u64(&mut self.h, name_bytes.len() as u64);
        self.h.update(name_bytes);
        xxh64_update_u8(&mut self.h, t.quant_type as u8);
        xxh64_update_u64(&mut self.h, t.shape.len() as u64);
        for &dim in &t.shape {
            xxh64_update_u32(&mut self.h, dim);
        }
        xxh64_update_u32(&mut self.h, t.group_size);
        xxh64_update_u64(&mut self.h, p.offset / TENSOR_ALIGN);
        xxh64_update_u64(&mut self.h, p.data_len);
        self.tensor_count += 1;
        self.payload_bytes += p.data_len;
    }

    fn start_stream_tensor(&mut self, t: &HfqStreamTensor, p: &PlannedTensor) {
        let name_bytes = t.name.as_bytes();
        xxh64_update_u64(&mut self.h, name_bytes.len() as u64);
        self.h.update(name_bytes);
        xxh64_update_u8(&mut self.h, t.quant_type as u8);
        xxh64_update_u64(&mut self.h, t.shape.len() as u64);
        for &dim in &t.shape {
            xxh64_update_u32(&mut self.h, dim);
        }
        xxh64_update_u32(&mut self.h, t.group_size);
        xxh64_update_u64(&mut self.h, p.offset / TENSOR_ALIGN);
        xxh64_update_u64(&mut self.h, p.data_len);
        self.tensor_count += 1;
        self.payload_bytes += p.data_len;
    }

    fn update_payload(&mut self, bytes: &[u8]) {
        self.h.update(bytes);
    }

    fn metadata(&self) -> serde_json::Value {
        serde_json::json!({
            "algorithm": "xxh64",
            "seed": 0,
            "scope": "hfq_v2_index_offsets_and_payload",
            "value": format!("{:016x}", self.h.digest()),
            "tensor_count": self.tensor_count,
            "payload_bytes": self.payload_bytes,
            "producer": {
                "package": "hipfire-quantize",
                "hipfire_version": env!("CARGO_PKG_VERSION"),
                "git_commit": git_commit(),
                "git_branch": git_branch(),
                "git_describe": git_describe(),
                "git_dirty": git_dirty(),
            },
        })
    }
}

fn placeholder_quantization_hash(tensor_count: usize, payload_bytes: u64) -> serde_json::Value {
    serde_json::json!({
        "algorithm": "xxh64",
        "seed": 0,
        "scope": "hfq_v2_index_offsets_and_payload",
        "value": "0000000000000000",
        "tensor_count": tensor_count,
        "payload_bytes": payload_bytes,
        "producer": {
            "package": "hipfire-quantize",
            "hipfire_version": env!("CARGO_PKG_VERSION"),
            "git_commit": git_commit(),
            "git_branch": git_branch(),
            "git_describe": git_describe(),
            "git_dirty": git_dirty(),
        },
    })
}

pub fn hfq_quantization_hash_metadata(
    tensors: &[HfqTensor],
    spill: Option<&TensorSpill>,
) -> std::io::Result<serde_json::Value> {
    let mut h = Xxh64::new(0);
    let mut payload_bytes = 0u64;
    h.update(b"hipfire-hfq-quantized-tensor-payload-v1");

    let mut spill_reader = if let Some(spill) = spill {
        Some(std::io::BufReader::new(File::open(&spill.path)?))
    } else {
        None
    };
    let mut buf = vec![0u8; 4 * 1024 * 1024];

    for t in tensors {
        let name_bytes = t.name.as_bytes();
        xxh64_update_u64(&mut h, name_bytes.len() as u64);
        h.update(name_bytes);
        xxh64_update_u8(&mut h, t.quant_type as u8);
        xxh64_update_u64(&mut h, t.shape.len() as u64);
        for &dim in &t.shape {
            xxh64_update_u32(&mut h, dim);
        }
        xxh64_update_u32(&mut h, t.group_size);
        let data_len = if t.spilled_len > 0 {
            t.spilled_len
        } else {
            t.data.len() as u64
        };
        xxh64_update_u64(&mut h, data_len);
        payload_bytes += data_len;

        if t.spilled_len > 0 {
            let reader = spill_reader
                .as_mut()
                .expect("spilled tensor requires spill reader");
            let mut remaining = t.spilled_len as usize;
            while remaining > 0 {
                let chunk = remaining.min(buf.len());
                use std::io::Read;
                reader.read_exact(&mut buf[..chunk])?;
                h.update(&buf[..chunk]);
                remaining -= chunk;
            }
        } else {
            h.update(&t.data);
        }
    }

    Ok(serde_json::json!({
        "algorithm": "xxh64",
        "seed": 0,
        "scope": "hfq_tensor_index_and_payload_v1",
        "value": format!("{:016x}", h.digest()),
        "tensor_count": tensors.len(),
        "payload_bytes": payload_bytes,
        "producer": {
            "package": "hipfire-quantize",
            "hipfire_version": env!("CARGO_PKG_VERSION"),
            "git_commit": git_commit(),
            "git_branch": git_branch(),
            "git_describe": git_describe(),
            "git_dirty": git_dirty(),
        },
    }))
}

pub fn metadata_with_quantization_hash(
    mut metadata: serde_json::Value,
    tensors: &[HfqTensor],
    spill: Option<&TensorSpill>,
) -> std::io::Result<String> {
    let hash = hfq_quantization_hash_metadata(tensors, spill)?;
    if let serde_json::Value::Object(ref mut map) = metadata {
        map.insert("quantization_hash".to_string(), hash);
    }
    serde_json::to_string(&metadata)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

pub fn command_stdout(cmd: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

pub fn git_commit() -> Option<String> {
    command_stdout("git", &["rev-parse", "HEAD"])
}

pub fn git_branch() -> Option<String> {
    command_stdout("git", &["rev-parse", "--abbrev-ref", "HEAD"])
}

pub fn git_describe() -> Option<String> {
    command_stdout("git", &["describe", "--always", "--dirty", "--tags"])
}

pub fn git_dirty() -> Option<bool> {
    let out = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(!out.stdout.is_empty())
}

/// Streaming tensor spill file. When the quantizer accumulates more than
/// `SPILL_THRESHOLD` bytes of tensor data in memory, it flushes completed
/// tensors to this file. At write_hfq time, spilled data is copied from
/// the spill file instead of from memory, keeping peak RSS bounded.
pub struct TensorSpill {
    file: std::io::BufWriter<File>,
    path: PathBuf,
    offset: u64,
}

impl TensorSpill {
    pub fn new(dir: &Path) -> std::io::Result<Self> {
        // PID-unique so concurrent quantize runs in the same output dir don't
        // share a spill path (a sibling run's Drop would otherwise delete this
        // run's spill file → write_hfq NotFound panic).
        let path = dir.join(format!(".hipfire_quant_spill.{}.tmp", std::process::id()));
        let file = std::io::BufWriter::with_capacity(4 * 1024 * 1024, File::create(&path)?);
        Ok(Self {
            file,
            path,
            offset: 0,
        })
    }

    /// Write tensor data to the spill file. Returns the byte count written.
    pub fn spill(&mut self, data: &[u8]) -> std::io::Result<u64> {
        use std::io::Write;
        self.file.write_all(data)?;
        self.offset += data.len() as u64;
        Ok(data.len() as u64)
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        use std::io::Write;
        self.file.flush()
    }

    pub fn cleanup(self) {
        // Explicit cleanup — Drop impl handles the actual removal.
        drop(self);
    }
}

impl Drop for TensorSpill {
    fn drop(&mut self) {
        // Ensure the temp file is removed even on panic.
        let _ = std::fs::remove_file(&self.path);
    }
}

pub fn write_hfq(
    path: &Path,
    arch: u32,
    metadata_json: &str,
    tensors: &[HfqTensor],
    mut spill: Option<&mut TensorSpill>,
) -> std::io::Result<()> {
    write_hfq_with_progress(
        path,
        arch,
        metadata_json,
        tensors,
        spill.as_deref_mut(),
        |_| {},
    )
}

pub fn write_hfq_with_progress(
    path: &Path,
    arch: u32,
    metadata_json: &str,
    tensors: &[HfqTensor],
    mut spill: Option<&mut TensorSpill>,
    mut progress: impl FnMut(HfqWriteProgress),
) -> std::io::Result<()> {
    if let Some(spill) = spill.as_mut() {
        spill.flush()?;
    }
    let (front_base, tail_bytes) = split_front_tail_metadata(metadata_json)?;
    let metadata_offset = HEADER_SIZE;

    let mut front_json = String::new();
    let mut index_bytes = Vec::new();
    #[allow(unused_assignments)]
    let mut planned = Vec::new();
    #[allow(unused_assignments)]
    let mut data_offset = 0u64;
    #[allow(unused_assignments)]
    let mut tail_offset = 0u64;
    for _ in 0..16 {
        data_offset = align_up(
            metadata_offset + front_json.len() as u64 + index_bytes.len() as u64,
            PAYLOAD_ALIGN,
        );
        planned = plan_offsets(tensors, data_offset);
        index_bytes = build_v2_index(tensors, &planned)?;
        let payload_end = planned
            .last()
            .map(|p| p.offset.saturating_add(p.data_len))
            .unwrap_or(data_offset);
        tail_offset = align_up(payload_end, TENSOR_ALIGN);
        let quant_hash =
            placeholder_quantization_hash(tensors.len(), planned.iter().map(|p| p.data_len).sum());
        let next_front =
            build_front_metadata(front_base.clone(), tail_offset, &tail_bytes, quant_hash)?;
        if next_front == front_json {
            break;
        }
        let old_len = front_json.len();
        front_json = next_front;
        if front_json.len() == old_len {
            data_offset = align_up(
                metadata_offset + front_json.len() as u64 + index_bytes.len() as u64,
                PAYLOAD_ALIGN,
            );
            planned = plan_offsets(tensors, data_offset);
            index_bytes = build_v2_index(tensors, &planned)?;
            tail_offset = align_up(
                planned
                    .last()
                    .map(|p| p.offset.saturating_add(p.data_len))
                    .unwrap_or(data_offset),
                TENSOR_ALIGN,
            );
            let quant_hash = placeholder_quantization_hash(
                tensors.len(),
                planned.iter().map(|p| p.data_len).sum(),
            );
            let stable_front =
                build_front_metadata(front_base.clone(), tail_offset, &tail_bytes, quant_hash)?;
            if stable_front.len() == front_json.len() {
                front_json = stable_front;
                break;
            }
            front_json = stable_front;
        }
    }
    data_offset = align_up(
        metadata_offset + front_json.len() as u64 + index_bytes.len() as u64,
        PAYLOAD_ALIGN,
    );
    planned = plan_offsets(tensors, data_offset);
    index_bytes = build_v2_index(tensors, &planned)?;
    tail_offset = align_up(
        planned
            .last()
            .map(|p| p.offset.saturating_add(p.data_len))
            .unwrap_or(data_offset),
        TENSOR_ALIGN,
    );
    let quant_hash =
        placeholder_quantization_hash(tensors.len(), planned.iter().map(|p| p.data_len).sum());
    front_json = build_front_metadata(front_base.clone(), tail_offset, &tail_bytes, quant_hash)?;
    let final_data_offset = align_up(
        metadata_offset + front_json.len() as u64 + index_bytes.len() as u64,
        PAYLOAD_ALIGN,
    );
    if final_data_offset != data_offset {
        data_offset = final_data_offset;
        planned = plan_offsets(tensors, data_offset);
        index_bytes = build_v2_index(tensors, &planned)?;
        tail_offset = align_up(
            planned
                .last()
                .map(|p| p.offset.saturating_add(p.data_len))
                .unwrap_or(data_offset),
            TENSOR_ALIGN,
        );
        let quant_hash =
            placeholder_quantization_hash(tensors.len(), planned.iter().map(|p| p.data_len).sum());
        front_json =
            build_front_metadata(front_base.clone(), tail_offset, &tail_bytes, quant_hash)?;
        let check_data_offset = align_up(
            metadata_offset + front_json.len() as u64 + index_bytes.len() as u64,
            PAYLOAD_ALIGN,
        );
        if check_data_offset != data_offset {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "HFQ v2 front metadata did not converge",
            ));
        }
    }

    let total_output_bytes = tail_offset.saturating_add(tail_bytes.len() as u64);
    let mut max_written_position = 0u64;
    let mut report_progress = |position: u64| {
        if position > max_written_position {
            max_written_position = position.min(total_output_bytes);
            progress(HfqWriteProgress {
                written_bytes: max_written_position,
                total_bytes: total_output_bytes,
            });
        }
    };

    let mut f = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(path)?;

    // Invalid magic until every section is durable enough to be parsed.
    f.write_all(INVALID_HFQ_MAGIC)?;
    f.write_all(&HFQ_VERSION.to_le_bytes())?;
    f.write_all(&arch.to_le_bytes())?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;
    f.write_all(front_json.as_bytes())?;
    f.write_all(&index_bytes)?;
    let pos = f.stream_position()?;
    report_progress(pos);
    if pos > data_offset {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("HFQ v2 front metadata/index exceeded data offset ({pos} > {data_offset})"),
        ));
    }
    write_zeroes(&mut f, data_offset - pos)?;
    report_progress(data_offset);

    let mut spill_reader = if let Some(spill) = spill {
        spill.flush()?;
        Some(std::io::BufReader::new(File::open(&spill.path)?))
    } else {
        None
    };
    let mut buf = vec![0u8; 4 * 1024 * 1024];
    let mut payload_hash = PayloadHashState::new();
    for (t, p) in tensors.iter().zip(&planned) {
        let pos = f.stream_position()?;
        if pos > p.offset {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "HFQ v2 tensor {} planned offset {} is behind writer position {pos}",
                    t.name, p.offset
                ),
            ));
        }
        write_zeroes(&mut f, p.offset - pos)?;
        payload_hash.start_tensor(t, p);
        if t.spilled_len > 0 {
            let reader = spill_reader
                .as_mut()
                .expect("spilled tensor requires spill reader");
            let mut remaining = t.spilled_len as usize;
            while remaining > 0 {
                let chunk = remaining.min(buf.len());
                reader.read_exact(&mut buf[..chunk])?;
                f.write_all(&buf[..chunk])?;
                payload_hash.update_payload(&buf[..chunk]);
                remaining -= chunk;
                report_progress(f.stream_position()?);
            }
        } else {
            f.write_all(&t.data)?;
            payload_hash.update_payload(&t.data);
            report_progress(f.stream_position()?);
        }
    }
    let pos = f.stream_position()?;
    if pos > tail_offset {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("HFQ v2 tail offset {tail_offset} is behind writer position {pos}"),
        ));
    }
    write_zeroes(&mut f, tail_offset - pos)?;
    report_progress(tail_offset);
    f.write_all(&tail_bytes)?;
    report_progress(total_output_bytes);
    f.flush()?;

    let final_front_json = build_front_metadata(
        front_base,
        tail_offset,
        &tail_bytes,
        payload_hash.metadata(),
    )?;
    if final_front_json.len() != front_json.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "HFQ v2 final front metadata length changed ({} != {})",
                final_front_json.len(),
                front_json.len()
            ),
        ));
    }
    f.seek(SeekFrom::Start(metadata_offset))?;
    f.write_all(final_front_json.as_bytes())?;
    f.write_all(&index_bytes)?;

    f.seek(SeekFrom::Start(0))?;
    f.write_all(HFQ_MAGIC)?;
    f.flush()?;
    Ok(())
}

pub fn write_hfq_streaming_with_progress(
    path: &Path,
    arch: u32,
    metadata_json: &str,
    tensors: &[HfqStreamTensor],
    mut write_tensor: impl FnMut(usize, &mut dyn Write) -> std::io::Result<()>,
    mut progress: impl FnMut(HfqWriteProgress),
) -> std::io::Result<()> {
    let (front_base, tail_bytes) = split_front_tail_metadata(metadata_json)?;
    let metadata_offset = HEADER_SIZE;

    let mut front_json = String::new();
    let mut index_bytes = Vec::new();
    #[allow(unused_assignments)]
    let mut planned = Vec::new();
    #[allow(unused_assignments)]
    let mut data_offset = 0u64;
    #[allow(unused_assignments)]
    let mut tail_offset = 0u64;
    for _ in 0..16 {
        data_offset = align_up(
            metadata_offset + front_json.len() as u64 + index_bytes.len() as u64,
            PAYLOAD_ALIGN,
        );
        planned = plan_stream_offsets(tensors, data_offset);
        index_bytes = build_v2_stream_index(tensors, &planned)?;
        let payload_end = planned
            .last()
            .map(|p| p.offset.saturating_add(p.data_len))
            .unwrap_or(data_offset);
        tail_offset = align_up(payload_end, TENSOR_ALIGN);
        let quant_hash =
            placeholder_quantization_hash(tensors.len(), planned.iter().map(|p| p.data_len).sum());
        let next_front =
            build_front_metadata(front_base.clone(), tail_offset, &tail_bytes, quant_hash)?;
        if next_front == front_json {
            break;
        }
        let old_len = front_json.len();
        front_json = next_front;
        if front_json.len() == old_len {
            break;
        }
    }
    data_offset = align_up(
        metadata_offset + front_json.len() as u64 + index_bytes.len() as u64,
        PAYLOAD_ALIGN,
    );
    planned = plan_stream_offsets(tensors, data_offset);
    index_bytes = build_v2_stream_index(tensors, &planned)?;
    tail_offset = align_up(
        planned
            .last()
            .map(|p| p.offset.saturating_add(p.data_len))
            .unwrap_or(data_offset),
        TENSOR_ALIGN,
    );
    let quant_hash =
        placeholder_quantization_hash(tensors.len(), planned.iter().map(|p| p.data_len).sum());
    front_json = build_front_metadata(front_base.clone(), tail_offset, &tail_bytes, quant_hash)?;
    let final_data_offset = align_up(
        metadata_offset + front_json.len() as u64 + index_bytes.len() as u64,
        PAYLOAD_ALIGN,
    );
    if final_data_offset != data_offset {
        data_offset = final_data_offset;
        planned = plan_stream_offsets(tensors, data_offset);
        index_bytes = build_v2_stream_index(tensors, &planned)?;
        tail_offset = align_up(
            planned
                .last()
                .map(|p| p.offset.saturating_add(p.data_len))
                .unwrap_or(data_offset),
            TENSOR_ALIGN,
        );
        let quant_hash =
            placeholder_quantization_hash(tensors.len(), planned.iter().map(|p| p.data_len).sum());
        front_json =
            build_front_metadata(front_base.clone(), tail_offset, &tail_bytes, quant_hash)?;
        let check_data_offset = align_up(
            metadata_offset + front_json.len() as u64 + index_bytes.len() as u64,
            PAYLOAD_ALIGN,
        );
        if check_data_offset != data_offset {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "HFQ v2 streaming front metadata did not converge",
            ));
        }
    }

    let total_output_bytes = tail_offset.saturating_add(tail_bytes.len() as u64);
    let mut max_written_position = 0u64;
    let mut report_progress = |position: u64| {
        if position > max_written_position {
            max_written_position = position.min(total_output_bytes);
            progress(HfqWriteProgress {
                written_bytes: max_written_position,
                total_bytes: total_output_bytes,
            });
        }
    };

    let mut f = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(path)?;

    f.write_all(INVALID_HFQ_MAGIC)?;
    f.write_all(&HFQ_VERSION.to_le_bytes())?;
    f.write_all(&arch.to_le_bytes())?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;
    f.write_all(front_json.as_bytes())?;
    f.write_all(&index_bytes)?;
    let pos = f.stream_position()?;
    report_progress(pos);
    if pos > data_offset {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("HFQ v2 front metadata/index exceeded data offset ({pos} > {data_offset})"),
        ));
    }
    write_zeroes(&mut f, data_offset - pos)?;
    report_progress(data_offset);

    let mut payload_hash = PayloadHashState::new();
    for (i, (t, p)) in tensors.iter().zip(&planned).enumerate() {
        let pos = f.stream_position()?;
        if pos > p.offset {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "HFQ v2 tensor {} planned offset {} is behind writer position {pos}",
                    t.name, p.offset
                ),
            ));
        }
        write_zeroes(&mut f, p.offset - pos)?;
        payload_hash.start_stream_tensor(t, p);
        let mut writer = HashingCountingWriter {
            inner: &mut f,
            hash: &mut payload_hash,
            written: 0,
        };
        write_tensor(i, &mut writer)?;
        if writer.written != p.data_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "HFQ v2 tensor {} wrote {} bytes, expected {}",
                    t.name, writer.written, p.data_len
                ),
            ));
        }
        report_progress(f.stream_position()?);
    }
    let pos = f.stream_position()?;
    if pos > tail_offset {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("HFQ v2 tail offset {tail_offset} is behind writer position {pos}"),
        ));
    }
    write_zeroes(&mut f, tail_offset - pos)?;
    report_progress(tail_offset);
    f.write_all(&tail_bytes)?;
    report_progress(total_output_bytes);
    f.flush()?;

    let final_front_json = build_front_metadata(
        front_base,
        tail_offset,
        &tail_bytes,
        payload_hash.metadata(),
    )?;
    if final_front_json.len() != front_json.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "HFQ v2 final front metadata length changed ({} != {})",
                final_front_json.len(),
                front_json.len()
            ),
        ));
    }
    f.seek(SeekFrom::Start(metadata_offset))?;
    f.write_all(final_front_json.as_bytes())?;
    f.write_all(&index_bytes)?;

    f.seek(SeekFrom::Start(0))?;
    f.write_all(HFQ_MAGIC)?;
    f.flush()?;
    Ok(())
}

pub struct LiveHfqWriter {
    f: File,
    front_json: String,
    index_bytes: Vec<u8>,
    planned: Vec<PlannedTensor>,
    tail_offset: u64,
    payload_hash: PayloadHashState,
    next_tensor: usize,
    total_output_bytes: u64,
    max_written_position: u64,
}

impl LiveHfqWriter {
    pub fn begin(
        path: &Path,
        arch: u32,
        placeholder_metadata_json: &str,
        tensors: &[HfqStreamTensor],
    ) -> std::io::Result<Self> {
        let (front_base, tail_bytes) = split_front_tail_metadata(placeholder_metadata_json)?;
        let metadata_offset = HEADER_SIZE;

        let mut front_json = String::new();
        let mut index_bytes = Vec::new();
        #[allow(unused_assignments)]
        let mut planned = Vec::new();
        #[allow(unused_assignments)]
        let mut data_offset = 0u64;
        #[allow(unused_assignments)]
        let mut tail_offset = 0u64;
        for _ in 0..16 {
            data_offset = align_up(
                metadata_offset + front_json.len() as u64 + index_bytes.len() as u64,
                PAYLOAD_ALIGN,
            );
            planned = plan_stream_offsets(tensors, data_offset);
            index_bytes = build_v2_stream_index(tensors, &planned)?;
            let payload_end = planned
                .last()
                .map(|p| p.offset.saturating_add(p.data_len))
                .unwrap_or(data_offset);
            tail_offset = align_up(payload_end, TENSOR_ALIGN);
            let quant_hash = placeholder_quantization_hash(
                tensors.len(),
                planned.iter().map(|p| p.data_len).sum(),
            );
            let next_front =
                build_front_metadata(front_base.clone(), tail_offset, &tail_bytes, quant_hash)?;
            if next_front == front_json {
                break;
            }
            let old_len = front_json.len();
            front_json = next_front;
            if front_json.len() == old_len {
                break;
            }
        }

        data_offset = align_up(
            metadata_offset + front_json.len() as u64 + index_bytes.len() as u64,
            PAYLOAD_ALIGN,
        );
        planned = plan_stream_offsets(tensors, data_offset);
        index_bytes = build_v2_stream_index(tensors, &planned)?;
        tail_offset = align_up(
            planned
                .last()
                .map(|p| p.offset.saturating_add(p.data_len))
                .unwrap_or(data_offset),
            TENSOR_ALIGN,
        );
        let quant_hash =
            placeholder_quantization_hash(tensors.len(), planned.iter().map(|p| p.data_len).sum());
        front_json =
            build_front_metadata(front_base.clone(), tail_offset, &tail_bytes, quant_hash)?;
        let final_data_offset = align_up(
            metadata_offset + front_json.len() as u64 + index_bytes.len() as u64,
            PAYLOAD_ALIGN,
        );
        if final_data_offset != data_offset {
            data_offset = final_data_offset;
            planned = plan_stream_offsets(tensors, data_offset);
            index_bytes = build_v2_stream_index(tensors, &planned)?;
            tail_offset = align_up(
                planned
                    .last()
                    .map(|p| p.offset.saturating_add(p.data_len))
                    .unwrap_or(data_offset),
                TENSOR_ALIGN,
            );
            let quant_hash = placeholder_quantization_hash(
                tensors.len(),
                planned.iter().map(|p| p.data_len).sum(),
            );
            front_json =
                build_front_metadata(front_base.clone(), tail_offset, &tail_bytes, quant_hash)?;
            let check_data_offset = align_up(
                metadata_offset + front_json.len() as u64 + index_bytes.len() as u64,
                PAYLOAD_ALIGN,
            );
            if check_data_offset != data_offset {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "HFQ v2 live front metadata did not converge",
                ));
            }
        }

        let total_output_bytes = tail_offset.saturating_add(tail_bytes.len() as u64);
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)?;

        f.write_all(INVALID_HFQ_MAGIC)?;
        f.write_all(&HFQ_VERSION.to_le_bytes())?;
        f.write_all(&arch.to_le_bytes())?;
        f.write_all(&(tensors.len() as u32).to_le_bytes())?;
        f.write_all(&metadata_offset.to_le_bytes())?;
        f.write_all(&data_offset.to_le_bytes())?;
        f.write_all(front_json.as_bytes())?;
        f.write_all(&index_bytes)?;
        let pos = f.stream_position()?;
        if pos > data_offset {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("HFQ v2 front metadata/index exceeded data offset ({pos} > {data_offset})"),
            ));
        }
        write_zeroes(&mut f, data_offset - pos)?;

        Ok(Self {
            f,
            front_json,
            index_bytes,
            planned,
            tail_offset,
            payload_hash: PayloadHashState::new(),
            next_tensor: 0,
            total_output_bytes,
            max_written_position: data_offset,
        })
    }

    pub fn write_next(
        &mut self,
        tensor: &HfqStreamTensor,
        mut write_payload: impl FnMut(&mut dyn Write) -> std::io::Result<()>,
        mut progress: impl FnMut(HfqWriteProgress),
    ) -> std::io::Result<()> {
        let p = self.planned.get(self.next_tensor).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "too many tensors written")
        })?;
        let pos = self.f.stream_position()?;
        if pos > p.offset {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "HFQ v2 tensor {} planned offset {} is behind writer position {pos}",
                    tensor.name, p.offset
                ),
            ));
        }
        write_zeroes(&mut self.f, p.offset - pos)?;
        self.payload_hash.start_stream_tensor(tensor, p);
        let mut writer = HashingCountingWriter {
            inner: &mut self.f,
            hash: &mut self.payload_hash,
            written: 0,
        };
        write_payload(&mut writer)?;
        if writer.written != p.data_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "HFQ v2 tensor {} wrote {} bytes, expected {}",
                    tensor.name, writer.written, p.data_len
                ),
            ));
        }
        let pos = self.f.stream_position()?;
        if pos > self.max_written_position {
            self.max_written_position = pos.min(self.total_output_bytes);
            progress(HfqWriteProgress {
                written_bytes: self.max_written_position,
                total_bytes: self.total_output_bytes,
            });
        }
        self.next_tensor += 1;
        Ok(())
    }

    pub fn finish(
        mut self,
        final_metadata_json: &str,
        mut progress: impl FnMut(HfqWriteProgress),
    ) -> std::io::Result<()> {
        if self.next_tensor != self.planned.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "HFQ v2 live writer finalized after {} tensors, expected {}",
                    self.next_tensor,
                    self.planned.len()
                ),
            ));
        }
        let (front_base, tail_bytes) = split_front_tail_metadata(final_metadata_json)?;
        let total_output_bytes = self.tail_offset.saturating_add(tail_bytes.len() as u64);
        let pos = self.f.stream_position()?;
        if pos > self.tail_offset {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "HFQ v2 tail offset {} is behind writer position {pos}",
                    self.tail_offset
                ),
            ));
        }
        write_zeroes(&mut self.f, self.tail_offset - pos)?;
        self.f.write_all(&tail_bytes)?;
        progress(HfqWriteProgress {
            written_bytes: total_output_bytes,
            total_bytes: total_output_bytes,
        });
        self.f.flush()?;

        let final_front_json = build_front_metadata(
            front_base,
            self.tail_offset,
            &tail_bytes,
            self.payload_hash.metadata(),
        )?;
        if final_front_json.len() != self.front_json.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "HFQ v2 final front metadata length changed ({} != {})",
                    final_front_json.len(),
                    self.front_json.len()
                ),
            ));
        }
        self.f.seek(SeekFrom::Start(HEADER_SIZE))?;
        self.f.write_all(final_front_json.as_bytes())?;
        self.f.write_all(&self.index_bytes)?;

        self.f.seek(SeekFrom::Start(0))?;
        self.f.write_all(HFQ_MAGIC)?;
        self.f.flush()?;
        Ok(())
    }
}

struct HashingCountingWriter<'a, W: Write> {
    inner: &'a mut W,
    hash: &'a mut PayloadHashState,
    written: u64,
}

impl<W: Write> Write for HashingCountingWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hash.update_payload(&buf[..n]);
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn write_zeroes(f: &mut File, mut n: u64) -> std::io::Result<()> {
    const ZEROES: [u8; 8192] = [0; 8192];
    while n > 0 {
        let chunk = (n as usize).min(ZEROES.len());
        f.write_all(&ZEROES[..chunk])?;
        n -= chunk as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json_end(bytes: &[u8]) -> usize {
        let mut depth = 0i32;
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
                    depth += 1;
                } else if b == b'}' {
                    depth -= 1;
                    if depth == 0 {
                        return i + 1;
                    }
                }
            }
        }
        panic!("json did not end")
    }

    #[test]
    fn write_hfq_v2_aligns_payloads_and_places_full_metadata_in_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("aligned.hfq");
        let tensors = vec![
            HfqTensor {
                name: "model.layers.0.mlp.experts.1.gate_up_proj.weight".to_string(),
                quant_type: QuantType::Q8F16,
                shape: vec![3, 5],
                group_size: 256,
                data: vec![1u8; 37],
                spilled_len: 0,
            },
            HfqTensor {
                name: "model.layers.0.mlp.experts.1.down_proj.weight".to_string(),
                quant_type: QuantType::Q8F16,
                shape: vec![5, 3],
                group_size: 256,
                data: vec![2u8; 19],
                spilled_len: 0,
            },
            HfqTensor {
                name: "model.norm.weight".to_string(),
                quant_type: QuantType::F16,
                shape: vec![7],
                group_size: 0,
                data: vec![3u8; 14],
                spilled_len: 0,
            },
        ];
        let metadata = serde_json::json!({
            "architecture": "test",
            "config": {"hidden_size": 7},
            "tokenizer": "{\"version\":\"1\"}",
            "tokenizer_config": {"chat_template": "{{ messages }}"},
        });
        write_hfq(
            &path,
            123,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
            None,
        )
        .unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], HFQ_MAGIC);
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            HFQ_VERSION
        );
        let data_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
        assert_eq!(data_offset % 4096, 0);
        let front = &bytes[32..data_offset];
        let front_end = json_end(front);
        let front_json: serde_json::Value = serde_json::from_slice(&front[..front_end]).unwrap();
        assert!(front_json.get("config").is_none());
        let tail_meta = front_json.get("tail_metadata").unwrap();
        let tail_offset = tail_meta["offset"].as_u64().unwrap() as usize;
        let tail_size = tail_meta["size"].as_u64().unwrap() as usize;
        let tail_json: serde_json::Value =
            serde_json::from_slice(&bytes[tail_offset..tail_offset + tail_size]).unwrap();
        assert_eq!(tail_json["metadata"]["config"]["hidden_size"], 7);
        assert_eq!(
            tail_json["metadata"]["tokenizer_config"]["chat_template"],
            "{{ messages }}"
        );

        let mut pos = 32 + front_end;
        let n = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        assert_eq!(n, tensors.len());
        let mut offsets = Vec::new();
        for _ in 0..n {
            let name_len = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2 + name_len;
            pos += 1;
            let n_dims = bytes[pos] as usize;
            pos += 1 + n_dims * 4 + 4;
            let _data_size = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
            pos += 8;
            let offset_div32 = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            offsets.push(offset_div32 * 32);
        }
        assert_eq!(offsets[0] % 4096, 0);
        assert_eq!(offsets[1] % 32, 0);
        assert_eq!(offsets[2] % 32, 0);
    }
}
