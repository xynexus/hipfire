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
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::hash::Hasher;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use twox_hash::XxHash64;

#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;

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

fn expert_role_rank(name: &str) -> u8 {
    let parts = name.split('.').collect::<Vec<_>>();
    let Some(expert_pos) = parts.iter().position(|part| *part == "experts") else {
        return u8::MAX;
    };
    match parts.get(expert_pos + 2).copied() {
        Some("gate_up_proj") | Some("gate_proj") => 0,
        Some("up_proj") => 1,
        Some("down_proj") => 2,
        _ => 3,
    }
}

/// Keep non-routed tensors in source order, then group routed tensors by
/// `(layer, expert)` and projection role. This makes every expert a contiguous
/// page-in unit while remaining independent of the model-family name.
fn canonical_tensor_order(tensors: &[HfqTensor]) -> Vec<&HfqTensor> {
    let mut always = Vec::new();
    let mut experts = BTreeMap::<(u16, u16), Vec<(u8, usize, &HfqTensor)>>::new();
    for (index, tensor) in tensors.iter().enumerate() {
        if let Some(key) = expert_key(&tensor.name) {
            experts
                .entry(key)
                .or_default()
                .push((expert_role_rank(&tensor.name), index, tensor));
        } else {
            always.push(tensor);
        }
    }
    for entries in experts.values_mut() {
        entries.sort_by_key(|(role, index, _)| (*role, *index));
        always.extend(entries.iter().map(|(_, _, tensor)| *tensor));
    }
    always
}

fn metadata_with_routed_modules(
    mut metadata: serde_json::Value,
    tensors: &[&HfqTensor],
    planned: &[PlannedTensor],
) -> std::io::Result<serde_json::Value> {
    if tensors.len() != planned.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "HFQ module planning tensor/offset count mismatch",
        ));
    }
    let mut groups = BTreeMap::<(u16, u16), Vec<(&HfqTensor, &PlannedTensor)>>::new();
    let mut last_key = None;
    let mut closed = std::collections::BTreeSet::new();
    for (tensor, plan) in tensors.iter().zip(planned) {
        let key = expert_key(&tensor.name);
        if key != last_key {
            if let Some(previous) = last_key {
                closed.insert(previous);
            }
            if key.is_some_and(|next| closed.contains(&next)) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "routed expert tensors are not contiguous in canonical HFQ order",
                ));
            }
            last_key = key;
        }
        if let Some(key) = key {
            groups.entry(key).or_default().push((tensor, plan));
        }
    }

    let mut modules = Vec::with_capacity(groups.len());
    for ((layer, expert), entries) in groups {
        let start = entries.first().map(|(_, plan)| plan.offset).unwrap_or(0);
        let end = entries
            .last()
            .map(|(_, plan)| plan.offset.saturating_add(plan.data_len))
            .unwrap_or(start);
        let module_tensors = entries
            .iter()
            .map(|(tensor, plan)| {
                serde_json::json!({
                    "name": tensor.name,
                    "quant_type": tensor.quant_type as u8,
                    "shape": tensor.shape,
                    "group_size": tensor.group_size,
                    "rel_offset": plan.offset.saturating_sub(start),
                    "data_size": plan.data_len,
                })
            })
            .collect::<Vec<_>>();
        modules.push(serde_json::json!({
            "module_id": format!("layers.{layer}.experts.{expert}"),
            "kind": "routed_expert",
            "layer": layer,
            "expert": expert,
            "placement_policy": "lazy_lru",
            "data_offset": start,
            "data_size": end.saturating_sub(start),
            "tensors": module_tensors,
        }));
    }
    let object = metadata.as_object_mut().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "HFQ metadata must be a JSON object",
        )
    })?;
    object.remove("hfqm_modules");
    if !modules.is_empty() {
        object.insert(
            "hfqm_modules".to_string(),
            serde_json::json!({
                "format": "hipfire.hfqm.modules.v1",
                "modules": modules,
            }),
        );
    }
    Ok(metadata)
}

fn split_front_tail_metadata(metadata_json: &str) -> std::io::Result<(serde_json::Value, Vec<u8>)> {
    let full: serde_json::Value = serde_json::from_str(metadata_json)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    split_front_tail_metadata_value(full)
}

fn split_front_tail_metadata_value(
    full: serde_json::Value,
) -> std::io::Result<(serde_json::Value, Vec<u8>)> {
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
            "hfqm_modules",
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedTensor {
    offset: u64,
    data_len: u64,
}

fn build_v2_index(tensors: &[&HfqTensor], planned: &[PlannedTensor]) -> std::io::Result<Vec<u8>> {
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

fn plan_offsets(tensors: &[&HfqTensor], payload_start: u64) -> Vec<PlannedTensor> {
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

    for t in canonical_tensor_order(tensors) {
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
            let (spill_offset, _) = spill
                .expect("spilled tensor requires spill state")
                .location(&t.name, t.spilled_len)?;
            reader.seek(SeekFrom::Start(spill_offset))?;
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
    locations: BTreeMap<String, (u64, u64)>,
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
            locations: BTreeMap::new(),
        })
    }

    /// Write named tensor data to the spill file. The recorded byte range lets
    /// the final writer emit tensors in canonical module order without loading
    /// them back into RAM or assuming spill order equals artifact order.
    pub fn spill(&mut self, name: &str, data: &[u8]) -> std::io::Result<u64> {
        use std::io::Write;
        if self.locations.contains_key(name) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("tensor {name} was already spilled"),
            ));
        }
        let start = self.offset;
        self.file.write_all(data)?;
        self.offset += data.len() as u64;
        self.locations
            .insert(name.to_string(), (start, data.len() as u64));
        Ok(data.len() as u64)
    }

    fn location(&self, name: &str, expected_len: u64) -> std::io::Result<(u64, u64)> {
        let location = self.locations.get(name).copied().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("spilled tensor {name} has no recorded byte range"),
            )
        })?;
        if location.1 != expected_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "spilled tensor {name} range is {} bytes, expected {expected_len}",
                    location.1
                ),
            ));
        }
        Ok(location)
    }

    /// Release filesystem blocks for a tensor after the final HFQ writer has
    /// copied and hashed it. Keeping the logical file length and location map
    /// intact lets the reader continue seeking to later tensors while peak
    /// disk usage stays close to one artifact instead of spill + artifact.
    fn release_location(&self, name: &str, expected_len: u64) -> std::io::Result<()> {
        let (offset, len) = self.location(name, expected_len)?;
        #[cfg(target_os = "linux")]
        {
            let result = unsafe {
                libc::fallocate(
                    self.file.get_ref().as_raw_fd(),
                    libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                    offset as libc::off_t,
                    len as libc::off_t,
                )
            };
            if result != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = (offset, len);
        Ok(())
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

/// Stamp `identity` into the metadata JSON so the container declares what it is
/// rather than leaving readers to infer it from the numeric header id.
///
/// Three deliberate behaviours:
///
/// - An `identity` the caller already supplied is left alone. The caller knows
///   the variant and role; this only derives the family from `arch`.
/// - An `arch` outside the frozen legacy map is left unstamped. Sidecar ids
///   (DFlash draft, MTP head) are tooling-only and never name a servable
///   family, so inventing one for them would be a lie.
/// - No header version bump. `version` selects the container's structural
///   layout and is read by two independent parsers; an added JSON key changes
///   no layout, so presence of `identity` is the signal instead.
fn stamp_identity(metadata_json: &str, arch: u32) -> std::io::Result<String> {
    let mut meta: serde_json::Value = serde_json::from_str(metadata_json)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let Some(object) = meta.as_object_mut() else {
        return Ok(metadata_json.to_string());
    };
    if object.contains_key("identity") {
        return Ok(metadata_json.to_string());
    }
    let Some(identity) = hipfire_arch_api::identity_for_legacy_arch_id(arch) else {
        return Ok(metadata_json.to_string());
    };
    object.insert(
        "identity".to_string(),
        hipfire_model::identity_json(identity),
    );
    Ok(meta.to_string())
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
    let metadata_json = &stamp_identity(metadata_json, arch)?;
    let metadata_base: serde_json::Value = serde_json::from_str(metadata_json)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let ordered = canonical_tensor_order(tensors);
    let metadata_offset = HEADER_SIZE;

    let mut front_json = String::new();
    let mut front_base = serde_json::json!({});
    let mut tail_bytes = Vec::new();
    let mut index_bytes = Vec::new();
    let mut planned = Vec::new();
    let mut data_offset = 0u64;
    let mut tail_offset = 0u64;
    let mut converged = false;
    for _ in 0..32 {
        let next_data_offset = align_up(
            metadata_offset + front_json.len() as u64 + index_bytes.len() as u64,
            PAYLOAD_ALIGN,
        );
        let next_planned = plan_offsets(&ordered, next_data_offset);
        let next_index = build_v2_index(&ordered, &next_planned)?;
        let full_metadata =
            metadata_with_routed_modules(metadata_base.clone(), &ordered, &next_planned)?;
        let (next_front_base, next_tail_bytes) = split_front_tail_metadata_value(full_metadata)?;
        let payload_end = next_planned
            .last()
            .map(|p| p.offset.saturating_add(p.data_len))
            .unwrap_or(next_data_offset);
        let next_tail_offset = align_up(payload_end, TENSOR_ALIGN);
        let quant_hash = placeholder_quantization_hash(
            ordered.len(),
            next_planned.iter().map(|p| p.data_len).sum(),
        );
        let next_front = build_front_metadata(
            next_front_base.clone(),
            next_tail_offset,
            &next_tail_bytes,
            quant_hash,
        )?;
        converged = next_data_offset == data_offset
            && next_tail_offset == tail_offset
            && next_planned == planned
            && next_index == index_bytes
            && next_tail_bytes == tail_bytes
            && next_front == front_json;
        data_offset = next_data_offset;
        tail_offset = next_tail_offset;
        planned = next_planned;
        index_bytes = next_index;
        front_base = next_front_base;
        tail_bytes = next_tail_bytes;
        front_json = next_front;
        if converged {
            break;
        }
    }
    if !converged {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "HFQ v2 module/front metadata did not converge",
        ));
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
    f.write_all(&(ordered.len() as u32).to_le_bytes())?;
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

    let mut spill_reader = if let Some(spill) = spill.as_ref() {
        Some(std::io::BufReader::new(File::open(&spill.path)?))
    } else {
        None
    };
    let mut buf = vec![0u8; 4 * 1024 * 1024];
    let mut payload_hash = PayloadHashState::new();
    for (t, p) in ordered.iter().zip(&planned) {
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
            let (spill_offset, _) = spill
                .as_ref()
                .expect("spilled tensor requires spill state")
                .location(&t.name, t.spilled_len)?;
            reader.seek(SeekFrom::Start(spill_offset))?;
            let mut remaining = t.spilled_len as usize;
            while remaining > 0 {
                let chunk = remaining.min(buf.len());
                reader.read_exact(&mut buf[..chunk])?;
                f.write_all(&buf[..chunk])?;
                payload_hash.update_payload(&buf[..chunk]);
                remaining -= chunk;
                report_progress(f.stream_position()?);
            }
            // Best effort: unsupported filesystems retain the old, safe
            // double-space behavior. Linux filesystems with hole punching
            // release copied spill blocks incrementally.
            let _ = spill
                .as_ref()
                .expect("spilled tensor requires spill state")
                .release_location(&t.name, t.spilled_len);
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
    let metadata_json = &stamp_identity(metadata_json, arch)?;
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
mod identity_stamp_tests {
    use super::*;

    fn identity_of(json: &str) -> Option<serde_json::Value> {
        serde_json::from_str::<serde_json::Value>(json)
            .unwrap()
            .get("identity")
            .cloned()
    }

    #[test]
    fn a_known_arch_id_stamps_its_family() {
        let out = stamp_identity(r#"{"config":{}}"#, hipfire_arch_api::ARCH_ID_ZAYA).unwrap();
        assert_eq!(
            identity_of(&out),
            Some(serde_json::json!({ "family": "zaya" })),
        );
    }

    #[test]
    fn a_caller_supplied_identity_is_left_alone() {
        // A3 will pass variant and role down from detection; the writer must
        // not flatten that back to a bare family.
        let rich = r#"{"identity":{"family":"gemma4","variant":"moe","role":"vl"},"config":{}}"#;
        let out = stamp_identity(rich, hipfire_arch_api::ARCH_ID_ZAYA).unwrap();
        assert_eq!(
            identity_of(&out),
            Some(serde_json::json!({"family":"gemma4","variant":"moe","role":"vl"})),
        );
    }

    #[test]
    fn a_tooling_only_arch_id_is_not_stamped() {
        // Sidecar ids (DFlash draft, MTP head) name no servable family;
        // inventing one for them would be a lie.
        for sidecar in [20u32, 21, 9999] {
            let out = stamp_identity(r#"{"config":{}}"#, sidecar).unwrap();
            assert_eq!(
                identity_of(&out),
                None,
                "arch {sidecar} must not be stamped"
            );
        }
    }

    #[test]
    fn a_stamped_container_resolves_back_to_the_same_identity() {
        let out = stamp_identity(r#"{"config":{}}"#, hipfire_arch_api::ARCH_ID_GEMMA4).unwrap();
        assert_eq!(
            hipfire_model::identity_from_metadata(&out),
            Some(hipfire_arch_api::ArchRef::base("gemma4")),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn spill_range_release_punches_copied_payload_without_touching_neighbor() {
        let dir = tempfile::tempdir().unwrap();
        let mut spill = TensorSpill::new(dir.path()).unwrap();
        let first = vec![0x11u8; 128 * 1024];
        let second = vec![0x22u8; 128 * 1024];
        spill.spill("first", &first).unwrap();
        spill.spill("second", &second).unwrap();
        spill.flush().unwrap();

        spill.release_location("first", first.len() as u64).unwrap();
        let bytes = std::fs::read(&spill.path).unwrap();
        assert!(bytes[..first.len()].iter().all(|&byte| byte == 0));
        assert_eq!(&bytes[first.len()..first.len() + second.len()], second);
    }

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
                quant_type: QuantType::F16,
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
        assert!(front_json.get("hfqm_modules").is_none());
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
        let modules = &tail_json["metadata"]["hfqm_modules"];
        assert_eq!(modules["format"], "hipfire.hfqm.modules.v1");
        assert_eq!(modules["modules"].as_array().unwrap().len(), 1);
        assert_eq!(modules["modules"][0]["kind"], "routed_expert");
        assert_eq!(modules["modules"][0]["layer"], 0);
        assert_eq!(modules["modules"][0]["expert"], 1);
        assert_eq!(modules["modules"][0]["tensors"][0]["quant_type"], 3);
        assert_eq!(modules["modules"][0]["tensors"][1]["quant_type"], 1);

        let mut pos = 32 + front_end;
        let n = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        assert_eq!(n, tensors.len());
        let mut names = Vec::new();
        let mut offsets = Vec::new();
        for _ in 0..n {
            let name_len = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            names.push(String::from_utf8(bytes[pos..pos + name_len].to_vec()).unwrap());
            pos += name_len;
            pos += 1;
            let n_dims = bytes[pos] as usize;
            pos += 1 + n_dims * 4 + 4;
            let _data_size = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
            pos += 8;
            let offset_div32 = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            offsets.push(offset_div32 * 32);
        }
        assert_eq!(
            names,
            [
                "model.norm.weight",
                "model.layers.0.mlp.experts.1.gate_up_proj.weight",
                "model.layers.0.mlp.experts.1.down_proj.weight",
            ]
        );
        assert_eq!(offsets[0] % 4096, 0);
        assert_eq!(offsets[1] % 4096, 0);
        assert_eq!(offsets[2] % 32, 0);
        assert_eq!(modules["modules"][0]["data_offset"], offsets[1]);
    }
}
