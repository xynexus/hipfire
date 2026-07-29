// SPDX-License-Identifier: Apache-2.0
//! Pass-two storage preflight — the quant-format byte math.
//!
//! Ported from `two_pass_quantize.pass_two_storage_preflight` and its helpers
//! (`_oq_block_bytes`, `_quantized_tensor_bytes`, `_routed_expert_identity`,
//! `_source_precision_output_bytes`, `_safetensors_index`, `_resolve_snapshot`).
//! The Python re-hardcoded the on-disk block geometry; here the fixed OQ4/OQ8/Q8
//! block sizes and group widths come from `hipfire_quant_format::QuantType`, the
//! single source of truth, rather than being re-derived. The mixed-precision
//! nominal (`130 + 2·overlays`) stays the wrapper's own estimate — a nominal OQ
//! width, deliberately not the real per-tensor OqPlus layout — matching Python.

use hipfire_quant_format::QuantType;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

const PASS_TWO_FIXED_SAFETY_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const PASS_TWO_RELATIVE_SAFETY: f64 = 0.10;
const PASS_TWO_CONTAINER_OVERHEAD_BYTES: u64 = 16 * 1024 * 1024;
const PASS_TWO_TENSOR_ALIGNMENT_BYTES: u64 = 4096;

fn dtype_bytes(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E4M3FN" | "F8_E5M2" => 1,
        "U16" | "I16" | "F16" | "BF16" => 2,
        "U32" | "I32" | "F32" => 4,
        "U64" | "I64" | "F64" => 8,
        _ => return None,
    })
}

#[derive(Clone, Debug)]
pub struct SafetensorTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub numel: u64,
    pub source_bytes: u64,
}

/// Resolve a Hugging Face cache root or snapshot directory to the snapshot that
/// holds `config.json`, matching `two_pass_quantize._resolve_snapshot`:
/// config.json in place → `refs/main` → newest `snapshots/*` by mtime.
pub fn resolve_snapshot(path: &Path) -> Result<PathBuf, String> {
    let path = super::python_resolve(path);
    if path.join("config.json").is_file() {
        return Ok(path);
    }
    let main_ref = path.join("refs").join("main");
    if main_ref.is_file() {
        if let Ok(rev) = std::fs::read_to_string(&main_ref) {
            let candidate = path.join("snapshots").join(rev.trim());
            if candidate.join("config.json").is_file() {
                return Ok(super::python_resolve(&candidate));
            }
        }
    }
    let snapshots = path.join("snapshots");
    if snapshots.is_dir() {
        let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&snapshots)
            .map_err(|e| format!("read {}: {e}", snapshots.display()))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|candidate| candidate.join("config.json").is_file())
            .filter_map(|candidate| {
                let mtime = std::fs::metadata(&candidate).and_then(|m| m.modified()).ok()?;
                Some((mtime, candidate))
            })
            .collect();
        candidates.sort_by_key(|(mtime, _)| *mtime);
        if let Some((_, candidate)) = candidates.last() {
            return Ok(super::python_resolve(candidate));
        }
    }
    Err(format!(
        "no Hugging Face snapshot/config.json under {}",
        path.display()
    ))
}

/// Read the safetensors shard headers (only) into a flat tensor index, the twin
/// of `two_pass_quantize._safetensors_index`.
pub fn safetensors_index(model: &Path) -> Result<Vec<SafetensorTensor>, String> {
    let snapshot = resolve_snapshot(model)?;
    let mut shards: Vec<PathBuf> = std::fs::read_dir(&snapshot)
        .map_err(|e| format!("read {}: {e}", snapshot.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(format!("no safetensors files under {}", snapshot.display()));
    }
    let mut tensors = Vec::new();
    for shard in &shards {
        let header = read_safetensors_header(shard)?;
        let map = header
            .as_object()
            .ok_or_else(|| format!("safetensors header is not an object: {}", shard.display()))?;
        for (name, value) in map {
            if name == "__metadata__" {
                continue;
            }
            let entry = value.as_object().ok_or_else(|| {
                format!("invalid safetensors index entry {name:?} in {}", shard.display())
            })?;
            let shape: Vec<u64> = entry
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("invalid safetensors index entry {name:?} in {}", shard.display()))?
                .iter()
                .map(|d| d.as_u64())
                .collect::<Option<Vec<u64>>>()
                .ok_or_else(|| format!("invalid shape for {name:?} in {}", shard.display()))?;
            let offsets = entry
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .filter(|a| a.len() == 2)
                .ok_or_else(|| format!("invalid data_offsets for {name:?} in {}", shard.display()))?;
            let begin = offsets[0]
                .as_u64()
                .ok_or_else(|| format!("invalid data_offsets for {name:?} in {}", shard.display()))?;
            let end = offsets[1]
                .as_u64()
                .ok_or_else(|| format!("invalid data_offsets for {name:?} in {}", shard.display()))?;
            if end < begin {
                return Err(format!("invalid data_offsets for {name:?} in {}", shard.display()));
            }
            let dtype = entry
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("invalid dtype for {name:?} in {}", shard.display()))?
                .to_string();
            let numel: u64 = shape.iter().product();
            let byte_len = end - begin;
            if let Some(width) = dtype_bytes(&dtype) {
                if byte_len != numel * width {
                    return Err(format!(
                        "safetensors byte length mismatch for {name}: {byte_len} != {numel}*{width}"
                    ));
                }
            }
            tensors.push(SafetensorTensor {
                name: name.clone(),
                dtype,
                shape,
                numel,
                source_bytes: byte_len,
            });
        }
    }
    Ok(tensors)
}

fn read_safetensors_header(shard: &Path) -> Result<Value, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|_| format!("truncated safetensors header prefix: {}", shard.display()))?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len > 1024 * 1024 * 1024 {
        return Err(format!(
            "unreasonable safetensors header size {header_len}: {}",
            shard.display()
        ));
    }
    let mut encoded = vec![0u8; header_len as usize];
    file.read_exact(&mut encoded)
        .map_err(|_| format!("truncated safetensors header: {}", shard.display()))?;
    serde_json::from_slice(&encoded)
        .map_err(|e| format!("invalid safetensors header json in {}: {e}", shard.display()))
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ExpertRole {
    GateUp,
    Down,
}

/// `(layer, explicit_expert, role)` if `name` is a routed-expert projection —
/// including grouped `[experts, ...]` tensors (`explicit_expert == None`) and
/// already-split layouts. Twin of `two_pass_quantize._routed_expert_identity`.
pub fn routed_expert_identity(name: &str) -> Option<(u64, Option<u64>, ExpertRole)> {
    let parts: Vec<&str> = name.split('.').collect();
    let layer_at = parts.iter().position(|p| *p == "layers")?;
    let layer: u64 = parts.get(layer_at + 1)?.parse().ok()?;
    // `experts` must appear at or after layer_at + 2.
    let expert_at = parts
        .iter()
        .enumerate()
        .skip(layer_at + 2)
        .find(|(_, p)| **p == "experts")
        .map(|(i, _)| i)?;
    let mut suffix: Vec<&str> = parts.get(expert_at + 1..)?.to_vec();
    if suffix.is_empty() {
        return None;
    }
    let mut expert = None;
    if suffix[0].chars().all(|c| c.is_ascii_digit()) && !suffix[0].is_empty() {
        expert = suffix.remove(0).parse().ok();
    }
    if suffix.is_empty() {
        return None;
    }
    let role = match suffix[0] {
        "gate_up_proj" | "gate_proj" | "up_proj" | "w1" | "w3" => ExpertRole::GateUp,
        "down_proj" | "w2" => ExpertRole::Down,
        _ => return None,
    };
    Some((layer, expert, role))
}

/// `(storage_bits_per_weight, nominal_on_disk_block_bytes)` for an OQ target
/// format string. Twin of `two_pass_quantize._oq_block_bytes`. The fixed OQ4/OQ8
/// blocks come from `QuantType::block_bytes` (130 / 258); mixed widths add
/// `2·overlays` to the OQ4 base, as the Python wrapper does.
pub fn oq_block_bytes(quant_format: &str) -> Result<(f64, u64), String> {
    let base = quant_format.strip_suffix("++").unwrap_or(quant_format);
    let base = base.strip_suffix('+').unwrap_or(base);
    let oq4_block = QuantType::Oq4G256.block_bytes().expect("Oq4G256 fixed") as u64; // 130
    let oq8_block = QuantType::Oq8G256.block_bytes().expect("Oq8G256 fixed") as u64; // 258
    if base == "oq4" {
        return Ok((4.0625, oq4_block));
    }
    if base == "oq8" {
        return Ok((8.0625, oq8_block));
    }
    // Mixed Opus width: `oq<int>.<frac>` exactly.
    let mixed = base
        .strip_prefix("oq")
        .filter(|rest| {
            let mut halves = rest.splitn(2, '.');
            match (halves.next(), halves.next()) {
                (Some(a), Some(b)) => {
                    !a.is_empty()
                        && !b.is_empty()
                        && a.chars().all(|c| c.is_ascii_digit())
                        && b.chars().all(|c| c.is_ascii_digit())
                }
                _ => false,
            }
        })
        .and_then(|rest| rest.parse::<f64>().ok());
    let requested = mixed.ok_or_else(|| {
        format!("pass-two storage admission does not know the on-disk block size for {quant_format:?}")
    })?;
    let overlays = ((requested - 4.0625) * 16.0).round() as i64;
    if !(1..=62).contains(&overlays) || (4.0625 + overlays as f64 / 16.0 - requested).abs() > 1e-6 {
        return Err(format!("invalid mixed Opus storage width {quant_format:?}"));
    }
    Ok((requested, oq4_block + 2 * overlays as u64))
}

fn source_precision_output_bytes(tensor: &SafetensorTensor) -> u64 {
    match tensor.dtype.as_str() {
        "BF16" | "F16" | "F32" => tensor.numel * 2,
        _ => tensor.source_bytes,
    }
}

/// `ceil(numel/group) * block`. Twin of `_quantized_tensor_bytes`: the Q8F16
/// ceiling (34 B / 32-group) is used when `q8`, otherwise the OQ 256-group block.
/// Both the block and group widths come from `QuantType`.
fn quantized_tensor_bytes(numel: u64, block_bytes: u64, q8: bool) -> u64 {
    let (block, group) = if q8 {
        (
            QuantType::Q8F16.block_bytes().expect("Q8F16 fixed") as u64, // 34
            QuantType::Q8F16.group_size() as u64,                        // 32
        )
    } else {
        (block_bytes, QuantType::Oq4G256.group_size() as u64) // 256
    };
    numel.div_ceil(group) * block
}

fn nearest_existing_path(path: &Path) -> Option<PathBuf> {
    let mut candidate = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()?.to_path_buf()
    };
    while !candidate.exists() {
        if !candidate.pop() {
            return None;
        }
    }
    Some(super::python_resolve(&candidate))
}

#[cfg(target_os = "linux")]
fn available_bytes_at(path: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c_path = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: c_path is NUL-terminated; statvfs initializes stats on success.
    if unsafe { libc::statvfs(c_path.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: statvfs returned 0.
    let stats = unsafe { stats.assume_init() };
    Some((stats.f_bavail as u64).saturating_mul(stats.f_frsize as u64))
}

#[cfg(not(target_os = "linux"))]
fn available_bytes_at(_path: &Path) -> Option<u64> {
    None
}

/// Estimate pass-two disk demand from the safetensors index only (no payload
/// reads). Byte-identical to `two_pass_quantize.pass_two_storage_preflight`.
///
/// `available_bytes` overrides the `statvfs` probe (used by tests); pass `None`
/// in production to probe the nearest existing ancestor of `output`.
pub fn pass_two_storage_preflight(
    model: &Path,
    output: &Path,
    quant_format: &str,
    calibration: &Value,
    available_bytes: Option<u64>,
) -> Result<Value, String> {
    let tensors = safetensors_index(model)?;
    let snapshot = resolve_snapshot(model)?.to_string_lossy().into_owned();
    preflight_from_index(
        &tensors,
        &snapshot,
        output,
        quant_format,
        calibration,
        available_bytes,
    )
}

/// The byte math over an already-built tensor index — split out so it is unit
/// testable against the Python golden without a real checkpoint on disk.
pub fn preflight_from_index(
    tensors: &[SafetensorTensor],
    snapshot: &str,
    output: &Path,
    quant_format: &str,
    calibration: &Value,
    available_bytes: Option<u64>,
) -> Result<Value, String> {
    let (storage_bits, block_bytes) = oq_block_bytes(quant_format)?;

    // Preserved (kept full precision) experts, keyed (layer, expert).
    let preserved_values = super::dig(Some(calibration), &["metadata", "preserve_high_precision"])
        .cloned()
        .unwrap_or(Value::Array(Vec::new()));
    let preserved_values = preserved_values
        .as_array()
        .ok_or("calibration preserve_high_precision is not a list")?;
    let mut preserved: HashSet<(i64, i64)> = HashSet::new();
    for value in preserved_values {
        let layer = value.get("layer").and_then(|v| v.as_i64());
        let expert = value.get("expert").and_then(|v| v.as_i64());
        match (value.as_object(), layer, expert) {
            (Some(_), Some(layer), Some(expert)) => {
                preserved.insert((layer, expert));
            }
            _ => {
                return Err(format!(
                    "invalid calibration preserve_high_precision entry: {value}"
                ))
            }
        }
    }

    let mut payload_bytes: u64 = 0;
    let mut nominal_payload_bytes: u64 = 0;
    let mut preserve_output_bytes: u64 = 0;
    let mut preserve_nominal_bytes: u64 = 0;
    let mut matched_roles: HashMap<(i64, i64), HashSet<ExpertRole>> =
        preserved.iter().map(|k| (*k, HashSet::new())).collect();
    let mut output_tensors: u64 = 0;
    let mut source_payload_bytes: u64 = 0;
    let mut source_parameters: u64 = 0;

    for tensor in tensors {
        source_payload_bytes += tensor.source_bytes;
        source_parameters += tensor.numel;
        if let Some((layer, explicit_expert, role)) = routed_expert_identity(&tensor.name) {
            let (expert_numel, experts): (u64, Vec<u64>) = match explicit_expert {
                None => {
                    if tensor.shape.len() < 2 || tensor.shape[0] < 1 {
                        return Err(format!(
                            "grouped routed-expert tensor has no expert dimension: {}",
                            tensor.name
                        ));
                    }
                    let count = tensor.shape[0];
                    let numel: u64 = tensor.shape[1..].iter().product();
                    (numel, (0..count).collect())
                }
                Some(expert) => (tensor.numel, vec![expert]),
            };
            for expert in experts {
                let nominal = quantized_tensor_bytes(expert_numel, block_bytes, false);
                nominal_payload_bytes += nominal;
                let key = (layer as i64, expert as i64);
                if preserved.contains(&key) {
                    let full = expert_numel * 2;
                    payload_bytes += full;
                    preserve_output_bytes += full;
                    preserve_nominal_bytes += nominal;
                    matched_roles.get_mut(&key).unwrap().insert(role);
                } else {
                    payload_bytes += nominal;
                }
                output_tensors += 1;
            }
            continue;
        }

        let is_weight = tensor.shape.len() >= 2 && tensor.name.ends_with(".weight");
        let encoded = if is_weight {
            quantized_tensor_bytes(tensor.numel, block_bytes, true)
        } else {
            source_precision_output_bytes(tensor)
        };
        payload_bytes += encoded;
        nominal_payload_bytes += encoded;
        output_tensors += 1;
    }

    if !preserved.is_empty() {
        let mut missing: Vec<(i64, i64)> = matched_roles
            .iter()
            .filter(|(_, roles)| roles.is_empty())
            .map(|(k, _)| *k)
            .collect();
        missing.sort();
        let both: HashSet<ExpertRole> = [ExpertRole::GateUp, ExpertRole::Down].into_iter().collect();
        let mut incomplete: Vec<(i64, i64)> = matched_roles
            .iter()
            .filter(|(_, roles)| !roles.is_empty() && **roles != both)
            .map(|(k, _)| *k)
            .collect();
        incomplete.sort();
        if !missing.is_empty() {
            return Err(format!(
                "calibration preserves {} experts with no routed-expert source tensors: {:?}",
                missing.len(),
                &missing[..missing.len().min(8)]
            ));
        }
        if !incomplete.is_empty() {
            return Err(format!(
                "calibration preserved experts lack both routed roles in source index: {:?}",
                &incomplete[..incomplete.len().min(8)]
            ));
        }
    }

    let alignment_bytes = output_tensors * PASS_TWO_TENSOR_ALIGNMENT_BYTES;
    let artifact_estimate = payload_bytes + alignment_bytes + PASS_TWO_CONTAINER_OVERHEAD_BYTES;
    let relative = (artifact_estimate as f64 * PASS_TWO_RELATIVE_SAFETY).ceil() as u64;
    let safety_margin = PASS_TWO_FIXED_SAFETY_BYTES.max(relative);
    let required_free = artifact_estimate + safety_margin;
    let probe_path = nearest_existing_path(output)
        .ok_or_else(|| format!("no existing parent for output path {}", output.display()))?;
    let available = match available_bytes {
        Some(bytes) => bytes,
        None => available_bytes_at(&probe_path)
            .ok_or_else(|| format!("statvfs failed for {}", probe_path.display()))?,
    };
    let sufficient = available >= required_free;
    let matched_experts = matched_roles
        .values()
        .filter(|roles| {
            **roles == [ExpertRole::GateUp, ExpertRole::Down].into_iter().collect::<HashSet<_>>()
        })
        .count() as u64;

    Ok(json!({
        "schema": "hipfire.pass_two_storage_preflight.v1",
        "index_only": true,
        "payload_values_read": false,
        "format": quant_format,
        "storage_bits_per_weight": storage_bits,
        "source": {
            "snapshot": snapshot,
            "tensors": tensors.len(),
            "parameters": source_parameters,
            "payload_bytes": source_payload_bytes,
        },
        "preserve_high_precision": {
            "requested_experts": preserved.len(),
            "matched_experts": matched_experts,
            "output_bytes": preserve_output_bytes,
            "nominal_quantized_bytes": preserve_nominal_bytes,
            "delta_bytes": preserve_output_bytes as i64 - preserve_nominal_bytes as i64,
        },
        "estimate": {
            "nominal_payload_bytes": nominal_payload_bytes,
            "mixed_payload_bytes": payload_bytes,
            "nonexpert_weight_ceiling": "q8f16",
            "tensor_alignment_bytes": alignment_bytes,
            "fixed_container_overhead_bytes": PASS_TWO_CONTAINER_OVERHEAD_BYTES,
            "completed_artifact_estimate_bytes": artifact_estimate,
            "safety_margin_bytes": safety_margin,
            "required_free_bytes": required_free,
        },
        "filesystem": {
            "probe_path": probe_path.to_string_lossy(),
            "available_bytes": available,
            "required_free_bytes": required_free,
            "sufficient": sufficient,
        },
    }))
}

/// Hard-fail when the preflight says the target filesystem cannot hold pass two.
/// Twin of `two_pass_quantize.require_pass_two_storage`.
pub fn require_pass_two_storage(preflight: &Value) -> Result<(), String> {
    let filesystem = preflight
        .get("filesystem")
        .ok_or("preflight has no filesystem section")?;
    if filesystem.get("sufficient").and_then(|v| v.as_bool()) != Some(true) {
        return Err(format!(
            "insufficient output storage for pass two: {} bytes available at {}, {} bytes required by the preserved-expert-aware estimate",
            filesystem.get("available_bytes").and_then(|v| v.as_u64()).unwrap_or(0),
            filesystem.get("probe_path").and_then(|v| v.as_str()).unwrap_or(""),
            filesystem.get("required_free_bytes").and_then(|v| v.as_u64()).unwrap_or(0),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oq_block_bytes_matches_python_golden() {
        assert_eq!(oq_block_bytes("oq4").unwrap(), (4.0625, 130));
        assert_eq!(oq_block_bytes("oq8").unwrap(), (8.0625, 258));
        assert_eq!(oq_block_bytes("oq4+").unwrap(), (4.0625, 130));
        assert_eq!(oq_block_bytes("oq4++").unwrap(), (4.0625, 130));
        assert_eq!(oq_block_bytes("oq8+").unwrap(), (8.0625, 258));
        assert_eq!(oq_block_bytes("oq4.25").unwrap(), (4.25, 136));
        assert_eq!(oq_block_bytes("oq4.25++").unwrap(), (4.25, 136));
        assert_eq!(oq_block_bytes("oq4.5++").unwrap(), (4.5, 144));
        assert_eq!(oq_block_bytes("oq4.125+").unwrap(), (4.125, 132));
        assert_eq!(oq_block_bytes("oq5.0").unwrap(), (5.0, 160));
        assert!(oq_block_bytes("mq4").is_err());
    }

    #[test]
    fn quantized_tensor_bytes_matches_python_golden() {
        // (numel, block, q8) -> bytes, from scripts/two_pass_quantize.py.
        let cases: &[(u64, u64, bool, u64)] = &[
            (0, 130, false, 0),
            (0, 130, true, 0),
            (1, 130, false, 130),
            (1, 130, true, 34),
            (255, 130, true, 272),
            (256, 258, false, 258),
            (257, 130, false, 260),
            (257, 136, false, 272),
            (257, 258, false, 516),
            (512, 136, false, 272),
            (1000, 258, false, 1032),
            (873438784, 130, false, 443543230),
            (873438784, 136, true, 928028708),
            (873438784, 258, false, 880262718),
        ];
        for &(numel, block, q8, expect) in cases {
            assert_eq!(quantized_tensor_bytes(numel, block, q8), expect, "{numel},{block},{q8}");
        }
    }

    #[test]
    fn routed_expert_identity_matches_python_golden() {
        assert_eq!(
            routed_expert_identity("model.layers.3.mlp.experts.7.gate_proj.weight"),
            Some((3, Some(7), ExpertRole::GateUp))
        );
        assert_eq!(
            routed_expert_identity("model.layers.3.mlp.experts.7.down_proj.weight"),
            Some((3, Some(7), ExpertRole::Down))
        );
        assert_eq!(
            routed_expert_identity("model.layers.3.mlp.experts.gate_up_proj"),
            Some((3, None, ExpertRole::GateUp))
        );
        assert_eq!(
            routed_expert_identity("model.layers.0.mlp.experts.w1.weight"),
            Some((0, None, ExpertRole::GateUp))
        );
        assert_eq!(
            routed_expert_identity("model.layers.0.mlp.experts.w2"),
            Some((0, None, ExpertRole::Down))
        );
        assert_eq!(routed_expert_identity("model.layers.0.self_attn.q_proj.weight"), None);
        assert_eq!(routed_expert_identity("model.embed_tokens.weight"), None);
        assert_eq!(routed_expert_identity("model.layers.2.mlp.experts.5.foo.weight"), None);
    }

    fn fake(name: &str, shape: &[u64]) -> SafetensorTensor {
        let numel: u64 = shape.iter().product();
        SafetensorTensor {
            name: name.into(),
            dtype: "BF16".into(),
            shape: shape.to_vec(),
            numel,
            source_bytes: numel * 2,
        }
    }

    #[test]
    fn moe_preflight_matches_python_golden() {
        // Synthetic MoE index + one preserved expert (layer 0, expert 3),
        // golden captured from a monkeypatched scripts/two_pass_quantize.py.
        let tensors = vec![
            fake("model.layers.0.mlp.experts.gate_up_proj", &[8, 512, 256]),
            fake("model.layers.0.mlp.experts.down_proj", &[8, 256, 512]),
            fake("model.layers.0.self_attn.q_proj.weight", &[512, 512]),
            fake("model.embed_tokens.weight", &[1000, 512]),
            fake("model.layers.0.input_layernorm.weight", &[512]),
        ];
        let calibration = json!({"metadata": {"preserve_high_precision": [{"layer": 0, "expert": 3}]}});
        let expect: &[(&str, u64, u64, u64, u64, u64, u64, u64, u64)] = &[
            // fmt, nominal, mixed, align, artifact, required, preserve_out, preserve_nominal, delta
            ("oq4", 1888512, 2279680, 77824, 19134720, 68738611456, 524288, 133120, 391168),
            ("oq8", 2937088, 3197184, 77824, 20052224, 68739528960, 524288, 264192, 260096),
            ("oq4.25++", 1937664, 2322688, 77824, 19177728, 68738654464, 524288, 139264, 385024),
        ];
        for &(fmt, nominal, mixed, align, artifact, required, p_out, p_nom, delta) in expect {
            let pf = preflight_from_index(
                &tensors,
                "/fake/snap",
                Path::new("/tmp/x.hfq"),
                fmt,
                &calibration,
                Some(1_000_000_000_000_000),
            )
            .unwrap();
            let e = &pf["estimate"];
            assert_eq!(e["nominal_payload_bytes"], nominal, "{fmt} nominal");
            assert_eq!(e["mixed_payload_bytes"], mixed, "{fmt} mixed");
            assert_eq!(e["tensor_alignment_bytes"], align, "{fmt} align");
            assert_eq!(e["completed_artifact_estimate_bytes"], artifact, "{fmt} artifact");
            assert_eq!(e["required_free_bytes"], required, "{fmt} required");
            let p = &pf["preserve_high_precision"];
            assert_eq!(p["requested_experts"], 1);
            assert_eq!(p["matched_experts"], 1, "{fmt} matched");
            assert_eq!(p["output_bytes"], p_out, "{fmt} preserve out");
            assert_eq!(p["nominal_quantized_bytes"], p_nom, "{fmt} preserve nominal");
            assert_eq!(p["delta_bytes"], delta, "{fmt} preserve delta");
            assert_eq!(pf["source"]["tensors"], 5);
            assert_eq!(pf["source"]["parameters"], 2871808);
            assert_eq!(pf["source"]["payload_bytes"], 5743616);
        }
    }

    #[test]
    fn real_dense_model_preflight_matches_python_golden() {
        // Exercises the safetensors header reader + snapshot resolution against a
        // real checkpoint. Golden from scripts/two_pass_quantize.py on Qwen3.5-0.8B.
        let model = Path::new(
            "/srv/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17",
        );
        if !model.join("config.json").is_file() {
            eprintln!("skipping: golden model absent");
            return;
        }
        let calibration = json!({"metadata": {"preserve_high_precision": []}});
        for fmt in ["oq4", "oq8", "oq4.25++", "oq4.5++"] {
            let pf = pass_two_storage_preflight(
                model,
                Path::new("/tmp/x.hfq"),
                fmt,
                &calibration,
                Some(1_000_000_000_000_000),
            )
            .unwrap();
            assert_eq!(pf["source"]["tensors"], 488, "{fmt}");
            assert_eq!(pf["source"]["parameters"], 873438784u64, "{fmt}");
            let e = &pf["estimate"];
            assert_eq!(e["nominal_payload_bytes"], 928204928u64, "{fmt}");
            assert_eq!(e["mixed_payload_bytes"], 928204928u64, "{fmt}");
            assert_eq!(e["tensor_alignment_bytes"], 1998848u64, "{fmt}");
            assert_eq!(e["completed_artifact_estimate_bytes"], 946980992u64, "{fmt}");
            assert_eq!(e["required_free_bytes"], 69666457728u64, "{fmt}");
        }
    }

    #[test]
    fn source_precision_matches_python_golden() {
        let mk = |dt: &str| SafetensorTensor {
            name: "t".into(),
            dtype: dt.into(),
            shape: vec![1000],
            numel: 1000,
            source_bytes: 12345,
        };
        assert_eq!(source_precision_output_bytes(&mk("BF16")), 2000);
        assert_eq!(source_precision_output_bytes(&mk("F16")), 2000);
        assert_eq!(source_precision_output_bytes(&mk("F32")), 2000);
        assert_eq!(source_precision_output_bytes(&mk("I8")), 12345);
        assert_eq!(source_precision_output_bytes(&mk("U8")), 12345);
    }
}
