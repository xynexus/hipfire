// SPDX-License-Identifier: Apache-2.0
//! Minimal `.hfq` (HFQM container) reader + in-place norm patcher for Path-A
//! export. We don't re-serialize the container — recovery only changes the fp
//! RMSNorm weights, which are stored BF16 at fixed offsets/sizes, so we overwrite
//! those bytes in place (same size) and leave codes/index/header untouched.
//!
//! Format (from hipfire-quantize write_hfq / its reader):
//!   [0:4]   "HFQM"      [4:8] version u32   [8:12] arch u32
//!   [12:16] n_tensors u32  [16:24] metadata_offset u64  [24:32] data_offset u64
//!   metadata: brace-matched JSON at metadata_offset
//!   index @ (metadata_offset + json_end): u32 count, then per tensor:
//!     u16 name_len, name, u8 quant_type, u8 n_dims, n_dims×u32 shape,
//!     u32 group_size, u64 data_size, and in **v2** a u64 `data_offset / 32`
//!   data @ data_offset (4096-aligned). In v1 tensors are concatenated in index
//!   order, so an entry's offset is the running sum; v2 stores each offset
//!   explicitly and does not require them to be contiguous.
//!
//! The version field used to be read and discarded, and every entry parsed as
//! v1. `hipfire-quantize` has emitted v2 since `HFQ_VERSION = 2`, so this
//! misread the 8 explicit-offset bytes as the next entry's header — on a real
//! `gemma-4-E2B-it--bf16.hfq` that surfaced as "invalid utf-8 sequence" from a
//! garbage name. Every offset after the first entry was wrong, and
//! `patch_norms_inplace` writes at those offsets.

use std::collections::HashMap;

const HFQ_MAGIC: &[u8; 4] = b"HFQM";
/// Highest container version this parser knows how to walk. Kept in step with
/// `hipfire_quantize::hfq_out::HFQ_VERSION` — a newer file is refused rather
/// than parsed with stale layout assumptions.
const HFQ_MAX_VERSION: u32 = 2;
/// v2 stores each entry's data offset divided by this.
const HFQM_V2_OFFSET_ALIGN: usize = 32;
const QT_BF16: u8 = 16; // QuantType::BF16 (=16; norms + down_proj are stored BF16)

#[derive(Debug, Clone)]
pub struct HfqEntry {
    pub name: String,
    pub quant_type: u8,
    pub shape: Vec<u32>,
    pub data_offset: usize, // absolute byte offset into the file
    pub data_size: usize,
}

/// Parse the HFQM header + index. Returns the tensor entries (with absolute data
/// offsets) and the metadata JSON string.
pub fn parse_hfq(bytes: &[u8]) -> Result<(Vec<HfqEntry>, String), String> {
    if bytes.len() < 32 || &bytes[0..4] != HFQ_MAGIC {
        return Err("not an HFQM container".into());
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version == 0 || version > HFQ_MAX_VERSION {
        return Err(format!(
            "HFQM version {version} is newer than this parser understands (max \
             {HFQ_MAX_VERSION}); refusing rather than misreading the index"
        ));
    }
    let n_tensors = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let metadata_offset = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    let data_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;

    // brace-match the metadata JSON. Bounds-checked: this returns a Result, so a
    // truncated container must be an Err, not a slice panic. A 4096-byte stub in
    // the model store used to panic here.
    if metadata_offset > data_offset || data_offset > bytes.len() {
        return Err(format!(
            "HFQM header out of range: metadata_offset {metadata_offset}, \
             data_offset {data_offset}, file {} bytes",
            bytes.len()
        ));
    }
    let meta = &bytes[metadata_offset..data_offset];
    let (mut depth, mut in_str, mut esc, mut json_end) = (0i32, false, false, None);
    for (i, &b) in meta.iter().enumerate() {
        if esc {
            esc = false;
            continue;
        }
        if b == b'\\' && in_str {
            esc = true;
            continue;
        }
        if b == b'"' {
            in_str = !in_str;
            continue;
        }
        if !in_str {
            if b == b'{' {
                depth += 1;
            } else if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    json_end = Some(i + 1);
                    break;
                }
            }
        }
    }
    let json_end = json_end.ok_or("HFQM metadata JSON did not end")?;
    let metadata_json = String::from_utf8(meta[..json_end].to_vec()).map_err(|e| e.to_string())?;

    let mut pos = metadata_offset + json_end;
    // Every read below goes through `take`, which bounds-checks against the end
    // of the index region rather than trusting a length from the file.
    let index_end = data_offset;
    let take = |pos: &mut usize, n: usize| -> Result<&[u8], String> {
        let end = pos.checked_add(n).ok_or("HFQM index offset overflow")?;
        if end > index_end || end > bytes.len() {
            return Err("HFQM index truncated".to_string());
        }
        let slice = &bytes[*pos..end];
        *pos = end;
        Ok(slice)
    };

    let idx_n = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap()) as usize;
    if idx_n != n_tensors {
        return Err(format!("index count {idx_n} != header count {n_tensors}"));
    }

    let mut entries = Vec::with_capacity(n_tensors);
    let mut cum = data_offset;
    for _ in 0..n_tensors {
        let name_len = u16::from_le_bytes(take(&mut pos, 2)?.try_into().unwrap()) as usize;
        let name =
            String::from_utf8(take(&mut pos, name_len)?.to_vec()).map_err(|e| e.to_string())?;
        let quant_type = take(&mut pos, 1)?[0];
        let n_dims = take(&mut pos, 1)?[0] as usize;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap()));
        }
        let _group_size = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap());
        let data_size = u64::from_le_bytes(take(&mut pos, 8)?.try_into().unwrap()) as usize;
        // v1 packs tensors back to back, so an entry's offset is the running
        // sum. v2 stores it explicitly, in 32-byte units, and does not promise
        // contiguity — deriving it from the running sum there is exactly the
        // misread this parser used to make.
        let entry_offset = if version >= 2 {
            let div32 = u64::from_le_bytes(take(&mut pos, 8)?.try_into().unwrap()) as usize;
            div32
                .checked_mul(HFQM_V2_OFFSET_ALIGN)
                .ok_or("HFQM v2 offset overflow")?
        } else {
            cum
        };
        entries.push(HfqEntry {
            name,
            quant_type,
            shape,
            data_offset: entry_offset,
            data_size,
        });
        cum += data_size;
    }
    Ok((entries, metadata_json))
}

/// Is this tensor one of the RMSNorm weights recovery tunes?
pub fn is_norm(name: &str) -> bool {
    name.ends_with(".input_layernorm.weight")
        || name.ends_with(".post_attention_layernorm.weight")
        || name == "model.norm.weight"
}

pub fn f32_to_bf16_bits(f: f32) -> u16 {
    hipfire_primitives::conv::f32_to_bf16_bits(f)
}

pub fn bf16_bits_to_f32(b: u16) -> f32 {
    hipfire_primitives::conv::bf16_bits_to_f32(b)
}

/// Patch a parsed HFQM byte buffer in place: overwrite each BF16 norm tensor
/// named in `tuned` (name → fp32 weights) with its tuned values. Same byte size,
/// so offsets/index/codes are untouched. Returns the number of tensors patched.
pub fn patch_norms_inplace(
    bytes: &mut [u8],
    entries: &[HfqEntry],
    tuned: &HashMap<String, Vec<f32>>,
) -> Result<usize, String> {
    let mut n = 0;
    for e in entries {
        let Some(vals) = tuned.get(&e.name) else {
            continue;
        };
        if e.quant_type != QT_BF16 {
            return Err(format!(
                "{}: expected BF16 norm (qt {}), refusing",
                e.name, e.quant_type
            ));
        }
        if vals.len() * 2 != e.data_size {
            return Err(format!(
                "{}: tuned len {} (×2={}) != data_size {}",
                e.name,
                vals.len(),
                vals.len() * 2,
                e.data_size
            ));
        }
        for (i, &v) in vals.iter().enumerate() {
            let off = e.data_offset + i * 2;
            bytes[off..off + 2].copy_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
        }
        n += 1;
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a container holding two BF16 tensors, in v1 or v2 encoding.
    ///
    /// The two encodings describe the SAME logical content, which is the point:
    /// `parse_hfq` must return identical entries for both. That equality is
    /// exactly what broke when the version field was read and discarded.
    fn container(version: u32) -> Vec<u8> {
        let meta = br#"{"arch":"test"}"#;
        let sizes = [4usize, 6];
        // v2 aligns each tensor to 32 bytes; v1 packs them back to back.
        let data_offset = 4096usize;
        let offsets: Vec<usize> = if version >= 2 {
            vec![data_offset, data_offset + 32]
        } else {
            vec![data_offset, data_offset + sizes[0]]
        };

        let mut index = Vec::new();
        index.extend_from_slice(&2u32.to_le_bytes());
        for (i, name) in ["model.norm.weight", "model.layers.0.input_layernorm.weight"]
            .iter()
            .enumerate()
        {
            index.extend_from_slice(&(name.len() as u16).to_le_bytes());
            index.extend_from_slice(name.as_bytes());
            index.push(QT_BF16);
            index.push(1u8);
            index.extend_from_slice(&((sizes[i] / 2) as u32).to_le_bytes());
            index.extend_from_slice(&0u32.to_le_bytes()); // group_size
            index.extend_from_slice(&(sizes[i] as u64).to_le_bytes());
            if version >= 2 {
                let div32 = (offsets[i] / HFQM_V2_OFFSET_ALIGN) as u64;
                index.extend_from_slice(&div32.to_le_bytes());
            }
        }

        let mut out = Vec::new();
        out.extend_from_slice(HFQ_MAGIC);
        out.extend_from_slice(&version.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // arch
        out.extend_from_slice(&2u32.to_le_bytes()); // n_tensors
        out.extend_from_slice(&32u64.to_le_bytes()); // metadata_offset
        out.extend_from_slice(&(data_offset as u64).to_le_bytes());
        out.extend_from_slice(meta);
        out.extend_from_slice(&index);
        out.resize(offsets[1] + sizes[1], 0);
        out
    }

    #[test]
    fn v1_and_v2_containers_parse_to_the_same_entries() {
        let (v1, _) = parse_hfq(&container(1)).expect("v1 parses");
        let (v2, _) = parse_hfq(&container(2)).expect("v2 parses");

        assert_eq!(v1.len(), 2);
        assert_eq!(v2.len(), 2);
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.quant_type, b.quant_type);
            assert_eq!(a.shape, b.shape);
            assert_eq!(a.data_size, b.data_size);
        }
        // The offsets differ by construction (v2 pads to 32), so check each
        // against its own encoding rather than against the other.
        assert_eq!(v1[0].data_offset, 4096);
        assert_eq!(v1[1].data_offset, 4100);
        assert_eq!(v2[0].data_offset, 4096);
        assert_eq!(
            v2[1].data_offset, 4128,
            "a v2 offset must come from the entry, not from a running sum"
        );
    }

    #[test]
    fn a_newer_version_is_refused_not_guessed_at() {
        let mut bytes = container(2);
        bytes[4..8].copy_from_slice(&3u32.to_le_bytes());
        let err = parse_hfq(&bytes).expect_err("v3 must be refused");
        assert!(err.contains("newer than this parser"), "{err}");
    }

    #[test]
    fn a_truncated_container_errors_rather_than_panicking() {
        let full = container(2);
        // Header intact, everything after it gone — the shape of the 4096-byte
        // stub in the model store that used to panic on a raw slice.
        let mut stub = full[..32].to_vec();
        stub.resize(4096, 0);
        assert!(parse_hfq(&stub).is_err());
        // And a cut mid-index.
        assert!(parse_hfq(&full[..40]).is_err());
    }
}
