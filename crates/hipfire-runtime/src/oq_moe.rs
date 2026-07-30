// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Family-neutral Opus-Quant routed-expert storage transforms.
//!
//! Portable HFQ OQ tensors use f16 scales.  The indexed routed-expert kernels
//! consume one independently addressable expert whose blocks carry f32 scales.
//! Keeping that conversion here gives eager architecture loaders and the
//! runtime weight pager one producer/consumer contract.

use hipfire_primitives::conv::f16_to_f32;
use hipfire_quant_format::QuantType;

pub const OQ4_MOE_BLOCK_BYTES: usize = 132;
pub const OQ8_MOE_BLOCK_BYTES: usize = 260;
pub const OQ8_CANONICAL_QT: u8 = QuantType::Oq8G256.code();
pub const OQPLUS_COMPACT_QT: u8 = QuantType::OqPlusCompact.code();
const GROUP_SIZE: usize = 256;

/// Whether routed OQ experts are loaded in the repacked indexed MoE block
/// layout (f32 scale) rather than the canonical HFQ layout (f16 scale).
///
/// `load_moe_expert` only repacks under `HIPFIRE_QWEN35_MOE_OQ_INDEXED=1`; every
/// consumer that strides an expert block must agree with it, so both read this.
pub fn moe_expert_blocks_repacked() -> bool {
    std::env::var("HIPFIRE_QWEN35_MOE_OQ_INDEXED")
        .ok()
        .as_deref()
        == Some("1")
}

fn checked_group_count(m: usize, k: usize, format: &str) -> Result<usize, String> {
    if k % GROUP_SIZE != 0 {
        return Err(format!(
            "{format} routed expert requires K % {GROUP_SIZE} == 0 (got K={k})"
        ));
    }
    m.checked_mul(k / GROUP_SIZE)
        .ok_or_else(|| format!("{format} routed expert group count overflow (M={m} K={k})"))
}

pub fn oq4_moe_packed_len(m: usize, k: usize) -> Result<usize, String> {
    checked_group_count(m, k, "OQ4")?
        .checked_mul(OQ4_MOE_BLOCK_BYTES)
        .ok_or_else(|| format!("OQ4 routed expert byte length overflow (M={m} K={k})"))
}

pub fn oq8_moe_packed_len(m: usize, k: usize) -> Result<usize, String> {
    checked_group_count(m, k, "OQ8")?
        .checked_mul(OQ8_MOE_BLOCK_BYTES)
        .ok_or_else(|| format!("OQ8 routed expert byte length overflow (M={m} K={k})"))
}

/// Convert canonical OQ4 `[f16 scale | 128 packed signed-int4]` blocks into
/// indexed routed-expert `[f32 scale | 128 packed signed-int4]` blocks.
pub fn oq4_canonical_to_moe_blocks(data: &[u8], m: usize, k: usize) -> Result<Vec<u8>, String> {
    const SRC_BLOCK_BYTES: usize = 130;
    let groups = checked_group_count(m, k, "OQ4")?;
    let expected = groups
        .checked_mul(SRC_BLOCK_BYTES)
        .ok_or_else(|| format!("OQ4 canonical byte length overflow (M={m} K={k})"))?;
    if data.len() != expected {
        return Err(format!(
            "OQ4 routed expert byte length {} != M*(K/256)*130 = {expected} (M={m} K={k})",
            data.len()
        ));
    }

    let mut out = vec![0u8; oq4_moe_packed_len(m, k)?];
    for block in 0..groups {
        let src = block * SRC_BLOCK_BYTES;
        let dst = block * OQ4_MOE_BLOCK_BYTES;
        let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
        out[dst..dst + 4].copy_from_slice(&scale.to_le_bytes());
        out[dst + 4..dst + OQ4_MOE_BLOCK_BYTES]
            .copy_from_slice(&data[src + 2..src + SRC_BLOCK_BYTES]);
    }
    Ok(out)
}

/// Convert canonical OQ8 `[f16 scale | 256 signed-int8]` blocks into indexed
/// routed-expert `[f32 scale | 256 signed-int8]` blocks.
pub fn oq8_canonical_to_moe_blocks(data: &[u8], m: usize, k: usize) -> Result<Vec<u8>, String> {
    const SRC_BLOCK_BYTES: usize = 258;
    let groups = checked_group_count(m, k, "OQ8")?;
    let expected = groups
        .checked_mul(SRC_BLOCK_BYTES)
        .ok_or_else(|| format!("OQ8 canonical byte length overflow (M={m} K={k})"))?;
    if data.len() != expected {
        return Err(format!(
            "OQ8 routed expert byte length {} != M*(K/256)*258 = {expected} (M={m} K={k})",
            data.len()
        ));
    }

    let mut out = vec![0u8; oq8_moe_packed_len(m, k)?];
    for block in 0..groups {
        let src = block * SRC_BLOCK_BYTES;
        let dst = block * OQ8_MOE_BLOCK_BYTES;
        let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
        out[dst..dst + 4].copy_from_slice(&scale.to_le_bytes());
        out[dst + 4..dst + OQ8_MOE_BLOCK_BYTES]
            .copy_from_slice(&data[src + 2..src + SRC_BLOCK_BYTES]);
    }
    Ok(out)
}

fn sign_extend_i4(nibble: u8) -> i8 {
    let value = (nibble & 0x0f) as i8;
    if value > 7 {
        value - 16
    } else {
        value
    }
}

/// Expand compact OQ+ `[f16 scale | 128 int4 | outlier pairs]` blocks into
/// the indexed OQ8 routed-expert layout.
pub fn oqplus_compact_to_moe_oq8_blocks(
    data: &[u8],
    m: usize,
    k: usize,
) -> Result<Vec<u8>, String> {
    let groups = checked_group_count(m, k, "OQ+C")?;
    if groups == 0 || data.is_empty() || data.len() % groups != 0 {
        return Err(format!(
            "OQ+C routed expert byte length {} is not divisible by {groups} groups (M={m} K={k})",
            data.len()
        ));
    }
    let block_bytes = data.len() / groups;
    if block_bytes < 132 || (block_bytes - 130) % 2 != 0 {
        return Err(format!(
            "OQ+C routed expert block size {block_bytes} is invalid (expected 130 + 2*N_out)"
        ));
    }
    let outliers = (block_bytes - 130) / 2;
    let mut out = vec![0u8; oq8_moe_packed_len(m, k)?];
    for block in 0..groups {
        let src = block * block_bytes;
        let dst = block * OQ8_MOE_BLOCK_BYTES;
        let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
        out[dst..dst + 4].copy_from_slice(&scale.to_le_bytes());
        for i in 0..128 {
            let byte = data[src + 2 + i];
            out[dst + 4 + 2 * i] = sign_extend_i4(byte & 0x0f) as u8;
            out[dst + 4 + 2 * i + 1] = sign_extend_i4(byte >> 4) as u8;
        }
        let table = src + 130;
        for outlier in 0..outliers {
            let index = data[table + 2 * outlier] as usize;
            let value = data[table + 2 * outlier + 1];
            out[dst + 4 + index] = value;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oq4_canonical_repack_expands_scale_without_changing_codes() {
        let mut canonical = vec![0u8; 130];
        canonical[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        for (index, byte) in canonical[2..].iter_mut().enumerate() {
            *byte = index as u8;
        }
        let packed = oq4_canonical_to_moe_blocks(&canonical, 1, 256).unwrap();
        assert_eq!(packed.len(), OQ4_MOE_BLOCK_BYTES);
        assert_eq!(&packed[..4], &1.0f32.to_le_bytes());
        assert_eq!(&packed[4..], &canonical[2..]);
    }

    #[test]
    fn oq8_canonical_repack_expands_scale_without_changing_codes() {
        let mut canonical = vec![0u8; 258];
        canonical[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
        for (index, byte) in canonical[2..].iter_mut().enumerate() {
            *byte = index as u8;
        }
        let packed = oq8_canonical_to_moe_blocks(&canonical, 1, 256).unwrap();
        assert_eq!(packed.len(), OQ8_MOE_BLOCK_BYTES);
        assert_eq!(&packed[..4], &0.5f32.to_le_bytes());
        assert_eq!(&packed[4..], &canonical[2..]);
    }

    #[test]
    fn rejects_non_group_aligned_shapes_and_wrong_lengths() {
        assert!(oq4_canonical_to_moe_blocks(&[], 1, 255).is_err());
        assert!(oq4_canonical_to_moe_blocks(&[0; 129], 1, 256).is_err());
        assert!(oq8_canonical_to_moe_blocks(&[0; 257], 1, 256).is_err());
    }
}
