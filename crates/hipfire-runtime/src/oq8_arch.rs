// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! OQ8-family device packing: the single source of truth for expanding the
//! W8A8/W4A8 on-disk formats into the `Oq8G256` *combined* device layout the
//! grouped-WMMA / GEMV kernels read.
//!
//! Three portable on-disk formats all resolve to the same combined buffer —
//! `[int8 weights m*k][f32 group scales m*(k/256)]` — so the forward derives the
//! weight-scale pointer via `sub_offset(m*k, ..)` and dispatches the one iu8
//! W8A8 path:
//!   * `Oq8G256` (qt 35): `[f16 scale][256 int8]`/group → copy int8 + f32 scale.
//!   * OQ4→OQ8 / W4A8 (qt 33): `[f16 scale][128 int4 nibbles]`/group → sign-extend
//!     the nibbles to int8 (weight values stay 4-bit; activations gain int8).
//!   * `OqPlusCompact` (qt 36): int4 bulk + sparse int8 outliers → expand the bulk
//!     and overlay the outliers.
//!
//! These were duplicated byte-for-byte in the gemma3 and qwen35 loaders; hosting
//! them here (beside [`crate::oq4_arch`]) keeps the transform and its length
//! contracts from drifting across crates. Minimax keeps its own variant because
//! it targets an *indexed-MoE-block* layout, not this dense combined one.
//!
//! Unlike OQ4 (which has a `…ArchPacked` on-disk code uploaded verbatim), these
//! always transform at load — there is no pre-packed OQ8 quant-type yet, so
//! `hipfire optimize` cannot pre-canonicalize them. Adding one is the follow-up
//! that would make OQ8/W4A8 weights page-in as pure copies.

use crate::quant::{f16_to_f32, QuantType};

/// Sign-extend a 4-bit nibble to `i8` (levels in `[-8, 7]`).
fn sext4(nib: u8) -> i8 {
    let v = (nib & 0xf) as i8;
    if v > 7 {
        v - 16
    } else {
        v
    }
}

/// `Oq8G256` (qt 35): `[f16 scale][256 int8]` per 256-group, row-contiguous →
/// combined `[int8 m*k][f32 scales m*ng]`.
pub fn oq8_combined(data: &[u8], m: usize, k: usize) -> Vec<u8> {
    const GROUP: usize = 256;
    // Single-sourced from hipfire-quant-format: Oq8G256 on-disk block = 258.
    const BLOCK: usize = QuantType::Oq8G256.block_bytes().unwrap();
    assert_eq!(k % GROUP, 0, "Oq8G256 requires K % 256 == 0 (got K={k})");
    let ng = k / GROUP;
    let expect = m * ng * BLOCK;
    assert_eq!(
        data.len(),
        expect,
        "Oq8G256 weight byte length {} != M*ng*258 = {expect} (M={m} K={k})",
        data.len()
    );
    let mut combined = vec![0u8; m * k + m * ng * 4];
    for r in 0..m {
        for g in 0..ng {
            let src = (r * ng + g) * BLOCK;
            let dst = r * k + g * GROUP;
            combined[dst..dst + GROUP].copy_from_slice(&data[src + 2..src + BLOCK]);
            let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
            let so = m * k + (r * ng + g) * 4;
            combined[so..so + 4].copy_from_slice(&scale.to_le_bytes());
        }
    }
    combined
}

/// OQ4→OQ8 / W4A8 (qt 33): on-disk bytes are `Oq4G256` (`[f16 scale][128 int4
/// nibbles]` per 256-group). Sign-extend the nibbles into int8 and tag the result
/// `Oq8G256` so it runs the W8A8 path with int8 activations — weight values stay
/// 4-bit, activations gain int8 precision.
pub fn oq4_to_oq8_combined(data: &[u8], m: usize, k: usize) -> Vec<u8> {
    const GROUP: usize = 256;
    // Single-sourced from hipfire-quant-format: Oq4G256 on-disk block = 130.
    const BLOCK: usize = QuantType::Oq4G256.block_bytes().unwrap();
    assert_eq!(k % GROUP, 0, "OQ4->OQ8 requires K % 256 == 0 (got K={k})");
    let ng = k / GROUP;
    let expect = m * ng * BLOCK;
    assert_eq!(
        data.len(),
        expect,
        "OQ4->OQ8 weight byte length {} != M*ng*130 = {expect} (M={m} K={k})",
        data.len()
    );
    let mut combined = vec![0u8; m * k + m * ng * 4];
    for r in 0..m {
        for g in 0..ng {
            let src = (r * ng + g) * BLOCK;
            let dst = r * k + g * GROUP;
            for i in 0..128 {
                let byte = data[src + 2 + i];
                combined[dst + 2 * i] = sext4(byte & 0xf) as u8;
                combined[dst + 2 * i + 1] = sext4(byte >> 4) as u8;
            }
            let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
            let so = m * k + (r * ng + g) * 4;
            combined[so..so + 4].copy_from_slice(&scale.to_le_bytes());
        }
    }
    combined
}

/// `OqPlusCompact` (qt 36): magnitude-tiered W4A8, on-disk block `[f16 scale]
/// [128 int4 nibbles][N_out × (u8 idx, i8 val)]` = `130 + 2·N_out` bytes. `N_out`
/// is derived from the byte length (uniform per tensor). Sign-extend the int4
/// bulk into int8, then overlay the sparse int8 outliers at their in-group
/// indices → the same combined `[int8 m*k][f32 scales m*ng]` layout.
pub fn oqplus_compact_to_oq8_combined(data: &[u8], m: usize, k: usize) -> Vec<u8> {
    const GROUP: usize = 256;
    assert_eq!(k % GROUP, 0, "OQ+C requires K % 256 == 0 (got K={k})");
    let ng = k / GROUP;
    let n_groups = m * ng;
    assert!(
        n_groups > 0 && !data.is_empty() && data.len() % n_groups == 0,
        "OQ+C weight byte length {} not divisible by n_groups {n_groups} (M={m} K={k})",
        data.len()
    );
    let block_bytes = data.len() / n_groups;
    assert!(
        block_bytes >= 132 && (block_bytes - 130) % 2 == 0,
        "OQ+C block_bytes {block_bytes} invalid (expected 130 + 2·N_out)"
    );
    let n_out = (block_bytes - 130) / 2;
    let mut combined = vec![0u8; m * k + m * ng * 4];
    for r in 0..m {
        for g in 0..ng {
            let src = (r * ng + g) * block_bytes;
            let dst = r * k + g * GROUP;
            // int4 bulk → int8 (read as signed char downstream).
            for i in 0..128 {
                let byte = data[src + 2 + i];
                combined[dst + 2 * i] = sext4(byte & 0xf) as u8;
                combined[dst + 2 * i + 1] = sext4(byte >> 4) as u8;
            }
            // Overlay the sparse int8 outliers: (u8 idx, i8 val) × N_out.
            let tbl = src + 130;
            for s in 0..n_out {
                let idx = data[tbl + 2 * s] as usize;
                let val = data[tbl + 2 * s + 1];
                combined[dst + idx] = val;
            }
            let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
            let so = m * k + (r * ng + g) * 4;
            combined[so..so + 4].copy_from_slice(&scale.to_le_bytes());
        }
    }
    combined
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1.0 as f16 bits, and its f32 little-endian bytes for scale asserts.
    const F16_ONE: u16 = 0x3C00;
    fn one_le() -> [u8; 4] {
        1.0f32.to_le_bytes()
    }

    #[test]
    fn oq8_combined_copies_int8_and_splits_scale() {
        // one 256-group: [f16 1.0][256 int8 = 0,1,..,255]
        let mut data = Vec::new();
        data.extend_from_slice(&F16_ONE.to_le_bytes());
        data.extend((0..256).map(|i| i as u8));
        let out = oq8_combined(&data, 1, 256);
        assert_eq!(out.len(), 256 + 4);
        assert_eq!(&out[0..256], &data[2..258]); // int8 weights verbatim
        assert_eq!(&out[256..260], &one_le()); // group f32 scale
    }

    #[test]
    fn oq4_to_oq8_sign_extends_nibbles() {
        // nibbles: byte 0x21 -> low 1, high 2 ; byte 0xF8 -> low -8, high -1
        let mut nibbles = vec![0u8; 128];
        nibbles[0] = 0x21;
        nibbles[1] = 0xF8;
        let mut data = Vec::new();
        data.extend_from_slice(&F16_ONE.to_le_bytes());
        data.extend_from_slice(&nibbles);
        let out = oq4_to_oq8_combined(&data, 1, 256);
        assert_eq!(out.len(), 256 + 4);
        assert_eq!(out[0] as i8, 1);
        assert_eq!(out[1] as i8, 2);
        assert_eq!(out[2] as i8, -8);
        assert_eq!(out[3] as i8, -1);
        assert_eq!(&out[256..260], &one_le());
    }

    #[test]
    fn oqplus_compact_overlays_outliers() {
        // block = [f16 1.0][128 nibbles all 0x11 -> int8 1][1 outlier: idx 5 val -100]
        let n_out = 1usize;
        let mut data = Vec::new();
        data.extend_from_slice(&F16_ONE.to_le_bytes());
        data.extend(std::iter::repeat(0x11u8).take(128)); // every int8 -> 1
        data.push(5u8); // outlier index
        data.push((-100i8) as u8); // outlier value
        assert_eq!(data.len(), 130 + 2 * n_out);
        let out = oqplus_compact_to_oq8_combined(&data, 1, 256);
        assert_eq!(out.len(), 256 + 4);
        assert_eq!(out[0] as i8, 1); // bulk
        assert_eq!(out[5] as i8, -100); // outlier overlaid
        assert_eq!(out[6] as i8, 1); // still bulk
        assert_eq!(&out[256..260], &one_le());
    }
}
