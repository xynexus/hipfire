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
use hipfire_rdna::DType;

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

/// `Oq3G256` (qt 38): symmetric W3, on-disk block `[f16 scale][8 × 3 u32
/// bit-planes]` = 98 B/group (3.0625 b/w). The 256 weights of a group are stored
/// as 8 sub-blocks of 32, each sub-block holding bit 0, bit 1 and bit 2 of its
/// 32 values as three separate little-endian u32 words — so weight `i` of
/// sub-block `s` is assembled bit-by-bit rather than read as a field.
///
/// Sign-extending int3 into an int8 container is EXACT (values live in [-4, 3])
/// and the f16 group scale carries over unchanged, so this upcast is lossless:
/// the served weights are bit-identical to what a native W3 decode would
/// produce. That is what lets 3-bit share the iu8 W8A8 kernels with oq4/oq8
/// instead of needing a dedicated W3 GEMV — the same trade `expand_oq2_to_oq8`
/// already makes for W2. Runtime VRAM is int8; the 3-bit win is on disk and on
/// the DMA path that reads it.
pub fn oq3_to_oq8_combined(data: &[u8], m: usize, k: usize) -> Vec<u8> {
    const GROUP: usize = 256;
    const BLOCK: usize = 98; // 2 (f16 scale) + 8 × 12 (three u32 bit-planes)
    assert_eq!(k % GROUP, 0, "OQ3->OQ8 requires K % 256 == 0 (got K={k})");
    let ng = k / GROUP;
    let expect = m * ng * BLOCK;
    assert_eq!(
        data.len(),
        expect,
        "OQ3->OQ8 weight byte length {} != M*ng*98 = {expect} (M={m} K={k})",
        data.len()
    );
    let mut combined = vec![0u8; m * k + m * ng * 4];
    for r in 0..m {
        for g in 0..ng {
            let src = (r * ng + g) * BLOCK;
            let dst = r * k + g * GROUP;
            for s in 0..8 {
                let bo = src + 2 + s * 12;
                let w = |o: usize| {
                    u32::from_le_bytes([
                        data[bo + o],
                        data[bo + o + 1],
                        data[bo + o + 2],
                        data[bo + o + 3],
                    ])
                };
                let (p0, p1, p2) = (w(0), w(4), w(8));
                for i in 0..32 {
                    let v = ((p0 >> i) & 1) | (((p1 >> i) & 1) << 1) | (((p2 >> i) & 1) << 2);
                    // 3-bit two's complement: codes 4..7 are -4..-1.
                    let signed = if v > 3 { v as i32 - 8 } else { v as i32 };
                    combined[dst + s * 32 + i] = signed as i8 as u8;
                }
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

/// Load-time dispatch for the OQ int8-activation family: expand the on-disk
/// W8A8/W4A8 codes into the combined `Oq8G256` device buffer. This is the
/// single arch-agnostic entry point every per-arch weight loader should call for
/// these codes — the OQ8 analog of [`crate::oq4_arch::oq4_arch_load`] for the
/// W4A4 family. It exists because the SAME 33/35/36 dispatch was open-coded (and
/// forgotten) in loader after loader (qwen2, the shared llama `load_weights_hfq`,
/// nemotron), each panicking on qt 35 until fixed one at a time; routing every
/// loader through here means a new family gets OQ8/OQ+ for free.
///
///   * qt 35 (`Oq8G256`)     — W8A8, int8 weights + int8 acts.
///   * qt 33 (`OqPlusG256`)  — W4A8, int4 weights sign-extended to int8.
///   * qt 36 (`OqPlusCompact`) — mixed W4A8, int4 bulk + int8 outliers.
///   * qt 38 (`Oq3G256`)     — W3, bit-planed int3 sign-extended to int8.
///
/// Returns `None` for any other code so the caller falls through to its own arms
/// (OQ4 via `oq4_arch_load`, plain dtypes, etc.). All three resolve to
/// `DType::Oq8G256`, dispatched by the generic iu8 GEMV/GEMM.
/// One dimension filter: true unless `var` holds a non-empty comma-separated
/// list of values that does not contain `value`. Unparseable entries are ignored
/// rather than fatal — this is a debugging handle, not a correctness gate, and
/// an empty or garbage list therefore means "no filter".
fn dim_selected(var: &hipfire_env::EnvVar, value: usize) -> bool {
    let Some(raw) = var.get() else {
        return true;
    };
    let mut any = false;
    for tok in raw.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        any = true;
        if tok.parse::<usize>() == Ok(value) {
            return true;
        }
    }
    !any
}

/// Diagnostic bisection filter for compact residency. Both
/// `HIPFIRE_OQ_COMPACT_RESIDENT_ONLY_K` and `..._ONLY_M` must admit the tensor,
/// so setting both narrows to a single (M, K) projection class — which is as
/// fine-grained as this hook can get, since it is handed the shape but never the
/// tensor name.
fn compact_shape_selected(m: usize, k: usize) -> bool {
    dim_selected(&hipfire_env::OQ_COMPACT_RESIDENT_ONLY_K, k)
        && dim_selected(&hipfire_env::OQ_COMPACT_RESIDENT_ONLY_M, m)
}

pub fn oq8_arch_load(qt: u8, data: &[u8], m: usize, k: usize) -> Option<(Vec<u8>, DType)> {
    // Compact residency: hand the OqPlusCompact blocks to the device untouched
    // so oq4.25++ stays ~4.25 bits/weight instead of being unpacked to one int8
    // per weight here. `gemm_oq_compact_grouped_wmma` decodes the nibbles and
    // applies the sparse overlay per tile, bit-identically to this expansion
    // (see hipfire-rdna examples/parity_gemm_oq_compact.rs).
    //
    // Opt-in while the end-to-end path is validated; once every consumer of
    // DType::OqCompactG256 is wired this becomes the default and the expansion
    // below can go — along with its two siblings in lfm2moe and minimax.
    //
    // HIPFIRE_OQ_COMPACT_RESIDENT_ONLY_K / _ONLY_M narrow this to chosen K and M
    // values so the compact-vs-expanded logit divergence can be bisected down to
    // a single (M, K) projection class. Purely diagnostic: unset (the normal
    // case) keeps every OqPlusCompact tensor compact, exactly as before. The
    // shape is the handle because this hook never sees the tensor name.
    if qt == QuantType::OqPlusCompact.code()
        && hipfire_env::OQ_COMPACT_RESIDENT.flag()
        && compact_shape_selected(m, k)
    {
        return Some((data.to_vec(), DType::OqCompactG256));
    }
    let bytes = match qt {
        c if c == QuantType::Oq8G256.code() => oq8_combined(data, m, k),
        c if c == QuantType::OqPlusG256.code() => oq4_to_oq8_combined(data, m, k),
        c if c == QuantType::OqPlusCompact.code() => oqplus_compact_to_oq8_combined(data, m, k),
        c if c == QuantType::Oq3G256.code() => oq3_to_oq8_combined(data, m, k),
        _ => return None,
    };
    Some((bytes, DType::Oq8G256))
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
    fn oq3_upcast_recovers_every_int3_code_losslessly() {
        // Pack one 256-group by hand in the on-disk bit-plane layout, using a
        // pattern that visits all 8 codes including the negative half, then check
        // the upcast reproduces them exactly. Sign extension is the whole point:
        // codes 4..7 must come back as -4..-1, not 4..7.
        let codes: Vec<i32> = (0..256).map(|i| i % 8).collect();
        let mut data = Vec::from(F16_ONE.to_le_bytes());
        for s in 0..8 {
            let (mut p0, mut p1, mut p2) = (0u32, 0u32, 0u32);
            for i in 0..32 {
                let v = codes[s * 32 + i] as u32;
                p0 |= (v & 1) << i;
                p1 |= ((v >> 1) & 1) << i;
                p2 |= ((v >> 2) & 1) << i;
            }
            for w in [p0, p1, p2] {
                data.extend_from_slice(&w.to_le_bytes());
            }
        }
        assert_eq!(data.len(), 98, "on-disk Oq3G256 block is 98 B");

        let out = oq3_to_oq8_combined(&data, 1, 256);
        assert_eq!(out.len(), 256 + 4);
        for i in 0..256 {
            let raw = codes[i];
            let want = if raw > 3 { raw - 8 } else { raw };
            assert_eq!(out[i] as i8 as i32, want, "code {raw} at {i}");
        }
        assert_eq!(&out[256..260], &one_le());

        // The dispatcher must route qt 38 here rather than returning None.
        let (bytes, dtype) = oq8_arch_load(QuantType::Oq3G256.code(), &data, 1, 256)
            .expect("qt 38 dispatches to the oq3 upcast");
        assert_eq!(dtype, DType::Oq8G256);
        assert_eq!(bytes, out);
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
    fn oq8_arch_load_dispatches_family_and_rejects_others() {
        // qt 35 (Oq8G256): routes to oq8_combined, tagged Oq8G256.
        let mut data = Vec::new();
        data.extend_from_slice(&F16_ONE.to_le_bytes());
        data.extend((0..256).map(|i| i as u8));
        let (bytes, dt) = oq8_arch_load(QuantType::Oq8G256.code(), &data, 1, 256)
            .expect("qt 35 is an OQ8-family code");
        assert_eq!(dt, DType::Oq8G256);
        assert_eq!(bytes, oq8_combined(&data, 1, 256));
        // A non-OQ8-family code (13 = MQ4G256) falls through so callers try their
        // own arms.
        assert!(oq8_arch_load(QuantType::MQ4G256.code(), &data, 1, 256).is_none());
        // qt 43 is the NPU-only ragged OQ8 layout. GPU loaders must not treat it
        // as the dense combined Oq8G256 layout.
        assert!(oq8_arch_load(QuantType::Oq8G256RowPadded.code(), &data, 1, 256).is_none());
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
