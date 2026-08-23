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
/// Prepare `OqPlusCompact` blocks for the kernels, in place: resolve duplicate
/// overlay indices, and zero the bulk nibble under every overlay index.
///
/// An overlay REPLACES the bulk nibble, so a kernel that applies it as a
/// correction has to add `(val - bulk)·x[idx]`. That `bulk` term carries two
/// costs, and both sit on the critical path of the compact decode GEMV:
///
/// - `bulk` lives in the registers of whichever lane owns that position, so only
///   that lane can apply the correction. Hence a divergent ownership test that
///   every lane pays for every entry, and a serial scan of the table whose
///   dependent `X[idx]` load cannot be hoisted.
/// - a repeated index means the LAST entry wins, which the kernel used to honour
///   with an inner `O(N_out^2)` rescan.
///
/// Zeroing the bulk under each overlay index makes the correction exactly
/// `val·x[idx]`, which depends on no lane's registers. Lane `e` can then apply
/// entry `e` in parallel and let the existing wave reduction sum the results.
/// Measured on `down [5120, 17408]` at the shipped N_out=3, ablating the overlay
/// loop was worth 25% of achieved bandwidth (178.3 -> 222.1 GB/s), so this is
/// the binding cost, not the 6% an N_out sweep alone suggests — every N_out,
/// including 1, pays the same fixed loop.
///
/// Transparent to every consumer: a kernel computing `val - bulk` now computes
/// `val - 0`, and one that replaces the nibble outright (the compact GEMM, and
/// the `oqplus_compact_to_oq8_combined*` expansion) overwrites the zero with the
/// value it always wrote. Duplicates are neutralised by zeroing the LOSER's
/// value, which is a no-op correction and still decodes last-wins.
///
/// The table keeps its declared size, so `block_stride` and every offset derived
/// from it are untouched. Idempotent — zeroing an already-zero nibble is a no-op.
pub fn normalize_compact_overlays(data: &mut [u8], m: usize, k: usize, group: usize) {
    let ng = k / group;
    let n_groups = m * ng;
    if n_groups == 0 || data.is_empty() || data.len() % n_groups != 0 {
        return;
    }
    let block_bytes = data.len() / n_groups;
    let header = 2 + group / 2;
    if block_bytes < header + 2 || (block_bytes - header) % 2 != 0 {
        return;
    }
    let n_out = (block_bytes - header) / 2;
    for b in 0..n_groups {
        let base = b * block_bytes;
        let tbl = base + header;
        for e in 0..n_out {
            let idx = data[tbl + 2 * e] as usize;
            if idx >= group {
                continue; // out-of-range index: leave the block alone, decode ignores it
            }
            // Superseded by any LATER entry on the same index? Then its
            // correction must vanish; the winner still lands on a zeroed bulk.
            if (e + 1..n_out).any(|e2| data[tbl + 2 * e2] as usize == idx) {
                data[tbl + 2 * e + 1] = 0;
            }
            // Zero the bulk nibble this entry sits on, so `val - bulk == val`.
            let byte = &mut data[base + 2 + idx / 2];
            *byte &= if idx % 2 == 0 { 0xf0 } else { 0x0f };
        }
    }
}

/// Re-lay-out `OqPlusCompact` blocks for the device: ALL nibble groups first,
/// then all `[f16 scale][overlay table]` side records. Same bytes, same order,
/// same 4.25 bits — only where they sit changes.
///
/// WHY: the interleaved block is `130 + 2*N_out` = 136 bytes at the shipped
/// N_out=3, which is not a power of two, so a row of `ng` blocks is 2720 bytes
/// and only every fourth row starts on a 128-byte line. A synthetic sweep over
/// the decode GEMV's exact access pattern (`hipfire-rdna` example
/// `bench_oq_layout`, M=69632 K=5120, ~4x the 32 MiB MALL) prices that:
///
///     flat dwordx4 stream (upper bound)      243.1 GB/s   100.0%
///     interleaved 136 + side reads           226.4         93.1%
///     interleaved 136, nibbles only          229.3         94.3%
///     128-byte stride, same instructions     235.6         96.9%
///     SPLIT PLANES                           234.7         96.6%
///
/// The nibble plane makes every row start `ng * (group/2)` bytes in, a multiple
/// of 128 at G=256, so every row is aligned. Splitting per ROW instead was tried
/// and does NOT work — 229.3 GB/s, no better than interleaved — because the row
/// stride is still 2720. It is row-start ALIGNMENT that pays, not within-row
/// contiguity, and only a tensor-wide split delivers it.
///
/// One allocation, not two: the side base is `m * ng * (group/2)`, derivable
/// from values every kernel already has, so no dispatch signature changes.
///
/// Run this AFTER [`normalize_compact_overlays`] — that one reads the
/// interleaved form.
pub fn split_compact_planes(data: &[u8], m: usize, k: usize, group: usize) -> Vec<u8> {
    let ng = k / group;
    let n_groups = m * ng;
    let nib = group / 2;
    if n_groups == 0 || data.is_empty() || data.len() % n_groups != 0 {
        return data.to_vec();
    }
    let block_bytes = data.len() / n_groups;
    let header = 2 + nib;
    if block_bytes < header + 2 || (block_bytes - header) % 2 != 0 {
        return data.to_vec();
    }
    let side = block_bytes - nib; // f16 scale + 2*N_out table bytes
    let mut out = vec![0u8; data.len()];
    let side_base = n_groups * nib;
    for b in 0..n_groups {
        let src = b * block_bytes;
        out[b * nib..(b + 1) * nib].copy_from_slice(&data[src + 2..src + 2 + nib]);
        let d = side_base + b * side;
        // scale, then the overlay table, contiguous.
        out[d..d + 2].copy_from_slice(&data[src..src + 2]);
        out[d + 2..d + side].copy_from_slice(&data[src + header..src + block_bytes]);
    }
    out
}

/// Inverse of [`split_compact_planes`]: rebuild the interleaved on-disk block
/// order from the device's split planes. Anything that DOWNLOADS a resident
/// compact weight and wants to decode it (the two-stage lm_head's coarse tier
/// builder is the live caller) has to come back through here first, because
/// `oqplus_compact_to_oq8_combined_g` reads the interleaved form.
pub fn unsplit_compact_planes(data: &[u8], m: usize, k: usize, group: usize) -> Vec<u8> {
    let ng = k / group;
    let n_groups = m * ng;
    let nib = group / 2;
    if n_groups == 0 || data.is_empty() || data.len() % n_groups != 0 {
        return data.to_vec();
    }
    let block_bytes = data.len() / n_groups;
    if block_bytes < nib + 4 {
        return data.to_vec();
    }
    let side = block_bytes - nib;
    let side_base = n_groups * nib;
    let mut out = vec![0u8; data.len()];
    for b in 0..n_groups {
        let dst = b * block_bytes;
        let sb = side_base + b * side;
        out[dst..dst + 2].copy_from_slice(&data[sb..sb + 2]);
        out[dst + 2..dst + 2 + nib].copy_from_slice(&data[b * nib..(b + 1) * nib]);
        out[dst + 2 + nib..dst + block_bytes].copy_from_slice(&data[sb + 2..sb + side]);
    }
    out
}

pub fn oqplus_compact_to_oq8_combined(data: &[u8], m: usize, k: usize) -> Vec<u8> {
    oqplus_compact_to_oq8_combined_g(data, m, k, 256)
}

/// Group-generic form of [`oqplus_compact_to_oq8_combined`]. The block header is
/// `2 + group/2` bytes (f16 scale + packed nibbles), so G=256 gives the familiar
/// 130 and G=128 gives 66; `N_out` is still inferred from what is left over.
/// G=128 exists to fit models whose K is not a multiple of 256 — see
/// docs/experiments/2026-08-06-oq-compact-group-size.md.
pub fn oqplus_compact_to_oq8_combined_g(data: &[u8], m: usize, k: usize, group: usize) -> Vec<u8> {
    #[allow(non_snake_case)]
    let GROUP: usize = group;
    assert_eq!(
        k % GROUP,
        0,
        "OQ+C requires K % group == 0 (got K={k} group={GROUP})"
    );
    let ng = k / GROUP;
    let n_groups = m * ng;
    assert!(
        n_groups > 0 && !data.is_empty() && data.len() % n_groups == 0,
        "OQ+C weight byte length {} not divisible by n_groups {n_groups} (M={m} K={k})",
        data.len()
    );
    let block_bytes = data.len() / n_groups;
    // f16 scale + group/2 packed nibbles; 130 at G=256, 66 at G=128.
    let header = 2 + GROUP / 2;
    assert!(
        block_bytes >= header + 2 && (block_bytes - header) % 2 == 0,
        "OQ+C block_bytes {block_bytes} invalid (expected {header} + 2·N_out for group {GROUP})"
    );
    let n_out = (block_bytes - header) / 2;
    let mut combined = vec![0u8; m * k + m * ng * 4];
    for r in 0..m {
        for g in 0..ng {
            let src = (r * ng + g) * block_bytes;
            let dst = r * k + g * GROUP;
            // int4 bulk → int8 (read as signed char downstream).
            for i in 0..GROUP / 2 {
                let byte = data[src + 2 + i];
                combined[dst + 2 * i] = sext4(byte & 0xf) as u8;
                combined[dst + 2 * i + 1] = sext4(byte >> 4) as u8;
            }
            // Overlay the sparse int8 outliers: (u8 idx, i8 val) × N_out.
            let tbl = src + header;
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
/// Compact residency is the DEFAULT as of 2026-08-20.
///
/// oq4.25++ IS a mixed-precision format — 4-bit bulk plus a sparse int8 outlier
/// overlay, decoded together by one kernel. That is the point of Opus. Unpacking
/// it into uniform int8 containers at load doubles the bytes every decode step
/// streams and changes not one weight value (`parity_gemm_oq_compact` and
/// `parity_gemv_oq_compact` both check the compact kernels against that exact
/// expansion). Both decode paths are W4A16, so the activation precision is the
/// same too; only the f32 accumulation order differs.
///
/// Measured, q8 KV, 64 tokens:
///   Qwen3.8-27B    dense  8.00 -> 9.10 tok/s (+14%), peak RSS 30 -> 21 GiB
///   Qwen3.5-35B-A3B MoE  36.80 -> 48.30 tok/s (+31%)
///
/// `HIPFIRE_OQ_COMPACT_RESIDENT=0` opts back out to the expansion. Keep that
/// escape hatch: a call site with no compact arm REFUSES loudly (see the
/// OqCompactG256 guards in qwen35), so an unwired path fails visibly rather than
/// corrupting, and unsetting is the documented workaround.
/// Whether OqPlusCompact tensors stay compact on the device.
///
/// Public because routed MoE experts consult it too (`load_moe_expert`), and
/// they must make the SAME choice as dense tensors from the SAME switch -- a
/// second env var for experts would let the two disagree silently.
pub fn compact_resident_enabled() -> bool {
    !matches!(
        hipfire_env::OQ_COMPACT_RESIDENT
            .get()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

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
    // DEFAULT as of 2026-08-20 (see `compact_resident_enabled`). The expansion
    // below is now the opt-out path and should eventually go, along with its two
    // siblings in lfm2moe and minimax.
    //
    // HIPFIRE_OQ_COMPACT_RESIDENT_ONLY_K / _ONLY_M narrow this to chosen K and M
    // values so the compact-vs-expanded logit divergence can be bisected down to
    // a single (M, K) projection class. Purely diagnostic: unset (the normal
    // case) keeps every OqPlusCompact tensor compact, exactly as before. The
    // shape is the handle because this hook never sees the tensor name.
    if compact_resident_enabled() && compact_shape_selected(m, k) {
        if qt == QuantType::OqPlusCompact.code() {
            let mut owned = data.to_vec();
            normalize_compact_overlays(&mut owned, m, k, 256);
            let split = split_compact_planes(&owned, m, k, 256);
            return Some((split, DType::OqCompactG256));
        }
        if qt == QuantType::OqPlusCompactG128.code() {
            let mut owned = data.to_vec();
            normalize_compact_overlays(&mut owned, m, k, 128);
            let split = split_compact_planes(&owned, m, k, 128);
            return Some((split, DType::OqCompactG128));
        }
    }
    let bytes = match qt {
        c if c == QuantType::Oq8G256.code() => oq8_combined(data, m, k),
        c if c == QuantType::OqPlusG256.code() => oq4_to_oq8_combined(data, m, k),
        c if c == QuantType::OqPlusCompact.code() => oqplus_compact_to_oq8_combined(data, m, k),
        // G=128 expands through the same path at a 128-element group; the result
        // is the identical combined Oq8G256 layout, so only the decode differs.
        c if c == QuantType::OqPlusCompactG128.code() => {
            oqplus_compact_to_oq8_combined_g(data, m, k, 128)
        }
        c if c == QuantType::Oq3G256.code() => oq3_to_oq8_combined(data, m, k),
        _ => return None,
    };
    Some((bytes, DType::Oq8G256))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reason G=128 exists. K=896 (7*128) is a real shape — Qwen2-0.5B's
    /// hidden size — and 896 % 256 == 128, so the G=256 compact format cannot
    /// represent it at all: the group must divide K. G=128 can, and expands to
    /// the same combined Oq8G256 layout, so nothing downstream changes.
    #[test]
    fn g128_covers_a_k_that_g256_cannot_divide() {
        const K: usize = 896;
        const M: usize = 2;
        assert_ne!(
            K % 256,
            0,
            "K=896 must NOT be divisible by 256 for this test"
        );
        assert_eq!(K % 128, 0, "K=896 must be divisible by 128");

        let group = 128usize;
        let n_out = 3usize;
        let header = 2 + group / 2;
        let block = header + 2 * n_out;
        let ng = K / group;
        let mut data = vec![0u8; M * ng * block];
        for b in 0..M * ng {
            let off = b * block;
            data[off..off + 2].copy_from_slice(&F16_ONE.to_le_bytes());
            // Distinct nibbles so a wrong header offset would misplace them.
            for i in 0..group / 2 {
                data[off + 2 + i] = 0x21;
            }
            for s in 0..n_out {
                data[off + header + 2 * s] = (s * 7) as u8;
                data[off + header + 2 * s + 1] = (100 + s) as u8;
            }
        }

        let combined = oqplus_compact_to_oq8_combined_g(&data, M, K, group);
        assert_eq!(combined.len(), M * K + M * ng * 4);
        // Bulk nibbles: low then high of 0x21 sign-extended = 1, 2. Position 1 is
        // bulk; position 0 is NOT, because overlay s=0 has index 0 and overrides
        // it — which is the precedence the format requires.
        assert_eq!(combined[1] as i8, 2);
        assert_eq!(combined[2] as i8, 1);
        // Overlays land at their in-group positions, proving the header offset is
        // 66 here and not the G=256 value of 130.
        assert_eq!(combined[0] as i8, 100, "position 0 is overlay s=0");
        assert_eq!(combined[7] as i8, 101, "position 7 is overlay s=1");
        assert_eq!(combined[14] as i8, 102, "position 14 is overlay s=2");
        // Scales are the f16 read back as f32, one per group.
        let so = M * K;
        assert_eq!(
            f32::from_le_bytes(combined[so..so + 4].try_into().unwrap()),
            1.0
        );
    }

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

#[cfg(test)]
mod overlay_normalize_tests {
    use super::{normalize_compact_overlays, oqplus_compact_to_oq8_combined};

    /// Neutralising a superseded entry must not change what the block DECODES to.
    /// The expansion is the oracle: it applies the table in order, so last-wins is
    /// already correct there, and normalization must leave its output identical.
    #[test]
    fn normalize_preserves_decoded_weights() {
        const G: usize = 256;
        let (m, k, n_out) = (3usize, G, 4usize);
        let stride = 2 + G / 2 + 2 * n_out;
        let mut blocks = vec![0u8; m * (k / G) * stride];
        let mut seed = 0x1234_5678u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 16) as u8
        };
        for b in 0..m {
            let off = b * stride;
            blocks[off] = 0x00;
            blocks[off + 1] = 0x3c; // f16 1.0
            for i in 0..G / 2 {
                blocks[off + 2 + i] = rnd();
            }
            // Deliberately collide: only two distinct indices across four entries.
            for e in 0..n_out {
                blocks[off + 2 + G / 2 + 2 * e] = if e % 2 == 0 { 7 } else { 200 };
                blocks[off + 2 + G / 2 + 2 * e + 1] = rnd();
            }
        }
        let before = oqplus_compact_to_oq8_combined(&blocks, m, k);
        let mut norm = blocks.clone();
        normalize_compact_overlays(&mut norm, m, k, G);
        let after = oqplus_compact_to_oq8_combined(&norm, m, k);
        assert_eq!(before, after, "normalization changed the decoded weights");
        assert_ne!(blocks, norm, "fixture had no duplicates to resolve");

        // The two properties the kernels are allowed to rely on: the bulk nibble
        // under every overlay index is ZERO (so `val - bulk == val`), and every
        // entry that still carries a nonzero value sits on a unique index.
        for b in 0..m {
            let tbl = b * stride + 2 + G / 2;
            let mut seen: Vec<u8> = Vec::new();
            for e in 0..n_out {
                let idx = norm[tbl + 2 * e];
                let val = norm[tbl + 2 * e + 1];
                let byte = norm[b * stride + 2 + idx as usize / 2];
                let bulk = if idx % 2 == 0 { byte & 0xf } else { byte >> 4 };
                assert_eq!(bulk, 0, "bulk nibble under overlay index {idx} not zeroed");
                if val != 0 {
                    assert!(!seen.contains(&idx), "duplicate live index {idx} survived");
                    seen.push(idx);
                }
            }
        }
    }

    /// Blocks with no duplicate indices still get their bulk nibbles zeroed —
    /// that part is unconditional, and it is what lets the kernel drop the
    /// `bulk` term. Decoding is unchanged and a second pass is a no-op.
    #[test]
    fn normalize_zeroes_bulk_and_is_idempotent() {
        const G: usize = 256;
        let (m, k, n_out) = (2usize, G, 3usize);
        let stride = 2 + G / 2 + 2 * n_out;
        let mut blocks = vec![0x11u8; m * stride];
        for b in 0..m {
            for e in 0..n_out {
                blocks[b * stride + 2 + G / 2 + 2 * e] = (e * 9) as u8; // all distinct
                blocks[b * stride + 2 + G / 2 + 2 * e + 1] = (7 + e) as u8;
            }
        }
        let before = oqplus_compact_to_oq8_combined(&blocks, m, k);
        let mut once = blocks.clone();
        normalize_compact_overlays(&mut once, m, k, G);
        assert_ne!(
            blocks, once,
            "bulk nibbles under the overlays were not zeroed"
        );
        assert_eq!(
            before,
            oqplus_compact_to_oq8_combined(&once, m, k),
            "zeroing the bulk changed the decoded weights"
        );
        let mut twice = once.clone();
        normalize_compact_overlays(&mut twice, m, k, G);
        assert_eq!(once, twice, "not idempotent");
    }

    /// A single-entry table has nothing to dedupe but still needs its bulk
    /// zeroed — the old normalizer bailed at `n_out < 2` and would have left the
    /// kernel adding `val` on top of a live nibble.
    /// The split layout must carry exactly the same information: decode every
    /// block back out of it and compare against the interleaved decode. Byte
    /// count is unchanged, so this also pins that no padding crept in.
    #[test]
    fn split_planes_preserve_every_block() {
        use super::split_compact_planes;
        const G: usize = 256;
        let (m, k, n_out) = (5usize, 2 * G, 3usize);
        let ng = k / G;
        let stride = 2 + G / 2 + 2 * n_out;
        let mut blocks = vec![0u8; m * ng * stride];
        let mut seed = 0xC0FF_EE11u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 16) as u8
        };
        for b in blocks.iter_mut() {
            *b = rnd();
        }
        let split = split_compact_planes(&blocks, m, k, G);
        assert_eq!(split.len(), blocks.len(), "split changed the byte count");
        let nib = G / 2;
        let side = stride - nib;
        let side_base = m * ng * nib;
        for b in 0..m * ng {
            let src = b * stride;
            assert_eq!(
                &split[b * nib..(b + 1) * nib],
                &blocks[src + 2..src + 2 + nib],
                "nibbles moved wrong for block {b}"
            );
            let d = side_base + b * side;
            assert_eq!(&split[d..d + 2], &blocks[src..src + 2], "scale wrong {b}");
            assert_eq!(
                &split[d + 2..d + side],
                &blocks[src + 2 + nib..src + stride],
                "overlay table wrong for block {b}"
            );
        }
    }

    /// split -> unsplit must be the identity, or anything that downloads a
    /// resident compact weight decodes garbage.
    #[test]
    fn split_planes_round_trip() {
        use super::{split_compact_planes, unsplit_compact_planes};
        for (group, k) in [(256usize, 512usize), (128, 384)] {
            let (m, n_out) = (4usize, 3usize);
            let ng = k / group;
            let stride = 2 + group / 2 + 2 * n_out;
            let mut blocks = vec![0u8; m * ng * stride];
            let mut seed = 0x1357_9BDFu32;
            for b in blocks.iter_mut() {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                *b = (seed >> 16) as u8;
            }
            let split = split_compact_planes(&blocks, m, k, group);
            let back = unsplit_compact_planes(&split, m, k, group);
            assert_eq!(back, blocks, "split/unsplit not an identity at G={group}");
        }
    }

    #[test]
    fn normalize_handles_n_out_one() {
        const G: usize = 256;
        let (m, k) = (1usize, G);
        let stride = 2 + G / 2 + 2;
        let mut blocks = vec![0x11u8; stride];
        blocks[0] = 0x00;
        blocks[1] = 0x3c; // f16 1.0
        blocks[2 + G / 2] = 5; // idx
        blocks[2 + G / 2 + 1] = (-100i8) as u8;
        let before = oqplus_compact_to_oq8_combined(&blocks, m, k);
        normalize_compact_overlays(&mut blocks, m, k, G);
        assert_eq!(
            blocks[2 + 5 / 2] & 0xf0,
            0,
            "high nibble of byte 2 not zeroed"
        );
        assert_eq!(before, oqplus_compact_to_oq8_combined(&blocks, m, k));
    }
}
