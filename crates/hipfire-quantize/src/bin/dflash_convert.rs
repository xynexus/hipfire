// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#![allow(
    clippy::collapsible_if,
    clippy::if_same_then_else,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_map_or
)]

//! dflash_convert: Convert a HuggingFace DFlash draft safetensors + config.json
//! into a hipfire `.hfq` file with a dflash metadata section.
//!
//! Usage:
//!     dflash_convert --input <dir_or_hf_id> --output <file.hfq> [--bf16 | --f16 | --keep-f32]
//!
//! Reads a single-file safetensors dump (the z-lab/Qwen3.5-*-DFlash draft
//! layout — no shards in practice at 1-4B params) and rewrites the tensors
//! into the hipfire HFQ container. BF16 weights are preserved by default.
//! `--f16` produces the compatibility artifact for older cards, while the
//! runtime can also convert a BF16 artifact to F16 when native BF16 WMMA is
//! unavailable. Pass `--keep-f32` to expand weights to F32.
//! Per-layer norms (`input_layernorm`, `post_attention_layernorm`,
//! `q_norm`, `k_norm`, `hidden_norm`, `norm`) are always F32.
//!
//! Metadata JSON layout:
//!
//! ```json
//! {
//!   "architecture": "dflash",
//!   "config": {<full HF config.json>},
//!   "dflash": {
//!     "block_size": 16,
//!     "mask_token_id": 248070,
//!     "target_layer_ids": [1, 8, 15, 22, 29],
//!     "num_target_layers": 32,
//!     "draft_dtype": "bf16"
//!   },
//!   "tokenizer": null
//! }
//! ```
//!
//! arch_id for the dflash draft is 20. The hipfire loader distinguishes
//! dflash drafts from Qwen3/Qwen3.5 by both arch_id and the presence of
//! the top-level `dflash` key in metadata.

use hipfire_primitives::conv::{
    f32_slice_to_bf16_bytes, f32_slice_to_f16_bytes, plain_dtype_to_f32 as to_f32,
};
use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

// ─── Safetensors Parser (single-file only) ─────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

struct SafetensorsFile {
    _file: File,
    mmap: Mmap,
    header_size: usize,
    tensors: HashMap<String, TensorMeta>,
}

impl SafetensorsFile {
    fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let header_json = std::str::from_utf8(&mmap[8..8 + header_len])
            .expect("safetensors header is not valid utf8");
        let raw: serde_json::Value =
            serde_json::from_str(header_json).expect("safetensors header JSON parse failed");
        let mut tensors = HashMap::new();
        if let serde_json::Value::Object(map) = raw {
            for (k, v) in map {
                if k == "__metadata__" {
                    continue;
                }
                let meta: TensorMeta = serde_json::from_value(v)
                    .unwrap_or_else(|e| panic!("tensor meta for {k}: {e}"));
                tensors.insert(k, meta);
            }
        }
        Ok(Self {
            _file: file,
            mmap,
            header_size: 8 + header_len,
            tensors,
        })
    }

    fn tensor_data(&self, name: &str) -> Option<(&TensorMeta, &[u8])> {
        let meta = self.tensors.get(name)?;
        let start = self.header_size + meta.data_offsets[0];
        let end = self.header_size + meta.data_offsets[1];
        Some((meta, &self.mmap[start..end]))
    }

    fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.keys().cloned().collect();
        names.sort();
        names
    }
}

fn f32_slice_to_f32_bytes(f32_data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(f32_data.len() * 4);
    for &v in f32_data {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

fn dflash_block_size(config: &serde_json::Value) -> Result<u32, String> {
    config
        .get("block_size")
        .and_then(|value| value.as_u64())
        .or_else(|| {
            config
                .get("dflash_config")
                .and_then(|value| value.get("block_size"))
                .and_then(|value| value.as_u64())
        })
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            "config.json missing positive block_size (top-level or dflash_config.block_size)"
                .to_string()
        })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DraftFormat {
    Bf16,
    F16,
    F32,
    Mq3,
    Mq4,
    Mq6,
    /// Non-rotated (plain-basis) int8 G256 — NPU W8A8/W8A16 projection path.
    Oq8,
    /// Non-rotated PURE int4 G256 (4.0625 b/w) — minimum-bandwidth NPU W4A8.
    Oq4,
    /// Non-rotated int4 bulk + `n` int8 overlays per G256 (mixed precision,
    /// e.g. oq4.25 = 3 overlays). Carries the overlay count.
    Oq4Mixed(usize),
}

impl DraftFormat {
    #[allow(clippy::too_many_arguments)]
    fn from_flags(
        use_f16: bool,
        keep_f32: bool,
        use_bf16: bool,
        use_mq3: bool,
        use_mq4: bool,
        use_mq6: bool,
        use_oq8: bool,
        use_oq4: bool,
        oq4_mixed_outliers: Option<usize>,
    ) -> Result<Self, String> {
        let selected = [
            use_f16,
            keep_f32,
            use_bf16,
            use_mq3,
            use_mq4,
            use_mq6,
            use_oq8,
            use_oq4,
            oq4_mixed_outliers.is_some(),
        ]
        .into_iter()
        .filter(|enabled| *enabled)
        .count();
        if selected > 1 {
            return Err(
                "--bf16, --f16, --keep-f32, --mq3, --mq4, --mq6, --oq8, --oq4, and --oq4.<bits> \
                 are mutually exclusive"
                    .to_string(),
            );
        }
        Ok(if use_f16 {
            Self::F16
        } else if keep_f32 {
            Self::F32
        } else if use_mq3 {
            Self::Mq3
        } else if use_mq4 {
            Self::Mq4
        } else if use_mq6 {
            Self::Mq6
        } else if use_oq8 {
            Self::Oq8
        } else if use_oq4 {
            Self::Oq4
        } else if let Some(n) = oq4_mixed_outliers {
            Self::Oq4Mixed(n)
        } else {
            // No flag and explicit --bf16 intentionally resolve identically.
            Self::Bf16
        })
    }

    fn metadata_name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::Mq3 => "mq3",
            Self::Mq4 => "mq4",
            Self::Mq6 => "mq6",
            Self::Oq8 => "oq8",
            // Base token; the mixed overlay count rides in the artifact filename
            // per the canonical naming (e.g. `.oq4.25+.hfq`). The loader keys on
            // per-tensor QuantType, not this string.
            Self::Oq4 | Self::Oq4Mixed(_) => "oq4",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16 (weights), F32 (norms)",
            Self::F16 => "F16 (weights), F32 (norms)",
            Self::F32 => "F32",
            Self::Mq3 => "MQ3-G256 (weights), F32 (norms)",
            Self::Mq4 => "MQ4-G256 (weights), F32 (norms)",
            Self::Mq6 => "MQ6-G256 (weights), F32 (norms)",
            Self::Oq8 => "OQ8-plain (non-rotated int8 G256 weights), F32 (norms) — NPU int8 W8A8/W8A16",
            Self::Oq4 => {
                "OQ4-plain (non-rotated PURE int4 G256, 130 B/group = 4.0625 b/w), \
                 F32 (norms) — NPU int4 W4A8, minimum bandwidth"
            }
            Self::Oq4Mixed(_) => {
                "OQ4-mixed-plain (non-rotated int4 bulk + int8 overlays per G256), \
                 F32 (norms) — NPU int4 W4A8"
            }
        }
    }
}

#[cfg(test)]
mod config_tests {
    use super::{dflash_block_size, DraftFormat};

    #[test]
    fn reads_current_zlab_nested_block_size() {
        let config = serde_json::json!({
            "block_size": null,
            "dflash_config": {"block_size": 16}
        });
        assert_eq!(dflash_block_size(&config).unwrap(), 16);
    }

    #[test]
    fn preserves_legacy_top_level_block_size() {
        let config = serde_json::json!({
            "block_size": 8,
            "dflash_config": {"block_size": 16}
        });
        assert_eq!(dflash_block_size(&config).unwrap(), 8);
    }

    #[test]
    fn bf16_is_the_default_draft_format() {
        assert_eq!(
            DraftFormat::from_flags(false, false, false, false, false, false, false, false, None)
                .unwrap(),
            DraftFormat::Bf16
        );
        assert_eq!(DraftFormat::Bf16.metadata_name(), "bf16");
        assert_eq!(super::QuantType::BF16 as u8, 16);
    }

    #[test]
    fn f16_remains_an_explicit_compatibility_format() {
        assert_eq!(
            DraftFormat::from_flags(true, false, false, false, false, false, false, false, None)
                .unwrap(),
            DraftFormat::F16
        );
        assert!(
            DraftFormat::from_flags(true, true, false, false, false, false, false, false, None)
                .is_err()
        );
    }
}

// ─── FWHT + MQ quantization ───────────────────────────────────────────────

/// MagnumQuant MQ3-G256: FWHT-rotated 3-bit quantization.
/// 104 bytes per 256 weights (0.406 B/w). Same binary layout as HFQ3-G256.
/// Lifted verbatim from hipfire-quantize/main.rs's `quantize_mq3g256`.
fn quantize_mq3g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 104;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        cpu_fwht_256(&mut group, signs1, signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 7.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        for chunk in 0..32 {
            let ci = chunk * 8;
            let mut q = [0u8; 8];
            for j in 0..8 {
                q[j] = ((group[ci + j] - min_val) * inv_scale + 0.5).clamp(0.0, 7.0) as u8;
            }
            let b0 = (q[0] & 7) | ((q[1] & 7) << 3) | ((q[2] & 3) << 6);
            let b1 = ((q[2] >> 2) & 1) | ((q[3] & 7) << 1) | ((q[4] & 7) << 4) | ((q[5] & 1) << 7);
            let b2 = ((q[5] >> 1) & 3) | ((q[6] & 7) << 2) | ((q[7] & 7) << 5);

            let bo = out_off + 8 + chunk * 3;
            output[bo] = b0;
            output[bo + 1] = b1;
            output[bo + 2] = b2;
        }
    }
    output
}

/// MagnumQuant MQ4-G256: FWHT-rotated 4-bit quantization.
/// 136 bytes per 256 weights (0.531 B/w). Same binary layout as HFQ4-G256;
/// the rotation is baked into the weights so the GEMM kernel just rotates
/// the input x instead of inverse-rotating W.
fn quantize_mq4g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 136;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        cpu_fwht_256(&mut group, signs1, signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());
        for i in 0..128 {
            let lo_q = ((group[2 * i] - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((group[2 * i + 1] - min_val) * inv_scale + 0.5) as u8;
            output[out_off + 8 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }
    output
}

/// MagnumQuant MQ6-G256: FWHT-rotated 6-bit quantization.
/// 200 bytes per 256 weights (0.781 B/w). Same binary layout as HFQ6-G256.
fn quantize_mq6g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 200;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);
        cpu_fwht_256(&mut group, signs1, signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 63.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        for i in (0..256).step_by(4) {
            let q0 = (((group[i] - min_val) * inv_scale + 0.5) as u8).min(63);
            let q1 = (((group[i + 1] - min_val) * inv_scale + 0.5) as u8).min(63);
            let q2 = (((group[i + 2] - min_val) * inv_scale + 0.5) as u8).min(63);
            let q3 = (((group[i + 3] - min_val) * inv_scale + 0.5) as u8).min(63);

            let byte_off = 8 + (i / 4) * 3;
            output[out_off + byte_off] = q0 | (q1 << 6);
            output[out_off + byte_off + 1] = (q1 >> 2) | (q2 << 4);
            output[out_off + byte_off + 2] = (q2 >> 4) | (q3 << 2);
        }
    }

    output
}

/// Non-rotated (plain-basis) OQ8: per-256-group symmetric signed int8, f16 scale.
/// Block = `[f16 scale][256 int8]` = 258 B/group. NO FWHT — the AIE2 int8 kernel
/// (W8A16/W8A8) reads these codes directly and needs no activation rotation.
///
/// Scale = max_abs / 127 (round-to-nearest-even quantization). At 8-bit,
/// clip-search is a no-op vs max-abs (measured: identical on the z-lab 9B
/// drafter — no outliers to clip at 256 levels), so this matches OQ8+ quality
/// while keeping the runtime activation path a plain per-group int8 quant.
/// The final partial group (if `n % 256 != 0`) is zero-padded; padded lanes are
/// never read by the kernel. Dequant oracle: `dequant_oq8_plain`.
fn quantize_oq8_plain(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256usize;
    let block_bytes = 2 + 256; // f16 scale + 256 signed int8
    let n = f32_data.len();
    let n_blocks = n.div_ceil(group_size);
    let mut output = vec![0u8; n_blocks * block_bytes];
    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let grp = &f32_data[start..end];
        let max_abs = grp.iter().fold(0.0f32, |a, &w| a.max(w.abs()));
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let inv = 1.0 / scale;
        let out_off = b * block_bytes;
        let scale_f16 = hipfire_primitives::conv::f32_to_f16(scale);
        output[out_off] = (scale_f16 & 0xFF) as u8;
        output[out_off + 1] = (scale_f16 >> 8) as u8;
        for (i, &w) in grp.iter().enumerate() {
            // round-to-nearest-even, clamp to symmetric signed int8 [-127, 127].
            let q = (w * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
            output[out_off + 2 + i] = q as u8;
        }
        // trailing lanes of a partial final group stay 0 (== quantized zero).
    }
    output
}

/// Symmetric clip-search: pick the per-group scale minimising round-trip SSE over
/// a shrink grid, rather than the naive `amax/qmax`. This is the `+` in `oq4.25+`.
/// Mirrors `codecs::symmetric_clipsearch`, kept local like the other plain codecs.
fn clipsearch_plain(group: &[f32], qmax: f32) -> f32 {
    const CLIP_GRID: [f32; 9] = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6];
    let amax = group.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let (mut best_scale, mut best_err) = (amax / qmax, f32::INFINITY);
    for &c in &CLIP_GRID {
        let scale = (c * amax / qmax).max(1e-12);
        let inv = 1.0 / scale;
        let mut err = 0.0f32;
        for &v in group.iter() {
            let q = (v * inv).round().clamp(-qmax, qmax);
            let d = v - q * scale;
            err += d * d;
        }
        if err < best_err {
            best_err = err;
            best_scale = scale;
        }
    }
    if best_scale > 0.0 { best_scale } else { 1.0 }
}

/// Rank group positions by how much promoting them from int4 to int8 reduces
/// squared error. The top `n_out` become the sparse overlay.
fn mixed_overlay_indices_plain(group: &[f32; 256], scale: f32) -> [usize; 256] {
    let inv = 1.0 / scale.max(1e-12);
    let gain = |index: usize| -> f32 {
        let value = group[index];
        let q4 = (value * inv).round().clamp(-7.0, 7.0);
        let q8 = (value * inv).round().clamp(-127.0, 127.0);
        let e4 = value - q4 * scale;
        let e8 = value - q8 * scale;
        e4 * e4 - e8 * e8
    };
    let mut indices: [usize; 256] = core::array::from_fn(|i| i);
    indices.sort_unstable_by(|&l, &r| {
        gain(r)
            .partial_cmp(&gain(l))
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    indices
}

/// SSE of the mixed encoding: outlier slots clamp to int8, the rest to int4.
fn mixed_overlay_error_plain(
    group: &[f32; 256],
    scale: f32,
    indices: &[usize; 256],
    n_out: usize,
) -> f32 {
    let inv = 1.0 / scale.max(1e-12);
    let mut is_w8 = [false; 256];
    for &index in &indices[..n_out] {
        is_w8[index] = true;
    }
    group
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let limit = if is_w8[index] { 127.0 } else { 7.0 };
            let q = (value * inv).round().clamp(-limit, limit);
            let e = value - q * scale;
            e * e
        })
        .sum()
}

/// Refit the scale knowing which slots will be int8 — a wider grid than the
/// int4-only search, because outliers no longer force the scale up.
fn refit_mixed_scale_plain(
    group: &[f32; 256],
    indices: &[usize; 256],
    n_out: usize,
    fallback: f32,
) -> f32 {
    const CLIP_GRID: [f32; 14] = [
        1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35,
    ];
    let amax = group.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let mut best_scale = fallback.max(1e-12);
    let mut best_error = mixed_overlay_error_plain(group, best_scale, indices, n_out);
    for clip in CLIP_GRID {
        let scale = (clip * amax / 7.0).max(1e-12);
        let error = mixed_overlay_error_plain(group, scale, indices, n_out);
        if error < best_error {
            best_scale = scale;
            best_error = error;
        }
    }
    best_scale
}

/// Joint scale/overlay selection: clip-search, pick outliers, refit, repeat once.
fn mixed_clipsearch_plain(group: &[f32; 256], n_out: usize) -> (f32, [usize; 256]) {
    let s0 = clipsearch_plain(group, 7.0);
    let i0 = mixed_overlay_indices_plain(group, s0);
    let s1 = refit_mixed_scale_plain(group, &i0, n_out, s0);
    let i1 = mixed_overlay_indices_plain(group, s1);
    let s2 = refit_mixed_scale_plain(group, &i1, n_out, s1);
    (s2, i1)
}

/// Number of int8 overlay slots per 256-group for a requested mixed bit-width.
///
/// Base cost is 4.0625 b/w (130 B/group = f16 scale + 128 nibbles); each overlay
/// entry is 2 B = 0.0625 b/w. So `bits = 4.0625 + n_out/16`, and `oq4.25` → 3.
/// Matches `hipfire-quantize::main::parse_opus_mixed_format` exactly so the
/// sidecar and the general HFQ pipeline agree on what a name means.
fn mixed_outliers_for_bits(bits: f32) -> Option<usize> {
    let exact = (bits - 4.0625) * 16.0;
    let n = exact.round() as isize;
    if !(1..=62).contains(&n) || (exact - n as f32).abs() > 1e-4 {
        return None;
    }
    Some(n as usize)
}

/// Non-rotated PURE int4 Opus quant — the minimum-bandwidth weight format.
///
/// Per 256-group: `[f16 scale][128 int4 nibbles]` = **130 B/group = 4.0625 b/w**.
/// This is `quantize_oq4_mixed_plain` with zero overlays, and it is the format the
/// AIE2 W4A8 projection kernel wants: the NPU weight path is bandwidth-bound, so
/// *bytes* are the only remaining lever (eight feed-side knobs measured null).
///
/// Deliberately NOT the mixed format. `Oq4MixedPlain` (qt=46) costs
/// `130 + 2·n_out` B/group and buys ~1 dB of SNR across the whole
/// 4.25 → 8.0 b/w range — second-order, while the bytes are first-order. Use this
/// unless a measured acceptance-rate result says otherwise.
///
/// Non-rotated for the same reason as `Oq8Plain`/`Oq4MixedPlain`: a rotated weight
/// basis requires the *activation* to be rotated per dispatch at runtime, which
/// cannot be baked in at HFQ creation. Distinct from the canonical rotated
/// `Oq4G256 = 34` so a rotated consumer can never mis-handle these bytes.
fn quantize_oq4_plain(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256usize;
    let block_bytes = 130; // [f16 scale][128 nibbles]
    let n = f32_data.len();
    let n_blocks = n.div_ceil(group_size);
    let mut output = vec![0u8; n_blocks * block_bytes];
    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        // Partial final group is zero-padded; padded lanes are never read at
        // inference and zero is exactly representable.
        let mut group = [0.0f32; 256];
        group[..end - start].copy_from_slice(&f32_data[start..end]);

        // Clip-search over the int4 grid (qmax=7) — the `+` in oq4+.
        let scale = clipsearch_plain(&group, 7.0);
        let inv = 1.0 / scale;
        let out_off = b * block_bytes;
        let scale_f16 = hipfire_primitives::conv::f32_to_f16(scale);
        output[out_off] = (scale_f16 & 0xFF) as u8;
        output[out_off + 1] = (scale_f16 >> 8) as u8;
        for i in 0..128 {
            let qlo = (group[2 * i] * inv).round().clamp(-7.0, 7.0) as i8;
            let qhi = (group[2 * i + 1] * inv).round().clamp(-7.0, 7.0) as i8;
            output[out_off + 2 + i] = ((qlo as u8) & 0xf) | (((qhi as u8) & 0xf) << 4);
        }
    }
    output
}

/// Dequant oracle for `quantize_oq4_plain` — for round-trip validation.
#[cfg(test)]
fn dequant_oq4_plain(data: &[u8], n: usize) -> Vec<f32> {
    let group_size = 256usize;
    let block_bytes = 130;
    let n_blocks = n.div_ceil(group_size);
    let mut out = Vec::with_capacity(n_blocks * group_size);
    for b in 0..n_blocks {
        let off = b * block_bytes;
        let scale =
            hipfire_primitives::conv::f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        for i in 0..128 {
            let byte = data[off + 2 + i];
            // sign-extend each nibble from 4 bits
            out.push(scale * ((((byte & 0xf) as i8) << 4 >> 4) as f32));
            out.push(scale * ((((byte >> 4) as i8) << 4 >> 4) as f32));
        }
    }
    out.truncate(n);
    out
}

/// Non-rotated mixed-precision Opus quant: int4 bulk plus a sparse int8 overlay.
///
/// Per 256-group: `[f16 scale][128 int4 nibbles][n_out × (u8 index, i8 value)]`
/// = `130 + 2·n_out` B. At `n_out = 3` that is 136 B/group = **4.25 b/w**.
///
/// Same layout and values as `codecs::quantize_oqplus_compact` (`OqPlusCompact`,
/// qt=36) but WITHOUT the FWHT rotation, for the same reason `Oq8Plain` exists:
/// a rotated weight basis requires the *activation* to be rotated per dispatch at
/// runtime, which cannot be baked in at HFQ creation. The AIE2 W4A8 projection
/// kernel consumes these bytes directly.
///
/// Nibble slots at outlier positions still carry the int4 clamp, so a consumer
/// that ignores the overlay table degrades gracefully to plain int4 rather than
/// reading garbage.
fn quantize_oq4_mixed_plain(f32_data: &[f32], n_out: usize) -> Vec<u8> {
    assert!(
        (1..=62).contains(&n_out),
        "n_out must be 1..=62, got {n_out}"
    );
    let group_size = 256usize;
    let block_bytes = 130 + 2 * n_out;
    let n = f32_data.len();
    let n_blocks = n.div_ceil(group_size);
    let mut output = vec![0u8; n_blocks * block_bytes];
    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        // Partial final group is zero-padded; padded lanes are never read at
        // inference, and zero is exactly representable.
        let mut group = [0.0f32; 256];
        group[..end - start].copy_from_slice(&f32_data[start..end]);

        let (scale, idx) = mixed_clipsearch_plain(&group, n_out);
        let inv = 1.0 / scale;
        let out_off = b * block_bytes;
        let scale_f16 = hipfire_primitives::conv::f32_to_f16(scale);
        output[out_off] = (scale_f16 & 0xFF) as u8;
        output[out_off + 1] = (scale_f16 >> 8) as u8;

        // Bulk int4 nibbles at every position (outlier slots overridden on load).
        for i in 0..128 {
            let qlo = (group[2 * i] * inv).round().clamp(-7.0, 7.0) as i8;
            let qhi = (group[2 * i + 1] * inv).round().clamp(-7.0, 7.0) as i8;
            output[out_off + 2 + i] = ((qlo as u8) & 0xf) | (((qhi as u8) & 0xf) << 4);
        }
        // Sparse int8 overlay: (u8 index-in-group, i8 value).
        let tbl = out_off + 130;
        for (s, &pos) in idx[..n_out].iter().enumerate() {
            let q8 = (group[pos] * inv).round().clamp(-127.0, 127.0) as i8;
            output[tbl + 2 * s] = pos as u8;
            output[tbl + 2 * s + 1] = q8 as u8;
        }
    }
    output
}

/// Dequant oracle for `quantize_oq4_mixed_plain` — for round-trip validation.
#[cfg(test)]
fn dequant_oq4_mixed_plain(data: &[u8], n: usize, n_out: usize) -> Vec<f32> {
    let group_size = 256usize;
    let block_bytes = 130 + 2 * n_out;
    let n_blocks = n.div_ceil(group_size);
    let mut out = Vec::with_capacity(n_blocks * group_size);
    for b in 0..n_blocks {
        let off = b * block_bytes;
        let scale =
            hipfire_primitives::conv::f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let mut grp = [0.0f32; 256];
        // int4 bulk: sign-extend each nibble from 4 bits.
        for i in 0..128 {
            let byte = data[off + 2 + i];
            let lo = ((byte & 0xf) as i8) << 4 >> 4;
            let hi = ((byte >> 4) as i8) << 4 >> 4;
            grp[2 * i] = scale * lo as f32;
            grp[2 * i + 1] = scale * hi as f32;
        }
        // int8 overlay wins where present.
        let tbl = off + 130;
        for s in 0..n_out {
            let pos = data[tbl + 2 * s] as usize;
            let val = data[tbl + 2 * s + 1] as i8;
            grp[pos] = scale * val as f32;
        }
        out.extend_from_slice(&grp);
    }
    out.truncate(n);
    out
}

/// Dequant oracle for `quantize_oq8_plain` — for the round-trip validation.
#[cfg(test)]
fn dequant_oq8_plain(data: &[u8], n: usize) -> Vec<f32> {
    let group_size = 256usize;
    let block_bytes = 2 + 256;
    let n_blocks = n.div_ceil(group_size);
    let mut out = Vec::with_capacity(n_blocks * group_size);
    for b in 0..n_blocks {
        let off = b * block_bytes;
        let scale =
            hipfire_primitives::conv::f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        for i in 0..group_size {
            let q = data[off + 2 + i] as i8;
            out.push(q as f32 * scale);
        }
    }
    out.truncate(n);
    out
}

// ─── HFQ File Format ──────────────────────────────────────────────────────

const HFQ_MAGIC: &[u8; 4] = b"HFQM";
const HFQ_VERSION: u32 = 1;
const ARCH_ID_DFLASH_DRAFT: u32 = 20;

#[repr(u8)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
enum QuantType {
    Q4F16G64 = 0,
    F16 = 1,
    F32 = 2,
    MQ4G256 = 13,
    MQ6G256 = 15,
    BF16 = 16,
    MQ3G256 = 17,
    /// Non-rotated (plain-basis) symmetric signed INT8, per-256-group f16 scale.
    /// On-disk block = [f16 scale][256 int8] = 258 B/group — same bytes as the
    /// canonical `Oq8G256 = 35` but WITHOUT the FWHT rotation, so the AIE2 int8
    /// W8A8/W8A16 projection kernel consumes it directly with no per-block
    /// activation rotation. Distinct id (and `"rotated": false` in metadata) so a
    /// rotated-OQ8 consumer can never mis-handle these bytes. NPU-only sidecar.
    Oq8Plain = 45,
    /// Non-rotated (plain-basis) MIXED Opus quant: int4 bulk + sparse int8
    /// overlay. Block = [f16 scale][128 nibbles][n_out × (u8 idx, i8 val)] =
    /// 130 + 2·n_out B/group. At n_out=3 that is 136 B/group = 4.25 b/w.
    /// Same layout as the canonical `OqPlusCompact = 36` but WITHOUT the FWHT
    /// rotation (see `Oq8Plain`), so the AIE2 W4A8 projection kernel consumes it
    /// with no per-block activation rotation. `n_out` is recoverable from the
    /// block length, and metadata carries it explicitly. NPU-only sidecar.
    Oq4MixedPlain = 46,
    /// Non-rotated (plain-basis) PURE int4. Block = [f16 scale][128 nibbles] =
    /// 130 B/group = 4.0625 b/w — the minimum-bandwidth weight format, and what
    /// the AIE2 W4A8 projection kernel wants. Distinct from the canonical ROTATED
    /// `Oq4G256 = 34` so a rotated consumer can never mis-handle these bytes.
    /// NPU-only sidecar.
    Oq4Plain = 47,
}

struct HfqTensor {
    name: String,
    quant_type: QuantType,
    shape: Vec<u32>,
    group_size: u32,
    data: Vec<u8>,
}

fn write_hfq(
    path: &Path,
    arch: u32,
    metadata_json: &str,
    tensors: &[HfqTensor],
) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    let metadata_bytes = metadata_json.as_bytes();

    let header_size = 32u64;
    let metadata_offset = header_size;
    let metadata_size = metadata_bytes.len() as u64;

    let index_offset = metadata_offset + metadata_size;
    let mut index_bytes = Vec::new();
    index_bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        let name_bytes = t.name.as_bytes();
        index_bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        index_bytes.extend_from_slice(name_bytes);
        index_bytes.push(t.quant_type as u8);
        index_bytes.push(t.shape.len() as u8);
        for &d in &t.shape {
            index_bytes.extend_from_slice(&d.to_le_bytes());
        }
        index_bytes.extend_from_slice(&t.group_size.to_le_bytes());
        index_bytes.extend_from_slice(&(t.data.len() as u64).to_le_bytes());
    }

    let data_start_unaligned = index_offset + index_bytes.len() as u64;
    let data_offset = (data_start_unaligned + 4095) & !4095;

    f.write_all(HFQ_MAGIC)?;
    f.write_all(&HFQ_VERSION.to_le_bytes())?;
    f.write_all(&arch.to_le_bytes())?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;

    f.write_all(metadata_bytes)?;
    f.write_all(&index_bytes)?;

    let pad_size = (data_offset - data_start_unaligned) as usize;
    f.write_all(&vec![0u8; pad_size])?;

    for t in tensors {
        f.write_all(&t.data)?;
    }

    Ok(())
}

// ─── Model discovery ───────────────────────────────────────────────────────

fn find_safetensors(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    files.sort();
    files
}

fn resolve_hf_cache_root(path: &Path) -> Option<PathBuf> {
    let snapshots_dir = path.join("snapshots");
    if !snapshots_dir.is_dir() {
        return None;
    }

    let refs_main = path.join("refs").join("main");
    if let Ok(revision) = std::fs::read_to_string(&refs_main) {
        let snapshot = snapshots_dir.join(revision.trim());
        if snapshot.join("config.json").exists() {
            return Some(snapshot);
        }
    }

    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&snapshots_dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path.join("config.json").exists())
        .collect();
    candidates.sort();
    candidates.into_iter().next()
}

fn resolve_model_path(input: &str) -> String {
    let path = Path::new(input);
    if path.join("config.json").exists() {
        return input.to_string();
    }

    if let Some(snapshot) = resolve_hf_cache_root(path) {
        return snapshot.to_string_lossy().into_owned();
    }

    if input.contains('/') {
        let parts: Vec<&str> = input.splitn(2, '/').collect();
        if parts.len() == 2 {
            let org = parts[0];
            let name = parts[1];
            let home = std::env::var("HOME").unwrap_or_default();
            let cache_root = PathBuf::from(format!(
                "{home}/.cache/huggingface/hub/models--{org}--{name}"
            ));
            if let Some(snapshot) = resolve_hf_cache_root(&cache_root) {
                return snapshot.to_string_lossy().into_owned();
            }
        }
    }
    input.to_string()
}

// ─── Tensor classification ────────────────────────────────────────────────

/// Returns true for tensors that must stay in F32 for numerical fidelity:
/// any RMSNorm weight. The rest (Q/K/V/O/fc/gate/up/down projections) can
/// use the selected draft weight format.
fn is_norm_tensor(name: &str) -> bool {
    name.contains("input_layernorm")
        || name.contains("post_attention_layernorm")
        || name.contains("q_norm")
        || name.contains("k_norm")
        || name == "hidden_norm.weight"
        || name == "norm.weight"
}

fn parse_int_array(json: &serde_json::Value) -> Vec<i64> {
    json.as_array()
        .map(|a| a.iter().filter_map(|v| v.as_i64()).collect())
        .unwrap_or_default()
}

// ─── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut input_dir: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut use_bf16 = false;
    let mut use_f16 = false;
    let mut keep_f32 = false;
    let mut use_mq4 = false;
    let mut use_mq6 = false;
    let mut use_mq3 = false;
    let mut use_oq8 = false;
    let mut use_oq4 = false;
    // Some(n_out) when a mixed --oq4.<bits> format was requested.
    let mut oq4_mixed_outliers: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                input_dir = Some(args[i + 1].clone());
                i += 2;
            }
            "--output" | "-o" => {
                output_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--keep-f32" => {
                keep_f32 = true;
                i += 1;
            }
            "--bf16" => {
                use_bf16 = true;
                i += 1;
            }
            "--f16" => {
                use_f16 = true;
                i += 1;
            }
            "--mq4" => {
                use_mq4 = true;
                i += 1;
            }
            "--mq6" => {
                use_mq6 = true;
                i += 1;
            }
            "--mq3" => {
                use_mq3 = true;
                i += 1;
            }
            "--oq8" => {
                use_oq8 = true;
                i += 1;
            }
            // PURE int4 (130 B/group, 4.0625 b/w). Matched before the
            // "--oq4." prefix arm below so it is not shadowed by it.
            "--oq4" => {
                use_oq4 = true;
                i += 1;
            }
            // Mixed-precision Opus quant, named by its exact storage width:
            // bits = 4.0625 + n_out/16, so --oq4.25 => 3 int8 overlays/group.
            // Accepts the whole canonical family (oq4.125 .. oq7.9375) rather
            // than hard-coding one width.
            other_fmt if other_fmt.starts_with("--oq4.") || other_fmt.starts_with("--oq5.") => {
                let bits_text = &other_fmt[2..].trim_start_matches("oq");
                match bits_text
                    .parse::<f32>()
                    .ok()
                    .and_then(mixed_outliers_for_bits)
                {
                    Some(n) => {
                        oq4_mixed_outliers = Some(n);
                        i += 1;
                    }
                    None => {
                        eprintln!(
                            "invalid mixed format {other_fmt}: storage bits must be \
                             4.0625 + n/16 for n in 1..=62 (e.g. --oq4.25 => 3 overlays)"
                        );
                        std::process::exit(1);
                    }
                }
            }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: dflash_convert --input <dir_or_hf_id> --output <file.hfq> \
                     [--bf16 | --f16 | --keep-f32 | --mq3 | --mq4 | --mq6 | --oq8 | --oq4 | --oq4.<bits>]"
                );
                eprintln!(
                    "  --oq4.<bits>  non-rotated mixed int4+int8 (e.g. --oq4.25 = 3 int8 \
                     overlays per 256-group, 136 B/group)"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let draft_format = DraftFormat::from_flags(
        use_f16,
        keep_f32,
        use_bf16,
        use_mq3,
        use_mq4,
        use_mq6,
        use_oq8,
        use_oq4,
        oq4_mixed_outliers,
    )
    .unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(1);
    });

    let input_dir = input_dir.expect("--input required");
    let output_path = output_path.expect("--output required");
    let input_dir = resolve_model_path(&input_dir);
    let input_dir = Path::new(&input_dir);
    let output_path = Path::new(&output_path);

    eprintln!("dflash_convert");
    eprintln!("  input : {}", input_dir.display());
    eprintln!("  output: {}", output_path.display());
    // For mixed OQ4 the exact overlay count / byte cost is dynamic, so print the
    // precise line; all fixed formats use the enum's static description().
    if let DraftFormat::Oq4Mixed(n) = draft_format {
        eprintln!(
            "  dtype : OQ{:.4}-plain (non-rotated int4 bulk + {n} int8 overlays per G256, \
             {} B/group), F32 (norms) — NPU int4 W4A8",
            4.0625 + n as f32 / 16.0,
            130 + 2 * n
        );
    } else {
        eprintln!("  dtype : {}", draft_format.description());
    }

    let config_path = input_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", config_path.display()));
    let config: serde_json::Value =
        serde_json::from_str(&config_str).expect("config.json parse failed");

    let architectures = config
        .get("architectures")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let is_dflash = architectures
        .iter()
        .any(|v| v.as_str() == Some("DFlashDraftModel"));
    if !is_dflash {
        eprintln!(
            "warning: config.json architectures = {architectures:?}; expected DFlashDraftModel"
        );
    }

    let dflash_cfg = config
        .get("dflash_config")
        .expect("config.json missing dflash_config block");
    let block_size = dflash_block_size(&config).unwrap_or_else(|error| panic!("{error}"));
    let mask_token_id = dflash_cfg
        .get("mask_token_id")
        .and_then(|v| v.as_u64())
        .expect("dflash_config missing mask_token_id") as u32;
    let target_layer_ids = parse_int_array(
        dflash_cfg
            .get("target_layer_ids")
            .expect("dflash_config missing target_layer_ids"),
    );
    let num_target_layers = config
        .get("num_target_layers")
        .and_then(|v| v.as_u64())
        .expect("config.json missing num_target_layers");

    let num_hidden_layers = config
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .expect("config.json missing num_hidden_layers") as usize;
    let hidden_size = config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .expect("config.json missing hidden_size") as usize;
    let num_attention_heads = config
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .expect("config.json missing num_attention_heads") as usize;
    let num_key_value_heads = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .expect("config.json missing num_key_value_heads") as usize;
    let head_dim = config
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(hidden_size / num_attention_heads);
    let intermediate_size = config
        .get("intermediate_size")
        .and_then(|v| v.as_u64())
        .expect("config.json missing intermediate_size") as usize;

    eprintln!(
        "  dflash: block_size={}, mask_token_id={}, target_layers={:?}, hidden_layers={}, hidden={}",
        block_size, mask_token_id, target_layer_ids, num_hidden_layers, hidden_size,
    );

    // Metadata JSON for the HFQ file.
    let draft_dtype = draft_format.metadata_name();
    // FWHT sign tables for MQ rotation. Seeds 42/1042 match the engine's
    // `hipfire_rdna::Gpu::ensure_mq_signs()` so quantized weights here can
    // be dequantized/used correctly on GPU at inference.
    let needs_fwht = matches!(
        draft_format,
        DraftFormat::Mq3 | DraftFormat::Mq4 | DraftFormat::Mq6
    );
    let signs1: Vec<f32> = if needs_fwht {
        gen_fwht_signs(42, 256)
    } else {
        Vec::new()
    };
    let signs2: Vec<f32> = if needs_fwht {
        gen_fwht_signs(1042, 256)
    } else {
        Vec::new()
    };
    let metadata = serde_json::json!({
        "architecture": "dflash",
        "config": config,
        "dflash": {
            "block_size": block_size,
            "mask_token_id": mask_token_id,
            "target_layer_ids": target_layer_ids,
            "num_target_layers": num_target_layers,
            "num_hidden_layers": num_hidden_layers,
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "intermediate_size": intermediate_size,
            "rms_norm_eps": config.get("rms_norm_eps").cloned().unwrap_or_else(|| serde_json::Value::from(1e-6)),
            "rope_theta": config.get("rope_theta").cloned().unwrap_or_else(|| serde_json::Value::from(10_000_000.0)),
            "vocab_size": config.get("vocab_size").cloned(),
            "draft_dtype": draft_dtype,
            // The -plain codecs (OQ8-plain, OQ4.x-plain) are non-rotated: the AIE2
            // kernel reads codes directly and must NOT rotate activations. Explicit
            // so a rotated consumer (canonical Oq8G256=35 / OqPlusCompact=36) can
            // never mis-handle these bytes.
            "rotated": if use_oq8 || use_oq4 || oq4_mixed_outliers.is_some() {
                serde_json::Value::Bool(false)
            } else {
                serde_json::Value::Null
            },
            // Mixed-precision descriptors. n_out is also recoverable from the block
            // length (130 + 2*n_out), but carrying it explicitly means a loader never
            // has to infer geometry from byte arithmetic.
            "mixed_outliers_per_group": match oq4_mixed_outliers {
                Some(n) => serde_json::Value::from(n),
                None => serde_json::Value::Null,
            },
            "mixed_storage_bits": match oq4_mixed_outliers {
                Some(n) => serde_json::Value::from(4.0625 + n as f64 / 16.0),
                None => serde_json::Value::Null,
            },
        },
        "tokenizer": serde_json::Value::Null,
    });
    let metadata_json = serde_json::to_string(&metadata).unwrap();

    // Load + convert all safetensors files (draft is typically one file).
    let st_files: Vec<SafetensorsFile> = find_safetensors(input_dir)
        .iter()
        .inspect(|p| eprintln!("  loading: {}", p.display()))
        .map(|p| SafetensorsFile::open(p).expect("safetensors open failed"))
        .collect();
    assert!(
        !st_files.is_empty(),
        "no .safetensors files found in input dir"
    );

    let mut name_to_file: Vec<(String, usize)> = Vec::new();
    for (fi, st) in st_files.iter().enumerate() {
        for name in st.tensor_names() {
            name_to_file.push((name, fi));
        }
    }
    name_to_file.sort_by_key(|(name, _)| name.clone());
    eprintln!("  tensors: {}", name_to_file.len());

    let mut hfq_tensors: Vec<HfqTensor> = Vec::with_capacity(name_to_file.len());
    let mut total_params = 0u64;
    let mut total_bytes_out = 0usize;

    for (name, fi) in &name_to_file {
        let (meta, raw) = st_files[*fi]
            .tensor_data(name)
            .expect("tensor lookup failed");
        let n_elements: usize = meta.shape.iter().product();
        total_params += n_elements as u64;

        let f32_data = to_f32(raw, &meta.dtype);

        // Classification rules:
        //   norms → always F32 (small, precision-critical).
        //   other (projections) → selected draft weight format. BF16 is the
        //                         precision-preserving default; F16 remains an
        //                         explicit compatibility artifact.
        // MQ divisibility: quantizers pad the final partial group with
        // zeros. That's safe for weights since the padded lanes are never read
        // at inference. We still require N ≥ 256 to ensure a full first group
        // (per-group scale/min carries meaning).
        let (quant_type, group_size, bytes) = if is_norm_tensor(name) {
            (QuantType::F32, 0u32, f32_slice_to_f32_bytes(&f32_data))
        } else if draft_format == DraftFormat::F32 {
            (QuantType::F32, 0u32, f32_slice_to_f32_bytes(&f32_data))
        } else if draft_format == DraftFormat::Mq4 && n_elements >= 256 {
            let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
            (QuantType::MQ4G256, 256u32, q)
        } else if draft_format == DraftFormat::Mq6 && n_elements >= 256 {
            let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
            (QuantType::MQ6G256, 256u32, q)
        } else if draft_format == DraftFormat::Mq3 && n_elements >= 256 {
            let q = quantize_mq3g256(&f32_data, &signs1, &signs2);
            (QuantType::MQ3G256, 256u32, q)
        } else if draft_format == DraftFormat::Oq8 && n_elements >= 256 {
            let q = quantize_oq8_plain(&f32_data);
            (QuantType::Oq8Plain, 256u32, q)
        } else if draft_format == DraftFormat::Oq4 && n_elements >= 256 {
            let q = quantize_oq4_plain(&f32_data);
            (QuantType::Oq4Plain, 256u32, q)
        } else if let DraftFormat::Oq4Mixed(n_out) = draft_format {
            if n_elements >= 256 {
                let q = quantize_oq4_mixed_plain(&f32_data, n_out);
                (QuantType::Oq4MixedPlain, 256u32, q)
            } else {
                (QuantType::BF16, 0u32, f32_slice_to_bf16_bytes(&f32_data))
            }
        } else if draft_format == DraftFormat::F16 {
            (QuantType::F16, 0u32, f32_slice_to_f16_bytes(&f32_data))
        } else {
            (QuantType::BF16, 0u32, f32_slice_to_bf16_bytes(&f32_data))
        };

        total_bytes_out += bytes.len();
        hfq_tensors.push(HfqTensor {
            name: name.clone(),
            quant_type,
            shape: meta.shape.iter().map(|d| *d as u32).collect(),
            group_size,
            data: bytes,
        });
    }

    eprintln!(
        "  total params: {:.3}B ({} tensors)",
        total_params as f64 / 1e9,
        hfq_tensors.len()
    );
    eprintln!(
        "  total out  : {:.2} MiB",
        total_bytes_out as f64 / (1024.0 * 1024.0)
    );

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("mkdir -p output parent");
        }
    }

    write_hfq(
        output_path,
        ARCH_ID_DFLASH_DRAFT,
        &metadata_json,
        &hfq_tensors,
    )
    .expect("write_hfq failed");

    eprintln!("  wrote: {}", output_path.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snr_db(w: &[f32], deq: &[f32]) -> f64 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (a, b) in w.iter().zip(deq) {
            num += ((*a - *b) as f64).powi(2);
            den += (*a as f64).powi(2);
        }
        10.0 * (den / num.max(1e-30)).log10()
    }

    #[test]
    fn oq8_plain_round_trip_smooth_is_near_lossless() {
        // Realistic (outlier-free) weight distribution: per-group int8 at 256
        // levels should clear ~45 dB. This is the weight-only quant ceiling the
        // W8A16 NPU path inherits (real 9B drafter measured ~33 dB end-to-end,
        // lower because errors compound across 5 layers + activation range).
        let n = 256 * 4;
        let w: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.011).sin()) * 0.06).collect();
        let packed = quantize_oq8_plain(&w);
        assert_eq!(packed.len(), 4 * (2 + 256));
        let deq = dequant_oq8_plain(&packed, n);
        let snr = snr_db(&w, &deq);
        assert!(snr > 44.0, "oq8-plain smooth SNR {snr:.1} dB too low");
    }

    #[test]
    fn oq8_plain_round_trip_with_outlier_is_int8_limited() {
        // A per-group outlier sets the group scale and crushes bulk resolution —
        // the known int8 limitation (not a bug). Documents the ~30 dB floor that
        // clip-search does NOT recover at 8 bits on real weights (measured).
        let n = 256 * 4;
        let w: Vec<f32> = (0..n)
            .map(|i| if i % 256 == 0 { 0.9 } else { (i as f32 * 0.017).sin() * 0.08 })
            .collect();
        let deq = dequant_oq8_plain(&quantize_oq8_plain(&w), n);
        let snr = snr_db(&w, &deq);
        assert!(snr > 30.0, "oq8-plain outlier SNR {snr:.1} dB unexpectedly low");
    }

    #[test]
    fn oq8_plain_zero_group_is_stable() {
        let w = vec![0.0f32; 256 * 2];
        let packed = quantize_oq8_plain(&w);
        let deq = dequant_oq8_plain(&packed, w.len());
        assert!(deq.iter().all(|&v| v == 0.0));
    }

    /// Decode the int4 bulk only, ignoring the overlay table. Same bytes, same
    /// scale — so diffing this against the full decode isolates exactly what the
    /// sparse int8 overlay contributes.
    fn dequant_bulk_only(data: &[u8], n: usize, n_out: usize) -> Vec<f32> {
        let block_bytes = 130 + 2 * n_out;
        let mut out = Vec::with_capacity(n);
        for b in 0..n.div_ceil(256) {
            let off = b * block_bytes;
            let scale = hipfire_primitives::conv::f16_to_f32(u16::from_le_bytes([
                data[off],
                data[off + 1],
            ]));
            for i in 0..128 {
                let byte = data[off + 2 + i];
                out.push(scale * ((((byte & 0xf) as i8) << 4 >> 4) as f32));
                out.push(scale * ((((byte >> 4) as i8) << 4 >> 4) as f32));
            }
        }
        out.truncate(n);
        out
    }

    #[test]
    fn mixed_bits_to_outliers_matches_canonical_formula() {
        // bits = 4.0625 + n_out/16. Must agree with
        // hipfire-quantize::main::parse_opus_mixed_format, or a sidecar and the
        // general pipeline would disagree about what "oq4.25" means.
        assert_eq!(mixed_outliers_for_bits(4.25), Some(3));
        assert_eq!(mixed_outliers_for_bits(4.125), Some(1));
        assert_eq!(mixed_outliers_for_bits(4.5), Some(7));
        // Not on the 1/16 lattice, or out of range.
        assert_eq!(mixed_outliers_for_bits(4.3), None);
        assert_eq!(mixed_outliers_for_bits(4.0625), None); // n_out = 0
        assert_eq!(mixed_outliers_for_bits(8.0), None);
    }

    #[test]
    fn oq4_25_block_geometry_is_136_bytes() {
        let n = 256 * 4;
        let w: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011).sin() * 0.06).collect();
        let packed = quantize_oq4_mixed_plain(&w, 3);
        // 130 + 2*3 = 136 B/group => 4.25 bits/weight.
        assert_eq!(packed.len(), 4 * 136);
        assert!((packed.len() as f32 * 8.0 / n as f32 - 4.25).abs() < 1e-6);
    }

    #[test]
    fn oq4_25_overlay_rescues_outlier_groups() {
        // THE POINT OF THE FORMAT. One large value per group sets the scale and
        // crushes int4 resolution for the other 255. Promoting the top-gain slots
        // to int8 should recover a large amount of SNR for 6 bytes per group.
        let n = 256 * 4;
        let w: Vec<f32> = (0..n)
            .map(|i| {
                if i % 256 == 0 {
                    0.9
                } else {
                    (i as f32 * 0.017).sin() * 0.08
                }
            })
            .collect();
        let packed = quantize_oq4_mixed_plain(&w, 3);
        let bulk = snr_db(&w, &dequant_bulk_only(&packed, n, 3));
        let full = snr_db(&w, &dequant_oq4_mixed_plain(&packed, n, 3));
        assert!(
            full > bulk + 6.0,
            "overlay bought only {:.1} dB (bulk {bulk:.1} -> full {full:.1}); \
             the 4.5% byte premium over pure int4 is not being earned",
            full - bulk
        );
    }

    #[test]
    fn oq4_25_sits_between_int4_and_int8_on_smooth_weights() {
        // Sanity on ordering: mixed must beat its own int4 bulk, and must not
        // beat int8 (it stores strictly less information than 8 bits/weight).
        let n = 256 * 4;
        let w: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011).sin() * 0.06).collect();
        let packed = quantize_oq4_mixed_plain(&w, 3);
        let bulk = snr_db(&w, &dequant_bulk_only(&packed, n, 3));
        let mixed = snr_db(&w, &dequant_oq4_mixed_plain(&packed, n, 3));
        let int8 = snr_db(&w, &dequant_oq8_plain(&quantize_oq8_plain(&w), n));
        assert!(mixed >= bulk, "mixed {mixed:.1} < its own bulk {bulk:.1}");
        assert!(
            mixed < int8,
            "mixed {mixed:.1} dB >= int8 {int8:.1} dB — implausible at 4.25 vs 8 b/w"
        );
    }

    #[test]
    fn oq4_25_zero_group_is_stable() {
        let w = vec![0.0f32; 256 * 2];
        let packed = quantize_oq4_mixed_plain(&w, 3);
        let deq = dequant_oq4_mixed_plain(&packed, w.len(), 3);
        assert!(deq.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn oq4_plain_block_geometry_is_130_bytes() {
        let n = 256 * 4;
        let w: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011).sin() * 0.06).collect();
        let packed = quantize_oq4_plain(&w);
        // [f16 scale][128 nibbles] = 130 B/group => 4.0625 bits/weight.
        assert_eq!(packed.len(), 4 * 130);
        assert!((packed.len() as f32 * 8.0 / n as f32 - 4.0625).abs() < 1e-6);
    }

    #[test]
    fn oq4_plain_is_cheaper_than_mixed_and_worse_than_int8() {
        // Documents the tradeoff the NPU cares about: pure W4 is the
        // minimum-BYTES format, and bytes are the binding constraint on the NPU
        // weight path. Quality ordering must be int4 < mixed < int8; byte
        // ordering must be the reverse.
        let n = 256 * 8;
        let w: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011).sin() * 0.06).collect();

        let p4 = quantize_oq4_plain(&w);
        let p425 = quantize_oq4_mixed_plain(&w, 3);
        let p8 = quantize_oq8_plain(&w);
        assert!(p4.len() < p425.len(), "pure W4 must be smaller than mixed");
        assert!(p425.len() < p8.len(), "mixed must be smaller than int8");

        let s4 = snr_db(&w, &dequant_oq4_plain(&p4, n));
        let s425 = snr_db(&w, &dequant_oq4_mixed_plain(&p425, n, 3));
        let s8 = snr_db(&w, &dequant_oq8_plain(&p8, n));
        assert!(s4 <= s425, "pure W4 {s4:.1} dB should not beat mixed {s425:.1}");
        assert!(s425 < s8, "mixed {s425:.1} dB should not beat int8 {s8:.1}");
        // Sanity floor only. Weight-level int4 round-trip on smooth data; the
        // ~22 dB int8->int4 gap is textbook (~5.5 dB/bit) and is NOT a defect.
        assert!(s4 > 15.0, "pure W4 SNR {s4:.1} dB implausibly low");
    }

    #[test]
    fn oq4_plain_zero_group_is_stable() {
        let w = vec![0.0f32; 256 * 2];
        let packed = quantize_oq4_plain(&w);
        assert!(dequant_oq4_plain(&packed, w.len())
            .iter()
            .all(|&v| v == 0.0));
    }

    #[test]
    fn oq4_plain_partial_final_group_round_trips() {
        let n = 256 * 2 + 100;
        let w: Vec<f32> = (0..n).map(|i| (i as f32 * 0.023).cos() * 0.05).collect();
        let packed = quantize_oq4_plain(&w);
        assert_eq!(packed.len(), 3 * 130);
        let deq = dequant_oq4_plain(&packed, n);
        assert_eq!(deq.len(), n);
        assert!(snr_db(&w, &deq) > 12.0);
    }

    #[test]
    fn oq4_25_partial_final_group_round_trips() {
        // 2.5 groups: the tail is zero-padded at encode and must not corrupt the
        // live lanes or panic on decode.
        let n = 256 * 2 + 100;
        let w: Vec<f32> = (0..n).map(|i| (i as f32 * 0.023).cos() * 0.05).collect();
        let packed = quantize_oq4_mixed_plain(&w, 3);
        assert_eq!(packed.len(), 3 * 136);
        let deq = dequant_oq4_mixed_plain(&packed, n, 3);
        assert_eq!(deq.len(), n);
        assert!(snr_db(&w, &deq) > 20.0);
    }
}
