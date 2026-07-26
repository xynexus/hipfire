// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Tensor payload encoders + a post-process quantizer for diffusion `.hfq`
//! artifacts. Reads a source artifact (whose weights decode to f32 via
//! [`cpu_tensor_from_hfq`]), re-encodes the large 2D+ `.weight` tensors into a
//! packed format, and copies every other entry (biases, norms, configs,
//! tokenizers) through verbatim. The decode path keys purely off each tensor's
//! `quant_type`, so the resulting artifact loads with no metadata changes beyond
//! the informational `weight_format` string.

use super::*;
use crate::quant_decode::{OQ_FWHT_SEED1, OQ_FWHT_SEED2};
use hipfire_quantize::codecs::{quantize_oq4g256, quantize_oq8g256};
use hipfire_quantize::gen_fwht_signs;
pub use hipfire_quantize::hessian_io::HessianSidecar;
use hipfire_quantize::ldlq::oq4_ldlq_pack;
use hipfire_runtime::hfq::{write_hfqm_package_streaming, HfqFile, HfqStreamEntry};
use std::path::Path;

/// Quantization formats this tool can emit. Both round-trip bit-exactly with the
/// matching decoder in `quant_decode.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionQuantFormat {
    /// q8_0: per-32 group, symmetric int8 with an f16 scale (34 bytes/block).
    Q8F16,
    /// Q4F16_G64: per-64 group, affine 4-bit with f16 scale+min (36 bytes/block).
    Q4F16G64,
    /// Q4F16_G64 storage, but encoded with per-group MSE clip-search (the `+` of
    /// `oq4+`): instead of using the raw group min/max, search the quantization
    /// range that minimizes reconstruction error, trading clipped outliers for
    /// finer resolution on the bulk of the distribution. Data-free.
    Q4F16G64Clip,
    /// Q4_K (llama.cpp k-quant): 256-superblock, 8x 32-element sub-blocks each
    /// with its own 6-bit scale+min under a per-superblock f16 d/dmin. Reuses the
    /// hipfire LLM-path codec; finer/hierarchical vs the flat group-64 affine.
    Q4K,
    /// Opus Quant 4-bit (FWHT-rotated, 256-group), RTN. Linears -> oq4, convs ->
    /// oq8 (convs are sensitive and have no Hessian). Data-free.
    Oq4,
    /// Opus Quant 4-bit, activation-calibrated: linears use LDLQ Hessian error
    /// feedback (oq4_ldlq_pack) when a `.calib.hfq` Hessian is available, else
    /// fall back to oq4 RTN; convs -> oq8. Requires `--calib`.
    Oq4PlusPlus,
    /// Opus Quant 8-bit (FWHT-rotated, 256-group), RTN. Near-lossless.
    Oq8,
    /// Plain (unrotated) unsigned **fold** format for the mixed-precision GEMM
    /// (`gemm_opus_tiled_wmma_u`): dense unsigned codes + per-group f32 scales,
    /// 256-group. Activation-aware **clip-calibrated** (the `+`) when a `.calib.hfq`
    /// imatrix is available, else RTN. The zero-point is folded out at GEMM time.
    /// 4/2/1-bit; 8-bit fold is unnecessary (oq8 is already near-lossless).
    OqFold4,
    OqFold2,
    OqFold1,
}

impl DiffusionQuantFormat {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "q8" | "q8f16" | "q80" => Some(Self::Q8F16),
            "q4" | "q4f16" | "q4f16g64" => Some(Self::Q4F16G64),
            "q4+" | "q4c" | "q4clip" | "q4f16g64clip" => Some(Self::Q4F16G64Clip),
            "q4k" | "q4_k" => Some(Self::Q4K),
            "oq4" => Some(Self::Oq4),
            "oq4+" | "oq4++" => Some(Self::Oq4PlusPlus),
            "oq8" => Some(Self::Oq8),
            "oqf4" | "oqfold4" => Some(Self::OqFold4),
            "oqf2" | "oqfold2" => Some(Self::OqFold2),
            "oqf1" | "oqfold1" => Some(Self::OqFold1),
            _ => None,
        }
    }

    /// Opus formats are FWHT-rotated, 256-group, and quantize per-tensor (linears
    /// vs convs differ), so they bypass the single-format `encode`/`quant_type`.
    fn is_opus(self) -> bool {
        matches!(self, Self::Oq4 | Self::Oq4PlusPlus | Self::Oq8)
    }

    /// Plain unsigned fold format (mixed-precision GEMM); returns the weight bit
    /// width. These bypass `encode`/`is_opus` and use `encode_fold_tensor`.
    fn fold_bits(self) -> Option<u32> {
        match self {
            Self::OqFold4 => Some(4),
            Self::OqFold2 => Some(2),
            Self::OqFold1 => Some(1),
            _ => None,
        }
    }

    fn is_fold(self) -> bool {
        self.fold_bits().is_some()
    }

    fn quant_type(self) -> u8 {
        match self {
            Self::Q8F16 => QT_DIFFUSION_TENSOR_Q8F16,
            Self::Q4F16G64 | Self::Q4F16G64Clip => QT_DIFFUSION_TENSOR_Q4F16_G64,
            Self::Q4K => QT_DIFFUSION_TENSOR_Q4_K,
            Self::Oq4 | Self::Oq4PlusPlus | Self::Oq8 => QT_DIFFUSION_TENSOR_OQ4_G256,
            Self::OqFold4 => QT_DIFFUSION_TENSOR_OQF_W4,
            Self::OqFold2 => QT_DIFFUSION_TENSOR_OQF_W2,
            Self::OqFold1 => QT_DIFFUSION_TENSOR_OQF_W1,
        }
    }

    fn group_size(self) -> u32 {
        match self {
            Self::Q8F16 => 32,
            Self::Q4F16G64 | Self::Q4F16G64Clip => 64,
            Self::Q4K
            | Self::Oq4
            | Self::Oq4PlusPlus
            | Self::Oq8
            | Self::OqFold4
            | Self::OqFold2
            | Self::OqFold1 => 256,
        }
    }

    fn weight_format_label(self) -> &'static str {
        match self {
            Self::Q8F16 => "q8",
            Self::Q4F16G64 => "q4",
            Self::Q4F16G64Clip => "q4+",
            Self::Q4K => "q4k",
            Self::Oq4 => "oq4",
            Self::Oq4PlusPlus => "oq4++",
            Self::Oq8 => "oq8",
            Self::OqFold4 => "oqf4",
            Self::OqFold2 => "oqf2",
            Self::OqFold1 => "oqf1",
        }
    }

    fn encode(self, data: &[f32]) -> Vec<u8> {
        match self {
            Self::Q8F16 => encode_q8f16(data),
            Self::Q4F16G64 => encode_q4f16_g64(data),
            Self::Q4F16G64Clip => encode_q4f16_g64_clipsearch(data),
            Self::Q4K => encode_q4k(data),
            // Opus/fold formats are handled per-tensor in quantize_diffusion_hfq.
            Self::Oq4 | Self::Oq4PlusPlus | Self::Oq8 => {
                unreachable!("opus uses encode_opus_tensor")
            }
            Self::OqFold4 | Self::OqFold2 | Self::OqFold1 => {
                unreachable!("fold uses encode_fold_tensor")
            }
        }
    }
}

/// Per-tensor Opus (oq4/oq8) encoding with optional LDLQ calibration.
/// Returns `(quant_type, group_size, bytes, ldlq_used)`. Convs (rank-4) and full
/// `oq8` runs go to oq8; linears go to oq4 (LDLQ when a Hessian is available,
/// else RTN).
#[allow(clippy::too_many_arguments)]
fn encode_opus_tensor(
    format: DiffusionQuantFormat,
    name: &str,
    shape: &[u32],
    data: &[f32],
    force_oq8: bool,
    calib: Option<&HessianSidecar>,
    signs1: &[f32],
    signs2: &[f32],
) -> (u8, u32, Vec<u8>, bool) {
    let is_conv = shape.len() == 4;
    if is_conv || force_oq8 || matches!(format, DiffusionQuantFormat::Oq8) {
        let bytes = quantize_oq8g256(data, signs1, signs2);
        return (QT_DIFFUSION_TENSOR_OQ8_G256, 256, bytes, false);
    }
    // Linear (rank-2) at oq4 / oq4++.
    let m = shape[0] as usize;
    let k = shape[1] as usize;
    if matches!(format, DiffusionQuantFormat::Oq4PlusPlus) && k % 256 == 0 && m * k == data.len() {
        if let Some(sc) = calib {
            let base = name.strip_suffix(".weight").unwrap_or(name);
            if let Some(href) = sc.get(base, 0) {
                if href.k == k {
                    let h: Vec<f32> = (0..k * k).map(|i| href.at(i / k, i % k) as f32).collect();
                    let diag: f64 = (0..k).map(|i| h[i * k + i] as f64).sum();
                    let damp = 0.01 * (diag / k as f64).max(1e-12);
                    if let Some(packed) = oq4_ldlq_pack(data, m, k, &h, signs1, signs2, damp) {
                        return (QT_DIFFUSION_TENSOR_OQ4_G256, 256, packed, true);
                    }
                }
            }
        }
    }
    let bytes = quantize_oq4g256(data, signs1, signs2);
    (QT_DIFFUSION_TENSOR_OQ4_G256, 256, bytes, false)
}

/// q8_0 encoder: groups of 32, symmetric int8, `scale = max_abs / 127`, stored
/// as `[f16 scale][32 x i8]` (34 bytes/block). Mirrors `dequant_q8f16`.
pub(crate) fn encode_q8f16(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len().div_ceil(32) * 34);
    for group in data.chunks(32) {
        let max_abs = group.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        bytes.extend_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
        for idx in 0..32 {
            let value = group.get(idx).copied().unwrap_or(0.0);
            let quantized = (value / scale).round().clamp(-128.0, 127.0) as i8;
            bytes.push(quantized as u8);
        }
    }
    bytes
}

/// Q4F16_G64 encoder: groups of 64, affine 4-bit, `scale = (max-min)/15`, stored
/// as `[f16 scale][f16 min][32 packed bytes]` (36 bytes/block) with the low 32
/// values in the low nibbles and the high 32 in the high nibbles. Mirrors
/// `decode_q4f16_g64_slice`.
pub(crate) fn encode_q4f16_g64(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len().div_ceil(64) * 36);
    for group in data.chunks(64) {
        let min = group.iter().copied().fold(f32::INFINITY, f32::min);
        let max = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if max > min { (max - min) / 15.0 } else { 1.0 };
        bytes.extend_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
        bytes.extend_from_slice(&f32_to_f16_bits(min).to_le_bytes());
        for idx in 0..32 {
            let lo = group.get(idx).copied().unwrap_or(min);
            let hi = group.get(idx + 32).copied().unwrap_or(min);
            let lo_q = ((lo - min) / scale).round().clamp(0.0, 15.0) as u8;
            let hi_q = ((hi - min) / scale).round().clamp(0.0, 15.0) as u8;
            bytes.push(lo_q | (hi_q << 4));
        }
    }
    bytes
}

/// Pack one 64-element group into a Q4F16_G64 block (`[f16 scale][f16 min][32
/// packed bytes]`) given an explicit affine range [`lo`, `lo + 15*scale`]. Mirrors
/// `decode_q4f16_g64_slice` so it round-trips bit-for-bit.
fn pack_q4_block(group: &[f32], lo: f32, scale: f32) -> [u8; 36] {
    let mut block = [0u8; 36];
    block[0..2].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
    block[2..4].copy_from_slice(&f32_to_f16_bits(lo).to_le_bytes());
    let q = |v: f32| ((v - lo) / scale).round().clamp(0.0, 15.0) as u8;
    for idx in 0..32 {
        let lo_q = q(group.get(idx).copied().unwrap_or(lo));
        let hi_q = q(group.get(idx + 32).copied().unwrap_or(lo));
        block[4 + idx] = lo_q | (hi_q << 4);
    }
    block
}

/// Reconstruction MSE of a group quantized to the affine range [`lo`, `lo +
/// 15*scale`], using the same f16-rounded scale/min the decoder will see so the
/// search optimizes the value that is actually stored.
fn q4_group_mse(group: &[f32], lo: f32, scale: f32) -> f32 {
    let lo = f16_bits_to_f32(f32_to_f16_bits(lo));
    let scale = f16_bits_to_f32(f32_to_f16_bits(scale)).max(1e-12);
    group
        .iter()
        .map(|&v| {
            let q = ((v - lo) / scale).round().clamp(0.0, 15.0);
            let recon = lo + q * scale;
            (v - recon) * (v - recon)
        })
        .sum()
}

/// Calibrated Q4F16_G64 encoder (the `+` in `oq4+`): per 64-group, search the
/// quantization range that minimizes reconstruction MSE rather than using the
/// raw min/max. The range is shrunk symmetrically around the group midpoint over
/// a grid of clip ratios; tighter ranges give finer resolution on the bulk of
/// the values at the cost of clipping outliers, which is a net win whenever the
/// group has heavy tails. Data-free (weight-only). rayon-parallel over groups.
pub(crate) fn encode_q4f16_g64_clipsearch(data: &[f32]) -> Vec<u8> {
    use rayon::prelude::*;
    // 17 clip ratios from 1.0 (raw min/max) down to 0.2 of the half-range.
    const RATIOS: usize = 17;
    let blocks: Vec<[u8; 36]> = data
        .par_chunks(64)
        .map(|group| {
            let min = group.iter().copied().fold(f32::INFINITY, f32::min);
            let max = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if !(max > min) {
                return pack_q4_block(group, min, 1.0);
            }
            let mid = 0.5 * (min + max);
            let half = 0.5 * (max - min);
            let mut best_lo = min;
            let mut best_scale = (max - min) / 15.0;
            let mut best_mse = q4_group_mse(group, best_lo, best_scale);
            for step in 1..RATIOS {
                let ratio = 1.0 - 0.8 * (step as f32) / ((RATIOS - 1) as f32);
                let lo = mid - ratio * half;
                let hi = mid + ratio * half;
                let scale = (hi - lo) / 15.0;
                if !(scale > 0.0) {
                    continue;
                }
                let mse = q4_group_mse(group, lo, scale);
                if mse < best_mse {
                    best_mse = mse;
                    best_lo = lo;
                    best_scale = scale;
                }
            }
            pack_q4_block(group, best_lo, best_scale)
        })
        .collect();
    let mut bytes = Vec::with_capacity(blocks.len() * 36);
    for block in blocks {
        bytes.extend_from_slice(&block);
    }
    bytes
}

/// Q4_K encoder ported from `hipfire_quantize::codecs::quantize_q4k` (the proven
/// LLM-path codec). 256-element super-blocks with 8 sub-blocks of 32, each with
/// its own 6-bit scale+min under a per-super-block f16 `d`/`dmin` — finer and
/// hierarchical vs the flat group-64 affine of `encode_q4f16_g64`. The byte
/// layout must match `hipfire_runtime::quant::dequant_q4k` (the diffusion
/// decoder); `q4k_encoder_round_trips_through_diffusion_decoder` guards that.
pub(crate) fn encode_q4k(f32_data: &[f32]) -> Vec<u8> {
    let super_block_size = 256;
    let block_bytes = 144;
    let n = f32_data.len();
    let n_blocks = n.div_ceil(super_block_size);
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let sb_start = b * super_block_size;
        let sb_end = (sb_start + super_block_size).min(n);
        let out_off = b * block_bytes;

        let mut sub_scales = [0.0f32; 8];
        let mut sub_mins = [0.0f32; 8];
        for sb in 0..8 {
            let start = sb_start + sb * 32;
            let end = (start + 32).min(sb_end);
            if start >= sb_end {
                break;
            }
            let group = &f32_data[start..end];
            let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = max_val - min_val;
            sub_scales[sb] = if range > 0.0 { range / 15.0 } else { 0.0 };
            sub_mins[sb] = min_val;
        }

        let max_scale = sub_scales.iter().cloned().fold(0.0f32, f32::max);
        let max_min = sub_mins.iter().map(|m| -m).fold(0.0f32, f32::max);
        let d = if max_scale > 0.0 {
            max_scale / 63.0
        } else {
            0.0
        };
        let dmin = if max_min > 0.0 { max_min / 63.0 } else { 0.0 };
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
        let inv_dmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };

        let mut scale_ints = [0u8; 8];
        let mut min_ints = [0u8; 8];
        for sb in 0..8 {
            scale_ints[sb] = (sub_scales[sb] * inv_d + 0.5).min(63.0) as u8;
            min_ints[sb] = ((-sub_mins[sb]) * inv_dmin + 0.5).min(63.0) as u8;
        }

        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        output[out_off + 2..out_off + 4].copy_from_slice(&f32_to_f16_bits(dmin).to_le_bytes());

        let sc = &mut output[out_off + 4..out_off + 16];
        for i in 0..4 {
            sc[i] = (scale_ints[i] & 63) | ((scale_ints[4 + i] >> 4) << 6);
            sc[4 + i] = (min_ints[i] & 63) | ((min_ints[4 + i] >> 4) << 6);
        }
        for i in 0..4 {
            sc[8 + i] = (scale_ints[4 + i] & 0xF) | ((min_ints[4 + i] & 0xF) << 4);
        }

        let qs = &mut output[out_off + 16..out_off + 144];
        for group in 0..4 {
            let sb_even = group * 2;
            let sb_odd = group * 2 + 1;
            let eff_scale_e = d * scale_ints[sb_even] as f32;
            let eff_min_e = dmin * min_ints[sb_even] as f32;
            let inv_se = if eff_scale_e > 0.0 {
                1.0 / eff_scale_e
            } else {
                0.0
            };
            let eff_scale_o = d * scale_ints[sb_odd] as f32;
            let eff_min_o = dmin * min_ints[sb_odd] as f32;
            let inv_so = if eff_scale_o > 0.0 {
                1.0 / eff_scale_o
            } else {
                0.0
            };
            for l in 0..32 {
                let idx_e = sb_start + group * 64 + l;
                let idx_o = sb_start + group * 64 + 32 + l;
                let val_e = if idx_e < sb_end { f32_data[idx_e] } else { 0.0 };
                let val_o = if idx_o < sb_end { f32_data[idx_o] } else { 0.0 };
                let q_e = ((val_e + eff_min_e) * inv_se + 0.5).clamp(0.0, 15.0) as u8;
                let q_o = ((val_o + eff_min_o) * inv_so + 0.5).clamp(0.0, 15.0) as u8;
                qs[group * 32 + l] = q_e | (q_o << 4);
            }
        }
    }
    output
}

/// True when a tensor entry is a large weight matrix worth quantizing: a
/// `.weight` with rank >= 2 (conv 4D / linear 2D), excluding 1D norm/bias
/// vectors which are cheap and precision-sensitive. Configs/tokenizers (rank-1
/// byte blobs) are excluded by the rank check.
fn is_quantizable_weight(name: &str, shape: &[u32]) -> bool {
    name.ends_with(".weight") && shape.len() >= 2 && shape.iter().all(|&d| d > 0)
}

fn is_opus_quantizable_weight(name: &str, shape: &[u32]) -> bool {
    is_quantizable_weight(name, shape) && name.starts_with("transformer/tensors/")
}

fn opus_precision_class(arch_id: u32, name: &str) -> hipfire_arch_api::PrecisionClass {
    use hipfire_arch_api::{default_precision_class, mmdit_role, ArchId};
    u16::try_from(arch_id)
        .ok()
        .and_then(|id| hipfire_archs::registry().get(ArchId(id)))
        .and_then(|arch| arch.caps.ingest)
        .map(|ingest| ingest.precision_class(name))
        .unwrap_or_else(|| default_precision_class(mmdit_role(name)))
}

fn opus_should_quantize(
    arch_id: u32,
    format: DiffusionQuantFormat,
    name: &str,
    shape: &[u32],
) -> bool {
    if !is_opus_quantizable_weight(name, shape) {
        return false;
    }
    !matches!(format, DiffusionQuantFormat::Oq8)
        || opus_precision_class(arch_id, name) < hipfire_arch_api::PrecisionClass::High
}

fn opus_uses_oq8(arch_id: u32, format: DiffusionQuantFormat, name: &str, shape: &[u32]) -> bool {
    if shape.len() == 4 || matches!(format, DiffusionQuantFormat::Oq8) {
        return true;
    }
    if !matches!(
        format,
        DiffusionQuantFormat::Oq4 | DiffusionQuantFormat::Oq4PlusPlus
    ) {
        return false;
    }
    opus_precision_class(arch_id, name) >= hipfire_arch_api::PrecisionClass::High
}

#[derive(Debug, Default)]
pub struct DiffusionQuantizeSummary {
    pub quantized_tensors: usize,
    pub copied_tensors: usize,
    /// Linears packed with LDLQ Hessian error feedback (Opus calibrated path).
    pub ldlq_tensors: usize,
    pub source_bytes: u64,
    pub output_bytes: u64,
}

/// Re-encode the weight tensors of `source` into `format`, copying all other
/// entries verbatim, and write the result to `output`. For the Opus formats
/// (`oq4`/`oq4++`/`oq8`), `calib` (a `.calib.hfq` opened via `HessianSidecar`)
/// supplies per-linear Hessians for LDLQ error feedback.
pub fn quantize_diffusion_hfq(
    source: &Path,
    output: &Path,
    format: DiffusionQuantFormat,
    calib: Option<&HessianSidecar>,
) -> anyhow::Result<DiffusionQuantizeSummary> {
    let hfq = HfqFile::open(source)?;
    let mut summary = DiffusionQuantizeSummary {
        source_bytes: std::fs::metadata(source)?.len(),
        ..Default::default()
    };
    let (signs1, signs2) = if format.is_opus() {
        (
            gen_fwht_signs(OQ_FWHT_SEED1, 256),
            gen_fwht_signs(OQ_FWHT_SEED2, 256),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    let infos = hfq.tensors().to_vec();
    let mut entries = Vec::with_capacity(infos.len());
    for info in &infos {
        let quantized = if format.is_opus() {
            opus_should_quantize(hfq.arch_id, format, &info.name, &info.shape)
        } else if format.is_fold() {
            fold_should_quantize(hfq.arch_id, &info.name, &info.shape)
        } else {
            is_quantizable_weight(&info.name, &info.shape)
        };
        let (quant_type, group_size, data_len) = if quantized {
            summary.quantized_tensors += 1;
            let elements = info.shape.iter().try_fold(1usize, |acc, &dim| {
                acc.checked_mul(dim as usize)
                    .ok_or_else(|| anyhow::anyhow!("tensor {:?} size overflows", info.name))
            })?;
            let (quant_type, group_size) = if format.is_opus() {
                let oq8 = opus_uses_oq8(hfq.arch_id, format, &info.name, &info.shape);
                (
                    if oq8 {
                        QT_DIFFUSION_TENSOR_OQ8_G256
                    } else {
                        QT_DIFFUSION_TENSOR_OQ4_G256
                    },
                    256,
                )
            } else {
                (format.quant_type(), format.group_size())
            };
            (
                quant_type,
                group_size,
                encoded_payload_len(format, elements, quant_type)?,
            )
        } else {
            summary.copied_tensors += 1;
            (info.quant_type, info.group_size, info.data_size as u64)
        };
        entries.push(HfqStreamEntry {
            name: info.name.clone(),
            quant_type,
            shape: info.shape.clone(),
            group_size,
            data_len,
        });
    }

    // Update the informational weight_format string; per-tensor decoding keys off
    // quant_type, so this does not affect loading. Stream one source tensor at a
    // time: retaining all encoded payloads made 4B diffusion packs swap heavily.
    let metadata_json = rewrite_weight_format(&hfq.metadata_json, format.weight_format_label());
    write_hfqm_package_streaming(
        output,
        hfq.arch_id,
        &metadata_json,
        &entries,
        |index, writer| {
            let info = &infos[index];
            let quantized = if format.is_opus() {
                opus_should_quantize(hfq.arch_id, format, &info.name, &info.shape)
            } else if format.is_fold() {
                fold_should_quantize(hfq.arch_id, &info.name, &info.shape)
            } else {
                is_quantizable_weight(&info.name, &info.shape)
            };
            if !quantized {
                let (_, bytes) = hfq.tensor_data_vec(&info.name).ok_or_else(|| {
                    std::io::Error::other(format!(
                        "tensor {:?} vanished from source index",
                        info.name
                    ))
                })?;
                return writer.write_all(&bytes);
            }
            let decoded = cpu_tensor_from_hfq(&hfq, &info.name).map_err(|error| {
                std::io::Error::other(format!("decode {:?}: {error}", info.name))
            })?;
            let data = if format.is_opus() {
                let (_, _, bytes, ldlq) = encode_opus_tensor(
                    format,
                    &info.name,
                    &info.shape,
                    &decoded.data,
                    opus_uses_oq8(hfq.arch_id, format, &info.name, &info.shape),
                    calib,
                    &signs1,
                    &signs2,
                );
                summary.ldlq_tensors += usize::from(ldlq);
                bytes
            } else if let Some(bits) = format.fold_bits() {
                encode_fold_tensor(bits, &info.name, &decoded.data, calib)
            } else {
                format.encode(&decoded.data)
            };
            writer.write_all(&data)
        },
    )?;
    summary.output_bytes = std::fs::metadata(output)?.len();
    Ok(summary)
}

/// Per-tensor weight reconstruction error between two diffusion `.hfq` artifacts,
/// over the quantizable `transformer/tensors/*.weight` set. Both sides decode to
/// f32 (so an on-disk oq8/oq4 tensor is compared against its bf16 reference in
/// *dequantized weight space*), and the error is summarized per tensor.
///
/// This is the sampler-independent quant-quality signal: if `rel_rms` is tiny
/// everywhere, the quantization is faithful and any rendered-image drift is
/// trajectory divergence (diffusion chaos), not weight corruption. Tensors that
/// were copied verbatim (bf16 on both sides) report zero error, so the nonzero
/// rows are exactly the tensors the quantizer actually touched.
#[derive(Debug, Clone)]
pub struct TensorQuantDiff {
    pub name: String,
    pub elements: usize,
    pub quant_type_ref: u8,
    pub quant_type_cand: u8,
    /// Mean absolute error over all elements.
    pub mae: f64,
    /// Maximum absolute error.
    pub max_abs: f64,
    /// Root-mean-square error.
    pub rms: f64,
    /// RMS error / reference-tensor RMS (relative L2); 0 when the reference tensor
    /// is all-zero.
    pub rel_rms: f64,
}

/// Compare the quantizable transformer weights of `reference` and `candidate`,
/// returning `(per-tensor diffs, warnings)`. Warnings collect tensors present in
/// the reference but absent/shape-mismatched in the candidate (skipped, not
/// fatal).
pub fn diff_quantized_transformer_tensors(
    reference: &Path,
    candidate: &Path,
) -> anyhow::Result<(Vec<TensorQuantDiff>, Vec<String>)> {
    let ref_hfq = HfqFile::open(reference)
        .map_err(|e| anyhow::anyhow!("open reference {reference:?}: {e}"))?;
    let cand_hfq = HfqFile::open(candidate)
        .map_err(|e| anyhow::anyhow!("open candidate {candidate:?}: {e}"))?;
    let cand_types: std::collections::HashMap<&str, u8> = cand_hfq
        .tensors()
        .iter()
        .map(|t| (t.name.as_str(), t.quant_type))
        .collect();

    let mut diffs = Vec::new();
    let mut warnings = Vec::new();
    for info in ref_hfq.tensors() {
        if !is_opus_quantizable_weight(&info.name, &info.shape) {
            continue;
        }
        let Some(&cand_type) = cand_types.get(info.name.as_str()) else {
            warnings.push(format!("{}: absent from candidate", info.name));
            continue;
        };
        let a = cpu_tensor_from_hfq(&ref_hfq, &info.name)
            .map_err(|e| anyhow::anyhow!("decode reference {}: {e}", info.name))?;
        let b = cpu_tensor_from_hfq(&cand_hfq, &info.name)
            .map_err(|e| anyhow::anyhow!("decode candidate {}: {e}", info.name))?;
        if a.data.len() != b.data.len() {
            warnings.push(format!(
                "{}: element count {} (ref) != {} (cand); skipped",
                info.name,
                a.data.len(),
                b.data.len()
            ));
            continue;
        }
        let n = a.data.len();
        if n == 0 {
            continue;
        }
        let mut sum_abs = 0.0f64;
        let mut max_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut ref_sum_sq = 0.0f64;
        for (&x, &y) in a.data.iter().zip(&b.data) {
            let d = (x - y) as f64;
            let ad = d.abs();
            sum_abs += ad;
            if ad > max_abs {
                max_abs = ad;
            }
            sum_sq += d * d;
            ref_sum_sq += (x as f64) * (x as f64);
        }
        let nf = n as f64;
        let rms = (sum_sq / nf).sqrt();
        let ref_rms = (ref_sum_sq / nf).sqrt();
        let rel_rms = if ref_rms > 0.0 { rms / ref_rms } else { 0.0 };
        diffs.push(TensorQuantDiff {
            name: info.name.clone(),
            elements: n,
            quant_type_ref: info.quant_type,
            quant_type_cand: cand_type,
            mae: sum_abs / nf,
            max_abs,
            rms,
            rel_rms,
        });
    }
    Ok((diffs, warnings))
}

/// One tensor's RTN-vs-clip calibration comparison for the fold format.
#[derive(Debug, Clone)]
pub struct FoldCalibRow {
    pub name: String,
    pub elements: usize,
    pub has_imatrix: bool,
    /// imatrix-weighted relative RMSE — the clip objective (unweighted when the
    /// tensor has no imatrix).
    pub rtn_weighted: f64,
    pub clip_weighted: f64,
    /// plain (unweighted) relative RMSE, for reference.
    pub rtn_unweighted: f64,
    pub clip_unweighted: f64,
}

/// For each fold-eligible transformer linear in `source`, quantize its bf16
/// weights to `bits` with RTN vs activation-aware clip (using the `.calib.hfq`
/// imatrix) and report the reconstruction error under both. Weight-space only,
/// no GPU — quantifies the calibration `+` before the consume path lands.
pub fn eval_fold_calibration(
    source: &Path,
    calib_path: &Path,
    bits: u32,
) -> anyhow::Result<Vec<FoldCalibRow>> {
    use hipfire_quantize::opus_lowbit::{
        quantize_symmetric, quantize_symmetric_clip, weighted_quant_error,
    };
    use rayon::prelude::*;
    const GROUP: usize = 256;
    let hfq = HfqFile::open(source).map_err(|e| anyhow::anyhow!("open source {source:?}: {e}"))?;
    let calib = open_calib_sidecar(calib_path)?;
    // Phase 1 (sequential I/O): decode each fold-eligible weight + its imatrix.
    let mut work: Vec<(String, Vec<f32>, Option<Vec<f32>>)> = Vec::new();
    for info in hfq.tensors() {
        if !fold_should_quantize(hfq.arch_id, &info.name, &info.shape) {
            continue;
        }
        let data = cpu_tensor_from_hfq(&hfq, &info.name)
            .map_err(|e| anyhow::anyhow!("decode {}: {e}", info.name))?
            .data;
        let base = info.name.strip_suffix(".weight").unwrap_or(&info.name);
        let imatrix: Option<Vec<f32>> = calib
            .imatrix(base)
            .filter(|im| im.k % GROUP == 0 && data.len() % im.k == 0)
            .map(|im| im.iter_f32().collect());
        work.push((info.name.clone(), data, imatrix));
    }
    // Phase 2 (parallel compute): RTN vs clip quantization + weighted errors.
    let rows = work
        .par_iter()
        .map(|(name, data, imatrix)| {
            let im = imatrix.as_deref();
            let (rtn_c, rtn_s) = quantize_symmetric(data, GROUP, bits);
            let (clip_c, clip_s) = quantize_symmetric_clip(data, GROUP, bits, im, 12, 0.2);
            FoldCalibRow {
                name: name.clone(),
                elements: data.len(),
                has_imatrix: im.is_some(),
                rtn_weighted: weighted_quant_error(data, &rtn_c, &rtn_s, GROUP, bits, im),
                clip_weighted: weighted_quant_error(data, &clip_c, &clip_s, GROUP, bits, im),
                rtn_unweighted: weighted_quant_error(data, &rtn_c, &rtn_s, GROUP, bits, None),
                clip_unweighted: weighted_quant_error(data, &clip_c, &clip_s, GROUP, bits, None),
            }
        })
        .collect();
    Ok(rows)
}

/// Fold-format eligibility (mixed-precision policy). A tensor is fold-quantized
/// only when it is a transformer 2-D linear with `K % 256 == 0` (the fold GEMM
/// needs `K % group == 0`, so `x_embedder` K=128 is excluded) **and** the arch
/// spec marks it below `High` precision — i.e. the *tolerant* tensors (the FF
/// up-projections). Sensitive roles (attention, residual writers, embeddings,
/// modulation, `proj_out`) stay bf16. This mirrors the near-lossless allocation
/// the oq8 experiment validated; uniform fold-everything is too lossy.
fn fold_should_quantize(arch_id: u32, name: &str, shape: &[u32]) -> bool {
    // Base eligibility: transformer 2-D linear, fold-GEMM-compatible input dim.
    if !(is_opus_quantizable_weight(name, shape)
        && shape.len() == 2
        && (shape[1] as usize) % 256 == 0)
    {
        return false;
    }
    // Data-driven allocation: HIPFIRE_DIFFUSION_FOLD_ROLES=<space-separated name
    // substrings> overrides the static precision gate, so a sensitivity-ablation
    // result can select exactly which roles to fold (e.g. attn Q/K/V, which the
    // static map protects but the ablation showed are tolerant). Unset ⇒ the
    // conservative default: only tensors the arch spec marks below `High`.
    match std::env::var("HIPFIRE_DIFFUSION_FOLD_ROLES") {
        Ok(roles) if !roles.trim().is_empty() => roles.split_whitespace().any(|s| name.contains(s)),
        _ => opus_precision_class(arch_id, name) < hipfire_arch_api::PrecisionClass::High,
    }
}

/// Per-tensor plain unsigned **fold** encoding for the mixed-precision GEMM.
/// Uses the `.calib.hfq` imatrix for activation-aware clip (the `+`) when present
/// and shaped for this tensor, else RTN. Output blob: `[dense codes | f32 scales]`.
fn encode_fold_tensor(
    bits: u32,
    name: &str,
    data: &[f32],
    calib: Option<&HessianSidecar>,
) -> Vec<u8> {
    use hipfire_quantize::opus_lowbit::{pack_dense, quantize_symmetric, quantize_symmetric_clip};
    const GROUP: usize = 256;
    let base = name.strip_suffix(".weight").unwrap_or(name);
    let imatrix: Option<Vec<f32>> = calib
        .and_then(|sc| sc.imatrix(base))
        .filter(|im| im.k % GROUP == 0 && data.len() % im.k == 0)
        .map(|im| im.iter_f32().collect());
    let (codes, scales) = match imatrix.as_deref() {
        Some(im) => quantize_symmetric_clip(data, GROUP, bits, Some(im), 12, 0.2),
        None => quantize_symmetric(data, GROUP, bits),
    };
    let mut blob = pack_dense(&codes, bits);
    for s in &scales {
        blob.extend_from_slice(&s.to_le_bytes());
    }
    blob
}

#[cfg(test)]
mod fold_tests {
    use super::*;

    #[test]
    fn fold_encode_decode_round_trips_rtn() {
        let (m, k, bits) = (2usize, 512usize, 4u32);
        let data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.011).sin()).collect();
        let blob = encode_fold_tensor(bits, "t.weight", &data, None);

        // Blob length matches the header estimate.
        let expected = encoded_payload_len(
            DiffusionQuantFormat::OqFold4,
            m * k,
            QT_DIFFUSION_TENSOR_OQF_W4,
        )
        .unwrap();
        assert_eq!(blob.len() as u64, expected);

        // decode_oqf_slice reconstructs the RTN dequant bit-for-bit.
        let decoded = crate::quant_decode::decode_oqf_slice("t", &blob, m * k, bits).unwrap();
        let (codes, scales) = hipfire_quantize::opus_lowbit::quantize_symmetric(&data, 256, bits);
        let z = 1i32 << (bits - 1);
        for i in 0..m * k {
            let want = (codes[i] as i32 - z) as f32 * scales[i / 256];
            assert!((decoded[i] - want).abs() < 1e-6, "mismatch at {i}");
        }
    }

    #[test]
    fn fold_only_quantizes_tolerant_256_aligned_transformer_linears() {
        let arch = hipfire_arch_api::ARCH_ID_FLUX2;
        // Tolerant (Compressed) FF up-projection, K%256==0: yes.
        assert!(fold_should_quantize(
            arch,
            "transformer/tensors/transformer_blocks.0.ff.linear_in.weight",
            &[9216, 3072]
        ));
        // Sensitive (High) attention out-projection: no — stays bf16.
        assert!(!fold_should_quantize(
            arch,
            "transformer/tensors/transformer_blocks.0.attn.to_out.0.weight",
            &[3072, 3072]
        ));
        // x_embedder (K=128): no. Non-transformer: no.
        assert!(!fold_should_quantize(
            arch,
            "transformer/tensors/x_embedder.weight",
            &[3072, 128]
        ));
        assert!(!fold_should_quantize(
            arch,
            "text_encoder/tensors/foo.weight",
            &[512, 512]
        ));
    }
}

fn encoded_payload_len(
    format: DiffusionQuantFormat,
    elements: usize,
    quant_type: u8,
) -> anyhow::Result<u64> {
    let bytes = match format {
        DiffusionQuantFormat::Q8F16 => elements.div_ceil(32).checked_mul(34),
        DiffusionQuantFormat::Q4F16G64 | DiffusionQuantFormat::Q4F16G64Clip => {
            elements.div_ceil(64).checked_mul(36)
        }
        DiffusionQuantFormat::Q4K => elements.div_ceil(256).checked_mul(144),
        DiffusionQuantFormat::Oq8 => elements.div_ceil(256).checked_mul(258),
        DiffusionQuantFormat::Oq4 | DiffusionQuantFormat::Oq4PlusPlus => {
            let block_bytes = if quant_type == QT_DIFFUSION_TENSOR_OQ8_G256 {
                258
            } else {
                debug_assert_eq!(quant_type, QT_DIFFUSION_TENSOR_OQ4_G256);
                130
            };
            elements.div_ceil(256).checked_mul(block_bytes)
        }
        DiffusionQuantFormat::OqFold4
        | DiffusionQuantFormat::OqFold2
        | DiffusionQuantFormat::OqFold1 => {
            // [dense packed codes (elements*bits/8) | f32 per-group scales (ng*4)]
            let bits = format.fold_bits().unwrap() as usize;
            elements
                .checked_mul(bits)
                .map(|b| b / 8)
                .and_then(|packed| elements.div_ceil(256).checked_mul(4).map(|sc| packed + sc))
        }
    }
    .ok_or_else(|| anyhow::anyhow!("encoded diffusion tensor size overflows"))?;
    u64::try_from(bytes).map_err(|_| anyhow::anyhow!("encoded diffusion tensor size exceeds u64"))
}

#[cfg(test)]
mod streaming_quantize_tests {
    use super::*;
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};

    #[test]
    fn general_quantizer_streams_declared_oq8_payload_lengths() {
        let suffix = format!(
            "{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        );
        let source = std::env::temp_dir().join(format!("diffusion_stream_source_{suffix}.hfq"));
        let output = std::env::temp_dir().join(format!("diffusion_stream_output_{suffix}.hfq"));
        let weights: Vec<f32> = (0..512).map(|index| index as f32 / 128.0 - 2.0).collect();
        let weight_bytes: Vec<u8> = weights
            .iter()
            .flat_map(|value| ((*value).to_bits() >> 16).to_le_bytes()[..2].to_vec())
            .collect();
        write_hfqm_package_mem(
            &source,
            0,
            r#"{"artifact_kind":"diffusion","quantization":{"weight_format":"bf16"}}"#,
            &[
                HfqMemTensor {
                    name: "transformer/tensors/test.weight".to_string(),
                    quant_type: QT_DIFFUSION_TENSOR_BF16,
                    shape: vec![2, 256],
                    group_size: 0,
                    data: weight_bytes,
                },
                HfqMemTensor {
                    name: "transformer/tensors/test.bias".to_string(),
                    quant_type: QT_DIFFUSION_TENSOR_BF16,
                    shape: vec![2],
                    group_size: 0,
                    data: vec![0; 4],
                },
                HfqMemTensor {
                    name: "text_encoder/tensors/test.weight".to_string(),
                    quant_type: QT_DIFFUSION_TENSOR_BF16,
                    shape: vec![2, 256],
                    group_size: 0,
                    data: vec![0; 1024],
                },
            ],
        )
        .expect("write source fixture");

        let summary = quantize_diffusion_hfq(&source, &output, DiffusionQuantFormat::Oq8, None)
            .expect("stream oq8 fixture");
        assert_eq!(summary.quantized_tensors, 1);
        assert_eq!(summary.copied_tensors, 2);
        let packed = HfqFile::open(&output).expect("open streamed output");
        let info = packed
            .tensors()
            .iter()
            .find(|info| info.name == "transformer/tensors/test.weight")
            .expect("packed weight");
        assert_eq!(info.quant_type, QT_DIFFUSION_TENSOR_OQ8_G256);
        assert_eq!(info.data_size, 2 * 258);
        let decoded = cpu_tensor_from_hfq(&packed, "transformer/tensors/test.weight")
            .expect("decode streamed weight");
        assert_eq!(decoded.shape, [2, 256]);
        assert!(decoded.data.iter().all(|value| value.is_finite()));
        let text_info = packed
            .tensors()
            .iter()
            .find(|info| info.name == "text_encoder/tensors/test.weight")
            .expect("copied text weight");
        assert_eq!(text_info.quant_type, QT_DIFFUSION_TENSOR_BF16);

        let _ = std::fs::remove_file(source);
        let _ = std::fs::remove_file(output);
    }

    #[test]
    fn flux2_oq4_policy_promotes_protected_roles_to_oq8() {
        let arch_id = hipfire_arch_api::ARCH_ID_FLUX2;
        assert!(opus_uses_oq8(
            arch_id,
            DiffusionQuantFormat::Oq4PlusPlus,
            "transformer/tensors/proj_out.weight",
            &[128, 3072],
        ));
        assert!(!opus_uses_oq8(
            arch_id,
            DiffusionQuantFormat::Oq4PlusPlus,
            "transformer/tensors/transformer_blocks.2.ff.linear_in.weight",
            &[12288, 3072],
        ));
        assert!(!opus_should_quantize(
            arch_id,
            DiffusionQuantFormat::Oq8,
            "transformer/tensors/proj_out.weight",
            &[128, 3072],
        ));
        assert!(opus_should_quantize(
            arch_id,
            DiffusionQuantFormat::Oq8,
            "transformer/tensors/transformer_blocks.2.ff.linear_in.weight",
            &[12288, 3072],
        ));
    }
}

/// Repack canonical `oq4g256` (`[f16 scale][128 nibbles]` per 256-group,
/// row-contiguous — the [`quantize_oq4g256`] / `decode_oq4g256_slice` format)
/// into the arch "combined" device layout consumed by
/// `hipfire_rdna::Gpu::gemm_oq4_grouped_f16_wmma` (W4A16). Delegates to the
/// single source of truth `hipfire_runtime::oq4_arch::oq4_pack_arch_combined`
/// (see `pack_oq4_arch_combined` below); both read the same qt-34 bytes.
///
/// Output (`[m,k]`, `ng=k/256`): `[nibbles m*(k/2)] [f32 scales m*ng]
/// [interleaved m*ng*132]`. The W4A16 GEMM reads the first two regions; the
/// interleaved tail is for decode GEMVs (kept for layout parity).
/// Byte length of [`pack_oq4_arch_combined`]'s output for an `[m, k]` matrix.
/// Thin re-export of the single source of truth in `hipfire_runtime::oq4_arch`.
pub fn oq4_arch_combined_len(m: usize, k: usize) -> usize {
    hipfire_runtime::hfq::oq4_arch_combined_len(m, k)
}

/// Repack canonical on-disk OQ4 into the arch combined device layout. Thin
/// wrapper over the single source of truth in `hipfire_runtime::oq4_arch`, kept
/// under this name for the diffusion encode/upload path and its public re-export.
pub fn pack_oq4_arch_combined(data: &[u8], m: usize, k: usize) -> Vec<u8> {
    hipfire_runtime::hfq::oq4_pack_arch_combined(data, m, k)
}

/// Open a `.calib.hfq` sidecar for use as the `calib` argument to
/// [`quantize_diffusion_hfq`].
pub fn open_calib_sidecar(path: &Path) -> anyhow::Result<HessianSidecar> {
    HessianSidecar::open(path).map_err(|e| anyhow::anyhow!("{e}"))
}

fn rewrite_weight_format(metadata_json: &str, label: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(metadata_json) {
        Ok(mut value) => {
            if let Some(quant) = value
                .get_mut("quantization")
                .and_then(|q| q.as_object_mut())
            {
                quant.insert(
                    "weight_format".to_string(),
                    serde_json::Value::String(label.to_string()),
                );
            }
            serde_json::to_string(&value).unwrap_or_else(|_| metadata_json.to_string())
        }
        Err(_) => metadata_json.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Plain (unrotated) Opus W4A8 / W8A8 on-disk quantizer — the artifact the tiled
// gemm_opus_tiled_wmma kernels consume directly (no runtime requant, no FWHT).
// Streams one tensor at a time (memory-safe on unified-memory boxes) and marks
// each quantized linear with QT_DIFFUSION_TENSOR_OQ4_PLAIN / _OQ8_PLAIN so the
// loader routes it to the w4/w8 kernel. Supports per-tensor mixed precision.
// ---------------------------------------------------------------------------

/// Plain on-disk quantization policy for the resident DiT linears.
#[derive(Clone, Copy, Debug)]
pub enum PlainOpusPolicy {
    /// Every resident linear → int4 (W4A8).
    AllW4,
    /// Every resident linear → int8 (W8A8).
    AllW8,
    /// Mixed: `oq8_fraction = None` uses the data-free heuristic (int8 for the
    /// first+last block and every FF down-projection — highest fan-in); `Some(f)`
    /// promotes the highest-fan-in resident linears to int8 until ~`f` of the
    /// quantized parameters are int8 (achieved average ≈ `4 + 4·f` bits).
    Mixed { oq8_fraction: Option<f32> },
}

impl PlainOpusPolicy {
    /// Parse plain-Opus policy tokens. Decimal tokens express a requested
    /// parameter-weighted average between four and eight bits; `mixed` keeps the
    /// separate legacy structural heuristic.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "oq4p" | "oq4-plain" => Some(Self::AllW4),
            "oq8p" | "oq8-plain" => Some(Self::AllW8),
            "oq4-mixed" | "mixed" => Some(Self::Mixed { oq8_fraction: None }),
            _ => {
                let bits = s.strip_prefix("oq")?.parse::<f32>().ok()?;
                (bits > 4.0 && bits < 8.0).then(|| Self::Mixed {
                    oq8_fraction: Some((bits - 4.0) / 4.0),
                })
            }
        }
    }
    /// Override the int8 fraction (from `--mix-fraction`); forces Mixed.
    pub fn with_fraction(f: f32) -> Self {
        Self::Mixed {
            oq8_fraction: Some(f.clamp(0.0, 1.0)),
        }
    }
}

/// The canonical Opus quant token for an achieved average bit-width: `oq4`/`oq8`
/// for pure runs, `oq<avg>` (2 dp, trailing zeros trimmed) for mixed — the name
/// is computed from what the quantizer actually produced, not what was requested.
pub fn opus_quant_token(avg_bits: f64) -> String {
    if (avg_bits - avg_bits.round()).abs() < 0.005 {
        return format!("oq{}", avg_bits.round() as i64);
    }
    let s = format!("{avg_bits:.2}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    format!("oq{s}")
}

/// The transformer_blocks.N.* suffixes that load via `linear_resident_weight_resident`
/// (attention q/k/v/out/gate + the joint-attention text stream + the FF linears).
/// Only these are safe to store as plain OQ*: other 2D weights (img_mod/txt_mod)
/// load via `cpu_tensor_from_hfq` and must stay in a decodable format.
const RESIDENT_LINEAR_SUFFIXES: &[&str] = &[
    // Attention projections (both streams).
    ".to_q.weight",
    ".to_k.weight",
    ".to_v.weight",
    ".to_out.0.weight",
    ".to_gate.weight",
    ".add_q_proj.weight",
    ".add_k_proj.weight",
    ".add_v_proj.weight",
    ".to_add_out.weight",
    // Krea2 gated FFN (the bulk of the DiT params) — transformer.rs loads these
    // as ResidentWeight.
    ".ff.up.weight",
    ".ff.gate.weight",
    ".ff.down.weight",
    // QwenImage-family FFN naming.
    ".net.0.proj.weight",
    ".net.2.weight",
];

fn transformer_block_index(name: &str) -> Option<usize> {
    let tag = "transformer_blocks.";
    let start = name.find(tag)? + tag.len();
    let rest = &name[start..];
    let end = rest.find('.')?;
    rest[..end].parse().ok()
}

/// Is `name` a 256-aligned resident DiT linear (safe to store as plain OQ*)?
fn is_resident_linear(name: &str, shape: &[u32]) -> bool {
    shape.len() == 2
        && shape[1] % 256 == 0
        && name.starts_with("transformer/tensors/transformer_blocks.")
        && RESIDENT_LINEAR_SUFFIXES.iter().any(|s| name.ends_with(s))
}

/// One quantizable-tensor candidate (index into the source tensor list).
struct QuantCandidate {
    idx: usize,
    params: u128,
    fan_in: u32,
    boundary: bool,
    /// Distance to the nearest boundary block. Used only as an
    /// architecture-neutral tiebreak after the arch-owned importance score.
    boundary_distance: usize,
    is_down: bool,
    /// Arch structural-saliency prior in `[0,255]` (higher = protect harder),
    /// used to rank the int8 promotion when `--arch-importance` is set.
    importance: u8,
}

/// Structural importance prior for one tensor. Prefers the container's own arch
/// `Ingest` (the arch owns its importance policy); falls back to the shared MMDiT
/// classifier for legacy/unknown-family diffusion containers.
fn tensor_importance(arch_id: u32, name: &str) -> u8 {
    use hipfire_arch_api::{default_importance, mmdit_role, ArchId};
    if let Ok(id) = u16::try_from(arch_id) {
        if let Some(ingest) = hipfire_archs::registry()
            .get(ArchId(id))
            .and_then(|a| a.caps.ingest)
        {
            return ingest.importance(name);
        }
    }
    default_importance(mmdit_role(name))
}

/// Choose which quantizable tensors go to int8 (the rest int4), returning the set
/// of source-tensor indices. `AllW4`/`AllW8` are trivial; `Mixed{None}` uses the
/// data-free heuristic (first/last block + FF down); `Mixed{Some(f)}` promotes the
/// top-ranked linears until ~`f` of the quantized parameters are int8.
///
/// `by_importance` selects the ranking for the fraction budget: when set, promote
/// the highest arch-importance tensors first (the arch's structural saliency
/// prior — embedders/attention/modulation/output over the FFN bulk); otherwise
/// the default highest-fan-in / down-proj-first heuristic. The budget (`f`) is
/// identical either way — only *which* tensors win the int8 promotion changes.
fn select_int8(
    cands: &[QuantCandidate],
    policy: PlainOpusPolicy,
    by_importance: bool,
) -> std::collections::HashSet<usize> {
    use std::collections::HashSet;
    match policy {
        PlainOpusPolicy::AllW4 => HashSet::new(),
        PlainOpusPolicy::AllW8 => cands.iter().map(|c| c.idx).collect(),
        PlainOpusPolicy::Mixed { oq8_fraction: None } => cands
            .iter()
            .filter(|c| c.boundary || c.is_down)
            .map(|c| c.idx)
            .collect(),
        PlainOpusPolicy::Mixed {
            oq8_fraction: Some(f),
        } => {
            let total: u128 = cands.iter().map(|c| c.params).sum();
            let target = (total as f64 * f as f64) as u128;
            let mut order: Vec<&QuantCandidate> = cands.iter().collect();
            if by_importance {
                // The architecture owns the primary score. For exact ties, keep
                // the allocation symmetric from the transformer boundaries,
                // then prefer the wider projection.
                order.sort_by(|a, b| {
                    b.importance
                        .cmp(&a.importance)
                        .then(a.boundary_distance.cmp(&b.boundary_distance))
                        .then(b.fan_in.cmp(&a.fan_in))
                        .then(a.idx.cmp(&b.idx))
                });
            } else {
                // Default: fan-in desc (down-projs first), boundary as tiebreak.
                order.sort_by(|a, b| {
                    b.fan_in
                        .cmp(&a.fan_in)
                        .then(b.boundary.cmp(&a.boundary))
                        .then(a.idx.cmp(&b.idx))
                });
            }
            let mut acc: u128 = 0;
            let mut set = HashSet::new();
            for c in order {
                if acc >= target {
                    break;
                }
                let next = acc + c.params;
                if acc.abs_diff(target) < next.abs_diff(target) {
                    break;
                }
                set.insert(c.idx);
                acc = next;
            }
            set
        }
    }
}

#[cfg(test)]
mod select_int8_tests {
    use super::*;
    use std::collections::HashSet;

    fn cand(idx: usize, params: u128, fan_in: u32, importance: u8) -> QuantCandidate {
        QuantCandidate {
            idx,
            params,
            fan_in,
            boundary: false,
            boundary_distance: usize::MAX,
            is_down: false,
            importance,
        }
    }

    #[test]
    fn importance_mode_promotes_salient_over_high_fan_in() {
        // Two equal-size candidates; a 0.5 budget promotes exactly one to int8.
        // idx 0: low importance (FFN bulk) but high fan-in.
        // idx 1: high importance (attention/output) but low fan-in.
        let cands = vec![cand(0, 1000, 4096, 128), cand(1, 1000, 512, 255)];
        let policy = PlainOpusPolicy::Mixed {
            oq8_fraction: Some(0.5),
        };
        // Default fan-in ranking promotes the high-fan-in bulk tensor.
        assert_eq!(select_int8(&cands, policy, false), HashSet::from([0]));
        // Arch-importance ranking promotes the salient tensor instead — same
        // budget, different selection.
        assert_eq!(select_int8(&cands, policy, true), HashSet::from([1]));
    }

    #[test]
    fn decimal_opus_token_maps_to_requested_average_bit_budget() {
        let Some(PlainOpusPolicy::Mixed {
            oq8_fraction: Some(fraction),
        }) = PlainOpusPolicy::parse("oq4.25")
        else {
            panic!("oq4.25 should be a numeric mixed-Opus policy");
        };
        assert!((fraction - 0.0625).abs() < f32::EPSILON);
        assert!(matches!(
            PlainOpusPolicy::parse("mixed"),
            Some(PlainOpusPolicy::Mixed { oq8_fraction: None })
        ));
    }

    #[test]
    fn importance_ties_promote_boundary_tensors_symmetrically() {
        let mut cands = vec![
            cand(0, 100, 1024, 253),
            cand(1, 100, 1024, 253),
            cand(26, 100, 1024, 253),
            cand(27, 100, 1024, 253),
        ];
        cands[0].boundary = true;
        cands[3].boundary = true;
        cands[0].boundary_distance = 0;
        cands[3].boundary_distance = 0;
        let policy = PlainOpusPolicy::Mixed {
            oq8_fraction: Some(0.5),
        };
        assert_eq!(select_int8(&cands, policy, true), HashSet::from([0, 27]));
    }

    #[test]
    fn fractional_budget_chooses_closest_prefix_without_forced_overshoot() {
        let cands = vec![cand(0, 100, 1024, 255), cand(1, 100, 1024, 254)];
        let policy = PlainOpusPolicy::Mixed {
            oq8_fraction: Some(0.2),
        };
        assert_eq!(select_int8(&cands, policy, true), HashSet::new());
    }
}

/// Summary of a plain-Opus quantization run.
#[derive(Default, Debug)]
pub struct PlainQuantizeSummary {
    pub w4_tensors: usize,
    pub w8_tensors: usize,
    pub copied_tensors: usize,
    pub source_bytes: u64,
    pub output_bytes: u64,
    pub avg_bits: f64,
}

/// Re-encode the resident DiT linears of `source` into plain (unrotated) Opus
/// W4A8/W8A8 blobs (`[packed | f32 per-group scales]`), copying every other entry
/// verbatim, and stream the result to `output`. One tensor is resident in RAM at
/// a time. The loader consumes these with zero runtime requant.
pub fn quantize_diffusion_hfq_plain(
    source: &Path,
    output: &Path,
    policy: PlainOpusPolicy,
    by_importance: bool,
) -> anyhow::Result<PlainQuantizeSummary> {
    const GROUP: usize = 256;
    let hfq = HfqFile::open(source)?;
    let infos = hfq.tensors().to_vec();
    let last_block = infos
        .iter()
        .filter_map(|t| transformer_block_index(&t.name))
        .max()
        .unwrap_or(0);

    let mut summary = PlainQuantizeSummary {
        source_bytes: std::fs::metadata(source)?.len(),
        ..Default::default()
    };

    // Pre-pass: collect quantizable candidates and choose the int8 set (per policy
    // / fraction) before building the index — the width decision is not per-tensor
    // local when a global bit budget is in play.
    let candidates: Vec<QuantCandidate> = infos
        .iter()
        .enumerate()
        .filter(|(_, t)| is_resident_linear(&t.name, &t.shape))
        .map(|(idx, t)| {
            let blk = transformer_block_index(&t.name).unwrap_or(usize::MAX);
            QuantCandidate {
                idx,
                params: (t.shape[0] as u128) * (t.shape[1] as u128),
                fan_in: t.shape[1],
                boundary: blk == 0 || blk == last_block,
                boundary_distance: blk.min(last_block.saturating_sub(blk)),
                is_down: t.name.ends_with(".ff.down.weight") || t.name.ends_with(".net.2.weight"),
                importance: tensor_importance(hfq.arch_id, &t.name),
            }
        })
        .collect();
    let int8_set = select_int8(&candidates, policy, by_importance);
    let quant_idx: std::collections::HashSet<usize> = candidates.iter().map(|c| c.idx).collect();

    // Per-entry plan + declared payload length (needed up front for the index).
    let mut entries: Vec<HfqStreamEntry> = Vec::with_capacity(infos.len());
    let mut plans: Vec<Option<u8>> = Vec::with_capacity(infos.len());
    let mut quant_bits_total: u128 = 0;
    let mut quant_elems_total: u128 = 0;
    for (i, t) in infos.iter().enumerate() {
        let bits = if quant_idx.contains(&i) {
            Some(if int8_set.contains(&i) { 8u8 } else { 4u8 })
        } else {
            None
        };
        plans.push(bits);
        let (quant_type, data_len) = match bits {
            Some(w) => {
                let m = t.shape[0] as usize;
                let k = t.shape[1] as usize;
                let ng = k / GROUP;
                let packed = if w == 4 { m * k / 2 } else { m * k };
                let len = (packed + ng * m * 4) as u64;
                if w == 4 {
                    summary.w4_tensors += 1;
                } else {
                    summary.w8_tensors += 1;
                }
                quant_bits_total += (w as u128) * (m as u128) * (k as u128);
                quant_elems_total += (m as u128) * (k as u128);
                let qt = if w == 4 {
                    QT_DIFFUSION_TENSOR_OQ4_PLAIN
                } else {
                    QT_DIFFUSION_TENSOR_OQ8_PLAIN
                };
                (qt, len)
            }
            None => {
                summary.copied_tensors += 1;
                (t.quant_type, t.data_size as u64)
            }
        };
        entries.push(HfqStreamEntry {
            name: t.name.clone(),
            quant_type,
            shape: t.shape.clone(),
            group_size: if bits.is_some() {
                GROUP as u32
            } else {
                t.group_size
            },
            data_len,
        });
    }
    summary.avg_bits = if quant_elems_total > 0 {
        quant_bits_total as f64 / quant_elems_total as f64
    } else {
        0.0
    };

    // The weight_format label reflects the achieved average, not the request.
    let token = opus_quant_token(summary.avg_bits);
    let metadata_json = rewrite_weight_format(&hfq.metadata_json, &token);
    write_hfqm_package_streaming(output, hfq.arch_id, &metadata_json, &entries, |i, w| {
        let name = &infos[i].name;
        let (_info, bytes) = hfq
            .tensor_data_vec(name)
            .ok_or_else(|| std::io::Error::other(format!("tensor {name:?} vanished")))?;
        match plans[i] {
            None => w.write_all(&bytes),
            Some(width) => {
                let m = infos[i].shape[0] as usize;
                let k = infos[i].shape[1] as usize;
                let ng = k / GROUP;
                // Source is bf16 (2 bytes/elem); read values straight from bytes.
                let read = |row_base: usize, elem: usize| -> f32 {
                    let b = row_base + elem * 2;
                    crate::bf16_byte_to_f32(bytes[b], bytes[b + 1])
                };
                let (packed, scales) = if width == 4 {
                    crate::quantize_w4a8_rows(m, k, ng, 2, read)
                } else {
                    crate::quantize_oq8_rows(m, k, ng, 2, read)
                };
                w.write_all(&packed)?;
                for s in &scales {
                    w.write_all(&s.to_le_bytes())?;
                }
                Ok(())
            }
        }
    })?;
    summary.output_bytes = std::fs::metadata(output)?.len();
    Ok(summary)
}
