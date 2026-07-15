// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Opus mixed-precision low-bit weight codec — CPU reference + specification.
//!
//! This is the source of truth the `gemm_opus_tiled_wmma.hip` kernel mirrors 1:1.
//! The Opus GEMM fixes the activation at dynamic int8 (A8) and always runs the
//! `wmma_i32_16x16x16_iu8` core, so the weight bit-width only changes how the
//! packed weight is *unpacked* into an int8 fragment. We store weights as
//! **unsigned** codes and flag the WMMA weight operand as unsigned, then fold the
//! symmetric zero-point out of the accumulator per group:
//!
//!   q_k = u_k - Z             (Z = 2^(bits-1), the signed zero-point)
//!   Σ_k q_k·x_k = (Σ_k u_k·x_k) - Z·(Σ_k x_k)
//!               = <unsigned WMMA result> - Z·<activation group sum>
//!
//! `Σ_k x_k` (the per-group activation sum) is produced once by the activation
//! quantizer, so the fold costs one int32 subtract per group in the rescale and
//! **removes all sign-extension from the unpack**. The unpack for every
//! power-of-two width is then pure branch-free bit-slicing (mask/shift, or
//! `v_perm_b32` on GPU), identical body, only the mask constant and Z change.
//!
//! Dense (byte-aligned) packing covers bits ∈ {1,2,4,8}. Non-power-of-two widths
//! (3,5,6,7) need the bit-plane layout (see [`plane`]); the fold and the iu8 core
//! are unchanged.

/// Signed zero-point for a symmetric `bits`-wide code: the stored unsigned code
/// `u ∈ [0, 2^bits - 1]` represents `q = u - Z ∈ [-2^(bits-1), 2^(bits-1)-1]`
/// (the standard signed-integer range — e.g. int4 → -8..=7).
#[inline]
pub const fn zero_point(bits: u32) -> i32 {
    1 << (bits - 1)
}

#[inline]
pub const fn code_max(bits: u32) -> u32 {
    (1u32 << bits) - 1
}

/// True for the dense-packable (byte-aligned) power-of-two widths.
#[inline]
pub const fn is_dense_width(bits: u32) -> bool {
    matches!(bits, 1 | 2 | 4 | 8)
}

/// Per-group symmetric quantization of `weights` into unsigned `bits`-wide codes.
/// Returns `(codes, scales)` with one scale per `group`. Mirrors the on-GPU
/// contract: dequant value ≈ `(code - Z) * scale`.
pub fn quantize_symmetric(weights: &[f32], group: usize, bits: u32) -> (Vec<u8>, Vec<f32>) {
    assert!((1..=8).contains(&bits), "bits must be 1..=8");
    assert!(group > 0 && weights.len() % group == 0, "len must be a multiple of group");
    let z = zero_point(bits);
    let qmin = -z;
    let qmax = z - 1;
    let mut codes = Vec::with_capacity(weights.len());
    let mut scales = Vec::with_capacity(weights.len() / group);
    for chunk in weights.chunks_exact(group) {
        let max_abs = chunk.iter().fold(0.0f32, |acc, &w| acc.max(w.abs()));
        // Map the largest magnitude onto the negative rail (Z levels), matching a
        // symmetric signed grid; guard the all-zero group.
        let scale = if max_abs > 0.0 { max_abs / z as f32 } else { 1.0 };
        scales.push(scale);
        for &w in chunk {
            let q = (w / scale).round() as i32;
            let q = q.clamp(qmin, qmax);
            codes.push((q + z) as u8); // unsigned storage in [0, 2^bits-1]
        }
    }
    (codes, scales)
}

/// Activation-aware per-group symmetric quant — the "+" (clip-search / AWQ) for
/// the plain (unrotated) fold format. For each group it searches `n_steps` clip
/// fractions in `[min_clip, 1]` and keeps the scale minimizing the **importance-
/// weighted** reconstruction error `Σ_c imp[c]·(w_c − dequant)²`, where `imp` is
/// the per-input-channel imatrix (`Σx²`, length K) from `hipfire diffusion
/// calibrate`. `None` importance ⇒ unweighted MSE clip-search. Same unsigned
/// output as [`quantize_symmetric`], so it drops straight into the fold GEMM.
///
/// This is the plain-basis complement to the rotated LDLQ path (`oq4_ldlq_pack`):
/// the fast fold GEMM doesn't rotate activations, so its codes must be calibrated
/// in the plain basis — clipping is the cheap, Hessian-free lever, and it matters
/// most at low bit (oq4/oq2/oq1) where a single outlier channel otherwise crushes
/// the whole group's scale.
pub fn quantize_symmetric_clip(
    weights: &[f32],
    group: usize,
    bits: u32,
    importance: Option<&[f32]>,
    n_steps: usize,
    min_clip: f32,
) -> (Vec<u8>, Vec<f32>) {
    assert!((1..=8).contains(&bits), "bits must be 1..=8");
    assert!(group > 0 && weights.len() % group == 0, "len must be a multiple of group");
    let n_steps = n_steps.max(1);
    let z = zero_point(bits);
    let (qmin, qmax) = (-z, z - 1);
    // Per-tensor channel count (for imatrix indexing) inferred from importance.
    let n_groups = importance.map(|im| {
        assert!(im.len() % group == 0 && weights.len() % im.len() == 0, "importance must be [K], K%group==0");
        im.len() / group
    });
    let mut codes = vec![0u8; weights.len()];
    let mut scales = vec![0.0f32; weights.len() / group];
    for (ci, chunk) in weights.chunks_exact(group).enumerate() {
        let imp: Option<&[f32]> = match (importance, n_groups) {
            (Some(im), Some(ng)) => {
                let gi = ci % ng;
                Some(&im[gi * group..gi * group + group])
            }
            _ => None,
        };
        let amax = chunk.iter().fold(0.0f32, |a, &w| a.max(w.abs()));
        let mut best_scale = if amax > 0.0 { amax / z as f32 } else { 1.0 };
        let mut best_err = f64::INFINITY;
        for s in 0..n_steps {
            let alpha = if n_steps == 1 {
                1.0
            } else {
                min_clip + (1.0 - min_clip) * (s as f32 / (n_steps - 1) as f32)
            };
            let clip = alpha * amax;
            let scale = if clip > 0.0 { clip / z as f32 } else { 1.0 };
            let inv = 1.0 / scale;
            let mut err = 0.0f64;
            for (i, &w) in chunk.iter().enumerate() {
                let q = ((w * inv).round() as i32).clamp(qmin, qmax);
                let d = (w - q as f32 * scale) as f64;
                let wgt = imp.map(|im| im[i] as f64).unwrap_or(1.0);
                err += wgt * d * d;
            }
            if err < best_err {
                best_err = err;
                best_scale = scale;
            }
        }
        scales[ci] = best_scale;
        let inv = 1.0 / best_scale;
        for (i, &w) in chunk.iter().enumerate() {
            let q = ((w * inv).round() as i32).clamp(qmin, qmax);
            codes[ci * group + i] = (q + z) as u8;
        }
    }
    (codes, scales)
}

/// Importance-weighted relative RMSE of a quantization, for calibration
/// evaluation: `sqrt(Σ imp·(w−ŵ)² / Σ imp·w²)`. `None` ⇒ unweighted.
pub fn weighted_quant_error(
    weights: &[f32],
    codes: &[u8],
    scales: &[f32],
    group: usize,
    bits: u32,
    importance: Option<&[f32]>,
) -> f64 {
    let z = zero_point(bits);
    let n_groups = importance.map(|im| im.len() / group);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (ci, chunk) in weights.chunks_exact(group).enumerate() {
        let scale = scales[ci];
        let imp: Option<&[f32]> = match (importance, n_groups) {
            (Some(im), Some(ng)) => {
                let gi = ci % ng;
                Some(&im[gi * group..gi * group + group])
            }
            _ => None,
        };
        for (i, &w) in chunk.iter().enumerate() {
            let q = (codes[ci * group + i] as i32 - z) as f32;
            let d = (w - q * scale) as f64;
            let wgt = imp.map(|im| im[i] as f64).unwrap_or(1.0);
            num += wgt * d * d;
            den += wgt * (w as f64) * (w as f64);
        }
    }
    (num / den.max(1e-30)).sqrt()
}

/// Dense LSB-first packing of unsigned `bits`-wide codes (only 1/2/4/8). Code `j`
/// occupies bits `[ (j*bits) % 8 .. +bits ]` of byte `j*bits/8`, low code = low
/// bits — matching the kernel's "low nibble = even k" convention.
pub fn pack_dense(codes: &[u8], bits: u32) -> Vec<u8> {
    assert!(is_dense_width(bits), "pack_dense only supports bits ∈ {{1,2,4,8}}");
    let per_byte = (8 / bits) as usize;
    let mask = code_max(bits) as u8;
    let mut out = vec![0u8; codes.len().div_ceil(per_byte)];
    for (i, &c) in codes.iter().enumerate() {
        let byte = i / per_byte;
        let slot = (i % per_byte) as u32;
        out[byte] |= (c & mask) << (slot * bits);
    }
    out
}

/// Inverse of [`pack_dense`]: recover `count` unsigned codes. Branch-free per
/// code (mask + shift) — the exact operation the device unpacker performs.
pub fn unpack_dense(packed: &[u8], count: usize, bits: u32) -> Vec<u8> {
    assert!(is_dense_width(bits), "unpack_dense only supports bits ∈ {{1,2,4,8}}");
    let per_byte = (8 / bits) as usize;
    let mask = code_max(bits) as u8;
    (0..count)
        .map(|i| {
            let byte = packed[i / per_byte];
            let slot = (i % per_byte) as u32;
            (byte >> (slot * bits)) & mask
        })
        .collect()
}

/// Dense byte length for `count` codes at `bits` (matches the kernel row stride).
#[inline]
pub const fn dense_len(count: usize, bits: u32) -> usize {
    let per_byte = (8 / bits) as usize;
    count.div_ceil(per_byte)
}

/// Bit-plane (sliced) packing for *any* width 1..=8, including 3/5/6/7. Plane `p`
/// holds bit `p` of every code, densely packed 8-per-byte. Unpack ORs `bits`
/// shifted planes — uniform, byte-aligned, sign-free, at the cost of `bits` loads.
pub mod plane {
    /// Pack `codes` into `bits` bit-planes; returns `bits` byte-planes concatenated.
    pub fn pack(codes: &[u8], bits: u32) -> Vec<u8> {
        assert!((1..=8).contains(&bits));
        let plane_bytes = codes.len().div_ceil(8);
        let mut out = vec![0u8; plane_bytes * bits as usize];
        for (i, &c) in codes.iter().enumerate() {
            for p in 0..bits {
                if (c >> p) & 1 != 0 {
                    out[p as usize * plane_bytes + i / 8] |= 1 << (i % 8);
                }
            }
        }
        out
    }

    /// Recover `count` codes from `bits` bit-planes.
    pub fn unpack(planes: &[u8], count: usize, bits: u32) -> Vec<u8> {
        let plane_bytes = count.div_ceil(8);
        (0..count)
            .map(|i| {
                let mut code = 0u8;
                for p in 0..bits {
                    let bit = (planes[p as usize * plane_bytes + i / 8] >> (i % 8)) & 1;
                    code |= bit << p;
                }
                code
            })
            .collect()
    }
}

/// Reference GEMM matching the kernel's math for one output element, using the
/// **unsigned codes + offset fold**: `Σ_g sw·sx·(Σ u·x − Z·Σx)`. `x_sum[g]` is the
/// per-group activation sum the quantizer precomputes.
#[allow(clippy::too_many_arguments)]
pub fn dot_offset_fold(
    codes_row: &[u8], // unsigned weight codes for one output row m, length K
    w_scales_row: &[f32], // one per group
    x_row: &[i8],     // int8 activations for one batch b, length K
    x_scales_row: &[f32],
    x_sum_row: &[i32], // Σ_{k∈g} x_row[k], one per group
    group: usize,
    bits: u32,
) -> f32 {
    let z = zero_point(bits);
    let n_groups = codes_row.len() / group;
    let mut acc = 0.0f32;
    for g in 0..n_groups {
        let base = g * group;
        let mut iacc = 0i32; // Σ u·x  (unsigned weight × signed activation)
        for k in 0..group {
            iacc += codes_row[base + k] as i32 * x_row[base + k] as i32;
        }
        let folded = iacc - z * x_sum_row[g];
        acc += folded as f32 * w_scales_row[g] * x_scales_row[g];
    }
    acc
}

/// Signed reference (`Σ_g sw·sx·Σ (u−Z)·x`) — no fold — used to prove the fold is
/// an exact integer identity.
pub fn dot_signed(
    codes_row: &[u8],
    w_scales_row: &[f32],
    x_row: &[i8],
    x_scales_row: &[f32],
    group: usize,
    bits: u32,
) -> f32 {
    let z = zero_point(bits);
    let n_groups = codes_row.len() / group;
    let mut acc = 0.0f32;
    for g in 0..n_groups {
        let base = g * group;
        let mut iacc = 0i32;
        for k in 0..group {
            iacc += (codes_row[base + k] as i32 - z) * x_row[base + k] as i32;
        }
        acc += iacc as f32 * w_scales_row[g] * x_scales_row[g];
    }
    acc
}

/// Per-group activation sums `Σ_{k∈g} x[k]` (what the activation quantizer emits
/// alongside the per-group scale so the fold has no in-kernel reduction).
pub fn group_sums_i8(x_row: &[i8], group: usize) -> Vec<i32> {
    x_row
        .chunks_exact(group)
        .map(|g| g.iter().map(|&v| v as i32).sum())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip_dense(bits: u32) {
        let count = 64;
        let codes: Vec<u8> = (0..count).map(|i| (i as u32 % (1 << bits)) as u8).collect();
        let packed = pack_dense(&codes, bits);
        assert_eq!(packed.len(), dense_len(count, bits));
        assert_eq!(unpack_dense(&packed, count, bits), codes);
    }

    #[test]
    fn dense_pack_unpack_round_trips_pow2() {
        for bits in [1, 2, 4, 8] {
            round_trip_dense(bits);
        }
    }

    #[test]
    fn plane_pack_unpack_round_trips_all_widths() {
        for bits in 1..=8u32 {
            let count = 100;
            let codes: Vec<u8> = (0..count).map(|i| (i as u32 % (1 << bits)) as u8).collect();
            let packed = plane::pack(&codes, bits);
            assert_eq!(plane::unpack(&packed, count, bits), codes, "bits={bits}");
        }
    }

    #[test]
    fn code_range_is_standard_signed() {
        // int4 → -8..=7, int2 → -2..=1, etc.
        for bits in 1..=8u32 {
            let z = zero_point(bits);
            assert_eq!(-z, -(1 << (bits - 1)));
            assert_eq!(code_max(bits) as i32 - z, (1 << (bits - 1)) - 1);
        }
    }

    #[test]
    fn offset_fold_is_exact_integer_identity() {
        // The unsigned-fold dot MUST equal the signed dot bit-for-bit (same f32
        // rounding of identical integer accumulators), for every width.
        let group = 16;
        let k = group * 3;
        for bits in [1, 2, 4, 8] {
            let z = zero_point(bits) as i32;
            let cmax = code_max(bits) as i32;
            let codes: Vec<u8> = (0..k).map(|i| ((i as i32 * 7 + 3) % (cmax + 1)) as u8).collect();
            let x: Vec<i8> = (0..k).map(|i| ((i as i32 * 5 - 40) % 100 - 50) as i8).collect();
            let ws: Vec<f32> = (0..k / group).map(|g| 0.01 * (g as f32 + 1.0)).collect();
            let xs: Vec<f32> = (0..k / group).map(|g| 0.02 * (g as f32 + 1.0)).collect();
            let xsum = group_sums_i8(&x, group);
            // sanity: fold term uses the same Z
            let _ = z;
            let folded = dot_offset_fold(&codes, &ws, &x, &xs, &xsum, group, bits);
            let signed = dot_signed(&codes, &ws, &x, &xs, group, bits);
            assert_eq!(folded.to_bits(), signed.to_bits(), "fold != signed at bits={bits}");
        }
    }

    #[test]
    fn clip_search_beats_rtn_on_weighted_error() {
        // Classic AWQ case: a high-magnitude but LOW-importance outlier channel
        // otherwise sets the whole group's scale and crushes the important small
        // channels. Clipping the outlier reduces the importance-weighted error.
        let group = 16;
        let bits = 2; // coarse grid → clipping matters most
        let mut w = vec![0.0f32; group];
        w[0] = 10.0; // outlier
        for (i, wi) in w.iter_mut().enumerate().skip(1) {
            *wi = ((i as f32 * 0.7).sin()) * 0.8; // |w| < 1, the signal we care about
        }
        let mut imp = vec![1.0f32; group];
        imp[0] = 0.01; // outlier barely matters

        let (rtn_codes, rtn_scales) = quantize_symmetric(&w, group, bits);
        let (clip_codes, clip_scales) =
            quantize_symmetric_clip(&w, group, bits, Some(&imp), 12, 0.1);

        let rtn_err = weighted_quant_error(&w, &rtn_codes, &rtn_scales, group, bits, Some(&imp));
        let clip_err = weighted_quant_error(&w, &clip_codes, &clip_scales, group, bits, Some(&imp));
        assert!(
            clip_err < rtn_err * 0.9,
            "clip should cut weighted error: rtn={rtn_err:.4} clip={clip_err:.4}"
        );
        // codes stay in range
        assert!(clip_codes.iter().all(|&c| (c as u32) <= code_max(bits)));
    }

    #[test]
    fn clip_search_no_importance_reduces_unweighted_mse() {
        // With a heavy-tailed group, unweighted clip-search should not be worse
        // than RTN (it may clip a lone outlier to help the bulk).
        let group = 32;
        let bits = 3;
        let w: Vec<f32> = (0..group)
            .map(|i| if i == 5 { 8.0 } else { (i as f32 * 0.31).cos() })
            .collect();
        let (rtn_codes, rtn_scales) = quantize_symmetric(&w, group, bits);
        let (clip_codes, clip_scales) = quantize_symmetric_clip(&w, group, bits, None, 12, 0.2);
        let rtn = weighted_quant_error(&w, &rtn_codes, &rtn_scales, group, bits, None);
        let clip = weighted_quant_error(&w, &clip_codes, &clip_scales, group, bits, None);
        assert!(clip <= rtn + 1e-9, "unweighted clip {clip:.4} should be <= rtn {rtn:.4}");
    }

    #[test]
    fn quantize_then_fold_matches_f32_within_grid_error() {
        let group = 16;
        let k = group * 4;
        let bits = 4;
        // A weight row and an activation row.
        let w: Vec<f32> = (0..k).map(|i| (i as f32 * 0.013).sin()).collect();
        let (codes, wscale) = quantize_symmetric(&w, group, bits);
        // dequantize back for the f32 truth
        let z = zero_point(bits);
        let w_hat: Vec<f32> = codes
            .iter()
            .enumerate()
            .map(|(i, &c)| (c as i32 - z) as f32 * wscale[i / group])
            .collect();
        let x: Vec<i8> = (0..k).map(|i| ((i as i32 % 40) - 20) as i8).collect();
        let xs = vec![0.05f32; k / group];
        let xsum = group_sums_i8(&x, group);
        // dequantized activations
        let x_f32: Vec<f32> = x.iter().enumerate().map(|(i, &q)| q as f32 * xs[i / group]).collect();
        let truth: f32 = w_hat.iter().zip(&x_f32).map(|(a, b)| a * b).sum();
        let folded = dot_offset_fold(&codes, &wscale, &x, &xs, &xsum, group, bits);
        assert!((folded - truth).abs() < 1e-3, "folded={folded} truth={truth}");
    }
}
