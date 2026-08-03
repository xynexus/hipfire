// SPDX-License-Identifier: Apache-2.0
// hipfire — QTIP-LDLQ (Phase C1e): output-aware (Hessian) trellis quantization.
//
// Clean-room: the LDLQ algorithm is implemented here; `faer` is only the
// SIMD/parallel Cholesky backend. MSE-optimal QTIP-2 failed PPL (125.6 vs MQ4
// 14.0) because reconstruction-optimal ≠ output-optimal; LDLQ minimizes the
// activation-weighted output error ‖(W−Ŵ)·√H‖ via OBS error feedback.

use faer::prelude::Solve;
use faer::{Mat, Side};

/// Lower factor `L` with `L·Lᵀ = (H + λI)⁻¹` (H row-major `k×k`, SPD).
///
/// The GPTQ/OBS upper factor `U` (`UᵀU = (H+λI)⁻¹`, used as: divisor
/// `U[step,step]`, propagation weight `U[step,next]`) is exactly `Uᵀ = L`, so
/// `U[step,step] = L[step,step]` and `U[step,next>step] = L[next,step]`. We
/// return `L` and index it transposed in the OBS loop — avoids a second
/// transpose pass. `None` on non-SPD breakdown.
pub fn inv_cholesky_lower(h_rowmajor: &[f32], k: usize, damp: f64) -> Option<Mat<f64>> {
    assert_eq!(h_rowmajor.len(), k * k);
    let hd = Mat::<f64>::from_fn(k, k, |i, j| {
        h_rowmajor[i * k + j] as f64 + if i == j { damp } else { 0.0 }
    });
    let chol = hd.llt(Side::Lower).ok()?;
    let hinv = chol.solve(Mat::<f64>::identity(k, k)); // H⁻¹ via Cholesky solve
    let chol2 = hinv.llt(Side::Lower).ok()?;
    Some(chol2.L().to_owned())
}

/// Per-256-block FWHT similarity transform of the Hessian, in place:
/// `H ← R H Rᵀ` where `R` is the engine's per-256 signed FWHT
/// (`crate::cpu_fwht_256`). Row-pass then column-pass (same R both sides),
/// putting H in the same incoherent domain as the FWHT-rotated weights so the
/// OBS feedback is consistent.
/// Rotate a [k, k] Hessian into the FWHT frame the trellis sees. Exposed so the
/// GPU-resident conditioning path can build L without duplicating this.
pub fn rotate_hessian(h: &mut [f64], k: usize, signs1: &[f32], signs2: &[f32]) {
    use rayon::prelude::*;
    // Rows are independent, so this pass parallelizes directly.
    let rows_pass = |h: &mut [f64]| {
        let nb = k / 256;
        h.par_chunks_mut(k).for_each(|row| {
            let mut buf = [0.0f32; 256];
            for b in 0..nb {
                for c in 0..256 {
                    buf[c] = row[b * 256 + c] as f32;
                }
                crate::cpu_fwht_256(&mut buf, signs1, signs2);
                for c in 0..256 {
                    row[b * 256 + c] = buf[c] as f64;
                }
            }
        });
    };
    // The column pass is the row pass on the transpose. Transposing twice costs
    // two O(k²) copies but buys the same parallelism, whereas striding down
    // columns of a `&mut [f64]` cannot be split by rayon without unsafe.
    //
    // Measured at 23% of greedy conditioning's wall-time (119.0s of 514s) while
    // fully SERIAL — the single cheapest speedup left in this path.
    let transpose = |src: &[f64], dst: &mut [f64]| {
        dst.par_chunks_mut(k).enumerate().for_each(|(i, drow)| {
            for (j, v) in drow.iter_mut().enumerate() {
                *v = src[j * k + i];
            }
        });
    };
    rows_pass(h);
    let mut t = vec![0.0f64; k * k];
    transpose(h, &mut t);
    rows_pass(&mut t);
    transpose(&t, h);
}

/// Lower Cholesky of (H_rot + lambda I)^-1. Exposed alongside rotate_hessian.
/// Same contract as [`inv_cholesky_lower_rotated`], without the identity solve.
///
/// The original computes `(H+λI)⁻¹` as `chol.solve(Mat::identity(k, k))`, which
/// materialises a k×k identity — 512 MB at k=8192, allocated per tensor PER
/// damping retry — and costs ~k³ (k triangular solves). `DenseSolveCore::inverse`
/// is the `potri` equivalent: it forms the inverse from the factor in ~k³/3 and
/// allocates only the result. Total drops from ~1.67·k³ to ~k³.
///
/// The second factorization is NOT redundant and must stay: `L·Lᵀ = (H+λI)⁻¹`
/// needs a LOWER factor of the inverse. `C⁻ᵀ` also satisfies `L·Lᵀ = A⁻¹` but is
/// UPPER triangular, and substituting it silently degrades LDLQ to RTN while
/// every residual check still passes — that cost 57 minutes earlier in this
/// project, so the shape is asserted in the test rather than assumed.
pub fn inv_cholesky_lower_rotated_fast(h: &[f64], k: usize, damp: f64) -> Option<Mat<f64>> {
    use faer::linalg::solvers::DenseSolveCore;
    let base = damp.max(1e-12);
    for mult in [1.0, 10.0, 100.0, 1000.0, 10000.0] {
        let lambda = base * mult;
        let hd = Mat::<f64>::from_fn(k, k, |i, j| {
            h[i * k + j] + if i == j { lambda } else { 0.0 }
        });
        let Ok(chol) = hd.llt(Side::Lower) else {
            continue;
        };
        let hinv = chol.inverse();
        let Ok(chol2) = hinv.llt(Side::Lower) else {
            continue;
        };
        return Some(chol2.L().to_owned());
    }
    None
}

/// REFERENCE implementation, kept as the equivalence oracle for
/// [`inv_cholesky_lower_rotated_fast`], which is what callers use. Retained
/// rather than deleted because a rewrite of this routine has already failed
/// silently once — comparing the two outputs directly is the cheapest check that
/// would have caught it.
pub fn inv_cholesky_lower_rotated(h: &[f64], k: usize, damp: f64) -> Option<Mat<f64>> {
    let base = damp.max(1e-12);
    for mult in [1.0, 10.0, 100.0, 1000.0, 10000.0] {
        let lambda = base * mult;
        let hd = Mat::<f64>::from_fn(k, k, |i, j| {
            h[i * k + j] + if i == j { lambda } else { 0.0 }
        });
        let Ok(chol) = hd.llt(Side::Lower) else {
            continue;
        };
        let hinv = chol.solve(Mat::<f64>::identity(k, k));
        let Ok(chol2) = hinv.llt(Side::Lower) else {
            continue;
        };
        return Some(chol2.L().to_owned());
    }
    None
}

/// QTIP-LDLQ: block-sequential **trellis** quantization with OBS error
/// feedback. Returns the dequantized weights in the ORIGINAL (un-rotated)
/// domain — the effective weight a fused QTIP kernel would produce — as
/// row-major `m×k` f32, for the simulated-quant PPL path. `None` on
/// Cholesky breakdown (caller falls back to plain QTIP).
///
/// FWHT-rotate H + W into the incoherent domain → `L` with `LLᵀ=(H_rot+λI)⁻¹`
/// (so OBS divisor `=L[c,c]`, propagation weight `U[c,f]=L[f,c]`) → for each
/// 256-column block in order: per row, trellis-encode the OBS-adjusted
/// `residual`, record dequant, propagate `(w−ŵ)/L[c,c]` to all later columns.
/// Intra-block coupling handled jointly by the trellis; inter-block by OBS.
#[allow(clippy::too_many_arguments)]
/// 2-bit LDLQ wrapper (preserves the original signature / callers).
pub fn qtip2_ldlq_dequant(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    h_rowmajor_f32: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    beam_width: usize,
    damp: f64,
) -> Option<Vec<f32>> {
    qtip_ldlq_dequant_bits(
        weights_f32,
        m,
        k,
        h_rowmajor_f32,
        signs1,
        signs2,
        beam_width,
        damp,
        2,
        &crate::qtip::build_codebook(),
    )
}

/// Bit-parametric QTIP-LDLQ: same block-trellis OBS encode for any bit-rate.
/// The bitshift codebook is indexed by the 12-bit trellis *state* (independent
/// of bits-per-weight), so only the per-step symbol count differs — encode /
/// optimal-scale / decode route to the `_bits` variants.
#[allow(clippy::too_many_arguments)]
pub fn qtip_ldlq_dequant_bits(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    h_rowmajor_f32: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    beam_width: usize,
    damp: f64,
    bits: u32,
    cb: &[f32],
) -> Option<Vec<f32>> {
    use rayon::prelude::*;
    assert_eq!(weights_f32.len(), m * k);
    assert_eq!(h_rowmajor_f32.len(), k * k);
    assert_eq!(k % 256, 0, "qtip_ldlq_dequant_bits requires k % 256 == 0");

    // Rotate the Hessian, then L with L·Lᵀ = (H_rot + λI)⁻¹.
    let mut h: Vec<f64> = h_rowmajor_f32.iter().map(|&v| v as f64).collect();
    rotate_hessian(&mut h, k, signs1, signs2);
    let l = inv_cholesky_lower_rotated_fast(&h, k, damp)?;

    // Rotate the weights into the same domain.
    let nb = k / 256;
    let mut residual = vec![0.0f64; m * k];
    residual
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(row, rr)| {
            let base = row * k;
            let mut buf = [0.0f32; 256];
            for b in 0..nb {
                for c in 0..256 {
                    buf[c] = weights_f32[base + b * 256 + c];
                }
                crate::cpu_fwht_256(&mut buf, signs1, signs2);
                for c in 0..256 {
                    rr[b * 256 + c] = buf[c] as f64;
                }
            }
        });

    // Codebook threaded from the caller (1MAD or 3INST) — must match the
    // beam_encode / decode the on-device kernel computes.
    let mut dequant = vec![0.0f64; m * k];

    for blk in 0..nb {
        let c0 = blk * 256;
        let c1 = c0 + 256;
        let errs: Vec<Vec<f64>> = dequant
            .par_chunks_mut(k)
            .enumerate()
            .map(|(row, dr)| {
                let rbase = row * k;
                let mut grp = [0.0f32; 256];
                for (c, g) in grp.iter_mut().enumerate() {
                    *g = residual[rbase + c0 + c] as f32;
                }
                let s0 = crate::qtip::group_scale(&grp);
                let sym = crate::qtip::beam_encode_group_bits(&grp, s0, cb, beam_width, bits);
                let s = crate::qtip::optimal_scale_bits(&grp, &sym, cb, bits);
                let deq = crate::qtip::decode_group_bits(&sym, s, cb, bits);
                let mut err = vec![0.0f64; 256];
                for c in 0..256 {
                    dr[c0 + c] = deq[c] as f64;
                    let ucc = l[(c0 + c, c0 + c)];
                    err[c] = if ucc > 0.0 {
                        (residual[rbase + c0 + c] - deq[c] as f64) / ucc
                    } else {
                        0.0
                    };
                }
                err
            })
            .collect();

        if c1 < k {
            residual
                .par_chunks_mut(k)
                .zip(errs.par_iter())
                .for_each(|(rr, err)| {
                    for c in 0..256 {
                        let ec = err[c];
                        if ec == 0.0 {
                            continue;
                        }
                        let col = c0 + c;
                        for f in c1..k {
                            let usf = l[(f, col)]; // U[col,f] = L[f,col]
                            if usf != 0.0 {
                                rr[f] -= ec * usf;
                            }
                        }
                    }
                });
        }
    }

    // Un-rotate (FWHT orthogonal → swap sign args).
    let mut out = vec![0.0f32; m * k];
    out.par_chunks_mut(k).enumerate().for_each(|(row, orow)| {
        let mut buf = [0.0f32; 256];
        for b in 0..nb {
            for c in 0..256 {
                buf[c] = dequant[row * k + b * 256 + c] as f32;
            }
            crate::cpu_fwht_256(&mut buf, signs2, signs1);
            orow[b * 256..b * 256 + 256].copy_from_slice(&buf);
        }
    });
    Some(out)
}

/// Opus W4A4 (oq4) LDLQ: Hessian error-feedback **symmetric int4** weight quant,
/// emitting the SAME packed `[f16 scale][128 signed nibbles]` per-256-group layout
/// as `codecs::quantize_oq4g256` (130 B/group, row-major), but with the GPTQ/OBS
/// error feedback the plain RTN codec lacks.
///
/// Same machinery as `qtip_ldlq_dequant_bits`: FWHT-rotate H + W into the
/// incoherent domain (oq4 stores PRE-rotated weights, so the packed output stays
/// in the rotated domain — no un-rotate), `L` with `LLᵀ=(H_rot+λI)⁻¹`, then for
/// each 256-column block in order, per row: clip-search scale + round the
/// OBS-adjusted residual to signed int4, pack, and propagate `(w−ŵ)/L[c,c]` to
/// all later columns. The inner per-group quant is oq4's symmetric round (the only
/// substitution vs the QTIP trellis encode). `None` on Cholesky breakdown →
/// caller falls back to plain `quantize_oq4g256`.
pub fn oq4_ldlq_pack(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    h_rowmajor_f32: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    damp: f64,
) -> Option<Vec<u8>> {
    use rayon::prelude::*;
    assert_eq!(weights_f32.len(), m * k);
    assert_eq!(h_rowmajor_f32.len(), k * k);
    assert_eq!(k % 256, 0, "oq4_ldlq_pack requires k % 256 == 0");

    let mut h: Vec<f64> = h_rowmajor_f32.iter().map(|&v| v as f64).collect();
    rotate_hessian(&mut h, k, signs1, signs2);
    let l = inv_cholesky_lower_rotated_fast(&h, k, damp)?;

    let nb = k / 256;
    // FWHT-rotate weights into the same domain (this is the residual we feed).
    let mut residual = vec![0.0f64; m * k];
    residual
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(row, rr)| {
            let base = row * k;
            let mut buf = [0.0f32; 256];
            for b in 0..nb {
                for c in 0..256 {
                    buf[c] = weights_f32[base + b * 256 + c];
                }
                crate::cpu_fwht_256(&mut buf, signs1, signs2);
                for c in 0..256 {
                    rr[b * 256 + c] = buf[c] as f64;
                }
            }
        });

    const BLOCK_BYTES: usize = 130; // 2 (f16 scale) + 128 nibbles
    let mut out = vec![0u8; m * nb * BLOCK_BYTES];

    for blk in 0..nb {
        let c0 = blk * 256;
        let c1 = c0 + 256;
        // Per row: clip-search scale, round to int4, pack, compute OBS error.
        let results: Vec<(Vec<f64>, [u8; BLOCK_BYTES])> = residual
            .par_chunks(k)
            .map(|rr_row| {
                let mut grp = [0.0f32; 256];
                for (c, g) in grp.iter_mut().enumerate() {
                    *g = rr_row[c0 + c] as f32;
                }
                let scale = crate::codecs::symmetric_clipsearch(&grp, 7.0);
                let inv = 1.0 / scale;
                let mut block = [0u8; BLOCK_BYTES];
                let s16 = crate::f32_to_f16(scale);
                block[0] = (s16 & 0xFF) as u8;
                block[1] = (s16 >> 8) as u8;
                let mut err = vec![0.0f64; 256];
                for i in 0..128 {
                    let qlo = (grp[2 * i] * inv).round().clamp(-7.0, 7.0);
                    let qhi = (grp[2 * i + 1] * inv).round().clamp(-7.0, 7.0);
                    block[2 + i] = ((qlo as i8 as u8) & 0xf) | (((qhi as i8 as u8) & 0xf) << 4);
                    let (clo, chi) = (c0 + 2 * i, c0 + 2 * i + 1);
                    let ulo = l[(clo, clo)];
                    let uhi = l[(chi, chi)];
                    err[2 * i] = if ulo > 0.0 {
                        (grp[2 * i] as f64 - (qlo * scale) as f64) / ulo
                    } else {
                        0.0
                    };
                    err[2 * i + 1] = if uhi > 0.0 {
                        (grp[2 * i + 1] as f64 - (qhi * scale) as f64) / uhi
                    } else {
                        0.0
                    };
                }
                (err, block)
            })
            .collect();

        for (row, (_, block)) in results.iter().enumerate() {
            let off = (row * nb + blk) * BLOCK_BYTES;
            out[off..off + BLOCK_BYTES].copy_from_slice(block);
        }

        if c1 < k {
            residual
                .par_chunks_mut(k)
                .zip(results.par_iter())
                .for_each(|(rr, (err, _))| {
                    for (c, &ec) in err.iter().enumerate() {
                        if ec == 0.0 {
                            continue;
                        }
                        let col = c0 + c;
                        for f in c1..k {
                            let usf = l[(f, col)];
                            if usf != 0.0 {
                                rr[f] -= ec * usf;
                            }
                        }
                    }
                });
        }
    }

    Some(out)
}

/// Opus W3A4 (oq3) LDLQ: Hessian error-feedback **symmetric int3** weight quant,
/// emitting the SAME bit-plane `[f16 scale][8×(3 u32)]` per-256-group layout as
/// `codecs::quantize_oq3g256` (98 B/group = 3.0625 b/w), but with GPTQ/OBS error
/// feedback the plain RTN codec lacks. At 3 bits the quant grid is coarse (±3), so
/// error feedback matters MORE than at int4 — this is the calibrated `oq3++` packer.
///
/// Same machinery as [`oq4_ldlq_pack`]: FWHT-rotate H + W into the incoherent domain
/// (oq3 stores PRE-rotated weights, so the packed output stays rotated — no
/// un-rotate), `L` with `LLᵀ=(H_rot+λI)⁻¹`, then for each 256-column block in order,
/// per row: clip-search scale + round the OBS-adjusted residual to signed int3
/// `[-3,3]`, bit-plane pack, and propagate `(w−ŵ)/L[c,c]` to all later columns. Only
/// the inner quant range + packing differ from `oq4_ldlq_pack`. `None` on Cholesky
/// breakdown → caller falls back to plain `quantize_oq3g256`.
pub fn oq3_ldlq_pack(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    h_rowmajor_f32: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    damp: f64,
) -> Option<Vec<u8>> {
    use rayon::prelude::*;
    assert_eq!(weights_f32.len(), m * k);
    assert_eq!(h_rowmajor_f32.len(), k * k);
    assert_eq!(k % 256, 0, "oq3_ldlq_pack requires k % 256 == 0");

    let mut h: Vec<f64> = h_rowmajor_f32.iter().map(|&v| v as f64).collect();
    rotate_hessian(&mut h, k, signs1, signs2);
    let l = inv_cholesky_lower_rotated_fast(&h, k, damp)?;

    let nb = k / 256;
    let mut residual = vec![0.0f64; m * k];
    residual
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(row, rr)| {
            let base = row * k;
            let mut buf = [0.0f32; 256];
            for b in 0..nb {
                for c in 0..256 {
                    buf[c] = weights_f32[base + b * 256 + c];
                }
                crate::cpu_fwht_256(&mut buf, signs1, signs2);
                for c in 0..256 {
                    rr[b * 256 + c] = buf[c] as f64;
                }
            }
        });

    const BLOCK_BYTES: usize = 98; // 2 (f16 scale) + 8×3 u32 bit-planes
    let mut out = vec![0u8; m * nb * BLOCK_BYTES];

    for blk in 0..nb {
        let c0 = blk * 256;
        let c1 = c0 + 256;
        let results: Vec<(Vec<f64>, [u8; BLOCK_BYTES])> = residual
            .par_chunks(k)
            .map(|rr_row| {
                let mut grp = [0.0f32; 256];
                for (c, g) in grp.iter_mut().enumerate() {
                    *g = rr_row[c0 + c] as f32;
                }
                let scale = crate::codecs::symmetric_clipsearch(&grp, 3.0);
                let inv = 1.0 / scale;
                let mut block = [0u8; BLOCK_BYTES];
                let s16 = crate::f32_to_f16(scale);
                block[0] = (s16 & 0xFF) as u8;
                block[1] = (s16 >> 8) as u8;
                let mut err = vec![0.0f64; 256];
                for s in 0..8 {
                    let (mut p0, mut p1, mut p2) = (0u32, 0u32, 0u32);
                    for i in 0..32 {
                        let idx = s * 32 + i;
                        let q = (grp[idx] * inv).round().clamp(-3.0, 3.0);
                        let u = (q as i8 as u8) & 7; // 3-bit two's-complement
                        p0 |= ((u & 1) as u32) << i;
                        p1 |= (((u >> 1) & 1) as u32) << i;
                        p2 |= (((u >> 2) & 1) as u32) << i;
                        let col = c0 + idx;
                        let ucc = l[(col, col)];
                        err[idx] = if ucc > 0.0 {
                            (grp[idx] as f64 - (q * scale) as f64) / ucc
                        } else {
                            0.0
                        };
                    }
                    let bo = 2 + s * 12;
                    block[bo..bo + 4].copy_from_slice(&p0.to_le_bytes());
                    block[bo + 4..bo + 8].copy_from_slice(&p1.to_le_bytes());
                    block[bo + 8..bo + 12].copy_from_slice(&p2.to_le_bytes());
                }
                (err, block)
            })
            .collect();

        for (row, (_, block)) in results.iter().enumerate() {
            let off = (row * nb + blk) * BLOCK_BYTES;
            out[off..off + BLOCK_BYTES].copy_from_slice(block);
        }

        if c1 < k {
            residual
                .par_chunks_mut(k)
                .zip(results.par_iter())
                .for_each(|(rr, (err, _))| {
                    for (c, &ec) in err.iter().enumerate() {
                        if ec == 0.0 {
                            continue;
                        }
                        let col = c0 + c;
                        for f in c1..k {
                            let usf = l[(f, col)];
                            if usf != 0.0 {
                                rr[f] -= ec * usf;
                            }
                        }
                    }
                });
        }
    }

    Some(out)
}

/// Opus W2 (oq2) LDLQ: Hessian error-feedback symmetric int2 weight quant,
/// emitting the same `[f16 scale][64 B 2-bit×256]` per-256-group layout as
/// `codecs::quantize_oq2g256`. This is the calibrated `oq2++` packer. The
/// error-feedback machinery is identical to [`oq3_ldlq_pack`]; only the grid
/// (±1, 3 levels) and packing (4 weights/byte) differ. At 2 bits the grid is
/// extremely coarse, so OBS error feedback matters most here — though 2-bit
/// stays quality-marginal (see project_lowbit_quant_findings). Served via the
/// Oq8 upcast loader (`expand_oq2_to_oq8`).
pub fn oq2_ldlq_pack(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    h_rowmajor_f32: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    damp: f64,
) -> Option<Vec<u8>> {
    use rayon::prelude::*;
    assert_eq!(weights_f32.len(), m * k);
    assert_eq!(h_rowmajor_f32.len(), k * k);
    assert_eq!(k % 256, 0, "oq2_ldlq_pack requires k % 256 == 0");

    let mut h: Vec<f64> = h_rowmajor_f32.iter().map(|&v| v as f64).collect();
    rotate_hessian(&mut h, k, signs1, signs2);
    let l = inv_cholesky_lower_rotated_fast(&h, k, damp)?;

    let nb = k / 256;
    let mut residual = vec![0.0f64; m * k];
    residual
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(row, rr)| {
            let base = row * k;
            let mut buf = [0.0f32; 256];
            for b in 0..nb {
                for c in 0..256 {
                    buf[c] = weights_f32[base + b * 256 + c];
                }
                crate::cpu_fwht_256(&mut buf, signs1, signs2);
                for c in 0..256 {
                    rr[b * 256 + c] = buf[c] as f64;
                }
            }
        });

    const BLOCK_BYTES: usize = 66; // 2 (f16 scale) + 64 (2-bit×256)
    let mut out = vec![0u8; m * nb * BLOCK_BYTES];

    for blk in 0..nb {
        let c0 = blk * 256;
        let c1 = c0 + 256;
        let results: Vec<(Vec<f64>, [u8; BLOCK_BYTES])> = residual
            .par_chunks(k)
            .map(|rr_row| {
                let mut grp = [0.0f32; 256];
                for (c, g) in grp.iter_mut().enumerate() {
                    *g = rr_row[c0 + c] as f32;
                }
                let scale = crate::codecs::symmetric_clipsearch(&grp, 1.0);
                let inv = 1.0 / scale;
                let mut block = [0u8; BLOCK_BYTES];
                let s16 = crate::f32_to_f16(scale);
                block[0] = (s16 & 0xFF) as u8;
                block[1] = (s16 >> 8) as u8;
                let mut err = vec![0.0f64; 256];
                for byte_i in 0..64 {
                    let mut packed = 0u8;
                    for j in 0..4 {
                        let idx = byte_i * 4 + j;
                        let q = (grp[idx] * inv).round().clamp(-1.0, 1.0);
                        packed |= ((q as i8 as u8) & 3) << (2 * j);
                        let col = c0 + idx;
                        let ucc = l[(col, col)];
                        err[idx] = if ucc > 0.0 {
                            (grp[idx] as f64 - (q * scale) as f64) / ucc
                        } else {
                            0.0
                        };
                    }
                    block[2 + byte_i] = packed;
                }
                (err, block)
            })
            .collect();

        for (row, (_, block)) in results.iter().enumerate() {
            let off = (row * nb + blk) * BLOCK_BYTES;
            out[off..off + BLOCK_BYTES].copy_from_slice(block);
        }

        if c1 < k {
            residual
                .par_chunks_mut(k)
                .zip(results.par_iter())
                .for_each(|(rr, (err, _))| {
                    for (c, &ec) in err.iter().enumerate() {
                        if ec == 0.0 {
                            continue;
                        }
                        let col = c0 + c;
                        for f in c1..k {
                            let usf = l[(f, col)];
                            if usf != 0.0 {
                                rr[f] -= ec * usf;
                            }
                        }
                    }
                });
        }
    }

    Some(out)
}

/// Opus W8A8 (oq8) LDLQ: Hessian error-feedback symmetric int8 weight quant,
/// emitting the same `[f16 scale][256 signed int8]` per-256-group layout as
/// `codecs::quantize_oq8g256`. This is the calibrated `op8+` packer: FWHT-rotate
/// H and W into the stored OQ8 domain, round the OBS-adjusted residual to int8,
/// and propagate the remaining error to later groups.
pub fn oq8_ldlq_pack(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    h_rowmajor_f32: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    damp: f64,
) -> Option<Vec<u8>> {
    use rayon::prelude::*;
    assert_eq!(weights_f32.len(), m * k);
    assert_eq!(h_rowmajor_f32.len(), k * k);
    assert_eq!(k % 256, 0, "oq8_ldlq_pack requires k % 256 == 0");

    let mut h: Vec<f64> = h_rowmajor_f32.iter().map(|&v| v as f64).collect();
    rotate_hessian(&mut h, k, signs1, signs2);
    let l = inv_cholesky_lower_rotated_fast(&h, k, damp)?;

    let nb = k / 256;
    let mut residual = vec![0.0f64; m * k];
    residual
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(row, rr)| {
            let base = row * k;
            let mut buf = [0.0f32; 256];
            for b in 0..nb {
                for c in 0..256 {
                    buf[c] = weights_f32[base + b * 256 + c];
                }
                crate::cpu_fwht_256(&mut buf, signs1, signs2);
                for c in 0..256 {
                    rr[b * 256 + c] = buf[c] as f64;
                }
            }
        });

    const BLOCK_BYTES: usize = 258; // 2 (f16 scale) + 256 int8
    let mut out = vec![0u8; m * nb * BLOCK_BYTES];

    for blk in 0..nb {
        let c0 = blk * 256;
        let c1 = c0 + 256;
        let results: Vec<(Vec<f64>, [u8; BLOCK_BYTES])> = residual
            .par_chunks(k)
            .map(|rr_row| {
                let mut grp = [0.0f32; 256];
                for (c, g) in grp.iter_mut().enumerate() {
                    *g = rr_row[c0 + c] as f32;
                }
                let scale = crate::codecs::symmetric_clipsearch(&grp, 127.0);
                let inv = 1.0 / scale;
                let mut block = [0u8; BLOCK_BYTES];
                let s16 = crate::f32_to_f16(scale);
                block[0] = (s16 & 0xFF) as u8;
                block[1] = (s16 >> 8) as u8;
                let mut err = vec![0.0f64; 256];
                for i in 0..256 {
                    let q = (grp[i] * inv).round().clamp(-127.0, 127.0);
                    block[2 + i] = q as i8 as u8;
                    let u = l[(c0 + i, c0 + i)];
                    err[i] = if u > 0.0 {
                        (grp[i] as f64 - (q * scale) as f64) / u
                    } else {
                        0.0
                    };
                }
                (err, block)
            })
            .collect();

        for (row, (_, block)) in results.iter().enumerate() {
            let off = (row * nb + blk) * BLOCK_BYTES;
            out[off..off + BLOCK_BYTES].copy_from_slice(block);
        }

        if c1 < k {
            residual
                .par_chunks_mut(k)
                .zip(results.par_iter())
                .for_each(|(rr, (err, _))| {
                    for (c, &ec) in err.iter().enumerate() {
                        if ec == 0.0 {
                            continue;
                        }
                        let col = c0 + c;
                        for f in c1..k {
                            let usf = l[(f, col)];
                            if usf != 0.0 {
                                rr[f] -= ec * usf;
                            }
                        }
                    }
                });
        }
    }

    Some(out)
}

/// LDLQ (GPTQ/OBS error-feedback) packer for the magnitude-tiered OQ+ (Opus
/// Plus) format: bulk int4 + top-`w8_frac` weights per group at int8, emitting the
/// **Oq8 on-disk layout** (`[f16 scale][256 signed int8]`, 258 B/group) so it
/// loads via the existing qt=35 path and the single iu8 W8A8 kernel.
///
/// Same machinery as [`oq4_ldlq_pack`] — FWHT-rotate H + W into the incoherent
/// domain, `L` with `LLᵀ=(H_rot+λI)⁻¹`, then per 256-block (in order) per row:
/// clip-search an INT4-tuned scale, pick the top-`w8_frac` positions by
/// int8-upgrade gain, round those to int8 `[-127,127]` and the bulk to int4
/// `[-7,7]` (same scale), and propagate `(w−ŵ)/L[c,c]` to later blocks. The win
/// vs plain tiering: the int8 outlier positions have ~zero residual, so the OBS
/// feedback spends its budget compensating only the int4 bulk.
#[allow(clippy::too_many_arguments)]
pub fn oqplus_tiered_ldlq_pack(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    h_rowmajor_f32: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    damp: f64,
    w8_frac: f32,
) -> Option<Vec<u8>> {
    use rayon::prelude::*;
    assert_eq!(weights_f32.len(), m * k);
    assert_eq!(h_rowmajor_f32.len(), k * k);
    assert_eq!(k % 256, 0, "oqplus_tiered_ldlq_pack requires k % 256 == 0");

    let mut h: Vec<f64> = h_rowmajor_f32.iter().map(|&v| v as f64).collect();
    rotate_hessian(&mut h, k, signs1, signs2);
    let l = inv_cholesky_lower_rotated_fast(&h, k, damp)?;

    let nb = k / 256;
    let mut residual = vec![0.0f64; m * k];
    residual
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(row, rr)| {
            let base = row * k;
            let mut buf = [0.0f32; 256];
            for b in 0..nb {
                for c in 0..256 {
                    buf[c] = weights_f32[base + b * 256 + c];
                }
                crate::cpu_fwht_256(&mut buf, signs1, signs2);
                for c in 0..256 {
                    rr[b * 256 + c] = buf[c] as f64;
                }
            }
        });

    const BLOCK_BYTES: usize = 258; // 2 (f16 scale) + 256 int8
    let n_out = ((w8_frac as f64 * 256.0).round() as usize).clamp(1, 256);
    let mut out = vec![0u8; m * nb * BLOCK_BYTES];

    for blk in 0..nb {
        let c0 = blk * 256;
        let c1 = c0 + 256;
        let results: Vec<(Vec<f64>, [u8; BLOCK_BYTES])> = residual
            .par_chunks(k)
            .map(|rr_row| {
                let mut grp = [0.0f32; 256];
                for (c, g) in grp.iter_mut().enumerate() {
                    *g = rr_row[c0 + c] as f32;
                }
                let scale = crate::codecs::symmetric_clipsearch(&grp, 7.0);
                let inv = 1.0 / scale;
                // Outliers: top n_out by int8-upgrade gain (int4_err² − int8_err²).
                let gain = |i: usize| -> f32 {
                    let v = grp[i];
                    let q4 = (v * inv).round().clamp(-7.0, 7.0);
                    let q8 = (v * inv).round().clamp(-127.0, 127.0);
                    let e4 = v - q4 * scale;
                    let e8 = v - q8 * scale;
                    e4 * e4 - e8 * e8
                };
                let mut idx: [usize; 256] = core::array::from_fn(|i| i);
                idx.sort_unstable_by(|&a, &c| {
                    gain(c)
                        .partial_cmp(&gain(a))
                        .unwrap_or(core::cmp::Ordering::Equal)
                        .then_with(|| a.cmp(&c))
                });
                let mut is_w8 = [false; 256];
                for &i in &idx[..n_out] {
                    is_w8[i] = true;
                }
                let mut block = [0u8; BLOCK_BYTES];
                let s16 = crate::f32_to_f16(scale);
                block[0] = (s16 & 0xFF) as u8;
                block[1] = (s16 >> 8) as u8;
                let mut err = vec![0.0f64; 256];
                for i in 0..256 {
                    let lim = if is_w8[i] { 127.0 } else { 7.0 };
                    let q = (grp[i] * inv).round().clamp(-lim, lim);
                    block[2 + i] = q as i8 as u8;
                    let u = l[(c0 + i, c0 + i)];
                    err[i] = if u > 0.0 {
                        (grp[i] as f64 - (q * scale) as f64) / u
                    } else {
                        0.0
                    };
                }
                (err, block)
            })
            .collect();

        for (row, (_, block)) in results.iter().enumerate() {
            let off = (row * nb + blk) * BLOCK_BYTES;
            out[off..off + BLOCK_BYTES].copy_from_slice(block);
        }

        if c1 < k {
            residual
                .par_chunks_mut(k)
                .zip(results.par_iter())
                .for_each(|(rr, (err, _))| {
                    for (c, &ec) in err.iter().enumerate() {
                        if ec == 0.0 {
                            continue;
                        }
                        let col = c0 + c;
                        for f in c1..k {
                            let usf = l[(f, col)];
                            if usf != 0.0 {
                                rr[f] -= ec * usf;
                            }
                        }
                    }
                });
        }
    }

    Some(out)
}

/// COMPACT (~4 b/w) variant of [`oqplus_tiered_ldlq_pack`]: identical GPTQ/OBS
/// error-feedback and tiered quantization, but emits the compact per-256-group
/// layout `[f16 scale][128 int4 nibbles][N_out × (u8 idx, i8 val)]` (matching
/// `codecs::quantize_oqplus_compact`) instead of the int8 Oq8 layout. The OBS
/// residual uses the actual tiered value (int8 for outliers → ~0 error → little
/// propagation; int4 for the bulk), so the feedback compensates the bulk exactly
/// as in the int8 packer — same dequantized values, ~half the bytes.
#[allow(clippy::too_many_arguments)]
pub fn oqplus_compact_ldlq_pack(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    h_rowmajor_f32: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    damp: f64,
    w8_frac: f32,
) -> Option<Vec<u8>> {
    use rayon::prelude::*;
    assert_eq!(weights_f32.len(), m * k);
    assert_eq!(h_rowmajor_f32.len(), k * k);
    assert_eq!(k % 256, 0, "oqplus_compact_ldlq_pack requires k % 256 == 0");

    let mut h: Vec<f64> = h_rowmajor_f32.iter().map(|&v| v as f64).collect();
    rotate_hessian(&mut h, k, signs1, signs2);
    let l = inv_cholesky_lower_rotated_fast(&h, k, damp)?;

    let nb = k / 256;
    let mut residual = vec![0.0f64; m * k];
    residual
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(row, rr)| {
            let base = row * k;
            let mut buf = [0.0f32; 256];
            for b in 0..nb {
                for c in 0..256 {
                    buf[c] = weights_f32[base + b * 256 + c];
                }
                crate::cpu_fwht_256(&mut buf, signs1, signs2);
                for c in 0..256 {
                    rr[b * 256 + c] = buf[c] as f64;
                }
            }
        });

    let n_out = ((w8_frac as f64 * 256.0).round() as usize).clamp(1, 255);
    let block_bytes = 130 + 2 * n_out; // [f16][128 nibbles][n_out×(u8 idx, i8 val)]
    let mut out = vec![0u8; m * nb * block_bytes];

    for blk in 0..nb {
        let c0 = blk * 256;
        let c1 = c0 + 256;
        let results: Vec<(Vec<f64>, Vec<u8>)> = residual
            .par_chunks(k)
            .map(|rr_row| {
                let mut grp = [0.0f32; 256];
                for (c, g) in grp.iter_mut().enumerate() {
                    *g = rr_row[c0 + c] as f32;
                }
                let scale = crate::codecs::symmetric_clipsearch(&grp, 7.0);
                let inv = 1.0 / scale;
                let gain = |i: usize| -> f32 {
                    let v = grp[i];
                    let q4 = (v * inv).round().clamp(-7.0, 7.0);
                    let q8 = (v * inv).round().clamp(-127.0, 127.0);
                    let e4 = v - q4 * scale;
                    let e8 = v - q8 * scale;
                    e4 * e4 - e8 * e8
                };
                let mut idx: [usize; 256] = core::array::from_fn(|i| i);
                idx.sort_unstable_by(|&a, &c| {
                    gain(c)
                        .partial_cmp(&gain(a))
                        .unwrap_or(core::cmp::Ordering::Equal)
                        .then_with(|| a.cmp(&c))
                });
                let mut is_w8 = [false; 256];
                for &i in &idx[..n_out] {
                    is_w8[i] = true;
                }
                let mut block = vec![0u8; block_bytes];
                let s16 = crate::f32_to_f16(scale);
                block[0] = (s16 & 0xFF) as u8;
                block[1] = (s16 >> 8) as u8;
                let mut err = vec![0.0f64; 256];
                // Quantize each position to its tier; nibbles store the int4 clamp
                // (outlier slots overridden by the sparse table on load), err uses
                // the ACTUAL tiered value.
                for i in 0..128 {
                    let q4lo = (grp[2 * i] * inv).round().clamp(-7.0, 7.0) as i8;
                    let q4hi = (grp[2 * i + 1] * inv).round().clamp(-7.0, 7.0) as i8;
                    block[2 + i] = ((q4lo as u8) & 0xf) | (((q4hi as u8) & 0xf) << 4);
                }
                for i in 0..256 {
                    let lim = if is_w8[i] { 127.0 } else { 7.0 };
                    let q = (grp[i] * inv).round().clamp(-lim, lim);
                    let u = l[(c0 + i, c0 + i)];
                    err[i] = if u > 0.0 {
                        (grp[i] as f64 - (q * scale) as f64) / u
                    } else {
                        0.0
                    };
                }
                let tbl = 130;
                for (s, &pos) in idx[..n_out].iter().enumerate() {
                    let q8 = (grp[pos] * inv).round().clamp(-127.0, 127.0) as i8;
                    block[tbl + 2 * s] = pos as u8;
                    block[tbl + 2 * s + 1] = q8 as u8;
                }
                (err, block)
            })
            .collect();

        for (row, (_, block)) in results.iter().enumerate() {
            let off = (row * nb + blk) * block_bytes;
            out[off..off + block_bytes].copy_from_slice(block);
        }

        if c1 < k {
            residual
                .par_chunks_mut(k)
                .zip(results.par_iter())
                .for_each(|(rr, (err, _))| {
                    for (c, &ec) in err.iter().enumerate() {
                        if ec == 0.0 {
                            continue;
                        }
                        let col = c0 + c;
                        for f in c1..k {
                            let usf = l[(f, col)];
                            if usf != 0.0 {
                                rr[f] -= ec * usf;
                            }
                        }
                    }
                });
        }
    }

    Some(out)
}

#[cfg(test)]
mod chol_tests {
    use super::*;

    /// Build an ill-conditioned SPD matrix: a rank-deficient Gram plus a small
    /// ridge, which is what a real Hessian looks like here and what makes the
    /// damping-retry loop matter.
    fn ill_conditioned(k: usize, ridge: f64) -> Vec<f64> {
        let r = k / 4; // rank-deficient by construction
        let mut a = vec![0.0f64; k * r];
        for i in 0..k {
            for j in 0..r {
                a[i * r + j] = ((i * 7 + j * 13) % 23) as f64 / 23.0 - 0.5;
            }
        }
        let mut h = vec![0.0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut acc = 0.0;
                for t in 0..r {
                    acc += a[i * r + t] * a[j * r + t];
                }
                h[i * k + j] = acc + if i == j { ridge } else { 0.0 };
            }
        }
        h
    }

    /// The identity-free variant must reproduce the original EXACTLY in shape and
    /// to tight tolerance in value. Comparing the outputs directly is both
    /// cheaper and stronger than comparing downstream KLD: a subtly wrong L could
    /// move KLD by less than run-to-run noise and pass, whereas any deviation
    /// shows up here.
    #[test]
    fn identity_free_inverse_cholesky_matches_the_original() {
        for (k, ridge, damp) in [(256usize, 1e-3, 1e-2), (256, 1e-8, 1e-6)] {
            let h = ill_conditioned(k, ridge);
            let a = inv_cholesky_lower_rotated(&h, k, damp).expect("original");
            let b = inv_cholesky_lower_rotated_fast(&h, k, damp).expect("fast");

            // Shape first: LOWER triangular. C^-T also satisfies L·Lᵀ = A⁻¹ but is
            // UPPER, and swapping it in degrades LDLQ to RTN with every residual
            // check still passing.
            let mut upper = 0.0f64;
            for i in 0..k {
                for j in (i + 1)..k {
                    upper = upper.max(b[(i, j)].abs());
                }
            }
            assert_eq!(upper, 0.0, "fast variant is not lower-triangular (k={k})");

            let scale = (0..k).map(|i| a[(i, i)].abs()).fold(0.0f64, f64::max);
            let mut worst = 0.0f64;
            for i in 0..k {
                for j in 0..=i {
                    worst = worst.max((a[(i, j)] - b[(i, j)]).abs());
                }
            }
            assert!(
                worst / scale < 1e-9,
                "fast variant differs by {worst} (rel {:.3e}, k={k}, ridge={ridge})",
                worst / scale
            );
        }
    }

    /// Not a correctness test — a throughput probe. The greedy pipeline spends
    /// ~400s of ~600s in this routine. MEASURED: faer already uses all 32 rayon
    /// threads by default (Par::rayon(0) resolves to current_num_threads) and
    /// setting it explicitly changes nothing, so the low rate is NOT a
    /// misconfiguration — Cholesky just scales badly at these sizes, because the
    /// panel factorization serialises. That is the argument for moving the
    /// trailing SYRK to the GPU rather than tuning the CPU path.
    #[test]
    fn cholesky_throughput_probe() {
        let k = 2048usize;
        let h = ill_conditioned(k, 1e-3);
        println!(
            "  rayon threads={}  faer par={:?}",
            rayon::current_num_threads(),
            faer::get_global_parallelism()
        );
        let t = std::time::Instant::now();
        let l = inv_cholesky_lower_rotated_fast(&h, k, 1e-2).expect("factorization");
        let secs = t.elapsed().as_secs_f64();

        // two Cholesky factorizations + one inverse, each ~k^3/3
        let flops = 3.0 * (k as f64).powi(3) / 3.0;
        println!(
            "  k={k}  {:.3}s  {:.1} GFLOP/s  (one AVX-512 core is ~20-50)",
            secs,
            flops / secs / 1e9
        );
        assert_eq!(l.nrows(), k);
    }

    /// The GPU right-looking factorization must reproduce faer's `llt` on the
    /// same matrix. Tolerance is f32-scale, not f64: the trailing update runs in
    /// f32 by design, so exact agreement is not the claim — agreement to within
    /// single precision is, and that is what decides whether this is usable.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_right_looking_matches_faer_llt() {
        let Ok(mut gpu) = hipfire_rdna::Gpu::init() else {
            eprintln!("no GPU — skipping");
            return;
        };
        let k = 512usize;
        let h = ill_conditioned(k, 1e-1); // well-conditioned enough for a clean compare
        let hd = Mat::<f64>::from_fn(k, k, |i, j| h[i * k + j]);
        let reference = hd.llt(Side::Lower).expect("faer llt").L().to_owned();

        let got = super::llt_lower_right_looking_gpu(&mut gpu, &h, k).expect("gpu llt");

        let scale = (0..k).map(|i| reference[(i, i)].abs()).fold(0.0f64, f64::max);
        let mut worst = 0.0f64;
        for i in 0..k {
            for j in 0..=i {
                worst = worst.max((reference[(i, j)] - got[i * k + j]).abs());
            }
        }
        println!("  gpu-vs-faer worst |dL| = {:.3e} (rel {:.3e})", worst, worst / scale);
        assert!(
            worst / scale < 1e-4,
            "GPU factorization differs by rel {:.3e} — f32 trailing update is not \
             accurate enough for this matrix",
            worst / scale
        );
    }

    /// Is the GPU path actually faster? The draft round-trips the whole k×k per
    /// block, so this is not a foregone conclusion — the transfers may cost more
    /// than the trailing update saves.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_right_looking_throughput_vs_faer() {
        let Ok(mut gpu) = hipfire_rdna::Gpu::init() else {
            eprintln!("no GPU — skipping");
            return;
        };
        for k in [1024usize, 2048] {
            let h = ill_conditioned(k, 1e-1);
            let hd = Mat::<f64>::from_fn(k, k, |i, j| h[i * k + j]);
            let t = std::time::Instant::now();
            let _ = hd.llt(Side::Lower).expect("faer");
            let cpu = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let _ = super::llt_lower_right_looking_gpu(&mut gpu, &h, k).expect("gpu");
            let gpu_s = t.elapsed().as_secs_f64();
            let f = (k as f64).powi(3) / 3.0;
            println!(
                "  k={k}  faer {:.3}s ({:.1} GF/s)   gpu {:.3}s ({:.1} GF/s)   speedup {:.2}x",
                cpu, f / cpu / 1e9, gpu_s, f / gpu_s / 1e9, cpu / gpu_s
            );
        }
    }

    /// Both variants must agree that a hopeless matrix is hopeless — the retry
    /// loop is where a rewrite most easily diverges, because it picks a different
    /// lambda on each pass.
    #[test]
    fn both_variants_agree_on_breakdown() {
        let k = 64usize;
        let h = vec![f64::NAN; k * k];
        assert!(inv_cholesky_lower_rotated(&h, k, 1e-6).is_none());
        assert!(inv_cholesky_lower_rotated_fast(&h, k, 1e-6).is_none());
    }
}

#[cfg(test)]
mod rotate_tests {
    use super::*;

    /// The parallel rotate_hessian does the column pass as `transpose ->
    /// row-pass -> transpose`. That is only valid if it reproduces the original
    /// serial row-then-column loops EXACTLY — a silently different rotation
    /// would put H in a frame the trellis does not see, and would surface as a
    /// quality regression rather than an error.
    fn rotate_hessian_serial(h: &mut [f64], k: usize, s1: &[f32], s2: &[f32]) {
        let nb = k / 256;
        let mut buf = [0.0f32; 256];
        for r in 0..k {
            for b in 0..nb {
                for c in 0..256 {
                    buf[c] = h[r * k + b * 256 + c] as f32;
                }
                crate::cpu_fwht_256(&mut buf, s1, s2);
                for c in 0..256 {
                    h[r * k + b * 256 + c] = buf[c] as f64;
                }
            }
        }
        for col in 0..k {
            for b in 0..nb {
                for r in 0..256 {
                    buf[r] = h[(b * 256 + r) * k + col] as f32;
                }
                crate::cpu_fwht_256(&mut buf, s1, s2);
                for r in 0..256 {
                    h[(b * 256 + r) * k + col] = buf[r] as f64;
                }
            }
        }
    }

    #[test]
    fn parallel_rotate_hessian_matches_the_serial_loops() {
        let k = 512usize; // two 256-blocks, so both passes cross a block boundary
        let s1 = crate::gen_fwht_signs(42, 256);
        let s2 = crate::gen_fwht_signs(1042, 256);
        // Symmetric, non-trivial, and not separable across blocks.
        let mut h = vec![0.0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                let v = ((i * 31 + j * 17) % 97) as f64 / 97.0 + if i == j { 3.0 } else { 0.0 };
                h[i * k + j] = v;
                h[j * k + i] = v;
            }
        }
        let mut a = h.clone();
        let mut b = h;
        rotate_hessian(&mut a, k, &s1, &s2);
        rotate_hessian_serial(&mut b, k, &s1, &s2);
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        assert!(worst == 0.0, "parallel rotation differs by {worst}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic LCG for a reproducible SPD matrix.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        }
    }

    #[test]
    fn inv_cholesky_reconstructs_inverse() {
        let k = 64;
        let mut rng = Lcg(0xABCD);
        // A = random; H = AᵀA (SPD) + ridge.
        let mut a = vec![0.0f64; k * k];
        for v in a.iter_mut() {
            *v = rng.next();
        }
        let mut h = vec![0.0f32; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0;
                for t in 0..k {
                    s += a[t * k + i] * a[t * k + j];
                }
                h[i * k + j] = s as f32;
            }
        }
        let damp = 1e-3;
        for i in 0..k {
            h[i * k + i] += damp as f32;
        }
        let l = inv_cholesky_lower(&h, k, 0.0).expect("SPD"); // ridge already in h

        // Verify (H)·(L·Lᵀ) ≈ I  ⇔  L·Lᵀ = H⁻¹.
        // Compute M = H · (L·Lᵀ), check ≈ identity.
        let llt = |r: usize, c: usize| -> f64 {
            // (L·Lᵀ)[r,c] = Σ_t L[r,t] L[c,t]
            let mut s = 0.0;
            for t in 0..=r.min(c) {
                s += l[(r, t)] * l[(c, t)];
            }
            s
        };
        let mut max_off = 0.0f64;
        let mut max_diag_err = 0.0f64;
        for i in 0..k {
            for j in 0..k {
                let mut m = 0.0;
                for t in 0..k {
                    m += h[i * k + t] as f64 * llt(t, j);
                }
                if i == j {
                    max_diag_err = max_diag_err.max((m - 1.0).abs());
                } else {
                    max_off = max_off.max(m.abs());
                }
            }
        }
        eprintln!("inv-cholesky: max|diag-1|={max_diag_err:.2e} max|offdiag|={max_off:.2e}");
        assert!(max_diag_err < 1e-4, "diag err {max_diag_err}");
        assert!(max_off < 1e-4, "offdiag err {max_off}");
    }

    /// The LDLQ claim: OBS feedback (real H) reduces the H-weighted *output*
    /// error vs no feedback (identity H) on column-correlated weights — the
    /// fix for the PPL-125 MSE-only QTIP-2.
    #[test]
    fn qtip2_ldlq_obs_beats_no_feedback() {
        let (m, k) = (1024usize, 512usize);
        let mut rng = Lcg(0x1357);
        let mut w = vec![0.0f32; m * k];
        for row in 0..m {
            let mut prev = 0.0f64;
            for c in 0..k {
                prev = 0.85 * prev + rng.next(); // AR(1) across columns
                w[row * k + c] = prev as f32;
            }
        }
        // H = (1/m) Σ w_rowᵀ w_row + ridge.
        let mut h = vec![0.0f32; k * k];
        for row in 0..m {
            let base = row * k;
            for i in 0..k {
                let wi = w[base + i];
                if wi == 0.0 {
                    continue;
                }
                for j in 0..k {
                    h[i * k + j] += wi * w[base + j];
                }
            }
        }
        for v in h.iter_mut() {
            *v /= m as f32;
        }
        for i in 0..k {
            h[i * k + i] += 1e-2;
        }
        let ident: Vec<f32> = (0..k * k)
            .map(|x| if x / k == x % k { 1.0 } else { 0.0 })
            .collect();
        let s1 = crate::gen_fwht_signs(42, 256);
        let s2 = crate::gen_fwht_signs(1042, 256);

        let deq_h = qtip2_ldlq_dequant(&w, m, k, &h, &s1, &s2, 64, 1e-2).expect("ldlq H");
        let deq_i = qtip2_ldlq_dequant(&w, m, k, &ident, &s1, &s2, 64, 1e-2).expect("ldlq I");

        let out_err = |deq: &[f32]| -> f64 {
            let mut tot = 0.0f64;
            for row in 0..m {
                let base = row * k;
                let d: Vec<f64> = (0..k)
                    .map(|c| (w[base + c] - deq[base + c]) as f64)
                    .collect();
                for i in 0..k {
                    if d[i] == 0.0 {
                        continue;
                    }
                    let mut acc = 0.0;
                    for j in 0..k {
                        acc += h[i * k + j] as f64 * d[j];
                    }
                    tot += d[i] * acc;
                }
            }
            tot
        };
        let (eh, ei) = (out_err(&deq_h), out_err(&deq_i));
        eprintln!(
            "qtip2-ldlq output-err: H-OBS={eh:.4} no-fb={ei:.4} ratio={:.3}",
            eh / ei
        );
        assert!(eh < ei, "OBS feedback must beat no-feedback: {eh} !< {ei}");
    }

    /// The int3 packer's OBS feedback (real H) must reduce the H-weighted *output*
    /// error vs no feedback (identity H) on column-correlated weights — the same
    /// LDLQ claim as the QTIP path, for the coarse ±3 grid where it matters most.
    /// Round-trips through the codec's `dequant_oq3g256` oracle (bit-plane decode).
    #[test]
    fn oq3_ldlq_obs_beats_no_feedback() {
        let (m, k) = (512usize, 512usize);
        let mut rng = Lcg(0x2468);
        let mut w = vec![0.0f32; m * k];
        for row in 0..m {
            let mut prev = 0.0f64;
            for c in 0..k {
                prev = 0.85 * prev + rng.next(); // AR(1) across columns
                w[row * k + c] = prev as f32;
            }
        }
        // H = (1/m) Σ w_rowᵀ w_row + ridge.
        let mut h = vec![0.0f32; k * k];
        for row in 0..m {
            let base = row * k;
            for i in 0..k {
                let wi = w[base + i];
                if wi == 0.0 {
                    continue;
                }
                for j in 0..k {
                    h[i * k + j] += wi * w[base + j];
                }
            }
        }
        for v in h.iter_mut() {
            *v /= m as f32;
        }
        for i in 0..k {
            h[i * k + i] += 1e-2;
        }
        let ident: Vec<f32> = (0..k * k)
            .map(|x| if x / k == x % k { 1.0 } else { 0.0 })
            .collect();
        let s1 = crate::gen_fwht_signs(42, 256);
        let s2 = crate::gen_fwht_signs(1042, 256);

        let pk_h = oq3_ldlq_pack(&w, m, k, &h, &s1, &s2, 1e-2).expect("oq3 ldlq H");
        let pk_i = oq3_ldlq_pack(&w, m, k, &ident, &s1, &s2, 1e-2).expect("oq3 ldlq I");
        let deq_h = crate::codecs::dequant_oq3g256(&pk_h, m * k, &s1, &s2);
        let deq_i = crate::codecs::dequant_oq3g256(&pk_i, m * k, &s1, &s2);

        let out_err = |deq: &[f32]| -> f64 {
            let mut tot = 0.0f64;
            for row in 0..m {
                let base = row * k;
                let d: Vec<f64> = (0..k)
                    .map(|c| (w[base + c] - deq[base + c]) as f64)
                    .collect();
                for i in 0..k {
                    if d[i] == 0.0 {
                        continue;
                    }
                    let mut acc = 0.0;
                    for j in 0..k {
                        acc += h[i * k + j] as f64 * d[j];
                    }
                    tot += d[i] * acc;
                }
            }
            tot
        };
        let (eh, ei) = (out_err(&deq_h), out_err(&deq_i));
        eprintln!(
            "oq3-ldlq output-err: H-OBS={eh:.4} no-fb={ei:.4} ratio={:.3}",
            eh / ei
        );
        assert!(eh < ei, "OBS feedback must beat no-feedback: {eh} !< {ei}");
    }

    #[test]
    fn oq2_ldlq_obs_beats_no_feedback() {
        let (m, k) = (512usize, 512usize);
        let mut rng = Lcg(0x2468);
        let mut w = vec![0.0f32; m * k];
        for row in 0..m {
            let mut prev = 0.0f64;
            for c in 0..k {
                prev = 0.85 * prev + rng.next(); // AR(1) across columns
                w[row * k + c] = prev as f32;
            }
        }
        // H = (1/m) Σ w_rowᵀ w_row + ridge.
        let mut h = vec![0.0f32; k * k];
        for row in 0..m {
            let base = row * k;
            for i in 0..k {
                let wi = w[base + i];
                if wi == 0.0 {
                    continue;
                }
                for j in 0..k {
                    h[i * k + j] += wi * w[base + j];
                }
            }
        }
        for v in h.iter_mut() {
            *v /= m as f32;
        }
        for i in 0..k {
            h[i * k + i] += 1e-2;
        }
        let ident: Vec<f32> = (0..k * k)
            .map(|x| if x / k == x % k { 1.0 } else { 0.0 })
            .collect();
        let s1 = crate::gen_fwht_signs(42, 256);
        let s2 = crate::gen_fwht_signs(1042, 256);

        let pk_h = oq2_ldlq_pack(&w, m, k, &h, &s1, &s2, 1e-2).expect("oq2 ldlq H");
        let pk_i = oq2_ldlq_pack(&w, m, k, &ident, &s1, &s2, 1e-2).expect("oq2 ldlq I");
        let deq_h = crate::codecs::dequant_oq2g256(&pk_h, m * k, &s1, &s2);
        let deq_i = crate::codecs::dequant_oq2g256(&pk_i, m * k, &s1, &s2);

        let out_err = |deq: &[f32]| -> f64 {
            let mut tot = 0.0f64;
            for row in 0..m {
                let base = row * k;
                let d: Vec<f64> = (0..k)
                    .map(|c| (w[base + c] - deq[base + c]) as f64)
                    .collect();
                for i in 0..k {
                    if d[i] == 0.0 {
                        continue;
                    }
                    let mut acc = 0.0;
                    for j in 0..k {
                        acc += h[i * k + j] as f64 * d[j];
                    }
                    tot += d[i] * acc;
                }
            }
            tot
        };
        let (eh, ei) = (out_err(&deq_h), out_err(&deq_i));
        eprintln!(
            "oq2-ldlq output-err: H-OBS={eh:.4} no-fb={ei:.4} ratio={:.3}",
            eh / ei
        );
        assert!(eh < ei, "OBS feedback must beat no-feedback: {eh} !< {ei}");
    }
}

// ─────────────────────────────────────────────────────────────────────────
// QTIP-3 conditioning (qtip3++ candidates)
// ─────────────────────────────────────────────────────────────────────────

/// Which conditioning to apply to the QTIP trellis encode. All three answer the
/// same question — how to spend Hessian information on a beam search — and they
/// sit at very different points on the cost curve, so they are measured rather
/// than argued about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QtipCondMode {
    /// Weight each step's squared error by the ROTATED Hessian diagonal.
    /// Cheapest; expected weakest, because the FWHT exists to flatten exactly
    /// that diagonal.
    Weighted,
    /// Plain beam per group, but feed the winning path's error forward to later
    /// groups through L. Captures cross-group error at almost no encode cost;
    /// no feedback within a group.
    Greedy,
    /// Carry a residual per beam candidate, so error feeds forward WITHIN a
    /// group too, and across groups as in `Greedy`. Faithful LDLQ; pays O(n) per
    /// candidate per step, so the caller must narrow the beam to afford it.
    BeamLdlq,
}

/// Conditioned QTIP encode. `rotated` is the already-FWHT-rotated weight matrix
/// [m, k]; the Hessian is rotated here to match that frame.
///
/// Returns `(symbols, targets)`. `targets` is what the symbols were actually
/// chosen to represent — identical to `rotated` for [`QtipCondMode::Weighted`],
/// but the RESIDUAL-adjusted values for the LDLQ modes. The caller must refit
/// the per-group scale against `targets`, not against `rotated`: under error
/// feedback a block encodes its adjusted target and the compensation for its own
/// error is carried by later blocks. Refitting against the original weights
/// would discard that compensation and quietly degrade to plain RTN-with-extra-
/// steps — the same trap that made an earlier LDLQ run silently do nothing.
#[allow(clippy::too_many_arguments)]
pub fn qtip_conditioned_encode(
    rotated: &[f32],
    m: usize,
    k: usize,
    h_rowmajor_f32: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    damp: f64,
    cb: &[f32],
    beam: usize,
    bits: u32,
    mode: QtipCondMode,
    // Optional external encoder for ONE block of `m` groups, laid out
    // contiguously as [m * 256]. `Greedy` needs only the chosen path, so it can
    // delegate to a stronger encoder (the GPU exact Viterbi) and still feed its
    // error forward — that combination is the point of this parameter, because
    // encoder quality was measured to dominate conditioning at small beams.
    // `BeamLdlq` ignores it: its feedback lives INSIDE the beam, so it cannot
    // delegate without the encoder itself carrying per-candidate residuals.
    block_encoder: Option<&dyn Fn(&[f32], usize) -> Vec<u8>>,
) -> Option<(Vec<u8>, Vec<f32>)> {
    use rayon::prelude::*;
    assert_eq!(rotated.len(), m * k);
    assert_eq!(h_rowmajor_f32.len(), k * k);
    assert_eq!(k % 256, 0, "qtip conditioning requires k % 256 == 0");

    let mut h: Vec<f64> = h_rowmajor_f32.iter().map(|&v| v as f64).collect();
    rotate_hessian(&mut h, k, signs1, signs2);

    if mode == QtipCondMode::Weighted {
        // Normalize the rotated diagonal to mean 1 so the cost stays on the same
        // scale as the unweighted encode (only the RELATIVE weighting matters,
        // and an unnormalized H would just rescale every candidate equally).
        let diag: Vec<f64> = (0..k).map(|i| h[i * k + i].max(0.0)).collect();
        let mean = diag.iter().sum::<f64>() / k as f64;
        let diag: Vec<f64> = if mean > 0.0 {
            diag.iter().map(|d| d / mean).collect()
        } else {
            vec![1.0; k]
        };
        let mut symbols = vec![0u8; m * k];
        symbols
            .par_chunks_mut(256)
            .enumerate()
            .zip(rotated.par_chunks(256))
            .for_each(|((gi, srow), grp)| {
                let c0 = (gi * 256) % k;
                let scale0 = crate::qtip::group_scale(grp);
                let sym = crate::qtip::beam_encode_group_bits_weighted(
                    grp,
                    scale0,
                    cb,
                    beam,
                    bits,
                    &diag[c0..c0 + 256],
                );
                srow.copy_from_slice(&sym);
            });
        return Some((symbols, rotated.to_vec()));
    }

    let l = inv_cholesky_lower_rotated_fast(&h, k, damp)?;
    drop(h);
    let nb = k / 256;
    let mut residual: Vec<f64> = rotated.iter().map(|&w| w as f64).collect();
    let mut symbols = vec![0u8; m * k];
    let mut targets = vec![0.0f32; m * k];

    for blk in 0..nb {
        let c0 = blk * 256;
        let c1 = c0 + 256;
        // The [256, 256] diagonal sub-block of L for this group, row-major.
        let l_block: Vec<f64> = if mode == QtipCondMode::BeamLdlq {
            let mut lb = vec![0.0f64; 256 * 256];
            for f in 0..256 {
                for c in 0..=f {
                    lb[f * 256 + c] = l[(c0 + f, c0 + c)];
                }
            }
            lb
        } else {
            Vec::new()
        };

        // Gather this block from every row into one contiguous [m * 256] buffer so
        // an external encoder sees the same layout it would for a whole tensor.
        let block_flat: Vec<f32> = if block_encoder.is_some() && mode == QtipCondMode::Greedy {
            let mut v = vec![0.0f32; m * 256];
            v.par_chunks_mut(256)
                .zip(residual.par_chunks(k))
                .for_each(|(dst, rr)| {
                    for (d, &sv) in dst.iter_mut().zip(&rr[c0..c1]) {
                        *d = sv as f32;
                    }
                });
            v
        } else {
            Vec::new()
        };
        let ext_syms: Option<Vec<u8>> = if block_flat.is_empty() {
            None
        } else {
            block_encoder.map(|f| f(&block_flat, m))
        };

        let per_row: Vec<(Vec<u8>, Vec<f32>, Vec<f64>)> = residual
            .par_chunks(k)
            .enumerate()
            .map(|(row, rr)| {
                let grp: Vec<f32> = rr[c0..c1].iter().map(|&v| v as f32).collect();
                let scale0 = crate::qtip::group_scale(&grp);
                let sym = match (&ext_syms, mode) {
                    (Some(es), _) => es[row * 256..row * 256 + 256].to_vec(),
                    (None, QtipCondMode::BeamLdlq) => crate::qtip::beam_encode_group_bits_ldlq(
                        &grp, scale0, cb, beam, bits, &l_block,
                    ),
                    (None, _) => crate::qtip::beam_encode_group_bits(&grp, scale0, cb, beam, bits),
                };
                // Refit the scale against what was actually encoded, then measure
                // the error THAT reconstruction leaves behind.
                let scale = crate::qtip::optimal_scale_bits(&grp, &sym, cb, bits);
                let deq = crate::qtip::decode_group_bits(&sym, scale, cb, bits);
                let mut err = vec![0.0f64; 256];
                for c in 0..256 {
                    let lcc = l[(c0 + c, c0 + c)];
                    err[c] = if lcc > 0.0 {
                        (grp[c] as f64 - deq[c] as f64) / lcc
                    } else {
                        0.0
                    };
                }
                (sym, grp, err)
            })
            .collect();

        for (row, (sym, grp, _)) in per_row.iter().enumerate() {
            symbols[row * k + c0..row * k + c1].copy_from_slice(sym);
            targets[row * k + c0..row * k + c1].copy_from_slice(grp);
        }

        if c1 < k {
            residual
                .par_chunks_mut(k)
                .zip(per_row.par_iter())
                .for_each(|(rr, (_, _, err))| {
                    for (c, &ec) in err.iter().enumerate() {
                        if ec == 0.0 {
                            continue;
                        }
                        let col = c0 + c;
                        for f in c1..k {
                            let lfc = l[(f, col)];
                            if lfc != 0.0 {
                                rr[f] -= ec * lfc;
                            }
                        }
                    }
                });
        }
    }
    Some((symbols, targets))
}

// ─────────────────────────────────────────────────────────────────────────
// Right-looking blocked Cholesky, trailing update on GPU
// ─────────────────────────────────────────────────────────────────────────

/// Lower Cholesky `A = L·Lᵀ` by right-looking blocks, with the trailing
/// submatrix update on the GPU.
///
/// MIXED PRECISION BY DESIGN. The panel (diagonal factorization + TRSM) runs on
/// the host in f64: it is O(k·jb²), negligible next to the trailing update, and
/// it is where an ill-conditioned Hessian actually breaks down. The trailing
/// update — the O(k³) — runs in f32 on the GPU, which is the only reason this is
/// worth doing at all: the CPU path measures ~20 GFLOP/s with all 32 threads
/// (verified not a misconfiguration), and Cholesky scales badly at these sizes
/// because the panel serialises.
///
/// Returns `None` on non-positive-definite input, matching faer's `llt`, so the
/// damping-retry loop above behaves identically.
///
/// `a` is `[k, k]` row-major, lower triangle read; the result overwrites the
/// lower triangle. The upper triangle is left untouched (the kernel skips it, and
/// writing it would break the symmetry each panel download assumes).
#[cfg(feature = "gpu")]
pub fn llt_lower_right_looking_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    a: &[f64],
    k: usize,
) -> Option<Vec<f64>> {
    const JB: usize = 128;

    // Device copy in f32. Only the lower triangle is meaningful from here on.
    // `host` is the authoritative copy throughout: panels are read from it and
    // trailing updates are downloaded back into it. Reading panels from a device
    // buffer that was uploaded once and never refreshed gives a factorization of
    // the ORIGINAL matrix — 98% relative error, which is what the first draft did.
    let mut host: Vec<f32> = a.iter().map(|&v| v as f32).collect();

    let mut j0 = 0usize;
    while j0 < k {
        let jb = JB.min(k - j0);
        let rows = k - j0;
        let mut panel = vec![0.0f64; rows * jb];
        for r in 0..rows {
            for c in 0..jb {
                panel[r * jb + c] = host[(j0 + r) * k + j0 + c] as f64;
            }
        }

        // Diagonal block: unblocked Cholesky in f64.
        for c in 0..jb {
            let mut d = panel[c * jb + c];
            for t in 0..c {
                d -= panel[c * jb + t] * panel[c * jb + t];
            }
            if !(d > 0.0) {
                return None; // not positive definite — caller retries with more damping
            }
            let d = d.sqrt();
            panel[c * jb + c] = d;
            for r in (c + 1)..rows {
                let mut v = panel[r * jb + c];
                for t in 0..c {
                    v -= panel[r * jb + t] * panel[c * jb + t];
                }
                panel[r * jb + c] = v / d;
            }
        }

        // Write the factored panel back and let the GPU take the trailing update.
        for r in 0..rows {
            for c in 0..jb.min(r + 1) {
                host[(j0 + r) * k + j0 + c] = panel[r * jb + c] as f32;
            }
        }
        if j0 + jb < k {
            let dev = gpu.upload_owned_f32(&host, &[k, k]).ok()?;
            gpu.chol_syrk_trailing(&dev, k, j0, jb).ok()?;
            gpu.device_synchronize().ok()?;
            host = gpu.download_f32(&dev).ok()?;
        }
        j0 += jb;
    }
    gpu.reclaim_pending();
    Some(host.iter().map(|&v| v as f64).collect())
}
