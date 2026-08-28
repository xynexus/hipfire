// SPDX-License-Identifier: Apache-2.0
//! A4 activation sim-quant — the activation side of the `Oq4G256` W4A4 path.
//!
//! `oqplus_quant` bakes the W4 *weight* error; this bakes the A4 *activation*
//! error so a rotation can be scored against the grid it actually deploys onto.
//! Runtime activation quant is online and cheap: per-group symmetric int4
//! (`q ∈ [-7,7]`) with an **absmax** scale (no weight-time clip-search — that is
//! a per-tensor offline luxury the per-token activation path can't afford). No
//! rotation happens here: R1 (residual) / R3 (KV) / R4 (down) are applied to the
//! activation *upstream*; this models only the int4 round-trip the rotated
//! activation then suffers.
//!
//! The point of a rotation is that it Gaussianizes the activation — spreads the
//! few high-kurtosis outlier channels (measured kurtosis > 200 in LLMs) across
//! the group so a single shared int4 scale no longer has to span an outlier and
//! clip everything else. Because an orthonormal `R` preserves per-row norm, the
//! reconstruction SNR of this round-trip *in the rotated basis* equals the
//! end-to-end activation SNR the original computation sees (see
//! [`crate::rotation::rotate_rows`]); so comparing [`snr_db`] across rotations is
//! a faithful, kernel-free measurement of a rotation's A4 quality.

use hipfire_rdna::{Gpu, GpuTensor, HipResult};

/// int4 activation group width (matches `Oq4G256`).
pub const GROUP: usize = 256;

/// Per-group symmetric int4 (absmax) round-trip of a `[rows, feat]` row-major
/// activation buffer. Groups tile the `feat` dim in [`GROUP`]-wide chunks; a
/// trailing partial group uses its own absmax. Returns the dequantized fp32.
pub fn a4_simquant(x: &[f32], rows: usize, feat: usize) -> Vec<f32> {
    simquant_bits(x, rows, feat, 4)
}

/// [`a4_simquant`] widened to any `bits` in `2..=8`, for sweeping the activation
/// tier (A4 vs A8 vs unquantized A16) against one fixed weight tier.
///
/// Same absmax scale and the same `feat`-dim [`GROUP`] tiling — only the code
/// width moves, so a sweep isolates the activation width and nothing else. The
/// deliberate absence of a clip search is documented in the module header and
/// applies at every width: online per-token activation quant cannot afford one.
pub fn simquant_bits(x: &[f32], rows: usize, feat: usize, bits: u32) -> Vec<f32> {
    debug_assert_eq!(x.len(), rows * feat);
    debug_assert!(
        (2..=8).contains(&bits),
        "simquant_bits: bits {bits} out of 2..=8"
    );
    let qmax = ((1i32 << (bits - 1)) - 1) as f32;
    let mut out = vec![0.0f32; rows * feat];
    for r in 0..rows {
        let row = &x[r * feat..r * feat + feat];
        let dst = &mut out[r * feat..r * feat + feat];
        let mut g = 0;
        while g < feat {
            let end = (g + GROUP).min(feat);
            let grp = &row[g..end];
            let amax = grp.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let scale = (amax / qmax).max(1e-12);
            let inv = 1.0 / scale;
            for (i, &v) in grp.iter().enumerate() {
                let q = (v * inv).round().clamp(-qmax, qmax);
                dst[g + i] = q * scale;
            }
            g = end;
        }
    }
    out
}

/// One activation grid: a bulk code width, optionally with the largest
/// magnitudes in each group promoted to int8.
///
/// `outliers = 0` is a uniform per-group absmax grid. `outliers = n` mirrors the
/// weight side's magnitude tier (`top-w8_frac` at int8, bulk at int4 — see
/// `hipfire-quantize::ldlq`) applied to activations instead. The payoff is not
/// the promoted values themselves but what removing them does to the BULK scale:
/// a single outlier inflates a group's absmax and coarsens all 256 codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActTier {
    /// Bulk code width in bits.
    pub bits: u32,
    /// Per-group count promoted to int8 by magnitude; 0 = uniform.
    pub outliers: usize,
}

impl ActTier {
    fn label(self) -> String {
        if self.outliers == 0 {
            format!("A{}", self.bits)
        } else {
            format!("A{}o{}", self.bits, self.outliers)
        }
    }
}

/// Per-group grid with the top-`n_out` magnitudes promoted to int8.
///
/// Both tiers keep the absmax scale — no clip search, for the reason in the
/// module header: online per-token activation quant cannot afford one, so
/// modelling one would score against a grid the runtime never deploys.
/// Promotion by magnitude IS affordable online (a threshold over 256 lanes),
/// which is what makes this worth measuring rather than merely simulating.
pub fn simquant_outlier(x: &[f32], rows: usize, feat: usize, bits: u32, n_out: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), rows * feat);
    if n_out == 0 {
        return simquant_bits(x, rows, feat, bits);
    }
    let qmax = ((1i32 << (bits - 1)) - 1) as f32;
    const OUT_QMAX: f32 = 127.0;
    let mut out = vec![0.0f32; rows * feat];
    let mut idx: Vec<usize> = Vec::with_capacity(GROUP);
    for r in 0..rows {
        let row = &x[r * feat..r * feat + feat];
        let dst = &mut out[r * feat..r * feat + feat];
        let mut g = 0;
        while g < feat {
            let end = (g + GROUP).min(feat);
            let grp = &row[g..end];
            let k = n_out.min(grp.len());
            idx.clear();
            idx.extend(0..grp.len());
            // Largest |v| first; the first k are promoted.
            idx.sort_unstable_by(|&a, &b| {
                grp[b]
                    .abs()
                    .partial_cmp(&grp[a].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut promoted = vec![false; grp.len()];
            let mut out_amax = 0.0f32;
            for &i in &idx[..k] {
                promoted[i] = true;
                out_amax = out_amax.max(grp[i].abs());
            }
            // The bulk scale is the whole point: it now sees the group WITHOUT
            // its outliers, so all remaining codes get finer resolution.
            let bulk_amax = idx[k..].iter().fold(0.0f32, |a, &i| a.max(grp[i].abs()));
            let bulk_scale = (bulk_amax / qmax).max(1e-12);
            let out_scale = (out_amax / OUT_QMAX).max(1e-12);
            for (i, &v) in grp.iter().enumerate() {
                let (scale, cap) = if promoted[i] {
                    (out_scale, OUT_QMAX)
                } else {
                    (bulk_scale, qmax)
                };
                let q = (v / scale).round().clamp(-cap, cap);
                dst[g + i] = q * scale;
            }
            g = end;
        }
    }
    out
}

/// Per-site activation tiers, one per linear-input the block quantizes.
///
/// Mixed widths are deployable, not hypothetical: each field maps to a distinct
/// GEMM, and the W4 kernels reach A8 by carrying the int8 activation as two
/// radix-16 int4 digits (`x = 16·x_hi + x_lo`) through the same iu4 matrix core
/// — see `kernels/src/gemm_oq_compact_iu4x2_wmma.hip`. The weight tile is
/// identical across the two passes, so A8 costs one extra activation plane and
/// WMMA issue, and adds nothing to weight traffic. Choosing A8 for one
/// projection and A4 for another is therefore a real runtime choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActTiers {
    /// q/k/v input (post-norm1 residual).
    pub xn1: Option<ActTier>,
    /// o_proj input (attention context).
    pub ctx: Option<ActTier>,
    /// gate/up input (post-norm2 residual).
    pub xn2: Option<ActTier>,
    /// down_proj input (SwiGLU output) — the widest, and one of the two
    /// high-crest-factor sites.
    pub act: Option<ActTier>,
}

impl ActTiers {
    /// Compact label for reports, e.g. `A4` or `A4(act=A8,ctx=A8)`.
    pub fn label(self) -> String {
        let w = |b: Option<ActTier>| match b {
            Some(t) => t.label(),
            None => "A16".to_string(),
        };
        if self.xn1 == self.ctx && self.xn1 == self.xn2 && self.xn1 == self.act {
            return w(self.xn1);
        }
        format!(
            "xn1={} ctx={} xn2={} act={}",
            w(self.xn1),
            w(self.ctx),
            w(self.xn2),
            w(self.act)
        )
    }

    /// True when nothing is quantized, i.e. the forward is byte-identical.
    pub fn is_noop(self) -> bool {
        self.xn1.is_none() && self.ctx.is_none() && self.xn2.is_none() && self.act.is_none()
    }
}

fn parse_act_tier(s: &str) -> Option<ActTier> {
    if s == "a16" {
        return None;
    }
    // `a<bits>` or `a<bits>o<n_out>`, e.g. `a4`, `a8`, `a4o8`.
    let body = s
        .strip_prefix('a')
        .unwrap_or_else(|| panic!("HIPFIRE_QAT_ACT: unknown tier {s:?} (want a16|a8|a4|a<b>o<n>)"));
    let (bits_s, out_s) = match body.split_once('o') {
        Some((b, o)) => (b, Some(o)),
        None => (body, None),
    };
    let bits: u32 = bits_s
        .parse()
        .unwrap_or_else(|_| panic!("HIPFIRE_QAT_ACT: bad bit width in {s:?}"));
    assert!(
        (2..=8).contains(&bits),
        "HIPFIRE_QAT_ACT: bits {bits} out of 2..=8 in {s:?}"
    );
    let outliers: usize = match out_s {
        Some(o) => o
            .parse()
            .unwrap_or_else(|_| panic!("HIPFIRE_QAT_ACT: bad outlier count in {s:?}")),
        None => 0,
    };
    assert!(
        outliers < GROUP,
        "HIPFIRE_QAT_ACT: outliers {outliers} must be < GROUP {GROUP} in {s:?}"
    );
    Some(ActTier { bits, outliers })
}

/// Activation tiers from `HIPFIRE_QAT_ACT`.
///
/// Grammar: `<base>[,<site>=<tier>]...` where tier is `a16|a8|a4` and site is
/// `xn1|ctx|xn2|act`. `a16` (the default) is a true no-op. Examples:
///
/// ```text
///   a4                    every site int4
///   a4,act=a8,ctx=a8      int4 bulk, int8 on the two high-crest inputs
/// ```
///
/// Deliberately **uncached**, matching [`crate::kv_noise::cfg_from_env`]: the QAT
/// example flips this off for the clean-teacher precompute and back on for the
/// student, which only works if every call re-reads the process env. Do not add
/// a `OnceLock` here — `ops::linear` caches its own precision env that way, and
/// copying that would freeze the gate at the teacher's clean state.
///
/// Panics on an unrecognised tier or site rather than silently running A16: a
/// typo that quietly disables the arm under test is the expensive failure here.
pub fn act_tiers_from_env() -> ActTiers {
    let spec = std::env::var("HIPFIRE_QAT_ACT")
        .unwrap_or_else(|_| "a16".into())
        .to_ascii_lowercase();
    let mut parts = spec.split(',');
    let base = parse_act_tier(parts.next().unwrap_or("a16").trim());
    let mut t = ActTiers {
        xn1: base,
        ctx: base,
        xn2: base,
        act: base,
    };
    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (site, tier) = part
            .split_once('=')
            .unwrap_or_else(|| panic!("HIPFIRE_QAT_ACT: expected <site>=<tier>, got {part:?}"));
        let bits = parse_act_tier(tier.trim());
        match site.trim() {
            "xn1" => t.xn1 = bits,
            "ctx" => t.ctx = bits,
            "xn2" => t.xn2 = bits,
            "act" => t.act = bits,
            other => {
                panic!("HIPFIRE_QAT_ACT: unknown site {other:?} (want xn1|ctx|xn2|act)")
            }
        }
    }
    t
}

/// Fake-quantize one linear's input activation **in place**, forward-only (STE).
/// No-op when `bits` is `None`.
///
/// `x` is `[rows, feat]` row-major with `feat` the projection's input width, so
/// groups tile channels — never tokens. Writing back into the *same* buffer is
/// what makes the STE free and self-consistent: the perturbed values are what
/// land in `BlockActivations`, so `linear_backward_w`'s `dw = dyᵀ·x` uses the
/// same `X` the forward multiplied, while `dx = dy·W` never reads `x` at all.
/// Perturbing a copy and leaving the original saved would differentiate a graph
/// that never ran.
///
/// In-place also avoids the alloc/free churn [`crate::kv_noise`] needs —
/// `GpuTensor` has no `Drop`, and this runs on four tensors per block per step.
pub fn maybe_quant_act(
    gpu: &mut Gpu,
    x: &GpuTensor,
    rows: usize,
    feat: usize,
    tier: Option<ActTier>,
) -> HipResult<()> {
    let Some(tier) = tier else { return Ok(()) };
    let host = gpu.download_f32(x)?;
    let q = simquant_outlier(&host, rows, feat, tier.bits, tier.outliers);
    let bytes = unsafe { std::slice::from_raw_parts(q.as_ptr() as *const u8, q.len() * 4) };
    gpu.memcpy_htod_auto(&x.buf, bytes)
}

/// Reconstruction SNR in dB: `10·log10(‖x‖² / ‖x − x̂‖²)`. Higher is better;
/// `+∞`-ish (capped) when the round-trip is lossless.
pub fn snr_db(orig: &[f32], recon: &[f32]) -> f32 {
    debug_assert_eq!(orig.len(), recon.len());
    let mut sig = 0.0f64;
    let mut noise = 0.0f64;
    for (&o, &r) in orig.iter().zip(recon.iter()) {
        sig += (o as f64) * (o as f64);
        let d = (o - r) as f64;
        noise += d * d;
    }
    if noise <= 0.0 {
        return 200.0;
    }
    (10.0 * (sig / noise).log10()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotation::{rotate_rows, Rotation};

    /// Heavy-tailed activations: a mostly-small Gaussian bulk plus a few large
    /// outlier channels — the regime int4 activation quant chokes on.
    fn heavy_tailed(rows: usize, feat: usize, seed: u64) -> Vec<f32> {
        let mut s = seed ^ 0xABCD_1234;
        let mut nxt = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0 // ~U(-1,1)
        };
        let mut x = vec![0.0f32; rows * feat];
        // A handful of persistent outlier channels shared across rows. The bulk
        // sits at σ≈1 (it carries real output energy, so destroying it must
        // hurt) while the outliers are ~10–14× larger — enough to inflate a
        // shared int4 group scale until the bulk rounds to zero. This is the
        // regime rotation is *for*; if the bulk were negligible (σ≪amax) you'd
        // just keep the outliers and identity would win.
        let outliers: Vec<usize> = (0..4).map(|k| (k * 61 + 7) % feat).collect();
        for r in 0..rows {
            for f in 0..feat {
                let base = nxt(); // ~U(-1,1) bulk
                x[r * feat + f] = if outliers.contains(&f) {
                    base + nxt().signum() * (10.0 + 4.0 * nxt().abs()) // 10–14× outlier
                } else {
                    base
                };
            }
        }
        x
    }

    /// The per-site grammar must actually override the base, and must leave the
    /// other three sites alone. A policy that silently parsed to uniform A4 would
    /// run the wrong arm for half an hour and look plausible doing it.
    #[test]
    fn act_tiers_parse_base_and_overrides() {
        let t = |bits: u32, outliers: usize| Some(ActTier { bits, outliers });
        let cases: &[(&str, ActTiers)] = &[
            (
                "a16",
                ActTiers {
                    xn1: None,
                    ctx: None,
                    xn2: None,
                    act: None,
                },
            ),
            (
                "a4",
                ActTiers {
                    xn1: t(4, 0),
                    ctx: t(4, 0),
                    xn2: t(4, 0),
                    act: t(4, 0),
                },
            ),
            (
                "a4,act=a8,ctx=a8",
                ActTiers {
                    xn1: t(4, 0),
                    ctx: t(8, 0),
                    xn2: t(4, 0),
                    act: t(8, 0),
                },
            ),
            (
                "a8,xn1=a4",
                ActTiers {
                    xn1: t(4, 0),
                    ctx: t(8, 0),
                    xn2: t(8, 0),
                    act: t(8, 0),
                },
            ),
            (
                "a4,act=a16",
                ActTiers {
                    xn1: t(4, 0),
                    ctx: t(4, 0),
                    xn2: t(4, 0),
                    act: None,
                },
            ),
            (
                "a4o8",
                ActTiers {
                    xn1: t(4, 8),
                    ctx: t(4, 8),
                    xn2: t(4, 8),
                    act: t(4, 8),
                },
            ),
            (
                "a4,act=a4o16",
                ActTiers {
                    xn1: t(4, 0),
                    ctx: t(4, 0),
                    xn2: t(4, 0),
                    act: t(4, 16),
                },
            ),
        ];
        for (spec, want) in cases {
            std::env::set_var("HIPFIRE_QAT_ACT", spec);
            assert_eq!(act_tiers_from_env(), *want, "spec {spec:?}");
        }
        std::env::remove_var("HIPFIRE_QAT_ACT");
        assert!(act_tiers_from_env().is_noop(), "unset must be a no-op");

        std::env::set_var("HIPFIRE_QAT_ACT", "a4");
        assert_eq!(act_tiers_from_env().label(), "A4");
        std::env::set_var("HIPFIRE_QAT_ACT", "a4,act=a8");
        assert!(act_tiers_from_env().label().contains("act=A8"));
        std::env::set_var("HIPFIRE_QAT_ACT", "a4o8");
        assert_eq!(act_tiers_from_env().label(), "A4o8");
        std::env::remove_var("HIPFIRE_QAT_ACT");
    }

    /// Promoting a group's largest magnitudes to int8 must beat the uniform grid
    /// at the same bulk width -- that is the entire premise of the magnitude
    /// tier, and it works by shrinking the BULK scale, not by the promoted values
    /// themselves. Heavy-tailed data is the case it is for.
    #[test]
    fn outlier_promotion_beats_uniform_at_same_bulk_width() {
        let (rows, feat) = (8, GROUP * 2);
        let x = heavy_tailed(rows, feat, 0x0271);
        let uniform = snr_db(&x, &simquant_outlier(&x, rows, feat, 4, 0));
        let mut prev = uniform;
        for n_out in [1usize, 4, 8, 16] {
            let snr = snr_db(&x, &simquant_outlier(&x, rows, feat, 4, n_out));
            assert!(
                snr > prev,
                "n_out={n_out} gave {snr} dB, not better than {prev} dB"
            );
            prev = snr;
        }
        // n_out = 0 must be exactly the uniform path, not merely close.
        assert_eq!(
            simquant_outlier(&x, rows, feat, 4, 0),
            simquant_bits(&x, rows, feat, 4)
        );
    }

    /// `simquant_bits` must reproduce `a4_simquant` exactly at bits=4 (the
    /// widening is not allowed to move the deployed A4 grid), and SNR must rise
    /// monotonically with width. A sign or off-by-one in `qmax` breaks one or
    /// the other.
    #[test]
    fn simquant_bits_matches_a4_and_is_monotone_in_width() {
        let (rows, feat) = (4, GROUP + 33); // include a trailing partial group
        let x = heavy_tailed(rows, feat, 0xA4);
        assert_eq!(
            simquant_bits(&x, rows, feat, 4),
            a4_simquant(&x, rows, feat),
            "bits=4 must be bit-identical to the a4 path"
        );
        let mut prev = f32::NEG_INFINITY;
        for bits in [2u32, 3, 4, 6, 8] {
            let snr = snr_db(&x, &simquant_bits(&x, rows, feat, bits));
            assert!(
                snr > prev,
                "SNR must increase with width: {bits} bits gave {snr} dB, not > {prev}"
            );
            prev = snr;
        }
    }

    #[test]
    fn a4_roundtrip_is_lossy_but_bounded() {
        let (rows, feat) = (4, GROUP);
        let x = heavy_tailed(rows, feat, 1);
        let q = a4_simquant(&x, rows, feat);
        let snr = snr_db(&x, &q);
        assert!(snr.is_finite() && snr > 0.0, "snr {snr} not sane");
    }

    /// Plain `y = x Wᵀ`, `x [rows,feat]`, `W [out,feat]`, both row-major.
    fn matmul_t(x: &[f32], w: &[f32], rows: usize, feat: usize, out: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; rows * out];
        for r in 0..rows {
            for o in 0..out {
                let mut acc = 0.0f32;
                for f in 0..feat {
                    acc += x[r * feat + f] * w[o * feat + f];
                }
                y[r * out + o] = acc;
            }
        }
        y
    }

    /// The core SpinQuant claim at the A4 grid, measured *end to end* through a
    /// weight — the faithful metric. Raw-activation reconstruction SNR is a poor
    /// proxy: its Frobenius norm is dominated by the outliers, which quantize
    /// well, so it rewards the identity basis for keeping them while ignoring
    /// that the bulk channels get crushed to zero. Through a (dense) weight, that
    /// crushed bulk propagates into the output and hurts — so a Hadamard rotation,
    /// which disperses the outliers and preserves the bulk, wins on output SNR.
    ///
    /// The rotated model consumes `A4(x Rᵀ)` with weight `W Rᵀ`, i.e.
    /// `A4(xRᵀ)·(WRᵀ)ᵀ`; identity `R` recovers the unrotated `A4(x)·Wᵀ`.
    #[test]
    fn hadamard_beats_identity_end_to_end() {
        let (rows, feat, out) = (8usize, GROUP, 32usize);
        let x = heavy_tailed(rows, feat, 42);
        // Dense random weight (each output mixes all feats, so bulk loss shows).
        let mut s = 0x5151u64;
        let w: Vec<f32> = (0..out * feat)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
            })
            .collect();
        let y_ref = matmul_t(&x, &w, rows, feat, out);
        let out_snr = |rot: &Rotation| {
            let xr = rotate_rows(&x, rot, rows); // x Rᵀ
            let xq = a4_simquant(&xr, rows, feat); // A4(x Rᵀ)
            let wr = rotate_rows(&w, rot, out); // W Rᵀ
            let yq = matmul_t(&xq, &wr, rows, feat, out); // A4(xRᵀ)(WRᵀ)ᵀ
            snr_db(&y_ref, &yq)
        };
        let s_ident = out_snr(&Rotation::identity(feat));
        let s_had = out_snr(&Rotation::hadamard(feat, 7));
        let s_rand = out_snr(&Rotation::random(feat, 7));
        println!(
            "A4 output SNR  identity={s_ident:.2} dB  hadamard={s_had:.2} dB  random={s_rand:.2} dB"
        );
        assert!(
            s_had > s_ident + 3.0,
            "hadamard {s_had:.2} not >3dB over identity {s_ident:.2}"
        );
        assert!(
            s_rand > s_ident + 3.0,
            "random {s_rand:.2} not >3dB over identity {s_ident:.2}"
        );
    }
}
