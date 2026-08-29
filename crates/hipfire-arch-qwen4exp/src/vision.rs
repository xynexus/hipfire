// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! The vision tower — CPU reference.
//!
//! A conventional pre-norm ViT, which makes it the least surprising part of this
//! model — but three details are easy to get wrong:
//!
//! * It uses **LayerNorm with bias**, not the RMSNorm the text trunk uses
//!   everywhere. Every projection here carries a bias too.
//! * There are **two different GELUs**. The block MLP uses `hidden_act`, which the
//!   shipped config sets to `gelu_pytorch_tanh` (the tanh approximation); the
//!   merger uses `nn.GELU()`, which is the EXACT erf form. They differ by ~1e-3 at
//!   moderate inputs — small enough to look like noise, large enough to fail a
//!   tight comparison.
//! * `qkv` is one fused projection laid out `[token][q|k|v][head][dim]`, so a
//!   head's q, k and v are NOT adjacent.

/// `gelu_pytorch_tanh` — the tanh approximation used by the block MLPs.
pub fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_56; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

/// Exact erf GELU — what `nn.GELU()` computes, used by the merger.
pub fn gelu_erf(x: f32) -> f32 {
    // erf via Abramowitz & Stegun 7.1.26; max error ~1.5e-7, well under the
    // tolerance any comparison here uses.
    let z = x / std::f32::consts::SQRT_2;
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let a = z.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_0) * t) + 1.421_413_7) * t - 0.284_496_74) * t
            + 0.254_829_59)
            * t
            * (-a * a).exp();
    0.5 * x * (1.0 + sign * y)
}

/// LayerNorm with weight and bias, over the last axis.
pub fn layer_norm(x: &[f32], w: &[f32], b: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row, chunk) in x.chunks_exact(dim).enumerate() {
        let mean = chunk.iter().sum::<f32>() / dim as f32;
        let var = chunk.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..dim {
            out[row * dim + i] = (chunk[i] - mean) * inv * w[i] + b[i];
        }
    }
    out
}

/// `y = W x + b` for each row of `x`, with `W` in `[out, in]`.
pub fn linear(x: &[f32], w: &[f32], b: Option<&[f32]>, n_in: usize, n_out: usize) -> Vec<f32> {
    let rows = x.len() / n_in;
    let mut out = vec![0.0f32; rows * n_out];
    for r in 0..rows {
        for o in 0..n_out {
            let mut acc = b.map_or(0.0, |b| b[o]);
            for i in 0..n_in {
                acc += w[o * n_in + i] * x[r * n_in + i];
            }
            out[r * n_out + o] = acc;
        }
    }
    out
}

/// Patch embedding.
///
/// The reference uses a `Conv3d` whose stride equals its kernel, so every patch is
/// independent and it is exactly a per-patch linear map from
/// `in_channels * temporal_patch_size * patch_size^2` to `hidden`.
pub fn patch_embed(
    patches: &[f32],
    w: &[f32],
    b: &[f32],
    per_patch: usize,
    hidden: usize,
) -> Vec<f32> {
    linear(patches, w, Some(b), per_patch, hidden)
}

pub struct VisionBlock<'a> {
    pub norm1_w: &'a [f32],
    pub norm1_b: &'a [f32],
    pub norm2_w: &'a [f32],
    pub norm2_b: &'a [f32],
    /// `[3 * hidden, hidden]` fused, laid out per token as `[q|k|v][head][dim]`.
    pub qkv_w: &'a [f32],
    pub qkv_b: &'a [f32],
    pub proj_w: &'a [f32],
    pub proj_b: &'a [f32],
    pub fc1_w: &'a [f32],
    pub fc1_b: &'a [f32],
    pub fc2_w: &'a [f32],
    pub fc2_b: &'a [f32],
    pub hidden: usize,
    pub n_heads: usize,
    pub intermediate: usize,
    pub eps: f32,
}

impl VisionBlock<'_> {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.n_heads
    }

    /// Attention over `n_tok` patches. Bidirectional — there is no causal mask.
    ///
    /// `cos`/`sin` are `[n_tok, head_dim]` and shared by every head.
    pub fn attention(&self, x: &[f32], n_tok: usize, cos: &[f32], sin: &[f32]) -> Vec<f32> {
        let (h, nh, hd) = (self.hidden, self.n_heads, self.head_dim());
        let qkv = linear(x, self.qkv_w, Some(self.qkv_b), h, 3 * h);
        let half = hd / 2;

        // Pull out per-token, per-head q/k/v and rotate q and k.
        let mut q = vec![0.0f32; n_tok * h];
        let mut k = vec![0.0f32; n_tok * h];
        let mut v = vec![0.0f32; n_tok * h];
        for t in 0..n_tok {
            for head in 0..nh {
                for d in 0..hd {
                    let base = t * 3 * h + head * hd + d;
                    q[t * h + head * hd + d] = qkv[base];
                    k[t * h + head * hd + d] = qkv[base + h];
                    v[t * h + head * hd + d] = qkv[base + 2 * h];
                }
            }
            for head in 0..nh {
                for arr in [&mut q, &mut k] {
                    let s = &mut arr[t * h + head * hd..t * h + (head + 1) * hd];
                    let orig: Vec<f32> = s.to_vec();
                    for d in 0..hd {
                        // rotate_half: cat(-x2, x1)
                        let rot = if d < half {
                            -orig[d + half]
                        } else {
                            orig[d - half]
                        };
                        s[d] = orig[d] * cos[t * hd + d] + rot * sin[t * hd + d];
                    }
                }
            }
        }

        let scale = 1.0 / (hd as f32).sqrt();
        let mut ctx = vec![0.0f32; n_tok * h];
        for t in 0..n_tok {
            for head in 0..nh {
                let qh = &q[t * h + head * hd..t * h + (head + 1) * hd];
                let mut score: Vec<f32> = (0..n_tok)
                    .map(|j| {
                        let kh = &k[j * h + head * hd..j * h + (head + 1) * hd];
                        qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let max = score.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut den = 0.0f32;
                for s in score.iter_mut() {
                    *s = (*s - max).exp();
                    den += *s;
                }
                for (j, p) in score.iter().enumerate() {
                    let vh = &v[j * h + head * hd..j * h + (head + 1) * hd];
                    let wgt = p / den;
                    for d in 0..hd {
                        ctx[t * h + head * hd + d] += wgt * vh[d];
                    }
                }
            }
        }
        linear(&ctx, self.proj_w, Some(self.proj_b), h, h)
    }

    pub fn mlp(&self, x: &[f32]) -> Vec<f32> {
        let mut t = linear(
            x,
            self.fc1_w,
            Some(self.fc1_b),
            self.hidden,
            self.intermediate,
        );
        for v in t.iter_mut() {
            *v = gelu_tanh(*v);
        }
        linear(
            &t,
            self.fc2_w,
            Some(self.fc2_b),
            self.intermediate,
            self.hidden,
        )
    }

    /// Pre-norm residual block: `x + attn(norm1(x))`, then `x + mlp(norm2(x))`.
    pub fn forward(&self, x: &[f32], n_tok: usize, cos: &[f32], sin: &[f32]) -> Vec<f32> {
        let n1 = layer_norm(x, self.norm1_w, self.norm1_b, self.hidden, self.eps);
        let a = self.attention(&n1, n_tok, cos, sin);
        let mut h: Vec<f32> = x.iter().zip(&a).map(|(p, q)| p + q).collect();
        let n2 = layer_norm(&h, self.norm2_w, self.norm2_b, self.hidden, self.eps);
        let m = self.mlp(&n2);
        for (p, q) in h.iter_mut().zip(&m) {
            *p += q;
        }
        h
    }
}

/// Patch merger: normalise per patch, then fold `spatial_merge_size^2` patches into
/// one token and project to the text hidden size.
///
/// The norm runs on the UNMERGED width (`use_postshuffle_norm` is false in the
/// shipped model), so it must happen before the reshape, not after.
#[allow(clippy::too_many_arguments)]
pub fn merger(
    x: &[f32],
    norm_w: &[f32],
    norm_b: &[f32],
    fc1_w: &[f32],
    fc1_b: &[f32],
    fc2_w: &[f32],
    fc2_b: &[f32],
    hidden: usize,
    merge_unit: usize,
    out_hidden: usize,
    eps: f32,
) -> Vec<f32> {
    let normed = layer_norm(x, norm_w, norm_b, hidden, eps);
    let wide = hidden * merge_unit;
    let mut t = linear(&normed, fc1_w, Some(fc1_b), wide, wide);
    for v in t.iter_mut() {
        *v = gelu_erf(*v);
    }
    linear(&t, fc2_w, Some(fc2_b), wide, out_hidden)
}

/// Per-patch `(row, col)` positions, in the order the tower consumes patches.
///
/// ⚠️ That order is **merge-block major**, not raster. With `merge = 2` on a 4x4
/// grid the sequence starts `(0,0) (0,1) (1,0) (1,1) (0,2) ...` — the four patches
/// that will be folded into one token are adjacent. Emitting plain raster order
/// gives the right SET of positions and the wrong assignment, which shows up as a
/// subtly scrambled image rather than an error.
pub fn position_ids(t: usize, h: usize, w: usize, merge: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(t * h * w);
    for _ in 0..t {
        for bh in 0..h / merge {
            for bw in 0..w / merge {
                for dh in 0..merge {
                    for dw in 0..merge {
                        out.push((bh * merge + dh, bw * merge + dw));
                    }
                }
            }
        }
    }
    out
}

/// Bilinear resample of the learned square position grid onto an image's grid.
///
/// Returns, per patch, the four corner indices into the `[num_grid^2, hidden]`
/// table and their weights. `align_corners = true`, so the mapping is
/// `src = pos * (num_grid - 1) / (dim - 1)` and the endpoints land exactly on the
/// first and last row/column.
pub fn pos_embed_interpolation(
    positions: &[(usize, usize)],
    h: usize,
    w: usize,
    num_grid: usize,
) -> Vec<([usize; 4], [f32; 4])> {
    // A single row or column has no interval to interpolate across; the reference's
    // align_corners mapping degenerates to 0 rather than dividing by zero.
    let map = |p: usize, dim: usize| -> (usize, f32) {
        if dim <= 1 {
            return (0, 0.0);
        }
        let src = p as f32 * (num_grid - 1) as f32 / (dim - 1) as f32;
        let lo = (src.floor() as usize).min(num_grid - 1);
        (lo, src - lo as f32)
    };
    positions
        .iter()
        .map(|&(ph, pw)| {
            let (r, fr) = map(ph, h);
            let (c, fc) = map(pw, w);
            let r1 = (r + 1).min(num_grid - 1);
            let c1 = (c + 1).min(num_grid - 1);
            (
                [
                    r * num_grid + c,
                    r * num_grid + c1,
                    r1 * num_grid + c,
                    r1 * num_grid + c1,
                ],
                [
                    (1.0 - fr) * (1.0 - fc),
                    (1.0 - fr) * fc,
                    fr * (1.0 - fc),
                    fr * fc,
                ],
            )
        })
        .collect()
}

/// The interpolated position embedding added after the patch embedding.
pub fn pos_embeds(table: &[f32], interp: &[([usize; 4], [f32; 4])], hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; interp.len() * hidden];
    for (t, (idx, wgt)) in interp.iter().enumerate() {
        for (i, &row) in idx.iter().enumerate() {
            let src = &table[row * hidden..(row + 1) * hidden];
            for d in 0..hidden {
                out[t * hidden + d] += wgt[i] * src[d];
            }
        }
    }
    out
}

/// Vision rotary frequencies: `[h * inv_freq | w * inv_freq]` per patch.
///
/// `dim` is `head_dim / 2` and `inv_freq` has `dim / 2` entries, so the result is
/// `dim` wide — the two axes concatenated, height first. The tower then
/// concatenates THAT with itself to reach `head_dim` before taking cos/sin, which
/// is what makes the `rotate_half` form correct.
pub fn rotary_frequencies(positions: &[(usize, usize)], dim: usize, theta: f32) -> Vec<f32> {
    let half = dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / dim as f32))
        .collect();
    let mut out = Vec::with_capacity(positions.len() * dim);
    for &(h, w) in positions {
        out.extend(inv.iter().map(|f| h as f32 * f));
        out.extend(inv.iter().map(|f| w as f32 * f));
    }
    out
}

/// `cos`/`sin` at `head_dim` width, from the half-width frequencies.
pub fn rotary_cos_sin(freqs: &[f32], n_tok: usize) -> (Vec<f32>, Vec<f32>) {
    let half = freqs.len() / n_tok;
    let hd = half * 2;
    let mut cos = vec![0.0f32; n_tok * hd];
    let mut sin = vec![0.0f32; n_tok * hd];
    for t in 0..n_tok {
        for i in 0..hd {
            let f = freqs[t * half + i % half];
            cos[t * hd + i] = f.cos();
            sin[t * hd + i] = f.sin();
        }
    }
    (cos, sin)
}

/// Splice merged vision tokens into the text embedding stream.
///
/// Each occurrence of `image_token_id` in the prompt is a PLACEHOLDER whose
/// embedding is replaced, in order, by the next merged vision token. The reference
/// does this as a `masked_scatter`.
///
/// Two things this does NOT do, both deliberate:
///
/// * It does not touch the token ids. The PLE n-gram block hashes the ORIGINAL
///   ids, placeholders included — the image content never reaches the n-gram
///   window.
/// * It does not resize anything. The vision tower's `out_hidden` must already
///   equal the text `hidden`; that is checked when the config is parsed, because
///   here it would only surface as a length mismatch.
///
/// Errors when the placeholder count and the vision-token count disagree, which is
/// the common prompt-construction mistake and silently corrupts the prompt if the
/// shorter of the two is just truncated.
pub fn splice_image_embeds(
    embeds: &mut [f32],
    token_ids: &[u32],
    image_embeds: &[f32],
    image_token_id: u32,
    hidden: usize,
) -> Result<usize, String> {
    let slots = token_ids.iter().filter(|&&t| t == image_token_id).count();
    let have = image_embeds.len() / hidden;
    if slots != have {
        return Err(format!(
            "{slots} image placeholders in the prompt but {have} vision tokens \
             (each {hidden} wide)"
        ));
    }
    let mut next = 0;
    for (t, &tok) in token_ids.iter().enumerate() {
        if tok == image_token_id {
            embeds[t * hidden..(t + 1) * hidden]
                .copy_from_slice(&image_embeds[next * hidden..(next + 1) * hidden]);
            next += 1;
        }
    }
    Ok(slots)
}
