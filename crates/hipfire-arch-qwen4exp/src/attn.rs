// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen Sparse Attention — CPU reference for the attention half.
//!
//! The SELECTION half lives in [`crate::qsa`]; this is what consumes its mask.
//!
//! Two things here are not the usual GQA shape:
//!
//! * `q_proj` emits **twice** `head_dim` per head. The first half is the query;
//!   the second is a per-channel **sigmoid output gate** applied to the attention
//!   result before `o_proj`. Sizing `q_proj` as `n_heads * head_dim` loads a
//!   checkpoint that is half the size it should be.
//! * The gate is laid out per head — `[query(head_dim), gate(head_dim)]` for each
//!   head in turn — not as two contiguous blocks over all heads. Reading it the
//!   other way still type-checks and still produces plausible output.
//!
//! RoPE is applied to q and k with the caller's `cos`/`sin`; deriving those
//! (mrope, with its interleaved sections) is a separate concern.

fn rms_norm_heads(x: &mut [f32], w: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    for h in 0..n_heads {
        let s = &mut x[h * head_dim..(h + 1) * head_dim];
        let inv = 1.0 / (s.iter().map(|v| v * v).sum::<f32>() / head_dim as f32 + eps).sqrt();
        for (i, v) in s.iter_mut().enumerate() {
            *v = *v * inv * (1.0 + w[i]);
        }
    }
}

/// `x * cos + rotate_half(x) * sin` over the first `rotary_dim` channels.
fn apply_rope(x: &mut [f32], cos: &[f32], sin: &[f32], n_heads: usize, head_dim: usize) {
    let rd = cos.len();
    let half = rd / 2;
    for h in 0..n_heads {
        let s = &mut x[h * head_dim..h * head_dim + rd];
        let orig: Vec<f32> = s[..rd].to_vec();
        for i in 0..rd {
            // rotate_half: cat(-x2, x1)
            let rot = if i < half {
                -orig[i + half]
            } else {
                orig[i - half]
            };
            s[i] = orig[i] * cos[i] + rot * sin[i];
        }
    }
}

pub struct QsaAttention<'a> {
    /// `[n_heads * head_dim * 2, hidden]` — query AND gate.
    pub q_proj: &'a [f32],
    /// `[n_kv * head_dim, hidden]`
    pub k_proj: &'a [f32],
    pub v_proj: &'a [f32],
    /// `[hidden, n_heads * head_dim]`
    pub o_proj: &'a [f32],
    /// `[head_dim]`, `1 + w` convention
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub eps: f32,
}

impl QsaAttention<'_> {
    /// Whole-sequence forward.
    ///
    /// `hs` is `[n_tok, hidden]`; `cos`/`sin` are `[n_tok, rotary_dim]`; `visible`
    /// is the `[n_tok, n_tok]` combined causal-and-selected mask, row-major by
    /// query. Returns `[n_tok, hidden]`.
    pub fn forward(
        &self,
        hs: &[f32],
        n_tok: usize,
        cos: &[f32],
        sin: &[f32],
        visible: &[bool],
    ) -> Vec<f32> {
        let (hd, nh, nkv) = (self.head_dim, self.n_heads, self.n_kv);
        assert_eq!(hs.len(), n_tok * self.hidden);
        assert_eq!(visible.len(), n_tok * n_tok);
        let rd = cos.len() / n_tok;
        let mv = |w: &[f32], x: &[f32], o: usize| -> Vec<f32> {
            (0..o)
                .map(|r| {
                    (0..self.hidden)
                        .map(|c| w[r * self.hidden + c] * x[c])
                        .sum()
                })
                .collect()
        };

        // Project, norm, rope — every position up front, since attention needs them all.
        let (mut qs, mut ks, mut vs, mut gates) = (vec![], vec![], vec![], vec![]);
        for t in 0..n_tok {
            let h = &hs[t * self.hidden..(t + 1) * self.hidden];
            let qg = mv(self.q_proj, h, nh * hd * 2);
            // Per head: [query | gate].
            let mut q: Vec<f32> = (0..nh)
                .flat_map(|x| qg[x * 2 * hd..x * 2 * hd + hd].to_vec())
                .collect();
            let g: Vec<f32> = (0..nh)
                .flat_map(|x| qg[x * 2 * hd + hd..(x + 1) * 2 * hd].to_vec())
                .collect();
            let mut k = mv(self.k_proj, h, nkv * hd);
            let v = mv(self.v_proj, h, nkv * hd);
            rms_norm_heads(&mut q, self.q_norm, nh, hd, self.eps);
            rms_norm_heads(&mut k, self.k_norm, nkv, hd, self.eps);
            apply_rope(
                &mut q,
                &cos[t * rd..(t + 1) * rd],
                &sin[t * rd..(t + 1) * rd],
                nh,
                hd,
            );
            apply_rope(
                &mut k,
                &cos[t * rd..(t + 1) * rd],
                &sin[t * rd..(t + 1) * rd],
                nkv,
                hd,
            );
            qs.push(q);
            ks.push(k);
            vs.push(v);
            gates.push(g);
        }

        let scaling = (hd as f32).powf(-0.5);
        let groups = nh / nkv;
        let mut out = vec![0.0f32; n_tok * self.hidden];
        for t in 0..n_tok {
            let mut ctx = vec![0.0f32; nh * hd];
            for h in 0..nh {
                let kv = h / groups;
                let qh = &qs[t][h * hd..(h + 1) * hd];
                let mut score = Vec::with_capacity(t + 1);
                for j in 0..n_tok {
                    if !visible[t * n_tok + j] {
                        continue;
                    }
                    let kh = &ks[j][kv * hd..(kv + 1) * hd];
                    score.push((
                        j,
                        qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scaling,
                    ));
                }
                let max = score.iter().map(|s| s.1).fold(f32::NEG_INFINITY, f32::max);
                let mut den = 0.0f32;
                for (_, s) in score.iter_mut() {
                    *s = (*s - max).exp();
                    den += *s;
                }
                for &(j, p) in &score {
                    let vh = &vs[j][kv * hd..(kv + 1) * hd];
                    let w = p / den;
                    for d in 0..hd {
                        ctx[h * hd + d] += w * vh[d];
                    }
                }
            }
            // Sigmoid output gate, then o_proj.
            for (c, g) in ctx.iter_mut().zip(&gates[t]) {
                *c *= 1.0 / (1.0 + (-g).exp());
            }
            for r in 0..self.hidden {
                out[t * self.hidden + r] = (0..nh * hd)
                    .map(|c| self.o_proj[r * nh * hd + c] * ctx[c])
                    .sum();
            }
        }
        out
    }
}

/// The QSA indexer: which cache positions each query is allowed to attend.
///
/// Scoring is **relu-then-sum over heads**, not sum-then-relu. Both are one line
/// and they differ only once a head scores negative, which is exactly when
/// selection matters; a GPU kernel written the other way passed a plausibility
/// check in this port's history before an exact-value control caught it.
///
/// Two structural rules:
/// * blocks are formed from the VISIBLE positions of that query, so the geometry
///   moves with the causal mask rather than being a fixed grid, and
/// * the ragged tail past the last complete block is ALWAYS selected — it never
///   competes for budget, and it is not representable in a block score at all.
///
/// The pooled block key is rotated at the position of the block's FIRST token.
pub struct Indexer<'a> {
    /// `[(n_heads + kv_heads) * head_dim, hidden]`
    pub qk_proj: &'a [f32],
    /// `[head_dim]`, `1 + w` convention
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub hidden: usize,
    pub n_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub budget: usize,
    pub compress_ratio: usize,
    pub eps: f32,
}

impl Indexer<'_> {
    pub fn block_topk(&self) -> usize {
        self.budget / self.compress_ratio
    }

    /// `[n_tok, n_tok]` selection mask, given the causal mask in the same shape.
    pub fn select_mask(
        &self,
        hs: &[f32],
        n_tok: usize,
        cos: &[f32],
        sin: &[f32],
        causal: &[bool],
    ) -> Vec<bool> {
        let (hd, nh) = (self.head_dim, self.n_heads);
        let rd = cos.len() / n_tok;
        let mv = |w: &[f32], x: &[f32], o: usize| -> Vec<f32> {
            (0..o)
                .map(|r| {
                    (0..self.hidden)
                        .map(|c| w[r * self.hidden + c] * x[c])
                        .sum()
                })
                .collect()
        };

        // Per position: the query heads (normed + rotated) and the raw token key.
        let mut qs = Vec::with_capacity(n_tok);
        let mut raw_keys = Vec::with_capacity(n_tok);
        for t in 0..n_tok {
            let h = &hs[t * self.hidden..(t + 1) * self.hidden];
            let qk = mv(self.qk_proj, h, (nh + self.kv_heads) * hd);
            let mut q = qk[..nh * hd].to_vec();
            rms_norm_heads(&mut q, self.q_norm, nh, hd, self.eps);
            apply_rope(
                &mut q,
                &cos[t * rd..(t + 1) * rd],
                &sin[t * rd..(t + 1) * rd],
                nh,
                hd,
            );
            qs.push(q);
            raw_keys.push(qk[nh * hd..].to_vec()); // un-normed, un-rotated
        }

        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![false; n_tok * n_tok];
        for t in 0..n_tok {
            let visible: Vec<usize> = (0..n_tok).filter(|&j| causal[t * n_tok + j]).collect();
            let n_blocks = visible.len() / self.compress_ratio;

            if n_blocks > 0 {
                let mut scores = Vec::with_capacity(n_blocks);
                for b in 0..n_blocks {
                    let toks = &visible[b * self.compress_ratio..(b + 1) * self.compress_ratio];
                    // Mean-pool the raw keys, then norm, then rotate at the block's start.
                    let mut pooled = vec![0.0f32; hd];
                    for &j in toks {
                        for d in 0..hd {
                            pooled[d] += raw_keys[j][d];
                        }
                    }
                    for v in pooled.iter_mut() {
                        *v /= self.compress_ratio as f32;
                    }
                    rms_norm_heads(&mut pooled, self.k_norm, 1, hd, self.eps);
                    let g = toks[0];
                    apply_rope(
                        &mut pooled,
                        &cos[g * rd..(g + 1) * rd],
                        &sin[g * rd..(g + 1) * rd],
                        1,
                        hd,
                    );

                    let s: f32 = (0..nh)
                        .map(|x| {
                            let qh = &qs[t][x * hd..(x + 1) * hd];
                            qh.iter()
                                .zip(&pooled)
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                                .max(0.0)
                        })
                        .sum::<f32>()
                        * scale;
                    scores.push((b, s));
                }
                // Largest first, ties by lower block index — torch.topk's order.
                let mut order: Vec<usize> = (0..n_blocks).collect();
                order.sort_by(|&a, &b| {
                    scores[b]
                        .1
                        .partial_cmp(&scores[a].1)
                        .unwrap()
                        .then(a.cmp(&b))
                });
                for &b in order.iter().take(self.block_topk().min(n_blocks)) {
                    for &j in &visible[b * self.compress_ratio..(b + 1) * self.compress_ratio] {
                        out[t * n_tok + j] = true;
                    }
                }
            }
            // The ragged tail is unconditional.
            for &j in &visible[n_blocks * self.compress_ratio..] {
                out[t * n_tok + j] = true;
            }
        }
        out
    }
}
