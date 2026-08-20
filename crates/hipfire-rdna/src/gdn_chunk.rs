// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Chunkwise-parallel gated DeltaNet — CPU reference.
//!
//! The serving recurrence (`kernels/src/gated_delta_net_*.hip`) walks tokens one
//! at a time, and every step depends on the previous one:
//!
//! ```text
//!     S_t = a_t * S_{t-1} * (I - b_t k_t k_t^T) + b_t v_t k_t^T
//!     o_t = S_t q_t
//! ```
//!
//! That is fine for decode, where there is only ever one token. It is what makes
//! BATCHED prefill — and therefore spec-decode verify — cost N serial decodes on
//! a hybrid stack: raising the batch amortizes every other kernel and does
//! nothing here. On Qwen3.8-27B, 48 of 64 layers are this recurrence, which is
//! the whole reason DFlash cannot pay on the family (see
//! `docs/experiments/2026-08-20-dflash2-qwen38-27b-performance.md`).
//!
//! This module is the chunkwise form: one chunk of `C` tokens becomes a few
//! matmuls plus one `C x C` triangular solve, with no dependence between tokens
//! inside the chunk. The FLOP count is the same order — the win is that the
//! `d^2` work turns into matmuls over the whole chunk instead of `C` dependent
//! rank-1 updates each gated on the last.
//!
//! # Derivation
//!
//! Write `P_t = I - b_t k_t k_t^T`, so `S_t = a_t S_{t-1} P_t + b_t v_t k_t^T`.
//! The gate `a_t` is a SCALAR, so it commutes with everything; substituting
//! `S_t = c_t S'_t` with `c_t = prod_{r<=t} a_r` cancels it entirely and leaves
//! the ungated delta rule
//!
//! ```text
//!     S'_t = S'_{t-1} + u_t k_t^T,     u_t = (b_t/c_t) v_t - b_t S'_{t-1} k_t
//! ```
//!
//! so `S'_t = S'_0 + sum_{s<=t} u_s k_s^T`. Substituting that back into `u_t`'s
//! own definition gives a closed triangular system in the `u`s:
//!
//! ```text
//!     u_t + b_t sum_{s<t} (k_s . k_t) u_s = (b_t/c_t) v_t - b_t S_0 k_t
//! ```
//!
//! Written directly, that system carries `1/c_t`, which grows as the chunk
//! decays and is the obvious way to make this blow up. Rescaling once more with
//! `U_t = c_t u_t` replaces every bare `1/c` with a RATIO `c_t/c_s = exp(L_t -
//! L_s)` for `t >= s`, where `L` is the cumulative log-gate. Every such factor is
//! `<= 1`, so the whole chunk is computed in decays and never in growths:
//!
//! ```text
//!     A[t,s] = b_t exp(L_t - L_s) (k_s . k_t)      for s < t, else 0
//!     B[t]   = b_t (v_t - exp(L_t) S_0 k_t)
//!     (I + A) U = B                                 (forward substitution)
//!     S_C    = exp(L_C) S_0 + sum_t exp(L_C - L_t) U_t k_t^T
//!     o_t    = exp(L_t) S_0 q_t + sum_{s<=t} exp(L_t - L_s) (k_s . q_t) U_s
//! ```
//!
//! At `C = 1` this collapses to exactly the kernel body: `U_1 = b(v - a S_0 k)`
//! is its `delta`, `S_1 = a S_0 + U_1 k^T` its state update, and `o_1 = a S_0 q +
//! (k.q) U_1` its output. `chunk_matches_sequential` checks that identity holds
//! for every chunk size, not just one.
//!
//! Layout matches the kernels: `q`/`k`/`v` are `[tokens, n_heads, head_dim]`,
//! `gate`/`beta` are `[tokens, n_heads]`, and state is `[n_heads, head_dim,
//! head_dim]` indexed `h*hd*hd + r*hd + c` — `r` the value/output dim, `c` the
//! key/query dim.

/// `HIPFIRE_GDN_CHUNK=1` — resolve a batched DeltaNet chunk with the chunkwise
/// kernel instead of the token-at-a-time recurrence.
///
/// Default OFF. The chunk form is the same recurrence but sums in a different
/// order, so it is NOT bit-identical to the serial kernel — and the serial
/// kernel is what plain decode runs. On a spec-decode verify that means
/// committed tokens come off slightly different logits than an AR baseline
/// would produce, which can flip a token at a decision boundary. Opt in,
/// measure the divergence, then decide.
pub fn chunk_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("HIPFIRE_GDN_CHUNK").ok().as_deref(),
            Some("1" | "true" | "on" | "yes")
        )
    })
}

/// Geometry of one gated-DeltaNet call.
#[derive(Clone, Copy, Debug)]
pub struct GdnDims {
    pub n_tokens: usize,
    pub n_heads: usize,
    pub head_dim: usize,
}

/// The serving recurrence, token by token. This is the definition the chunked
/// form has to reproduce; it is not used in production, only to check it.
pub fn gdn_sequential_f64(
    d: GdnDims,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &mut [f64],
) -> Vec<f64> {
    let (hd, nh) = (d.head_dim, d.n_heads);
    let stride = nh * hd;
    let mut out = vec![0.0f64; d.n_tokens * stride];
    for t in 0..d.n_tokens {
        for h in 0..nh {
            let alpha = (gate[t * nh + h] as f64).exp();
            let b = beta[t * nh + h] as f64;
            let base = t * stride + h * hd;
            for r in 0..hd {
                let row = h * hd * hd + r * hd;
                let kv: f64 = (0..hd).map(|c| state[row + c] * k[base + c] as f64).sum();
                let delta = (v[base + r] as f64 - alpha * kv) * b;
                let mut o = 0.0f64;
                for c in 0..hd {
                    let s = alpha * state[row + c] + k[base + c] as f64 * delta;
                    state[row + c] = s;
                    o += s * q[base + c] as f64;
                }
                out[base + r] = o;
            }
        }
    }
    out
}

/// The same thing, chunkwise. `chunk` tokens are resolved together; the state is
/// carried across chunks, so `chunk = 1` degenerates to the sequential form and
/// `chunk >= n_tokens` does the whole call in one shot.
pub fn gdn_chunked_f64(
    d: GdnDims,
    chunk: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &mut [f64],
) -> Vec<f64> {
    assert!(chunk > 0, "chunk must be positive");
    let (hd, nh) = (d.head_dim, d.n_heads);
    let stride = nh * hd;
    let mut out = vec![0.0f64; d.n_tokens * stride];

    let mut t0 = 0usize;
    while t0 < d.n_tokens {
        let c = chunk.min(d.n_tokens - t0);
        for h in 0..nh {
            let s0 = h * hd * hd;
            let at = |t: usize| (t0 + t) * stride + h * hd;

            // Cumulative log-gate inside the chunk. Every exp() below is of a
            // DIFFERENCE L_t - L_s with t >= s, i.e. a decay in [0, 1].
            let mut lcum = vec![0.0f64; c];
            let mut acc = 0.0f64;
            for (t, l) in lcum.iter_mut().enumerate() {
                acc += gate[(t0 + t) * nh + h] as f64;
                *l = acc;
            }
            let bt = |t: usize| beta[(t0 + t) * nh + h] as f64;

            // B[t] = b_t (v_t - exp(L_t) S_0 k_t), then solved in place into U.
            let mut u = vec![0.0f64; c * hd];
            for t in 0..c {
                let (base, e_t) = (at(t), lcum[t].exp());
                for r in 0..hd {
                    let row = s0 + r * hd;
                    let sk: f64 = (0..hd)
                        .map(|cc| state[row + cc] * k[base + cc] as f64)
                        .sum();
                    u[t * hd + r] = bt(t) * (v[base + r] as f64 - e_t * sk);
                }
            }
            // Forward substitution for (I + A) U = B. A is strictly lower, so
            // row t only ever reads rows already resolved.
            for t in 1..c {
                let base_t = at(t);
                for s in 0..t {
                    let base_s = at(s);
                    let kk: f64 = (0..hd)
                        .map(|cc| k[base_s + cc] as f64 * k[base_t + cc] as f64)
                        .sum();
                    let a = bt(t) * (lcum[t] - lcum[s]).exp() * kk;
                    if a == 0.0 {
                        continue;
                    }
                    for r in 0..hd {
                        u[t * hd + r] -= a * u[s * hd + r];
                    }
                }
            }

            // Outputs read the ENTRY state plus the chunk-local contributions of
            // every token at or before them.
            for t in 0..c {
                let (base, e_t) = (at(t), lcum[t].exp());
                for r in 0..hd {
                    let row = s0 + r * hd;
                    let sq: f64 = (0..hd)
                        .map(|cc| state[row + cc] * q[base + cc] as f64)
                        .sum();
                    out[base + r] = e_t * sq;
                }
                for s in 0..=t {
                    let base_s = at(s);
                    let kq: f64 = (0..hd)
                        .map(|cc| k[base_s + cc] as f64 * q[base + cc] as f64)
                        .sum();
                    let w = (lcum[t] - lcum[s]).exp() * kq;
                    for r in 0..hd {
                        out[base + r] += w * u[s * hd + r];
                    }
                }
            }

            // State advances once for the whole chunk.
            let e_c = lcum[c - 1].exp();
            for r in 0..hd {
                let row = s0 + r * hd;
                for cc in 0..hd {
                    state[row + cc] *= e_c;
                }
            }
            for s in 0..c {
                let (base_s, w) = (at(s), (lcum[c - 1] - lcum[s]).exp());
                for r in 0..hd {
                    let row = s0 + r * hd;
                    let ur = w * u[s * hd + r];
                    for cc in 0..hd {
                        state[row + cc] += ur * k[base_s + cc] as f64;
                    }
                }
            }
        }
        t0 += c;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: u32, n: usize) -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
                ((s as f32 + 0.5) / 2_147_483_648.0) * 2.0 - 1.0
            })
            .collect()
    }

    /// `gate` is a LOG decay, so it must be <= 0 — a positive gate would make the
    /// state grow without bound and is not a state the model produces.
    fn fixture(
        d: GdnDims,
        seed: u32,
        decay: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f64>) {
        let n = d.n_tokens * d.n_heads * d.head_dim;
        let g: Vec<f32> = lcg(seed + 3, d.n_tokens * d.n_heads)
            .iter()
            .map(|x| -(x.abs()) * decay)
            .collect();
        let b: Vec<f32> = lcg(seed + 4, d.n_tokens * d.n_heads)
            .iter()
            .map(|x| x.abs())
            .collect();
        let s0 = lcg(seed + 5, d.n_heads * d.head_dim * d.head_dim)
            .iter()
            .map(|&x| x as f64 * 0.05)
            .collect();
        (lcg(seed, n), lcg(seed + 1, n), lcg(seed + 2, n), g, b, s0)
    }

    fn max_abs(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn chunk_matches_sequential() {
        // head_dim 32 rather than the kernel's 128 keeps the O(C*hd^2) reference
        // quick; the derivation has no head_dim-dependent step.
        let d = GdnDims {
            n_tokens: 24,
            n_heads: 2,
            head_dim: 32,
        };
        for &decay in &[0.05f32, 0.5, 2.0] {
            let (q, k, v, g, b, s0) = fixture(d, 11, decay);
            let mut s_seq = s0.clone();
            let want = gdn_sequential_f64(d, &q, &k, &v, &g, &b, &mut s_seq);
            // Chunk sizes that do and do not divide n_tokens, so the ragged tail
            // is covered too.
            for &c in &[1usize, 2, 5, 8, 16, 24, 64] {
                let mut s_ch = s0.clone();
                let got = gdn_chunked_f64(d, c, &q, &k, &v, &g, &b, &mut s_ch);
                let (do_, ds) = (max_abs(&got, &want), max_abs(&s_ch, &s_seq));
                assert!(
                    do_ < 1e-9 && ds < 1e-9,
                    "decay={decay} chunk={c}: output {do_:.3e}, state {ds:.3e}"
                );
            }
        }
    }

    /// The point of the `U_t = c_t u_t` rescaling: a hard-decaying chunk is where
    /// the naive `1/c_t` form loses its digits. Nothing here may grow with C.
    #[test]
    fn hard_decay_stays_conditioned() {
        let d = GdnDims {
            n_tokens: 32,
            n_heads: 1,
            head_dim: 32,
        };
        let (q, k, v, g, b, s0) = fixture(d, 7, 6.0);
        let mut s_seq = s0.clone();
        let want = gdn_sequential_f64(d, &q, &k, &v, &g, &b, &mut s_seq);
        let mut s_ch = s0.clone();
        let got = gdn_chunked_f64(d, 32, &q, &k, &v, &g, &b, &mut s_ch);
        assert!(
            max_abs(&got, &want) < 1e-9 && max_abs(&s_ch, &s_seq) < 1e-9,
            "hard decay: output {:.3e}, state {:.3e}",
            max_abs(&got, &want),
            max_abs(&s_ch, &s_seq)
        );
    }
}
