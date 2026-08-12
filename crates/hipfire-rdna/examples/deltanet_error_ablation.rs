// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Attribute the DeltaNet multi-step error to the specific operation that causes
//! it, now that `parity_gated_delta_net_f64acc{,_routed}` have established an
//! oracle worth measuring against.
//!
//! The f32 kernel has four places precision can be lost per token:
//!
//!   TILE   the LDS tile is `float`, so the state is rounded to f32 after EVERY
//!          token's update — a per-step round-trip on a recurrent accumulator
//!   KV     `kv = <S[r,:], k>`: a 128-term dot product plus a 5-level
//!          `__shfl_down` tree, all f32
//!   UPD    `s = alpha*s + k*delta`
//!   OUT    `out = <S[r,:], q>`: another 128-term dot + tree
//!
//! Each is independently switchable here. Everything else runs in f64, so the
//! error of a configuration is attributable to exactly the terms left in f32.
//!
//! This runs on the CPU on purpose: it needs no kernel variants, no dispatcher
//! plumbing and no kernel cache, and it reproduces the GPU reduction ORDER
//! exactly (the 32-lane tree, 4 values per lane) rather than approximating it
//! with a serial sum. The check that it is faithful is the ALL-F32 row: it has to
//! land near the GPU f32 kernel's measured error against the same reference
//! (2.997e-7 plain / 1.570e-7 routed). If it does not, the model is wrong and
//! none of the attribution below means anything.
//!
//!   cargo run --release -p hipfire-rdna --example deltanet_error_ablation

const HD: usize = 128;
const N_HEADS: usize = 2;

/// L2-normalise each (token, head) vector, matching the model's qk norm.
fn l2_norm_per_head(x: &[f32], n_tokens: usize) -> Vec<f32> {
    let mut out = x.to_vec();
    for t in 0..n_tokens {
        for h in 0..N_HEADS {
            let base = t * N_HEADS * HD + h * HD;
            let n: f32 = out[base..base + HD].iter().map(|v| v * v).sum::<f32>().sqrt();
            if n > 0.0 {
                for v in out[base..base + HD].iter_mut() {
                    *v /= n;
                }
            }
        }
    }
    out
}

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            ((s as f32 + 0.5) / 2_147_483_648.0) * 2.0 - 1.0
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Cfg {
    tile_f32: bool,
    kv_f32: bool,
    upd_f32: bool,
    out_f32: bool,
    /// Compensated (Neumaier) summation for the KV dot + its reduction tree,
    /// still in f32 storage. Models the proposed kernel change: carry a running
    /// correction term so the bits that fall off the bottom of each add are
    /// recovered, rather than discarded.
    kv_kahan: bool,
    /// Everything in f16 — the "fp16/fp16" variant: half-precision STORAGE and
    /// half-precision ARITHMETIC, not just storage. This is the lower bound of
    /// the question "how little precision does the recurrence need", and it is
    /// the configuration nobody has measured.
    all_f16: bool,
    /// Kahan PLUS two-product (Dekker/FMA): `e = fma(a, b, -a*b)` recovers the
    /// bits lost by each MULTIPLY, which compensation of the adds alone cannot
    /// reach. On RDNA an FMA is one instruction, so this is +1 FMA per product.
    kv_dekker: bool,
}

/// f32 two-sum: exact error term of `a + b` when |a| >= |b| is not assumed
/// (Neumaier's variant, which handles either ordering).
/// Round an f64 through IEEE binary16 and back. Rust has no stable f16, so this
/// does the rounding explicitly: 10 explicit mantissa bits, subnormals below
/// ~6.1e-5, and a hard ceiling at 65504.
fn to_f16(x: f64) -> f64 {
    let v = x as f32;
    if !v.is_finite() {
        return v as f64;
    }
    let a = v.abs();
    if a > 65504.0 {
        return if v < 0.0 { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    if a < 6.103_515_6e-5 {
        // subnormal: quantise to the fixed 2^-24 grid
        let step = 5.960_464_5e-8f32;
        return ((v / step).round() * step) as f64;
    }
    let bits = v.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let step = (2.0f32).powi(exp - 10);
    ((v / step).round() * step) as f64
}

fn two_sum_f32(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let c = if a.abs() >= b.abs() {
        (a - s) + b
    } else {
        (b - s) + a
    };
    (s, c)
}

/// Exact residual of an f32 product: `a*b` rounds, and `fma(a, b, -(a*b))`
/// recovers precisely what the rounding discarded. Requires a true fused
/// multiply-add — with contraction disabled this returns 0 and silently degrades
/// to plain Kahan, which is why the table below must show it BEATING Kahan to be
/// believed.
fn two_prod_f32(a: f32, b: f32) -> (f32, f32) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

/// Kahan + two-product: compensates the multiplies as well as the adds, so the
/// f32 dot product becomes near-exact and only the final narrowing remains.
fn dot_tree_dekker(a: &[f64], b: &[f64]) -> f64 {
    let mut sums = [0.0f32; 32];
    let mut corr = [0.0f32; 32];
    for l in 0..32 {
        let c = l * 4;
        let (mut s, mut k) = (0.0f32, 0.0f32);
        for j in 0..4 {
            let (p, pe) = two_prod_f32(a[c + j] as f32, b[c + j] as f32);
            let (ns, nc) = two_sum_f32(s, p);
            s = ns;
            k += nc + pe; // both the add's residual AND the product's
        }
        sums[l] = s;
        corr[l] = k;
    }
    let mut o = 16;
    while o > 0 {
        for l in 0..o {
            let (ns, nc) = two_sum_f32(sums[l], sums[l + o]);
            sums[l] = ns;
            corr[l] += nc + corr[l + o];
        }
        o >>= 1;
    }
    (sums[0] + corr[0]) as f64
}

/// Compensated version of `dot_tree`: every lane keeps (sum, correction) in f32
/// and the tree combines both halves, so the correction survives the reduction
/// instead of being dropped at each level. This is what the kernel change would
/// do — one extra f32 register per lane and a second shuffle per level.
fn dot_tree_kahan(a: &[f64], b: &[f64]) -> f64 {
    let mut sums = [0.0f32; 32];
    let mut corr = [0.0f32; 32];
    for l in 0..32 {
        let c = l * 4;
        let (mut s, mut k) = (0.0f32, 0.0f32);
        for j in 0..4 {
            let p = (a[c + j] as f32) * (b[c + j] as f32);
            let (ns, nc) = two_sum_f32(s, p);
            s = ns;
            k += nc;
        }
        sums[l] = s;
        corr[l] = k;
    }
    let mut o = 16;
    while o > 0 {
        for l in 0..o {
            let (ns, nc) = two_sum_f32(sums[l], sums[l + o]);
            sums[l] = ns;
            corr[l] += nc + corr[l + o];
        }
        o >>= 1;
    }
    (sums[0] + corr[0]) as f64
}

/// The same tree, but every product and every partial sum rounded to f16.
fn dot_tree_f16(a: &[f64], b: &[f64]) -> f64 {
    let mut lanes = [0.0f64; 32];
    for (l, lane) in lanes.iter_mut().enumerate() {
        let c = l * 4;
        let mut acc = 0.0f64;
        for j in 0..4 {
            acc = to_f16(acc + to_f16(to_f16(a[c + j]) * to_f16(b[c + j])));
        }
        *lane = acc;
    }
    let mut o = 16;
    while o > 0 {
        for l in 0..o {
            lanes[l] = to_f16(lanes[l] + lanes[l + o]);
        }
        o >>= 1;
    }
    lanes[0]
}

/// Dot product in the GPU's order: lane `l` holds 4 contiguous values, then a
/// 5-level halving tree across 32 lanes. Reduction order changes the rounding,
/// so a serial sum here would not model the kernel.
fn dot_tree(a: &[f64], b: &[f64], as_f32: bool) -> f64 {
    let mut lanes = [0.0f64; 32];
    for (l, lane) in lanes.iter_mut().enumerate() {
        let c = l * 4;
        let mut acc = if as_f32 {
            let p0 = (a[c] as f32) * (b[c] as f32);
            let p1 = (a[c + 1] as f32) * (b[c + 1] as f32);
            let p2 = (a[c + 2] as f32) * (b[c + 2] as f32);
            let p3 = (a[c + 3] as f32) * (b[c + 3] as f32);
            (((p0 + p1) + p2) + p3) as f64
        } else {
            a[c] * b[c] + a[c + 1] * b[c + 1] + a[c + 2] * b[c + 2] + a[c + 3] * b[c + 3]
        };
        if as_f32 {
            acc = acc as f32 as f64;
        }
        *lane = acc;
    }
    let mut o = 16;
    while o > 0 {
        for l in 0..o {
            lanes[l] += lanes[l + o];
            if as_f32 {
                lanes[l] = lanes[l] as f32 as f64;
            }
        }
        o >>= 1;
    }
    lanes[0]
}

fn run(cfg: Cfg, n_tokens: usize, q: &[f32], k: &[f32], v: &[f32], gate: &[f32], beta: &[f32], s0: &[f32]) -> Vec<f64> {
    let stride = N_HEADS * HD;
    let mut s: Vec<f64> = s0.iter().map(|&x| x as f64).collect();
    for t in 0..n_tokens {
        for h in 0..N_HEADS {
            let alpha = (gate[t * N_HEADS + h] as f64).exp();
            let beta_v = beta[t * N_HEADS + h] as f64;
            let base = t * stride + h * HD;
            let kt: Vec<f64> = (0..HD).map(|c| k[base + c] as f64).collect();
            let qt: Vec<f64> = (0..HD).map(|c| q[base + c] as f64).collect();
            for r in 0..HD {
                let row = h * HD * HD + r * HD;
                let srow: Vec<f64> = s[row..row + HD].to_vec();
                let kv = if cfg.all_f16 {
                    dot_tree_f16(&srow, &kt)
                } else if cfg.kv_dekker {
                    dot_tree_dekker(&srow, &kt)
                } else if cfg.kv_kahan {
                    dot_tree_kahan(&srow, &kt)
                } else {
                    dot_tree(&srow, &kt, cfg.kv_f32)
                };
                let mut delta = (v[base + r] as f64 - alpha * kv) * beta_v;
                if cfg.all_f16 {
                    delta = to_f16(delta);
                } else if cfg.upd_f32 {
                    delta = delta as f32 as f64;
                }
                for c in 0..HD {
                    let mut nv = if cfg.all_f16 {
                        to_f16(to_f16(to_f16(alpha) * to_f16(s[row + c]))
                            + to_f16(to_f16(kt[c]) * to_f16(delta)))
                    } else if cfg.upd_f32 {
                        ((alpha as f32) * (s[row + c] as f32) + (kt[c] as f32) * (delta as f32))
                            as f64
                    } else {
                        alpha * s[row + c] + kt[c] * delta
                    };
                    // The LDS tile is `float`, so the state is re-rounded here
                    // every token — this is the per-step round-trip term.
                    if cfg.all_f16 {
                        nv = to_f16(nv);
                    } else if cfg.tile_f32 {
                        nv = nv as f32 as f64;
                    }
                    s[row + c] = nv;
                }
                let srow2: Vec<f64> = s[row..row + HD].to_vec();
                let _ = dot_tree(&srow2, &qt, cfg.out_f32);
            }
        }
    }
    s
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let mut n = 0.0;
    let mut d = 0.0;
    for (x, y) in a.iter().zip(b) {
        n += (x - y).powi(2);
        d += y * y;
    }
    (n / d.max(1e-300)).sqrt()
}

fn main() {
    let stride = N_HEADS * HD;
    for &n_tokens in &[24usize, 96, 384] {
        let q = l2_norm_per_head(&lcg(1, n_tokens * stride), n_tokens);
        // k and q are L2-NORMALISED per head in the real model — that is what
        // `fused_qk_l2_norm_scale_f32` does, and it appears in every DeltaNet
        // launch trace. Without it ||k||^2 is ~43 for random unit-ish data, the
        // delta rule's stability condition beta*||k||^2 < 2 is violated, and the
        // f64 REFERENCE itself diverges to NaN by 384 tokens. Two earlier
        // versions of this harness chased that NaN through the gate and then
        // beta before the normalisation turned out to be the missing piece.
        let k = l2_norm_per_head(&lcg(2, n_tokens * stride), n_tokens);
        let v = lcg(3, n_tokens * stride);
        // alpha = exp(gate) multiplies the state every token, so gate MUST be
        // <= 0 or the recurrence grows without bound. An earlier version used
        // +/-0.02 and the f64 reference itself went NaN by 384 tokens — a
        // harness artifact that looked like a precision result. Real gates decay.
        let gate: Vec<f32> = lcg(4, n_tokens * N_HEADS)
            .iter()
            .map(|x| -(x.abs() * 0.05) - 0.001)
            .collect();
        // beta is a gate in [0,1] in the real model (a sigmoid output). Signed
        // beta inverts the delta rule into POSITIVE feedback and the recurrence
        // diverges — which is what actually produced the NaNs at 384 tokens,
        // not the gate and not any precision effect.
        let beta: Vec<f32> = lcg(5, n_tokens * N_HEADS)
            .iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        let s0: Vec<f32> = lcg(6, N_HEADS * HD * HD).iter().map(|x| x * 0.1).collect();

        let exact = run(
            Cfg { tile_f32: false, kv_f32: false, upd_f32: false, out_f32: false, kv_kahan: false, kv_dekker: false, all_f16: false },
            n_tokens, &q, &k, &v, &gate, &beta, &s0,
        );

        let cases: [(&str, Cfg); 11] = [
            ("all f32 (models the kernel)", Cfg { tile_f32: true, kv_f32: true, upd_f32: true, out_f32: true, kv_kahan: false, kv_dekker: false, all_f16: false }),
            ("only TILE f32", Cfg { tile_f32: true, kv_f32: false, upd_f32: false, out_f32: false, kv_kahan: false, kv_dekker: false, all_f16: false }),
            ("only KV dot f32", Cfg { tile_f32: false, kv_f32: true, upd_f32: false, out_f32: false, kv_kahan: false, kv_dekker: false, all_f16: false }),
            ("only UPDATE f32", Cfg { tile_f32: false, kv_f32: false, upd_f32: true, out_f32: false, kv_kahan: false, kv_dekker: false, all_f16: false }),
            ("only OUT dot f32", Cfg { tile_f32: false, kv_f32: false, upd_f32: false, out_f32: true, kv_kahan: false, kv_dekker: false, all_f16: false }),
            ("all f32 EXCEPT tile", Cfg { tile_f32: false, kv_f32: true, upd_f32: true, out_f32: true, kv_kahan: false, kv_dekker: false, all_f16: false }),
            ("all f32 + KAHAN kv", Cfg { tile_f32: true, kv_f32: true, upd_f32: true, out_f32: true, kv_kahan: true, kv_dekker: false, all_f16: false }),
            ("only KV f32, KAHAN", Cfg { tile_f32: false, kv_f32: true, upd_f32: false, out_f32: false, kv_kahan: true, kv_dekker: false, all_f16: false }),
            ("all f32 + DEKKER kv", Cfg { tile_f32: true, kv_f32: true, upd_f32: true, out_f32: true, kv_kahan: false, kv_dekker: true, all_f16: false }),
            ("only KV f32, DEKKER", Cfg { tile_f32: false, kv_f32: true, upd_f32: false, out_f32: false, kv_kahan: false, kv_dekker: true, all_f16: false }),
            ("ALL f16 (storage+arith)", Cfg { tile_f32: false, kv_f32: false, upd_f32: false, out_f32: false, kv_kahan: false, kv_dekker: false, all_f16: true }),
        ];
        println!("\n=== {n_tokens} tokens, heads={N_HEADS}, head_dim={HD} ===");
        println!("{:<30} {:>14}", "configuration", "rel L2 err");
        for (name, cfg) in cases {
            let got = run(cfg, n_tokens, &q, &k, &v, &gate, &beta, &s0);
            println!("{name:<30} {:>14.4e}", rel(&got, &exact));
        }
    }
    println!(
        "\nFidelity check: 'all f32' must land near the GPU f32 kernel's measured\n\
         error vs the same style of f64 reference (2.997e-7 at 24 tokens). If it\n\
         does not, this model is wrong and the attribution above is void."
    );
    println!(
        "\nTwo rows are structural, not bugs, and the table must not be read as a\n\
         clean decomposition:\n\
         * 'only OUT dot f32' is EXACTLY 0 because out_v is written to the output\n\
           and never fed back into S. It cannot move the STATE at any token count.\n\
           It does move the logits, which this example does not measure.\n\
         * 'all f32 EXCEPT tile' EQUALS 'all f32' because an f32 UPDATE already\n\
           produces an f32-valued result, so the tile's rounding is then a no-op.\n\
           UPD subsumes TILE; the terms are NOT orthogonal and do not sum. The\n\
           isolated storage cost is the 'only TILE f32' row, where the update runs\n\
           in f64 and only the store rounds."
    );
}
