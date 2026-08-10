// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! P0 of `docs/plans/2026-08-10-admm-quant-and-qat.md`: does an ADMM loop reach
//! the same quantization objective as the LDLQ column sweep, WITHOUT forming or
//! factorizing the Hessian?
//!
//! This is a controlled optimizer comparison, not a pipeline change. All three
//! methods below minimise the SAME OBS proxy loss
//!
//!     L(Ŵ) = tr( (W − Ŵ) H (W − Ŵ)ᵀ ),   H = XᵀX
//!
//! onto the SAME grid (per-256-group symmetric int4 with a clip-searched scale,
//! `codecs::symmetric_clipsearch` — the exact grid `oq4_ldlq_pack` uses). They
//! are run in one common domain, without the FWHT rotation, so the measurement
//! isolates the OPTIMIZER. The rotation is orthogonal and applies identically to
//! all three, so it cannot change their ranking.
//!
//! What matters is NOT byte-equality with the current packer — ADMM is a
//! different optimizer and will not reproduce LDLQ's output. The question is
//! whether it reaches an equal or lower objective, because that is the only
//! thing the quantizer is trying to minimise.
//!
//! Run: cargo run --release -p hipfire-quantize --example admm_probe

use hipfire_quantize::codecs::symmetric_clipsearch;
use hipfire_quantize::ldlq::inv_cholesky_lower_rotated_fast;

const GROUP: usize = 256;
const QMAX: f32 = 7.0; // symmetric int4: -7..=7

/// Deterministic LCG — no rand dependency, and reproducible across runs.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 33) as f32) / (u32::MAX as f32 / 2.0)) - 1.0
    }
    /// Sum of 3 uniforms ≈ normal; activations are heavier-tailed than uniform
    /// and that is what makes the Hessian anisotropic enough to be a real test.
    fn next_gauss(&mut self) -> f32 {
        (self.next_f32() + self.next_f32() + self.next_f32()) * 0.6
    }
}

/// Quantize one group to the oq4 grid and return the DEQUANTIZED values.
fn quant_group(group: &[f32], out: &mut [f32]) {
    let scale = symmetric_clipsearch(group, QMAX);
    let inv = 1.0 / scale;
    for (i, &g) in group.iter().enumerate() {
        out[i] = (g * inv).round().clamp(-QMAX, QMAX) * scale;
    }
}

/// tr( (W−Ŵ) H (W−Ŵ)ᵀ ) — the objective all three methods minimise.
fn proxy_loss(w: &[f32], wq: &[f32], h: &[f64], m: usize, k: usize) -> f64 {
    let mut total = 0.0;
    for r in 0..m {
        let d: Vec<f64> = (0..k)
            .map(|c| (w[r * k + c] - wq[r * k + c]) as f64)
            .collect();
        for i in 0..k {
            if d[i] == 0.0 {
                continue;
            }
            let mut acc = 0.0;
            for j in 0..k {
                acc += h[i * k + j] * d[j];
            }
            total += d[i] * acc;
        }
    }
    total
}

/// Baseline: quantize each group independently. No error feedback at all.
fn rtn(w: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * k];
    for r in 0..m {
        for b in 0..(k / GROUP) {
            let s = r * k + b * GROUP;
            let (src, dst) = (&w[s..s + GROUP], &mut out[s..s + GROUP]);
            let mut tmp = vec![0.0f32; GROUP];
            quant_group(src, &mut tmp);
            dst.copy_from_slice(&tmp);
        }
    }
    out
}

/// The current algorithm: LDLQ / GPTQ column sweep with exact error feedback.
/// Mirrors `ldlq::oq4_ldlq_pack` minus the rotation and the packing.
fn ldlq_sweep(w: &[f32], h: &[f64], m: usize, k: usize, damp: f64) -> Option<Vec<f32>> {
    let l = inv_cholesky_lower_rotated_fast(h, k, damp)?;
    let mut residual: Vec<f64> = w.iter().map(|&v| v as f64).collect();
    let mut out = vec![0.0f32; m * k];

    for blk in 0..(k / GROUP) {
        let c0 = blk * GROUP;
        for r in 0..m {
            let grp: Vec<f32> = (0..GROUP)
                .map(|c| residual[r * k + c0 + c] as f32)
                .collect();
            let mut deq = vec![0.0f32; GROUP];
            quant_group(&grp, &mut deq);
            for c in 0..GROUP {
                out[r * k + c0 + c] = deq[c];
            }
            // Propagate this column's error into every later column, scaled by
            // the inverse-Cholesky factor. This is the sequential dependency
            // that ADMM is trying to break.
            for c in 0..GROUP {
                let col = c0 + c;
                let ucc = l[(col, col)];
                if ucc <= 0.0 {
                    continue;
                }
                let err = (grp[c] as f64 - deq[c] as f64) / ucc;
                if err == 0.0 {
                    continue;
                }
                for f in (c0 + GROUP)..k {
                    let usf = l[(f, col)];
                    if usf != 0.0 {
                        residual[r * k + f] -= err * usf;
                    }
                }
            }
        }
    }
    Some(out)
}

/// H·v for ONE row vector, computed as the operator it is. The whole point:
/// ADMM never needs `H` factorized, and in the real pipeline would not even
/// need it FORMED — `H·v = Xᵀ(X·v)`.
fn hmul(h: &[f64], v: &[f64], k: usize, out: &mut [f64]) {
    for i in 0..k {
        let mut acc = 0.0;
        let row = &h[i * k..i * k + k];
        for j in 0..k {
            acc += row[j] * v[j];
        }
        out[i] = acc;
    }
}

/// ADMM. Splits into continuous `w` (gradient/solve step), quantized `z`
/// (projection onto the grid), and dual `u` (consensus).
///
///   w ← argmin ‖(w0−w)‖²_H + (ρ/2)‖w − z + u‖²  ⇒  (H + ρI) w = H w0 + ρ(z−u)
///   z ← quantize(w + u)
///   u ← u + w − z
///
/// The w-update is solved by conjugate gradient — matvecs only, fully parallel,
/// no factorization. That is the property that would remove both the K³ Cholesky
/// and the `calib_hessian_outer_f32` capture from the pipeline.
fn admm(
    w: &[f32],
    h: &[f64],
    m: usize,
    k: usize,
    rho: f64,
    iters: usize,
    cg_iters: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * k];
    for r in 0..m {
        let w0: Vec<f64> = (0..k).map(|c| w[r * k + c] as f64).collect();
        // Warm start from RTN: ADMM converges much faster from a sane point,
        // and it guarantees we never do worse than the baseline by accident.
        let mut z = vec![0.0f64; k];
        {
            let mut deq = vec![0.0f32; GROUP];
            for b in 0..(k / GROUP) {
                let grp: Vec<f32> = (0..GROUP).map(|c| w0[b * GROUP + c] as f32).collect();
                quant_group(&grp, &mut deq);
                for c in 0..GROUP {
                    z[b * GROUP + c] = deq[c] as f64;
                }
            }
        }
        let mut u = vec![0.0f64; k];
        let mut x = z.clone();

        let mut hw0 = vec![0.0f64; k];
        hmul(h, &w0, k, &mut hw0);

        for _ in 0..iters {
            // ---- w-update: CG on (H + ρI) x = H w0 + ρ(z − u)
            let rhs: Vec<f64> = (0..k).map(|i| hw0[i] + rho * (z[i] - u[i])).collect();
            let mut ax = vec![0.0f64; k];
            hmul(h, &x, k, &mut ax);
            for i in 0..k {
                ax[i] += rho * x[i];
            }
            let mut rr: Vec<f64> = (0..k).map(|i| rhs[i] - ax[i]).collect();
            let mut p = rr.clone();
            let mut rs: f64 = rr.iter().map(|v| v * v).sum();
            for _ in 0..cg_iters {
                if rs.sqrt() < 1e-10 {
                    break;
                }
                let mut ap = vec![0.0f64; k];
                hmul(h, &p, k, &mut ap);
                for i in 0..k {
                    ap[i] += rho * p[i];
                }
                let denom: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
                if denom.abs() < 1e-30 {
                    break;
                }
                let alpha = rs / denom;
                for i in 0..k {
                    x[i] += alpha * p[i];
                    rr[i] -= alpha * ap[i];
                }
                let rs_new: f64 = rr.iter().map(|v| v * v).sum();
                let beta = rs_new / rs;
                for i in 0..k {
                    p[i] = rr[i] + beta * p[i];
                }
                rs = rs_new;
            }

            // ---- z-update: project (x + u) onto the quant grid
            let mut deq = vec![0.0f32; GROUP];
            for b in 0..(k / GROUP) {
                let grp: Vec<f32> = (0..GROUP)
                    .map(|c| (x[b * GROUP + c] + u[b * GROUP + c]) as f32)
                    .collect();
                quant_group(&grp, &mut deq);
                for c in 0..GROUP {
                    z[b * GROUP + c] = deq[c] as f64;
                }
            }

            // ---- u-update
            for i in 0..k {
                u[i] += x[i] - z[i];
            }
        }
        for c in 0..k {
            out[r * k + c] = z[c] as f32;
        }
    }
    out
}

fn main() {
    let m: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let k: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let n: usize = std::env::args()
        .nth(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    assert_eq!(k % GROUP, 0, "k must be a multiple of {GROUP}");

    let mut rng = Lcg(0x9E3779B97F4A7C15);
    let w: Vec<f32> = (0..m * k).map(|_| rng.next_gauss()).collect();

    // H = XᵀX from real-shaped activations, so it is PSD and anisotropic —
    // a diagonal or identity H would make the sweep and ADMM trivially agree
    // and prove nothing.
    let x: Vec<f32> = (0..n * k)
        .map(|i| {
            let ch = i % k;
            // Per-channel scale spread: a few channels dominate, as in real
            // activations. This is what error feedback exists to exploit.
            let s = 1.0 + 6.0 * ((ch % 17) as f32 / 17.0).powi(3);
            rng.next_gauss() * s
        })
        .collect();
    let mut h = vec![0.0f64; k * k];
    for row in 0..n {
        let xr = &x[row * k..row * k + k];
        for i in 0..k {
            let xi = xr[i] as f64;
            if xi == 0.0 {
                continue;
            }
            for j in 0..k {
                h[i * k + j] += xi * xr[j] as f64;
            }
        }
    }
    let damp = 0.01 * (0..k).map(|i| h[i * k + i]).sum::<f64>() / k as f64;

    println!("shape m={m} k={k} n={n}  damp={damp:.4}\n");

    let t = std::time::Instant::now();
    let q_rtn = rtn(&w, m, k);
    let l_rtn = proxy_loss(&w, &q_rtn, &h, m, k);
    let dt_rtn = t.elapsed();

    let t = std::time::Instant::now();
    let q_ldlq = ldlq_sweep(&w, &h, m, k, damp).expect("cholesky");
    let l_ldlq = proxy_loss(&w, &q_ldlq, &h, m, k);
    let dt_ldlq = t.elapsed();

    println!(
        "{:<26} {:>14}  {:>9}  {}",
        "method", "proxy loss", "vs RTN", "time"
    );
    println!(
        "  {:<24} {:>14.4}  {:>8.1}%  {:?}",
        "RTN (no feedback)", l_rtn, 0.0, dt_rtn
    );
    println!(
        "  {:<24} {:>14.4}  {:>8.1}%  {:?}",
        "LDLQ sweep (current)",
        l_ldlq,
        100.0 * (l_ldlq - l_rtn) / l_rtn,
        dt_ldlq
    );

    // rho is swept relative to the Hessian's OWN scale (damp = 0.01·mean diag,
    // so mean diag = 100·damp). Too small and the w-update ignores consensus and
    // drifts off the grid; too large and it never moves off the warm start.
    let sweep: Vec<(usize, usize, f64)> = vec![
        (15, 16, 1.0),
        (15, 16, 10.0),
        (15, 16, 100.0),
        (15, 16, 300.0),
        (15, 16, 1000.0),
        (40, 16, 100.0),
        (40, 16, 300.0),
    ];
    for &(iters, cg, rho_mul) in &sweep {
        let rho = rho_mul * damp;
        let t = std::time::Instant::now();
        let q = admm(&w, &h, m, k, rho, iters, cg);
        let l = proxy_loss(&w, &q, &h, m, k);
        let dt = t.elapsed();
        let label = format!("ADMM it={iters} cg={cg} ρ={rho_mul}λ");
        println!(
            "  {:<24} {:>14.4}  {:>8.1}%  {:?}{}",
            label,
            l,
            100.0 * (l - l_rtn) / l_rtn,
            dt,
            if l <= l_ldlq { "   <= LDLQ" } else { "" }
        );
    }
    // GUARD: a lower objective is only meaningful if the output is actually
    // storable. Every group must be exactly {code · scale, |code| <= 7} for a
    // single scale, or we are comparing against something the format cannot
    // represent and the win is fictional.
    let check = admm(&w, &h, m, k, 1000.0 * damp, 15, 16);
    let mut bad = 0usize;
    let mut max_code = 0.0f32;
    for r in 0..m {
        for b in 0..(k / GROUP) {
            let g = &check[r * k + b * GROUP..r * k + (b + 1) * GROUP];
            let amax = g.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            if amax == 0.0 {
                continue;
            }
            // Recover the implied scale as the smallest nonzero |value| step.
            let mut step = f32::MAX;
            for &v in g {
                let a = v.abs();
                if a > 1e-12 && a < step {
                    step = a;
                }
            }
            for &v in g {
                let code = v / step;
                let err = (code - code.round()).abs();
                max_code = max_code.max(code.abs());
                if err > 1e-3 || code.abs() > 7.5 {
                    bad += 1;
                }
            }
        }
    }
    println!(
        "\nrepresentability: {} / {} values off-grid, max |code| = {:.2} (must be <= 7)",
        bad,
        m * k,
        max_code
    );
    println!("(lower proxy loss is better; RTN is the no-feedback floor to beat)");
}
