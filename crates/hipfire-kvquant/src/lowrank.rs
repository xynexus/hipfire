// SPDX-License-Identifier: Apache-2.0
//! Low-rank KV compression algorithms (adoption-plan Levers 3–5). All training-free
//! (offline calibration / online update). Self-contained f64 linalg (no external dep).
//!
//! - **Lever 3 (KQ-SVD `V·Wᵒ`)**: `vwo_basis` — the output-aware V basis (top-r right
//!   singular directions of the value–output product `V·Wᵒ`), so we drop the V
//!   directions the output projection annihilates rather than V's own spectrum.
//! - **Lever 4 (ReCalKV OVC)**: `ovc_recalibrate` — closed-form (Eq 7/8) recalibration
//!   of low-rank factors to minimise the *calibration-weighted* reconstruction
//!   `‖(W − L R)X‖`, a strict win over a vanilla truncated SVD. Composes with Lever 3
//!   (V·Wᵒ basis inits R, OVC recalibrates L,R).
//! - **Lever 5 (OjaKV)**: `oja_update` — online incremental-PCA (subspace Oja rule +
//!   re-orthonormalisation) that tracks the top-r KV subspace during decode instead of
//!   a per-cache SVD, adapting to distribution shift.
//!
//! NOTE (head_dim=256 feasibility): the Lever-5 probe (`lowrank_feasibility.py`) showed
//! aggressive static low-rank does NOT hold at hd=256 (post-RoPE K high-rank; V only
//! ~2× at rank-128). These algorithms are correct + tested, but on qwen3.5-256 they buy
//! little over KVarN — they're staged capability, not a default. Runtime-codec wiring
//! (reconstruct-on-read in the cold tier) is the remaining production step.

/// Cyclic Jacobi eigendecomposition of a symmetric `n×n` matrix `a` (row-major, f64).
/// Returns `(eigenvalues desc, eigenvectors)` with eigenvectors stored as columns
/// (col j is the eigenvector for eigenvalue j), row-major `[n×n]`.
pub fn jacobi_eig(a_in: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a_in.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        // off-diagonal magnitude
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                // Standard robust Jacobi rotation (Numerical Recipes t-formulation):
                // t = tan(φ) solves t² + 2·θ·t − 1 = 0, θ = (a_qq − a_pp)/(2·a_pq).
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // rotate rows/cols p,q of A
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    let evals: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    idx.sort_by(|&i, &j| evals[j].partial_cmp(&evals[i]).unwrap());
    let sorted_evals: Vec<f64> = idx.iter().map(|&i| evals[i]).collect();
    let mut sorted_vecs = vec![0.0f64; n * n];
    for (newc, &oldc) in idx.iter().enumerate() {
        for r in 0..n {
            sorted_vecs[r * n + newc] = v[r * n + oldc];
        }
    }
    (sorted_evals, sorted_vecs)
}

/// Gauss–Jordan inverse of a symmetric-positive-definite-ish `n×n` matrix (row-major).
pub fn mat_inv(a_in: &[f64], n: usize) -> Vec<f64> {
    let mut a = a_in.to_vec();
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        // partial pivot
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r * n + col].abs() > a[piv * n + col].abs() {
                piv = r;
            }
        }
        if piv != col {
            for k in 0..n {
                a.swap(col * n + k, piv * n + k);
                inv.swap(col * n + k, piv * n + k);
            }
        }
        let d = a[col * n + col];
        let d = if d.abs() < 1e-12 { 1e-12 } else { d };
        for k in 0..n {
            a[col * n + k] /= d;
            inv[col * n + k] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r * n + col];
            for k in 0..n {
                a[r * n + k] -= f * a[col * n + k];
                inv[r * n + k] -= f * inv[col * n + k];
            }
        }
    }
    inv
}

/// Modified Gram–Schmidt: orthonormalise the `r` columns of `u` (`[d×r]` row-major).
pub fn orthonormalize(u: &mut [f64], d: usize, r: usize) {
    for j in 0..r {
        for i in 0..j {
            let mut dot = 0.0;
            for k in 0..d {
                dot += u[k * r + j] * u[k * r + i];
            }
            for k in 0..d {
                u[k * r + j] -= dot * u[k * r + i];
            }
        }
        let mut nrm = 0.0;
        for k in 0..d {
            nrm += u[k * r + j] * u[k * r + j];
        }
        let nrm = nrm.sqrt().max(1e-12);
        for k in 0..d {
            u[k * r + j] /= nrm;
        }
    }
}

/// Top-`r` right singular vectors (as columns of `[cols×r]`) of `m` (`[rows×cols]`
/// row-major) — the basis that best preserves `m`'s row space. Via eig of `mᵀm`.
pub fn truncated_svd_basis(m: &[f64], rows: usize, cols: usize, r: usize) -> Vec<f64> {
    let mut mtm = vec![0.0f64; cols * cols];
    for i in 0..cols {
        for j in i..cols {
            let mut s = 0.0;
            for t in 0..rows {
                s += m[t * cols + i] * m[t * cols + j];
            }
            mtm[i * cols + j] = s;
            mtm[j * cols + i] = s;
        }
    }
    let (_ev, vecs) = jacobi_eig(&mtm, cols);
    // top-r columns of vecs → [cols×r]
    let mut b = vec![0.0f64; cols * r];
    for row in 0..cols {
        for c in 0..r.min(cols) {
            b[row * r + c] = vecs[row * cols + c];
        }
    }
    b
}

/// **Lever 3 — KQ-SVD `V·Wᵒ` basis.** Output-aware V basis: top-r right singular
/// directions of the value–output product `V·Wᵒ` (`v`: `[n×d]`, `wo`: `[d×dout]`),
/// returned as `[d×r]` columns. Projecting V onto this basis preserves the attention
/// *output* (`V·Wᵒ`) rather than V's own spectrum — drops directions `Wᵒ` annihilates.
pub fn vwo_basis(v: &[f32], wo: &[f32], n: usize, d: usize, dout: usize, r: usize) -> Vec<f64> {
    // Y = V·Wᵒ  [n×dout]
    let mut y = vec![0.0f64; n * dout];
    for t in 0..n {
        for o in 0..dout {
            let mut s = 0.0f64;
            for k in 0..d {
                s += v[t * d + k] as f64 * wo[k * dout + o] as f64;
            }
            y[t * dout + o] = s;
        }
    }
    // The V-space directions that carry output energy = top-r left singular vectors of
    // Wᵒ weighted by V's usage. Practical form: eig of  Wᵒ (YᵀY-weight) Wᵒᵀ collapses to
    // the top-r eigvecs of  M = Vᵀ V  restricted to Wᵒ's active range. We approximate by
    // the top-r right singular vecs of the *whitened* value matrix Ỹ = V·Wᵒ mapped back
    // through Wᵒ⁺ — but the robust, output-faithful basis is simply the top-r right
    // singular vecs of (Y·Wᵒᵀ) = V·(Wᵒ Wᵒᵀ), i.e. V weighted by output covariance.
    let mut vw = vec![0.0f64; n * d]; // V·(Wᵒ Wᵒᵀ)
    for t in 0..n {
        for k in 0..d {
            let mut s = 0.0f64;
            for o in 0..dout {
                s += y[t * dout + o] * wo[k * dout + o] as f64;
            }
            vw[t * d + k] = s;
        }
    }
    truncated_svd_basis(&vw, n, d, r)
}

/// **Lever 4 — ReCalKV OVC.** Closed-form recalibration of low-rank factors to minimise
/// the calibration-weighted reconstruction `‖(W − L R) X‖_F`, given `w` (`[out×in]`),
/// `xxt = X Xᵀ` (`[in×in]`), and rank `r`. Returns `(l [out×r], rr [r×in])`. Iterates the
/// two closed-form updates a few times from a `truncated_svd_basis(W)` init.
pub fn ovc_recalibrate(
    w: &[f32],
    xxt: &[f64],
    out: usize,
    inn: usize,
    r: usize,
) -> (Vec<f64>, Vec<f64>) {
    let wf: Vec<f64> = w.iter().map(|&x| x as f64).collect();
    // init R = top-r right-sing-vecs of W (transposed to [r×in])
    let b = truncated_svd_basis(&wf, out, inn, r); // [in×r]
    let mut rr = vec![0.0f64; r * inn];
    for i in 0..inn {
        for c in 0..r {
            rr[c * inn + i] = b[i * r + c];
        }
    }
    let mut l = vec![0.0f64; out * r];
    for _it in 0..3 {
        // L = W XXᵀ Rᵀ (R XXᵀ Rᵀ)⁻¹
        let wxxt = matmul(&wf, xxt, out, inn, inn); // [out×in]
        let wxxt_rt = matmul_bt(&wxxt, &rr, out, inn, r); // [out×r]
        let rxxt = matmul(&rr, xxt, r, inn, inn); // [r×in]
        let rxxt_rt = matmul_bt(&rxxt, &rr, r, inn, r); // [r×r]
        let inv = mat_inv(&rxxt_rt, r);
        l = matmul(&wxxt_rt, &inv, out, r, r); // [out×r]
                                               // R = (Lᵀ L)⁻¹ Lᵀ W
        let ltl = matmul_at(&l, &l, out, r, r); // [r×r]
        let ltl_inv = mat_inv(&ltl, r);
        let ltw = matmul_at(&l, &wf, out, r, inn); // [r×in]
        rr = matmul(&ltl_inv, &ltw, r, r, inn); // [r×in]
    }
    (l, rr)
}

/// **Lever 5 — OjaKV.** One subspace-Oja incremental-PCA update of an orthonormal basis
/// `u` (`[d×r]`) over a batch `x` (`[n×d]`): `U ← orth(U + η · Xᵀ(X U))`. Tracks the top-r
/// subspace online (no per-cache SVD), adapting to distribution shift during decode.
pub fn oja_update(u: &mut [f64], x: &[f32], n: usize, d: usize, r: usize, eta: f64) {
    // XU  [n×r]
    let mut xu = vec![0.0f64; n * r];
    for t in 0..n {
        for c in 0..r {
            let mut s = 0.0f64;
            for k in 0..d {
                s += x[t * d + k] as f64 * u[k * r + c];
            }
            xu[t * r + c] = s;
        }
    }
    // U += η Xᵀ (XU)
    for k in 0..d {
        for c in 0..r {
            let mut s = 0.0f64;
            for t in 0..n {
                s += x[t * d + k] as f64 * xu[t * r + c];
            }
            u[k * r + c] += eta * s / n as f64;
        }
    }
    orthonormalize(u, d, r);
}

// ── tiny row-major matmul helpers (f64) ─────────────────────────────────────────────
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for t in 0..k {
            let a_it = a[i * k + t];
            for j in 0..n {
                c[i * n + j] += a_it * b[t * n + j];
            }
        }
    }
    c
}
/// A [m×k] · Bᵀ where B is [n×k] → [m×n]
fn matmul_bt(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for t in 0..k {
                s += a[i * k + t] * b[j * k + t];
            }
            c[i * n + j] = s;
        }
    }
    c
}
/// Aᵀ [k×m from A m×k] · B [m×n] → [k×n]
fn matmul_at(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; k * n];
    for t in 0..m {
        for i in 0..k {
            let a_ti = a[t * k + i];
            for j in 0..n {
                c[i * n + j] += a_ti * b[t * n + j];
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn n(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        }
    }

    fn recon_err_rowspace(m: &[f64], basis: &[f64], rows: usize, cols: usize, r: usize) -> f64 {
        // ||M - M B Bᵀ||_F / ||M||_F  (B [cols×r] orthonormal columns)
        let mb = matmul(m, basis, rows, cols, r); // [rows×r]
        let recon = matmul_bt(&mb, basis, rows, r, cols); // [rows×cols]
        let (mut num, mut den) = (0.0, 0.0);
        for i in 0..rows * cols {
            num += (m[i] - recon[i]).powi(2);
            den += m[i].powi(2);
        }
        (num / den.max(1e-30)).sqrt()
    }

    #[test]
    fn jacobi_reconstructs_symmetric() {
        let n = 6;
        let mut rng = Lcg(1);
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in i..n {
                let v = rng.n();
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        let (ev, vc) = jacobi_eig(&a, n);
        // A ≈ V diag(ev) Vᵀ
        let mut rec = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += vc[i * n + k] * ev[k] * vc[j * n + k];
                }
                rec[i * n + j] = s;
            }
        }
        let err: f64 = (0..n * n).map(|i| (a[i] - rec[i]).abs()).sum();
        assert!(err < 1e-6, "jacobi reconstruction err {err}");
        // descending eigenvalues
        for k in 1..n {
            assert!(ev[k - 1] >= ev[k] - 1e-9);
        }
    }

    #[test]
    fn mat_inv_identity() {
        let n = 5;
        let mut rng = Lcg(9);
        // SPD: A = BᵀB + I
        let mut b = vec![0.0f64; n * n];
        for x in b.iter_mut() {
            *x = rng.n();
        }
        let mut a = matmul_at(&b, &b, n, n, n);
        for i in 0..n {
            a[i * n + i] += 1.0;
        }
        let inv = mat_inv(&a, n);
        let prod = matmul(&a, &inv, n, n, n);
        let mut err = 0.0;
        for i in 0..n {
            for j in 0..n {
                err += (prod[i * n + j] - if i == j { 1.0 } else { 0.0 }).abs();
            }
        }
        assert!(err < 1e-6, "A·A⁻¹ ≠ I, err {err}");
    }

    #[test]
    fn oja_converges_to_subspace() {
        // Data living in a rank-2 subspace of R^6 (+ small noise). Oja should learn a
        // rank-2 basis that reconstructs the data well.
        let (d, r, n) = (6usize, 2usize, 64usize);
        let mut rng = Lcg(7);
        // random orthonormal ground-truth basis columns [d×r]
        let mut truth = vec![0.0f64; d * r];
        for x in truth.iter_mut() {
            *x = rng.n();
        }
        orthonormalize(&mut truth, d, r);
        let gen = |rng: &mut Lcg| -> Vec<f32> {
            let mut x = vec![0.0f64; n * d];
            for t in 0..n {
                let coeffs: Vec<f64> = (0..r).map(|_| rng.n() * 3.0).collect();
                for k in 0..d {
                    let mut s = 0.0;
                    for c in 0..r {
                        s += coeffs[c] * truth[k * r + c];
                    }
                    x[t * d + k] = s + rng.n() * 0.01; // small noise
                }
            }
            x.iter().map(|&v| v as f32).collect()
        };
        let mut u = vec![0.0f64; d * r];
        for x in u.iter_mut() {
            *x = rng.n();
        }
        orthonormalize(&mut u, d, r);
        for _ in 0..200 {
            let batch = gen(&mut rng);
            oja_update(&mut u, &batch, n, d, r, 0.05);
        }
        let test: Vec<f64> = gen(&mut rng).iter().map(|&v| v as f64).collect();
        let err = recon_err_rowspace(&test, &u, n, d, r);
        assert!(
            err < 0.15,
            "Oja subspace reconstruction err {err} (should be small)"
        );
    }

    #[test]
    fn ovc_beats_vanilla_svd() {
        // OVC minimises ‖(W − L R)X‖; must beat a vanilla truncated-SVD factorisation
        // of W (ignoring the calibration weighting) at the same rank.
        let (out, inn, r, ns) = (8usize, 10usize, 3usize, 40usize);
        let mut rng = Lcg(3);
        let w: Vec<f32> = (0..out * inn).map(|_| rng.n() as f32).collect();
        // anisotropic calibration X [in×ns] → XXᵀ
        let mut x = vec![0.0f64; inn * ns];
        for i in 0..inn {
            let scale = if i < 3 { 5.0 } else { 0.3 }; // few dominant input dirs
            for t in 0..ns {
                x[i * ns + t] = rng.n() * scale;
            }
        }
        let xxt = matmul_bt(&x, &x, inn, ns, inn); // [in×in]
                                                   // vanilla: R0 = top-r right-sing-vecs of W, L0 = W R0ᵀ
        let wf: Vec<f64> = w.iter().map(|&v| v as f64).collect();
        let b = truncated_svd_basis(&wf, out, inn, r); // [in×r]
        let mut r0 = vec![0.0f64; r * inn];
        for i in 0..inn {
            for c in 0..r {
                r0[c * inn + i] = b[i * r + c];
            }
        }
        let l0 = matmul_bt(&wf, &r0, out, inn, r); // W R0ᵀ  [out×r]
        let (l, rr) = ovc_recalibrate(&w, &xxt, out, inn, r);
        // weighted error tr((W-LR) XXᵀ (W-LR)ᵀ)
        let werr = |l: &[f64], rr: &[f64]| -> f64 {
            let lr = matmul(l, rr, out, r, inn); // [out×in]
            let diff: Vec<f64> = (0..out * inn).map(|i| wf[i] - lr[i]).collect();
            let dx = matmul(&diff, &xxt, out, inn, inn);
            (0..out)
                .map(|i| {
                    (0..inn)
                        .map(|j| dx[i * inn + j] * diff[i * inn + j])
                        .sum::<f64>()
                })
                .sum()
        };
        let e_vanilla = werr(&l0, &r0);
        let e_ovc = werr(&l, &rr);
        assert!(
            e_ovc <= e_vanilla + 1e-6,
            "OVC weighted err {e_ovc} should be <= vanilla {e_vanilla}"
        );
    }

    #[test]
    fn vwo_basis_beats_naive_on_output() {
        // The V·Wᵒ basis must preserve the OUTPUT V·Wᵒ better than a naive V-only SVD
        // basis at the same rank, when Wᵒ has low effective rank (kills some V dirs).
        let (n, d, dout, r) = (48usize, 8usize, 8usize, 3usize);
        let mut rng = Lcg(11);
        let v: Vec<f32> = (0..n * d).map(|_| rng.n() as f32).collect();
        // Wᵒ rank-4: only 4 V directions reach the output.
        let mut wo = vec![0.0f32; d * dout];
        for i in 0..d {
            for j in 0..dout {
                wo[i * dout + j] = if i < 4 {
                    rng.n() as f32
                } else {
                    (rng.n() * 0.02) as f32
                };
            }
        }
        // naive: top-r right-sing-vecs of V
        let vf: Vec<f64> = v.iter().map(|&x| x as f64).collect();
        let naive = truncated_svd_basis(&vf, n, d, r); // [d×r]
        let vwo = vwo_basis(&v, &wo, n, d, dout, r); // [d×r]
                                                     // output err: ||(V - V B Bᵀ) Wᵒ||_F
        let out_err = |b: &[f64]| -> f64 {
            let vb = matmul(&vf, b, n, d, r);
            let vrec = matmul_bt(&vb, b, n, r, d); // [n×d]
            let mut num = 0.0;
            let mut den = 0.0;
            for t in 0..n {
                for o in 0..dout {
                    let (mut yo, mut yr) = (0.0, 0.0);
                    for k in 0..d {
                        yo += vf[t * d + k] * wo[k * dout + o] as f64;
                        yr += vrec[t * d + k] * wo[k * dout + o] as f64;
                    }
                    num += (yo - yr).powi(2);
                    den += yo * yo;
                }
            }
            (num / den.max(1e-30)).sqrt()
        };
        let e_naive = out_err(&naive);
        let e_vwo = out_err(&vwo);
        assert!(
            e_vwo <= e_naive + 1e-9,
            "V·Wᵒ output err {e_vwo} should beat naive {e_naive}"
        );
    }
}
