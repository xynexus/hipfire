// SPDX-License-Identifier: Apache-2.0
//! Static rank-r latent-KV sim, as a forward-only (STE) perturbation on post-RoPE
//! K and V — the retraining-lever probe for the hierarchical calibrated latent-KV
//! line (docs/plans/2026-07-11-latent-kv-large-model-confirmation.md).
//!
//! Every static / equivariant rank-32 basis on the *frozen* model was rejected on
//! 0.8B/4B/9B: the KV low-rank structure lives in arbitrary per-cache directions.
//! The only remaining lever is per-model adaptation — co-train the model (LoRA)
//! to *live in* a fixed rank-r subspace. This module injects that fixed subspace.
//!
//! Per (layer, kv-head) we calibrate a rank-r projector `P = U Uᵀ` from the
//! top-r eigenvectors of the post-RoPE K (and, separately, V) covariance, fit on
//! a calibration pass with projection OFF. During training the student's post-RoPE
//! K and V are replaced by `P·k` / `P_v·v` in place (same shape), which realizes
//! the shared-basis rank-r latent score `qᵀ P k` while keeping the attention op
//! unchanged. It is stored into `BlockActivations`, so `gqa_backward` treats it as
//! a straight-through estimator — gradient flows to LoRA(q/v) + norms as identity.
//!
//! Enabled purely by presence of calibrated projectors in the thread-local store
//! (`set_projectors`); the teacher and the calibration pass run with the store
//! empty, so their forward is byte-identical to baseline.

use hipfire_rdna::{Gpu, GpuTensor, HipResult};
use std::cell::RefCell;

/// Per-layer projectors: `k[kv_head]` / `v[kv_head]` are row-major `[hd*hd]`
/// symmetric idempotent projection matrices onto the calibrated rank-r subspace.
#[derive(Clone, Default)]
pub struct LayerProjectors {
    pub head_dim: usize,
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

thread_local! {
    static STORE: RefCell<Option<Vec<LayerProjectors>>> = const { RefCell::new(None) };
}

/// Install calibrated projectors (enables the latent-KV path for this thread).
pub fn set_projectors(p: Vec<LayerProjectors>) {
    STORE.with(|s| *s.borrow_mut() = Some(p));
}

/// Remove projectors (restores byte-identical baseline forward).
pub fn clear_projectors() {
    STORE.with(|s| *s.borrow_mut() = None);
}

/// True while calibrated projectors are installed.
pub fn active() -> bool {
    STORE.with(|s| s.borrow().is_some())
}

/// Apply `y = P x` for each length-`hd` head sub-vector of a `[seq, n_head*hd]`
/// token-major host buffer, using per-kv-head projector `proj[h]` (`[hd*hd]`).
fn project_host(buf: &mut [f32], seq: usize, dim: usize, hd: usize, proj: &[Vec<f32>]) {
    let n_head = dim / hd;
    let mut y = vec![0.0f32; hd];
    for t in 0..seq {
        for h in 0..n_head {
            let p = &proj[h];
            let off = t * dim + h * hd;
            let x = &buf[off..off + hd];
            for (i, yi) in y.iter_mut().enumerate() {
                let row = &p[i * hd..i * hd + hd];
                let mut acc = 0.0f32;
                for j in 0..hd {
                    acc += row[j] * x[j];
                }
                *yi = acc;
            }
            buf[off..off + hd].copy_from_slice(&y);
        }
    }
}

/// Project post-RoPE K and V onto the calibrated rank-r subspace for `layer_idx`.
/// No-op (identity) when no projectors are installed. `kvd = n_kv * head_dim`.
pub fn maybe_project(
    layer_idx: usize,
    gpu: &mut Gpu,
    k_r: GpuTensor,
    v: GpuTensor,
    seq: usize,
    kvd: usize,
    head_dim: usize,
) -> HipResult<(GpuTensor, GpuTensor)> {
    let proj = STORE.with(|s| s.borrow().as_ref().and_then(|p| p.get(layer_idx).cloned()));
    let Some(proj) = proj else {
        return Ok((k_r, v));
    };
    let mut kh = gpu.download_f32(&k_r)?;
    let mut vh = gpu.download_f32(&v)?;
    project_host(&mut kh, seq, kvd, head_dim, &proj.k);
    project_host(&mut vh, seq, kvd, head_dim, &proj.v);
    let k_new = gpu.upload_f32(&kh, &[seq * kvd])?;
    let v_new = gpu.upload_f32(&vh, &[seq * kvd])?;
    gpu.free_tensor(k_r)?;
    gpu.free_tensor(v)?;
    Ok((k_new, v_new))
}

/// Cyclic Jacobi eigensolver for a small dense symmetric `n×n` matrix (row-major,
/// mutated to diagonal). Returns eigenvalues and eigenvectors (columns), unsorted.
fn jacobi_eig(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut vecs = vec![0.0f64; n * n];
    for i in 0..n {
        vecs[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        // largest off-diagonal magnitude
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off < 1e-24 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-30 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = 0.5 * (aqq - app) / apq;
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
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
                    let vkp = vecs[k * n + p];
                    let vkq = vecs[k * n + q];
                    vecs[k * n + p] = c * vkp - s * vkq;
                    vecs[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let evals: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (evals, vecs)
}

/// Build a rank-`r` projector `P = U Uᵀ` (row-major `[n*n]` f32) from a symmetric
/// covariance `cov` (`[n*n]` f64), using its top-`r` eigenvectors.
fn projector_from_cov(cov: &[f64], n: usize, r: usize) -> Vec<f32> {
    let mut a = cov.to_vec();
    let (evals, vecs) = jacobi_eig(&mut a, n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| evals[j].partial_cmp(&evals[i]).unwrap());
    let top = &order[..r.min(n)];
    // P[i][j] = sum_{k in top} U[i,k] U[j,k], U[i,k] = vecs[i*n + k]
    let mut p = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for &k in top {
                acc += vecs[i * n + k] * vecs[j * n + k];
            }
            p[i * n + j] = acc as f32;
        }
    }
    p
}

/// Accumulate per-kv-head covariance of the length-`hd` head sub-vectors of a
/// `[seq, n_kv*hd]` token-major host buffer into `cov[h]` (`[hd*hd]` f64).
fn accumulate_cov(buf: &[f32], seq: usize, kvd: usize, hd: usize, cov: &mut [Vec<f64>]) {
    let n_kv = kvd / hd;
    for t in 0..seq {
        for h in 0..n_kv {
            let off = t * kvd + h * hd;
            let x = &buf[off..off + hd];
            let c = &mut cov[h];
            for i in 0..hd {
                let xi = x[i] as f64;
                for j in 0..hd {
                    c[i * hd + j] += xi * x[j] as f64;
                }
            }
        }
    }
}

/// Streaming per-(layer, kv-head) covariance accumulator for calibration.
pub struct CovAccum {
    pub n_layers: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    kcov: Vec<Vec<Vec<f64>>>,
    vcov: Vec<Vec<Vec<f64>>>,
}

impl CovAccum {
    pub fn new(n_layers: usize, n_kv: usize, head_dim: usize) -> Self {
        let mk = || {
            (0..n_layers)
                .map(|_| (0..n_kv).map(|_| vec![0.0f64; head_dim * head_dim]).collect())
                .collect::<Vec<Vec<Vec<f64>>>>()
        };
        Self { n_layers, n_kv, head_dim, kcov: mk(), vcov: mk() }
    }

    /// Add one calibration sequence's post-RoPE K/V (downloaded host buffers).
    pub fn add_layer(&mut self, layer: usize, k_host: &[f32], v_host: &[f32], seq: usize) {
        let kvd = self.n_kv * self.head_dim;
        accumulate_cov(k_host, seq, kvd, self.head_dim, &mut self.kcov[layer]);
        accumulate_cov(v_host, seq, kvd, self.head_dim, &mut self.vcov[layer]);
    }

    /// Finalize into rank-`r` projectors per layer.
    pub fn finish(&self, rank: usize) -> Vec<LayerProjectors> {
        (0..self.n_layers)
            .map(|l| LayerProjectors {
                head_dim: self.head_dim,
                k: (0..self.n_kv)
                    .map(|h| projector_from_cov(&self.kcov[l][h], self.head_dim, rank))
                    .collect(),
                v: (0..self.n_kv)
                    .map(|h| projector_from_cov(&self.vcov[l][h], self.head_dim, rank))
                    .collect(),
            })
            .collect()
    }
}
