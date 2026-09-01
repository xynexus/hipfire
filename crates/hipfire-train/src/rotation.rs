// SPDX-License-Identifier: Apache-2.0
//! SpinQuant R1: rotation-invariant residual-stream transform (Phase 0).
//!
//! The full-precision pre-norm transformer is *rotation-invariant*: inserting an
//! orthonormal `R` on the residual stream and its inverse at matched points
//! leaves the fp output unchanged, but rotates the basis the quantizer sees
//! (SpinQuant, arXiv:2405.16406). This module bakes such an `R1` into the fp32
//! weights of [`crate::model::LlamaModel`] so a later quantize sees the rotated
//! (better-conditioned) grid. Phase 0 only proves the invariance contract; the
//! learned optimizer (Cayley SGD) lands in a later phase.
//!
//! The transform has two moves, applied together by [`apply_r1`]:
//!
//! 1. **Fold each RMSNorm scale `α` into the following weight.** LLaMA RMSNorm is
//!    `y = (x/rms(x)) ⊙ α`; the elementwise `α` is what breaks rotation
//!    invariance (`α ⊙ (xRᵀ) ≠ (α ⊙ x) Rᵀ`). Folding `α` into the *columns* of
//!    every weight that reads the norm output (SliceGPT-style) leaves a
//!    scale-free RMSNorm `y = x/rms(x)`, which **is** rotation-equivariant
//!    (`rms(xRᵀ)=rms(x)` for orthonormal `R`).
//! 2. **Rotate residual readers and writers by `R`.** A *reader* (q/k/v/gate/up,
//!    embedding-as-input, lm_head) consumes the residual: `W → W Rᵀ` on its
//!    input (`h`) dimension. A *writer* (o_proj, down_proj) adds into the
//!    residual: `W → R W` on its output (`h`) dimension. Every block shares the
//!    one global `R` (the residual basis is model-wide).
//!
//! Result: every intermediate (q,k,v,ctx,gate,up,act) is bit-for-bit unchanged
//! and the residual stream is carried in the rotated basis `x Rᵀ`, so the logits
//! match the original up to fp reassociation.
//!
//! Tied embeddings: the input embedding needs `E Rᵀ` (rotate rows) while the
//! output head needs `α_f` folded first — incompatible in one shared matrix. So
//! [`apply_r1`] **unties** the head (materializes `lm_head` from `embed`) before
//! folding/rotating. The forward path already supports an untied `lm_head`; the
//! tied-only backward is a Phase 2 concern (learning R needs the untied head
//! grad wired anyway).

use crate::model::LlamaModel;
use hipfire_primitives::fwht::{gen_fwht_signs, signed_fwht};
use hipfire_rdna::{Gpu, GpuTensor, HipResult};

/// An orthonormal `[h,h]` rotation, row-major (`r[i*h + j]`). Invariant: `R Rᵀ = I`.
#[derive(Clone)]
pub struct Rotation {
    pub h: usize,
    pub r: Vec<f32>,
}

impl Rotation {
    /// Identity rotation (`apply_r1` with this is a pure norm-scale fold — a
    /// useful control: fold alone must already be bit-exact-equivalent).
    pub fn identity(h: usize) -> Self {
        let mut r = vec![0.0f32; h * h];
        for i in 0..h {
            r[i * h + i] = 1.0;
        }
        Self { h, r }
    }

    /// A random orthonormal matrix: Gram–Schmidt on a deterministic Gaussian
    /// (Box–Muller over a splitmix64 stream). Offline `O(h³)`; fine for a bake.
    pub fn random(h: usize, seed: u64) -> Self {
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next_u64 = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut normal = || {
            // Box–Muller; two uniforms in (0,1].
            let u1 = ((next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
            let u2 = ((next_u64() >> 11) as f64) / (1u64 << 53) as f64;
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        };
        let mut r: Vec<f32> = (0..h * h).map(|_| normal()).collect();
        gram_schmidt_rows(&mut r, h);
        Self { h, r }
    }

    /// A random-sign normalized Hadamard (`h` must be a power of two): the
    /// QuaRot / SpinQuant *fixed* rotation (+0.9 dB tier). Sylvester construction
    /// scaled by `1/√h`, with each column flipped by a deterministic ±1 sign so
    /// the rotation is data-agnostic but not the bare Hadamard. Panics if `h` is
    /// not a power of two (the residual `h` is; sub-block Hadamards for odd dims
    /// are a later concern).
    pub fn hadamard(h: usize, seed: u64) -> Self {
        assert!(h.is_power_of_two(), "hadamard size {h} not a power of two");
        let scale = 1.0 / (h as f32).sqrt();
        let mut r = vec![0.0f32; h * h];
        for i in 0..h {
            for j in 0..h {
                // Sylvester entry sign = (-1)^popcount(i & j).
                let parity = (i & j).count_ones() & 1;
                r[i * h + j] = if parity == 0 { scale } else { -scale };
            }
        }
        // Random column signs (still orthonormal: diag(±1) is orthonormal).
        let mut state = seed ^ 0xD1B5_4A32_D192_ED03;
        for j in 0..h {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if (state >> 63) & 1 == 1 {
                for i in 0..h {
                    r[i * h + j] = -r[i * h + j];
                }
            }
        }
        Self { h, r }
    }

    /// The Oq4G256 codec's per-256-group FWHT as a dense block-diagonal `[h,h]`
    /// rotation `F` (signs 42/1042 — the production seeds). `rotate_rows(x, F)`
    /// reproduces the codec's `fwht_rows`, so `F` is the rotation the deployed
    /// int4 pipeline already applies. `h % 256 == 0` required.
    pub fn block_fwht(h: usize) -> Self {
        Self::block_fwht_with(h, 256, 42, 1042)
    }

    /// The general form: a block-diagonal `[h,h]` rotation whose blocks are a
    /// `group`-wide signed FWHT with the given sign seeds.
    ///
    /// [`block_fwht`] is this with the Oq4G256 codec's parameters (256, 42/1042).
    /// The width and the seeds both have to match the codec you are baking for,
    /// because the bake identity is `F·(Fᵀ M) = M` and a mismatched `F` on either
    /// side leaves a residual rotation rather than cancelling — which is silent,
    /// since the result is still an orthonormal matrix.
    ///
    /// The other live width is 128 with seeds 43/1043, which is what
    /// `paro_la_gates_codec`'s MQ4G128 encoder and the `gemv_hfq4g128` kernel use
    /// (uploaded by `ensure_mq_signs_128`). That is the width that makes a
    /// per-head R2 deployable — see [`bake_for_per_head_codec`].
    pub fn block_fwht_with(h: usize, group: usize, seed1: u32, seed2: u32) -> Self {
        assert!(
            group.is_power_of_two(),
            "block_fwht group {group} is not a power of two"
        );
        assert_eq!(
            h % group,
            0,
            "block_fwht size {h} not a multiple of {group}"
        );
        let (s1, s2) = (gen_fwht_signs(seed1, group), gen_fwht_signs(seed2, group));
        // Column i of the block = signed_fwht(e_i) (so rotate_rows == fwht).
        let mut block = vec![0.0f32; group * group];
        for i in 0..group {
            let mut e = vec![0.0f32; group];
            e[i] = 1.0;
            signed_fwht(&mut e, &s1, &s2);
            for (j, &v) in e.iter().enumerate() {
                block[j * group + i] = v;
            }
        }
        let mut r = vec![0.0f32; h * h];
        for b in 0..(h / group) {
            let off = b * group;
            for j in 0..group {
                for i in 0..group {
                    r[(off + j) * h + (off + i)] = block[j * group + i];
                }
            }
        }
        Self { h, r }
    }

    /// Transpose (= inverse, since orthonormal).
    pub fn transpose(&self) -> Self {
        let h = self.h;
        let mut r = vec![0.0f32; h * h];
        for i in 0..h {
            for j in 0..h {
                r[j * h + i] = self.r[i * h + j];
            }
        }
        Self { h, r }
    }

    /// Matrix product `self · rhs` (both `[h,h]`); orthonormal × orthonormal ⇒
    /// orthonormal.
    pub fn compose(&self, rhs: &Rotation) -> Self {
        assert_eq!(self.h, rhs.h);
        let h = self.h;
        let mut r = vec![0.0f32; h * h];
        for i in 0..h {
            for k in 0..h {
                let a = self.r[i * h + k];
                if a == 0.0 {
                    continue;
                }
                for j in 0..h {
                    r[i * h + j] += a * rhs.r[k * h + j];
                }
            }
        }
        Self { h, r }
    }

    /// Re-orthonormalize the rows in place (modified Gram–Schmidt). Cheap guard
    /// against drift when an iterative process (e.g. the approximate Cayley
    /// inverse in [`crate::learn_rotation`]) accumulates small departures from
    /// `R Rᵀ = I` over many steps.
    pub fn reorthonormalize(&mut self) {
        gram_schmidt_rows(&mut self.r, self.h);
    }

    /// `max |R Rᵀ − I|` — the orthonormality residual (a correctness probe).
    pub fn orthonormality_error(&self) -> f32 {
        let h = self.h;
        let mut worst = 0.0f32;
        for i in 0..h {
            for j in 0..h {
                let mut dot = 0.0f32;
                for k in 0..h {
                    dot += self.r[i * h + k] * self.r[j * h + k];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((dot - target).abs());
            }
        }
        worst
    }
}

/// In-place modified Gram–Schmidt orthonormalization of the `h` rows of a row-
/// major `[h,h]` matrix.
fn gram_schmidt_rows(m: &mut [f32], h: usize) {
    for i in 0..h {
        // Subtract projections onto previously-orthonormalized rows.
        for j in 0..i {
            let mut dot = 0.0f32;
            for k in 0..h {
                dot += m[i * h + k] * m[j * h + k];
            }
            for k in 0..h {
                m[i * h + k] -= dot * m[j * h + k];
            }
        }
        let mut norm = 0.0f32;
        for k in 0..h {
            norm += m[i * h + k] * m[i * h + k];
        }
        let inv = 1.0 / norm.sqrt().max(1e-20);
        for k in 0..h {
            m[i * h + k] *= inv;
        }
    }
}

/// Multiply column `i` of a row-major `[out, cols]` weight by `alpha[i]` (folds
/// an RMSNorm scale into the following reader weight).
fn fold_cols(w: &mut [f32], alpha: &[f32], out: usize, cols: usize) {
    debug_assert_eq!(alpha.len(), cols);
    for o in 0..out {
        let row = &mut w[o * cols..o * cols + cols];
        for (v, &a) in row.iter_mut().zip(alpha.iter()) {
            *v *= a;
        }
    }
}

/// Rotate the `rows` of a row-major `[rows, h]` activation buffer by `Rᵀ` (each
/// row `x → x Rᵀ`) — the residual stream as the R1-transformed model carries it.
/// Orthonormal `R` preserves per-row norm, so this is a pure basis change: the
/// SNR of a later A4 round-trip in this basis equals the end-to-end activation
/// SNR in the original basis (the SpinQuant measurement). Same op as the reader
/// weight rotate, exposed for the A4-SNR probe.
pub fn rotate_rows(x: &[f32], rot: &Rotation, rows: usize) -> Vec<f32> {
    rotate_input(x, rot, rows)
}

/// Reader rotate: `W → W Rᵀ` on the input (`h`) dimension of a `[out, h]` weight.
/// `new[o,j] = Σ_i W[o,i]·R[j,i]`.
fn rotate_input(w: &[f32], rot: &Rotation, out: usize) -> Vec<f32> {
    let h = rot.h;
    let mut o_out = vec![0.0f32; out * h];
    for o in 0..out {
        let src = &w[o * h..o * h + h];
        let dst = &mut o_out[o * h..o * h + h];
        for (j, d) in dst.iter_mut().enumerate() {
            let rrow = &rot.r[j * h..j * h + h];
            let mut acc = 0.0f32;
            for (s, rr) in src.iter().zip(rrow.iter()) {
                acc += s * rr;
            }
            *d = acc;
        }
    }
    o_out
}

/// R2 writer: rotate each head's `head_dim` **output rows** by `R2` — for
/// `v_proj [n_heads·hd, cols]`, `new[head,a,:] = Σ_b R2[a,b]·W[head,b,:]`.
fn rotate_head_rows(w: &[f32], r2: &Rotation, n_heads: usize, cols: usize) -> Vec<f32> {
    let hd = r2.h;
    let mut out = vec![0.0f32; w.len()];
    for head in 0..n_heads {
        let base = head * hd;
        for a in 0..hd {
            let dst = &mut out[(base + a) * cols..(base + a) * cols + cols];
            for b in 0..hd {
                let rab = r2.r[a * hd + b];
                if rab == 0.0 {
                    continue;
                }
                let src = &w[(base + b) * cols..(base + b) * cols + cols];
                for (o, s) in dst.iter_mut().zip(src.iter()) {
                    *o += rab * s;
                }
            }
        }
    }
    out
}

/// R2 reader: rotate each head's `head_dim` **input columns** by `R2ᵀ` — for
/// `o_proj [rows, n_heads·hd]`, `new[d,head,a] = Σ_b W[d,head,b]·R2[a,b]`.
fn rotate_head_cols(w: &[f32], r2: &Rotation, n_heads: usize, rows: usize) -> Vec<f32> {
    let hd = r2.h;
    let qd = n_heads * hd;
    let mut out = vec![0.0f32; w.len()];
    for d in 0..rows {
        let row = &w[d * qd..d * qd + qd];
        let orow = &mut out[d * qd..d * qd + qd];
        for head in 0..n_heads {
            let base = head * hd;
            for a in 0..hd {
                let rr = &r2.r[a * hd..a * hd + hd];
                let mut acc = 0.0f32;
                for (b, &rab) in rr.iter().enumerate() {
                    acc += row[base + b] * rab;
                }
                orow[base + a] = acc;
            }
        }
    }
    out
}

/// Writer rotate: `W → R W` on the output (`h`) dimension of a `[h, cols]`
/// weight. `new[e,c] = Σ_d R[e,d]·W[d,c]`.
fn rotate_output(w: &[f32], rot: &Rotation, cols: usize) -> Vec<f32> {
    let h = rot.h;
    let mut o_out = vec![0.0f32; h * cols];
    for e in 0..h {
        let rrow = &rot.r[e * h..e * h + h];
        let dst = &mut o_out[e * cols..e * cols + cols];
        for (d, &rval) in rrow.iter().enumerate() {
            if rval == 0.0 {
                continue;
            }
            let src = &w[d * cols..d * cols + cols];
            for (o, s) in dst.iter_mut().zip(src.iter()) {
                *o += rval * s;
            }
        }
    }
    o_out
}

/// Download → transform → re-upload, replacing `slot` and freeing the old device
/// buffer (GpuTensor has no Drop).
fn replace_tensor<F>(gpu: &mut Gpu, slot: &mut GpuTensor, f: F) -> HipResult<()>
where
    F: FnOnce(Vec<f32>) -> Vec<f32>,
{
    let host = gpu.download_f32(slot)?;
    let shape = slot.shape.clone();
    let new_host = f(host);
    debug_assert_eq!(new_host.len(), shape.iter().product::<usize>());
    let newt = gpu.upload_f32(&new_host, &shape)?;
    let old = std::mem::replace(slot, newt);
    gpu.free_tensor(old)?;
    Ok(())
}

/// Set a norm weight to all-ones (its scale has been folded into the readers).
fn set_ones(gpu: &mut Gpu, slot: &mut GpuTensor) -> HipResult<()> {
    replace_tensor(gpu, slot, |v| vec![1.0f32; v.len()])
}

/// Bake SpinQuant `R1` into `model` in place: fold every RMSNorm scale into its
/// readers, then rotate residual readers/writers/embedding/head by `R`. The fp32
/// forward is left invariant (up to fp reassociation); the residual stream is now
/// carried in the `x Rᵀ` basis. Unties the head if tied (see module docs).
///
/// `R.h` must equal the model hidden size.
pub fn apply_r1(gpu: &mut Gpu, model: &mut LlamaModel, rot: &Rotation) -> HipResult<()> {
    let h = model.dims.h;
    assert_eq!(rot.h, h, "rotation size {} != hidden {}", rot.h, h);
    let qd = model.dims.q_dim();
    let kvd = model.dims.kv_dim();
    let inter = model.dims.inter;
    let vocab = model.vocab;

    // ── Head: untie, then fold final_norm α_f into lm_head columns. ───────────
    // The input embedding and the (folded) output head diverge here, so we must
    // untie before touching either.
    if model.lm_head.is_none() {
        let embed_host = gpu.download_f32(&model.embed)?;
        let lmh = gpu.upload_f32(&embed_host, &[vocab * h])?;
        model.lm_head = Some(lmh);
    }
    let alpha_f = gpu.download_f32(&model.final_norm)?;
    {
        let lmh = model.lm_head.as_mut().expect("untied above");
        replace_tensor(gpu, lmh, |mut w| {
            fold_cols(&mut w, &alpha_f, vocab, h);
            w
        })?;
    }
    set_ones(gpu, &mut model.final_norm)?;

    // ── Per-layer: fold norms, rotate readers (input) and writers (output). ───
    for (w, _lora) in model.layers.iter_mut() {
        let a1 = gpu.download_f32(&w.norm1)?;
        for (proj, out) in [(&mut w.wq, qd), (&mut w.wk, kvd), (&mut w.wv, kvd)] {
            replace_tensor(gpu, proj, |mut m| {
                fold_cols(&mut m, &a1, out, h);
                rotate_input(&m, rot, out)
            })?;
        }
        set_ones(gpu, &mut w.norm1)?;
        replace_tensor(gpu, &mut w.wo, |m| rotate_output(&m, rot, qd))?;

        let a2 = gpu.download_f32(&w.norm2)?;
        for proj in [&mut w.wgate, &mut w.wup] {
            replace_tensor(gpu, proj, |mut m| {
                fold_cols(&mut m, &a2, inter, h);
                rotate_input(&m, rot, inter)
            })?;
        }
        set_ones(gpu, &mut w.norm2)?;
        replace_tensor(gpu, &mut w.wdown, |m| rotate_output(&m, rot, inter))?;
    }

    // ── Embedding (input writer) and head (output reader): rotate on `h`. ─────
    // Both rotate on their `h` columns (`E → E Rᵀ`, `lm_head → lm_head Rᵀ`).
    replace_tensor(gpu, &mut model.embed, |m| rotate_input(&m, rot, vocab))?;
    {
        let lmh = model.lm_head.as_mut().expect("untied above");
        replace_tensor(gpu, lmh, |m| rotate_input(&m, rot, vocab))?;
    }

    Ok(())
}

/// Bake SpinQuant `R2` (head-wise `[head_dim, head_dim]`) into `model` in place.
/// `R2` rotates the value subspace of each attention head: it's merged into
/// `v_proj` (writer, per-KV-head output rows `Wv → R2·Wv`) and `o_proj` (reader,
/// per-query-head input columns `Wo → Wo·R2ᵀ`). Attention is linear in `V`, so
/// the rotated value flows into the context unchanged after `o_proj` un-rotates
/// it — the fp output is invariant, but the quantizer sees a better-conditioned
/// per-head value/o_proj basis. Composes with [`apply_r1`] (different axes: R1 on
/// the hidden dim, R2 on the head dim); apply either order. Shares one `R2` across
/// heads (SpinQuant's construction).
pub fn apply_r2(gpu: &mut Gpu, model: &mut LlamaModel, r2: &Rotation) -> HipResult<()> {
    let hd = model.dims.head_dim;
    assert_eq!(r2.h, hd, "R2 size {} != head_dim {}", r2.h, hd);
    let (n_kv, n_heads, h) = (model.dims.n_kv, model.dims.n_heads, model.dims.h);
    for (w, _lora) in model.layers.iter_mut() {
        // v_proj [kv_dim, h]: rotate each KV head's head_dim output rows by R2.
        replace_tensor(gpu, &mut w.wv, |m| rotate_head_rows(&m, r2, n_kv, h))?;
        // o_proj [h, q_dim]: rotate each query head's head_dim input cols by R2ᵀ.
        replace_tensor(gpu, &mut w.wo, |m| rotate_head_cols(&m, r2, n_heads, h))?;
    }
    Ok(())
}

/// Bake a learned rotation `M` for deployment through the Oq4G256 codec: returns
/// `R1 = Fᵀ M` where `F` is the codec's per-256-group FWHT. Passing this to
/// [`apply_r1`] merges it into the weights; the codec then applies `F`, and
/// `F·(Fᵀ M) = M`, so the int4 grid sees the learned rotation. The result is a
/// **standard** `Oq4G256` `.hfq` — R1 is fully merged (zero runtime cost, no
/// loader changes); export is just "`apply_r1(bake_for_oq4_recipe(M))` then run
/// the normal Oq4 quantizer on the rotated weights".
pub fn bake_for_oq4_recipe(m: &Rotation) -> Rotation {
    Rotation::block_fwht(m.h).transpose().compose(m)
}

/// Sign seeds and group width of the live 128-wide FWHT codec — the one
/// `paro_la_gates_codec`'s MQ4G128 encoder writes and `gemv_hfq4g128` reads
/// (tables uploaded by `ensure_mq_signs_128`).
pub const G128_CODEC: (usize, u32, u32) = (128, 43, 1043);

/// Bake a learned per-head rotation `R2` for deployment through a codec whose
/// FWHT group width EQUALS the head dimension.
///
/// This is what unblocks R2 deploy. The Oq4G256 path cannot carry R2 at all: its
/// FWHT is 256 wide, so with `head_dim = 128` a group spans two heads and the net
/// transform is `F₂₅₆ ∘ blockdiag(R2)`, which is not `R2` and cannot be un-baked
/// by any per-head pre-rotation. Nothing about that is fixable in the plumbing —
/// it is true of the safetensors-round-trip route too, because that runs the same
/// codec.
///
/// When the group width equals `head_dim`, each FWHT group is exactly one head,
/// so the identity that makes R1 work applies per head: pre-rotate by `Fᵀ R2`,
/// the codec applies `F`, and `F·(Fᵀ R2) = R2` leaves the int4 grid in the learned
/// per-head basis. `r2` is the single `[head_dim, head_dim]` block; `apply_r2`
/// already expands one block across heads via `rotate_head_cols`.
///
/// `group` must equal `r2.h`, and the seeds must be the target codec's — a
/// mismatch on either leaves a residual rotation instead of cancelling, and does
/// so silently, since the product is still orthonormal.
pub fn bake_for_per_head_codec(r2: &Rotation, group: usize, seed1: u32, seed2: u32) -> Rotation {
    assert_eq!(
        r2.h, group,
        "per-head bake needs the codec group width ({group}) to equal head_dim ({}); \
         a group that spans two heads cannot be un-baked per head",
        r2.h
    );
    Rotation::block_fwht_with(group, group, seed1, seed2)
        .transpose()
        .compose(r2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The R2 deploy proof: with the codec's FWHT group width equal to head_dim,
    /// pre-rotating by `Fᵀ R2` and then applying the codec's per-group FWHT leaves
    /// exactly `R2`. This is the per-head analogue of
    /// `bake_cancels_codec_fwht_leaving_m` in hipfire-quantize's rotate.rs.
    #[test]
    fn per_head_bake_cancels_the_codec_fwht_leaving_r2() {
        let (group, seed1, seed2) = G128_CODEC;
        let n_heads = 3usize;
        let h = group * n_heads;
        let r2 = Rotation::hadamard(group, 7);
        let baked = bake_for_per_head_codec(&r2, group, seed1, seed2);

        // A deterministic weight [out, h] whose h dim is head-major.
        let out = 9usize;
        let mut w = vec![0.0f32; out * h];
        let mut st = 0x9E37_79B9u64;
        for v in w.iter_mut() {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
        }

        // Path 1: rotate per head by the BAKED R2 using the SAME helper the
        // production `apply_r2` uses, then apply the codec FWHT per group —
        // which is per head, because group == head_dim. Using the helper rather
        // than a hand-rolled multiply is deliberate: it is the convention that
        // ships, and a transposed convention here would silently pass or fail
        // for reasons unrelated to the bake.
        let mut w1 = rotate_head_cols(&w, &baked, n_heads, out);
        let (s1, s2) = (gen_fwht_signs(seed1, group), gen_fwht_signs(seed2, group));
        for o in 0..out {
            for head in 0..n_heads {
                let base = o * h + head * group;
                let mut buf = w1[base..base + group].to_vec();
                signed_fwht(&mut buf, &s1, &s2);
                w1[base..base + group].copy_from_slice(&buf);
            }
        }

        // Path 2: rotate per head directly by R2 — what the int4 grid should see.
        let w2 = rotate_head_cols(&w, &r2, n_heads, out);

        let worst = w1
            .iter()
            .zip(&w2)
            .fold(0.0f32, |acc, (&a, &b)| acc.max((a - b).abs()));
        assert!(
            worst < 1e-3,
            "per-head bake did not cancel the codec FWHT: max|delta| {worst:.2e}"
        );
    }

    /// The OTHER silent failure the doc comment warns about: right width, wrong
    /// SEEDS. `Fᵀ_a` then `F_b` is still an orthonormal matrix, so nothing errors
    /// and the artifact quantizes cleanly — it just is not in the learned basis.
    /// Only a numeric check catches it, so here is the numeric check.
    #[test]
    fn baking_against_the_wrong_sign_seeds_does_not_cancel() {
        let (group, seed1, seed2) = G128_CODEC;
        let n_heads = 2usize;
        let out = 5usize;
        let h = group * n_heads;
        let r2 = Rotation::hadamard(group, 7);
        // Baked with the Oq4G256 codec's seeds (42/1042) instead of this codec's.
        let wrong = bake_for_per_head_codec(&r2, group, 42, 1042);

        let mut w = vec![0.0f32; out * h];
        let mut st = 0xDEAD_BEEFu64;
        for v in w.iter_mut() {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
        }

        let mut w1 = rotate_head_cols(&w, &wrong, n_heads, out);
        let (s1, s2) = (gen_fwht_signs(seed1, group), gen_fwht_signs(seed2, group));
        for o in 0..out {
            for head in 0..n_heads {
                let base = o * h + head * group;
                let mut buf = w1[base..base + group].to_vec();
                signed_fwht(&mut buf, &s1, &s2);
                w1[base..base + group].copy_from_slice(&buf);
            }
        }
        let w2 = rotate_head_cols(&w, &r2, n_heads, out);
        let worst = w1
            .iter()
            .zip(&w2)
            .fold(0.0f32, |acc, (&a, &b)| acc.max((a - b).abs()));
        assert!(
            worst > 1e-2,
            "wrong seeds appeared to cancel (max|delta| {worst:.2e}) — then the \
             seeds do not matter and the doc comment is wrong"
        );
    }

    /// The negative control, and the reason this needed a new codec width at all:
    /// a 256-wide FWHT over `head_dim = 128` spans TWO heads, so no per-head
    /// pre-rotation can cancel it. Without this, the test above would pass just as
    /// happily against a width that cannot work, and prove nothing.
    #[test]
    fn a_group_spanning_two_heads_cannot_be_un_baked() {
        let head_dim = 128usize;
        let r2 = Rotation::hadamard(head_dim, 7);
        // The 256 codec's block over a 256-wide span is NOT block-diagonal in
        // 128-blocks, so Fᵀ over 256 does not commute with a per-head R2.
        let f256 = Rotation::block_fwht(256);
        let mut cross = 0.0f32;
        for j in 0..256 {
            for i in 0..256 {
                let same_head = (j < head_dim) == (i < head_dim);
                if !same_head {
                    cross = cross.max(f256.r[j * 256 + i].abs());
                }
            }
        }
        assert!(
            cross > 1e-3,
            "F256 has no cross-head energy ({cross:.2e}) — if this ever holds, the \
             256 codec could carry R2 and bake_for_per_head_codec is unnecessary"
        );
        assert_eq!(r2.h, head_dim);
    }

    #[test]
    fn random_rotation_is_orthonormal() {
        for &h in &[4usize, 8, 16, 33] {
            let rot = Rotation::random(h, 12345 + h as u64);
            let err = rot.orthonormality_error();
            assert!(err < 1e-4, "h={h} orthonormality err {err:e}");
        }
    }

    #[test]
    fn identity_is_orthonormal() {
        assert!(Rotation::identity(8).orthonormality_error() < 1e-6);
    }

    #[test]
    fn hadamard_is_orthonormal() {
        for &h in &[2usize, 4, 8, 16, 64] {
            let err = Rotation::hadamard(h, 3).orthonormality_error();
            assert!(err < 1e-5, "h={h} hadamard orthonormality err {err:e}");
        }
    }

    #[test]
    fn block_fwht_matches_cpu_fwht_and_is_orthonormal() {
        use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
        let f = Rotation::block_fwht(512);
        assert!(
            f.orthonormality_error() < 1e-4,
            "block_fwht not orthonormal"
        );
        // rotate_rows(x, F) must reproduce the codec's per-256-group fwht.
        let x: Vec<f32> = (0..2 * 512).map(|i| (i as f32 * 0.013).sin()).collect();
        let via_rot = rotate_rows(&x, &f, 2);
        let (s1, s2) = (gen_fwht_signs(42, 256), gen_fwht_signs(1042, 256));
        let mut manual = x.clone();
        for r in 0..2 {
            for seg in 0..2 {
                let base = r * 512 + seg * 256;
                let mut buf = [0.0f32; 256];
                buf.copy_from_slice(&manual[base..base + 256]);
                cpu_fwht_256(&mut buf, &s1, &s2);
                manual[base..base + 256].copy_from_slice(&buf);
            }
        }
        let worst = via_rot
            .iter()
            .zip(&manual)
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        assert!(worst < 1e-4, "block_fwht != codec fwht: {worst:e}");
    }

    /// The export bake identity: applying `R1 = Fᵀ M` and then the codec's FWHT
    /// `F` reproduces the direct learned rotation `M` — so a standard Oq4G256
    /// quantize of the `apply_r1(bake_for_oq4_recipe(M))` weights carries the
    /// learned rotation, no loader changes.
    #[test]
    fn bake_composes_to_learned_through_codec_fwht() {
        let h = 512usize;
        let m = Rotation::random(h, 5);
        let f = Rotation::block_fwht(h);
        let r1_bake = bake_for_oq4_recipe(&m);
        assert!(
            r1_bake.orthonormality_error() < 1e-3,
            "baked R1 not orthonormal"
        );
        let x: Vec<f32> = (0..3 * h).map(|i| (i as f32 * 0.007).cos()).collect();
        // Deployed: apply R1_bake (merged into weights) then the codec FWHT.
        let deployed = rotate_rows(&rotate_rows(&x, &r1_bake, 3), &f, 3);
        // Direct: the learned rotation M.
        let direct = rotate_rows(&x, &m, 3);
        let worst = deployed
            .iter()
            .zip(&direct)
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        assert!(worst < 1e-3, "bake identity broken: {worst:e}");
    }

    /// A reader followed by the residual rotation reproduces the pre-rotation
    /// activation: `(x Rᵀ) · (W Rᵀ)ᵀ = x Wᵀ`. This is the invariance identity the
    /// whole transform rests on, checked on tiny random data.
    #[test]
    fn reader_rotation_preserves_activation() {
        let (h, out) = (8usize, 5usize);
        let rot = Rotation::random(h, 7);
        // Random x [1,h] and W [out,h].
        let x: Vec<f32> = (0..h).map(|i| (i as f32 * 0.37).sin()).collect();
        let w: Vec<f32> = (0..out * h).map(|i| (i as f32 * 0.11).cos()).collect();
        // Original activation y = x Wᵀ.
        let y: Vec<f32> = (0..out)
            .map(|o| (0..h).map(|i| x[i] * w[o * h + i]).sum::<f32>())
            .collect();
        // Rotated residual x̃ = x Rᵀ  (x̃[j] = Σ_i x[i] R[j,i]).
        let xr: Vec<f32> = (0..h)
            .map(|j| (0..h).map(|i| x[i] * rot.r[j * h + i]).sum::<f32>())
            .collect();
        // Rotated reader W Rᵀ, then activation ỹ = x̃ (W Rᵀ)ᵀ.
        let wr = rotate_input(&w, &rot, out);
        let yr: Vec<f32> = (0..out)
            .map(|o| (0..h).map(|j| xr[j] * wr[o * h + j]).sum::<f32>())
            .collect();
        let worst = y
            .iter()
            .zip(&yr)
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        assert!(worst < 1e-4, "reader-rotation mismatch {worst:e}");
    }

    /// R2 merge correctness: rotating `v_proj`'s per-head output rows by `R2` and
    /// `o_proj`'s per-head input columns by `R2ᵀ` composes to identity through the
    /// value→(identity attention)→o_proj pipeline — so the fp output is preserved.
    /// `attn = Wo·(Wv·x)` must equal `Wo_new·(Wv_new·x)`.
    #[test]
    fn r2_headwise_merge_is_identity() {
        let (hd, n_heads, h) = (4usize, 3usize, 8usize);
        let qd = n_heads * hd; // n_kv == n_heads here (ctx = v, identity attention)
        let r2 = Rotation::random(hd, 3);
        let x: Vec<f32> = (0..h).map(|i| (i as f32 * 0.31).sin()).collect();
        let wv: Vec<f32> = (0..qd * h).map(|i| (i as f32 * 0.07).cos()).collect(); // [qd,h]
        let wo: Vec<f32> = (0..h * qd).map(|i| (i as f32 * 0.05).sin()).collect(); // [h,qd]
        let mul = |w: &[f32], v: &[f32], rows: usize, inner: usize| -> Vec<f32> {
            (0..rows)
                .map(|o| (0..inner).map(|k| w[o * inner + k] * v[k]).sum())
                .collect()
        };
        let attn = mul(&wo, &mul(&wv, &x, qd, h), h, qd);
        let wv_new = rotate_head_rows(&wv, &r2, n_heads, h);
        let wo_new = rotate_head_cols(&wo, &r2, n_heads, h);
        let attn_new = mul(&wo_new, &mul(&wv_new, &x, qd, h), h, qd);
        let worst = attn
            .iter()
            .zip(&attn_new)
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        assert!(worst < 1e-4, "R2 merge not identity: {worst:e}");
    }

    /// The residual rotation applied to a writer output reproduces the rotated
    /// contribution: `R (Wᵀ c)` written by `(R W)`. Checks `writer` matches
    /// rotating the plain output.
    #[test]
    fn writer_rotation_matches_rotated_output() {
        let (h, cols) = (8usize, 5usize);
        let rot = Rotation::random(h, 9);
        let c: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.53).sin()).collect();
        let w: Vec<f32> = (0..h * cols).map(|i| (i as f32 * 0.17).cos()).collect();
        // Plain output o = W c  (o[d] = Σ_c W[d,c] c[c]), then rotate: õ = R o.
        let o: Vec<f32> = (0..h)
            .map(|d| (0..cols).map(|cc| w[d * cols + cc] * c[cc]).sum::<f32>())
            .collect();
        let or: Vec<f32> = (0..h)
            .map(|e| (0..h).map(|d| rot.r[e * h + d] * o[d]).sum::<f32>())
            .collect();
        // Rotated writer R W, then output = (R W) c.
        let wr = rotate_output(&w, &rot, cols);
        let ow: Vec<f32> = (0..h)
            .map(|e| (0..cols).map(|cc| wr[e * cols + cc] * c[cc]).sum::<f32>())
            .collect();
        let worst = or
            .iter()
            .zip(&ow)
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        assert!(worst < 1e-4, "writer-rotation mismatch {worst:e}");
    }
}
