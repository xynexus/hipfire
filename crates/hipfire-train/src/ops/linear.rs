// SPDX-License-Identifier: Apache-2.0
//! Linear (no bias): `Y = X · Wᵀ`, with `X:[M,K]` (tokens×in), `W:[N,K]`
//! (HF row-major `[out, in]`), `Y:[M,N]`.
//!
//! All three products go through `gemm_f32_train` (general transpose flags):
//!   forward  Y[M,N]  = X·Wᵀ  : trans_b=true
//!   dX[M,K]          = dY·W   : no transpose
//!   dW[N,K]          = dYᵀ·X  : trans_a=true
//! See the mapping table in docs/plans/2026-06-17-hipfire-train-phase0.md §1.

use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};
use std::sync::OnceLock;

// WMMA backward (opt-in `HIPFIRE_TRAIN_BWD=bf16x2`): dX (contracts N) and dW
// (contracts M) are reformulated as NT via `transpose_f32` + the split-precision
// `gemm_bf16x2_train_nt`, so both backward matmuls run on the matrix cores at
// near-f32 accuracy. Off by default so the finite-difference gradchecks keep the
// exact f32 path.
//
// HISTORY (2026-07-08): a first attempt was reverted after the isolation overfit
// "diverged" (best_eval 3.60→8.5 vs an f32 run's 1.63), blamed on a "scale bug"
// in transpose_f32 / the scratch+add accumulate / the gfx1151 m128 GEMM variant.
// That diagnosis was WRONG. Three independent checks show the backward is
// correct, and the divergence was chaotic amplification, not a defect:
//   1. `examples/gemm_bf16x2_backward_parity` validates EACH piece against the
//      f32 reference at real training dims — m128 variant (dX at M>=16), LDS-free
//      variant (M<16), the dW transposes, the scratch+add accumulate, and
//      `sub_offset` views — all match to <=1.2e-3, cos 1.0.
//   2. The overfit train-loss curve is BIT-IDENTICAL f32-vs-bf16x2 for the first
//      ~5 epochs, then differs at ~1e-4 (bf16x2's rounding floor) before the
//      curves wander apart — the fingerprint of a tiny perturbation amplified by
//      an unstable regime (the f32 loss itself spikes 16.8→23.9 and bounces).
//   3. Two IDENTICAL f32 runs diverge from EACH OTHER just as much (best 5.70 vs
//      4.25). The training is nondeterministic — `rmsnorm_train.hip` accumulates
//      dw via atomicAdd (order-dependent) — so the end-loss is chaos-dominated
//      and cannot discriminate a correct low-precision backward from f32 at all.
// ⇒ Validate the backward by gradient parity + curve-tracking, NOT overfit
// end-loss. The reformulation lives in ONE seam (below) so there is no
// per-call-site wiring to get wrong. Blocker 2 stands: the small-M body backward
// (contract = block = 7) is overhead-bound and wants Phase A window-batching
// before WMMA is a net speed win — so this is correctness-proven but gated OFF
// by default (also keeps the finite-difference gradchecks on the exact path).

/// Low-precision compute mode for the linear FORWARD, selected by
/// `HIPFIRE_TRAIN_LOWP` (`f16` | `bf16`; anything else / unset = f32).
#[derive(Clone, Copy, PartialEq)]
enum LowpFwd {
    F32,
    F16,
    Bf16,
}

/// Opt-in (`HIPFIRE_TRAIN_LOWP=f16|bf16`): run the linear FORWARD on a WMMA GEMM
/// (matrix cores, f32 accumulate) that casts the f32 operands to f16/bf16, vs the
/// scalar f32 kernel. Off by default so the finite-difference gradchecks — which
/// need the forward to resolve perturbations below f16/bf16 precision — keep the
/// exact f32 path. The backward matmuls stay f32 regardless (they contract axes
/// the NT WMMA kernel does not cover, and are ~17% of step time). f16 (10
/// mantissa bits) preserves convergence better than bf16 (7) for the drafter's
/// rmsnorm'd activations. Cached: the forward is on the hot path.
fn lowp_forward() -> LowpFwd {
    static MODE: OnceLock<LowpFwd> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("HIPFIRE_TRAIN_LOWP").as_deref() {
        Ok("f16") | Ok("fp16") => LowpFwd::F16,
        Ok("bf16") => LowpFwd::Bf16,
        _ => LowpFwd::F32,
    })
}

/// `y = x · wᵀ`. Shapes: `x:[m*k]`, `w:[n*k]`, `y:[m*n]` (all flat fp32).
pub fn linear_forward(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    match lowp_forward() {
        LowpFwd::F16 => gpu.gemm_f16c_train_nt(x, w, y, m, k, n),
        LowpFwd::Bf16 => gpu.gemm_bf16c_train_nt(x, w, y, m, k, n),
        LowpFwd::F32 => gpu.gemm_f32_train(x, w, y, m, n, k, k, k, false, true),
    }
}

/// Precision mode for the vocab-head forward, selected by `HIPFIRE_TRAIN_HEADS`
/// (`bf16x2` = split-precision WMMA; anything else / unset = f32).
#[derive(Clone, Copy, PartialEq)]
enum HeadsFwd {
    F32,
    Bf16x2,
}

fn heads_forward_mode() -> HeadsFwd {
    static MODE: OnceLock<HeadsFwd> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("HIPFIRE_TRAIN_HEADS").as_deref() {
        Ok("bf16x2") => HeadsFwd::Bf16x2,
        _ => HeadsFwd::F32,
    })
}

/// `y = x · wᵀ` for the vocab heads (lm-head / markov / confidence). Their logits
/// feed the softmax/CE loss and are precision-sensitive, so plain bf16 is off the
/// table (it degrades convergence); the choice is f32 (default — also what the
/// gradchecks need) vs `HIPFIRE_TRAIN_HEADS=bf16x2`, the split-precision WMMA GEMM
/// that reaches ~f32 accuracy (~16 mantissa bits) on the matrix cores. The
/// low-precision *body* forward (`HIPFIRE_TRAIN_LOWP`) is independent of this.
pub fn linear_forward_heads(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    match heads_forward_mode() {
        HeadsFwd::Bf16x2 => gpu.gemm_bf16x2_train_nt(x, w, y, m, k, n),
        HeadsFwd::F32 => gpu.gemm_f32_train(x, w, y, m, n, k, k, k, false, true),
    }
}

/// Low-precision compute mode for the linear BACKWARD, selected by
/// `HIPFIRE_TRAIN_BWD` (`bf16` = single-pass 8-bit mantissa; `f16` = single-pass
/// 10-bit mantissa but 5-bit exponent — overflows on large operands; `bf16x2` =
/// 3-pass split ~16-bit; else / unset = f32).
#[derive(Clone, Copy, PartialEq)]
enum LowpBwd {
    F32,
    Bf16,
    F16,
    F16s,
    F16sC2,
    Bf16x2,
}

fn bwd_mode() -> LowpBwd {
    static MODE: OnceLock<LowpBwd> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("HIPFIRE_TRAIN_BWD").as_deref() {
        Ok("bf16") => LowpBwd::Bf16,
        Ok("f16") | Ok("fp16") => LowpBwd::F16,
        Ok("f16s") | Ok("fp16s") => LowpBwd::F16s,
        Ok("f16sc2") => LowpBwd::F16sC2,
        Ok("bf16x2") => LowpBwd::Bf16x2,
        _ => LowpBwd::F32,
    })
}

/// The NT WMMA GEMM for an *unscaled* low-precision backward:
/// `y[m,n] = Σ_k x[m,k]·w[n,k]`. `bf16`/`f16` = one pass; `bf16x2` = 3-pass split
/// (~16-bit). Same contract as `gemm_f32_train(x, w, y, m, n, k, k, k, false,
/// true)`. `f16s` is scaled and handled separately (see [`f16s_scale`]).
fn bwd_gemm_nt(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
    mode: LowpBwd,
) -> HipResult<()> {
    match mode {
        LowpBwd::Bf16 => gpu.gemm_bf16c_train_nt(x, w, y, m, k, n),
        LowpBwd::F16 => gpu.gemm_f16c_train_nt(x, w, y, m, k, n),
        LowpBwd::Bf16x2 => gpu.gemm_bf16x2_train_nt(x, w, y, m, k, n),
        LowpBwd::F32 | LowpBwd::F16s | LowpBwd::F16sC2 => {
            unreachable!("bwd_gemm_nt: unscaled path only")
        }
    }
}

/// Per-tensor power-of-two scale for the scaled-f16 (`f16s`) backward: bring
/// `max|x|` up to ~2^14 (one bit under f16's 2^15 ceiling) so the whole operand
/// fits f16 range with headroom. Power-of-two ⇒ the scale is an exact, reversible
/// `ldexp` (no rounding). Zero/negative ⇒ 1.0.
fn pow2_from_amax(amax: f32) -> f32 {
    if !(amax > 0.0) {
        return 1.0;
    }
    (16384.0_f32 / amax).log2().floor().exp2()
}

/// [`pow2_from_amax`] from a standalone abs-max reduction, for the one f16s
/// operand not produced by a transpose (dY in `linear_backward_x`). Operands that
/// come out of a transpose get their abs-max fused into it (`transpose_f32_amax`).
fn f16s_scale(gpu: &mut Gpu, x: &GpuTensor) -> HipResult<f32> {
    Ok(pow2_from_amax(gpu.abs_max_f32(x)?))
}

/// f16s backward strategy crossover (measured, gfx1151): when the strided
/// operand's contract/output axis exceeds this, the C2 no-transpose kernels'
/// UNCOALESCED strided reads lose to the transpose path (coalesced read + write +
/// read, whose overhead amortizes over the few huge matmuls) — the lm-head regime
/// (N=vocab~262k). Below it (the body's many small matmuls) the transpose
/// overhead dominates and the strided C2 kernels win ~6.8×. Measured: body_bwd
/// 1059→156 ms (C2) vs heads_bwd 174 (transpose) < 238 (C2).
const F16S_STRIDED_MAX: usize = 65536;

/// dX = dY·W (contract N). `force_c2` pins the strided kernel; otherwise dispatch
/// on N: C2 NN (strided, no transpose) for small N, transpose+NT for huge N.
#[allow(clippy::too_many_arguments)]
fn f16s_backward_x(
    gpu: &mut Gpu,
    dy: &GpuTensor,
    w: &GpuTensor,
    dx: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
    accumulate: bool,
    force_c2: bool,
) -> HipResult<()> {
    let s_dy = f16s_scale(gpu, dy)?;
    if force_c2 || n <= F16S_STRIDED_MAX {
        // C2 NN: read W strided, no transpose pass.
        let s_w = f16s_scale(gpu, w)?;
        if accumulate {
            let scratch = gpu.zeros(&[m * k], DType::F32)?;
            gpu.gemm_f16s_nn_train(dy, w, &scratch, m, n, k, s_dy, s_w)?;
            gpu.add_inplace_f32(dx, &scratch)?;
            gpu.free_tensor(scratch)
        } else {
            gpu.gemm_f16s_nn_train(dy, w, dx, m, n, k, s_dy, s_w)
        }
    } else {
        // Huge N (lm-head): transpose W (abs-max fused) then NT — coalesced.
        let wt = gpu.zeros(&[k * n], DType::F32)?;
        let s_wt = pow2_from_amax(gpu.transpose_f32_amax(w, &wt, n, k)?);
        if accumulate {
            let scratch = gpu.zeros(&[m * k], DType::F32)?;
            gpu.gemm_f16s_train_nt(dy, &wt, &scratch, m, n, k, s_dy, s_wt)?;
            gpu.add_inplace_f32(dx, &scratch)?;
            gpu.free_tensor(scratch)?;
        } else {
            gpu.gemm_f16s_train_nt(dy, &wt, dx, m, n, k, s_dy, s_wt)?;
        }
        gpu.free_tensor(wt)
    }
}

/// dW = dYᵀ·X (contract M). `force_c2` pins the strided kernel; otherwise
/// dispatch on the output-row axis N: C2 TN for small N, transpose+NT for huge N.
#[allow(clippy::too_many_arguments)]
fn f16s_backward_w(
    gpu: &mut Gpu,
    dy: &GpuTensor,
    x: &GpuTensor,
    dw: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
    accumulate: bool,
    force_c2: bool,
) -> HipResult<()> {
    if force_c2 || n <= F16S_STRIDED_MAX {
        // C2 TN: read both operands strided, no transpose passes.
        let s_dy = f16s_scale(gpu, dy)?;
        let s_x = f16s_scale(gpu, x)?;
        if accumulate {
            let scratch = gpu.zeros(&[n * k], DType::F32)?;
            gpu.gemm_f16s_tn_train(dy, x, &scratch, m, n, k, s_dy, s_x)?;
            gpu.add_inplace_f32(dw, &scratch)?;
            gpu.free_tensor(scratch)
        } else {
            gpu.gemm_f16s_tn_train(dy, x, dw, m, n, k, s_dy, s_x)
        }
    } else {
        // Huge N: transpose dY/X (abs-max fused) then NT.
        let dyt = gpu.zeros(&[n * m], DType::F32)?;
        let s_dyt = pow2_from_amax(gpu.transpose_f32_amax(dy, &dyt, m, n)?);
        let xt = gpu.zeros(&[k * m], DType::F32)?;
        let s_xt = pow2_from_amax(gpu.transpose_f32_amax(x, &xt, m, k)?);
        if accumulate {
            let scratch = gpu.zeros(&[n * k], DType::F32)?;
            gpu.gemm_f16s_train_nt(&dyt, &xt, &scratch, n, m, k, s_dyt, s_xt)?;
            gpu.add_inplace_f32(dw, &scratch)?;
            gpu.free_tensor(scratch)?;
        } else {
            gpu.gemm_f16s_train_nt(&dyt, &xt, dw, n, m, k, s_dyt, s_xt)?;
        }
        gpu.free_tensor(dyt)?;
        gpu.free_tensor(xt)
    }
}

/// Gradient w.r.t. the input: `dx = dy · w`. `dy:[m*n]`, `w:[n*k]`, `dx:[m*k]`.
/// When `accumulate`, does `dx += dy·w` (for inputs feeding multiple consumers,
/// e.g. the rmsnorm output that fans into q/k/v).
pub fn linear_backward_x(
    gpu: &mut Gpu,
    dy: &GpuTensor,
    w: &GpuTensor,
    dx: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
    accumulate: bool,
) -> HipResult<()> {
    match bwd_mode() {
        LowpBwd::F32 => {
            if accumulate {
                gpu.gemm_f32_train_accum(dy, w, dx, m, k, n, n, k, false, false, 1.0)
            } else {
                gpu.gemm_f32_train(dy, w, dx, m, k, n, n, k, false, false)
            }
        }
        // Scaled f16: dispatch strided-C2 vs transpose by N (see f16s_backward_x).
        LowpBwd::F16s => f16s_backward_x(gpu, dy, w, dx, m, k, n, accumulate, false),
        // Force the strided C2 kernels everywhere (tuning / bench).
        LowpBwd::F16sC2 => f16s_backward_x(gpu, dy, w, dx, m, k, n, accumulate, true),
        // dX[m,k] = Σ_n dY[m,n]·W[n,k] (contract N). NT form: dX = NT(dY, Wᵀ),
        // Wᵀ = transpose(w[n,k]) → [k,n]; bwd_gemm_nt(dY, Wᵀ, ·, m, n, k).
        mode => {
            let wt = gpu.zeros(&[k * n], DType::F32)?;
            gpu.transpose_f32(w, &wt, n, k)?; // w [n,k] → [k,n]
            if accumulate {
                let scratch = gpu.zeros(&[m * k], DType::F32)?;
                bwd_gemm_nt(gpu, dy, &wt, &scratch, m, n, k, mode)?;
                gpu.add_inplace_f32(dx, &scratch)?;
                gpu.free_tensor(scratch)?;
            } else {
                bwd_gemm_nt(gpu, dy, &wt, dx, m, n, k, mode)?;
            }
            gpu.free_tensor(wt)
        }
    }
}

/// Gradient w.r.t. the weight: `dw = dyᵀ · x`. `dy:[m*n]`, `x:[m*k]`, `dw:[n*k]`.
pub fn linear_backward_w(
    gpu: &mut Gpu,
    dy: &GpuTensor,
    x: &GpuTensor,
    dw: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
    accumulate: bool,
) -> HipResult<()> {
    match bwd_mode() {
        LowpBwd::F32 => {
            if accumulate {
                gpu.gemm_f32_train_accum(dy, x, dw, n, k, m, n, k, true, false, 1.0)
            } else {
                gpu.gemm_f32_train(dy, x, dw, n, k, m, n, k, true, false)
            }
        }
        // Scaled f16: dispatch strided-C2 vs transpose by N (see f16s_backward_w).
        LowpBwd::F16s => f16s_backward_w(gpu, dy, x, dw, m, k, n, accumulate, false),
        // Force the strided C2 kernels everywhere (tuning / bench).
        LowpBwd::F16sC2 => f16s_backward_w(gpu, dy, x, dw, m, k, n, accumulate, true),
        // dW[n,k] = Σ_m dY[m,n]·X[m,k] (contract M). NT form: dW = NT(dYᵀ, Xᵀ),
        // dYᵀ = transpose(dy)[n,m], Xᵀ = transpose(x)[k,m];
        // bwd_gemm_nt(dYᵀ, Xᵀ, ·, n, m, k).
        mode => {
            let dyt = gpu.zeros(&[n * m], DType::F32)?;
            gpu.transpose_f32(dy, &dyt, m, n)?; // dy [m,n] → [n,m]
            let xt = gpu.zeros(&[k * m], DType::F32)?;
            gpu.transpose_f32(x, &xt, m, k)?; // x [m,k] → [k,m]
            if accumulate {
                let scratch = gpu.zeros(&[n * k], DType::F32)?;
                bwd_gemm_nt(gpu, &dyt, &xt, &scratch, n, m, k, mode)?;
                gpu.add_inplace_f32(dw, &scratch)?;
                gpu.free_tensor(scratch)?;
            } else {
                bwd_gemm_nt(gpu, &dyt, &xt, dw, n, m, k, mode)?;
            }
            gpu.free_tensor(dyt)?;
            gpu.free_tensor(xt)
        }
    }
}
