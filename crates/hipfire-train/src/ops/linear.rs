// SPDX-License-Identifier: Apache-2.0
//! Linear (no bias): `Y = X · Wᵀ`, with `X:[M,K]` (tokens×in), `W:[N,K]`
//! (HF row-major `[out, in]`), `Y:[M,N]`.
//!
//! All three products go through `gemm_f32_train` (general transpose flags):
//!   forward  Y[M,N]  = X·Wᵀ  : trans_b=true
//!   dX[M,K]          = dY·W   : no transpose
//!   dW[N,K]          = dYᵀ·X  : trans_a=true
//! See the mapping table in docs/plans/2026-06-17-hipfire-train-phase0.md §1.

use hipfire_rdna::{Gpu, GpuTensor, HipResult};
use std::sync::OnceLock;

// NOTE (2026-07-08): a WMMA backward — reformulating dX (contracts N) and dW
// (contracts M) as NT via `transpose_f32` + `gemm_bf16x2_train_nt` — was tried
// and REVERTED for TWO independent reasons:
//   1. It DIVERGES training. Isolated (f32 forward + bf16x2 backward), the
//      overfit reached best_eval 3.60 then climbed to 8.5 (vs f32's 1.63). This
//      is NOT a precision problem: the bf16x2 GEMM matches f32 to ~1e-5 on every
//      shape incl. the backward mappings (see gemm_bf16c_parity), so more split
//      terms would not help. It is a SCALE-ONLY bug in the backward wrapper
//      (transpose_f32 / the scratch+add accumulate / the gfx1151 LDS m128 GEMM
//      variant used with the backward's large m,k,n) that BOTH the gradcheck
//      (small toy dims → only the simple GEMM variant, no LDS) AND the GEMM
//      parity (GEMM output only, not the wrapper) miss. Root-cause it against the
//      real training tensors (large dims + sub_offset slices) before retrying.
//   2. Even correct, it is ~5x SLOWER: the body backward is dominated by small-M
//      GEMMs (contract = block = 7) where the transpose + 3-pass split + per-call
//      scratch alloc dwarf the tiny scalar-f32 cost.
// So the backward stays f32. The real win needs batching the drafter body across
// windows first (raises the backward's M, amortizes the transposes), which also
// makes the divergence easier to bisect. Revisit the two together.

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
    if accumulate {
        gpu.gemm_f32_train_accum(dy, w, dx, m, k, n, n, k, false, false, 1.0)
    } else {
        gpu.gemm_f32_train(dy, w, dx, m, k, n, n, k, false, false)
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
    if accumulate {
        gpu.gemm_f32_train_accum(dy, x, dw, n, k, m, n, k, true, false, 1.0)
    } else {
        gpu.gemm_f32_train(dy, x, dw, n, k, m, n, k, true, false)
    }
}
