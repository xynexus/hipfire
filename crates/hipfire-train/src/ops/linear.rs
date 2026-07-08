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

/// `y = x · wᵀ`, ALWAYS f32 — ignores `HIPFIRE_TRAIN_LOWP`. Used by the vocab
/// heads (lm-head / markov / confidence), whose logits feed the softmax/CE loss
/// and are precision-sensitive: the low-precision forward is scoped to the
/// rmsnorm-bounded body, where it is safe, and kept off the loss-critical logits.
pub fn linear_forward_f32(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    gpu.gemm_f32_train(x, w, y, m, n, k, k, k, false, true)
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
