// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// DFlash2 candidate-selector transition scores for ONE position.
//
//   score[k] = unary[k] + Σ_r predecessor[r] · hidden_proj[r] · successor[k, r]
//
// The predecessor codebook row is gated ELEMENTWISE by the projected hidden and
// then dotted against each candidate's successor row — a rank-`rank` TRIPLE
// product, not a plain bilinear form. Semantics pinned by
// `hipfire_runtime::dflash2::selector_scores`, itself parity-checked against
// z-lab/dflash on real checkpoint weights.
//
// The full selector is a GREEDY left-to-right trace: argmax per position, no
// backtracking, each pick seeding the next predecessor. That sequential
// dependency is why this kernel scores one position per launch — the caller
// picks the argmax and gathers the next predecessor row between calls.
//
//   Unary : [top_k]        f32
//   Pred  : [rank]         f32   predecessor codebook row
//   Hp    : [rank]         f32   hidden_projection(hidden) at this position
//   Suc   : [top_k, rank]  f32   successor codebook rows for the candidates
//   Out   : [top_k]        f32
//
// One wave per candidate, lanes striding `rank`, then a wave reduction. rank is
// 256 and top_k 16, so this is a handful of microseconds.
//
// The block is ONE wave (32 threads on RDNA). It must stay one wave: the
// reduction is `__shfl_down`, which only crosses lanes WITHIN a wave, so a
// 64-thread block on wave32 silently drops the upper wave's partial sum. That
// showed up as max|Δ|=6.7e-2 against the CPU reference before this was pinned.

#include <hip/hip_runtime.h>

__launch_bounds__(32) extern "C" __global__ void dflash2_candidate_selector(
    const float* __restrict__ Unary,
    const float* __restrict__ Pred,
    const float* __restrict__ Hp,
    const float* __restrict__ Suc,
    float*       __restrict__ Out,
    int top_k, int rank
) {
    const int k = blockIdx.x;
    if (k >= top_k) return;
    const int lane = threadIdx.x;
    const int width = blockDim.x;

    const float* suc = Suc + (long long)k * rank;
    float acc = 0.0f;
    for (int r = lane; r < rank; r += width) {
        // Gate the predecessor row by the projected hidden, then dot with this
        // candidate's successor row.
        acc = fmaf(Pred[r] * Hp[r], suc[r], acc);
    }
    // Wave reduction over the lanes that split `rank`.
    for (int off = width >> 1; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off, width);
    }
    if (lane == 0) {
        Out[k] = Unary[k] + acc;
    }
}
