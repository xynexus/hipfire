// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Qwen Sparse Attention block scoring for qwen4_exp.
//
// One score per micro-block of `compress_ratio` cached tokens:
//
//   block_key[b] = mean over the block's tokens of K[pos]
//   score[b]     = sum_h relu(q[h] . block_key[b]) / sqrt(head_dim)
//
// Three things differ from DeepSeek V4's indexer, which is the nearest sibling:
//
//   * keys are MEAN-POOLED over raw cached K here, where DSV4 scores a
//     learned-compressed cache — same `compress_ratio`, different algorithm;
//   * the per-head scores are ReLU'd BEFORE being summed (a sum-then-ReLU lets a
//     strongly negative head cancel a strongly positive one);
//   * there is NO learned per-head weight. DSV4 has `indexer_weights_proj`; this
//     family's checkpoint carries no such tensor, so a port that keeps it would
//     need to invent one.
//
// `visible` lists the causally-legal cache positions in increasing order, so a
// block's tokens need not be contiguous in the cache — a masked or paged cache
// still scores correctly.
//
// Differenced against `hipfire_arch_qwen4exp::qsa::score_block`.
//
// Grid: (n_blocks). Block = ONE WAVE (32 lanes), striding over head_dim.
//
// 32, not 64: gfx1151 is wave32, and `__shfl_xor` does not cross a wave. A 64-lane
// block with a reduction step at offset 32 makes that step return the caller's own
// value, so the dot silently DOUBLES — caught here by the relu-before-sum control
// reading exactly 2x its expected value. `gated_norm_f32` uses the same 32-lane
// shape for the same reason.

#define LANES 32
#define MAX_HEAD_DIM 256

__launch_bounds__(LANES)
extern "C" __global__ void qsa_block_score(
    const float* __restrict__ q,        // [n_heads, head_dim]
    const float* __restrict__ keys,     // [n_positions, head_dim]
    const int*   __restrict__ visible,  // [n_visible] cache positions, increasing
    float*       __restrict__ scores,   // [n_blocks]
    int n_heads,
    int head_dim,
    int compress_ratio
) {
    const int b = blockIdx.x;
    const int lane = threadIdx.x;
    if (head_dim > MAX_HEAD_DIM) return;

    // Pool this block's keys into shared memory once, then reuse across heads.
    __shared__ float pooled[MAX_HEAD_DIM];
    for (int i = lane; i < head_dim; i += LANES) {
        float acc = 0.0f;
        for (int t = 0; t < compress_ratio; ++t) {
            const int pos = visible[b * compress_ratio + t];
            acc += keys[(long long)pos * head_dim + i];
        }
        pooled[i] = acc / (float)compress_ratio;
    }
    __syncthreads();

    float total = 0.0f;
    for (int h = 0; h < n_heads; ++h) {
        const float* qh = q + (long long)h * head_dim;
        float dot = 0.0f;
        for (int i = lane; i < head_dim; i += LANES) {
            dot = fmaf(qh[i], pooled[i], dot);
        }
        for (int o = LANES / 2; o > 0; o >>= 1) {  // 16, 8, 4, 2, 1 — all in-wave
            dot += __shfl_xor(dot, o);
        }
        // ReLU per head, INSIDE the sum.
        total += fmaxf(dot, 0.0f);
    }

    if (lane == 0) {
        scores[b] = total / sqrtf((float)head_dim);
    }
}
