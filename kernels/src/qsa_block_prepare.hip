// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// QSA block-key preparation, split from scoring.
//
// `qsa_block_score` pools its keys INTERNALLY and scores in the same launch. That
// cannot express what the reference actually does, which is three steps with two
// transforms wedged between the pool and the score:
//
//     pooled  = mean(raw_keys[block])
//     pooled  = k_layernorm(pooled)          <-- not expressible inside that kernel
//     blockk  = rope(pooled, at the block's FIRST position)   <-- nor this
//     score   = sum_h relu(q_h . blockk) / sqrt(head_dim)
//
// So the pooling is separated here, the rotation happens on the prepared keys via
// the normal batched RoPE path, and `qsa_score_prepared` scores what comes out.
// `qsa_block_score` stays for the pool-and-score case it was written for.

#define BLOCK 256

// Mean-pool each block's raw keys, then per-block RMSNorm with a shared weight.
// Also emits each block's FIRST visible position, which is the position its
// pooled key is rotated at.
__launch_bounds__(BLOCK)
extern "C" __global__ void qsa_pool_norm_blocks(
    const float* __restrict__ keys,     // [n_positions, head_dim] raw, un-normed
    const int*   __restrict__ visible,  // [n_visible] cache positions, increasing
    const float* __restrict__ weight,   // [head_dim], shared across blocks
    float*       __restrict__ out,      // [n_blocks, head_dim]
    int*         __restrict__ starts,   // [n_blocks] first position of each block
    int   n_blocks,
    int   compress_ratio,
    int   head_dim,
    float eps
) {
    const int b = blockIdx.x;
    if (b >= n_blocks) return;
    const long long dst = (long long)b * head_dim;

    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        float acc = 0.0f;
        for (int t = 0; t < compress_ratio; ++t) {
            acc += keys[(long long)visible[b * compress_ratio + t] * head_dim + i];
        }
        out[dst + i] = acc / (float)compress_ratio;
    }
    if (threadIdx.x == 0) starts[b] = visible[b * compress_ratio];
    __syncthreads();

    float acc = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = out[dst + i];
        acc = fmaf(v, v, acc);
    }
    __shared__ float red[BLOCK];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float scale = rsqrtf(red[0] / (float)head_dim + eps);
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        out[dst + i] = out[dst + i] * scale * (1.0f + weight[i]);
    }
}

// Score already-prepared block keys: ReLU per head, INSIDE the sum over heads.
//
// relu-then-sum, not sum-then-relu. They differ exactly when a head scores
// negative, which is when selection matters.
__launch_bounds__(BLOCK)
extern "C" __global__ void qsa_score_prepared(
    const float* __restrict__ q,           // [n_heads, head_dim]
    const float* __restrict__ block_keys,  // [n_blocks, head_dim]
    float*       __restrict__ scores,      // [n_blocks]
    int n_heads,
    int head_dim,
    int n_blocks
) {
    const int b = blockIdx.x;
    if (b >= n_blocks) return;
    const float* __restrict__ bk = block_keys + (long long)b * head_dim;

    __shared__ float red[BLOCK];
    float total = 0.0f;
    for (int h = 0; h < n_heads; ++h) {
        const float* __restrict__ qh = q + (long long)h * head_dim;
        float dot = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += BLOCK) dot = fmaf(qh[i], bk[i], dot);
        red[threadIdx.x] = dot;
        __syncthreads();
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
            if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        total += fmaxf(red[0], 0.0f);
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[b] = total / sqrtf((float)head_dim);
}
