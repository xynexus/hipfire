// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Portable (any RDNA3+) no-LDS flash-decode attention — same ABI/math as
// attention_f32 but a register-resident online softmax removes the scores[seq_len]
// LDS buffer. Fixed grid [n_heads], zero shared mem, reads seq_len from pos_buf →
// position-independent and safe to bake into a hipGraph. (The gfx1103 variant is
// `#if __gfx1103__`-gated and compiles to a no-op elsewhere; this one is not.)
// One wave32/head; lane l owns dims { l, l+32, ... }. head_dim % 32 == 0, ≤ 512.
__device__ __forceinline__ float hipfire_attn_nolds_allsum32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor(v, offset, 32);
    return v;
}
extern "C" __launch_bounds__(32, 1) __global__ void attention_f32_nolds(
    const float* __restrict__ q, const float* __restrict__ k_cache,
    const float* __restrict__ v_cache, float* __restrict__ out,
    const int* __restrict__ pos_buf, int n_heads, int n_kv_heads, int head_dim,
    int max_seq, float scale
) {
    (void)max_seq;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int seq_len = pos_buf[0] + 1;
    const int lane = threadIdx.x;
    const int kv_h = h / (n_heads / n_kv_heads);
    const int kv_stride = n_kv_heads * head_dim;
    const int kv_off = kv_h * head_dim;
    const float* q_head = q + h * head_dim;
    float q_reg[16];
    float acc[16];
    int ndl = 0;
    for (int d = lane; d < head_dim; d += 32) { q_reg[ndl] = q_head[d]; acc[ndl] = 0.0f; ndl++; }
    float m = -1e30f, l = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        const float* k_t = k_cache + (size_t)t * kv_stride + kv_off;
        float part = 0.0f;
        int di = 0;
        for (int d = lane; d < head_dim; d += 32) part += q_reg[di++] * k_t[d];
        const float s = hipfire_attn_nolds_allsum32(part) * scale;
        const float m_new = fmaxf(m, s);
        const float alpha = __expf(m - m_new);
        const float p = __expf(s - m_new);
        l = l * alpha + p;
        const float* v_t = v_cache + (size_t)t * kv_stride + kv_off;
        di = 0;
        for (int d = lane; d < head_dim; d += 32) { acc[di] = acc[di] * alpha + p * v_t[d]; di++; }
        m = m_new;
    }
    float* out_head = out + h * head_dim;
    const float inv_l = 1.0f / l;
    int di = 0;
    for (int d = lane; d < head_dim; d += 32) out_head[d] = acc[di++] * inv_l;
}

extern "C" __global__ void attention_f32(
    const float* __restrict__ q,          // [n_heads * head_dim]
    const float* __restrict__ k_cache,    // [max_seq * n_kv_heads * head_dim]
    const float* __restrict__ v_cache,    // [max_seq * n_kv_heads * head_dim]
    float* __restrict__ out,              // [n_heads * head_dim]
    const int* __restrict__ pos_buf,      // GPU buffer: seq_len = pos_buf[0] + 1
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,        // max sequence length (stride for cache indexing)
    float scale         // 1/sqrt(head_dim)
) {
    int seq_len = pos_buf[0] + 1;
    extern __shared__ float sdata[];

    const int h = blockIdx.x;  // query head index
    if (h >= n_heads) return;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const float* q_head = q + h * head_dim;

    // Phase 1: compute attention scores (Q @ K^T) for all positions
    // Each thread handles a subset of positions
    float* scores = sdata;  // [seq_len] — reused for softmax

    // Find max for numerical stability
    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += nthreads) {
        const float* k_t = k_cache + t * n_kv_heads * head_dim + kv_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_head[d] * k_t[d];
        }
        float s = dot * scale;
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }

    // Reduce max across threads
    // Workspace starts at sdata[seq_len] to avoid overlapping scores[0..seq_len-1]
    float* workspace = sdata + seq_len;
    workspace[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] = fmaxf(workspace[tid], workspace[tid + s]);
        __syncthreads();
    }
    float max_val = workspace[0];
    __syncthreads();

    // Exp and sum
    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += nthreads) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }

    workspace[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] += workspace[tid + s];
        __syncthreads();
    }
    float sum_val = workspace[0];
    __syncthreads();

    // Normalize
    for (int t = tid; t < seq_len; t += nthreads) {
        scores[t] /= sum_val;
    }
    __syncthreads();

    // Phase 2: weighted sum of values
    // Each thread computes a subset of output dimensions
    float* out_head = out + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            val += scores[t] * v_cache[t * n_kv_heads * head_dim + kv_h * head_dim + d];
        }
        out_head[d] = val;
    }
}
