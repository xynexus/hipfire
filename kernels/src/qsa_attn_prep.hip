// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#define BLOCK 256

// Per-head RMSNorm with a weight SHARED across heads.
//
// Distinct from `hc_grouped_rmsnorm`, whose weight is full-width `[n_groups,
// group]`. The q/k norms here carry one `[head_dim]` vector reused by every head,
// so indexing it per-group would read past the end of the tensor.
//
// Same `1 + w` convention as the rest of this family's `Qwen4ExpTextRMSNorm`.
__launch_bounds__(BLOCK)
extern "C" __global__ void rms_norm_heads_shared_w(
    const float* __restrict__ x,       // [n_heads, head_dim]
    const float* __restrict__ weight,  // [head_dim], shared
    float*       __restrict__ out,     // [n_heads, head_dim]
    int   head_dim,
    int   n_heads,
    float eps
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const long long base = (long long)h * head_dim;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = x[base + i];
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
        out[base + i] = x[base + i] * scale * (1.0f + weight[i]);
    }
}

// Split the doubled `q_proj` output into the query and its output gate.
//
// The layout is PER HEAD — `[query(head_dim) | gate(head_dim)]` repeated for each
// head — not two contiguous blocks of `n_heads * head_dim`. Reading it the other
// way is shape-compatible and silently mixes head 0's gate into head 1's query.
extern "C" __global__ void qsa_split_query_gate(
    const float* __restrict__ qg,     // [n_heads, 2 * head_dim]
    float*       __restrict__ query,  // [n_heads, head_dim]
    float*       __restrict__ gate,   // [n_heads, head_dim]
    int n_heads,
    int head_dim
) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    const int total = n_heads * head_dim;
    if (i >= total) return;
    const int h = i / head_dim;
    const int d = i - h * head_dim;
    const long long src = (long long)h * 2 * head_dim + d;
    query[i] = qg[src];
    gate[i] = qg[src + head_dim];
}

// `out = ctx * sigmoid(gate)` — the attention output gate, applied before o_proj.
extern "C" __global__ void qsa_apply_output_gate(
    const float* __restrict__ ctx,
    const float* __restrict__ gate,
    float*       __restrict__ out,
    int n
) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    out[i] = ctx[i] / (1.0f + __expf(-gate[i]));
}

// Write one token's K (or V) into a slot-major cache at `pos`.
//
// The cache is `[n_kv, max_seq, head_dim]`, so a token's heads are NOT contiguous
// — they are `max_seq * head_dim` apart. A flat copy would write head 0's data
// across the first heads' slots and silently corrupt the cache.
extern "C" __global__ void qsa_cache_write(
    const float* __restrict__ src,    // [n_kv, head_dim]
    float*       __restrict__ cache,  // [n_kv, max_seq, head_dim]
    int pos,
    int n_kv,
    int head_dim,
    int max_seq
) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n_kv * head_dim) return;
    const int kv = i / head_dim;
    const int d = i - kv * head_dim;
    cache[((long long)kv * max_seq + pos) * head_dim + d] = src[i];
}

// Accumulate one expert's contribution into the MoE output.
//
// `accumulate == 0` OVERWRITES, so the first routed expert seeds the buffer and no
// separate zero-fill pass is needed. Every later slot adds.
extern "C" __global__ void moe_accum_scaled(
    float*       __restrict__ acc,
    const float* __restrict__ src,
    const float* __restrict__ weights,  // [k] routing weights, on device
    int slot,
    int n,
    int accumulate
) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    const float v = weights[slot] * src[i];
    acc[i] = accumulate ? (acc[i] + v) : v;
}

// `out = a * sigmoid(g)` where `g` is a single scalar read from device memory —
// the always-on shared expert's gate.
extern "C" __global__ void moe_shared_gate(
    const float* __restrict__ a,
    const float* __restrict__ g,   // [1]
    float*       __restrict__ out,
    int n
) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] / (1.0f + __expf(-g[0]));
}
