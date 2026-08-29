// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// QSA sparse attention, gather half.
//
// There is NO new attention kernel here, deliberately. `attention_cold_slots`
// already attends over an arbitrary compacted slot set with no causal assumption
// — which is exactly the shape QSA needs — and its layout 0 is slot-major f32.
// So sparse attention is: turn the block mask into a slot list, gather K/V into
// slots, call the kernel that already exists.
//
// Two kernels:
//   qsa_select_indices — block mask -> ASCENDING cache-position list + count
//   qsa_gather_kv      — gather those positions into [n_kv, n_slots, head_dim]
//
// Why ascending and not an atomicAdd compaction: softmax is permutation
// invariant, so attention would be *correct* either way — but the flash
// accumulation over slots is a float sum, so a run-varying permutation makes the
// output run-varying too. Ordering costs one scan and buys determinism.

#define THREADS 256

// Block mask -> the cache positions attention should see, ascending.
//
// The ragged tail past the last complete block is ALWAYS visible: it has no block
// to be selected, so it cannot be derived from `block_mask` at all and is
// appended unconditionally. Matches `hipfire_arch_qwen4exp::qsa::select`.
//
// Grid: (1). Single block — the scan carries across chunks.
__launch_bounds__(THREADS)
extern "C" __global__ void qsa_select_indices(
    const unsigned int* __restrict__ block_mask, // [n_blocks], 1 = selected
    const int* __restrict__ visible,             // [n_visible] cache positions
    int* __restrict__ out_idx,                   // [n_visible] capacity
    int* __restrict__ out_count,                 // [1]
    int n_visible, int n_blocks, int compress_ratio
) {
    const int tid = threadIdx.x;
    __shared__ int scan[THREADS];
    __shared__ int carry;
    if (tid == 0) carry = 0;
    __syncthreads();

    for (int base = 0; base < n_blocks; base += THREADS) {
        const int b = base + tid;
        const int sel = (b < n_blocks && block_mask[b] != 0u) ? 1 : 0;
        scan[tid] = sel;
        __syncthreads();
        for (int off = 1; off < THREADS; off <<= 1) {   // inclusive scan
            const int v = (tid >= off) ? scan[tid - off] : 0;
            __syncthreads();
            scan[tid] += v;
            __syncthreads();
        }
        if (sel) {
            const int dst = (carry + scan[tid] - 1) * compress_ratio;
            const int src = b * compress_ratio;
            for (int j = 0; j < compress_ratio; ++j) out_idx[dst + j] = visible[src + j];
        }
        __syncthreads();
        if (tid == THREADS - 1) carry += scan[THREADS - 1];
        __syncthreads();
    }

    const int tail_start = n_blocks * compress_ratio;
    const int head = carry * compress_ratio;
    for (int j = tid; j < n_visible - tail_start; j += THREADS) {
        out_idx[head + j] = visible[tail_start + j];
    }
    if (tid == 0) *out_count = head + (n_visible - tail_start);
}

// Gather selected cache positions into slot-major f32, the layout
// `attention_cold_slots` reads with k_layout/v_layout = 0.
//
// Grid: (n_slots, n_kv_heads). Block = head_dim rounded to a wave.
extern "C" __global__ void qsa_gather_kv(
    const float* __restrict__ cache,   // [n_kv_heads, cache_stride, head_dim]
    const int* __restrict__ slot_idx,  // [n_slots] cache positions
    float* __restrict__ out,           // [n_kv_heads, n_slots, head_dim]
    int n_kv_heads, int n_slots, int head_dim, int cache_stride
) {
    const int s = blockIdx.x;
    const int kv = blockIdx.y;
    if (s >= n_slots || kv >= n_kv_heads) return;
    const int pos = slot_idx[s];
    const long long src = ((long long)kv * cache_stride + pos) * head_dim;
    const long long dst = ((long long)kv * n_slots + s) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) out[dst + d] = cache[src + d];
}
