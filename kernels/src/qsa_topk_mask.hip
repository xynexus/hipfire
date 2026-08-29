// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Exact top-k over QSA block scores, as a block mask.
//
// The indexer top-k kernels already in this tree are an O(N*K) SERIAL selection
// sort on decode and an O(N^2) rank-by-scan when batched, both written for
// N <= 2048. QSA at this model's native 262144 context is top-512 of up to 65536
// blocks, where neither is viable.
//
// So: binary-search the k-th largest VALUE, then partition around it. Each probe
// is an O(N) parallel count, and the whole search runs inside ONE block so it is
// one launch rather than 64 round trips to the host.
//
// Tie handling is the subtle part and must match
// `hipfire_arch_qwen4exp::qsa::topk_by_threshold` exactly. The search converges to
// a value strictly BETWEEN the k-th and (k+1)-th DISTINCT scores, never equal to
// one of them — so the tied band is `== max{s : s <= t}`, NOT `== t`. Testing
// `== t` selects nothing from the band and silently returns FEWER than k blocks,
// which reads downstream as attention seeing less than its budget allows, with no
// error anywhere.
//
// Grid: (1). Block = THREADS. Single block by design: the search is sequential in
// its probes and each probe is cheap.

#define THREADS 256

__launch_bounds__(THREADS)
extern "C" __global__ void qsa_topk_mask(
    const float* __restrict__ scores,  // [n_blocks]
    unsigned int* __restrict__ mask,   // [n_blocks], 1 = selected
    int n_blocks,
    int k
) {
    const int tid = threadIdx.x;
    __shared__ float red[THREADS];
    __shared__ int   ired[THREADS];
    __shared__ float s_lo, s_hi, s_band;
    __shared__ int   s_above;

    if (k >= n_blocks) {
        for (int i = tid; i < n_blocks; i += THREADS) mask[i] = 1u;
        return;
    }
    for (int i = tid; i < n_blocks; i += THREADS) mask[i] = 0u;
    if (k <= 0) return;

    // ── bounds ───────────────────────────────────────────────────────────────
    float lo = INFINITY, hi = -INFINITY;
    for (int i = tid; i < n_blocks; i += THREADS) {
        lo = fminf(lo, scores[i]);
        hi = fmaxf(hi, scores[i]);
    }
    red[tid] = lo; __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fminf(red[tid], red[tid + s]);
        __syncthreads();
    }
    if (tid == 0) s_lo = red[0];
    red[tid] = hi; __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    if (tid == 0) s_hi = red[0];
    __syncthreads();

    // ── binary search for the largest t with count(score > t) <= k ───────────
    for (int it = 0; it < 64; ++it) {
        const float mid = 0.5f * (s_lo + s_hi);
        if (mid <= s_lo || mid >= s_hi) break;
        int c = 0;
        for (int i = tid; i < n_blocks; i += THREADS) c += (scores[i] > mid) ? 1 : 0;
        ired[tid] = c; __syncthreads();
        for (int s = THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) ired[tid] += ired[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            if (ired[0] > k) s_lo = mid; else s_hi = mid;
        }
        __syncthreads();
    }
    const float t = s_hi;

    // ── everything strictly above the threshold ──────────────────────────────
    int c = 0;
    for (int i = tid; i < n_blocks; i += THREADS) {
        if (scores[i] > t) { mask[i] = 1u; c += 1; }
    }
    ired[tid] = c; __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) ired[tid] += ired[tid + s];
        __syncthreads();
    }
    if (tid == 0) s_above = ired[0];
    __syncthreads();
    if (s_above >= k) return;

    // ── the tied band: max{s : s <= t}, NOT `== t` ───────────────────────────
    float band = -INFINITY;
    for (int i = tid; i < n_blocks; i += THREADS) {
        if (scores[i] <= t) band = fmaxf(band, scores[i]);
    }
    red[tid] = band; __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    if (tid == 0) s_band = red[0];
    __syncthreads();

    // Fill the remaining slots from the band in INCREASING INDEX order, which is
    // the tie-break the CPU sort uses. Serial on one thread, but only over band
    // members and only until `k` is reached.
    if (tid == 0) {
        int need = k - s_above;
        for (int i = 0; i < n_blocks && need > 0; ++i) {
            if (scores[i] == s_band) { mask[i] = 1u; --need; }
        }
    }
}
