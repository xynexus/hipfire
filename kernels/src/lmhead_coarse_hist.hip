// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// GPU top-K pass 2/3 for the two-stage lm_head: histogram the coarse scores (as
// order-preserving u32 keys) into `nbins` LINEAR bins over the value range [lo, hi].
// lo/hi are read from the TAIL of the same `stats` buffer that the min/max pass wrote
// (`stats[nbins]`, `stats[nbins+1]`) — no host round-trip between the passes. Bins are
// written in place at `stats[0..nbins)` (pre-zeroed by the caller). Binning in the
// actual key range (not the full u32 range) keeps selectivity high however the scores
// cluster. The host then scans bins top-down to the top-K threshold. grid-strided.
__device__ __forceinline__ unsigned int zmk_mono_h(float f) {
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

extern "C" __global__ void lmhead_coarse_hist(
    const float* __restrict__ coarse, int V, unsigned int* __restrict__ stats, int nbins
) {
    const unsigned int lo = stats[nbins];
    const unsigned int hi = stats[nbins + 1];
    const unsigned long long range = (unsigned long long)hi - (unsigned long long)lo + 1ull;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < V; i += gridDim.x * blockDim.x) {
        const unsigned int k = zmk_mono_h(coarse[i]);
        const unsigned long long off = (unsigned long long)(k - lo);
        int b = (int)(off * (unsigned long long)nbins / range);
        if (b < 0) b = 0;
        if (b >= nbins) b = nbins - 1;
        atomicAdd(&stats[b], 1u);
    }
}
