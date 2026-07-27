// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// GPU top-K pass 1/3 for the two-stage lm_head: global min/max of the coarse scores as
// order-preserving u32 keys, written into the TAIL of the shared `stats` buffer
// (`stats[nbins]` = min key, init 0xFFFFFFFF; `stats[nbins+1]` = max key, init 0). The
// histogram pass then reads lo/hi straight from `stats` on the device — folding min/max
// and the histogram into one buffer means only a single host download for the whole
// top-K, instead of one round-trip per pass. LDS block-reduce → one atomic PER BLOCK
// (not per row), so there is no atomic-contention hit. block=256, grid-strided.
__device__ __forceinline__ unsigned int zmk_mono(float f) {
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

extern "C" __global__ void lmhead_coarse_minmax(
    const float* __restrict__ coarse, int V, unsigned int* __restrict__ stats, int nbins
) {
    unsigned int lmin = 0xFFFFFFFFu, lmax = 0u;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < V; i += gridDim.x * blockDim.x) {
        unsigned int k = zmk_mono(coarse[i]);
        lmin = min(lmin, k);
        lmax = max(lmax, k);
    }
    __shared__ unsigned int smin[256];
    __shared__ unsigned int smax[256];
    const int t = threadIdx.x;
    smin[t] = lmin;
    smax[t] = lmax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            smin[t] = min(smin[t], smin[t + s]);
            smax[t] = max(smax[t], smax[t + s]);
        }
        __syncthreads();
    }
    if (t == 0) {
        atomicMin(&stats[nbins], smin[0]);
        atomicMax(&stats[nbins + 1], smax[0]);
    }
}
