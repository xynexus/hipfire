// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// GPU top-K pass 3/3 for the two-stage lm_head: stream-compact the row indices whose
// coarse key >= `tau` (the host-computed top-K threshold) into `idx`, bumping `counter`
// atomically. The result is a SUPERSET of the exact top-K (the boundary bin is included
// wholesale) — harmless, since the fine bf16 pass rescores every selected row exactly.
// `cap` bounds `idx`; the host sizes it from the histogram tail count + slack so it never
// truncates in practice. Order is arbitrary (the fine gather is order-independent).
__device__ __forceinline__ unsigned int zmk_mono_c(float f) {
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

extern "C" __global__ void lmhead_coarse_compact(
    const float* __restrict__ coarse, int V, unsigned int tau,
    unsigned int* __restrict__ idx, unsigned int* __restrict__ counter, int cap
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < V; i += gridDim.x * blockDim.x) {
        if (zmk_mono_c(coarse[i]) >= tau) {
            unsigned int p = atomicAdd(counter, 1u);
            if (p < (unsigned int)cap) idx[p] = (unsigned int)i;
        }
    }
}
