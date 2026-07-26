// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
// Sum-of-squares reduction for global-norm gradient clipping (training). Each
// call ACCUMULATES Σ x[i]² into `out[0]` (the caller pre-zeros `out` once and
// calls this over every gradient tensor, so out ends as Σ_all g²; the global
// L2 norm is sqrt of that). The cross-block combine is an atomicAdd — the
// resulting norm is not bit-reproducible to the last ulp, which is irrelevant
// for a clip threshold and does NOT touch the precision-parity/oracle paths
// (clipping only rescales grads by a scalar when the norm is large).

#include <hip/hip_runtime.h>

extern "C" __global__ void sum_sq_f32(
    const float* __restrict__ x,
    float* __restrict__ out,   // [1], pre-zeroed by the caller; accumulated
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (; i < n; i += stride) {
        float v = x[i];
        acc += v * v;
    }
    __shared__ float red[256];
    int tid = threadIdx.x;
    red[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(out, red[0]);
    }
}
