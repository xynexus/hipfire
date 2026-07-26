// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void transpose_f32(
    const float* __restrict__ in_data,
    float* __restrict__ out_data,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int r = idx / cols;
    int c = idx % cols;
    out_data[c * rows + r] = in_data[idx];
}

// Transpose + fused per-tensor abs-max of the input, in ONE pass over the data.
// Since transpose only permutes elements, max|in| == max|out|, so the caller
// gets the transposed operand's abs-max for free — no separate reduction pass
// (a big saving for the lm-head weight: ~2.7 GB otherwise re-streamed). `amax`
// is a single u32, pre-zeroed; the atomicMax on the abs value's bit pattern is
// order-independent, so it does not perturb determinism. Read back + reinterpret
// as f32 to get max|in| (used to pick the f16s scale).
extern "C" __global__ void transpose_f32_amax(
    const float* __restrict__ in_data,
    float* __restrict__ out_data,
    int rows, int cols,
    unsigned int* __restrict__ amax   // [1], pre-zeroed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    float a = 0.0f;
    if (idx < total) {
        int r = idx / cols;
        int c = idx % cols;
        float v = in_data[idx];
        out_data[c * rows + r] = v;
        a = fabsf(v);
    }
    __shared__ float red[256];
    int tid = threadIdx.x;
    red[tid] = a;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        atomicMax(amax, __float_as_uint(red[0]));
    }
}
