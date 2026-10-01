// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
extern "C" __global__ void argmax_f32(
    const float* __restrict__ data, int* __restrict__ result, int n
) {
    extern __shared__ float s[];
    int* si = (int*)(s + blockDim.x);
    float lmax = -1e30f; int lidx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (data[i] > lmax) { lmax = data[i]; lidx = i; }
    }
    s[threadIdx.x] = lmax; si[threadIdx.x] = lidx;
    __syncthreads();
    for (int sz = blockDim.x/2; sz > 0; sz >>= 1) {
        if (threadIdx.x < sz && s[threadIdx.x + sz] > s[threadIdx.x]) {
            s[threadIdx.x] = s[threadIdx.x + sz]; si[threadIdx.x] = si[threadIdx.x + sz];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) result[0] = si[0];
}

// One argmax per pointer, all in one launch: block b reduces `ptrs[b]`. Same
// block size, loop and reduction as `argmax_f32`, so each result — ties
// included — is exactly what `argmax_f32` returns for that tensor. A batched
// decode step used to call `argmax_f32` once per session: one 256-thread block
// over the 248K-entry vocab plus a blocking readback, 64 times a step at 64
// sessions (~0.35 ms each, 5.6% of the step).
extern "C" __global__ void argmax_f32_ptrs(
    const unsigned long long* __restrict__ ptrs, int* __restrict__ result, int n
) {
    const float* __restrict__ data = (const float*)ptrs[blockIdx.x];
    extern __shared__ float s[];
    int* si = (int*)(s + blockDim.x);
    float lmax = -1e30f; int lidx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (data[i] > lmax) { lmax = data[i]; lidx = i; }
    }
    s[threadIdx.x] = lmax; si[threadIdx.x] = lidx;
    __syncthreads();
    for (int sz = blockDim.x/2; sz > 0; sz >>= 1) {
        if (threadIdx.x < sz && s[threadIdx.x + sz] > s[threadIdx.x]) {
            s[threadIdx.x] = s[threadIdx.x + sz]; si[threadIdx.x] = si[threadIdx.x + sz];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) result[blockIdx.x] = si[0];
}
