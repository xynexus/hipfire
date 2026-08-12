// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Per-row symmetric Q2 GEMV (coarse lm_head scorer): y[m] = scale[m] * sum_k q2[m,k]*x[k].
// Weight layout: `w2` = [M, K/4] bytes, 4 signed 2-bit (twos-complement: 00→0, 01→1,
// 10→-2, 11→-1) per byte — dim 4i+j in bits [2j, 2j+1]. `scale` = [M] f32. This is the
// most aggressive coarse quant, viable only with row-wise L2 normalisation (the norm is
// separated + exact; the 2-bit direction is 3σ-clipped). gfx1103 wave32, F32 accumulate.
//
// Vectorized: each lane consumes a uint32 (16 2-bit values) per load, so the unpack ALU
// stops throttling the weight stream (byte-at-a-time sat ~110 GB/s; see gemv_q4sym_f32).
// Row stride 4-byte aligned (K/4 % 4 == 0 for K a multiple of 16). grid=[M], wave32.
__device__ __forceinline__ int zq2_sx(int n) { return (n << 30) >> 30; } // sign-extend int2

__launch_bounds__(32, 16) extern "C" __global__ void gemv_q2sym_f32(
    const unsigned char* __restrict__ w2,
    const float* __restrict__ scale,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int lane = threadIdx.x; // wave32
    const int n32 = K / 16;       // uint32 words per row (16 2-bit values each)
    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        const unsigned int* __restrict__ w32 =
            reinterpret_cast<const unsigned int*>(w2 + (size_t)row * (K / 4));
        float sum = 0.0f;
        for (int i = lane; i < n32; i += 32) {
            const unsigned int p = w32[i];
            const float* __restrict__ xr = x + (size_t)16 * i;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                sum += (float)zq2_sx((int)((p >> (2 * j)) & 0x3)) * xr[j];
            }
        }
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_down(sum, off);
        if (lane == 0) y[row] = scale[row] * sum;
    }
}
