// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Per-row symmetric Q4 GEMV (coarse lm_head scorer): y[m] = scale[m] * sum_k q4[m,k]*x[k].
// Weight layout: `w4` = [M, K/2] bytes, 2 signed 4-bit (twos-complement, range [-7,7])
// per byte — dim 2i in the low nibble, dim 2i+1 in the high nibble. `scale` = [M] f32.
// This is the CANDIDATE-SELECTION stage of the two-stage lm_head: it only has to keep
// the true winner inside the top-K shortlist, so per-row Q4 (one scale / 2048-dim row)
// is deliberately the aggressive/worst-case coarse quant. gfx1103 wave32, F32 accumulate.
//
// Vectorized: each lane consumes a uint32 (8 nibbles) per load instead of a byte, so the
// nibble-unpack ALU no longer throttles the load-issue rate (byte-at-a-time sat at
// ~110 GB/s vs the bf16 gemv's ~170; the unpack ops were starving the weight stream).
// Row stride 4-byte aligned (K/2 % 4 == 0 for K a multiple of 8). grid=[M], wave32.
__device__ __forceinline__ int zq4_sx(int nib) { return (nib ^ 0x8) - 0x8; } // sign-extend int4

__launch_bounds__(32, 16) extern "C" __global__ void gemv_q4sym_f32(
    const unsigned char* __restrict__ w4,
    const float* __restrict__ scale,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int lane = threadIdx.x; // wave32
    const int n32 = K / 8;        // uint32 words per row (8 nibbles each)
    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        const unsigned int* __restrict__ w32 =
            reinterpret_cast<const unsigned int*>(w4 + (size_t)row * (K / 2));
        float sum = 0.0f;
        for (int i = lane; i < n32; i += 32) {
            const unsigned int p = w32[i];
            const float* __restrict__ xr = x + (size_t)8 * i;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                sum += (float)zq4_sx((int)((p >> (4 * j)) & 0xF)) * xr[j];
            }
        }
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_down(sum, off);
        if (lane == 0) y[row] = scale[row] * sum;
    }
}
