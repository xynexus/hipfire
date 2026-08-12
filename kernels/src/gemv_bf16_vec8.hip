// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Plain-BF16 GEMV with the SAME access shape as gemv_bf16l3: 8 consecutive
// elements per lane, fetched as one aligned 16-byte load, walked in 256-element
// blocks. Identical maths to gemv_bf16_bf16 (which loads at stride 32 instead).
//
// This exists purely as the honest baseline for the BF16L3 comparison. Against
// gemv_bf16_bf16 it isolates the coalescing effect; against gemv_bf16l3 it
// isolates the compression effect, because everything except the weight
// encoding is held constant.
//
// wave32, zero LDS, F32 accumulate, BF16 round on store. Requires K % 256 == 0.
#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
__device__ __forceinline__ unsigned short f32_to_bf16_rne(float f) {
    unsigned int b = __float_as_uint(f);
    if ((b & 0x7fffffffu) > 0x7f800000u) return (unsigned short)(b >> 16); // NaN
    unsigned int bias = 0x7fffu + ((b >> 16) & 1u);
    return (unsigned short)((b + bias) >> 16);
}

__launch_bounds__(32)
extern "C" __global__ void gemv_bf16_vec8(
    const unsigned short* __restrict__ W,   // [M, K] BF16
    const unsigned short* __restrict__ x,   // [K] BF16
    unsigned short*       __restrict__ y,   // [M] BF16
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const unsigned short* __restrict__ w_row = W + (size_t)row * (size_t)K;

    const int blocks_per_row = K / 256;
    float sum = 0.0f;

    for (int j = 0; j < blocks_per_row; ++j) {
        const size_t base = (size_t)j * 256 + 8 * tid;
        const uint4 wv = *(const uint4* __restrict__)(w_row + base);
        const uint4 xv = *(const uint4* __restrict__)(x + base);
        const unsigned int wb[4] = { wv.x, wv.y, wv.z, wv.w };
        const unsigned int xb[4] = { xv.x, xv.y, xv.z, xv.w };
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const unsigned short w =
                (unsigned short)((wb[s >> 1] >> (16 * (s & 1))) & 0xffffu);
            const unsigned short xs =
                (unsigned short)((xb[s >> 1] >> (16 * (s & 1))) & 0xffffu);
            sum += bf16_to_f32(w) * bf16_to_f32(xs);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset);
    if (tid == 0) y[row] = f32_to_bf16_rne(sum);
}
