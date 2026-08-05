// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

// Mixed-precision GEMV: BF16 weight x F32 vector -> F32.
// y[m] = sum_k W_bf16[m,k] * x_f32[k]. wave32, zero LDS, F32 accumulate.
//
// The bf16 analogue of `gemv_f16_xf32_residual`, which F16 weights have had all
// along. Without it, a bf16 weight had two bad options: a batch-1 WMMA GEMM
// (4.8x slower — it is built for batched work and collapses at one row), or
// `gemv_bf16_f32`, which needs the ACTIVATION in bf16 too and so rounds it.
//
// Rounding x is a real loss and an avoidable one: the weight is genuinely bf16,
// so widening it to f32 recovers nothing, but x arrives at full precision and
// staging it down costs rel rms 3.4e-4 / worst rel 1.6e-3 at 128256 x 2048.
// Reading x as f32 directly keeps that precision at no bandwidth cost — x is
// K elements against an M x K weight read.
#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float bf16_to_f32_x(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

__launch_bounds__(32)
extern "C" __global__ void gemv_bf16_xf32(
    const unsigned short* __restrict__ W,   // [M, K] BF16
    const float*          __restrict__ x,   // [K] F32
    float*                __restrict__ y,   // [M] F32
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const unsigned short* w_row = W + (size_t)row * K;

    float sum = 0.0f;
    for (int i = tid; i < K; i += 32) {
        sum += bf16_to_f32_x(w_row[i]) * x[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
