// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
// bf16-OUTPUT variant of gemm_oq4_grouped_wmma (see that file for the algorithm).
// Identical iu4 WMMA compute + f32 accumulation; the ONLY difference is the
// epilogue writes each result as bf16 (top 16 bits of the f32, truncated) into a
// uint16 Y buffer — halving the output-write bandwidth. This is the "output-memory
// lever" probe: at these prefill shapes the f32 output write is 72-77% of memory
// traffic, so halving it should move the (roofline-)memory-bound iu4 GEMM toward
// compute-bound. bf16 (not f16) keeps f32's exponent range for activation outliers.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t __attribute__((ext_vector_type(2))) int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;
typedef float   __attribute__((ext_vector_type(8))) float8_t;

static __device__ __forceinline__ int32x2_t load_u4x16(const uint8_t* p) {
    return (int32x2_t){(int32_t)*(const uint32_t*)(p + 0),
                       (int32_t)*(const uint32_t*)(p + 4)};
}

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_oq4_grouped_wmma_bf16out(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ Ws,
    const uint8_t* __restrict__ X,
    const float*   __restrict__ Xs,
    uint16_t*      __restrict__ Y,   // [B, M] bf16
    int M, int K, int B, int group
) {
    const int tid  = threadIdx.x;
    const int lane = tid & 15;
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    if (row_start >= M || batch_start >= B) return;

    const int a_row = row_start + lane;
    const int x_row = batch_start + lane;
    const int safe_a = (a_row < M) ? a_row : (M - 1);
    const int safe_x = (x_row < B) ? x_row : (B - 1);
    const int stride = K / 2;
    const int n_groups = K / group;
    const int tiles_per_group = group / 16;

    const uint8_t* w_row = W + (long long)safe_a * stride;
    const uint8_t* x_prow = X + (long long)safe_x * stride;

    float8_t facc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int g = 0; g < n_groups; ++g) {
        int32x8_t iacc = {0, 0, 0, 0, 0, 0, 0, 0};
        const int g_byte = g * (group / 2);
        for (int kt = 0; kt < tiles_per_group; ++kt) {
            const int32x2_t a = load_u4x16(w_row + g_byte + kt * 8);
            const int32x2_t b = load_u4x16(x_prow + g_byte + kt * 8);
            iacc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, iacc, false);
        }
        const int b_col = batch_start + lane;
        const float sx = (b_col < B) ? Xs[(long long)b_col * n_groups + g] : 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int m_row = row_start + 2 * j + (tid >> 4);
            const float sw = (m_row < M) ? Ws[(long long)m_row * n_groups + g] : 0.f;
            facc[j] += (float)iacc[j] * sw * sx;
        }
    }

    const int out_col = batch_start + lane;
    if (out_col < B) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                // Truncate f32 -> bf16 (top 16 bits). Exact rounding is irrelevant
                // for this bandwidth probe; bf16 keeps f32 range.
                Y[(long long)out_col * M + out_row] =
                    (uint16_t)(__float_as_uint(facc[j]) >> 16);
            }
        }
    }
}
