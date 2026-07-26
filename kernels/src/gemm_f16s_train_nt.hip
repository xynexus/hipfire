// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
// SCALED F16-compute training GEMM (NT), F32 in / F32 out — f16's 10-bit
// mantissa WITHOUT its 5-bit-exponent range trap. Each operand is multiplied by
// a per-tensor scale (chosen so max|operand|·scale sits near the top of f16
// range, ~16384) BEFORE the f16 cast, so values that would overflow un-scoped
// f16 (activations/gradients > 65504) land inside range; the f32 accumulator is
// unscaled by 1/(sa·sx) on store. This is the software realization of MXFP16-
// style microscaling on RDNA3.5 (no hardware scaled-MMA): a shared exponent per
// tensor, applied around the existing f16 WMMA path.
//
// Same layout/contract as gemm_f16c_train_nt (see that file). The scales are
// plain f32 multipliers; callers pass powers of two so the scale is exact and
// bit-reversible (an ldexp), computed from `abs_max_f32` below.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_f16s_train_nt(
    const float* __restrict__ A,   // [M, K] F32
    const float* __restrict__ X,   // [B, K] F32
    float*       __restrict__ Y,   // [B, M] F32
    int M, int K, int B,
    float sa,                      // scale for A (=w): brings into f16 range
    float sx                       // scale for X
) {
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid  = threadIdx.x;
    const int lane = tid & 15;

    if (row_start >= M || batch_start >= B) return;

    int my_m = row_start + lane;
    int my_b = batch_start + lane;
    int safe_m = (my_m < M) ? my_m : (M - 1);
    int safe_b = (my_b < B) ? my_b : (B - 1);

    const float* w_row = A + (long long)safe_m * K;
    const float* x_row = X + (long long)safe_b * K;

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int k = 0; k < K; k += 16) {
        half16_t a_frag;
        half16_t x_frag;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int kk = k + i;
            float av = (kk < K) ? w_row[kk] * sa : 0.0f;
            float xv = (kk < K) ? x_row[kk] * sx : 0.0f;
            a_frag[i] = (_Float16)av;
            x_frag[i] = (_Float16)xv;
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, x_frag, acc);
    }

    const float inv = 1.0f / (sa * sx);
    const int out_col = batch_start + lane;
    if (out_col < B) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[j] * inv;
            }
        }
    }
}

// Deterministic per-tensor abs-max: out (a single u32, pre-zeroed) receives
// max_i |x[i]| via an atomicMax on the IEEE bit pattern of the absolute value.
// For non-negative floats the uint order matches float order, so this is a true
// maximum; max is associative/commutative/exact, so the result is bit-identical
// regardless of block/atomic ordering (unlike a sum) — it does NOT reintroduce
// nondeterminism. Read back and reinterpret as f32 to get the scale.
extern "C" __global__ void abs_max_f32(
    const float* __restrict__ x,
    unsigned int* __restrict__ out,   // [1], pre-zeroed
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float local = 0.0f;
    for (; i < n; i += stride) {
        local = fmaxf(local, fabsf(x[i]));
    }
    __shared__ float red[256];
    int tid = threadIdx.x;
    red[tid] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        atomicMax(out, __float_as_uint(red[0]));
    }
}
