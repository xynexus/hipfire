// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
// BF16-compute training GEMM (NT), F32 in / F32 out.
//
// Computes  Y[B, M] = X[B, K] · A[M, K]^T  with A and X read as F32 and cast to
// BF16 (round-to-nearest-even) IN REGISTER for the WMMA multiply; the product is
// accumulated and stored in F32. This is exactly the forward of the training
// linear op  Y[Mtok, Nout] = X[Mtok, K] · W[Nout, K]^T  (map A=W → M=Nout,
// X=X → B=Mtok, Y=Y).
//
// Why an f32-in cast variant rather than the bf16-in gemm_bf16_x_bf16_wmma:
// training weights + activations live in F32 and change every step, so a
// separate f32->bf16 staging pass (and its buffers) would just add bandwidth.
// Casting inline during the K-loop keeps the master data F32 (optimizer + the
// f32 backward are unchanged) and needs no staging. The fragment/output layout
// is IDENTICAL to gemm_bf16_x_bf16_wmma — only the load dtype differs — so the
// WMMA sees the same BF16 fragments a pre-converted bf16 buffer would produce,
// making this a drop-in bf16-compute replacement for the scalar gemm_f32_train
// forward. gfx1103/gfx1151 wave32, register-tiled, ZERO LDS. acc[j] at lane t
// maps to (row = 2*j + (t>>4), col = t & 15). K need not be a multiple of 16
// (the tail is zero-masked).

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short __attribute__((ext_vector_type(16))) b16x16_t;
typedef float __attribute__((ext_vector_type(8)))  float8_t;

// F32 -> BF16, round-to-nearest-even (matches gemm_bf16_bf16_wmma).
__device__ __forceinline__ unsigned short f32_to_bf16_rne(float f) {
    unsigned int b = __float_as_uint(f);
    if ((b & 0x7fffffffu) > 0x7f800000u) return (unsigned short)(b >> 16); // NaN
    unsigned int bias = 0x7fffu + ((b >> 16) & 1u);
    return (unsigned short)((b + bias) >> 16);
}

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_bf16c_train_nt(
    const float* __restrict__ A,   // [M, K] F32
    const float* __restrict__ X,   // [B, K] F32
    float*       __restrict__ Y,   // [B, M] F32
    int M, int K, int B
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
        b16x16_t a_frag;
        b16x16_t x_frag;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int kk = k + i;
            float av = (kk < K) ? w_row[kk] : 0.0f;
            float xv = (kk < K) ? x_row[kk] : 0.0f;
            a_frag[i] = (short)f32_to_bf16_rne(av);
            x_frag[i] = (short)f32_to_bf16_rne(xv);
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, x_frag, acc);
    }

    const int out_col = batch_start + lane;
    if (out_col < B) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
