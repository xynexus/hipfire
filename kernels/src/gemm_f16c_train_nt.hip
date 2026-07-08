// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
// F16-compute training GEMM (NT), F32 in / F32 out. Identical structure to
// gemm_bf16c_train_nt but casts to _Float16 (10 mantissa bits) instead of BF16
// (7), for the forward matmuls whose rounding feeds the loss/gradient — the
// drafter's rmsnorm'd activations sit well inside F16's range, so the extra
// mantissa recovers convergence quality the BF16 cast gives up, at the same
// WMMA throughput.
//
// Computes  Y[B, M] = X[B, K] · A[M, K]^T  with A and X read as F32 and cast to
// F16 in-register for a WMMA f32-accumulate multiply; output is F32. Map onto
// the training linear forward  Y[Mtok, Nout] = X[Mtok, K] · W[Nout, K]^T  as
// A=W (M=Nout), X=X (B=Mtok). Fragment/output layout matches
// gemm_f16_x_f16_wmma. gfx1103/gfx1151 wave32, register-tiled, ZERO LDS.
// acc[j] at lane t maps to (row = 2*j + (t>>4), col = t & 15). K need not be a
// multiple of 16 (tail zero-masked).

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_f16c_train_nt(
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
        half16_t a_frag;
        half16_t x_frag;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int kk = k + i;
            float av = (kk < K) ? w_row[kk] : 0.0f;
            float xv = (kk < K) ? x_row[kk] : 0.0f;
            a_frag[i] = (_Float16)av;
            x_frag[i] = (_Float16)xv;
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, x_frag, acc);
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
