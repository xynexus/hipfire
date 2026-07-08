// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
// Phase C2 — dedicated NN / TN scaled-f16 WMMA kernels for the training backward,
// so the two backward matmuls run on the matrix cores WITHOUT a separate
// transpose_f32 pass. The transpose reformulation (Phase C1) re-streams the
// strided operand (read W + write Wt + read Wt = 3x W traffic; the transposes
// dominate the low-precision backward — see docs/plans/2026-07-08-dspark-wmma-
// training.md sec 4.1). Here the strided operand is read DIRECTLY (strided
// global loads fold the transpose into the fragment load), so W/dY/X are each
// streamed once.
//
// Scaled f16 (MXFP16-style): each operand is multiplied by a per-tensor scale
// before the f16 cast (f16's 10-bit mantissa without its 5-bit-exponent range
// trap); the f32 accumulator is unscaled by 1/(sa*sb) on store. gfx1103/gfx1151
// wave32, register-tiled, ZERO LDS (strided reads, not LDS-staged — keeps it off
// the gfx1103 HIP-719 LDS fault; an LDS-coalesced variant is a later tuning).
//
// WMMA convention (from gemm_f16c_train_nt): wmma(a,b) computes
// C[i,j] = sum_c a_row_i[c] * b_row_j[c]; each lane loads one row's 16 contract
// values; acc[j] at lane t maps to (a-row = 2j+(t>>4), b-row = t&15).

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

// dX[M,K] = sum_N dY[M,N] * W[N,K]   (contract N).
//   a = dY[M,N]  — contract N is the last axis (contiguous load).
//   b = W[N,K] read as [K,N] — lane k gathers W[n,k] over n (STRIDED by K).
__launch_bounds__(32, 8)
extern "C" __global__ void gemm_f16s_nn_train(
    const float* __restrict__ dY,   // [M, N] F32
    const float* __restrict__ W,    // [N, K] F32
    float*       __restrict__ dX,   // [M, K] F32  out
    int M, int N, int K,
    float s_dy, float s_w
) {
    const int m_start = blockIdx.y * 16;   // a-rows (M)
    const int k_start = blockIdx.x * 16;   // b-rows (K, free axis of W)
    const int tid  = threadIdx.x;
    const int lane = tid & 15;
    if (m_start >= M || k_start >= K) return;

    int my_m = m_start + lane;
    int safe_m = (my_m < M) ? my_m : (M - 1);
    int my_k = k_start + lane;
    int safe_k = (my_k < K) ? my_k : (K - 1);
    const float* dy_row = dY + (long long)safe_m * N;

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int n = 0; n < N; n += 16) {
        half16_t a_frag, b_frag;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int nn = n + i;
            float av = (nn < N) ? dy_row[nn] * s_dy : 0.0f;                    // dY[m, nn]
            float bv = (nn < N) ? W[(long long)nn * K + safe_k] * s_w : 0.0f;  // W[nn, k] strided
            a_frag[i] = (_Float16)av;
            b_frag[i] = (_Float16)bv;
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, acc);
    }

    const float inv = 1.0f / (s_dy * s_w);
    const int out_k = k_start + lane;   // b-index = k
    if (out_k < K) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_m = m_start + 2 * j + (tid >> 4);   // a-index = m
            if (out_m < M) {
                dX[(long long)out_m * K + out_k] = acc[j] * inv;
            }
        }
    }
}

// dW[N,K] = sum_M dY[M,N] * X[M,K]   (contract M).
//   a = dY[M,N] read as [N,M] — lane n gathers dY[m,n] over m (STRIDED by N).
//   b = X[M,K]  read as [K,M] — lane k gathers X[m,k] over m (STRIDED by K).
// Both operands are strided (the contract axis M is the leading axis of both),
// so neither is contiguous — this is the TN case, and both fold their transpose
// into the strided fragment load.
__launch_bounds__(32, 8)
extern "C" __global__ void gemm_f16s_tn_train(
    const float* __restrict__ dY,   // [M, N] F32
    const float* __restrict__ X,    // [M, K] F32
    float*       __restrict__ dW,   // [N, K] F32  out
    int M, int N, int K,
    float s_dy, float s_x
) {
    const int n_start = blockIdx.y * 16;   // a-rows (N)
    const int k_start = blockIdx.x * 16;   // b-rows (K)
    const int tid  = threadIdx.x;
    const int lane = tid & 15;
    if (n_start >= N || k_start >= K) return;

    int my_n = n_start + lane;
    int safe_n = (my_n < N) ? my_n : (N - 1);
    int my_k = k_start + lane;
    int safe_k = (my_k < K) ? my_k : (K - 1);

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int m = 0; m < M; m += 16) {
        half16_t a_frag, b_frag;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int mm = m + i;
            float av = (mm < M) ? dY[(long long)mm * N + safe_n] * s_dy : 0.0f; // dY[mm, n] strided
            float bv = (mm < M) ? X[(long long)mm * K + safe_k] * s_x : 0.0f;   // X[mm, k]  strided
            a_frag[i] = (_Float16)av;
            b_frag[i] = (_Float16)bv;
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, acc);
    }

    const float inv = 1.0f / (s_dy * s_x);
    const int out_k = k_start + lane;   // b-index = k
    if (out_k < K) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_n = n_start + 2 * j + (tid >> 4);   // a-index = n
            if (out_n < N) {
                dW[(long long)out_n * K + out_k] = acc[j] * inv;
            }
        }
    }
}
