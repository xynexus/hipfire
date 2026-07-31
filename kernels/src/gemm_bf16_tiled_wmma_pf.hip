// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
// Software-pipelined (prefetched) variants of the production register-tiled bf16
// DiT GEMM (gemm_bf16_tiled_wmma). Same math, same WMMA order, same f32
// accumulation ⇒ BIT-EXACT to gemm_bf16_tiled_wmma_<MB>x<NB>.
//
// WHY: the free-ALU probe (plan §12c) showed the production 4x4 kernel gets ~15%
// FASTER when independent VALU work is injected between its WMMAs — i.e. it
// stalls on its dependent global fragment loads and any independent work fills
// those slots. The principled fix is to prefetch the NEXT K-step's fragments into
// registers before issuing the current step's WMMAs, so the load latency overlaps
// the matrix-core work instead of stalling it.
//
// The constraint is VGPRs: acc alone is TILE_MB*TILE_NB*8 (128 for 4x4), and each
// b16x16_t fragment is 8 VGPRs, so a full double-buffer of (MB+NB) fragments adds
// (MB+NB)*8 more. 4x4 + both-prefetch = 128+64+64 = 256 → spills. Hence the sweep:
// prefetch X only (cheap) at 4x4, and both at the smaller tiles that have room.
//
//   A : [M, K] BF16 row-major   X : [B, K] BF16 row-major   Y : [B, M] F32
//   grid = (ceil(M/(16*MB)), ceil(B/(16*NB))), block = 32 (one wave).
//   Requires K % 32 == 0 (two 16-wide K-steps per pipelined iteration).

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short __attribute__((ext_vector_type(16))) b16x16_t;
typedef float __attribute__((ext_vector_type(8)))  float8_t;

// PF_A / PF_X: prefetch the A / X fragments one K-step ahead.
#define DEFINE_PF_GEMM(NAME, TILE_MB, TILE_NB, MINWAVES, PF_A, PF_X)            \
__launch_bounds__(32, MINWAVES)                                                 \
extern "C" __global__ void NAME(                                                \
    const unsigned short* __restrict__ A,                                       \
    const unsigned short* __restrict__ X,                                       \
    float*               __restrict__ Y,                                        \
    int M, int K, int B                                                         \
) {                                                                             \
    const int row_start   = blockIdx.x * (16 * TILE_MB);                        \
    const int batch_start = blockIdx.y * (16 * TILE_NB);                        \
    const int tid  = threadIdx.x;                                               \
    const int lane = tid & 15;                                                  \
    if (row_start >= M || batch_start >= B) return;                             \
    const unsigned short* w_row[TILE_MB];                                       \
    const unsigned short* x_row[TILE_NB];                                       \
    _Pragma("unroll")                                                           \
    for (int mi = 0; mi < TILE_MB; ++mi) {                                      \
        int m = row_start + mi * 16 + lane;                                     \
        w_row[mi] = A + (long long)((m < M) ? m : (M - 1)) * K;                 \
    }                                                                           \
    _Pragma("unroll")                                                           \
    for (int ni = 0; ni < TILE_NB; ++ni) {                                      \
        int b = batch_start + ni * 16 + lane;                                   \
        x_row[ni] = X + (long long)((b < B) ? b : (B - 1)) * K;                 \
    }                                                                           \
    float8_t acc[TILE_MB][TILE_NB];                                             \
    _Pragma("unroll")                                                           \
    for (int mi = 0; mi < TILE_MB; ++mi)                                        \
        _Pragma("unroll")                                                       \
        for (int ni = 0; ni < TILE_NB; ++ni)                                    \
            acc[mi][ni] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};             \
    /* Fragment registers. When prefetching, `nx` holds the next K-step. */     \
    b16x16_t af[TILE_MB], xf[TILE_NB];                                          \
    b16x16_t af_nx[PF_A ? TILE_MB : 1], xf_nx[PF_X ? TILE_NB : 1];              \
    _Pragma("unroll")                                                           \
    for (int mi = 0; mi < TILE_MB; ++mi) af[mi] = *(const b16x16_t*)(w_row[mi]); \
    _Pragma("unroll")                                                           \
    for (int ni = 0; ni < TILE_NB; ++ni) xf[ni] = *(const b16x16_t*)(x_row[ni]); \
    for (int k = 0; k < K; k += 16) {                                          \
        const int knext = k + 16;                                               \
        /* Issue the NEXT step's loads BEFORE this step's WMMAs so the memory    \
           latency overlaps the matrix-core work (no wait between). */          \
        if (knext < K) {                                                        \
            if (PF_A) {                                                         \
                _Pragma("unroll")                                               \
                for (int mi = 0; mi < TILE_MB; ++mi)                            \
                    af_nx[PF_A ? mi : 0] =                                      \
                        *(const b16x16_t*)(w_row[mi] + knext);                  \
            }                                                                   \
            if (PF_X) {                                                         \
                _Pragma("unroll")                                               \
                for (int ni = 0; ni < TILE_NB; ++ni)                            \
                    xf_nx[PF_X ? ni : 0] =                                      \
                        *(const b16x16_t*)(x_row[ni] + knext);                  \
            }                                                                   \
        }                                                                       \
        _Pragma("unroll")                                                       \
        for (int mi = 0; mi < TILE_MB; ++mi)                                    \
            _Pragma("unroll")                                                   \
            for (int ni = 0; ni < TILE_NB; ++ni)                                \
                acc[mi][ni] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(      \
                    af[mi], xf[ni], acc[mi][ni]);                               \
        if (knext < K) {                                                        \
            _Pragma("unroll")                                                   \
            for (int mi = 0; mi < TILE_MB; ++mi)                                \
                af[mi] = PF_A ? af_nx[PF_A ? mi : 0]                            \
                              : *(const b16x16_t*)(w_row[mi] + knext);          \
            _Pragma("unroll")                                                   \
            for (int ni = 0; ni < TILE_NB; ++ni)                                \
                xf[ni] = PF_X ? xf_nx[PF_X ? ni : 0]                            \
                              : *(const b16x16_t*)(x_row[ni] + knext);          \
        }                                                                       \
    }                                                                           \
    _Pragma("unroll")                                                           \
    for (int ni = 0; ni < TILE_NB; ++ni) {                                      \
        const int out_col = batch_start + ni * 16 + lane;                       \
        if (out_col >= B) continue;                                             \
        _Pragma("unroll")                                                       \
        for (int mi = 0; mi < TILE_MB; ++mi) {                                  \
            _Pragma("unroll")                                                   \
            for (int j = 0; j < 8; ++j) {                                       \
                const int out_row = row_start + mi * 16 + 2 * j + (tid >> 4);   \
                if (out_row < M) {                                              \
                    Y[(long long)out_col * M + out_row] = acc[mi][ni][j];       \
                }                                                               \
            }                                                                   \
        }                                                                       \
    }                                                                           \
}

// acc VGPRs = MB*NB*8; each prefetched fragment set adds MB*8 / NB*8.
DEFINE_PF_GEMM(gemm_bf16_pf_4x4_x,    4, 4, 2, 0, 1)  // 128 + 64 + 32 = 224
DEFINE_PF_GEMM(gemm_bf16_pf_4x4_a,    4, 4, 2, 1, 0)  // 128 + 64 + 32 = 224
DEFINE_PF_GEMM(gemm_bf16_pf_4x4_both, 4, 4, 2, 1, 1)  // 256 — spill probe
DEFINE_PF_GEMM(gemm_bf16_pf_4x2_both, 4, 2, 2, 1, 1)  //  64 + 48 + 48 = 160
DEFINE_PF_GEMM(gemm_bf16_pf_2x2_both, 2, 2, 4, 1, 1)  //  32 + 32 + 32 =  96
