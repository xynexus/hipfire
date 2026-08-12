// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
// DIAGNOSTIC-ONLY compute-headroom probe for the diffusion-DiT bf16 GEMM.
// NOT routed into any model path (mirrors bench_iu_wmma_gfx1151's contract).
//
// WHY: the DiT bf16 GEMM runs at ~10% of the bf16-WMMA peak and is COMPUTE-bound
// (AI >> roofline ridge), so the SIMD is idle most cycles waiting on
// memory/dependency latency — it is not issue-saturated. If that idle issue
// bandwidth can absorb unrelated VALU work "for free", then a codebook /
// trellis (QTIP, LO-BCQ) weight DECODE — and any correction branch (low-rank
// add, error feedback) — can hide in the GEMM's shadow, which is precisely the
// blocker that parks Lever D ("gated on the hardest kernel question": a fast
// codebook-indexed GEMM). This kernel measures that budget directly.
//
// HOW: an exact copy of the production register-tiled 4x4 kernel
// (gemm_bf16_tiled_wmma_4x4) plus NALU independent side FMAs per K-step. Sweep
// NALU and find the knee where wall-time starts rising: everything below the
// knee is free compute. The side chain is float (FP non-associativity blocks
// strength-reduction/hoisting) and is loop-carried across K (blocks LICM); a
// runtime `sink` arg (always passed 0) blocks dead-code elimination. The sweep
// is taken high enough that time DOES eventually rise, which self-proves the
// work is real and not being optimized away.
//
// BUDGET MATH: per K-step each lane materializes TILE_MB * 16 = 64 bf16 weight
// elements (4 A-fragments), so an affordable decode cost is
//     ops per weight element = NALU_free / 64.
// (QTIP trellis decode ~5-10 ops/elem; LO-BCQ LUT ~2-4 ops + a gather.)
//
//   A : [M, K] BF16 row-major   X : [B, K] BF16 row-major   Y : [B, M] F32
//   grid = (ceil(M/64), ceil(B/64)), block = 32 (one wave).

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short __attribute__((ext_vector_type(16))) b16x16_t;
typedef float __attribute__((ext_vector_type(8)))  float8_t;

// 8 independent chains x 4 unrolled = 32 FMAs per inner iteration, so NALU is a
// multiple of 32 and loop overhead is ~6% of the added ops. `unroll 1` on the
// outer loop keeps the K-loop body I-cache-resident at large NALU.
#define SIDE_ALU(NALU)                                                          \
    _Pragma("unroll 1")                                                         \
    for (int q = 0; q < (NALU) / 32; ++q) {                                     \
        _Pragma("unroll")                                                       \
        for (int u = 0; u < 4; ++u) {                                           \
            d0 = __builtin_fmaf(d0, 1.0000001f, 1e-8f);                         \
            d1 = __builtin_fmaf(d1, 1.0000001f, 1e-8f);                         \
            d2 = __builtin_fmaf(d2, 1.0000001f, 1e-8f);                         \
            d3 = __builtin_fmaf(d3, 1.0000001f, 1e-8f);                         \
            d4 = __builtin_fmaf(d4, 1.0000001f, 1e-8f);                         \
            d5 = __builtin_fmaf(d5, 1.0000001f, 1e-8f);                         \
            d6 = __builtin_fmaf(d6, 1.0000001f, 1e-8f);                         \
            d7 = __builtin_fmaf(d7, 1.0000001f, 1e-8f);                         \
        }                                                                       \
    }

#define DEFINE_ALU_GEMM(NAME, TILE_MB, TILE_NB, MINWAVES, NALU)                 \
__launch_bounds__(32, MINWAVES)                                                 \
extern "C" __global__ void NAME(                                                \
    const unsigned short* __restrict__ A,                                       \
    const unsigned short* __restrict__ X,                                       \
    float*               __restrict__ Y,                                        \
    int M, int K, int B, int sink                                               \
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
    float d0 = (float)tid + 1.0f, d1 = d0 + 0.125f, d2 = d0 + 0.25f;            \
    float d3 = d0 + 0.375f, d4 = d0 + 0.5f, d5 = d0 + 0.625f;                   \
    float d6 = d0 + 0.75f, d7 = d0 + 0.875f;                                    \
    for (int k = 0; k < K; k += 16) {                                          \
        b16x16_t a_frag[TILE_MB];                                               \
        b16x16_t x_frag[TILE_NB];                                               \
        _Pragma("unroll")                                                       \
        for (int mi = 0; mi < TILE_MB; ++mi)                                    \
            a_frag[mi] = *(const b16x16_t*)(w_row[mi] + k);                     \
        _Pragma("unroll")                                                       \
        for (int ni = 0; ni < TILE_NB; ++ni)                                    \
            x_frag[ni] = *(const b16x16_t*)(x_row[ni] + k);                     \
        _Pragma("unroll")                                                       \
        for (int mi = 0; mi < TILE_MB; ++mi)                                    \
            _Pragma("unroll")                                                   \
            for (int ni = 0; ni < TILE_NB; ++ni)                                \
                acc[mi][ni] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(      \
                    a_frag[mi], x_frag[ni], acc[mi][ni]);                       \
        SIDE_ALU(NALU)                                                          \
    }                                                                           \
    /* DCE guard: `sink` is always 0 at runtime, but the compiler cannot prove   \
       it, so the whole side chain must be computed. */                         \
    if (sink != 0) {                                                            \
        Y[0] = d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7;                           \
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

// NALU = side FMAs per K-step. ops/weight-element = NALU / 64.
DEFINE_ALU_GEMM(gemm_bf16_alu0,    4, 4, 2, 0)
DEFINE_ALU_GEMM(gemm_bf16_alu64,   4, 4, 2, 64)
DEFINE_ALU_GEMM(gemm_bf16_alu128,  4, 4, 2, 128)
DEFINE_ALU_GEMM(gemm_bf16_alu256,  4, 4, 2, 256)
DEFINE_ALU_GEMM(gemm_bf16_alu512,  4, 4, 2, 512)
DEFINE_ALU_GEMM(gemm_bf16_alu1024, 4, 4, 2, 1024)
DEFINE_ALU_GEMM(gemm_bf16_alu2048, 4, 4, 2, 2048)
