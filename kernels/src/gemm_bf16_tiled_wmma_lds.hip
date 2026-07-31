// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
// LDS-staged, double-buffered, register-super-tiled BF16×BF16→F32 WMMA GEMM for
// gfx1103 (RDNA3, wave32) — the diffusion-DiT throughput kernel. Beats the
// zero-LDS register-tiled gemm_bf16_tiled_wmma_4x4: that kernel is 1 wave/block
// with 16 f32x8 accumulators (128 VGPRs → 2 waves/SIMD, register-starved) and no
// operand reuse across waves / no prefetch, pinning it near ~8% of peak. This
// kernel mirrors the structure of gemm_oq4_grouped_wmma_lds.hip (the iu4 LDS win)
// with the quant machinery removed — bf16 accumulates directly in the f32
// fragment, so there is no int4 unpack, no per-group scale, no i32→f32 rescale.
//
//   A : [M, K] BF16 row-major (raw uint16)   X : [B, K] BF16 row-major
//   Y : [B, M] F32 row-major
//
// 8 wave32 waves / block computing a BM=64 × BN=128 block tile:
//   • WMt×WNt = 2×2 register super-tile → 4 INDEPENDENT WMMA chains / wave (deep
//     ILP), only 32 acc VGPRs / wave (vs 128 in the 4×4 zero-LDS kernel → far
//     higher occupancy).
//   • BK=64 K-strips staged in double-buffered LDS with a register-staged
//     global→reg prefetch of the next strip DURING compute (RDNA has no cp.async).
//   • Every wave reuses the block's LDS tile → the redundant DRAM re-reads that
//     capped the zero-LDS kernel are gone. LDS = 2·(64+128)·64·2 B = 48 KiB (< 64).
// Same f32 accumulation order over K + same WMMA builtin as the tiled kernel ⇒
// BIT-EXACT (parity_gemm_bf16_tiled_wmma_lds). Requires K % 64 == 0.
// gfx1103/RDNA3 wave32 (the gfx1103 LDS-wedge hazard is nullified).

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short   __attribute__((ext_vector_type(16))) b16x16_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;
typedef int32_t __attribute__((ext_vector_type(4)))  i32x4;

#define BK 64                    // K-strip depth (bf16 elements)
#define WARPS_M 2
#define WARPS_N 4
#define NWARP (WARPS_M * WARPS_N) // 8 wave32 waves
#define BLOCK (NWARP * 32)        // 256 threads
#define WMt 2                     // M sub-tiles per wave
#define WNt 2                     // N sub-tiles per wave
#define BM (WARPS_M * WMt * 16)   // 64
#define BN (WARPS_N * WNt * 16)   // 128

__launch_bounds__(BLOCK, 2)
extern "C" __global__ void gemm_bf16_tiled_wmma_lds(
    const unsigned short* __restrict__ A,
    const unsigned short* __restrict__ X,
    float*                __restrict__ Y,
    int M, int K, int B
) {
    __shared__ unsigned short s_A[2][BM * BK];
    __shared__ unsigned short s_X[2][BN * BK];

    const int tid  = threadIdx.x;
    const int wid  = tid >> 5;            // 0..7
    const int wtid = tid & 31;            // wave-local 0..31
    const int lane = wtid & 15;           // 0..15 (mirrored WMMA input row)
    const int wm   = wid / WARPS_N;       // 0..1
    const int wn   = wid % WARPS_N;       // 0..3
    const int hrow = wtid >> 4;           // 0/1 output row-half

    const int bm = blockIdx.x * BM;
    const int bn = blockIdx.y * BN;
    const int nstrip = K / BK;

    // Staging counts. A row-strip = BK bf16 = BK/2 u32; X read as i32x4 (8 bf16).
    const int UA  = BK / 2;               // u32 per A row-strip (=32)
    const int UXv = BK / 8;               // i32x4 per X row-strip (=8)
    const int NA  = BM * UA / BLOCK;      // A u32 / thread (=8)
    const int NXv = BN * UXv / BLOCK;     // X i32x4 / thread (=4)

    float8_t acc[WMt][WNt];
#pragma unroll
    for (int i = 0; i < WMt; i++)
#pragma unroll
        for (int j = 0; j < WNt; j++)
            acc[i][j] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    uint32_t rA[NA];
    i32x4    rX[NXv];

    // Global -> registers. kk is a K-offset in bf16 elements.
#define LOADG(kk) \
    _Pragma("unroll") for (int l = 0; l < NA; l++) { \
        int idx = tid + l * BLOCK, row = idx / UA, c = idx % UA; \
        int r = bm + row < M ? bm + row : M - 1; \
        rA[l] = *(const uint32_t*)(A + (long long)r * K + (kk) + c * 2); } \
    _Pragma("unroll") for (int l = 0; l < NXv; l++) { \
        int q = tid + l * BLOCK, row = q / UXv, c = q % UXv; \
        int r = bn + row < B ? bn + row : B - 1; \
        rX[l] = *(const i32x4*)(X + (long long)r * K + (kk) + c * 8); }
#define STS(bf) \
    _Pragma("unroll") for (int l = 0; l < NA; l++) { int idx = tid + l * BLOCK; ((uint32_t*)s_A[bf])[idx] = rA[l]; } \
    _Pragma("unroll") for (int l = 0; l < NXv; l++) { int q = tid + l * BLOCK; ((i32x4*)s_X[bf])[q] = rX[l]; }

    { LOADG(0); STS(0); }
    __syncthreads();

    for (int k = 0; k < nstrip; k++) {
        const int buf = k & 1;
        if (k + 1 < nstrip) { LOADG((k + 1) * BK); }
#pragma unroll
        for (int s = 0; s < BK / 16; s++) {
            b16x16_t af[WMt], xf[WNt];
#pragma unroll
            for (int i = 0; i < WMt; i++)
                af[i] = *(const b16x16_t*)(&s_A[buf][((wm * WMt + i) * 16 + lane) * BK + s * 16]);
#pragma unroll
            for (int j = 0; j < WNt; j++)
                xf[j] = *(const b16x16_t*)(&s_X[buf][((wn * WNt + j) * 16 + lane) * BK + s * 16]);
#pragma unroll
            for (int i = 0; i < WMt; i++)
#pragma unroll
                for (int j = 0; j < WNt; j++)
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[i], xf[j], acc[i][j]);
        }
        if (k + 1 < nstrip) { STS((k + 1) & 1); }
        __syncthreads();
    }
#undef LOADG
#undef STS

#pragma unroll
    for (int i = 0; i < WMt; i++)
#pragma unroll
        for (int j = 0; j < WNt; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
                int r = bm + (wm * WMt + i) * 16 + 2 * e + hrow;
                int c = bn + (wn * WNt + j) * 16 + lane;
                if (r < M && c < B) {
                    Y[(long long)c * M + r] = acc[i][j][e];
                }
            }
}
