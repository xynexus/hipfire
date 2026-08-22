// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Rung 5 of the first-principles ladder: REAL GEMM TILING.
//
// Rungs 1-4 fed the matrix core from LDS, folded per group, and staged from
// global with a double buffer -- but their addressing was synthetic: a small
// resident tile with slots cycled by the loop counter. A real GEMM indexes a
// workgroup tile by (wave, m-subtile, n-subtile, k-step), which is a different
// address pattern, a different reuse pattern, and different register pressure.
//
// This adds exactly that and nothing else, so the drop from rung 4 is the cost
// of real tiling.
//
//   BM=64 BN=128 BK=64, 8 wave32 waves arranged 2x4, WMt=WNt=2.
//   Per wave per k-step: 2 A-frags + 2 B-frags -> 4 WMMA = 1.00 ds_load/WMMA,
//   which is what the shipping kernel achieves (0.31) rather than what the
//   synthetic probe did (2.00).
//   LDS double-buffered: 2*(2048+4096) = 12288 B.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t i32x2 __attribute__((ext_vector_type(2)));
typedef int32_t i32x4 __attribute__((ext_vector_type(4)));
typedef int32_t i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
#define W32(a, b, c) __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, false)

#define TBM 64
#define TBN 128
#define TBK 64
#define TWM 2          // wave grid rows
#define TWN 4          // wave grid cols
#define TWMt 2         // m-subtiles per wave
#define TWNt 2         // n-subtiles per wave
#define TAB (TBM * (TBK / 2))   // 2048 B
#define TBB (TBN * (TBK / 2))   // 4096 B

__launch_bounds__(256) extern "C" __global__ void wmma_iu4_tiled_w32(
    int32_t* __restrict__ out, const int32_t* __restrict__ src,
    int kstrips, int srcmask) {
    __shared__ uint8_t As[2][TAB];
    __shared__ uint8_t Bs[2][TBB];
    const int t = threadIdx.x;
    const int wid = t >> 5, lane = t & 31;
    const int wm = wid / TWN, wn = wid % TWN;

    i32x8 acc[TWMt][TWNt];
    f32x8 facc[TWMt][TWNt];
#pragma unroll
    for (int i = 0; i < TWMt; i++)
#pragma unroll
        for (int j = 0; j < TWNt; j++) {
            acc[i][j] = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
            facc[i][j] = (f32x8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }

    // Staging: (TAB + TBB)/16 = 384 dwordx4 per strip over 256 threads.
    const int NA = TAB / 16 / 256;   // 0 -> handled by the guard below
    const int NB = TBB / 16 / 256;
    int gbase = blockIdx.x * (TAB + TBB) / 4;
    i32x4 ra[2], rb[2];
    // prime buffer 0
    for (int l = 0; l * 256 + t < TAB / 16; ++l)
        *(i32x4*)(As[0] + (l * 256 + t) * 16) =
            *(const i32x4*)(src + ((gbase + (l * 256 + t) * 4) & srcmask));
    for (int l = 0; l * 256 + t < TBB / 16; ++l)
        *(i32x4*)(Bs[0] + (l * 256 + t) * 16) =
            *(const i32x4*)(src + ((gbase + TAB / 4 + (l * 256 + t) * 4) & srcmask));
    gbase += (TAB + TBB) / 4;
    __syncthreads();

    for (int k = 0; k < kstrips; ++k) {
        const int buf = k & 1;
        // issue next strip's loads; they retire under the WMMA below
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            const int i = l * 256 + t;
            if (i < TAB / 16)
                ra[l] = *(const i32x4*)(src + ((gbase + i * 4) & srcmask));
            if (i < TBB / 16)
                rb[l] = *(const i32x4*)(src + ((gbase + TAB / 4 + i * 4) & srcmask));
        }
        gbase += (TAB + TBB) / 4;

        // REAL TILED ADDRESSING: fragment for (wave, subtile, k-step).
#pragma unroll
        for (int s = 0; s < TBK / 16; ++s) {
            i32x2 af[TWMt], bf[TWNt];
#pragma unroll
            for (int i = 0; i < TWMt; i++)
                af[i] = *(const i32x2*)(As[buf] +
                    ((wm * TWMt + i) * 16 + lane) * (TBK / 2) + s * 8);
#pragma unroll
            for (int j = 0; j < TWNt; j++)
                bf[j] = *(const i32x2*)(Bs[buf] +
                    ((wn * TWNt + j) * 16 + lane) * (TBK / 2) + s * 8);
#pragma unroll
            for (int i = 0; i < TWMt; i++)
#pragma unroll
                for (int j = 0; j < TWNt; j++)
                    acc[i][j] = W32(af[i], bf[j], acc[i][j]);
        }

        // per-group fold: a 256-weight group is 4 strips at BK=64
        if ((k & 3) == 3) {
            const float sw = __builtin_amdgcn_rcpf((float)(k + 2));
#pragma unroll
            for (int i = 0; i < TWMt; i++)
#pragma unroll
                for (int j = 0; j < TWNt; j++) {
#pragma unroll
                    for (int e = 0; e < 8; ++e)
                        facc[i][j][e] += (float)acc[i][j][e] * sw;
                    acc[i][j] = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
                }
        }

        // commit next strip behind ONE barrier
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            const int i = l * 256 + t;
            if (i < TAB / 16) *(i32x4*)(As[buf ^ 1] + i * 16) = ra[l];
            if (i < TBB / 16) *(i32x4*)(Bs[buf ^ 1] + i * 16) = rb[l];
        }
        __syncthreads();
    }

    float s = 0.f;
#pragma unroll
    for (int i = 0; i < TWMt; i++)
#pragma unroll
        for (int j = 0; j < TWNt; j++)
#pragma unroll
            for (int e = 0; e < 8; ++e) s += facc[i][j][e];
    if (t == 255) out[blockIdx.x] = (int32_t)s;
    (void)NA; (void)NB;
}
