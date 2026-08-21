// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64
//
// Compact-resident Opus W4A4 on the TUNED wave64 structure.
//
// This is `gemm_iu4_i32_wmma_lds`'s geometry — wave64, BK=64 K-strip, N-heavy
// BM=64/BN=256, register-staged double buffer, b128 fragment reads — carrying
// the compact format's per-group scales. The wave32 compact kernel it replaces
// measured 33 TOPS; the structure below measures 50-69 on the same shapes,
// exactly where the recorded tuning arc says a wave32 double-buffered design
// tops out.
//
// Two facts make the port narrow rather than a rewrite:
//
//  * THE COMPACT NIBBLE PLANE IS ALREADY A DENSE [M, K/2] int4 ARRAY.
//    `split_compact_planes` lays row m's groups contiguously at m*(ng*128), and
//    group g at g*128 within it — precisely the A operand this structure wants.
//    A addressing is therefore UNCHANGED from the dense kernel.
//  * wave64 `_w64` WMMA holds a 16x16 tile in 4 accumulator GPRs, not 8 (64
//    lanes x 4 = 256), which is what affords WNt=8. N-heavy also halves the
//    rescale cost per WMMA: 256 WMMA per group per wave against 64 output
//    elements, where the wave32 2x8 shape had 128 against 64.
//
// The one structural difference from the dense kernel: it accumulates i32 across
// the whole of K, while a compact group carries its own f16 scale every 256
// weights. 256 = 4 K-strips at BK=64, so the i32 accumulator is folded into an
// f32 one and zeroed every fourth strip. That costs a second accumulator set
// (i32x4 + float4 per tile) and is inherent to per-group scales.
//
//   W  : compact SPLIT-PLANE blocks — nibble plane first, then [f16 scale][table]
//   X  : [B, K/2] packed signed int4     Xs : [B, K/G] f32
//   Y  : [B, M] f32
//
// Does NOT apply the sparse overlay; that is a separate pass, as for the wave32
// twin. Requires K % 256 == 0 (Oq4G256 guarantees it).
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t i32x2 __attribute__((ext_vector_type(2)));
typedef int32_t i32x4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
#define W64(a, b, c) __builtin_amdgcn_wmma_i32_16x16x16_iu4_w64(true, a, true, b, c, false)

#define BK 64
#define BLOCK 256   // 4 wave64 waves
#define WARPS_N 2
#define WMt 2
#ifndef WNt
#define WNt 8
#endif
#define BM (2 * WMt * 16)
#define BN (2 * WNt * 16)
#define STRIPS_PER_GROUP (256 / BK)   // 4

__launch_bounds__(BLOCK) extern "C" __global__ void gemm_oq_compact_iu4_w64(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ X,
    const float* __restrict__ Xs, float* __restrict__ Y,
    int M, int K, int B, int block_stride) {
    __shared__ uint8_t As[2][BM * (BK / 2)], Xls[2][BN * (BK / 2)];
    const int bm = blockIdx.x * BM, bn = blockIdx.y * BN, st = K / 2;
    const int wid = threadIdx.x >> 6, wtid = threadIdx.x & 63, lane = wtid & 15;
    const int wm = wid / WARPS_N, wn = wid % WARPS_N, nstrip = K / BK;
    const int n_groups = K / 256;
    const int side_stride = block_stride - 128;
    const long long side_base = (long long)M * n_groups * 128LL;
    const int UA = (BK / 2) / 4;
    const int NA = BM * UA / BLOCK;
    const int NXv = BN * UA / 4 / BLOCK;

    i32x4 acc[WMt][WNt];
    f32x4 facc[WMt][WNt];
#pragma unroll
    for (int i = 0; i < WMt; i++)
#pragma unroll
        for (int j = 0; j < WNt; j++) {
            acc[i][j] = (i32x4){0, 0, 0, 0};
            facc[i][j] = (f32x4){0.f, 0.f, 0.f, 0.f};
        }

    // Scale pointers hoisted: the row/col indices vary only in the group, so a
    // 64-bit multiply per (group, tile) becomes one per tile for the kernel.
    const uint8_t* swp[WMt][4];
    const float* sxp[WNt];
#pragma unroll
    for (int i = 0; i < WMt; i++)
#pragma unroll
        for (int g = 0; g < 4; g++) {
            const int r0 = bm + wm * WMt * 16 + i * 16 + 4 * g + (wtid >> 4);
            const int rs = (r0 < M) ? r0 : (M - 1);
            swp[i][g] = W + side_base
                      + (long long)rs * n_groups * (long long)side_stride;
        }
#pragma unroll
    for (int j = 0; j < WNt; j++) {
        const int c0 = bn + wn * WNt * 16 + j * 16 + lane;
        const int cs = (c0 < B) ? c0 : (B - 1);
        sxp[j] = Xs + (long long)cs * n_groups;
    }

    uint32_t rA[NA];
    i32x4 rX[NXv];
#define LOADG(kk) \
    _Pragma("unroll") for (int l = 0; l < NA; l++) { int idx = threadIdx.x + l * BLOCK, row = idx / UA, c = idx % UA; int r = bm + row < M ? bm + row : M - 1; rA[l] = *(const uint32_t*)(W + (long long)r * st + (kk) / 2 + c * 4); } \
    _Pragma("unroll") for (int l = 0; l < NXv; l++) { int q = threadIdx.x + l * BLOCK, base = q * 4, row = base / UA, c = base % UA; int r = bn + row < B ? bn + row : B - 1; rX[l] = *(const i32x4*)(X + (long long)r * st + (kk) / 2 + c * 4); }
#define STS(bf) \
    _Pragma("unroll") for (int l = 0; l < NA; l++) { int idx = threadIdx.x + l * BLOCK; ((uint32_t*)As[bf])[idx] = rA[l]; } \
    _Pragma("unroll") for (int l = 0; l < NXv; l++) { int q = threadIdx.x + l * BLOCK; ((i32x4*)Xls[bf])[q] = rX[l]; }

    { LOADG(0); STS(0); }
    __syncthreads();
    for (int k = 0; k < nstrip; k++) {
        int buf = k & 1;
        if (k + 1 < nstrip) { LOADG((k + 1) * BK); }
#pragma unroll
        for (int s = 0; s < BK / 16; s++) {
            i32x2 af[WMt], xf[WNt];
#pragma unroll
            for (int i = 0; i < WMt; i++) af[i] = *(const i32x2*)(As[buf] + (wm * WMt * 16 + i * 16 + lane) * (BK / 2) + s * 8);
#pragma unroll
            for (int j = 0; j < WNt; j++) xf[j] = *(const i32x2*)(Xls[buf] + (wn * WNt * 16 + j * 16 + lane) * (BK / 2) + s * 8);
#pragma unroll
            for (int i = 0; i < WMt; i++)
#pragma unroll
                for (int j = 0; j < WNt; j++) acc[i][j] = W64(af[i], xf[j], acc[i][j]);
        }
        // A compact group is 256 weights = STRIPS_PER_GROUP strips. Fold the i32
        // accumulator into f32 with this group's scales and reset it.
        if ((k + 1) % STRIPS_PER_GROUP == 0) {
            const int grp = k / STRIPS_PER_GROUP;
            const int g_off = grp * side_stride;
            float sw[WMt][4];
#pragma unroll
            for (int i = 0; i < WMt; i++)
#pragma unroll
                for (int g = 0; g < 4; g++)
                    sw[i][g] = (float)(*(const _Float16*)(swp[i][g] + g_off));
#pragma unroll
            for (int j = 0; j < WNt; j++) {
                const float sx = sxp[j][grp];
#pragma unroll
                for (int i = 0; i < WMt; i++) {
#pragma unroll
                    for (int g = 0; g < 4; g++) {
                        facc[i][j][g] += (float)acc[i][j][g] * sw[i][g] * sx;
                    }
                    acc[i][j] = (i32x4){0, 0, 0, 0};
                }
            }
        }
        if (k + 1 < nstrip) { STS((k + 1) & 1); }
        __syncthreads();
    }
#undef LOADG
#undef STS
#pragma unroll
    for (int i = 0; i < WMt; i++)
#pragma unroll
        for (int j = 0; j < WNt; j++) {
            int oc = bn + wn * WNt * 16 + j * 16 + lane;
            if (oc >= B) continue;
#pragma unroll
            for (int g = 0; g < 4; g++) {
                int r = bm + wm * WMt * 16 + i * 16 + 4 * g + (wtid >> 4);
                if (r < M) Y[(long long)oc * M + r] = facc[i][j][g];
            }
        }
}
