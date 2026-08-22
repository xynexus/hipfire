// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64
//
// Compact-resident Opus EXACT W4A8 on the tuned wave64 structure.
//
// This is `gemm_oq_compact_iu4_w64`'s geometry — wave64, BK=64 K-strip,
// register-staged double buffer, b128 fragment reads — carrying the exact
// two-pass activation split of `gemm_oq_compact_iu4x2_wmma`:
//
//     x = 16 * x_hi + x_lo      (hi digit SIGNED, lo digit UNSIGNED)
//
// so one int8 activation is consumed by two iu4 WMMA passes against ONE weight
// fragment. iu4 is 16 cycles against iu8's 32 on gfx1151, so two iu4 passes cost
// the same WMMA cycles as a single iu8 pass while feeding the compact nibbles
// RAW — no unpack in the inner loop, which is what makes the compact format
// beat both oq8 (2.4x slower) and dense bf16 on this hardware.
//
// WNt is halved against the 1-pass twin (4, not 8). Two i32 accumulator sets at
// WNt=8 would be 128 VGPRs of accumulator alone before `facc`; the wave32 2-pass
// made exactly the same trade when it halved NB.
//
//   W  : compact SPLIT-PLANE blocks — nibble plane first, then [f16 scale][table]
//   X  : [B, K] int8, as `quantize_act_oq8_batched` leaves it. The two digit
//        planes are split during the LDS store: K/2 + K/2 == K, so this is the
//        same byte volume as pre-split planes and costs a few VALU ops.
//   Xs : [B, K/G] f32     Y : [B, M] f32
//
// Does NOT apply the sparse overlay; that is a separate pass, as for both twins.
// Requires K % 256 == 0 (Oq4G256 guarantees it).
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t i32x2 __attribute__((ext_vector_type(2)));
typedef int32_t i32x4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
// hi pass: B operand SIGNED. lo pass: B operand UNSIGNED. A (weights) always signed.
#define W64H(a, b, c) __builtin_amdgcn_wmma_i32_16x16x16_iu4_w64(true, a, true, b, c, false)
#define W64L(a, b, c) __builtin_amdgcn_wmma_i32_16x16x16_iu4_w64(true, a, false, b, c, false)

// Pack the low nibble of each of 4 bytes into the low 16 bits, in order.
// Input must already be masked to 0x0f0f0f0f.
__device__ __forceinline__ uint32_t oqx2w64_pack4(uint32_t t) {
    t = (t | (t >> 4)) & 0x00ff00ffu;
    return (t | (t >> 8)) & 0x0000ffffu;
}

#define BK 64
#define BLOCK 256   // 4 wave64 waves
#define WARPS_N 2
#define WMt 2
#ifndef WNt
#define WNt 4      // halved vs the 1-pass twin: two i32 accumulator sets.
                   // WNt=8 spills 113 VGPRs (2 waves); WNt=6 breaks the
                   // load macros (BN*UA/4 must divide BLOCK). So: 4.
#endif
#define BM (2 * WMt * 16)
#define BN (2 * WNt * 16)
#define STRIPS_PER_GROUP (256 / BK)   // 4

__launch_bounds__(BLOCK) extern "C" __global__ void gemm_oq_compact_iu4x2_w64(
    const uint8_t* __restrict__ W, const int8_t* __restrict__ X,
    const float* __restrict__ Xs, float* __restrict__ Y,
    int M, int K, int B, int block_stride) {
    __shared__ uint8_t As[2][BM * (BK / 2)], Xhs[2][BN * (BK / 2)], Xls[2][BN * (BK / 2)];
    const int bm = blockIdx.x * BM, bn = blockIdx.y * BN, st = K / 2;
    const int xst = K;               // int8 activation row stride
    const int wid = threadIdx.x >> 6, wtid = threadIdx.x & 63, lane = wtid & 15;
    const int wm = wid / WARPS_N, wn = wid % WARPS_N, nstrip = K / BK;
    const int n_groups = K / 256;
    const int side_stride = block_stride - 128;
    const long long side_base = (long long)M * n_groups * 128LL;
    const int UA = (BK / 2) / 4;
    const int NA = BM * UA / BLOCK;
    const int NXv = BN * UA / 4 / BLOCK;

    i32x4 acch[WMt][WNt], accl[WMt][WNt];
    f32x4 facc[WMt][WNt];
#pragma unroll
    for (int i = 0; i < WMt; i++)
#pragma unroll
        for (int j = 0; j < WNt; j++) {
            acch[i][j] = (i32x4){0, 0, 0, 0};
            accl[i][j] = (i32x4){0, 0, 0, 0};
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
    i32x4 rXh[NXv], rXl[NXv];
#define LOADG(kk) \
    _Pragma("unroll") for (int l = 0; l < NA; l++) { int idx = threadIdx.x + l * BLOCK, row = idx / UA, c = idx % UA; int r = bm + row < M ? bm + row : M - 1; rA[l] = *(const uint32_t*)(W + (long long)r * st + (kk) / 2 + c * 4); } \
    _Pragma("unroll") for (int l = 0; l < NXv; l++) { \
        int q = threadIdx.x + l * BLOCK, base = q * 4, row = base / UA, c = base % UA; \
        int r = bn + row < B ? bn + row : B - 1; \
        const int8_t* xp = X + (long long)r * xst + (kk) + c * 8; \
        /* 16 output nibble-bytes = 32 int8 = two i32x4 loads. */ \
        const i32x4 s0 = *(const i32x4*)(xp); \
        const i32x4 s1 = *(const i32x4*)(xp + 16); \
        i32x4 vh, vl; \
        _Pragma("unroll") for (int t = 0; t < 2; t++) { \
            const i32x4 sv = t ? s1 : s0; \
            uint32_t ph = 0, pl = 0, ph2 = 0, pl2 = 0; \
            ph  = oqx2w64_pack4(((uint32_t)sv[0] >> 4) & 0x0f0f0f0fu) | (oqx2w64_pack4(((uint32_t)sv[1] >> 4) & 0x0f0f0f0fu) << 16); \
            ph2 = oqx2w64_pack4(((uint32_t)sv[2] >> 4) & 0x0f0f0f0fu) | (oqx2w64_pack4(((uint32_t)sv[3] >> 4) & 0x0f0f0f0fu) << 16); \
            pl  = oqx2w64_pack4((uint32_t)sv[0] & 0x0f0f0f0fu) | (oqx2w64_pack4((uint32_t)sv[1] & 0x0f0f0f0fu) << 16); \
            pl2 = oqx2w64_pack4((uint32_t)sv[2] & 0x0f0f0f0fu) | (oqx2w64_pack4((uint32_t)sv[3] & 0x0f0f0f0fu) << 16); \
            vh[2 * t] = (int)ph; vh[2 * t + 1] = (int)ph2; \
            vl[2 * t] = (int)pl; vl[2 * t + 1] = (int)pl2; \
        } \
        rXh[l] = vh; rXl[l] = vl; }
#define STS(bf) \
    _Pragma("unroll") for (int l = 0; l < NA; l++) { int idx = threadIdx.x + l * BLOCK; ((uint32_t*)As[bf])[idx] = rA[l]; } \
    _Pragma("unroll") for (int l = 0; l < NXv; l++) { int q = threadIdx.x + l * BLOCK; ((i32x4*)Xhs[bf])[q] = rXh[l]; ((i32x4*)Xls[bf])[q] = rXl[l]; }

    { LOADG(0); STS(0); }
    __syncthreads();
    for (int k = 0; k < nstrip; k++) {
        int buf = k & 1;
        if (k + 1 < nstrip) { LOADG((k + 1) * BK); }
#pragma unroll
        for (int s = 0; s < BK / 16; s++) {
            i32x2 af[WMt], xfh[WNt], xfl[WNt];
#pragma unroll
            for (int i = 0; i < WMt; i++) af[i] = *(const i32x2*)(As[buf] + (wm * WMt * 16 + i * 16 + lane) * (BK / 2) + s * 8);
#pragma unroll
            for (int j = 0; j < WNt; j++) {
                const int xo = (wn * WNt * 16 + j * 16 + lane) * (BK / 2) + s * 8;
                xfh[j] = *(const i32x2*)(Xhs[buf] + xo);
                xfl[j] = *(const i32x2*)(Xls[buf] + xo);
            }
#pragma unroll
            for (int i = 0; i < WMt; i++)
#pragma unroll
                for (int j = 0; j < WNt; j++) {
                    // ONE weight fragment, BOTH digit passes -- the whole reason
                    // two passes are not two kernels' worth of weight traffic.
                    acch[i][j] = W64H(af[i], xfh[j], acch[i][j]);
                    accl[i][j] = W64L(af[i], xfl[j], accl[i][j]);
                }
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
                        // Exact radix-16 recombination in i32 before the scale.
                        const int v = 16 * acch[i][j][g] + accl[i][j][g];
                        facc[i][j][g] += (float)v * sw[i][g] * sx;
                    }
                    acch[i][j] = (i32x4){0, 0, 0, 0};
                    accl[i][j] = (i32x4){0, 0, 0, 0};
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
