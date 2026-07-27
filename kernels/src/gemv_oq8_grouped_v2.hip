// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Opus Quant W8A16 DECODE GEMV v2 (batch=1): bandwidth-optimized rewrite of
// gemv_oq8_grouped. Same numerics (inline int8 weight dequant × f32 activation,
// per-group f32 scales), but:
//   - 128-bit (int4 / dwordx4) coalesced weight loads — 16 int8/lane/load, 4×
//     fewer memory instructions than the 4-byte int32 loads of v1.
//   - Each wave32 covers TWO groups per round (16 lanes/group), giving two
//     independent 16-wide dot products in flight → better ILP / latency hiding.
//   - float4 activation loads.
// Requires group == 256 and K % 512 == 0 (ng even). Callers fall back to v1
// otherwise (small router GEMVs).
//
//   W  : [M, K] signed int8            row stride = K
//   Ws : [M, K/256] f32 group scales
//   X  : [K] f32 activation (rotated)
//   Y  : [M] f32

#include <hip/hip_runtime.h>
#include <stdint.h>

// Sign-extend the n-th int8 lane (0..3) of a packed int32 word.
#define OQ8V2_SXT(word, n) ((int)(((uint32_t)(word) << (24 - 8 * (n)))) >> 24)

__launch_bounds__(32) extern "C" __global__ void gemv_oq8_grouped_v2(
    const int8_t* __restrict__ W,
    const float* __restrict__ Ws,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int group /* == 256 */
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x;            // wave32
    const int ng = K / 256;                  // groups (even)
    const int gir = lane >> 4;               // 0 or 1 : which of the 2 groups
    const int lig = lane & 15;               // 0..15 : 16 int8 within the group
    const int8_t* w_row = W + (long long)row * K;
    const float* ws_row = Ws + (long long)row * ng;
    const int elt = lig * 16;                // int8/float offset within the group

    float facc = 0.0f;
    for (int r = 0; r < ng; r += 2) {
        const int gi = r + gir;              // this lane's group this round
        const int off = gi * 256 + elt;      // 16-byte aligned
        const int4 wv = *(const int4*)(w_row + off);
        const float4 x0 = *(const float4*)(X + off);
        const float4 x1 = *(const float4*)(X + off + 4);
        const float4 x2 = *(const float4*)(X + off + 8);
        const float4 x3 = *(const float4*)(X + off + 12);
        float s = 0.0f;
        s += (float)OQ8V2_SXT(wv.x, 0) * x0.x + (float)OQ8V2_SXT(wv.x, 1) * x0.y
           + (float)OQ8V2_SXT(wv.x, 2) * x0.z + (float)OQ8V2_SXT(wv.x, 3) * x0.w;
        s += (float)OQ8V2_SXT(wv.y, 0) * x1.x + (float)OQ8V2_SXT(wv.y, 1) * x1.y
           + (float)OQ8V2_SXT(wv.y, 2) * x1.z + (float)OQ8V2_SXT(wv.y, 3) * x1.w;
        s += (float)OQ8V2_SXT(wv.z, 0) * x2.x + (float)OQ8V2_SXT(wv.z, 1) * x2.y
           + (float)OQ8V2_SXT(wv.z, 2) * x2.z + (float)OQ8V2_SXT(wv.z, 3) * x2.w;
        s += (float)OQ8V2_SXT(wv.w, 0) * x3.x + (float)OQ8V2_SXT(wv.w, 1) * x3.y
           + (float)OQ8V2_SXT(wv.w, 2) * x3.z + (float)OQ8V2_SXT(wv.w, 3) * x3.w;
        facc += s * ws_row[gi];
    }
    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down(facc, off);
    if (lane == 0) Y[row] = facc;
}
