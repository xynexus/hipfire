// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Compact-resident Opus decode GEMV v3 (batch=1): WIDE-LOAD twin of
// `gemv_oq_compact_grouped_v2`. Identical math and the same overlay contract
// (duplicate indices resolved and the bulk nibble zeroed at load by
// `normalize_compact_overlays`); what changes is the shape of the weight read.
//
// v2 gives each group 16 lanes, so a lane loads 8 nibble bytes — a `dwordx2`.
// The pure-read stream this kernel is chasing moves 250 GB/s with `dwordx4`,
// and ablations said the residual gap is weight-load SHAPE rather than bytes,
// ALU (2.5%) or activation traffic (~0 at one wave per workgroup). So v3 gives
// each group 8 lanes and a `dwordx4`: 16 bytes = 32 weights per lane. One wave32
// then covers FOUR groups per round instead of two, which halves the weight VMEM
// instruction count per byte moved and quadruples the groups in flight.
//
// Requires group == 256 and ng % 4 == 0 (K % 1024 == 0); callers fall back to v2
// otherwise. Qwen3.8-27B's K values (5120, 6144, 17408) all qualify.
//
// Numerics are v2's exactly: inline int4 dequant + sparse int8 overlay against
// the FULL-PRECISION f32 activation (W4A16 at decode).
//
//   W : compact blocks, [M, K/256] blocks of block_stride bytes
//   X : [K] f32 activation (batch=1, FWHT-rotated)
//   Y : [M] f32

#include <hip/hip_runtime.h>
#include <stdint.h>

#define OQCV3_LO(w, j) ((int)((((uint32_t)(w)) >> (8 * (j))) & 0xf) << 28 >> 28)
#define OQCV3_HI(w, j) ((int)((((uint32_t)(w)) >> (8 * (j) + 4)) & 0xf) << 28 >> 28)

// One 32-bit word = 8 nibbles = 8 weights, against two float4 of activation.
// The word parameter is `ww`, NOT `w`: a parameter named `w` would make the
// `(xa).w` component selector expand to the caller's argument.
#define OQCV3_WORD(ww, xa, xb)                                                 \
    ( (float)OQCV3_LO(ww, 0) * (xa).x + (float)OQCV3_HI(ww, 0) * (xa).y        \
    + (float)OQCV3_LO(ww, 1) * (xa).z + (float)OQCV3_HI(ww, 1) * (xa).w        \
    + (float)OQCV3_LO(ww, 2) * (xb).x + (float)OQCV3_HI(ww, 2) * (xb).y        \
    + (float)OQCV3_LO(ww, 3) * (xb).z + (float)OQCV3_HI(ww, 3) * (xb).w )

__launch_bounds__(32) extern "C" __global__ void gemv_oq_compact_grouped_v3(
    const uint8_t* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int group /* == 256 */, int block_stride
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x;   // wave32
    const int ng   = K / 256;       // groups (multiple of 4)
    const int gir  = lane >> 3;     // 0..3 : which of the 4 groups this round
    const int lig  = lane & 7;      // 0..7 : 32 weights within the group
    const int n_ov = (block_stride - 130) >> 1;
    const int elt  = lig * 32;      // weight/activation offset within the group

    const uint8_t* w_row = W + (long long)row * ng * (long long)block_stride;

    float facc = 0.0f;
    for (int r = 0; r < ng; r += 4) {
        const int gi = r + gir;
        const uint8_t* blk = w_row + (long long)gi * block_stride;

        const float sw = (float)(*(const _Float16*)blk);

        // 16 bytes = 32 weights per lane. The window is 2 mod 16, so state the
        // width and let the ISA take the unaligned load — measured neutral here
        // (an 8-byte-ALIGNED ablation of v2 scored 74.08 ms against 74.41, i.e.
        // alignment is worth nothing on this access pattern).
        uint32_t wv[4];
        __builtin_memcpy(wv, blk + 2 + lig * 16, 16);

        const int xoff = gi * 256 + elt;
        const float4 x0 = *(const float4*)(X + xoff);
        const float4 x1 = *(const float4*)(X + xoff + 4);
        const float4 x2 = *(const float4*)(X + xoff + 8);
        const float4 x3 = *(const float4*)(X + xoff + 12);
        const float4 x4 = *(const float4*)(X + xoff + 16);
        const float4 x5 = *(const float4*)(X + xoff + 20);
        const float4 x6 = *(const float4*)(X + xoff + 24);
        const float4 x7 = *(const float4*)(X + xoff + 28);

        float s = OQCV3_WORD(wv[0], x0, x1) + OQCV3_WORD(wv[1], x2, x3)
                + OQCV3_WORD(wv[2], x4, x5) + OQCV3_WORD(wv[3], x6, x7);

        // Overlays REPLACE the bulk nibble, and the loader ZEROES that nibble, so
        // a correction is just val*x[idx] and depends on no lane's registers.
        // BRANCHLESS ON PURPOSE: the `lig < n_ov` guard used to be a divergent
        // branch, and that branch — not the table loads, not the scattered
        // gather, not alignment — was what the overlay actually cost. Every lane
        // now issues the same loads and the inactive ones multiply by zero,
        // worth 3.2% of decode (73.17 -> 70.83 ms/token on Qwen3.8-27B).
        // Load-merging into one ushort (+1%), 8-byte block alignment (-0.5%),
        // K-depth 2 (+14.6%) and pipelining the table one group ahead (+2.7%)
        // were all tried against this same cost and all LOST; the branch was the
        // thing. Entries beyond one per lane fall to the residual loop, whose
        // bound is wave-uniform.
        const uint8_t* table = blk + 130;
        {
            const int e   = (lig < n_ov) ? lig : 0;
            const int idx = (int)table[2 * e];
            const int val = (lig < n_ov) ? (int)(int8_t)table[2 * e + 1] : 0;
            s += (float)val * X[gi * 256 + idx];
        }
        for (int e = lig + 8; e < n_ov; e += 8) {
            const int idx = (int)table[2 * e];
            const int val = (int)(int8_t)table[2 * e + 1];
            s += (float)val * X[gi * 256 + idx];
        }

        facc += s * sw;
    }

    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down(facc, off);
    if (lane == 0) Y[row] = facc;
}
