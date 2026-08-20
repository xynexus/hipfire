// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Opus Quant COMPACT-RESIDENT decode GEMV v2 (batch=1): bandwidth-optimized
// rewrite of gemv_oq_compact_grouped, standing in the same relation to it as
// gemv_oq8_grouped_v2 does to v1, and for the same reasons:
//   - 64-bit coalesced nibble loads — 8 bytes = 16 weights per lane per load,
//     against v1's 4 bytes. (Not 128-bit: at 16 lanes/group a 256-group holds
//     only 8 nibble bytes per lane. Half the weight bytes is the whole point.)
//   - Each wave32 covers TWO groups per round (16 lanes/group), giving two
//     independent 16-wide dot products in flight → ILP / latency hiding. This
//     is what v1 lacked; profiling put v1 at ~1.8x the per-call cost of
//     gemv_oq8_grouped_v2 despite reading half the bytes.
//   - float4 activation loads.
//
// Requires group == 256 and ng even (K % 512 == 0); callers fall back to v1
// otherwise, mirroring the oq8 pair.
//
// Numerics are v1's: inline int4 dequant + sparse int8 overlay against the
// FULL-PRECISION f32 activation (W4A16 at decode). No activation quantization —
// at B=1 there is no WMMA to feed.
//
// On-disk block, one per (row, group) of 256 weights, read as it sits:
//
//   [ f16 scale | 128 packed signed-int4 | N_out * (u8 idx_in_group, i8 val) ]
//     2 bytes      128 bytes                2*N_out bytes
//   block_stride = 130 + 2*N_out
//
//   W : compact blocks, [M, K/256] blocks of block_stride bytes
//   X : [K] f32 activation (batch=1, FWHT-rotated)
//   Y : [M] f32

#include <hip/hip_runtime.h>
#include <stdint.h>

// Sign-extend the low/high nibble of byte `j` (0..3) within a packed int32 word.
#define OQCV2_LO(w, j) ((int)((((uint32_t)(w)) >> (8 * (j))) & 0xf) << 28 >> 28)
#define OQCV2_HI(w, j) ((int)((((uint32_t)(w)) >> (8 * (j) + 4)) & 0xf) << 28 >> 28)

__launch_bounds__(32) extern "C" __global__ void gemv_oq_compact_grouped_v2(
    const uint8_t* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int group /* == 256 */, int block_stride
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x;   // wave32
    const int ng   = K / 256;       // groups (even)
    const int gir  = lane >> 4;     // 0 or 1 : which of the 2 groups this round
    const int lig  = lane & 15;     // 0..15 : 16 weights within the group
    const int n_ov = (block_stride - 130) >> 1;
    const int elt  = lig * 16;      // weight/activation offset within the group

    const uint8_t* w_row = W + (long long)row * ng * (long long)block_stride;

    float facc = 0.0f;
    for (int r = 0; r < ng; r += 2) {
        const int gi = r + gir;
        const uint8_t* blk = w_row + (long long)gi * block_stride;

        // The block's f16 scale — one load per lane-half, broadcast, instead of
        // the scattered per-output-row scale fetches the WMMA path needed.
        const float sw = (float)(*(const _Float16*)blk);

        // Nibbles start at blk+2, so this 8-byte window is 2 mod 8. State the
        // width and let the ISA take the unaligned load (gfx11 does it natively)
        // rather than decaying to byte reads.
        uint32_t wv[2];
        __builtin_memcpy(wv, blk + 2 + lig * 8, 8);
        const uint32_t w0 = wv[0], w1 = wv[1];

        const int xoff = gi * 256 + elt;
        const float4 x0 = *(const float4*)(X + xoff);
        const float4 x1 = *(const float4*)(X + xoff + 4);
        const float4 x2 = *(const float4*)(X + xoff + 8);
        const float4 x3 = *(const float4*)(X + xoff + 12);

        float s = 0.0f;
        s += (float)OQCV2_LO(w0, 0) * x0.x + (float)OQCV2_HI(w0, 0) * x0.y
           + (float)OQCV2_LO(w0, 1) * x0.z + (float)OQCV2_HI(w0, 1) * x0.w;
        s += (float)OQCV2_LO(w0, 2) * x1.x + (float)OQCV2_HI(w0, 2) * x1.y
           + (float)OQCV2_LO(w0, 3) * x1.z + (float)OQCV2_HI(w0, 3) * x1.w;
        s += (float)OQCV2_LO(w1, 0) * x2.x + (float)OQCV2_HI(w1, 0) * x2.y
           + (float)OQCV2_LO(w1, 1) * x2.z + (float)OQCV2_HI(w1, 1) * x2.w;
        s += (float)OQCV2_LO(w1, 2) * x3.x + (float)OQCV2_HI(w1, 2) * x3.y
           + (float)OQCV2_LO(w1, 3) * x3.z + (float)OQCV2_HI(w1, 3) * x3.w;

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
        for (int e = lig + 16; e < n_ov; e += 16) {
            const int idx = (int)table[2 * e];
            const int val = (int)(int8_t)table[2 * e + 1];
            s += (float)val * X[gi * 256 + idx];
        }

        facc += s * sw;
    }

    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down(facc, off);
    if (lane == 0) Y[row] = facc;
}
