// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Compact-resident Opus decode GEMV, multi-wave + K-unrolled. Identical math to
// `gemv_oq_compact_grouped_v2` and the same overlay contract (duplicate indices
// resolved at load by `normalize_compact_overlays`); what changes is how the
// work is scheduled.
//
// Two levers, both chosen from measurement rather than guessed:
//
//  * WAVES per workgroup (from blockDim.x). RDNA caps workgroups per CU, so one
//    wave per workgroup leaves resident waves unused. Worth +1.9% end-to-end at
//    2 (10.40 -> 10.60 tok/s, interleaved A/B).
//
//  * TWO GROUPS' weight loads issued before either is consumed. FETCH_SIZE says
//    this kernel fetches 48019 KB for 46240 KB of weights — 1.038x, i.e. the
//    access pattern is already near-perfect and nothing is being wasted — yet it
//    runs at 165 GB/s against a 250 GB/s pure-read stream. The loop was one
//    8-byte weight load followed by ~48 VALU ops and four activation loads, so a
//    lane never had more than ONE weight load outstanding. The shortfall is
//    memory-level parallelism, not waste, and depth is the lever that addresses
//    it. Worth +8.5% end-to-end (10.60 -> 11.50 tok/s at 2 waves), validated
//    cross-process against the committed baseline BINARY rather than against a
//    previous bench run — 4 alternating reps each, zero variance.
//
//    DEPTH 4 WAS TRIED AND LOST: 10.50 dense, 49.50 MoE against 11.50 / 52.10.
//    Two is the sweet spot. This reproduces the lesson already recorded in
//    `.claude/skills/hipfire-kernel-tuning/levers.md` §3, where K2 is the shipped
//    baseline and the K4 / k2x32 experiments are negative results: past a point
//    register pressure gates, and more parallel work does not help when it
//    cannot be pipelined.
//
// Ruled out before landing here, so none is re-attempted: multi-row (rows=2 ->
// 9.70, rows=4 -> 8.70 tok/s against 10.60 — fewer workgroups costs more than
// the shared activation loads save); block re-alignment (~3%, and an aligned +8
// offset scores the same as the block's +2, so it is base-offset drift);
// A8 activations (`gemv_oq_compact_w8a8_grouped`, slower — int4 weights must be
// widened for V_DOT4 and at batch 1 the activation is a rounding error in the
// byte count); and fusing the small projections (0.9% of GEMV time).
//
//   W : compact blocks, [M, K/256] blocks of block_stride bytes
//   X : [K] f32 activation (batch=1, FWHT-rotated)
//   Y : [M] f32

#include <hip/hip_runtime.h>
#include <stdint.h>

#define OQCV2_LO(w, j) ((int)((((uint32_t)(w)) >> (8 * (j))) & 0xf) << 28 >> 28)
#define OQCV2_HI(w, j) ((int)((((uint32_t)(w)) >> (8 * (j) + 4)) & 0xf) << 28 >> 28)

// One group's contribution, given its already-loaded nibbles. Taking the loaded
// words as arguments is what lets the caller issue several groups' loads before
// any of them is consumed.
__device__ __forceinline__ float oqc_group(
    uint32_t w0, uint32_t w1, const uint8_t* __restrict__ blk,
    const float* __restrict__ X, int gi, int elt, int lig, int n_ov
) {
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

    // Overlays REPLACE the bulk nibble, so add the difference. No supersede
    // rescan — duplicates were resolved at load.
    const uint8_t* table = blk + 130;
    for (int e = 0; e < n_ov; ++e) {
        const int idx = (int)table[2 * e];
        if ((idx >> 4) != lig) continue;
        const int slot = idx & 15;
        const int j    = slot >> 1;
        const uint32_t word = (j < 4) ? w0 : w1;
        const int sh   = 8 * (j & 3) + ((slot & 1) ? 4 : 0);
        const int bulk = (int)((word >> sh) & 0xf) << 28 >> 28;
        const int val  = (int)(int8_t)table[2 * e + 1];
        s += (float)(val - bulk) * X[gi * 256 + idx];
    }
    return s;
}

__launch_bounds__(256) extern "C" __global__ void gemv_oq_compact_grouped_mw(
    const uint8_t* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int group /* == 256 */, int block_stride
) {
    const int wave  = threadIdx.x >> 5;
    const int waves = blockDim.x >> 5;
    const int row   = blockIdx.x * waves + wave;
    if (row >= M) return;

    const int lane = threadIdx.x & 31;
    const int ng   = K / 256;
    const int gir  = lane >> 4;   // which of the two groups in flight
    const int lig  = lane & 15;   // 16 weights within the group
    const int n_ov = (block_stride - 130) >> 1;
    const int elt  = lig * 16;

    const uint8_t* w_row = W + (long long)row * ng * (long long)block_stride;

    float facc = 0.0f;
    int r = 0;
    for (; r + 4 <= ng; r += 4) {
        const int gA = r + gir, gB = r + 2 + gir;
        const uint8_t* blkA = w_row + (long long)gA * block_stride;
        const uint8_t* blkB = w_row + (long long)gB * block_stride;
        // Both weight loads issued before either is consumed.
        uint32_t a[2], b[2];
        __builtin_memcpy(a, blkA + 2 + lig * 8, 8);
        __builtin_memcpy(b, blkB + 2 + lig * 8, 8);
        const float swA = (float)(*(const _Float16*)blkA);
        const float swB = (float)(*(const _Float16*)blkB);
        facc += oqc_group(a[0], a[1], blkA, X, gA, elt, lig, n_ov) * swA;
        facc += oqc_group(b[0], b[1], blkB, X, gB, elt, lig, n_ov) * swB;
    }
    for (; r < ng; r += 2) {
        const int gi = r + gir;
        const uint8_t* blk = w_row + (long long)gi * block_stride;
        uint32_t v[2];
        __builtin_memcpy(v, blk + 2 + lig * 8, 8);
        const float sw = (float)(*(const _Float16*)blk);
        facc += oqc_group(v[0], v[1], blk, X, gi, elt, lig, n_ov) * sw;
    }

    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down(facc, off);
    if (lane == 0) Y[row] = facc;
}
