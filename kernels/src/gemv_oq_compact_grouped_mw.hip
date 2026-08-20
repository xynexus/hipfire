// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Multi-wave + MULTI-ROW compact decode GEMV. Identical math to
// `gemv_oq_compact_grouped_v2` and the same overlay contract (duplicates
// resolved at load by `normalize_compact_overlays`); what changes is how work is
// packed.
//
// Two levers, both about the ACTIVATION, which is what the inner loop actually
// spends its load slots on. Per group a lane issues ONE 8-byte weight load and
// FOUR 16-byte activation loads — so at one row per wave, 4 of every 5 loads are
// re-reading the same tiny activation that every other row already read.
//
//   * WAVES waves per workgroup: RDNA caps workgroups per CU, so one wave per
//     workgroup leaves resident waves unused. Worth +1.9% end-to-end at WAVES=2
//     (10.40 -> 10.60 tok/s, interleaved A/B, reproducible to the digit).
//
// MULTI-ROW WAS TRIED AND LOST, so it is not re-attempted. Hoisting the four
// activation loads out and sharing them across 2 or 4 rows per wave cuts
// activation load issue by that factor, and measured 10.60 -> 9.70 (rows=2) and
// 8.70 (rows=4) end-to-end. Fewer rows per wave means more workgroups, and that
// parallelism is worth more here than the load-issue saving — the activation was
// never the constraint. Standalone bandwidth agreed once the cache-resident
// shapes were discounted.
//
// Ruled out before landing here, so neither is re-attempted: the block's 2-byte
// misalignment costs ~3% (measured, `bench_read_bw --align`, and an aligned +8
// offset scores the same as +2, so it is base-offset drift not misalignment);
// and fusing the small projections is ~1% (they are 0.9% of GEMV time).
//
//   W : compact blocks, [M, K/256] blocks of block_stride bytes
//   X : [K] f32 activation (batch=1, FWHT-rotated)
//   Y : [M] f32

#include <hip/hip_runtime.h>
#include <stdint.h>

#define OQCV2_LO(w, j) ((int)((((uint32_t)(w)) >> (8 * (j))) & 0xf) << 28 >> 28)
#define OQCV2_HI(w, j) ((int)((((uint32_t)(w)) >> (8 * (j) + 4)) & 0xf) << 28 >> 28)

#define OQC_MR_KERNEL(NAME, ROWS)                                              \
__launch_bounds__(256) extern "C" __global__ void NAME(                        \
    const uint8_t* __restrict__ W,                                             \
    const float* __restrict__ X,                                               \
    float* __restrict__ Y,                                                     \
    int M, int K, int group /* == 256 */, int block_stride                     \
) {                                                                            \
    const int wave  = threadIdx.x >> 5;                                        \
    const int waves = blockDim.x >> 5;                                         \
    const int row0  = (blockIdx.x * waves + wave) * ROWS;                      \
    if (row0 >= M) return;                                                     \
    const int lane = threadIdx.x & 31;                                         \
    const int ng   = K / 256;                                                  \
    const int gir  = lane >> 4;                                                \
    const int lig  = lane & 15;                                                \
    const int n_ov = (block_stride - 130) >> 1;                                \
    const int elt  = lig * 16;                                                 \
                                                                               \
    const uint8_t* w_rows[ROWS];                                               \
    float facc[ROWS];                                                          \
    _Pragma("unroll")                                                          \
    for (int t = 0; t < ROWS; ++t) {                                           \
        const int rr = (row0 + t < M) ? (row0 + t) : (M - 1);                   \
        w_rows[t] = W + (long long)rr * ng * (long long)block_stride;           \
        facc[t] = 0.0f;                                                        \
    }                                                                          \
                                                                               \
    for (int r = 0; r < ng; r += 2) {                                          \
        const int gi = r + gir;                                                \
        const int xoff = gi * 256 + elt;                                       \
        /* Hoisted: one activation read serves every row below. */             \
        const float4 x0 = *(const float4*)(X + xoff);                          \
        const float4 x1 = *(const float4*)(X + xoff + 4);                      \
        const float4 x2 = *(const float4*)(X + xoff + 8);                      \
        const float4 x3 = *(const float4*)(X + xoff + 12);                     \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < ROWS; ++t) {                                       \
            const uint8_t* blk = w_rows[t] + (long long)gi * block_stride;      \
            const float sw = (float)(*(const _Float16*)blk);                   \
            uint32_t wv[2];                                                    \
            __builtin_memcpy(wv, blk + 2 + lig * 8, 8);                        \
            const uint32_t w0 = wv[0], w1 = wv[1];                             \
            float s = 0.0f;                                                    \
            s += (float)OQCV2_LO(w0, 0) * x0.x + (float)OQCV2_HI(w0, 0) * x0.y \
               + (float)OQCV2_LO(w0, 1) * x0.z + (float)OQCV2_HI(w0, 1) * x0.w;\
            s += (float)OQCV2_LO(w0, 2) * x1.x + (float)OQCV2_HI(w0, 2) * x1.y \
               + (float)OQCV2_LO(w0, 3) * x1.z + (float)OQCV2_HI(w0, 3) * x1.w;\
            s += (float)OQCV2_LO(w1, 0) * x2.x + (float)OQCV2_HI(w1, 0) * x2.y \
               + (float)OQCV2_LO(w1, 1) * x2.z + (float)OQCV2_HI(w1, 1) * x2.w;\
            s += (float)OQCV2_LO(w1, 2) * x3.x + (float)OQCV2_HI(w1, 2) * x3.y \
               + (float)OQCV2_LO(w1, 3) * x3.z + (float)OQCV2_HI(w1, 3) * x3.w;\
            const uint8_t* table = blk + 130;                                  \
            for (int e = 0; e < n_ov; ++e) {                                   \
                const int idx = (int)table[2 * e];                             \
                if ((idx >> 4) != lig) continue;                               \
                const int slot = idx & 15;                                     \
                const int j    = slot >> 1;                                    \
                const uint32_t word = (j < 4) ? w0 : w1;                       \
                const int sh   = 8 * (j & 3) + ((slot & 1) ? 4 : 0);           \
                const int bulk = (int)((word >> sh) & 0xf) << 28 >> 28;        \
                const int val  = (int)(int8_t)table[2 * e + 1];                \
                s += (float)(val - bulk) * X[gi * 256 + idx];                  \
            }                                                                  \
            facc[t] += s * sw;                                                 \
        }                                                                      \
    }                                                                          \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int t = 0; t < ROWS; ++t) {                                           \
        float v = facc[t];                                                     \
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off);        \
        if (lane == 0 && row0 + t < M) Y[row0 + t] = v;                        \
    }                                                                          \
}

OQC_MR_KERNEL(gemv_oq_compact_grouped_mw, 1)
