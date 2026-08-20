// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Fine pass of the two-stage lm_head when the head is COMPACT-RESIDENT Opus
// (qt 36) rather than bf16: exact W4A16 dot for a shortlist of vocab rows.
//
// The bf16 twin is `gemv_bf16_gather_f32`; this is the same contract —
// `out[idx[k]]` is written (not `out[k]`), so the scatter is fused and the
// caller pre-fills `out` with -inf — but the weight rows are OqPlusCompact
// blocks decoded in-kernel, bit-identically to `gemv_oq_compact_grouped_v3`.
//
// Deliberately the SIMPLE shape: one wave per selected row, all 32 lanes on one
// group at a time (32 lanes x 8 weights = 256). v2/v3's two- and four-groups-in-
// flight layouts exist to saturate DRAM across a whole [M, K] sweep; this pass
// touches ~128 rows, is a rounding error in the token, and has no shape
// constraint on ng to satisfy. Correctness and generality win here.
//
//   W    : compact blocks, [V, K/256] blocks of block_stride bytes
//   idx  : [K_sel] selected row indices, 0xFFFFFFFF = unused slot
//   X    : [K] f32 FWHT-rotated activation (the same one the coarse tier scored)
//   out  : [V] f32, pre-filled -inf; this kernel overwrites the shortlist

#include <hip/hip_runtime.h>
#include <stdint.h>

#define OQCG_LO(ww, j) ((int)((((uint32_t)(ww)) >> (8 * (j))) & 0xf) << 28 >> 28)
#define OQCG_HI(ww, j) ((int)((((uint32_t)(ww)) >> (8 * (j) + 4)) & 0xf) << 28 >> 28)

__launch_bounds__(32) extern "C" __global__ void gemv_oq_compact_gather_f32(
    const uint8_t* __restrict__ W,
    const unsigned int* __restrict__ idx,
    const float* __restrict__ X,
    float* __restrict__ out,
    int K_sel, int K, int block_stride
) {
    const int lane = threadIdx.x;   // wave32
    const int ng   = K / 256;
    const int n_ov = (block_stride - 130) >> 1;
    const int elt  = lane * 8;      // this lane's 8 weights within the group

    for (int s = blockIdx.x; s < K_sel; s += gridDim.x) {
        const unsigned int row = idx[s];
        if (row == 0xFFFFFFFFu) continue;   // sentinel: top-K over-allocates

        const uint8_t* w_row = W + (long long)row * ng * (long long)block_stride;
        float facc = 0.0f;

        for (int g = 0; g < ng; ++g) {
            const uint8_t* blk = w_row + (long long)g * block_stride;
            const float sw = (float)(*(const _Float16*)blk);

            uint32_t w0;
            __builtin_memcpy(&w0, blk + 2 + lane * 4, 4);

            const int xoff = g * 256 + elt;
            const float4 x0 = *(const float4*)(X + xoff);
            const float4 x1 = *(const float4*)(X + xoff + 4);

            float v = (float)OQCG_LO(w0, 0) * x0.x + (float)OQCG_HI(w0, 0) * x0.y
                    + (float)OQCG_LO(w0, 1) * x0.z + (float)OQCG_HI(w0, 1) * x0.w
                    + (float)OQCG_LO(w0, 2) * x1.x + (float)OQCG_HI(w0, 2) * x1.y
                    + (float)OQCG_LO(w0, 3) * x1.z + (float)OQCG_HI(w0, 3) * x1.w;

            // Same overlay contract as the streaming kernels: the loader zeroed
            // the bulk nibble under each entry, so a correction is val*x[idx].
            // Branchless for the same reason it is there — the divergent guard,
            // not the loads, is what the overlay costs.
            const uint8_t* table = blk + 130;
            {
                const int e   = (lane < n_ov) ? lane : 0;
                const int oi  = (int)table[2 * e];
                const int ov  = (lane < n_ov) ? (int)(int8_t)table[2 * e + 1] : 0;
                v += (float)ov * X[g * 256 + oi];
            }
            for (int e = lane + 32; e < n_ov; e += 32) {
                const int oi = (int)table[2 * e];
                const int ov = (int)(int8_t)table[2 * e + 1];
                v += (float)ov * X[g * 256 + oi];
            }

            facc += v * sw;
        }

        for (int off = 16; off > 0; off >>= 1) facc += __shfl_down(facc, off);
        if (lane == 0) out[row] = facc;
    }
}
