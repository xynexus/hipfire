// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Compact-resident Opus SMALL-BATCH GEMM: Y[b, m] = sum_k W[m,k] * Xq[b,k].
//
// `gemm_oq_compact_grouped_wmma` tiles B by 16, so at spec-decode's verify width
// (B = K+1, typically 8) it computes a 16-wide WMMA tile for 8 useful columns and
// has no N to amortize the int4 decode + overlay scan over. Measured on
// Qwen3.8-27B oq4.25++, a B=8 verify costs 64 us per draft token against plain
// decode's 66 — verify is barely cheaper per token than just decoding, which is
// why spec-decode loses on this family.
//
// This reads each weight row ONCE, exactly like the decode GEMV, and accumulates
// B columns against it: one weight sweep for all B tokens instead of one sweep
// per token. Deliberately the SIMPLEST correct shape — ONE group per wave round,
// 32 lanes x 8 weights = 256 — so there is no lane-group split and no ng
// divisibility requirement.
//
//   W  : compact SPLIT-PLANE blocks (all nibbles, then all [f16 scale][table])
//   Xq : [B, K] int8   Xs : [B, K/256] f32   Y : [B, M] f32

#include <hip/hip_runtime.h>
#include <stdint.h>

#define OQCM_MAX_B 16

__launch_bounds__(256) extern "C" __global__ void gemv_oq_compact_multicol(
    const uint8_t* __restrict__ W,
    const int8_t*  __restrict__ Xq,
    const float*   __restrict__ Xs,
    float*         __restrict__ Y,
    int M, int K, int B, int block_stride
) {
    const int lane = threadIdx.x & 31;
    const int wpb  = blockDim.x >> 5;
    const int wid  = blockIdx.x * wpb + (threadIdx.x >> 5);
    const int nw   = gridDim.x * wpb;
    const int ng   = K / 256;
    const int n_ov = (block_stride - 130) >> 1;

    const int side_stride = block_stride - 128;
    const long long side_base = (long long)M * ng * 128LL;

    for (int row = wid; row < M; row += nw) {
        const uint8_t* n_row = W + (long long)row * ng * 128LL;
        const uint8_t* s_row =
            W + side_base + (long long)row * ng * (long long)side_stride;

        float facc[OQCM_MAX_B];
        #pragma unroll
        for (int b = 0; b < OQCM_MAX_B; ++b) facc[b] = 0.0f;

        for (int g = 0; g < ng; ++g) {
            const uint8_t* sb = s_row + (long long)g * side_stride;
            const uint8_t* table = sb + 2;
            const float sw = (float)(*(const _Float16*)sb);

            // 4 bytes = this lane's 8 weights. 32 lanes x 8 = 256 = one group.
            uint32_t w4;
            __builtin_memcpy(&w4, n_row + (long long)g * 128 + lane * 4, 4);
            int8_t q[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                q[2 * j]     = (int8_t)((int)((w4 >> (8 * j)) & 0xf) << 28 >> 28);
                q[2 * j + 1] = (int8_t)((int)((w4 >> (8 * j + 4)) & 0xf) << 28 >> 28);
            }

            // Overlays REPLACE the bulk nibble and the loader ZEROED it, so a
            // correction is val*x[idx] and needs no lane's registers. Lane `lane`
            // takes entry `lane`; the wave reduction sums them wherever they land.
            const int has_ov = (lane < n_ov);
            const int oe     = has_ov ? lane : 0;
            const int oidx   = (int)table[2 * oe];
            const int oval   = has_ov ? (int)(int8_t)table[2 * oe + 1] : 0;

            const int elt = lane * 8;
            for (int b = 0; b < B; ++b) {
                const int8_t* xq = Xq + (long long)b * K + (long long)g * 256;
                int acc = 0;
                #pragma unroll
                for (int i = 0; i < 8; ++i) acc += (int)q[i] * (int)xq[elt + i];
                acc += oval * (int)xq[oidx];
                // Tables deeper than one entry per lane (32 here) never occur in
                // shipped artifacts, but must not be silently dropped.
                for (int e = lane + 32; e < n_ov; e += 32) {
                    acc += (int)(int8_t)table[2 * e + 1] * (int)xq[(int)table[2 * e]];
                }
                // Reduce the group's dot to an EXACT INTEGER before scaling, so
                // this is BIT-IDENTICAL to the WMMA GEMM rather than merely close.
                // Integer addition is associative, so the lane split does not
                // change the sum; scaling per-lane partials in f32 would. The
                // WMMA path accumulates the whole 256-wide group in int32 and
                // then does one `+= (float)iacc * sw * sx` per group, and this
                // now matches that order exactly.
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off);
                facc[b] += (float)acc * sw * Xs[(long long)b * ng + g];
            }
        }

        // Every lane holds the same facc after the per-group butterfly, so no
        // final reduction is needed.
        if (lane == 0) {
            for (int b = 0; b < B; ++b) Y[(long long)b * M + row] = facc[b];
        }
    }
}
