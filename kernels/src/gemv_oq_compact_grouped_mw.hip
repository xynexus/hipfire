// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Multi-wave twin of `gemv_oq_compact_grouped_v2`. Identical math, identical
// overlay contract (duplicates resolved at load by `normalize_compact_overlays`);
// the ONLY change is that a workgroup carries several waves, each owning one
// output row, instead of one wave per workgroup.
//
// Why: this kernel is 92 % of Opus decode and every large launch runs at
// ~155-161 GB/s against a 250 GB/s pure-read wall. It is not the inner loop —
// the grid is one wave per workgroup, and RDNA caps workgroups per CU, so
// resident waves are left on the table. Widening the block is the same lever
// that bought +12 % on gemm_oq8_grouped_wmma.
//
//   W  : compact blocks, [M, K/256] blocks of block_stride bytes
//   X  : [K] f32 activation (batch=1, FWHT-rotated)
//   Y  : [M] f32

#include <hip/hip_runtime.h>
#include <stdint.h>

#define OQCV2_LO(w, j) ((int)((((uint32_t)(w)) >> (8 * (j))) & 0xf) << 28 >> 28)
#define OQCV2_HI(w, j) ((int)((((uint32_t)(w)) >> (8 * (j) + 4)) & 0xf) << 28 >> 28)

__launch_bounds__(256) extern "C" __global__ void gemv_oq_compact_grouped_mw(
    const uint8_t* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int group /* == 256 */, int block_stride
) {
    // One wave per output row, SEVERAL waves per workgroup. Same work and same
    // numerics as the v2 kernel; only the packing into workgroups changes.
    // RDNA caps workgroups-per-CU, so a grid of one-wave workgroups leaves waves
    // on the table — widening the block lifts occupancy without touching the
    // inner loop. Block width comes from blockDim.x, so the caller sweeps it.
    const int wave  = threadIdx.x >> 5;
    const int waves = blockDim.x >> 5;
    const int row   = blockIdx.x * waves + wave;
    if (row >= M) return;
    const int lane = threadIdx.x & 31;   // lane within this wave
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

        // Overlays REPLACE the bulk nibble, so add the difference, applied by
        // whichever lane owns the position. Duplicate indices are resolved at
        // LOAD (`normalize_compact_overlays`) rather than rescanned here — the
        // rescan was O(N_out^2) and divergent, and measured as the thing holding
        // this kernel off the memory wall (1 -> 8 corrections cost ~4x).
        const uint8_t* table = blk + 130;
        for (int e = 0; e < n_ov; ++e) {
            const int idx = (int)table[2 * e];
            if ((idx >> 4) != lig) continue;
            // No supersede rescan: duplicate indices are resolved at load by
            // `normalize_compact_overlays`, which neutralises a superseded entry
            // (value := the bulk nibble under it, so the difference is zero).
            const int slot = idx & 15;
            const int j    = slot >> 1;
            const uint32_t word = (j < 4) ? w0 : w1;
            const int sh   = 8 * (j & 3) + ((slot & 1) ? 4 : 0);
            const int bulk = (int)((word >> sh) & 0xf) << 28 >> 28;
            const int val  = (int)(int8_t)table[2 * e + 1];
            s += (float)(val - bulk) * X[gi * 256 + idx];
        }

        facc += s * sw;
    }

    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down(facc, off);
    if (lane == 0) Y[row] = facc;
}
