// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Opus Quant COMPACT-RESIDENT decode GEMV (batch=1):
//   y[m] = Σ_g sw[m,g] · Σ_{p∈g} qw[m,p] · x[g·G + p]
// reading OqPlusCompact (qt=36) blocks DIRECTLY. One wave32 per output row,
// ZERO LDS, one wave-shuffle reduction. Structure mirrors gemv_oq4_grouped.
//
// Why this exists: the compact path had no GEMV. Decode ran
// `gemm_oq_compact_grouped_wmma` at n=1 — a 16x16x16 WMMA with ONE useful
// column, carrying per-tile nibble decode and an overlay scan across 16 tiles.
// That made compact residency ~8% SLOWER than the int8-expanded path despite
// reading 136 B/group against 260, because at n=1 the kernel is bound by decode
// ALU and wasted matrix-tile capacity, not by weight bytes. Three memory-side
// fixes (wider nibble loads, spill removal, a hoisted scale plane) each measured
// no change, which is what pointed at the tile shape rather than the bytes.
//
// Two structural wins over the GEMM-at-n=1:
//   * No wasted tile: every lane's work lands in the single output column.
//   * The per-group weight scale is read ONCE per wave and broadcast, instead of
//     8 scattered 2-byte loads (one cache line each, since consecutive m-rows
//     sit n_groups*block_stride apart in the block stream).
//
// DECODE numerics, same call as gemv_oq4_grouped: at B=1 there is no WMMA, so
// the activation is NOT quantized to int8 — the unpacked weight multiplies the
// FULL-PRECISION f32 activation (W4A16 at decode). This is strictly more
// accurate than the expanded path's int8 activations, so results are close to
// but not bit-identical with it. The AWQ fold (W·s)·(x/s)=W·x is unaffected.
//
// On-disk block, one per (row, group) of G weights — unchanged, this kernel
// reads the artifact bytes as they sit:
//
//   [ f16 scale | G/2 packed signed-int4 | N_out * (u8 idx_in_group, i8 val) ]
//     2 bytes      G/2 bytes                2*N_out bytes
//   block_stride = (2 + G/2) + 2*N_out
//
// Overlay semantics match `oqplus_compact_to_oq8_combined`: an entry REPLACES
// the bulk nibble at its in-group index, and later duplicates win. Applied here
// as a correction (val - bulk)·x[idx] by whichever lane owns that position, so
// no lane needs a dynamically indexed local array.
//
//   W : compact blocks, [M, K/G] blocks of block_stride bytes
//   X : [K] f32 activation (batch=1, FWHT-rotated)
//   Y : [M] f32
//
// G must be 256 or 128 (u8 overlay index caps it at 256) and divide K, and
// G/2 must be a multiple of 32 so each lane owns a whole number of bytes.

#include <hip/hip_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ float oqcg_f16(const uint8_t* p) {
    return (float)(*(const _Float16*)p);
}

// Sign-extend a 4-bit code held in the low nibble.
static __device__ __forceinline__ int oqcg_sext4(int q) {
    return (q << 28) >> 28;
}

__launch_bounds__(32)
extern "C" __global__ void gemv_oq_compact_grouped(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ X,
    float*         __restrict__ Y,
    int M, int K, int group, int block_stride
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x; // 0..31

    const int ng     = K / group;
    const int header = 2 + (group >> 1);
    const int n_ov   = (block_stride - header) >> 1;
    const int per_lane = group >> 5;        // positions per lane: 8 (G256), 4 (G128)
    const int bpl      = per_lane >> 1;     // nibble bytes per lane: 4 or 2

    const uint8_t* w_row = W + (long long)row * ng * (long long)block_stride;

    float facc = 0.0f;
    for (int g = 0; g < ng; ++g) {
        const uint8_t* blk     = w_row + (long long)g * block_stride;
        const uint8_t* nibbles = blk + 2;
        const uint8_t* table   = blk + header;

        // The block's f16 scale: ONE load per group for the whole wave, not one
        // per lane per output row as the WMMA path needed.
        const float sw = oqcg_f16(blk);

        // `nibbles` is blk+2, so this is 2 mod 4 — state the width and let the
        // ISA handle the unaligned load rather than falling back to byte reads.
        uint32_t wv = 0;
        __builtin_memcpy(&wv, nibbles + lane * bpl, bpl);

        const int xoff = g * group + lane * per_lane;
        float gsum = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            if (n < per_lane) {
                const int qw = oqcg_sext4((int)((wv >> (4 * n)) & 0xf));
                gsum += (float)qw * X[xoff + n];
            }
        }

        // Overlays replace the bulk nibble, so add the difference. Only the lane
        // that owns the position contributes; every lane reads the (tiny, L1-hot)
        // table.
        //
        // Corrections SUM, so a duplicated index would apply twice and land on
        // v1+v2-bulk instead of the oracle's v2. Skip an entry that a later one
        // supersedes — O(n_ov^2), which is 9 comparisons at oq4.25++'s N_out=3.
        for (int e = 0; e < n_ov; ++e) {
            const int idx = (int)table[2 * e];
            bool superseded = false;
            for (int e2 = e + 1; e2 < n_ov; ++e2) {
                if ((int)table[2 * e2] == idx) { superseded = true; break; }
            }
            if (superseded) continue;
            if (idx / per_lane == lane) {
                const int slot = idx - lane * per_lane;
                const int bulk = oqcg_sext4((int)((wv >> (4 * slot)) & 0xf));
                const int val  = (int)(int8_t)table[2 * e + 1];
                gsum += (float)(val - bulk) * X[g * group + idx];
            }
        }

        facc += gsum * sw;
    }

    for (int off = 16; off > 0; off >>= 1) {
        facc += __shfl_down(facc, off);
    }
    if (lane == 0) {
        Y[row] = facc;
    }
}
