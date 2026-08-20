// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Compact-resident Opus DECODE GEMV, W4A8 (batch=1): the A8 twin of
// `gemv_oq_compact_grouped_v2`.
//
// Status: A8 REACHES the memory wall — 236 GB/s on o_proj against a 250 GB/s
// pure-read stream, 94% — but only when the sparse overlay is out of the way.
// At the production N_out=3 the overlay dominates BOTH this kernel and its A16
// sibling, and this one comes out behind. Not wired into a dispatch route yet
// for that reason.
//
//   N_out   A16 gate/up   A8 gate/up   A16 o_proj   A8 o_proj
//     1       197.7         171.2        250.2        236.4     <- bandwidth-bound
//     3       146.9          91.6        202.3        133.7     <- production
//     8        45.7          40.8         62.6         58.1
//
// The overlay loop is O(N_out^2) and divergent: every correction rescans the
// remaining table entries looking for a superseding duplicate, and the ownership
// test is a `continue`, so all 16 lanes of a group pay for every iteration even
// though one lane applies it. That is what 1 -> 8 costs 4x on both kernels, and
// it is the thing to fix before the dot matters. Deduping the overlay table
// offline (last-wins already applied) would delete the inner rescan outright.
//
// What was ruled out along the way, so it is not re-derived: the dot IS emitted
// (4x v_dot4_i32_iu8 in the ISA); this kernel uses FEWER resources than the A16
// sibling (100 VALU vs 127, 27 VGPRs vs 47, 8 loads vs 10, no spills); and
// breaking the four dots out of a single accumulator into independent ones
// changed nothing. The nibble widening was optimized twice — naive
// extract-and-repack, then SWAR + V_PERM_B32 — for no measurable difference,
// which is what rules the widening out too.
//
// Numerics track `gemv_oq_compact_grouped_v2` exactly, including the overlay
// contract: an overlay REPLACES the bulk nibble, so it is applied as a
// difference, and a duplicate index means LAST WINS — an entry a later one
// supersedes must be skipped or the correction lands on v1+v2-bulk.
//
// On-disk block, one per (row, group) of 256 weights:
//   [ f16 scale | 128 packed signed-int4 | N_out * (u8 idx_in_group, i8 val) ]
//   block_stride = 130 + 2*N_out
//
//   W  : compact blocks, [M, K/256] blocks of block_stride bytes
//   Xq : [K] int8 activation (FWHT-rotated, per-group absmax quantized)
//   Xs : [K/256] f32 per-group activation scales
//   Y  : [M] f32

#include <hip/hip_runtime.h>
#include <stdint.h>

#define OQC8_LO(w, j) ((int)((((uint32_t)(w)) >> (8 * (j))) & 0xf) << 28 >> 28)
#define OQC8_HI(w, j) ((int)((((uint32_t)(w)) >> (8 * (j) + 4)) & 0xf) << 28 >> 28)

// SWAR nibble widening. Extracting 8 nibbles one at a time and re-packing them
// costs more ALU than V_DOT4 saves — measured 0.59-0.84x against the A16 kernel
// before this. These do 4 at a time in three ops each.
//
// Sign-extension without a per-byte branch or a borrowing subtract: a 4-bit
// value is negative exactly when bit 3 is set, and 0x08 * 0x1E == 0xF0 stays
// inside its byte, so the multiply broadcasts the sign into the high nibble of
// each byte independently.
#define OQC8_SEXT(t) ((t) | (((t) & 0x08080808u) * 0x1Eu))
// Low nibbles of the 4 bytes  -> elements 0,2,4,6 of the run.
#define OQC8_WIDEN_LO(w) OQC8_SEXT((w) & 0x0f0f0f0fu)
// High nibbles                -> elements 1,3,5,7.
#define OQC8_WIDEN_HI(w) OQC8_SEXT(((w) >> 4) & 0x0f0f0f0fu)

__launch_bounds__(32) extern "C" __global__ void gemv_oq_compact_w8a8_grouped(
    const uint8_t* __restrict__ W,
    const signed char* __restrict__ Xq,
    const float* __restrict__ Xs,
    float* __restrict__ Y,
    int M, int K, int group /* == 256 */, int block_stride
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x;   // wave32
    const int ng   = K / 256;       // groups (even)
    const int gir  = lane >> 4;     // which of the two groups in flight
    const int lig  = lane & 15;     // 0..15 : 16 weights within the group
    const int n_ov = (block_stride - 130) >> 1;
    const int elt  = lig * 16;      // weight/activation offset within the group

    // SPLIT PLANES: this tensor holds ALL nibble groups first, then all
    // [f16 scale][overlay table] side records — see
    // `hipfire_runtime::oq8_arch::split_compact_planes`. Same bytes, same 4.25
    // bits; what it buys is that a nibble row starts at ng*(group/2), a multiple
    // of 128 at G=256, so EVERY row is cache-line aligned. The interleaved
    // 136-byte block left only every fourth row aligned, worth ~3.7% of achieved
    // bandwidth (examples/bench_oq_layout: 226.4 -> 234.8 GB/s).
    const int side_stride = block_stride - 128;
    const uint8_t* n_row = W + (long long)row * ng * 128LL;
    const uint8_t* s_row = W + (long long)M * ng * 128LL
                             + (long long)row * ng * (long long)side_stride;

    float facc = 0.0f;
    for (int r = 0; r < ng; r += 2) {
        const int gi = r + gir;
        const uint8_t* sb = s_row + (long long)gi * side_stride;
        const float sw = (float)(*(const _Float16*)sb);

        uint32_t wv[2];
        __builtin_memcpy(wv, n_row + (long long)gi * 128 + lig * 8, 8);
        const uint32_t w0 = wv[0], w1 = wv[1];

        // SWAR-widen into even/odd halves. That leaves the weights INTERLEAVED
        // (elements 0,2,4,6 then 1,3,5,7), so the activation is de-interleaved to
        // match with two V_PERM_B32 instead of the weights being re-packed into
        // sequential order — the permute is one instruction where the re-pack was
        // seven.
        const uint32_t we0 = OQC8_WIDEN_LO(w0), wo0 = OQC8_WIDEN_HI(w0);
        const uint32_t we1 = OQC8_WIDEN_LO(w1), wo1 = OQC8_WIDEN_HI(w1);

        const int xoff = gi * 256 + elt;
        const uint32_t xa = *(const uint32_t*)(Xq + xoff);
        const uint32_t xb = *(const uint32_t*)(Xq + xoff + 4);
        const uint32_t xc = *(const uint32_t*)(Xq + xoff + 8);
        const uint32_t xd = *(const uint32_t*)(Xq + xoff + 12);

        // v_perm_b32(hi, lo, sel): byte selectors 0-3 index `lo`, 4-7 index `hi`.
        // 0x06040200 gathers the even bytes of the 8-byte run, 0x07050301 the odd.
        const uint32_t xe0 = __builtin_amdgcn_perm(xb, xa, 0x06040200u);
        const uint32_t xo0 = __builtin_amdgcn_perm(xb, xa, 0x07050301u);
        const uint32_t xe1 = __builtin_amdgcn_perm(xd, xc, 0x06040200u);
        const uint32_t xo1 = __builtin_amdgcn_perm(xd, xc, 0x07050301u);

        // sudot4 = V_DOT4_I32_IU8 with per-operand sign flags; both signed here.
        // FOUR INDEPENDENT accumulators, not one chained: each v_dot4 carries its
        // own accumulate operand, so writing them into a single `idot` serializes
        // the four on that register's latency with nothing to overlap. Integer
        // adds are exact and associative, so the final sum is identical.
        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        d0 = __builtin_amdgcn_sudot4(true, (int)we0, true, (int)xe0, d0, false);
        d1 = __builtin_amdgcn_sudot4(true, (int)wo0, true, (int)xo0, d1, false);
        d2 = __builtin_amdgcn_sudot4(true, (int)we1, true, (int)xe1, d2, false);
        d3 = __builtin_amdgcn_sudot4(true, (int)wo1, true, (int)xo1, d3, false);
        int idot = (d0 + d1) + (d2 + d3);

        // Overlay corrections stay INTEGER — (val - bulk) * xq is exact in i32,
        // so the whole group still needs only one f32 rescale.
        const uint8_t* table = sb + 2;
        for (int e = 0; e < n_ov; ++e) {
            const int idx = (int)table[2 * e];
            if ((idx >> 4) != lig) continue;
            // No supersede rescan: `normalize_compact_overlays` resolves duplicate
            // indices at load, neutralising a superseded entry by setting its value
            // to the bulk nibble under it, so its difference is exactly zero and it
            // can be applied unconditionally. That rescan was O(N_out^2) and
            // divergent, and it is what held this kernel off the memory wall.
            const int slot = idx & 15;
            const int j    = slot >> 1;
            const uint32_t word = (j < 4) ? w0 : w1;
            const int sh   = 8 * (j & 3) + ((slot & 1) ? 4 : 0);
            const int bulk = (int)((word >> sh) & 0xf) << 28 >> 28;
            const int val  = (int)(int8_t)table[2 * e + 1];
            idot += (val - bulk) * (int)Xq[gi * 256 + idx];
        }

        facc += (float)idot * (sw * Xs[gi]);
    }

    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down(facc, off);
    if (lane == 0) Y[row] = facc;
}
