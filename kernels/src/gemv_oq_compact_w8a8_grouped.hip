// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Compact-resident Opus DECODE GEMV, W4A8 (batch=1): the A8 twin of
// `gemv_oq_compact_grouped_v2`.
//
// ⚠️ MEASURED SLOWER THAN THE A16 SIBLING. Kept because it is exact and because
// the reason is worth not re-deriving. Not wired into any dispatch route.
//
//   shape                    A16        A8
//   gate/up [17408, 5120]  145 GB/s   93 GB/s   0.64x
//   down    [5120, 17408]  128        93        0.72x
//   o_proj  [5120,  5120]  204       153        0.75x
//
// The premise was that A8 doubles dot throughput, so the kernel would stop being
// issue-bound and fall back onto the memory wall. It does not, for a batch-1
// reason: the ACTIVATION is a rounding error in the byte count. Per group a lane
// reads block_stride (136) bytes of weights against 16 bytes of int8 activation;
// quantizing the activation saves 3*K bytes against M*K*0.53 of weights. There is
// no bandwidth to recover, so A8 can only pay in compute — and for int4 weights
// it COSTS compute, because V_DOT4 needs int8 lanes and the nibbles must be
// widened first. `gemv_oq8_w8a8_grouped` wins on ZAYA precisely because its
// weights are already dense int8 and it widens nothing.
//
// The widening was optimized before concluding this: extracting and re-packing 16
// nibbles measured 0.59-0.84x, and SWAR widening plus a V_PERM_B32 activation
// de-interleave measured 0.64-0.75x. Nearly unchanged, which is what rules the
// widening out as the cost and points at the batch-1 argument above.
//
// Where A8 DOES pay is batched, where the activation is B*K and the int8 WMMA
// path far outruns f32 — which is why the batched Opus GEMM already uses W8A8.
//
// oq4.25++ is a MIXED-precision format — 4-bit bulk plus a sparse int8 outlier
// overlay, decoded together — so the weights arrive as int4 and the dot needs
// int8 lanes. The nibbles are therefore widened into packed int8 before the dot;
// there is no int4xint8 dot on this ISA (`v_dot8_i32_iu4` needs BOTH operands
// int4). The widening is register work that replaces 16 f32 converts.
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

    const uint8_t* w_row = W + (long long)row * ng * (long long)block_stride;

    float facc = 0.0f;
    for (int r = 0; r < ng; r += 2) {
        const int gi = r + gir;
        const uint8_t* blk = w_row + (long long)gi * block_stride;
        const float sw = (float)(*(const _Float16*)blk);

        // 8 bytes of nibbles = this lane's 16 weights. Offset is 2 mod 8; gfx11
        // takes the unaligned load natively rather than decaying to byte reads.
        uint32_t wv[2];
        __builtin_memcpy(wv, blk + 2 + lig * 8, 8);
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
        int idot = 0;
        idot = __builtin_amdgcn_sudot4(true, (int)we0, true, (int)xe0, idot, false);
        idot = __builtin_amdgcn_sudot4(true, (int)wo0, true, (int)xo0, idot, false);
        idot = __builtin_amdgcn_sudot4(true, (int)we1, true, (int)xe1, idot, false);
        idot = __builtin_amdgcn_sudot4(true, (int)wo1, true, (int)xo1, idot, false);

        // Overlay corrections stay INTEGER — (val - bulk) * xq is exact in i32,
        // so the whole group still needs only one f32 rescale.
        const uint8_t* table = blk + 130;
        for (int e = 0; e < n_ov; ++e) {
            const int idx = (int)table[2 * e];
            if ((idx >> 4) != lig) continue;
            bool superseded = false;
            for (int e2 = e + 1; e2 < n_ov; ++e2) {
                if ((int)table[2 * e2] == idx) { superseded = true; break; }
            }
            if (superseded) continue;
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
