// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// GEMV against BF16L3-compressed weights: y[m] = sum_k W[m,k] * x_bf16[k],
// with W stored as the lossless 3-bit-exponent-LUT format and decoded in
// registers. Weights are never materialised as BF16 in VRAM, so the ~1.38x
// size reduction is a reduction in weight *bandwidth* — which is what decode
// is bound by on a UMA APU.
//
// wave32, zero LDS, F32 accumulate, BF16 round on store. grid=[M], block=[32],
// matching gemv_bf16_bf16 so the two are directly comparable.
//
// Requires K % 256 == 0: the format blocks the flattened tensor, so rows only
// start on a block boundary when K is a whole number of blocks. The dispatch
// caller enforces this.
//
// Layout constants MUST mirror hipfire_primitives::bf16_lut3::Layout.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define BF16L3_BLOCK       256
#define BF16L3_LUT_STRIDE    8
#define BF16L3_CODE_STRIDE  96   // 256 codes x 3 bits
#define BF16L3_HEADER       16
#define BF16L3_ESCAPE        7

__device__ __forceinline__ float bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
__device__ __forceinline__ unsigned short f32_to_bf16_rne(float f) {
    unsigned int b = __float_as_uint(f);
    if ((b & 0x7fffffffu) > 0x7f800000u) return (unsigned short)(b >> 16); // NaN
    unsigned int bias = 0x7fffu + ((b >> 16) & 1u);
    return (unsigned short)((b + bias) >> 16);
}
__device__ __forceinline__ size_t align16(size_t x) { return (x + 15) & ~(size_t)15; }

__launch_bounds__(32)
extern "C" __global__ void gemv_bf16l3(
    const unsigned char*  __restrict__ W,   // BF16L3 payload of the [M, K] matrix
    const unsigned short* __restrict__ x,   // [K] BF16
    unsigned short*       __restrict__ y,   // [M] BF16
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;            // lane 0..31

    // Plane bases, recomputed from n exactly as the host-side Layout does.
    const size_t n        = (size_t)M * (size_t)K;
    const size_t nb       = (n + BF16L3_BLOCK - 1) / BF16L3_BLOCK;
    const size_t esc_tab  = BF16L3_HEADER;
    const size_t lut_off  = esc_tab + align16(4 * nb);
    const size_t mant_off = lut_off + align16(BF16L3_LUT_STRIDE * nb);
    const size_t code_off = mant_off + BF16L3_BLOCK * nb;
    const size_t esc_off  = code_off + BF16L3_CODE_STRIDE * nb;

    const int blocks_per_row = K / BF16L3_BLOCK;
    float sum = 0.0f;

    for (int j = 0; j < blocks_per_row; ++j) {
        const size_t blk = (size_t)row * (size_t)blocks_per_row + (size_t)j;

        // The one indexed load in the whole format; issue it before the
        // dependent work so it overlaps the code/mantissa fetches.
        const unsigned int esc_start =
            ((const unsigned int* __restrict__)(W + esc_tab))[blk];

        // The whole 8-entry LUT as two registers. Indexing it in memory instead
        // would be a divergent dependent byte load per element — 256 load
        // instructions per block against 8 bytes. Here it is a shift and a
        // select.
        const uint2 lv =
            *(const uint2* __restrict__)(W + lut_off + BF16L3_LUT_STRIDE * blk);
        const unsigned char* __restrict__ mant  = W + mant_off + BF16L3_BLOCK * blk;
        const unsigned char* __restrict__ codes = W + code_off + BF16L3_CODE_STRIDE * blk;
        const unsigned char* __restrict__ esc   = W + esc_off  + esc_start;

        // Lane t owns elements [8t, 8t+8) — exactly the 3 code bytes at 3t.
        // That correspondence is why the codes are packed 8-per-3-bytes.
        unsigned int acc = (unsigned int)codes[3 * tid]
                         | ((unsigned int)codes[3 * tid + 1] << 8)
                         | ((unsigned int)codes[3 * tid + 2] << 16);

        // 8 mantissa bytes as one aligned 8-byte load: the wave then covers the
        // block's 256 mantissa bytes contiguously instead of at stride 8.
        const uint2 mv = *(const uint2* __restrict__)(mant + 8 * tid);
        // 8 activations as one aligned 16-byte load, same reason.
        const uint4 xv = *(const uint4* __restrict__)(x + (size_t)j * BF16L3_BLOCK + 8 * tid);

        // How many of this lane's 8 elements escaped.
        int local = 0;
        #pragma unroll
        for (int s = 0; s < 8; ++s) local += (((acc >> (3 * s)) & 7u) == BF16L3_ESCAPE);

        // Wave-wide exclusive prefix sum -> this lane's cursor into the block's
        // escape run. Replaces an O(i) per-element rank scan with 5 shuffles.
        int scan = local;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int up = __shfl_up(scan, d);
            if (tid >= d) scan += up;
        }
        int cursor = scan - local;

        const unsigned int mb[2] = { mv.x, mv.y };
        const unsigned int xb[4] = { xv.x, xv.y, xv.z, xv.w };
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const unsigned int code = (acc >> (3 * s)) & 7u;
            const unsigned int lsel = (code < 4 ? lv.x : lv.y) >> (8 * (code & 3));
            // Keep the escape read behind the branch. A branchless variant that
            // always loads esc[cursor] and selects measured *worse* (1.086x vs
            // 1.196x): at a ~2.4% escape rate, 8 unconditional byte loads per
            // lane per block cost more than the divergence they remove.
            const unsigned int e = (code == BF16L3_ESCAPE) ? (unsigned int)esc[cursor++]
                                                           : (lsel & 0xffu);
            const unsigned int sm = (mb[s >> 2] >> (8 * (s & 3))) & 0xffu;
            const unsigned short w =
                (unsigned short)(((sm & 0x80u) << 8) | (e << 7) | (sm & 0x7fu));
            const unsigned short xs =
                (unsigned short)((xb[s >> 1] >> (16 * (s & 1))) & 0xffffu);
            sum += bf16_to_f32(w) * bf16_to_f32(xs);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset);
    if (tid == 0) y[row] = f32_to_bf16_rne(sum);
}
