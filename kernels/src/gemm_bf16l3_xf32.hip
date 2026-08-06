// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

// BF16L3 weight x F32 activations -> F32, batched over N columns.
//
// The batched form of gemv_bf16l3_xf32. Same plane layout, same wave-wide
// escape prefix sum, same branch on escape — the ONLY change is that the
// decoded weight is reused across up to BF16L3_NT activation columns instead of
// one.
//
// Why this exists: the GEMV form reads the entire packed head per output
// position. On an lm_head that is 367 MB per call, and a KLD reference build
// calls it once per scored position — 1023 per 2048-token chunk, 748 GB of
// weight traffic, measured at 1.99 s and 62% of all GPU time in a reference
// build (rocprofv3, gfx1151). Decoding is per weight element and independent of
// the activation, so a tile of columns amortizes both the read and the decode:
// BF16L3_NT columns per pass cuts the weight traffic by that factor.
//
// The host loops column tiles, passing `col_base`; `n_cols` is the live width of
// the final partial tile. Accumulators are registers, so BF16L3_NT trades
// register pressure against the amortization factor — 8 keeps the same 32-lane
// launch bounds as the GEMV.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define BF16L3_BLOCK       256
#define BF16L3_LUT_STRIDE    8
#define BF16L3_CODE_STRIDE  96   // 256 codes x 3 bits
#define BF16L3_HEADER       16
#define BF16L3_ESCAPE        7
#define BF16L3_NT            8   // activation columns per pass

__device__ __forceinline__ float bf16l3_gemm_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
__device__ __forceinline__ size_t bf16l3_gemm_align16(size_t x) {
    return (x + 15) & ~(size_t)15;
}

__launch_bounds__(32)
extern "C" __global__ void gemm_bf16l3_xf32(
    const unsigned char* __restrict__ W,   // BF16L3 payload of the [M, K] matrix
    const float*         __restrict__ x,   // [N, K] F32, row-major per column
    float*               __restrict__ y,   // [N, M] F32, row-major per column
    int M, int K, int N, int col_base, int n_cols
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;            // lane 0..31

    // Plane bases, recomputed from n exactly as the host-side Layout does.
    const size_t n        = (size_t)M * (size_t)K;
    const size_t nb       = (n + BF16L3_BLOCK - 1) / BF16L3_BLOCK;
    const size_t esc_tab  = BF16L3_HEADER;
    const size_t lut_off  = esc_tab + bf16l3_gemm_align16(4 * nb);
    const size_t mant_off = lut_off + bf16l3_gemm_align16(BF16L3_LUT_STRIDE * nb);
    const size_t code_off = mant_off + BF16L3_BLOCK * nb;
    const size_t esc_off  = code_off + BF16L3_CODE_STRIDE * nb;

    const int blocks_per_row = K / BF16L3_BLOCK;

    float sum[BF16L3_NT];
    #pragma unroll
    for (int c = 0; c < BF16L3_NT; ++c) sum[c] = 0.0f;

    for (int j = 0; j < blocks_per_row; ++j) {
        const size_t blk = (size_t)row * (size_t)blocks_per_row + (size_t)j;

        const unsigned int esc_start =
            ((const unsigned int* __restrict__)(W + esc_tab))[blk];
        const uint2 lv =
            *(const uint2* __restrict__)(W + lut_off + BF16L3_LUT_STRIDE * blk);
        const unsigned char* __restrict__ mant  = W + mant_off + BF16L3_BLOCK * blk;
        const unsigned char* __restrict__ codes = W + code_off + BF16L3_CODE_STRIDE * blk;
        const unsigned char* __restrict__ esc   = W + esc_off  + esc_start;

        // Lane t owns elements [8t, 8t+8) — exactly the 3 code bytes at 3t.
        unsigned int acc = (unsigned int)codes[3 * tid]
                         | ((unsigned int)codes[3 * tid + 1] << 8)
                         | ((unsigned int)codes[3 * tid + 2] << 16);
        const uint2 mv = *(const uint2* __restrict__)(mant + 8 * tid);

        // How many of this lane's 8 elements escaped.
        int local = 0;
        #pragma unroll
        for (int s = 0; s < 8; ++s) local += (((acc >> (3 * s)) & 7u) == BF16L3_ESCAPE);

        // Wave-wide exclusive prefix sum -> this lane's cursor into the block's
        // escape run.
        int scan = local;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int up = __shfl_up(scan, d);
            if (tid >= d) scan += up;
        }
        int cursor = scan - local;

        const unsigned int mb[2] = { mv.x, mv.y };
        const size_t k0 = (size_t)j * BF16L3_BLOCK + 8 * tid;

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const unsigned int code = (acc >> (3 * s)) & 7u;
            const unsigned int lsel = (code < 4 ? lv.x : lv.y) >> (8 * (code & 3));
            const unsigned int e = (code == BF16L3_ESCAPE) ? (unsigned int)esc[cursor++]
                                                           : (lsel & 0xffu);
            const unsigned int sm = (mb[s >> 2] >> (8 * (s & 3))) & 0xffu;
            const unsigned short wq =
                (unsigned short)(((sm & 0x80u) << 8) | (e << 7) | (sm & 0x7fu));
            // Decoded ONCE, spent across the column tile. This reuse is the
            // entire point of the kernel.
            const float wf = bf16l3_gemm_to_f32(wq);
            #pragma unroll
            for (int c = 0; c < BF16L3_NT; ++c) {
                if (c < n_cols) {
                    sum[c] += wf * x[(size_t)(col_base + c) * (size_t)K + k0 + s];
                }
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < BF16L3_NT; ++c) {
        float v = sum[c];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down(v, offset);
        if (tid == 0 && c < n_cols) {
            y[(size_t)(col_base + c) * (size_t)M + row] = v;
        }
    }
}
