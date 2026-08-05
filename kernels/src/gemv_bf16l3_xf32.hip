// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

// BF16L3 weight x F32 vector -> F32. The mixed-precision form of gemv_bf16l3.
//
// gemv_bf16l3 takes a BF16 activation and emits BF16 logits. For an lm_head
// both are avoidable losses: staging x down costs rel rms 3.4e-4, and rounding
// the logits discards precision the sampler wants. x is K elements against an
// M x K weight read, so reading it as f32 is free on bandwidth.
//
// Everything else — the plane layout, the wave-wide escape prefix sum, the
// branch on escape — is gemv_bf16l3's, unchanged. Measured at 128256 x 2048 on
// gfx1151: gemv_bf16l3 1.92 ms against gemv_bf16_xf32's 3.20 ms, the packed
// form being both 1.396x smaller (376.3 MB vs 525.3 MB) and better at
// saturating (196.3 vs 164.1 GB/s).
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
extern "C" __global__ void gemv_bf16l3_xf32(
    const unsigned char* __restrict__ W,   // BF16L3 payload of the [M, K] matrix
    const float*         __restrict__ x,   // [K] F32
    float*               __restrict__ y,   // [M] F32
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
        const float* __restrict__ xp = x + (size_t)j * BF16L3_BLOCK + 8 * tid;
        const float4 xv0 = *(const float4* __restrict__)(xp);
        const float4 xv1 = *(const float4* __restrict__)(xp + 4);

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
        const float xb[8] = { xv0.x, xv0.y, xv0.z, xv0.w,
                              xv1.x, xv1.y, xv1.z, xv1.w };
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
            sum += bf16_to_f32(w) * xb[s];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset);
    if (tid == 0) y[row] = sum;
}
