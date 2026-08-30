// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Compact-resident Opus EXACT W4A8, built from the first-principles ladder.
//
// The shipping wave64 kernel spends 234 VGPRs on register blocking and runs 3
// waves/SIMD, reaching 48.2 TOPS of iu4 issue. The ladder (docs/perf/
// gfx1151-gemm-levers.md §E) showed every ingredient it needs is cheap -- LDS
// operand supply 1.07x, per-group fold 1.08x, double-buffered staging 1.01x,
// real tiling 1.04x -- and that the SAME BM x BN tile split across more, smaller
// waves reaches 84.5 TOPS. This is that shape carrying the real contract.
//
//   BM=64 weight rows x BN=128 batch cols, BK=64, 8 wave32 waves as 2x4,
//   WMt=WNt=2, double-buffered LDS (A 2048 + Xh 4096 + Xl 4096) x2 = 20480 B.
//
// Exact W4A8 via two iu4 passes on ONE weight fragment: x = 16*x_hi + x_lo, hi
// digit signed, lo unsigned. iu4 is 16 cyc against iu8's 32 on gfx1151, so two
// passes cost the same WMMA cycles as one iu8 pass while feeding the compact
// nibbles RAW -- no unpack in the inner loop.
//
//   W    : compact SPLIT-PLANE blocks. Nibble plane [M][n_groups][128 B] first,
//          then side records [M][n_groups][side_stride] holding [f16 scale][table].
//   Xq   : [B, K] int8, as quantize_act_oq8_batched leaves it. Digit planes are
//          split during the LDS store (K/2 + K/2 == K, so no extra bytes).
//   Xs   : [B, K/256] f32     Y : [B, M] f32
//
// Does NOT apply the sparse overlay -- separate pass, as for both existing twins.
// Requires K % 256 == 0.
// Quant-basis contract: input activation `x` must already be rotated into the
// weights' quantization basis by the dispatch caller (rotate_x_*); a raw `x`
// runs but yields garbage. See kernels/AGENTS.md.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t i32x2 __attribute__((ext_vector_type(2)));
typedef int32_t i32x4 __attribute__((ext_vector_type(4)));
typedef int32_t i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
// hi pass: B operand SIGNED. lo pass: B operand UNSIGNED. A (weights) signed.
#define WH(a, b, c) __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, false)
#define WL(a, b, c) __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, false, b, c, false)

#define TBM 64
#define TBN 128
#define TBK 64
#define TWM 2            // wave grid rows
#define TWN 4            // wave grid cols
#define TWMt 2           // m-subtiles per wave
#define TWNt 2           // n-subtiles per wave
#define TAB (TBM * (TBK / 2))    // 2048
#define TXB (TBN * (TBK / 2))    // 4096 per digit plane
#define GROUP 256
#define STRIPS_PER_GROUP (GROUP / TBK)   // 4

// Pack the low nibble of each of 4 bytes into the low 16 bits, in order.
// Input must already be masked to 0x0f0f0f0f.
__device__ __forceinline__ uint32_t tpack4(uint32_t t) {
    t = (t | (t >> 4)) & 0x00ff00ffu;
    return (t | (t >> 8)) & 0x0000ffffu;
}
__device__ __forceinline__ float t_f16(const uint8_t* p) {
    return (float)(*(const _Float16*)p);
}

__launch_bounds__(256) extern "C" __global__ void gemm_oq_compact_iu4x2_tiled(
    const uint8_t* __restrict__ W, const int8_t* __restrict__ Xq,
    const float* __restrict__ Xs, float* __restrict__ Y,
    int M, int K, int B, int block_stride) {
    __shared__ uint8_t As[2][TAB];
    __shared__ uint8_t Xh[2][TXB];
    __shared__ uint8_t Xl[2][TXB];

    const int t = threadIdx.x;
    const int wid = t >> 5, wtid = t & 31;
    const int lane = wtid & 15, half = wtid >> 4;
    const int wm = wid / TWN, wn = wid % TWN;
    const int bm = blockIdx.x * TBM;
    const int bn = blockIdx.y * TBN;
    const int n_groups = K / GROUP;
    const int nstrip = K / TBK;
    const int side_stride = block_stride - 128;
    const long long side_base = (long long)M * n_groups * 128LL;

    i32x8 ah[TWMt][TWNt], al[TWMt][TWNt];
    f32x8 facc[TWMt][TWNt];
#pragma unroll
    for (int i = 0; i < TWMt; i++)
#pragma unroll
        for (int j = 0; j < TWNt; j++) {
            ah[i][j] = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
            al[i][j] = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
            facc[i][j] = (f32x8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }

    // ---- staging index math, hoisted ----
    // A: 64 rows x 32 B = 2048 B over 256 threads = 8 B each (one i32x2).
    const int a_row = t >> 2;              // 0..63
    const int a_col = (t & 3) * 8;         // byte within the 32 B strip
    const int a_src_row = (bm + a_row < M) ? (bm + a_row) : (M - 1);
    // X: 128 cols x 64 int8 over 256 threads = 2 threads per column, 32 B each.
    const int x_col = t >> 1;              // 0..127
    const int x_half = t & 1;              // which 32 B of the 64
    const int x_src_col = (bn + x_col < B) ? (bn + x_col) : (B - 1);

#define STAGE(buf, strip)                                                       \
    {                                                                           \
        const int g = (strip) / STRIPS_PER_GROUP;                               \
        const int sg = (strip) % STRIPS_PER_GROUP;                              \
        const uint8_t* ap = W + ((long long)a_src_row * n_groups + g) * 128      \
                          + sg * (TBK / 2) + a_col;                             \
        *(i32x2*)(As[buf] + a_row * (TBK / 2) + a_col) = *(const i32x2*)ap;      \
        const int8_t* xp = Xq + (long long)x_src_col * K + (long long)(strip) * TBK \
                         + x_half * 32;                                          \
        uint32_t hh[4], ll[4];                                                   \
        _Pragma("unroll") for (int q = 0; q < 4; ++q) {                          \
            const uint32_t d0 = *(const uint32_t*)(xp + q * 8);                  \
            const uint32_t d1 = *(const uint32_t*)(xp + q * 8 + 4);              \
            hh[q] = tpack4((d0 >> 4) & 0x0f0f0f0fu)                              \
                  | (tpack4((d1 >> 4) & 0x0f0f0f0fu) << 16);                     \
            ll[q] = tpack4(d0 & 0x0f0f0f0fu)                                     \
                  | (tpack4(d1 & 0x0f0f0f0fu) << 16);                            \
        }                                                                        \
        *(i32x4*)(Xh[buf] + x_col * (TBK / 2) + x_half * 16) = *(i32x4*)hh;      \
        *(i32x4*)(Xl[buf] + x_col * (TBK / 2) + x_half * 16) = *(i32x4*)ll;      \
    }

    STAGE(0, 0)
    __syncthreads();

    for (int strip = 0; strip < nstrip; ++strip) {
        const int buf = strip & 1;
        const uint8_t* pa = As[buf] + (wm * TWMt * 16 + lane) * (TBK / 2);
        const uint8_t* ph = Xh[buf] + (wn * TWNt * 16 + lane) * (TBK / 2);
        const uint8_t* pl = Xl[buf] + (wn * TWNt * 16 + lane) * (TBK / 2);
#pragma unroll
        for (int s = 0; s < TBK / 16; ++s) {
            i32x2 af[TWMt], bh[TWNt], bl[TWNt];
#pragma unroll
            for (int i = 0; i < TWMt; i++)
                af[i] = *(const i32x2*)(pa + i * 16 * (TBK / 2) + s * 8);
#pragma unroll
            for (int j = 0; j < TWNt; j++) {
                bh[j] = *(const i32x2*)(ph + j * 16 * (TBK / 2) + s * 8);
                bl[j] = *(const i32x2*)(pl + j * 16 * (TBK / 2) + s * 8);
            }
#pragma unroll
            for (int i = 0; i < TWMt; i++)
#pragma unroll
                for (int j = 0; j < TWNt; j++) {
                    // ONE weight fragment, BOTH digit passes.
                    ah[i][j] = WH(af[i], bh[j], ah[i][j]);
                    al[i][j] = WL(af[i], bl[j], al[i][j]);
                }
        }

        // A compact group is STRIPS_PER_GROUP strips: fold with this group's
        // scales and reset, exactly as the shipping twins do.
        if ((strip + 1) % STRIPS_PER_GROUP == 0) {
            const int g = strip / STRIPS_PER_GROUP;
#pragma unroll
            for (int i = 0; i < TWMt; i++) {
                float sw[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int mr = bm + wm * TWMt * 16 + i * 16 + 2 * e + half;
                    sw[e] = (mr < M) ? t_f16(W + side_base
                        + ((long long)mr * n_groups + g) * (long long)side_stride) : 0.f;
                }
#pragma unroll
                for (int j = 0; j < TWNt; j++) {
                    const int bc = bn + wn * TWNt * 16 + j * 16 + lane;
                    const float sx = (bc < B) ? Xs[(long long)bc * n_groups + g] : 0.f;
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        // Exact radix-16 recombination in i32 before the scale.
                        const int v = 16 * ah[i][j][e] + al[i][j][e];
                        facc[i][j][e] += (float)v * sw[e] * sx;
                    }
                    ah[i][j] = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
                    al[i][j] = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
                }
            }
        }

        if (strip + 1 < nstrip) {
            __syncthreads();
            STAGE(buf ^ 1, strip + 1)
            __syncthreads();
        }
    }

#pragma unroll
    for (int i = 0; i < TWMt; i++)
#pragma unroll
        for (int j = 0; j < TWNt; j++) {
            const int bc = bn + wn * TWNt * 16 + j * 16 + lane;
            if (bc >= B) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int mr = bm + wm * TWMt * 16 + i * 16 + 2 * e + half;
                if (mr < M) Y[(long long)bc * M + mr] = facc[i][j][e];
            }
        }
}
