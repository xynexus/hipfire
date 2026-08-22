// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Sparse overlay correction, K-MAJOR activation variant.
//
// `gemv_oq_compact_overlay_correct` documents three layouts, all far too
// expensive, and names the reason for the best of them exactly: "a gather's 32
// lanes read `x[b, g*G+idx]` for 32 different b, an activation row apart
// (K/2 = 2560 B at K=5120), so every gather touches 32 cache lines."
//
// All three kept the activation in its native [B, K/2] b-major layout. That is
// the whole problem: b is the axis the lanes vary along, and it is the OUTER
// axis of the array. Transposing the activation to K-major makes the same
// gather 32 CONTIGUOUS bytes — one cache line instead of 32 — without changing
// a single arithmetic operation.
//
// The transpose is cheap and amortises: the activation is quantized once per
// layer and shared across every projection that reads it (gate+up, q/k/v), and
// it is only B*K bytes (1.3 MB at B=256, K=5120) against the 44.6 MB of weights
// the GEMM streams.
//
//   XT : [K, B] int8, one sign-extended value per byte (not packed)
//   XsT: [n_groups, B] f32
//   W  : compact SPLIT-PLANE blocks
//   Y  : [B, M] f32, ACCUMULATED IN PLACE
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef OQCO_MAX_OV
#define OQCO_MAX_OV 16
#endif

static __device__ __forceinline__ float oqcot_f16(const uint8_t* p) {
    return (float)(*(const _Float16*)p);
}

// [B, K/2] packed signed int4  ->  [K, B] int8, and [B, ng] f32 -> [ng, B] f32.
// One byte per activation: the correction gathers single values, so packing
// buys nothing here and costs an unpack per gather.
__launch_bounds__(256) extern "C" __global__ void oq_compact_x4_transpose(
    const uint8_t* __restrict__ X,
    const float*   __restrict__ Xs,
    int8_t*        __restrict__ XT,
    float*         __restrict__ XsT,
    int B, int K, int n_groups
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    if (b >= B || k >= K) return;
    const uint8_t packed = X[(long long)b * (K / 2) + (k >> 1)];
    const int q = (k & 1) ? (((int)(packed >> 4) << 28) >> 28)
                          : (((int)(packed & 0xf) << 28) >> 28);
    XT[(long long)k * B + b] = (int8_t)q;          // contiguous across lanes
    if (k < n_groups) XsT[(long long)k * B + b] = Xs[(long long)b * n_groups + k];
}

// int8 twin of the above, for the exact W4A8 (iu4x2) path whose activation is
// already one byte per element. Same [K, B] / [ng, B] output layout.
__launch_bounds__(256) extern "C" __global__ void oq_compact_x8_transpose(
    const int8_t* __restrict__ X,
    const float*  __restrict__ Xs,
    int8_t*       __restrict__ XT,
    float*        __restrict__ XsT,
    int B, int K, int n_groups
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    if (b >= B || k >= K) return;
    XT[(long long)k * B + b] = X[(long long)b * K + k];
    if (k < n_groups) XsT[(long long)k * B + b] = Xs[(long long)b * n_groups + k];
}

__launch_bounds__(256) extern "C" __global__ void oq_compact_overlay_correct_t(
    const uint8_t* __restrict__ W,
    const int8_t*  __restrict__ XT,
    const float*   __restrict__ XsT,
    float*         __restrict__ Y,
    int M, int K, int B, int group, int block_stride
) {
    const int wave  = threadIdx.x >> 5;
    const int waves = blockDim.x >> 5;
    const int lane  = threadIdx.x & 31;
    const int row   = blockIdx.x * waves + wave;
    if (row >= M) return;

    const int n_groups = K / group;
    const int nib_bytes = group >> 1;
    const int side_stride = block_stride - nib_bytes;
    const int n_ov_all = (side_stride - 2) >> 1;
    const int n_ov = (n_ov_all < OQCO_MAX_OV) ? n_ov_all : OQCO_MAX_OV;
    const long long side_base = (long long)M * n_groups * nib_bytes;
    const uint8_t* s_row = W + side_base + (long long)row * n_groups * side_stride;

    // B-SLICE IN REGISTERS, TABLE READ ONCE, GATHER AS A DWORD.
    //
    // Three things the b-major original gets wrong, in order of cost:
    //   1. the activation is [B, K/2], so 32 lanes reading 32 different b touch
    //      32 cache lines (its own comment says so). K-major fixes that.
    //   2. the overlay table is read INSIDE the b0 loop -- B/32 times over, and
    //      idx/val are wave-uniform so it is redundant scalar work too.
    //   3. a byte gather puts 32 lanes x 1 B = 32 B into a 64 B line, half wasted.
    //      Giving each lane FOUR CONSECUTIVE b makes it a dword gather: 32 lanes
    //      x 4 B = 128 contiguous bytes, two full lines, nothing wasted.
    //
    // XT rows are B-aligned and the block base is a multiple of 4, so the dword
    // load is always aligned.
    constexpr int MAXQ = 4;                    // dword blocks per lane, B <= 512
    const int nblk = (B + 127) >> 7;           // 128 b per wave-pass
    float acc[MAXQ][4];
    #pragma unroll
    for (int q = 0; q < MAXQ; ++q)
        #pragma unroll
        for (int t = 0; t < 4; ++t) acc[q][t] = 0.0f;

    for (int g = 0; g < n_groups; ++g) {
        const uint8_t* sb = s_row + (long long)g * side_stride;
        const uint8_t* table = sb + 2;
        const int kbase = g * group;
        const float sw = oqcot_f16(sb);
        int isum[MAXQ][4];
        #pragma unroll
        for (int q = 0; q < MAXQ; ++q)
            #pragma unroll
            for (int t = 0; t < 4; ++t) isum[q][t] = 0;

        for (int e = 0; e < n_ov; ++e) {
            const int idx = (int)table[2 * e];
            const int val = (int)(int8_t)table[2 * e + 1];
            const int8_t* xr = XT + (long long)(kbase + idx) * B;
            for (int q = 0; q < nblk; ++q) {
                const int b0 = (q << 7) + (lane << 2);
                if (b0 >= B) break;
                const uint32_t w4 = *(const uint32_t*)(xr + b0);
                #pragma unroll
                for (int t = 0; t < 4; ++t)
                    isum[q][t] += val * (int)(int8_t)((w4 >> (t * 8)) & 0xff);
            }
        }
        const float* xsg = XsT + (long long)g * B;
        for (int q = 0; q < nblk; ++q) {
            const int b0 = (q << 7) + (lane << 2);
            if (b0 >= B) break;
            #pragma unroll
            for (int t = 0; t < 4; ++t)
                acc[q][t] += (float)isum[q][t] * sw * xsg[b0 + t];
        }
    }
    for (int q = 0; q < nblk; ++q) {
        const int b0 = (q << 7) + (lane << 2);
        if (b0 >= B) break;
        #pragma unroll
        for (int t = 0; t < 4; ++t)
            Y[(long long)(b0 + t) * M + row] += acc[q][t];
    }
}
