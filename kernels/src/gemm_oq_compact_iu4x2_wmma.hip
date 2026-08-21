// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Compact-resident Opus W4-A8-via-TWO-iu4-PASSES.
//
// The contraction is linear, so an activation can be carried as radix-16 digits
// and reassembled after the matrix core:
//
//     x = 16*x_hi + x_lo   ⇒   W·x = 16*(W·x_hi) + (W·x_lo)
//
// with x_hi SIGNED int4 [-8,7] and x_lo UNSIGNED int4 [0,15], which together
// span exactly int8's range. `wmma_i32_16x16x16_iu4_w32` takes per-operand
// signedness flags, so both digits feed the same instruction. K=5120 terms of
// int4xint4 peak near 327k, far inside i32, so the 16x recombination in the
// epilogue is exact rather than saturating.
//
// WHY THIS IS NOT JUST "2x THE WORK", which is what it looks like and is the
// mistake worth not repeating: **the weight tile is identical across passes**.
// Only the activation digit changes. Measured decomposition of the 1-pass
// kernel on gate/up B=256 — activation loads 27.6%, weight loads 31.3%, WMMA
// issue and the rest 41.1% — so a second pass adds WMMA and one more int4
// activation plane and adds NOTHING to weight traffic. Two int4 planes are also
// exactly the bytes of one int8 activation, so nothing is wasted there either.
//
// That only holds if `a` is loaded once and consumed by both passes, which
// forces TWO live i32 accumulator sets. At OQC4X2_NB=8 that is 128 VGPRs of
// accumulator before `facc`, so this variant halves NB to 4. Re-reading the
// weight tile per pass instead would restore the naive 2x and defeat the point.
//
//   W    : compact SPLIT-PLANE blocks (bulk nibbles fed to the core RAW)
//   Xhi  : [B, K/2] packed SIGNED   int4 high digits
//   Xlo  : [B, K/2] packed UNSIGNED int4 low digits
//   Xs   : [B, K/G] f32     Y : [B, M] f32
//
// Does NOT apply the sparse weight overlay — same contract as the 1-pass twin.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t __attribute__((ext_vector_type(2))) int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;
typedef float   __attribute__((ext_vector_type(8))) float8_t;

#ifndef OQC4X2_NB
#define OQC4X2_NB 4
#endif
#ifndef OQC4X2_MW
#define OQC4X2_MW 16
#endif
#define OQC4X2_COLS (OQC4X2_NB * 16)
#define OQC4X2_PAD  4

static __device__ __forceinline__ int32x2_t oqc4x2_load(const uint8_t* p) {
    return (int32x2_t){(int32_t)*(const uint32_t*)(p + 0),
                       (int32_t)*(const uint32_t*)(p + 4)};
}

static __device__ __forceinline__ float oqc4x2_f16(const uint8_t* b) {
    return (float)(*(const _Float16*)b);
}

__launch_bounds__(32 * OQC4X2_MW, 1)
extern "C" __global__ void gemm_oq_compact_iu4x2_wmma(
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ Xhi,
    const uint8_t* __restrict__ Xlo,
    const float*   __restrict__ Xs,
    float*         __restrict__ Y,
    int M, int K, int B, int group, int block_stride
) {
    const int tid   = threadIdx.x;
    const int wave  = tid >> 5;
    const int wlane = tid & 31;
    const int lane  = wlane & 15;
    const int half  = wlane >> 4;
    const int row_start   = (blockIdx.x * OQC4X2_MW + wave) * 16;
    const int batch_start = blockIdx.y * OQC4X2_COLS;
    // Only the workgroup-uniform condition may early-return: a per-wave return
    // would desert the __syncthreads() below and hang the workgroup.
    if (batch_start >= B) return;

    const int a_row  = row_start + lane;
    const int safe_a = (a_row < M) ? a_row : (M - 1);
    const int n_groups = K / group;
    const int tiles_per_group = group / 16;
    const int nib_bytes   = group >> 1;
    const int side_stride = block_stride - nib_bytes;
    const long long side_base = (long long)M * n_groups * nib_bytes;
    const int x_stride = K / 2;

    int b_col_nb[OQC4X2_NB];
    float8_t facc[OQC4X2_NB];
    #pragma unroll
    for (int nb = 0; nb < OQC4X2_NB; ++nb) {
        b_col_nb[nb] = batch_start + nb * 16 + lane;
        facc[nb] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    // Both digit planes staged; one group's tile is COLS * (group/2) bytes each.
    __shared__ uint8_t xh[OQC4X2_COLS * (128 + OQC4X2_PAD)];
    __shared__ uint8_t xl[OQC4X2_COLS * (128 + OQC4X2_PAD)];
    const int lds_stride = nib_bytes + OQC4X2_PAD;
    const int dwords_per_col = nib_bytes >> 2;
    // Power of two (16 or 32) — shift/mask, not a DIVISION. See the 1-pass twin.
    const int dpc_shift = __builtin_ctz(dwords_per_col);
    const int dpc_mask  = dwords_per_col - 1;
    const int stage_total = OQC4X2_COLS * dwords_per_col;
    const int cols = (B - batch_start < OQC4X2_COLS) ? (B - batch_start) : OQC4X2_COLS;

    for (int g = 0; g < n_groups; ++g) {
        __syncthreads();
        for (int i = tid; i < stage_total; i += blockDim.x) {
            const int c = i >> dpc_shift;
            const int d = i & dpc_mask;
            const int br = batch_start + c;
            const int sb = (br < B) ? br : (B - 1);
            const long long off = (long long)sb * x_stride + g * nib_bytes + d * 4;
            *(uint32_t*)(xh + c * lds_stride + d * 4) = *(const uint32_t*)(Xhi + off);
            *(uint32_t*)(xl + c * lds_stride + d * 4) = *(const uint32_t*)(Xlo + off);
        }
        __syncthreads();

        const uint8_t* nibbles = W + ((long long)safe_a * n_groups + g) * nib_bytes;
        int32x8_t iacc_hi[OQC4X2_NB], iacc_lo[OQC4X2_NB];
        #pragma unroll
        for (int nb = 0; nb < OQC4X2_NB; ++nb) {
            iacc_hi[nb] = (int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
            iacc_lo[nb] = (int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
        }
        for (int kt = 0; kt < tiles_per_group; ++kt) {
            // ONE weight load, consumed by BOTH digit passes. This is the whole
            // reason two passes are not two kernels' worth of work.
            const int32x2_t a = oqc4x2_load(nibbles + kt * 8);
            #pragma unroll
            for (int nb = 0; nb < OQC4X2_NB; ++nb) {
                const int off = (nb * 16 + lane) * lds_stride + kt * 8;
                const int32x2_t bh = oqc4x2_load(xh + off);
                const int32x2_t bl = oqc4x2_load(xl + off);
                // high digit SIGNED, low digit UNSIGNED — together exactly int8.
                iacc_hi[nb] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                    true, a, true, bh, iacc_hi[nb], false);
                iacc_lo[nb] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                    true, a, false, bl, iacc_lo[nb], false);
            }
        }

        float sw[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int m_row = row_start + 2 * j + half;
            sw[j] = (m_row < M)
                ? oqc4x2_f16(W + side_base
                      + ((long long)m_row * n_groups + g) * (long long)side_stride)
                : 0.f;
        }
        #pragma unroll
        for (int nb = 0; nb < OQC4X2_NB; ++nb) {
            const int b_col = b_col_nb[nb];
            const float sx = (b_col < B) ? Xs[(long long)b_col * n_groups + g] : 0.f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                // Exact radix-16 recombination in i32 before the f32 scale.
                const int v = 16 * iacc_hi[nb][j] + iacc_lo[nb][j];
                facc[nb][j] += (float)v * sw[j] * sx;
            }
        }
    }

    if (row_start >= M) return;
    #pragma unroll
    for (int nb = 0; nb < OQC4X2_NB; ++nb) {
        const int b_col = b_col_nb[nb];
        if (b_col >= B) continue;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int m_row = row_start + 2 * j + half;
            if (m_row < M) Y[(long long)b_col * M + m_row] = facc[nb][j];
        }
    }
}
