// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// i8 WMMA MMQ MoE grouped-GEMM port for gfx12 (RDNA4 - R9700/gfx1201,
// gfx1200) - k4 (deeper K-tile pipeline) variant for HFQ4G128 ParoQuant.
//
// Combines three existing kernels:
//   - `gemm_hfq4g256_moe_grouped_mmq_k4.gfx12.hip` (gfx12 k4 pipeline
//     pattern: 8 WMMAs into 4 independent cacc accumulators, interleaved
//     across sub-blocks, then 4 scale FMAs resolve at the end).
//   - `gemm_paro_q4g128_moe_grouped_mmq.gfx12.hip` (gfx12 G128 layout:
//     72 B/group stride, ONE Q8_1 mmq block per HFQ4 group).
//   - `gemm_paro_q4g128_moe_grouped_mmq_k8.gfx1151.hip` (gfx1151 K8
//     PARO sibling - same G128 nibble layout, different WMMA encoding).
//
// === K4 PIPELINE STRUCTURE ===
//
// Per HFQ4G128 group (single Q8_1 mmq block of K=128):
//   1. Pre-load all 8 (sb, kt) operand pairs into pk_arr[8], b_arr[8].
//      Already done by the k2 sibling.
//   2. Materialize a_i32/b_i32 vectors for all 8 slots upfront.
//   3. Issue 8 WMMAs into 4 INDEPENDENT accumulators (cacc0..cacc3) -
//      one per sub-block - interleaved so consecutive WMMAs target
//      distinct destination registers.
//   4. After all 8 WMMAs complete, resolve 4 scale FMAs.
//
// vs k2 (gemm_paro_q4g128_moe_grouped_mmq.gfx12.hip):
//   k2 serialises scale FMA inside the sub-block loop, gating the
//   compiler's ability to interleave WMMAs across sub-blocks. K4
//   amortises that chain over more WMMA dispatches per scale FMA
//   boundary - matches the gfx12 HFQ4G256 k4 pattern that gave +X% on
//   that kernel family.
//
// Output is numerically equivalent to k2 modulo floating-point
// summation order - same `acc[j] += sc*d_x*dot + zp*sum_x` per
// sub-block, same per-sub-block resolution boundary.
//
// === KEY GFX12 DIFFERENCES vs gfx1151 sibling ===
// (Same as the k2 gfx12 PARO sibling - see
// `gemm_paro_q4g128_moe_grouped_mmq.gfx12.hip` for full prose.)
//
// 1. Intrinsic: `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`.
// 2. Per-lane operand width: int32x2 (8 int8 / lane) vs int32x4 on gfx11.
// 3. K-tile split across 2 lane groups (k_grp = tid >> 4).
// 4. C-output: `acc[j] = C[8*k_grp + j][m_lane]`.
// 5. Per-row sc/zp broadcast: `src_lane = 8*k_grp + j`.
//
// === DIFFERENCES vs hfq4g256 k4 gfx12 sister ===
//
// 1. Stride: 72 B/group (G128) vs 136 B/group (G256).
// 2. ONE Q8_1 block per HFQ4 group (G128 = 128 K-elements / 128 = 1) vs
//    TWO (G256 = 256 / 128). Removes the outer `kb_in_group` loop.
// 3. K-offset: `sb*32 + kt*16 + k_grp*8` (no kb_in_group * 128 term).
//
// `__launch_bounds__(32, 2)` - matches k2 sibling.
//
// Compile gate: gfx1200 / gfx1201 only.

#if defined(__gfx1200__) || defined(__gfx1201__)

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};

static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

typedef int8_t  __attribute__((ext_vector_type(8)))  int8x8_t;
typedef int32_t __attribute__((ext_vector_type(2)))  int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8)))  int32x8_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_paro_q4g128_moe_grouped_mmq_k4_gfx12(
    const unsigned long long*  __restrict__ expert_weight_ptrs, // [E]
    const int*                 __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                 __restrict__ sorted_slot_index,  // [m_total]
    const block_q8_1_mmq*      __restrict__ X_q8_src,           // [K/128 × x_src_rows]
    float*                     __restrict__ Y_grouped,          // [m_total × M]
    int M, int K, int x_row_div, int m_total, int x_src_rows
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int slot_idx = slot_start + m_lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }

    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int groups_per_row = K / 128;
    const char* row_base = A + (long long)safe_row * groups_per_row * 72;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 72;
        const float sc_self = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp_self = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        float sc_row[8];
        float zp_row[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int src_lane = 8 * k_grp + j;
            sc_row[j] = __shfl(sc_self, src_lane);
            zp_row[j] = __shfl(zp_self, src_lane);
        }

        // PARO G128: 1 Q8_1 mmq block per HFQ4 group. Direct map g to Q8_1 block.
        const int kb_q8 = g;
        const block_q8_1_mmq* xb = (x_row >= 0)
            ? (X_q8_src + (long long)kb_q8 * x_src_rows + x_row)
            : nullptr;

        // Pre-load all 8 (sb, kt) operand pairs upfront.
        unsigned int pk_arr[8];
        int8x8_t b_arr[8];
        float d_x_arr[4];
        float sum_x_arr[4];

        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            if (xb != nullptr) {
                half2 dm = xb->ds4[sb];
                float2 dm_f = __half22float2(dm);
                d_x_arr[sb]   = dm_f.x;
                sum_x_arr[sb] = dm_f.y;
            } else {
                d_x_arr[sb]   = 0.0f;
                sum_x_arr[sb] = 0.0f;
            }
            #pragma unroll
            for (int kt = 0; kt < 2; kt++) {
                const int slot = sb * 2 + kt;
                const int k_off = sb * 32 + kt * 16 + k_grp * 8;
                pk_arr[slot] = *(const unsigned int*)(gp + 8 + k_off / 2);
                if (xb != nullptr) {
                    const int8_t* qs_p = xb->qs + sb * 32 + kt * 16 + k_grp * 8;
                    b_arr[slot] = *(const int8x8_t*)qs_p;
                } else {
                    b_arr[slot] = (int8x8_t){0, 0, 0, 0, 0, 0, 0, 0};
                }
            }
        }

        // === K4 pipeline ===
        // 4 independent int32 accumulators, one per sub-block.
        int32x8_t cacc0 = {0, 0, 0, 0, 0, 0, 0, 0};
        int32x8_t cacc1 = {0, 0, 0, 0, 0, 0, 0, 0};
        int32x8_t cacc2 = {0, 0, 0, 0, 0, 0, 0, 0};
        int32x8_t cacc3 = {0, 0, 0, 0, 0, 0, 0, 0};

        // Materialize a_i32 / b_i32 for all 8 slots upfront. Nibble unpack
        // is hoisted before the WMMA chain so the compiler can interleave
        // it with the prefetch loads.
        int32x2_t a_i32_arr[8];
        int32x2_t b_i32_arr[8];
        #pragma unroll
        for (int slot = 0; slot < 8; slot++) {
            const unsigned int pk = pk_arr[slot];
            int8x8_t a_vec;
            a_vec[0] = (int8_t)((pk >>  0) & 0xFu);
            a_vec[1] = (int8_t)((pk >>  4) & 0xFu);
            a_vec[2] = (int8_t)((pk >>  8) & 0xFu);
            a_vec[3] = (int8_t)((pk >> 12) & 0xFu);
            a_vec[4] = (int8_t)((pk >> 16) & 0xFu);
            a_vec[5] = (int8_t)((pk >> 20) & 0xFu);
            a_vec[6] = (int8_t)((pk >> 24) & 0xFu);
            a_vec[7] = (int8_t)((pk >> 28) & 0xFu);
            a_i32_arr[slot] = *(const int32x2_t*)&a_vec;
            b_i32_arr[slot] = *(const int32x2_t*)&b_arr[slot];
        }

        // Issue 8 WMMAs into 4 independent accumulators - interleaved
        // across sub-blocks (cacc0/1/2/3) so consecutive WMMAs target
        // distinct destination registers (maximises scheduler overlap).
        //
        // signed_a = false (HFQ4 nibbles 0..15, unsigned-safe);
        // signed_b = true (Q8_1 signed -127..127); clamp = true.
        cacc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            false, a_i32_arr[0], true, b_i32_arr[0], cacc0, true);
        cacc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            false, a_i32_arr[2], true, b_i32_arr[2], cacc1, true);
        cacc2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            false, a_i32_arr[4], true, b_i32_arr[4], cacc2, true);
        cacc3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            false, a_i32_arr[6], true, b_i32_arr[6], cacc3, true);
        cacc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            false, a_i32_arr[1], true, b_i32_arr[1], cacc0, true);
        cacc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            false, a_i32_arr[3], true, b_i32_arr[3], cacc1, true);
        cacc2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            false, a_i32_arr[5], true, b_i32_arr[5], cacc2, true);
        cacc3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            false, a_i32_arr[7], true, b_i32_arr[7], cacc3, true);

        // Resolve 4 scale FMAs in order. Numerically equivalent to k2
        // (same per-sub-block resolution).
        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            const float d_x_sb   = d_x_arr[sb];
            const float sum_x_sb = sum_x_arr[sb];
            const int32x8_t cacc_sb = (sb == 0) ? cacc0
                                    : (sb == 1) ? cacc1
                                    : (sb == 2) ? cacc2
                                                : cacc3;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float dot_f = (float)cacc_sb[j];
                acc[j] += sc_row[j] * d_x_sb * dot_f
                        + zp_row[j] * sum_x_sb;
            }
        }
    } // for g

    // gfx12 wave32 WMMA C-mapping: acc[j] = C[8*k_grp + j][m_lane].
    const int out_col = slot_start + m_lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 8 * k_grp + j;
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}

#else

// Stub for non-gfx12 archs.
extern "C" __global__ void gemm_paro_q4g128_moe_grouped_mmq_k4_gfx12(
    const unsigned long long*, const int*, const int*,
    const void*, float*, int, int, int, int, int
) {}

#endif
