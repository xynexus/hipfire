// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// i8 WMMA MMQ MoE grouped-GEMM for HFQ4G128 (ParoQuant) on gfx12 (RDNA4 —
// R9700/gfx1201, gfx1200).
//
// Combines two prior ports:
//   - `gemm_hfq4g256_moe_grouped_mmq.gfx12.hip`: gfx12 WMMA conventions
//     (k_grp lane split, _gfx12 intrinsic suffix, 8 int8/lane operand width,
//     acc[j] = C[8*k_grp + j][m_lane] output mapping).
//   - `gemm_paro_q4g128_moe_grouped_mmq.gfx1151.hip`: HFQ4G128 layout
//     (72 B/group stride, 128 K-elements per HFQ4 group, ONE Q8_1 mmq block
//     per HFQ4 group — no outer kb_in_group loop).
//
// Dispatch contract (mirrors both siblings):
//   - X pre-quantized to Q8_1 (block_q8_1_mmq) via `quantize_q8_1_mmq_ds4`
//     before launch (X_q8_src layout = [K/128 × x_src_rows]).
//   - Per-tile (blockIdx.x, blockIdx.y) owns a 16-row × 16-slot output tile.
//   - All 16 slots in this tile share the same expert (scatter pad guarantees
//     expert_offsets aligned to BLOCK_M = 16).
//   - Per lane (m_lane = tid & 15) routes to ONE slot via sorted_slot_index;
//     -1 padding lanes contribute zero Q8_1 to the B operand.
//   - HFQ4 scale/zero applied per-row via Q8_1 (d, sum) correction at the
//     FMA step (`sum_f32 += sc * d_x * C_frag + zp * sum_x`).
//
// === KEY DIFFERENCES vs paro_q4g128 gfx1151 sister ===
//
// 1. Intrinsic: `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` (with
//    `_gfx12` suffix); gfx11/gfx1151 omit the suffix. Same 16x16x16 shape,
//    different operand widths (see #2) and C-layout (see #4).
//
// 2. Operand width per lane: gfx12 wave32 packs 8 int8 / lane (int32x2)
//    vs gfx11/gfx1151's 16 int8 / lane (int32x4). Two WMMA calls per
//    K=32 sub-block (K=16 each).
//
// 3. K-tile lane split: gfx12 wave32 splits 16-K WMMA across `k_grp = tid >> 4`.
//    Lanes 0..15 carry K[0..7], lanes 16..31 carry K[8..15].
//
// 4. C-output mapping: gfx12 `acc[j] = C[8*k_grp + j][m_lane]` (j=0..7
//    contiguous rows). gfx1151 sister: `C[2*j + (tid>>4)][m_lane]`.
//
// 5. sc/zp shuffle pattern: gfx12 src_lane = `8 * k_grp + j` (single broadcast
//    per output row). gfx1151: src_lane = `2 * j + (tid>>4)`.
//
// === DIFFERENCES vs hfq4g256 gfx12 sister ===
//
// 1. Stride: 72 B/group (G128) vs 136 B/group (G256). qweight payload = 64 B
//    (G128) vs 128 B (G256).
//
// 2. Q8_1 blocks per HFQ4 group: ONE (G128 = 128 K-elements ÷ 128 K per Q8_1
//    mmq block) vs TWO (G256 = 256 K-elements ÷ 128). Removes the outer
//    `kb_in_group` loop — `kb_q8 = g` directly.
//
// 3. K-offset formula: `sb*32 + kt*16 + k_grp*8` (no `kb_in_group * 128` term).
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

// `x_row_div` controls how sorted_slot_index maps to X_q8_src rows:
//   gate_up: X_src_f32 was [N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src_f32 was [N*K_TOP × K], x_row_div = 1 → x_row = flat
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_paro_q4g128_moe_grouped_mmq_gfx12(
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

    // Sentinel early-return: tile_y beyond live m_total.
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int m_lane = tid & 15;   // 0..15: slot index within tile
    const int k_grp  = tid >> 4;   // 0/1: which K-half this lane carries

    // X gather: each lane (m_lane) gets its own routed slot. Padding lanes
    // (sorted_slot_index == -1) substitute zero Q8_1 contributions.
    const int slot_idx = slot_start + m_lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }

    // A-row selection: pad OOB to last valid row (same trick as FP16 sister).
    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int groups_per_row = K / 128;                       // G128
    const char* row_base = A + (long long)safe_row * groups_per_row * 72;  // 72 B/group

    // Output accumulator (gfx12 wave32 layout): acc[j] = Y[8*k_grp + j][m_lane].
    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 72;
        // HFQ4 per-group scale (sc) and zero (zp) for THIS lane's row.
        const float sc_self = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp_self = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        // gfx12 acc[j] = Y[8*k_grp + j][m_lane]. Output row r = row_start + 8*k_grp + j
        // is owned by THIS lane (m_lane is the slot/N-dim). The sc/zp for row r belongs
        // to the lane whose own row is r — i.e. lane with m_lane = 8*k_grp + j.
        // sc_self/zp_self at lane L holds row L's value, so shuffle from
        // src_lane = 8*k_grp + j (within the same lane group of 16; mirrored across
        // the wave32 halves for the 16x16 A-operand).
        float sc_row[8];
        float zp_row[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int src_lane = 8 * k_grp + j;
            sc_row[j] = __shfl(sc_self, src_lane);
            zp_row[j] = __shfl(zp_self, src_lane);
        }

        // HFQ4G128: 1 Q8_1 mmq block per HFQ4 group (128 K-elements / 128 = 1).
        // No outer kb_in_group loop — directly map g → Q8_1 block.
        const int kb_q8 = g;
        const block_q8_1_mmq* xb = (x_row >= 0)
            ? (X_q8_src + (long long)kb_q8 * x_src_rows + x_row)
            : nullptr;

        // Pre-load all 8 (sb=0..3, kt=0..1) A+B operands to overlap memory
        // latency with WMMA execution. Mirrors the G256 gfx12 sister.
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
                // K-offset within HFQ4G128 group:
                //   sb * 32  (sub-block within group)
                // + kt * 16  (WMMA call within sub-block)
                // + k_grp * 8 (lane-group's K-half).
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

        // 4 sub-blocks of K=32. Per sub-block: TWO i8 WMMA calls (K=16
        // each) on an int32 accumulator that resets at start-of-sub-block,
        // then scale FMA across (sc_row, zp_row, d_x, sum_x).
        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            int32x8_t cacc = {0, 0, 0, 0, 0, 0, 0, 0};

            #pragma unroll
            for (int kt = 0; kt < 2; kt++) {
                const int slot = sb * 2 + kt;
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

                int32x2_t a_i32 = *(const int32x2_t*)&a_vec;
                int32x2_t b_i32 = *(const int32x2_t*)&b_arr[slot];

                // signed_a = false (HFQ4 nibbles are 0..15, unsigned-safe);
                // signed_b = true (Q8_1 is signed -127..127). clamp = true
                // (safe: K=16 chunk max |dot| = 16 * 15 * 127 ≈ 30k).
                cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                    false, a_i32, true, b_i32, cacc, true
                );
            } // for kt

            // Scale FMA: per output element (j in [0,8)).
            //   dot_real = sum_k((sc*q4 + zp) * d_x*q8)
            //            = sc * d_x * sum_k(q4*q8) + zp * d_x * sum_k(q8)
            //            = sc * d_x * cacc[j] + zp * sum_real_x
            // Q8_1 stores sum of REAL x values directly in ds4[].y, so we
            // multiply by sum_x_self only (not d_x_self * sum_x_self).
            const float d_x_self   = d_x_arr[sb];
            const float sum_x_self = sum_x_arr[sb];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float dot_f = (float)cacc[j];
                acc[j] += sc_row[j] * d_x_self * dot_f
                        + zp_row[j] * sum_x_self;
            }
        } // for sb
    } // for g

    // gfx12 wave32 WMMA C-mapping: acc[j] = C[8*k_grp + j][m_lane].
    // Direct store (=) because Y_grouped is fresh per call (caller's combine
    // fans K_TOP slots into x_residual; this kernel writes ONE slot per cell).
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

// Stub for non-gfx12 archs (gfx11/gfx1151 has its own MMQ port with different
// WMMA intrinsics + operand widths; gfx906 has dp4a-based MMQ siblings).
extern "C" __global__ void gemm_paro_q4g128_moe_grouped_mmq_gfx12(
    const unsigned long long*, const int*, const int*,
    const void*, float*, int, int, int, int, int
) {}

#endif
