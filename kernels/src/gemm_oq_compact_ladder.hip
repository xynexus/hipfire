// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Compact-resident Opus exact W4A8, built as a LADDER OF REAL KERNELS.
//
// Every rung here is a complete, correct GEMM that passes parity against the CPU
// oracle. The previous attempt built a synthetic probe, measured a ceiling on it,
// then wrote the whole real kernel in one step -- and neither the ceiling nor two
// of the probe's conclusions (double buffering, the occupancy rule) transferred.
// So: one change per rung, parity every rung, measured every rung, and the Rust
// binary REBUILT every rung (`include_str!` bakes this source in; deleting the
// .hsaco only re-JITs the string already compiled into the binary).
//
// Contract, identical at every rung:
//   W    : compact SPLIT-PLANE blocks. Nibble plane [M][n_groups][128 B], then
//          side records [M][n_groups][side_stride] = [f16 scale][overlay table].
//   Xq   : [B, K] int8      Xs : [B, K/256] f32      Y : [B, M] f32
//   exact W4A8: x = 16*x_hi + x_lo, hi signed, lo unsigned, two iu4 passes on
//   one weight fragment. No sparse overlay (separate pass, as for every twin).
//
// ---- RUNG 0 (optimized): no LDS, no tiling, no blocking ----
//
// First cut measured 2.552 ms / 35.8 TOPS and 81 INSTRUCTIONS PER WMMA. The
// census said the waste was not the nibble packing (~20/WMMA, inherent to the
// format) but two things that are pure bookkeeping:
//
//   64-bit address arithmetic     ~10.5/WMMA  (v_add_co + v_add_co_ci +
//                                              v_mad_u64_u32 + v_mul_lo)
//   bounds guards inside the fold  ~3.75/WMMA (v_cmpx + s_cbranch_execz)
//
// Both are loop-invariant and were being recomputed every group. This rung
// hoists every base pointer out of the group loop and advances it by a constant
// stride, and pre-clamps the fold's row indices so the inner loop has no
// branches at all.
// One wave32 computes one 16x16 output tile. Both fragments come straight from
// global every k-step. This is the floor: it does the least possible work per
// byte moved, and every later rung has to beat it.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t i32x2 __attribute__((ext_vector_type(2)));
typedef int32_t i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
#define LWH(a, b, c) __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, false)
#define LWL(a, b, c) __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, false, b, c, false)

#define LGROUP 256
#define LWAVES 8            // waves per workgroup -> BN = 8*16 = 128

__device__ __forceinline__ uint32_t lpack4(uint32_t t) {
    t = (t | (t >> 4)) & 0x00ff00ffu;
    return (t | (t >> 8)) & 0x0000ffffu;
}
__device__ __forceinline__ float l_f16(const uint8_t* p) {
    return (float)(*(const _Float16*)p);
}

__launch_bounds__(32 * LWAVES) extern "C" __global__ void gemm_oq_compact_ladder(
    const uint8_t* __restrict__ W, const int8_t* __restrict__ Xq,
    const float* __restrict__ Xs, float* __restrict__ Y,
    int M, int K, int B, int block_stride) {
    const int t = threadIdx.x;
    const int wid = t >> 5, wtid = t & 31;
    const int lane = wtid & 15, half = wtid >> 4;
    const int n_groups = K / LGROUP;
    const int side_stride = block_stride - 128;
    const long long side_base = (long long)M * n_groups * 128LL;

    const int m0 = blockIdx.x * 16;                    // 16 weight rows
    const int b0 = blockIdx.y * (LWAVES * 16) + wid * 16;  // this wave's 16 cols

    // A fragment: lane L holds row (m0+L)'s 16 K nibbles.
    const int a_row = (m0 + lane < M) ? (m0 + lane) : (M - 1);
    // B fragment: lane L holds col (b0+L)'s 16 K values.
    const int b_col = (b0 + lane < B) ? (b0 + lane) : (B - 1);

    i32x8 ah = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
    i32x8 al = ah;
    f32x8 facc = (f32x8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // Hoisted: every base is computed ONCE in 64-bit, then advanced by a
    // constant stride per group, so the loop body is 32-bit pointer bumps.
    const uint8_t* arow = W + (long long)a_row * n_groups * 128;
    const int8_t* xrow = Xq + (long long)b_col * K;
    const float* xsp = Xs + (long long)b_col * n_groups;
    // Fold scale pointers: the row for element e is m0 + 2*e + half, which does
    // not depend on the group. Pre-clamp here so the fold has NO branches.
    const uint8_t* swp[8];
    float swm[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int mr = m0 + 2 * e + half;
        const int rs = (mr < M) ? mr : (M - 1);
        swp[e] = W + side_base + (long long)rs * n_groups * (long long)side_stride;
        swm[e] = (mr < M) ? 1.0f : 0.0f;
    }

    for (int g = 0; g < n_groups; ++g) {
#pragma unroll 4
        for (int s = 0; s < LGROUP / 16; ++s) {
            // weights: 16 nibbles = 8 bytes, already the operand layout
            const i32x2 af = *(const i32x2*)(arow + s * 8);
            // activations: 16 int8 -> 8 bytes of hi nibbles + 8 of lo
            const uint32_t d0 = *(const uint32_t*)(xrow + s * 16);
            const uint32_t d1 = *(const uint32_t*)(xrow + s * 16 + 4);
            const uint32_t d2 = *(const uint32_t*)(xrow + s * 16 + 8);
            const uint32_t d3 = *(const uint32_t*)(xrow + s * 16 + 12);
            i32x2 bh, bl;
            bh[0] = (int)(lpack4((d0 >> 4) & 0x0f0f0f0fu)
                        | (lpack4((d1 >> 4) & 0x0f0f0f0fu) << 16));
            bh[1] = (int)(lpack4((d2 >> 4) & 0x0f0f0f0fu)
                        | (lpack4((d3 >> 4) & 0x0f0f0f0fu) << 16));
            bl[0] = (int)(lpack4(d0 & 0x0f0f0f0fu) | (lpack4(d1 & 0x0f0f0f0fu) << 16));
            bl[1] = (int)(lpack4(d2 & 0x0f0f0f0fu) | (lpack4(d3 & 0x0f0f0f0fu) << 16));
            ah = LWH(af, bh, ah);
            al = LWL(af, bl, al);
        }
        // fold: this group's weight scale (per row) x activation scale (per col)
        const float sx = *xsp;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            // branch-free: out-of-range rows carry swm=0 and contribute nothing
            facc[e] += (float)(16 * ah[e] + al[e]) * (l_f16(swp[e]) * swm[e]) * sx;
        }
        ah = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
        al = ah;
        arow += 128;
        xrow += LGROUP;
        xsp += 1;
#pragma unroll
        for (int e = 0; e < 8; ++e) swp[e] += side_stride;
    }

    if (b0 + lane >= B) return;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int mr = m0 + 2 * e + half;
        if (mr < M) Y[(long long)(b0 + lane) * M + mr] = facc[e];
    }
}
