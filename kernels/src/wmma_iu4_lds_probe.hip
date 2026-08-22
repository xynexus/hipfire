// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// iu4 WMMA OPERAND-SUPPLY probe (wave32).
//
// `wmma_iu4_noop_w32` deliberately holds its operands loop-invariant, so it
// measures ISSUE and says so. That leaves the next question unanswered, and it
// is the one a real kernel actually pays: once both operands must be re-read
// from LDS every instruction, what rate survives?
//
// This is the same CHAINS sweep and the same accumulate-and-sink shape, with one
// change: A and B are read from LDS each iteration at an address that varies
// with the loop counter, so nothing can be hoisted into registers.
//
// Why it should be nearly free, and therefore why measuring it matters:
//   * one WMMA is 16 cycles (`v_wmma_i32_16x16x16_iu4`, gfx1151);
//   * its operands are 2 x ds_read_b64 per lane = 2 x 256 B per wave32;
//   * LDS is 64 banks x 4 B (rdna35:703-706), so a wave32 ds_read_b64 is one
//     conflict-free cycle when lanes are contiguous.
// So the LDS pipe needs ~2 cycles per 16 cycles of matrix work. If the measured
// rate drops much below the issue-rate curve, operand supply is NOT free and
// that -- not the matrix core -- is what a GEMM has to be designed around.
//
// The stride is 8 B x lane so each lane reads its own contiguous 8 B: 32 lanes x
// 8 B = 256 B, exactly one full bank sweep, no conflicts.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t i32x2 __attribute__((ext_vector_type(2)));
typedef int32_t i32x8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));
#define W32(a, b, c) __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, false)

// 8 KB per operand plane: big enough that the compiler cannot cache it, small
// enough to leave occupancy alone (16 KB total, 4 workgroups per 64 KB half).
#define LDSP_WORDS 2048
#define LDSP_RING  255      // (LDSP_WORDS/8 - 1), power-of-two-minus-one

#define LDS_PROBE_W32(NAME, C)                                                 \
__launch_bounds__(256) extern "C" __global__ void NAME(                        \
    int32_t* __restrict__ out, int iters) {                                    \
    __shared__ int32_t sa[LDSP_WORDS];                                         \
    __shared__ int32_t sb[LDSP_WORDS];                                         \
    const int t = threadIdx.x;                                                 \
    for (int i = t; i < LDSP_WORDS; i += 256) {                                \
        sa[i] = i * 7 + 1;                                                     \
        sb[i] = i * 5 + 2;                                                     \
    }                                                                          \
    __syncthreads();                                                           \
    const int lane = t & 31;                                                   \
    i32x8 acc[C];                                                              \
    _Pragma("unroll") for (int c = 0; c < (C); ++c)                            \
        acc[c] = (i32x8){c, c+1, c+2, c+3, c+4, c+5, c+6, c+7};                \
    for (int i = 0; i < iters; ++i) {                                          \
        /* vary the slot with i so the loads cannot be hoisted; lane*2 keeps   \
           each lane on its own contiguous 8 B => conflict-free */             \
        const int slot = ((i & LDSP_RING) * 8) & (LDSP_WORDS - 1);             \
        const int off = (slot + lane * 2) & (LDSP_WORDS - 2);                  \
        _Pragma("unroll") for (int c = 0; c < (C); ++c) {                      \
            const i32x2 a = *(const i32x2*)(sa + off);                         \
            const i32x2 b = *(const i32x2*)(sb + off);                         \
            acc[c] = W32(a, b, acc[c]);                                        \
        }                                                                      \
    }                                                                          \
    int32_t s = 0;                                                             \
    _Pragma("unroll") for (int c = 0; c < (C); ++c)                            \
        s += acc[c][0]+acc[c][1]+acc[c][2]+acc[c][3]+acc[c][4]+acc[c][5]+acc[c][6]+acc[c][7]; \
    if (t == 255) out[blockIdx.x] = s;                                         \
}

// Rung 2: same LDS-fed chain, plus the PER-GROUP FOLD our compact GEMM must do.
// A compact group is 256 weights = 16 WMMA at K=16, after which the i32
// accumulator has to be converted to f32, scaled by the group's f16 weight scale
// times the activation scale, added into an f32 accumulator, and zeroed. That is
// the structure `gemm_oq_compact_iu4x2_wmma` and its wave64 twin both carry, and
// it is what forces a second accumulator set and caps WNt at 4.
//
// Everything else is identical to the plain LDS probe, so the delta between the
// two curves is the fold and nothing else.
#define LDS_FOLD_W32(NAME, C)                                                  \
__launch_bounds__(256) extern "C" __global__ void NAME(                        \
    int32_t* __restrict__ out, int iters) {                                    \
    __shared__ int32_t sa[LDSP_WORDS];                                         \
    __shared__ int32_t sb[LDSP_WORDS];                                         \
    const int t = threadIdx.x;                                                 \
    for (int i = t; i < LDSP_WORDS; i += 256) {                                \
        sa[i] = i * 7 + 1;                                                     \
        sb[i] = i * 5 + 2;                                                     \
    }                                                                          \
    __syncthreads();                                                           \
    const int lane = t & 31;                                                   \
    i32x8 acc[C];                                                              \
    float8 facc[C];                                                            \
    _Pragma("unroll") for (int c = 0; c < (C); ++c) {                          \
        acc[c] = (i32x8){c, c+1, c+2, c+3, c+4, c+5, c+6, c+7};                \
        facc[c] = (float8){0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};                   \
    }                                                                          \
    const float sx = 1.0009765625f;                                            \
    for (int i = 0; i < iters; ++i) {                                          \
        const int slot = ((i & LDSP_RING) * 8) & (LDSP_WORDS - 1);             \
        const int off = (slot + lane * 2) & (LDSP_WORDS - 2);                  \
        _Pragma("unroll") for (int c = 0; c < (C); ++c) {                      \
            const i32x2 a = *(const i32x2*)(sa + off);                         \
            const i32x2 b = *(const i32x2*)(sb + off);                         \
            acc[c] = W32(a, b, acc[c]);                                        \
        }                                                                      \
        /* 16 WMMA per 256-weight group -> fold and reset, as the real kernel */\
        if ((i & 15) == 15) {                                                  \
            const float sw = __builtin_amdgcn_rcpf((float)(i + 2));            \
            _Pragma("unroll") for (int c = 0; c < (C); ++c) {                  \
                _Pragma("unroll") for (int e = 0; e < 8; ++e)                  \
                    facc[c][e] += (float)acc[c][e] * sw * sx;                  \
                acc[c] = (i32x8){0,0,0,0,0,0,0,0};                             \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    float s = 0.f;                                                             \
    _Pragma("unroll") for (int c = 0; c < (C); ++c)                            \
        _Pragma("unroll") for (int e = 0; e < 8; ++e) s += facc[c][e];         \
    if (t == 255) out[blockIdx.x] = (int32_t)s;                                \
}

LDS_FOLD_W32(wmma_iu4_fold_w32_c1, 1)
LDS_FOLD_W32(wmma_iu4_fold_w32_c2, 2)
LDS_FOLD_W32(wmma_iu4_fold_w32_c4, 4)
LDS_FOLD_W32(wmma_iu4_fold_w32_c8, 8)
LDS_FOLD_W32(wmma_iu4_fold_w32_c16, 16)

LDS_PROBE_W32(wmma_iu4_lds_w32_c1, 1)
LDS_PROBE_W32(wmma_iu4_lds_w32_c2, 2)
LDS_PROBE_W32(wmma_iu4_lds_w32_c4, 4)
LDS_PROBE_W32(wmma_iu4_lds_w32_c8, 8)
LDS_PROBE_W32(wmma_iu4_lds_w32_c16, 16)
