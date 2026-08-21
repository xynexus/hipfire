// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// No HIPFIRE_COMPILER_FLAGS line: wave32 is the RDNA default, and there is no
// `-mwavefrontsize32` argument — clang rejects it outright.
//
// Pure iu4 WMMA ISSUE-RATE probe, WAVE32 twin of the w64 file. No global memory, no LDS, no address
// arithmetic — a dependency chain of WMMA and nothing else.
//
// This exists because the "ceiling" numbers in use were both indirect. The
// 99.2 TOPS figure comes from a separate synthetic probe, and the ~70 TOPS
// figure came from ABLATING a real kernel, which still carried its register
// pressure, loop bookkeeping and occupancy. Neither answers the question a
// kernel designer actually needs: at a given number of INDEPENDENT accumulator
// chains, what rate can the matrix core sustain?
//
// CHAINS is the sweep variable. One WMMA depends on the previous WMMA writing
// the same accumulator, so a single chain exposes the full instruction latency;
// C chains give the scheduler C independent instructions to interleave. Where
// the curve flattens is the minimum accumulator count a real kernel needs, and
// its value is the honest ceiling to design against.
//
// Operands are loop-invariant on purpose: this measures ISSUE, not operand
// supply. `out` is written once per block from the accumulators so nothing is
// dead-code eliminated.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t i32x2 __attribute__((ext_vector_type(2)));
typedef int32_t i32x4 __attribute__((ext_vector_type(4)));
typedef int32_t i32x8 __attribute__((ext_vector_type(8)));
#define W32(a, b, c) __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, false)

#define NOOP_W32(NAME, C)                                                      \
__launch_bounds__(256) extern "C" __global__ void NAME(                        \
    int32_t* __restrict__ out, int iters) {                                    \
    const int t = threadIdx.x;                                                 \
    i32x2 a = (i32x2){(int)(t * 7 + 1), (int)(t * 13 + 3)};                    \
    i32x2 b = (i32x2){(int)(t * 5 + 2), (int)(t * 11 + 5)};                    \
    i32x8 acc[C];                                                              \
    _Pragma("unroll") for (int c = 0; c < (C); ++c)                            \
        acc[c] = (i32x8){c, c+1, c+2, c+3, c+4, c+5, c+6, c+7};                              \
    for (int i = 0; i < iters; ++i) {                                          \
        _Pragma("unroll") for (int c = 0; c < (C); ++c)                        \
            acc[c] = W32(a, b, acc[c]);                                        \
    }                                                                          \
    int32_t s = 0;                                                             \
    _Pragma("unroll") for (int c = 0; c < (C); ++c)                            \
        s += acc[c][0]+acc[c][1]+acc[c][2]+acc[c][3]+acc[c][4]+acc[c][5]+acc[c][6]+acc[c][7];                    \
    if (t == 255) out[blockIdx.x] = s;                                         \
}

NOOP_W32(wmma_iu4_noop_w32_c1, 1)
NOOP_W32(wmma_iu4_noop_w32_c2, 2)
NOOP_W32(wmma_iu4_noop_w32_c4, 4)
NOOP_W32(wmma_iu4_noop_w32_c8, 8)
NOOP_W32(wmma_iu4_noop_w32_c16, 16)
NOOP_W32(wmma_iu4_noop_w32_c32, 32)
