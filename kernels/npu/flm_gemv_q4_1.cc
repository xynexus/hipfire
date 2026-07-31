// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// Plain q4_1 decode GEMV entry point. The arithmetic lives in
// `flm_q4_1_tile.h`, shared with the fused FFN kernel; this file exists so the
// symbol has its own translation unit (IRON compiles each ExternalFunction's
// source separately, so two entry points in one file link twice).

#include "flm_q4_1_tile.h"

// The one definition of the shared activation block-sum array. alignas is
// load-bearing: it is vector-loaded, and an unaligned 512-bit load returns
// garbage rather than faulting.
alignas(64) bfloat16 g_asum[DIM_K / 32];

extern "C" __attribute__((noinline)) void
flm_gemv_q4_1(const bfloat16 *restrict act, const uint8 *restrict wtile,
              float *restrict out) {
  flm_q4_1_tile(act, wtile, out);
}
