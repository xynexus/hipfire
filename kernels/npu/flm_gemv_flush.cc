// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// K-chunked q4_1 GEMV, final chunk: adds this chunk, applies the residual from
// the broadcast's aux half, emits, and clears the slot for the next token.
//
// See flm_gemv_acc.cc for why down_proj is chunked at all.

#include "flm_q4_1_tile.h"

#ifndef DIM_ACCN
#define DIM_ACCN 128
#endif

extern float g_acc_down[];

#ifndef RESID_FROM_STASH
#define RESID_FROM_STASH 0
#endif
#if RESID_FROM_STASH
#ifndef DIM_RESN
#define DIM_RESN 128
#endif
// Written by flm_gemv_residual in phase P3, on this same core. See that file
// for why the residual does not travel through the broadcast.
extern float g_resid[];
#endif

// Emits **bf16**, not f32. This is the layer's output `x_out`, which is the
// next layer's residual stream, and every inter-phase value in the fused layer
// is bf16 — the broadcast object is bf16, so an f32 result would have to be
// narrowed before the next phase could read it anyway. It also halves the
// result traffic and lets one result object shape serve every phase.
extern "C" __attribute__((noinline)) void
flm_gemv_flush(const bfloat16 *restrict act_aux, const uint8 *restrict wtile,
               bfloat16 *restrict out) {
  float part[NROWS];
  flm_q4_1_tile(act_aux, wtile, part);
  const int base = tile_row_base(wtile);
  const int slot = base % DIM_ACCN;
  const bfloat16 *restrict aux = act_aux + K;
  for (int r = 0; r < NROWS; ++r) {
#if RESID_FROM_STASH
    const float res = g_resid[(base + r) % DIM_RESN];
#else
    const float res = float(aux[base + r]);   // standalone harnesses
#endif
    out[r] = bfloat16(g_acc_down[slot + r] + part[r] + res);
    g_acc_down[slot + r] = 0.0f;   // ready for the next token
  }
}
