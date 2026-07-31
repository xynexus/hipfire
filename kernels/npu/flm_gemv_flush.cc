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

extern "C" __attribute__((noinline)) void
flm_gemv_flush(const bfloat16 *restrict act_aux, const uint8 *restrict wtile,
               float *restrict out) {
  float part[NROWS];
  flm_q4_1_tile(act_aux, wtile, part);
  const int base = tile_row_base(wtile);
  const int slot = base % DIM_ACCN;
  const bfloat16 *restrict aux = act_aux + K;
  for (int r = 0; r < NROWS; ++r) {
    out[r] = g_acc_down[slot + r] + part[r] + float(aux[base + r]);
    g_acc_down[slot + r] = 0.0f;   // ready for the next token
  }
}
