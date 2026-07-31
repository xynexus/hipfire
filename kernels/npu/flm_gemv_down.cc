// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// K-chunked q4_1 GEMV for down_proj: flm_gemv_acc and flm_gemv_flush as one
// entry point, selected by the tile's own flag.
//
// **This exists to remove a LOOP BODY, not to save kernel bytes.** Merging the
// two kernels saves only ~80 B of code — measured. What it saves is a phase
// body: P5 currently has a `range_` over the accumulating chunks plus a
// separate flush block, and each body costs ~2800 B of generated acquire /
// release / prepare structure. With one kernel P5 becomes a single loop over
// NCHUNK, and body count is what the 16 KB budget is actually spent on.
//
// The flag rides the tile trailer, so the choice is per tile at runtime — the
// same core runs three accumulating chunks and one flushing chunk in one
// `range_`, which a compile-time switch cannot express.
//
// Compile-time: -DDIM_ACCN, -DRESID_FROM_STASH, -DXOUT_TO_STASH, -DDIM_RESN.

#include "flm_q4_1_tile.h"

#ifndef DIM_ACCN
#define DIM_ACCN 128
#endif

alignas(64) float g_acc_down[DIM_ACCN];

#ifndef RESID_FROM_STASH
#define RESID_FROM_STASH 0
#endif
#ifndef XOUT_TO_STASH
#define XOUT_TO_STASH 0
#endif
#if RESID_FROM_STASH || XOUT_TO_STASH
#ifndef DIM_RESN
#define DIM_RESN 256
#endif
extern float g_resid[];
#endif

extern "C" __attribute__((noinline)) void
flm_gemv_down(const bfloat16 *restrict act_aux, const uint8 *restrict wtile,
              bfloat16 *restrict out) {
  float part[NROWS];
  flm_q4_1_tile(act_aux, wtile, part);
  const int base = tile_row_base(wtile);
  const int slot = base % DIM_ACCN;

  if (!tile_flags(wtile)) {                 // an accumulating chunk
    for (int r = 0; r < NROWS; ++r)
      g_acc_down[slot + r] += part[r];
    return;
  }

  const bfloat16 *restrict aux = act_aux + K;
  for (int r = 0; r < NROWS; ++r) {
#if RESID_FROM_STASH
    const float res = g_resid[(base + r) % DIM_RESN];
#else
    const float res = float(aux[base + r]);
#endif
    const float v = g_acc_down[slot + r] + part[r] + res;
#if XOUT_TO_STASH
    g_resid[(base + r) % DIM_RESN] = v;
    (void)out;
#else
    out[r] = bfloat16(v);
#endif
    g_acc_down[slot + r] = 0.0f;
  }
}
