// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// K-chunked q4_1 GEMV, accumulate phase. See flm_gemv_flush.cc for the last
// chunk.
//
// WHY down_proj is chunked. Its K is 8192, so a monolithic tile needs a 16384 B
// activation (32768 double-buffered) and the 64 KB tile memory then allows only
// **2 rows per weight tile**, against 16 for every other projection. Measured,
// that costs ~54 us/layer of pure geometry. Splitting K into 4 chunks of 2048
// makes down_proj the same shape as everything else: a 4096 B activation and
// 16-row tiles.
//
// WHY it is exact. The GEMV identity is linear in blocks —
//   out[n] = sum_b ( d[n,b]*sum_t q[n,b,t]*a[b,t] + m[n,b]*sum_t a[b,t] )
// — so summing four 2048-wide partials is the same arithmetic as one 8192-wide
// pass, up to float accumulation order. And the container's planar row splits
// on a chunk boundary with no repacking of codes: chunk c is d[64c:64c+64],
// m[64c:64c+64], codes[1024c:1024c+1024], which is exactly a K=2048 tile.
//
// Compile-time: -DDIM_K (2048, the chunk), -DDIM_NROWS, -DDIM_ACCN (rows/core).

#include "flm_q4_1_tile.h"

#ifndef DIM_ACCN
#define DIM_ACCN 128
#endif

// Partial sums across chunks, one slot per output row this core owns.
// `row_base % DIM_ACCN` is the slot: core c owns rows [ACCN*c, ACCN*c+ACCN),
// so the modulo lands each tile on its own slice with no per-core state.
alignas(64) float g_acc_down[DIM_ACCN];

extern "C" __attribute__((noinline)) void
flm_gemv_acc(const bfloat16 *restrict act_aux, const uint8 *restrict wtile) {
  float part[NROWS];
  flm_q4_1_tile(act_aux, wtile, part);
  const int slot = tile_row_base(wtile) % DIM_ACCN;
  for (int r = 0; r < NROWS; ++r)
    g_acc_down[slot + r] += part[r];
}
