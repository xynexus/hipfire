// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// gate_proj half of the FFN's first stage: a plain q4_1 GEMV whose result is
// stashed in-core for the matching up_proj tile. See flm_gemv_up_swiglu.cc.
//
// WHY not the single fused tile that flm_ffn_gate_up.cc uses. That kernel takes
// ONE tile carrying both gate and up weights, so the tile is 2x the size and
// the 64 KB tile memory then allows only **8 rows**, against 16 everywhere
// else — and 8 rows measured ~25% worse than 16 on the plain GEMV. Streaming
// them as ALTERNATING acquires of single tiles keeps each tile at 16 rows; the
// weight stream is simply reordered offline to
// [gate t0][up t0][gate t1][up t1]... — same bytes, no extra DMA channel, no
// extra dispatch, and the 8192-wide intermediate is still never materialised.
//
// The stash is 64 B per core.

#include "flm_q4_1_tile.h"

alignas(64) float g_gate[DIM_NROWS];

extern "C" __attribute__((noinline)) void
flm_gemv_gate(const bfloat16 *restrict act, const uint8 *restrict wtile) {
  flm_q4_1_tile(act, wtile, g_gate);
}
