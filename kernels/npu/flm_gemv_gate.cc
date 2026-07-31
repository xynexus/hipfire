// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// gate_proj half of the FFN's first stage: a plain q4_1 GEMV whose result is
// stashed in-core for the matching up_proj tile. See flm_gemv_up_swiglu.cc.
//
// WHY not the single fused tile that flm_ffn_gate_up.cc uses. That kernel takes
// ONE tile carrying both gate and up weights, so the tile is 2x the size and
// the 64 KB tile memory then allows only 8 rows, against 16 everywhere else.
// Streaming them as ALTERNATING acquires of single tiles keeps each tile at 16
// rows; the weight stream is reordered offline to
// [gate t0][up t0][gate t1][up t1]... — same bytes, no extra DMA channel, no
// extra dispatch, and the 8192-wide intermediate is still never materialised.
//
// **It is not faster, and the row count was never why gate/up runs below its
// ceiling.** That was this kernel's original rationale (8 rows measured ~25%
// worse than 16 on the plain GEMV) and it was measured and falsified: 526.6 us
// against the single fused tile's 514.0 on the same 21.0 MB. Both sit ~55-65 us
// above what the transfer size allows, so the gap is the stage's arithmetic,
// not its geometry. What keeps this kernel is the L1 budget: a fused gate/up
// tile at 16 rows needs 82176 B of double-buffered operand and does not fit at
// all, and at 8 rows its object is 20608 B against every other phase's 20544,
// so the fused layer cannot use one operand shape across its phases with it.
// See docs/npu/flm-refe-log.md, 2026-07-31.
//
// The stash is 64 B per core.

#include "flm_q4_1_tile.h"

alignas(64) float g_gate[DIM_NROWS];

extern "C" __attribute__((noinline)) void
flm_gemv_gate(const bfloat16 *restrict act, const uint8 *restrict wtile) {
  flm_q4_1_tile(act, wtile, g_gate);
}
