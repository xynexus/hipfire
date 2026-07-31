// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// Fused gate/up projection + SwiGLU for the reproduced decoder layer.
//
// The FFN's first half is embarrassingly local: gate and up read the SAME
// activation and produce the SAME output rows, and SwiGLU combines them
// element-wise. So one core can do all three for its slice with **no cross-core
// dependency and nothing leaving the tile** — the 8192-wide intermediate is
// never materialised in memory at all, which is the point of fusing.
//
// (down_proj is the opposite: its activation is the whole 8192-wide SwiGLU
// output, so it needs every core's slice and is a separate phase.)
//
//   wtile = [gate tile][up tile], each the standard q4_1 tile for NROWS rows
//   out   = NROWS bf16 of silu(gate) * up
//
// Compile-time: -DDIM_K -DDIM_NROWS, as for the plain GEMV.

#include "flm_q4_1_tile.h"

namespace {
// exp(-x) = exp2(-x * log2 e). The vector exp2 is the only exponential on the
// core — there is no scalar libm (`undefined symbol: exp2f`).
constexpr float LOG2E = 1.4426950408889634f;
constexpr int SLANES = 32;
static_assert(NROWS <= SLANES, "SwiGLU evaluates one 32-lane exp2 per tile");
} // namespace

extern "C" __attribute__((noinline)) void
flm_ffn_gate_up(const bfloat16 *restrict act, const uint8 *restrict wtile,
                bfloat16 *restrict out) {
  float g[NROWS], u[NROWS];
  flm_q4_1_tile(act, wtile, g);
  flm_q4_1_tile(act, wtile + TILE_TOTAL, u);   // past the gate tile AND its trailer

  // One 32-lane exp2 covers the whole tile; NROWS is 8 or 16, so the spare
  // lanes are wasted but a second call would not be cheaper.
  aie::vector<float, SLANES> e;
  e = aie::zeros<float, SLANES>();
  for (int r = 0; r < NROWS; ++r)
    e[r] = -g[r] * LOG2E;
  const auto s = aie::exp2<bfloat16>(e);

  for (int r = 0; r < NROWS; ++r)
    out[r] = bfloat16(g[r] / (1.0f + float(s[r])) * u[r]);
}
