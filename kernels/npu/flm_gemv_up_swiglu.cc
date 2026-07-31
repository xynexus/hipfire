// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// up_proj half of the FFN's first stage, with SwiGLU applied in-core against
// the gate result stashed by flm_gemv_gate.cc:
//
//     out[r] = silu(gate[r]) * up[r]
//
// silu(g) = g / (1 + exp(-g)), and exp(-x) = exp2(-x * log2 e) — the vector
// exp2 is the only exponential on the core (no scalar libm). Its accuracy is
// the AIE2P NLF floor, 3.54% mean / 5.86% max, which is where this stage's
// error sits; the scalar polynomial in tools/npu/softmax_bf16.cc is ~10x more
// accurate and is the upgrade path, not the first build.

#include "flm_q4_1_tile.h"

namespace {
constexpr float LOG2E = 1.4426950408889634f;
constexpr int SLANES = 32;
static_assert(NROWS <= SLANES, "one 32-lane exp2 covers the tile");
} // namespace

extern float g_gate[];

extern "C" __attribute__((noinline)) void
flm_gemv_up_swiglu(const bfloat16 *restrict act, const uint8 *restrict wtile,
                   bfloat16 *restrict out) {
  float u[NROWS];
  flm_q4_1_tile(act, wtile, u);

  aie::vector<float, SLANES> e = aie::zeros<float, SLANES>();
  for (int r = 0; r < NROWS; ++r)
    e[r] = -g_gate[r] * LOG2E;
  const auto s = aie::exp2<bfloat16>(e);

  for (int r = 0; r < NROWS; ++r)
    out[r] = bfloat16(g_gate[r] / (1.0f + float(s[r])) * u[r]);
}
