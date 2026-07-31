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

  // Build the exp2 argument with a LOAD, not element-by-element.
  //
  // The obvious form
  //     aie::vector<float, SLANES> e = aie::zeros<float, SLANES>();
  //     for (int r = 0; r < NROWS; ++r) e[r] = -g_gate[r] * LOG2E;
  // compiles clean and **silently does nothing** — `operator[]` on an
  // aie::vector yields a temporary, so every write is dropped and `e` stays
  // zero. exp2(0) is 1, the sigmoid collapses to a constant 1/(1+1) = 1/2, and
  // the kernel computes `g*u/2` for every row. That is close enough to
  // SwiGLU on small activations to pass a loose tolerance (ffn_alt.py, on
  // unnormalised inputs, showed 1.6e-04) and only shows up once the
  // activations are RMSNorm-scaled: the measured sigmoid was 0.4990-0.5008
  // for every row whatever g was, and `g*u/2` matched the device to 4 digits.
  //
  // g_gate is SLANES wide and static, so lanes [NROWS, SLANES) are zero for
  // the life of the program and their exp2 is a harmless 1.
  const auto gv = aie::load_v<SLANES>(g_gate);
  const auto e = aie::mul(gv, aie::broadcast<float, SLANES>(-LOG2E))
                     .template to_vector<float>();
  const auto s = aie::exp2<bfloat16>(e);

  for (int r = 0; r < NROWS; ++r)
    out[r] = bfloat16(g_gate[r] / (1.0f + float(s[r])) * u[r]);
}
