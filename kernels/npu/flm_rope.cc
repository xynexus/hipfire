// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// RoPE for the reproduced llama-3.2-1B decoder layer — FULL rotary, half-split.
//
// The existing `tools/npu/rope_rotate_bf16.cc` cannot be reused: it is
// Qwen3.5-specific with `partial_rotary_factor = 0.25`, rotating only
// head_dim/4 dims and passing the rest through. llama-3.2 rotates all of them.
//
// Half-split convention, matching HF's `rotate_half`:
//   for i in [0, HEAD/2):
//     x = in[i], y = in[i + HEAD/2]
//     out[i]          = x*cos[i] - y*sin[i]
//     out[i + HEAD/2] = y*cos[i] + x*sin[i]
//
// `cs` is [cos(HEAD/2)][sin(HEAD/2)] for this token's position — the same
// buffer for every head, so the caller acquires it once.
//
// The subtraction uses `aie::msc` (multiply-subtract). That is the same
// `vmsc.f` that `docs/npu/flm-layer-dataflow.md` used to LOCATE RoPE in FLM's
// array: it occurs in layer.xclbin only in the cols 3-4 cores, because
// `x*cos - y*sin` is essentially the only thing that needs it.
//
// The llama3 frequency scaling is NOT computed here. FLM's container stores
// `rope_freqs.weight` [HEAD/2] = the per-frequency llama3 **divisor**
// ([1,1,...,1, 1.648, 3.297, 9.688, 32,...,32]), so the caller gets the scaled
// frequencies with `inv_freq = base_inv_freq / rope_freqs` — verified to 0.21%
// against a from-scratch llama3 wavelength interpolation.
//
// Compile-time: -DDIM_HEAD (head_dim).

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_HEAD
#define DIM_HEAD 64
#endif

namespace {
constexpr int HEAD = DIM_HEAD;
constexpr int HALF = HEAD / 2;
static_assert(HALF % 16 == 0, "half a head must be a whole number of vectors");
} // namespace

extern "C" __attribute__((noinline)) void
flm_rope(const bfloat16 *restrict in, const bfloat16 *restrict cs,
         bfloat16 *restrict out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const auto x = aie::load_v<HALF>(in);
  const auto y = aie::load_v<HALF>(in + HALF);
  const auto c = aie::load_v<HALF>(cs);
  const auto s = aie::load_v<HALF>(cs + HALF);

  auto lo = aie::mul(x, c);          // x*cos
  lo = aie::msc(lo, y, s);           // -= y*sin
  aie::store_v(out, lo.template to_vector<bfloat16>());

  auto hi = aie::mul(y, c);          // y*cos
  hi = aie::mac(hi, x, s);           // += x*sin
  aie::store_v(out + HALF, hi.template to_vector<bfloat16>());
}
