// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// Measure `aie::exp2<bfloat16>` against float64, one vector at a time.
//
// **This exists because the attention floor has no model.** Four normalizers
// were fitted and discarded; the mechanism was eventually read out of
// `flm_attn_decode.cc` instead — the softmax weights and the online-rescale
// factor are both `aie::exp2<bfloat16>` results — but the coefficient still does
// not close, because `aie::exp2` is a HARDWARE APPROXIMATION and nothing in this
// repo has ever measured how far off it is.
//
// Two errors are tangled in that call and this separates them:
//
//   * bf16 rounding of an exact result   -- known, 2^-9 relative
//   * the NLF's own approximation error  -- unknown, and the point of this probe
//
// The host compares the device result against BOTH `2**x` in float64 and
// `bfloat16(2**x)`. If the second matches, the NLF is exact to bf16 and the
// attention floor is pure rounding. If it does not, the difference IS the NLF
// error, and it is the missing term.
//
// Input is float32 and output bfloat16, matching the call in the attention
// kernel exactly (`aie::exp2<bfloat16>` over a float vector).
//
// Compile-time: -DDIM_NPROBE (elements, multiple of 32).

#include <aie_api/aie.hpp>

#ifndef DIM_NPROBE
#define DIM_NPROBE 1024
#endif

namespace {
constexpr int NPROBE = DIM_NPROBE;
constexpr int V = 32;                   // the attention kernel's TSEQ width
static_assert(NPROBE % V == 0, "probe length must tile the vector width");
} // namespace

extern "C" __attribute__((noinline)) void
flm_exp2_probe(const float *restrict in, bfloat16 *restrict out) {
  for (int i = 0; i < NPROBE; i += V) {
    // Exactly the shape attention uses: a float vector in, bf16 lanes out.
    const aie::vector<bfloat16, V> y = aie::exp2<bfloat16>(aie::load_v<V>(in + i));
    aie::store_v(out + i, y);
  }
}
