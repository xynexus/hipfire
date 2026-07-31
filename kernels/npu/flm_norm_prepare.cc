// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// RMSNorm fused into the GEMV's activation prologue — replaces
// `flm_asum_prepare` where the projection is preceded by a norm.
//
// WHY this exists rather than a standalone RMSNorm dispatch. At the measured
// **92.9 us per dispatch**, an operator's cost is set by its size, and a
// layer's RMSNorm moves 4 KB: it is ~99.9% fixed cost. Two norms per layer over
// 16 layers is **2.97 ms/token** — a fifth of the entire 13.55 ms streaming
// floor — spent on 128 KB of actual data. Elementwise operators have to ride
// along with a large one, and the GEMV prologue is exactly the right place:
// `flm_asum_prepare` ALREADY walks the whole activation to compute the q4_1
// block sums, so the norm's sum-of-squares rides in a pass that is already paid
// for.
//
// Passes over the activation:
//   standalone RMSNorm (2) + asum_prepare (1) = 3
//   this                                      = 2
// so fusing is not merely dispatch-free, it is one pass cheaper than the two
// operators were separately.
//
// The activation is normalised **in place** in the ObjectFifo buffer, so the
// GEMV that follows needs no change at all — it reads the same pointer.
//
// The norm weight rides in the SAME buffer, immediately after the activation:
// `[act K][norm_weight K]`. A separate fifo for it would be a third DMA input
// on the core tile, and a core tile has only **2 input channels** — the placer
// rejects it with "reduce the LTO's DMA fanin".
//
// out = x * rsqrt(mean(x^2) + 1e-5) * w   (llama-3.2-1B's rms_norm_eps)
//
// Compile-time: -DDIM_K, -DDIM_NROWS (for g_asum's extent, shared with the GEMV).

#include "flm_q4_1_tile.h"

namespace {
constexpr float RMS_EPS = 1.0e-5f;
} // namespace

extern "C" __attribute__((noinline)) void
flm_norm_prepare(bfloat16 *restrict actnw) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  bfloat16 *restrict act = actnw;              // [0, K)
  const bfloat16 *restrict nw = actnw + K;     // [K, 2K)

  // Pass 1: sum of squares, in float.
  aie::accum<accfloat, LANES> ss;
  ss.from_vector(aie::zeros<float, LANES>());
  for (int i = 0; i < K; i += LANES) {
    const auto v = aie::load_v<LANES>(act + i);
    ss = aie::mac(ss, v, v);
  }
  const float mean_sq =
      aie::reduce_add(ss.template to_vector<float>()) / float(K);

  // No scalar libm on the core: rsqrt via the vector unit, lane 0 extracted.
  const float inv =
      aie::invsqrt(aie::broadcast<float, LANES>(mean_sq + RMS_EPS))[0];
  const auto iv = aie::broadcast<bfloat16, LANES>(bfloat16(inv));

  // Pass 2: normalise in place, and accumulate the q4_1 block sums of the
  // NORMALISED activation in the same sweep. A q4_1 block is BLK=32 elements,
  // which is exactly one 32-lane vector, so a block is one iteration.
  static_assert(BLK == LANES, "one q4_1 block is one vector");
  const auto ones = aie::broadcast<bfloat16, LANES>(bfloat16(1.0f));
  for (int b = 0; b < NB; ++b) {
    const int i = b * BLK;
    const auto xn = aie::mul(aie::load_v<LANES>(act + i), iv)
                        .template to_vector<bfloat16>();
    const auto o = aie::mul(xn, aie::load_v<LANES>(nw + i))
                       .template to_vector<bfloat16>();
    aie::store_v(act + i, o);
    aie::accum<accfloat, LANES> t;
    t.from_vector(aie::zeros<float, LANES>());
    t = aie::mac(t, o, ones);
    g_asum[b] = bfloat16(aie::reduce_add(t.template to_vector<float>()));
  }
}
