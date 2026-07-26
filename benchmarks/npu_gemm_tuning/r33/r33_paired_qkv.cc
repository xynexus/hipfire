// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include "aie_kernels/aie_kernel_utils.h"
#include <stdint.h>

namespace {
constexpr int LM = 3;
constexpr int LN = 2;
constexpr int KT = 32;
using W8_MMUL = aie::mmul<8, 8, 8, int8, int8>;
constexpr int SA = W8_MMUL::size_A;
constexpr int SB = 2 * W8_MMUL::size_B;
constexpr int SC = 2 * W8_MMUL::size_C;
constexpr int A_DATA = LM * KT * SA;
constexpr int W_DATA = LN * KT * SB;
constexpr int PAIRED_SCALE_BASE = 6272;
constexpr int PAIRED_SCALE_STRIDE = 64;

static inline aie::vector<int32, 16>
join_rows(aie::vector<int32, W8_MMUL::size_C> lo,
          aie::vector<int32, W8_MMUL::size_C> hi, int row) {
  switch (row) {
  case 0: return aie::concat(lo.extract<8>(0), hi.extract<8>(0));
  case 1: return aie::concat(lo.extract<8>(1), hi.extract<8>(1));
  case 2: return aie::concat(lo.extract<8>(2), hi.extract<8>(2));
  case 3: return aie::concat(lo.extract<8>(3), hi.extract<8>(3));
  case 4: return aie::concat(lo.extract<8>(4), hi.extract<8>(4));
  case 5: return aie::concat(lo.extract<8>(5), hi.extract<8>(5));
  case 6: return aie::concat(lo.extract<8>(6), hi.extract<8>(6));
  default: return aie::concat(lo.extract<8>(7), hi.extract<8>(7));
  }
}

__attribute__((noinline))
void projection_group(const int8_t *restrict activations,
                      const int8_t *restrict weights,
                      const float *restrict weight_scales,
                      float *restrict output, bool accumulate) {
  const float *activation_scales =
      reinterpret_cast<const float *>(activations + A_DATA);
  for (int im = 0; im < LM; ++im)
    for (int jn = 0; jn < LN; ++jn) {
      W8_MMUL lo, hi;
      auto a = aie::load_v<SA>(activations + (im * KT) * SA);
      const int8_t *w = weights + (jn * KT) * SB;
      lo.mul(a, aie::load_v<W8_MMUL::size_B>(w));
      hi.mul(a, aie::load_v<W8_MMUL::size_B>(w + W8_MMUL::size_B));
      for (int k = 1; k < KT; ++k) {
        a = aie::load_v<SA>(activations + (im * KT + k) * SA);
        w = weights + (jn * KT + k) * SB;
        lo.mac(a, aie::load_v<W8_MMUL::size_B>(w));
        hi.mac(a, aie::load_v<W8_MMUL::size_B>(w + W8_MMUL::size_B));
      }
      const auto vlo = lo.to_vector<int32>();
      const auto vhi = hi.to_vector<int32>();
      const auto weight_scale = aie::load_v<16>(weight_scales + jn * 16);
      for (int row = 0; row < 8; ++row) {
        const int offset = (im * LN + jn) * SC + row * 16;
        auto scaled =
            aie::mul(aie::to_float(join_rows(vlo, vhi, row)), weight_scale)
                .to_vector<float>();
        scaled =
            aie::mul(scaled, activation_scales[im * 8 + row]).to_vector<float>();
        if (accumulate)
          scaled = aie::add(scaled, aie::load_v<16>(output + offset));
        aie::store_v(output + offset, scaled);
      }
    }
}
} // namespace

extern "C" {
void r33_w8_projection_group_pair(const int8_t *activations,
                                  const int8_t *paired_weights, float *accum0,
                                  float *accum1, int32_t pair_index,
                                  int32_t accumulate) {
  const float *scales = reinterpret_cast<const float *>(
      activations + PAIRED_SCALE_BASE) + pair_index * PAIRED_SCALE_STRIDE;
  projection_group(activations, paired_weights, scales, accum0,
                   accumulate != 0);
  projection_group(activations, paired_weights + W_DATA,
                   scales + PAIRED_SCALE_STRIDE / 2, accum1,
                   accumulate != 0);
}
}
