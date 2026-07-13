// SPDX-License-Identifier: Apache-2.0
// Size-conscious R15 W4 projection: one implementation for init and accumulate.

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int LM = 6;
constexpr int LN = 6;
constexpr int KT = 16;
using MMUL = aie::mmul<4, 16, 16, int8, int4>;
constexpr int SA = MMUL::size_A;
constexpr int SB = MMUL::size_B / 2;
constexpr int SC = MMUL::size_C;
constexpr int A_DATA = LM * KT * SA;
constexpr int W_DATA = LN * KT * SB;
} // namespace

extern "C" void r70_w4_scaled_group(
    const int8 *__restrict activation_payload,
    const int8 *__restrict weight_payload, int32 *__restrict output_bits,
    int32_t accumulate) {
  aie::set_rounding(aie::rounding_mode::floor);
  aie::set_saturation(aie::saturation_mode::none);
  const float *activation_scales =
      reinterpret_cast<const float *>(activation_payload + A_DATA);
  const float *weight_scales =
      reinterpret_cast<const float *>(weight_payload + W_DATA);
  float *output = reinterpret_cast<float *>(output_bits);

  for (int im = 0; im < LM; ++im)
    for (int jn = 0; jn < LN; ++jn) {
      auto sum = aie::zeros<int32, SC>();
      for (int k = 0; k < KT; k += 2) {
        const auto a0 =
            aie::load_v<SA>(activation_payload + (im * KT + k) * SA);
        const auto w0 = aie::load_v<MMUL::size_B>(
            reinterpret_cast<const int4 *>(weight_payload +
                                            (jn * KT + k) * SB));
        MMUL partial;
        partial.mul(a0, w0);
        const auto a1 =
            aie::load_v<SA>(activation_payload + (im * KT + k + 1) * SA);
        const auto w1 = aie::load_v<MMUL::size_B>(
            reinterpret_cast<const int4 *>(weight_payload +
                                            (jn * KT + k + 1) * SB));
        partial.mac(a1, w1);
        sum = aie::add(sum, partial.to_vector<int32>());
      }
      const auto weight_scale = aie::load_v<16>(weight_scales + jn * 16);
      for (int row = 0; row < 4; ++row) {
        const int offset = (im * LN + jn) * SC + row * 16;
        auto scaled =
            aie::mul(aie::to_float(sum.extract<16>(row)), weight_scale)
                .to_vector<float>();
        scaled =
            aie::mul(scaled,
                     aie::broadcast<float, 16>(activation_scales[im * 4 + row]))
                .to_vector<float>();
        if (accumulate)
          scaled = aie::add(scaled, aie::load_v<16>(output + offset));
        aie::store_v(output + offset, scaled);
      }
    }
}
