#include <aie_api/aie.hpp>
#include <stdint.h>

using MMUL = aie::mmul<4, 16, 16, int8, int4>;
constexpr unsigned A_SCALE_OFFSET = 6144;
constexpr unsigned W_SCALE_OFFSET = 12288;

static inline aie::vector<int8, MMUL::size_A>
load_r34_a(const int8_t *__restrict activations, unsigned kt) {
  aie::vector<int8, MMUL::size_A> a;
  const unsigned first = kt * 128;
  for (unsigned row = 0; row < 4; ++row)
    for (unsigned k = 0; k < 16; ++k) {
      const unsigned source =
          k < 8 ? first + row * 8 + k
                : first + 64 + row * 8 + (k - 8);
      a.set(activations[source], row * 16 + k);
    }
  return a;
}

extern "C" void r60_scaled_group(const int8_t *__restrict activations,
                                  const int8_t *__restrict packed_weights,
                                  float *__restrict output,
                                  const int32_t accumulate) {
  MMUL product;
  for (unsigned kt = 0; kt < 16; ++kt) {
    const auto *weights = reinterpret_cast<const int4 *>(packed_weights + kt * 128);
    const auto a = load_r34_a(activations, kt);
    if (kt == 0)
      product.mul(a, aie::load_v<MMUL::size_B>(weights));
    else
      product.mac(a, aie::load_v<MMUL::size_B>(weights));
  }

  const auto dots = product.template to_vector<int32>();
  const auto weight_scales = aie::load_v<16>(
      reinterpret_cast<const float *>(packed_weights + W_SCALE_OFFSET));
  const auto *activation_scales =
      reinterpret_cast<const float *>(activations + A_SCALE_OFFSET);
  for (unsigned row = 0; row < 4; ++row) {
    auto scaled = aie::mul(aie::to_float(dots.extract<16>(row)), weight_scales)
                      .to_vector<float>();
    scaled = aie::mul(scaled, activation_scales[row]).to_vector<float>();
    if (accumulate)
      scaled = aie::add(scaled, aie::load_v<16>(output + row * 16));
    aie::store_v(output + row * 16, scaled);
  }
}
