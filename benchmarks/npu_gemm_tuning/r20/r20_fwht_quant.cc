// Vector-AWQ experiment against R19's exact activation preprocessing contract.
#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>

__attribute__((noinline)) static void
awq_sign16(const float *__restrict input, const float *__restrict awq,
           const float *__restrict signs, float *__restrict scratch) {
  auto values = aie::load_v<16>(input);
  auto divisors = aie::load_v<16>(awq);
  auto sign = aie::load_v<16>(signs);
  auto divided = aie::div(values, divisors).template to_vector<float>();
  aie::store_v(scratch,
               aie::mul(divided, sign).template to_vector<float>());
}

__attribute__((noinline)) static float
post_sign16(float *__restrict scratch, const float *__restrict signs) {
  auto values = aie::load_v<16>(scratch);
  auto sign = aie::load_v<16>(signs);
  auto scaled = aie::mul(aie::mul(values, sign).template to_vector<float>(),
                         aie::broadcast<float, 16>(0.0625f))
                    .template to_vector<float>();
  aie::store_v(scratch, scaled);
  return aie::reduce_max(aie::abs(scaled));
}

template <unsigned STRIDE>
__attribute__((noinline)) static void
fwht16_stage(float *__restrict scratch) {
  for (int block = 0; block < 256; block += 16) {
    auto values = aie::load_v<16>(scratch + block);
    auto a = aie::filter_even(values, STRIDE);
    auto b = aie::filter_odd(values, STRIDE);
    auto sum = aie::add(a, b);
    auto difference = aie::sub(a, b);
    aie::store_v(scratch + block,
                 aie::concat(aie::interleave_zip(sum, difference, STRIDE)));
  }
}

extern "C" void r20_fwht_quant(const float *__restrict input,
                                const float *__restrict param,
                                int8 *__restrict output,
                                float *__restrict scratch) {
  constexpr int GROUP = 256;
  constexpr int GROUPS = 5;
  constexpr int PAD_K = GROUP * GROUPS;
  const float *awq = param;
  const float *signs1 = param + PAD_K;
  const float *signs2 = signs1 + GROUP;
  float *scales = reinterpret_cast<float *>(output + PAD_K);
  for (int group = 0; group < GROUPS; group++) {
    const int base = group * GROUP;
    for (int i = 0; i < GROUP; i += 16)
      awq_sign16(input + base + i, awq + base + i, signs1 + i, scratch + i);

    fwht16_stage<1>(scratch);
    fwht16_stage<2>(scratch);
    fwht16_stage<4>(scratch);
    fwht16_stage<8>(scratch);
    for (int stride = 16; stride < GROUP; stride <<= 1)
      for (int block = 0; block < GROUP; block += 2 * stride)
        for (int i = 0; i < stride; i += 16) {
          auto a = aie::load_v<16>(scratch + block + i);
          auto b = aie::load_v<16>(scratch + block + i + stride);
          aie::store_v(scratch + block + i, aie::add(a, b));
          aie::store_v(scratch + block + i + stride, aie::sub(a, b));
        }

    float max_abs = 0.0f;
    for (int i = 0; i < GROUP; i += 16) {
      const float local_max = post_sign16(scratch + i, signs2 + i);
      if (local_max > max_abs) max_abs = local_max;
    }
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
    scales[group] = scale;

    const auto old_rounding =
        aie::swap_rounding(aie::rounding_mode::symmetric_inf);
    const auto old_saturation =
        aie::swap_saturation(aie::saturation_mode::symmetric);
    if (scale > 0.0f) {
      for (int i = 0; i < GROUP; i += 16) {
        auto values = aie::load_v<16>(scratch + i);
        auto normalized =
            aie::div(values, scale).template to_vector<float>();
        aie::store_v(output + base + i, aie::to_fixed<int8>(normalized));
      }
    } else {
      for (int i = 0; i < GROUP; i++) output[base + i] = 0;
    }
    aie::set_saturation(old_saturation);
    aie::set_rounding(old_rounding);
  }
  for (int i = GROUPS; i < 8; i++) scales[i] = 0.0f;
}
