// Vector canonical activation pack consumed immediately by the R15 W4 down MMUL.
#include "../r15/r15_w4_scaled.cc"
#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>

constexpr int GROUP = 256;
constexpr int GROUPS = 5;
constexpr int PAD_K = GROUP * GROUPS;
constexpr int A_DATA = 6144;
constexpr int W_DATA = 12288;
constexpr int W_COLS = 96;
constexpr int GROUP_PARAM_OFFSET = W_DATA + W_COLS * sizeof(float);

template <unsigned STRIDE>
__attribute__((noinline)) static void fwht16_stage(float *__restrict scratch) {
  for (int block = 0; block < GROUP; block += 16) {
    auto values = aie::load_v<16>(scratch + block);
    auto a = aie::filter_even(values, STRIDE);
    auto b = aie::filter_odd(values, STRIDE);
    auto sum = aie::add(a, b);
    auto difference = aie::sub(a, b);
    aie::store_v(scratch + block,
                 aie::concat(aie::interleave_zip(sum, difference, STRIDE)));
  }
}

extern "C" void r21_w4_pack_row(const float *__restrict input,
                                 const int8 *__restrict weight_payload,
                                 int8 *__restrict activation_payload,
                                 float *__restrict scratch, int local_row) {
  const float *params = reinterpret_cast<const float *>(
      weight_payload + GROUP_PARAM_OFFSET);
  const float *awq = params;
  const float *signs1 = awq + GROUP;
  const float *signs2 = signs1 + GROUP;
  for (int i = 0; i < GROUP; i += 16) {
    auto values = aie::load_v<16>(input + i);
    auto divisors = aie::load_v<16>(awq + i);
    auto signs = aie::load_v<16>(signs1 + i);
    auto divided = aie::div(values, divisors).template to_vector<float>();
    aie::store_v(
        scratch + i,
        aie::mul(divided, signs).template to_vector<float>());
  }

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
    auto values = aie::load_v<16>(scratch + i);
    auto signs = aie::load_v<16>(signs2 + i);
    auto scaled = aie::mul(
                      aie::mul(values, signs).template to_vector<float>(),
                      aie::broadcast<float, 16>(0.0625f))
                      .template to_vector<float>();
    aie::store_v(scratch + i, scaled);
    const float local_max = aie::reduce_max(aie::abs(scaled));
    if (local_max > max_abs) max_abs = local_max;
  }
  const float scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
  reinterpret_cast<float *>(activation_payload + A_DATA)[local_row] = scale;

  const int lm = local_row / 4;
  const int row = local_row % 4;
  const auto old_rounding =
      aie::swap_rounding(aie::rounding_mode::symmetric_inf);
  const auto old_saturation =
      aie::swap_saturation(aie::saturation_mode::symmetric);
  for (int kt = 0; kt < 16; kt++) {
    const int target = (lm * 16 + kt) * 64 + row * 16;
    if (scale > 0.0f) {
      auto values = aie::load_v<16>(scratch + kt * 16);
      auto normalized = aie::div(values, scale).template to_vector<float>();
      aie::store_v(activation_payload + target,
                   aie::to_fixed<int8>(normalized));
    } else {
      aie::store_v(activation_payload + target, aie::zeros<int8, 16>());
    }
  }
  aie::set_saturation(old_saturation);
  aie::set_rounding(old_rounding);
}
