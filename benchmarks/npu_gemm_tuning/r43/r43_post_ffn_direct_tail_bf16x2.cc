// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int ROWS = 2;
constexpr int HIDDEN = 768;
constexpr int POST_NORM_BYTES = HIDDEN * sizeof(bfloat16);
constexpr int EPSILON_OFFSET = POST_NORM_BYTES;
constexpr float INV_HIDDEN = 1.0f / HIDDEN;
} // namespace

extern "C" {

__attribute__((noinline, minsize)) void
r43_copy_params(const int8_t *restrict input, int8_t *restrict output) {
  for (int offset = 0; offset < POST_NORM_BYTES; offset += 32)
    aie::store_v(output + offset, aie::load_v<32>(input + offset));
  *reinterpret_cast<float *>(output + EPSILON_OFFSET) =
      *reinterpret_cast<const float *>(input + EPSILON_OFFSET);
}

void r43_post_ffn_direct_tail_bf16x2(int8_t *restrict output_bytes,
                                     const int8_t *restrict ffn_bytes,
                                     const int8_t *restrict params_bytes) {
  auto *output = reinterpret_cast<bfloat16 *>(output_bytes);
  const auto *combined = reinterpret_cast<const bfloat16 *>(ffn_bytes);
  const auto *post_norm = reinterpret_cast<const bfloat16 *>(params_bytes);
  const float epsilon =
      *reinterpret_cast<const float *>(params_bytes + EPSILON_OFFSET);

  for (int row = 0; row < ROWS; ++row) {
    const bfloat16 *ffn_row = combined + row * 3 * HIDDEN;
    const bfloat16 *ffn_low = ffn_row + HIDDEN;
    const bfloat16 *residual_row = ffn_low + HIDDEN;
    float sum = 0.0f;
    for (int hidden = 0; hidden < HIDDEN; hidden += 16) {
      const auto high =
          aie::mul(aie::load_v<16>(ffn_row + hidden),
                   aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
              .to_vector<float>();
      const auto low =
          aie::mul(aie::load_v<16>(ffn_low + hidden),
                   aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
              .to_vector<float>();
      const auto value = aie::add(high, low);
      sum += aie::reduce_add(aie::mul(value, value).to_vector<float>());
    }
    const float post_inverse = aie::invsqrt(sum * INV_HIDDEN + epsilon);
    bfloat16 *output_high = output + row * 2 * HIDDEN;
    bfloat16 *output_low = output_high + HIDDEN;
    for (int hidden = 0; hidden < HIDDEN; hidden += 16) {
      const auto residual =
          aie::mul(aie::load_v<16>(residual_row + hidden),
                   aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
              .to_vector<float>();
      const auto norm =
          aie::mul(aie::load_v<16>(post_norm + hidden),
                   aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
              .to_vector<float>();
      const auto ffn_high =
          aie::mul(aie::load_v<16>(ffn_row + hidden),
                   aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
              .to_vector<float>();
      const auto ffn_low_value =
          aie::mul(aie::load_v<16>(ffn_low + hidden),
                   aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
              .to_vector<float>();
      const auto normalized =
          aie::mul(aie::add(ffn_high, ffn_low_value), norm).to_vector<float>();
      const auto result =
          aie::add(residual,
                   aie::mul(normalized, post_inverse).to_vector<float>());
      const auto high = aie::mul(result, 1.0f).to_vector<bfloat16>();
      const auto high_f32 =
          aie::mul(high, aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
              .to_vector<float>();
      const auto low = aie::sub(result, high_f32);
      aie::store_v(output_high + hidden, high);
      aie::store_v(output_low + hidden,
                   aie::mul(low, 1.0f).to_vector<bfloat16>());
    }
  }
}

} // extern "C"
