// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int ROWS = 8;
constexpr int HIDDEN = 768;
constexpr int POST_NORM_BYTES = HIDDEN * sizeof(bfloat16);
constexpr int EPSILON_OFFSET = POST_NORM_BYTES;
constexpr float INV_HIDDEN = 1.0f / HIDDEN;
} // namespace

extern "C" {

__attribute__((noinline, minsize)) void
r40_copy_residual(const int8_t *restrict residual_bytes,
                  int8_t *restrict output_bytes) {
  const auto *residual = reinterpret_cast<const bfloat16 *>(residual_bytes);
  auto *output = reinterpret_cast<bfloat16 *>(output_bytes);
  for (int offset = 0; offset < ROWS * HIDDEN; offset += 32)
    aie::store_v(output + offset, aie::load_v<32>(residual + offset));
}

__attribute__((noinline, minsize)) void
r40_copy_params(const int8_t *restrict input, int8_t *restrict output) {
  for (int offset = 0; offset < POST_NORM_BYTES; offset += 32)
    aie::store_v(output + offset, aie::load_v<32>(input + offset));
  *reinterpret_cast<float *>(output + EPSILON_OFFSET) =
      *reinterpret_cast<const float *>(input + EPSILON_OFFSET);
}

void r40_post_ffn_direct_tail(int8_t *restrict output_bytes,
                              const int8_t *restrict ffn_bytes,
                              const int8_t *restrict params_bytes) {
  auto *output = reinterpret_cast<bfloat16 *>(output_bytes);
  const auto *ffn = reinterpret_cast<const bfloat16 *>(ffn_bytes);
  const auto *post_norm =
      reinterpret_cast<const bfloat16 *>(params_bytes);
  const float epsilon =
      *reinterpret_cast<const float *>(params_bytes + EPSILON_OFFSET);

  for (int row = 0; row < ROWS; ++row) {
    const bfloat16 *ffn_row = ffn + row * HIDDEN;
    float sum = 0.0f;
    for (int hidden = 0; hidden < HIDDEN; hidden += 16) {
      const auto value = aie::load_v<16>(ffn_row + hidden);
      sum += aie::reduce_add(aie::mul(value, value).to_vector<float>());
    }
    const float post_inverse = aie::invsqrt(sum * INV_HIDDEN + epsilon);
    bfloat16 *output_row = output + row * HIDDEN;
    for (int hidden = 0; hidden < HIDDEN; hidden += 16) {
      const auto residual =
          aie::mul(aie::load_v<16>(output_row + hidden),
                   aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
              .to_vector<float>();
      const auto normalized =
          aie::mul(aie::load_v<16>(ffn_row + hidden),
                   aie::load_v<16>(post_norm + hidden))
              .to_vector<float>();
      const auto result_float =
          aie::add(residual,
                   aie::mul(normalized, post_inverse).to_vector<float>());
      const auto result =
          aie::mul(result_float, 1.0f).to_vector<bfloat16>();
      aie::store_v(output_row + hidden, result);
    }
  }
}

} // extern "C"
