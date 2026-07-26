// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int ROWS = 8;
constexpr int HIDDEN = 768;
constexpr int PRE_RECIP_BYTES = HIDDEN * sizeof(float);
constexpr int POST_NORM_BYTES = HIDDEN * sizeof(bfloat16);
constexpr int EPSILON_OFFSET = PRE_RECIP_BYTES + POST_NORM_BYTES;
constexpr float INV_HIDDEN = 1.0f / HIDDEN;
} // namespace

extern "C" {

__attribute__((noinline, minsize)) void
r39_copy_hidden(const int8_t *restrict hidden_bytes,
                int8_t *restrict output_bytes) {
  const auto *hidden = reinterpret_cast<const bfloat16 *>(hidden_bytes);
  auto *output = reinterpret_cast<bfloat16 *>(output_bytes);
  for (int offset = 0; offset < ROWS * HIDDEN; offset += 32) {
    aie::store_v(output + offset, aie::load_v<32>(hidden + offset));
  }
}

__attribute__((noinline, minsize)) void
r39_copy_inverse(const int8_t *restrict metadata_bytes,
                 float *restrict inverse) {
  const auto *metadata = reinterpret_cast<const float *>(metadata_bytes);
  for (int row = 0; row < ROWS; ++row)
    inverse[row] = metadata[row];
}

__attribute__((noinline, minsize)) void
r39_copy_params(const int8_t *restrict input, int8_t *restrict output) {
  for (int offset = 0; offset < EPSILON_OFFSET; offset += 32) {
    aie::store_v(output + offset, aie::load_v<32>(input + offset));
  }
  *reinterpret_cast<float *>(output + EPSILON_OFFSET) =
      *reinterpret_cast<const float *>(input + EPSILON_OFFSET);
}

void r39_post_ffn_tail(int8_t *restrict output_bytes,
                       const int8_t *restrict ffn_bytes,
                       const int8_t *restrict params_bytes,
                       const float *restrict pre_inverse) {
  auto *output = reinterpret_cast<bfloat16 *>(output_bytes);
  const auto *ffn = reinterpret_cast<const bfloat16 *>(ffn_bytes);
  const auto *pre_recip = reinterpret_cast<const float *>(params_bytes);
  const auto *post_norm = reinterpret_cast<const bfloat16 *>(
      params_bytes + PRE_RECIP_BYTES);
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
    const float residual_scale = aie::inv(pre_inverse[row]);
    bfloat16 *output_row = output + row * HIDDEN;
    for (int hidden = 0; hidden < HIDDEN; hidden += 16) {
      const auto hidden_value =
          aie::mul(aie::load_v<16>(output_row + hidden),
                   aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
              .to_vector<float>();
      const auto residual =
          aie::mul(hidden_value, aie::load_v<16>(pre_recip + hidden))
              .to_vector<float>();
      const auto normalized =
          aie::mul(aie::load_v<16>(ffn_row + hidden),
                   aie::load_v<16>(post_norm + hidden))
              .to_vector<float>();
      const auto result_float =
          aie::add(aie::mul(residual, residual_scale).to_vector<float>(),
                   aie::mul(normalized, post_inverse).to_vector<float>());
      const auto result =
          aie::mul(result_float, 1.0f).to_vector<bfloat16>();
      aie::store_v(output_row + hidden, result);
    }
  }
}

} // extern "C"
