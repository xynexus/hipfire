// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int ROWS = 8;
constexpr int HIDDEN = 768;
constexpr int BLOCK = 256;
constexpr int PARAM_POST_NORM = ROWS * HIDDEN;
constexpr int PARAM_PRE_NORM = PARAM_POST_NORM + HIDDEN;
constexpr int PARAM_EPSILON_BYTES = (PARAM_PRE_NORM + HIDDEN) * 2;
constexpr float INV_HIDDEN = 1.0f / (float)HIDDEN;

} // namespace

extern "C" __attribute__((noinline, minsize, cold)) void
r90_post_residual_pre_ffn_split(int8_t *restrict prefix,
                                int8_t *restrict tail,
                                const int8_t *restrict params_bytes) {
  const auto *params = reinterpret_cast<const bfloat16 *>(params_bytes);
  const auto *residual = params;
  const auto *post_norm = params + PARAM_POST_NORM;
  const auto *pre_norm = params + PARAM_PRE_NORM;
  const float epsilon =
      *reinterpret_cast<const float *>(params_bytes + PARAM_EPSILON_BYTES);
  bfloat16 *blocks[] = {
      reinterpret_cast<bfloat16 *>(prefix),
      reinterpret_cast<bfloat16 *>(prefix + 4096),
      reinterpret_cast<bfloat16 *>(tail),
  };

  for (int row = 0; row < ROWS; ++row) {
    float output_sum = 0.0f;
    for (int block = 0; block < 3; ++block) {
      const bfloat16 *values = blocks[block] + row * BLOCK;
      for (int dim = 0; dim < BLOCK; dim += 16) {
        const auto value = aie::load_v<16>(values + dim);
        output_sum += aie::reduce_add(aie::mul(value, value).to_vector<float>());
      }
    }
    const float post_inverse =
        aie::invsqrt(output_sum * INV_HIDDEN + epsilon);

    float residual_sum = 0.0f;
    for (int block = 0; block < 3; ++block) {
      bfloat16 *values = blocks[block] + row * BLOCK;
      const int hidden_base = block * BLOCK;
      for (int dim = 0; dim < BLOCK; dim += 16) {
        const int hidden = hidden_base + dim;
        auto normalized =
            aie::mul(aie::load_v<16>(values + dim),
                     aie::load_v<16>(post_norm + hidden))
                .to_vector<float>();
        normalized = aie::mul(normalized, post_inverse).to_vector<float>();
        const auto x = aie::add(
            normalized,
            aie::mul(aie::load_v<16>(residual + row * HIDDEN + hidden),
                     aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
                .to_vector<float>());
        residual_sum += aie::reduce_add(aie::mul(x, x).to_vector<float>());
        aie::store_v(values + dim,
                     aie::mul(x, 1.0f).to_vector<bfloat16>());
      }
    }
    const float pre_inverse =
        aie::invsqrt(residual_sum * INV_HIDDEN + epsilon);
    for (int block = 0; block < 3; ++block) {
      bfloat16 *values = blocks[block] + row * BLOCK;
      const int hidden_base = block * BLOCK;
      for (int dim = 0; dim < BLOCK; dim += 16) {
        const auto normalized =
            aie::mul(aie::load_v<16>(values + dim),
                     aie::load_v<16>(pre_norm + hidden_base + dim))
                .to_vector<float>();
        aie::store_v(values + dim,
                     aie::mul(normalized, pre_inverse).to_vector<bfloat16>());
      }
    }
  }
}
