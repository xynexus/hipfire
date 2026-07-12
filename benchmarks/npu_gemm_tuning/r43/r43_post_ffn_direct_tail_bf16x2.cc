// SPDX-License-Identifier: Apache-2.0

#include "aie_kernels/aie_kernel_utils.h"
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

#ifdef R46_SPLIT_RESIDUAL
__attribute__((noinline, minsize)) void
r46_x_source(const int8_t *restrict input, int8_t *restrict local) {
  constexpr int LOCAL_BYTES = ROWS * HIDDEN * sizeof(bfloat16);
  for (int offset = 0; offset < LOCAL_BYTES; offset += 64)
    aie::store_v(local + offset, aie::load_v<64>(input + offset));
  const int *words = reinterpret_cast<const int *>(input + LOCAL_BYTES);
  for (int word = 0; word < 3 * LOCAL_BYTES / (int)sizeof(int); ++word)
    put_ms(words[word]);
}

__attribute__((noinline, minsize)) void
r46_x_relay(int8_t *restrict local, int32_t forward_chunks) {
  constexpr int LOCAL_WORDS = ROWS * HIDDEN * sizeof(bfloat16) / sizeof(int);
  int *local_words = reinterpret_cast<int *>(local);
  for (int word = 0; word < LOCAL_WORDS; ++word)
    local_words[word] = get_ss_int();
  for (int chunk = 0; chunk < forward_chunks; ++chunk)
    for (int word = 0; word < LOCAL_WORDS; ++word)
      put_ms(get_ss_int());
}
#endif

__attribute__((noinline, minsize)) void
r43_copy_params(const int8_t *restrict input, int8_t *restrict output) {
  for (int offset = 0; offset < POST_NORM_BYTES; offset += 32)
    aie::store_v(output + offset, aie::load_v<32>(input + offset));
  *reinterpret_cast<float *>(output + EPSILON_OFFSET) =
      *reinterpret_cast<const float *>(input + EPSILON_OFFSET);
}

void r43_post_ffn_direct_tail_bf16x2(int8_t *restrict output_bytes,
                                     const int8_t *restrict ffn_bytes,
#ifdef R46_SPLIT_RESIDUAL
                                     const int8_t *restrict residual_bytes,
#endif
                                     const int8_t *restrict params_bytes) {
  auto *output = reinterpret_cast<bfloat16 *>(output_bytes);
  const auto *combined = reinterpret_cast<const bfloat16 *>(ffn_bytes);
#ifdef R46_SPLIT_RESIDUAL
  const auto *residual = reinterpret_cast<const bfloat16 *>(residual_bytes);
#endif
  const auto *post_norm = reinterpret_cast<const bfloat16 *>(params_bytes);
  const float epsilon =
      *reinterpret_cast<const float *>(params_bytes + EPSILON_OFFSET);

  for (int row = 0; row < ROWS; ++row) {
#ifdef R46_SPLIT_RESIDUAL
    const bfloat16 *ffn_row = combined + row * 2 * HIDDEN;
#else
    const bfloat16 *ffn_row = combined + row * 3 * HIDDEN;
#endif
    const bfloat16 *ffn_low = ffn_row + HIDDEN;
#ifdef R46_SPLIT_RESIDUAL
    const bfloat16 *residual_row = residual + row * HIDDEN;
#else
    const bfloat16 *residual_row = ffn_low + HIDDEN;
#endif
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
