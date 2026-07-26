// SPDX-License-Identifier: Apache-2.0

// Final EmbeddingGemma RMSNorm plus mean pooling. The single-core schedule is
// deliberate: it keeps the 768-wide reduction accumulator local while the DMA
// engine streams all 256 compensated BF16x2 rows through a two-slot FIFO.

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int ROWS = 256;
constexpr int HIDDEN = 768;
constexpr float INV_HIDDEN = 1.0f / (float)HIDDEN;
constexpr float INV_ROWS = 1.0f / (float)ROWS;
constexpr int EPSILON_OFFSET = HIDDEN * sizeof(float);
} // namespace

extern "C" {

void r49_final_norm_mean_row(const int8_t *restrict input_bytes,
                             const int8_t *restrict param_bytes,
                             float *restrict pooled, int32_t row) {
  const auto *high = reinterpret_cast<const bfloat16 *>(input_bytes);
  const auto *low = high + HIDDEN;
  const auto *norm = reinterpret_cast<const float *>(param_bytes);
  const float epsilon =
      *reinterpret_cast<const float *>(param_bytes + EPSILON_OFFSET);
  const auto one = aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  float sum = 0.0f;
  for (int hidden = 0; hidden < HIDDEN; hidden += 16) {
    const auto high_f =
        aie::mul(aie::load_v<16>(high + hidden), one).to_vector<float>();
    const auto low_f =
        aie::mul(aie::load_v<16>(low + hidden), one).to_vector<float>();
    const auto value = aie::add(high_f, low_f);
    sum += aie::reduce_add(aie::mul(value, value).to_vector<float>());
  }
  const float inverse = aie::invsqrt(sum * INV_HIDDEN + epsilon);
  for (int hidden = 0; hidden < HIDDEN; hidden += 16) {
    const auto high_f =
        aie::mul(aie::load_v<16>(high + hidden), one).to_vector<float>();
    const auto low_f =
        aie::mul(aie::load_v<16>(low + hidden), one).to_vector<float>();
    const auto value = aie::add(high_f, low_f);
    const auto normalized =
        aie::mul(value, aie::load_v<16>(norm + hidden)).to_vector<float>();
    auto accumulated = row == 0 ? aie::zeros<float, 16>()
                                : aie::load_v<16>(pooled + hidden);
    accumulated = aie::add(accumulated,
                           aie::mul(normalized, inverse).to_vector<float>());
    if (row == ROWS - 1)
      accumulated =
          aie::mul(accumulated, INV_ROWS).to_vector<float>();
    aie::store_v(pooled + hidden, accumulated);
  }
}

} // extern "C"
