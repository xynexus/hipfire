// SPDX-License-Identifier: Apache-2.0

// EmbeddingGemma sentence-transformer Dense(768->3072->768) plus L2 norm.
// The two weight matrices stream one output row at a time while the pooled
// input and 3072-element intermediate remain in one AIE core's local memory.

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int INPUT = 768;
constexpr int INTERMEDIATE = 3072;
constexpr int OUTPUT = 768;
} // namespace

extern "C" {

void r51_copy_input(const float *restrict source, float *restrict input,
                    int32_t half) {
  const int base = half * (INPUT / 2);
  for (int column = 0; column < INPUT / 2; column += 16)
    aie::store_v(input + base + column, aie::load_v<16>(source + column));
}

void r51_dense0_row(const float *restrict input,
                    const bfloat16 *restrict weights,
                    float *restrict intermediate, int32_t row) {
  const auto one = aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  float sum = 0.0f;
  for (int column = 0; column < INPUT; column += 16) {
    const auto weight =
        aie::mul(aie::load_v<16>(weights + column), one).to_vector<float>();
    sum += aie::reduce_add(
        aie::mul(weight, aie::load_v<16>(input + column)).to_vector<float>());
  }
  intermediate[row] = sum;
}

void r51_dense1_row(const float *restrict intermediate,
                    const bfloat16 *restrict weights,
                    float *restrict output, int32_t row) {
  const auto one = aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  float sum = 0.0f;
  for (int column = 0; column < INTERMEDIATE; column += 16) {
    const auto weight =
        aie::mul(aie::load_v<16>(weights + column), one).to_vector<float>();
    sum += aie::reduce_add(
        aie::mul(weight, aie::load_v<16>(intermediate + column))
            .to_vector<float>());
  }
  output[row] = sum;
}

void r51_l2_normalize(float *restrict output) {
  float sum = 0.0f;
  for (int column = 0; column < OUTPUT; column += 16) {
    const auto value = aie::load_v<16>(output + column);
    sum += aie::reduce_add(aie::mul(value, value).to_vector<float>());
  }
  const float inverse = sum > 0.0f ? aie::invsqrt(sum) : 1.0f;
  for (int column = 0; column < OUTPUT; column += 16) {
    aie::store_v(output + column,
                 aie::mul(aie::load_v<16>(output + column), inverse)
                     .to_vector<float>());
  }
}

} // extern "C"
