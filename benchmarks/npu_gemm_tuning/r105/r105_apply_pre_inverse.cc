// SPDX-License-Identifier: Apache-2.0

// Canonical direct-X to unit-RMS BF16. X is read once from external memory,
// retained in tile memory, and traversed twice locally (sum, then normalize).
// Immutable pre-FFN norm is folded into the following W4 activation divisor.

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int ROWS = 8;
constexpr int HIDDEN = 768;
constexpr float INV_HIDDEN = 1.0f / (float)HIDDEN;
constexpr float EPSILON = 1.0e-6f;
}

extern "C" void r105_direct_x_unit_rms(const int8_t *restrict input_bytes,
                                        int8_t *restrict output_bytes) {
  const auto *input = reinterpret_cast<const bfloat16 *>(input_bytes);
  auto *output = reinterpret_cast<bfloat16 *>(output_bytes);

  for (int row = 0; row < ROWS; ++row) {
    float sum = 0.0f;
    for (int hidden = 0; hidden < HIDDEN; hidden += 16) {
      const auto values = aie::load_v<16>(input + row * HIDDEN + hidden);
      sum += aie::reduce_add(aie::mul(values, values).to_vector<float>());
    }
    const float inverse = aie::invsqrt(sum * INV_HIDDEN + EPSILON);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(48)
    for (int hidden = 0; hidden < HIDDEN; hidden += 16) {
      const auto values = aie::load_v<16>(input + row * HIDDEN + hidden);
      const auto expanded =
          aie::mul(values, aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
              .to_vector<float>();
      aie::store_v(output + row * HIDDEN + hidden,
                   aie::mul(expanded, inverse).to_vector<bfloat16>());
    }
  }
}
