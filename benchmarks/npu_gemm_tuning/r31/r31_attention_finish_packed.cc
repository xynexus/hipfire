// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void r31_attention_finish_packed(
    const float *restrict accum, const float *restrict stats,
    int8_t *restrict output_bytes) {
  constexpr int queries = 4;
  constexpr int dimension = 256;
  bfloat16 *output = reinterpret_cast<bfloat16 *>(output_bytes);
  for (int query = 0; query < queries; ++query) {
    const float inv_sum = aie::inv(stats[queries + query]);
    for (int dim = 0; dim < dimension; dim += 16) {
      const auto value =
          aie::mul(aie::load_v<16>(accum + query * dimension + dim), inv_sum)
              .to_vector<bfloat16>();
      aie::store_v(output + (dim / 8 * queries + query) * 8,
                   value.extract<8>(0));
      aie::store_v(output + ((dim / 8 + 1) * queries + query) * 8,
                   value.extract<8>(1));
    }
  }
}
