// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
void finish_one(const float *restrict accum, const float *restrict stats,
                bfloat16 *restrict output) {
  constexpr int queries = 4;
  constexpr int dimension = 256;
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
} // namespace

extern "C" void r32_attention_finish_pair_packed(
    const float *restrict accum0, const float *restrict stats0,
    const float *restrict accum1, const float *restrict stats1,
    int8_t *restrict output_bytes) {
  auto *output = reinterpret_cast<bfloat16 *>(output_bytes);
  finish_one(accum0, stats0, output);
  finish_one(accum1, stats1, output + 1024);
}
