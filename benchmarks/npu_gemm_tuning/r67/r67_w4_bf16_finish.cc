// SPDX-License-Identifier: Apache-2.0
// Extract one 8-token x 32-column BF16 tile from an R15 24x96 accumulator.

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int LN = 6;
constexpr int MR = 4;
constexpr int VEC = 16;
constexpr int OUTPUT_COLS = 32;
} // namespace

extern "C" void r67_w4_finish_bf16_group(
    const int32_t *__restrict accumulator_bits,
    int8_t *__restrict output_bytes, int32_t slice, int32_t token_group) {
  const float *accumulator =
      reinterpret_cast<const float *>(accumulator_bits);
  bfloat16 *output = reinterpret_cast<bfloat16 *>(output_bytes);
  const int first_im = token_group * 2;
  const int first_jn = slice * 2;

  for (int local_im = 0; local_im < 2; ++local_im)
    for (int row = 0; row < MR; ++row)
      for (int local_jn = 0; local_jn < 2; ++local_jn) {
        const int im = first_im + local_im;
        const int jn = first_jn + local_jn;
        const int source = (im * LN + jn) * MR * VEC + row * VEC;
        const int target = (local_im * MR + row) * OUTPUT_COLS + local_jn * VEC;
        const auto value =
            aie::mul(aie::load_v<VEC>(accumulator + source), 1.0f)
                .to_vector<bfloat16>();
        aie::store_v(output + target, value);
      }
}
