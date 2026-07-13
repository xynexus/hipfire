// SPDX-License-Identifier: Apache-2.0
// Convert one completed R15 W4 f32 accumulator tile to a padded 24x32 BF16
// record. The three slice calls cover the tile's 24x96 logical result.

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int LM = 6;
constexpr int LN = 6;
constexpr int MR = 4;
constexpr int VEC = 16;
constexpr int ACC_ELEMS = LM * LN * MR * VEC;
constexpr int OUTPUT_ROWS = 32;
constexpr int OUTPUT_COLS = 32;
} // namespace

extern "C" void r65_w4_finish_bf16_slice(
    const int32_t *__restrict accumulator_bits,
    int8_t *__restrict output_bytes, int32_t slice) {
  const float *accumulator =
      reinterpret_cast<const float *>(accumulator_bits);
  bfloat16 *output = reinterpret_cast<bfloat16 *>(output_bytes);
  const int first_jn = slice * 2;

  for (int im = 0; im < LM; ++im)
    for (int row = 0; row < MR; ++row)
      for (int local_jn = 0; local_jn < 2; ++local_jn) {
        const int jn = first_jn + local_jn;
        const int source = (im * LN + jn) * MR * VEC + row * VEC;
        const int target = (im * MR + row) * OUTPUT_COLS + local_jn * VEC;
        const auto value =
            aie::mul(aie::load_v<VEC>(accumulator + source), 1.0f)
                .to_vector<bfloat16>();
        aie::store_v(output + target, value);
      }

  for (int index = LM * MR * OUTPUT_COLS;
       index < OUTPUT_ROWS * OUTPUT_COLS; index += VEC)
    aie::store_v(output + index, aie::zeros<bfloat16, VEC>());
}
