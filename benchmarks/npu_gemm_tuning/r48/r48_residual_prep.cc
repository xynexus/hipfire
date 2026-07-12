// SPDX-License-Identifier: Apache-2.0

// Convert R46's token-major compensated BF16x2 state into the padded records
// consumed by R48. Each value reconstructs high+low in float and rounds once
// to BF16 nearest-even, matching the former host bridge. Each AIE core owns
// eight rows and writes one 16 KiB record.

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int HIDDEN = 768;
constexpr int HIGH_BYTES = HIDDEN * sizeof(bfloat16);
constexpr int ROWS = 8;
constexpr int PAYLOAD_BYTES = ROWS * HIGH_BYTES;
constexpr int RECORD_BYTES = 16384;
} // namespace

extern "C" {

void r48_copy_residual_row(const int8_t *restrict input,
                           int8_t *restrict output, int32_t row) {
  const auto *high = reinterpret_cast<const bfloat16 *>(input);
  const auto *low = high + HIDDEN;
  auto *target = reinterpret_cast<bfloat16 *>(output) + row * HIDDEN;
  const auto one = aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  const auto old_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int hidden = 0; hidden < HIDDEN; hidden += 16) {
    const auto high_f =
        aie::mul(aie::load_v<16>(high + hidden), one).to_vector<float>();
    const auto low_f =
        aie::mul(aie::load_v<16>(low + hidden), one).to_vector<float>();
    const auto reconstructed = aie::add(high_f, low_f);
    aie::store_v(target + hidden,
                 aie::mul(reconstructed, 1.0f).to_vector<bfloat16>());
  }
  aie::set_rounding(old_rounding);
  if (row == 0)
    for (int offset = PAYLOAD_BYTES; offset < RECORD_BYTES; offset += 64)
      aie::store_v(output + offset, aie::zeros<int8_t, 64>());
}

} // extern "C"
