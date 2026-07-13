// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include "aie_kernels/aie_kernel_utils.h"
#include <stdint.h>

namespace {
using MMUL = aie::mmul<8, 8, 8, int8, int8>;
constexpr int K_TILES = 32;
constexpr int A_SCALE_OFFSET = K_TILES * MMUL::size_A;
constexpr int W_SCALE_OFFSET = K_TILES * 2 * MMUL::size_B;

static inline aie::vector<int32, 16>
join_rows(aie::vector<int32, MMUL::size_C> lo,
          aie::vector<int32, MMUL::size_C> hi, int row) {
  switch (row) {
  case 0: return aie::concat(lo.extract<8>(0), hi.extract<8>(0));
  case 1: return aie::concat(lo.extract<8>(1), hi.extract<8>(1));
  case 2: return aie::concat(lo.extract<8>(2), hi.extract<8>(2));
  case 3: return aie::concat(lo.extract<8>(3), hi.extract<8>(3));
  case 4: return aie::concat(lo.extract<8>(4), hi.extract<8>(4));
  case 5: return aie::concat(lo.extract<8>(5), hi.extract<8>(5));
  case 6: return aie::concat(lo.extract<8>(6), hi.extract<8>(6));
  default: return aie::concat(lo.extract<8>(7), hi.extract<8>(7));
  }
}
} // namespace

extern "C" {

void r116_compact_group_n16(const int8_t *restrict activation_slot,
                            const int8_t *restrict weight_record,
                            float *restrict output, int32_t group) {
  alignas(32) float prior[8 * 16];
  if (group != 0)
    for (int row = 0; row < 8; ++row)
      aie::store_v(prior + row * 16, aie::load_v<16>(output + row * 16));

  MMUL lo, hi;
  auto a = aie::load_v<MMUL::size_A>(activation_slot);
  lo.mul(a, aie::load_v<MMUL::size_B>(weight_record));
  hi.mul(a, aie::load_v<MMUL::size_B>(weight_record + MMUL::size_B));
  for (int kt = 1; kt < K_TILES; ++kt) {
    a = aie::load_v<MMUL::size_A>(activation_slot + kt * MMUL::size_A);
    const int8_t *w = weight_record + kt * 2 * MMUL::size_B;
    lo.mac(a, aie::load_v<MMUL::size_B>(w));
    hi.mac(a, aie::load_v<MMUL::size_B>(w + MMUL::size_B));
  }

  const auto vlo = lo.to_vector<int32>();
  const auto vhi = hi.to_vector<int32>();
  const auto weight_scale =
      aie::load_v<16>(reinterpret_cast<const float *>(weight_record +
                                                       W_SCALE_OFFSET));
  const auto *activation_scale =
      reinterpret_cast<const float *>(activation_slot + A_SCALE_OFFSET);
  for (int row = 0; row < 8; ++row) {
    auto scaled =
        aie::mul(aie::to_float(join_rows(vlo, vhi, row)), weight_scale)
            .to_vector<float>();
    scaled = aie::mul(scaled, activation_scale[row]).to_vector<float>();
    if (group != 0)
      scaled = aie::add(scaled, aie::load_v<16>(prior + row * 16));
    aie::store_v(output + row * 16, scaled);
  }
  // The next group immediately reloads this same local output object. Keep the
  // inter-call store/load dependency visible to the scheduler.
  chess_separator_scheduler(1);
  chess_separator();
}

} // extern "C"
