// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// R58 NIBBLE_DECODE: consume the production WholeScaledV1 W4 tile without
// changing its global block order. Each 32-byte load contains 64 signed int4
// values, low nibble first. `unpack_sign(true)` lowers to AIE2P's single
// hardware int4-to-int8 unpack instruction.

#include <aie_api/aie.hpp>
#include "aie_kernels/aie_kernel_utils.h"
#include <stdint.h>

#ifndef TILE_BYTES
#define TILE_BYTES 16384
#endif
#ifndef DATA_BYTES
#define DATA_BYTES 12288
#endif

static_assert(DATA_BYTES <= TILE_BYTES);
static_assert(DATA_BYTES % 32 == 0);

namespace {
aie::vector<int16, 64> decode_lane_sums(const int8_t *__restrict tile) {
  aie::vector<int16, 64> local = aie::zeros<int16, 64>();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (unsigned offset = 0; offset < DATA_BYTES; offset += 32) {
    const auto *packed = reinterpret_cast<const int4 *>(tile + offset);
    aie::vector<int4, 64> nibbles = aie::load_v<64>(packed);
    aie::vector<int8, 64> decoded = nibbles.unpack_sign(true);
    local = aie::add(local, decoded.unpack_sign(true));
  }
  return local;
}
} // namespace

extern "C" {

// Accumulate one signed sum for every lane modulo 64. This forces every packed
// nibble through the hardware decoder while keeping output traffic negligible.
// A separate first-vector output below proves the exact low/high lane order.
void r58_decode_guard(const int8_t *__restrict tile,
                      int32_t *__restrict lane_sums) {
  aie::vector<int16, 64> local = decode_lane_sums(tile);
  aie::vector<int32, 64> prior = aie::load_v<64>(lane_sums);
  aie::store_v(lane_sums, aie::add(prior, local.unpack_sign(true)));
  const auto *packed = reinterpret_cast<const int4 *>(tile);
  aie::vector<int4, 64> first = aie::load_v<64>(packed);
  aie::store_v(reinterpret_cast<int8_t *>(lane_sums + 64),
               first.unpack_sign(true));
}

}
