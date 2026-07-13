#include <aie_api/aie.hpp>
#include "aie_kernels/aie_kernel_utils.h"
#include <stdint.h>

#ifndef DATA_BYTES
#define DATA_BYTES 12288
#endif

namespace {
volatile int16_t sink_lane;
}

extern "C" void r58_decode_sink(const int8_t *__restrict tile) {
  aie::vector<int16, 64> local = aie::zeros<int16, 64>();
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (unsigned offset = 0; offset < DATA_BYTES; offset += 32) {
    const auto *packed = reinterpret_cast<const int4 *>(tile + offset);
    aie::vector<int4, 64> nibbles = aie::load_v<64>(packed);
    local = aie::add(local, nibbles.unpack_sign(true).unpack_sign(true));
  }
  sink_lane = local.get(0);
}
