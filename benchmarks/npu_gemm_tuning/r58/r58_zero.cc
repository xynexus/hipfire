#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void r58_zero_guard(int32_t *__restrict lane_sums) {
  aie::store_v(lane_sums, aie::zeros<int32, 64>());
}
