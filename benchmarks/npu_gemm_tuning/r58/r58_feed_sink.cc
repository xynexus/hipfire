#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
volatile int32_t sink_guard;
}

extern "C" void r58_feed_sink(const int8_t *__restrict tile) {
  sink_guard = aie::reduce_add(aie::load_v<64>(tile));
}
