#include <aie_api/aie.hpp>

#ifndef TILE_N
#define TILE_N 16384
#endif

namespace {
volatile int32_t sink_guard;
}

extern "C" void feed_sum(const int8 *__restrict tile, int32 *__restrict acc) {
  acc[0] = aie::reduce_add(aie::load_v<64>(tile));
}

// Broadcast consumers that are not traced keep the read live without creating
// another southbound ObjectFIFO. The volatile tile-local store is the DCE guard.
extern "C" void feed_sink(const int8 *__restrict tile) {
  sink_guard = aie::reduce_add(aie::load_v<64>(tile));
}
