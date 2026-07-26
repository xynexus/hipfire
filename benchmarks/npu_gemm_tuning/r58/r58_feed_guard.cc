#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void r58_feed_guard(const int8_t *__restrict tile,
                                int32_t *__restrict guard) {
  guard[0] = aie::reduce_add(aie::load_v<64>(tile));
}
