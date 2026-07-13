#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void r60_weight_sink(const int8_t *__restrict tile) {
  volatile int8_t value = aie::reduce_add(aie::load_v<64>(tile));
  (void)value;
}
