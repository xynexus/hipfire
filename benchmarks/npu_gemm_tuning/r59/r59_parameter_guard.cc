#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void r59_parameter_guard(const int8_t *__restrict parameters,
                                     int32_t *__restrict guard) {
  int32_t total = 0;
  for (int offset = 0; offset < 4096; offset += 64)
    total += aie::reduce_add(aie::load_v<64>(parameters + offset));
  guard[4] = total;
}
