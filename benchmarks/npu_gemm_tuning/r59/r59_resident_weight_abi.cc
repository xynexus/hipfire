#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {
void r59_weight_guard(const int8_t *__restrict tile,
                      int32_t *__restrict guard, const int32_t role) {
  guard[role] = aie::reduce_add(aie::load_v<64>(tile));
}
} // extern "C"
