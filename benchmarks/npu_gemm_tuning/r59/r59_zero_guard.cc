#include <stdint.h>

extern "C" void r59_zero_guard(int32_t *__restrict guard) {
  for (int index = 0; index < 8; ++index)
    guard[index] = 0;
}
