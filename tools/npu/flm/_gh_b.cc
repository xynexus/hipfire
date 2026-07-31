// SPDX-License-Identifier: Apache-2.0
#include <aie_api/aie.hpp>
#include <stdint.h>
extern bfloat16 g_hand[];
extern "C" __attribute__((noinline)) void
gh_read(const bfloat16 *restrict unused, bfloat16 *restrict out) {
  (void)unused;
  for (int i = 0; i < 64; i += 32)
    aie::store_v(out + i, aie::load_v<32>(g_hand + i));
}
