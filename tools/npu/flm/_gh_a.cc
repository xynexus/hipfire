// SPDX-License-Identifier: Apache-2.0
#include <aie_api/aie.hpp>
#include <stdint.h>
alignas(64) bfloat16 g_hand[64];
extern "C" __attribute__((noinline)) void
gh_write(const bfloat16 *restrict in) {
  for (int i = 0; i < 64; i += 32)
    aie::store_v(g_hand + i, aie::load_v<32>(in + i));
}
