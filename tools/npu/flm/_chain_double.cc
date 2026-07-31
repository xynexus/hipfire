// SPDX-License-Identifier: Apache-2.0
// Doubles its input. The probe needs an operation whose repeated application
// is visible in the result, so that a broken chain is distinguishable from a
// chain that ran the wrong number of times.
#include <aie_api/aie.hpp>
#include <stdint.h>
#ifndef DIM_N
#define DIM_N 256
#endif
extern "C" __attribute__((noinline)) void
chain_double(const bfloat16 *restrict in, bfloat16 *restrict out) {
  for (int i = 0; i < DIM_N; i += 32)
    aie::store_v(out + i, aie::mul(aie::load_v<32>(in + i),
                                   bfloat16(2.0f)).to_vector<bfloat16>());
}
