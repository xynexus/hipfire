// SPDX-License-Identifier: Apache-2.0
// Emits one head. The probe is about the DMA pattern, not the arithmetic, so
// the body just forwards a head-sized object.
#include <aie_api/aie.hpp>
#include <stdint.h>
#ifndef DIM_HEAD
#define DIM_HEAD 64
#endif
extern "C" __attribute__((noinline)) void
kv_emit(const float *restrict in, float *restrict out) {
  for (int i = 0; i < DIM_HEAD; i += 16)
    aie::store_v(out + i, aie::load_v<16>(in + i));
}
