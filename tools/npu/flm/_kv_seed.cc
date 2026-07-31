// SPDX-License-Identifier: Apache-2.0
// Writes a head into g_stage, standing in for flm_gemv_qkv's rotate-and-stage
// epilogue. The append scheme is what is under test, not the projection.
#include <aie_api/aie.hpp>
#include <stdint.h>
#ifndef DIM_HEAD
#define DIM_HEAD 64
#endif
// DEFINES g_stage here. In the layer it is flm_gemv_qkv.cc's staging buffer and
// this kernel does not exist; isolating the append means something else has to
// own it.
alignas(64) bfloat16 g_stage[DIM_HEAD];
extern "C" __attribute__((noinline)) void
kv_seed(const bfloat16 *restrict in) {
  for (int i = 0; i < DIM_HEAD; i += 32)
    aie::store_v(g_stage + i, aie::load_v<32>(in + i));
}
