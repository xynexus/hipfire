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
#ifdef SEED_CONST
  // split test: ignore the input and write a constant. If the cache then shows
  // 5.0, the g_stage handoff works and the input fifo is at fault; if it stays
  // zero, the cross-TU g_stage link is.
  for (int i = 0; i < DIM_HEAD; i += 32)
    aie::store_v(g_stage + i, aie::broadcast<bfloat16, 32>(bfloat16(5.0f)));
#else
  for (int i = 0; i < DIM_HEAD; i += 32)
    aie::store_v(g_stage + i, aie::load_v<32>(in + i));
#endif
}
