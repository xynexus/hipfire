// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// Emit one finished q/k/v head from the in-core stage to the result fifo.
// Companion to flm_gemv_qkv.cc; see that file for the phase design.
//
// This is a separate entry point rather than the tail of `flm_gemv_qkv`
// because the result fifo is acquired by the worker, not by the kernel: only
// one tile in four closes a head, so the acquire cannot sit inside a call that
// runs on every tile without stalling the other three.
//
// 128 B per head, which is the result object for phase P1.
//
// Compile-time: -DDIM_HEAD.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_HEAD
#define DIM_HEAD 64
#endif

namespace {
constexpr int HEAD = DIM_HEAD;
constexpr int VLANES = 32;
static_assert(HEAD % VLANES == 0, "a head must be a whole number of vectors");
} // namespace

extern bfloat16 g_stage[];

extern "C" __attribute__((noinline)) void
flm_qkv_emit(bfloat16 *restrict out) {
  for (int i = 0; i < HEAD; i += VLANES)
    aie::store_v(out + i, aie::load_v<VLANES>(g_stage + i));
}
