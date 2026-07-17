// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// C1: the fixed dispatch floor.
//
// The kernel does as close to nothing as the compiler will allow: touch each
// input once so it is not dead, write one output word. Whatever time a dispatch
// costs with this kernel is cost that no real kernel can avoid.
//
// Sweeping NARGS separates the per-command cost from the per-BO cost:
//   t(n_bos) = c_cmd + c_bo * n_bos
//
// This matters more than it sounds: R64 measured a warm production wrapper as
// 76.6% preparation/submit/sync/deblock, and R117 doubled useful work while
// getting 9.8% *faster* — both say the floor, not the arithmetic, sets the
// scale for small consumers.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef NARGS
#define NARGS 1
#endif

extern "C" void c1_null(const int32_t *__restrict i0,
#if NARGS >= 2
                        const int32_t *__restrict i1,
#endif
#if NARGS >= 3
                        const int32_t *__restrict i2,
#endif
#if NARGS >= 4
                        const int32_t *__restrict i3,
#endif
                        int32_t *__restrict out) {
  // Touch every input exactly once: enough to keep the BO live, cheap enough
  // that the arithmetic is not what we are measuring.
  int32_t acc = i0[0];
#if NARGS >= 2
  acc += i1[0];
#endif
#if NARGS >= 3
  acc += i2[0];
#endif
#if NARGS >= 4
  acc += i3[0];
#endif
  out[0] = acc;
  out[1] = NARGS;
}
