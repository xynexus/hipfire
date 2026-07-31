// SPDX-License-Identifier: Apache-2.0
// Emits one head, tagged so the host can tell which one it was. The probe is
// about the routing, not the arithmetic.
#include <aie_api/aie.hpp>
#include <stdint.h>
#ifndef DIM_HEAD
#define DIM_HEAD 64
#endif
int g_head_ix;

extern "C" __attribute__((noinline)) void
route_emit(bfloat16 *restrict out) {
  // g_head_ix persists across calls (and across dispatches — see
  // static_persist_probe.py), so each head carries a distinct tag and a drain
  // that took the wrong count shifts the tags visibly.
  // Every element of head h is h+1 — a small integer, exact in bf16 (which is
  // only exact on integers to 256, so a 100*h+i ramp would add rounding noise
  // on top of any routing error and confuse the two).
  const bfloat16 tag = bfloat16(float(g_head_ix) + 1.0f);
  for (int i = 0; i < DIM_HEAD; i += 16)
    aie::store_v(out + i, aie::broadcast<bfloat16, 16>(tag));
  ++g_head_ix;
}
