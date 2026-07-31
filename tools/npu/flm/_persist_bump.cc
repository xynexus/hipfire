// SPDX-License-Identifier: Apache-2.0
// Increments a file-scope counter and reports it. The counter is ordinary .bss
// with external linkage, which is exactly what the k' carry would be.
#include <aie_api/aie.hpp>
#include <stdint.h>
#ifndef DIM_N
#define DIM_N 32
#endif
#ifndef PERSIST_TAG
#define PERSIST_TAG 0
#endif

alignas(64) float g_persist_count[DIM_N];

extern "C" __attribute__((noinline)) void
persist_bump(bfloat16 *restrict out) {
#if PERSIST_TAG == 1
  // CONTROL: no state at all. If repeated dispatches do not all read 7, the
  // probe's own output path is broken and it cannot answer anything about .bss.
  for (int i = 0; i < DIM_N; ++i)
    out[i] = bfloat16(7.0f);
#else
  for (int i = 0; i < DIM_N; ++i) {
    g_persist_count[i] += 1.0f;
    out[i] = bfloat16(g_persist_count[i]);
  }
#endif
}
