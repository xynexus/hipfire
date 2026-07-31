// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// The k' column-pair write, shared by the standalone `flm_kv_emit` entry point
// and by `flm_qkv_emit`'s k branch. One definition, because both are real call
// sites and a second copy would be a second thing to keep correct.
//
// Appending one token to the channel-major K cache is a stride-TSEQ scatter,
// and a DMA cannot do it one element at a time: transfer sizes must be
// multiples of 4 bytes and offsets 4-byte aligned, so a lone 2-byte bf16 per
// destination is an illegal size and an odd column an unreachable offset
// (`tools/npu/flm/kv_append_probe.py`). The narrowest legal write covers two
// columns from an even one, so:
//
//     even t:  (k'_t, 0)          -> column pair (t, t+1)
//     odd  t:  (k'_{t-1}, k'_t)   -> column pair (t-1, t)
//
// Each token lands in its own column, each pair is written twice, and the zero
// at even t is what attention's `npad` correction wants from a padded position.
// Closing a pair needs the previous token, which `g_kprev` carries across the
// dispatch boundary — core `.bss` survives while the xclbin stays loaded
// (`tools/npu/flm/static_persist_probe.py`). That is what keeps K in bf16
// rather than an f32 cache costing ~6 tok/s.
//
// Verified end to end by `tools/npu/flm/kv_emit_verify.py`: five dispatches,
// both carries exact.

#pragma once
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_HEAD
#define DIM_HEAD 64
#endif

// This core's previous-token k', carried between dispatches.
inline bfloat16 g_kprev[DIM_HEAD] __attribute__((aligned(64)));

// `src` is this token's k' head; `out` receives 2*DIM_HEAD interleaved values.
inline __attribute__((noinline)) void
flm_kv_pair(const bfloat16 *restrict src, int t, bfloat16 *restrict out) {
  if (t & 1) {
    for (int d = 0; d < DIM_HEAD; ++d) {
      out[2 * d] = g_kprev[d];
      out[2 * d + 1] = src[d];
    }
  } else {
    for (int d = 0; d < DIM_HEAD; ++d) {
      out[2 * d] = src[d];
      out[2 * d + 1] = bfloat16(0.0f);
      g_kprev[d] = src[d];
    }
  }
}
