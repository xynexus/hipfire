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
//
// **One carry per LAYER.** A design that runs several layers on the same core
// between dispatches — the fused single dispatch does all sixteen — would
// otherwise have layer 0 of the next token close its column pair with layer 15's
// key. Even positions never READ the carry, which is why that stayed invisible
// through every even-position result. `FLM_KV_LAYERS` defaults to 1, so every
// one-layer design keeps a single 128-byte array and the layer index is the
// constant 0 that `pack_tile` writes.

#pragma once
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_HEAD
#define DIM_HEAD 64
#endif

// Layers this core runs between dispatches. A power of two, so the index can be
// masked into range for free rather than trusted.
#ifndef FLM_KV_LAYERS
#define FLM_KV_LAYERS 1
#endif
static_assert((FLM_KV_LAYERS & (FLM_KV_LAYERS - 1)) == 0,
              "FLM_KV_LAYERS must be a power of two");

// This core's previous-token k', per layer, carried between dispatches.
inline bfloat16 g_kprev[FLM_KV_LAYERS][DIM_HEAD] __attribute__((aligned(64)));

// `src` is this token's k' head; `out` receives 2*DIM_HEAD interleaved values.
// `lay` selects the carry, and is `tile_layer(wtile)` at both call sites.
inline __attribute__((noinline)) void
flm_kv_pair(const bfloat16 *restrict src, int t, int lay,
            bfloat16 *restrict out) {
  bfloat16 *restrict prev = g_kprev[lay & (FLM_KV_LAYERS - 1)];
  if (t & 1) {
    for (int d = 0; d < DIM_HEAD; ++d) {
      out[2 * d] = prev[d];
      out[2 * d + 1] = src[d];
    }
  } else {
    for (int d = 0; d < DIM_HEAD; ++d) {
      out[2 * d] = src[d];
      out[2 * d + 1] = bfloat16(0.0f);
      prev[d] = src[d];
    }
  }
}
