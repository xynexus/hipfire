// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// Emit one k' head into the channel-major KV cache — the P1→P2 seam.
//
// Attention stores K as [HEAD][TSEQ] so scores accumulate across the head
// dimension with no horizontal reduce (`flm_attn_decode.cc`), which makes
// appending one token a stride-TSEQ scatter. **A DMA cannot do that one element
// at a time**: transfer sizes must be multiples of 4 bytes and offsets must be
// 4-byte aligned, so a lone 2-byte bf16 per destination is an illegal size and
// an odd column is an unreachable offset (measured in
// `tools/npu/flm/kv_append_probe.py`).
//
// The narrowest legal write therefore covers TWO columns starting at an even
// one, and this kernel supplies the two values for it:
//
//     even t:  (k'_t, 0)          written at column pair (t, t+1)
//     odd  t:  (k'_{t-1}, k'_t)   written at column pair (t-1, t)
//
// so every token lands in its own column and each pair is written twice — once
// opening it, once closing it. The zero at even t is not filler: attention
// requires padded positions to hold K=0 so their softmax contribution is
// exactly the `exp2(-m)` that `flm_attn_finish` subtracts.
//
// Closing a pair needs the PREVIOUS token's k', which is one dispatch earlier.
// `g_kprev` holds it, and that works because **core .bss survives between
// dispatches** while the xclbin stays loaded — measured in
// `tools/npu/flm/static_persist_probe.py` (a counter reads 1,2,3,... across
// separate dispatches). It is the whole reason K stays bf16: the alternative is
// an f32 cache, which doubles KV traffic and costs ~6 tok/s.
//
// Output is 2*HEAD interleaved, and the drain consumes it as TWO fifo objects
// (channels 0..HEAD/2-1 from the first, the rest from the second) — see
// `tools/npu/flm/qkv_route_probe.py`, where getting that accounting wrong made
// correct hardware look broken.
//
// Compile-time: -DDIM_HEAD.

#include "flm_q4_1_tile.h"
#include "flm_kv_pair.h"

extern bfloat16 g_stage[];      // the rotated head, from flm_gemv_qkv

// Thin entry point. The write itself lives in flm_kv_pair.h because
// flm_qkv_emit's k branch is the other real call site.
extern "C" __attribute__((noinline)) void
flm_kv_emit(const uint8 *restrict wtile, bfloat16 *restrict out) {
  flm_kv_pair(g_stage, tile_flags(wtile), out);
}
