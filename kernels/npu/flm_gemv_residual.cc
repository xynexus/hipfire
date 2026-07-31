// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// q4_1 decode GEMV with the residual add fused into the epilogue:
//
//     out[r] = W_tile[r] . act + residual[r]
//
// WHY. A decoder layer has two residual adds (after o_proj and after
// down_proj). Each moves 4 KB and, at the measured **92.9 us per dispatch**, is
// ~99.9% fixed cost — 2.97 ms/token over 16 layers for 128 KB of data. Like the
// norms, they must ride along with a large operator rather than take a dispatch.
//
// HOW the residual reaches the core. A core tile has only **2 input DMA
// channels**, and the activation and the weight stream already use both — a
// third fifo is rejected by the placer ("reduce the LTO's DMA fanin"). So the
// residual slice for this tile rides in the **weight tile itself**, appended
// after the codes:
//
//     [NROWS*NB bf16 d][NROWS*NB bf16 m][NROWS*K/2 codes][64 B residual region]
//
// That is 64 B on a tile of TILE_BYTES — 0.3% at NROWS=16, K=2048 — to remove a
// whole dispatch. It also solves the addressing problem for free: each core gets
// exactly the residual rows it is computing, with no notion of its own row
// offset.
//
// **The region is a fixed 64 bytes, not NROWS*2, and that is load-bearing.**
// With NROWS*2 the tile is 20512 B, so the double-buffered ObjectFifo places
// buffer 1 at a 32-byte-aligned address and the vectorised residual load off it
// reads garbage. The symptom is exact: **even tiles correct to 1e-7, odd tiles
// wrong by ~1.0** — alternating, because the fifo alternates buffers. Padding
// the tile to a 64-byte multiple aligns both buffers.
//
// Compile-time: -DDIM_K, -DDIM_NROWS, as for the plain GEMV.

#include "flm_q4_1_tile.h"

extern "C" __attribute__((noinline)) void
flm_gemv_q4_1_residual(const bfloat16 *restrict act,
                       const uint8 *restrict wtile, float *restrict out) {
  flm_q4_1_tile(act, wtile, out);
  static_assert(NROWS * 2 <= 64, "residual region holds NROWS bf16");
  static_assert(TILE_BYTES % 64 == 0, "tile must already be 64-byte aligned");
  const auto *res = reinterpret_cast<const bfloat16 *>(wtile + TILE_BYTES);
  for (int r = 0; r < NROWS; ++r)
    out[r] += float(res[r]);
}
