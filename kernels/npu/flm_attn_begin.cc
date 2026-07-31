// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// Decode attention over a KV cache for AIE2P — one GQA group per core.
//
// llama-3.2-1B decodes one token at a time against a growing cache: 32 query
// heads, 8 KV heads (GQA ratio 4), head_dim 64. One core owns one KV head and
// the GQA query heads that share it, so the KV it streams is private and the
// 8 groups map onto 8 cores with no broadcast.
//
// **Operand orientation is the whole design, and it follows the reverse
// engineering** (`docs/npu/flm-attn-dataflow.md`, and the plan's note that "K
// records are channel-major [head_dim x GROUP]"):
//
//   K tile  channel-major [HEAD_DIM][TSEQ] — score[t] accumulates ACROSS d, so
//           each `mac` adds a whole 32-position vector and there is **no
//           horizontal reduce anywhere in the score path**.
//   V tile  position-major [TSEQ][HEAD_DIM] — out[d] accumulates ACROSS t, so
//           each `mac` adds a whole head vector.
//
// The two want opposite layouts, which is exactly why FLM does the KV layout
// transform in the memtile DMA (free, while the data moves) rather than in a
// core.
//
// Softmax is online (flash-style) so the cache is streamed once. The scale
// 1/sqrt(head_dim) AND log2(e) are folded into Q on the host, so the
// exponential is the hardware `exp2` on an accumulator with no pre-multiply.
//
// Call order per token: begin() -> tile() per KV tile -> finish().
// Compile-time: -DDIM_GQA -DDIM_HEAD -DDIM_TSEQ.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_GQA
#define DIM_GQA 4
#endif
#ifndef DIM_HEAD
#define DIM_HEAD 64
#endif
#ifndef DIM_TSEQ
#define DIM_TSEQ 32
#endif
//
// NOTE: one entry point per translation unit. IRON compiles each
// ExternalFunction's source separately, so N entry points in one file are
// linked N times and fail with `duplicate symbol`. The softmax state lives
// in flm_attn_decode.cc and is reached by `extern` from here.

namespace {
constexpr int GQA = DIM_GQA;
constexpr int HEAD = DIM_HEAD;
constexpr int TSEQ = DIM_TSEQ;
constexpr int HALF = HEAD / 2;
} // namespace

extern float g_m[];
extern float g_l[];
extern float g_acc[];

// State init only — **RoPE is NOT applied here**, and this is a change.
//
// It used to be, and the comment that lived here argued it *had* to be: "at 16
// rows per weight tile a q4_1 tile produces a quarter of a head_dim-64 head,
// and RoPE needs whole heads — NROWS=64 would make the tile 81920 B, far past
// the 64 KB tile memory." The premise was right and the conclusion was wrong.
// The qkv projection does not need a 64-row tile to rotate a whole head; it
// needs somewhere to put four 16-row tiles, and 128 B of core memory is
// somewhere. `flm_gemv_qkv.cc` stages the head and rotates it when the fourth
// tile closes it, from the tile's own `row_base` trailer.
//
// That matters beyond tidiness: in the fused layer, k' must be rotated *before*
// it is appended to the KV cache, and the cache is written by phase P1. Leaving
// the rotation in attention would rotate q only, and every cached k would be
// wrong. So P1 owns RoPE for both, and by the time attention runs, q' and every
// cached k' are already rotated.
//
// `q` is the core's [GQA][HEAD] query block. It is untouched here — the
// parameter is kept because IRON binds each kernel to the fifo objects it is
// called with, and this is the one that names the phase's Q buffer.
extern "C" __attribute__((noinline)) void flm_attn_begin(bfloat16 *restrict q) {
  (void)q;
  for (int h = 0; h < GQA; ++h) {
    g_m[h] = -3.0e38f;      // -inf; the first tile always wins the max
    g_l[h] = 0.0f;
  }
  for (int i = 0; i < GQA * HEAD; i += HALF)
    aie::store_v(g_acc + i, aie::zeros<float, HALF>());
}
