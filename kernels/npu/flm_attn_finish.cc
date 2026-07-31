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

extern "C" __attribute__((noinline)) void
flm_attn_finish(bfloat16 *restrict out) {      // [GQA][HEAD]
  for (int h = 0; h < GQA; ++h) {
    const float inv = 1.0f / g_l[h];
    const auto iv = aie::broadcast<float, HALF>(inv);
    for (int off = 0; off < HEAD; off += HALF)
      aie::store_v(out + h * HEAD + off,
                   aie::mul(aie::load_v<HALF>(g_acc + h * HEAD + off), iv)
                       .template to_vector<bfloat16>());
  }
}
