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
#ifndef DIM_QSTRIDE
#define DIM_QSTRIDE DIM_HEAD
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
constexpr int QSTRIDE = DIM_QSTRIDE;
} // namespace

extern float g_m[];
extern float g_l[];
extern float g_acc[];

// `npad` is how many padded positions the last KV tile carried. The cache is
// streamed in whole TSEQ tiles, so a sequence length that is not a multiple of
// TSEQ leaves the tail padded with K=0, V=0. Those add nothing to `g_acc`
// (V=0), but each contributes `exp2(0 - m) = exp2(-m)` to the softmax
// denominator, because a zero K gives a zero score, not -inf. Subtracting them
// is the whole correction.
//
// It arrives as **f32, not bf16**: bf16 represents integers exactly only up to
// 256, and the same slot carries counts up to 2047 in the fused layer, so the
// wider type is the one that is always right.
//
// It rides in the tail of the Q buffer rather than in a fifo of its own,
// because **a core tile has 2 input DMA channels and attention already uses
// both** (Q and the KV stream). Giving npad a third is not a tight fit, it is
// a compile error: `tile (0,3) requires 3 input/1 output DMA channels, but only
// 2 input/2 output available`. Same reason the norm weight rides inside the
// activation in flm_norm_prepare and cs_q/cs_k ride inside the broadcast in
// flm_gemv_qkv — on this device, extra operands are packed, never given a fifo.
//
// So `q` is [GQA*HEAD bf16 q'][1 f32 npad], and the f32 is read through a cast.
// The offset is GQA*HEAD bf16 = 512 B, so it is 4-byte aligned by construction.
extern "C" __attribute__((noinline)) void
flm_attn_finish(bfloat16 *restrict out,        // [GQA][HEAD]
                const bfloat16 *restrict q) {
  const float npad_f =
      *reinterpret_cast<const float *>(q + GQA * QSTRIDE);
  for (int h = 0; h < GQA; ++h) {
    // No scalar libm on the core (`undefined symbol: exp2f`), so this goes
    // through the vector exp2 and extracts one lane — the same idiom the
    // rescale factor in flm_attn_decode.cc uses.
    g_l[h] -= npad_f * float(aie::exp2<bfloat16>(
        aie::broadcast<float, HALF>(-g_m[h]))[0]);
    const float inv = 1.0f / g_l[h];
    const auto iv = aie::broadcast<float, HALF>(inv);
    for (int off = 0; off < HEAD; off += HALF)
      aie::store_v(out + h * HEAD + off,
                   aie::mul(aie::load_v<HALF>(g_acc + h * HEAD + off), iv)
                       .template to_vector<bfloat16>());
  }
}
