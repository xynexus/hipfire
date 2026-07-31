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
// Elements between consecutive query heads. HEAD when the block is packed, which
// is what the standalone harnesses use. Phase P1 emits every head as a 2*HEAD
// object — only k' needs the doubled form, but the result-fifo object size is
// fixed — and a drain cannot skip source elements, so P1's q' lands strided.
// This is the one place q is indexed; everything else is core-local.
#ifndef DIM_QSTRIDE
#define DIM_QSTRIDE DIM_HEAD
#endif
// How many whole KV tiles one acquire delivers. 1 for the standalone harness,
// where the fifo object is exactly a tile; >1 in the fused layer, where every
// phase shares one 20544 B operand object and a lone 8192 B KV tile would waste
// 61% of it. Two tiles fill 16384 of 20544 — 20% waste on the KV stream, which
// at S=2048 is +2.8% of a layer's bytes against +16.6% for one tile. TSEQ
// itself cannot just be doubled: the score vector is one 32-lane register.
#ifndef DIM_KVPER
#define DIM_KVPER 1
#endif

namespace {
constexpr int GQA = DIM_GQA;
constexpr int HEAD = DIM_HEAD;
constexpr int TSEQ = DIM_TSEQ;
constexpr int HALF = HEAD / 2;      // head vectors run 32 lanes at a time
constexpr int KVPER = DIM_KVPER;
constexpr int KVSTRIDE = 2 * TSEQ * HEAD;   // bf16 elements in one [K][V] tile
constexpr int QSTRIDE = DIM_QSTRIDE;        // elements between query heads

static_assert(TSEQ == 32, "score vectors are one 32-lane register");
static_assert(HEAD % 32 == 0, "head_dim must be a multiple of 32");

// Running softmax state, per query head. Persistent across tile() calls, which
// is what makes the cache a single streaming pass.
} // namespace

// Softmax state, defined here and reached by `extern` from the begin/finish
// translation units.
alignas(64) float g_m[GQA];                 // running max
alignas(64) float g_l[GQA];                 // running denominator
alignas(64) float g_acc[GQA * HEAD];        // running weighted sum of V

extern "C" __attribute__((noinline)) void
flm_attn_tile(const bfloat16 *restrict q,       // [GQA][HEAD], pre-scaled
              const bfloat16 *restrict kvpack) {  // KVPER x [K][V], see below
  // The tile is one sequential read of the cache: K channel-major
  // [HEAD][TSEQ] followed by V position-major [TSEQ][HEAD]. Keeping them in a
  // single object means one DMA stream per core rather than two.
  //
  // KVPER tiles are processed per call. The online softmax state is carried in
  // g_m/g_l/g_acc across calls anyway, so folding several tiles into one call
  // changes nothing arithmetically — it only decouples the fifo object size
  // from TSEQ, which the fused layer needs.
  for (int u = 0; u < KVPER; ++u) {
  const bfloat16 *restrict kt = kvpack + u * KVSTRIDE;
  const bfloat16 *restrict vt = kt + TSEQ * HEAD;
  for (int h = 0; h < GQA; ++h) {
    // ---- scores: accumulate across the head dim, 32 positions at a time ----
    aie::accum<accfloat, TSEQ> s;
    s.from_vector(aie::zeros<float, TSEQ>());
    for (int d = 0; d < HEAD; ++d)
      s = aie::mac(s, aie::load_v<TSEQ>(kt + d * TSEQ),
                   aie::broadcast<bfloat16, TSEQ>(q[h * QSTRIDE + d]));

    const auto sv = s.template to_vector<float>();

    // ---- online softmax update ----
    const float m_old = g_m[h];
    const float m_tile = aie::reduce_max(sv);
    const float m_new = m_old > m_tile ? m_old : m_tile;

    // exp2, not exp: the 1/sqrt(head_dim) and log2(e) factors are already in q.
    const auto p = aie::exp2<bfloat16>(
        aie::sub(sv, aie::broadcast<float, TSEQ>(m_new)));
    // No scalar libm on the core (`undefined symbol: exp2f`), so the rescale
    // factor goes through the same vector exp2 and one lane is extracted.
    const float corr = float(aie::exp2<bfloat16>(
        aie::broadcast<float, TSEQ>(m_old - m_new))[0]);
    // Accumulate the tile's probability mass in FLOAT. `p` is bf16, and
    // reduce_add on a bf16 vector rounds at every step of the tree — ~1% on 32
    // values, and this is the softmax denominator, so it scales the output
    // directly. (Same fault as the activation block-sums in the GEMV.)
    aie::accum<accfloat, TSEQ> pa;
    pa.from_vector(aie::zeros<float, TSEQ>());
    pa = aie::mac(pa, p, aie::broadcast<bfloat16, TSEQ>(bfloat16(1.0f)));
    g_l[h] = g_l[h] * corr + aie::reduce_add(pa.template to_vector<float>());

    // ---- accumulate p . V, rescaling the running sum by corr ----
    const auto cv = aie::broadcast<float, HALF>(corr);
    for (int off = 0; off < HEAD; off += HALF) {
      aie::accum<accfloat, HALF> a;
      // rescale the old accumulator into the new max's frame
      a.from_vector(aie::mul(aie::load_v<HALF>(g_acc + h * HEAD + off), cv)
                        .template to_vector<float>());
      for (int t = 0; t < TSEQ; ++t)
        a = aie::mac(a, aie::load_v<HALF>(vt + t * HEAD + off),
                     aie::broadcast<bfloat16, HALF>(p[t]));
      aie::store_v(g_acc + h * HEAD + off, a.template to_vector<float>());
    }
    g_m[h] = m_new;
  }
  }
}
