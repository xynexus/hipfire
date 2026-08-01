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
//
// **Operands arrive as `uint8`, not `bfloat16`, and are cast here.** The fused
// layer has ONE operand fifo per pair carrying q4_1 weight tiles in P1/P3/P4/P5
// and q'/KV in P2 — a core has two input DMA channels and the broadcast takes
// one, so a second data fifo does not exist. A fifo has a single object type and
// IRON requires the kernel's declared argument type to match it exactly:
// `func.call op operand type mismatch: expected memref<128xbf16>, but provided
// memref<256xui8>`. uint8 is the type the weight tiles need, so attention casts.
//
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
} // namespace

// The KV operand object's 64-byte trailer, the same convention `row_base` uses
// on a weight tile. The object is DIM_KVOBJ bytes and the KV tiles occupy
// 2*KVPER*TSEQ*HEAD of it, so the tail is free.
//
// It carries this core's offset into the SHARED q' block. q' cannot ride the
// operand fifo — an object held across other acquire/release cycles on one fifo
// does not stay valid (measured: tools/npu/flm/attn_phase one-fifo variant,
// 2.93e-02 against a 1.08e-03 tolerance, unchanged at fifo depth 2/3/4) — so it
// rides the broadcast, where every core sees all 32 heads and needs to be told
// which 4 are its own.
#ifndef DIM_KVOBJ
#define DIM_KVOBJ 20544
#endif
#ifndef DIM_NPADOFF
#define DIM_NPADOFF 4096        // bf16 elements: npad sits past all 32 heads
#endif
namespace {
// Off by default: harnesses that give each core its own packed q block have no
// trailer to read, and a garbage offset indexes out of the block.
#ifndef QOFF_FROM_KV
#define QOFF_FROM_KV 0
#endif
inline int kv_qoff(const uint8 *restrict kv) {
#if QOFF_FROM_KV
  return int(reinterpret_cast<const float *>(kv + DIM_KVOBJ - 64)[0]);
#else
  (void)kv;
  return 0;
#endif
}
} // namespace
namespace {

// The scores for one tile are TSEQ floats and there is no TSEQ-lane register
// for an arbitrary TSEQ. They are carried as a 32-lane vector plus an optional
// TAIL vector -- also 32 lanes, of which only SCORE1 are valid. TSEQ may be 32
// (one vector; the tail compiles out entirely and the code is textually what it
// always was) or 40 (32 + 8 valid).
//
// The tail is a FULL 32-lane vector, so it reads 32 columns from column SCORE0,
// of which the last 24 run past K's row into the next channel's -- and are
// masked off before they can reach the running max, the denominator or p.V. The
// read stays inside the KV object (V follows K), so it is garbage, never a fault.
//
// It is 32 lanes rather than 8 for a reason that is only half established. An
// 8-lane tail failed in the backend with `unable to legalize instruction:
// <16 x s32> G_FADD`; widening it to 32 did NOT fix that, and what did was
// folding the tail's denominator into the SAME accumulator as the head's (its
// masked lanes exp2 to exactly 0) and its max into one reduction over
// `aie::max(sv, sv_t)`. So the cause was the second accumulator and the second
// reduce_add, not the width -- and an 8-lane tail was never retried after the
// fold. Recording that rather than the tidier story, because the tidier story is
// the one the first version of this comment told and it was wrong.
//
// 40 is not arbitrary. The fused layer's KV tile is `2 * HEAD * TSEQ` bf16 and
// has to fit the one operand size every fifo in that design shares, 10304 B,
// which is also group C's q4nx weight tile: 32 -> 8192 B, 40 -> 10240, 44 ->
// 11264 and it no longer fits. At 40, `KVSTRIDE = max(KTILE + VTILE,
// OPERAND / 2) = max(5120, 5152) = 5152` — today's stride — so the cache layout
// does not move either.
#define SCORE0 32
#define SCORE1 (DIM_TSEQ - SCORE0)
static_assert(TSEQ == 32 || TSEQ == 40,
              "scores are one 32-lane register plus an optional tail");
static_assert(SCORE1 >= 0 && SCORE1 <= SCORE0, "the tail is at most one vector");
static_assert(HEAD % 32 == 0, "head_dim must be a multiple of 32");

// K is channel-major with a row stride of TSEQ, so `kt + d * TSEQ` is 64-byte
// aligned for every d only when TSEQ is a multiple of the 32-lane vector. At
// TSEQ = 40 three loads in four are not, and `aie::load_v` on a misaligned
// pointer does NOT fault — it reads the wrong bytes, which is worse. It cost one
// build here: pos 0 was bit-exact (its softmax is over one entry, so no score
// reaches the output) and every position past it collapsed to a cosine of 0.05
// to 0.28 against the oracle.
//
// The rows are 40 bf16 = 80 B apart, so the pointer is always a multiple of 8
// elements and never of 32; `load_unaligned_v` is told the 8 it can rely on.
#if SCORE1
#define KLOAD(p) aie::load_unaligned_v<SCORE0>((p), 8)
#else
#define KLOAD(p) aie::load_v<SCORE0>(p)
#endif

// Running softmax state, per query head. Persistent across tile() calls, which
// is what makes the cache a single streaming pass.
} // namespace

// Softmax state, defined here and reached by `extern` from the begin/finish
// translation units.
// ATTN_MASK_PAD: drop the padded lanes instead of correcting for them later.
//
// The cache streams in whole TSEQ tiles, so a sequence that is not a multiple of
// TSEQ leaves a padded tail of K=0, V=0. A zero K gives a score of 0, not -inf,
// so each padded lane contributes exp2(0 - m) to the denominator.
// `flm_attn_finish` compensates by subtracting npad * exp2(-m) at the end.
//
// That is exact in real arithmetic and badly conditioned in floating point. At
// position 0 there are 31 padded lanes against 1 real one: the denominator is
// built up to ~31 and then nearly all of it is subtracted away, leaving the real
// term plus the rounding the discarded terms deposited. Measured 30.8 ULP at
// pos 0, decaying to the 0.69 ULP floor by pos 31 as the padded fraction shrinks.
//
// Masking is the conditioning fix: the padded lanes never enter g_l or g_m at
// all, so there is nothing to subtract. It also fixes a second-order error the
// subtraction could not — a padded score of 0 can WIN the running max when every
// real score is negative, which shifts the whole softmax frame.
//
// Off by default; `flm_attn_finish` must be built with the same flag, since the
// two corrections would otherwise both apply.
#ifndef ATTN_MASK_PAD
#define ATTN_MASK_PAD 0
#endif

#if ATTN_MASK_PAD || SCORE1
namespace {
// No iota in the vector API and no runtime init on a core, so the lane indices
// are a constexpr table. It runs to TSEQ + SCORE0 rather than TSEQ when there is
// a tail, because the tail's mask loads 32 lanes starting at column SCORE0 and
// only SCORE1 of them are real columns -- the comparison has to see indices past
// TSEQ so those lanes lose it. At TSEQ = 32 there is no tail and the table is
// TSEQ entries, exactly as it was.
constexpr int LANES_N = TSEQ + (SCORE1 ? SCORE0 : 0);
struct LaneIdx {
  float v[LANES_N];
  constexpr LaneIdx() : v() {
    for (int i = 0; i < LANES_N; ++i)
      v[i] = float(i);
  }
};
alignas(64) constexpr LaneIdx g_lane{};
} // namespace
#endif

alignas(64) float g_m[GQA];                 // running max
alignas(64) float g_l[GQA];                 // running denominator
alignas(64) float g_acc[GQA * HEAD];        // running weighted sum of V

extern "C" __attribute__((noinline)) void
flm_attn_tile(const uint8 *restrict q_raw,      // [GQA][HEAD] bf16, pre-scaled
              const uint8 *restrict kv_raw) {     // KVPER x [K][V] bf16
  // q_raw is the shared broadcast block; the trailer says which slice.
  const auto *restrict q =
      reinterpret_cast<const bfloat16 *>(q_raw) + kv_qoff(kv_raw);
  const auto *restrict kvpack =
      reinterpret_cast<const bfloat16 *>(kv_raw);
  // The tile is one sequential read of the cache: K channel-major
  // [HEAD][TSEQ] followed by V position-major [TSEQ][HEAD]. Keeping them in a
  // single object means one DMA stream per core rather than two.
  //
  // KVPER tiles are processed per call. The online softmax state is carried in
  // g_m/g_l/g_acc across calls anyway, so folding several tiles into one call
  // changes nothing arithmetically — it only decouples the fifo object size
  // from TSEQ, which the fused layer needs.
#if ATTN_MASK_PAD
  // npad must be read from wherever THIS build put it. Reading the KV trailer
  // unconditionally was wrong: designs that leave NPAD_FROM_KV at 0 write npad
  // into the q tail and never touch the KV trailer, so the mask consumed
  // garbage and made the result far worse (p1p2_chain seq=2: 9.0e-02 -> 4.0e-01).
  // Mirror flm_attn_finish exactly rather than assuming one layout.
#if NPAD_FROM_KV
  const float npad_dec =
      *reinterpret_cast<const float *>(kv_raw + DIM_KVOBJ - 60);
#else
  const float npad_dec =
      *reinterpret_cast<const float *>(
          reinterpret_cast<const bfloat16 *>(q_raw) + DIM_NPADOFF);
#endif
#endif
  for (int u = 0; u < KVPER; ++u) {
  const bfloat16 *restrict kt = kvpack + u * KVSTRIDE;
  const bfloat16 *restrict vt = kt + TSEQ * HEAD;
  for (int h = 0; h < GQA; ++h) {
    // ---- scores: accumulate across the head dim, 32 positions at a time ----
    aie::accum<accfloat, SCORE0> s;
    s.from_vector(aie::zeros<float, SCORE0>());
    for (int d = 0; d < HEAD; ++d)
      s = aie::mac(s, KLOAD(kt + d * TSEQ),
                   aie::broadcast<bfloat16, SCORE0>(q[h * QSTRIDE + d]));

    auto sv = s.template to_vector<float>();
#if SCORE1
    // The tail lanes. K is channel-major [HEAD][TSEQ], so this segment is the
    // same rows at a +SCORE0 column offset -- one more load per head dim.
    aie::accum<accfloat, SCORE0> s_t;
    s_t.from_vector(aie::zeros<float, SCORE0>());
    for (int d = 0; d < HEAD; ++d)
      s_t = aie::mac(s_t, KLOAD(kt + d * TSEQ + SCORE0),
                     aie::broadcast<bfloat16, SCORE0>(q[h * QSTRIDE + d]));
    auto sv_t = s_t.template to_vector<float>();
    // Only SCORE1 of these lanes are this channel's. Killing the rest here,
    // unconditionally, is what makes the over-read safe -- it must not depend
    // on ATTN_MASK_PAD, which is a different correction and may be off.
    sv_t = aie::select(aie::broadcast<float, SCORE0>(-3.0e38f), sv_t,
                       aie::lt(aie::load_v<SCORE0>(g_lane.v),
                               aie::broadcast<float, SCORE0>(float(SCORE1))));
#endif
#if ATTN_MASK_PAD
    // Valid lanes in THIS tile. npad rides the object's trailer and covers the
    // whole object, so it is spent tile by tile from the end.
    {
      const int nv_i = TSEQ * KVPER - int(npad_dec) - u * TSEQ;
      const int nv = nv_i < 0 ? 0 : (nv_i > TSEQ ? TSEQ : nv_i);
      sv = aie::select(
          aie::broadcast<float, SCORE0>(-3.0e38f), sv,
          aie::lt(aie::load_v<SCORE0>(g_lane.v),
                  aie::broadcast<float, SCORE0>(float(nv))));
#if SCORE1
      sv_t = aie::select(
          aie::broadcast<float, SCORE0>(-3.0e38f), sv_t,
          aie::lt(aie::load_v<SCORE0>(g_lane.v + SCORE0),
                  aie::broadcast<float, SCORE0>(float(nv))));
#endif
    }
#endif

    // ---- online softmax update ----
    // The running max and the denominator reduce over BOTH segments: a tail
    // lane that wins the max and is not seen would leave the whole softmax
    // frame wrong, and one that is not summed would leave the denominator short.
    const float m_old = g_m[h];
#if SCORE1
    // One reduction over the elementwise max of the two segments, rather than
    // two reductions: the masked tail lanes are -3.0e38 and lose it.
    const float m_tile = aie::reduce_max(aie::max(sv, sv_t));
#else
    const float m_tile = aie::reduce_max(sv);
#endif
    const float m_new = m_old > m_tile ? m_old : m_tile;

    // exp2, not exp: the 1/sqrt(head_dim) and log2(e) factors are already in q.
    const auto p = aie::exp2<bfloat16>(
        aie::sub(sv, aie::broadcast<float, SCORE0>(m_new)));
#if SCORE1
    const auto p_t = aie::exp2<bfloat16>(
        aie::sub(sv_t, aie::broadcast<float, SCORE0>(m_new)));
#endif
    // No scalar libm on the core (`undefined symbol: exp2f`), so the rescale
    // factor goes through the same vector exp2 and one lane is extracted.
    const float corr = float(aie::exp2<bfloat16>(
        aie::broadcast<float, SCORE0>(m_old - m_new))[0]);
    // Accumulate the tile's probability mass in FLOAT. `p` is bf16, and
    // reduce_add on a bf16 vector rounds at every step of the tree — ~1% on 32
    // values, and this is the softmax denominator, so it scales the output
    // directly. (Same fault as the activation block-sums in the GEMV.)
    aie::accum<accfloat, SCORE0> pa;
    pa.from_vector(aie::zeros<float, SCORE0>());
    pa = aie::mac(pa, p, aie::broadcast<bfloat16, SCORE0>(bfloat16(1.0f)));
#if SCORE1
    // Into the SAME accumulator: a masked lane's score is -3.0e38, so its
    // exp2 is exactly 0 and it adds nothing. One reduction, not two.
    pa = aie::mac(pa, p_t, aie::broadcast<bfloat16, SCORE0>(bfloat16(1.0f)));
#endif
    g_l[h] = g_l[h] * corr + aie::reduce_add(pa.template to_vector<float>());

    // ---- accumulate p . V, rescaling the running sum by corr ----
    // V is position-major [TSEQ][HEAD], so the tail is just the last SCORE1
    // rows and needs no second layout -- only its own probabilities.
    const auto cv = aie::broadcast<float, HALF>(corr);
    for (int off = 0; off < HEAD; off += HALF) {
      aie::accum<accfloat, HALF> a;
      // rescale the old accumulator into the new max's frame
      a.from_vector(aie::mul(aie::load_v<HALF>(g_acc + h * HEAD + off), cv)
                        .template to_vector<float>());
      for (int t = 0; t < SCORE0; ++t)
        a = aie::mac(a, aie::load_v<HALF>(vt + t * HEAD + off),
                     aie::broadcast<bfloat16, HALF>(p[t]));
#if SCORE1
      for (int t = 0; t < SCORE1; ++t)
        a = aie::mac(a, aie::load_v<HALF>(vt + (SCORE0 + t) * HEAD + off),
                     aie::broadcast<bfloat16, HALF>(p_t[t]));
#endif
      aie::store_v(g_acc + h * HEAD + off, a.template to_vector<float>());
    }
    g_m[h] = m_new;
  }
  }
}
