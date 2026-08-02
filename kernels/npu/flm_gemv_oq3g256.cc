// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// Opus Quant W3A4 (`Oq3G256`) GEMV for AIE2P — the bandwidth probe for the
// memory-ceiling format.
//
// WHY THIS EXISTS. Decode here is 92.4% DMA, so bits per weight IS speed, and
// oq3 is 98 B per 256-group = 3.0625 b/w against oq4's 130 B / 4.0625. That is
// 25% less weight traffic: 475 MB/token against 630, ~104 tok/s against ~78,
// on a path where FLM measures 61.18. hipfire's own format docs call it "the
// memory-ceiling lever". If it streams, it is worth more than oq4 by a wide
// margin — and it needs no repack for the GPU path because its bit-plane
// storage already IS that kernel layout.
//
// THE FORMAT (codecs.rs `quantize_oq3g256`). Per 256-group:
// `[f16 scale][8 x (3 u32 bit-planes)]`. Sub-block s covers values 32s..32s+31;
// plane b of that sub-block has bit i set iff bit b of `u` is set, where
// `u = (q as u8) & 7` is the 3-bit two's-complement code and q is in [-3, 3]
// (symmetric, avoiding the asymmetric -4 endpoint, as oq4 avoids -8).
//
// So reconstruction per lane is `u = p0 | p1<<1 | p2<<2` then sign-extend 3
// bits. This kernel does it as:
//
//     base = (m0 ? 1 : 0) + (m1 ? 2 : 0)      // 0..3
//     q    = m2 ? base - 4 : base             // 3-bit two's complement
//
// which is exact: u<4 gives q=u, and u>=4 gives q=u-8 because base is u&3.
// A bit-plane IS a 32-lane mask, so `aie::mask<32>::from_uint32` turns each
// plane into one directly and `aie::select` spreads it — no scalar loop, no
// per-lane extract.
//
// THE REPACK, and it is not the on-disk one. On disk the three planes of a
// sub-block are adjacent: p0,p1,p2 every 12 bytes. Twelve is not a vector
// stride, and this tree has already paid for a misaligned `aie::load_v` that
// does not fault but reads the wrong bytes. Since the NPU repack is ours to
// choose (this is exactly what the format's per-arch repack exists for), the
// planes are stored PLANE-MAJOR within a group:
//
//     [8 u32 of p0][8 u32 of p1][8 u32 of p2]   = 96 B per 256-group
//
// Each plane for a whole group is then 32 contiguous, aligned bytes. Byte count
// is identical, so the bandwidth question is answered honestly.
//
// Tile, planar, same shape as the oq4 probe so the two are comparable:
//
//     [NROWS*NG bf16 scales, padded to 64][NROWS * K/8*3 plane bytes]
//
// At NROWS=16, K=2048: 256 B of scale (already 64-aligned) + 12288 B of planes
// = 12544 B/tile, which is 3.0625 b/w exactly.
//
// WHAT TO EXPECT, and how to read a bad number. The oq4 probe went
// 16.3 -> 31.4 -> 35.5 -> 55.5 GB/s on formulation alone with its bytes never
// changing; the first formulation looked like a verdict on the format and was
// not. oq3 gets 75% of oq4's time per 256 weights (it moves 75% of the bytes)
// and spends roughly 4x the arithmetic to unpack them, so it is the harder
// case. Measure it against the control tier (`oq3_1s`), which streams these
// exact bytes with trivial arithmetic: if the control reaches the ceiling and
// this does not, the gap is the unpack and not the layout.
//
// NOT MODELLED: the FWHT rotation, the clip-search, and SpinQuant. The format
// doc is explicit that 3-bit is only viable with the SpinQuant learned rotation
// ON TOP of the FWHT — "the FWHT here is the fixed-rotation floor" — so oq3
// carries a real accuracy precondition that this file does not test and no
// accuracy claim may rest on. Bytes are unaffected, which is all this measures.
//
// Compile-time: -DDIM_K -DDIM_NROWS, and -DOQ3_ONE_SCALE=1 for the control.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_K
#define DIM_K 2048
#endif
#ifndef DIM_NROWS
#define DIM_NROWS 16
#endif

namespace {
constexpr int K = DIM_K;
constexpr int NROWS = DIM_NROWS;
constexpr int LANES = 32;          // one bit-plane word == one 32-lane mask
constexpr int GROUP = 256;
constexpr int NG = K / GROUP;      // scale groups per output row
constexpr int SUB = GROUP / LANES; // 8 sub-blocks of 32 weights per group
constexpr int PBYTES = K / 8 * 3;  // 3 bits a weight = 768 B per row at K=2048
constexpr int SBYTES = ((NROWS * NG * 2) + 63) & ~63;

static_assert(K % GROUP == 0, "K must divide into whole 256-element groups");
} // namespace

extern "C" __attribute__((noinline)) void
flm_gemv_oq3g256(const bfloat16 *restrict act, const uint8 *restrict wtile,
                 float *restrict out) {
  aie::set_rounding(aie::rounding_mode::conv_even);

  const bfloat16 *scale = reinterpret_cast<const bfloat16 *>(wtile);
  const uint8 *ps = wtile + SBYTES;

  const auto v_zero = aie::zeros<int16, LANES>();
  const auto v_one = aie::broadcast<int16, LANES>(1);
  const auto v_two = aie::broadcast<int16, LANES>(2);
  const auto v_four = aie::broadcast<int16, LANES>(4);
  int16_t lane_init[LANES];
  for (int i = 0; i < LANES; ++i) lane_init[i] = static_cast<int16_t>(i + 1);
  const auto v_lane = aie::load_v<LANES>(lane_init);
  const auto w_zero = aie::zeros<int8, 2 * LANES>();
  const auto w_one = aie::broadcast<int8, 2 * LANES>(1);
  const auto w_two = aie::broadcast<int8, 2 * LANES>(2);
  const auto w_four = aie::broadcast<int8, 2 * LANES>(4);

  for (int r = 0; r < NROWS; ++r) {
    const uint32_t *prow =
        reinterpret_cast<const uint32_t *>(ps + r * PBYTES);

    aie::accum<accfloat, LANES> rowacc;
    rowacc.from_vector(aie::zeros<float, LANES>());

#if OQ3_ONE_SCALE
    // THE DMA CONTROL. Streams the identical tile and TOUCHES EVERY WORD --
    // that part matters, because a control whose loads get folded away measures
    // an empty dispatch and would report a flattering ceiling. It accumulates
    // the raw plane words with no spread, no hash, no per-group work, and is
    // numerically meaningless by construction. If this reaches the coarse
    // tier's GB/s and the real path does not, the gap is the unpack.
    {
      // VECTOR loads, not scalar. The first version of this control summed the
      // 192 plane words with a scalar loop -- 3072 scalar loads a tile -- and
      // measured SCALAR LOAD THROUGHPUT, not DMA. It reported 36.2 GB/s and
      // held that number across a 50% change in tile size, which is exactly
      // what a compute-bound control does and exactly what a DMA control does
      // not. A control that is itself bound on the wrong resource answers no
      // question at all.
      aie::vector<int32, LANES> vacc = aie::zeros<int32, LANES>();
      const int32 *pw = reinterpret_cast<const int32 *>(prow);
      for (int w = 0; w < NG * 3 * SUB; w += LANES)
        vacc = aie::add(vacc, aie::load_v<LANES>(pw + w));
      out[r] = static_cast<float>(scale[r * NG]) *
               static_cast<float>(aie::reduce_add(vacc) & 0xFF);
      continue;
    }
#endif

#if OQ3_SUMS
    // THE SCALE READ, isolated: sum_g scale_g * (g+1). Position-weighted so a
    // permuted or mis-indexed scale plane cannot pass. Everything else --
    // spread, to_float, MAC, act -- is already verified exactly.
    {
      float acc = 0.0f;
      for (int g = 0; g < NG; ++g)
        acc += static_cast<float>(scale[r * NG + g]) * static_cast<float>(g + 1);
      out[r] = acc;
      continue;
    }
#endif

#if OQ3_DOTQ
    // to_float + MAC + act, WITHOUT the scale. The spread is already verified
    // exactly (lane-weighted, 128256/128256), and the packed tile's scale and
    // code planes are verified on the host, so this is the last untested step:
    // `to_float<bfloat16>` on the decoded codes, and the MAC against the real
    // activation. If this matches the host dot product, only the scale multiply
    // is left; if it does not, to_float is the fault.
    {
      aie::accum<accfloat, LANES> dacc;
      dacc.from_vector(aie::zeros<float, LANES>());
      for (int g = 0; g < NG; ++g) {
        const uint32_t *b0 = prow + g * (3 * SUB);
        for (int s2 = 0; s2 < SUB; ++s2) {
          const auto m0 = aie::mask<LANES>::from_uint32(b0[s2]);
          const auto m1 = aie::mask<LANES>::from_uint32(b0[SUB + s2]);
          const auto m2 = aie::mask<LANES>::from_uint32(b0[2 * SUB + s2]);
          auto q = aie::select(v_zero, v_one, m0);
          q = aie::add(q, aie::select(v_zero, v_two, m1));
          q = aie::sub(q, aie::select(v_zero, v_four, m2));
          dacc = aie::mac(dacc, aie::to_float<bfloat16>(q),
                          aie::load_v<LANES>(act + g * GROUP + s2 * LANES));
        }
      }
      out[r] = aie::reduce_add(dacc.template to_vector<float>());
      continue;
    }
#endif

#if OQ3_SUMQ
    // SPREAD ISOLATION. out[r] = sum of the DECODED codes over the row, with no
    // activation and no scale, so it compares against a host q.sum(1) directly.
    // Every other suspect (act indexing, scale folding, bf16 rounding, the MAC)
    // drops out; if this disagrees, the bug is the spread and nothing else.
    {
      int acc = 0;
      for (int g = 0; g < NG; ++g) {
        const uint32_t *b0 = prow + g * (3 * SUB);
        for (int s2 = 0; s2 < SUB; ++s2) {
          const auto m0 = aie::mask<LANES>::from_uint32(b0[s2]);
          const auto m1 = aie::mask<LANES>::from_uint32(b0[SUB + s2]);
          const auto m2 = aie::mask<LANES>::from_uint32(b0[2 * SUB + s2]);
          auto q = aie::select(v_zero, v_one, m0);
          q = aie::add(q, aie::select(v_zero, v_two, m1));
          q = aie::sub(q, aie::select(v_zero, v_four, m2));
          // Weighted by LANE INDEX, not a plain sum: a plain row sum is
          // permutation-invariant, so it matched 128256/128256 while the codes
          // were landing in the wrong lanes. A check that cannot see the
          // failure mode is not a check.
          acc += aie::reduce_add(aie::mul(q, v_lane).template to_vector<int16>());
        }
      }
      out[r] = static_cast<float>(acc);
      continue;
    }
#endif

    for (int g = 0; g < NG; ++g) {
      // Plane-major within the group: p0 words, then p1 words, then p2 words.
      const uint32_t *p0 = prow + g * (3 * SUB);
      const uint32_t *p1 = p0 + SUB;
      const uint32_t *p2 = p1 + SUB;

      // The scale is folded into the WEIGHTS here, as in the shipped oq4
      // kernel: that formulation is what reached the ceiling there (55.5 GB/s
      // against 35.5 for scaling a per-group accumulator), because it removes
      // the per-group accumulator round-trip entirely. It costs precision --
      // `scale * code` rounds to bf16 before the MAC -- which is recorded as an
      // open question for chained layers in docs/npu/next-phase-goals.md.
      const auto sb2 = aie::broadcast<bfloat16, 2 * LANES>(
          static_cast<bfloat16>(scale[r * NG + g]));
#if OQ3_ACCUM_SCALE
      aie::accum<accfloat, LANES> gacc;
      gacc.from_vector(aie::zeros<float, LANES>());
#endif

      // SIXTY-FOUR LANES AT A TIME. int8 at 64 lanes is AIE2P's NATIVE integer
      // vector and the width oq4's proven path uses; at 32 lanes every spread op
      // does half a register's work. `from_uint32` is variadic, so a mask<64>
      // takes the TWO plane words for sub-blocks s and s+1 -- which the
      // plane-major repack already stores adjacently, covering 64 contiguous
      // weights.
      //
      // Per 64 weights: 3 masks, 3 selects, an add, a sub, one to_float, one
      // mul, two MACs = 12 ops, against 22 for the same weights at 32 lanes.
      //
      // No unroll pragma: `unroll(full)` on this loop produced rel 2.8e-1
      // against the rolled form's 2.8e-3, same source. It is not trusted here.
      for (int s = 0; s < SUB; s += 2) {
        const auto m0 = aie::mask<2 * LANES>::from_uint32(p0[s], p0[s + 1]);
        const auto m1 = aie::mask<2 * LANES>::from_uint32(p1[s], p1[s + 1]);
        const auto m2 = aie::mask<2 * LANES>::from_uint32(p2[s], p2[s + 1]);

        auto q = aie::select(w_zero, w_one, m0);
        q = aie::add(q, aie::select(w_zero, w_two, m1));
        q = aie::sub(q, aie::select(w_zero, w_four, m2));

        const auto w = aie::mul(aie::to_float<bfloat16>(q), sb2)
                           .template to_vector<bfloat16>();
        const int k0i = g * GROUP + s * LANES;
        rowacc = aie::mac(rowacc, w.template extract<LANES>(0),
                          aie::load_v<LANES>(act + k0i));
        rowacc = aie::mac(rowacc, w.template extract<LANES>(1),
                          aie::load_v<LANES>(act + k0i + LANES));
      }
#if OQ3_ACCUM_SCALE
      rowacc = aie::mac(rowacc, gacc.template to_vector<float>(),
                        aie::broadcast<float, LANES>(
                            static_cast<float>(scale[r * NG + g])));
#endif
    }

    out[r] = aie::reduce_add(rowacc.template to_vector<float>());
  }
}
