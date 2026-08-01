// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// Opus Quant W4A4 (`Oq4G256`) GEMV for AIE2P — the bandwidth probe for the
// oq4++ decode port.
//
// WHAT THIS ANSWERS. Every tok/s projection in docs/npu/next-phase-goals.md
// assumes an oq4 tile streams at the same ~54.7 GB/s the q4_1 and coarse tiers
// reach. That assumption was untested, and it is the one the whole plan rests
// on. lm_head is the right place to test it: 128256 x 2048 is big enough that
// the dispatch floor is noise, and the coarse tier gives a same-size control.
//
// The comparison is unusually clean. At K=2048 this format is
// 262.7M weights / 256 x 130 B = 133.4 MB against the coarse tier's 131.8 MB —
// within 1.2%. Same bytes, different tile shape, so any GB/s difference is
// attributable to the SHAPE and to nothing else.
//
// THE FORMAT, and why the on-disk block is not what streams here.
// `hipfire-quant-format` stores Oq4G256 as `[f16 scale][128 nibbles]` = 130 B
// per 256-group, symmetric signed-INT4, FWHT-rotated, per-group scale. That
// block is 130 bytes, so within a row the nibbles of group g start at byte
// 130g + 2 — a stride that is not a multiple of 4, let alone of 32. Vector
// loading it directly is exactly the failure this tree has already paid for:
// `aie::load_v` on a misaligned pointer does NOT fault, it reads the wrong
// bytes, and at TSEQ=40 that stayed bit-exact at position 0 while every later
// position collapsed to cosine 0.05.
//
// That is why the format has a separate `Oq4G256ArchPacked` id and why the
// loader repacks per arch: the 130 B block is a STORAGE form, never a kernel
// form. So this picks the NPU repack, and picks it planar:
//
//     [NROWS*NG bf16 scales, padded to 64][NROWS * K/2 packed nibbles]
//
// Byte count is unchanged — 2 B of scale and 128 B of codes per 256 weights,
// 4.0625 bits/weight — so the bandwidth question is answered honestly while
// every vector load lands on a 32-byte boundary. At NROWS=16, K=2048 the scale
// plane is 16*8*2 = 256 B, already a multiple of 64, so the padding is zero and
// the codes start at a natural boundary with nothing wasted.
//
// AGAINST THE COARSE TIER, which this is deliberately shaped to be comparable
// to: that kernel carries ONE scale for a whole output row and reduces its
// accumulator once. This carries K/256 = 8 scales per row and therefore
// reduces 8 times per row. That is the only arithmetic difference, and on a
// design running at 97% of the fabric roof it should be invisible — which is
// itself part of what this measures. If 8x the reductions DID show up, the
// format would be paying for its finer scales in time as well as bytes.
//
// Codes are loaded as native `int4`, not `uint4`: Opus Quant is SYMMETRIC
// signed, so the sign extension rides the same `vldb.unpack` + `vups.4x` the
// unsigned path gets and costs nothing. Extracting the sign by hand would be
// two extra vector ops per 64 weights for what the load already gives.
//
// NOT MODELLED HERE: the FWHT rotation of the activation. It is an
// activation-side transform — it changes neither the weight bytes nor how they
// stream, and at K=2048 it is ~22K ops against 2048 MACs a row over 16 rows.
// It belongs in the forward path, not in a bandwidth probe, and leaving it out
// is what keeps this measuring the thing it claims to measure. The GEMV below
// is a correct symmetric-int4 grouped dequant-and-dot for whatever basis the
// weights were packed in.
//
// Compile-time: -DDIM_K (input dims) -DDIM_NROWS (output rows per tile).

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
constexpr int LANES = 32;          // bf16 lanes per MAC, as in flm_q4_1_tile
constexpr int GROUP = 256;         // Oq4G256 — the G in the format name
constexpr int NG = K / GROUP;      // scale groups per output row
constexpr int QBYTES = K / 2;      // packed code bytes per output row
// bf16 scales, rounded up to a 64-byte boundary so the codes start aligned.
// At NROWS=16, K=2048 this is exactly 256 and the round-up is a no-op.
constexpr int SBYTES = ((NROWS * NG * 2) + 63) & ~63;

static_assert(K % GROUP == 0, "K must divide into whole 256-element groups");
static_assert(GROUP % (2 * LANES) == 0, "a group must be whole 64-code loads");
} // namespace

// One oq4 weight tile against one activation:
//     out[r] = sum over groups g of  scale[r][g] * (Q[r][g] . act[g])
//
// `noinline` for the reason `flm_q4_1_tile` is: a core tile has 16 KB of
// program memory and inlining a GEMV body into every caller is what once put
// the fused layer at 103% of it.
extern "C" __attribute__((noinline)) void
flm_gemv_oq4g256(const bfloat16 *restrict act, const uint8 *restrict wtile,
                 float *restrict out) {
  // Round to nearest even on every accum->bf16 conversion. The default mode
  // TRUNCATES, which biases each rounded product one way; a GEMV over 2048
  // terms with heavy cancellation keeps a systematic bias and loses a
  // symmetric one. Measured at 13% on a real row in the q4_1 body.
  aie::set_rounding(aie::rounding_mode::conv_even);

  const bfloat16 *scale = reinterpret_cast<const bfloat16 *>(wtile);
  const uint8 *qs = wtile + SBYTES;

  for (int r = 0; r < NROWS; ++r) {
    const uint8 *qrow = qs + r * QBYTES;

    // ONE reduction per row, not one per group. The first version of this
    // kernel reduced each group's accumulator and summed the scalars, which is
    // NG=8 `reduce_add`s a row -- 128 per tile against the coarse tier's 16 --
    // and each is a 5-step log shuffle-add. Measured, that ran the dispatch at
    // 16.3 GB/s against coarse's 54.9: 3.4x the time for 1.2% more bytes,
    // because the design had stopped being DMA-bound and become bound on MY
    // arithmetic. The bytes were never the problem.
    //
    // Instead each group's vector accumulator is folded into a row accumulator
    // with its scale BROADCAST -- a vector FMA, NG of them, no reduction -- and
    // the row reduces once at the end. Same arithmetic to fp associativity,
    // 8x fewer reductions.
    aie::accum<accfloat, LANES> rowacc;
    rowacc.from_vector(aie::zeros<float, LANES>());

#if OQ4_ONE_SCALE
    // THE DMA CONTROL. Streams the identical oq4 tile -- same bytes, same
    // layout, same loads -- but does the COARSE tier's arithmetic on it: one
    // accumulator across the whole row, one scale, one reduce. It is
    // numerically wrong on purpose (it applies group 0's scale to all NG
    // groups) and exists only to separate "this tile shape cannot stream" from
    // "my per-group arithmetic is too slow". If this reaches the coarse tier's
    // GB/s then the layout is fine and the gap is entirely arithmetic.
    {
      aie::accum<accfloat, LANES> vacc;
      vacc.from_vector(aie::zeros<float, LANES>());
      for (int i = 0; i < K; i += 2 * LANES) {
        const auto packed = aie::load_v<2 * LANES>(
            reinterpret_cast<const int4 *>(qrow + i / 2));
        const auto wide = aie::to_float<bfloat16>(aie::unpack(packed));
        vacc = aie::mac(vacc, wide.template extract<LANES>(0),
                        aie::load_v<LANES>(act + i));
        vacc = aie::mac(vacc, wide.template extract<LANES>(1),
                        aie::load_v<LANES>(act + i + LANES));
      }
      out[r] = static_cast<float>(scale[r * NG]) *
               aie::reduce_add(vacc.template to_vector<float>());
      continue;
    }
#endif

    for (int g = 0; g < NG; ++g) {
      const int k0 = g * GROUP;

      // One 32-lane float accumulator per GROUP, reduced at the group's end.
      // 32 is forced as well as preferred: a 16-lane float accumulator does not
      // legalize on this Peano build (`G_FADD <16 x s32>`), recorded in
      // flm_q4_1_tile.h. Accumulating all NG groups into one vector and scaling
      // afterwards is not available — each group carries its OWN scale, which
      // is the entire point of a grouped format.
      aie::accum<accfloat, LANES> vacc;
      vacc.from_vector(aie::zeros<float, LANES>());

      for (int i = 0; i < GROUP; i += 2 * LANES) {
        // 32 bytes loaded AS 64 signed int4 lanes; the hardware widens them to
        // int8 and converts to bf16. Codes are in [-8, 7], exact in bf16's
        // 8-bit significand, so nothing is lost before the MAC.
        //
        // Alignment holds by construction: qrow is a multiple of QBYTES=1024
        // from a 64-aligned base, and (k0+i)/2 is a multiple of 32. The 130 B
        // on-disk block would break exactly this, which is why it is repacked.
        const auto packed = aie::load_v<2 * LANES>(
            reinterpret_cast<const int4 *>(qrow + (k0 + i) / 2));
        const auto wide = aie::to_float<bfloat16>(aie::unpack(packed));
        const auto lo = wide.template extract<LANES>(0);
        const auto hi = wide.template extract<LANES>(1);

        vacc = aie::mac(vacc, lo, aie::load_v<LANES>(act + k0 + i));
        vacc = aie::mac(vacc, hi, aie::load_v<LANES>(act + k0 + i + LANES));
      }

      // Fold this group into the row with its scale broadcast across the lanes.
      // No reduce here -- that is the whole point.
      rowacc = aie::mac(rowacc, vacc.template to_vector<float>(),
                        aie::broadcast<float, LANES>(
                            static_cast<float>(scale[r * NG + g])));
    }

    out[r] = aie::reduce_add(rowacc.template to_vector<float>());
  }
}
