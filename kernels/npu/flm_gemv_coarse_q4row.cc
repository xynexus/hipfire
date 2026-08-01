// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// Coarse-tier GEMV for the two-pass lm_head, AIE2P.
//
// This is NOT flm_q4_1_tile with a flag. The q4_1 format is asymmetric —
// `w = d*q + m` with a bf16 scale AND a bf16 min per 32 input dims, codes
// 0..15 — and its whole body is organised around folding the min term into a
// shared activation block-sum. The coarse tier is symmetric, has no min, and
// carries ONE scale for a whole output row:
//
//     out[n] = scale[n] * sum_k q[n,k] * a[k],   q in [-7, 7]
//
// so there is no zero-point term, no `g_asum`, no per-block broadcast, and no
// prepare kernel. Bending the q4_1 body into this would leave every one of
// those costs in place to compute a thing that does not need them.
//
// WHY IT EXISTS: lm_head is bandwidth bound. Measured, it runs 163.7 MB at
// 54.7 GB/s — 97% of the 56.5 GB/s fabric roof — so the arithmetic is not the
// lever and never was. The coarse tier is 4 bits flat plus 4 bytes a row
// against q4_1's 5.00, which is ~20% fewer bytes and therefore ~20% less time.
// The accuracy it gives up is spent where it is free: the coarse pass only has
// to keep the true argmax inside a small top-K, and the host rescores that
// shortlist exactly.
//
// TILE LAYOUT, planar, `[NROWS f32 scales][NROWS*K/2 packed nibbles]`:
//
//   * The scale is f32 here, not the f16 the container format stores. At
//     NROWS=16 that is exactly 64 bytes — one alignment unit, no padding —
//     whereas 16 bf16 scales are 32 bytes and would need 32 bytes of padding
//     to keep the tile a multiple of 64 (which the log records as load-bearing:
//     a misaligned double-buffered fifo corrupted alternating tiles). So the
//     wider scale is FREE against the aligned alternative, and it removes the
//     question of whether rounding the row scale costs recall.
//   * There is no 64-byte trailer. Every other tile in this tree carries one
//     for `row_base`, which replaces per-core indexing — residual slot, RoPE
//     head identity, down-chunk accumulator. lm_head needs none of that: the
//     output is NROWS floats and the fifo already places them. Omitting it
//     saves 0.39%, and 0.39% of a 2.4 ms dispatch is not nothing.
//
// Nibbles are in plain element order — byte j carries element 2j in its LOW
// half and 2j+1 in its HIGH half — matching `build_coarse_q4row`.
//
// **The codes are loaded as native `int4`, not `uint4`.** `flm_q4_1_tile`
// loads uint4 because q4_1 codes are unsigned 0..15; these are two's-complement
// signed nibbles, and `int4` is a first-class vector element type on this
// arch (`native_vector_type<int4, 64> = v64int4`). So the sign extension rides
// the same `vldb.unpack` + `vups.4x` the unsigned path gets, and costs nothing.
// Extracting the sign by hand — compare against 7 and subtract 16 — would be
// two extra vector ops per 64 weights for a result the load already gives.
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
constexpr int QBYTES = K / 2;      // packed bytes per output row
// f32 scales, rounded up to a 64-byte boundary so the codes start aligned.
constexpr int SBYTES = ((NROWS * 4) + 63) & ~63;

static_assert(K % 64 == 0, "K must be a multiple of 64 (two 32-lane halves)");
} // namespace

// One coarse weight tile against one activation: out[0..NROWS) = scale * (Q . act).
//
// `noinline` for the same reason `flm_q4_1_tile` is: a core tile has 16 KB of
// program memory and inlining a GEMV body into every entry point that calls it
// is what put the fused layer at 103% of it. Here there is one caller, so this
// is cheap insurance rather than a fix, and the call is one per ~2000 MACs.
extern "C" __attribute__((noinline)) void
flm_gemv_coarse_q4row(const bfloat16 *restrict act, const uint8 *restrict wtile,
                      float *restrict out) {
  // Round to nearest even on every accum->bf16 conversion. The default mode
  // TRUNCATES, which biases each rounded product one way; a GEMV over 2048
  // terms with heavy cancellation keeps a systematic bias and loses a
  // symmetric one. Measured at 13% on a real row in the q4_1 body.
  aie::set_rounding(aie::rounding_mode::conv_even);

  const float *scale = reinterpret_cast<const float *>(wtile);
  const uint8 *qs = wtile + SBYTES;

  for (int r = 0; r < NROWS; ++r) {
    const uint8 *qrow = qs + r * QBYTES;

    // One 32-lane float accumulator, reduced once at the end. 32 is forced as
    // well as preferred: a 16-lane float accumulator does not legalize on this
    // Peano build (`G_FADD <16 x s32>`), which is recorded in flm_q4_1_tile.h.
    aie::accum<accfloat, LANES> vacc;
    vacc.from_vector(aie::zeros<float, LANES>());

    for (int i = 0; i < K; i += 2 * LANES) {
      // 32 bytes loaded AS 64 signed uint4 lanes; the hardware widens them to
      // int8 and converts to bf16. Codes are in [-7, 7], exact in bf16's 8-bit
      // significand, so nothing is lost before the MAC.
      const auto packed =
          aie::load_v<2 * LANES>(reinterpret_cast<const int4 *>(qrow + i / 2));
      const auto wide = aie::to_float<bfloat16>(aie::unpack(packed));
      const auto lo = wide.template extract<LANES>(0);
      const auto hi = wide.template extract<LANES>(1);

      // No per-block scale to fold in: the row's single scale comes out of the
      // sum at the end, which is one multiply per row instead of K/32 per row.
      vacc = aie::mac(vacc, lo, aie::load_v<LANES>(act + i));
      vacc = aie::mac(vacc, hi, aie::load_v<LANES>(act + i + LANES));
    }

    out[r] = scale[r] * aie::reduce_add(vacc.template to_vector<float>());
  }
}
