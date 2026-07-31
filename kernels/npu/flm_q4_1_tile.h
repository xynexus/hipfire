// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// q4_1 decode GEMV tile body for AIE2P — shared header.
//
// The compute body of the `layer.xclbin` reproduction, factored out so the
// plain GEMV and the fused FFN kernel run identical arithmetic instead of
// diverging copies.
//
// Weight format, established from FLM's own container in
// `docs/npu/flm-refe-log.md`: asymmetric q4_1, **32 contiguous input dims per
// block**, one bf16 scale `d` and one bf16 min `m` per block, planar per tile:
//
//     [NROWS*NB bf16 d][NROWS*NB bf16 m][NROWS*K/2 bytes of packed nibbles]
//
// which is 5.00 bits/weight exactly, matching FLM byte for byte. Nibbles are
// in plain element order — byte j carries element 2j in its low nibble and
// element 2j+1 in its high nibble — because that is what lets the codes be
// loaded as a native uint4 vector and widened by the hardware.
//
// The dequant folds out of the inner loop. With w = d*q + m and the GEMV
// summing over K,
//
//     out[n] = sum_b ( d[n,b] * sum_t q[n,b,t]*a[b,t]  +  m[n,b] * sum_t a[b,t] )
//
// so the zero-point term collapses to one scalar per block against an
// activation block-sum that is shared by every output row, and the codes go
// into the MAC as small integers. FLM instead spends a 42-op dequant chain
// materialising bf16 weights; that chain is not reproduced here, and it does
// not need to be — the weight supply is 2.57 MACs/cycle/core against the MAC
// unit's 512, so the body is built for correctness and for bytes.
//
// Compile-time: -DDIM_K (input dims) -DDIM_NROWS (output rows per tile).

#pragma once
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_K
#define DIM_K 2048
#endif
#ifndef DIM_NROWS
#define DIM_NROWS 8
#endif

namespace {
constexpr int K = DIM_K;
constexpr int NROWS = DIM_NROWS;
constexpr int BLK = 32;            // weights per q4_1 block
constexpr int NB = K / BLK;        // blocks per output row
constexpr int HALF = BLK / 2;      // lanes per nibble half within one block
constexpr int LANES = 2 * HALF;    // two blocks per iteration — see below
constexpr int QBYTES = K / 2;      // packed bytes per output row
// Bytes in one weight tile's PAYLOAD:
//   [NROWS*NB bf16 d][NROWS*NB bf16 m][NROWS*K/2 codes]
constexpr int TILE_BYTES = 2 * NROWS * NB * 2 + NROWS * QBYTES;
// ... followed by a universal 64-byte trailer: [f32 row_base][f32 flags][pad].
// `row_base` is the global output-row index of this tile's first row, and it
// replaces every per-core index the kernels would otherwise need — residual
// indexing, RoPE head identity, down-chunk accumulator slot — with no runtime
// scalar arguments and no static cursors. 64 also keeps the tile a multiple of
// 64 so both halves of a double-buffered fifo stay aligned.
constexpr int TILE_TRAILER = 64;
constexpr int TILE_TOTAL = TILE_BYTES + TILE_TRAILER;

inline int tile_row_base(const uint8 *restrict wtile) {
  return int(reinterpret_cast<const float *>(wtile + TILE_BYTES)[0]);
}

// The trailer's second f32. Carries the token's KV-cache position for the k'
// emit, which needs its parity to decide whether this step opens a column pair
// or closes one. f32 rather than bf16 because bf16 is exact on integers only to
// 256 and the position runs to the context length.
inline int tile_flags(const uint8 *restrict wtile) {
  return int(reinterpret_cast<const float *>(wtile + TILE_BYTES)[1]);
}

static_assert(NB % 2 == 0, "K must be a multiple of 64");
static_assert(NB % LANES == 0, "K/32 must be a multiple of 32 for the zero-point reduction");

// Two blocks are unpacked at a time, not one. A 16-lane uint8 vector is 128
// bits, which the AIE2P backend cannot legalize (`unable to legalize
// G_AND <4 x s32>`); 32 lanes is 256 bits and is native. So the byte-domain
// work runs 32-wide while the bf16 domain stays 16-wide per sub-block.
//
// A 32-byte group holds two whole blocks in element order, so unpacking gives
// 64 bf16 lanes that line up with a contiguous 64-element activation run.
// Widening keeps the codes exact — 0..15 is well inside bf16's 8-bit
// significand, so nothing is lost before the MAC.
//
// The native uint4 operand is 75 instructions for this loop against 103 for a
// masked-uint8 version, because `vband` disappears and the widening rides the
// load. The body is op-bound, not bandwidth-bound, at this size, so that is
// the axis that matters.
//
// **Plain element order is a convenience here, NOT a requirement.** Any nibble
// order can be consumed for ~1 extra instruction using the shuffle network:
// llama.cpp's split form (byte j -> elements j and j+16 of one block) unpacks
// to lanes [e0,e16,e1,e17,...], which `aie::interleave_unzip(lo, hi, 1)`
// separates in one op — measured at **76 instructions against this loop's
// 75**. An earlier version of this comment claimed that form cost 25 vector
// ops per 64 weights against 18; that was true only of a `aie::concat`-based
// gather and is wrong as a statement about the format. It matters because
// FLM's own nibble order is not yet known, and consuming it will be cheap.
//
// Note `aie::downshift` on a uint8 vector **segfaults the AIE2P backend**
// (fine on uint16 — see `tools/npu/flm/README.md`), which is what ruled out
// the obvious shift-based nibble extraction in the first place.
inline void unpack_codes(const uint8 *restrict qs,
                         aie::vector<bfloat16, LANES> &lo,
                         aie::vector<bfloat16, LANES> &hi) {
  // Load the 32 bytes AS 64 uint4 lanes and let the hardware widen them. This
  // is the native int4 operand path: it issues `vldb.unpack` (widening folded
  // into the load) and `vups.4x`, which are exactly the instructions FLM's
  // GEMM cores carry (`vunpack:64 vups.4x:64`).
  const auto packed = aie::load_v<2 * LANES>(reinterpret_cast<const uint4 *>(qs));
  const auto wide = aie::to_float<bfloat16>(aie::unpack(packed));
  lo = wide.template extract<LANES>(0);   // block b   -> elements 0..31
  hi = wide.template extract<LANES>(1);   // block b+1 -> elements 32..63
}
} // namespace

// Block sums of the activation, shared by every output row AND by every weight
// tile. They depend only on the activation, which is constant for a whole
// dispatch, so they are computed once by `flm_asum_prepare` and read from here
// rather than recomputed per tile. Recomputing cost ~13 bundles x K/32 blocks =
// **12% of every tile call**, paid 116 times per core for one useful result.
//
// alignas is load-bearing: this array is vector-loaded, and an unaligned
// 512-bit load returns garbage (the symptom is NaN in every output, not a
// fault).
// An `inline` variable (C++17), which is exactly the right tool here and worth
// stating why. This header is included by several translation units — the plain
// GEMV, the residual-fused GEMV, the fused FFN, the fused norm+prepare — and
// IRON compiles each ExternalFunction's source separately, then links only the
// objects whose entry points a given design actually calls. A plain definition
// in the header gives `duplicate symbol` when two includers are linked; a plain
// `extern` with the definition in one entry point's file gives
// `undefined symbol` for any design that does not happen to use that entry
// point (e.g. residual-GEMV + fused-norm). An `inline` variable is one object
// across every TU that includes it, and is emitted by whichever ones survive
// the link — so every combination works.
//
// Filled by whichever prepare entry point runs: `flm_asum_prepare` (block sums
// only) or `flm_norm_prepare` (RMSNorm fused in). alignas is load-bearing: it
// is vector-loaded, and an unaligned 512-bit load returns garbage.
alignas(64) inline bfloat16 g_asum[DIM_K / 32];

// One weight tile against one activation: out[0..NROWS) = W_tile . act.
// Shared by the plain GEMV entry point and by the fused FFN kernel, which
// calls it twice (gate, then up) before applying SwiGLU in-core.
//
// **`noinline` is load-bearing, and it is about program memory, not speed.** A
// core tile has 16 KB of it, and the fused layer needs every phase's code
// resident at once because a Worker is one program per core. Inlined, this body
// is duplicated into all six GEMV entry points and the linker has nothing to
// fold: measured, the fused set links to 14,272 B, and with attention on cores
// 0-7 that is **16,896 B = 103% of 16 KB — one dispatch per layer does not
// fit.** As one out-of-line `linkonce_odr` definition the duplicates fold away,
// 14,608 B of objects link to 10,208, and the same set lands at 12,832 B = 78%
// with 3.5 KB of headroom for control flow.
//
// It is free: the call happens once per weight tile and the loop inside it runs
// K/32 blocks, so the overhead is one call per ~2000 MACs. Measured on
// gemv_bench at 16 cores, throughput is unchanged.
inline __attribute__((noinline)) void flm_q4_1_tile(const bfloat16 *restrict act,
                          const uint8 *restrict wtile,
                          float *restrict out) {
  // Round to nearest even on every accum->bf16 conversion. The default mode
  // TRUNCATES, which puts a one-sided bias on each rounded product; the GEMV
  // sums ~2000 terms whose magnitudes total far more than the result (heavy
  // cancellation), so a systematic bias survives the reduction where a
  // symmetric error would not. Measured cost of leaving it unset: 13% on a
  // real row, against 0.19% for correctly-rounded bf16 operands.
  aie::set_rounding(aie::rounding_mode::conv_even);

  const auto *dq = reinterpret_cast<const bfloat16 *>(wtile);
  const auto *mq = dq + NROWS * NB;
  const uint8 *qs = wtile + 2 * NROWS * NB * sizeof(bfloat16);

  for (int r = 0; r < NROWS; ++r) {
    const bfloat16 *drow = dq + r * NB;
    const bfloat16 *mrow = mq + r * NB;
    const uint8 *qrow = qs + r * QBYTES;

    // ponytail: single accumulator, so both MACs in an iteration serialise on
    // its latency. Splitting them into two independent accumulators is the
    // obvious ILP fix and it does not compile — a second 32-lane float
    // accumulator makes the backend emit a 16-lane float add it cannot
    // legalize (`G_FADD <16 x s32>`), the same limit that forced the
    // zero-point term into this accumulator. Revisit if the Peano backend
    // grows 16-lane float support; see the throughput section of
    // docs/npu/flm-reproduction-results.md.
    aie::accum<accfloat, LANES> vacc;
    vacc.from_vector(aie::zeros<float, LANES>());

    for (int b = 0; b < NB; b += 2) {
      // One 32-byte load carries both blocks in element order, so each
      // unpacked 32-lane code vector lines up with a CONTIGUOUS 32-lane
      // activation load and the scale is one broadcast per block.
      aie::vector<bfloat16, LANES> lo, hi;
      unpack_codes(qrow + b * HALF, lo, hi);

      // Fold the block scale into the activation rather than onto the 32
      // decoded weights: one multiply per block instead of a per-weight
      // rescale, and the codes stay exact going into the MAC.
      const auto d0 = aie::broadcast<bfloat16, LANES>(drow[b]);
      const auto d1 = aie::broadcast<bfloat16, LANES>(drow[b + 1]);

      const auto a0 = aie::mul(aie::load_v<LANES>(act + b * BLK), d0)
                          .template to_vector<bfloat16>();
      const auto a1 = aie::mul(aie::load_v<LANES>(act + (b + 1) * BLK), d1)
                          .template to_vector<bfloat16>();

      vacc = aie::mac(vacc, lo, a0);
      vacc = aie::mac(vacc, hi, a1);
    }

    // Zero-point term, sum_b m[b]*asum[b], as a vector MAC.
    //
    // This was a scalar loop and it MISCOMPILED: written as
    // `msum += float(mrow[b])*asum[b] + float(mrow[b+1])*asum[b+1]` inside the
    // stride-2 loop above, the device dropped every b+1 term — the result
    // matched "even blocks only" to six digits, while `mrow[b]`, `asum[b]`,
    // `sum_b mrow[b]` and `sum_b asum[b]` all read back correct on their own.
    // A plain `for (b=0..NB) m2 += float(mq[b])*asum[b]` produced a third,
    // different wrong value. The vector form below is correct; scalar
    // reductions mixing bf16 loads with a float accumulator are not to be
    // trusted on this Peano build.
    // It accumulates into the SAME 32-lane accumulator as the dot term, since
    // both are summed into one scalar at the end anyway — one reduction, not
    // two. A separate 16-lane accumulator also fails to legalize here
    // (`G_FADD <16 x s32>`), so the width is forced as well as preferred.
    for (int b = 0; b < NB; b += LANES)
      vacc = aie::mac(vacc, aie::load_v<LANES>(mrow + b),
                      aie::load_v<LANES>(g_asum + b));

    out[r] = aie::reduce_add(vacc.template to_vector<float>());
  }
}
