// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// Activation block-sums for the q4_1 decode GEMV — see flm_gemv_q4_1.cc.
//
// Split into its own translation unit so it can be a second entry point on
// the same core: IRON compiles each ExternalFunction's source separately, so
// two entry points in ONE file link twice and fail with duplicate symbols.
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
constexpr int BLK = 32;
constexpr int NB = K / BLK;
constexpr int HALF = BLK / 2;
} // namespace

extern bfloat16 g_asum[];

// Call once per activation, before the tile loop.
extern "C" __attribute__((noinline)) void
flm_asum_prepare(const bfloat16 *restrict act) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int b = 0; b < NB; ++b) {
    const auto ones = aie::broadcast<bfloat16, HALF>(bfloat16(1.0f));
    aie::accum<accfloat, HALF> t;
    t.from_vector(aie::zeros<float, HALF>());
    t = aie::mac(t, aie::load_v<HALF>(act + b * BLK), ones);
    t = aie::mac(t, aie::load_v<HALF>(act + b * BLK + HALF), ones);
    // float accumulation, rounded to bf16 only at the end: reducing a bf16
    // vector rounds at every step of the tree and loses ~1% on a 32-element sum.
    g_asum[b] = bfloat16(aie::reduce_add(t.template to_vector<float>()));
  }
}

