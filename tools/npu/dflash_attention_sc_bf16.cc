// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Single-core NON-CAUSAL cross-attention for ONE DFlash-drafter head.
//
// Plain row-major bf16 layout (no mmul tiling) so the host harness stages
// contiguous tensors — deliberately simple/correct-first for Gate C on npu1
// (4 columns; the 8-col segmented_attention kernel is aie2p-only). Perf/fusion
// is Gate D's concern; this validates the non-causal cross-attention numerics.
//
//   Q : [q_len,  head_dim] bf16   (the block's queries for one q-head)
//   K : [kv_len, head_dim] bf16   (concat(ctx, block) keys for the kv-head)
//   V : [kv_len, head_dim] bf16
//   O : [q_len,  head_dim] bf16
//   scalars: q_len, kv_len, head_dim  (head_dim is the IRON auto-appended tile 'n')
//
// Bidirectional (non-causal): every query attends to every key (all kv_len
// visible). scale = 1/sqrt(head_dim). Full softmax (kv_len small ≤ ~512, fits).
// Matches dflash/model.py eager attention with attention_mask=None.

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

namespace {
constexpr int HEAD_DIM = 128;
constexpr int LANES = 16;
constexpr int DV = HEAD_DIM / LANES;  // 8 vectors per head row
}  // namespace

// Compile-time drafter shapes (baked per build like softmax bakes ctx-len).
// One (q_head) head per call: Q[q_len,128], K[kv_len,128], V[kv_len,128] -> O.
#ifndef HIPFIRE_Q_LEN
#define HIPFIRE_Q_LEN 16
#endif
#ifndef HIPFIRE_KV_LEN
#define HIPFIRE_KV_LEN 48
#endif

extern "C" {

// KV = [K(kv_len,128) | V(kv_len,128)] in one buffer (2 input DMA channels max
// per tile → Q + KV, not Q + K + V).
void dflash_attention_sc_bf16(bfloat16 *restrict Q, bfloat16 *restrict KV,
                              bfloat16 *restrict O) {
  const int32_t q_len = HIPFIRE_Q_LEN;
  const int32_t kv_len = HIPFIRE_KV_LEN;
  const bfloat16 *K = KV;
  const bfloat16 *V = KV + kv_len * HEAD_DIM;
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float scale = 0.08838834764831845f;  // 1/sqrt(128)
  alignas(64) float scores[HIPFIRE_KV_LEN];

  for (int qi = 0; qi < q_len; ++qi) {
    const bfloat16 *qrow = Q + qi * HEAD_DIM;
    aie::vector<bfloat16, LANES> qv[DV];
    for (int d = 0; d < DV; ++d) qv[d] = aie::load_v<LANES>(qrow + d * LANES);

    // scores[ki] = scale * (Q[qi] . K[ki])
    float m = -3.0e30f;
    for (int ki = 0; ki < kv_len; ++ki) {
      const bfloat16 *krow = K + ki * HEAD_DIM;
      aie::vector<float, LANES> partial = aie::zeros<float, LANES>();
      for (int d = 0; d < DV; ++d) {
        aie::vector<float, LANES> prod =
            aie::mul(qv[d], aie::load_v<LANES>(krow + d * LANES)).template to_vector<float>();
        partial = aie::add(partial, prod);
      }
      const float s = aie::reduce_add(partial) * scale;
      scores[ki] = s;
      if (s > m) m = s;
    }

    // softmax (non-causal: all keys visible), exp computed INLINE.
    // exp(x) = 2^iy * 2^fy: 2^iy via an IEEE-754 exponent bit-pack (like
    // softmax_bf16.cc); 2^fy via a degree-6 exp series on w = fy*ln2 ∈ (-ln2,0].
    // Two toolchain traps this form avoids:
    //   1. Inlining the exp keeps the scalar `sum += result` reduction correct —
    //      an __attribute__((noinline)) exp helper computed correct exp values
    //      but the reduction around the call miscompiled to ~1.0 (the peak term
    //      only), leaving the output unnormalised.
    //   2. Degree 6 (not the degree-2 poly softmax_bf16 uses, which is ~9% off
    //      near fy=-1) keeps accuracy; degree-2 showed up as ~3.0 output error.
    const float log2e = 1.442695040888963f;
    const float ln2 = 0.6931471805599453f;
    float sum = 0.0f;
    for (int ki = 0; ki < kv_len; ++ki) {
      const float x = scores[ki] - m;
      const float y = x * log2e;
      int32_t iy = (int32_t)y;          // truncate toward zero (y ≤ 0)
      const float fy = y - (float)iy;   // fractional part ∈ (-1, 0]
      float result;
      if (iy < -127) {
        result = 0.0f;                  // underflow clamp
      } else {
        iy = (iy + 127) << 23;          // pack into IEEE-754 float exponent
        float pow2_iy;
        memcpy(&pow2_iy, &iy, sizeof(float));
        const float w = fy * ln2;       // 2^fy = exp(w), w ∈ (-ln2, 0]
        const float pow2_fy =
            1.0f + w * (1.0f + w * (0.5f + w * (0.1666666666666667f +
            w * (0.0416666666666667f + w * (0.0083333333333333f +
            w * 0.0013888888888889f)))));
        result = pow2_iy * pow2_fy;
      }
      scores[ki] = result;
      sum += result;
    }
    const float inv = 1.0f / (sum + 1e-7f);

    // O[qi] = inv * Σ_ki exp_ki * V[ki].
    // NOTE: broadcast the runtime weight as *float* (aie::broadcast<float>),
    // NOT bfloat16. On this toolchain aie::broadcast<bfloat16>(runtime_scalar)
    // miscompiles — a runtime-varying bf16 broadcast gives cos~0.09, while a
    // compile-time-constant bf16 broadcast is fine. A float broadcast of a
    // runtime scalar (as qwen3_final_pool_l2_bf16.cc uses) works correctly.
    // mul(bf16_vec, float_vec)->float + add is the reduction the score
    // dot-product already validates.
    const aie::vector<bfloat16, LANES> bf_one =
        aie::broadcast<bfloat16, LANES>(bfloat16(1.0f));  // bf16->float via const mul
    aie::vector<float, LANES> ov[DV];
    for (int d = 0; d < DV; ++d) ov[d] = aie::zeros<float, LANES>();
    for (int ki = 0; ki < kv_len; ++ki) {
      const bfloat16 *vrow = V + ki * HEAD_DIM;
      const aie::vector<float, LANES> wv = aie::broadcast<float, LANES>(scores[ki]);
      for (int d = 0; d < DV; ++d) {
        aie::vector<float, LANES> vf =
            aie::mul(aie::load_v<LANES>(vrow + d * LANES), bf_one).template to_vector<float>();
        ov[d] = aie::add(ov[d], aie::mul(vf, wv).template to_vector<float>());
      }
    }
    bfloat16 *orow = O + qi * HEAD_DIM;
    const aie::vector<float, LANES> invv = aie::broadcast<float, LANES>(inv);
    for (int d = 0; d < DV; ++d)
      aie::store_v(orow + d * LANES,
                   aie::mul(ov[d], invv).template to_vector<bfloat16>());
  }
}

}  // extern "C"
