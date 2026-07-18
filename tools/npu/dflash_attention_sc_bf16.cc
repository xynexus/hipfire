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

namespace {
constexpr int HEAD_DIM = 128;
constexpr int LANES = 16;
constexpr int DV = HEAD_DIM / LANES;  // 8 vectors per head row

// exp2-domain exp (same polynomial as segmented_attention_bf16.cc).
__attribute__((noinline)) float exp_f32(float x) {
  // exp(x) = exp2(x * log2e)
  const float v = x * 1.4426950408889634f;
  if (v <= -126.0f) return 0.0f;
  int e = (int)v;
  if ((float)e > v) --e;
  const float z = (v - (float)e) * 0.6931471805599453f;
  const float p = 1.0f + z * (1.0f + z * (0.5f + z * (0.1666666666666667f +
                  z * (0.0416666666666667f + z * (0.0083333333333333f +
                  z * 0.0013888888888889f)))));
  float s = 1.0f;
  if (e >= 0) { for (int i = 0; i < e; ++i) s *= 2.0f; }
  else { for (int i = 0; i < -e; ++i) s *= 0.5f; }
  return s * p;
}
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
  float scores[HIPFIRE_KV_LEN];

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

    // softmax (non-causal: all keys visible)
    float sum = 0.0f;
    for (int ki = 0; ki < kv_len; ++ki) {
      const float w = exp_f32(scores[ki] - m);
      scores[ki] = w;
      sum += w;
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    // O[qi] = Σ_ki softmax[ki] * V[ki]  (float accum, bf16 weight broadcast)
    aie::accum<accfloat, LANES> ov[DV];
    for (int d = 0; d < DV; ++d) ov[d] = aie::zeros<accfloat, LANES>();
    for (int ki = 0; ki < kv_len; ++ki) {
      const bfloat16 *vrow = V + ki * HEAD_DIM;
      const auto wv = aie::broadcast<bfloat16, LANES>((bfloat16)(scores[ki] * inv));
      for (int d = 0; d < DV; ++d)
        ov[d] = aie::mac(ov[d], aie::load_v<LANES>(vrow + d * LANES), wv);
    }
    bfloat16 *orow = O + qi * HEAD_DIM;
    for (int d = 0; d < DV; ++d)
      aie::store_v(orow + d * LANES, ov[d].to_vector<bfloat16>());
  }
}

}  // extern "C"
