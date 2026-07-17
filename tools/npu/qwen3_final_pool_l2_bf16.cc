// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef HIPFIRE_HIDDEN_SIZE
#define HIPFIRE_HIDDEN_SIZE 1024
#endif

namespace {
constexpr int K = HIPFIRE_HIDDEN_SIZE;
constexpr int VEC = 16;
constexpr int HIDDEN_PAIR_BYTES = 2 * K * sizeof(bfloat16);
}

extern "C" __attribute__((noinline, minsize)) void
hipfire_qwen3_select_last(const int8_t *restrict input_and_lengths,
                          bfloat16 *restrict selected, int32_t token) {
  const auto *input = reinterpret_cast<const bfloat16 *>(input_and_lengths);
  const auto *lengths =
      reinterpret_cast<const int32_t *>(input_and_lengths + HIDDEN_PAIR_BYTES);
  for (int document = 0; document < 2; ++document) {
    if (lengths[document] == token + 1) {
      for (int inner = 0; inner < K; inner += VEC)
        aie::store_v(selected + document * K + inner,
                     aie::load_v<VEC>(input + document * K + inner));
    }
  }
}

extern "C" __attribute__((noinline, minsize)) void
hipfire_qwen3_final_norm_l2(const bfloat16 *restrict selected,
                            const float *restrict weight_and_epsilon,
                            float *restrict scratch, float *restrict output) {
  const float epsilon = weight_and_epsilon[K];
  for (int document = 0; document < 2; ++document) {
    const auto *input = selected + document * K;
    auto *temporary = scratch + document * K;
    auto *destination = output + document * K;
    float sum_sq = 0.0f;
    for (int inner = 0; inner < K; inner += VEC) {
      const auto values = aie::load_v<VEC>(input + inner);
      sum_sq += aie::reduce_add(aie::mul(values, values).to_vector<float>());
    }
    const float inv_rms =
        aie::invsqrt(aie::broadcast<float, VEC>(sum_sq / float(K) + epsilon))[0];
    float l2_sq = 0.0f;
    for (int inner = 0; inner < K; inner += VEC) {
      auto values =
          aie::mul(aie::load_v<VEC>(input + inner),
                   aie::broadcast<bfloat16, VEC>(bfloat16(1.0f)))
              .template to_vector<float>();
      auto scale = aie::mul(aie::load_v<VEC>(weight_and_epsilon + inner),
                            aie::broadcast<float, VEC>(inv_rms))
                       .template to_vector<float>();
      auto normalized = aie::mul(values, scale).template to_vector<float>();
      aie::store_v(temporary + inner, normalized);
      l2_sq += aie::reduce_add(aie::mul(normalized, normalized).to_vector<float>());
    }
    const float inv_l2 =
        aie::invsqrt(aie::broadcast<float, VEC>(l2_sq))[0];
    for (int inner = 0; inner < K; inner += VEC)
      aie::store_v(destination + inner,
                   aie::mul(aie::load_v<VEC>(temporary + inner),
                            aie::broadcast<float, VEC>(inv_l2))
                       .template to_vector<float>());
  }
}
