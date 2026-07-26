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
}

extern "C" __attribute__((noinline, minsize)) void
hipfire_qwen3_residual_rmsnorm(const int8_t *restrict input_pair,
                              const float *restrict weight_and_epsilon,
                              int8_t *restrict output, int32_t pair_lane) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr int tensor_bytes = K * sizeof(bfloat16);
  constexpr int record_bytes = 2 * tensor_bytes;
  input_pair += pair_lane * record_bytes;
  const auto *residual = reinterpret_cast<const bfloat16 *>(input_pair);
  const auto *delta = residual + K;
  auto *completed = reinterpret_cast<bfloat16 *>(output);
  auto *normalized = completed + K;

  float sum_sq = 0.0f;
  for (int inner = 0; inner < K; inner += VEC) {
    auto hidden = aie::add(aie::load_v<VEC>(residual + inner),
                           aie::load_v<VEC>(delta + inner));
    aie::store_v(completed + inner, hidden);
    sum_sq += aie::reduce_add(aie::mul(hidden, hidden).to_vector<float>());
  }
  const float epsilon = weight_and_epsilon[K];
  const float inv_rms =
      aie::invsqrt(aie::broadcast<float, VEC>(sum_sq / float(K) + epsilon))[0];
  for (int inner = 0; inner < K; inner += VEC) {
    auto hidden = aie::load_v<VEC>(completed + inner);
    auto hidden_f32 =
        aie::mul(hidden, aie::broadcast<bfloat16, VEC>(bfloat16(1.0f)))
            .template to_vector<float>();
    auto scale = aie::mul(aie::load_v<VEC>(weight_and_epsilon + inner),
                          aie::broadcast<float, VEC>(inv_rms))
                     .template to_vector<float>();
    aie::store_v(normalized + inner,
                 aie::mul(hidden_f32, scale).template to_vector<bfloat16>());
  }
}
