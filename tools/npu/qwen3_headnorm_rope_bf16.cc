// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef HIPFIRE_HEAD_DIM
#define HIPFIRE_HEAD_DIM 128
#endif

namespace {
constexpr int D = HIPFIRE_HEAD_DIM;
constexpr int HALF = D / 2;
constexpr int VEC = 16;
constexpr int HEAD_BYTES = D * sizeof(bfloat16);
}

extern "C" __attribute__((noinline, minsize)) void
hipfire_qwen3_headnorm_rope(const int8_t *restrict input_pair,
                           const int8_t *restrict parameter_bytes,
                           int8_t *restrict output_bytes, int32_t pair_lane) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const auto *input = reinterpret_cast<const bfloat16 *>(
      input_pair + pair_lane * HEAD_BYTES);
  auto *output = reinterpret_cast<bfloat16 *>(output_bytes + pair_lane * HEAD_BYTES);
  const auto *weight = reinterpret_cast<const bfloat16 *>(parameter_bytes);
  const auto *cosine = weight + D;
  const auto *sine = cosine + HALF;
  const float epsilon =
      *reinterpret_cast<const float *>(parameter_bytes + 2 * D * sizeof(bfloat16));

  float sum_sq = 0.0f;
  for (int inner = 0; inner < D; inner += VEC) {
    auto values = aie::load_v<VEC>(input + inner);
    sum_sq += aie::reduce_add(aie::mul(values, values).to_vector<float>());
  }
  const float inv_rms =
      aie::invsqrt(aie::broadcast<float, VEC>(sum_sq / float(D) + epsilon))[0];
  const auto inv = aie::broadcast<float, VEC>(inv_rms);
  const auto one = aie::broadcast<bfloat16, VEC>(bfloat16(1.0f));
  for (int inner = 0; inner < HALF; inner += VEC) {
    auto x = aie::mul(
                 aie::mul(aie::load_v<VEC>(input + inner), one)
                     .template to_vector<float>(),
                 inv)
                 .template to_vector<bfloat16>();
    auto y = aie::mul(
                 aie::mul(aie::load_v<VEC>(input + HALF + inner), one)
                     .template to_vector<float>(),
                 inv)
                 .template to_vector<bfloat16>();
    x = aie::mul(x, aie::load_v<VEC>(weight + inner))
            .template to_vector<bfloat16>();
    y = aie::mul(y, aie::load_v<VEC>(weight + HALF + inner))
            .template to_vector<bfloat16>();
    const auto c = aie::load_v<VEC>(cosine + inner);
    const auto s = aie::load_v<VEC>(sine + inner);
    aie::store_v(output + inner,
                 aie::sub(aie::mul(x, c), aie::mul(y, s))
                     .template to_vector<bfloat16>());
    aie::store_v(output + HALF + inner,
                 aie::add(aie::mul(y, c), aie::mul(x, s))
                     .template to_vector<bfloat16>());
  }
}
