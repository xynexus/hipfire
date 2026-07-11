// SPDX-License-Identifier: Apache-2.0
// Fused EmbeddingGemma Q/K headnorm + RoPE and direct R27 MMUL packing.

#include <aie_api/aie.hpp>
#include "aie_kernels/aie_kernel_utils.h"
#include <stdint.h>

namespace {
constexpr int HEAD_DIM = 256;
constexpr int HALF = HEAD_DIM / 2;
constexpr int VEC = 16;
constexpr int Q_ROWS = 4;
constexpr int KV_ROWS = 8;
constexpr int Q_TILE_ELEMS = Q_ROWS * HEAD_DIM;
constexpr int PARAM_QNORM = 0;
constexpr int PARAM_KNORM = 512;
constexpr int PARAM_EPS = 1024;
constexpr int RAW_PAIR_ELEMS = 2 * Q_TILE_ELEMS;
constexpr int CS_ROW_ELEMS = HEAD_DIM;

float inverse_rms(const bfloat16 *input, float epsilon) {
  float sum = 0.0f;
  for (int dim = 0; dim < HEAD_DIM; dim += VEC) {
    const auto value = aie::load_v<VEC>(input + dim);
    const aie::vector<float, VEC> squared =
        aie::mul(value, value).to_vector<float>();
    sum += aie::reduce_add(squared);
  }
  return aie::invsqrt(sum / (float)HEAD_DIM + epsilon);
}

aie::vector<bfloat16, VEC>
rotate_half(aie::vector<bfloat16, VEC> input,
            aie::vector<bfloat16, VEC> paired,
            aie::vector<bfloat16, VEC> weight,
            aie::vector<bfloat16, VEC> paired_weight,
            aie::vector<bfloat16, VEC> cosine,
            aie::vector<bfloat16, VEC> sine, float inverse_rms, bool upper) {
  aie::vector<float, VEC> normalized =
      aie::mul(input, weight).to_vector<float>();
  normalized = aie::mul(normalized, inverse_rms).to_vector<float>();
  aie::vector<float, VEC> paired_normalized =
      aie::mul(paired, paired_weight).to_vector<float>();
  paired_normalized =
      aie::mul(paired_normalized, inverse_rms).to_vector<float>();
  aie::vector<float, VEC> cosine_f =
      aie::mul(cosine, aie::broadcast<bfloat16, VEC>((bfloat16)1.0f))
          .to_vector<float>();
  aie::vector<float, VEC> sine_f =
      aie::mul(sine, aie::broadcast<bfloat16, VEC>((bfloat16)1.0f))
          .to_vector<float>();
  if (upper) {
    aie::vector<float, VEC> result =
        aie::add(aie::mul(normalized, cosine_f).to_vector<float>(),
                 aie::mul(paired_normalized, sine_f).to_vector<float>());
    return aie::mul(result, 1.0f).to_vector<bfloat16>();
  }
  aie::vector<float, VEC> result =
      aie::sub(aie::mul(normalized, cosine_f).to_vector<float>(),
               aie::mul(paired_normalized, sine_f).to_vector<float>());
  return aie::mul(result, 1.0f).to_vector<bfloat16>();
}

__attribute__((noinline)) void q_lower_store(
    const bfloat16 *restrict input, const bfloat16 *restrict weight,
    const bfloat16 *restrict cs, bfloat16 *restrict packed, float inverse_rms,
    int32_t dim, int32_t row) {
  const auto rotated = rotate_half(
      aie::load_v<VEC>(input + dim), aie::load_v<VEC>(input + HALF + dim),
      aie::load_v<VEC>(weight + dim),
      aie::load_v<VEC>(weight + HALF + dim), aie::load_v<VEC>(cs + dim),
      aie::load_v<VEC>(cs + HALF + dim), inverse_rms, false);
  aie::store_v(packed + (dim / 8) * 32 + row * 8,
               rotated.extract<8>(0));
  aie::store_v(packed + (dim / 8 + 1) * 32 + row * 8,
               rotated.extract<8>(1));
}

__attribute__((noinline)) void q_upper_store(
    const bfloat16 *restrict input, const bfloat16 *restrict weight,
    const bfloat16 *restrict cs, bfloat16 *restrict packed, float inverse_rms,
    int32_t dim, int32_t row) {
  const auto rotated = rotate_half(
      aie::load_v<VEC>(input + HALF + dim), aie::load_v<VEC>(input + dim),
      aie::load_v<VEC>(weight + HALF + dim),
      aie::load_v<VEC>(weight + dim), aie::load_v<VEC>(cs + dim),
      aie::load_v<VEC>(cs + HALF + dim), inverse_rms, true);
  const int32_t upper = dim + HALF;
  aie::store_v(packed + (upper / 8) * 32 + row * 8,
               rotated.extract<8>(0));
  aie::store_v(packed + (upper / 8 + 1) * 32 + row * 8,
               rotated.extract<8>(1));
}

const bfloat16 *qnorm(const int8_t *params) {
  return reinterpret_cast<const bfloat16 *>(params + PARAM_QNORM);
}
const bfloat16 *knorm(const int8_t *params) {
  return reinterpret_cast<const bfloat16 *>(params + PARAM_KNORM);
}
float epsilon(const int8_t *params) {
  return *reinterpret_cast<const float *>(params + PARAM_EPS);
}
} // namespace

extern "C" {

void r28_pack_q(const int8_t *restrict raw_pair,
                const int8_t *restrict params,
                int8_t *restrict packed_bytes, int32_t pair_lane) {
  const bfloat16 *pair = reinterpret_cast<const bfloat16 *>(raw_pair);
  const bfloat16 *raw = pair + pair_lane * Q_TILE_ELEMS;
  const bfloat16 *cs = pair + RAW_PAIR_ELEMS +
                       pair_lane * Q_ROWS * CS_ROW_ELEMS;
  bfloat16 *packed = reinterpret_cast<bfloat16 *>(packed_bytes);
  const bfloat16 *weight = qnorm(params);
  const float eps = epsilon(params);
  for (int row = 0; row < Q_ROWS; ++row) {
    const bfloat16 *input = raw + row * HEAD_DIM;
    const float inv = inverse_rms(input, eps);
    const bfloat16 *row_cs = cs + row * CS_ROW_ELEMS;
    for (int dim = 0; dim < HALF; dim += VEC) {
      q_lower_store(input, weight, row_cs, packed, inv, dim, row);
      q_upper_store(input, weight, row_cs, packed, inv, dim, row);
    }
  }
}

void r28_pack_k(const int8_t *restrict raw_pair,
                const int8_t *restrict params, int8_t *restrict packed_bytes,
                float *restrict inverse_rms_rows, int32_t upper_half) {
  const bfloat16 *raw = reinterpret_cast<const bfloat16 *>(raw_pair);
  const bfloat16 *cs = raw + RAW_PAIR_ELEMS;
  bfloat16 *packed = reinterpret_cast<bfloat16 *>(packed_bytes);
  const bfloat16 *weight = knorm(params);
  const float eps = epsilon(params);
  if (upper_half == 0)
    for (int row = 0; row < KV_ROWS; ++row)
      inverse_rms_rows[row] = inverse_rms(raw + row * HEAD_DIM, eps);

  int output = 0;
  alignas(32) bfloat16 rotated_rows[KV_ROWS * VEC];
  for (int dim = 0; dim < HALF; dim += VEC) {
    for (int row = 0; row < KV_ROWS; ++row) {
      const bfloat16 *input = raw + row * HEAD_DIM;
      const auto x = aie::load_v<VEC>(input + dim);
      const auto y = aie::load_v<VEC>(input + dim + HALF);
      const auto wx = aie::load_v<VEC>(weight + dim);
      const auto wy = aie::load_v<VEC>(weight + dim + HALF);
      const bfloat16 *row_cs = cs + row * CS_ROW_ELEMS;
      const auto cosine = aie::load_v<VEC>(row_cs + dim);
      const auto sine = aie::load_v<VEC>(row_cs + HALF + dim);
      const auto rotated = upper_half
                               ? rotate_half(y, x, wy, wx, cosine, sine,
                                             inverse_rms_rows[row], true)
                               : rotate_half(x, y, wx, wy, cosine, sine,
                                             inverse_rms_rows[row], false);
      aie::store_v(rotated_rows + row * VEC, rotated);
    }
    aie::vector<bfloat16, 64> tile;
    for (int row = 0; row < KV_ROWS; ++row)
      tile.insert(row, aie::load_v<8>(rotated_rows + row * VEC));
    aie::store_v(packed + output, aie::transpose(tile, 8, 8));
    output += 64;
    for (int row = 0; row < KV_ROWS; ++row)
      tile.insert(row, aie::load_v<8>(rotated_rows + row * VEC + 8));
    aie::store_v(packed + output, aie::transpose(tile, 8, 8));
    output += 64;
  }
}

void r28_pack_v(const int8_t *restrict raw_pair,
                int8_t *restrict packed_bytes, int32_t upper_half) {
  const bfloat16 *raw = reinterpret_cast<const bfloat16 *>(raw_pair);
  bfloat16 *packed = reinterpret_cast<bfloat16 *>(packed_bytes);
  int output = 0;
  const int base = upper_half * HALF;
  for (int dim = 0; dim < HALF; dim += VEC) {
    aie::vector<bfloat16, 64> tile0;
    aie::vector<bfloat16, 64> tile1;
    for (int row = 0; row < KV_ROWS; ++row) {
      const auto value = aie::load_v<VEC>(raw + row * HEAD_DIM + base + dim);
      tile0.insert(row, value.extract<8>(0));
      tile1.insert(row, value.extract<8>(1));
    }
    aie::store_v(packed + output, tile0);
    output += 64;
    aie::store_v(packed + output, tile1);
    output += 64;
  }
}

} // extern "C"
