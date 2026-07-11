// SPDX-License-Identifier: Apache-2.0
// Resident W8 QKV projection followed by headnorm/RoPE and R27 packing.

#include <aie_api/aie.hpp>
#include "aie_kernels/aie_kernel_utils.h"
#include <stdint.h>

namespace {
constexpr int VEC = 16;

// R15 W8 scaled full-K projection geometry.
constexpr int LM = 3;
constexpr int LN = 2;
constexpr int KT = 32;
using W8_MMUL = aie::mmul<8, 8, 8, int8, int8>;
constexpr int SA = W8_MMUL::size_A;
constexpr int SB = 2 * W8_MMUL::size_B;
constexpr int SC = 2 * W8_MMUL::size_C;
constexpr int A_DATA = LM * KT * SA;
constexpr int W_DATA = LN * KT * SB;
constexpr int ACC_ELEMS = LM * LN * SC;

// Five exact 256-column roles: Q0, Q1, Q2, K, V.
constexpr int HEAD_DIM = 256;
constexpr int HALF = HEAD_DIM / 2;
constexpr int Q_ROWS = 4;
constexpr int KV_ROWS = 8;
constexpr int RAW_ROWS = 8;
constexpr int RAW_ELEMS = RAW_ROWS * HEAD_DIM;
constexpr int CS_ELEMS = RAW_ROWS * HEAD_DIM;
constexpr int PARAM_QNORM = 0;
constexpr int PARAM_KNORM = 512;
constexpr int PARAM_EPS = 1024;
constexpr int PARAM_OFFSET = (RAW_ELEMS + CS_ELEMS) * 2;

static inline aie::vector<int32, 16>
join_rows(aie::vector<int32, W8_MMUL::size_C> lo,
          aie::vector<int32, W8_MMUL::size_C> hi, int row) {
  switch (row) {
  case 0: return aie::concat(lo.extract<8>(0), hi.extract<8>(0));
  case 1: return aie::concat(lo.extract<8>(1), hi.extract<8>(1));
  case 2: return aie::concat(lo.extract<8>(2), hi.extract<8>(2));
  case 3: return aie::concat(lo.extract<8>(3), hi.extract<8>(3));
  case 4: return aie::concat(lo.extract<8>(4), hi.extract<8>(4));
  case 5: return aie::concat(lo.extract<8>(5), hi.extract<8>(5));
  case 6: return aie::concat(lo.extract<8>(6), hi.extract<8>(6));
  default: return aie::concat(lo.extract<8>(7), hi.extract<8>(7));
  }
}

template <bool ACCUMULATE>
void projection_group(const int8_t *restrict activations,
                      const int8_t *restrict weights,
                      float *restrict output) {
  const float *activation_scales =
      reinterpret_cast<const float *>(activations + A_DATA);
  const float *weight_scales =
      reinterpret_cast<const float *>(weights + W_DATA);
  for (int im = 0; im < LM; ++im)
    for (int jn = 0; jn < LN; ++jn) {
      W8_MMUL lo, hi;
      auto a = aie::load_v<SA>(activations + (im * KT) * SA);
      const int8_t *w = weights + (jn * KT) * SB;
      lo.mul(a, aie::load_v<W8_MMUL::size_B>(w));
      hi.mul(a, aie::load_v<W8_MMUL::size_B>(w + W8_MMUL::size_B));
      for (int k = 1; k < KT; ++k) {
        a = aie::load_v<SA>(activations + (im * KT + k) * SA);
        w = weights + (jn * KT + k) * SB;
        lo.mac(a, aie::load_v<W8_MMUL::size_B>(w));
        hi.mac(a, aie::load_v<W8_MMUL::size_B>(w + W8_MMUL::size_B));
      }
      const auto vlo = lo.to_vector<int32>();
      const auto vhi = hi.to_vector<int32>();
      const auto weight_scale = aie::load_v<16>(weight_scales + jn * 16);
      for (int row = 0; row < 8; ++row) {
        const int offset = (im * LN + jn) * SC + row * 16;
        auto scaled =
            aie::mul(aie::to_float(join_rows(vlo, vhi, row)), weight_scale)
                .to_vector<float>();
        scaled =
            aie::mul(scaled, activation_scales[im * 8 + row]).to_vector<float>();
        if constexpr (ACCUMULATE)
          scaled = aie::add(scaled, aie::load_v<16>(output + offset));
        aie::store_v(output + offset, scaled);
      }
    }
}

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
  const aie::vector<float, VEC> cosine_f =
      aie::mul(cosine, aie::broadcast<bfloat16, VEC>((bfloat16)1.0f))
          .to_vector<float>();
  const aie::vector<float, VEC> sine_f =
      aie::mul(sine, aie::broadcast<bfloat16, VEC>((bfloat16)1.0f))
          .to_vector<float>();
  aie::vector<float, VEC> result =
      upper ? aie::add(aie::mul(normalized, cosine_f).to_vector<float>(),
                       aie::mul(paired_normalized, sine_f).to_vector<float>())
            : aie::sub(aie::mul(normalized, cosine_f).to_vector<float>(),
                       aie::mul(paired_normalized, sine_f).to_vector<float>());
  return aie::mul(result, 1.0f).to_vector<bfloat16>();
}

const bfloat16 *raw_values(const int8_t *pair) {
  return reinterpret_cast<const bfloat16 *>(pair);
}
const bfloat16 *raw_cs(const int8_t *pair) {
  return raw_values(pair) + RAW_ELEMS;
}
const int8_t *raw_params(const int8_t *pair) { return pair + PARAM_OFFSET; }
const bfloat16 *qnorm(const int8_t *pair) {
  return reinterpret_cast<const bfloat16 *>(raw_params(pair) + PARAM_QNORM);
}
const bfloat16 *knorm(const int8_t *pair) {
  return reinterpret_cast<const bfloat16 *>(raw_params(pair) + PARAM_KNORM);
}
float epsilon(const int8_t *pair) {
  return *reinterpret_cast<const float *>(raw_params(pair) + PARAM_EPS);
}

__attribute__((noinline)) void q_lower_store(
    const bfloat16 *restrict input, const bfloat16 *restrict weight,
    const bfloat16 *restrict cs, bfloat16 *restrict packed, float inv,
    int32_t dim, int32_t row) {
  const auto value = rotate_half(
      aie::load_v<VEC>(input + dim), aie::load_v<VEC>(input + HALF + dim),
      aie::load_v<VEC>(weight + dim),
      aie::load_v<VEC>(weight + HALF + dim), aie::load_v<VEC>(cs + dim),
      aie::load_v<VEC>(cs + HALF + dim), inv, false);
  aie::store_v(packed + (dim / 8) * 32 + row * 8, value.extract<8>(0));
  aie::store_v(packed + (dim / 8 + 1) * 32 + row * 8,
               value.extract<8>(1));
}

__attribute__((noinline)) void q_upper_store(
    const bfloat16 *restrict input, const bfloat16 *restrict weight,
    const bfloat16 *restrict cs, bfloat16 *restrict packed, float inv,
    int32_t dim, int32_t row) {
  const auto value = rotate_half(
      aie::load_v<VEC>(input + HALF + dim), aie::load_v<VEC>(input + dim),
      aie::load_v<VEC>(weight + HALF + dim),
      aie::load_v<VEC>(weight + dim), aie::load_v<VEC>(cs + dim),
      aie::load_v<VEC>(cs + HALF + dim), inv, true);
  const int upper = dim + HALF;
  aie::store_v(packed + (upper / 8) * 32 + row * 8,
               value.extract<8>(0));
  aie::store_v(packed + (upper / 8 + 1) * 32 + row * 8,
               value.extract<8>(1));
}

__attribute__((noinline)) void k_lower_store(
    const bfloat16 *restrict input, const bfloat16 *restrict weight,
    const bfloat16 *restrict cs, bfloat16 *restrict output, float inv,
    int32_t dim) {
  const auto value = rotate_half(
      aie::load_v<VEC>(input + dim), aie::load_v<VEC>(input + HALF + dim),
      aie::load_v<VEC>(weight + dim),
      aie::load_v<VEC>(weight + HALF + dim), aie::load_v<VEC>(cs + dim),
      aie::load_v<VEC>(cs + HALF + dim), inv, false);
  aie::store_v(output, value);
}

__attribute__((noinline)) void k_upper_store(
    const bfloat16 *restrict input, const bfloat16 *restrict weight,
    const bfloat16 *restrict cs, bfloat16 *restrict output, float inv,
    int32_t dim) {
  const auto value = rotate_half(
      aie::load_v<VEC>(input + HALF + dim), aie::load_v<VEC>(input + dim),
      aie::load_v<VEC>(weight + HALF + dim),
      aie::load_v<VEC>(weight + dim), aie::load_v<VEC>(cs + dim),
      aie::load_v<VEC>(cs + HALF + dim), inv, true);
  aie::store_v(output, value);
}
} // namespace

extern "C" {

void r29_w8_projection_init(const int8_t *a, const int8_t *w, float *acc) {
  projection_group<false>(a, w, acc);
}
void r29_w8_projection_accum(const int8_t *a, const int8_t *w, float *acc) {
  projection_group<true>(a, w, acc);
}
void r29_w8_projection_finish(const float *acc, int8_t *output_bytes) {
  bfloat16 *output = reinterpret_cast<bfloat16 *>(output_bytes);
  for (int im = 0; im < LM; ++im)
    for (int row = 0; row < 8; ++row)
      for (int jn = 0; jn < LN; ++jn) {
        const int source = (im * LN + jn) * SC + row * VEC;
        const int target = (im * 8 + row) * 32 + jn * VEC;
        const auto value =
            aie::mul(aie::load_v<VEC>(acc + source), 1.0f)
                .to_vector<bfloat16>();
        aie::store_v(output + target, value);
      }
  for (int index = 24 * 32; index < 32 * 32; index += VEC)
    aie::store_v(output + index, aie::zeros<bfloat16, VEC>());
}

void r29_pack_q(const int8_t *restrict pair, int8_t *restrict packed_bytes,
                int32_t pair_lane) {
  const bfloat16 *raw = raw_values(pair) + pair_lane * Q_ROWS * HEAD_DIM;
  const bfloat16 *cs = raw_cs(pair) + pair_lane * Q_ROWS * HEAD_DIM;
  bfloat16 *packed = reinterpret_cast<bfloat16 *>(packed_bytes);
  const bfloat16 *weight = qnorm(pair);
  const float eps = epsilon(pair);
  for (int row = 0; row < Q_ROWS; ++row) {
    const bfloat16 *input = raw + row * HEAD_DIM;
    const bfloat16 *row_cs = cs + row * HEAD_DIM;
    const float inv = inverse_rms(input, eps);
    for (int dim = 0; dim < HALF; dim += VEC) {
      q_lower_store(input, weight, row_cs, packed, inv, dim, row);
      q_upper_store(input, weight, row_cs, packed, inv, dim, row);
    }
  }
}

void r29_pack_k(const int8_t *restrict pair, int8_t *restrict packed_bytes,
                float *restrict inverse_rows, int32_t upper_half) {
  const bfloat16 *raw = raw_values(pair);
  const bfloat16 *cs = raw_cs(pair);
  const bfloat16 *weight = knorm(pair);
  const float eps = epsilon(pair);
  bfloat16 *packed = reinterpret_cast<bfloat16 *>(packed_bytes);
  for (int row = 0; row < KV_ROWS; ++row)
    inverse_rows[row] = inverse_rms(raw + row * HEAD_DIM, eps);
  alignas(32) bfloat16 rotated_rows[KV_ROWS * VEC];
  int output = 0;
  for (int dim = 0; dim < HALF; dim += VEC) {
    for (int row = 0; row < KV_ROWS; ++row) {
      const bfloat16 *input = raw + row * HEAD_DIM;
      const bfloat16 *row_cs = cs + row * HEAD_DIM;
      if (upper_half)
        k_upper_store(input, weight, row_cs, rotated_rows + row * VEC,
                      inverse_rows[row], dim);
      else
        k_lower_store(input, weight, row_cs, rotated_rows + row * VEC,
                      inverse_rows[row], dim);
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

void r29_pack_v(const int8_t *restrict pair, int8_t *restrict packed_bytes,
                int32_t upper_half) {
  const bfloat16 *raw = raw_values(pair);
  bfloat16 *packed = reinterpret_cast<bfloat16 *>(packed_bytes);
  int output = 0;
  const int base = upper_half * HALF;
  for (int dim = 0; dim < HALF; dim += VEC) {
    aie::vector<bfloat16, 64> tile0;
    aie::vector<bfloat16, 64> tile1;
    for (int row = 0; row < KV_ROWS; ++row) {
      const auto value =
          aie::load_v<VEC>(raw + row * HEAD_DIM + base + dim);
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
