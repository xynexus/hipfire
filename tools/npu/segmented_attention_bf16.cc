// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

// Online-softmax causal attention for one Qwen3 AIE2P query tile. The graph
// passes absolute query/key positions plus the document's real length; masked
// keys receive zero probability and padded queries produce an all-zero row.

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef HIPFIRE_HEAD_DIM
#define HIPFIRE_HEAD_DIM 128
#endif

namespace {
constexpr int QUERIES = 4;
constexpr int KEYS = 16;
constexpr int HEAD_DIM = HIPFIRE_HEAD_DIM;
constexpr int LANES = 16;
constexpr int MMUL_K = 8;
constexpr int MMUL_N = 8;
constexpr int DIM_TILES = HEAD_DIM / MMUL_K;
constexpr int KEY_TILES = KEYS / MMUL_N;
constexpr int QUERY_TILE_ELEMS = QUERIES * HEAD_DIM;
constexpr int QUERY_PAIR_ELEMS = 2 * QUERY_TILE_ELEMS;
constexpr float INV_SQRT_HEAD = HEAD_DIM == 128 ? 0.08838834764831845f : 0.0625f;
constexpr float LOG2E = 1.4426950408889634f;
using BF16_MMUL =
    aie::mmul<QUERIES, MMUL_K, MMUL_N, bfloat16, bfloat16>;

__attribute__((noinline)) float exp2_f32(float value) {
  if (value <= -126.0f)
    return 0.0f;
  if (value >= 0.0f)
    return 1.0f;
  int exponent = (int)value;
  if ((float)exponent > value)
    --exponent;
  const float z = (value - (float)exponent) * 0.6931471805599453f;
  const float polynomial =
      1.0f +
      z * (1.0f +
           z * (0.5f +
                z * (0.1666666666666667f +
                     z * (0.0416666666666667f +
                          z * (0.0083333333333333f +
                               z * 0.0013888888888889f)))));
  float scale = 1.0f;
  for (int power = 0; power < -exponent; ++power)
    scale *= 0.5f;
  return scale * polynomial;
}

inline bool visible(int query, int key, int real_length, int causal,
                    int sliding_window) {
  if (query < 0 || query >= real_length || key < 0 || key >= real_length)
    return false;
  if (causal != 0 && key > query)
    return false;
  if (sliding_window > 0 &&
      (key < query - sliding_window || key > query + sliding_window))
    return false;
  return true;
}
} // namespace

extern "C" {

void hipfire_segmented_attention_init(float *restrict accum,
                                      float *restrict stats) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int query = 0; query < QUERIES; ++query) {
    stats[query] = -3.0e30f;
    stats[QUERIES + query] = 0.0f;
  }
  for (int index = 0; index < QUERIES * HEAD_DIM; index += LANES)
    aie::store_v(accum + index, aie::zeros<float, LANES>());
}

void hipfire_segmented_attention_block(
    const bfloat16 *restrict queries, const bfloat16 *restrict key_value,
    float *restrict accum, float *restrict stats, int32_t pair_lane,
    int32_t query_base, int32_t key_base, int32_t causal,
    int32_t sliding_window) {
  const int32_t real_length =
      *reinterpret_cast<const int32_t *>(queries + QUERY_PAIR_ELEMS);
  queries += pair_lane * QUERY_TILE_ELEMS;
  const bfloat16 *keys = key_value;
  const bfloat16 *values = key_value + KEYS * HEAD_DIM;
  alignas(32) float scores[QUERIES * KEYS];
  alignas(32) bfloat16 weights_lo[QUERIES * MMUL_K];
  alignas(32) bfloat16 weights_hi[QUERIES * MMUL_K];
  alignas(32) float alpha[QUERIES];

  BF16_MMUL score_lo;
  BF16_MMUL score_hi;
  for (int dim_tile = 0; dim_tile < DIM_TILES; ++dim_tile) {
    const auto q = aie::load_v<BF16_MMUL::size_A>(
        queries + dim_tile * BF16_MMUL::size_A);
    const auto k_lo = aie::load_v<BF16_MMUL::size_B>(
        keys + (0 * DIM_TILES + dim_tile) * BF16_MMUL::size_B);
    const auto k_hi = aie::load_v<BF16_MMUL::size_B>(
        keys + (1 * DIM_TILES + dim_tile) * BF16_MMUL::size_B);
    if (dim_tile == 0) {
      score_lo.mul(q, k_lo);
      score_hi.mul(q, k_hi);
    } else {
      score_lo.mac(q, k_lo);
      score_hi.mac(q, k_hi);
    }
  }
  auto scores_lo = aie::mul(score_lo.to_vector<float>(), INV_SQRT_HEAD)
                       .template to_vector<float>();
  auto scores_hi = aie::mul(score_hi.to_vector<float>(), INV_SQRT_HEAD)
                       .template to_vector<float>();
  for (int query = 0; query < QUERIES; ++query) {
    aie::store_v(scores + query * KEYS,
                 scores_lo.template extract<MMUL_N>(query));
    aie::store_v(scores + query * KEYS + MMUL_N,
                 scores_hi.template extract<MMUL_N>(query));
  }

  for (int query = 0; query < QUERIES; ++query) {
    const int absolute_query = query_base + query;
    float block_max = -3.0e30f;
    bool any_visible = false;
    for (int key = 0; key < KEYS; ++key) {
      if (visible(absolute_query, key_base + key, real_length, causal,
                  sliding_window)) {
        block_max = scores[query * KEYS + key] > block_max
                        ? scores[query * KEYS + key]
                        : block_max;
        any_visible = true;
      } else {
        scores[query * KEYS + key] = -3.0e30f;
      }
    }
    if (!any_visible) {
      alpha[query] = 1.0f;
      for (int key = 0; key < MMUL_K; ++key) {
        weights_lo[query * MMUL_K + key] = (bfloat16)0.0f;
        weights_hi[query * MMUL_K + key] = (bfloat16)0.0f;
      }
      continue;
    }

    const float old_max = stats[query];
    const float old_sum = stats[QUERIES + query];
    const float new_max = old_max > block_max ? old_max : block_max;
    alpha[query] = old_sum == 0.0f ? 0.0f : exp2_f32((old_max - new_max) * LOG2E);
    float block_sum = 0.0f;
    for (int key = 0; key < KEYS; ++key) {
      const float weight = exp2_f32((scores[query * KEYS + key] - new_max) * LOG2E);
      block_sum += weight;
      if (key < MMUL_K) {
        weights_lo[query * MMUL_K + key] = (bfloat16)weight;
      } else {
        weights_hi[query * MMUL_K + key - MMUL_K] = (bfloat16)weight;
      }
    }
    for (int key = 0; key < KEYS; ++key) {
      if (!visible(absolute_query, key_base + key, real_length, causal,
                   sliding_window)) {
        if (key < MMUL_K) {
          weights_lo[query * MMUL_K + key] = (bfloat16)0.0f;
        } else {
          weights_hi[query * MMUL_K + key - MMUL_K] = (bfloat16)0.0f;
        }
      }
    }
    stats[query] = new_max;
    stats[QUERIES + query] = old_sum * alpha[query] + block_sum;
  }

  const auto weight_vector_lo =
      aie::load_v<BF16_MMUL::size_A>(weights_lo);
  const auto weight_vector_hi =
      aie::load_v<BF16_MMUL::size_A>(weights_hi);
  for (int dim_tile = 0; dim_tile < DIM_TILES; dim_tile += 2) {
    BF16_MMUL pv0;
    BF16_MMUL pv1;
    pv0.mul(weight_vector_lo,
            aie::load_v<BF16_MMUL::size_B>(
                values + (dim_tile * KEY_TILES + 0) * BF16_MMUL::size_B));
    pv0.mac(weight_vector_hi,
            aie::load_v<BF16_MMUL::size_B>(
                values + (dim_tile * KEY_TILES + 1) * BF16_MMUL::size_B));
    pv1.mul(weight_vector_lo,
            aie::load_v<BF16_MMUL::size_B>(
                values + ((dim_tile + 1) * KEY_TILES + 0) *
                             BF16_MMUL::size_B));
    pv1.mac(weight_vector_hi,
            aie::load_v<BF16_MMUL::size_B>(
                values + ((dim_tile + 1) * KEY_TILES + 1) *
                             BF16_MMUL::size_B));
    auto contribution0 = pv0.to_vector<float>();
    auto contribution1 = pv1.to_vector<float>();
    for (int query = 0; query < QUERIES; ++query) {
      float *out = accum + query * HEAD_DIM + dim_tile * MMUL_N;
      auto retained = aie::mul(aie::load_v<LANES>(out), alpha[query])
                          .template to_vector<float>();
      auto contribution =
          aie::concat(contribution0.template extract<MMUL_N>(query),
                      contribution1.template extract<MMUL_N>(query));
      aie::store_v(out, aie::add(retained, contribution));
    }
  }
}

void hipfire_segmented_attention_finish(const float *restrict accum,
                                        const float *restrict stats,
                                        bfloat16 *restrict output) {
  for (int query = 0; query < QUERIES; ++query) {
    const float sum = stats[QUERIES + query];
    const float inv_sum = sum > 0.0f ? aie::inv(sum) : 0.0f;
    const float *input = accum + query * HEAD_DIM;
    bfloat16 *out = output + query * HEAD_DIM;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (int dim = 0; dim < HEAD_DIM; dim += LANES) {
      auto normalized = aie::mul(aie::load_v<LANES>(input + dim), inv_sum)
                            .template to_vector<bfloat16>();
      aie::store_v(out + dim, normalized);
    }
  }
}

} // extern "C"
