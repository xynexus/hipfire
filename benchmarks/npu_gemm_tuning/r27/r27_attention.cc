// SPDX-License-Identifier: Apache-2.0
// Fused online-softmax bidirectional attention tile for EmbeddingGemma R27.

#include <aie_api/aie.hpp>
#include "aie_kernels/aie_kernel_utils.h"
#include <stdint.h>

namespace {
constexpr int QUERIES = 4;
constexpr int KEYS = 16;
constexpr int HEAD_DIM = 256;
constexpr int LANES = 16;
constexpr int MMUL_K = 8;
constexpr int MMUL_N = 8;
constexpr int DIM_TILES = HEAD_DIM / MMUL_K;
constexpr int KEY_TILES = KEYS / MMUL_N;
constexpr int QUERY_TILE_ELEMS = QUERIES * HEAD_DIM;
constexpr float INV_SQRT_HEAD = 0.0625f;
constexpr float LOG2E = 1.4426950408889634f;
using BF16_MMUL = aie::mmul<QUERIES, MMUL_K, MMUL_N, bfloat16, bfloat16>;

__attribute__((noinline)) bfloat16 exp_bf16(float value) {
  auto input = aie::broadcast<float, LANES>(value * LOG2E);
  return aie::exp2<bfloat16>(input)[0];
}
} // namespace

extern "C" {

void r27_attention_init(float *restrict accum, float *restrict stats) {
  for (int query = 0; query < QUERIES; ++query) {
    stats[query] = -3.0e30f;
    stats[QUERIES + query] = 0.0f;
  }
  for (int index = 0; index < QUERIES * HEAD_DIM; index += LANES) {
    aie::store_v(accum + index, aie::zeros<float, LANES>());
  }
}

void r27_attention_block(const bfloat16 *restrict queries,
                         const bfloat16 *restrict key_value,
                         float *restrict accum, float *restrict stats,
                         int32_t pair_lane) {
  queries += pair_lane * QUERY_TILE_ELEMS;
  const bfloat16 *keys = key_value;
  const bfloat16 *values = key_value + KEYS * HEAD_DIM;
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
    const auto lo = scores_lo.template extract<MMUL_N>(query);
    const auto hi = scores_hi.template extract<MMUL_N>(query);
    const float block_max =
        aie::max(aie::reduce_max(lo), aie::reduce_max(hi));
    const float old_max = stats[query];
    const float old_sum = stats[QUERIES + query];
    const float new_max = old_max > block_max ? old_max : block_max;
    alpha[query] = old_sum == 0.0f ? 0.0f : (float)exp_bf16(old_max - new_max);

    auto shifted_lo =
        aie::sub(lo, aie::broadcast<float, MMUL_N>(new_max));
    auto shifted_hi =
        aie::sub(hi, aie::broadcast<float, MMUL_N>(new_max));
    auto weight_lo = aie::exp2<bfloat16>(
        aie::mul(shifted_lo, LOG2E).template to_vector<float>());
    auto weight_hi = aie::exp2<bfloat16>(
        aie::mul(shifted_hi, LOG2E).template to_vector<float>());
    aie::store_v(weights_lo + query * MMUL_K, weight_lo);
    aie::store_v(weights_hi + query * MMUL_K, weight_hi);
    const auto all_weights = aie::concat(weight_lo, weight_hi);
    const float block_sum = aie::reduce_add(
        aie::mul(all_weights, aie::broadcast<bfloat16, KEYS>((bfloat16)1.0f))
            .template to_vector<float>());
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
      auto current = aie::load_v<LANES>(out);
      auto retained =
          aie::mul(current, alpha[query]).template to_vector<float>();
      auto contribution =
          aie::concat(contribution0.template extract<MMUL_N>(query),
                      contribution1.template extract<MMUL_N>(query));
      aie::store_v(out, aie::add(retained, contribution));
    }
  }
}

void r27_attention_finish(const float *restrict accum,
                          const float *restrict stats,
                          bfloat16 *restrict output) {
  for (int query = 0; query < QUERIES; ++query) {
    const float inv_sum = aie::inv(stats[QUERIES + query]);
    const float *input = accum + query * HEAD_DIM;
    bfloat16 *out = output + query * HEAD_DIM;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int dim = 0; dim < HEAD_DIM; dim += LANES) {
      auto normalized =
          aie::mul(aie::load_v<LANES>(input + dim), inv_sum)
              .template to_vector<bfloat16>();
      aie::store_v(out + dim, normalized);
    }
  }
}

} // extern "C"
