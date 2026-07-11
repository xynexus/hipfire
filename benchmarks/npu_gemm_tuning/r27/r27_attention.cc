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
constexpr int QUERY_TILE_ELEMS = QUERIES * HEAD_DIM;
constexpr float INV_SQRT_HEAD = 0.0625f;
constexpr float LOG2E = 1.4426950408889634f;

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
  alignas(32) float scores[KEYS];
  alignas(32) bfloat16 weights[KEYS];

  for (int query = 0; query < QUERIES; ++query) {
    const bfloat16 *q = queries + query * HEAD_DIM;
    for (int key = 0; key < KEYS; ++key) {
      const bfloat16 *k = keys + key * HEAD_DIM;
      auto dot = aie::zeros<accfloat, LANES>();
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_MIN_ITERATION_COUNT(16)
      for (int dim = 0; dim < HEAD_DIM; dim += LANES) {
        dot = aie::add(dot, aie::mul(aie::load_v<LANES>(q + dim),
                                     aie::load_v<LANES>(k + dim)));
      }
      scores[key] =
          aie::reduce_add(dot.template to_vector<float>()) * INV_SQRT_HEAD;
    }

    auto score_vector = aie::load_v<KEYS>(scores);
    const float block_max = aie::reduce_max(score_vector);
    const float old_max = stats[query];
    const float old_sum = stats[QUERIES + query];
    const float new_max = old_max > block_max ? old_max : block_max;
    const bfloat16 alpha = old_sum == 0.0f ? (bfloat16)0.0f
                                          : exp_bf16(old_max - new_max);

    auto shifted = aie::sub(score_vector, aie::broadcast<float, KEYS>(new_max));
    auto weight_vector = aie::exp2<bfloat16>(
        aie::mul(shifted, LOG2E).template to_vector<float>());
    aie::store_v(weights, weight_vector);
    const float block_sum = aie::reduce_add(
        aie::mul(weight_vector, aie::broadcast<bfloat16, KEYS>((bfloat16)1.0f))
            .template to_vector<float>());

    float *out = accum + query * HEAD_DIM;
    for (int dim = 0; dim < HEAD_DIM; dim += LANES) {
      auto current = aie::load_v<LANES>(out + dim);
      aie::store_v(
          out + dim,
          aie::mul(current, (float)alpha).template to_vector<float>());
    }
    for (int key = 0; key < KEYS; ++key) {
      const bfloat16 weight = weights[key];
      const bfloat16 *v = values + key * HEAD_DIM;
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_MIN_ITERATION_COUNT(16)
      for (int dim = 0; dim < HEAD_DIM; dim += LANES) {
        auto current = aie::load_v<LANES>(out + dim);
        auto contribution =
            aie::mul(aie::load_v<LANES>(v + dim), weight)
                .template to_vector<float>();
        aie::store_v(out + dim, aie::add(current, contribution));
      }
    }
    stats[query] = new_max;
    stats[QUERIES + query] = old_sum * (float)alpha + block_sum;
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
