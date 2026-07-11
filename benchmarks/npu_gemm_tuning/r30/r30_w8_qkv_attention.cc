// SPDX-License-Identifier: Apache-2.0
// R29 resident W8 QKV pack followed by R27 BF16 bidirectional attention.

#include "../r29/r29_w8_qkv_attention_pack.cc"

namespace {
constexpr int ATTN_QUERIES = 4;
constexpr int ATTN_KEYS = 16;
constexpr int ATTN_DIM = 256;
constexpr int ATTN_MMUL_K = 8;
constexpr int ATTN_MMUL_N = 8;
constexpr int ATTN_DIM_TILES = ATTN_DIM / ATTN_MMUL_K;
constexpr int ATTN_KEY_TILES = ATTN_KEYS / ATTN_MMUL_N;
constexpr int ATTN_QUERY_ELEMS = ATTN_QUERIES * ATTN_DIM;
constexpr float ATTN_SCALE = 0.0625f;
constexpr float LOG2E = 1.4426950408889634f;
using ATTN_MMUL =
    aie::mmul<ATTN_QUERIES, ATTN_MMUL_K, ATTN_MMUL_N, bfloat16,
              bfloat16>;

__attribute__((noinline)) bfloat16 exp_bf16(float value) {
  return aie::exp2<bfloat16>(aie::broadcast<float, 16>(value * LOG2E))[0];
}
} // namespace

extern "C" {

void r30_attention_load_q(const int8_t *restrict q_join,
                          int8_t *restrict q_pair, int32_t pair_col) {
  const bfloat16 *source =
      reinterpret_cast<const bfloat16 *>(q_join) + pair_col * 2 * ATTN_QUERY_ELEMS;
  bfloat16 *target = reinterpret_cast<bfloat16 *>(q_pair);
  for (int index = 0; index < 2 * ATTN_QUERY_ELEMS; index += 16)
    aie::store_v(target + index, aie::load_v<16>(source + index));
}

void r30_attention_init(float *restrict accum, float *restrict stats) {
  for (int query = 0; query < ATTN_QUERIES; ++query) {
    stats[query] = -3.0e30f;
    stats[ATTN_QUERIES + query] = 0.0f;
  }
  for (int index = 0; index < ATTN_QUERIES * ATTN_DIM; index += 16)
    aie::store_v(accum + index, aie::zeros<float, 16>());
}

void r30_attention_block(const int8_t *restrict query_bytes,
                         const int8_t *restrict key_value_bytes,
                         float *restrict accum, float *restrict stats,
                         int32_t pair_lane) {
  const bfloat16 *queries = reinterpret_cast<const bfloat16 *>(query_bytes) +
                            pair_lane * ATTN_QUERY_ELEMS;
  const bfloat16 *keys =
      reinterpret_cast<const bfloat16 *>(key_value_bytes);
  const bfloat16 *values = keys + ATTN_KEYS * ATTN_DIM;
  alignas(32) bfloat16 weights_lo[ATTN_QUERIES * ATTN_MMUL_K];
  alignas(32) bfloat16 weights_hi[ATTN_QUERIES * ATTN_MMUL_K];
  alignas(32) float alpha[ATTN_QUERIES];

  ATTN_MMUL score_lo;
  ATTN_MMUL score_hi;
  for (int tile = 0; tile < ATTN_DIM_TILES; ++tile) {
    const auto q = aie::load_v<ATTN_MMUL::size_A>(
        queries + tile * ATTN_MMUL::size_A);
    const auto k_lo = aie::load_v<ATTN_MMUL::size_B>(
        keys + (0 * ATTN_DIM_TILES + tile) * ATTN_MMUL::size_B);
    const auto k_hi = aie::load_v<ATTN_MMUL::size_B>(
        keys + (1 * ATTN_DIM_TILES + tile) * ATTN_MMUL::size_B);
    if (tile == 0) {
      score_lo.mul(q, k_lo);
      score_hi.mul(q, k_hi);
    } else {
      score_lo.mac(q, k_lo);
      score_hi.mac(q, k_hi);
    }
  }
  const auto scores_lo =
      aie::mul(score_lo.to_vector<float>(), ATTN_SCALE).to_vector<float>();
  const auto scores_hi =
      aie::mul(score_hi.to_vector<float>(), ATTN_SCALE).to_vector<float>();

  for (int query = 0; query < ATTN_QUERIES; ++query) {
    const auto lo = scores_lo.extract<ATTN_MMUL_N>(query);
    const auto hi = scores_hi.extract<ATTN_MMUL_N>(query);
    const float block_max = aie::max(aie::reduce_max(lo), aie::reduce_max(hi));
    const float old_max = stats[query];
    const float old_sum = stats[ATTN_QUERIES + query];
    const float new_max = old_max > block_max ? old_max : block_max;
    alpha[query] = old_sum == 0.0f ? 0.0f : (float)exp_bf16(old_max - new_max);
    const auto shifted_lo =
        aie::sub(lo, aie::broadcast<float, ATTN_MMUL_N>(new_max));
    const auto shifted_hi =
        aie::sub(hi, aie::broadcast<float, ATTN_MMUL_N>(new_max));
    const auto weight_lo = aie::exp2<bfloat16>(
        aie::mul(shifted_lo, LOG2E).to_vector<float>());
    const auto weight_hi = aie::exp2<bfloat16>(
        aie::mul(shifted_hi, LOG2E).to_vector<float>());
    aie::store_v(weights_lo + query * ATTN_MMUL_K, weight_lo);
    aie::store_v(weights_hi + query * ATTN_MMUL_K, weight_hi);
    const auto all_weights = aie::concat(weight_lo, weight_hi);
    const float block_sum = aie::reduce_add(
        aie::mul(all_weights,
                 aie::broadcast<bfloat16, ATTN_KEYS>((bfloat16)1.0f))
            .to_vector<float>());
    stats[query] = new_max;
    stats[ATTN_QUERIES + query] = old_sum * alpha[query] + block_sum;
  }

  const auto weight_vector_lo =
      aie::load_v<ATTN_MMUL::size_A>(weights_lo);
  const auto weight_vector_hi =
      aie::load_v<ATTN_MMUL::size_A>(weights_hi);
  for (int tile = 0; tile < ATTN_DIM_TILES; tile += 2) {
    ATTN_MMUL pv0;
    ATTN_MMUL pv1;
    pv0.mul(weight_vector_lo,
            aie::load_v<ATTN_MMUL::size_B>(
                values + (tile * ATTN_KEY_TILES + 0) * ATTN_MMUL::size_B));
    pv0.mac(weight_vector_hi,
            aie::load_v<ATTN_MMUL::size_B>(
                values + (tile * ATTN_KEY_TILES + 1) * ATTN_MMUL::size_B));
    pv1.mul(weight_vector_lo,
            aie::load_v<ATTN_MMUL::size_B>(
                values + ((tile + 1) * ATTN_KEY_TILES + 0) *
                             ATTN_MMUL::size_B));
    pv1.mac(weight_vector_hi,
            aie::load_v<ATTN_MMUL::size_B>(
                values + ((tile + 1) * ATTN_KEY_TILES + 1) *
                             ATTN_MMUL::size_B));
    const auto contribution0 = pv0.to_vector<float>();
    const auto contribution1 = pv1.to_vector<float>();
    for (int query = 0; query < ATTN_QUERIES; ++query) {
      float *output = accum + query * ATTN_DIM + tile * ATTN_MMUL_N;
      const auto retained =
          aie::mul(aie::load_v<16>(output), alpha[query]).to_vector<float>();
      const auto contribution =
          aie::concat(contribution0.extract<ATTN_MMUL_N>(query),
                      contribution1.extract<ATTN_MMUL_N>(query));
      aie::store_v(output, aie::add(retained, contribution));
    }
  }
}

void r30_attention_finish(const float *restrict accum,
                          const float *restrict stats,
                          int8_t *restrict output_bytes) {
  bfloat16 *output = reinterpret_cast<bfloat16 *>(output_bytes);
  for (int query = 0; query < ATTN_QUERIES; ++query) {
    const float inv_sum = aie::inv(stats[ATTN_QUERIES + query]);
    for (int dim = 0; dim < ATTN_DIM; dim += 16) {
      const auto normalized =
          aie::mul(aie::load_v<16>(accum + query * ATTN_DIM + dim), inv_sum)
              .to_vector<bfloat16>();
      aie::store_v(output + query * ATTN_DIM + dim, normalized);
    }
  }
}
} // extern "C"
