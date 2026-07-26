// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef HIPFIRE_HEAD_DIM
#define HIPFIRE_HEAD_DIM 128
#endif

namespace {
constexpr int QUERIES = 4;
constexpr int HEAD_DIM = HIPFIRE_HEAD_DIM;
constexpr int MMUL_K = 8;
constexpr int Q_TILE_ELEMENTS = QUERIES * HEAD_DIM;
constexpr int Q_TILE_BYTES = Q_TILE_ELEMENTS * sizeof(bfloat16);
constexpr int LENGTH_TRAILER_BYTES = 512;
}

extern "C" void hipfire_qwen3_pack_query_pair(
    const int8_t *restrict query_pair, int8_t *restrict packed_pair,
    int32_t real_length) {
  const auto *q0 = reinterpret_cast<const bfloat16 *>(query_pair);
  const auto *q1 = reinterpret_cast<const bfloat16 *>(query_pair + Q_TILE_BYTES);
  auto *output = reinterpret_cast<bfloat16 *>(packed_pair);
  for (int dim_tile = 0; dim_tile < HEAD_DIM / MMUL_K; ++dim_tile) {
    for (int query = 0; query < QUERIES; ++query) {
      const int source = query * HEAD_DIM + dim_tile * MMUL_K;
      const int destination =
          (dim_tile * QUERIES + query) * MMUL_K;
      aie::store_v(output + destination, aie::load_v<MMUL_K>(q0 + source));
      aie::store_v(output + Q_TILE_ELEMENTS + destination,
                   aie::load_v<MMUL_K>(q1 + source));
    }
  }
  auto *trailer = packed_pair + 2 * Q_TILE_BYTES;
  *reinterpret_cast<int32_t *>(trailer) = real_length;
  for (int offset = sizeof(int32_t); offset < 16; ++offset)
    trailer[offset] = 0;
  for (int offset = 16; offset < LENGTH_TRAILER_BYTES; offset += 16)
    aie::store_v(trailer + offset, aie::zeros<int8_t, 16>());
}
