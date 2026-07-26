// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef HIPFIRE_HEAD_DIM
#define HIPFIRE_HEAD_DIM 128
#endif

namespace {
constexpr int KEYS = 16;
constexpr int HEAD_DIM = HIPFIRE_HEAD_DIM;
constexpr int TILE = 8;
constexpr int HEAD_ELEMENTS = KEYS * HEAD_DIM;
}

extern "C" void hipfire_qwen3_pack_kv_block(
    const int8_t *restrict key_bytes, const int8_t *restrict value_bytes,
    int8_t *restrict packed_bytes) {
  const auto *keys = reinterpret_cast<const bfloat16 *>(key_bytes);
  const auto *values = reinterpret_cast<const bfloat16 *>(value_bytes);
  auto *packed_keys = reinterpret_cast<bfloat16 *>(packed_bytes);
  auto *packed_values = packed_keys + HEAD_ELEMENTS;
  for (int key_tile = 0; key_tile < KEYS / TILE; ++key_tile) {
    for (int dim_tile = 0; dim_tile < HEAD_DIM / TILE; ++dim_tile) {
      auto *destination = packed_keys +
                          (key_tile * (HEAD_DIM / TILE) + dim_tile) *
                              TILE * TILE;
      for (int dim_lane = 0; dim_lane < TILE; ++dim_lane) {
        for (int key_lane = 0; key_lane < TILE; ++key_lane) {
          destination[dim_lane * TILE + key_lane] =
              keys[(key_tile * TILE + key_lane) * HEAD_DIM +
                   dim_tile * TILE + dim_lane];
        }
      }
    }
  }
  for (int dim_tile = 0; dim_tile < HEAD_DIM / TILE; ++dim_tile) {
    for (int key_tile = 0; key_tile < KEYS / TILE; ++key_tile) {
      auto *destination = packed_values +
                          (dim_tile * (KEYS / TILE) + key_tile) *
                              TILE * TILE;
      for (int key_lane = 0; key_lane < TILE; ++key_lane) {
        const auto *source = values + (key_tile * TILE + key_lane) * HEAD_DIM +
                             dim_tile * TILE;
        aie::store_v(destination + key_lane * TILE,
                     aie::load_v<TILE>(source));
      }
    }
  }
}
