// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef HIPFIRE_HEAD_DIM
#define HIPFIRE_HEAD_DIM 128
#endif

namespace {
constexpr int BYTES = 4 * HIPFIRE_HEAD_DIM * sizeof(bfloat16);
}

extern "C" void hipfire_qwen3_copy_attention_tile(
    const int8_t *restrict input, int8_t *restrict output) {
  for (int offset = 0; offset < BYTES; offset += 64) {
    aie::store_v(output + offset, aie::load_v<64>(input + offset));
  }
}
