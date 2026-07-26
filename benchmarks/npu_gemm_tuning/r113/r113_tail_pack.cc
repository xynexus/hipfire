// SPDX-License-Identifier: Apache-2.0

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../r47/r47_next_layer_prep.cc"

namespace {
constexpr int R113_ROWS = 8;
constexpr int R113_GROUPS = 3;
constexpr int R113_GROUP = 256;
constexpr int R113_COMPLETED_ROW_BYTES =
    2 * R113_GROUPS * R113_GROUP * sizeof(bfloat16);
constexpr int R113_PHASE_ROWS = 2;
constexpr int R113_PHASE_BYTES = R113_PHASE_ROWS * R113_COMPLETED_ROW_BYTES;
constexpr int R113_GROUP_PARAM_BYTES =
    2 * R113_GROUP * sizeof(float) + 2 * R113_GROUP * sizeof(bfloat16);
constexpr int R113_NEXT_PARAM_BYTES = R113_GROUPS * R113_GROUP_PARAM_BYTES;
constexpr int R113_CHUNK_BYTES =
    R113_ROWS * R113_GROUP + R113_ROWS * sizeof(float);
constexpr int R113_OUTPUT_BYTES = R113_PHASE_BYTES;
} // namespace

extern "C" {

__attribute__((noinline, minsize)) void
r113_copy_next_params(const int8_t *restrict input,
                      int8_t *restrict params) {
  for (int offset = 0; offset < R113_NEXT_PARAM_BYTES; offset += 32)
    aie::store_v(params + offset, aie::load_v<32>(input + offset));
}

void r113_pack_phase(const int8_t *restrict completed,
                     const int8_t *restrict params, int8_t *restrict chunks,
                     float *restrict scratch, float *restrict inverse,
                     int32_t phase) {
  for (int local_row = 0; local_row < R113_PHASE_ROWS; ++local_row) {
    const int row = phase * R113_PHASE_ROWS + local_row;
    const int8_t *source =
        completed + local_row * R113_COMPLETED_ROW_BYTES;
    r47_accumulate_row(source, inverse, row);
    for (int group = 0; group < R113_GROUPS; ++group)
      r47_pack_row(source, params + group * R113_GROUP_PARAM_BYTES,
                   chunks + group * R113_CHUNK_BYTES, scratch, inverse, row,
                   group);
  }
}

__attribute__((noinline, minsize)) void
r113_emit_pack_group(const int8_t *restrict chunks, int8_t *restrict output,
                     int32_t group) {
  for (int offset = 0; offset < R113_OUTPUT_BYTES; offset += 32)
    aie::store_v(output + offset, aie::zeros<int8_t, 32>());
  const int8_t *source = chunks + group * R113_CHUNK_BYTES;
  for (int offset = 0; offset < R113_CHUNK_BYTES; offset += 32)
    aie::store_v(output + offset, aie::load_v<32>(source + offset));
}

} // extern "C"
