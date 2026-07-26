// SPDX-License-Identifier: Apache-2.0

#include "../r113/r113_tail_pack.cc"

namespace {
constexpr int R114_CHUNK_BYTES = 8 * 256 + 8 * sizeof(float);
constexpr int R114_BLOCK_BYTES = 3 * R114_CHUNK_BYTES;
constexpr int R114_Q_BYTES = 3 * 8 * 256;
constexpr int R114_SCALE_BYTES = 3 * 8 * sizeof(float);
} // namespace

extern "C" {

__attribute__((noinline, minsize)) void
r114_copy_chunk(const int8_t *restrict input, int8_t *restrict output) {
  for (int offset = 0; offset < R114_CHUNK_BYTES; offset += 32)
    aie::store_v(output + offset, aie::load_v<32>(input + offset));
}

__attribute__((noinline, minsize)) void
r114_copy_group(const int8_t *restrict chunks, int8_t *restrict output,
                int32_t group) {
  r114_copy_chunk(chunks + group * R114_CHUNK_BYTES, output);
}

__attribute__((noinline, minsize)) void
r114_place_chunk(int8_t *restrict block, const int8_t *restrict chunk,
                 int32_t lm) {
  copy_chunk_to_block(chunk, block, lm);
}

__attribute__((noinline, minsize)) void
r114_place_group(int8_t *restrict block, const int8_t *restrict chunks,
                 int32_t lm, int32_t group) {
  copy_chunk_to_block(chunks + group * R114_CHUNK_BYTES, block, lm);
}

__attribute__((noinline, minsize)) void
r114_copy_block(const int8_t *restrict input, int8_t *restrict output) {
  for (int offset = 0; offset < R114_BLOCK_BYTES; offset += 32)
    aie::store_v(output + offset, aie::load_v<32>(input + offset));
}

__attribute__((noinline, minsize)) void
r114_zero_output(int8_t *restrict output) {
  for (int offset = 0; offset < R114_Q_BYTES; offset += 32)
    aie::store_v(output + offset, aie::zeros<int8_t, 32>());
}

__attribute__((noinline, minsize)) void
r114_emit_q(const int8_t *restrict block, int8_t *restrict output) {
  for (int offset = 0; offset < R114_Q_BYTES; offset += 32)
    aie::store_v(output + offset, aie::load_v<32>(block + offset));
}

__attribute__((noinline, minsize)) void
r114_emit_scales(const int8_t *restrict block, int8_t *restrict output) {
  r114_zero_output(output);
  for (int offset = 0; offset < R114_SCALE_BYTES; offset += 32)
    aie::store_v(output + offset,
                 aie::load_v<32>(block + R114_Q_BYTES + offset));
}

} // extern "C"
