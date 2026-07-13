// SPDX-License-Identifier: Apache-2.0

#include "../r85/r85_output_projection_m8_reuse_a.cc"

namespace {
constexpr int R89_ROWS = 8;
constexpr int R89_SLICE = 32;
constexpr int R89_PAIR = 64;
constexpr int R89_PREFIX_BYTES = 8192;

static inline bfloat16 *r89_storage(int8_t *prefix, int8_t *tail,
                                    int element) {
  const int byte = element * 2;
  return reinterpret_cast<bfloat16 *>(
      byte < R89_PREFIX_BYTES ? prefix + byte : tail + byte - R89_PREFIX_BYTES);
}

static inline const bfloat16 *r89_storage(const int8_t *prefix,
                                          const int8_t *tail, int element) {
  const int byte = element * 2;
  return reinterpret_cast<const bfloat16 *>(
      byte < R89_PREFIX_BYTES ? prefix + byte : tail + byte - R89_PREFIX_BYTES);
}
} // namespace

extern "C" {
void r89_output_projection_finish_pair_bf16_split(
    const float *restrict accum0, const float *restrict accum1,
    int8_t *restrict prefix, int8_t *restrict tail, int32_t pair) {
  for (int slice = 0; slice < 2; ++slice) {
    const float *accum = slice == 0 ? accum0 : accum1;
    for (int mt = 0; mt < 2; ++mt)
      for (int row = 0; row < 4; ++row)
        for (int nt = 0; nt < 4; ++nt) {
          const int source = (mt * 4 + nt) * 32 + row * 8;
          const int column = pair * R89_PAIR + slice * R89_SLICE + nt * 8;
          const int block = column >> 8;
          const int target = block * R89_ROWS * 256 + (mt * 4 + row) * 256 +
                             (column & 255);
          aie::store_v(r89_storage(prefix, tail, target),
                       aie::mul(aie::load_v<8>(accum + source), 1.0f)
                           .to_vector<bfloat16>());
        }
  }
}

__attribute__((noinline, minsize)) void
r89_emit_bf16_chunk(const int8_t *restrict prefix,
                    const int8_t *restrict tail, int8_t *restrict output_bytes,
                    int32_t chunk) {
  const int8_t *source =
      chunk < 4 ? prefix + chunk * 2048 : tail + (chunk - 4) * 2048;
  for (int offset = 0; offset < 2048; offset += 64)
    aie::store_v(output_bytes + offset, aie::load_v<64>(source + offset));
}
}
