// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include "aie_kernels/aie_kernel_utils.h"
#include <stdint.h>

namespace {
using MMUL = aie::mmul<8, 8, 8, int8, int8>;
constexpr int K_TILES = 32;
constexpr int CHUNK_BYTES = 8 * 256 + 8 * sizeof(float);
constexpr int CHUNK_STRIDE = 2112;
constexpr int A_SCALE_OFFSET = K_TILES * MMUL::size_A;
constexpr int WEIGHT_SLICE_BYTES = K_TILES * 2 * MMUL::size_B;
constexpr int W_SCALE_OFFSET = 2 * WEIGHT_SLICE_BYTES;

static inline aie::vector<int32, 16>
join_rows(aie::vector<int32, MMUL::size_C> lo,
          aie::vector<int32, MMUL::size_C> hi, int row) {
  switch (row) {
  case 0: return aie::concat(lo.extract<8>(0), hi.extract<8>(0));
  case 1: return aie::concat(lo.extract<8>(1), hi.extract<8>(1));
  case 2: return aie::concat(lo.extract<8>(2), hi.extract<8>(2));
  case 3: return aie::concat(lo.extract<8>(3), hi.extract<8>(3));
  case 4: return aie::concat(lo.extract<8>(4), hi.extract<8>(4));
  case 5: return aie::concat(lo.extract<8>(5), hi.extract<8>(5));
  case 6: return aie::concat(lo.extract<8>(6), hi.extract<8>(6));
  default: return aie::concat(lo.extract<8>(7), hi.extract<8>(7));
  }
}

} // namespace

extern "C" {

void r118_stage_chunk(const int8_t *restrict activation_slot,
                      int8_t *restrict activation_stage, int32_t group) {
  int8_t *target = activation_stage + group * CHUNK_STRIDE;
  for (int offset = 0; offset < CHUNK_BYTES; offset += 32)
    aie::store_v(target + offset, aie::load_v<32>(activation_slot + offset));
  chess_separator_scheduler(1);
  chess_separator();
}

void r118_compact_group_n32(const int8_t *restrict activation_stage,
                            const int8_t *restrict weight_record,
                            float *restrict output, int32_t group) {
  const int8_t *activation_slot = activation_stage + group * CHUNK_STRIDE;
  alignas(32) float prior[8 * 32];
  if (group != 0)
    for (int row = 0; row < 8; ++row) {
      aie::store_v(prior + row * 32, aie::load_v<16>(output + row * 32));
      aie::store_v(prior + row * 32 + 16,
                   aie::load_v<16>(output + row * 32 + 16));
    }

  MMUL lo0, hi0, lo1, hi1;
  auto a = aie::load_v<MMUL::size_A>(activation_slot);
  const int8_t *w0 = weight_record;
  const int8_t *w1 = weight_record + WEIGHT_SLICE_BYTES;
  lo0.mul(a, aie::load_v<MMUL::size_B>(w0));
  hi0.mul(a, aie::load_v<MMUL::size_B>(w0 + MMUL::size_B));
  lo1.mul(a, aie::load_v<MMUL::size_B>(w1));
  hi1.mul(a, aie::load_v<MMUL::size_B>(w1 + MMUL::size_B));
  for (int kt = 1; kt < K_TILES; ++kt) {
    a = aie::load_v<MMUL::size_A>(activation_slot + kt * MMUL::size_A);
    w0 = weight_record + kt * 2 * MMUL::size_B;
    w1 = weight_record + WEIGHT_SLICE_BYTES + kt * 2 * MMUL::size_B;
    lo0.mac(a, aie::load_v<MMUL::size_B>(w0));
    hi0.mac(a, aie::load_v<MMUL::size_B>(w0 + MMUL::size_B));
    lo1.mac(a, aie::load_v<MMUL::size_B>(w1));
    hi1.mac(a, aie::load_v<MMUL::size_B>(w1 + MMUL::size_B));
  }

  const auto vlo0 = lo0.to_vector<int32>();
  const auto vhi0 = hi0.to_vector<int32>();
  const auto vlo1 = lo1.to_vector<int32>();
  const auto vhi1 = hi1.to_vector<int32>();
  const auto *weight_scales =
      reinterpret_cast<const float *>(weight_record + W_SCALE_OFFSET);
  const auto weight_scale0 = aie::load_v<16>(weight_scales);
  const auto weight_scale1 = aie::load_v<16>(weight_scales + 16);
  const auto *activation_scale =
      reinterpret_cast<const float *>(activation_slot + A_SCALE_OFFSET);
  for (int row = 0; row < 8; ++row) {
    auto scaled0 =
        aie::mul(aie::to_float(join_rows(vlo0, vhi0, row)), weight_scale0)
            .to_vector<float>();
    auto scaled1 =
        aie::mul(aie::to_float(join_rows(vlo1, vhi1, row)), weight_scale1)
            .to_vector<float>();
    scaled0 = aie::mul(scaled0, activation_scale[row]).to_vector<float>();
    scaled1 = aie::mul(scaled1, activation_scale[row]).to_vector<float>();
    if (group != 0) {
      scaled0 = aie::add(scaled0, aie::load_v<16>(prior + row * 32));
      scaled1 =
          aie::add(scaled1, aie::load_v<16>(prior + row * 32 + 16));
    }
    aie::store_v(output + row * 32, scaled0);
    aie::store_v(output + row * 32 + 16, scaled1);
  }
  chess_separator_scheduler(1);
  chess_separator();
}

void r129_compact_group_n32_b2(const int8_t *restrict activation_stage0,
                               const int8_t *restrict activation_stage1,
                               const int8_t *restrict weight_record,
                               float *restrict output, int32_t group) {
  r118_compact_group_n32(activation_stage0, weight_record, output, group);
  r118_compact_group_n32(activation_stage1, weight_record, output + 8 * 32,
                         group);
}

void r129_compact_group_n32_b4(const int8_t *restrict activation_stage0,
                               const int8_t *restrict activation_stage1,
                               const int8_t *restrict activation_stage2,
                               const int8_t *restrict activation_stage3,
                               const int8_t *restrict weight_record,
                               float *restrict output, int32_t group) {
  r118_compact_group_n32(activation_stage0, weight_record, output, group);
  r118_compact_group_n32(activation_stage1, weight_record, output + 8 * 32,
                         group);
  r118_compact_group_n32(activation_stage2, weight_record, output + 2 * 8 * 32,
                         group);
  r118_compact_group_n32(activation_stage3, weight_record, output + 3 * 8 * 32,
                         group);
}

} // extern "C"
