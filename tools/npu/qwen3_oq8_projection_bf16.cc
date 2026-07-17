// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int ROWS = 8;
constexpr int GROUP = 256;
constexpr int OUTPUTS = 16;
constexpr int MMUL_M = 4;
constexpr int MMUL_K = 8;
constexpr int MMUL_N = 8;
constexpr int K_TILES = GROUP / MMUL_K;
constexpr int A_ELEMENTS = ROWS * GROUP;
using MMUL = aie::mmul<MMUL_M, MMUL_K, MMUL_N, bfloat16, bfloat16>;

static void pack_activations(const bfloat16 *restrict input,
                             bfloat16 *restrict packed) {
  for (int k_tile = 0; k_tile < K_TILES; ++k_tile)
    for (int row = 0; row < ROWS; ++row)
      aie::store_v(packed + (k_tile * ROWS + row) * MMUL_K,
                   aie::load_v<MMUL_K>(input + row * GROUP + k_tile * MMUL_K));
}
} // namespace

extern "C" __attribute__((noinline, minsize)) void
hipfire_qwen3_oq8_projection_group(
    const int8_t *restrict input_bytes, const int8_t *restrict weight_bytes,
    float *restrict accumulator, int8_t *restrict packed_bytes,
    int32_t initialize, int32_t wave, int32_t pair_lane) {
  const auto *input = reinterpret_cast<const bfloat16 *>(input_bytes) +
                      pair_lane * A_ELEMENTS;
  const auto *weights = reinterpret_cast<const bfloat16 *>(weight_bytes);
  auto *packed = reinterpret_cast<bfloat16 *>(packed_bytes);
  accumulator += wave * ROWS * OUTPUTS;
  pack_activations(input, packed);

  MMUL row0_out0, row0_out1, row1_out0, row1_out1;
  for (int k_tile = 0; k_tile < K_TILES; ++k_tile) {
    const auto a0 = aie::load_v<MMUL::size_A>(
        packed + (k_tile * ROWS + 0 * MMUL_M) * MMUL_K);
    const auto a1 = aie::load_v<MMUL::size_A>(
        packed + (k_tile * ROWS + 1 * MMUL_M) * MMUL_K);
    const auto b0 = aie::load_v<MMUL::size_B>(
        weights + (k_tile * 2 + 0) * MMUL_K * MMUL_N);
    const auto b1 = aie::load_v<MMUL::size_B>(
        weights + (k_tile * 2 + 1) * MMUL_K * MMUL_N);
    if (k_tile == 0) {
      row0_out0.mul(a0, b0);
      row0_out1.mul(a0, b1);
      row1_out0.mul(a1, b0);
      row1_out1.mul(a1, b1);
    } else {
      row0_out0.mac(a0, b0);
      row0_out1.mac(a0, b1);
      row1_out0.mac(a1, b0);
      row1_out1.mac(a1, b1);
    }
  }
  const auto results00 = row0_out0.template to_vector<float>();
  const auto results01 = row0_out1.template to_vector<float>();
  const auto results10 = row1_out0.template to_vector<float>();
  const auto results11 = row1_out1.template to_vector<float>();
  for (int row = 0; row < ROWS; ++row) {
    const int local = row % MMUL_M;
    const auto low = row < MMUL_M
                         ? results00.template extract<MMUL_N>(local)
                         : results10.template extract<MMUL_N>(local);
    const auto high = row < MMUL_M
                          ? results01.template extract<MMUL_N>(local)
                          : results11.template extract<MMUL_N>(local);
    auto values = aie::concat(low, high);
    if (!initialize)
      values = aie::add(values,
                        aie::load_v<OUTPUTS>(accumulator + row * OUTPUTS));
    aie::store_v(accumulator + row * OUTPUTS, values);
  }
}

extern "C" __attribute__((noinline, minsize)) void
hipfire_qwen3_oq8_projection_finish(const float *restrict accumulator,
                                    int8_t *restrict output_bytes,
                                    int32_t wave) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  accumulator += wave * ROWS * OUTPUTS;
  auto *output = reinterpret_cast<bfloat16 *>(output_bytes);
  for (int row = 0; row < ROWS; ++row) {
    auto values = aie::mul(aie::load_v<OUTPUTS>(accumulator + row * OUTPUTS),
                           aie::broadcast<float, OUTPUTS>(1.0f))
                      .template to_vector<bfloat16>();
    aie::store_v(output + row * OUTPUTS, values);
  }
}
