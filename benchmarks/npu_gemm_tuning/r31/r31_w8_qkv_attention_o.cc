// SPDX-License-Identifier: Apache-2.0
// BF16 output-projection phase appended to the admitted R30 object.

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int O_M = 32;
constexpr int O_K = 256;
constexpr int O_N = 32;
constexpr int O_MMUL_M = 4;
constexpr int O_MMUL_K = 8;
constexpr int O_MMUL_N = 8;
constexpr int O_M_TILES = O_M / O_MMUL_M;
constexpr int O_K_TILES = O_K / O_MMUL_K;
constexpr int O_N_TILES = O_N / O_MMUL_N;
using O_MMUL = aie::mmul<O_MMUL_M, O_MMUL_K, O_MMUL_N, bfloat16,
                         bfloat16>;

void output_projection_group(const bfloat16 *restrict activations,
                             const bfloat16 *restrict weights,
                             float *restrict output, bool accumulate) {
  for (int mt = 0; mt < O_M_TILES; ++mt)
    for (int nt = 0; nt < O_N_TILES; ++nt) {
      O_MMUL result;
      for (int kt = 0; kt < O_K_TILES; ++kt) {
        const auto a = aie::load_v<O_MMUL::size_A>(
            activations + (mt * O_K_TILES + kt) * O_MMUL::size_A);
        const auto b = aie::load_v<O_MMUL::size_B>(
            weights + (nt * O_K_TILES + kt) * O_MMUL::size_B);
        if (kt == 0)
          result.mul(a, b);
        else
          result.mac(a, b);
      }
      auto value = result.to_vector<float>();
      const int offset = (mt * O_N_TILES + nt) * O_MMUL::size_C;
      if (accumulate)
        value = aie::add(value, aie::load_v<O_MMUL::size_C>(output + offset));
      aie::store_v(output + offset, value);
    }
}
} // namespace

extern "C" {
void r31_output_projection_group(const int8_t *activations_bytes,
                                 const int8_t *weights_bytes,
                                 float *output, int32_t accumulate) {
  output_projection_group(
      reinterpret_cast<const bfloat16 *>(activations_bytes),
      reinterpret_cast<const bfloat16 *>(weights_bytes), output,
      accumulate != 0);
}

void r31_output_projection_finish(const float *restrict accum,
                                  int8_t *restrict output_bytes) {
  float *output = reinterpret_cast<float *>(output_bytes);
  for (int mt = 0; mt < O_M_TILES; ++mt)
    for (int row = 0; row < O_MMUL_M; ++row)
      for (int nt = 0; nt < O_N_TILES; ++nt) {
        const int source = (mt * O_N_TILES + nt) * O_MMUL::size_C +
                           row * O_MMUL_N;
        const int target = (mt * O_MMUL_M + row) * O_N + nt * O_MMUL_N;
        const auto value = aie::load_v<O_MMUL_N>(accum + source);
        aie::store_v(output + target, value);
      }
}
} // extern "C"
