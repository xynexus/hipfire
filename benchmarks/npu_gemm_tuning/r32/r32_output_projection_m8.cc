// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int M = 8;
constexpr int K = 256;
constexpr int N = 32;
constexpr int MMUL_M = 4;
constexpr int MMUL_K = 8;
constexpr int MMUL_N = 8;
constexpr int M_TILES = M / MMUL_M;
constexpr int K_TILES = K / MMUL_K;
constexpr int N_TILES = N / MMUL_N;
using MMUL = aie::mmul<MMUL_M, MMUL_K, MMUL_N, bfloat16, bfloat16>;
} // namespace

extern "C" {
void r32_output_projection_group_m8(const int8_t *activations_bytes,
                                    const int8_t *weights_bytes, float *output,
                                    int32_t accumulate) {
  const auto *activations =
      reinterpret_cast<const bfloat16 *>(activations_bytes);
  const auto *weights = reinterpret_cast<const bfloat16 *>(weights_bytes);
  for (int mt = 0; mt < M_TILES; ++mt)
    for (int nt = 0; nt < N_TILES; ++nt) {
      MMUL result;
      for (int kt = 0; kt < K_TILES; ++kt) {
        const auto a = aie::load_v<MMUL::size_A>(
            activations + (mt * K_TILES + kt) * MMUL::size_A);
        const auto b = aie::load_v<MMUL::size_B>(
            weights + (nt * K_TILES + kt) * MMUL::size_B);
        if (kt == 0)
          result.mul(a, b);
        else
          result.mac(a, b);
      }
      auto value = result.to_vector<float>();
      const int offset = (mt * N_TILES + nt) * MMUL::size_C;
      if (accumulate)
        value = aie::add(value, aie::load_v<MMUL::size_C>(output + offset));
      aie::store_v(output + offset, value);
    }
}

void r32_output_projection_finish_pair_m8(const float *restrict accum0,
                                          const float *restrict accum1,
                                          int8_t *restrict output_bytes) {
  auto *output = reinterpret_cast<float *>(output_bytes);
  for (int slice = 0; slice < 2; ++slice) {
    const float *accum = slice == 0 ? accum0 : accum1;
    for (int mt = 0; mt < M_TILES; ++mt)
      for (int row = 0; row < MMUL_M; ++row)
        for (int nt = 0; nt < N_TILES; ++nt) {
          const int source =
              (mt * N_TILES + nt) * MMUL::size_C + row * MMUL_N;
          const int target =
              (mt * MMUL_M + row) * (2 * N) + slice * N + nt * MMUL_N;
          aie::store_v(output + target, aie::load_v<MMUL_N>(accum + source));
        }
  }
}
}
