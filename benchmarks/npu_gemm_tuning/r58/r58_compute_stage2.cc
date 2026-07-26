#include <aie_api/aie.hpp>
#include "aie_kernels/aie_kernel_utils.h"
#include <stdint.h>

#define LM 6
#define LN 6
#define KT 16

using MMUL = aie::mmul<4, 16, 16, int8, int4>;
static constexpr unsigned WEIGHT_TILE_BYTES = MMUL::size_B / 2;
static constexpr unsigned WEIGHT_DATA_BYTES = LN * KT * WEIGHT_TILE_BYTES;
static_assert(WEIGHT_DATA_BYTES == 12288);

extern "C" void r58_compute_stage2(const int8_t *__restrict tile,
                                    int32_t *__restrict sums) {
  aie::vector<int32, MMUL::size_C> local = aie::zeros<int32, MMUL::size_C>();
  for (unsigned im = 0; im < LM; ++im) {
    aie::vector<int8, MMUL::size_A> activation =
        aie::broadcast<int8, MMUL::size_A>(static_cast<int8_t>(im + 1));
    for (unsigned jn = 0; jn < LN; ++jn) {
      aie::vector<int32, MMUL::size_C> partial_sum =
          aie::zeros<int32, MMUL::size_C>();
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_MIN_ITERATION_COUNT(KT / 2)
      for (unsigned k = 0; k < KT; k += 2) {
        const auto *w0 = reinterpret_cast<const int4 *>(
            tile + (jn * KT + k) * WEIGHT_TILE_BYTES);
        MMUL partial;
        partial.mul(activation, aie::load_v<MMUL::size_B>(w0));
        const auto *w1 = reinterpret_cast<const int4 *>(
            tile + (jn * KT + k + 1) * WEIGHT_TILE_BYTES);
        partial.mac(activation, aie::load_v<MMUL::size_B>(w1));
        partial_sum =
            aie::add(partial_sum, partial.template to_vector<int32>());
      }
      local = aie::add(local, partial_sum);
    }
  }
  aie::store_v(sums, aie::add(aie::load_v<MMUL::size_C>(sums), local));
}
