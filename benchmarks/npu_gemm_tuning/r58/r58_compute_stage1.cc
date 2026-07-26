#include <aie_api/aie.hpp>
#include "aie_kernels/aie_kernel_utils.h"
#include <stdint.h>

#ifndef DATA_BYTES
#define DATA_BYTES 12288
#endif

using MMUL = aie::mmul<4, 16, 16, int8, int4>;
static constexpr unsigned WEIGHT_TILE_BYTES = MMUL::size_B / 2;
static_assert(DATA_BYTES % WEIGHT_TILE_BYTES == 0);

extern "C" void r58_compute_stage1(const int8_t *__restrict tile,
                                    int32_t *__restrict sums) {
  aie::vector<int8, MMUL::size_A> activation =
      aie::broadcast<int8, MMUL::size_A>(1);
  aie::vector<int32, MMUL::size_C> local = aie::zeros<int32, MMUL::size_C>();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (unsigned offset = 0; offset < DATA_BYTES; offset += WEIGHT_TILE_BYTES) {
    const auto *packed = reinterpret_cast<const int4 *>(tile + offset);
    aie::vector<int4, MMUL::size_B> weights =
        aie::load_v<MMUL::size_B>(packed);
    MMUL partial;
    partial.mul(activation, weights);
    local = aie::add(local, partial.template to_vector<int32>());
  }

  aie::vector<int32, MMUL::size_C> prior = aie::load_v<MMUL::size_C>(sums);
  aie::store_v(sums, aie::add(prior, local));
}
