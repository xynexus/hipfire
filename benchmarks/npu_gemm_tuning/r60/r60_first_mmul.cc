#include <aie_api/aie.hpp>
#include <stdint.h>

using MMUL = aie::mmul<4, 16, 16, int8, int4>;

extern "C" void r60_first_mmul(const int8_t *__restrict activations,
                                const int8_t *__restrict packed_weights,
                                int32_t *__restrict output) {
  // R34 stores eight K values for one row contiguously. Join two adjacent
  // K tiles for four rows into the row-major 4x16 A operand expected by MMUL.
  aie::vector<int8, MMUL::size_A> a;
  for (unsigned row = 0; row < 4; ++row)
    for (unsigned k = 0; k < 16; ++k) {
      const unsigned source =
          k < 8 ? row * 8 + k : 64 + row * 8 + (k - 8);
      a.set(activations[source], row * 16 + k);
    }

  // WholeScaledV1 stores one row-major 16x16 W4 tile in the first 128 bytes.
  // The signed nibbles stay packed until this load; no global block reorder is
  // performed in the kernel.
  const auto *weights = reinterpret_cast<const int4 *>(packed_weights);
  MMUL product;
  product.mul(a, aie::load_v<MMUL::size_B>(weights));
  aie::store_v(output, product.template to_vector<int32>());
}
