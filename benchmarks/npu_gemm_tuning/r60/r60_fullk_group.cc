#include <aie_api/aie.hpp>
#include <stdint.h>

using MMUL = aie::mmul<4, 16, 16, int8, int4>;

static inline aie::vector<int8, MMUL::size_A>
load_r34_a(const int8_t *__restrict activations, unsigned kt) {
  aie::vector<int8, MMUL::size_A> a;
  const unsigned first = kt * 128;
  for (unsigned row = 0; row < 4; ++row)
    for (unsigned k = 0; k < 16; ++k) {
      const unsigned source =
          k < 8 ? first + row * 8 + k
                : first + 64 + row * 8 + (k - 8);
      a.set(activations[source], row * 16 + k);
    }
  return a;
}

extern "C" void r60_fullk_group(const int8_t *__restrict activations,
                                 const int8_t *__restrict packed_weights,
                                 int32_t *__restrict output) {
  MMUL product;
  for (unsigned kt = 0; kt < 16; ++kt) {
    const auto *weights = reinterpret_cast<const int4 *>(packed_weights + kt * 128);
    const auto a = load_r34_a(activations, kt);
    if (kt == 0)
      product.mul(a, aie::load_v<MMUL::size_B>(weights));
    else
      product.mac(a, aie::load_v<MMUL::size_B>(weights));
  }
  aie::store_v(output, product.template to_vector<int32>());
}
