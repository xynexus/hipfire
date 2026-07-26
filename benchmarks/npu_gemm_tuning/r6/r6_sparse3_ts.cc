// Compact OQ4.25 sparse residual: three (K-index, int8 delta) pairs per output column.
// A is row-major (MT*4)x256 int8, W is 64 columns x 6 bytes, C is row-major
// (MT*4)x64 int32. The correction is tiny enough that scalar gathers are preferable
// to feeding a dense W8 matrix.
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef MT
#define MT 4
#endif
#ifndef NT
#define NT 4
#endif
#ifndef KCHUNK
#define KCHUNK 16
#endif

extern "C" void r6_mac(const int8 *__restrict pA, const int8 *__restrict sparse,
                       int32 *__restrict pC) {
  static_assert(NT == 4, "sparse3 kernel requires NT=4");
  static_assert(KCHUNK == 16, "sparse3 kernel requires K=256");
  constexpr int rows = MT * 4;
  constexpr int cols = NT * 16;
  static_assert(MT == 4, "sparse3 vector kernel requires 16 resident rows");
  for (int col = 0; col < cols; ++col) {
    const uint8_t *entry = reinterpret_cast<const uint8_t *>(sparse + col * 6);
    aie::vector<int16, 16> a0, a1, a2;
    for (int row = 0; row < rows; ++row) {
      const int8 *a = pA + row * 256;
      a0.set(static_cast<int16_t>(a[entry[0]]), row);
      a1.set(static_cast<int16_t>(a[entry[2]]), row);
      a2.set(static_cast<int16_t>(a[entry[4]]), row);
    }
    auto d0 = aie::broadcast<int16, 16>(static_cast<int8_t>(entry[1]));
    auto d1 = aie::broadcast<int16, 16>(static_cast<int8_t>(entry[3]));
    auto d2 = aie::broadcast<int16, 16>(static_cast<int8_t>(entry[5]));
    auto acc = aie::mul(a0, d0);
    acc = aie::mac(acc, a1, d1);
    acc = aie::mac(acc, a2, d2);
    auto values = acc.template to_vector<int32>();
    for (int row = 0; row < rows; ++row) {
      pC[row * cols + col] = values[row];
    }
  }
}
