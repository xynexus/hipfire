// Dense W8A8 8x8x8 GEMM with row-major A for mixed-Opus residuals.
//
// The standard W8M8 kernel consumes tile-major A. Mixed full-K scheduling must
// reuse the same row-major activation FIFO as the W4 base, so this variant uses
// an AIE tensor buffer stream to form each 8x8 activation tile in-core.
#include <aie_api/aie.hpp>

#ifndef MT
#define MT 4
#endif
#ifndef NT
#define NT 4
#endif
#ifndef KCHUNK
#define KCHUNK 32
#endif

using MMUL = aie::mmul<8, 8, 8, int8, int8>;

static inline aie::vector<int8, MMUL::size_B>
load_w(const int8 *weights, int nt, int half, int k) {
  return aie::load_v<MMUL::size_B>(
      weights + ((nt * KCHUNK + k) * 2 + half) * MMUL::size_B);
}

static inline aie::vector<int32, 16>
join_rows(aie::vector<int32, MMUL::size_C> low,
          aie::vector<int32, MMUL::size_C> high, int row) {
  switch (row) {
  case 0:
    return aie::concat(low.template extract<8>(0), high.template extract<8>(0));
  case 1:
    return aie::concat(low.template extract<8>(1), high.template extract<8>(1));
  case 2:
    return aie::concat(low.template extract<8>(2), high.template extract<8>(2));
  case 3:
    return aie::concat(low.template extract<8>(3), high.template extract<8>(3));
  case 4:
    return aie::concat(low.template extract<8>(4), high.template extract<8>(4));
  case 5:
    return aie::concat(low.template extract<8>(5), high.template extract<8>(5));
  case 6:
    return aie::concat(low.template extract<8>(6), high.template extract<8>(6));
  default:
    return aie::concat(low.template extract<8>(7), high.template extract<8>(7));
  }
}

static inline void store_rows(int32 *output, int mt, int nt,
                              aie::vector<int32, MMUL::size_C> low,
                              aie::vector<int32, MMUL::size_C> high) {
  int32 *base = output + mt * NT * 8 * 16 + nt * 16;
  for (int row = 0; row < 8; ++row) {
    auto value = join_rows(low, high, row);
#ifdef R6_W8_ACCUMULATE
    value = aie::add(aie::load_v<16>(base + row * NT * 16), value);
#endif
    aie::store_v(base + row * NT * 16, value);
  }
}

#ifndef R6_W8_ENTRY
#define R6_W8_ENTRY r6_w8_residual
#endif

extern "C" void R6_W8_ENTRY(const int8 *__restrict activations,
                            const int8 *__restrict weights,
                            int32 *__restrict output) {
  static_assert(NT == 4, "row-major W8 kernel requires NT=4");
  static_assert(KCHUNK % 2 == 0, "row-major W8 kernel requires even KCHUNK");
  // AIE2P's smallest native int8 vector is 16 lanes. Read two adjacent
  // 8-wide K tiles from every row, concatenate the eight rows, then unzip in
  // 8-byte chunks to form the two 8x8 MMUL operands.
  auto a_desc = aie::make_tensor_descriptor<int8, 16>(
      aie::tensor_dim(MT, 4 * KCHUNK),
      aie::tensor_dim(KCHUNK / 2, 1),
      aie::tensor_dim(8u, KCHUNK / 2));
  auto stream_a = aie::make_tensor_buffer_stream(activations, a_desc);

  for (int mt = 0; mt < MT; ++mt) {
    MMUL c0l, c0h, c1l, c1h, c2l, c2h, c3l, c3h;
    aie::vector<int8, 16> r0, r1, r2, r3, r4, r5, r6, r7;
    stream_a >> r0 >> r1 >> r2 >> r3 >> r4 >> r5 >> r6 >> r7;
    auto rows = aie::concat(r0, r1, r2, r3, r4, r5, r6, r7);
    auto a = aie::filter_even(rows, 8);
    auto next_a = aie::filter_odd(rows, 8);
    c0l.mul(a, load_w(weights, 0, 0, 0));
    c0h.mul(a, load_w(weights, 0, 1, 0));
    c1l.mul(a, load_w(weights, 1, 0, 0));
    c1h.mul(a, load_w(weights, 1, 1, 0));
    c2l.mul(a, load_w(weights, 2, 0, 0));
    c2h.mul(a, load_w(weights, 2, 1, 0));
    c3l.mul(a, load_w(weights, 3, 0, 0));
    c3h.mul(a, load_w(weights, 3, 1, 0));
    c0l.mac(next_a, load_w(weights, 0, 0, 1));
    c0h.mac(next_a, load_w(weights, 0, 1, 1));
    c1l.mac(next_a, load_w(weights, 1, 0, 1));
    c1h.mac(next_a, load_w(weights, 1, 1, 1));
    c2l.mac(next_a, load_w(weights, 2, 0, 1));
    c2h.mac(next_a, load_w(weights, 2, 1, 1));
    c3l.mac(next_a, load_w(weights, 3, 0, 1));
    c3h.mac(next_a, load_w(weights, 3, 1, 1));
    for (int k = 2; k < KCHUNK; k += 2) {
      stream_a >> r0 >> r1 >> r2 >> r3 >> r4 >> r5 >> r6 >> r7;
      rows = aie::concat(r0, r1, r2, r3, r4, r5, r6, r7);
      a = aie::filter_even(rows, 8);
      next_a = aie::filter_odd(rows, 8);
      c0l.mac(a, load_w(weights, 0, 0, k));
      c0h.mac(a, load_w(weights, 0, 1, k));
      c1l.mac(a, load_w(weights, 1, 0, k));
      c1h.mac(a, load_w(weights, 1, 1, k));
      c2l.mac(a, load_w(weights, 2, 0, k));
      c2h.mac(a, load_w(weights, 2, 1, k));
      c3l.mac(a, load_w(weights, 3, 0, k));
      c3h.mac(a, load_w(weights, 3, 1, k));
      c0l.mac(next_a, load_w(weights, 0, 0, k + 1));
      c0h.mac(next_a, load_w(weights, 0, 1, k + 1));
      c1l.mac(next_a, load_w(weights, 1, 0, k + 1));
      c1h.mac(next_a, load_w(weights, 1, 1, k + 1));
      c2l.mac(next_a, load_w(weights, 2, 0, k + 1));
      c2h.mac(next_a, load_w(weights, 2, 1, k + 1));
      c3l.mac(next_a, load_w(weights, 3, 0, k + 1));
      c3h.mac(next_a, load_w(weights, 3, 1, k + 1));
    }
    store_rows(output, mt, 0, c0l.template to_vector<int32>(),
               c0h.template to_vector<int32>());
    store_rows(output, mt, 1, c1l.template to_vector<int32>(),
               c1h.template to_vector<int32>());
    store_rows(output, mt, 2, c2l.template to_vector<int32>(),
               c2h.template to_vector<int32>());
    store_rows(output, mt, 3, c3l.template to_vector<int32>(),
               c3h.template to_vector<int32>());
  }
}
