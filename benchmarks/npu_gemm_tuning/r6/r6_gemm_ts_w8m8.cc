// R6-TS-W8M8: AIE2P dense W8A8 GEMM using 8x8x8 int8 mmuls.
//
// This keeps W8 separate from the AIE2/W4 kernels. Each output tile is 8x16,
// formed from two 8x8 N halves. KCHUNK counts 8-wide K tiles.
//
// pA: TILE-MAJOR MT x KCHUNK x 8x8 int8 A tiles.
// pW: TILE-MAJOR NT x KCHUNK x two N-halves x 8x8 int8 B tiles.
// pC: ROW-MAJOR (MT*8) x (NT*16) int32.
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

static inline aie::vector<int8, MMUL::size_B> ldW(const int8 *wbytes, int nt,
                                                  int n_half, int k) {
  return aie::load_v<MMUL::size_B>(
      wbytes + ((nt * KCHUNK + k) * 2 + n_half) * MMUL::size_B);
}

static inline aie::vector<int32, 16> join_rows(aie::vector<int32, MMUL::size_C> lo,
                                               aie::vector<int32, MMUL::size_C> hi,
                                               int row) {
  switch (row) {
  case 0:
    return aie::concat(lo.template extract<8>(0), hi.template extract<8>(0));
  case 1:
    return aie::concat(lo.template extract<8>(1), hi.template extract<8>(1));
  case 2:
    return aie::concat(lo.template extract<8>(2), hi.template extract<8>(2));
  case 3:
    return aie::concat(lo.template extract<8>(3), hi.template extract<8>(3));
  case 4:
    return aie::concat(lo.template extract<8>(4), hi.template extract<8>(4));
  case 5:
    return aie::concat(lo.template extract<8>(5), hi.template extract<8>(5));
  case 6:
    return aie::concat(lo.template extract<8>(6), hi.template extract<8>(6));
  default:
    return aie::concat(lo.template extract<8>(7), hi.template extract<8>(7));
  }
}

static inline void store_nt_rows(int32 *pC, int mt, int nt,
                                 aie::vector<int32, MMUL::size_C> lo,
                                 aie::vector<int32, MMUL::size_C> hi) {
  int32 *base = pC + mt * NT * 8 * 16 + nt * 16;
  aie::store_v(base + 0 * NT * 16, join_rows(lo, hi, 0));
  aie::store_v(base + 1 * NT * 16, join_rows(lo, hi, 1));
  aie::store_v(base + 2 * NT * 16, join_rows(lo, hi, 2));
  aie::store_v(base + 3 * NT * 16, join_rows(lo, hi, 3));
  aie::store_v(base + 4 * NT * 16, join_rows(lo, hi, 4));
  aie::store_v(base + 5 * NT * 16, join_rows(lo, hi, 5));
  aie::store_v(base + 6 * NT * 16, join_rows(lo, hi, 6));
  aie::store_v(base + 7 * NT * 16, join_rows(lo, hi, 7));
}

extern "C" void r6_mac(const int8 *__restrict pA, const int8 *__restrict wbytes,
                       int32 *__restrict pC) {
  static_assert(NT == 4, "kernel unrolled for NT=4 accumulators");

  for (int mt = 0; mt < MT; mt++) {
    MMUL c0l, c0h, c1l, c1h, c2l, c2h, c3l, c3h;
    const int8 *abase = pA + mt * KCHUNK * MMUL::size_A;
    auto a = aie::load_v<MMUL::size_A>(abase);
    c0l.mul(a, ldW(wbytes, 0, 0, 0));
    c0h.mul(a, ldW(wbytes, 0, 1, 0));
    c1l.mul(a, ldW(wbytes, 1, 0, 0));
    c1h.mul(a, ldW(wbytes, 1, 1, 0));
    c2l.mul(a, ldW(wbytes, 2, 0, 0));
    c2h.mul(a, ldW(wbytes, 2, 1, 0));
    c3l.mul(a, ldW(wbytes, 3, 0, 0));
    c3h.mul(a, ldW(wbytes, 3, 1, 0));

    for (int k = 1; k < KCHUNK; k++) {
      a = aie::load_v<MMUL::size_A>(abase + k * MMUL::size_A);
      c0l.mac(a, ldW(wbytes, 0, 0, k));
      c0h.mac(a, ldW(wbytes, 0, 1, k));
      c1l.mac(a, ldW(wbytes, 1, 0, k));
      c1h.mac(a, ldW(wbytes, 1, 1, k));
      c2l.mac(a, ldW(wbytes, 2, 0, k));
      c2h.mac(a, ldW(wbytes, 2, 1, k));
      c3l.mac(a, ldW(wbytes, 3, 0, k));
      c3h.mac(a, ldW(wbytes, 3, 1, k));
    }

    auto v0l = c0l.template to_vector<int32>();
    auto v0h = c0h.template to_vector<int32>();
    auto v1l = c1l.template to_vector<int32>();
    auto v1h = c1h.template to_vector<int32>();
    auto v2l = c2l.template to_vector<int32>();
    auto v2h = c2h.template to_vector<int32>();
    auto v3l = c3l.template to_vector<int32>();
    auto v3h = c3h.template to_vector<int32>();
    store_nt_rows(pC, mt, 0, v0l, v0h);
    store_nt_rows(pC, mt, 1, v1l, v1h);
    store_nt_rows(pC, mt, 2, v2l, v2h);
    store_nt_rows(pC, mt, 3, v3l, v3h);
  }
}
