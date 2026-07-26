// R6-TS-W8: M-parallel tensor-stream GEMM with int8 activations and int8 weights
// (W8A8). This mirrors r6_gemm_ts.cc's row-major A/C contract and objectfifo
// layout, but doubles the W tile payload. AIE2P's direct 4x16x8 int8 shape uses
// sparse-B storage, so the dense path computes one 4x16x16 K tile as four
// dense 4x8x8 mmuls: two K halves times two N halves. KCHUNK>1 needs host-side
// grouping for now; loop-carried AIE2P 4x8x8 accumulators produced row leakage.
//
// pA: TILE-MAJOR MT x KCHUNK x two packed 4x8 int8 A halves.
// pW: TILE-MAJOR NT x KCHUNK x two K-halves x two N-halves x 8x8 int8 B tiles.
// pC: ROW-MAJOR (MT*4) x (NT*16) int32.
#include <aie_api/aie.hpp>

#ifndef MT
#define MT 8
#endif
#ifndef NT
#define NT 4
#endif
#ifndef KCHUNK
#define KCHUNK 8
#endif

using MMUL = aie::mmul<4, 8, 8, int8, int8>;

static inline aie::vector<int8, MMUL::size_B>
ldW(const int8 *wbytes, int nt, int k_half, int n_half, int k) {
  return aie::load_v<MMUL::size_B>(
      wbytes + ((nt * KCHUNK + k) * 4 + k_half * 2 + n_half) * MMUL::size_B);
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
  default:
    return aie::concat(lo.template extract<8>(3), hi.template extract<8>(3));
  }
}

extern "C" void r6_mac(const int8 *__restrict pA, const int8 *__restrict wbytes,
                       int32 *__restrict pC) {
  static_assert(NT == 4, "kernel unrolled for NT=4 accumulators");
  static_assert(KCHUNK == 1, "W8 dense kernel currently requires KCHUNK=1");

  auto c_desc = aie::make_tensor_descriptor<int32, 16>(
      aie::tensor_dim(MT, 4 * NT), aie::tensor_dim(NT, 1), aie::tensor_dim(4u, NT));
  auto tsC = aie::make_tensor_buffer_stream(pC, c_desc);

  for (int mt = 0; mt < MT; mt++) {
    MMUL c0l, c0h, c1l, c1h, c2l, c2h, c3l, c3h;
    const int8 *abase = pA + mt * KCHUNK * 64;
    auto a0 = aie::load_v<MMUL::size_A>(abase);
    auto a1 = aie::load_v<MMUL::size_A>(abase + MMUL::size_A);
    c0l.mul(a0, ldW(wbytes, 0, 0, 0, 0));
    c0l.mac(a1, ldW(wbytes, 0, 1, 0, 0));
    c0h.mul(a0, ldW(wbytes, 0, 0, 1, 0));
    c0h.mac(a1, ldW(wbytes, 0, 1, 1, 0));
    c1l.mul(a0, ldW(wbytes, 1, 0, 0, 0));
    c1l.mac(a1, ldW(wbytes, 1, 1, 0, 0));
    c1h.mul(a0, ldW(wbytes, 1, 0, 1, 0));
    c1h.mac(a1, ldW(wbytes, 1, 1, 1, 0));
    c2l.mul(a0, ldW(wbytes, 2, 0, 0, 0));
    c2l.mac(a1, ldW(wbytes, 2, 1, 0, 0));
    c2h.mul(a0, ldW(wbytes, 2, 0, 1, 0));
    c2h.mac(a1, ldW(wbytes, 2, 1, 1, 0));
    c3l.mul(a0, ldW(wbytes, 3, 0, 0, 0));
    c3l.mac(a1, ldW(wbytes, 3, 1, 0, 0));
    c3h.mul(a0, ldW(wbytes, 3, 0, 1, 0));
    c3h.mac(a1, ldW(wbytes, 3, 1, 1, 0));
    auto v0l = c0l.template to_vector<int32>();
    auto v0h = c0h.template to_vector<int32>();
    auto v1l = c1l.template to_vector<int32>();
    auto v1h = c1h.template to_vector<int32>();
    auto v2l = c2l.template to_vector<int32>();
    auto v2h = c2h.template to_vector<int32>();
    auto v3l = c3l.template to_vector<int32>();
    auto v3h = c3h.template to_vector<int32>();
    tsC << join_rows(v0l, v0h, 0) << join_rows(v0l, v0h, 1)
        << join_rows(v0l, v0h, 2) << join_rows(v0l, v0h, 3);
    tsC << join_rows(v1l, v1h, 0) << join_rows(v1l, v1h, 1)
        << join_rows(v1l, v1h, 2) << join_rows(v1l, v1h, 3);
    tsC << join_rows(v2l, v2h, 0) << join_rows(v2l, v2h, 1)
        << join_rows(v2l, v2h, 2) << join_rows(v2l, v2h, 3);
    tsC << join_rows(v3l, v3h, 0) << join_rows(v3l, v3h, 1)
        << join_rows(v3l, v3h, 2) << join_rows(v3l, v3h, 3);
  }
}
