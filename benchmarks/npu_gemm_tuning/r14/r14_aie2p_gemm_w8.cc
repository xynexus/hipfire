// AIE2P whole-array dense W8A8 block kernel. Each logical 8x16 output
// tile uses two 8x8x8 MMULs; a 1x2 tile keeps four accumulators live.
#include <aie_api/aie.hpp>

#define MT 1
#define NT 2
#define LM 3
#define LN 3
#define KT 32

using MMUL = aie::mmul<8, 8, 8, int8, int8>;
static constexpr int SA = MMUL::size_A;
static constexpr int SB = 2 * MMUL::size_B;
static constexpr int SC = 2 * MMUL::size_C;

static inline aie::vector<int32, 16>
join_rows(aie::vector<int32, MMUL::size_C> lo,
          aie::vector<int32, MMUL::size_C> hi, int row) {
  switch (row) {
  case 0: return aie::concat(lo.template extract<8>(0), hi.template extract<8>(0));
  case 1: return aie::concat(lo.template extract<8>(1), hi.template extract<8>(1));
  case 2: return aie::concat(lo.template extract<8>(2), hi.template extract<8>(2));
  case 3: return aie::concat(lo.template extract<8>(3), hi.template extract<8>(3));
  case 4: return aie::concat(lo.template extract<8>(4), hi.template extract<8>(4));
  case 5: return aie::concat(lo.template extract<8>(5), hi.template extract<8>(5));
  case 6: return aie::concat(lo.template extract<8>(6), hi.template extract<8>(6));
  default: return aie::concat(lo.template extract<8>(7), hi.template extract<8>(7));
  }
}

extern "C" void r14_aie2p_gemm_w8(const int8 *__restrict activations,
                                   const int8 *__restrict weights,
                                   int32 *__restrict output) {
  for (int im = 0; im < LM; im += MT)
    for (int jn = 0; jn < LN; jn += NT) {
      MMUL accumulators[NT][2];
      {
        auto a = aie::load_v<SA>(activations + (im * KT) * SA);
#pragma unroll
        for (int j = 0; j < NT; j++) {
          const int8 *w = weights + ((jn + j) * KT) * SB;
          accumulators[j][0].mul(a, aie::load_v<MMUL::size_B>(w));
          accumulators[j][1].mul(a, aie::load_v<MMUL::size_B>(w + MMUL::size_B));
        }
      }
      for (int k = 1; k < KT; k++)
        chess_prepare_for_pipelining {
          auto a = aie::load_v<SA>(activations + (im * KT + k) * SA);
#pragma unroll
          for (int j = 0; j < NT; j++) {
            const int8 *w = weights + ((jn + j) * KT + k) * SB;
            accumulators[j][0].mac(a, aie::load_v<MMUL::size_B>(w));
            accumulators[j][1].mac(a, aie::load_v<MMUL::size_B>(w + MMUL::size_B));
          }
        }
#pragma unroll
      for (int j = 0; j < NT; j++) {
        auto lo = accumulators[j][0].template to_vector<int32>();
        auto hi = accumulators[j][1].template to_vector<int32>();
#pragma unroll
        for (int row = 0; row < 8; row++)
          aie::store_v(output + ((im * LN + jn + j) * SC) + row * 16,
                       join_rows(lo, hi, row));
      }
    }
}
