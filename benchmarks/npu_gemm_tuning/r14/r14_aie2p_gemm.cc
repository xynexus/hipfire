// AIE2P whole-array W4A8 block kernel. AIE2P's int4 primitive is
// <4,16,16>, so a 2x2 register tile has the same four-accumulator footprint
// as Phoenix's 3x3/<4,16,8> optimum without overflowing acc32 registers.
#include <aie_api/aie.hpp>

#define MT 2
#define NT 2
#define LM 6
#define LN 6
#define KT 16

using MMUL = aie::mmul<4, 16, 16, int8, int4>;
static constexpr int SA = MMUL::size_A;
static constexpr int SB = MMUL::size_B / 2;
static constexpr int SC = MMUL::size_C;

extern "C" void r14_aie2p_gemm(const int8 *__restrict activations,
                                const int8 *__restrict packed_weights,
                                int32 *__restrict output) {
  for (int im = 0; im < LM; im += MT)
    for (int jn = 0; jn < LN; jn += NT) {
      MMUL accumulators[MT][NT];
      {
        aie::vector<int8, SA> a[MT];
        aie::vector<int4, MMUL::size_B> w[NT];
#pragma unroll
        for (int i = 0; i < MT; i++)
          a[i] = aie::load_v<SA>(activations + ((im + i) * KT) * SA);
#pragma unroll
        for (int j = 0; j < NT; j++)
          w[j] = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(
              packed_weights + ((jn + j) * KT) * SB));
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
          for (int j = 0; j < NT; j++)
            accumulators[i][j].mul(a[i], w[j]);
      }
      for (int k = 1; k < KT; k++)
        chess_prepare_for_pipelining {
          aie::vector<int8, SA> a[MT];
          aie::vector<int4, MMUL::size_B> w[NT];
#pragma unroll
          for (int i = 0; i < MT; i++)
            a[i] = aie::load_v<SA>(activations + ((im + i) * KT + k) * SA);
#pragma unroll
          for (int j = 0; j < NT; j++)
            w[j] = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(
                packed_weights + ((jn + j) * KT + k) * SB));
#pragma unroll
          for (int i = 0; i < MT; i++)
#pragma unroll
            for (int j = 0; j < NT; j++)
              accumulators[i][j].mac(a[i], w[j]);
        }
#pragma unroll
      for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
          aie::store_v(output + ((im + i) * LN + (jn + j)) * SC,
                       accumulators[i][j].template to_vector<int32>());
    }
}
