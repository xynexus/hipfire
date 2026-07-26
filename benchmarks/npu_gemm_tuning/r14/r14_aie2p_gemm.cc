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
    for (int jn = 0; jn < LN; jn += NT)
      for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++) {
          // The native int8/int4 MMUL chain becomes invalid once realistic
          // FWHT activations drive a single chain beyond about signed-i16
          // range, despite exposing acc32. Keep each MMUL chain to K=32 and
          // accumulate its exact int32 vectors explicitly.
          auto sum = aie::zeros<int32, SC>();
          for (int k = 0; k < KT; k += 2) {
            auto a0 = aie::load_v<SA>(activations + ((im + i) * KT + k) * SA);
            auto w0 = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(
                packed_weights + ((jn + j) * KT + k) * SB));
            MMUL partial;
            partial.mul(a0, w0);
            auto a1 = aie::load_v<SA>(activations + ((im + i) * KT + k + 1) * SA);
            auto w1 = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(
                packed_weights + ((jn + j) * KT + k + 1) * SB));
            partial.mac(a1, w1);
            sum = aie::add(sum, partial.template to_vector<int32>());
          }
          aie::store_v(output + ((im + i) * LN + (jn + j)) * SC,
                       sum);
        }
}
