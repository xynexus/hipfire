// AIE2P whole-array W8A8 GEMM with on-core f32 group reconstruction.
#include <aie_api/aie.hpp>

#define LM 3
#define LN 3
#define KT 32
using MMUL = aie::mmul<8, 8, 8, int8, int8>;
static constexpr int SA = MMUL::size_A;
static constexpr int SB = 2 * MMUL::size_B;
static constexpr int SC = 2 * MMUL::size_C;
static constexpr int A_DATA = LM * KT * SA;
static constexpr int W_DATA = LN * KT * SB;

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

template <bool ACCUMULATE>
static void scaled_impl(const int8 *__restrict activations,
                        const int8 *__restrict weights,
                        int32 *__restrict output_bits) {
  const float *activation_scales =
      reinterpret_cast<const float *>(activations + A_DATA);
  const float *weight_scales =
      reinterpret_cast<const float *>(weights + W_DATA);
  float *output = reinterpret_cast<float *>(output_bits);
  for (int im = 0; im < LM; im++)
    for (int jn = 0; jn < LN; jn++) {
      MMUL lo, hi;
      auto a = aie::load_v<SA>(activations + (im * KT) * SA);
      const int8 *w = weights + (jn * KT) * SB;
      lo.mul(a, aie::load_v<MMUL::size_B>(w));
      hi.mul(a, aie::load_v<MMUL::size_B>(w + MMUL::size_B));
      for (int k = 1; k < KT; k++) {
        a = aie::load_v<SA>(activations + (im * KT + k) * SA);
        w = weights + (jn * KT + k) * SB;
        lo.mac(a, aie::load_v<MMUL::size_B>(w));
        hi.mac(a, aie::load_v<MMUL::size_B>(w + MMUL::size_B));
      }
      auto vlo = lo.template to_vector<int32>();
      auto vhi = hi.template to_vector<int32>();
      auto weight_scale = aie::load_v<16>(weight_scales + jn * 16);
#pragma unroll
      for (int row = 0; row < 8; row++) {
        const int offset = (im * LN + jn) * SC + row * 16;
        auto values = aie::to_float(join_rows(vlo, vhi, row));
        auto scaled = aie::mul(values, weight_scale).template to_vector<float>();
        scaled = aie::mul(
                     scaled,
                     aie::broadcast<float, 16>(activation_scales[im * 8 + row]))
                     .template to_vector<float>();
        if constexpr (ACCUMULATE)
          scaled = aie::add(scaled, aie::load_v<16>(output + offset));
        aie::store_v(output + offset, scaled);
      }
    }
}

extern "C" void r15_w8_scaled_init(const int8 *a, const int8 *w, int32 *c) {
  scaled_impl<false>(a, w, c);
}
extern "C" void r15_w8_scaled_accum(const int8 *a, const int8 *w, int32 *c) {
  scaled_impl<true>(a, w, c);
}
