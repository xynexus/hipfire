// AIE2P whole-array W4A8 GEMM with on-core f32 group reconstruction.
// The surrounding graph calls init for group zero and accum for later groups,
// retaining one final f32 C tile instead of returning int32 group partials.
#include <aie_api/aie.hpp>

#define LM 6
#define LN 6
#define KT 16

using MMUL = aie::mmul<4, 16, 16, int8, int4>;
static constexpr int SA = MMUL::size_A;
static constexpr int SB = MMUL::size_B / 2;
static constexpr int SC = MMUL::size_C;
static constexpr int AB = LM * KT * SA;
static constexpr int WB = LN * KT * SB;

template <bool ACCUMULATE>
static void scaled_impl(const int8 *__restrict activation_payload,
                        const int8 *__restrict weight_payload,
                        int32 *__restrict output_bits) {
  aie::set_rounding(aie::rounding_mode::floor);
  aie::set_saturation(aie::saturation_mode::none);
  const float *activation_scales =
      reinterpret_cast<const float *>(activation_payload + AB);
  const float *weight_scales =
      reinterpret_cast<const float *>(weight_payload + WB);
  float *output = reinterpret_cast<float *>(output_bits);

  for (int im = 0; im < LM; im++)
    for (int jn = 0; jn < LN; jn++) {
      auto sum = aie::zeros<int32, SC>();
      for (int k = 0; k < KT; k += 2) {
        auto a0 = aie::load_v<SA>(activation_payload + (im * KT + k) * SA);
        auto w0 = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(
            weight_payload + (jn * KT + k) * SB));
        MMUL partial;
        partial.mul(a0, w0);
        auto a1 = aie::load_v<SA>(activation_payload + (im * KT + k + 1) * SA);
        auto w1 = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(
            weight_payload + (jn * KT + k + 1) * SB));
        partial.mac(a1, w1);
        sum = aie::add(sum, partial.template to_vector<int32>());
      }
      auto weight_scale = aie::load_v<16>(weight_scales + jn * 16);
#pragma unroll
      for (int row = 0; row < 4; row++) {
        const int offset = (im * LN + jn) * SC + row * 16;
        auto values = aie::to_float(sum.template extract<16>(row));
        auto scaled = aie::mul(values, weight_scale).template to_vector<float>();
        scaled = aie::mul(
                     scaled,
                     aie::broadcast<float, 16>(activation_scales[im * 4 + row]))
                     .template to_vector<float>();
        if constexpr (ACCUMULATE)
          scaled = aie::add(scaled, aie::load_v<16>(output + offset));
        aie::store_v(output + offset, scaled);
      }
    }
}

extern "C" void r15_w4_scaled_dynamic(
    const int8 *__restrict activation_payload,
    const int8 *__restrict weight_payload, int32 *__restrict output_bits,
    int accumulate) {
  aie::set_rounding(aie::rounding_mode::floor);
  aie::set_saturation(aie::saturation_mode::none);
  const float *activation_scales =
      reinterpret_cast<const float *>(activation_payload + AB);
  const float *weight_scales =
      reinterpret_cast<const float *>(weight_payload + WB);
  float *output = reinterpret_cast<float *>(output_bits);

  for (int im = 0; im < LM; im++)
    for (int jn = 0; jn < LN; jn++) {
      auto sum = aie::zeros<int32, SC>();
      for (int k = 0; k < KT; k += 2) {
        auto a0 = aie::load_v<SA>(activation_payload + (im * KT + k) * SA);
        auto w0 = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(
            weight_payload + (jn * KT + k) * SB));
        MMUL partial;
        partial.mul(a0, w0);
        auto a1 = aie::load_v<SA>(activation_payload + (im * KT + k + 1) * SA);
        auto w1 = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(
            weight_payload + (jn * KT + k + 1) * SB));
        partial.mac(a1, w1);
        sum = aie::add(sum, partial.template to_vector<int32>());
      }
      auto weight_scale = aie::load_v<16>(weight_scales + jn * 16);
#pragma unroll
      for (int row = 0; row < 4; row++) {
        const int offset = (im * LN + jn) * SC + row * 16;
        auto values = aie::to_float(sum.template extract<16>(row));
        auto scaled = aie::mul(values, weight_scale).template to_vector<float>();
        scaled = aie::mul(
                     scaled,
                     aie::broadcast<float, 16>(activation_scales[im * 4 + row]))
                     .template to_vector<float>();
        if (accumulate)
          scaled = aie::add(scaled, aie::load_v<16>(output + offset));
        aie::store_v(output + offset, scaled);
      }
    }
}

#ifndef R15_DYNAMIC_ONLY
extern "C" void r15_w4_scaled_init(const int8 *__restrict activations,
                                    const int8 *__restrict weights,
                                    int32 *__restrict output) {
  scaled_impl<false>(activations, weights, output);
}

extern "C" void r15_w4_scaled_accum(const int8 *__restrict activations,
                                     const int8 *__restrict weights,
                                     int32 *__restrict output) {
  scaled_impl<true>(activations, weights, output);
}
#endif
