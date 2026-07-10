// Scale and accumulate one exact int32 R6 slab on the following AIE row.
#include <aie_api/aie.hpp>

#ifndef MT
#define MT 8
#endif
static constexpr int ROWS = MT * 4;

template <bool ACCUMULATE>
static void scale_impl(const int32 *__restrict integers,
                       const int8 *__restrict scale_payload,
                       int32 *__restrict output_bits) {
  const float *activation_scales =
      reinterpret_cast<const float *>(scale_payload);
  const float *weight_scales = activation_scales + ROWS;
  float *output = reinterpret_cast<float *>(output_bits);
  for (int row = 0; row < ROWS; row++) {
    auto row_scale = aie::broadcast<float, 16>(activation_scales[row]);
    for (int block = 0; block < 4; block++) {
      const int offset = row * 64 + block * 16;
      auto weight_scale = aie::load_v<16>(weight_scales + block * 16);
      auto integer_values = aie::load_v<16>(integers + offset);
      auto values = aie::to_float(integer_values);
      auto scaled = aie::mul(values, weight_scale).template to_vector<float>();
      scaled = aie::mul(scaled, row_scale).template to_vector<float>();
      if constexpr (ACCUMULATE)
        scaled = aie::add(scaled, aie::load_v<16>(output + offset));
      aie::store_v(output + offset, scaled);
    }
  }
}

extern "C" void r6_scale_init(const int32 *__restrict integers,
                              const int8 *__restrict scale_payload,
                              int32 *__restrict output_bits) {
  scale_impl<false>(integers, scale_payload, output_bits);
}

extern "C" void r6_scale_accum(const int32 *__restrict integers,
                               const int8 *__restrict scale_payload,
                               int32 *__restrict output_bits) {
  scale_impl<true>(integers, scale_payload, output_bits);
}
