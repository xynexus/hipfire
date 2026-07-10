// W4 scaled whole-array projection followed by tile-local GeGLU.
#include "../r15/r15_w4_scaled.cc"
#include "aie_kernels/aie_kernel_utils.h"

static inline aie::vector<float, 16> r18_geglu16(aie::vector<float, 16> gate,
                                                  aie::vector<float, 16> up) {
  auto one = aie::broadcast<float, 16>(1.0f);
  auto half = aie::broadcast<float, 16>(0.5f);
  auto beta = aie::broadcast<float, 16>(0.044715f);
  auto factor = aie::broadcast<float, 16>(0.7978845608028654f);
  auto g2 = aie::mul(gate, gate).template to_vector<float>();
  auto g3 = aie::mul(g2, gate).template to_vector<float>();
  auto inner = aie::add(gate, aie::mul(g3, beta).template to_vector<float>());
  auto arg = aie::mul(inner, factor).template to_vector<float>();
  auto tanh_bf16 = aie::tanh<bfloat16>(arg);
  auto bf16_one = aie::broadcast<bfloat16, 16>(1.0f);
  auto tanh_f32 = aie::mul(tanh_bf16, bf16_one).template to_vector<float>();
  auto cdf = aie::mul(aie::add(one, tanh_f32), half).template to_vector<float>();
  return aie::mul(aie::mul(gate, cdf).template to_vector<float>(), up)
      .template to_vector<float>();
}

extern "C" void r18_w4_geglu(const int32 *__restrict accumulator_bits,
                              int32 *__restrict output_bits) {
  const float *accumulator = reinterpret_cast<const float *>(accumulator_bits);
  float *output = reinterpret_cast<float *>(output_bits);
  constexpr int SC = 64;
  constexpr int OUT_WIDTH = 48;
  AIE_PREPARE_FOR_PIPELINING
  for (int im = 0; im < 6; im++)
    for (int row = 0; row < 4; row++)
      for (int jn = 0; jn < 3; jn++) {
        const int gate_offset = (im * 6 + jn) * SC + row * 16;
        const int up_offset = (im * 6 + jn + 3) * SC + row * 16;
        auto gate = aie::load_v<16>(accumulator + gate_offset);
        auto up = aie::load_v<16>(accumulator + up_offset);
        aie::store_v(output + (im * 4 + row) * OUT_WIDTH + jn * 16,
                     r18_geglu16(gate, up));
      }
}
