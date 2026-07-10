// W8 scaled whole-array projection followed by tile-local GeGLU.
#include "../r15/r15_w8_scaled.cc"
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

__attribute__((noinline)) static void
r18_geglu_store16(const float *__restrict gate,
                  const float *__restrict up,
                  float *__restrict output) {
  aie::store_v(output,
               r18_geglu16(aie::load_v<16>(gate), aie::load_v<16>(up)));
}

__attribute__((noinline)) static void
r18_geglu_store8(const float *__restrict gate_up,
                 float *__restrict output) {
  auto zero = aie::zeros<float, 8>();
  auto gate = aie::concat(aie::load_v<8>(gate_up), zero);
  auto up = aie::concat(aie::load_v<8>(gate_up + 8), zero);
  aie::store_v(output, r18_geglu16(gate, up));
}

extern "C" void r18_w8_geglu(const int32 *__restrict accumulator_bits,
                              int32 *__restrict output_bits) {
  const float *accumulator = reinterpret_cast<const float *>(accumulator_bits);
  float *output = reinterpret_cast<float *>(output_bits);
  constexpr int SC = 128;
  constexpr int OUT_WIDTH = 32;
  AIE_PREPARE_FOR_PIPELINING
  for (int im = 0; im < 3; im++)
    for (int row = 0; row < 8; row++) {
      const int j0 = (im * 3) * SC + row * 16;
      const int j1 = (im * 3 + 1) * SC + row * 16;
      const int j2 = (im * 3 + 2) * SC + row * 16;
      float *dst = output + (im * 8 + row) * OUT_WIDTH;
      r18_geglu_store16(accumulator + j0, accumulator + j1, dst);
      r18_geglu_store8(accumulator + j2, dst + 16);
    }
}
