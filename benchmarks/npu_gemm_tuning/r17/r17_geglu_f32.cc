// AIE2P GeGLU transition for EmbeddingGemma's resident FFN.
// Input/output are f32; the hardware tanh result is bf16, matching the AIE2P
// nonlinear primitive while keeping the surrounding polynomial/multiply in f32.
#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>

extern "C" void r17_geglu_f32(const float *__restrict input,
                               float *__restrict output) {
  constexpr int N = 1152;
  const float *__restrict gate = input;
  const float *__restrict up = input + N;
  auto half = aie::broadcast<float, 16>(0.5f);
  auto one = aie::broadcast<float, 16>(1.0f);
  auto beta = aie::broadcast<float, 16>(0.044715f);
  auto sqrt_2_over_pi = aie::broadcast<float, 16>(0.7978845608028654f);
  auto one_bf16 = aie::broadcast<bfloat16, 16>(1.0f);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int offset = 0; offset < N; offset += 16) {
    auto g = aie::load_v<16>(gate + offset);
    auto u = aie::load_v<16>(up + offset);
    auto g2 = aie::mul(g, g).template to_vector<float>();
    auto g3 = aie::mul(g2, g).template to_vector<float>();
    auto cubic = aie::mul(g3, beta).template to_vector<float>();
    auto inner = aie::add(g, cubic);
    auto tanh_arg = aie::mul(inner, sqrt_2_over_pi).template to_vector<float>();
    auto tanh_bf16 = aie::tanh<bfloat16>(tanh_arg);
    auto tanh_f32 = aie::mul(tanh_bf16, one_bf16).template to_vector<float>();
    auto cdf = aie::mul(aie::add(one, tanh_f32), half).template to_vector<float>();
    auto gelu = aie::mul(g, cdf).template to_vector<float>();
    auto result = aie::mul(gelu, u).template to_vector<float>();
    aie::store_v(output + offset, result);
  }
}
