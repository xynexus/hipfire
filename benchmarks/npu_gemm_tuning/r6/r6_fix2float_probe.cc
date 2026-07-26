// Isolate AIE2P int32 -> float/BF16 conversion from GEMM/layout code.
#include <aie_api/aie.hpp>

extern "C" void r6_fix2float_probe(const int32 *__restrict input,
                                   const float *__restrict scales,
                                   float *__restrict output_f32,
                                   float *__restrict output_scaled) {
  auto integers = aie::load_v<16>(input);
  auto floats = aie::to_float(integers);
  auto scale = aie::load_v<16>(scales);
  auto scaled = aie::mul(floats, scale).template to_vector<float>();
  aie::store_v(output_f32, floats);
  aie::store_v(output_scaled, scaled);
}
