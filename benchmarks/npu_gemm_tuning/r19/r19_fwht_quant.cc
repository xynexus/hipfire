// Canonical Opus activation preprocessing for one padded K=1280 row.
// param = [awq(1280), signs1(256), signs2(256)].
#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>

__attribute__((noinline)) static void
awq_sign8(const float *__restrict input, const float *__restrict awq,
          const float *__restrict signs, float *__restrict scratch) {
  scratch[0] = (input[0] / awq[0]) * signs[0];
  scratch[1] = (input[1] / awq[1]) * signs[1];
  scratch[2] = (input[2] / awq[2]) * signs[2];
  scratch[3] = (input[3] / awq[3]) * signs[3];
  scratch[4] = (input[4] / awq[4]) * signs[4];
  scratch[5] = (input[5] / awq[5]) * signs[5];
  scratch[6] = (input[6] / awq[6]) * signs[6];
  scratch[7] = (input[7] / awq[7]) * signs[7];
}

__attribute__((noinline)) static float
post_sign8(float *__restrict scratch, const float *__restrict signs) {
  float max_abs = 0.0f;
  for (int i = 0; i < 8; i++) {
    scratch[i] *= 0.0625f * signs[i];
    const float magnitude = __builtin_fabsf(scratch[i]);
    if (magnitude > max_abs) max_abs = magnitude;
  }
  return max_abs;
}

extern "C" void r19_fwht_quant(const float *__restrict input,
                                const float *__restrict param,
                                int8 *__restrict output,
                                float *__restrict scratch) {
  constexpr int GROUP = 256;
  constexpr int GROUPS = 5;
  constexpr int PAD_K = GROUP * GROUPS;
  const float *awq = param;
  const float *signs1 = param + PAD_K;
  const float *signs2 = signs1 + GROUP;
  float *scales = reinterpret_cast<float *>(output + PAD_K);
  for (int group = 0; group < GROUPS; group++) {
    const int base = group * GROUP;
    for (int i = 0; i < GROUP; i += 8)
      awq_sign8(input + base + i, awq + base + i, signs1 + i, scratch + i);

    for (int stride = 1; stride < GROUP; stride <<= 1)
      for (int block = 0; block < GROUP; block += 2 * stride)
        for (int i = 0; i < stride; i++) {
          const float a = scratch[block + i];
          const float b = scratch[block + i + stride];
          scratch[block + i] = a + b;
          scratch[block + i + stride] = a - b;
        }

    float max_abs = 0.0f;
    for (int i = 0; i < GROUP; i += 8) {
      const float local_max = post_sign8(scratch + i, signs2 + i);
      if (local_max > max_abs) max_abs = local_max;
    }
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
    scales[group] = scale;

    const auto old_rounding = aie::swap_rounding(aie::rounding_mode::symmetric_floor);
    const auto old_saturation = aie::swap_saturation(aie::saturation_mode::symmetric);
    if (scale > 0.0f) {
      for (int i = 0; i < GROUP; i++) {
        const float normalized = scratch[i] / scale;
        const float biased = normalized + (normalized >= 0.0f ? 0.5f : -0.5f);
        output[base + i] = aie::to_fixed<int8>(biased);
      }
    } else {
      for (int i = 0; i < GROUP; i++) output[base + i] = 0;
    }
    aie::set_saturation(old_saturation);
    aie::set_rounding(old_rounding);
  }
  for (int i = GROUPS; i < 8; i++) scales[i] = 0.0f;
}
