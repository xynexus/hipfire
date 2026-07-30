// Scale and accumulate one exact int32 R6 slab on the following AIE row.
#include <aie_api/aie.hpp>

#ifndef MT
#define MT 8
#endif
static constexpr int ROWS = MT * 4;
// ROWS_PADDED_DEF: see r6_scale_accum.cc — 64-byte alignment for load_v<16>.
static constexpr int ROWS_PADDED = ((ROWS + 15) / 16) * 16;

template <bool ACCUMULATE>
static void scale_impl(const int32 *__restrict integers,
                       const int8 *__restrict scale_payload,
                       int32 *__restrict output_bits) {
  const float *activation_scales =
      reinterpret_cast<const float *>(scale_payload);
  const float *weight_scales = activation_scales + ROWS_PADDED;
  float *output = reinterpret_cast<float *>(output_bits);
  // OBSERVATION PROBE: ignore the GEMM entirely and copy the scale payload this
  // core actually receives into the output, so the host can compare it against
  // what `copy_scale_payload` wrote. Distinguishes a ROUTING fault (the core
  // gets the wrong bytes) from an INTERPRETATION fault (right bytes, wrong math)
  // — which no amount of arithmetic variation could separate.
  if constexpr (!ACCUMULATE) {
    for (int i = 0; i < ROWS + 64; i++)
      output[i] = activation_scales[i];
    for (int i = ROWS + 64; i < 64; i++)
      output[i] = 0.0f;
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
