// Observation probe: record the scale payload EVERY group receives.
//
// r6_scale_dump_probe.cc only implements the !ACCUMULATE branch, so it observes
// group 0 (r6_scale_init) and nothing else. Groups 1..KGROUPS-1 arrive through
// r6_scale_accum and had never been observed — which is exactly where the
// per-group weight scale stops being applied.
//
// The kernel is not told its group index, so derive it: the scale core calls
// init once per slab then accum for each remaining group, so a static counter
// reset in init counts groups within a slab. Each group writes its first
// activation and weight scale to output[2*group], output[2*group+1]. The output
// DMA maps tile offset 0..63 to row 0, columns slab*64.. of this core's C
// region, so the host reads them back as out[0..2*KGROUPS].
#include <aie_api/aie.hpp>

#ifndef MT
#define MT 8
#endif
static constexpr int ROWS = MT * 4;
// ROWS_PADDED_DEF: see r6_scale_accum.cc — 64-byte alignment for load_v<16>.
static constexpr int ROWS_PADDED = ((ROWS + 15) / 16) * 16;

static int group_index = 0;

template <bool ACCUMULATE>
static void scale_impl(const int32 *__restrict integers,
                       const int8 *__restrict scale_payload,
                       int32 *__restrict output_bits) {
  const float *activation_scales =
      reinterpret_cast<const float *>(scale_payload);
  const float *weight_scales = activation_scales + ROWS_PADDED;
  float *output = reinterpret_cast<float *>(output_bits);
  if constexpr (ACCUMULATE) {
    group_index++;
  } else {
    group_index = 0;
    for (int i = 0; i < 64; i++)
      output[i] = 0.0f;
  }
  // Third slot is the first INTEGER this group was handed. If the @fr stream is
  // permuted relative to @fs, the sum stays right under uniform scales
  // (addition commutes) but per-group scales pair the wrong partial with the
  // wrong scale — which is exactly the observed signature.
  const int slot = 3 * group_index;
  if (slot + 2 < 64) {
    output[slot] = activation_scales[0];
    output[slot + 1] = weight_scales[0];
    output[slot + 2] = (float)integers[0];
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
