// Parallel exact activation pack feeding W4 down plus arbitrary compact W8 overlays.
#include "../r22/r22_w4_ring_down.cc"

constexpr int MIXED_OFFSET = GROUP_PARAM_OFFSET + 3 * GROUP * sizeof(float);

extern "C" void r24_mixed_scaled(const int8 *__restrict activations,
                                  const int8 *__restrict weights,
                                  int32 *__restrict output_bits,
                                  int accumulate, int overlays) {
  if (accumulate)
    r15_w4_scaled_accum(activations, weights, output_bits);
  else
    r15_w4_scaled_init(activations, weights, output_bits);

  const float *activation_scales =
      reinterpret_cast<const float *>(activations + A_DATA);
  const float *weight_scales =
      reinterpret_cast<const float *>(weights + W_DATA);
  const uint8_t *overlay =
      reinterpret_cast<const uint8_t *>(weights + MIXED_OFFSET);
  float *output = reinterpret_cast<float *>(output_bits);
  for (int im = 0; im < 6; im++)
    for (int row = 0; row < 4; row++) {
      const float activation_scale = activation_scales[im * 4 + row];
      for (int col = 0; col < W_COLS; col++) {
        int residual = 0;
        const uint8_t *entries = overlay + col * overlays * 2;
        for (int entry = 0; entry < overlays; entry++) {
          const int inner = entries[2 * entry];
          const int delta = static_cast<int8_t>(entries[2 * entry + 1]);
          const int kt = inner / 16;
          const int kk = inner % 16;
          const int activation_offset = (im * 16 + kt) * 64 + row * 16 + kk;
          residual += activations[activation_offset] * delta;
        }
        const int jn = col / 16;
        const int nn = col % 16;
        const int output_offset = (im * 6 + jn) * 64 + row * 16 + nn;
        output[output_offset] += residual * activation_scale * weight_scales[col];
      }
    }
}
