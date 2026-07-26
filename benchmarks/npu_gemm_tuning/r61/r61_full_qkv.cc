#include <aie_api/aie.hpp>
#include <stdint.h>

using MMUL = aie::mmul<4, 16, 16, int8, int4>;
constexpr unsigned LM = 6;
constexpr unsigned LN = 6;
constexpr unsigned KT = 16;
constexpr unsigned SC = MMUL::size_C;
constexpr unsigned A_SCALE_OFFSET = 6144;
constexpr unsigned W_SCALE_OFFSET = 12288;

static inline aie::vector<int8, MMUL::size_A>
load_r34_a(const int8_t *__restrict activations, unsigned im, unsigned kt) {
  const unsigned lm8 = im / 2;
  const unsigned local_row = (im % 2) * 4;
  const unsigned first = (lm8 * 32 + kt * 2) * 64 + local_row * 8;
  const auto lo = aie::load_v<32>(activations + first);
  const auto hi = aie::load_v<32>(activations + first + 64);
  return aie::concat(aie::interleave_zip(lo, hi, 8));
}

extern "C" void r61_full_qkv_block(const int8_t *__restrict activations,
                                    const int8_t *__restrict packed_weights,
                                    float *__restrict output,
                                    const int32_t accumulate) {
  alignas(32) int8_t local_activations[LM * KT * MMUL::size_A];
  for (unsigned im = 0; im < LM; ++im)
    for (unsigned kt = 0; kt < KT; ++kt)
      aie::store_v(local_activations + (im * KT + kt) * MMUL::size_A,
                   load_r34_a(activations, im, kt));

  const auto *activation_scales =
      reinterpret_cast<const float *>(activations + A_SCALE_OFFSET);
  const auto *weight_scales =
      reinterpret_cast<const float *>(packed_weights + W_SCALE_OFFSET);

  for (unsigned im = 0; im < LM; ++im)
    for (unsigned jn = 0; jn < LN; ++jn) {
      auto sum = aie::zeros<int32, SC>();
      for (unsigned kt = 0; kt < KT; kt += 2) {
        const auto *w0 = reinterpret_cast<const int4 *>(
            packed_weights + (jn * KT + kt) * 128);
        MMUL partial;
        partial.mul(aie::load_v<MMUL::size_A>(
                        local_activations + (im * KT + kt) * MMUL::size_A),
                    aie::load_v<MMUL::size_B>(w0));
        const auto *w1 = reinterpret_cast<const int4 *>(
            packed_weights + (jn * KT + kt + 1) * 128);
        partial.mac(aie::load_v<MMUL::size_A>(
                        local_activations + (im * KT + kt + 1) * MMUL::size_A),
                    aie::load_v<MMUL::size_B>(w1));
        sum = aie::add(sum, partial.template to_vector<int32>());
      }
      const auto weight_scale = aie::load_v<16>(weight_scales + jn * 16);
      for (unsigned row = 0; row < 4; ++row) {
        const unsigned offset = (im * LN + jn) * SC + row * 16;
        auto scaled =
            aie::mul(aie::to_float(sum.extract<16>(row)), weight_scale)
                .to_vector<float>();
        scaled = aie::mul(scaled, activation_scales[im * 4 + row])
                     .to_vector<float>();
        if (accumulate)
          scaled = aie::add(scaled, aie::load_v<16>(output + offset));
        aie::store_v(output + offset, scaled);
      }
    }
}

extern "C" void r61_full_qkv_init(const int8_t *__restrict activations,
                                   const int8_t *__restrict packed_weights,
                                   float *__restrict output) {
  r61_full_qkv_block(activations, packed_weights, output, 0);
}

extern "C" void r61_full_qkv_accum(const int8_t *__restrict activations,
                                    const int8_t *__restrict packed_weights,
                                    float *__restrict output) {
  r61_full_qkv_block(activations, packed_weights, output, 1);
}
