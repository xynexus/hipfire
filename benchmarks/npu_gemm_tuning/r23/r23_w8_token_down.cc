// Parallel exact Opus activation pack feeding the resident scaled W8 down MMUL.
#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>

constexpr int W8_LM = 3;
constexpr int W8_LN = 3;
constexpr int W8_KT = 32;
using W8_MMUL = aie::mmul<8, 8, 8, int8, int8>;
constexpr int W8_SA = W8_MMUL::size_A;
constexpr int W8_SB = 2 * W8_MMUL::size_B;
constexpr int W8_SC = 2 * W8_MMUL::size_C;
constexpr int A_DATA = W8_LM * W8_KT * W8_SA;
constexpr int W_DATA = W8_LN * W8_KT * W8_SB;
constexpr int GROUP = 256;
constexpr int W_COLS = 48;
constexpr int GROUP_PARAM_OFFSET = W_DATA + W_COLS * sizeof(float);
constexpr int FRAGMENT_ROWS = 3;
constexpr int FRAGMENT_BYTES = FRAGMENT_ROWS * GROUP + 16;
constexpr int FRAGMENT_WORDS = FRAGMENT_BYTES / sizeof(int);

static inline aie::vector<int32, 16>
join_rows(aie::vector<int32, W8_MMUL::size_C> lo,
          aie::vector<int32, W8_MMUL::size_C> hi, int row) {
  switch (row) {
  case 0: return aie::concat(lo.template extract<8>(0), hi.template extract<8>(0));
  case 1: return aie::concat(lo.template extract<8>(1), hi.template extract<8>(1));
  case 2: return aie::concat(lo.template extract<8>(2), hi.template extract<8>(2));
  case 3: return aie::concat(lo.template extract<8>(3), hi.template extract<8>(3));
  case 4: return aie::concat(lo.template extract<8>(4), hi.template extract<8>(4));
  case 5: return aie::concat(lo.template extract<8>(5), hi.template extract<8>(5));
  case 6: return aie::concat(lo.template extract<8>(6), hi.template extract<8>(6));
  default: return aie::concat(lo.template extract<8>(7), hi.template extract<8>(7));
  }
}

extern "C" void r23_w8_scaled(const int8 *__restrict activations,
                               const int8 *__restrict weights,
                               int32 *__restrict output_bits,
                               int accumulate) {
  const float *activation_scales =
      reinterpret_cast<const float *>(activations + A_DATA);
  const float *weight_scales =
      reinterpret_cast<const float *>(weights + W_DATA);
  float *output = reinterpret_cast<float *>(output_bits);
  for (int im = 0; im < W8_LM; im++)
    for (int jn = 0; jn < W8_LN; jn++) {
      W8_MMUL lo, hi;
      auto a = aie::load_v<W8_SA>(activations + (im * W8_KT) * W8_SA);
      const int8 *w = weights + (jn * W8_KT) * W8_SB;
      lo.mul(a, aie::load_v<W8_MMUL::size_B>(w));
      hi.mul(a, aie::load_v<W8_MMUL::size_B>(w + W8_MMUL::size_B));
      for (int k = 1; k < W8_KT; k++) {
        a = aie::load_v<W8_SA>(activations + (im * W8_KT + k) * W8_SA);
        w = weights + (jn * W8_KT + k) * W8_SB;
        lo.mac(a, aie::load_v<W8_MMUL::size_B>(w));
        hi.mac(a, aie::load_v<W8_MMUL::size_B>(w + W8_MMUL::size_B));
      }
      auto vlo = lo.template to_vector<int32>();
      auto vhi = hi.template to_vector<int32>();
      auto weight_scale = aie::load_v<16>(weight_scales + jn * 16);
      for (int row = 0; row < 8; row++) {
        const int offset = (im * W8_LN + jn) * W8_SC + row * 16;
        auto values = aie::to_float(join_rows(vlo, vhi, row));
        auto scaled = aie::mul(values, weight_scale).template to_vector<float>();
        scaled = aie::mul(
                     scaled,
                     aie::broadcast<float, 16>(activation_scales[im * 8 + row]))
                     .template to_vector<float>();
        if (accumulate) scaled = aie::add(scaled, aie::load_v<16>(output + offset));
        aie::store_v(output + offset, scaled);
      }
    }
}

template <unsigned STRIDE>
__attribute__((noinline)) static void fwht16_stage(float *__restrict scratch) {
  for (int block = 0; block < GROUP; block += 16) {
    auto values = aie::load_v<16>(scratch + block);
    auto a = aie::filter_even(values, STRIDE);
    auto b = aie::filter_odd(values, STRIDE);
    aie::store_v(scratch + block,
                 aie::concat(aie::interleave_zip(aie::add(a, b),
                                                 aie::sub(a, b), STRIDE)));
  }
}

static void pack_row(const float *__restrict input,
                     const int8 *__restrict weight_payload,
                     int8 *__restrict quantized, float *__restrict scratch,
                     float &scale) {
  const float *params = reinterpret_cast<const float *>(
      weight_payload + GROUP_PARAM_OFFSET);
  const float *awq = params;
  const float *signs1 = awq + GROUP;
  const float *signs2 = signs1 + GROUP;
  for (int i = 0; i < GROUP; i += 16) {
    auto divided = aie::div(aie::load_v<16>(input + i),
                            aie::load_v<16>(awq + i))
                       .template to_vector<float>();
    aie::store_v(scratch + i,
                 aie::mul(divided, aie::load_v<16>(signs1 + i))
                     .template to_vector<float>());
  }
  fwht16_stage<1>(scratch);
  fwht16_stage<2>(scratch);
  fwht16_stage<4>(scratch);
  fwht16_stage<8>(scratch);
  for (int stride = 16; stride < GROUP; stride <<= 1)
    for (int block = 0; block < GROUP; block += 2 * stride)
      for (int i = 0; i < stride; i += 16) {
        auto a = aie::load_v<16>(scratch + block + i);
        auto b = aie::load_v<16>(scratch + block + i + stride);
        aie::store_v(scratch + block + i, aie::add(a, b));
        aie::store_v(scratch + block + i + stride, aie::sub(a, b));
      }
  float max_abs = 0.0f;
  for (int i = 0; i < GROUP; i += 16) {
    auto scaled =
        aie::mul(aie::mul(aie::load_v<16>(scratch + i),
                          aie::load_v<16>(signs2 + i))
                     .template to_vector<float>(),
                 aie::broadcast<float, 16>(0.0625f))
            .template to_vector<float>();
    aie::store_v(scratch + i, scaled);
    const float local_max = aie::reduce_max(aie::abs(scaled));
    if (local_max > max_abs) max_abs = local_max;
  }
  scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
  const auto old_rounding =
      aie::swap_rounding(aie::rounding_mode::symmetric_inf);
  const auto old_saturation =
      aie::swap_saturation(aie::saturation_mode::symmetric);
  for (int i = 0; i < GROUP; i += 16) {
    auto output = scale > 0.0f
                      ? aie::to_fixed<int8>(
                            aie::div(aie::load_v<16>(scratch + i), scale)
                                .template to_vector<float>())
                      : aie::zeros<int8, 16>();
    aie::store_v(quantized + i, output);
  }
  aie::set_saturation(old_saturation);
  aie::set_rounding(old_rounding);
}

extern "C" void r23_pack3(const float *__restrict input,
                           const int8 *__restrict weight_payload,
                           int8 *__restrict activation_payload,
                           float *__restrict scratch,
                           int8 *__restrict fragment, int owner) {
  (void)activation_payload;
  const float *owned_input = input + (owner & 1) * FRAGMENT_ROWS * GROUP;
  float *scales = reinterpret_cast<float *>(fragment + FRAGMENT_ROWS * GROUP);
  for (int row = 0; row < FRAGMENT_ROWS; row++)
    pack_row(owned_input + row * GROUP, weight_payload,
             fragment + row * GROUP, scratch, scales[row]);
  reinterpret_cast<int *>(fragment)[FRAGMENT_WORDS - 1] = 0;
}

extern "C" void r23_insert_fragment(const int8 *__restrict fragment,
                                     int8 *__restrict activation_payload,
                                     int owner) {
  float *scales = reinterpret_cast<float *>(activation_payload + A_DATA);
  for (int row = 0; row < FRAGMENT_ROWS; row++) {
    const int local_row = owner * FRAGMENT_ROWS + row;
    const int lm = local_row / 8;
    const int rr = local_row % 8;
    for (int kt = 0; kt < 32; kt++) {
      const int target = (lm * 32 + kt) * 64 + rr * 8;
      const int source = row * GROUP + kt * 8;
      reinterpret_cast<int *>(activation_payload + target)[0] =
          reinterpret_cast<const int *>(fragment + source)[0];
      reinterpret_cast<int *>(activation_payload + target)[1] =
          reinterpret_cast<const int *>(fragment + source)[1];
    }
    scales[local_row] =
        reinterpret_cast<const float *>(fragment + FRAGMENT_ROWS * GROUP)[row];
  }
}

extern "C" void r23_send_fragment(const int8 *__restrict fragment) {
  const int *words = reinterpret_cast<const int *>(fragment);
  for (int word = 0; word < FRAGMENT_WORDS; word++) put_ms(words[word]);
}

extern "C" void r23_receive_fragment(int8 *__restrict fragment) {
  int *words = reinterpret_cast<int *>(fragment);
  for (int word = 0; word < FRAGMENT_WORDS; word++) words[word] = get_ss_int();
}
