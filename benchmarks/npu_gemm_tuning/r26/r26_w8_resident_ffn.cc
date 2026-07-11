// Complete resident dense-W8 gate/up + GeGLU -> activation pack -> down FFN.
#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>

constexpr int R26_LM = 3;
constexpr int R26_LN = 3;
constexpr int R26_KT = 32;
using R26_MMUL = aie::mmul<8, 8, 8, int8, int8>;
constexpr int R26_SA = R26_MMUL::size_A;
constexpr int R26_SB = 2 * R26_MMUL::size_B;
constexpr int R26_SC = 2 * R26_MMUL::size_C;
constexpr int R26_A_DATA = R26_LM * R26_KT * R26_SA;
constexpr int R26_W_DATA = R26_LN * R26_KT * R26_SB;
constexpr int R26_GROUP = 256;
constexpr int R26_W_COLS = 48;
constexpr int R26_PARAM_OFFSET = R26_W_DATA + R26_W_COLS * sizeof(float);
constexpr int R26_FRAGMENT_ROWS = 3;
constexpr int R26_FRAGMENT_BYTES = R26_FRAGMENT_ROWS * R26_GROUP + 16;
constexpr int R26_FRAGMENT_WORDS = R26_FRAGMENT_BYTES / sizeof(int);

static inline aie::vector<int32, 16>
r26_join_rows(aie::vector<int32, R26_MMUL::size_C> lo,
              aie::vector<int32, R26_MMUL::size_C> hi, int row) {
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

// `output_ln=3,nmacro=0` is the gate/up tile. `output_ln=6,nmacro=0/1`
// retains both N=384 down macros in one DMA-friendly accumulator.
template <int OUTPUT_LN, int NMACRO>
__attribute__((noinline, minsize)) static void
r26_w8_scaled(const int8 *__restrict activations,
              const int8 *__restrict weights, int32 *__restrict output_bits,
              int accumulate) {
  const float *activation_scales =
      reinterpret_cast<const float *>(activations + R26_A_DATA);
  const float *weight_scales =
      reinterpret_cast<const float *>(weights + R26_W_DATA);
  float *output = reinterpret_cast<float *>(output_bits);
  for (int im = 0; im < R26_LM; im++)
    for (int jn = 0; jn < R26_LN; jn++) {
      R26_MMUL lo, hi;
      auto a = aie::load_v<R26_SA>(activations + (im * R26_KT) * R26_SA);
      const int8 *w = weights + (jn * R26_KT) * R26_SB;
      lo.mul(a, aie::load_v<R26_MMUL::size_B>(w));
      hi.mul(a, aie::load_v<R26_MMUL::size_B>(w + R26_MMUL::size_B));
      for (int k = 1; k < R26_KT; k++) {
        a = aie::load_v<R26_SA>(activations + (im * R26_KT + k) * R26_SA);
        w = weights + (jn * R26_KT + k) * R26_SB;
        lo.mac(a, aie::load_v<R26_MMUL::size_B>(w));
        hi.mac(a, aie::load_v<R26_MMUL::size_B>(w + R26_MMUL::size_B));
      }
      auto vlo = lo.template to_vector<int32>();
      auto vhi = hi.template to_vector<int32>();
      auto weight_scale = aie::load_v<16>(weight_scales + jn * 16);
      for (int row = 0; row < 8; row++) {
        const int offset = [&] {
          if constexpr (OUTPUT_LN == 2 * R26_LN)
            return (im * 8 + row) * 96 + NMACRO * 48 + jn * 16;
          else
            return (im * OUTPUT_LN + NMACRO * R26_LN + jn) * R26_SC +
                   row * 16;
        }();
        auto values = aie::to_float(r26_join_rows(vlo, vhi, row));
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

extern "C" void r26_gate_scaled(const int8 *a, const int8 *w, int32 *c,
                                int accumulate) {
  r26_w8_scaled<R26_LN, 0>(a, w, c, accumulate);
}

extern "C" void r26_down0_scaled(const int8 *a, const int8 *w, int32 *c,
                                 int accumulate) {
  r26_w8_scaled<2 * R26_LN, 0>(a, w, c, accumulate);
}

extern "C" void r26_down1_scaled(const int8 *a, const int8 *w, int32 *c,
                                 int accumulate) {
  r26_w8_scaled<2 * R26_LN, 1>(a, w, c, accumulate);
}

static inline aie::vector<float, 16>
r26_geglu16(aie::vector<float, 16> gate, aie::vector<float, 16> up) {
  auto one = aie::broadcast<float, 16>(1.0f);
  auto half = aie::broadcast<float, 16>(0.5f);
  auto beta = aie::broadcast<float, 16>(0.044715f);
  auto factor = aie::broadcast<float, 16>(0.7978845608028654f);
  auto g2 = aie::mul(gate, gate).template to_vector<float>();
  auto g3 = aie::mul(g2, gate).template to_vector<float>();
  auto inner = aie::add(gate, aie::mul(g3, beta).template to_vector<float>());
  auto arg = aie::mul(inner, factor).template to_vector<float>();
  auto tanh_bf16 = aie::tanh<bfloat16>(arg);
  auto tanh_f32 =
      aie::mul(tanh_bf16, aie::broadcast<bfloat16, 16>(1.0f))
          .template to_vector<float>();
  auto cdf = aie::mul(aie::add(one, tanh_f32), half).template to_vector<float>();
  return aie::mul(aie::mul(gate, cdf).template to_vector<float>(), up)
      .template to_vector<float>();
}

// Store each core's 24 logical columns in a 96-wide physical row. The last
// eight lanes use scalar
// stores only after the MMUL accumulator has been materialized, avoiding the
// unstable live-register 8-lane store attempted in R18.
extern "C" __attribute__((noinline, minsize)) void
r26_geglu_padded(const int32 *__restrict accumulator_bits,
                 float *__restrict output) {
  const float *accumulator = reinterpret_cast<const float *>(accumulator_bits);
  for (int im = 0; im < 3; im++)
    for (int row = 0; row < 8; row++) {
      const int j0 = (im * 3) * R26_SC + row * 16;
      const int j1 = (im * 3 + 1) * R26_SC + row * 16;
      const int j2 = (im * 3 + 2) * R26_SC + row * 16;
      float *destination = output + (im * 8 + row) * 96;
      aie::store_v(destination,
                   r26_geglu16(aie::load_v<16>(accumulator + j0),
                               aie::load_v<16>(accumulator + j1)));
      auto gate = aie::concat(aie::load_v<8>(accumulator + j2),
                              aie::zeros<float, 8>());
      auto up = aie::concat(aie::load_v<8>(accumulator + j2 + 8),
                            aie::zeros<float, 8>());
      auto tail = r26_geglu16(gate, up);
      for (int lane = 0; lane < 8; lane++) destination[16 + lane] = tail[lane];
    }
}

template <unsigned STRIDE>
__attribute__((noinline)) static void r26_fwht16(float *__restrict scratch) {
  for (int block = 0; block < R26_GROUP; block += 16) {
    auto values = aie::load_v<16>(scratch + block);
    auto a = aie::filter_even(values, STRIDE);
    auto b = aie::filter_odd(values, STRIDE);
    aie::store_v(scratch + block,
                 aie::concat(aie::interleave_zip(aie::add(a, b),
                                                 aie::sub(a, b), STRIDE)));
  }
}

static void r26_pack_row(const float *__restrict input,
                         const int8 *__restrict weight_payload,
                         int8 *__restrict quantized, float *__restrict scratch,
                         float &scale) {
  const float *params = reinterpret_cast<const float *>(
      weight_payload + R26_PARAM_OFFSET);
  const float *awq = params;
  const float *signs1 = awq + R26_GROUP;
  const float *signs2 = signs1 + R26_GROUP;
  for (int i = 0; i < R26_GROUP; i += 16) {
    auto divided = aie::div(aie::load_unaligned_v<16>(input + i),
                            aie::load_v<16>(awq + i))
                       .template to_vector<float>();
    aie::store_v(scratch + i,
                 aie::mul(divided, aie::load_v<16>(signs1 + i))
                     .template to_vector<float>());
  }
  r26_fwht16<1>(scratch);
  r26_fwht16<2>(scratch);
  r26_fwht16<4>(scratch);
  r26_fwht16<8>(scratch);
  for (int stride = 16; stride < R26_GROUP; stride <<= 1)
    for (int block = 0; block < R26_GROUP; block += 2 * stride)
      for (int i = 0; i < stride; i += 16) {
        auto a = aie::load_v<16>(scratch + block + i);
        auto b = aie::load_v<16>(scratch + block + i + stride);
        aie::store_v(scratch + block + i, aie::add(a, b));
        aie::store_v(scratch + block + i + stride, aie::sub(a, b));
      }
  float max_abs = 0.0f;
  for (int i = 0; i < R26_GROUP; i += 16) {
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
  const auto old_rounding = aie::swap_rounding(aie::rounding_mode::symmetric_inf);
  const auto old_saturation = aie::swap_saturation(aie::saturation_mode::symmetric);
  for (int i = 0; i < R26_GROUP; i += 16) {
    auto values = scale > 0.0f
                      ? aie::to_fixed<int8>(
                            aie::div(aie::load_v<16>(scratch + i), scale)
                                .template to_vector<float>())
                      : aie::zeros<int8, 16>();
    aie::store_v(quantized + i, values);
  }
  aie::set_saturation(old_saturation);
  aie::set_rounding(old_rounding);
}

extern "C" __attribute__((minsize)) void
r26_pack3(const int8 *__restrict input_bytes,
          const int8 *__restrict weight_payload,
          int8 *__restrict activation_payload, float *__restrict scratch,
          int8 *__restrict fragment, int owner, int group) {
  (void)activation_payload;
  const float *input = reinterpret_cast<const float *>(input_bytes);
  constexpr int ROW_WINDOW = 288;
  const int skip = (group * R26_GROUP) % 24;
  const float *owned = input + (owner & 1) * R26_FRAGMENT_ROWS * ROW_WINDOW;
  float *scales = reinterpret_cast<float *>(fragment + 3 * R26_GROUP);
  for (int row = 0; row < R26_FRAGMENT_ROWS; row++)
    r26_pack_row(owned + row * ROW_WINDOW + skip, weight_payload,
                 fragment + row * R26_GROUP, scratch, scales[row]);
  reinterpret_cast<int *>(fragment)[R26_FRAGMENT_WORDS - 1] = 0;
}

extern "C" __attribute__((minsize)) void
r26_insert_fragment(const int8 *__restrict fragment,
                    int8 *__restrict activation_payload, int owner) {
  float *scales = reinterpret_cast<float *>(activation_payload + R26_A_DATA);
  for (int row = 0; row < 3; row++) {
    const int local_row = owner * 3 + row;
    const int im = local_row / 8;
    const int rr = local_row % 8;
    for (int kt = 0; kt < 32; kt++) {
      const int target = (im * 32 + kt) * 64 + rr * 8;
      const int source = row * R26_GROUP + kt * 8;
      reinterpret_cast<int *>(activation_payload + target)[0] =
          reinterpret_cast<const int *>(fragment + source)[0];
      reinterpret_cast<int *>(activation_payload + target)[1] =
          reinterpret_cast<const int *>(fragment + source)[1];
    }
    scales[local_row] =
        reinterpret_cast<const float *>(fragment + 3 * R26_GROUP)[row];
  }
}

extern "C" __attribute__((noinline, minsize)) void
r26_send_fragment(const int8 *__restrict fragment) {
  const int *words = reinterpret_cast<const int *>(fragment);
  for (int word = 0; word < R26_FRAGMENT_WORDS; word++) put_ms(words[word]);
}

extern "C" __attribute__((noinline, minsize)) void
r26_receive_fragment(int8 *__restrict fragment) {
  int *words = reinterpret_cast<int *>(fragment);
  for (int word = 0; word < R26_FRAGMENT_WORDS; word++) words[word] = get_ss_int();
}
