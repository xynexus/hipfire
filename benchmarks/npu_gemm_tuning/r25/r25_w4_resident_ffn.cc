// Complete resident W4 gate/up + GeGLU -> activation pack -> down projection.
#include "../r18/r18_w4_fused.cc"

constexpr int R25_GROUP = 256;
constexpr int R25_TILE_COLS = 48;
constexpr int R25_TILE_ROWS = 24;
constexpr int R25_TILE_FLOATS = R25_TILE_ROWS * R25_TILE_COLS;
constexpr int R25_LOGICAL_COLS = 8 * R25_TILE_COLS;
constexpr int R25_FRAGMENT_ROWS = 3;
constexpr int R25_FRAGMENT_BYTES = R25_FRAGMENT_ROWS * R25_GROUP + 16;
constexpr int R25_FRAGMENT_WORDS = R25_FRAGMENT_BYTES / sizeof(int);
constexpr int R25_A_DATA = 6144;
constexpr int R25_A_BYTES = R25_A_DATA + 24 * sizeof(float);
constexpr int R25_W_DATA = 12288;
constexpr int R25_W_COLS = 96;
constexpr int R25_PARAM_OFFSET = R25_W_DATA + R25_W_COLS * sizeof(float);
constexpr int R25_WEIGHT_BYTES = 15872;
#ifndef R25_PROBE_ROW
#define R25_PROBE_ROW 0
#endif
#ifndef R25_RAW_SOURCE_JN
#define R25_RAW_SOURCE_JN -1
#endif
extern "C" __attribute__((noinline, minsize)) void
r25_wait_weight(const int8 *__restrict weights, float *__restrict scratch) {
  uint32_t hash = 2166136261u;
  for (int i = 0; i < R25_WEIGHT_BYTES; i++)
    hash = (hash ^ static_cast<uint8_t>(weights[i])) * 16777619u;
  reinterpret_cast<volatile uint32_t *>(scratch)[0] = hash;
}

extern "C" void r25_zero(int32 *__restrict output_bits) {
  float *output = reinterpret_cast<float *>(output_bits);
  for (int i = 0; i < 2304; i += 16)
    aie::store_v(output + i, aie::zeros<float, 16>());
}

extern "C" __attribute__((minsize)) void
r25_down(const int8 *a, const int8 *w, int32 *c, int accumulate) {
  using down_fn = void (*)(const int8 *, const int8 *, int32 *);
  const down_fn fn = accumulate ? r15_w4_scaled_accum : r15_w4_scaled_init;
  fn(a, w, c);
}

extern "C" void r25_touch_weight(const int8 *__restrict weights,
                                  float *__restrict scratch) {
  uint32_t hash = 2166136261u;
  for (int i = 0; i < R25_WEIGHT_BYTES; i++)
    hash = (hash ^ static_cast<uint8_t>(weights[i])) * 16777619u;
  reinterpret_cast<uint32_t *>(scratch)[0] = hash;
}

extern "C" __attribute__((noinline)) void
r25_touch_weight_tail(const int8 *__restrict weights, float *__restrict scratch) {
  reinterpret_cast<volatile int8 *>(scratch)[0] = weights[R25_WEIGHT_BYTES - 1];
}

extern "C" void r25_snapshot_raw_gate(const int32 *__restrict accumulator_bits,
                                        float *__restrict snapshot) {
  constexpr int row = R25_PROBE_ROW;
  constexpr int im = row / 4;
  constexpr int rr = row % 4;
#if R25_RAW_SOURCE_JN >= 0
  for (int lane = 0; lane < 16; lane++) {
    const int source = (im * 6 + R25_RAW_SOURCE_JN) * 64 + rr * 16 + lane;
    reinterpret_cast<int32 *>(snapshot)[lane] = accumulator_bits[source];
  }
#else
  for (int col = 0; col < 96; col++) {
    const int offset = (im * 6 + col / 16) * 64 + rr * 16 + col % 16;
    reinterpret_cast<int32 *>(snapshot)[col] = accumulator_bits[offset];
  }
#endif
  chess_separator_scheduler(1);
  chess_separator();
}

extern "C" void r25_emit_raw_gate(const float *__restrict snapshot,
                                    int32 *__restrict output_bits) {
  constexpr int row = R25_PROBE_ROW;
  constexpr int im = row / 4;
  constexpr int rr = row % 4;
#if R25_RAW_SOURCE_JN >= 0
  for (int lane = 0; lane < 16; lane++) {
    const int destination = (im * 6 + 2) * 64 + rr * 16 + lane;
    output_bits[destination] = reinterpret_cast<const int32 *>(snapshot)[lane];
  }
#else
  for (int col = 0; col < 96; col++) {
    const int destination = (im * 6 + col / 16) * 64 + rr * 16 + col % 16;
    output_bits[destination] = reinterpret_cast<const int32 *>(snapshot)[col];
  }
#endif
}

extern "C" void r25_probe_raw_samples(const int32 *__restrict accumulator_bits,
                                       int32 *__restrict output_bits) {
  constexpr int row = R25_PROBE_ROW;
  constexpr int im = row / 4;
  constexpr int rr = row % 4;
  constexpr int source_jn = R25_RAW_SOURCE_JN < 0 ? 0 : R25_RAW_SOURCE_JN;
  const int source = (im * 6 + source_jn) * 64 + rr * 16;
  const int destination = im * 6 * 64 + rr * 16;
  aie::store_v(output_bits + destination,
               aie::load_v<16>(accumulator_bits + source));
}

extern "C" __attribute__((minsize)) void
r25_geglu_inplace(int32 *__restrict accumulator_bits,
                  float *__restrict scratch) {
  float *accumulator = reinterpret_cast<float *>(accumulator_bits);
  constexpr int INPUT_STRIDE = 6 * 64;
  for (int im = 0; im < 6; im++) {
    for (int i = 0; i < INPUT_STRIDE; i += 16)
      aie::store_v(scratch + i,
                   aie::load_v<16>(accumulator + im * INPUT_STRIDE + i));
    for (int row = 0; row < 4; row++)
      for (int jn = 0; jn < 3; jn++) {
        const int gate_offset = jn * 64 + row * 16;
        const int up_offset = (jn + 3) * 64 + row * 16;
        const int output_offset = (im * 4 + row) * R25_TILE_COLS + jn * 16;
        aie::store_v(accumulator + output_offset,
                     r18_geglu16(aie::load_v<16>(scratch + gate_offset),
                                 aie::load_v<16>(scratch + up_offset)));
      }
  }
  chess_separator_scheduler_local();
  chess_separator();
}

template <unsigned STRIDE>
__attribute__((noinline)) static void r25_fwht16(float *__restrict scratch) {
  for (int block = 0; block < R25_GROUP; block += 16) {
    auto values = aie::load_v<16>(scratch + block);
    auto a = aie::filter_even(values, STRIDE);
    auto b = aie::filter_odd(values, STRIDE);
    aie::store_v(scratch + block,
                 aie::concat(aie::interleave_zip(aie::add(a, b),
                                                 aie::sub(a, b), STRIDE)));
  }
}

static void r25_pack_row(const float *__restrict input,
                         const float *__restrict carry,
                         const int8 *__restrict weight_payload,
                         int8 *quantized,
                         float *__restrict scratch, float *__restrict scale_out,
                         int row, int kind) {
  const float *params = reinterpret_cast<const float *>(
      weight_payload + R25_PARAM_OFFSET);
  const float *awq = params;
  const float *signs1 = awq + R25_GROUP;
  const float *signs2 = signs1 + R25_GROUP;
  for (int i = 0; i < R25_GROUP; i += 16) {
    aie::vector<float, 16> values;
    if (kind == 1)
      values = i < 128 ? aie::load_v<16>(carry + row * 128 + i)
                       : aie::load_v<16>(input + row * 384 + i - 128);
    else if (kind == 2)
      values = aie::load_v<16>(input + row * 256 + i);
    else if (kind == 4)
      values = i < 128 ? aie::load_v<16>(input + row * 128 + i)
                       : aie::zeros<float, 16>();
    else
      values = aie::load_v<16>(input + row * 384 + i);
    auto divided =
        aie::div(values, aie::load_v<16>(awq + i)).template to_vector<float>();
    aie::store_v(scratch + i,
                 aie::mul(divided, aie::load_v<16>(signs1 + i))
                     .template to_vector<float>());
  }
  r25_fwht16<1>(scratch);
  r25_fwht16<2>(scratch);
  r25_fwht16<4>(scratch);
  r25_fwht16<8>(scratch);
  for (int stride = 16; stride < R25_GROUP; stride <<= 1)
    for (int block = 0; block < R25_GROUP; block += 2 * stride)
      for (int i = 0; i < stride; i += 16) {
        auto a = aie::load_v<16>(scratch + block + i);
        auto b = aie::load_v<16>(scratch + block + i + stride);
        aie::store_v(scratch + block + i, aie::add(a, b));
        aie::store_v(scratch + block + i + stride, aie::sub(a, b));
      }
  float max_abs = 0.0f;
  for (int i = 0; i < R25_GROUP; i += 16) {
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
  const float scale =
      max_abs > 0.0f
          ? aie::div(aie::broadcast<float, 16>(max_abs),
                     aie::broadcast<float, 16>(127.0f))
                .template to_vector<float>()[0]
          : 0.0f;
  float *scale_base = scale_out - row;
  auto scale_values =
      row == 0 ? aie::zeros<float, 4>() : aie::load_v<4>(scale_base);
  scale_values.set(scale, row);
  aie::store_v(scale_base, scale_values);
  aie::set_rounding(aie::rounding_mode::symmetric_inf);
  aie::set_saturation(aie::saturation_mode::symmetric);
  for (int i = 0; i < R25_GROUP; i += 16) {
    auto output = scale > 0.0f
                      ? aie::to_fixed<int8>(
                            aie::div(aie::load_v<16>(scratch + i), scale)
                                .template to_vector<float>())
                      : aie::zeros<int8, 16>();
    aie::store_v(quantized + i, output);
  }
  aie::set_saturation(aie::saturation_mode::none);
  aie::set_rounding(aie::rounding_mode::floor);
}

extern "C" __attribute__((minsize)) void r25_pack3(const float *__restrict input,
                           const float *__restrict carry,
                           const int8 *__restrict weights,
                           float *__restrict scratch,
                           float *__restrict scales,
                           int8 *__restrict fragment, int kind) {
  for (int row = 0; row < 3; row++)
    r25_pack_row(input, carry, weights, fragment + row * R25_GROUP,
                 scratch, scales + row, row, kind);
  aie::store_v(reinterpret_cast<float *>(fragment + 3 * R25_GROUP),
               aie::load_v<4>(scales));
}

extern "C" __attribute__((noinline, minsize)) void
r25_save_tile(const float *__restrict input, float *__restrict output,
              int offset, int count) {
  for (int row = 0; row < 3; row++)
    for (int i = 0; i < count; i += 16)
      aie::store_v(output + row * count + i,
                   aie::load_v<16>(input + row * 384 + offset + i));
}

extern "C" void r25_save_carry_generic(const float *, float *, int, int)
    __attribute__((alias("r25_save_tile")));

extern "C" __attribute__((minsize)) void
r25_restore_tile(const float *__restrict input, float *__restrict output,
                 int count) {
  for (int row = 0; row < 3; row++)
    for (int i = 0; i < count; i += 16)
      aie::store_v(output + row * count + i,
                   aie::load_v<16>(input + row * count + i));
}

extern "C" __attribute__((noinline, minsize)) void
r25_spill_down_bf16(const int32 *__restrict accumulator_bits,
                    float *__restrict saved, int8 *__restrict own,
                    int8 *__restrict transit) {
  const uint32_t *input = reinterpret_cast<const uint32_t *>(accumulator_bits);
  uint16_t *saved_bf16 = reinterpret_cast<uint16_t *>(saved);
  uint16_t *own_bf16 = reinterpret_cast<uint16_t *>(own);
  uint16_t *transit_bf16 = reinterpret_cast<uint16_t *>(transit);
  for (int i = 0; i < 2304; i++) {
    const uint32_t bits = input[i];
    const uint16_t value = static_cast<uint16_t>(bits >> 16);
    if (i < 1536)
      saved_bf16[i] = value;
    else if (i < 1928)
      own_bf16[i - 1536] = value;
    else
      transit_bf16[i - 1928] = value;
  }
}

extern "C" __attribute__((noinline, minsize)) void
r25_restore_down_bf16(const float *__restrict saved,
                      const int8 *__restrict own,
                      const int8 *__restrict transit,
                      int32 *__restrict accumulator_bits) {
  const uint16_t *saved_bf16 = reinterpret_cast<const uint16_t *>(saved);
  const uint16_t *own_bf16 = reinterpret_cast<const uint16_t *>(own);
  const uint16_t *transit_bf16 = reinterpret_cast<const uint16_t *>(transit);
  uint32_t *output = reinterpret_cast<uint32_t *>(accumulator_bits);
  for (int i = 0; i < 2304; i++) {
    const uint16_t value = i < 1536   ? saved_bf16[i]
                           : i < 1928 ? own_bf16[i - 1536]
                                      : transit_bf16[i - 1928];
    output[i] = static_cast<uint32_t>(value) << 16;
  }
}

extern "C" __attribute__((minsize)) void r25_extract_local(const float *__restrict tile,
                                   float *__restrict transposed, int owner,
                                   int destination) {
  const int row0 = destination * 3;
  for (int row = 0; row < 3; row++)
    for (int col = 0; col < 48; col += 16)
      aie::store_v(transposed + row * 384 + owner * 48 + col,
                   aie::load_v<16>(tile + (row0 + row) * 48 + col));
}

extern "C" __attribute__((noinline, minsize)) void
r25_send_words(const int *__restrict words, int count) {
  for (int word = 0; word < count; word++) put_ms(words[word]);
}

extern "C" __attribute__((minsize)) void r25_receive_tile(float *__restrict transposed, int owner,
                                  int destination, int forward) {
  const int before = destination * 3 * 48;
  const int last = before + 3 * 48;
  int local = 0;
  int *output = reinterpret_cast<int *>(transposed) + owner * 48;
  for (int word = 0; word < R25_TILE_FLOATS; word++) {
    const int value = get_ss_int();
    if (word >= before && word < last) {
      *output++ = value;
      local++;
      if (local == 48) {
        output += 384 - 48;
        local = 0;
      }
    }
    if (forward) put_ms(value);
  }
}

extern "C" __attribute__((noinline, minsize)) void
r25_insert_fragment(const int8 *__restrict fragment,
                    int8 *__restrict activations, int owner) {
  float *scales = reinterpret_cast<float *>(activations + R25_A_DATA);
  for (int row = 0; row < 3; row++) {
    const int local_row = owner * 3 + row;
    const int im = local_row / 4;
    const int rr = local_row % 4;
    for (int kt = 0; kt < 16; kt++)
      aie::store_v(activations + (im * 16 + kt) * 64 + rr * 16,
                   aie::load_v<16>(fragment + row * 256 + kt * 16));
    scales[local_row] = reinterpret_cast<const float *>(fragment + 768)[row];
  }
}

extern "C" __attribute__((noinline, minsize)) void
r25_send_fragment(const int8 *__restrict fragment) {
  r25_send_words(reinterpret_cast<const int *>(fragment), R25_FRAGMENT_WORDS);
}

extern "C" __attribute__((noinline, minsize)) void
r25_receive_fragment(int8 *__restrict fragment) {
  int *words = reinterpret_cast<int *>(fragment);
  for (int word = 0; word < R25_FRAGMENT_WORDS; word++) words[word] = get_ss_int();
}

extern "C" void r25_probe_activation_rows(const int8 *__restrict activations,
                                           int32 *__restrict output_bits,
                                           int owner) {
  constexpr int row = R25_PROBE_ROW;
  constexpr int im = row / 4;
  constexpr int rr = row % 4;
  for (int lane = 0; lane < 32; lane++) {
    const int inner = owner * 32 + lane;
    const int source = (im * 16 + inner / 16) * 64 + rr * 16 + inner % 16;
    const int destination = (im * 6 + lane / 16) * 64 + rr * 16 + lane % 16;
    output_bits[destination] = activations[source];
  }
}

extern "C" void r25_probe_gate_row(const float *__restrict tile,
                                    int32 *__restrict output_bits) {
  constexpr int row = R25_PROBE_ROW;
  constexpr int im = row / 4;
  constexpr int rr = row % 4;
  for (int col = 0; col < R25_TILE_COLS; col++) {
    const int destination = (im * 6 + col / 16) * 64 + rr * 16 + col % 16;
    output_bits[destination] = reinterpret_cast<const int *>(tile)[row * R25_TILE_COLS + col];
  }
}

extern "C" void r25_probe_gate_inputs(const int8 *__restrict activations,
                                       const int8 *__restrict weights,
                                       int32 *__restrict output_bits,
                                       int slot) {
  constexpr int row = R25_PROBE_ROW;
  constexpr int im = row / 4;
  constexpr int rr = row % 4;
  uint32_t activation_hash = 2166136261u;
  uint32_t weight_hash = 2166136261u;
  for (int i = 0; i < R25_A_BYTES; i++)
    activation_hash = (activation_hash ^ static_cast<uint8_t>(activations[i])) * 16777619u;
  for (int i = 0; i < R25_WEIGHT_BYTES; i++)
    weight_hash = (weight_hash ^ static_cast<uint8_t>(weights[i])) * 16777619u;
  const int base = im * 6 * 64 + rr * 16;
  output_bits[base + 2 * slot] = activation_hash;
  output_bits[base + 2 * slot + 1] = weight_hash;
}
