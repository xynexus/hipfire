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
r98_finish_interleaved_bf16x2(int32 *__restrict accumulator_bits) {
  float *input = reinterpret_cast<float *>(accumulator_bits);
  bfloat16 *output = reinterpret_cast<bfloat16 *>(accumulator_bits);
  for (int offset = 0; offset < 2 * R25_TILE_FLOATS; offset += 16) {
    const auto values = aie::load_v<16>(input + offset);
    const auto high =
        aie::mul(values, 1.0f).template to_vector<bfloat16>();
    const auto high_float =
        aie::mul(high, aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
            .template to_vector<float>();
    const auto low = aie::mul(aie::sub(values, high_float), 1.0f)
                         .template to_vector<bfloat16>();
    aie::store_v(output + 2 * offset,
                 aie::concat(aie::interleave_zip(high, low, 1)));
  }
}

extern "C" __attribute__((noinline, minsize)) void
r25_wait_weight(const int8 *__restrict weights, float *__restrict scratch) {
  uint32_t hash = 2166136261u;
  for (int i = 0; i < R25_WEIGHT_BYTES; i++)
    hash = (hash ^ static_cast<uint8_t>(weights[i])) * 16777619u;
  reinterpret_cast<volatile uint32_t *>(scratch)[0] = hash;
}

extern "C" __attribute__((noinline, minsize)) void
r97_bf16_to_f32_3(const int8 *__restrict input_bytes,
                  float *__restrict output) {
  const auto *input = reinterpret_cast<const bfloat16 *>(input_bytes);
  for (int row = 0; row < 3; ++row)
    for (int inner = 0; inner < R25_GROUP; inner += 16) {
      const auto values =
          aie::mul(aie::load_v<16>(input + row * R25_GROUP + inner),
                   aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
              .to_vector<float>();
      aie::store_v(output + row * R25_GROUP + inner, values);
    }
}

extern "C" __attribute__((noinline, minsize)) void
r101_bf16_x_inverse_to_f32_3(const int8 *__restrict input_bytes,
                             float *__restrict output, int32_t group) {
  for (int row = 0; row < 3; ++row) {
    const int8 *row_bytes = input_bytes + row * 1664;
    const auto *input = reinterpret_cast<const bfloat16 *>(row_bytes);
    const float inverse =
        *reinterpret_cast<const float *>(row_bytes + 1536);
    const auto scale = aie::broadcast<float, 16>(inverse);
    for (int inner = 0; inner < R25_GROUP; inner += 16) {
      const auto values =
          aie::mul(aie::load_v<16>(input + group * R25_GROUP + inner),
                   aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
              .to_vector<float>();
      aie::store_v(output + row * R25_GROUP + inner,
                   aie::mul(values, scale).to_vector<float>());
    }
  }
}

extern "C" __attribute__((noinline, minsize)) void
r104_rms_accumulate3(const int8 *__restrict input_bytes,
#ifdef R104_FULL_X_OBJECT
                     float *__restrict inverse) {
#else
                     float *__restrict inverse, int32_t group) {
#endif
  const auto *input = reinterpret_cast<const bfloat16 *>(input_bytes);
  for (int row = 0; row < 3; ++row) {
#ifdef R104_FULL_X_OBJECT
    float sum = 0.0f;
    for (int inner = 0; inner < 3 * R25_GROUP; inner += 16) {
      const auto values =
          aie::mul(aie::load_v<16>(input + row * 3 * R25_GROUP + inner),
                   aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
              .to_vector<float>();
      sum += aie::reduce_add(aie::mul(values, values).to_vector<float>());
    }
#else
    float sum = group == 0 ? 0.0f : inverse[row];
    for (int inner = 0; inner < R25_GROUP; inner += 16) {
      const auto values =
          aie::mul(aie::load_v<16>(input + row * R25_GROUP + inner),
                   aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
              .to_vector<float>();
      sum += aie::reduce_add(aie::mul(values, values).to_vector<float>());
    }
#endif
#ifdef R104_FULL_X_OBJECT
    {
#else
    if (group == 2) {
#endif
      const auto mean =
          aie::mul(aie::broadcast<float, 16>(sum),
                   aie::broadcast<float, 16>(1.0f / 768.0f))
              .template to_vector<float>();
      inverse[row] = aie::invsqrt(mean[0] + 1.0e-6f);
#ifndef R104_FULL_X_OBJECT
    } else {
      inverse[row] = sum;
#endif
    }
  }
}

extern "C" __attribute__((noinline, minsize)) void
r104_bf16_x_inverse_to_f32_3(const int8 *__restrict input_bytes,
                             const float *__restrict inverse,
                             float *__restrict output
#ifdef R104_FULL_X_OBJECT
                             ,
                             int32_t group
#endif
                             ) {
  const auto *input = reinterpret_cast<const bfloat16 *>(input_bytes);
  for (int row = 0; row < 3; ++row) {
    const auto scale = aie::broadcast<float, 16>(inverse[row]);
    for (int inner = 0; inner < R25_GROUP; inner += 16) {
      const auto values =
          aie::mul(aie::load_v<16>(input
#ifdef R104_FULL_X_OBJECT
                                   + row * 3 * R25_GROUP + group * R25_GROUP
#else
                                   + row * R25_GROUP
#endif
                                   + inner),
                   aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
              .to_vector<float>();
      aie::store_v(output + row * R25_GROUP + inner,
                   aie::mul(values, scale).to_vector<float>());
    }
  }
}

extern "C" __attribute__((noinline, minsize)) void
r97_activation_fence(const int8 *__restrict activations) {
  const volatile int8 *visible = activations;
  int sink = visible[0] + visible[R25_A_DATA - 1] +
             visible[R25_A_BYTES - (int)sizeof(float)];
  (void)sink;
  chess_separator_scheduler(1);
  chess_separator();
}

extern "C" __attribute__((noinline, minsize)) void
r97_scan_activation(const int8 *__restrict activations,
                    float *__restrict scratch) {
  const volatile int8 *visible = activations;
  uint32_t hash = 2166136261u;
  for (int offset = 0; offset < R25_A_BYTES; ++offset)
    hash = (hash ^ static_cast<uint8_t>(visible[offset])) * 16777619u;
  reinterpret_cast<volatile uint32_t *>(scratch)[0] = hash;
  chess_separator_scheduler(1);
  chess_separator();
}

extern "C" __attribute__((noinline, minsize)) void
r97_snapshot_activation_hash(const int8 *__restrict activations,
                             float *__restrict snapshot, int slot) {
  uint32_t hash = 2166136261u;
  for (int offset = 0; offset < R25_A_BYTES; ++offset)
    hash = (hash ^ static_cast<uint8_t>(activations[offset])) * 16777619u;
  reinterpret_cast<uint32_t *>(snapshot)[slot] = hash;
}

extern "C" __attribute__((noinline, minsize)) void
r97_snapshot_weight_hash(const int8 *__restrict weights,
                         float *__restrict snapshot, int slot) {
  uint32_t hash = 2166136261u;
  for (int offset = 0; offset < R25_WEIGHT_BYTES; ++offset)
    hash = (hash ^ static_cast<uint8_t>(weights[offset])) * 16777619u;
  reinterpret_cast<uint32_t *>(snapshot)[slot] = hash;
}

extern "C" __attribute__((noinline, minsize)) void
r97_emit_activation_hashes(const float *__restrict snapshot,
                           int32 *__restrict output) {
  constexpr int row = R25_PROBE_ROW;
  constexpr int im = row / 4;
  constexpr int rr = row % 4;
  for (int slot = 0; slot < 6; ++slot) {
    const int col = 48 + slot;
    const int destination = (im * 6 + col / 16) * 64 + rr * 16 + col % 16;
    output[destination] = reinterpret_cast<const int32 *>(snapshot)[slot];
  }
}

extern "C" __attribute__((noinline, minsize)) void
r97_copy_activation(const int8 *__restrict input, int8 *__restrict output) {
  for (int offset = 0; offset < R25_A_BYTES; offset += 16)
    aie::store_v(output + offset, aie::load_v<16>(input + offset));
}

extern "C" __attribute__((noinline, minsize)) void
r97_probe_source(const int8 *__restrict input, int32 *__restrict output,
                 int slot) {
  uint32_t hash = 2166136261u;
  for (int offset = 0; offset < 3 * R25_GROUP * (int)sizeof(bfloat16); ++offset)
    hash = (hash ^ static_cast<uint8_t>(input[offset])) * 16777619u;
  constexpr int row = R25_PROBE_ROW;
  constexpr int im = row / 4;
  constexpr int rr = row % 4;
  const int col = 38 + slot;
  const int destination = (im * 6 + col / 16) * 64 + rr * 16 + col % 16;
  output[destination] = static_cast<int32_t>(hash);
}

extern "C" void r25_zero(int32 *__restrict output_bits) {
  float *output = reinterpret_cast<float *>(output_bits);
  for (int i = 0; i < 2304; i += 16)
    aie::store_v(output + i, aie::zeros<float, 16>());
}

extern "C" __attribute__((minsize)) void
r25_down(const int8 *a, const int8 *w, int32 *c, int accumulate) {
#ifdef R15_DYNAMIC_ONLY
  r15_w4_scaled_dynamic(a, w, c, accumulate);
#else
  using down_fn = void (*)(const int8 *, const int8 *, int32 *);
  const down_fn fn = accumulate ? r15_w4_scaled_accum : r15_w4_scaled_init;
  fn(a, w, c);
#endif
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

__attribute__((noinline)) static void
r25_fwht16(float *__restrict scratch, unsigned stride) {
  for (int block = 0; block < R25_GROUP; block += 16) {
    auto values = aie::load_v<16>(scratch + block);
    auto a = aie::filter_even(values, stride);
    auto b = aie::filter_odd(values, stride);
    aie::store_v(scratch + block,
                 aie::concat(aie::interleave_zip(aie::add(a, b),
                                                 aie::sub(a, b), stride)));
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
  for (unsigned stride = 1; stride <= 8; stride <<= 1)
    r25_fwht16(scratch, stride);
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

extern "C" __attribute__((noinline, minsize)) void
r25_exchange_fragments(int8 *__restrict activations,
                       const int8 *__restrict own,
                       int8 *__restrict transit, int owner) {
  r25_insert_fragment(own, activations, owner);
  for (int source = 0; source < 8; ++source) {
    if (owner == source) {
      r25_send_fragment(own);
    } else {
      r25_receive_fragment(transit);
      r25_insert_fragment(transit, activations, source);
      if (owner != (source + 7) % 8)
        r25_send_fragment(transit);
    }
  }
}

extern "C" void r25_probe_activation_rows(const int8 *__restrict activations,
                                           int32 *__restrict output_bits,
                                           int owner) {
  for (int row = 0; row < R25_TILE_ROWS; ++row) {
    const int im = row / 4;
    const int rr = row % 4;
    for (int lane = 0; lane < 32; lane++) {
      const int inner = owner * 32 + lane;
      const int source = (im * 16 + inner / 16) * 64 + rr * 16 + inner % 16;
      const int destination = (im * 6 + lane / 16) * 64 + rr * 16 + lane % 16;
      output_bits[destination] = activations[source];
    }
    constexpr int scale_col = 48;
    const int scale_destination =
        (im * 6 + scale_col / 16) * 64 + rr * 16 + scale_col % 16;
    output_bits[scale_destination] = reinterpret_cast<const int32 *>(
        activations + R25_A_DATA)[row];
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
  const int activation_col = 32 + 2 * slot;
  const int weight_col = activation_col + 1;
  const int activation_destination =
      (im * 6 + activation_col / 16) * 64 + rr * 16 + activation_col % 16;
  const int weight_destination =
      (im * 6 + weight_col / 16) * 64 + rr * 16 + weight_col % 16;
  output_bits[activation_destination] = activation_hash;
  output_bits[weight_destination] = weight_hash;
}
