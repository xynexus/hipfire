// SPDX-License-Identifier: Apache-2.0

// Resident cross-layer preprocessing for EmbeddingGemma.  The source is the
// compensated BF16x2 completed state emitted by R46.  Each core owns eight
// rows, computes the input RMS inverse, applies the next layer's learned norm
// and optional AWQ scale, performs the canonical signed FWHT-256, and emits
// one physical R34 activation chunk per K group.

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int ROWS = 8;
constexpr int GROUP = 256;
constexpr int GROUPS = 3;
constexpr int HIDDEN = GROUP * GROUPS;
constexpr int INPUT_BYTES = 2 * HIDDEN * sizeof(bfloat16);
constexpr int Q_BYTES = ROWS * GROUP;
constexpr int SCALE_BYTES = ROWS * sizeof(float);
constexpr int CHUNK_BYTES = Q_BYTES + SCALE_BYTES;
constexpr int BLOCK_BYTES = 3 * Q_BYTES + 3 * SCALE_BYTES;
constexpr int PARAM_NORM = 0;
constexpr int PARAM_AWQ = PARAM_NORM + GROUP * sizeof(float);
constexpr int PARAM_SIGN1 = PARAM_AWQ + GROUP * sizeof(float);
constexpr int PARAM_SIGN2 = PARAM_SIGN1 + GROUP * sizeof(bfloat16);
constexpr int PARAM_BYTES = PARAM_SIGN2 + GROUP * sizeof(bfloat16);
constexpr float INV_HIDDEN = 1.0f / (float)HIDDEN;
constexpr float EPSILON = 1.0e-6f;

static inline float bf16_to_float(bfloat16 value) {
  return (float)value;
}

__attribute__((noinline)) static void
prepare8(const bfloat16 *restrict high, const bfloat16 *restrict low,
         const float *restrict norm, const float *restrict awq,
         const bfloat16 *restrict signs, float inverse,
         float *restrict scratch) {
  scratch[0] = (bf16_to_float(high[0]) + bf16_to_float(low[0])) * inverse
               * norm[0] / awq[0] * (float)signs[0];
  scratch[1] = (bf16_to_float(high[1]) + bf16_to_float(low[1])) * inverse
               * norm[1] / awq[1] * (float)signs[1];
  scratch[2] = (bf16_to_float(high[2]) + bf16_to_float(low[2])) * inverse
               * norm[2] / awq[2] * (float)signs[2];
  scratch[3] = (bf16_to_float(high[3]) + bf16_to_float(low[3])) * inverse
               * norm[3] / awq[3] * (float)signs[3];
  scratch[4] = (bf16_to_float(high[4]) + bf16_to_float(low[4])) * inverse
               * norm[4] / awq[4] * (float)signs[4];
  scratch[5] = (bf16_to_float(high[5]) + bf16_to_float(low[5])) * inverse
               * norm[5] / awq[5] * (float)signs[5];
  scratch[6] = (bf16_to_float(high[6]) + bf16_to_float(low[6])) * inverse
               * norm[6] / awq[6] * (float)signs[6];
  scratch[7] = (bf16_to_float(high[7]) + bf16_to_float(low[7])) * inverse
               * norm[7] / awq[7] * (float)signs[7];
}

__attribute__((noinline)) static float
post_sign16(float *restrict scratch, const bfloat16 *restrict signs) {
  const auto values = aie::load_v<16>(scratch);
  const auto sign =
      aie::mul(aie::load_v<16>(signs),
               aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
          .to_vector<float>();
  const auto scaled =
      aie::mul(aie::mul(values, sign).to_vector<float>(),
               aie::broadcast<float, 16>(0.0625f))
          .to_vector<float>();
  aie::store_v(scratch, scaled);
  return aie::reduce_max(aie::abs(scaled));
}

static inline void copy_chunk_to_block(const int8_t *restrict chunk,
                                       int8_t *restrict block, int lm) {
  const int q_base = lm * Q_BYTES;
  const int scale_base = 3 * Q_BYTES + lm * SCALE_BYTES;
  // R111 keeps three chunks live. The allocator guarantees 32-byte alignment
  // for each buffer, but the second and third chunks need not be 64-byte
  // aligned. AIE load_v requires vector-size alignment, so use the guaranteed
  // width for that schedule rather than issuing an invalid 64-byte load.
#ifdef R111_ALIGN_SAFE_CHUNK_COPY
  for (int offset = 0; offset < Q_BYTES; offset += 32)
    aie::store_v(block + q_base + offset,
                 aie::load_v<32>(chunk + offset));
#else
  for (int offset = 0; offset < Q_BYTES; offset += 64)
    aie::store_v(block + q_base + offset,
                 aie::load_v<64>(chunk + offset));
#endif
  for (int offset = 0; offset < SCALE_BYTES; offset += 32)
    aie::store_v(block + scale_base + offset,
                 aie::load_v<32>(chunk + Q_BYTES + offset));
}

} // namespace

extern "C" {

__attribute__((noinline, minsize)) void
r111_copy_row(const int8_t *restrict input, int8_t *restrict output) {
  for (int offset = 0; offset < INPUT_BYTES; offset += 32)
    aie::store_v(output + offset, aie::load_v<32>(input + offset));
}

void r47_accumulate_row(const int8_t *restrict input_bytes,
                        float *restrict sums, int32_t row) {
  const auto *high = reinterpret_cast<const bfloat16 *>(input_bytes);
  const auto *low = high + HIDDEN;
  float sum = 0.0f;
  for (int i = 0; i < HIDDEN; ++i) {
    const float value = bf16_to_float(high[i]) + bf16_to_float(low[i]);
    sum += value * value;
  }
  sums[row] = aie::invsqrt(sum * INV_HIDDEN + EPSILON);
}

void r47_pack_row(const int8_t *restrict input_bytes,
                  const int8_t *restrict param_bytes,
                  int8_t *restrict chunk, float *restrict scratch,
                  const float *restrict inverse, int32_t row,
                  int32_t group) {
  const auto *high = reinterpret_cast<const bfloat16 *>(input_bytes)
                     + group * GROUP;
  const auto *low = reinterpret_cast<const bfloat16 *>(input_bytes)
                    + HIDDEN + group * GROUP;
  const auto *norm = reinterpret_cast<const float *>(param_bytes + PARAM_NORM);
  const auto *awq = reinterpret_cast<const float *>(param_bytes + PARAM_AWQ);
  const auto *sign1 =
      reinterpret_cast<const bfloat16 *>(param_bytes + PARAM_SIGN1);
  const auto *sign2 =
      reinterpret_cast<const bfloat16 *>(param_bytes + PARAM_SIGN2);

  for (int i = 0; i < GROUP; i += 8)
    prepare8(high + i, low + i, norm + i, awq + i, sign1 + i,
             inverse[row], scratch + i);
  for (int stride = 1; stride < GROUP; stride <<= 1)
    for (int block = 0; block < GROUP; block += 2 * stride)
      for (int i = 0; i < stride; ++i) {
        const float a = scratch[block + i];
        const float b = scratch[block + i + stride];
        scratch[block + i] = a + b;
        scratch[block + i + stride] = a - b;
      }

  float max_abs = 0.0f;
  for (int i = 0; i < GROUP; i += 16) {
    const float local_max = post_sign16(scratch + i, sign2 + i);
    if (local_max > max_abs)
      max_abs = local_max;
  }
  const float scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
  *reinterpret_cast<float *>(chunk + Q_BYTES + row * sizeof(float)) = scale;

  const auto old_rounding =
      aie::swap_rounding(aie::rounding_mode::symmetric_floor);
  const auto old_saturation =
      aie::swap_saturation(aie::saturation_mode::symmetric);
  for (int i = 0; i < GROUP; ++i) {
    int8_t quantized = 0;
    if (scale > 0.0f) {
      const float normalized = scratch[i] / scale;
      const float biased =
          normalized + (normalized >= 0.0f ? 0.5f : -0.5f);
      quantized = aie::to_fixed<int8_t>(biased);
    }
    const int kt = i / 8;
    const int kk = i & 7;
    chunk[kt * 64 + row * 8 + kk] = quantized;
  }
  aie::set_saturation(old_saturation);
  aie::set_rounding(old_rounding);
}

__attribute__((noinline, minsize)) void
r47_copy_param(const int8_t *restrict input, int8_t *restrict params,
               int32_t group) {
  (void)group;
  int8_t *output = params;
  for (int offset = 0; offset < PARAM_BYTES; offset += 32)
    aie::store_v(output + offset, aie::load_v<32>(input + offset));
}

__attribute__((noinline, minsize)) void
r47_send_chunk(const int8_t *restrict chunk) {
  const int32_t *words = reinterpret_cast<const int32_t *>(chunk);
  for (int word = 0; word < CHUNK_BYTES / (int)sizeof(int32_t); ++word)
    put_ms(words[word]);
}

__attribute__((noinline, minsize)) void
r47_relay_then_send(const int8_t *restrict chunk) {
  for (int word = 0; word < CHUNK_BYTES / (int)sizeof(int32_t); ++word)
    put_ms(get_ss_int());
  r47_send_chunk(chunk);
}

__attribute__((noinline, minsize)) void
r47_assemble_block(int8_t *restrict block, const int8_t *restrict own,
                   int32_t predecessors) {
  for (int offset = 0; offset < BLOCK_BYTES; offset += 32)
    aie::store_v(block + offset, aie::zeros<int8_t, 32>());
  for (int lm = 0; lm < predecessors; ++lm) {
    for (int word = 0; word < Q_BYTES / (int)sizeof(int32_t); ++word)
      reinterpret_cast<int32_t *>(block + lm * Q_BYTES)[word] = get_ss_int();
    for (int word = 0; word < SCALE_BYTES / (int)sizeof(int32_t); ++word)
      reinterpret_cast<int32_t *>(block + 3 * Q_BYTES + lm * SCALE_BYTES)[word] =
          get_ss_int();
  }
  copy_chunk_to_block(own, block, predecessors);
}

__attribute__((noinline, minsize)) void
r47_emit_block(const int8_t *restrict block, int8_t *restrict output) {
  for (int offset = 0; offset < BLOCK_BYTES; offset += 32)
    aie::store_v(output + offset, aie::load_v<32>(block + offset));
}

} // extern "C"
