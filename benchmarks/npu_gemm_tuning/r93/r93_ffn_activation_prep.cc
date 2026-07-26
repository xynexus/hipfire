// SPDX-License-Identifier: Apache-2.0

// Canonical BF16 pre-FFN state to the exact resident R25 W4 activation ABI.
// This is mutable activation preparation only: immutable weight/block order
// remains an offline .rdna2.hfp concern.

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int ROWS = 8;
constexpr int GROUP = 256;
constexpr int HIDDEN = 768;
constexpr int Q_BYTES = ROWS * GROUP;
constexpr int SCALE_BYTES = ROWS * sizeof(float);
constexpr int CHUNK_BYTES = Q_BYTES + SCALE_BYTES;
constexpr int BLOCK_BYTES = 3 * Q_BYTES + 3 * SCALE_BYTES;
#ifdef R93_VECTOR_PREP
constexpr int PARAM_AWQ = 0;
constexpr int PARAM_SIGN1 = PARAM_AWQ + GROUP * sizeof(float);
constexpr int PARAM_SIGN2 = PARAM_SIGN1 + GROUP * sizeof(float);
constexpr int PARAM_BYTES = PARAM_SIGN2 + GROUP * sizeof(float);
#else
constexpr int PARAM_NORM = 0;
constexpr int PARAM_AWQ = PARAM_NORM + GROUP * sizeof(float);
constexpr int PARAM_SIGN1 = PARAM_AWQ + GROUP * sizeof(float);
constexpr int PARAM_SIGN2 = PARAM_SIGN1 + GROUP * sizeof(bfloat16);
constexpr int PARAM_BYTES = PARAM_SIGN2 + GROUP * sizeof(bfloat16);
#endif

#ifndef R93_VECTOR_PREP
__attribute__((noinline)) static void
prepare8(const bfloat16 *restrict input, const float *restrict norm,
         const float *restrict awq, const bfloat16 *restrict signs,
         float *restrict scratch) {
  scratch[0] = (float)input[0] * norm[0] / awq[0] * (float)signs[0];
  scratch[1] = (float)input[1] * norm[1] / awq[1] * (float)signs[1];
  scratch[2] = (float)input[2] * norm[2] / awq[2] * (float)signs[2];
  scratch[3] = (float)input[3] * norm[3] / awq[3] * (float)signs[3];
  scratch[4] = (float)input[4] * norm[4] / awq[4] * (float)signs[4];
  scratch[5] = (float)input[5] * norm[5] / awq[5] * (float)signs[5];
  scratch[6] = (float)input[6] * norm[6] / awq[6] * (float)signs[6];
  scratch[7] = (float)input[7] * norm[7] / awq[7] * (float)signs[7];
}
#endif

#ifndef R93_VECTOR_PREP
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
#else
__attribute__((noinline)) static float
post_sign16(float *restrict scratch, const float *restrict signs) {
  const auto scaled =
      aie::mul(aie::mul(aie::load_v<16>(scratch),
                        aie::load_unaligned_v<16>(signs))
                   .to_vector<float>(),
               aie::broadcast<float, 16>(0.0625f))
          .to_vector<float>();
  aie::store_v(scratch, scaled);
  return aie::reduce_max(aie::abs(scaled));
}
#endif

static inline void copy_chunk_to_block(const int8_t *restrict chunk,
                                       int8_t *restrict block, int owner) {
  const int q_base = owner * Q_BYTES;
  const int scale_base = 3 * Q_BYTES + owner * SCALE_BYTES;
  for (int offset = 0; offset < Q_BYTES; offset += 64)
    aie::store_v(block + q_base + offset,
                 aie::load_v<64>(chunk + offset));
  aie::store_v(block + scale_base,
               aie::load_v<32>(chunk + Q_BYTES));
}
} // namespace

extern "C" {

void r93_pack_rows(const int8_t *restrict input_bytes,
                   const int8_t *restrict param_bytes,
                   int8_t *restrict chunk, float *restrict scratch,
                   int32_t row, int32_t group) {
#ifdef R93_ROUTE_PROBE
  (void)input_bytes;
  (void)param_bytes;
  (void)scratch;
  const int8_t marker = (int8_t)(1 + group * ROWS + row);
  for (int i = 0; i < GROUP; ++i) {
    const int local_m = row >> 2;
    const int local_row = row & 3;
    const int kt = i >> 4;
    const int kk = i & 15;
    chunk[(local_m * 16 + kt) * 64 + local_row * 16 + kk] = marker;
  }
  *reinterpret_cast<float *>(chunk + Q_BYTES + row * sizeof(float)) =
      (float)marker;
  return;
#else
  const auto *input = reinterpret_cast<const bfloat16 *>(input_bytes)
                      + group * GROUP;
  const auto *awq = reinterpret_cast<const float *>(param_bytes + PARAM_AWQ);
#ifdef R93_VECTOR_PREP
  const auto *sign1 = reinterpret_cast<const float *>(param_bytes + PARAM_SIGN1);
  const auto *sign2 = reinterpret_cast<const float *>(param_bytes + PARAM_SIGN2);
#else
  const auto *norm = reinterpret_cast<const float *>(param_bytes + PARAM_NORM);
  const auto *sign1 =
      reinterpret_cast<const bfloat16 *>(param_bytes + PARAM_SIGN1);
  const auto *sign2 =
      reinterpret_cast<const bfloat16 *>(param_bytes + PARAM_SIGN2);
#endif

#ifdef R93_LOAD_PROBE
#ifdef R93_VECTOR_PREP
  const float loaded = (float)input[0] / awq[0] * sign1[0] * sign2[0];
#else
  const float loaded = (float)input[0] * norm[0] / awq[0]
                       * (float)sign1[0] * (float)sign2[0];
#endif
  const int8_t marker = loaded >= 0.0f ? 37 : -37;
  for (int i = 0; i < GROUP; ++i) {
    const int local_m = row >> 2;
    const int local_row = row & 3;
    const int kt = i >> 4;
    const int kk = i & 15;
    chunk[(local_m * 16 + kt) * 64 + local_row * 16 + kk] = marker;
  }
  *reinterpret_cast<float *>(chunk + Q_BYTES + row * sizeof(float)) = loaded;
  return;
#endif

#ifdef R93_VECTOR_PREP
  for (int i = 0; i < GROUP; i += 16) {
  const auto values =
        aie::mul(aie::load_v<16>(input + i),
                 aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
            .to_vector<float>();
    const auto divided =
        aie::div(values, aie::load_unaligned_v<16>(awq + i)).to_vector<float>();
    aie::store_v(scratch + i,
                 aie::mul(divided, aie::load_unaligned_v<16>(sign1 + i))
                     .to_vector<float>());
  }
#else
  for (int i = 0; i < GROUP; i += 8)
    prepare8(input + i, norm + i, awq + i, sign1 + i, scratch + i);
#endif
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
    const int local_m = row >> 2;
    const int local_row = row & 3;
    const int kt = i >> 4;
    const int kk = i & 15;
    chunk[(local_m * 16 + kt) * 64 + local_row * 16 + kk] = quantized;
  }
  aie::set_saturation(old_saturation);
  aie::set_rounding(old_rounding);
#endif
}

__attribute__((noinline, minsize)) void
r93_copy_param(const int8_t *restrict input, int8_t *restrict params) {
  for (int offset = 0; offset < PARAM_BYTES; offset += 32)
    aie::store_v(params + offset, aie::load_v<32>(input + offset));
}

__attribute__((noinline, minsize)) void
r93_send_chunk(const int8_t *restrict chunk) {
  const int32_t *words = reinterpret_cast<const int32_t *>(chunk);
  for (int word = 0; word < CHUNK_BYTES / (int)sizeof(int32_t); ++word)
    put_ms(words[word]);
}

__attribute__((noinline, minsize)) void
r93_relay_then_send(const int8_t *restrict chunk) {
  for (int word = 0; word < CHUNK_BYTES / (int)sizeof(int32_t); ++word)
    put_ms(get_ss_int());
  r93_send_chunk(chunk);
}

__attribute__((noinline, minsize)) void
r93_assemble_block(int8_t *restrict block, const int8_t *restrict own,
                   int32_t predecessors) {
  for (int offset = 0; offset < BLOCK_BYTES; offset += 32)
    aie::store_v(block + offset, aie::zeros<int8_t, 32>());
  for (int owner = 0; owner < predecessors; ++owner) {
    for (int word = 0; word < Q_BYTES / (int)sizeof(int32_t); ++word)
      reinterpret_cast<int32_t *>(block + owner * Q_BYTES)[word] = get_ss_int();
    for (int word = 0; word < SCALE_BYTES / (int)sizeof(int32_t); ++word)
      reinterpret_cast<int32_t *>(block + 3 * Q_BYTES
                                  + owner * SCALE_BYTES)[word] = get_ss_int();
  }
  copy_chunk_to_block(own, block, predecessors);
}

__attribute__((noinline, minsize)) void
r93_emit_block(const int8_t *restrict block, int8_t *restrict output) {
  for (int offset = 0; offset < BLOCK_BYTES; offset += 32)
    aie::store_v(output + offset, aie::load_v<32>(block + offset));
}

} // extern "C"
