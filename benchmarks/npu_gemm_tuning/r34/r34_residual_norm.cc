// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int ROWS = 8;
constexpr int HIDDEN = 768;
constexpr int BLOCK_COLS = 256;
constexpr int SLICE_COLS = 64;
constexpr int PARAM_RESIDUAL = 0;
constexpr int PARAM_POST_NORM = ROWS * HIDDEN;
constexpr int PARAM_PRE_NORM = PARAM_POST_NORM + HIDDEN;
constexpr int PARAM_EPSILON_BYTES = (PARAM_PRE_NORM + HIDDEN) * 2;
#ifdef R42_EXCEPTION_COLUMN
constexpr int EXCEPTION_PHYSICAL_COLUMN = R42_EXCEPTION_COLUMN;
constexpr int EXCEPTION_DIM = EXCEPTION_PHYSICAL_COLUMN % BLOCK_COLS;
#if R42_EXCEPTION_COLUMN < 256
#define R42_EXCEPTION_BLOCK block0
#elif R42_EXCEPTION_COLUMN < 512
#define R42_EXCEPTION_BLOCK block1
#else
#define R42_EXCEPTION_BLOCK block2
#endif
#endif
constexpr float INV_HIDDEN = 1.0f / (float)HIDDEN;

static inline bfloat16 *select_block(int8_t *block0, int8_t *block1,
                                    int8_t *block2, int index) {
  return reinterpret_cast<bfloat16 *>(index == 0 ? block0
                                      : index == 1 ? block1
                                                   : block2);
}

} // namespace

extern "C" {
#if !defined(R42_SPLIT_OBJECTS) || defined(R42_BUILD_OUTPUT)
void r34_output_projection_finish_pair_bf16(
    const float *restrict accum0, const float *restrict accum1,
    int8_t *restrict block0, int8_t *restrict block1, int8_t *restrict block2,
    int32_t pair_index) {
  bfloat16 *output = select_block(block0, block1, block2, pair_index >> 2);
  const int column_base = (pair_index & 3) * SLICE_COLS;
  for (int slice = 0; slice < 2; ++slice) {
    const float *accum = slice == 0 ? accum0 : accum1;
    for (int mt = 0; mt < 2; ++mt)
      for (int row = 0; row < 4; ++row)
        for (int nt = 0; nt < 4; ++nt) {
          const int source = (mt * 4 + nt) * 32 + row * 8;
          const int target = (mt * 4 + row) * BLOCK_COLS + column_base
                             + slice * 32 + nt * 8;
          aie::store_v(output + target,
                       aie::mul(aie::load_v<8>(accum + source), 1.0f)
                           .to_vector<bfloat16>());
        }
  }
}
#endif

#if !defined(R42_SPLIT_OBJECTS) || defined(R42_BUILD_POST)
void r34_post_residual_pre_ffn(int8_t *restrict block0,
                               int8_t *restrict block1,
                               int8_t *restrict block2,
                               const int8_t *restrict params_bytes,
                               int8_t *restrict metadata_bytes,
                               int32_t wave) {
  const auto *params = reinterpret_cast<const bfloat16 *>(params_bytes);
  const auto *residual = params + PARAM_RESIDUAL;
  const auto *post_norm = params + PARAM_POST_NORM;
  const auto *pre_norm = params + PARAM_PRE_NORM;
  auto *metadata = reinterpret_cast<float *>(metadata_bytes) + wave * ROWS;
  const float epsilon =
      *reinterpret_cast<const float *>(params_bytes + PARAM_EPSILON_BYTES);
  bfloat16 *blocks[] = {
      reinterpret_cast<bfloat16 *>(block0),
      reinterpret_cast<bfloat16 *>(block1),
      reinterpret_cast<bfloat16 *>(block2),
  };

  for (int row = 0; row < ROWS; ++row) {
    float output_sum = 0.0f;
    for (int block = 0; block < 3; ++block) {
      const bfloat16 *values = blocks[block] + row * BLOCK_COLS;
      for (int dim = 0; dim < BLOCK_COLS; dim += 16) {
        const auto value = aie::load_v<16>(values + dim);
        output_sum += aie::reduce_add(aie::mul(value, value).to_vector<float>());
      }
    }
    const float post_inverse =
        aie::invsqrt(output_sum * INV_HIDDEN + epsilon);

    float residual_sum = 0.0f;
    for (int block = 0; block < 3; ++block) {
      bfloat16 *values = blocks[block] + row * BLOCK_COLS;
      const int hidden_base = block * BLOCK_COLS;
      for (int dim = 0; dim < BLOCK_COLS; dim += 16) {
        const int hidden = hidden_base + dim;
        auto normalized =
            aie::mul(aie::load_v<16>(values + dim),
                     aie::load_v<16>(post_norm + hidden))
                .to_vector<float>();
        normalized = aie::mul(normalized, post_inverse).to_vector<float>();
        const auto residual_value =
            aie::mul(aie::load_v<16>(residual + row * HIDDEN + hidden),
                     aie::broadcast<bfloat16, 16>((bfloat16)1.0f))
                .to_vector<float>();
        const auto x = aie::add(normalized, residual_value);
        residual_sum += aie::reduce_add(aie::mul(x, x).to_vector<float>());
        aie::store_v(values + dim,
                     aie::mul(x, 1.0f).to_vector<bfloat16>());
      }
    }
    const float pre_inverse =
        aie::invsqrt(residual_sum * INV_HIDDEN + epsilon);
#ifdef R42_EXCEPTION_COLUMN
    metadata[row] = pre_inverse;
    // Keep the selected exception block distinct from the three-pointer array
    // used above. Without chess_copy Peano aliases block0's late reload with
    // block2 while optimizing the restricted pointers.
    const auto *exception_block = reinterpret_cast<const bfloat16 *>(
        chess_copy(R42_EXCEPTION_BLOCK));
    const auto exception_vector = aie::load_v<16>(
        exception_block + row * BLOCK_COLS + (EXCEPTION_DIM & -16));
    reinterpret_cast<bfloat16 *>(metadata_bytes + 64 + wave * ROWS * 4)
        [row * 2] = exception_vector[EXCEPTION_DIM & 15];
#else
    metadata[row] = pre_inverse;
#endif
    for (int block = 0; block < 3; ++block) {
      bfloat16 *values = blocks[block] + row * BLOCK_COLS;
      const int hidden_base = block * BLOCK_COLS;
      for (int dim = 0; dim < BLOCK_COLS; dim += 16) {
        const auto normalized =
            aie::mul(aie::load_v<16>(values + dim),
                     aie::load_v<16>(pre_norm + hidden_base + dim))
                .to_vector<float>();
        aie::store_v(values + dim,
                     aie::mul(normalized, pre_inverse).to_vector<bfloat16>());
      }
    }
  }
}
#endif

#ifdef R42_EXCEPTION_COLUMN
#undef R42_EXCEPTION_BLOCK
#endif

#if !defined(R42_SPLIT_OBJECTS) || defined(R42_BUILD_EMIT)
__attribute__((noinline, minsize, cold)) void
r34_emit_norm_half(const int8_t *restrict block, int8_t *restrict output,
                   int32_t half) {
  const int source_half = half * 2048;
  block += source_half;
  for (int chunk = 0; chunk < 32; ++chunk) {
    aie::store_v(output, aie::load_v<64>(block));
    block += 64;
    output += 64;
  }
}
#endif

#if !defined(R42_SPLIT_OBJECTS) || defined(R42_BUILD_RELAY)
__attribute__((noinline, minsize)) void
r38_relay_pre_inverse(const int8_t *restrict input,
                      int8_t *restrict output) {
  const auto values =
      aie::load_v<16>(reinterpret_cast<const float *>(input));
  aie::store_v(reinterpret_cast<float *>(output), values.extract<8>(0));
  aie::store_v(reinterpret_cast<float *>(output + 1024),
               values.extract<8>(1));
#ifdef R42_EXCEPTION_COLUMN
  const auto exception_state =
      aie::load_v<16>(reinterpret_cast<const uint32_t *>(input + 64));
  aie::store_v(reinterpret_cast<uint32_t *>(output + 32),
               exception_state.extract<8>(0));
  aie::store_v(reinterpret_cast<uint32_t *>(output + 1056),
               exception_state.extract<8>(1));
#endif
}
#endif

}
