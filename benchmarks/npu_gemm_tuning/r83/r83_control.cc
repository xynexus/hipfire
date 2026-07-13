// SPDX-License-Identifier: Apache-2.0
// Keep Peano from cloning the complete attention driver sixteen times.

#include <stdint.h>

extern "C" __attribute__((noinline, minsize)) int32_t r83_attention_blocks() {
  return 16;
}

extern "C" __attribute__((noinline, minsize)) int32_t r83_projection_slices() {
  return 3;
}

extern "C" __attribute__((noinline, minsize)) int32_t r84_output_pairs() {
  return 12;
}

extern "C" __attribute__((noinline, minsize)) int32_t r84_output_waves() {
  return 2;
}

extern "C" __attribute__((noinline, minsize)) int32_t r89_q_groups() {
  return 6;
}

extern "C" __attribute__((noinline, minsize)) int32_t r89_output_chunks() {
  return 6;
}

extern "C" __attribute__((noinline, minsize)) int32_t r90_projection_blocks() {
  return 18;
}
