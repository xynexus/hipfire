// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// C5: drain bandwidth. The mirror of C2's sink.
//
// The core produces tiles as cheaply as it can — write one word, leave the rest
// of the buffer alone — so the DMA drain is the only thing scaling with bytes.
// R64 measured the output DMA starved for 198 us of a 241 us device span, so
// the drain path deserves its own constant rather than an assumption that it
// matches the feed.

#include <stdint.h>

extern "C" void c5_src(int32_t *__restrict out) {
  // One word per tile. The DMA still moves the whole tile.
  // (No loop-index argument: range_()'s induction variable is MLIR `index`,
  // which does not implicitly convert to the i32 the kernel would declare.)
  out[0] = 1;
}
