// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// C2/C5: feed and drain bandwidth.
//
// The DMA moves a whole tile regardless of how much of it the core reads, so
// the cheapest possible consumer isolates the transfer: touch one word per
// tile, accumulate, and let the ObjectFIFO do the work. Any time that scales
// with bytes is transport, not compute.
//
// Deliberately NOT a sum over the tile: summing would make the core the
// limiter and we would measure arithmetic instead of bandwidth.

#include <stdint.h>

extern "C" void c2_sink(const int32_t *__restrict in, int32_t *__restrict acc) {
  // One word per tile: enough to keep the buffer live, cheap enough to stay
  // far below the DMA's delivery rate.
  acc[0] += in[0];
}
