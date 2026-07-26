// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// D1: does DMA feed actually OVERLAP core compute?
//
// The model's central composition claim is
//
//     T_device ~= fill + max(t_feed, t_core, t_drain) + tail
//
// a MAX, not a sum. That is what lets it reproduce R117 (more work, same fixed
// cost, LESS time). But nothing validated it: family B is feed-only with a
// trivial core, family C is compute-only with no feed. Both terms were checked
// in isolation and the composition never was.
//
// This kernel does both: it consumes a streamed tile AND runs MMULS mmul chains
// on resident operands per tile. Sweeping MMULS moves compute from far below the
// feed time to far above it:
//
//   max model:  time/tile = max(t_feed, t_core)  -> FLAT while t_core < t_feed,
//                                                   then rises with slope 1
//   sum model:  time/tile = t_feed + t_core      -> rises from the very first
//                                                   step
//
// At the crossover the two predictions differ by 2x, so a handful of points
// settles it.
//
// The tile is touched (acc) so the DMA cannot be elided; the mmul chain is
// resident so it adds compute without adding traffic.

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef MMULS
#define MMULS 0
#endif
#ifndef CHAINS
#define CHAINS 4
#endif

using MMUL = aie::mmul<4, 8, 8, int8, int8>;

extern "C" void d1_overlap(const int32_t *__restrict in, int32_t *__restrict acc) {
  // Touch the streamed tile: keeps the DMA live and real.
  acc[0] += in[0];

#if MMULS > 0
  // Resident operands: compute with no extra traffic. Four independent chains,
  // matching K1's structure so the VMAC pipe is saturated rather than
  // latency-bound.
  const int8 *p = (const int8 *)in;
  aie::vector<int8, MMUL::size_A> a0 = aie::load_v<MMUL::size_A>(p);
  aie::vector<int8, MMUL::size_B> b0 = aie::load_v<MMUL::size_B>(p + MMUL::size_A);
  MMUL c0, c1, c2, c3;
  c0.mul(a0, b0);
  c1.mul(a0, b0);
  c2.mul(a0, b0);
  c3.mul(a0, b0);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < MMULS; i++) {
    c0.mac(a0, b0);
    c1.mac(a0, b0);
    c2.mac(a0, b0);
    c3.mac(a0, b0);
  }
  auto s = aie::add(aie::add(c0.template to_vector<int32>(), c1.template to_vector<int32>()),
                    aie::add(c2.template to_vector<int32>(), c3.template to_vector<int32>()));
  // DCE guard that cannot perturb the accumulator's value in practice.
  if (s.get(0) == 0x7FFFFFFF) acc[1] = 1;
#endif
}
