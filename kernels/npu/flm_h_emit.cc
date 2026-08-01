// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// Emit this core's whole slice of `h` from `g_resid` into one result object.
//
// **This exists so P4 can be broadcast a dense `h`.** P3 emits one object per
// tile with NROWS live values in a DIM_HEAD*2-element object, which is 12%
// dense; a drain consumes its source linearly, so those padding elements cannot
// be dropped on the way out and the gathered vector P4 needs cannot be built
// from them. Writing all of a core's tiles into a single object instead makes
// the drain dense.
//
// It costs nothing extra to compute: `flm_gemv_q4_1_residual` already stashes
// every row it produces in `g_resid`, because P5's flush needs exactly those
// rows on exactly this core. This kernel only copies that slice out.
//
// The sizes line up with no padding at all: a core owns DIM_RESN = 128 rows
// (2048 over 16 cores), which is p3tiles * NROWS = 8 * 16, and equals the
// 2*DIM_HEAD result object P3 shares with P1. One object, fully live.
//
// Compile-time: -DDIM_RESN, -DDIM_HEAD.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_RESN
#define DIM_RESN 256            // a PAIR's row span — see the gather below
#endif
#ifndef DIM_HEAD
#define DIM_HEAD 64
#endif
#ifndef DIM_NROWS
#define DIM_NROWS 16
#endif
#ifndef DIM_P3TILES
#define DIM_P3TILES 8
#endif
#ifndef DIM_GROUP
#define DIM_GROUP 2             // cores sharing one operand fifo: 2 pair, 4 quad
#endif

namespace {
constexpr int RESN = DIM_RESN;
constexpr int NR = DIM_NROWS;
constexpr int TILES = DIM_P3TILES;
constexpr int GRP = DIM_GROUP;
// A core's slice must FIT the shared result object, not fill it. At 16 cores in
// pairs it happens to fill it exactly (8*16 == 2*64); at 32 cores in quads a core
// owns half as many rows and the object is half-written, which is legal — the
// drain reads only TILES*NR per core either way.
static_assert(TILES * NR <= 2 * DIM_HEAD,
              "h's slice must fit the shared result object");
static_assert(RESN == GRP * TILES * NR, "g_resid spans the GROUP's rows");
} // namespace

extern float g_resid[];

#include "flm_q4_1_tile.h"       // tile_row_base

extern "C" __attribute__((noinline)) void
flm_h_emit(const uint8 *restrict wtile, bfloat16 *restrict out) {
  // g_resid is indexed `base % DIM_RESN` and DIM_RESN spans a PAIR, so a
  // single core's rows are SCATTERED through it: the two cores of a pair
  // interleave at NROWS granularity, giving this core slots
  // {t*2*NR + j*NR + r}. Gather them back into a dense slice.
  //
  // A core-sized DIM_RESN would collide instead of scattering — rows 0 and 128
  // land in the same slot — which is why resid_chain sizes it per pair.
  const int base = tile_row_base(wtile);
  // Which core within the group: rows interleave at NROWS granularity across all
  // GRP cores, so the modulus spans the group, not just a pair.
  const int j = (base % (GRP * NR)) / NR;
  // **Do not let this unroll.** TILES*NR is 128 iterations, and fully unrolled
  // it costs over 960 B of program memory — enough to push a core running
  // P1+P3+P4+P5 past 16 KB, which is measured, not hypothetical. As a rolled
  // loop it is a handful of instructions and the copy is off the critical path.
#pragma clang loop unroll(disable)
  for (int i = 0; i < TILES * NR; ++i) {
    const int t = i / NR, r = i - t * NR;
    out[i] = bfloat16(g_resid[t * GRP * NR + j * NR + r]);
  }
}
