// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// Emit one finished head of phase P1 — q', k' or v' — in the form its
// destination needs. The ROUTING is done by the runtime sequence, one drain per
// destination taking successive parts of the stream
// (`tools/npu/flm/qkv_route_probe.py`); this kernel only sets the form.
//
// Separate from `flm_qkv_emit`, which stays the plain 64-element emit used by
// `qkv_verify.py`. Folding the branch into that kernel changes its result-object
// size from HEAD to 2*HEAD and breaks every existing caller, so the two forms
// are two entry points.
//
// **One kernel handles all three head types because the core cannot choose at
// trace time.** All 16 cores run the same body, and which of a core's three
// heads is q, k or v depends on where they fall in 0..47 — so the branch is at
// run time, on the head index the weight tile's `row_base` already carries.
//
// The result object is 2*HEAD for every head. Only k' needs the doubled form (it
// writes a column pair), so q' and v' use the first HEAD and their drains skip
// the rest with a stride. The waste is result bandwidth only: 40 heads x 128 B
// = 5 KB per layer, alongside 38 MB of weights.
//
// This is a separate entry point rather than the tail of `flm_gemv_qkv`
// because the result fifo is acquired by the worker, not by the kernel: only
// one tile in four closes a head, so the acquire cannot sit inside a call that
// runs on every tile without stalling the other three.
//
// 256 B per head: 2*HEAD bf16, of which q' and v' use the first half.
//
// Compile-time: -DDIM_HEAD.

#include "flm_q4_1_tile.h"
#include "flm_kv_pair.h"

#ifndef DIM_QHEADS
#define DIM_QHEADS 32           // q heads: 2048 / 64
#endif
#ifndef DIM_QKHEADS
#define DIM_QKHEADS 40          // + k heads: 512 / 64
#endif
#ifndef DIM_QGROUP
#define DIM_QGROUP 1            // q heads per RESULT OBJECT
#endif

namespace {
constexpr int HEAD = DIM_HEAD;
// How many q heads share one result object. 1 is the paired design: one head per
// object, written at offset 0. The role architecture needs GQA of them in ONE
// object, because attention acquires its q once and holds it across the whole KV
// loop, reading q[h * QSTRIDE + d] for h in 0..GQA-1 — four separate objects
// would not be contiguous.
constexpr int QGROUP = DIM_QGROUP;
constexpr int VLANES = 32;
static_assert(HEAD % VLANES == 0, "a head must be a whole number of vectors");
} // namespace

extern bfloat16 g_stage[];

extern "C" __attribute__((noinline)) void
flm_p1_emit(const uint8 *restrict wtile, bfloat16 *restrict out) {
  const int head = tile_row_base(wtile) / HEAD;
  if (head >= DIM_QHEADS && head < DIM_QKHEADS) {
    // k': the column-pair form, carrying the previous token when this step
    // closes a pair. tile_flags() is the KV-cache position.
    flm_kv_pair(g_stage, tile_flags(wtile), out);
    return;
  }
  // Which slot of the result object this head occupies. Only Q heads pack into
  // a group: they share one object so attention can acquire it once and read
  // q[h * QSTRIDE + d] across all GQA. A v' head gets its own object at offset
  // 0 — grouping it would put it at head % QGROUP, which varies with the head
  // index and is not where the cache drain looks.
  const int slot = (QGROUP == 1 || head >= DIM_QHEADS) ? 0 : (head % QGROUP);
  bfloat16 *restrict dst = out + slot * 2 * HEAD;

  // q' and v': the head in the first half of its slot, ZEROS in the second.
  //
  // The zeros are not padding for its own sake. A drain consumes its source
  // LINEARLY — `sizes`/`strides` shape only the destination walk — so there is
  // no way to skip the unused half on the way out. Whatever is in it gets
  // written. For v' that lands on cache row pos+1, which is a future position:
  // zero is exactly what attention's `npad` correction wants there, and the
  // next token overwrites it. For q' the host simply ignores it.
  for (int i = 0; i < HEAD; i += VLANES) {
    aie::store_v(dst + i, aie::load_v<VLANES>(g_stage + i));
    aie::store_v(dst + HEAD + i, aie::zeros<bfloat16, VLANES>());
  }
}
