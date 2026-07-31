// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// Fused q/k/v projection with RoPE in the epilogue — phase P1 of the fused
// decoder layer (`docs/npu/flm-fused-layer-plan.md` §1.4, Task 5).
//
// q, k and v read the same activation, so they are ONE N=3072 projection
// (2048 q + 512 k + 512 v), not three. FLM does the same — its 6144 B split
// core. At 16 cores that is 192 rows/core, and 192 = 3 x 64, so **every core
// owns exactly three whole heads and RoPE never straddles a core**. That is
// the reason this phase runs on 16 cores and not 32.
//
// A head is 64 rows = 4 tiles at NROWS=16. Tiles arrive in order, so the
// kernel stages each tile's rows into a 128 B in-core buffer and, on the tile
// that completes a head (`row_base % 64 == 48`), rotates it in place. The
// caller then emits the head with `flm_qkv_emit`. Nothing but the finished
// head ever leaves the core, and no q/k/v intermediate is materialised in
// memory.
//
// Which head this is comes from the weight tile's `row_base` trailer, so there
// are no runtime scalar arguments and no static cursor to get out of step:
//
//     head = row_base / 64;  q if head < 32,  k if head < 40,  else v (no RoPE)
//
// `cs_q` and `cs_k` ride the tail of the broadcast object
// ([act][aux][cs_q][cs_k]), so this needs no third DMA input — a core tile has
// only two. `cs_q` is pre-multiplied by `head_dim^-0.5 * log2(e)` at pack
// time, which is why attention's `exp2` needs no pre-scale and the host never
// touches q' mid-dispatch.
//
// **Only the half-split pairing (i, i+HEAD/2) is implemented, and that is
// sufficient for both conventions.** The plan asked for a `-DROPE_INTERLEAVED`
// flag, but the interleaved pairing (2i, 2i+1) is the half-split pairing
// applied to a permuted row order — which is exactly *why* llama.cpp's
// converter permutes q/k weights rather than shipping a second RoPE. So the
// convention is selected where it belongs, at **pack time**, by reordering the
// tile's rows within a head; the kernel stays one code path and 8 instructions
// shorter than a shuffle-network version measured against it.
//
// That is safe for attention because q.k is a dot product over the head
// dimension: **any permutation shared by q and k leaves it unchanged**, and v
// is never rotated so it keeps the model's order for o_proj. See
// `tools/npu/flm/qkv_verify.py --interleaved`, which packs the permuted order
// and checks it against an interleaved numpy reference.
//
// Which convention the container actually wants is still OPEN. It cannot be
// settled from the weights — they are transformed, and
// `tools/npu/flm/ground_truth.py` shows the container's blocks are not the
// checkpoint's under any arrangement. A probe that tried to read the pairing
// out of the row order was built and discarded: its v_proj control, which gets
// no RoPE at all, showed the same signal more strongly. See
// `docs/npu/flm-refe-log.md`, 2026-07-31.
//
// Compile-time: -DDIM_K -DDIM_NROWS -DDIM_HEAD -DDIM_ACT.

#include "flm_q4_1_tile.h"

#ifndef DIM_HEAD
#define DIM_HEAD 64
#endif
#ifndef DIM_ACT                 // bf16 elements in the broadcast's act half
#define DIM_ACT 2048
#endif
namespace {
constexpr int HEAD = DIM_HEAD;
constexpr int RHALF = HEAD / 2;          // one rotation plane per pair
constexpr int Q_HEADS = 32;              // 2048 / 64
constexpr int QK_HEADS = 40;             // + 512 / 64
// [act DIM_ACT][aux DIM_ACT][cs_q RHALF cos + RHALF sin][cs_k ...]
constexpr int CS_Q = 2 * DIM_ACT;
constexpr int CS_K = CS_Q + HEAD;
static_assert(HEAD % NROWS == 0, "a head must be a whole number of tiles");
static_assert(RHALF % 16 == 0, "half a head must be a whole number of vectors");
} // namespace

// The head under construction. 128 B, and alignas is load-bearing — an
// unaligned 512-bit load returns garbage rather than faulting.
alignas(64) bfloat16 g_stage[HEAD];

namespace {
// Rotate the staged head in place. `cs` is [cos RHALF][sin RHALF].
// HF `rotate_half`: the pair is (i, i + HEAD/2), already contiguous halves, so
// this is two loads and no shuffle. The interleaved convention reaches the same
// result through the pack-time row order — see the header comment.
inline void rope_stage(const bfloat16 *restrict cs) {
  const auto c = aie::load_v<RHALF>(cs);
  const auto s = aie::load_v<RHALF>(cs + RHALF);
  const auto x = aie::load_v<RHALF>(g_stage);
  const auto y = aie::load_v<RHALF>(g_stage + RHALF);

  auto lo = aie::mul(x, c);              // x*cos
  lo = aie::msc(lo, y, s);               //      - y*sin
  auto hi = aie::mul(y, c);              // y*cos
  hi = aie::mac(hi, x, s);               //      + x*sin

  aie::store_v(g_stage, lo.template to_vector<bfloat16>());
  aie::store_v(g_stage + RHALF, hi.template to_vector<bfloat16>());
}
} // namespace

extern "C" __attribute__((noinline)) void
flm_gemv_qkv(const bfloat16 *restrict bcast, const uint8 *restrict wtile) {
  float acc[NROWS];
  flm_q4_1_tile(bcast, wtile, acc);

  const int row_base = tile_row_base(wtile);
  const int off = row_base % HEAD;
  for (int r = 0; r < NROWS; ++r)
    g_stage[off + r] = bfloat16(acc[r]);

  if (off + NROWS == HEAD) {             // this tile closed the head
    const int head = row_base / HEAD;
    if (head < Q_HEADS)
      rope_stage(bcast + CS_Q);
    else if (head < QK_HEADS)
      rope_stage(bcast + CS_K);
    // v is not rotated — it leaves the stage untouched.
  }
}
