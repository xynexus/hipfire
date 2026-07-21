// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// FLASH-STYLE streaming DFlash non-causal cross-attention, one q-head per
// "iteration", KV delivered in fixed-size tiles.
//
// Replaces dflash_attention_sc_bf16.cc's two structural limits:
//
//   1. Throughput. The old kernel scored each (q,k) pair with 8 aie::mul +
//      8 aie::add + a full aie::reduce_add (~2400 core cycles per 128-length
//      dot). Here both GEMMs (S = Q·Kᵀ and O += P·V) run through
//      aie::mmul<4,8,4,bfloat16,bfloat16> — the same shape AMD's own
//      aie_kernels/aie2/mm.cc uses for bf16 — at 128 MACs/instruction.
//
//   2. tot ≤ 55. The old design pinned one kv-head's ENTIRE KV in core-tile
//      L1 (memKV = 512·tot bytes), so tot=56 blew the 64 KiB data memory.
//      Here L1 holds ONE KV tile (KV_TILE rows), so core memory is independent
//      of tot and the cap is gone. That forces ONLINE softmax: a running max
//      and running sum, with the O accumulator rescaled as each tile arrives.
//
// Data layouts are mmul-tiled (pre-tiled by the host, see
// build_dflash_attention_flash.py). All tiles row-major, tiles themselves in
// row-major order — the convention documented in aie_kernels/aie2/mm.cc.
//
//   Q  : A-layout, (Q_LEN/4) x (128/8) tiles of 4x8 bf16
//   KV : per tile, [ Kt | V | M ]
//          Kt : B-layout of Kᵀ (128 x KV_TILE), (128/8) x (KV_TILE/4) tiles of 8x4
//          V  : B-layout of V  (KV_TILE x 128), (KV_TILE/8) x (128/4) tiles of 8x4
//          M  : KV_TILE float32 additive score mask, occupying 2*KV_TILE of the
//               bfloat16-typed buffer's trailing slots (see below)
//   O  : C-layout, (Q_LEN/4) x (128/4) tiles of 4x4 bf16
//
// TAIL MASKING. v1 required KV_TILE to divide `tot` exactly, which the
// spec-decode loop cannot honour (tot = L + B grows by one token per cycle).
// The fix is an ADDITIVE per-KV-row score mask carried as runtime data in the
// tile itself: 0.0f for a real KV row, MASK_NEG for a padding row. `tot` is
// padded up to a multiple of KV_TILE host-side and the pad rows are masked, so
// only N_TILES stays compile-time and it changes once per KV_TILE tokens
// instead of every cycle.
//
// The mask MUST be additive-negative, not zeroed K/V: with online softmax a
// zeroed K row scores 0, and exp(0 - m) is a perfectly ordinary weight that
// inflates the running sum `l` and scales the output down. Only a
// -inf-equivalent score removes the row from the distribution.
//
// The mask is float32 but the ObjectFifo is typed bfloat16 (one element type
// per fifo), so it rides in the buffer's trailing 2*KV_TILE bfloat16 slots and
// is read back through a float* reinterpret. That keeps its offset
// (2*128*KV_TILE elements = 512*KV_TILE bytes) 64-byte aligned for any
// KV_TILE that is a multiple of 16, which the static_assert below already
// requires.
//
// Call protocol (driven from the IRON core_fn): O is acquired ONCE per head and
// held across the whole tile loop, so a single entry point suffices —
//
//   step(Q, KV, O) x N_TILES
//
// State auto-initialises on tile 0 and the normalised result is written (and
// the state reset) on tile N_TILES-1. One entry point matters practically:
// IRON compiles each ExternalFunction from the whole .cc, so two exported
// symbols in one file collide at link time as duplicates.
//
// Softmax state lives in file-scope statics (core .bss), not on the stack, so
// the ObjectFifo buffers keep the tile budget.

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef HIPFIRE_Q_LEN
#define HIPFIRE_Q_LEN 16
#endif
#ifndef HIPFIRE_KV_TILE
#define HIPFIRE_KV_TILE 32
#endif
#ifndef HIPFIRE_N_TILES
#define HIPFIRE_N_TILES 3
#endif

namespace {

constexpr int D = 128;              // head_dim
constexpr int QLEN = HIPFIRE_Q_LEN;
constexpr int KVT = HIPFIRE_KV_TILE;
constexpr int NTILES = HIPFIRE_N_TILES;

constexpr int MR = 4, MS = 8, MT = 4;   // aie::mmul<r, s, t>
constexpr int SZ_A = MR * MS;           // 32
constexpr int SZ_B = MS * MT;           // 32
constexpr int SZ_C = MR * MT;           // 16

constexpr int QB = QLEN / MR;   // q tile-rows
constexpr int DB = D / MS;      // reduction blocks for S  (16)
constexpr int KB = KVT / MT;    // output blocks for S
constexpr int VB = KVT / MS;    // reduction blocks for PV
constexpr int OB = D / MT;      // output blocks for PV    (32)
constexpr int NV = KVT / 16;    // 16-wide float vectors per score row

static_assert(QLEN % MR == 0, "Q_LEN must be a multiple of 4");
static_assert(KVT % 16 == 0, "KV_TILE must be a multiple of 16 (vector width)");
static_assert(QLEN % 16 == 0, "Q_LEN must be a multiple of 16 (vector width)");

using MMUL = aie::mmul<MR, MS, MT, bfloat16, bfloat16, accauto>;

// scale = 1/sqrt(128)
constexpr float SCALE = 0.08838834764831845f;
constexpr float NEG_BIG = -3.0e30f;
// Argument floor for the exp poly. exp(-80) ~ 1.8e-35: below any weight that
// can matter, and it keeps the (int32_t)y cast in-range when the running max
// is still the -3e30 sentinel on the first tile.
constexpr float EXP_FLOOR = -80.0f;

// ── persistent per-head softmax state (core .bss, NOT stack) ──
alignas(32) float g_oacc[QLEN * D];   // C-layout, unnormalised O accumulator
alignas(32) float g_srow[QLEN * KVT]; // scores, plain row-major q x kv
alignas(32) bfloat16 g_ptile[QLEN * KVT];  // probabilities, A-layout for P·V
alignas(32) float g_m[QLEN];          // running max
alignas(32) float g_l[QLEN];          // running sum
alignas(32) float g_alpha[QLEN];      // this tile's rescale factor
alignas(32) float g_delta[QLEN];      // m_old - m_new, input to the rescale exp
alignas(32) float g_sum[QLEN];        // this tile's probability sum
int32_t g_tile = 0;

// exp(x) for x <= 0, computed INLINE. Same form (and the same two toolchain
// traps) as dflash_attention_sc_bf16.cc: an out-of-line helper miscompiled the
// surrounding scalar reduction, and a degree-2 poly is ~9% off near fy=-1.
__attribute__((always_inline)) inline float exp_neg(float x) {
  const float log2e = 1.442695040888963f;
  const float ln2 = 0.6931471805599453f;
  if (x < EXP_FLOOR) x = EXP_FLOOR;
  const float y = x * log2e;
  int32_t iy = (int32_t)y;         // truncate toward zero (y <= 0)
  const float fy = y - (float)iy;  // fractional part in (-1, 0]
  if (iy < -127) return 0.0f;
  iy = (iy + 127) << 23;           // pack into the IEEE-754 exponent
  float pow2_iy;
  __builtin_memcpy(&pow2_iy, &iy, sizeof(float));
  const float w = fy * ln2;        // 2^fy = exp(w), w in (-ln2, 0]
  const float pow2_fy =
      1.0f + w * (1.0f + w * (0.5f + w * (0.1666666666666667f +
      w * (0.0416666666666667f + w * (0.0083333333333333f +
      w * 0.0013888888888889f)))));
  return pow2_iy * pow2_fy;
}

// ── vectorised exp for x <= 0 ────────────────────────────────────────────────
// MEASURED: on AIE2 the SCALAR float path, not the dot product, is what pins
// this kernel. With both GEMMs already on aie::mmul the kernel still cost
// ~1900 core cycles per (q,k) pair — indistinguishable from the pre-mmul
// kernel's ~2400 — because every pair still ran one SCALAR exp. AIE2's scalar
// unit has no fast float datapath; the vector unit does. So the exp has to be
// vectorised for the mmul work to be visible at all.
//
// exp(x) = 2^(x*log2e) with round-to-nearest split via the classic magic
// constant: adding 1.5*2^23 to z parks round(z) in the mantissa, so the integer
// part falls out of an int32 bitcast and (float)n falls out of subtracting the
// magic back. Requires round-to-nearest-even, which set_rounding() establishes.
// f then lands in [-0.5, 0.5] where a degree-5 exp series on w = f*ln2 is exact
// to well under bf16 resolution.
constexpr int VL = 16;
using vf = aie::vector<float, VL>;
using vi = aie::vector<int32_t, VL>;
constexpr float MAGIC_F = 12582912.0f;      // 1.5 * 2^23
constexpr int32_t MAGIC_I = 0x4B400000;     // its bit pattern

__attribute__((always_inline)) inline vf exp_neg_v(const vf &x) {
  const float log2e = 1.442695040888963f;
  const float ln2 = 0.6931471805599453f;
  const vf magic = aie::broadcast<float, VL>(MAGIC_F);
  // clamp so the exponent pack stays in normal range (2^-126 underflows to 0
  // through the poly anyway, and no attention weight that small can matter)
  const vf z = aie::max(aie::mul(x, log2e).to_vector<float>(),
                        aie::broadcast<float, VL>(-126.0f));
  const vf t = aie::add(z, magic);
  const vi n = aie::sub(t.template cast_to<int32_t>(),
                        aie::broadcast<int32_t, VL>(MAGIC_I));   // round(z)
  const vf f = aie::sub(z, aie::sub(t, magic));                  // z - (float)n
  const vf p2n = aie::upshift(aie::add(n, aie::broadcast<int32_t, VL>(127)), 23)
                     .template cast_to<float>();                 // 2^n
  const vf w = aie::mul(f, ln2).to_vector<float>();
  vf poly = aie::broadcast<float, VL>(0.0083333333333333f);
  poly = aie::add(aie::mul(poly, w).to_vector<float>(), 0.0416666666666667f);
  poly = aie::add(aie::mul(poly, w).to_vector<float>(), 0.1666666666666667f);
  poly = aie::add(aie::mul(poly, w).to_vector<float>(), 0.5f);
  poly = aie::add(aie::mul(poly, w).to_vector<float>(), 1.0f);
  poly = aie::add(aie::mul(poly, w).to_vector<float>(), 1.0f);
  return aie::mul(p2n, poly).to_vector<float>();
}

}  // namespace

extern "C" {

// One KV tile. Q is re-read every call (it stays resident in its ObjectFifo
// buffer for the whole head); KV is a fresh tile each call.
void dflash_flash_step(bfloat16 *restrict Q, bfloat16 *restrict KV,
                       bfloat16 *restrict O) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *restrict Kt = KV;              // D x KVT, B-layout
  const bfloat16 *restrict Vt = KV + D * KVT;    // KVT x D, B-layout
  // Additive score mask, float32 aliased over the trailing bf16 slots.
  const float *restrict Mk =
      reinterpret_cast<const float *>(KV + 2 * D * KVT);

  if (g_tile == 0) {
    const aie::vector<float, 16> z = aie::zeros<float, 16>();
#pragma clang loop unroll(disable)
    for (int i = 0; i < QLEN * D; i += 16) aie::store_v(g_oacc + i, z);
    for (int q = 0; q < QLEN; ++q) {
      g_m[q] = NEG_BIG;
      g_l[q] = 0.0f;
    }
  }

  // ── 1. S = Q · Kᵀ  (mmul-tiled), scattered into row-major g_srow ──
#pragma clang loop unroll(disable)
  for (int qb = 0; qb < QB; ++qb) {
#pragma clang loop unroll(disable)
    for (int kb = 0; kb < KB; ++kb) {
      MMUL C;  // default-constructed => zero-start accumulation
      const bfloat16 *pA = Q + (qb * DB) * SZ_A;
      for (int db = 0; db < DB; ++db) {
        C.mac(aie::load_v<SZ_A>(pA + db * SZ_A),
              aie::load_v<SZ_B>(Kt + (db * KB + kb) * SZ_B));
      }
      const aie::vector<float, SZ_C> cv = C.template to_vector<float>();
#pragma clang loop unroll(disable)
      for (int qi = 0; qi < MR; ++qi)
        aie::store_v(g_srow + (qb * MR + qi) * KVT + kb * MT,
                     cv.template extract<MT>(qi));
    }
  }

  // ── 2. online softmax over this tile ──
  // Two passes over the q rows so the only per-row scalar float work left is
  // the running-max compare; the exps for BOTH the rescale factors (one per
  // row) and the probabilities (KVT per row) go through exp_neg_v.
#pragma clang loop unroll(disable)
  for (int q = 0; q < QLEN; ++q) {
    float *restrict sr = g_srow + q * KVT;
    float m_new = g_m[q];
#pragma clang loop unroll(disable)
    for (int nv = 0; nv < NV; ++nv) {
      // scale, then apply the additive tail mask. Masked lanes land at
      // ~MASK_NEG, which exp_neg_v's -126 exponent clamp turns into 2^-126:
      // the same "structurally zero" mechanism the NEG_BIG running-max
      // sentinel already relies on, so no new numerical path is introduced.
      const vf sv = aie::add(aie::mul(aie::load_v<VL>(sr + nv * VL), SCALE)
                                 .template to_vector<float>(),
                             aie::load_v<VL>(Mk + nv * VL));
      aie::store_v(sr + nv * VL, sv);
      const float r = aie::reduce_max(sv);
      if (r > m_new) m_new = r;
    }
    g_delta[q] = g_m[q] - m_new;   // <= 0
    g_m[q] = m_new;
  }
  // rescale factors: QLEN/VL vector exps instead of QLEN scalar ones
#pragma clang loop unroll(disable)
  for (int nq = 0; nq < QLEN / VL; ++nq)
    aie::store_v(g_alpha + nq * VL, exp_neg_v(aie::load_v<VL>(g_delta + nq * VL)));

#pragma clang loop unroll(disable)
  for (int q = 0; q < QLEN; ++q) {
    float *restrict sr = g_srow + q * KVT;
    const vf msub = aie::broadcast<float, VL>(-g_m[q]);
    float sum = 0.0f;
#pragma clang loop unroll(disable)
    for (int nv = 0; nv < NV; ++nv) {
      const vf pv = exp_neg_v(aie::add(aie::load_v<VL>(sr + nv * VL), msub));
      aie::store_v(sr + nv * VL, pv);
      sum += aie::reduce_add(pv);
    }
    g_sum[q] = sum;
    // scatter into the mmul A-layout for P·V: row q of tile (qb, vb) at
    // (qb*VB + vb)*SZ_A + qi*MS.
    const int qb = q / MR, qi = q % MR;
#pragma clang loop unroll(disable)
    for (int vb = 0; vb < VB; ++vb) {
      const aie::vector<float, MS> p8 = aie::load_v<MS>(sr + vb * MS);
      aie::store_v(g_ptile + (qb * VB + vb) * SZ_A + qi * MS,
                   aie::mul(p8, 1.0f).template to_vector<bfloat16>());
    }
  }
  // l = l*alpha + sum, vectorised
#pragma clang loop unroll(disable)
  for (int nq = 0; nq < QLEN / VL; ++nq) {
    const vf l = aie::mul(aie::load_v<VL>(g_l + nq * VL),
                          aie::load_v<VL>(g_alpha + nq * VL))
                     .template to_vector<float>();
    aie::store_v(g_l + nq * VL, aie::add(l, aie::load_v<VL>(g_sum + nq * VL)));
  }

  // ── 3. rescale the O accumulator, then O += P · V ──
#pragma clang loop unroll(disable)
  for (int qb = 0; qb < QB; ++qb) {
    alignas(32) float av[SZ_C];
    for (int qi = 0; qi < MR; ++qi)
      for (int t = 0; t < MT; ++t) av[qi * MT + t] = g_alpha[qb * MR + qi];
    const aie::vector<float, SZ_C> avv = aie::load_v<SZ_C>(av);
#pragma clang loop unroll(disable)
    for (int ob = 0; ob < OB; ++ob) {
      float *restrict pc = g_oacc + (qb * OB + ob) * SZ_C;
      MMUL C(aie::mul(aie::load_v<SZ_C>(pc), avv).template to_vector<float>());
      for (int vb = 0; vb < VB; ++vb) {
        C.mac(aie::load_v<SZ_A>(g_ptile + (qb * VB + vb) * SZ_A),
              aie::load_v<SZ_B>(Vt + (vb * OB + ob) * SZ_B));
      }
      aie::store_v(pc, C.template to_vector<float>());
    }
  }

  // ── 4. last tile: normalise by the running sum, emit O, reset for the next head ──
  if (g_tile == NTILES - 1) {
#pragma clang loop unroll(disable)
    for (int qb = 0; qb < QB; ++qb) {
      alignas(32) float iv[SZ_C];
      for (int qi = 0; qi < MR; ++qi) {
        const float r = 1.0f / (g_l[qb * MR + qi] + 1e-7f);
        for (int t = 0; t < MT; ++t) iv[qi * MT + t] = r;
      }
      const aie::vector<float, SZ_C> ivv = aie::load_v<SZ_C>(iv);
#pragma clang loop unroll(disable)
      for (int ob = 0; ob < OB; ++ob) {
        const int off = (qb * OB + ob) * SZ_C;
        aie::store_v(O + off,
                     aie::mul(aie::load_v<SZ_C>(g_oacc + off), ivv)
                         .template to_vector<bfloat16>());
      }
    }
    g_tile = 0;
  } else {
    g_tile += 1;
  }
}

}  // extern "C"
