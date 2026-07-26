// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// K1: derive the AIE compute clock (f_H).
//
// npu1 exposes no clock: xrt-smi implements only {aie-partitions, all, host,
// platform} and none reports npu_clk_max (halo reports 1800 MHz via a report
// this device does not have). Without f_H, t_core's `cyc_mmul / f_H` is only
// ever fitted as a product, so the whole core term is unidentifiable.
//
// Method (adapted from benchmarks/npu_gemm_tuning/r0/r0b_throughput.cc, which
// established it on aie2p): CHAINS independent accumulator chains issue
// back-to-back VMACs from resident L1 tiles. Enough independent chains hide the
// accumulator latency, so a saturated pipe runs at II=1 — one VMAC per cycle.
// Then measured VMAC/s == f_H, with no separate cycle-count assumption.
//
// The II=1 premise is not assumed, it is tested: sweep CHAINS and confirm
// throughput plateaus (more chains stop helping => latency is hidden), and
// count VMAC bundles in the disassembly.
//
// AIE2P needs more chains than AIE2 to hide the accumulator latency, and the
// original 8x8x8 accumulator (64 int32) spills the register file at 8 chains
// before the pipe saturates — the sweep collapsed instead of plateauing. So
// CHAINS is parameterised up to 16 here, and the clock probe drives it with the
// smaller MR=4 shape (32-int32 accumulator) so the extra chains fit in
// registers. Shape changes MACs/VMAC, never the VMAC issue rate, so a smaller
// tile is still a valid clock probe.
//
// Everything is resident and hoisted, so no DMA or memory term contaminates the
// slope. The host sweeps ITERS and fits; only the slope is used, so fixed
// dispatch cost cancels.

#include "aie_kernels/aie_kernel_utils.h"  // AIE_PREPARE_FOR_PIPELINING et al
#include <aie_api/aie.hpp>

#ifndef ITERS
#define ITERS 100000
#endif
// AIE2 int8 native shape.
#ifndef MR
#define MR 4
#define MK 8
#define MN 8
#endif
#ifndef CHAINS
#define CHAINS 4
#endif
#if CHAINS > 16
#error "K1 supports at most 16 chains; add operand tiles before raising this"
#endif
// Operand types. C4 sweeps these to find which (dtype, shape) pairs are native:
// int8 x int8 tops out at 256 MACs/VMAC, but a narrower operand type may pack
// more MACs into the same native instruction (aie2p's R58 used int8 x int4).
#ifndef TA
#define TA int8
#endif
#ifndef TB
#define TB int8
#endif

using MMUL = aie::mmul<MR, MK, MN, TA, TB>;

// Four distinct (a_i, b_j) products cycle across the chains: chain i reads
// a1 when bit 1 of i is set and b1 when bit 0 is set. Chains beyond four reuse
// the same operand pairs into fresh accumulators, which is all the latency
// hiding needs — the accumulators are independent even when the inputs repeat.
#define A_OF(i) (((i) & 2) ? a1 : a0)
#define B_OF(i) (((i) & 1) ? b1 : b0)

extern "C" void k1_vmac_chain(const TA *__restrict pA, const TB *__restrict pB, int32 *__restrict pOut) {
  aie::vector<TA, MMUL::size_A> a0 = aie::load_v<MMUL::size_A>(pA);
  aie::vector<TA, MMUL::size_A> a1 = aie::load_v<MMUL::size_A>(pA + MMUL::size_A);
  aie::vector<TB, MMUL::size_B> b0 = aie::load_v<MMUL::size_B>(pB);
  aie::vector<TB, MMUL::size_B> b1 = aie::load_v<MMUL::size_B>(pB + MMUL::size_B);

  // Declare and seed exactly CHAINS independent accumulators.
#define DECL(i) MMUL c##i; c##i.mul(A_OF(i), B_OF(i));
  DECL(0)
#if CHAINS > 1
  DECL(1)
#endif
#if CHAINS > 2
  DECL(2)
#endif
#if CHAINS > 3
  DECL(3)
#endif
#if CHAINS > 4
  DECL(4)
#endif
#if CHAINS > 5
  DECL(5)
#endif
#if CHAINS > 6
  DECL(6)
#endif
#if CHAINS > 7
  DECL(7)
#endif
#if CHAINS > 8
  DECL(8)
#endif
#if CHAINS > 9
  DECL(9)
#endif
#if CHAINS > 10
  DECL(10)
#endif
#if CHAINS > 11
  DECL(11)
#endif
#if CHAINS > 12
  DECL(12)
#endif
#if CHAINS > 13
  DECL(13)
#endif
#if CHAINS > 14
  DECL(14)
#endif
#if CHAINS > 15
  DECL(15)
#endif
#undef DECL

  // ITERS * CHAINS VMACs. Runtime is linear in ITERS; the host fits the slope.
#define STEP(i) c##i.mac(A_OF(i), B_OF(i));
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < ITERS; i++) {
    STEP(0)
#if CHAINS > 1
    STEP(1)
#endif
#if CHAINS > 2
    STEP(2)
#endif
#if CHAINS > 3
    STEP(3)
#endif
#if CHAINS > 4
    STEP(4)
#endif
#if CHAINS > 5
    STEP(5)
#endif
#if CHAINS > 6
    STEP(6)
#endif
#if CHAINS > 7
    STEP(7)
#endif
#if CHAINS > 8
    STEP(8)
#endif
#if CHAINS > 9
    STEP(9)
#endif
#if CHAINS > 10
    STEP(10)
#endif
#if CHAINS > 11
    STEP(11)
#endif
#if CHAINS > 12
    STEP(12)
#endif
#if CHAINS > 13
    STEP(13)
#endif
#if CHAINS > 14
    STEP(14)
#endif
#if CHAINS > 15
    STEP(15)
#endif
  }
#undef STEP

  // DCE guard: every chain must reach a store or the loop vanishes.
#define RED(i) s = aie::add(s, c##i.template to_vector<int32>());
  auto s = c0.template to_vector<int32>();
#if CHAINS > 1
  RED(1)
#endif
#if CHAINS > 2
  RED(2)
#endif
#if CHAINS > 3
  RED(3)
#endif
#if CHAINS > 4
  RED(4)
#endif
#if CHAINS > 5
  RED(5)
#endif
#if CHAINS > 6
  RED(6)
#endif
#if CHAINS > 7
  RED(7)
#endif
#if CHAINS > 8
  RED(8)
#endif
#if CHAINS > 9
  RED(9)
#endif
#if CHAINS > 10
  RED(10)
#endif
#if CHAINS > 11
  RED(11)
#endif
#if CHAINS > 12
  RED(12)
#endif
#if CHAINS > 13
  RED(13)
#endif
#if CHAINS > 14
  RED(14)
#endif
#if CHAINS > 15
  RED(15)
#endif
#undef RED
  aie::store_v(pOut + 8, s);

  pOut[0] = ITERS;
  pOut[1] = CHAINS;
  pOut[2] = ITERS * CHAINS;  // total VMACs
  pOut[3] = MR * MK * MN;    // MACs per VMAC
  // Halo's current XRTHostRuntime path may overwrite output word zero while
  // collecting command timing. Keep a complete second guard block so K1 can
  // validate the image independently of that runtime-owned word.
  pOut[4] = ITERS;
  pOut[5] = CHAINS;
  pOut[6] = ITERS * CHAINS;
  pOut[7] = MR * MK * MN;
}
