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

extern "C" void k1_vmac_chain(const TA *__restrict pA, const TB *__restrict pB, int32 *__restrict pOut) {
  // Two A tiles x two B tiles give four independent (a_i, b_j) chains, all
  // resident in registers across the loop.
  aie::vector<TA, MMUL::size_A> a0 = aie::load_v<MMUL::size_A>(pA);
  aie::vector<TA, MMUL::size_A> a1 = aie::load_v<MMUL::size_A>(pA + MMUL::size_A);
  aie::vector<TB, MMUL::size_B> b0 = aie::load_v<MMUL::size_B>(pB);
  aie::vector<TB, MMUL::size_B> b1 = aie::load_v<MMUL::size_B>(pB + MMUL::size_B);

  MMUL c0, c1, c2, c3, c4, c5, c6, c7;
  c0.mul(a0, b0);
  c1.mul(a0, b1);
  c2.mul(a1, b0);
  c3.mul(a1, b1);
#if CHAINS > 4
  c4.mul(a0, b0);
  c5.mul(a0, b1);
  c6.mul(a1, b0);
  c7.mul(a1, b1);
#endif

  // ITERS * CHAINS VMACs. Runtime is linear in ITERS; the host fits the slope.
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < ITERS; i++) {
    c0.mac(a0, b0);
#if CHAINS > 1
    c1.mac(a0, b1);
#endif
#if CHAINS > 2
    c2.mac(a1, b0);
#endif
#if CHAINS > 3
    c3.mac(a1, b1);
#endif
#if CHAINS > 4
    c4.mac(a0, b0);
    c5.mac(a0, b1);
    c6.mac(a1, b0);
    c7.mac(a1, b1);
#endif
  }

  // DCE guard: every chain must reach a store or the loop vanishes.
  auto s = aie::add(aie::add(c0.template to_vector<int32>(), c1.template to_vector<int32>()),
                    aie::add(c2.template to_vector<int32>(), c3.template to_vector<int32>()));
#if CHAINS > 4
  s = aie::add(s, aie::add(aie::add(c4.template to_vector<int32>(), c5.template to_vector<int32>()),
                           aie::add(c6.template to_vector<int32>(), c7.template to_vector<int32>())));
#endif
  aie::store_v(pOut + 8, s);
  pOut[0] = ITERS;
  pOut[1] = CHAINS;
  pOut[2] = ITERS * CHAINS;  // total VMACs
  pOut[3] = MR * MK * MN;    // MACs per VMAC
}
