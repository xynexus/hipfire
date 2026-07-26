// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// AIE2 BF16 SwiGLU kernel: out[i] = silu(gate[i]) * up[i]
//
// Used in the Qwen3.5 dense FFN NPU path after the gate/up projections have
// already been computed on the GPU.  Adapted from the aie2/swiglu.cc kernel
// shipped with mlir_aie; simplified to remove the weight-matmul fuse.
//
// ── Inputs / output ──────────────────────────────────────────────────────────
//   gate   : [hidden_size] bf16   — gate projection output (tiled by IRON)
//   up     : [hidden_size] bf16   — up projection output   (tiled by IRON)
//   out    : [hidden_size] bf16   — silu(gate) * up
//   n      : int32               — tile_size (auto-appended by IRON)
//
// ── Computation ──────────────────────────────────────────────────────────────
//   sigmoid(g) = 1 / (1 + exp(-g))
//   silu(g)    = g * sigmoid(g)
//   out[i]     = silu(gate[i]) * up[i]
//
// ── IRON dispatch pattern ────────────────────────────────────────────────────
//   transform_parallel_binary — work split equally across all 4 NPU columns.
//   Default tile_size=16; hidden_size must be divisible by tile_size × num_cols
//   = 64 on NPU1 (4 compute columns).
//
// ── Shape constraints ────────────────────────────────────────────────────────
//   tile_size   % 16 == 0   (vector alignment)
//   hidden_size % 64 == 0   (tile_size × 4 columns; NPU1 has 4 compute cols)
//
// ── Built xclbins (target/npu/) ──────────────────────────────────────────────
//   qwen35-swiglu-3584.xclbin   — 0.8B FFN intermediate (hidden=1024)
//   qwen35-swiglu-6144.xclbin   — 2B   FFN intermediate (hidden=2048)
//   qwen35-swiglu-8960.xclbin   — 1.5B FFN intermediate (hidden=1536)
//   qwen35-swiglu-9216.xclbin   — 4B   FFN intermediate (hidden=2560)
//   qwen35-swiglu-12288.xclbin  — (reserved)
//   qwen35-swiglu-18944.xclbin  — 9B   FFN intermediate (hidden=4096)
//   Naming: qwen35-swiglu-{hidden_size}.xclbin + -instr.bin
//
// ── Integration status ───────────────────────────────────────────────────────
//   Wired: HIPFIRE_QWEN35_FFN_BF16=xdna1  (opt-in env var)
//   Perf note: net-slower than GPU at 0.8B — dispatch overhead (~180 µs × 24
//   layers = 4.3 ms) exceeds GPU SwiGLU cost (~0.3 ms total). Only beneficial
//   when GPU is fully saturated or model decode is >100 ms/tok.
//
// ── AIE2 vs AIE2P ───────────────────────────────────────────────────────────
//   AIE2  (__AIE_ARCH__==20, NPU1/Phoenix): getTanhBf16() LUT-based tanh
//   AIE2P (__AIE_ARCH__==21, NPU2/Strix+): scalar F32 exp/sigmoid with BF16
//          rounding after SiLU and after the final multiply. This matches the
//          PyTorch BF16 reference closely enough for full-encoder admission.

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
// AIE2 (NPU1, __AIE_ARCH__==20): getTanhBf16 is LUT-based.
// Include the .h (declares getTanhBf16 inline + extern table decls) then the
// .cpp (provides the actual table definitions) — the .cpp does NOT include the
// .h itself so both are required.
// AIE2P (NPU2, __AIE_ARCH__==21): hardware aie::tanh — no LUT needed.
#if __AIE_ARCH__ == 20
#  include "lut_based_ops.h"
#  include "lut_based_ops.cpp"
#endif
#include <stdint.h>

using namespace aie;

#if __AIE_ARCH__ == 21
__attribute__((noinline)) float exp2_f32(float value) {
  if (value <= -126.0f)
    return 0.0f;
  if (value >= 0.0f)
    return 1.0f;
  int exponent = (int)value;
  if ((float)exponent > value)
    --exponent;
  const float z = (value - (float)exponent) * 0.6931471805599453f;
  const float polynomial =
      1.0f +
      z * (1.0f +
           z * (0.5f +
                z * (0.1666666666666667f +
                     z * (0.0416666666666667f +
                          z * (0.0083333333333333f +
                               z * 0.0013888888888889f)))));
  float scale = 1.0f;
  for (int power = 0; power < -exponent; ++power)
    scale *= 0.5f;
  return scale * polynomial;
}
#endif

static void silu_mul_bf16_inner(bfloat16 *restrict gate, bfloat16 *restrict up,
                                bfloat16 *restrict output, const int32_t n) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  auto it_gate = aie::begin_restrict_vector<16>(gate);
  auto it_up   = aie::begin_restrict_vector<16>(up);
#if __AIE_ARCH__ == 20
  auto it_out = aie::begin_restrict_vector<16>(output);
  aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  aie::vector<bfloat16, 16> one  = aie::broadcast<bfloat16, 16>(1.0f);
#endif

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < n; i += 16) {
    aie::vector<bfloat16, 16> g = *it_gate++;
    aie::vector<bfloat16, 16> u = *it_up++;

    // sigmoid(g) = 0.5 * (1 + tanh(0.5 * g))
    // Use explicit vector<bfloat16,16> for all intermediates — aie::mul returns
    // accum<__accfloat,16> and using auto keeps it as accum, which breaks chained
    // mul/add calls (no vec×accum overload exists).
    //
    // AIE2 uses the runtime LUT. AIE2P uses the scalar stable sigmoid so its
    // BF16 rounding points match the framework oracle.
#if __AIE_ARCH__ == 20
    aie::vector<bfloat16, 16> half_g     = aie::mul(g, half);
    aie::vector<bfloat16, 16> tanh_hg    = getTanhBf16(half_g);
    aie::vector<bfloat16, 16> tanh_plus1 = aie::add(tanh_hg, one);
    aie::vector<bfloat16, 16> sig_g      = aie::mul(tanh_plus1, half);
    aie::vector<bfloat16, 16> silu_g = aie::mul(g, sig_g);
    auto result = aie::mul(silu_g, u);
    *it_out++ = result.to_vector<bfloat16>();
#else
    for (int lane = 0; lane < 16; ++lane) {
      const float gate_f32 = float(g[lane]);
      const float exponential =
          exp2_f32(-__builtin_fabsf(gate_f32) * 1.4426950408889634f);
      const float reciprocal = exponential == 0.0f
                                   ? 1.0f
                                   : aie::inv(aie::broadcast<float, 16>(
                                                  1.0f + exponential))[0];
      const float sigmoid = gate_f32 >= 0.0f ? reciprocal
                                              : exponential * reciprocal;
      const bfloat16 silu = bfloat16(gate_f32 * sigmoid);
      output[i + lane] = bfloat16(float(silu) * float(u[lane]));
    }
#endif
  }
}

extern "C" {

void silu_mul_bf16(bfloat16 *restrict gate, bfloat16 *restrict up,
                   bfloat16 *restrict output, const int32_t n) {
  silu_mul_bf16_inner(gate, up, output, n);
}

} // extern "C"
