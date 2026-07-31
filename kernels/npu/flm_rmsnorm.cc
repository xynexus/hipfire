// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// RMSNorm for the reproduced decoder layer, over the existing
// `tools/npu/rms_norm_weighted_bf16.cc`, whose math already matches llama:
// out = x * rsqrt(mean(x^2) + eps) * weight, with eps = 1e-5, which is
// llama-3.2-1B's `rms_norm_eps` exactly.
//
// The wrapper binds the column count at compile time. That kernel takes it as a
// runtime `const int32_t`, and passing a scalar through IRON's
// `ExternalFunction` arg_types makes it allocate a *buffer* for the scalar; the
// design then fails to fit tile memory. Same trap as the SwiGLU wrapper.
//
// TWO precision notes about the underlying kernel, both measured:
//
// 1. It never calls `aie::set_rounding`, so every bf16 conversion in it
//    TRUNCATES. Confirmed exactly: a truncating emulation reproduces the
//    device's error to the last digit (1.2320e-02 both), where a
//    round-to-nearest emulation gives 8.81e-03 — so truncation costs ~40% more
//    error for nothing. This wrapper sets the mode before calling; the mode is
//    core-global, so setting it here fixes the shared kernel's behaviour on our
//    path without editing a kernel other paths depend on.
// 2. It broadcasts `inv_rms` in bf16, so the normalisation scale carries a
//    ~0.28% error applied UNIFORMLY to the whole vector — a systematic gain
//    error on every activation entering the projections, not a random one.
//    That one is inherent to the kernel and is left alone here.

#include "../../tools/npu/rms_norm_weighted_bf16.cc"

#ifndef DIM_COLS
#define DIM_COLS 2048
#endif

extern "C" __attribute__((noinline)) void
flm_rmsnorm(bfloat16 *restrict input, bfloat16 *restrict weight,
            bfloat16 *restrict output) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  rms_norm_weighted_bf16(input, weight, output, DIM_COLS);
}
