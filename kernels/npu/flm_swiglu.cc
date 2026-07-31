// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.
//
// SwiGLU for the reproduced decoder layer: out[i] = silu(gate[i]) * up[i].
//
// This is a thin wrapper over the existing `tools/npu/silu_mul_bf16.cc`, which
// is already AIE2P-aware (hardware `aie::tanh`, no LUT). The wrapper exists for
// one reason: that kernel takes the element count as a runtime `const int32_t`,
// and passing a scalar through IRON's `ExternalFunction` arg_types makes it
// allocate a *buffer* for the scalar — the design then fails to fit tile memory
// with `Basic sequential allocation also failed`. Fixing the count at compile
// time via -DDIM_TILE avoids the scalar argument entirely.

#include "../../tools/npu/silu_mul_bf16.cc"

#ifndef DIM_TILE
#define DIM_TILE 1024
#endif

extern "C" __attribute__((noinline)) void
flm_swiglu(bfloat16 *restrict gate, bfloat16 *restrict up,
           bfloat16 *restrict out) {
  silu_mul_bf16(gate, up, out, DIM_TILE);
}
