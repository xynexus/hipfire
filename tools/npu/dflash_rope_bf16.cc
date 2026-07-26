// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// AIE2/AIE2P BF16 RoPE for the DFlash drafter — FULL neox rotation.
//
// The DFlash draft model is Qwen3 (NOT Qwen3.5): it rotates the ENTIRE head_dim
// (partial_rotary_factor = 1.0), half-split/neox convention, matching the
// runtime `rope_batched.hip` (which the F16 GPU golden was validated against).
// This differs from `rope_rotate_bf16.cc`, which is hard-coded to Qwen3.5's
// n_rot = head_dim/4 partial rotary — hence a separate kernel.
//
//   n_rot  = head_dim            (full rotation, no passthrough)
//   n_rot2 = head_dim / 2        (number of (x, y) pairs = frequency count)
//
// cos/sin buffer (half-split): cs[0..n_rot2) = cos, cs[n_rot2..head_dim) = sin,
// where cos/sin are of (pos * inv_freq[i]), inv_freq[i] = theta^(-2i/head_dim).
// The host builds `cs` with the sidecar's rope_theta (1e7 for the z-lab 9B
// drafter); the kernel is theta-agnostic (applies whatever cs it is given).
//
// Rotation math (per pair i in [0, n_rot2)):
//   x = input[i], y = input[i + n_rot2]
//   output[i]          = x*cos[i] - y*sin[i]
//   output[i + n_rot2] = y*cos[i] + x*sin[i]
//
// IRON dispatch: _transform_gen single-core, tile_size = head_dim,
// N_div_n = n_heads (Q) or n_kv_heads (K). cs is a tensor param acquired once.

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

template <typename T, unsigned N>
static void dflash_rope_impl(const T *restrict input, T *restrict output,
                             const T *restrict cs, int32_t head_dim) {
    const int32_t n_rot2 = head_dim / 2;  // full rotation → pairs = head_dim/2

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int32_t i = 0; i < n_rot2; i += N) {
        ::aie::vector<T, N> xv = ::aie::load_v<N>(input + i);
        ::aie::vector<T, N> yv = ::aie::load_v<N>(input + i + n_rot2);
        ::aie::vector<T, N> cv = ::aie::load_v<N>(cs + i);
        ::aie::vector<T, N> sv = ::aie::load_v<N>(cs + i + n_rot2);

        ::aie::vector<T, N> xcos = ::aie::mul(xv, cv);
        ::aie::vector<T, N> ysin = ::aie::mul(yv, sv);
        ::aie::vector<T, N> xrot = ::aie::sub(xcos, ysin);

        ::aie::vector<T, N> ycos = ::aie::mul(yv, cv);
        ::aie::vector<T, N> xsin = ::aie::mul(xv, sv);
        ::aie::vector<T, N> yrot = ::aie::add(ycos, xsin);

        ::aie::store_v(output + i, xrot);
        ::aie::store_v(output + i + n_rot2, yrot);
    }
}

extern "C" {

void dflash_rope_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                      bfloat16 *restrict cs, int32_t head_dim) {
    dflash_rope_impl<bfloat16, 16>(input, output, cs, head_dim);
}

} // extern "C"
