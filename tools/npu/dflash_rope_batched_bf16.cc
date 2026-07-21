// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// AIE2/AIE2P BF16 FULL-neox RoPE for the DFlash drafter — BATCHED variant.
//
// Identical rotation math to dflash_rope_bf16.cc, but the cos/sin buffer is a
// SECOND TILED INPUT rather than a once-acquired tensor param. This lets one
// dispatch rotate a whole [rows*n_heads] batch where each head-tile carries its
// OWN cs (position-dependent), which the param-based kernel cannot do (a param
// is acquired once and shared by every tile in the dispatch, forcing one
// position — hence one row — per dispatch).
//
// _transform_gen dispatch: inputs = [qk_buf, cs_buf], both [rows*n_heads*head_dim],
// tile_size = head_dim, N_div_n = rows*n_heads. The framework calls
//   dflash_rope_batched_bf16(qk_tile, cs_tile, out_tile, head_dim)
// per tile (arg order: *inputs, output, n). The host packs cs_buf so each of the
// n_heads tiles within a row repeats that row's cs (all heads at a row share the
// token position); across rows the cs differs.
//
// cs layout (half-split, full rotation n_rot=head_dim):
//   cs[0..n_rot2)      = cos(pos * inv_freq[i])
//   cs[n_rot2..head_dim) = sin(pos * inv_freq[i])
// Rotation (per pair i in [0, n_rot2)):
//   x=in[i], y=in[i+n_rot2]
//   out[i]        = x*cos[i] - y*sin[i]
//   out[i+n_rot2] = y*cos[i] + x*sin[i]

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

template <typename T, unsigned N>
static void dflash_rope_batched_impl(const T *restrict input,
                                     const T *restrict cs, T *restrict output,
                                     int32_t head_dim) {
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

void dflash_rope_batched_bf16(bfloat16 *restrict input, bfloat16 *restrict cs,
                              bfloat16 *restrict output, int32_t head_dim) {
    dflash_rope_batched_impl<bfloat16, 16>(input, cs, output, head_dim);
}

} // extern "C"
