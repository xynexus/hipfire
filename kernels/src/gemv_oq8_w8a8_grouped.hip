// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Opus Quant W8A8 DECODE GEMV (batch=1): int8 weight × int8 activation, per-group
// scales on BOTH sides. RDNA-native integer path — group dot in int32, then a
// single f32 rescale per group. Distinct from `gemv_oq8_grouped` (W8A16, f32
// activation dequant-inline): here the activation is pre-quantized to int8 (via a
// fused rmsnorm+rotate+quantize) so the whole block shares one int8 activation.
//
//   W  : [M, K] signed int8 weights                 row stride = K
//   Ws : [M, K/group] f32 per-group weight scales
//   Xq : [K] signed int8 activation (FWHT-rotated, per-group absmax quantized)
//   Xs : [K/group] f32 per-group activation scales
//   Y  : [M] f32 output
//   group MUST be 256 (group/32 = 8 int8 per lane = two aligned int32 loads).
//
//   y[m] = Σ_g Ws[m,g]·Xs[g]·( Σ_{k∈g} W_i8[m,k]·Xq[k] )
//
// One wave32 per output row, ZERO LDS, single wave-shuffle reduction.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_oq8_w8a8_grouped(
    const signed char* __restrict__ W,
    const float*       __restrict__ Ws,
    const signed char* __restrict__ Xq,
    const float*       __restrict__ Xs,
    float*             __restrict__ Y,
    int M, int K, int group
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x; // wave32
    const int ng = K / group;
    const signed char* w_row = W + (long long)row * K;
    const float* ws_row = Ws + (long long)row * ng;

    float facc = 0.0f;
    for (int g = 0; g < ng; g++) {
        const int boff = g * group + lane * 8; // 8 int8 / lane = two aligned int32 loads
        // int8·int8 → int32 lane-dot over 8 elements via V_DOT4_I32_IU8. gfx1151
        // (RDNA3.5) DOES have it, and its per-operand NEG modifiers select signed
        // inputs — so __builtin_amdgcn_sdot4 gives the signed×signed dot the earlier
        // comment thought impossible. Two 4-way dots + coalesced int32 loads replace
        // eight scalar muls + eight byte loads (leaner loop ⇒ higher load-issue rate).
        const int w0 = *(const int*)(w_row + boff);
        const int w1 = *(const int*)(w_row + boff + 4);
        const int x0 = *(const int*)(Xq + boff);
        const int x1 = *(const int*)(Xq + boff + 4);
        // sudot4 = V_DOT4_I32_IU8 with per-operand sign flags (both signed here).
        int idot = 0;
        idot = __builtin_amdgcn_sudot4(true, w0, true, x0, idot, false);
        idot = __builtin_amdgcn_sudot4(true, w1, true, x1, idot, false);
        // Group scale (weight × activation) applied after the int32 lane-dot.
        facc += (float)idot * (ws_row[g] * Xs[g]);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        facc += __shfl_down(facc, offset);
    if (lane == 0) Y[row] = facc;
}
