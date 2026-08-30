// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Opus Quant W4A8 DECODE GEMV (batch=1): y[m] = Σ_g sw[m,g]·Σ_{k∈g} qw·x[k].
// One wave32 per output row, ZERO LDS, single wave-shuffle reduction.
//
// DECODE numerics note: at B=1 there is no WMMA, so we do NOT quantize the
// activation to int8 (that exists only to feed the iu8 WMMA tiles in the
// batched prefill path). Instead we dequantize the int8 weight inline and
// multiply by the FULL-PRECISION f32 activation — exactly like the mq4
// decode GEMV (fused_qkvza_hfq4g256). This (a) eliminates the separate
// quantize_act_oq8 launch per projection (the dominant decode launch cost),
// and (b) is numerically MORE accurate than W8A8 (acts stay f32). The AWQ
// math (W·s)·(x/s)=W·x still holds: the awq divide is folded into x_rot by
// the fused rmsnorm, the s is baked into the int8 weight at quantize time.
// group is 256 or 128: per_lane = group/32 int8 per lane, so 8 (two aligned
// int32 loads) or 4 (one). The G256 arm is kept byte-for-byte as it was, and the
// two arms are selected by a WAVE-UNIFORM branch (`group` is a scalar kernel
// argument), so neither pays for the other and G256 numerics are unchanged —
// the accumulation ORDER differs between the arms, so sharing one loop would
// have silently re-rounded every existing G256 baseline.
//
//   W  : [M, K] signed int8 weights              row stride = K
//   Ws : [M, K/group] f32 per-group weight scales
//   X  : [K] f32 activation (batch=1, rotated)
//   Y  : [M] f32

// Quant-basis contract: input activation `x` must already be rotated into the
// weights' quantization basis by the dispatch caller (rotate_x_*); a raw `x`
// runs but yields garbage. See kernels/AGENTS.md.

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void gemv_oq8_grouped(
    const int8_t*  __restrict__ W,
    const float*   __restrict__ Ws,
    const float*   __restrict__ X,
    float*         __restrict__ Y,
    int M, int K, int group
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x; // 0..31
    const int ng = K / group;
    const int8_t* w_row = W + (long long)row * K; // K int8 bytes per row
    const float* ws_row = Ws + (long long)row * ng;

    float facc = 0.0f;
    if ((group >> 5) == 8) {
        // G256: 8 int8 per lane, two aligned int32 loads. UNCHANGED.
        for (int g = 0; g < ng; ++g) {
            const int boff = g * group + lane * 8; // 8 int8 = two int32 per lane
            const int32_t w0 = *(const int32_t*)(w_row + boff);
            const int32_t w1 = *(const int32_t*)(w_row + boff + 4);
            float gsum = 0.0f;
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                int qw0 = (int)((w0 >> (8 * n)) & 0xff); qw0 = (qw0 << 24) >> 24; // sext int8
                int qw1 = (int)((w1 >> (8 * n)) & 0xff); qw1 = (qw1 << 24) >> 24;
                gsum += (float)qw0 * X[boff + n];
                gsum += (float)qw1 * X[boff + 4 + n];
            }
            facc += gsum * ws_row[g];
        }
    } else {
        // G128: 4 int8 per lane, one aligned int32 load. `lane * 4` keeps the
        // load 4-byte aligned for any group that is a multiple of 32.
        for (int g = 0; g < ng; ++g) {
            const int boff = g * group + lane * 4;
            const int32_t w0 = *(const int32_t*)(w_row + boff);
            float gsum = 0.0f;
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                int qw0 = (int)((w0 >> (8 * n)) & 0xff); qw0 = (qw0 << 24) >> 24; // sext int8
                gsum += (float)qw0 * X[boff + n];
            }
            facc += gsum * ws_row[g];
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        facc += __shfl_down(facc, off);
    }
    if (lane == 0) {
        Y[row] = facc;
    }
}
