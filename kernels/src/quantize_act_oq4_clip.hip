// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
// Clip-search variant of quantize_act_oq4 (Stream B, Lever A "+"/AWQ on the
// ACTIVATION side). Same contract and output format as quantize_act_oq4 — the
// existing gemm_oq4_grouped_wmma[_lds] consumes it unchanged — but instead of
// scale = absmax/7 (which a single in-group outlier inflates, coarsening the
// bulk), it grid-searches a per-group clip ratio a in (0,1] and picks the scale
// = a*absmax/7 that MINIMIZES the group's int4 reconstruction MSE. Clipping the
// rare large values buys finer resolution on the dense bulk. This is the
// activation analog of the codec's weight-side clip pass; it is dynamic
// per-token/per-group and adds no bytes (Xs still one f32/group). gfx1103 wave32,
// ZERO LDS (wave-shuffle reductions).
//
//   X  : [B, K] f32 (already FWHT-rotated), Xq : [B,K/2] s4, Xs : [B,K/group] f32
//   grid = [K/group, B], block = [32]. group % 32 == 0, group/32 even.

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void quantize_act_oq4_clip(
    const float*   __restrict__ X,
    uint8_t*       __restrict__ Xq,
    float*         __restrict__ Xs,
    int B, int K, int group
) {
    const int b = blockIdx.y;
    const int g = blockIdx.x;
    if (b >= B) return;
    const int lane = threadIdx.x;
    const int per_lane = group / 32;
    const int n_groups = K / group;
    const long long g0 = (long long)g * group;
    const float* row = X + (long long)b * K + g0;

    // Cache this lane's group elements in registers so the clip-ratio search
    // reuses them (no 9× DRAM re-read → search is nearly free vs plain absmax).
    // per_lane = group/32 ≤ 16 for group ≤ 512 (oq4 uses 256 → per_lane 8).
    float vals[16];
    for (int i = 0; i < per_lane; ++i) {
        vals[i] = row[lane * per_lane + i];
    }

    // Group absmax (wave-cooperative).
    float amax = 0.0f;
    for (int i = 0; i < per_lane; ++i) {
        amax = fmaxf(amax, fabsf(vals[i]));
    }
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_down(amax, off));
    }
    amax = __shfl(amax, 0);

    if (amax <= 0.0f) {
        if (lane == 0) Xs[(long long)b * n_groups + g] = 1.0f;
        uint8_t* out0 = Xq + (long long)b * (K / 2) + g0 / 2;
        for (int i = 0; i < per_lane; i += 2) {
            out0[(lane * per_lane + i) / 2] = 0;
        }
        return;
    }

    // Grid-search clip ratio a: scale = a*amax/7 ; pick min group MSE. a=1.0 is
    // the plain absmax baseline, so the search can never do worse than it.
    const float ALPHAS[9] = {1.0f, 0.92f, 0.85f, 0.78f, 0.71f, 0.64f, 0.57f, 0.50f, 0.43f};
    float best_scale = amax / 7.0f;
    float best_err = 3.4e38f;
    #pragma unroll
    for (int a = 0; a < 9; ++a) {
        const float scale = ALPHAS[a] * amax / 7.0f;
        const float inv = 1.0f / scale;
        float err = 0.0f;
        for (int i = 0; i < per_lane; ++i) {
            const float v = vals[i];
            int q = (int)lroundf(v * inv);
            q = max(-7, min(7, q));
            const float d = v - (float)q * scale;
            err += d * d;
        }
        // wave-reduce the squared error across the group
        for (int off = 16; off > 0; off >>= 1) {
            err += __shfl_down(err, off);
        }
        err = __shfl(err, 0);
        if (err < best_err) {
            best_err = err;
            best_scale = scale;
        }
    }

    const float scale = best_scale;
    const float inv = 1.0f / scale;
    if (lane == 0) {
        Xs[(long long)b * n_groups + g] = scale;
    }
    uint8_t* out = Xq + (long long)b * (K / 2) + g0 / 2;
    for (int i = 0; i < per_lane; i += 2) {
        const int idx = lane * per_lane + i;
        int qlo = (int)lroundf(vals[i] * inv);
        int qhi = (int)lroundf(vals[i + 1] * inv);
        qlo = max(-7, min(7, qlo));
        qhi = max(-7, min(7, qhi));
        out[idx / 2] = ((uint8_t)qlo & 0xf) | (((uint8_t)qhi & 0xf) << 4);
    }
}
