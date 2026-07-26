// SPDX-License-Identifier: Apache-2.0
// hipfire — head_dim=128 variant of flash_tier_merge (see that file for the full
// design). Identical online-softmax tier combine; only CHD/CPL change (128 →
// 4 elements per 32-lane wave). Selected by the runtime when head_dim==128.

#include <hip/hip_runtime.h>

#define CHD 128   // head_dim
#define CPL 4     // CHD / 32 elements per lane

__launch_bounds__(32)
extern "C" __global__ void flash_tier_merge_128(
    const float* __restrict__ OA, const float* __restrict__ MA, const float* __restrict__ LA, // hot
    const float* __restrict__ OB, const float* __restrict__ MB, const float* __restrict__ LB, // cold
    float* __restrict__ OUT,   // [n_heads * 128] merged, normalized
    float* __restrict__ M_OUT, // [n_heads] combined max (or null)
    float* __restrict__ L_OUT, // [n_heads] combined denom (or null)
    int n_heads
) {
    const int hq = blockIdx.x;
    if (hq >= n_heads) return;
    const int lane = threadIdx.x;

    const float ma = MA[hq], mb = MB[hq];
    const float la = LA[hq], lb = LB[hq];
    const float m  = fmaxf(ma, mb);
    const float wa = la * __expf(ma - m);
    const float wb = lb * __expf(mb - m);
    const float wsum = wa + wb;
    const float inv  = (wsum > 0.0f) ? (1.0f / wsum) : 0.0f;

    const float* oa = OA + (long long)hq * CHD + lane * CPL;
    const float* ob = OB + (long long)hq * CHD + lane * CPL;
    float* o = OUT + (long long)hq * CHD + lane * CPL;
    #pragma unroll
    for (int i = 0; i < CPL; ++i) o[i] = (wa * oa[i] + wb * ob[i]) * inv;

    if (lane == 0) {
        if (M_OUT) M_OUT[hq] = m;
        if (L_OUT) L_OUT[hq] = wsum;
    }
}
