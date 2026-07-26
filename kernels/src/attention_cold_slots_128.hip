// SPDX-License-Identifier: Apache-2.0
// hipfire — head_dim=128 variant of attention_cold_slots (see that file for the
// full design). Identical kernel; only CHD/CPL change (128 head_dim → 4 elements
// per 32-lane wave instead of 8). Selected by the runtime when head_dim==128
// (the FWHT-256 vs FWHT-128 split mirrors mq_rotate_x vs mq_rotate_x_128).
//
// Parity oracle: hipfire_kvquant::ColdTier::two_tier_attend with n_hot = 0.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define CHD 128       // head_dim
#define CPL 4         // CHD / 32 elements per lane

// Read this lane's CPL elements of slot `s` for one operand, per its layout:
//   0 = SLOT-major  f32  : element (s,d) at slot_base + s*CHD + (d within lane)
//   1 = CHANNEL-major f16 : element (s,d) at ch_base + (d)*sstride + s
//   2 = SLOT-major  f16  : element (s,d) at slot_base + s*CHD + (d within lane)
__device__ __forceinline__ void cold_read_slot(
    const void* __restrict__ P, int layout,
    long long slot_base, long long ch_base, int s, int sstride, float* out
) {
    if (layout == 0) {
        const float* p = (const float*)P + slot_base + (long long)s * CHD;
        #pragma unroll
        for (int i = 0; i < CPL; ++i) out[i] = p[i];
    } else if (layout == 2) {
        const __half* p = (const __half*)P + slot_base + (long long)s * CHD;
        #pragma unroll
        for (int i = 0; i < CPL; ++i) out[i] = __half2float(p[i]);
    } else { // 1: channel-major f16
        const __half* p = (const __half*)P + ch_base + s;
        #pragma unroll
        for (int i = 0; i < CPL; ++i) out[i] = __half2float(p[(long long)i * sstride]);
    }
}

__launch_bounds__(32)
extern "C" __global__ void attention_cold_slots_128(
    const float* __restrict__ Q,   // [n_heads * 128]
    const float* __restrict__ K,   // [n_kv_heads * n_slots * 128]
    const float* __restrict__ V,   // [n_kv_heads * n_slots * 128]
    float*       __restrict__ OUT, // [n_heads * 128] (normalized)
    float*       __restrict__ M_OUT, // [n_heads] flash max (or null) — for tier merge
    float*       __restrict__ L_OUT, // [n_heads] flash denom (or null)
    int n_heads, int n_kv_heads, int n_slots, float scale,
    int k_layout, int v_layout,
    int slot_stride,
    float* __restrict__ MASS_OUT
) {
    const int sstride = (slot_stride > 0) ? slot_stride : n_slots;
    const int hq = blockIdx.x;
    if (hq >= n_heads) return;
    const int lane = threadIdx.x;          // 0..31
    const int kv = hq / (n_heads / n_kv_heads);

    // Load this lane's CPL query elements.
    float q[CPL];
    const float* qh = Q + (long long)hq * CHD + lane * CPL;
    #pragma unroll
    for (int i = 0; i < CPL; ++i) q[i] = qh[i];

    float m = -1e30f, l = 0.0f;
    float acc[CPL];
    #pragma unroll
    for (int i = 0; i < CPL; ++i) acc[i] = 0.0f;

    const long long slot_base = (long long)kv * sstride * CHD + lane * CPL; // slot-major
    const long long ch_base   = (long long)kv * CHD * sstride + lane * CPL * (long long)sstride; // channel-major
    for (int s = 0; s < n_slots; ++s) {
        float kk[CPL], vv[CPL];
        cold_read_slot(K, k_layout, slot_base, ch_base, s, sstride, kk);
        cold_read_slot(V, v_layout, slot_base, ch_base, s, sstride, vv);
        float d = 0.0f;
        #pragma unroll
        for (int i = 0; i < CPL; ++i) d += q[i] * kk[i];
        // butterfly all-reduce across the 32-lane wave → every lane holds the sum.
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) d += __shfl_xor(d, off);
        d *= scale;

        const float m_new = fmaxf(m, d);
        const float corr = __expf(m - m_new);   // <= 1
        const float p = __expf(d - m_new);
        l = l * corr + p;
        #pragma unroll
        for (int i = 0; i < CPL; ++i) acc[i] = acc[i] * corr + p * vv[i];
        m = m_new;
    }

    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    float* oh = OUT + (long long)hq * CHD + lane * CPL;
    #pragma unroll
    for (int i = 0; i < CPL; ++i) oh[i] = acc[i] * inv;

    if (lane == 0) {
        if (M_OUT) M_OUT[hq] = m;
        if (L_OUT) L_OUT[hq] = l;
    }

    if (MASS_OUT) {
        for (int s = 0; s < n_slots; ++s) {
            const float* k = K + slot_base + (long long)s * CHD;
            float d = 0.0f;
            #pragma unroll
            for (int i = 0; i < CPL; ++i) d += q[i] * k[i];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) d += __shfl_xor(d, off);
            d *= scale;
            if (lane == 0) {
                atomicAdd(&MASS_OUT[s], __expf(d - m) * inv);
            }
        }
    }
}
