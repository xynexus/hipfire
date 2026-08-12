// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Per-EXPERT AWQ rotate for the indexed MoE path. Derived from
// `rotate_x_mq_awq` (same FWHT, same normalisation); the only difference is
// WHICH awq_scale each output slot divides by, and that the output is expanded
// per (token, krank).
//
// Why this kernel has to exist. AWQ stores W·s offline and the forward must
// compute (W·s)·(x/s). The indexed MoE path divided x by ONE awq_scale per
// layer — the shared expert's — and handed that single buffer to all K_TOP
// routed experts. Routed-expert scales are NOT identical (measured on a
// 35B-A3B oq4.25++: expert0/expert1/expert7/shared all differ), because routing
// sends each expert a different token subset, so each gets a different imatrix
// and a different s. Every routed expert was therefore off by s_e/s_shared per
// channel from layer 0.
//
// Folding the divide into the GEMV instead is NOT possible: the divide must
// precede the FWHT, and the FWHT mixes channels, so FWHT(x/s) != FWHT(x)/s.
// Hence a genuinely per-expert rotated buffer.
//
//   x_in            [N x K]              — shared post-rmsnorm activation
//   expert_awq_ptrs [n_exp] u64          — device pointer per expert to [K] f32
//   topk_indices    [N x K_TOP] i32
//   x_out           [N x K_TOP x K]      — one rotated basis per (token, krank)
//
// Grid (K/256, N*K_TOP, 1), block 32. blockIdx.y is the flattened slot.
//
// A null awq pointer means that expert carries no sidecar; the slot then gets
// the plain rotation, matching `rotate_x_mq_batched` byte-for-byte so mixed
// artifacts (some experts AWQ-scaled, some not) stay correct. A null TABLE
// (expert_awq_ptrs itself) means no expert at this layer has one: every slot
// takes the plain rotation, so this kernel is also the per-slot expander for
// non-AWQ artifacts and callers need no second code path. The expansion is
// required regardless of AWQ — the indexed gate_up GEMVs read x per slot.

extern "C" __launch_bounds__(32, 16)
__global__ void rotate_x_mq_awq_indexed_batched(
    const float* __restrict__ x_in,
    const unsigned long long* __restrict__ expert_awq_ptrs,
    const int* __restrict__ topk_indices,
    float* __restrict__ x_out,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int K,
    int K_TOP
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    if (group >= groups_total) return;

    const int slot = blockIdx.y;          // token * K_TOP + krank
    const int token = slot / K_TOP;

    // Input is shared across kranks (every routed expert consumes the same
    // post-rmsnorm x); only the scale and the destination differ.
    const float* __restrict__ x_in_b = x_in + (long long)token * K;
    float* __restrict__ x_out_b = x_out + (long long)slot * K;
    const float* __restrict__ awq =
        expert_awq_ptrs == nullptr
            ? nullptr
            : reinterpret_cast<const float*>(expert_awq_ptrs[topk_indices[slot]]);

    int d0 = tid * 8;
    int base = group * 256 + d0;

    // AWQ divide BEFORE the FWHT — see rotate_x_mq_awq.hip. awq is indexed in
    // the unrotated x basis.
    float v0, v1, v2, v3, v4, v5, v6, v7;
    if (awq != nullptr) {
        v0 = (x_in_b[base    ] / awq[base    ]) * signs1[d0    ];
        v1 = (x_in_b[base + 1] / awq[base + 1]) * signs1[d0 + 1];
        v2 = (x_in_b[base + 2] / awq[base + 2]) * signs1[d0 + 2];
        v3 = (x_in_b[base + 3] / awq[base + 3]) * signs1[d0 + 3];
        v4 = (x_in_b[base + 4] / awq[base + 4]) * signs1[d0 + 4];
        v5 = (x_in_b[base + 5] / awq[base + 5]) * signs1[d0 + 5];
        v6 = (x_in_b[base + 6] / awq[base + 6]) * signs1[d0 + 6];
        v7 = (x_in_b[base + 7] / awq[base + 7]) * signs1[d0 + 7];
    } else {
        v0 = x_in_b[base    ] * signs1[d0    ];
        v1 = x_in_b[base + 1] * signs1[d0 + 1];
        v2 = x_in_b[base + 2] * signs1[d0 + 2];
        v3 = x_in_b[base + 3] * signs1[d0 + 3];
        v4 = x_in_b[base + 4] * signs1[d0 + 4];
        v5 = x_in_b[base + 5] * signs1[d0 + 5];
        v6 = x_in_b[base + 6] * signs1[d0 + 6];
        v7 = x_in_b[base + 7] * signs1[d0 + 7];
    }

    // Local butterfly: strides 1, 2, 4
    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;

    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;

    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    // Wave butterfly: strides 1-16 in thread space via ds_swizzle.
    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while(0)

    #define HBFLY8(pat, str) \
        HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
        HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)

    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8
    #undef HBFLY

    const float s = 0.0625f;
    x_out_b[base    ] = v0 * s * signs2[d0    ];
    x_out_b[base + 1] = v1 * s * signs2[d0 + 1];
    x_out_b[base + 2] = v2 * s * signs2[d0 + 2];
    x_out_b[base + 3] = v3 * s * signs2[d0 + 3];
    x_out_b[base + 4] = v4 * s * signs2[d0 + 4];
    x_out_b[base + 5] = v5 * s * signs2[d0 + 5];
    x_out_b[base + 6] = v6 * s * signs2[d0 + 6];
    x_out_b[base + 7] = v7 * s * signs2[d0 + 7];
}
