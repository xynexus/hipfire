// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Per-EXPERT AWQ variant of `fused_silu_mul_mq_rotate_awq`, for the indexed
// MoE down_proj input stage.
//
// Same bug, same fix as `rotate_x_mq_awq_indexed_batched` on the gate_up side:
// AWQ stores W·s offline, so the forward must divide by that expert's own s,
// and routed experts do NOT share one (measured on a 35B-A3B oq4.25++). The
// indexed path divided the whole [N x K_TOP x MI] SwiGLU buffer by ONE scale —
// `experts[0].down`'s — so every slot routed elsewhere was off by s_e/s_0 per
// channel. The divide cannot fold into the GEMV because it must precede the
// FWHT, which mixes channels.
//
// Unlike the gate_up side this needs no reshape: gate, up and x_rot are ALREADY
// one row per (token, krank), so only the scale lookup changes. blockIdx.y is
// the flattened slot and indexes topk_indices directly.
//
//   gate, up        [N x K_TOP x MI]     — routed SwiGLU operands
//   expert_awq_ptrs [n_exp] u64          — device pointer per expert to [MI] f32
//   topk_indices    [N x K_TOP] i32
//   x_rot           [N x K_TOP x MI]     — rotated down_proj input
//
// Grid (MI/256, N*K_TOP, 1), block 32.
//
// A null per-expert pointer, or a null TABLE, means no sidecar applies and the
// slot takes the plain rotation — byte-identical to `fused_silu_mul_mq_rotate`.

extern "C" __launch_bounds__(32, 16)
__global__ void fused_silu_mul_mq_rotate_awq_indexed(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const unsigned long long* __restrict__ expert_awq_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    int K
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    if (group >= groups_total) return;

    // grid.y is the flattened slot (token * K_TOP + krank). gate, up and x_rot
    // are already expanded per slot, so the row offset is the same one
    // `fused_silu_mul_mq_rotate_awq` uses for its batch dim.
    const int slot = blockIdx.y;
    const long long batch_off = (long long)slot * K;
    const float* gate_b  = gate  + batch_off;
    const float* up_b    = up    + batch_off;
          float* x_rot_b = x_rot + batch_off;

    const float* __restrict__ awq =
        expert_awq_ptrs == nullptr
            ? nullptr
            : reinterpret_cast<const float*>(expert_awq_ptrs[topk_indices[slot]]);

    int d0 = tid * 8;
    int base = group * 256 + d0;

    // Phase 1: read gate + up, compute silu(gate)*up, AWQ divide, signs1.
    // silu(z) = z / (1 + exp(-z)). AWQ divide happens BEFORE signs1+FWHT
    // to mirror the offline weight-side pre-scaling W' = W · diag(s).
    // awq is indexed in the unrotated feature basis (1D, length K).
    #define SILU_MUL(g, u) ((g) / (1.0f + expf(-(g))) * (u))
    float g0 = gate_b[base];     float u0 = up_b[base];
    float g1 = gate_b[base + 1]; float u1 = up_b[base + 1];
    float g2 = gate_b[base + 2]; float u2 = up_b[base + 2];
    float g3 = gate_b[base + 3]; float u3 = up_b[base + 3];
    float g4 = gate_b[base + 4]; float u4 = up_b[base + 4];
    float g5 = gate_b[base + 5]; float u5 = up_b[base + 5];
    float g6 = gate_b[base + 6]; float u6 = up_b[base + 6];
    float g7 = gate_b[base + 7]; float u7 = up_b[base + 7];

    float v0, v1, v2, v3, v4, v5, v6, v7;
    if (awq != nullptr) {
        v0 = (SILU_MUL(g0, u0) / awq[base    ]) * signs1[d0    ];
        v1 = (SILU_MUL(g1, u1) / awq[base + 1]) * signs1[d0 + 1];
        v2 = (SILU_MUL(g2, u2) / awq[base + 2]) * signs1[d0 + 2];
        v3 = (SILU_MUL(g3, u3) / awq[base + 3]) * signs1[d0 + 3];
        v4 = (SILU_MUL(g4, u4) / awq[base + 4]) * signs1[d0 + 4];
        v5 = (SILU_MUL(g5, u5) / awq[base + 5]) * signs1[d0 + 5];
        v6 = (SILU_MUL(g6, u6) / awq[base + 6]) * signs1[d0 + 6];
        v7 = (SILU_MUL(g7, u7) / awq[base + 7]) * signs1[d0 + 7];
    } else {
        v0 = SILU_MUL(g0, u0) * signs1[d0    ];
        v1 = SILU_MUL(g1, u1) * signs1[d0 + 1];
        v2 = SILU_MUL(g2, u2) * signs1[d0 + 2];
        v3 = SILU_MUL(g3, u3) * signs1[d0 + 3];
        v4 = SILU_MUL(g4, u4) * signs1[d0 + 4];
        v5 = SILU_MUL(g5, u5) * signs1[d0 + 5];
        v6 = SILU_MUL(g6, u6) * signs1[d0 + 6];
        v7 = SILU_MUL(g7, u7) * signs1[d0 + 7];
    }
    #undef SILU_MUL

    // Local butterfly: strides 1, 2, 4 — identical to mq_rotate_x.
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

    // Wave butterfly via ds_swizzle — same as mq_rotate_x.
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
    x_rot_b[base    ] = v0 * s * signs2[d0    ];
    x_rot_b[base + 1] = v1 * s * signs2[d0 + 1];
    x_rot_b[base + 2] = v2 * s * signs2[d0 + 2];
    x_rot_b[base + 3] = v3 * s * signs2[d0 + 3];
    x_rot_b[base + 4] = v4 * s * signs2[d0 + 4];
    x_rot_b[base + 5] = v5 * s * signs2[d0 + 5];
    x_rot_b[base + 6] = v6 * s * signs2[d0 + 6];
    x_rot_b[base + 7] = v7 * s * signs2[d0 + 7];
}
