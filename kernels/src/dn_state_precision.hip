// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Widen / narrow the Gated DeltaNet S state between its persistent f16 storage
// and a transient f32 shadow.
//
// WHY THIS EXISTS. `gated_delta_net_f16` keeps S as an FP32 working copy in LDS
// for the whole launch and narrows to f16 exactly once, at the end. So the number
// of narrowings equals the number of LAUNCHES: batched prefill (one launch per
// chunk) rounds once per chunk, while the per-token fallback (one launch per
// token) rounds once per token. That cadence difference — not any arithmetic in
// the kernel, which is already f32 — is the measured hidden-state divergence
// between the two prefill paths.
//
// These let the per-token path hold S in f32 across the whole loop and round on
// the SAME boundaries the batched path rounds on.
//
// ⚠️ The narrow MUST reproduce `f32_to_f16_dither` from `gated_delta_net_f16.hip`
// bit-for-bit, INCLUDING its index derivation. There, the state pointer is
// `state + h*HD*HD + row_start*HD` and the dither index is
// `(row_start*HD + i) ^ (h << 19)` — i.e. the element's flat offset WITHIN ITS
// HEAD, xor the head index shifted. Using a flat global element index instead
// produces different dither for every head but head 0, and the paths still
// diverge — the cadence would be fixed and the numbers still wrong.

#define BLOCK 256

// Verbatim from gated_delta_net_f16.hip. Duplicated deliberately: if that one
// changes, this must change with it, and a shared header would hide that a
// divergence-critical definition is shared.
__device__ __forceinline__ _Float16 f32_to_f16_dither(float x, unsigned int idx) {
    unsigned int u = __float_as_uint(x);
    unsigned int r = u ^ (idx * 2654435761u);
    r ^= r >> 15; r *= 2246822519u; r ^= r >> 13; r *= 3266489917u; r ^= r >> 16;
    unsigned int dithered = (u + (r & 0x1FFFu)) & 0xFFFFE000u;
    return (_Float16)__uint_as_float(dithered);
}

// f16 -> f32. Exact: every f16 is representable in f32, so seeding a shadow from
// a live state loses nothing. Needed because the shadow CANNOT simply be zeroed —
// prefill runs mid-sequence (DFlash's FullPrefill rollback replays a committed
// prefix), where S is already non-zero.
extern "C" __global__ void dn_state_widen_f16_f32(
    const _Float16* __restrict__ src,
    float*          __restrict__ dst,
    int n
) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    dst[i] = (float)src[i];
}

// f32 -> f16, matching the in-kernel narrowing exactly.
//
// `elems_per_head` is HD*HD (16384 at head_dim 128); the head index and the
// within-head offset are recovered from it so the dither index matches what
// `gated_delta_net_f16` would have produced for the same element.
extern "C" __global__ void dn_state_narrow_f32_f16(
    const float*    __restrict__ src,
    _Float16*       __restrict__ dst,
    int n,
    int elems_per_head,
    int stochastic_round
) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    if (stochastic_round) {
        const unsigned int h = (unsigned int)(i / elems_per_head);
        const unsigned int within = (unsigned int)(i - (int)h * elems_per_head);
        dst[i] = f32_to_f16_dither(src[i], within ^ (h << 19));
    } else {
        dst[i] = (_Float16)src[i];
    }
}
