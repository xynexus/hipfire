// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

#define HD 128
#define TILE_ROWS 4

// Routed FP16-state Gated Delta Net recurrence for independent request sessions
// — the f16 counterpart of `gated_delta_net_f32_routed_batch_seq`.
//
// Grid: [n_heads, HD/TILE_ROWS, n_sessions]; block [32].
//
// Each block owns one session/head/S-row-tile and scans the flattened
// round-major rows, applying only those whose `row_session_indices[b]` matches
// its `blockIdx.z`. That preserves exact per-session recurrence order while
// sessions share one launch.
//
// This is the entry point that MATTERS for f16. State is per-SEQUENCE, so its
// cost scales with concurrency, and cross-session batching is precisely the
// high-concurrency case: `state_ptrs` holds one state per session, so halving
// each one halves the whole fleet's resident state. On 27B that is ~149 MB ->
// ~75 MB per session, doubled again while spec-decoding.
//
// Storage f16, arithmetic FP32, identical to the linear f16 kernel: the tile
// widens into LDS once, the recurrence runs in f32 registers, and it narrows
// once on store — so a session pays ONE rounding per launch, not one per row.
//
// The pointers in `state_ptrs` therefore address f16 buffers (2 bytes/element),
// HALF the f32 stride. A caller that allocated f32 state and reached this kernel
// would read every element at the wrong offset — which is why the state's
// precision and the kernel choice are decided together at the call site rather
// than independently.
//
// Lane mapping is `col = tid * 4`, matching both the FP32 routed kernel and the
// linear f16 kernel, so per-session results stay comparable element-for-element
// across all four variants.
extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_f16_routed_batch_seq(
    const float* __restrict__ q,                    // [batch_rows * n_heads * HD]
    const float* __restrict__ k,                    // [batch_rows * n_heads * HD]
    const float* __restrict__ v,                    // [batch_rows * n_heads * HD]
    const float* __restrict__ gate,                 // [batch_rows * n_heads]
    const float* __restrict__ beta,                 // [batch_rows * n_heads]
    const uint64_t* __restrict__ state_ptrs,        // [n_sessions * ptr_layer_stride], -> f16
    const int* __restrict__ row_session_indices,    // [batch_rows]
    float* __restrict__ output,                     // [batch_rows * n_heads * HD]
    int ptr_layer_stride,
    int delta_layer_index,
    int batch_rows,
    int n_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    const int tile = blockIdx.y;
    const int session_idx = blockIdx.z;
    if (h >= n_heads || head_dim != HD) return;

    const int tid = threadIdx.x;
    const int col = tid * 4;
    const int stride = n_heads * HD;
    const int row_start = tile * TILE_ROWS;

    __shared__ float S_tile[TILE_ROWS * HD];  // FP32 working copy

    _Float16* state = reinterpret_cast<_Float16*>(
        static_cast<uintptr_t>(state_ptrs[(size_t)session_idx * ptr_layer_stride + delta_layer_index])
    );
    _Float16* S_global = state + h * HD * HD + row_start * HD;
    for (int i = tid; i < TILE_ROWS * HD; i += 32) {
        S_tile[i] = (float)S_global[i];
    }

    for (int b = 0; b < batch_rows; b++) {
        if (row_session_indices[b] != session_idx) continue;

        const float* q_t = q + b * stride + h * HD;
        const float* k_t = k + b * stride + h * HD;
        const float* v_t = v + b * stride + h * HD;
        float alpha = expf(gate[b * n_heads + h]);
        float beta_v = beta[b * n_heads + h];

        float k0 = k_t[col], k1 = k_t[col + 1], k2 = k_t[col + 2], k3 = k_t[col + 3];
        float q0 = q_t[col], q1 = q_t[col + 1], q2 = q_t[col + 2], q3 = q_t[col + 3];

        for (int r = 0; r < TILE_ROWS; r++) {
            int row = row_start + r;
            float* sr = S_tile + r * HD + col;
            float s0 = sr[0], s1 = sr[1], s2 = sr[2], s3 = sr[3];

            float kv = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
            for (int o = 16; o > 0; o >>= 1) {
                kv += __shfl_down(kv, o);
            }
            kv = __shfl(kv, 0);

            float delta = (v_t[row] - alpha * kv) * beta_v;

            s0 = alpha * s0 + k0 * delta;
            s1 = alpha * s1 + k1 * delta;
            s2 = alpha * s2 + k2 * delta;
            s3 = alpha * s3 + k3 * delta;
            sr[0] = s0;
            sr[1] = s1;
            sr[2] = s2;
            sr[3] = s3;

            float out_v = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
            for (int o = 16; o > 0; o >>= 1) {
                out_v += __shfl_down(out_v, o);
            }
            if (tid == 0) {
                output[b * stride + h * HD + row] = out_v;
            }
        }
    }

    for (int i = tid; i < TILE_ROWS * HD; i += 32) {
        S_global[i] = (_Float16)S_tile[i];
    }
}
