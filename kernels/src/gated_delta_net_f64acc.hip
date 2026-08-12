// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// FP64-accumulate DeltaNet — the numerical ORACLE for the f32 and f16 state
// paths, not a serving kernel.
//
// State storage stays FP32, identical to `gated_delta_net.hip`; only the working
// tile and the arithmetic move to double. That isolates one term: the error the
// f32 kernel accrues inside its own tile (dot products, the delta update, and
// the `__shfl_down` reductions across HD terms), separate from any storage
// round-trip.
//
// Why this exists. Every FP16-vs-FP32 state number measured so far is a
// comparison against `gated_delta_net.hip`, which is itself a drifting f32
// accumulator that nothing has ever validated. Without an oracle there is no way
// to say how much of a measured FP16 penalty is FP16 and how much is the
// reference moving underneath it. The f16 kernel's error decomposes as
//
//     f16 error = (f32 arithmetic error)  +  (per-step narrowing error)
//                  ^ shared with f32         ^ unique to f16
//
// and this kernel measures the first term by removing it.
//
// NOT for serving: gfx1103 is a consumer RDNA3 part, so fp64 runs at a small
// fraction of the fp32 rate. This is an oracle build — correctness reference,
// measured offline, selected by `HIPFIRE_DN_STATE_F64_ORACLE=1`.
//
// Lane mapping and loop structure match `gated_delta_net.hip` exactly (`col =
// tid * 4`, TILE_ROWS rows per block) so the two are comparable
// element-for-element; the only difference is accumulation precision.

#include <hip/hip_runtime.h>

#define HD 128
#define TILE_ROWS 4

extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_f64acc(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ output,
    int n_tokens,
    int n_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    const int tile = blockIdx.y;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int col = tid * 4;
    const int stride = n_heads * HD;
    const int row_start = tile * TILE_ROWS;

    // TILE_ROWS(4) x 128 doubles = 4 KiB, against 2 KiB for the f32 tile.
    // NOTE: TILE_ROWS MUST match gated_delta_net.hip — the dispatcher launches
    // 128/TILE_ROWS blocks, so a mismatch silently overruns the row range. The
    // f32 kernel's own comment says "TILE_ROWS x 128 floats = 4KB", which is
    // stale (4x128x4 = 2KB) and is what made an earlier copy of this file use 8.
    __shared__ double S_tile[TILE_ROWS * HD];

    float* S_global = state + h * HD * HD + row_start * HD;
    // Widening f32 -> f64 is exact, so the load contributes no error.
    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_tile[i] = (double)S_global[i];

    for (int t = 0; t < n_tokens; t++) {
        const float* q_t = q + t * stride + h * HD;
        const float* k_t = k + t * stride + h * HD;
        const float* v_t = v + t * stride + h * HD;
        double alpha = exp((double)gate[t * n_heads + h]);
        double beta_v = (double)beta[t * n_heads + h];

        double k0 = k_t[col], k1 = k_t[col+1], k2 = k_t[col+2], k3 = k_t[col+3];
        double q0 = q_t[col], q1 = q_t[col+1], q2 = q_t[col+2], q3 = q_t[col+3];

        for (int r = 0; r < TILE_ROWS; r++) {
            int row = row_start + r;
            double* sr = S_tile + r * HD + col;
            double s0 = sr[0], s1 = sr[1], s2 = sr[2], s3 = sr[3];

            double kv = s0*k0 + s1*k1 + s2*k2 + s3*k3;
            // HIP provides 64-bit __shfl_down (two 32-bit moves underneath), so
            // the cross-lane reduction stays in double as well — reducing in
            // f32 here would reintroduce the very term this kernel removes.
            for (int o = 16; o > 0; o >>= 1)
                kv += __shfl_down(kv, o);
            kv = __shfl(kv, 0);

            double delta = ((double)v_t[row] - alpha * kv) * beta_v;

            s0 = alpha * s0 + k0 * delta;
            s1 = alpha * s1 + k1 * delta;
            s2 = alpha * s2 + k2 * delta;
            s3 = alpha * s3 + k3 * delta;
            sr[0] = s0; sr[1] = s1; sr[2] = s2; sr[3] = s3;

            double out_v = s0*q0 + s1*q1 + s2*q2 + s3*q3;
            for (int o = 16; o > 0; o >>= 1)
                out_v += __shfl_down(out_v, o);
            if (tid == 0)
                output[t * stride + h * HD + row] = (float)out_v;
        }
    }

    // Narrow back to the f32 storage format. This IS a rounding, but it happens
    // once per call on a value that carried full double precision through the
    // whole token loop — the point is to remove the accumulated arithmetic
    // error, not to change the storage format.
    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_global[i] = (float)S_tile[i];
}
