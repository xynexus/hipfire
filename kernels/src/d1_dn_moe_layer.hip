// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// D1: one Qwen3.6-35B-A3B DeltaNetMoe layer as a counter-scheduled persistent
// kernel (docs/plans/2026-10-09-d1-persistent-layer-kernel.md).
//
// COVERS the attention half: attn rmsnorm; for each of wqkv / wz / w_beta /
// w_alpha a per-weight AWQ rotate + int8 quantize and the wide multicol W4A8
// GEMV; the beta/alpha gates; the routed conv1d + silu + split; q/k l2-norm and
// the 16 -> 32 head repeat; the routed f16-state gated delta-net; the gated
// norm; and wo (rotate + quantize + multicol) added into the residual. The FFN
// half still runs as the op chain, from the buffers this writes.
//
// SCHEDULING. Ops are split into work items (a row's norm, a row's rotate+
// quantize for one weight, 8 waves x RW rows of a GEMV, the gates). Items are
// numbered in a fixed topological order; a workgroup takes the next number with
// one atomic, waits until every op its item depends on has completed (per-op
// counters, agent-scope acquire), runs it, and bumps its op's counter (release).
// The launch is cooperative, so every workgroup is resident: an item waited on
// was taken earlier by a running workgroup, and the waits cannot deadlock.
//
// NUMERICS. Every op body is the arithmetic of its chain kernel, in its order:
//   rmsnorm_f32_gfx1151, rotate_x_mq_awq, quantize_act_oq8,
//   gemv_oq_compact_multicol_w{N} (oqcmw_body<N, RW>), fused_sigmoid_alpha_gate_f32,
//   conv1d_silu_split_routed_f32, fused_qk_l2_norm_scale_f32,
//   repeat_interleave_qk_f32_batched, gated_delta_net_f16_routed_batch_seq,
//   gated_norm_f32, add_inplace_f32.
// Rotate and quantize are fused per wave: a rotate lane holds exactly the eight
// elements its quantize lane reads (8L..8L+7 of the group), so the int8 output
// is the chain's, without the f32 round trip. The layer's outputs must be
// BIT-IDENTICAL to the chain's (HIPFIRE_D1_CHECK).
//
// D1_N (rows) is compiled in -- one module per row count, like the multicol
// entry points -- because the GEMV's per-row accumulators are sized by it.

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef D1_N
#define D1_N 1
#endif
#define D1_RW (D1_N <= 16 ? 3 : 2)
#define D1_WAVES 8
#define HD 128
#define TILE_ROWS 4
#define D1_LDS_GEMV (4 * D1_N * 256)
#define D1_LDS_GDN (D1_WAVES * TILE_ROWS * HD * 4)
#define D1_LDS (D1_LDS_GEMV > D1_LDS_GDN ? D1_LDS_GEMV : D1_LDS_GDN)

struct D1Proj {
    const uint8_t* w;      // OqCompact split-plane blocks
    const float* awq;      // [K] AWQ scale (the activation is divided by it)
    float* y;              // [n x m]
    int8_t* xq;            // [n x K] int8 activation for this weight
    float* xs;             // [n x K/256] its group scales
    float* resid;          // wo: [n x m] residual the output is added into, else null
    int m;
    int k;
    int block_stride;
    int pad;
};

struct D1Params {
    const float* x;        // [n x dim] residual stream (wo adds into proj[4].resid)
    const float* attn_norm;
    float* xn;             // [n x dim] normed (the chain's x_rot_batch)
    const float* signs1;
    const float* signs2;
    const float* dt_bias;
    const float* a_log;
    int* counters;         // [0] next item, [1 + op] items done, [1 + D1_NOPS] exited
    float* q_raw;          // [n x k_dim]
    float* k_raw;
    float* v;              // [n x v_dim]
    float* q;              // [n x n_v_heads*HD] after the head repeat
    float* k;
    const float* conv_weight;   // [conv channels x 4]
    const uint64_t* conv_ptrs;  // per (session, layer) conv ring, f32 [channels x 3]
    const uint64_t* s_ptrs;     // per (session, layer) delta-net state, f16
    const int* row_session;     // [n]
    float* attn_out;       // [n x v_dim]
    float* normed;         // [n x v_dim]
    const float* norm_weight;   // [HD]
    D1Proj proj[5];        // wqkv, wz, w_beta, w_alpha, wo
    int n;
    int dim;
    int n_v_heads;
    int n_k_heads;
    int k_dim;
    int v_dim;
    int sessions;
    int ptr_stride;
    int layer;
    int stochastic_round;
    float eps;
    float q_scale;
};

// Ops. GEMV + 4 is wo; its rotate + quantize runs inside OP_GN.
enum {
    OP_NORM = 0, OP_ROTQ = 1, OP_GEMV = 5, OP_GATE = 10, OP_CONV = 11, OP_QK = 12,
    OP_GDN = 13, OP_GN = 14, D1_NOPS = 15
};

// ── op bodies ──────────────────────────────────────────────────────────────

static __device__ __forceinline__ float d1_wave_sum32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down(v, offset, 32);
    return v;
}

// rmsnorm_f32_gfx1151, one row per item (256 threads).
static __device__ void d1_rmsnorm_row(const D1Params& p, int r, float* wave_sums) {
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int wave_id = tid >> 5;
    const int num_waves = (blockDim.x + 31) >> 5;
    const int n = p.dim;
    const long long row_off = (long long)r * n;
    const float* x_row = p.x + row_off;
    float* out_row = p.xn + row_off;
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = x_row[i];
        local_sum += v * v;
    }
    const float wave_sum = d1_wave_sum32(local_sum);
    if (lane_id == 0) wave_sums[wave_id] = wave_sum;
    __syncthreads();
    float total = 0.0f;
    if (wave_id == 0) {
        total = (lane_id < num_waves) ? wave_sums[lane_id] : 0.0f;
        total = d1_wave_sum32(total);
        if (lane_id == 0) wave_sums[0] = total;
    }
    __syncthreads();
    const float inv_rms = rsqrtf(wave_sums[0] / (float)n + p.eps);
    for (int i = tid; i < n; i += blockDim.x) out_row[i] = x_row[i] * p.attn_norm[i] * inv_rms;
}

// rotate_x_mq_awq then quantize_act_oq8 (group 256) on one 256-group of row b,
// one wave.
static __device__ void d1_rotq_group(const D1Params& p, const D1Proj& pj, const float* src,
                                     int b, int group) {
    const int tid = threadIdx.x & 31;
    const int K = pj.k;
    const long long batch_off = (long long)b * K;
    const float* x_in_b = src + batch_off;
    const float* awq_scale = pj.awq;
    const float* signs1 = p.signs1;
    const float* signs2 = p.signs2;
    int d0 = tid * 8;
    int base = group * 256 + d0;
    float v0 = (x_in_b[base    ] / awq_scale[base    ]) * signs1[d0    ];
    float v1 = (x_in_b[base + 1] / awq_scale[base + 1]) * signs1[d0 + 1];
    float v2 = (x_in_b[base + 2] / awq_scale[base + 2]) * signs1[d0 + 2];
    float v3 = (x_in_b[base + 3] / awq_scale[base + 3]) * signs1[d0 + 3];
    float v4 = (x_in_b[base + 4] / awq_scale[base + 4]) * signs1[d0 + 4];
    float v5 = (x_in_b[base + 5] / awq_scale[base + 5]) * signs1[d0 + 5];
    float v6 = (x_in_b[base + 6] / awq_scale[base + 6]) * signs1[d0 + 6];
    float v7 = (x_in_b[base + 7] / awq_scale[base + 7]) * signs1[d0 + 7];
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
    float r[8];
    r[0] = v0 * s * signs2[d0    ];
    r[1] = v1 * s * signs2[d0 + 1];
    r[2] = v2 * s * signs2[d0 + 2];
    r[3] = v3 * s * signs2[d0 + 3];
    r[4] = v4 * s * signs2[d0 + 4];
    r[5] = v5 * s * signs2[d0 + 5];
    r[6] = v6 * s * signs2[d0 + 6];
    r[7] = v7 * s * signs2[d0 + 7];

    // quantize_act_oq8: lane `tid` owns elements tid*8 .. tid*8+7 of the group.
    const int lane = tid;
    const int per_lane = 8;
    const int n_groups = K / 256;
    float amax = 0.0f;
    for (int i = 0; i < per_lane; ++i) amax = fmaxf(amax, fabsf(r[i]));
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_down(amax, off));
    amax = __shfl(amax, 0);
    const float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    if (lane == 0) pj.xs[(long long)b * n_groups + group] = scale;
    int8_t* out = pj.xq + (long long)b * K + (long long)group * 256;
    for (int i = 0; i < per_lane; ++i) {
        const int idx = lane * per_lane + i;
        int q = (int)lroundf(r[i] * inv);
        q = max(-127, min(127, q));
        out[idx] = (int8_t)q;
    }
}

static __device__ __forceinline__ int d1_dot4_i8(int a, int b, int acc) {
    return __builtin_amdgcn_sudot4(true, a, true, b, acc, false);
}
static __device__ __forceinline__ void d1_decode(uint32_t w, uint32_t* p0, uint32_t* p1) {
    const uint32_t lo4 = w & 0x0F0F0F0Fu;
    const uint32_t hi4 = (w >> 4) & 0x0F0F0F0Fu;
    uint32_t a = __builtin_amdgcn_perm(hi4, lo4, 0x05010400u);
    uint32_t b = __builtin_amdgcn_perm(hi4, lo4, 0x07030602u);
    a |= ((a >> 3) & 0x01010101u) * 0xF0u;
    b |= ((b >> 3) & 0x01010101u) * 0xF0u;
    *p0 = a;
    *p1 = b;
}

// oqcmw_body<D1_N, D1_RW>, one trip: this workgroup's 8 waves take waves
// item*8 .. item*8+7 of the chain kernel's wave numbering.
static __device__ void d1_multicol_item(const D1Proj& pj, int item, int8_t* lds_x) {
    constexpr int BC = D1_N;
    constexpr int RW = D1_RW;
    const uint8_t* __restrict__ W = pj.w;
    const int8_t* __restrict__ Xq = pj.xq;
    const float* __restrict__ Xs = pj.xs;
    float* __restrict__ Y = pj.y;
    const int M = pj.m;
    const int K = pj.k;
    const int block_stride = pj.block_stride;
    const int lane = threadIdx.x & 31;
    const int wid = item * D1_WAVES + (threadIdx.x >> 5);
    const int ng = K / 256;
    const int n_ov = (block_stride - 130) >> 1;
    const int side_stride = block_stride - 128;
    const long long side_base = (long long)M * ng * 128LL;
    const int gir = lane >> 3;
    const int lig = lane & 7;
    const int elt = lig * 32;

    const int row0 = wid * RW;
    const bool row_active = row0 < M;
    const uint8_t* n_row[RW];
    const uint8_t* s_row[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        const int row = (row_active && row0 + r < M) ? (row0 + r) : (M - 1);
        n_row[r] = W + (long long)row * ng * 128LL;
        s_row[r] = W + side_base + (long long)row * ng * (long long)side_stride;
    }
    float facc[RW][BC];
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int b = 0; b < BC; ++b) facc[r][b] = 0.0f;

    for (int g0 = 0; g0 < ng; g0 += 4) {
        __syncthreads();
        {
            const int tid = threadIdx.x;
            const int nthr = blockDim.x;
            const int bytes = 4 * BC * 256;
            for (int off = tid * 16; off < bytes; off += nthr * 16) {
                const int gl = off / (BC * 256);
                const int rem = off - gl * (BC * 256);
                const int b = rem >> 8;
                const int byte = rem & 255;
                const int gsrc = g0 + gl;
                const int8_t* src = (gsrc < ng)
                    ? Xq + (long long)b * K + (long long)gsrc * 256 + byte
                    : Xq;
                __builtin_memcpy(&lds_x[off], src, 16);
            }
        }
        __syncthreads();
        const int gi = g0 + gir;
        uint32_t wv[RW][4];
#pragma unroll
        for (int r = 0; r < RW; ++r)
            __builtin_memcpy(wv[r], n_row[r] + (long long)gi * 128 + lig * 16, 16);
        uint32_t pp[RW][8];
#pragma unroll
        for (int r = 0; r < RW; ++r)
#pragma unroll
            for (int j = 0; j < 4; ++j) d1_decode(wv[r][j], &pp[r][2 * j], &pp[r][2 * j + 1]);
        float sw[RW];
        const uint8_t* table[RW];
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const uint8_t* sb = s_row[r] + (long long)gi * side_stride;
            table[r] = sb + 2;
            sw[r] = (float)(*(const _Float16*)sb);
        }
        const bool has_ov = lig < n_ov;
        int oidx[RW], oval[RW];
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const int oe = has_ov ? lig : 0;
            oidx[r] = (int)table[r][2 * oe];
            oval[r] = has_ov ? (int)(int8_t)table[r][2 * oe + 1] : 0;
        }
        float sx[BC];
#pragma unroll
        for (int b = 0; b < BC; ++b) sx[b] = Xs[(long long)b * ng + gi];
#pragma unroll
        for (int b = 0; b < BC; ++b) {
            const int8_t* xq = &lds_x[((gir * BC) + b) * 256];
            uint32_t xv[8];
            __builtin_memcpy(&xv[0], xq + elt, 16);
            __builtin_memcpy(&xv[4], xq + elt + 16, 16);
#pragma unroll
            for (int r = 0; r < RW; ++r) {
                int acc = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j) acc = d1_dot4_i8((int)pp[r][j], (int)xv[j], acc);
                acc += oval[r] * (int)xq[oidx[r]];
                for (int e = lig + 8; e < n_ov; e += 8)
                    acc += (int)(int8_t)table[r][2 * e + 1] * (int)xq[(int)table[r][2 * e]];
                facc[r][b] += (float)acc * sw[r] * sx[b];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int b = 0; b < BC; ++b) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) facc[r][b] += __shfl_xor(facc[r][b], off);
        }
    if (lane == 0 && row_active) {
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            if (row0 + r >= M) break;
#pragma unroll
            for (int b = 0; b < BC; ++b) {
                // wo: the chain writes a scratch and add_inplace_f32's it in.
                if (pj.resid) pj.resid[(long long)b * M + row0 + r] += facc[r][b];
                else Y[(long long)b * M + row0 + r] = facc[r][b];
            }
        }
    }
}

// fused_sigmoid_alpha_gate_f32 over every row, one item.
static __device__ void d1_gate(const D1Params& p) {
    const int nv = p.n_v_heads;
    float* beta = p.proj[2].y;
    float* alpha = p.proj[3].y;
    for (int idx = threadIdx.x; idx < p.n * nv; idx += blockDim.x) {
        const int i = idx % nv;
        beta[idx] = 1.0f / (1.0f + expf(-beta[idx]));
        float biased = alpha[idx] + p.dt_bias[i];
        float sp = (biased > 20.0f) ? biased
                 : ((biased < -20.0f) ? expf(biased) : logf(1.0f + expf(biased)));
        alpha[idx] = sp * (-expf(p.a_log[i]));
    }
}

// conv1d_silu_split_routed_f32: channel block `cb` (256 channels) of one
// session, its rows in order.
static __device__ void d1_conv(const D1Params& p, int cb, int session_idx) {
    float* __restrict__ q_out = p.q_raw;
    float* __restrict__ k_out = p.k_raw;
    float* __restrict__ v_out = p.v;
    const float* __restrict__ input = p.proj[0].y;
    const float* __restrict__ weight = p.conv_weight;
    const int k_dim = p.k_dim;
    const int v_dim = p.v_dim;
    int c = cb * blockDim.x + threadIdx.x;
    int n_channels = 2 * k_dim + v_dim;
    if (c >= n_channels) return;

    float* state = reinterpret_cast<float*>(
        static_cast<uintptr_t>(p.conv_ptrs[(size_t)session_idx * p.ptr_stride + p.layer])
    );

    for (int b = 0; b < p.n; b++) {
        if (p.row_session[b] != session_idx) continue;

        float x = input[b * n_channels + c];
        float s0 = state[c * 3];
        float s1 = state[c * 3 + 1];
        float s2 = state[c * 3 + 2];

        // The chain kernel's contraction, pinned: under -ffp-contract=fast
        // it compiles to x*w3 then three fmacs in source order, and the
        // inlined copy here was free to contract differently.
        float y = __builtin_fmaf(weight[c * 4], s2,
                  __builtin_fmaf(weight[c * 4 + 1], s1,
                  __builtin_fmaf(weight[c * 4 + 2], s0, weight[c * 4 + 3] * x)));

        float result = y / (1.0f + expf(-y));

        state[c * 3 + 2] = s1;
        state[c * 3 + 1] = s0;
        state[c * 3] = x;

        if (c < k_dim) {
            q_out[b * k_dim + c] = result;
        } else if (c < 2 * k_dim) {
            k_out[b * k_dim + (c - k_dim)] = result;
        } else {
            v_out[b * v_dim + (c - 2 * k_dim)] = result;
        }
    }
}

// fused_qk_l2_norm_scale_f32, one (row, key head), one wave.
static __device__ void d1_qk_l2_head(const D1Params& p, int b, int h) {
    const int n_heads = p.n_k_heads;
    const int head_dim = HD;
    const float q_scale = p.q_scale;
    const float eps = p.eps;
    const int tid = threadIdx.x & 31;

    const long long batch_off = (long long)b * n_heads * head_dim;
    float* q_head = p.q_raw + batch_off + h * head_dim;
    float* k_head = p.k_raw + batch_off + h * head_dim;

    float q_sq = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        q_sq += q_head[i] * q_head[i];
    for (int o = 16; o > 0; o >>= 1)
        q_sq += __shfl_xor(q_sq, o);
    float q_inv_norm = rsqrtf(q_sq + eps);
    for (int i = tid; i < head_dim; i += 32)
        q_head[i] *= q_inv_norm;
    for (int i = tid; i < head_dim; i += 32)
        q_head[i] *= q_scale;

    float k_sq = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        k_sq += k_head[i] * k_head[i];
    for (int o = 16; o > 0; o >>= 1)
        k_sq += __shfl_xor(k_sq, o);
    float k_inv_norm = rsqrtf(k_sq + eps);
    for (int i = tid; i < head_dim; i += 32)
        k_head[i] *= k_inv_norm;
}

// One row: the q/k l2-norm of every key head, then repeat_interleave_qk_f32_batched.
static __device__ void d1_qk_row(const D1Params& p, int b) {
    for (int h = threadIdx.x >> 5; h < p.n_k_heads; h += D1_WAVES) d1_qk_l2_head(p, b, h);
    __syncthreads();
    const int n_key_heads = p.n_k_heads;
    const int ratio = p.n_v_heads / p.n_k_heads;
    const int head_dim = HD;
    int total_out = n_key_heads * ratio * head_dim;
    for (int idx = threadIdx.x; idx < total_out; idx += blockDim.x) {
        int d = idx % head_dim;
        int kh_r = idx / head_dim;
        int kh = kh_r / ratio;

        int src_stride = n_key_heads * head_dim;
        int dst_stride = total_out;

        int src_off = (long long)b * src_stride + kh * head_dim + d;
        int dst_off = (long long)b * dst_stride + idx;

        p.q[dst_off] = p.q_raw[src_off];
        p.k[dst_off] = p.k_raw[src_off];
    }
}

__device__ __forceinline__ _Float16 d1_f32_to_f16_dither(float x, unsigned int idx) {
    unsigned int u = __float_as_uint(x);
    unsigned int r = u ^ (idx * 2654435761u);
    r ^= r >> 15; r *= 2246822519u; r ^= r >> 13; r *= 3266489917u; r ^= r >> 16;
    unsigned int dithered = (u + (r & 0x1FFFu)) & 0xFFFFE000u;
    return (_Float16)__uint_as_float(dithered);
}

// gated_delta_net_f16_routed_batch_seq, one (head, S-row tile, session), one wave.
static __device__ void d1_gdn(const D1Params& p, int h, int tile, int session_idx, float* S_tile) {
// The chain kernel pins contraction (see its header); so must its copy.
#pragma clang fp contract(on)
    const float* __restrict__ q = p.q;
    const float* __restrict__ k = p.k;
    const float* __restrict__ v = p.v;
    const float* __restrict__ gate = p.proj[3].y;
    const float* __restrict__ beta = p.proj[2].y;
    const int* __restrict__ row_session_indices = p.row_session;
    float* __restrict__ output = p.attn_out;
    const int batch_rows = p.n;
    const int n_heads = p.n_v_heads;
    const int stochastic_round = p.stochastic_round;

    const int tid = threadIdx.x & 31;
    const int g = tid >> 3;
    const int l = tid & 7;
    const int col = l * 16;
    const int row_in_tile = g;
    const int stride = n_heads * HD;
    const int row_start = tile * TILE_ROWS;

    _Float16* state = reinterpret_cast<_Float16*>(
        static_cast<uintptr_t>(p.s_ptrs[(size_t)session_idx * p.ptr_stride + p.layer])
    );
    _Float16* S_global = state + h * HD * HD + row_start * HD;
    for (int i = tid; i < TILE_ROWS * HD; i += 32) {
        S_tile[i] = (float)S_global[i];
    }

    for (int b0 = 0; b0 < batch_rows; b0 += 32) {
    const int bl = b0 + tid;
    unsigned long long mine =
        __ballot(bl < batch_rows && row_session_indices[bl] == session_idx);
    while (mine) {
        const int b = b0 + (__ffsll((long long)mine) - 1);
        mine &= mine - 1;

        const float* q_t = q + b * stride + h * HD;
        const float* k_t = k + b * stride + h * HD;
        const float* v_t = v + b * stride + h * HD;
        float alpha = expf(gate[b * n_heads + h]);
        float beta_v = beta[b * n_heads + h];

        float sreg[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) sreg[i] = S_tile[row_in_tile * HD + col + i];

        float kk[16], qq[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) { kk[i] = k_t[col + i]; qq[i] = q_t[col + i]; }

        float part[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) part[i] = sreg[i] * kk[i];
        #pragma unroll
        for (int w = 8; w > 0; w >>= 1)
            #pragma unroll
            for (int i = 0; i < w; i++) part[i] += part[i + w];
        float kv = part[0];
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) kv += __shfl_down(kv, o, 8);
        kv = __shfl(kv, 0, 8);

        const int row = row_start + row_in_tile;
        const float delta = (v_t[row] - alpha * kv) * beta_v;

        #pragma unroll
        for (int i = 0; i < 16; i++) sreg[i] = alpha * sreg[i] + kk[i] * delta;
        #pragma unroll
        for (int i = 0; i < 16; i++) part[i] = sreg[i] * qq[i];
        #pragma unroll
        for (int w = 8; w > 0; w >>= 1)
            #pragma unroll
            for (int i = 0; i < w; i++) part[i] += part[i + w];
        float out_v = part[0];
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) out_v += __shfl_down(out_v, o, 8);

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const unsigned int nidx =
                (unsigned int)(row_start * HD + row_in_tile * HD + col + i) ^ ((unsigned int)h << 19);
            const _Float16 nv =
                stochastic_round ? d1_f32_to_f16_dither(sreg[i], nidx) : (_Float16)sreg[i];
            sreg[i] = (float)nv;
            S_tile[row_in_tile * HD + col + i] = sreg[i];
        }

        if (l == 0)
            output[b * stride + h * HD + row] = out_v;
    }
    }

    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_global[i] = (_Float16)S_tile[i];
}

// gated_norm_f32, one (row, value head), one wave.
static __device__ void d1_gated_norm_head(const D1Params& p, int b, int h) {
    const int n_heads = p.n_v_heads;
    const int head_dim = HD;
    const float eps = p.eps;
    const int tid = threadIdx.x & 31;

    const long long batch_off = (long long)b * n_heads * head_dim;
    const float* xh = p.attn_out + batch_off + h * head_dim;
    const float* zh = p.proj[1].y + batch_off + h * head_dim;
    float* oh = p.normed + batch_off + h * head_dim;
    const float* weight = p.norm_weight;

    float sq_sum = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        sq_sum += xh[i] * xh[i];
    for (int o = 16; o > 0; o >>= 1)
        sq_sum += __shfl_xor(sq_sum, o);
    float inv_rms = rsqrtf(sq_sum / (float)head_dim + eps);

    for (int i = tid; i < head_dim; i += 32) {
        float normed = xh[i] * inv_rms * weight[i];
        float z_val = zh[i];
        float silu_z = z_val / (1.0f + expf(-z_val));
        oh[i] = normed * silu_z;
    }
}

// One row: the gated norm of every value head, then wo's rotate + quantize.
static __device__ void d1_gn_row(const D1Params& p, int b) {
    for (int h = threadIdx.x >> 5; h < p.n_v_heads; h += D1_WAVES) d1_gated_norm_head(p, b, h);
    __syncthreads();
    const int ng = p.proj[4].k / 256;
    for (int g = threadIdx.x >> 5; g < ng; g += D1_WAVES) d1_rotq_group(p, p.proj[4], p.normed, b, g);
}

// ── the scheduler ──────────────────────────────────────────────────────────

// Item order: the beta/alpha weights first so the gates can run while the
// large projections stream; wz before the conv, which waits on wqkv alone.
__device__ __constant__ int d1_order[D1_NOPS] = {
    OP_NORM, OP_ROTQ + 2, OP_ROTQ + 3, OP_ROTQ + 0, OP_ROTQ + 1,
    OP_GEMV + 2, OP_GEMV + 3, OP_GATE, OP_GEMV + 0, OP_GEMV + 1,
    OP_CONV, OP_QK, OP_GDN, OP_GN, OP_GEMV + 4,
};

static __device__ int d1_deps(int op, int* d) {
    if (op >= OP_ROTQ && op < OP_GEMV) { d[0] = OP_NORM; return 1; }
    if (op >= OP_GEMV && op < OP_GEMV + 4) { d[0] = OP_ROTQ + (op - OP_GEMV); return 1; }
    if (op == OP_GEMV + 4) { d[0] = OP_GN; return 1; }
    if (op == OP_GATE) { d[0] = OP_GEMV + 2; d[1] = OP_GEMV + 3; return 2; }
    if (op == OP_CONV) { d[0] = OP_GEMV + 0; return 1; }
    if (op == OP_QK) { d[0] = OP_CONV; return 1; }
    if (op == OP_GDN) { d[0] = OP_QK; d[1] = OP_GATE; return 2; }
    if (op == OP_GN) { d[0] = OP_GDN; d[1] = OP_GEMV + 1; return 2; }
    return 0;
}

extern "C" __launch_bounds__(256) __global__ void d1_dn_moe_layer(D1Params p) {
    __shared__ __attribute__((aligned(16))) int8_t lds[D1_LDS];
    __shared__ float wave_sums[D1_WAVES];
    __shared__ int s_item;
    __shared__ int s_count[D1_NOPS];
    __shared__ int s_start[D1_NOPS + 1];   // by position in d1_order
    const int tid = threadIdx.x;
    const int gdn_per_session = p.n_v_heads * (HD / TILE_ROWS / D1_WAVES);

    if (tid == 0) {
        for (int o = 0; o < D1_NOPS; ++o) {
            int c;
            if (o < OP_GEMV) c = p.n;   // norm, rotate + quantize: a row
            else if (o < OP_GATE) {
                const int total_blocks = (p.proj[o - OP_GEMV].m + D1_RW - 1) / D1_RW;
                c = (total_blocks + D1_WAVES - 1) / D1_WAVES;
            } else if (o == OP_GATE) c = 1;
            else if (o == OP_CONV) c = (2 * p.k_dim + p.v_dim + 255) / 256 * p.sessions;
            else if (o == OP_GDN) c = gdn_per_session * p.sessions;
            else c = p.n;              // q/k, gated norm: a row
            s_count[o] = c;
        }
        s_start[0] = 0;
        for (int k = 0; k < D1_NOPS; ++k) s_start[k + 1] = s_start[k] + s_count[d1_order[k]];
    }
    __syncthreads();
    const int total = s_start[D1_NOPS];
    const int conv_blocks = (2 * p.k_dim + p.v_dim + 255) / 256;

    for (;;) {
        if (tid == 0) s_item = atomicAdd(&p.counters[0], 1);
        __syncthreads();
        const int item = s_item;
        if (item >= total) break;
        int pos = 0;
        while (item >= s_start[pos + 1]) ++pos;
        const int op = d1_order[pos];
        const int local = item - s_start[pos];

        // Wait for this op's prerequisites.
        if (tid == 0) {
            int deps[2];
            const int nd = d1_deps(op, deps);
            for (int d = 0; d < nd; ++d) {
                while (__hip_atomic_load(&p.counters[1 + deps[d]], __ATOMIC_ACQUIRE,
                                         __HIP_MEMORY_SCOPE_AGENT) < s_count[deps[d]])
                    __builtin_amdgcn_s_sleep(1);
            }
        }
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

        if (op == OP_NORM) {
            d1_rmsnorm_row(p, local, wave_sums);
        } else if (op < OP_GEMV) {
            const D1Proj& pj = p.proj[op - OP_ROTQ];
            const int ng = pj.k / 256;
            for (int g = tid >> 5; g < ng; g += D1_WAVES) d1_rotq_group(p, pj, p.xn, local, g);
        } else if (op < OP_GATE) {
            d1_multicol_item(p.proj[op - OP_GEMV], local, lds);
        } else if (op == OP_GATE) {
            d1_gate(p);
        } else if (op == OP_CONV) {
            d1_conv(p, local % conv_blocks, local / conv_blocks);
        } else if (op == OP_QK) {
            d1_qk_row(p, local);
        } else if (op == OP_GDN) {
            const int s = local / gdn_per_session;
            const int rem = local - s * gdn_per_session;
            const int per_head = HD / TILE_ROWS / D1_WAVES;
            const int h = rem / per_head;
            const int tile = (rem - h * per_head) * D1_WAVES + (tid >> 5);
            d1_gdn(p, h, tile, s, reinterpret_cast<float*>(lds) + (tid >> 5) * TILE_ROWS * HD);
        } else {
            d1_gn_row(p, local);
        }

        // Publish: every thread's writes before the barrier, then one release.
        __threadfence();
        __syncthreads();
        if (tid == 0)
            __hip_atomic_fetch_add(&p.counters[1 + op], 1, __ATOMIC_RELEASE,
                                   __HIP_MEMORY_SCOPE_AGENT);
    }

    // The last workgroup out zeroes the counters for the next launch: every
    // other one has left the loop and touches them no more.
    if (tid == 0 && atomicAdd(&p.counters[1 + D1_NOPS], 1) == (int)gridDim.x - 1) {
        for (int c = 0; c < 2 + D1_NOPS; ++c) p.counters[c] = 0;
    }
}
