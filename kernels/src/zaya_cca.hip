// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// ZAYA1 CCA + EDA/MoD custom elementwise kernels. All are register-only
// (NO __shared__/LDS — gfx1103 wedge hazard): one thread per output element,
// global reads only. Faithful to crates/hipfire-arch-zaya/src/cpu.rs.

#include <hip/hip_runtime.h>

// Embedding gather: out[t, i] = embed[ids[t]*hidden + i]. n = s*hidden.
// NOTE: `ids[t] * hidden` is int32 arithmetic, so this is bounded by
// vocab * hidden < 2^31. ZAYA1-8B sits at 262272 * 2048 = 537M, only 4x under
// that ceiling — a 4x larger embedding table overflows into a wild read rather
// than failing. Widen both this index and `n` to int64 before reusing it for a
// bigger vocabulary or hidden size.
extern "C" __global__ void zaya_embed_gather_f32(
    float* __restrict__ out, const float* __restrict__ embed,
    const int* __restrict__ ids, int hidden, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int t = i / hidden;
    int c = i % hidden;
    out[i] = embed[ids[t] * hidden + c];
}

// ── streamed-calibration MoE row plumbing ────────────────────────────────────
// A position slice routes its rows to different experts, so each expert's rows
// have to be made contiguous for one GEMM and then put back. These three run
// once per expert group instead of once per row, which is the whole point:
// at a slice width of 128 the per-row form issued 3 launches per row against
// ~2 GEMMs per expert.

// Gather whole rows: `out[r, c] = src[idx[r], c]`.
extern "C" __global__ void zaya_gather_rows_f32(
    float* __restrict__ out, const float* __restrict__ src,
    const int* __restrict__ idx, int width, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int r = i / width;
    int c = i - r * width;
    out[i] = src[idx[r] * width + c];
}

// Batched SwiGLU over the INTERLEAVED `[rows, 2*inter]` result of a gate_up
// GEMM: `out[r, c] = silu(gate_up[r, c]) * gate_up[r, inter + c]`. The gate and
// up halves are interleaved per row (that is the weight's layout), so the flat
// `silu_mul_f32` cannot span the batch — hence the explicit row stride here.
// Uses `expf`, matching `silu_mul_f32` exactly so the batched form stays
// bit-identical to the per-row one.
extern "C" __global__ void zaya_silu_mul_gate_up_f32(
    float* __restrict__ out, const float* __restrict__ gate_up, int inter, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int r = i / inter;
    int c = i - r * inter;
    const float* row = &gate_up[r * 2 * inter];
    float v = row[c];
    out[i] = (v / (1.0f + expf(-v))) * row[inter + c];
}

// Scatter-accumulate whole rows with a per-row scale:
//   `dst[idx[r], c] += scale[r] * src[r, c]`
// Routing is top-1, so each row of the slice belongs to exactly one expert
// group and every destination row is written by exactly one source row — no
// atomics needed. Matches `scaled_add_inplace_cpu_scalar_f32`'s `y += c * x`.
extern "C" __global__ void zaya_scatter_scaled_add_f32(
    float* __restrict__ dst, const float* __restrict__ src,
    const int* __restrict__ idx, const float* __restrict__ scale,
    int width, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int r = i / width;
    int c = i - r * width;
    dst[idx[r] * width + c] += scale[r] * src[i];
}

// Broadcast bias add: x[i] += bias[i % d]. n = s*d.
extern "C" __global__ void zaya_bias_add_f32(
    float* __restrict__ x, const float* __restrict__ bias, int d, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += bias[i % d];
}

// Global input residual affine on the embeddings: out[i] = (x[i]+bias[c])*scale[c],
// c = i % d. n = seq*d.
extern "C" __global__ void zaya_affine_input_f32(
    float* __restrict__ out, const float* __restrict__ x,
    const float* __restrict__ scale, const float* __restrict__ bias,
    int d, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i % d;
    out[i] = (x[i] + bias[c]) * scale[c];
}

// Learned residual merge: out[i] = (h[i]+hb[c])*hs[c] + (res[i]+rb[c])*rs[c].
extern "C" __global__ void zaya_affine_residual_f32(
    float* __restrict__ out, const float* __restrict__ h, const float* __restrict__ res,
    const float* __restrict__ hs, const float* __restrict__ hb,
    const float* __restrict__ rs, const float* __restrict__ rb,
    int d, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i % d;
    out[i] = (h[i] + hb[c]) * hs[c] + (res[i] + rb[c]) * rs[c];
}

// Scatter q[s,q_dim] and k[s,k_dim] into a left-padded channel-major stream
// [conv_ch, pad+s] (pad-region pre-zeroed by the caller). conv_ch = q_dim+k_dim.
// One thread per (t, channel) over conv_ch*s.
extern "C" __global__ void zaya_qk_stream_f32(
    float* __restrict__ stream, const float* __restrict__ q, const float* __restrict__ k,
    int s, int q_dim, int k_dim, int pad
) {
    int conv_ch = q_dim + k_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= conv_ch * s) return;
    int c = idx / s;
    int t = idx % s;
    int plen = pad + s;
    float v = (c < q_dim) ? q[t * q_dim + c] : k[t * k_dim + (c - q_dim)];
    stream[c * plen + pad + t] = v;
}

// Generic VALID causal grouped conv1d, channel-major. in: [channels, in_len];
// out: [channels, out_len] with out_len = in_len - kernel + 1. weight layout
// [out_channels, in_per_group, kernel] row-major. in==out==channels here so
// in_per_group == out_per_group == channels/groups. Depthwise == groups==channels.
extern "C" __global__ void zaya_conv1d_valid_f32(
    float* __restrict__ out, const float* __restrict__ in,
    const float* __restrict__ weight, const float* __restrict__ bias,
    int channels, int groups, int kernel, int in_len, int out_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= channels * out_len) return;
    int c = idx / out_len;
    int t = idx % out_len;
    int per_group = channels / groups;          // in_per_group == out_per_group
    int base = (c / per_group) * per_group;      // first input channel of this group
    float acc = bias[c];
    for (int j = 0; j < per_group; ++j) {
        const float* wr = &weight[(c * per_group + j) * kernel];
        const float* ir = &in[(base + j) * in_len + t];
        for (int kk = 0; kk < kernel; ++kk) {
            acc += wr[kk] * ir[kk];
        }
    }
    out[c * out_len + t] = acc;
}

// q/k residual paths. query_residual[t,head,d] = (q[t,head,d] + k[t,head/groups,d]) * 0.5;
// key_residual[t,kh,d] = mean over the `groups` q-heads of query_residual.
// Launch over s*nq*hd (writes query_residual) then s*nkv*hd (writes key_residual)
// via two calls distinguished by `mode` (0 = query, 1 = key).
extern "C" __global__ void zaya_qk_residual_f32(
    float* __restrict__ query_res, float* __restrict__ key_res,
    const float* __restrict__ q, const float* __restrict__ k,
    int s, int nq, int nkv, int hd, int mode
) {
    int groups = nq / nkv;
    if (mode == 0) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= s * nq * hd) return;
        int d = idx % hd;
        int head = (idx / hd) % nq;
        int t = idx / (hd * nq);
        int kh = head / groups;
        float qv = q[t * (nq * hd) + head * hd + d];
        float kv = k[t * (nkv * hd) + kh * hd + d];
        query_res[(t * nq + head) * hd + d] = (qv + kv) * 0.5f;
    } else {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= s * nkv * hd) return;
        int d = idx % hd;
        int kh = (idx / hd) % nkv;
        int t = idx / (hd * nkv);
        float acc = 0.0f;
        for (int g = 0; g < groups; ++g) {
            acc += query_res[(t * nq + kh * groups + g) * hd + d];
        }
        key_res[(t * nkv + kh) * hd + d] = acc / (float)groups;
    }
}

// Add conv output (channel-major [conv_ch, s]) back to the residual paths and
// emit head-major q [s,nq,hd] and k [s,nkv,hd]. mode 0 = query, 1 = key.
extern "C" __global__ void zaya_add_conv_residual_f32(
    float* __restrict__ out, const float* __restrict__ conv, const float* __restrict__ res,
    int s, int heads, int hd, int q_dim, int mode
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= s * heads * hd) return;
    int d = idx % hd;
    int head = (idx / hd) % heads;
    int t = idx / (hd * heads);
    int conv_ch = (mode == 0) ? head * hd + d : q_dim + head * hd + d;
    int conv_seq = s; // conv stored [conv_ch, s]
    out[(t * heads + head) * hd + d] = conv[conv_ch * conv_seq + t] + res[(t * heads + head) * hd + d];
}

// Compositional value: head 0 = current-token v, head 1 = previous-token delayed v.
// value [s, nkv=2, hd]; v_cur/v_del [s, hd].
extern "C" __global__ void zaya_value_compose_f32(
    float* __restrict__ value, const float* __restrict__ v_cur, const float* __restrict__ v_del,
    int s, int nkv, int hd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= s * hd) return;
    int d = idx % hd;
    int t = idx / hd;
    value[(t * nkv + 0) * hd + d] = v_cur[t * hd + d];
    value[(t * nkv + 1) * hd + d] = (t == 0) ? 0.0f : v_del[(t - 1) * hd + d];
}

// L2-normalize each [head, hd] row to `scale` (= sqrt(head_dim)); if `temp` is
// non-null, multiply each head by temp[head] (key path). No LDS: each thread
// owns one (t, head) row and loops hd.
extern "C" __global__ void zaya_qk_l2norm_temp_f32(
    float* __restrict__ x, const float* __restrict__ temp,
    int s, int heads, int hd, float scale, float eps, int has_temp
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= s * heads) return;
    float* v = &x[row * hd];
    float nrm = 0.0f;
    for (int d = 0; d < hd; ++d) nrm += v[d] * v[d];
    float inv = scale / fmaxf(sqrtf(nrm), eps);
    float tmul = 1.0f;
    if (has_temp != 0) {
        int head = row % heads;
        tmul = temp[head];
    }
    for (int d = 0; d < hd; ++d) v[d] = v[d] * inv * tmul;
}

// Exact erf-GELU (A&S 7.1.26), in place. n elements.
extern "C" __global__ void zaya_gelu_exact_f32(float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float a = v * 0.70710678118654752440f; // x/sqrt(2)
    float sign = (a < 0.0f) ? -1.0f : 1.0f;
    float ax = fabsf(a);
    float tt = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * tt - 1.453152027f) * tt) + 1.421413741f) * tt - 0.284496736f) * tt + 0.254829592f) * tt * expf(-ax * ax);
    float erf = sign * y;
    x[i] = 0.5f * v * (1.0f + erf);
}

// Partial rotary (half-split), position = pos_base + token index. x [s,heads,hd];
// only first n_rot dims rotate. One thread per (token, head). pos_base=0 for
// prefill; pos_base=current position for single-token decode.
extern "C" __global__ void zaya_rope_partial_f32(
    float* __restrict__ x, int s, int heads, int hd, int n_rot, float theta, int pos_base
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= s * heads) return;
    int head = idx % heads;
    int t = (idx / heads) + pos_base;
    (void)head;
    float* v = &x[idx * hd];
    int half = n_rot / 2;
    for (int d = 0; d < half; ++d) {
        float inv_freq = powf(theta, -(2.0f * (float)d) / (float)n_rot);
        float ang = (float)t * inv_freq;
        float c = cosf(ang), sn = sinf(ang);
        float a = v[d];
        float b = v[d + half];
        v[d] = a * c - b * sn;
        v[d + half] = b * c + a * sn;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Fused decode-glue kernels: each combines a pair/triple of the tiny per-block
// elementwise ops above into ONE launch. Single-token decode issues ~46
// kernels/block × 40 blocks ≈ 1840 launches/token and is dispatch-overhead
// bound (~8µs/dispatch), so halving the glue-op launch count is a direct win.
// All are behaviourally identical to the singles they replace; s-general.
// ─────────────────────────────────────────────────────────────────────────

// erf-GELU scalar (A&S 7.1.26) — shared by the fused bias+gelu kernel.
__device__ __forceinline__ float zaya_gelu_scalar(float v) {
    float a = v * 0.70710678118654752440f;
    float sign = (a < 0.0f) ? -1.0f : 1.0f;
    float ax = fabsf(a);
    float tt = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * tt - 1.453152027f) * tt) + 1.421413741f) * tt - 0.284496736f) * tt + 0.254829592f) * tt * expf(-ax * ax);
    return 0.5f * v * (1.0f + sign * y);
}

// Fused broadcast bias-add + exact GELU, in place: x[i] = gelu(x[i] + bias[i%d]).
// Replaces zaya_bias_add_f32 + zaya_gelu_exact_f32 (router fc1/fc2).
extern "C" __global__ void zaya_bias_gelu_f32(
    float* __restrict__ x, const float* __restrict__ bias, int d, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = zaya_gelu_scalar(x[i] + bias[i % d]);
}

// Fused partial-RoPE over query AND key in one launch. Threads [0, s*nq) rotate
// q [s,nq,hd]; [s*nq, s*nq+s*nkv) rotate k [s,nkv,hd]. Identical math to
// zaya_rope_partial_f32; replaces the two per-block calls.
extern "C" __global__ void zaya_rope_partial_qk_f32(
    float* __restrict__ q, float* __restrict__ k,
    int s, int nq, int nkv, int hd, int n_rot, float theta, int pos_base
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= s * (nq + nkv)) return;
    float* x;
    int local;
    if (idx < s * nq) { x = q; local = idx; }
    else              { x = k; local = idx - s * nq; }
    int heads = (idx < s * nq) ? nq : nkv;
    int t = (local / heads) + pos_base;
    float* v = &x[local * hd];
    int half = n_rot / 2;
    for (int d = 0; d < half; ++d) {
        float inv_freq = powf(theta, -(2.0f * (float)d) / (float)n_rot);
        float ang = (float)t * inv_freq;
        float c = cosf(ang), sn = sinf(ang);
        float a = v[d];
        float b = v[d + half];
        v[d] = a * c - b * sn;
        v[d + half] = b * c + a * sn;
    }
}

// Device-position variant of zaya_rope_partial_qk_f32: reads the base position
// from `pos_buf[0]` (device i32) instead of an immediate. Capture-safe — the
// per-token position lives in a device buffer the host updates between graph
// replays, so nothing position-dependent is baked into the graph.
extern "C" __global__ void zaya_rope_partial_qk_posbuf_f32(
    float* __restrict__ q, float* __restrict__ k, const int* __restrict__ pos_buf,
    int s, int nq, int nkv, int hd, int n_rot, float theta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= s * (nq + nkv)) return;
    int pos_base = pos_buf[0];
    float* x;
    int local;
    if (idx < s * nq) { x = q; local = idx; }
    else              { x = k; local = idx - s * nq; }
    int heads = (idx < s * nq) ? nq : nkv;
    int t = (local / heads) + pos_base;
    float* v = &x[local * hd];
    int half = n_rot / 2;
    for (int d = 0; d < half; ++d) {
        float inv_freq = powf(theta, -(2.0f * (float)d) / (float)n_rot);
        float ang = (float)t * inv_freq;
        float c = cosf(ang), sn = sinf(ang);
        float a = v[d];
        float b = v[d + half];
        v[d] = a * c - b * sn;
        v[d + half] = b * c + a * sn;
    }
}

// Fused L2-norm+scale over query (no temp) AND key (per-head temp) in one launch.
// Threads [0, s*nq) do q; [s*nq, ...) do k with tmul = temp[kh]. Replaces the two
// per-block zaya_qk_l2norm_temp_f32 calls.
extern "C" __global__ void zaya_qk_l2norm_qk_f32(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ temp,
    int s, int nq, int nkv, int hd, float scale, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= s * (nq + nkv)) return;
    float* x;
    float tmul = 1.0f;
    if (idx < s * nq) { x = q; }
    else {
        int local = idx - s * nq;
        int kh = local % nkv;
        x = k;
        tmul = temp[kh];
        idx = local; // reuse for row addressing below
    }
    float* v = &x[idx * hd];
    float nrm = 0.0f;
    for (int d = 0; d < hd; ++d) nrm += v[d] * v[d];
    float inv = scale / fmaxf(sqrtf(nrm), eps);
    for (int d = 0; d < hd; ++d) v[d] = v[d] * inv * tmul;
}

// Fused add-conv-residual over query AND key in one launch (both modes of
// zaya_add_conv_residual_f32). conv is channel-major [conv_ch, s]; q uses conv
// channels [head*hd+d], k uses [q_dim + head*hd+d].
extern "C" __global__ void zaya_add_conv_residual_qk_f32(
    float* __restrict__ query, float* __restrict__ key, const float* __restrict__ conv,
    const float* __restrict__ qres, const float* __restrict__ kres,
    int s, int nq, int nkv, int hd, int q_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nq_elems = s * nq * hd;
    if (idx >= nq_elems + s * nkv * hd) return;
    float* out; const float* res; int heads; int base; int j;
    if (idx < nq_elems) { out = query; res = qres; heads = nq; base = 0;     j = idx; }
    else                { out = key;   res = kres; heads = nkv; base = q_dim; j = idx - nq_elems; }
    int d = j % hd;
    int head = (j / hd) % heads;
    int t = j / (hd * heads);
    int conv_ch = base + head * hd + d;
    out[(t * heads + head) * hd + d] = conv[conv_ch * s + t] + res[(t * heads + head) * hd + d];
}

// Fused single-token value assembly + delayed-value advance (decode only):
//   value[0:vh]     = v_cur              (head 0 = current)
//   value[vh:2vh]   = delayed_v (OLD)    (head 1 = previous token's delayed v)
//   delayed_v       = v_del  (advance)
// Each thread owns one dim d and reads OLD delayed_v[d] before overwriting it, so
// the three writes are race-free. Replaces 3× zaya_write_at_f32.
extern "C" __global__ void zaya_value_assemble_decode_f32(
    float* __restrict__ value, const float* __restrict__ v_cur,
    float* __restrict__ delayed_v, const float* __restrict__ v_del, int vh
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= vh) return;
    float old_del = delayed_v[d];
    value[d] = v_cur[d];
    value[vh + d] = old_del;
    delayed_v[d] = v_del[d];
}

// Fused decode pre-conv prep (single workgroup): qk_residual (both modes) + the
// channel-major qk-stream column, in ONE launch. query_res[head,d]=(q+k[head/g])*0.5;
// key_res[kh,d]=mean_g query_res[kh*g+g',d]; cur_qk[c]=c<q_dim?q[c]:k[c-q_dim].
// Replaces 2× zaya_qk_residual_f32 + zaya_qk_stream_f32 (3 launches → 1). s=1 decode.
extern "C" __global__ void zaya_qk_prep_decode_f32(
    const float* __restrict__ q, const float* __restrict__ k,
    float* __restrict__ query_res, float* __restrict__ key_res, float* __restrict__ cur_qk,
    int nq, int nkv, int hd, int q_dim, int k_dim
) {
    const int groups = nq / nkv;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    // channel-major current qk column (independent).
    for (int c = tid; c < q_dim + k_dim; c += nthreads)
        cur_qk[c] = (c < q_dim) ? q[c] : k[c - q_dim];
    // query_res.
    for (int i = tid; i < nq * hd; i += nthreads) {
        const int d = i % hd;
        const int head = i / hd;
        const int kh = head / groups;
        query_res[i] = (q[head * hd + d] + k[kh * hd + d]) * 0.5f;
    }
    __syncthreads();
    // key_res = mean over the `groups` q-heads of query_res.
    for (int i = tid; i < nkv * hd; i += nthreads) {
        const int d = i % hd;
        const int kh = i / hd;
        float acc = 0.0f;
        for (int g = 0; g < groups; ++g) acc += query_res[(kh * groups + g) * hd + d];
        key_res[i] = acc / (float)groups;
    }
}

// Fused router-input prep (decode): rhid += bias; if has_eda, rhid += prev*scale
// (Exponential Depth Averaging); then prev := rhid (save next-block EDA state).
// `prev` (router_state) is read-old then written-new per element (race-free).
// Replaces zaya_bias_add_f32 + zaya_eda_add_f32 + zaya_write_at_f32.
extern "C" __global__ void zaya_router_prep_f32(
    float* __restrict__ rhid, const float* __restrict__ bias,
    float* __restrict__ prev, const float* __restrict__ scale,
    int rh, int n, int has_eda
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i % rh;
    float v = rhid[i] + bias[c];
    if (has_eda != 0) v += prev[i] * scale[c];
    rhid[i] = v;
    prev[i] = v;
}

// Simple GQA causal attention (bring-up; O(s^2) per row, NO LDS). q [s,nq,hd],
// k/v [s,nkv,hd] head-major; out [s,nq,hd]. scaling applied to scores. One
// thread per (query position i, head). For short prompts; flash path later.
extern "C" __global__ void zaya_gqa_attn_f32(
    float* __restrict__ out, const float* __restrict__ q,
    const float* __restrict__ k, const float* __restrict__ v,
    int s, int nq, int nkv, int hd, float scaling
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= s * nq) return;
    int head = idx % nq;
    int i = idx / nq;
    int groups = nq / nkv;
    int kh = head / groups;
    const float* qv = &q[(i * nq + head) * hd];
    // online softmax over j = 0..i
    float maxv = -1e30f;
    for (int j = 0; j <= i; ++j) {
        const float* kv = &k[(j * nkv + kh) * hd];
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) dot += qv[d] * kv[d];
        dot *= scaling;
        if (dot > maxv) maxv = dot;
    }
    float denom = 0.0f;
    float* o = &out[(i * nq + head) * hd];
    for (int d = 0; d < hd; ++d) o[d] = 0.0f;
    for (int j = 0; j <= i; ++j) {
        const float* kv = &k[(j * nkv + kh) * hd];
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) dot += qv[d] * kv[d];
        float p = expf(dot * scaling - maxv);
        denom += p;
        const float* vv = &v[(j * nkv + kh) * hd];
        for (int d = 0; d < hd; ++d) o[d] += p * vv[d];
    }
    float inv = 1.0f / denom;
    for (int d = 0; d < hd; ++d) o[d] *= inv;
}

// Strided row copy: dst[r*dst_stride + j] = src[r*src_stride + src_off + j],
// j in 0..len, r in 0..rows. (Used to seed the decode conv ring from the prefill
// stream's last `pad` columns.) One thread per (row, j).
extern "C" __global__ void zaya_strided_copy_f32(
    float* __restrict__ dst, const float* __restrict__ src,
    int rows, int dst_stride, int src_stride, int src_off, int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * len) return;
    int r = idx / len;
    int j = idx % len;
    dst[r * dst_stride + j] = src[r * src_stride + src_off + j];
}

// Decode conv window + ring update. Build window [conv_ch, pad+1] = [ring | cur]
// (channel-major) and advance the ring (drop oldest, push cur). One thread/channel.
extern "C" __global__ void zaya_conv_window_f32(
    float* __restrict__ window, float* __restrict__ ring, const float* __restrict__ cur,
    int conv_ch, int pad
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_ch) return;
    int wlen = pad + 1;
    for (int j = 0; j < pad; ++j) window[c * wlen + j] = ring[c * pad + j];
    window[c * wlen + pad] = cur[c];
    for (int j = 0; j + 1 < pad; ++j) ring[c * pad + j] = ring[c * pad + j + 1];
    ring[c * pad + (pad - 1)] = cur[c];
}

// Copy src[0..n] into dst[offset..offset+n] (KV-cache append). One thread/elem.
extern "C" __global__ void zaya_write_at_f32(
    float* __restrict__ dst, const float* __restrict__ src, int offset, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[offset + i] = src[i];
}

// Single-token GQA decode attention over the KV cache. q [nq,hd]; k_cache/v_cache
// [max_seq, nkv, hd]; out [nq,hd]; attends positions 0..=pos. One thread/(head).
extern "C" __global__ void zaya_gqa_decode_f32(
    float* __restrict__ out, const float* __restrict__ q,
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    int pos, int nq, int nkv, int hd, float scaling
) {
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= nq) return;
    int groups = nq / nkv;
    int kh = head / groups;
    const float* qv = &q[head * hd];
    int kvdim = nkv * hd;
    float maxv = -1e30f;
    for (int j = 0; j <= pos; ++j) {
        const float* kv = &k_cache[j * kvdim + kh * hd];
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) dot += qv[d] * kv[d];
        dot *= scaling;
        if (dot > maxv) maxv = dot;
    }
    float denom = 0.0f;
    float* o = &out[head * hd];
    for (int d = 0; d < hd; ++d) o[d] = 0.0f;
    for (int j = 0; j <= pos; ++j) {
        const float* kv = &k_cache[j * kvdim + kh * hd];
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) dot += qv[d] * kv[d];
        float p = expf(dot * scaling - maxv);
        denom += p;
        const float* vv = &v_cache[j * kvdim + kh * hd];
        for (int d = 0; d < hd; ++d) o[d] += p * vv[d];
    }
    float inv = 1.0f / denom;
    for (int d = 0; d < hd; ++d) o[d] *= inv;
}

// EDA cross-layer state add: router_hidden[t,i] += prev[t,i] * scale[i]. n = s*rh.
extern "C" __global__ void zaya_eda_add_f32(
    float* __restrict__ router_hidden, const float* __restrict__ prev,
    const float* __restrict__ scale, int rh, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i % rh;
    router_hidden[i] += prev[i] * scale[c];
}

// On-device top-1 router select for the EDA/MoD gate (decode, single token).
// Replaces the per-block `download_f32(rlogits)` + host softmax/argmax so the
// whole decode MoE path stays device-resident. Faithful to the host reference
// in cpu.rs / the previous gpu_decode host loop: softmax over `n_route` logits,
// then argmax over (prob + balancing_bias); the applied gate is the *unbiased*
// softmax prob of the winner. `n_route = n_exp + 1`; the last slot (`n_exp`) is
// the "null"/skip expert — selecting it yields gate 0 so the combine adds
// nothing (the indexed GEMV still runs a bounded alias expert, see the pointer
// table in gpu.rs). Single-thread: `n_route` is tiny (e.g. 17).
extern "C" __global__ void zaya_router_select_f32(
    const float* __restrict__ logits, // [n_route]
    const float* __restrict__ bias,   // [n_route] balancing biases
    int* __restrict__ sel_idx,        // [1] out: winning expert id in [0, n_route)
    float* __restrict__ sel_gate,     // [1] out: unbiased softmax gate (0 if null)
    int n_route
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float maxv = -3.0e38f;
    for (int e = 0; e < n_route; e++) maxv = fmaxf(maxv, logits[e]);
    float denom = 0.0f;
    for (int e = 0; e < n_route; e++) denom += expf(logits[e] - maxv);
    int best = 0;
    float bestv = -3.0e38f;
    float best_prob = 0.0f;
    for (int e = 0; e < n_route; e++) {
        float p = expf(logits[e] - maxv) / denom;
        float v = p + bias[e];
        if (v > bestv) {
            bestv = v;
            best = e;
            best_prob = p;
        }
    }
    const int n_exp = n_route - 1;
    sel_idx[0] = best;
    sel_gate[0] = (best == n_exp) ? 0.0f : best_prob;
}

// FWHT-256 in-place on LDS `buf[256]`, performed by wave0 (threads 0-31), 8
// elems/lane. Mirrors rotate_x_mq exactly: signs1 → local butterfly (strides 1,2,4)
// → ds_swizzle wave butterfly (strides 1..16) → ×0.0625 → signs2.
__device__ __forceinline__ void zaya_fwht256_lds(
    float* buf, const float* signs1, const float* signs2, int tid
) {
    if (tid < 32) {
        int d0 = tid * 8;
        float v0 = buf[d0] * signs1[d0], v1 = buf[d0+1] * signs1[d0+1];
        float v2 = buf[d0+2] * signs1[d0+2], v3 = buf[d0+3] * signs1[d0+3];
        float v4 = buf[d0+4] * signs1[d0+4], v5 = buf[d0+5] * signs1[d0+5];
        float v6 = buf[d0+6] * signs1[d0+6], v7 = buf[d0+7] * signs1[d0+7];
        float t;
        t=v0;v0=v0+v1;v1=t-v1; t=v2;v2=v2+v3;v3=t-v3; t=v4;v4=v4+v5;v5=t-v5; t=v6;v6=v6+v7;v7=t-v7;
        t=v0;v0=v0+v2;v2=t-v2; t=v1;v1=v1+v3;v3=t-v3; t=v4;v4=v4+v6;v6=t-v6; t=v5;v5=v5+v7;v7=t-v7;
        t=v0;v0=v0+v4;v4=t-v4; t=v1;v1=v1+v5;v5=t-v5; t=v2;v2=v2+v6;v6=t-v6; t=v3;v3=v3+v7;v7=t-v7;
        #define ZHB(v,pat,str) do{ float _p=__int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v),(pat))); if(tid&(str)){(v)=_p-(v);}else{(v)=(v)+_p;} }while(0)
        #define ZHB8(pat,str) ZHB(v0,pat,str);ZHB(v1,pat,str);ZHB(v2,pat,str);ZHB(v3,pat,str);ZHB(v4,pat,str);ZHB(v5,pat,str);ZHB(v6,pat,str);ZHB(v7,pat,str)
        ZHB8(0x041F,1); ZHB8(0x081F,2); ZHB8(0x101F,4); ZHB8(0x201F,8); ZHB8(0x401F,16);
        #undef ZHB8
        #undef ZHB
        const float s = 0.0625f;
        buf[d0]=v0*s*signs2[d0]; buf[d0+1]=v1*s*signs2[d0+1]; buf[d0+2]=v2*s*signs2[d0+2]; buf[d0+3]=v3*s*signs2[d0+3];
        buf[d0+4]=v4*s*signs2[d0+4]; buf[d0+5]=v5*s*signs2[d0+5]; buf[d0+6]=v6*s*signs2[d0+6]; buf[d0+7]=v7*s*signs2[d0+7];
    }
}

// Fused router MLP (decode, single workgroup, 256 threads): down_proj(Oq8, input
// pre-rotated pa_xrot) → prep(+bias,+EDA,save state) → rmsnorm → [FWHT→fc1(Oq8)→gelu]
// → [FWHT→fc2(Oq8)→gelu] → [FWHT→out(Oq8)] → select. Replaces ~9 launches with 1.
// All Oq8 weights are planar `[int8 M*K | f32 scales M*(K/256)]`, FwhtG256 basis
// (K=256 ⇒ 1 scale/row). Bit-mirrors the current op chain. rh<=256, n_route<=32.
extern "C" __global__ void zaya_router_mlp_fused(
    const float* __restrict__ pa_xrot,     // [h] rotated post-attn activation
    const signed char* __restrict__ dp_w, const float* __restrict__ dp_b,
    float* __restrict__ rstate, const float* __restrict__ escale,
    const float* __restrict__ rnorm_w,
    const signed char* __restrict__ fc1_w, const float* __restrict__ fc1_b,
    const signed char* __restrict__ fc2_w, const float* __restrict__ fc2_b,
    const signed char* __restrict__ out_w,
    const float* __restrict__ bbias,
    const float* __restrict__ signs1, const float* __restrict__ signs2,
    int* __restrict__ sel_idx, float* __restrict__ sel_gate,
    int h, int rh, int n_route, float eps, int has_eda
) {
    __shared__ float buf[256];
    __shared__ float red[8];
    __shared__ float logit[32];
    const int tid = threadIdx.x;

    // Phase 1: down_proj (thread m → rhid[m]) + prep. K=h, ng=h/256.
    if (tid < rh) {
        const int ng = h / 256;
        const signed char* wr = dp_w + (long long)tid * h;
        const float* ws = (const float*)(dp_w + (long long)rh * h) + tid * ng;
        float acc = 0.0f;
        for (int g = 0; g < ng; ++g) {
            float ss = 0.0f;
            for (int k = 0; k < 256; ++k) ss += (float)wr[g * 256 + k] * pa_xrot[g * 256 + k];
            acc += ss * ws[g];
        }
        float v = acc + dp_b[tid];
        if (has_eda) v += rstate[tid] * escale[tid];
        rstate[tid] = v;
        buf[tid] = v;
    }
    __syncthreads();

    // Phase 2: rmsnorm(buf).
    float sq = (tid < rh) ? buf[tid] * buf[tid] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_down(sq, o);
    if ((tid & 31) == 0) red[tid >> 5] = sq;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        const int nw = blockDim.x >> 5;
        for (int i = 0; i < nw; ++i) s += red[i];
        red[0] = rsqrtf(s / (float)rh + eps);
    }
    __syncthreads();
    const float rms = red[0];
    if (tid < rh) buf[tid] = buf[tid] * rms * rnorm_w[tid];
    __syncthreads();

    // Phase 3: FWHT → fc1 → gelu.
    zaya_fwht256_lds(buf, signs1, signs2, tid);
    __syncthreads();
    float a = 0.0f;
    if (tid < rh) {
        const signed char* wr = fc1_w + (long long)tid * rh;
        const float sc = *((const float*)(fc1_w + (long long)rh * rh) + tid);
        float ss = 0.0f;
        for (int k = 0; k < rh; ++k) ss += (float)wr[k] * buf[k];
        a = zaya_gelu_scalar(fc1_b[tid] + sc * ss);
    }
    __syncthreads();
    if (tid < rh) buf[tid] = a;
    __syncthreads();

    // Phase 4: FWHT → fc2 → gelu.
    zaya_fwht256_lds(buf, signs1, signs2, tid);
    __syncthreads();
    a = 0.0f;
    if (tid < rh) {
        const signed char* wr = fc2_w + (long long)tid * rh;
        const float sc = *((const float*)(fc2_w + (long long)rh * rh) + tid);
        float ss = 0.0f;
        for (int k = 0; k < rh; ++k) ss += (float)wr[k] * buf[k];
        a = zaya_gelu_scalar(fc2_b[tid] + sc * ss);
    }
    __syncthreads();
    if (tid < rh) buf[tid] = a;
    __syncthreads();

    // Phase 5: FWHT → out (n_route rows) → select.
    zaya_fwht256_lds(buf, signs1, signs2, tid);
    __syncthreads();
    if (tid < n_route) {
        const signed char* wr = out_w + (long long)tid * rh;
        const float sc = *((const float*)(out_w + (long long)n_route * rh) + tid);
        float ss = 0.0f;
        for (int k = 0; k < rh; ++k) ss += (float)wr[k] * buf[k];
        logit[tid] = sc * ss;
    }
    __syncthreads();
    if (tid == 0) {
        float maxv = -3.0e38f;
        for (int e = 0; e < n_route; ++e) maxv = fmaxf(maxv, logit[e]);
        float denom = 0.0f;
        for (int e = 0; e < n_route; ++e) denom += expf(logit[e] - maxv);
        int best = 0; float bestv = -3.0e38f, bp = 0.0f;
        for (int e = 0; e < n_route; ++e) {
            float p = expf(logit[e] - maxv) / denom;
            float vv = p + bbias[e];
            if (vv > bestv) { bestv = vv; best = e; bp = p; }
        }
        sel_idx[0] = best;
        sel_gate[0] = (best == n_route - 1) ? 0.0f : bp;
    }
}

// Indexed OQ8 (Oq8G256) top-1 expert GEMV over zaya's PLANAR `oq8_combined`
// buffer layout — distinct from the shared MoE kernels' interleaved 260B groups.
// Each expert buffer `A` is `[M*K signed int8 row-major | M*ng f32 scales]`,
// ng = K/256, scale[row][g] = scales[row*ng + g], dequant w = scale * int8.
// `expert_ptrs[sel_idx[0]]` selects the expert on-device (no host readback); the
// null slot aliases expert 0 and is gated to zero weight downstream. wave32, 8
// int8/lane, one workgroup per output row.
//
// CONTRACT: `x` MUST already be FWHT-rotated (Opus Quant weights live in the
// rotated basis) — the caller runs `rotate_x_mq` before this kernel. See
// kernels/AGENTS.md "Quant activation-basis contract".
//
// gate_up: fused — rows [0, M/2) → y_gate, [M/2, M) → y_up (M = 2*moe_int).
extern "C" __global__ void zaya_moe_gate_up_oq8_planar_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_route]
    const int* __restrict__ sel_idx,                    // [1]
    const float* __restrict__ x,                        // [K]
    float* __restrict__ y_gate,                         // [M/2]
    float* __restrict__ y_up,                           // [M/2]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x; // wave32
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[sel_idx[0]]);
    const signed char* __restrict__ w8 = reinterpret_cast<const signed char*>(A);
    const float* __restrict__ scales =
        reinterpret_cast<const float*>(A + (long long)M * K);
    const int ng = K / 256;
    const int boff = tid * 8; // 8 int8 / lane = two aligned int32 loads
    float acc = 0.0f;
    for (int g = 0; g < ng; g++) {
        const float sc = scales[row * ng + g];
        const signed char* wr = w8 + (long long)row * K + g * 256 + boff;
        const int w0 = __builtin_nontemporal_load((const int*)(wr));
        const int w1 = __builtin_nontemporal_load((const int*)(wr + 4));
        const float* xr = x + g * 256 + boff;
        float gsum = 0.0f;
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int q = (w0 >> (8 * n)) & 0xff;
            q = (q << 24) >> 24; // sign-extend int8
            gsum += (float)q * xr[n];
        }
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int q = (w1 >> (8 * n)) & 0xff;
            q = (q << 24) >> 24;
            gsum += (float)q * xr[4 + n];
        }
        acc += sc * gsum;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (tid == 0) {
        const int mi = M >> 1;
        if (row < mi) {
            y_gate[row] = acc;
        } else {
            y_up[row - mi] = acc;
        }
    }
}

// down projection: y[row] = expert_down(x), same planar layout. M = hidden.
extern "C" __global__ void zaya_moe_down_oq8_planar_indexed(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ sel_idx,
    const float* __restrict__ x, // [K = moe_int]
    float* __restrict__ y,       // [M = hidden]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[sel_idx[0]]);
    const signed char* __restrict__ w8 = reinterpret_cast<const signed char*>(A);
    const float* __restrict__ scales =
        reinterpret_cast<const float*>(A + (long long)M * K);
    const int ng = K / 256;
    const int boff = tid * 8; // 8 int8 / lane = two aligned int32 loads
    float acc = 0.0f;
    for (int g = 0; g < ng; g++) {
        const float sc = scales[row * ng + g];
        const signed char* wr = w8 + (long long)row * K + g * 256 + boff;
        const int w0 = __builtin_nontemporal_load((const int*)(wr));
        const int w1 = __builtin_nontemporal_load((const int*)(wr + 4));
        const float* xr = x + g * 256 + boff;
        float gsum = 0.0f;
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int q = (w0 >> (8 * n)) & 0xff;
            q = (q << 24) >> 24;
            gsum += (float)q * xr[n];
        }
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int q = (w1 >> (8 * n)) & 0xff;
            q = (q << 24) >> 24;
            gsum += (float)q * xr[4 + n];
        }
        acc += sc * gsum;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (tid == 0) y[row] = acc;
}

// ── LDS activation-staged variants (HIPFIRE_ZAYA_MOE_LDS) ──
// The one-warp-per-row indexed gemvs above are memory-LATENCY bound (~93% MemUnitBusy but
// only ~113 GB/s of ~211 peak, measured via rocprofv3). HALF the per-group global loads
// are the activation x, which EVERY one of the M row-waves re-reads. Here a block of
// blockDim/32 waves stages x[K] into LDS once (cooperative), then each wave reads x from
// LDS — removing x from the memory-unit request stream so the weight fetches get its full
// outstanding-request budget. Bit-identical math. (LDS now allowed: gfx1103 wedge fixed.)
extern "C" __global__ void zaya_moe_gate_up_oq8_planar_indexed_lds(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ sel_idx,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int M, int K
) {
    extern __shared__ float xs[]; // K floats, staged once per block
    const int tib = threadIdx.x;
    for (int i = tib; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int lane = tib & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (tib >> 5);
    if (row >= M) return;
    const char* __restrict__ A = reinterpret_cast<const char*>(expert_ptrs[sel_idx[0]]);
    const signed char* __restrict__ w8 = reinterpret_cast<const signed char*>(A);
    const float* __restrict__ scales = reinterpret_cast<const float*>(A + (long long)M * K);
    const int ng = K / 256;
    const int boff = lane * 8;
    float acc = 0.0f;
    for (int g = 0; g < ng; g++) {
        const float sc = scales[row * ng + g];
        const signed char* wr = w8 + (long long)row * K + g * 256 + boff;
        const int w0 = __builtin_nontemporal_load((const int*)(wr));
        const int w1 = __builtin_nontemporal_load((const int*)(wr + 4));
        const float* xr = xs + g * 256 + boff;
        float gsum = 0.0f;
#pragma unroll
        for (int n = 0; n < 4; n++) { int q = (w0 >> (8 * n)) & 0xff; q = (q << 24) >> 24; gsum += (float)q * xr[n]; }
#pragma unroll
        for (int n = 0; n < 4; n++) { int q = (w1 >> (8 * n)) & 0xff; q = (q << 24) >> 24; gsum += (float)q * xr[4 + n]; }
        acc += sc * gsum;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0) {
        const int mi = M >> 1;
        if (row < mi) y_gate[row] = acc; else y_up[row - mi] = acc;
    }
}
extern "C" __global__ void zaya_moe_down_oq8_planar_indexed_lds(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ sel_idx,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    extern __shared__ float xs[];
    const int tib = threadIdx.x;
    for (int i = tib; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int lane = tib & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (tib >> 5);
    if (row >= M) return;
    const char* __restrict__ A = reinterpret_cast<const char*>(expert_ptrs[sel_idx[0]]);
    const signed char* __restrict__ w8 = reinterpret_cast<const signed char*>(A);
    const float* __restrict__ scales = reinterpret_cast<const float*>(A + (long long)M * K);
    const int ng = K / 256;
    const int boff = lane * 8;
    float acc = 0.0f;
    for (int g = 0; g < ng; g++) {
        const float sc = scales[row * ng + g];
        const signed char* wr = w8 + (long long)row * K + g * 256 + boff;
        const int w0 = __builtin_nontemporal_load((const int*)(wr));
        const int w1 = __builtin_nontemporal_load((const int*)(wr + 4));
        const float* xr = xs + g * 256 + boff;
        float gsum = 0.0f;
#pragma unroll
        for (int n = 0; n < 4; n++) { int q = (w0 >> (8 * n)) & 0xff; q = (q << 24) >> 24; gsum += (float)q * xr[n]; }
#pragma unroll
        for (int n = 0; n < 4; n++) { int q = (w1 >> (8 * n)) & 0xff; q = (q << 24) >> 24; gsum += (float)q * xr[4 + n]; }
        acc += sc * gsum;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0) y[row] = acc;
}

// ── W8A8 int8-activation variants (HIPFIRE_ZAYA_MOE_W8A8) ──
// The base kernels read the activation as f32 (W8A16): per group each lane loads 8 f32
// x (32 B) but only 8 int8 weight (8 B), so the f32 activation is ~80% of the load bytes
// (disasm: 2 dwordx4 x-loads : 2 int32 w-loads / iter) and saturates the memory unit.
// Here the activation is pre-quantized to int8 (per-256-group scale, like the mq4 Q8_1
// path): 4× less x-traffic + a signed V_DOT4_I32_IU8 (sudot4) int8 dot. Xq = [K] int8
// activation, Xs = [K/256] f32 act scales; y = Σ_g ws[m,g]·Xs[g]·Σ_k w8·xq. gfx1151 wave32.
extern "C" __global__ void zaya_moe_gate_up_oq8_planar_indexed_w8a8(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ sel_idx,
    const signed char* __restrict__ Xq,
    const float* __restrict__ Xs,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const char* __restrict__ A = reinterpret_cast<const char*>(expert_ptrs[sel_idx[0]]);
    const signed char* __restrict__ w8 = reinterpret_cast<const signed char*>(A);
    const float* __restrict__ ws = reinterpret_cast<const float*>(A + (long long)M * K);
    const int ng = K / 256;
    const int boff = lane * 8;
    float acc = 0.0f;
    for (int g = 0; g < ng; g++) {
        const signed char* wr = w8 + (long long)row * K + g * 256 + boff;
        const signed char* xr = Xq + g * 256 + boff;
        const int w0 = __builtin_nontemporal_load((const int*)(wr));
        const int w1 = __builtin_nontemporal_load((const int*)(wr + 4));
        const int x0 = *(const int*)(xr);
        const int x1 = *(const int*)(xr + 4);
        int idot = 0;
        idot = __builtin_amdgcn_sudot4(true, w0, true, x0, idot, false);
        idot = __builtin_amdgcn_sudot4(true, w1, true, x1, idot, false);
        acc += (float)idot * (ws[row * ng + g] * Xs[g]);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0) {
        const int mi = M >> 1;
        if (row < mi) y_gate[row] = acc; else y_up[row - mi] = acc;
    }
}
extern "C" __global__ void zaya_moe_down_oq8_planar_indexed_w8a8(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ sel_idx,
    const signed char* __restrict__ Xq,
    const float* __restrict__ Xs,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const char* __restrict__ A = reinterpret_cast<const char*>(expert_ptrs[sel_idx[0]]);
    const signed char* __restrict__ w8 = reinterpret_cast<const signed char*>(A);
    const float* __restrict__ ws = reinterpret_cast<const float*>(A + (long long)M * K);
    const int ng = K / 256;
    const int boff = lane * 8;
    float acc = 0.0f;
    for (int g = 0; g < ng; g++) {
        const signed char* wr = w8 + (long long)row * K + g * 256 + boff;
        const signed char* xr = Xq + g * 256 + boff;
        const int w0 = __builtin_nontemporal_load((const int*)(wr));
        const int w1 = __builtin_nontemporal_load((const int*)(wr + 4));
        const int x0 = *(const int*)(xr);
        const int x1 = *(const int*)(xr + 4);
        int idot = 0;
        idot = __builtin_amdgcn_sudot4(true, w0, true, x0, idot, false);
        idot = __builtin_amdgcn_sudot4(true, w1, true, x1, idot, false);
        acc += (float)idot * (ws[row * ng + g] * Xs[g]);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0) y[row] = acc;
}

// ── Multi-row register-blocked variants (ZMOE_R rows per wave) ──
// The one-warp-per-row indexed MoE gemvs above are latency-bound (~97 GB/s of the
// ~211 GB/s peak: max clock + 100% busy, halving weight bytes via oq4.25 gives ~0
// speedup — see docs/perf EXP-24/decomposition). Each wave here owns ZMOE_R output
// rows and interleaves their per-group weight loads (x loaded once, shared) →
// ZMOE_R× outstanding memory requests → saturates DRAM latency. Bit-identical math.
#ifndef ZMOE_R
#define ZMOE_R 4
#endif

extern "C" __global__ void zaya_moe_gate_up_oq8_planar_indexed_mrow(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ sel_idx,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int M, int K
) {
    const int lane = threadIdx.x; // wave32
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[sel_idx[0]]);
    const signed char* __restrict__ w8 = reinterpret_cast<const signed char*>(A);
    const float* __restrict__ scales = reinterpret_cast<const float*>(A + (long long)M * K);
    const int ng = K / 256;
    const int boff = lane * 8;
    const int mi = M >> 1;
    for (int row0 = blockIdx.x * ZMOE_R; row0 < M; row0 += gridDim.x * ZMOE_R) {
        float acc[ZMOE_R];
        #pragma unroll
        for (int r = 0; r < ZMOE_R; ++r) acc[r] = 0.0f;
        for (int g = 0; g < ng; g++) {
            const float* xr = x + g * 256 + boff;
            const float x0 = xr[0], x1 = xr[1], x2 = xr[2], x3 = xr[3];
            const float x4 = xr[4], x5 = xr[5], x6 = xr[6], x7 = xr[7];
            #pragma unroll
            for (int r = 0; r < ZMOE_R; ++r) {
                const int row = row0 + r;
                if (row >= M) continue;
                const float sc = scales[row * ng + g];
                const signed char* wr = w8 + (long long)row * K + g * 256 + boff;
                const int w0 = __builtin_nontemporal_load((const int*)(wr));
                const int w1 = __builtin_nontemporal_load((const int*)(wr + 4));
                float gs = 0.0f;
                gs += (float)((w0 << 24) >> 24) * x0;
                gs += (float)((w0 << 16) >> 24) * x1;
                gs += (float)((w0 << 8) >> 24) * x2;
                gs += (float)(w0 >> 24) * x3;
                gs += (float)((w1 << 24) >> 24) * x4;
                gs += (float)((w1 << 16) >> 24) * x5;
                gs += (float)((w1 << 8) >> 24) * x6;
                gs += (float)(w1 >> 24) * x7;
                acc[r] += sc * gs;
            }
        }
        #pragma unroll
        for (int r = 0; r < ZMOE_R; ++r) {
            float v = acc[r];
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off);
            const int row = row0 + r;
            if (lane == 0 && row < M) {
                if (row < mi) y_gate[row] = v; else y_up[row - mi] = v;
            }
        }
    }
}

extern "C" __global__ void zaya_moe_down_oq8_planar_indexed_mrow(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ sel_idx,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int lane = threadIdx.x;
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[sel_idx[0]]);
    const signed char* __restrict__ w8 = reinterpret_cast<const signed char*>(A);
    const float* __restrict__ scales = reinterpret_cast<const float*>(A + (long long)M * K);
    const int ng = K / 256;
    const int boff = lane * 8;
    for (int row0 = blockIdx.x * ZMOE_R; row0 < M; row0 += gridDim.x * ZMOE_R) {
        float acc[ZMOE_R];
        #pragma unroll
        for (int r = 0; r < ZMOE_R; ++r) acc[r] = 0.0f;
        for (int g = 0; g < ng; g++) {
            const float* xr = x + g * 256 + boff;
            const float x0 = xr[0], x1 = xr[1], x2 = xr[2], x3 = xr[3];
            const float x4 = xr[4], x5 = xr[5], x6 = xr[6], x7 = xr[7];
            #pragma unroll
            for (int r = 0; r < ZMOE_R; ++r) {
                const int row = row0 + r;
                if (row >= M) continue;
                const float sc = scales[row * ng + g];
                const signed char* wr = w8 + (long long)row * K + g * 256 + boff;
                const int w0 = __builtin_nontemporal_load((const int*)(wr));
                const int w1 = __builtin_nontemporal_load((const int*)(wr + 4));
                float gs = 0.0f;
                gs += (float)((w0 << 24) >> 24) * x0;
                gs += (float)((w0 << 16) >> 24) * x1;
                gs += (float)((w0 << 8) >> 24) * x2;
                gs += (float)(w0 >> 24) * x3;
                gs += (float)((w1 << 24) >> 24) * x4;
                gs += (float)((w1 << 16) >> 24) * x5;
                gs += (float)((w1 << 8) >> 24) * x6;
                gs += (float)(w1 >> 24) * x7;
                acc[r] += sc * gs;
            }
        }
        #pragma unroll
        for (int r = 0; r < ZMOE_R; ++r) {
            float v = acc[r];
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off);
            const int row = row0 + r;
            if (lane == 0 && row < M) y[row] = v;
        }
    }
}
