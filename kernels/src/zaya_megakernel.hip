// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// ZAYA decode cooperative megakernel — Phase 0 (megakernel-B, the MLP half).
//
// Fuses stages 12–17 of the per-block decode body into ONE cooperative launch,
// keeping the GPU saturated across the whole MLP half so the DRAM clock ramps
// (see docs/plans/2026-07-24-zaya-cooperative-megakernel.md and
// docs/perf/zaya-decode-optimization.md). Grid-strided gemvs over all resident
// workgroups, `cg::this_grid().sync()` between dependent phases.
//
// Stages (all for a single block `li`, input `g_res2`, output `hidden`):
//   12  post-attn rmsnorm + FWHT-256 rotate  → pa_xrot           [block 0]
//   13  router MLP (down→prep→rmsnorm→[FWHT→fc→gelu]×2→FWHT→out)
//       + top-1 select                       → sel_idx, sel_gate [block 0]
//   14  MoE gate_up (indexed Oq8 planar)     → gate_up           [grid-strided]
//   15  silu(gate)*up + FWHT-256 rotate      → xr_act            [grid-strided]
//   16  MoE down (indexed Oq8 planar)        + ...
//   17  affine residual (fused into 16)      → hidden            [grid-strided]
//
// Stages 16 and 17 are fused per output row: the warp that computes the down
// projection for hidden-row `r` immediately folds in the affine residual for
// that row (`moe_out[r] = sel_gate * down[r]`), so no `moe_out` scratch and no
// extra grid.sync are needed. Result: 3 grid.sync / block.
//
// Correctness anchors (cf. plan §4.4): every Oq8 gemv reads the FWHT-rotated
// activation (weights live rotated); the down-proj activation is FWHT-rotated
// after silu_mul; the router uses prob+balancing_bias argmax with the null
// (MoD-skip) slot → gate 0. This kernel is a faithful fusion of the reference
// op chain (fused_rmsnorm_mq_rotate_plain, zaya_router_mlp_fused,
// zaya_moe_{gate_up,down}_oq8_planar_indexed, fused_silu_mul_mq_rotate,
// zaya_affine_residual_f32) — diff each stage vs the reference before trusting.

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

// ── device helpers (copied verbatim from zaya_cca.hip) ──

// GELU with the same Abramowitz-Stegun erf approximation as zaya_gelu_scalar
// in zaya_cca.hip — MUST match bit-for-bit (the router path uses it).
__device__ __forceinline__ float zmk_gelu_scalar(float v) {
    float a = v * 0.70710678118654752440f;
    float sign = (a < 0.0f) ? -1.0f : 1.0f;
    float ax = fabsf(a);
    float tt = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * tt - 1.453152027f) * tt) + 1.421413741f) * tt - 0.284496736f) * tt + 0.254829592f) * tt * expf(-ax * ax);
    return 0.5f * v * (1.0f + sign * y);
}

// FWHT-256 in-place on LDS `buf[256]`, performed by wave0 (threads 0-31), 8
// elems/lane. Mirrors rotate_x_mq: signs1 → local butterfly (strides 1,2,4) →
// ds_swizzle wave butterfly (strides 1..16) → ×0.0625 → signs2.
__device__ __forceinline__ void zmk_fwht256_lds(
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
        #define ZMKHB(v,pat,str) do{ float _p=__int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v),(pat))); if(tid&(str)){(v)=_p-(v);}else{(v)=(v)+_p;} }while(0)
        #define ZMKHB8(pat,str) ZMKHB(v0,pat,str);ZMKHB(v1,pat,str);ZMKHB(v2,pat,str);ZMKHB(v3,pat,str);ZMKHB(v4,pat,str);ZMKHB(v5,pat,str);ZMKHB(v6,pat,str);ZMKHB(v7,pat,str)
        ZMKHB8(0x041F,1); ZMKHB8(0x081F,2); ZMKHB8(0x101F,4); ZMKHB8(0x201F,8); ZMKHB8(0x401F,16);
        #undef ZMKHB8
        #undef ZMKHB
        const float s = 0.0625f;
        buf[d0]=v0*s*signs2[d0]; buf[d0+1]=v1*s*signs2[d0+1]; buf[d0+2]=v2*s*signs2[d0+2]; buf[d0+3]=v3*s*signs2[d0+3];
        buf[d0+4]=v4*s*signs2[d0+4]; buf[d0+5]=v5*s*signs2[d0+5]; buf[d0+6]=v6*s*signs2[d0+6]; buf[d0+7]=v7*s*signs2[d0+7];
    }
}

// One warp (32 lanes, 8 int8/lane) computes one Oq8-planar dot product:
// acc = sum_g scale[row,g] * sum_{k in group g} w8[row, g*256+k] * x[g*256+k].
// Planar layout: A = [M*K signed int8 row-major | M*ng f32 scales], ng = K/256.
// Returns the reduced value on lane 0 (garbage on other lanes).
__device__ __forceinline__ float zmk_oq8_planar_row(
    const signed char* __restrict__ w8, const float* __restrict__ scales,
    const float* __restrict__ x, long long row, int K, int lane
) {
    const int ng = K / 256;
    const int boff = lane * 8; // 8 int8 / lane = two aligned int32 loads
    float acc = 0.0f;
    for (int g = 0; g < ng; g++) {
        const float sc = scales[row * ng + g];
        const signed char* wr = w8 + row * K + g * 256 + boff;
        const int w0 = *(const int*)(wr);
        const int w1 = *(const int*)(wr + 4);
        const float* xr = x + g * 256 + boff;
        float gsum = 0.0f;
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int q = (w0 >> (8 * n)) & 0xff; q = (q << 24) >> 24;
            gsum += (float)q * xr[n];
        }
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int q = (w1 >> (8 * n)) & 0xff; q = (q << 24) >> 24;
            gsum += (float)q * xr[4 + n];
        }
        acc += sc * gsum;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    return acc;
}

// ── the megakernel ──
//
// Grid: [GRID, 1, 1] cooperative (GRID = resident workgroups, e.g. 160 on
// gfx1151). Block: [256, 1, 1]. 8 wave32 warps/block ⇒ GRID*8 warps total.
extern "C" __launch_bounds__(256, 1)
__global__ void zaya_decode_megakernel_b(
    // ── inputs / outputs ──
    // g_res2 is the post-attn residual (stage 12 input). When fold_oproj != 0 it
    // is WRITABLE scratch that the folded stages 10–11 compute in-kernel from ctx;
    // otherwise the caller pre-fills it (Phase 0/1 path).
    float* __restrict__ g_res2,            // [h]
    float* __restrict__ hidden,            // [h] out (stage 17); ALSO read by the
                                           // folded attn residual (stage 11)
    // ── Phase 2: folded o_proj (stage 10) + attn affine residual (stage 11) ──
    float* __restrict__ ctx,               // [q_dim] attention output (written when
                                           // fold_attn) → o_proj input
    const signed char* __restrict__ o_proj_w, // planar [h*q_dim | h*(q_dim/256)]
    float* __restrict__ ctx_rot,           // [q_dim] scratch: FWHT-rotated ctx
    const float* __restrict__ pa_hs,       // pa_rs[0] [h]  (attn affine scales)
    const float* __restrict__ pa_hb,       // pa_rs[1] [h]
    const float* __restrict__ pa_rs2,      // pa_rs[2] [h]
    const float* __restrict__ pa_rb,       // pa_rs[3] [h]
    int q_dim, int fold_oproj,
    // ── Phase 3: folded attention (stage 9: KV write + flash-decode) ──
    const float* __restrict__ query,       // [q_dim] rotated query (from megakernel-A)
    const float* __restrict__ key,         // [nkv*hd] current key
    const float* __restrict__ value,       // [nkv*hd] current value
    float* __restrict__ k_cache,           // [max_seq*nkv*hd]
    float* __restrict__ v_cache,           // [max_seq*nkv*hd]
    const int* __restrict__ pos_buf,       // seq_len = pos_buf[0]+1
    int nkv, int hd, int max_seq, float attn_scale, int fold_attn,
    // ── stage 12: post-attn rmsnorm + rotate ──
    const float* __restrict__ post_attn_ln,// [h]
    const float* __restrict__ signs1,      // [256]
    const float* __restrict__ signs2,      // [256]
    float* __restrict__ pa_xrot,           // [h] scratch: rotated activation
    float* __restrict__ mk_norm,           // [h] scratch: rmsnormed pre-FWHT
    // ── stage 13: router MLP + select ──
    const signed char* __restrict__ dp_w,  // planar [rh*h | rh*(h/256)]
    const float* __restrict__ dp_b,        // [rh]
    float* __restrict__ rstate,            // [rh] EDA state (persists across blocks)
    const float* __restrict__ escale,      // [rh] EDA scale (or dummy if !has_eda)
    const float* __restrict__ rnorm_w,     // [rh]
    const signed char* __restrict__ fc1_w, // planar [rh*rh | rh]
    const float* __restrict__ fc1_b,       // [rh]
    const signed char* __restrict__ fc2_w, // planar [rh*rh | rh]
    const float* __restrict__ fc2_b,       // [rh]
    const signed char* __restrict__ out_w, // planar [n_route*rh | n_route]
    const float* __restrict__ bbias,       // [n_route]
    int* __restrict__ sel_idx,             // [1] out: winning expert id
    float* __restrict__ sel_gate,          // [1] out: unbiased softmax gate
    // ── stage 14/16: indexed MoE experts ──
    const unsigned long long* __restrict__ gate_up_ptrs, // [n_route]
    const unsigned long long* __restrict__ down_ptrs,    // [n_route]
    float* __restrict__ gate_up,           // [2*moe_int] scratch
    float* __restrict__ xr_act,            // [moe_int] scratch
    // ── stage 17: affine residual (pm_rs) ──
    const float* __restrict__ pm_hs,       // pm_rs[0] [h]
    const float* __restrict__ pm_hb,       // pm_rs[1] [h]
    const float* __restrict__ pm_rs,       // pm_rs[2] [h]
    const float* __restrict__ pm_rb,       // pm_rs[3] [h]
    // ── dims ──
    int h, int rh, int n_route, int moe_int, float eps, int has_eda
) {
    cg::grid_group grid = cg::this_grid();
    const int tid = threadIdx.x;                 // 0..255
    const int lane = tid & 31;
    const int warp_in_block = tid >> 5;          // 0..7
    const int warps_per_block = blockDim.x >> 5; // 8
    const int global_warp = blockIdx.x * warps_per_block + warp_in_block;
    const int total_warps = gridDim.x * warps_per_block;

    __shared__ float sm[256];   // router work buffer (rh == 256)
    __shared__ float red[8];    // warp partials
    __shared__ float logit[32]; // router logits (n_route <= 32)

    // ══ Phase 3 head (fold_attn): stage 9 KV write + flash-decode attention ══
    // Faithful fusion of kv_cache_write ×2 + attention_f32_nolds (register-resident
    // online softmax, one wave32/head). Writes ctx, consumed by the o_proj fold below.
    if (fold_attn) {
        const int kvdim = nkv * hd;
        const int pos = pos_buf[0];
        const int gthread = blockIdx.x * blockDim.x + tid;
        const int gthreads = gridDim.x * blockDim.x;
        // 9a: append current K/V at the device position.
        for (int i = gthread; i < kvdim; i += gthreads) {
            k_cache[(size_t)pos * kvdim + i] = key[i];
            v_cache[(size_t)pos * kvdim + i] = value[i];
        }
        grid.sync();
        // 9b: per-head flash-decode attention → ctx (heads grid-strided over warps).
        const int nq = q_dim / hd;
        const int seq_len = pos + 1;
        const int kv_group = nq / nkv;
        for (int hh = global_warp; hh < nq; hh += total_warps) {
            const int kv_h = hh / kv_group;
            const int kv_off = kv_h * hd;
            const float* q_head = query + hh * hd;
            float q_reg[8], acc[8];
            int ndl = 0;
            for (int d = lane; d < hd; d += 32) { q_reg[ndl] = q_head[d]; acc[ndl] = 0.0f; ndl++; }
            float m = -1e30f, l = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                const float* k_t = k_cache + (size_t)t * kvdim + kv_off;
                float part = 0.0f;
                int di = 0;
                for (int d = lane; d < hd; d += 32) part += q_reg[di++] * k_t[d];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) part += __shfl_xor(part, off, 32);
                const float s = part * attn_scale;
                const float m_new = fmaxf(m, s);
                const float alpha = __expf(m - m_new);
                const float p = __expf(s - m_new);
                l = l * alpha + p;
                const float* v_t = v_cache + (size_t)t * kvdim + kv_off;
                di = 0;
                for (int d = lane; d < hd; d += 32) { acc[di] = acc[di] * alpha + p * v_t[d]; di++; }
                m = m_new;
            }
            float* out_head = ctx + hh * hd;
            const float inv_l = 1.0f / l;
            int di = 0;
            for (int d = lane; d < hd; d += 32) out_head[d] = acc[di++] * inv_l;
        }
        grid.sync();
    }

    // ══ Phase 2 head (fold_oproj): stage 10 o_proj + stage 11 attn affine ══
    if (fold_oproj) {
        // Stage 10a (block 0): FWHT-256 rotate ctx → ctx_rot (q_dim/256 groups,
        // NO rmsnorm — o_proj is not preceded by a norm). Opus weights live rotated.
        if (blockIdx.x == 0) {
            const int groups = q_dim / 256;
            for (int group = warp_in_block; group < groups; group += warps_per_block) {
                const int d0 = lane * 8;
                const int base = group * 256 + d0;
                float v0 = ctx[base]   * signs1[d0];
                float v1 = ctx[base+1] * signs1[d0+1];
                float v2 = ctx[base+2] * signs1[d0+2];
                float v3 = ctx[base+3] * signs1[d0+3];
                float v4 = ctx[base+4] * signs1[d0+4];
                float v5 = ctx[base+5] * signs1[d0+5];
                float v6 = ctx[base+6] * signs1[d0+6];
                float v7 = ctx[base+7] * signs1[d0+7];
                float t;
                t=v0;v0=v0+v1;v1=t-v1; t=v2;v2=v2+v3;v3=t-v3; t=v4;v4=v4+v5;v5=t-v5; t=v6;v6=v6+v7;v7=t-v7;
                t=v0;v0=v0+v2;v2=t-v2; t=v1;v1=v1+v3;v3=t-v3; t=v4;v4=v4+v6;v6=t-v6; t=v5;v5=v5+v7;v7=t-v7;
                t=v0;v0=v0+v4;v4=t-v4; t=v1;v1=v1+v5;v5=t-v5; t=v2;v2=v2+v6;v6=t-v6; t=v3;v3=v3+v7;v7=t-v7;
                #define ZMKHB(v,pat,str) do{ float _p=__int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v),(pat))); if(lane&(str)){(v)=_p-(v);}else{(v)=(v)+_p;} }while(0)
                #define ZMKHB8(pat,str) ZMKHB(v0,pat,str);ZMKHB(v1,pat,str);ZMKHB(v2,pat,str);ZMKHB(v3,pat,str);ZMKHB(v4,pat,str);ZMKHB(v5,pat,str);ZMKHB(v6,pat,str);ZMKHB(v7,pat,str)
                ZMKHB8(0x041F,1); ZMKHB8(0x081F,2); ZMKHB8(0x101F,4); ZMKHB8(0x201F,8); ZMKHB8(0x401F,16);
                #undef ZMKHB8
                #undef ZMKHB
                const float s = 0.0625f;
                ctx_rot[base]   = v0*s*signs2[d0];
                ctx_rot[base+1] = v1*s*signs2[d0+1];
                ctx_rot[base+2] = v2*s*signs2[d0+2];
                ctx_rot[base+3] = v3*s*signs2[d0+3];
                ctx_rot[base+4] = v4*s*signs2[d0+4];
                ctx_rot[base+5] = v5*s*signs2[d0+5];
                ctx_rot[base+6] = v6*s*signs2[d0+6];
                ctx_rot[base+7] = v7*s*signs2[d0+7];
            }
        }
        grid.sync();
        // Stage 10b + 11 fused per output row: attn_out[r] = o_proj·ctx_rot; then
        // g_res2[r] = (attn_out[r]+pa_hb)*pa_hs + (hidden[r]+pa_rb)*pa_rs2.
        {
            const float* oscales =
                reinterpret_cast<const float*>(o_proj_w + (long long)h * q_dim);
            for (int row = global_warp; row < h; row += total_warps) {
                float acc = zmk_oq8_planar_row(o_proj_w, oscales, ctx_rot, row, q_dim, lane);
                if (lane == 0) {
                    g_res2[row] = (acc + pa_hb[row]) * pa_hs[row]
                                + (hidden[row] + pa_rb[row]) * pa_rs2[row];
                }
            }
        }
        grid.sync();
    }

    // ══ Phase A (block 0 only): stage 12 rmsnorm+rotate, then stage 13 router ══
    if (blockIdx.x == 0) {
        // ---- stage 12: rmsnorm(g_res2) -> mk_norm, FWHT-256 rotate -> pa_xrot ----
        // Pass 1: sum of squares over h.
        float ls = 0.0f;
        for (int i = tid; i < h; i += blockDim.x) {
            float v = g_res2[i];
            ls += v * v;
        }
        for (int o = 16; o > 0; o >>= 1) ls += __shfl_down(ls, o);
        if (lane == 0) red[warp_in_block] = ls;
        __syncthreads();
        float rms;
        {
            float s = 0.0f;
            for (int i = 0; i < warps_per_block; i++) s += red[i];
            rms = rsqrtf(s / (float)h + eps);
        }
        // Pass 2: normalize -> mk_norm (global).
        for (int i = tid; i < h; i += blockDim.x) {
            mk_norm[i] = g_res2[i] * post_attn_ln[i] * rms;
        }
        __syncthreads();
        // Pass 3: FWHT-256 per group (h/256 groups), one warp per group, no LDS.
        const int groups = h / 256;
        for (int group = warp_in_block; group < groups; group += warps_per_block) {
            const int d0 = lane * 8;
            const int base = group * 256 + d0;
            float v0 = mk_norm[base]   * signs1[d0];
            float v1 = mk_norm[base+1] * signs1[d0+1];
            float v2 = mk_norm[base+2] * signs1[d0+2];
            float v3 = mk_norm[base+3] * signs1[d0+3];
            float v4 = mk_norm[base+4] * signs1[d0+4];
            float v5 = mk_norm[base+5] * signs1[d0+5];
            float v6 = mk_norm[base+6] * signs1[d0+6];
            float v7 = mk_norm[base+7] * signs1[d0+7];
            float t;
            t=v0;v0=v0+v1;v1=t-v1; t=v2;v2=v2+v3;v3=t-v3; t=v4;v4=v4+v5;v5=t-v5; t=v6;v6=v6+v7;v7=t-v7;
            t=v0;v0=v0+v2;v2=t-v2; t=v1;v1=v1+v3;v3=t-v3; t=v4;v4=v4+v6;v6=t-v6; t=v5;v5=v5+v7;v7=t-v7;
            t=v0;v0=v0+v4;v4=t-v4; t=v1;v1=v1+v5;v5=t-v5; t=v2;v2=v2+v6;v6=t-v6; t=v3;v3=v3+v7;v7=t-v7;
            #define ZMKHB(v,pat,str) do{ float _p=__int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v),(pat))); if(lane&(str)){(v)=_p-(v);}else{(v)=(v)+_p;} }while(0)
            #define ZMKHB8(pat,str) ZMKHB(v0,pat,str);ZMKHB(v1,pat,str);ZMKHB(v2,pat,str);ZMKHB(v3,pat,str);ZMKHB(v4,pat,str);ZMKHB(v5,pat,str);ZMKHB(v6,pat,str);ZMKHB(v7,pat,str)
            ZMKHB8(0x041F,1); ZMKHB8(0x081F,2); ZMKHB8(0x101F,4); ZMKHB8(0x201F,8); ZMKHB8(0x401F,16);
            #undef ZMKHB8
            #undef ZMKHB
            const float s = 0.0625f;
            pa_xrot[base]   = v0*s*signs2[d0];
            pa_xrot[base+1] = v1*s*signs2[d0+1];
            pa_xrot[base+2] = v2*s*signs2[d0+2];
            pa_xrot[base+3] = v3*s*signs2[d0+3];
            pa_xrot[base+4] = v4*s*signs2[d0+4];
            pa_xrot[base+5] = v5*s*signs2[d0+5];
            pa_xrot[base+6] = v6*s*signs2[d0+6];
            pa_xrot[base+7] = v7*s*signs2[d0+7];
        }
        __syncthreads();

        // ---- stage 13: router MLP + select (single-WG, faithful to zaya_router_mlp_fused) ----
        // Phase 1: down_proj (thread m -> rhid[m]) + prep. K=h, ng=h/256.
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
            sm[tid] = v;
        }
        __syncthreads();
        // Phase 2: rmsnorm(sm).
        float sq = (tid < rh) ? sm[tid] * sm[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_down(sq, o);
        if ((tid & 31) == 0) red[tid >> 5] = sq;
        __syncthreads();
        if (tid == 0) {
            float s = 0.0f;
            for (int i = 0; i < warps_per_block; ++i) s += red[i];
            red[0] = rsqrtf(s / (float)rh + eps);
        }
        __syncthreads();
        const float rrms = red[0];
        if (tid < rh) sm[tid] = sm[tid] * rrms * rnorm_w[tid];
        __syncthreads();
        // Phase 3: FWHT -> fc1 -> gelu.
        zmk_fwht256_lds(sm, signs1, signs2, tid);
        __syncthreads();
        float av = 0.0f;
        if (tid < rh) {
            const signed char* wr = fc1_w + (long long)tid * rh;
            const float sc = *((const float*)(fc1_w + (long long)rh * rh) + tid);
            float ss = 0.0f;
            for (int k = 0; k < rh; ++k) ss += (float)wr[k] * sm[k];
            av = zmk_gelu_scalar(fc1_b[tid] + sc * ss);
        }
        __syncthreads();
        if (tid < rh) sm[tid] = av;
        __syncthreads();
        // Phase 4: FWHT -> fc2 -> gelu.
        zmk_fwht256_lds(sm, signs1, signs2, tid);
        __syncthreads();
        av = 0.0f;
        if (tid < rh) {
            const signed char* wr = fc2_w + (long long)tid * rh;
            const float sc = *((const float*)(fc2_w + (long long)rh * rh) + tid);
            float ss = 0.0f;
            for (int k = 0; k < rh; ++k) ss += (float)wr[k] * sm[k];
            av = zmk_gelu_scalar(fc2_b[tid] + sc * ss);
        }
        __syncthreads();
        if (tid < rh) sm[tid] = av;
        __syncthreads();
        // Phase 5: FWHT -> out (n_route rows) -> select.
        zmk_fwht256_lds(sm, signs1, signs2, tid);
        __syncthreads();
        if (tid < n_route) {
            const signed char* wr = out_w + (long long)tid * rh;
            const float sc = *((const float*)(out_w + (long long)n_route * rh) + tid);
            float ss = 0.0f;
            for (int k = 0; k < rh; ++k) ss += (float)wr[k] * sm[k];
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
    grid.sync();

    // ══ Phase B (all warps, grid-strided rows): stage 14 gate_up gemv ══
    // M = 2*moe_int rows, K = h. gate_up[row] = expert_gate_up(pa_xrot)[row].
    {
        const int M = 2 * moe_int;
        const signed char* A =
            reinterpret_cast<const signed char*>(gate_up_ptrs[sel_idx[0]]);
        const float* scales = reinterpret_cast<const float*>(A + (long long)M * h);
        for (int row = global_warp; row < M; row += total_warps) {
            float acc = zmk_oq8_planar_row(A, scales, pa_xrot, row, h, lane);
            if (lane == 0) gate_up[row] = acc;
        }
    }
    grid.sync();

    // ══ Phase C (grid-strided groups): stage 15 silu(gate)*up + FWHT rotate ══
    // gate = gate_up[0..moe_int], up = gate_up[moe_int..2*moe_int]. Per 256-group.
    {
        const int groups = moe_int / 256;
        const float* gate = gate_up;
        const float* up = gate_up + moe_int;
        for (int group = global_warp; group < groups; group += total_warps) {
            const int d0 = lane * 8;
            const int base = group * 256 + d0;
            #define ZMK_SILU(g, u) ((g) / (1.0f + expf(-(g))) * (u))
            float v0 = ZMK_SILU(gate[base],   up[base])   * signs1[d0];
            float v1 = ZMK_SILU(gate[base+1], up[base+1]) * signs1[d0+1];
            float v2 = ZMK_SILU(gate[base+2], up[base+2]) * signs1[d0+2];
            float v3 = ZMK_SILU(gate[base+3], up[base+3]) * signs1[d0+3];
            float v4 = ZMK_SILU(gate[base+4], up[base+4]) * signs1[d0+4];
            float v5 = ZMK_SILU(gate[base+5], up[base+5]) * signs1[d0+5];
            float v6 = ZMK_SILU(gate[base+6], up[base+6]) * signs1[d0+6];
            float v7 = ZMK_SILU(gate[base+7], up[base+7]) * signs1[d0+7];
            #undef ZMK_SILU
            float t;
            t=v0;v0=v0+v1;v1=t-v1; t=v2;v2=v2+v3;v3=t-v3; t=v4;v4=v4+v5;v5=t-v5; t=v6;v6=v6+v7;v7=t-v7;
            t=v0;v0=v0+v2;v2=t-v2; t=v1;v1=v1+v3;v3=t-v3; t=v4;v4=v4+v6;v6=t-v6; t=v5;v5=v5+v7;v7=t-v7;
            t=v0;v0=v0+v4;v4=t-v4; t=v1;v1=v1+v5;v5=t-v5; t=v2;v2=v2+v6;v6=t-v6; t=v3;v3=v3+v7;v7=t-v7;
            #define ZMKHB(v,pat,str) do{ float _p=__int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v),(pat))); if(lane&(str)){(v)=_p-(v);}else{(v)=(v)+_p;} }while(0)
            #define ZMKHB8(pat,str) ZMKHB(v0,pat,str);ZMKHB(v1,pat,str);ZMKHB(v2,pat,str);ZMKHB(v3,pat,str);ZMKHB(v4,pat,str);ZMKHB(v5,pat,str);ZMKHB(v6,pat,str);ZMKHB(v7,pat,str)
            ZMKHB8(0x041F,1); ZMKHB8(0x081F,2); ZMKHB8(0x101F,4); ZMKHB8(0x201F,8); ZMKHB8(0x401F,16);
            #undef ZMKHB8
            #undef ZMKHB
            const float s = 0.0625f;
            xr_act[base]   = v0*s*signs2[d0];
            xr_act[base+1] = v1*s*signs2[d0+1];
            xr_act[base+2] = v2*s*signs2[d0+2];
            xr_act[base+3] = v3*s*signs2[d0+3];
            xr_act[base+4] = v4*s*signs2[d0+4];
            xr_act[base+5] = v5*s*signs2[d0+5];
            xr_act[base+6] = v6*s*signs2[d0+6];
            xr_act[base+7] = v7*s*signs2[d0+7];
        }
    }
    grid.sync();

    // ══ Phase D (grid-strided rows): stage 16 down gemv + stage 17 affine (fused) ══
    // M = h rows, K = moe_int. hidden[r] = (sel_gate*down[r] + pm_hb[r])*pm_hs[r]
    //                                     + (g_res2[r] + pm_rb[r])*pm_rs[r].
    {
        const signed char* A =
            reinterpret_cast<const signed char*>(down_ptrs[sel_idx[0]]);
        const float* scales = reinterpret_cast<const float*>(A + (long long)h * moe_int);
        const float sg = sel_gate[0];
        for (int row = global_warp; row < h; row += total_warps) {
            float acc = zmk_oq8_planar_row(A, scales, xr_act, row, moe_int, lane);
            if (lane == 0) {
                float moe = sg * acc;
                hidden[row] = (moe + pm_hb[row]) * pm_hs[row]
                            + (g_res2[row] + pm_rb[row]) * pm_rs[row];
            }
        }
    }
}

// ── megakernel-A (Phase 1): the front half (stages 1–8) ──
//
// Fuses the CCA in-projection + conv glue into ONE cooperative launch:
//   1  input rmsnorm + FWHT-256 rotate      → qkv_xrot           [block 0]
//   2  fused qkv/z/beta/alpha Oq8 gemv       → q,k,vcur,vdel      [grid-strided]
//   3  qk-prep (qres, kres, cur_qk column)   →                    [block 0]
//   4  conv window build + ring advance      → window             [grid-strided]
//   5  depthwise conv1d                       → dw                 [grid-strided]
//   6  grouped conv1d                         → gw                 [grid-strided]
//   7  add-conv-residual q+k; value assemble  → query,key,value    [grid-strided]
//   8  q+k L2-norm+scale; partial RoPE        → query,key          [grid-strided]
//
// Attention + o_proj stay as separate launches after this (Phase 1). Faithful
// fusion of fused_qkvza_oq8_gemv + zaya_{qk_prep_decode, conv_window,
// conv1d_valid×2, add_conv_residual_qk, value_assemble_decode, qk_l2norm_qk,
// rope_partial_qk_posbuf}. Single-token decode (s = 1).
extern "C" __launch_bounds__(256, 1)
__global__ void zaya_decode_megakernel_a(
    const float* __restrict__ hidden,      // [h] residual (input)
    const float* __restrict__ input_ln,    // [h]
    const float* __restrict__ signs1,      // [256]
    const float* __restrict__ signs2,      // [256]
    float* __restrict__ qkv_xrot,          // [h] scratch (rmsnorm+rotate, in-place FWHT)
    // stage 2: qkv gemv (four planar Oq8 weight buffers, K = h)
    const signed char* __restrict__ w_q,   // [q_dim*h | q_dim*(h/256)]
    const signed char* __restrict__ w_k,   // [k_dim*h | ...]
    const signed char* __restrict__ w_vc,  // [v_half*h | ...]
    const signed char* __restrict__ w_vd,  // [v_half*h | ...]
    float* __restrict__ q,                 // [q_dim] scratch
    float* __restrict__ k,                 // [k_dim] scratch
    float* __restrict__ vcur,              // [v_half] scratch
    float* __restrict__ vdel,              // [v_half] scratch
    // stage 3: qk-prep
    float* __restrict__ qres,              // [q_dim] scratch
    float* __restrict__ kres,              // [k_dim] scratch
    float* __restrict__ cur_qk,            // [conv_ch] scratch
    // stage 4-6: conv
    const float* __restrict__ conv_dw_w,   // [conv_ch*1*dwk]
    const float* __restrict__ conv_dw_b,   // [conv_ch]
    const float* __restrict__ conv_gr_w,   // [conv_ch*per_group_gr*grk]
    const float* __restrict__ conv_gr_b,   // [conv_ch]
    float* __restrict__ conv_ring,         // [conv_ch*pad] state (in/out)
    float* __restrict__ window,            // [conv_ch*(pad+1)] scratch
    float* __restrict__ dw,                // [conv_ch*dw_len] scratch
    float* __restrict__ gw,                // [conv_ch] scratch
    // stage 7: residual add + value assemble
    float* __restrict__ delayed_v,         // [v_half] state (in/out)
    float* __restrict__ query,             // [q_dim] out
    float* __restrict__ key,               // [k_dim] out
    float* __restrict__ value,             // [k_dim] out (2 heads × hd)
    // stage 8: l2norm + rope
    const float* __restrict__ qk_temp,     // [nkv] per-head key temp
    const int* __restrict__ pos_buf,       // [1] device position
    // dims
    int h, int nq, int nkv, int hd, int q_dim, int k_dim, int v_half,
    int conv_ch, int pad, int dwk, int grk, int dw_len, int n_rot,
    float rope_theta, float l2_scale, float eps
) {
    cg::grid_group grid = cg::this_grid();
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_in_block = tid >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int global_warp = blockIdx.x * warps_per_block + warp_in_block;
    const int total_warps = gridDim.x * warps_per_block;
    const int gthread = blockIdx.x * blockDim.x + tid;
    const int gthreads = gridDim.x * blockDim.x;

    __shared__ float red[8];

    // ══ Stage 1 (block 0): input rmsnorm + FWHT-256 rotate → qkv_xrot ══
    if (blockIdx.x == 0) {
        float ls = 0.0f;
        for (int i = tid; i < h; i += blockDim.x) { float v = hidden[i]; ls += v * v; }
        for (int o = 16; o > 0; o >>= 1) ls += __shfl_down(ls, o);
        if (lane == 0) red[warp_in_block] = ls;
        __syncthreads();
        float rms;
        { float s = 0.0f; for (int i = 0; i < warps_per_block; i++) s += red[i];
          rms = rsqrtf(s / (float)h + eps); }
        for (int i = tid; i < h; i += blockDim.x)
            qkv_xrot[i] = hidden[i] * input_ln[i] * rms;
        __syncthreads();
        const int groups = h / 256;
        for (int group = warp_in_block; group < groups; group += warps_per_block) {
            const int d0 = lane * 8;
            const int base = group * 256 + d0;
            float v0 = qkv_xrot[base]   * signs1[d0];
            float v1 = qkv_xrot[base+1] * signs1[d0+1];
            float v2 = qkv_xrot[base+2] * signs1[d0+2];
            float v3 = qkv_xrot[base+3] * signs1[d0+3];
            float v4 = qkv_xrot[base+4] * signs1[d0+4];
            float v5 = qkv_xrot[base+5] * signs1[d0+5];
            float v6 = qkv_xrot[base+6] * signs1[d0+6];
            float v7 = qkv_xrot[base+7] * signs1[d0+7];
            float t;
            t=v0;v0=v0+v1;v1=t-v1; t=v2;v2=v2+v3;v3=t-v3; t=v4;v4=v4+v5;v5=t-v5; t=v6;v6=v6+v7;v7=t-v7;
            t=v0;v0=v0+v2;v2=t-v2; t=v1;v1=v1+v3;v3=t-v3; t=v4;v4=v4+v6;v6=t-v6; t=v5;v5=v5+v7;v7=t-v7;
            t=v0;v0=v0+v4;v4=t-v4; t=v1;v1=v1+v5;v5=t-v5; t=v2;v2=v2+v6;v6=t-v6; t=v3;v3=v3+v7;v7=t-v7;
            #define ZMKHB(v,pat,str) do{ float _p=__int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v),(pat))); if(lane&(str)){(v)=_p-(v);}else{(v)=(v)+_p;} }while(0)
            #define ZMKHB8(pat,str) ZMKHB(v0,pat,str);ZMKHB(v1,pat,str);ZMKHB(v2,pat,str);ZMKHB(v3,pat,str);ZMKHB(v4,pat,str);ZMKHB(v5,pat,str);ZMKHB(v6,pat,str);ZMKHB(v7,pat,str)
            ZMKHB8(0x041F,1); ZMKHB8(0x081F,2); ZMKHB8(0x101F,4); ZMKHB8(0x201F,8); ZMKHB8(0x401F,16);
            #undef ZMKHB8
            #undef ZMKHB
            const float s = 0.0625f;
            qkv_xrot[base]   = v0*s*signs2[d0];
            qkv_xrot[base+1] = v1*s*signs2[d0+1];
            qkv_xrot[base+2] = v2*s*signs2[d0+2];
            qkv_xrot[base+3] = v3*s*signs2[d0+3];
            qkv_xrot[base+4] = v4*s*signs2[d0+4];
            qkv_xrot[base+5] = v5*s*signs2[d0+5];
            qkv_xrot[base+6] = v6*s*signs2[d0+6];
            qkv_xrot[base+7] = v7*s*signs2[d0+7];
        }
    }
    grid.sync();

    // ══ Stage 2 (grid-strided rows): fused qkv/z/beta/alpha Oq8 gemv ══
    {
        const int m_tot = q_dim + k_dim + v_half + v_half;
        for (int r = global_warp; r < m_tot; r += total_warps) {
            const signed char* W; float* Y; int Mrows, local;
            if (r < q_dim)                       { W = w_q;  Y = q;    Mrows = q_dim;  local = r; }
            else if (r < q_dim + k_dim)          { W = w_k;  Y = k;    Mrows = k_dim;  local = r - q_dim; }
            else if (r < q_dim + k_dim + v_half) { W = w_vc; Y = vcur; Mrows = v_half; local = r - q_dim - k_dim; }
            else                                 { W = w_vd; Y = vdel; Mrows = v_half; local = r - q_dim - k_dim - v_half; }
            const float* scales = reinterpret_cast<const float*>(W + (long long)Mrows * h);
            float acc = zmk_oq8_planar_row(W, scales, qkv_xrot, local, h, lane);
            if (lane == 0) Y[local] = acc;
        }
    }
    grid.sync();

    // ══ Stage 3 (block 0): qk-prep → qres, kres, cur_qk ══
    if (blockIdx.x == 0) {
        const int groups = nq / nkv;
        for (int c = tid; c < q_dim + k_dim; c += blockDim.x)
            cur_qk[c] = (c < q_dim) ? q[c] : k[c - q_dim];
        for (int i = tid; i < nq * hd; i += blockDim.x) {
            const int d = i % hd;
            const int head = i / hd;
            const int kh = head / groups;
            qres[i] = (q[head * hd + d] + k[kh * hd + d]) * 0.5f;
        }
        __syncthreads();
        for (int i = tid; i < nkv * hd; i += blockDim.x) {
            const int d = i % hd;
            const int kh = i / hd;
            float acc = 0.0f;
            for (int g = 0; g < groups; ++g) acc += qres[(kh * groups + g) * hd + d];
            kres[i] = acc / (float)groups;
        }
    }
    grid.sync();

    // ══ Stage 4 (grid-strided channels): conv window + ring advance ══
    {
        const int wlen = pad + 1;
        for (int c = gthread; c < conv_ch; c += gthreads) {
            for (int j = 0; j < pad; ++j) window[c * wlen + j] = conv_ring[c * pad + j];
            window[c * wlen + pad] = cur_qk[c];
            for (int j = 0; j + 1 < pad; ++j) conv_ring[c * pad + j] = conv_ring[c * pad + j + 1];
            conv_ring[c * pad + (pad - 1)] = cur_qk[c];
        }
    }
    grid.sync();

    // ══ Stage 5 (grid-strided): depthwise conv1d (groups = conv_ch) → dw ══
    {
        const int in_len = pad + 1;
        for (int idx = gthread; idx < conv_ch * dw_len; idx += gthreads) {
            const int c = idx / dw_len;
            const int t = idx % dw_len;
            // per_group = 1, base = c.
            float acc = conv_dw_b[c];
            const float* wr = &conv_dw_w[c * dwk];
            const float* ir = &window[c * in_len + t];
            for (int kk = 0; kk < dwk; ++kk) acc += wr[kk] * ir[kk];
            dw[c * dw_len + t] = acc;
        }
    }
    grid.sync();

    // ══ Stage 6 (grid-strided): grouped conv1d (groups = nq+nkv, out_len=1) → gw ══
    {
        const int groups_gr = nq + nkv;
        const int per_group = conv_ch / groups_gr;
        const int in_len = dw_len;
        for (int c = gthread; c < conv_ch; c += gthreads) {
            const int base = (c / per_group) * per_group;
            float acc = conv_gr_b[c];
            for (int j = 0; j < per_group; ++j) {
                const float* wr = &conv_gr_w[(c * per_group + j) * grk];
                const float* ir = &dw[(base + j) * in_len + 0];
                for (int kk = 0; kk < grk; ++kk) acc += wr[kk] * ir[kk];
            }
            gw[c] = acc;
        }
    }
    grid.sync();

    // ══ Stage 7 (grid-strided): add-conv-residual q+k; value assemble ══
    {
        // add-conv-residual (s = 1): conv is channel-major gw[conv_ch].
        for (int j = gthread; j < q_dim + k_dim; j += gthreads) {
            if (j < q_dim) {
                const int d = j % hd, head = j / hd;
                query[head * hd + d] = gw[head * hd + d] + qres[head * hd + d];
            } else {
                const int jj = j - q_dim;
                const int d = jj % hd, head = jj / hd;
                key[head * hd + d] = gw[q_dim + head * hd + d] + kres[head * hd + d];
            }
        }
        // value assemble (head0 = current v, head1 = previous delayed v; advance).
        for (int d = gthread; d < v_half; d += gthreads) {
            float old_del = delayed_v[d];
            value[d] = vcur[d];
            value[v_half + d] = old_del;
            delayed_v[d] = vdel[d];
        }
    }
    grid.sync();

    // ══ Stage 8a (grid-strided rows): q+k L2-norm + scale ══
    {
        const int rows = nq + nkv;
        for (int row = gthread; row < rows; row += gthreads) {
            float* v; float tmul;
            if (row < nq) { v = &query[row * hd]; tmul = 1.0f; }
            else          { int kh = (row - nq) % nkv; v = &key[(row - nq) * hd]; tmul = qk_temp[kh]; }
            float nrm = 0.0f;
            for (int d = 0; d < hd; ++d) nrm += v[d] * v[d];
            float inv = l2_scale / fmaxf(sqrtf(nrm), eps);
            for (int d = 0; d < hd; ++d) v[d] = v[d] * inv * tmul;
        }
    }
    grid.sync();

    // ══ Stage 8b (grid-strided rows): partial RoPE (device position) ══
    {
        const int pos_base = pos_buf[0];
        const int rows = nq + nkv;
        const int half = n_rot / 2;
        for (int row = gthread; row < rows; row += gthreads) {
            float* v = (row < nq) ? &query[row * hd] : &key[(row - nq) * hd];
            const int t = pos_base; // s = 1 ⇒ all heads share the token position
            for (int d = 0; d < half; ++d) {
                float inv_freq = powf(rope_theta, -(2.0f * (float)d) / (float)n_rot);
                float ang = (float)t * inv_freq;
                float c = cosf(ang), sn = sinf(ang);
                float a = v[d], b = v[d + half];
                v[d] = a * c - b * sn;
                v[d + half] = b * c + a * sn;
            }
        }
    }
}

// ── grid.sync() micro-benchmark (EXP-19 diagnostic) ──
// Times the cost of a cooperative grid barrier at the megakernel's launch shape
// (160 blocks × 256 threads). `scratch[0]` write prevents the loop being elided.
extern "C" __launch_bounds__(256, 1)
__global__ void zaya_grid_sync_bench(int* __restrict__ scratch, int n_syncs) {
    cg::grid_group grid = cg::this_grid();
    const int gt = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < n_syncs; i++) {
        if (gt == 0) scratch[0] = i;
        grid.sync();
    }
}
