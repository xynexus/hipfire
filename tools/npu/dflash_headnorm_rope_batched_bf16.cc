// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// AIE2/AIE2P BF16 fused per-head RMSNorm + FULL-neox RoPE for the DFlash
// drafter — BATCHED variant (Phase D stage fusion, step 1).
//
// Fuses what are today TWO dispatches per tensor (qwen35-headnorm-* then
// dflash-rope-*) into ONE. Applied to both Q and K this saves 2 dispatches per
// layer (13.6 -> 11.6).
//
// Relationship to the existing kernels:
//   * rms_norm_head_bf16.cc          — the norm half (weight = once-acquired param)
//   * dflash_rope_batched_bf16.cc    — the rope half (cs = 2nd TILED input, so each
//                                      head-tile carries its own token position)
//   * headnorm_rope_bf16.cc          — the Qwen3.5 fused kernel, but hard-coded to
//                                      PARTIAL rotary (n_rot = head_dim/4) with cs
//                                      packed into the weight param. The DFlash
//                                      drafter needs FULL head_dim neox and
//                                      per-row positions, hence this variant.
//
// ── Why TWO heads per tile (the DMA-channel constraint) ─────────────────────
// The obvious fusion — tiled input + tiled cs + a weight tensor param — needs
// THREE inbound DMA channels on the core tile, but an AIE2 core tile has only
// 2 input / 2 output. aiecc rejects it outright:
//     "tile (0,3) requires 3 input/1 output DMA channels, but only 2 input/2
//      output available"
// So the norm weight and the cos/sin must share ONE input stream. They are
// both [head_dim], so the tile is widened to 2*head_dim and each invocation
// processes a PAIR of heads:
//
//   input  tile [2*head_dim] : two consecutive heads (head A, head B)
//   coeff  tile [2*head_dim] : [ gamma(head_dim) , cs(head_dim) ]
//   output tile [2*head_dim] : the two normed + rotated heads
//
// This works because gamma is shared by every head, and the two heads in a
// pair always belong to the SAME row (n_heads is even: 32 for Q, 8 for K) and
// therefore share the same token position, hence the same cs. Two inbound
// streams, one outbound — fits the tile.
//
// ── Inputs / output ──────────────────────────────────────────────────────────
//   input  : [2*head_dim] bf16 — two heads of Q or K        (tiled input 0)
//   coeff  : [2*head_dim] bf16 — [gamma | cs] for that row  (tiled input 1)
//   output : [2*head_dim] bf16 — the two normed + rotated heads
//   n      : int32             — tile size = 2*head_dim (auto-appended by IRON)
//
//   C signature (from _transform_gen: *inputs, output, n):
//     dflash_headnorm_rope_batched_bf16(input, coeff, output, n)
//
// ── cs layout (half-split, FULL rotation n_rot = head_dim) ───────────────────
//   cs[0 .. n_rot2)        = cos(pos * inv_freq[i])
//   cs[n_rot2 .. head_dim) = sin(pos * inv_freq[i])      n_rot2 = head_dim/2
//
// ── Algorithm, per head (twice per invocation) ──────────────────────────────
//   Pass 1: sum(x[i]^2) over [0, head_dim) -> inv_rms = 1/sqrt(mean + eps)
//   Pass 2 (all pairs i in [0, n_rot2) — FULL rotation, no passthrough region):
//     x_n = x[i]        * inv_rms * gamma[i]
//     y_n = x[i+n_rot2] * inv_rms * gamma[i+n_rot2]
//     output[i]        = x_n*cos[i] - y_n*sin[i]
//     output[i+n_rot2] = y_n*cos[i] + x_n*sin[i]
//
// ── IRON dispatch pattern ────────────────────────────────────────────────────
//   _transform_gen, single-core: tile_size = 2*head_dim,
//   N_div_n = rows*n_heads/2 head-PAIR tiles.
//
// ── Shape constraints ────────────────────────────────────────────────────────
//   head_dim % 32 == 0  (VEC=16 over both halves; head_dim/2 % 16 == 0)
//   n_heads even        (a head pair must not straddle a row / position)
//   DFlash drafter: head_dim=128 (n_rot2=64), NH=32, NKV=8. OK.
//
// ── AIE traps observed in this tree (see dflash_attention_sc_bf16.cc) ────────
//   Runtime aie::broadcast<bfloat16>(scalar) miscompiles — inv_rms is applied as
//   a bfloat16 SCALAR operand to aie::mul (the headnorm_rope_bf16.cc pattern),
//   not via a bf16 broadcast. The float broadcast feeding aie::invsqrt is fine.
//   The per-head body is kept INLINE (no noinline helper) — a noinline wrapper
//   around a scalar reduction has been observed to miscompile here.

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr unsigned VEC = 16;
static const float HeadNorm_EPS = 1.0e-5f;

template <typename T, unsigned N>
static void dflash_headnorm_rope_batched_impl(const T *restrict input,
                                              const T *restrict coeff,
                                              T *restrict output,
                                              int32_t n) {
    const int32_t head_dim = n / 2;       // two heads per tile
    const int32_t n_rot2 = head_dim / 2;  // full rotation -> pairs = head_dim/2
    const T *restrict weight = coeff;
    const T *restrict cs = coeff + head_dim;

    for (int32_t h = 0; h < 2; ++h) {
        const T *restrict in_h = input + h * head_dim;
        T *restrict out_h = output + h * head_dim;

        // Pass 1: sum of squares in float32 (mirrors rms_norm_head_bf16.cc).
        ::aie::accum<accfloat, N> acc = ::aie::zeros<accfloat, N>();
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_MIN_ITERATION_COUNT(1)
        for (int32_t i = 0; i < head_dim; i += N) {
            ::aie::vector<T, N> v = ::aie::load_v<N>(in_h + i);
            acc = ::aie::add(acc, ::aie::mul_square(v));
        }
        float sum_sq = ::aie::reduce_add(acc.template to_vector<float>());
        // Vector invsqrt — the scalar form calls sqrtf, unavailable bare-metal.
        float inv_rms = ::aie::invsqrt(::aie::broadcast<float, N>(
            sum_sq / (float)head_dim + HeadNorm_EPS))[0];
        T inv_rms_s = (T)inv_rms;

        // Pass 2: normalize + rotate every pair (full neox — no passthrough).
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_MIN_ITERATION_COUNT(1)
        for (int32_t i = 0; i < n_rot2; i += N) {
            ::aie::vector<T, N> xv = ::aie::load_v<N>(in_h + i);
            ::aie::vector<T, N> yv = ::aie::load_v<N>(in_h + i + n_rot2);
            ::aie::vector<T, N> wx = ::aie::load_v<N>(weight + i);
            ::aie::vector<T, N> wy = ::aie::load_v<N>(weight + i + n_rot2);
            ::aie::vector<T, N> cv = ::aie::load_v<N>(cs + i);
            ::aie::vector<T, N> sv = ::aie::load_v<N>(cs + i + n_rot2);

            // Head norm: x_n = x * inv_rms * gamma
            ::aie::vector<T, N> xn =
                ::aie::mul(::aie::mul(xv, inv_rms_s).template to_vector<T>(), wx)
                    .template to_vector<T>();
            ::aie::vector<T, N> yn =
                ::aie::mul(::aie::mul(yv, inv_rms_s).template to_vector<T>(), wy)
                    .template to_vector<T>();

            // RoPE: [x_n*c - y_n*s, y_n*c + x_n*s]
            ::aie::store_v(out_h + i,
                           ::aie::sub(::aie::mul(xn, cv), ::aie::mul(yn, sv))
                               .template to_vector<T>());
            ::aie::store_v(out_h + i + n_rot2,
                           ::aie::add(::aie::mul(yn, cv), ::aie::mul(xn, sv))
                               .template to_vector<T>());
        }
    }
}

extern "C" {

void dflash_headnorm_rope_batched_bf16(bfloat16 *restrict input,
                                       bfloat16 *restrict coeff,
                                       bfloat16 *restrict output, int32_t n) {
    dflash_headnorm_rope_batched_impl<bfloat16, VEC>(input, coeff, output, n);
}

} // extern "C"
