// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched multi-token forward (prefill) for qwen4_exp.
//!
//! [`decode_step_into`](crate::trunk_gpu::decode_step_into) advances one position
//! and sweeps every weight once per token, so a prompt prefilled through it costs
//! `len` full weight sweeps (~5.8 tok/s on the 180B). This runs a chunk of `n`
//! tokens LAYER-MAJOR instead: every layer sees all `n` rows before the next layer
//! runs, so each linear is one `weight_gemm` and each routed expert is read once
//! for all the rows routed to it.
//!
//! Layer-major is exact, not an approximation: every piece of sequence state
//! (GDN recurrence + conv ring, PLE dilated conv ring, QSA KV cache) belongs to
//! ONE layer and is advanced in token order within that layer, which is all the
//! causal order the model has. The numbers still differ from the per-token path
//! in the last bits — Opus GEMMs quantise activations (W8A8/W4A8) where the GEMV
//! does not, and the expert sum is accumulated in a different order — so the two
//! are compared statistically (argmax, max|Δlogit|, PPL), never by hash.
//!
//! What still runs per row: the hyper-connection norm/collapse/inject, the PLE
//! tail, the QSA projection split and block selection + attention, and top-k
//! routing. None of them read a weight matrix, so they cost launches, not
//! bandwidth. Fuse them if a trace says they matter.

use crate::attn_gpu::{qsa_select_attend, QsaCache, QsaScratch, QsaWeights};
use crate::config::Qwen4ExpConfig;
use crate::gdn::{GdnState, GdnWeights};
use crate::hc_gpu::HcWeights;
use crate::moe_gpu::{capture_routed_input, ExpertStack, MoeWeights, ROUTING_TRACE};
use crate::ngram::NgramHasher;
use crate::ngram_rows::NgramRows;
use crate::ple_gpu::ple_tail;
use crate::trunk_gpu::{TokenMixer, TrunkScratch, TrunkState, TrunkWeights};
use hipfire_rdna::{DType, Gpu, GpuTensor, HipError, HipResult};
use hipfire_runtime::weight_pager::page_timing;
use hipfire_runtime::weights::{weight_gemm_oq4_act, weight_gemv, WeightTensor};

/// Every batched linear in this module. Oq4G256 runs at f16 ACTIVATIONS rather
/// than `weight_gemm`'s default small-batch int4: on the shipped 180B, held-out
/// PPL (4095 tokens) was 4.0868 at the default, 4.0709 at A8, 4.0550 at A16,
/// against 4.0411 per-token — and A16 was no slower (128 s vs 136 s).
fn weight_gemm(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    weight_gemm_oq4_act(gpu, w, x, y, n, Some(16))
}

/// Row `t` of a row-major `[rows, width]` buffer.
fn row(t: &GpuTensor, r: usize, width: usize) -> GpuTensor {
    t.sub_offset(r * width, width)
}

/// The first `n` rows of a row-major `[rows, width]` buffer.
fn rows(t: &GpuTensor, n: usize, width: usize) -> GpuTensor {
    t.sub_offset(0, n * width)
}

fn bits(v: usize) -> f32 {
    f32::from_bits(v as u32)
}

/// Chunk-wide scratch, `rows` tokens deep. Separate from [`TrunkScratch`] (one
/// row) so the decode path's buffers — and therefore its output — are untouched.
pub struct BatchScratch {
    rows: usize,
    wide: GpuTensor,
    mixed: GpuTensor,
    block_out: GpuTensor,
    collapsed: GpuTensor,
    // hyper-connections
    hc_normed: GpuTensor,
    hc_lowrank: GpuTensor,
    hc_mix: GpuTensor,
    hc_inject: GpuTensor,
    // PLE
    ple_key: GpuTensor,
    ple_value: GpuTensor,
    ple_out: GpuTensor,
    // Gated DeltaNet
    g_qkv: GpuTensor,
    g_z: GpuTensor,
    g_a: GpuTensor,
    g_b: GpuTensor,
    g_alpha: GpuTensor,
    g_beta: GpuTensor,
    g_q_raw: GpuTensor,
    g_k_raw: GpuTensor,
    g_v: GpuTensor,
    g_q: GpuTensor,
    g_k: GpuTensor,
    g_attn: GpuTensor,
    g_gated: GpuTensor,
    // sparse attention
    q_qg: GpuTensor,
    q_query: GpuTensor,
    q_gate: GpuTensor,
    q_qn: GpuTensor,
    q_k: GpuTensor,
    q_kn: GpuTensor,
    q_v: GpuTensor,
    q_ix_qk: GpuTensor,
    q_ix_q: GpuTensor,
    q_gated: GpuTensor,
    // MoE
    m_logits: GpuTensor,
    m_idx: GpuTensor,
    m_w: GpuTensor,
    /// Token rows grouped by expert, `[rows * k]` i32-as-f32 — one upload per layer.
    m_rows: GpuTensor,
    m_scale: GpuTensor,
    m_x: GpuTensor,
    m_gu: GpuTensor,
    m_inter: GpuTensor,
    m_eo: GpuTensor,
    m_sg: GpuTensor,
    m_su: GpuTensor,
    m_sinter: GpuTensor,
    m_sgate: GpuTensor,
    /// `[rows, vocab]`, only when a caller asks for every position's logits.
    logits: Option<GpuTensor>,
}

impl BatchScratch {
    pub fn new(gpu: &mut Gpu, cfg: &Qwen4ExpConfig, rows: usize) -> HipResult<Self> {
        let hc = cfg.gated_residual.count;
        let width = hc * cfg.hidden;
        let d = &cfg.deltanet;
        let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let ix = &cfg.indexer;
        let m = &cfg.moe;
        let k_span = d.key_heads * d.key_head_dim;
        let v_span = d.z_dim();
        let z = |g: &mut Gpu, per_row: usize| g.zeros(&[(rows * per_row).max(1)], DType::F32);
        Ok(Self {
            rows,
            wide: z(gpu, width)?,
            mixed: z(gpu, cfg.hidden)?,
            block_out: z(gpu, cfg.hidden)?,
            collapsed: z(gpu, cfg.hidden)?,
            hc_normed: z(gpu, width)?,
            hc_lowrank: z(gpu, cfg.gated_residual.lowrank)?,
            hc_mix: z(gpu, width)?,
            hc_inject: z(gpu, hc)?,
            ple_key: z(gpu, width)?,
            ple_value: z(gpu, cfg.hidden)?,
            ple_out: z(gpu, width)?,
            g_qkv: z(gpu, d.qkv_dim())?,
            g_z: z(gpu, v_span)?,
            g_a: z(gpu, d.value_heads)?,
            g_b: z(gpu, d.value_heads)?,
            g_alpha: z(gpu, d.value_heads)?,
            g_beta: z(gpu, d.value_heads)?,
            g_q_raw: z(gpu, k_span)?,
            g_k_raw: z(gpu, k_span)?,
            g_v: z(gpu, v_span)?,
            g_q: z(gpu, v_span)?,
            g_k: z(gpu, v_span)?,
            g_attn: z(gpu, v_span)?,
            g_gated: z(gpu, v_span)?,
            q_qg: z(gpu, nh * hd * 2)?,
            q_query: z(gpu, nh * hd)?,
            q_gate: z(gpu, nh * hd)?,
            q_qn: z(gpu, nh * hd)?,
            q_k: z(gpu, nkv * hd)?,
            q_kn: z(gpu, nkv * hd)?,
            q_v: z(gpu, nkv * hd)?,
            q_ix_qk: z(gpu, (ix.n_heads + ix.kv_heads) * ix.head_dim)?,
            q_ix_q: z(gpu, ix.n_heads * ix.head_dim)?,
            q_gated: z(gpu, nh * hd)?,
            m_logits: z(gpu, m.num_experts)?,
            m_idx: z(gpu, m.experts_per_tok)?,
            m_w: z(gpu, m.experts_per_tok)?,
            m_rows: z(gpu, m.experts_per_tok)?,
            m_scale: z(gpu, m.experts_per_tok)?,
            m_x: z(gpu, cfg.hidden)?,
            m_gu: z(gpu, 2 * m.intermediate)?,
            m_inter: z(gpu, m.intermediate)?,
            m_eo: z(gpu, cfg.hidden)?,
            m_sg: z(gpu, m.shared_intermediate)?,
            m_su: z(gpu, m.shared_intermediate)?,
            m_sinter: z(gpu, m.shared_intermediate)?,
            m_sgate: z(gpu, 1)?,
            logits: None,
        })
    }

    /// Tokens per chunk this scratch holds.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Row `r`'s final WIDE residual from the last [`prefill_chunk_into`] — the
    /// MTP head's trunk input for that position.
    pub fn wide_row(&self, cfg: &Qwen4ExpConfig, r: usize) -> GpuTensor {
        row(&self.wide, r, cfg.gated_residual.count * cfg.hidden)
    }

    /// The first `n` rows of the final WIDE residual (`[n, width]`).
    pub fn wide_rows(&self, cfg: &Qwen4ExpConfig, n: usize) -> GpuTensor {
        rows(&self.wide, n, cfg.gated_residual.count * cfg.hidden)
    }

    /// `[rows, vocab]` logits of the last [`prefill_chunk_into`] that asked for
    /// them; row `t` predicts the token after chunk position `t`.
    pub fn all_logits(&self) -> Option<&GpuTensor> {
        self.logits.as_ref()
    }

    pub fn free(self, gpu: &mut Gpu) {
        let Self {
            rows: _,
            wide,
            mixed,
            block_out,
            collapsed,
            hc_normed,
            hc_lowrank,
            hc_mix,
            hc_inject,
            ple_key,
            ple_value,
            ple_out,
            g_qkv,
            g_z,
            g_a,
            g_b,
            g_alpha,
            g_beta,
            g_q_raw,
            g_k_raw,
            g_v,
            g_q,
            g_k,
            g_attn,
            g_gated,
            q_qg,
            q_query,
            q_gate,
            q_qn,
            q_k,
            q_kn,
            q_v,
            q_ix_qk,
            q_ix_q,
            q_gated,
            m_logits,
            m_idx,
            m_w,
            m_rows,
            m_scale,
            m_x,
            m_gu,
            m_inter,
            m_eo,
            m_sg,
            m_su,
            m_sinter,
            m_sgate,
            logits,
        } = self;
        for t in [
            wide, mixed, block_out, collapsed, hc_normed, hc_lowrank, hc_mix, hc_inject, ple_key,
            ple_value, ple_out, g_qkv, g_z, g_a, g_b, g_alpha, g_beta, g_q_raw, g_k_raw, g_v, g_q,
            g_k, g_attn, g_gated, q_qg, q_query, q_gate, q_qn, q_k, q_kn, q_v, q_ix_qk, q_ix_q,
            q_gated, m_logits, m_idx, m_w, m_rows, m_scale, m_x, m_gu, m_inter, m_eo, m_sg, m_su,
            m_sinter, m_sgate,
        ]
        .into_iter()
        .chain(logits)
        {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Batched [`crate::hc_gpu::hc_read`] over `n` rows of `streams`.
fn hc_read_rows(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    w: &HcWeights,
    b: &BatchScratch,
    streams: &GpuTensor,
    mixed_out: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    let hc = cfg.gated_residual.count;
    let hidden = cfg.hidden;
    let width = hc * hidden;
    let lr = cfg.gated_residual.lowrank;
    let inv = 1.0 / hc as f32;

    // The norm weight spans ONE row's hc streams, so the kernel cannot take
    // n*hc groups — it would index the weight past its end.
    for t in 0..n {
        gpu.hc_grouped_rmsnorm(
            &row(streams, t, width),
            &w.hc_norm,
            &row(&b.hc_normed, t, width),
            hidden as i32,
            hc as i32,
            cfg.rms_norm_eps,
        )?;
    }
    weight_gemm(gpu, &w.mix_down, &b.hc_normed, &b.hc_lowrank, n)?;
    gpu.hc_scaled_silu(&b.hc_lowrank, &b.hc_lowrank, (n * lr) as i32, inv)?;
    weight_gemm(gpu, &w.mix_up, &b.hc_lowrank, &b.hc_mix, n)?;
    gpu.hc_sigmoid(&b.hc_mix, &b.hc_mix, (n * width) as i32)?;
    for t in 0..n {
        gpu.hc_input_map_perchannel(
            &row(&b.hc_mix, t, width),
            &row(&b.hc_normed, t, width),
            &row(mixed_out, t, hidden),
            hidden as i32,
            hc as i32,
        )?;
    }
    if let Some(bi) = w.block_inject.as_ref() {
        weight_gemm(gpu, bi, &b.hc_normed, &b.hc_inject, n)?;
        gpu.hc_inject_gate(&b.hc_inject, &b.hc_inject, (n * hc) as i32, inv)?;
    }
    Ok(())
}

/// Batched [`crate::hc_gpu::hc_write`]: needs the inject gates the matching
/// [`hc_read_rows`] left in `b.hc_inject`.
fn hc_write_rows(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    b: &BatchScratch,
    streams: &GpuTensor,
    block_out: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    let hc = cfg.gated_residual.count;
    let width = hc * cfg.hidden;
    for t in 0..n {
        gpu.hc_residual_inject(
            &row(streams, t, width),
            &row(block_out, t, cfg.hidden),
            &row(&b.hc_inject, t, hc),
            cfg.hidden as i32,
            hc as i32,
        )?;
    }
    Ok(())
}

/// Batched [`crate::gdn::gdn_decode_step`]: the recurrence and the conv ring both
/// advance `n` steps inside one launch, in token order.
#[allow(clippy::too_many_arguments)]
fn gdn_rows(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    w: &GdnWeights,
    b: &BatchScratch,
    st: &GdnState,
    x: &GpuTensor,
    y: &GpuTensor,
    n: usize,
    snap: Option<(&GpuTensor, &GpuTensor)>,
) -> HipResult<()> {
    let d = &cfg.deltanet;
    let hd = d.key_head_dim;
    let vh = d.value_heads;

    weight_gemm(gpu, &w.in_proj_qkv, x, &b.g_qkv, n)?;
    weight_gemm(gpu, &w.in_proj_z, x, &b.g_z, n)?;
    weight_gemm(gpu, &w.in_proj_a, x, &b.g_a, n)?;
    weight_gemm(gpu, &w.in_proj_b, x, &b.g_b, n)?;

    gpu.memcpy_dtod_auto(&b.g_beta.buf, &b.g_b.buf, n * vh * 4)?;
    gpu.memcpy_dtod_auto(&b.g_alpha.buf, &b.g_a.buf, n * vh * 4)?;
    gpu.fused_sigmoid_alpha_gate_f32_batched(&b.g_beta, &b.g_alpha, &w.dt_bias, &w.a_log, vh, n)?;

    // With `snap`, the per-row state checkpoints land in it (row t's state
    // AFTER token t), which is what speculative rollback restores from.
    gpu.conv1d_silu_split_f32_n_opts(
        &b.g_q_raw,
        &b.g_k_raw,
        &b.g_v,
        &b.g_qkv,
        &w.conv_weight,
        &st.conv,
        d.key_heads * hd,
        d.z_dim(),
        n,
        snap.map(|(_, conv)| conv),
    )?;
    gpu.fused_qk_l2_norm_scale_f32_batched(
        &b.g_q_raw,
        &b.g_k_raw,
        d.key_heads,
        hd,
        1.0 / (hd as f32).sqrt(),
        cfg.rms_norm_eps,
        n,
    )?;
    let ratio = d.value_per_key();
    if ratio > 1 {
        gpu.repeat_interleave_qk_f32_batched(
            &b.g_q_raw,
            &b.g_k_raw,
            &b.g_q,
            &b.g_k,
            d.key_heads,
            ratio,
            hd,
            n,
        )?;
    } else {
        let bytes = n * d.key_heads * hd * 4;
        gpu.memcpy_dtod_auto(&b.g_q.buf, &b.g_q_raw.buf, bytes)?;
        gpu.memcpy_dtod_auto(&b.g_k.buf, &b.g_k_raw.buf, bytes)?;
    }
    match snap {
        None => gpu.gated_delta_net_f32(
            &b.g_q,
            &b.g_k,
            &b.g_v,
            &b.g_alpha,
            &b.g_beta,
            &st.recurrent,
            &b.g_attn,
            n,
            vh,
            d.value_head_dim,
        )?,
        Some((rec, _)) => gpu.gated_delta_net_f32_batch_seq_opts(
            &b.g_q,
            &b.g_k,
            &b.g_v,
            &b.g_alpha,
            &b.g_beta,
            &st.recurrent,
            &b.g_attn,
            n,
            vh,
            d.value_head_dim,
            Some(rec),
        )?,
    }
    if d.output_gate_sigmoid {
        gpu.gated_norm_sigmoid_f32(
            &b.g_attn,
            &b.g_z,
            &w.norm_weight,
            &b.g_gated,
            vh as i32,
            d.value_head_dim as i32,
            n as i32,
            cfg.rms_norm_eps,
        )?;
    } else {
        let span = d.z_dim();
        for t in 0..n {
            gpu.gated_norm_f32(
                &row(&b.g_attn, t, span),
                &row(&b.g_z, t, span),
                &w.norm_weight,
                &row(&b.g_gated, t, span),
                vh,
                d.value_head_dim,
                cfg.rms_norm_eps,
            )?;
        }
    }
    weight_gemm(gpu, &w.out_proj, &b.g_gated, y, n)
}

/// Batched [`crate::attn_gpu::qsa_decode_step`]. The whole chunk's K/V/indexer
/// keys are written first; each row then selects over `visible[..=pos]`, which
/// never names a later position, so writing ahead is exact.
#[allow(clippy::too_many_arguments)]
fn qsa_rows(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    w: &QsaWeights,
    b: &BatchScratch,
    s: &QsaScratch,
    cache: &mut QsaCache,
    x: &GpuTensor,
    out: &GpuTensor,
    pos0: usize,
    n: usize,
    visible: &GpuTensor,
) -> HipResult<()> {
    qsa_project_write(gpu, cfg, w, b, cache, x, pos0, n, visible)?;
    let (qw, ix_qw) = (
        cfg.n_heads * cfg.head_dim,
        cfg.indexer.n_heads * cfg.indexer.head_dim,
    );
    for t in 0..n {
        let n_vis = pos0 + t + 1;
        qsa_select_attend(
            gpu,
            cfg,
            w,
            s,
            cache,
            &row(&b.q_qn, t, qw),
            &row(&b.q_ix_q, t, ix_qw),
            &row(&b.q_gate, t, qw),
            &visible.sub_offset(0, n_vis),
            n_vis,
            &row(&b.q_gated, t, qw),
        )?;
    }
    weight_gemm(gpu, &w.o_proj, &b.q_gated, out, n)
}

/// The projection half of [`qsa_rows`]: every row's query (normed + rotated,
/// left in `b`) and its K/V/indexer key WRITTEN to `cache`. On its own it is
/// what an MTP head needs for positions whose output nobody reads — a later
/// position's attention only ever sees this layer's cache, never its output.
#[allow(clippy::too_many_arguments)]
fn qsa_project_write(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    w: &QsaWeights,
    b: &BatchScratch,
    cache: &mut QsaCache,
    x: &GpuTensor,
    pos0: usize,
    n: usize,
    visible: &GpuTensor,
) -> HipResult<()> {
    let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let ix = &cfg.indexer;
    let eps = cfg.rms_norm_eps;
    let ms = cache.max_seq();
    let (qw, kvw) = (nh * hd, nkv * hd);
    let ix_row = (ix.n_heads + ix.kv_heads) * ix.head_dim;
    let ix_qw = ix.n_heads * ix.head_dim;

    weight_gemm(gpu, &w.q_proj, x, &b.q_qg, n)?;
    for t in 0..n {
        gpu.qsa_split_query_gate(
            &row(&b.q_qg, t, 2 * qw),
            &row(&b.q_query, t, qw),
            &row(&b.q_gate, t, qw),
            nh as i32,
            hd as i32,
        )?;
    }
    // One shared per-head weight, so n rows of heads are just more heads.
    gpu.rms_norm_heads_shared_w(
        &rows(&b.q_query, n, qw),
        &w.q_norm,
        &rows(&b.q_qn, n, qw),
        hd as i32,
        (n * nh) as i32,
        eps,
    )?;
    weight_gemm(gpu, &w.k_proj, x, &b.q_k, n)?;
    gpu.rms_norm_heads_shared_w(
        &rows(&b.q_k, n, kvw),
        &w.k_norm,
        &rows(&b.q_kn, n, kvw),
        hd as i32,
        (n * nkv) as i32,
        eps,
    )?;
    weight_gemm(gpu, &w.v_proj, x, &b.q_v, n)?;

    // `visible` is 0..pos0+n in the same i32-as-f32 encoding RoPE reads.
    let pos_buf = visible.sub_offset(pos0, n);
    gpu.rope_partial_interleaved_f32_batched(
        &b.q_qn,
        &b.q_kn,
        &pos_buf,
        nh,
        nkv,
        hd,
        cfg.rotary_dim(),
        cfg.rotary_dim(),
        cfg.rope_theta,
        n,
        0,
    )?;

    weight_gemm(gpu, &w.ix_qk_proj, x, &b.q_ix_qk, n)?;
    for t in 0..n {
        let p = (pos0 + t) as i32;
        gpu.qsa_cache_write(
            &row(&b.q_kn, t, kvw),
            &cache.k,
            p,
            nkv as i32,
            hd as i32,
            ms as i32,
        )?;
        gpu.qsa_cache_write(
            &row(&b.q_v, t, kvw),
            &cache.v,
            p,
            nkv as i32,
            hd as i32,
            ms as i32,
        )?;
        // The indexer key is cached RAW, exactly as the decode step does.
        let ix_key = b.q_ix_qk.sub_offset(t * ix_row + ix_qw, ix.head_dim);
        gpu.qsa_cache_write(&ix_key, &cache.ix_keys, p, 1, ix.head_dim as i32, ms as i32)?;
        gpu.rms_norm_heads_shared_w(
            &row(&b.q_ix_qk, t, ix_row),
            &w.ix_q_norm,
            &row(&b.q_ix_q, t, ix_qw),
            ix.head_dim as i32,
            ix.n_heads as i32,
            eps,
        )?;
    }
    cache.len = cache.len.max(pos0 + n);
    gpu.rope_partial_interleaved_f32_batched(
        &b.q_ix_q,
        &b.q_ix_q,
        &pos_buf,
        ix.n_heads,
        0,
        ix.head_dim,
        ix.head_dim,
        ix.head_dim,
        cfg.rope_theta,
        n,
        0,
    )
}

/// Batched [`crate::moe_gpu::moe_forward`]. Routing runs per row; the rows are
/// then grouped by expert on the host, so each expert routed to at least once
/// runs ONE gather → gate_up GEMM → SwiGLU → down GEMM → scaled scatter.
#[allow(clippy::too_many_arguments)]
fn moe_rows(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    w: &MoeWeights,
    b: &BatchScratch,
    x: &GpuTensor,
    out: &GpuTensor,
    n: usize,
    layer: Option<usize>,
) -> HipResult<()> {
    let m = &cfg.moe;
    let (hidden, mi, ne, k) = (cfg.hidden, m.intermediate, m.num_experts, m.experts_per_tok);
    let smi = m.shared_intermediate;

    // ── shared expert first: its output SEEDS `out`, so the routed scatter can
    // accumulate without a separate zero-fill.
    weight_gemm(gpu, &w.shared_gate, x, &b.m_sg, n)?;
    weight_gemm(gpu, &w.shared_up, x, &b.m_su, n)?;
    gpu.silu_mul_f32(
        &rows(&b.m_sg, n, smi),
        &rows(&b.m_su, n, smi),
        &rows(&b.m_sinter, n, smi),
    )?;
    weight_gemm(gpu, &w.shared_down, &b.m_sinter, out, n)?;
    weight_gemm(gpu, &w.shared_expert_gate, x, &b.m_sgate, n)?;
    for t in 0..n {
        let o = row(out, t, hidden);
        gpu.moe_shared_gate(&o, &row(&b.m_sgate, t, 1), &o, hidden as i32)?;
    }

    // ── routing
    weight_gemm(gpu, &w.router, x, &b.m_logits, n)?;
    for t in 0..n {
        gpu.moe_softmax_topk_renorm(
            &row(&b.m_logits, t, ne),
            &row(&b.m_idx, t, k),
            &row(&b.m_w, t, k),
            ne,
            m.norm_topk_prob,
            k,
        )?;
    }
    let idx: Vec<usize> = gpu.download_f32(&rows(&b.m_idx, n, k))?[..n * k]
        .iter()
        .map(|v| (v.to_bits() as i32).max(0) as usize)
        .collect();
    let wts: Vec<f32> = gpu.download_f32(&rows(&b.m_w, n, k))?[..n * k].to_vec();

    use hipfire_runtime::dispatch::moe_telemetry as telemetry;
    if let Some(layer) = layer.filter(|_| telemetry::moe_router_histogram_active()) {
        for t in 0..n {
            let r = t * k..(t + 1) * k;
            telemetry::record_moe_router_selection(layer, t, &idx[r.clone()], &wts[r]);
        }
    }
    // Call order here is layer-major (all rows of layer l, then l+1), where the
    // decode path appends token-major. Probes that index the trace by token must
    // drive decode, not prefill.
    if let Some(tr) = ROUTING_TRACE
        .lock()
        .unwrap()
        .as_mut()
        .filter(|_| layer.is_some())
    {
        tr.ids.extend(idx.iter().map(|&e| e as u16));
        tr.logits
            .extend(gpu.download_f32(&rows(&b.m_logits, n, ne))?);
    }

    // ── group rows by expert: one host pass, one upload of each table.
    let mut by_expert: Vec<Vec<(usize, f32)>> = vec![Vec::new(); ne];
    for (slot, (&e, &wt)) in idx.iter().zip(&wts).enumerate() {
        if e >= ne {
            return Err(HipError::new(
                0,
                &format!("qwen4_exp prefill: router picked expert {e} of {ne}"),
            ));
        }
        by_expert[e].push((slot / k, wt));
    }
    let mut row_ids: Vec<f32> = Vec::with_capacity(n * k);
    let mut scales: Vec<f32> = Vec::with_capacity(n * k);
    let mut spans: Vec<(usize, usize, usize)> = Vec::new();
    for (e, picks) in by_expert.iter().enumerate() {
        if picks.is_empty() {
            continue;
        }
        spans.push((e, row_ids.len(), picks.len()));
        for &(t, wt) in picks {
            row_ids.push(bits(t));
            scales.push(wt);
        }
    }
    let as_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
    gpu.memcpy_htod_auto(&b.m_rows.buf, &as_bytes(&row_ids))?;
    gpu.memcpy_htod_auto(&b.m_scale.buf, &as_bytes(&scales))?;

    for &(e, start, cnt) in &spans {
        let ids = b.m_rows.sub_offset(start, cnt);
        let sc = b.m_scale.sub_offset(start, cnt);
        let xg = rows(&b.m_x, cnt, hidden);
        let gu_out = rows(&b.m_gu, cnt, 2 * mi);
        let inter = rows(&b.m_inter, cnt, mi);
        let eo = rows(&b.m_eo, cnt, hidden);
        gpu.zaya_gather_rows_f32(&xg, x, &ids.buf, hidden, cnt * hidden)?;
        // Ensure residency immediately before use: with a tight budget, paging in
        // a later expert may evict this one.
        let (gu, dn) = ExpertStack::expert_pair(&w.gate_up, &w.down, gpu, e)?;
        // A single routed row is a GEMV; the GEMM tiles would waste 15/16 of
        // their columns on it.
        let lin = |gpu: &mut Gpu, wt: &WeightTensor, x: &GpuTensor, y: &GpuTensor| {
            if cnt == 1 {
                weight_gemv(gpu, wt, x, y)
            } else {
                weight_gemm(gpu, wt, x, y, cnt)
            }
        };
        capture_routed_input(gpu, layer, e, "gate_up_proj", &xg, cnt, hidden);
        page_timing::timed(page_timing::Phase::ExpertGemv, || {
            lin(gpu, &gu, &xg, &gu_out)
        })?;
        // Contiguous halves per row, gate FIRST — same convention as decode.
        gpu.zaya_silu_mul_gate_up_f32(&inter, &gu_out, mi, cnt * mi)?;
        capture_routed_input(gpu, layer, e, "down_proj", &inter, cnt, mi);
        page_timing::timed(page_timing::Phase::ExpertGemv, || {
            lin(gpu, &dn, &inter, &eo)
        })?;
        // Each row appears at most once per expert (top-k picks are distinct),
        // which is the scatter's no-atomics precondition.
        gpu.zaya_scatter_scaled_add_f32(out, &eo, &ids.buf, &sc, hidden, cnt * hidden)?;
    }
    page_timing::admission(n * k);
    Ok(())
}

/// Run positions `pos0 .. history.len()` through the trunk as one chunk.
///
/// `history` is the WHOLE token stream through the end of the chunk (the n-gram
/// window reaches back across chunk boundaries). Leaves, exactly as
/// [`crate::trunk_gpu::decode_step_into`] would after the chunk's last token:
/// `s.logits`, `s.wide`, `s.collapsed`. With `all_logits`, every row's logits
/// also land in [`BatchScratch::all_logits`].
#[allow(clippy::too_many_arguments)]
pub fn prefill_chunk_into(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    w: &TrunkWeights,
    st: &mut TrunkState,
    s: &mut TrunkScratch,
    b: &mut BatchScratch,
    embed_table: &[f32],
    ngram_rows: Option<&dyn NgramRows>,
    history: &[u32],
    pos0: usize,
    eos: u32,
    all_logits: bool,
    ckpt: Option<&Row0Checkpoint>,
) -> HipResult<()> {
    let n = history.len().saturating_sub(pos0);
    if ckpt.is_some() && n > Row0Checkpoint::ROWS {
        return Err(HipError::new(
            0,
            "qwen4_exp: a row-0 checkpoint covers 2-row verifies only",
        ));
    }
    if n == 0 || n > b.rows {
        return Err(HipError::new(
            0,
            &format!(
                "qwen4_exp prefill chunk of {n} rows (scratch holds {})",
                b.rows
            ),
        ));
    }
    let (hidden, hc) = (cfg.hidden, cfg.gated_residual.count);
    let width = hc * hidden;

    // Seed every stream with a COPY of the embedding, as decode does.
    let mut seeded: Vec<f32> = Vec::with_capacity(n * width);
    for &tok in &history[pos0..] {
        let e = &embed_table[tok as usize * hidden..(tok as usize + 1) * hidden];
        for _ in 0..hc {
            seeded.extend_from_slice(e);
        }
    }
    let up = gpu.upload_f32(&seeded, &[n * width])?;
    gpu.memcpy_dtod_auto(&b.wide.buf, &up.buf, n * width * 4)?;
    gpu.free_tensor(up)?;

    // One causal position list for the chunk; row t sees its prefix.
    let vis: Vec<f32> = (0..pos0 + n).map(bits).collect();
    let visible = gpu.upload_f32(&vis, &[vis.len()])?;

    let wide = rows(&b.wide, n, width);
    let result = (|| -> HipResult<()> {
        for l in 0..cfg.layers {
            let lw = &w.layers[l];

            if let (Some(pw), Some(ps), Some(ng), Some(src)) = (
                lw.ple.as_ref(),
                st.ple[l].as_mut(),
                cfg.ngram.as_ref(),
                ngram_rows,
            ) {
                let hasher = NgramHasher::from_config(ng, cfg.vocab as u64, eos);
                let hd = ng.head_dim();
                let ctx_len = ng.ngram_size - 1;
                let mut hist: Vec<u32> = vec![eos; ctx_len];
                hist.extend_from_slice(history);
                let mut emb: Vec<f32> = Vec::with_capacity(n * ng.embed_dim);
                for t in 0..n {
                    let i = ctx_len + pos0 + t;
                    let preds: Vec<Option<u32>> = hist[..i].iter().map(|&v| Some(v)).collect();
                    let r = hasher.rows(hist[i], &preds);
                    emb.extend(src.gather(&r, hd).map_err(|e| HipError::new(0, &e))?);
                }
                let g_emb = gpu.upload_f32(&emb, &[emb.len()])?;
                let r = (|| -> HipResult<()> {
                    weight_gemm(gpu, &pw.key_proj, &g_emb, &b.ple_key, n)?;
                    weight_gemm(gpu, &pw.value_proj, &g_emb, &b.ple_value, n)?;
                    for t in 0..n {
                        ple_tail(
                            gpu,
                            cfg,
                            pw,
                            ps,
                            &row(&b.wide, t, width),
                            &row(&b.ple_key, t, width),
                            &row(&b.ple_value, t, hidden),
                            &row(&b.ple_out, t, width),
                        )?;
                        // The dilated ring is per row, so its row-0 checkpoint is
                        // a copy taken right here.
                        if let (0, Some(c)) = (t, ckpt.and_then(|c| c.ple[l].as_ref())) {
                            let bytes = ps.conv_state.numel() * 4;
                            gpu.memcpy_dtod_auto(&c.buf, &ps.conv_state.buf, bytes)?;
                        }
                    }
                    gpu.add_inplace_f32(&wide, &rows(&b.ple_out, n, width))
                })();
                gpu.free_tensor(g_emb)?;
                r?;
            }

            hc_read_rows(gpu, cfg, &lw.attn_hc, b, &b.wide, &b.mixed, n)?;
            match &lw.mixer {
                TokenMixer::Gdn(gw) => {
                    let gs = st.gdn[l].as_ref().expect("gdn state on a linear layer");
                    let snap = ckpt.map(|c| {
                        (
                            c.rec[l].as_ref().expect("gdn checkpoint"),
                            c.conv[l].as_ref().expect("gdn checkpoint"),
                        )
                    });
                    gdn_rows(gpu, cfg, gw, b, gs, &b.mixed, &b.block_out, n, snap)?;
                }
                TokenMixer::Qsa(qw) => {
                    let cache = st.qsa[l].as_mut().expect("kv cache on a sparse-attn layer");
                    qsa_rows(
                        gpu,
                        cfg,
                        qw,
                        b,
                        &s.qsa,
                        cache,
                        &b.mixed,
                        &b.block_out,
                        pos0,
                        n,
                        &visible,
                    )?;
                }
            }
            hc_write_rows(gpu, cfg, b, &b.wide, &b.block_out, n)?;

            hc_read_rows(gpu, cfg, &lw.mlp_hc, b, &b.wide, &b.mixed, n)?;
            moe_rows(gpu, cfg, &lw.moe, b, &b.mixed, &b.block_out, n, Some(l))?;
            hc_write_rows(gpu, cfg, b, &b.wide, &b.block_out, n)?;
        }

        hc_read_rows(gpu, cfg, &w.mixer, b, &b.wide, &b.collapsed, n)?;
        let last = n - 1;
        gpu.memcpy_dtod_auto(&s.wide.buf, &row(&b.wide, last, width).buf, width * 4)?;
        gpu.memcpy_dtod_auto(
            &s.collapsed.buf,
            &row(&b.collapsed, last, hidden).buf,
            hidden * 4,
        )?;
        if all_logits {
            if b.logits.is_none() {
                b.logits = Some(gpu.zeros(&[b.rows * cfg.vocab], DType::F32)?);
            }
            let lg = b.logits.as_ref().expect("just allocated");
            weight_gemm(gpu, &w.lm_head, &b.collapsed, lg, n)?;
            gpu.memcpy_dtod_auto(&s.logits.buf, &row(lg, last, cfg.vocab).buf, cfg.vocab * 4)
        } else {
            weight_gemv(gpu, &w.lm_head, &s.collapsed, &s.logits)
        }
    })();
    gpu.free_tensor(visible)?;
    result
}

/// The trunk's sequence state as it stood after ROW 0 of a 2-row verify, so a
/// rejected draft (row 1) can be undone without re-running row 0.
///
/// GDN recurrence and conv ring come from the kernels' own per-row snapshot
/// outputs (the DFlash rollback mechanism); the PLE ring runs per row, so it is
/// copied after row 0; QSA caches only need their length rewound — a position
/// past it is never visible and is overwritten on the next write.
pub struct Row0Checkpoint {
    rec: Vec<Option<GpuTensor>>,
    conv: Vec<Option<GpuTensor>>,
    ple: Vec<Option<GpuTensor>>,
}

impl Row0Checkpoint {
    /// Rows a checkpointed forward may carry: the token and one draft.
    pub const ROWS: usize = 2;

    pub fn new(gpu: &mut Gpu, cfg: &Qwen4ExpConfig, st: &TrunkState) -> HipResult<Self> {
        let n = Self::ROWS;
        let (mut rec, mut conv, mut ple) = (Vec::new(), Vec::new(), Vec::new());
        for l in 0..cfg.layers {
            match st.gdn[l].as_ref() {
                Some(g) => {
                    rec.push(Some(gpu.zeros(&[n * g.recurrent.numel()], DType::F32)?));
                    conv.push(Some(gpu.zeros(&[n * g.conv.numel()], DType::F32)?));
                }
                None => {
                    rec.push(None);
                    conv.push(None);
                }
            }
            ple.push(match st.ple[l].as_ref() {
                Some(p) => Some(gpu.zeros(&[p.conv_state.numel()], DType::F32)?),
                None => None,
            });
        }
        Ok(Self { rec, conv, ple })
    }

    /// Put `st` back to its state after row 0; `len` is the sequence length that
    /// row 0 ended (its position + 1).
    pub fn restore(&self, gpu: &mut Gpu, st: &mut TrunkState, len: usize) -> HipResult<()> {
        for l in 0..st.gdn.len() {
            if let (Some(g), Some(r), Some(c)) = (
                st.gdn[l].as_ref(),
                self.rec[l].as_ref(),
                self.conv[l].as_ref(),
            ) {
                // Slot 0 is the first `numel` floats of each snapshot.
                gpu.memcpy_dtod_auto(&g.recurrent.buf, &r.buf, g.recurrent.numel() * 4)?;
                gpu.memcpy_dtod_auto(&g.conv.buf, &c.buf, g.conv.numel() * 4)?;
            }
            if let (Some(p), Some(c)) = (st.ple[l].as_ref(), self.ple[l].as_ref()) {
                gpu.memcpy_dtod_auto(&p.conv_state.buf, &c.buf, p.conv_state.numel() * 4)?;
            }
            if let Some(q) = st.qsa[l].as_mut() {
                q.len = q.len.min(len);
            }
        }
        Ok(())
    }

    pub fn free(self, gpu: &mut Gpu) {
        let Self { rec, conv, ple } = self;
        for t in rec.into_iter().chain(conv).chain(ple).flatten() {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Write the MTP head's attention cache for positions `pos0 .. pos0+n` WITHOUT
/// running its attention, MoE or mixer.
///
/// The head is one trunk layer over a fused input, and a position's cache entry
/// depends only on that position's fused input — never on an earlier
/// position's OUTPUT. So the prompt (and a speculative step that was accepted
/// without the head drafting from it) only needs fuse → attn read → K/V/indexer
/// write. `trunk_wide` is `[n, width]`, `embeds` the `n` next-token embeddings.
#[allow(clippy::too_many_arguments)]
pub fn mtp_cache_rows(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    head: &crate::mtp_gpu::MtpWeightsGpu,
    hb: &BatchScratch,
    cache: &mut QsaCache,
    trunk_wide: &GpuTensor,
    embeds: &[f32],
    pos0: usize,
    n: usize,
    fusion: crate::mtp::Fusion,
) -> HipResult<()> {
    let (hidden, hc) = (cfg.hidden, cfg.gated_residual.count);
    let width = hc * hidden;
    let eps = cfg.rms_norm_eps;
    if n == 0 || n > hb.rows {
        return Err(HipError::new(0, "qwen4_exp mtp_cache_rows: bad row count"));
    }
    // Same arithmetic as `mtp_gpu::fuse_inputs`, batched: `fc_hidden` runs on
    // each of the n*hc streams (one GEMM over [n*hc, hidden]), `fc_embedding`
    // once per row, and the embedding projection is added per `fusion`.
    for t in 0..n {
        gpu.hc_grouped_rmsnorm(
            &row(trunk_wide, t, width),
            &head.pre_fc_norm_hidden,
            &row(&hb.ple_out, t, width),
            hidden as i32,
            hc as i32,
            eps,
        )?;
    }
    let e = gpu.upload_f32(embeds, &[n * hidden])?;
    let r = (|| -> HipResult<()> {
        for t in 0..n {
            gpu.hc_grouped_rmsnorm(
                &row(&e, t, hidden),
                &head.pre_fc_norm_embedding,
                &row(&hb.mixed, t, hidden),
                hidden as i32,
                1,
                eps,
            )?;
        }
        weight_gemm(gpu, &head.fc_embedding, &hb.mixed, &hb.block_out, n)?;
        weight_gemm(gpu, &head.fc_hidden, &hb.ple_out, &hb.wide, n * hc)?;
        for t in 0..n {
            for st in 0..hc {
                let take = match fusion {
                    crate::mtp::Fusion::BroadcastAllStreams => true,
                    crate::mtp::Fusion::Stream0Only => st == 0,
                };
                if take {
                    gpu.add_inplace_f32(
                        &row(&hb.wide, t * hc + st, hidden),
                        &row(&hb.block_out, t, hidden),
                    )?;
                }
            }
        }
        Ok(())
    })();
    gpu.free_tensor(e)?;
    r?;
    hc_read_rows(gpu, cfg, &head.attn_hc, hb, &hb.wide, &hb.mixed, n)?;
    let vis: Vec<f32> = (0..pos0 + n).map(bits).collect();
    let visible = gpu.upload_f32(&vis, &[vis.len()])?;
    let r = qsa_project_write(gpu, cfg, &head.qsa, hb, cache, &hb.mixed, pos0, n, &visible);
    gpu.free_tensor(visible)?;
    r
}
