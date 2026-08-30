// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! The QSA attention block on the GPU, decode-shaped. Mirrors
//! [`crate::attn::QsaAttention`] and [`crate::attn::Indexer`].
//!
//! There is **no new attention kernel**: `attention_cold_slots` already attends an
//! arbitrary compacted slot set with no causal assumption, which is exactly what a
//! sparse selection produces. The block is therefore
//! project → norm → rotate → cache → select → gather → that kernel → gate → project.
//!
//! The KV cache is slot-major f32 (`attention_cold_slots` layout 0), sized to
//! `max_seq`. Quantised KV is a later concern; correctness first.

use crate::config::Qwen4ExpConfig;
use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

pub struct QsaWeights {
    /// `[n_heads * head_dim * 2, hidden]` — query AND output gate.
    pub q_proj: GpuTensor,
    /// `[n_kv * head_dim, hidden]`
    pub k_proj: GpuTensor,
    pub v_proj: GpuTensor,
    /// `[hidden, n_heads * head_dim]`
    pub o_proj: GpuTensor,
    /// `[head_dim]`, shared across heads
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    /// `[(ix_heads + ix_kv) * ix_head_dim, hidden]`
    pub ix_qk_proj: GpuTensor,
    /// `[ix_head_dim]` each
    pub ix_q_norm: GpuTensor,
    pub ix_k_norm: GpuTensor,
}

/// Per-sequence cache and scratch. `max_seq` bounds the whole thing.
pub struct QsaCache {
    pub k: GpuTensor,
    pub v: GpuTensor,
    /// `[max_seq, ix_head_dim]` — the indexer's own RAW keys (un-normed,
    /// un-rotated; both are applied to the POOLED key, not the per-token one).
    pub ix_keys: GpuTensor,
    pub len: usize,
    max_seq: usize,
}

impl QsaCache {
    pub fn new(gpu: &mut Gpu, cfg: &Qwen4ExpConfig, max_seq: usize) -> HipResult<Self> {
        let ix = &cfg.indexer;
        Ok(Self {
            k: gpu.zeros(&[cfg.n_kv_heads * max_seq * cfg.head_dim], DType::F32)?,
            v: gpu.zeros(&[cfg.n_kv_heads * max_seq * cfg.head_dim], DType::F32)?,
            ix_keys: gpu.zeros(&[max_seq * ix.head_dim], DType::F32)?,
            len: 0,
            max_seq,
        })
    }
}

pub struct QsaScratch {
    qg: GpuTensor,
    query: GpuTensor,
    gate: GpuTensor,
    qn: GpuTensor,
    k: GpuTensor,
    kn: GpuTensor,
    v: GpuTensor,
    ix_qk: GpuTensor,
    ix_q: GpuTensor,
    block_keys: GpuTensor,
    starts: GpuTensor,
    scores: GpuTensor,
    block_mask: GpuTensor,
    slots: GpuTensor,
    count: GpuTensor,
    gath_k: GpuTensor,
    gath_v: GpuTensor,
    ctx: GpuTensor,
    gated: GpuTensor,
    m: GpuTensor,
    l: GpuTensor,
}

impl QsaScratch {
    pub fn new(gpu: &mut Gpu, cfg: &Qwen4ExpConfig, max_seq: usize) -> HipResult<Self> {
        let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let ix = &cfg.indexer;
        let max_blocks = max_seq / ix.compress_ratio + 1;
        let z = |g: &mut Gpu, n: usize| g.zeros(&[n.max(1)], DType::F32);
        Ok(Self {
            qg: z(gpu, nh * hd * 2)?,
            query: z(gpu, nh * hd)?,
            gate: z(gpu, nh * hd)?,
            qn: z(gpu, nh * hd)?,
            k: z(gpu, nkv * hd)?,
            kn: z(gpu, nkv * hd)?,
            v: z(gpu, nkv * hd)?,
            ix_qk: z(gpu, (ix.n_heads + ix.kv_heads) * ix.head_dim)?,
            ix_q: z(gpu, ix.n_heads * ix.head_dim)?,
            block_keys: z(gpu, max_blocks * ix.head_dim)?,
            starts: z(gpu, max_blocks)?,
            scores: z(gpu, max_blocks)?,
            block_mask: z(gpu, max_blocks)?,
            slots: z(gpu, max_seq)?,
            count: z(gpu, 1)?,
            gath_k: z(gpu, nkv * max_seq * hd)?,
            gath_v: z(gpu, nkv * max_seq * hd)?,
            ctx: z(gpu, nh * hd)?,
            gated: z(gpu, nh * hd)?,
            m: z(gpu, nh)?,
            l: z(gpu, nh)?,
        })
    }
}

/// One decode step. `x` is `[hidden]`, `out` is `[hidden]`. Appends to the cache.
///
/// `visible_positions` is the caller's causal/paged view of the cache — the
/// selection is over THOSE positions, so block geometry follows the mask rather
/// than a fixed grid.
pub fn qsa_decode_step(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    w: &QsaWeights,
    s: &mut QsaScratch,
    cache: &mut QsaCache,
    x: &GpuTensor,
    pos: usize,
    visible_positions: &[usize],
    out: &GpuTensor,
) -> HipResult<()> {
    let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let ix = &cfg.indexer;
    let eps = cfg.rms_norm_eps;
    let ms = cache.max_seq;

    // Query and its output gate share one doubled projection, per head.
    gpu.gemv_f32(&w.q_proj, x, &s.qg)?;
    gpu.qsa_split_query_gate(&s.qg, &s.query, &s.gate, nh as i32, hd as i32)?;
    gpu.rms_norm_heads_shared_w(&s.query, &w.q_norm, &s.qn, hd as i32, nh as i32, eps)?;
    gpu.gemv_f32(&w.k_proj, x, &s.k)?;
    gpu.rms_norm_heads_shared_w(&s.k, &w.k_norm, &s.kn, hd as i32, nkv as i32, eps)?;
    gpu.gemv_f32(&w.v_proj, x, &s.v)?;

    let pos_buf = gpu.upload_f32(&[f32::from_bits(pos as u32)], &[1])?;
    gpu.rope_partial_interleaved_f32_batched(
        &s.qn,
        &s.kn,
        &pos_buf,
        nh,
        nkv,
        hd,
        cfg.rotary_dim(),
        cfg.rotary_dim(),
        cfg.rope_theta,
        1,
        0,
    )?;
    gpu.qsa_cache_write(
        &s.kn, &cache.k, pos as i32, nkv as i32, hd as i32, ms as i32,
    )?;
    gpu.qsa_cache_write(&s.v, &cache.v, pos as i32, nkv as i32, hd as i32, ms as i32)?;
    cache.len = cache.len.max(pos + 1);

    // ── indexer ─────────────────────────────────────────────────────────────
    gpu.gemv_f32(&w.ix_qk_proj, x, &s.ix_qk)?;
    // The indexer's key is stored RAW; norm and rotation apply to the pooled
    // block key, not to the per-token one.
    let ix_key = s.ix_qk.sub_offset(ix.n_heads * ix.head_dim, ix.head_dim);
    gpu.qsa_cache_write(
        &ix_key,
        &cache.ix_keys,
        pos as i32,
        1,
        ix.head_dim as i32,
        ms as i32,
    )?;
    gpu.rms_norm_heads_shared_w(
        &s.ix_qk,
        &w.ix_q_norm,
        &s.ix_q,
        ix.head_dim as i32,
        ix.n_heads as i32,
        eps,
    )?;
    gpu.rope_partial_interleaved_f32_batched(
        &s.ix_q,
        &s.ix_q,
        &pos_buf,
        ix.n_heads,
        0,
        ix.head_dim,
        ix.head_dim,
        ix.head_dim,
        cfg.rope_theta,
        1,
        0,
    )?;

    let vis_f: Vec<f32> = visible_positions
        .iter()
        .map(|v| f32::from_bits(*v as u32))
        .collect();
    let n_vis = vis_f.len();
    // ponytail: fresh upload per step. The list is `n_vis` i32s and this is the
    // correctness path; a persistent buffer is the obvious win once it matters.
    let visible = gpu.upload_f32(&vis_f, &[n_vis])?;
    let n_blocks = n_vis / ix.compress_ratio;
    if n_blocks > 0 {
        gpu.qsa_pool_norm_blocks(
            &cache.ix_keys,
            &visible,
            &w.ix_k_norm,
            &s.block_keys,
            &s.starts,
            n_blocks as i32,
            ix.compress_ratio as i32,
            ix.head_dim as i32,
            eps,
        )?;
        // Each pooled key rotates at its block's FIRST position.
        gpu.rope_partial_interleaved_f32_batched(
            &s.block_keys,
            &s.block_keys,
            &s.starts,
            1,
            0,
            ix.head_dim,
            ix.head_dim,
            ix.head_dim,
            cfg.rope_theta,
            n_blocks,
            0,
        )?;
        gpu.qsa_score_prepared(
            &s.ix_q,
            &s.block_keys,
            &s.scores,
            ix.n_heads as i32,
            ix.head_dim as i32,
            n_blocks as i32,
        )?;
        gpu.qsa_topk_mask(
            &s.scores,
            &s.block_mask,
            n_blocks as i32,
            ix.block_topk().min(n_blocks) as i32,
        )?;
    }
    gpu.qsa_select_indices(
        &s.block_mask,
        &visible,
        &s.slots,
        &s.count,
        n_vis as i32,
        n_blocks as i32,
        ix.compress_ratio as i32,
    )?;
    let n_sel = gpu.download_f32(&s.count)?[0].to_bits() as usize;

    // ── gather + attend ─────────────────────────────────────────────────────
    for (src, dst) in [(&cache.k, &s.gath_k), (&cache.v, &s.gath_v)] {
        gpu.qsa_gather_kv(
            src,
            &s.slots,
            dst,
            nkv as i32,
            n_sel as i32,
            hd as i32,
            ms as i32,
        )?;
    }
    gpu.attention_cold_slots(
        &s.qn,
        &s.gath_k,
        &s.gath_v,
        &s.ctx,
        &s.m,
        &s.l,
        nh,
        nkv,
        n_sel,
        1.0 / (hd as f32).sqrt(),
        0,
        0,
        0,
        None,
        hd,
    )?;
    gpu.qsa_apply_output_gate(&s.ctx, &s.gate, &s.gated, (nh * hd) as i32)?;
    gpu.gemv_f32(&w.o_proj, &s.gated, out)
}
