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
use hipfire_runtime::weights::{weight_gemv, WeightTensor};

pub struct QsaWeights {
    /// `[n_heads * head_dim * 2, hidden]` — query AND output gate.
    pub q_proj: WeightTensor,
    /// `[n_kv * head_dim, hidden]`
    pub k_proj: WeightTensor,
    pub v_proj: WeightTensor,
    /// `[hidden, n_heads * head_dim]`
    pub o_proj: WeightTensor,
    /// `[head_dim]`, shared across heads
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    /// `[(ix_heads + ix_kv) * ix_head_dim, hidden]`
    pub ix_qk_proj: WeightTensor,
    /// `[ix_head_dim]` each
    pub ix_q_norm: GpuTensor,
    pub ix_k_norm: GpuTensor,
}

/// Per-sequence cache and scratch. `max_seq` bounds the whole thing.
pub struct QsaCache {
    kv: QsaKv,
    /// `[max_seq, ix_head_dim]` — the indexer's own RAW keys (un-normed,
    /// un-rotated; both are applied to the POOLED key, not the per-token one).
    /// Stored f16 (pooled and normed in f32): they decide WHICH slots are
    /// attended, so the narrowing was measured, not assumed — see the
    /// continuous-batching plan's P0a log.
    pub ix_keys: GpuTensor,
    pub len: usize,
    max_seq: usize,
}

/// Tokens per KVarN K block (one `kvarn_quantize_tile` tile per KV head).
const KVARN_GROUP: usize = 128;

/// How a sparse-attention layer stores its K and V.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QsaKvMode {
    /// Full f32 K and V — the reference.
    F32,
    /// KVarN K (`bits`-bit variance-normalised tiles of 128 tokens, an f32
    /// window for the open tile) + Q8_0 V: the tree's asymmetric KVarN scheme.
    Kvarn { bits: usize },
}

impl QsaKvMode {
    /// The `kv_cache` setting for this family. `auto` is **KVarN-8**; `f32`
    /// keeps the reference for A/B; `kvarn`/`kvarn4` mean 4-bit, as they do for
    /// every other family.
    ///
    /// Measured on the 180B oq4, held-out wikitext2 (2026-11-27):
    ///
    /// | KV | 4K PPL | 32K PPL |
    /// |---|---|---|
    /// | f32 | 4.0550 | 4.4188 |
    /// | KVarN-4, un-rotated | 4.0880 (+0.81%) | — |
    /// | KVarN-4, rotated | 4.0411 | 4.4341 (+0.35%) |
    /// | KVarN-8, un-rotated | 4.0345 | 4.4135 |
    /// | KVarN-8, rotated (this default) | 4.0306 | — |
    ///
    /// The Hadamard rotation is what un-rotated KVarN-4 was missing. Rotated
    /// 4-bit passes the 0.5% bar at 32K too, but its error GROWS with context
    /// (−0.34% at 4K, +0.35% at 32K) while 8-bit stays flat, and once K is
    /// quantised the Q8 V and indexer keys dominate, so 8-bit costs only
    /// ~100 MB per 32K session over 4-bit. 8 stays the default.
    ///
    /// The deprecated single-tier modes are not implemented here, so they are
    /// NAMED and served as the default rather than silently honoured or refused.
    pub fn from_setting(mode: &str) -> Self {
        match mode {
            "f32" | "fp32" => Self::F32,
            "kvarn2" => Self::Kvarn { bits: 2 },
            "kvarn" | "kvarn4" => Self::Kvarn { bits: 4 },
            "auto" | "kvarn8" => Self::Kvarn { bits: 8 },
            other => {
                eprintln!(
                    "qwen4_exp: kv_cache={other} is not implemented for this family; \
                     using kvarn8 (set kv_cache=kvarn8, or f32 for the reference)"
                );
                Self::Kvarn { bits: 8 }
            }
        }
    }
}

enum QsaKv {
    F32 {
        /// `[n_kv, max_seq, head_dim]`
        k: GpuTensor,
        v: GpuTensor,
    },
    Kvarn {
        /// `[n_blocks, n_kv, record_bytes]` — completed 128-token K blocks.
        recs: GpuTensor,
        /// `[KVARN_GROUP, n_kv * head_dim]` f32 — the open block (slot = pos % 128).
        window: GpuTensor,
        /// `[max_seq, kv_dim/32, 34]` — Q8_0 V by position.
        v_q8: GpuTensor,
        /// `[n_kv, head_dim * KVARN_GROUP]` f32 scratch for a closing block.
        tiles: GpuTensor,
        bits: usize,
        record_bytes: usize,
        /// K is stored in the Hadamard-rotated frame (`rotate_x_mq`, orthonormal
        /// per 256 channels), and the query is rotated the same way at attention
        /// time, so scores are unchanged. This is the first half of the KVarN
        /// method — "a Hadamard rotation followed by a dual-scaling variance
        /// normalization" (2606.03458) — which the GPU KVarN path had never
        /// implemented (see `hipfire_kvquant::kvarn::hadamard_rows`).
        rotated: bool,
    },
}

/// `HIPFIRE_QSA_KVARN_NO_ROTATE=1` — DEBUG A/B only: store KVarN K un-rotated
/// (the pre-2026-11-27 behaviour) to measure what the rotation buys.
fn kvarn_rotate() -> bool {
    std::env::var("HIPFIRE_QSA_KVARN_NO_ROTATE").as_deref() != Ok("1")
}

impl QsaCache {
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// An f32 cache — the reference the CPU-parity examples compare against.
    pub fn new(gpu: &mut Gpu, cfg: &Qwen4ExpConfig, max_seq: usize) -> HipResult<Self> {
        Self::with_kv(gpu, cfg, max_seq, QsaKvMode::F32)
    }

    pub fn with_kv(
        gpu: &mut Gpu,
        cfg: &Qwen4ExpConfig,
        max_seq: usize,
        mode: QsaKvMode,
    ) -> HipResult<Self> {
        let ix = &cfg.indexer;
        let (nkv, hd) = (cfg.n_kv_heads, cfg.head_dim);
        let kv = match mode {
            QsaKvMode::F32 => QsaKv::F32 {
                k: gpu.zeros(&[nkv * max_seq * hd], DType::F32)?,
                v: gpu.zeros(&[nkv * max_seq * hd], DType::F32)?,
            },
            QsaKvMode::Kvarn { bits } => {
                let record_bytes =
                    hipfire_kvquant::kvarn::kvarn_record_bytes_bits(hd, KVARN_GROUP, bits);
                let n_blocks = max_seq.div_ceil(KVARN_GROUP);
                QsaKv::Kvarn {
                    recs: gpu.zeros(&[n_blocks * nkv * record_bytes], DType::Raw)?,
                    window: gpu.zeros(&[KVARN_GROUP * nkv * hd], DType::F32)?,
                    v_q8: gpu.zeros(&[max_seq * (nkv * hd / 32) * 34], DType::Raw)?,
                    tiles: gpu.zeros(&[nkv * hd * KVARN_GROUP], DType::F32)?,
                    bits,
                    record_bytes,
                    // The MQ FWHT works in 256-wide groups; a head that is not a
                    // whole number of them cannot be rotated per head.
                    rotated: kvarn_rotate() && hd % 256 == 0,
                }
            }
        };
        Ok(Self {
            kv,
            ix_keys: gpu.zeros(&[max_seq * ix.head_dim], DType::F16)?,
            len: 0,
            max_seq,
        })
    }

    /// The query as attention must see it: `qn` itself, or `qn` rotated into
    /// `scratch` when K is stored rotated. `qn` is `[n_heads * head_dim]`.
    pub fn query_for_attention<'a>(
        &self,
        gpu: &mut Gpu,
        cfg: &Qwen4ExpConfig,
        qn: &'a GpuTensor,
        scratch: &'a GpuTensor,
    ) -> HipResult<&'a GpuTensor> {
        match &self.kv {
            QsaKv::Kvarn { rotated: true, .. } => {
                gpu.rotate_x_mq(qn, scratch, cfg.n_heads * cfg.head_dim)?;
                Ok(scratch)
            }
            _ => Ok(qn),
        }
    }

    /// Whether K is held in the Hadamard-rotated frame (KVarN with rotation).
    pub fn k_rotated(&self) -> bool {
        matches!(&self.kv, QsaKv::Kvarn { rotated: true, .. })
    }

    pub fn kv_mode(&self) -> QsaKvMode {
        match &self.kv {
            QsaKv::F32 { .. } => QsaKvMode::F32,
            QsaKv::Kvarn { bits, .. } => QsaKvMode::Kvarn { bits: *bits },
        }
    }

    /// Store position `pos`'s K and V rows (`[n_kv * head_dim]` each, K normed
    /// and rotated). `pos_dev` is the same position as a device i32 (the
    /// f32-bit-cast encoding the RoPE and slot lists use). Does NOT advance
    /// `len` — callers do, after the whole row set is written.
    ///
    /// KVarN: the row lands in the open-block window; when it is the block's
    /// last slot the whole block is quantised into its record. A block can be
    /// closed more than once (a rolled-back speculative row re-written at the
    /// same position) — the record is simply rebuilt from the window, which
    /// still holds every row of the block until the next block starts.
    pub fn write(
        &mut self,
        gpu: &mut Gpu,
        cfg: &Qwen4ExpConfig,
        k_row: &GpuTensor,
        v_row: &GpuTensor,
        pos: usize,
        pos_dev: &GpuTensor,
    ) -> HipResult<()> {
        let (nkv, hd, ms) = (cfg.n_kv_heads, cfg.head_dim, self.max_seq);
        let kv_dim = nkv * hd;
        match &self.kv {
            QsaKv::F32 { k, v } => {
                gpu.qsa_cache_write(k_row, k, pos as i32, nkv as i32, hd as i32, ms as i32)?;
                gpu.qsa_cache_write(v_row, v, pos as i32, nkv as i32, hd as i32, ms as i32)
            }
            QsaKv::Kvarn {
                recs,
                window,
                v_q8,
                tiles,
                bits,
                record_bytes,
                rotated,
            } => {
                let slot = window.sub_offset((pos % KVARN_GROUP) * kv_dim, kv_dim);
                if *rotated {
                    gpu.rotate_x_mq(k_row, &slot, kv_dim)?;
                } else {
                    gpu.memcpy_dtod_auto(&slot.buf, &k_row.buf, kv_dim * 4)?;
                }
                gpu.kv_cache_write_q8_0(v_q8, v_row, &pos_dev.buf, nkv, hd)?;
                if pos % KVARN_GROUP == KVARN_GROUP - 1 {
                    let blk = pos / KVARN_GROUP;
                    gpu.kvarn_gather_k_tiles(window, tiles, 1, nkv, hd, KVARN_GROUP)?;
                    let rec = recs.sub_offset(blk * nkv * record_bytes, nkv * record_bytes);
                    gpu.kvarn_quantize_tile(
                        tiles,
                        &rec,
                        nkv,
                        hd,
                        KVARN_GROUP,
                        *record_bytes,
                        *bits,
                    )?;
                }
                Ok(())
            }
        }
    }

    /// Gather the `n_sel` positions in `slots` into `out_k`/`out_v`
    /// (`[n_kv, n_sel, head_dim]` f32), dequantising in KVarN mode.
    pub fn gather(
        &self,
        gpu: &mut Gpu,
        cfg: &Qwen4ExpConfig,
        slots: &GpuTensor,
        n_sel: usize,
        out_k: &GpuTensor,
        out_v: &GpuTensor,
    ) -> HipResult<()> {
        let (nkv, hd, ms) = (cfg.n_kv_heads, cfg.head_dim, self.max_seq);
        match &self.kv {
            QsaKv::F32 { k, v } => {
                for (src, dst) in [(k, out_k), (v, out_v)] {
                    gpu.qsa_gather_kv(
                        src,
                        slots,
                        dst,
                        nkv as i32,
                        n_sel as i32,
                        hd as i32,
                        ms as i32,
                    )?;
                }
                Ok(())
            }
            QsaKv::Kvarn {
                recs,
                window,
                v_q8,
                bits,
                record_bytes,
                ..
            } => gpu.qsa_gather_kvarn(
                recs,
                window,
                v_q8,
                slots,
                out_k,
                out_v,
                nkv,
                n_sel,
                hd,
                KVARN_GROUP,
                *record_bytes,
                *bits,
                // Only CLOSED blocks have records; the open one is the window.
                self.len / KVARN_GROUP,
            ),
        }
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
    /// `[n_heads * head_dim]` — the rotated query, when K is stored rotated.
    q_rot: GpuTensor,
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
            q_rot: z(gpu, nh * hd)?,
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
    weight_gemv(gpu, &w.q_proj, x, &s.qg)?;
    gpu.qsa_split_query_gate(&s.qg, &s.query, &s.gate, nh as i32, hd as i32)?;
    gpu.rms_norm_heads_shared_w(&s.query, &w.q_norm, &s.qn, hd as i32, nh as i32, eps)?;
    weight_gemv(gpu, &w.k_proj, x, &s.k)?;
    gpu.rms_norm_heads_shared_w(&s.k, &w.k_norm, &s.kn, hd as i32, nkv as i32, eps)?;
    weight_gemv(gpu, &w.v_proj, x, &s.v)?;

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
    cache.write(gpu, cfg, &s.kn, &s.v, pos, &pos_buf)?;
    cache.len = cache.len.max(pos + 1);

    // ── indexer ─────────────────────────────────────────────────────────────
    weight_gemv(gpu, &w.ix_qk_proj, x, &s.ix_qk)?;
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
    // ponytail: fresh upload per step, O(context) bytes. A persistent prefix
    // buffer (as the batched prefill uses) is the upgrade if decode traces show it.
    let visible = gpu.upload_f32(&vis_f, &[n_vis])?;
    let r = qsa_select_attend(
        gpu, cfg, w, s, cache, &s.qn, &s.ix_q, &s.gate, &visible, n_vis, &s.gated,
    );
    // Both are pooled uploads with no Drop. `visible` is `pos+1` entries, so
    // leaking it cost O(context^2) device memory over a generation.
    gpu.free_tensor(visible)?;
    gpu.free_tensor(pos_buf)?;
    r?;
    weight_gemv(gpu, &w.o_proj, &s.gated, out)
}

/// Block selection, gather, and attention for ONE query row, leaving the gated
/// context (the `o_proj` input) in `gated`.
///
/// `qn` is the normed+rotated query `[n_heads * head_dim]`, `ix_q` the indexer
/// query `[ix_heads * ix_head_dim]`, `gate` the output gate. `visible` holds
/// `n_vis` positions (f32-bit-cast i32). The cache must already hold every
/// visible position — the batched prefill writes the whole chunk first, which
/// is exact because the causal `visible` list never names a later position.
#[allow(clippy::too_many_arguments)]
pub fn qsa_select_attend(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    w: &QsaWeights,
    s: &QsaScratch,
    cache: &QsaCache,
    qn: &GpuTensor,
    ix_q: &GpuTensor,
    gate: &GpuTensor,
    visible: &GpuTensor,
    n_vis: usize,
    gated: &GpuTensor,
) -> HipResult<()> {
    let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let ix = &cfg.indexer;
    let eps = cfg.rms_norm_eps;
    let n_blocks = n_vis / ix.compress_ratio;
    if n_blocks > 0 {
        gpu.qsa_pool_norm_blocks(
            &cache.ix_keys,
            visible,
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
            ix_q,
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
        visible,
        &s.slots,
        &s.count,
        n_vis as i32,
        n_blocks as i32,
        ix.compress_ratio as i32,
    )?;
    let n_sel = gpu.download_f32(&s.count)?[0].to_bits() as usize;

    // ── gather + attend ─────────────────────────────────────────────────────
    cache.gather(gpu, cfg, &s.slots, n_sel, &s.gath_k, &s.gath_v)?;
    let qn = cache.query_for_attention(gpu, cfg, qn, &s.q_rot)?;
    gpu.attention_cold_slots(
        qn,
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
    gpu.qsa_apply_output_gate(&s.ctx, gate, gated, (nh * hd) as i32)?;
    Ok(())
}

// ── GPU teardown ────────────────────────────────────────────────────────────
//
// Exhaustive destructures, so a field added later fails to compile until someone
// decides whether it needs freeing. See the note in `hc_gpu.rs`.

impl QsaWeights {
    pub fn free(self, gpu: &mut Gpu) {
        let Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            ix_qk_proj,
            ix_q_norm,
            ix_k_norm,
        } = self;
        for t in [q_proj, k_proj, v_proj, o_proj, ix_qk_proj] {
            let _ = gpu.free_tensor(t.buf);
        }
        for t in [q_norm, k_norm, ix_q_norm, ix_k_norm] {
            let _ = gpu.free_tensor(t);
        }
    }
}

impl QsaCache {
    /// Rewind to an empty cache. Only `len` needs clearing — the K/V and indexer
    /// key buffers are written before they are read at every position < len, so
    /// stale bytes past the cursor are never visible.
    pub fn reset(&mut self) {
        self.len = 0;
    }

    pub fn free(self, gpu: &mut Gpu) {
        let Self {
            kv,
            ix_keys,
            len: _,
            max_seq: _,
        } = self;
        let _ = gpu.free_tensor(ix_keys);
        match kv {
            QsaKv::F32 { k, v } => {
                let _ = gpu.free_tensor(k);
                let _ = gpu.free_tensor(v);
            }
            QsaKv::Kvarn {
                recs,
                window,
                v_q8,
                tiles,
                bits: _,
                record_bytes: _,
                rotated: _,
            } => {
                for t in [recs, window, v_q8, tiles] {
                    let _ = gpu.free_tensor(t);
                }
            }
        }
    }
}

impl QsaScratch {
    pub fn free(self, gpu: &mut Gpu) {
        let Self {
            qg,
            query,
            gate,
            qn,
            k,
            kn,
            v,
            ix_qk,
            ix_q,
            block_keys,
            starts,
            scores,
            block_mask,
            slots,
            count,
            gath_k,
            gath_v,
            ctx,
            gated,
            q_rot,
            m,
            l,
        } = self;
        for t in [
            qg, query, gate, qn, k, kn, v, ix_qk, ix_q, block_keys, starts, scores, block_mask,
            slots, count, gath_k, gath_v, ctx, gated, q_rot, m, l,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}
