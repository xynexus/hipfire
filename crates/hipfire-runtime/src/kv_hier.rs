// SPDX-License-Identifier: Apache-2.0
//! Deferred-hierarchical KV cache (Phase 2b sub-task 4c, flag-gated).
//!
//! When `HIPFIRE_KV_HIERARCHICAL=1`, the KVarN decode path is replaced by a
//! two-tier cache:
//!   * HOT tier — the most recent `hot_budget` tokens, kept as an f16 ring
//!     `[n_kv_heads × hot_budget × head_dim]` (slot-major; f16 halves the exact-tier
//!     VRAM and is near-lossless — measured PPL-identical to f32, far above the cold
//!     2-bit floor). For a single decode query at the last position every hot token
//!     is causally visible, so it is read by `attention_cold_slots` slot-major-f16
//!     (layout 2), which already emits the flash partials (m,l).
//!   * COLD tier — older tokens, compacted by `compact_cold_kv` (KVarN 4-bit,
//!     importance-weighted m:1 merge) into segments that stay 4-bit-resident on
//!     GPU and are dequantized on-the-fly each step (`kvarn_dequant_tile` → f16)
//!     and read by the channel-major mode of `attention_cold_slots`.
//!
//! The two tiers are folded by `flash_tier_merge` (online softmax). The hot tier
//! being f16 (not 4-bit) costs `hot_budget × kv_dim × 2` B/layer — small; the
//! storage win lives in the compacted cold tier that holds the bulk of a long
//! context. head_dim is fixed at 256 (the kernels' CHD).
//!
//! Migration (hot → cold) has two paths: an overflow fallback on the critical path
//! (`migrate_n(migrate_batch)` when the ring fills), and `idle_compact` — the
//! deferred drain run between turns (off the latency path; see
//! `qwen35_prefill_active_session`). Both fold a token range into ONE cold segment
//! via `compact_cold_kv`.
//!
//! Cold compaction is tunable (defaults shown):
//!   * importance (`HIPFIRE_KV_IMPORTANCE`): vnorm (best) | uniform | knorm |
//!     kvnorm | attn — ranks which cold tokens stay exact (core) and weights the
//!     merge average; vnorm beats the others (attn underperforms — see commit log).
//!   * merge (`HIPFIRE_KV_FOLD_M`=4, `HIPFIRE_KV_CORE_FRAC`=0.125,
//!     `HIPFIRE_KV_POS_LOCAL`=on): m:1 importance-weighted average of the non-core
//!     tail, grouped by adjacent position to limit RoPE-phase blur (the dominant
//!     merge cost). fold_m=1 = no merge (lossless, no compression).
//!   * precision (`HIPFIRE_KV_COLD_BITS`=4): 2 halves cold-code storage at ~+1.6%
//!     PPL — quant is cheap even at 2-bit (Sinkhorn variance-norm does the
//!     incoherence job a rotation would, so `rotate=false` and no ConQuR needed).
//!
//! Window/drain knobs: `HIPFIRE_KV_HOT_BUDGET`(512), `HIPFIRE_KV_MIGRATE_BATCH`(128),
//! `HIPFIRE_KV_IDLE_KEEP`(0 = full between-turns drain).
//!
//! Constraint: this is an inherently per-token-attention feature (it lives in
//! `kv_cache_attention_dispatch`); the batched session-batch prefill bypasses that
//! and is guarded against hier. Parity oracle: `ColdTier::two_tier_attend` + the
//! GPU kernels validated in hipfire-rdna/examples/parity_{attention_cold_slots,
//! flash_tier_merge,flash_partials_ml,two_tier_e2e,cold_4bit_read} and
//! hipfire-runtime/examples/parity_kv_hier.

use crate::triattn::TriAttnCenters;
use hipfire_kvquant::kv_compact::{compact_cold_kv, ColdTier};
use hipfire_kvquant::kvarn::{
    dequantize_tile, kvarn_record_bytes_bits, pack_kvarn_tile_bits, unpack_kvarn_tile_bits,
};
use hipfire_primitives::conv::f16_to_f32;
use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

/// Per-token importance proxy used to rank/weight cold compaction.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ImportanceMode {
    Uniform,
    VNorm,
    KNorm,
    KvNorm,
    /// Real accumulated attention mass (CASK): Σ over q-heads & decode steps of the
    /// normalized attention weight each token received while in the hot window.
    Attn,
    /// Calibrated TriAttention importance: score(key) = Σ_band ‖E[q_f]‖·‖k_band‖ from
    /// a calibrated TRIA sidecar (query-energy-weighted key magnitude), GQA-aggregated.
    /// The "real CASK" importance signal (vs the while-hot `Attn` proxy). Needs
    /// `HIPFIRE_KV_TRIATTN_SIDECAR`; falls back to vnorm if centers are missing.
    TriAttn,
}

impl ImportanceMode {
    fn from_str(s: &str) -> Self {
        match s {
            "uniform" => ImportanceMode::Uniform,
            "knorm" => ImportanceMode::KNorm,
            "kvnorm" => ImportanceMode::KvNorm,
            "attn" => ImportanceMode::Attn,
            "triattn" => ImportanceMode::TriAttn,
            _ => ImportanceMode::VNorm, // default
        }
    }
}

/// One compacted cold segment, 4-bit-resident on GPU (per kv-head record tiles).
pub struct ColdSegmentGpu {
    pub k_recs: GpuTensor, // [n_kv_heads × rec_bytes] as f32-view (bytes/4 elems)
    pub v_recs: GpuTensor,
    pub n_valid: usize,     // real slots attended
    pub n_slots: usize,     // padded tile width (= slot_stride for the cold read)
    pub rec_bytes: usize,   // K record stride
    pub bits: usize,        // K quant bits per code (4 or 2) — for the dequant unpack
    pub v_rec_bytes: usize, // V record stride (may differ from K when v_bits != bits)
    pub v_bits: usize,      // V quant bits per code
    pub v_perslot: bool,    // V tile is slot-major [n_slots × HD] (per-token quant)
}

/// Reusable read scratch (lazily sized to the largest cold segment seen).
struct HierScratch {
    acc_m: GpuTensor, // [n_heads] accumulator flash max
    acc_l: GpuTensor, // [n_heads] accumulator flash denom
    out_c: GpuTensor,
    m_c: GpuTensor,
    l_c: GpuTensor,
    deq_k: GpuTensor, // f16 [n_kv_heads × HD × max_slots]
    deq_v: GpuTensor,
    max_slots: usize,
}

/// 8-bit hot-ring store (Phase 1): head-major slot-major symmetric-absmax int8,
/// replacing one f16 ring. `codes` = int8 [nkv × hot_budget × HD] (raw bytes),
/// `scale` = f32 [nkv × hot_budget] (one per slot per head). Written by
/// `kv_hot_quant_q8`, read back via `kv_hot_dequant_q8` → f16 slot-major scratch.
struct Q8Ring {
    codes: GpuTensor,
    scale: GpuTensor,
}

impl Q8Ring {
    fn alloc(gpu: &mut Gpu, nkv: usize, hb: usize, head_dim: usize) -> HipResult<Self> {
        let n = nkv * hb * head_dim;
        Ok(Q8Ring {
            codes: gpu.upload_raw(&vec![0u8; n], &[n])?,
            scale: gpu.zeros(&[nkv * hb], DType::F32)?,
        })
    }
    /// 1-element placeholder for a non-KV (e.g. linear-attention) layer — never
    /// written/read, mirrors the base cache's `alloc_k_v_filtered` placeholders so
    /// absolute layer indexing is preserved without full-ring VRAM.
    fn placeholder(gpu: &mut Gpu) -> HipResult<Self> {
        Ok(Q8Ring {
            codes: gpu.upload_raw(&[0u8], &[1])?,
            scale: gpu.zeros(&[1], DType::F32)?,
        })
    }
}

pub struct HierKvState {
    pub enabled: bool,
    pub hot_budget: usize,
    pub migrate_batch: usize,
    pub core_frac: f32,
    pub fold_m: usize,
    /// Per-token importance signal for cold compaction (core selection + merge
    /// weighting). "uniform" (meaningless, average merge), "vnorm" (‖V_t‖),
    /// "knorm" (‖K_t‖), "kvnorm" (‖K_t‖·‖V_t‖). A real attention-mass signal
    /// (CASK) would need per-key accumulation in the hot read; norms are the
    /// zero-tracking proxy.
    pub importance_mode: ImportanceMode,
    /// Group merged (non-core) cold tokens by adjacent position (similar RoPE
    /// phase → less merge blur) rather than importance rank. Default on.
    pub position_local: bool,
    /// CASK content-similarity merge grouping (`HIPFIRE_KV_MERGE=similarity`): fold
    /// near-DUPLICATE keys (K-cosine) instead of position-adjacent ones, so averaging
    /// is ~lossless (fixes the content-merge loss). Takes precedence over
    /// position_local. Default off (byte-identical).
    pub similarity_merge: bool,
    /// Max quant code for cold K tiles (15=4-bit default, 3=2-bit probe).
    pub cold_qmax: f32,
    /// Bits per cold K code (4 or 2) — drives real sub-nibble packing + dequant.
    pub cold_bits: usize,
    /// Max quant code for cold V tiles (independent of K — asymmetric K2V4 etc.).
    pub cold_v_qmax: f32,
    /// Bits per cold V code (`HIPFIRE_KV_COLD_V_BITS`, defaults to `cold_bits`).
    pub cold_v_bits: usize,
    /// Store cold V per-slot (token axis) instead of per-channel — V's natural
    /// quant axis (`HIPFIRE_KV_COLD_V_PERSLOT=1`, default off).
    pub cold_v_perslot: bool,
    /// Idle-time cold-segment defrag threshold (`HIPFIRE_KV_DEFRAG_SEGMENTS`, 0 =
    /// off). When a layer holds more than this many cold segments, `idle_compact`
    /// folds them all into one wider tile (bounds the per-segment two-tier read
    /// cost + amortizes per-channel scale overhead). Off by default (byte-identical).
    pub defrag_segments: usize,
    /// PyramidKV per-layer budget schedule (`HIPFIRE_KV_PYRAMID=1`): upper layers fold
    /// MORE aggressively (concentrated attention → cheaper) + keep less core; lower
    /// layers fold LESS (diffuse attention → need more tokens). Varies fold_m/core_frac
    /// around the base by ±`pyramid_amp` linearly in layer index. Default off.
    pub pyramid: bool,
    pub pyramid_amp: f32,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hot_k: Vec<GpuTensor>, // [n_layers] slot-major [nkv × hot_budget × HD] f32
    pub hot_v: Vec<GpuTensor>,
    /// Per-layer per-hot-slot accumulated attention mass [hot_budget] f32 (CASK
    /// importance). Filled by the hot read's mass pass; only used when
    /// importance_mode == Attn. Zeroed at reset, shifted on migrate.
    pub attn_mass: Vec<GpuTensor>,
    pub hot_count: Vec<usize>,          // live hot tokens per layer
    pub migrated: Vec<usize>,           // tokens already moved to cold per layer
    pub cold: Vec<Vec<ColdSegmentGpu>>, // [n_layers][segments]
    scr: Option<HierScratch>,
    /// Reused f16 [n_kv_heads × HD] scratch for casting an incoming f32 token before
    /// it is placed into the f16 hot ring (avoids a per-token alloc). `None` when
    /// disabled.
    hot_cast: Option<GpuTensor>,
    /// Calibrated TriAttention centers for `ImportanceMode::TriAttn` (loaded from
    /// `HIPFIRE_KV_TRIATTN_SIDECAR`). `None` unless that mode is active with a sidecar.
    /// Read at `migrate_n` to rank the cold merge by calibrated query-energy alignment.
    centers: Option<TriAttnCenters>,
    /// Phase 0 rotated-frame probe (`HIPFIRE_KV_HOT_ROTATE=1`, default off): FWHT-
    /// rotate the hot K on write AND the query on read with the same orthonormal
    /// `rotate_x_mq`, so `q_rot·K_rot = q·K` and the whole cache lives in one
    /// rotated frame — the cold tier inherits it via migrate (`rotate=false` on
    /// already-rotated K, no double-rotate). K-only; V stays un-rotated so the
    /// attention output needs no inverse. Groundwork for the 8-bit hot ring
    /// (docs/plans/2026-07-15-8bit-hot-ring-kv-hier.md Phase 0). Norm-based
    /// importance is rotation-invariant, so migrate/importance are unaffected.
    pub hot_rotate: bool,
    /// f32 [n_kv_heads × HD] scratch: rotated hot K before the f16 cast. Some when hot_rotate.
    rot_k: Option<GpuTensor>,
    /// f32 [n_heads × HD] scratch: rotated query for the two-tier read. Some when hot_rotate.
    q_rot: Option<GpuTensor>,
    /// 8-bit hot tier (`HIPFIRE_KV_HOT_BITS`, DEFAULT 8; 16 = f16 for A/B). When on,
    /// the hot ring is per-token symmetric-absmax int8 (`hot_kq`/`hot_vq`) instead
    /// of the f16 `hot_k`/`hot_v` rings (which are then left empty), halving hot-tier
    /// VRAM. Forces `hot_rotate` (symmetric q8 needs the FWHT-centered frame). The
    /// read/migrate dequant into a shared f16 scratch (`hot_deq_k`/`hot_deq_v`) and
    /// reuse the existing layout-2 read + widen/compact path.
    pub hot_q8: bool,
    hot_kq: Vec<Q8Ring>, // [n_layers] rotated-K int8 ring; empty unless hot_q8
    hot_vq: Vec<Q8Ring>, // [n_layers] V int8 ring; empty unless hot_q8
    /// Count of KV-bearing (full-attention) layers — hybrid arches (Qwen3.5:
    /// DeltaNet + FullAttention) allocate full hot rings only for these; the rest
    /// get 1-element placeholders. Drives honest `hot_tier_bytes` accounting.
    n_kv_layers: usize,
    /// Shared f16 [nkv × hot_budget × HD] dequant scratch (one ring's worth, reused
    /// across layers/reads). Some when hot_q8.
    hot_deq_k: Option<GpuTensor>,
    hot_deq_v: Option<GpuTensor>,
}

impl HierKvState {
    /// Read `HIPFIRE_KV_HIERARCHICAL` / `HIPFIRE_KV_HOT_BUDGET` /
    /// `HIPFIRE_KV_MIGRATE_BATCH`. Returns a disabled state when the flag is off.
    pub fn from_env(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<Self> {
        // Full hot rings only for KV-bearing (full-attention) layers; hybrid arches
        // (Qwen3.5 DeltaNet + FullAttention) get 1-element placeholders for the rest,
        // mirroring the base cache's `alloc_k_v_filtered`. Absolute layer indexing is
        // preserved. `is_kv_layer` = base-cache mask (layer_types == FullAttention).
        let n_layers = is_kv_layer.len();
        let n_kv_layers = is_kv_layer.iter().filter(|b| **b).count();
        // FWHT-256 and FWHT-128 both supported (attention_cold_slots + _128,
        // rotate_x_mq + _128); other head_dims fall back to non-hier KVarN.
        let enabled = std::env::var("HIPFIRE_KV_HIERARCHICAL").ok().as_deref() == Some("1")
            && (head_dim == 256 || head_dim == 128);
        // Default 512: the knee sweep showed hot=512 clearly beats 256 (PPL 27.5 vs
        // 29.0 at fold=4/2-bit), and the f16 ring makes 512 cost the same VRAM the old
        // f32-256 default did. Hot budget is the primary quality dial now that the
        // dephasing merge lever is closed (content loss, not phase).
        let hot_budget = std::env::var("HIPFIRE_KV_HOT_BUDGET")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512usize);
        let migrate_batch = std::env::var("HIPFIRE_KV_MIGRATE_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(128usize)
            .min(hot_budget / 2)
            .max(1);
        // Cold-tier compaction knobs. fold_m=1 disables the m:1 merge (cold = pure
        // 4-bit KVarN, no token reduction, no RoPE-phase blur); higher = more
        // compression but more blur. core_frac keeps the top fraction exact (1 slot).
        let fold_m = std::env::var("HIPFIRE_KV_FOLD_M")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4usize)
            .max(1);
        let core_frac = std::env::var("HIPFIRE_KV_CORE_FRAC")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.125f32);
        let importance_mode = ImportanceMode::from_str(
            &std::env::var("HIPFIRE_KV_IMPORTANCE").unwrap_or_else(|_| "vnorm".to_string()),
        );
        // TriAttn importance needs calibrated centers from a TRIA sidecar. Load once
        // here; if missing, migrate_n falls back to vnorm ranking (never fails hard).
        let centers = if importance_mode == ImportanceMode::TriAttn {
            match std::env::var("HIPFIRE_KV_TRIATTN_SIDECAR") {
                Ok(p) => match TriAttnCenters::load(std::path::Path::new(&p)) {
                    Ok(c) => {
                        eprintln!(
                            "[kv_hier] TriAttn importance: centers {}L x {}H (hd={}) from {p}",
                            c.n_layers, c.n_heads, c.head_dim
                        );
                        Some(c)
                    }
                    Err(e) => {
                        eprintln!("[kv_hier] TRIA sidecar load failed ({e}); using vnorm");
                        None
                    }
                },
                Err(_) => {
                    eprintln!(
                        "[kv_hier] ImportanceMode::TriAttn but HIPFIRE_KV_TRIATTN_SIDECAR unset; using vnorm"
                    );
                    None
                }
            }
        } else {
            None
        };
        let position_local = std::env::var("HIPFIRE_KV_POS_LOCAL").ok().as_deref() != Some("0");
        let similarity_merge =
            std::env::var("HIPFIRE_KV_MERGE").ok().as_deref() == Some("similarity");
        // Cold-tile quant precision probe: code max = 2^bits - 1 (4-bit=15 default,
        // 2-bit=3). Same nibble storage; this measures lower-precision quant QUALITY.
        let cold_bits: u32 = std::env::var("HIPFIRE_KV_COLD_BITS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4)
            .clamp(1, 4);
        let cold_qmax = ((1u32 << cold_bits) - 1) as f32;
        // V may carry a DIFFERENT bit width than K (asymmetric cold, e.g. K2V4):
        // V is the "easy" operand (weighted average, no outlier channels), so it
        // can hold more bits than an aggressive cold K for a small extra cost, or
        // match it. Defaults to `cold_bits` (symmetric) when unset.
        let cold_v_bits: u32 = std::env::var("HIPFIRE_KV_COLD_V_BITS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(cold_bits)
            .clamp(1, 4);
        let cold_v_qmax = ((1u32 << cold_v_bits) - 1) as f32;
        let cold_v_perslot =
            std::env::var("HIPFIRE_KV_COLD_V_PERSLOT").ok().as_deref() == Some("1");
        let defrag_segments = std::env::var("HIPFIRE_KV_DEFRAG_SEGMENTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0usize);
        let pyramid = std::env::var("HIPFIRE_KV_PYRAMID").ok().as_deref() == Some("1");
        let pyramid_amp = std::env::var("HIPFIRE_KV_PYRAMID_AMP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5f32);
        // Hot-tier bit-width (`HIPFIRE_KV_HOT_BITS`): 8 = per-token symmetric-absmax
        // int8 ring (DEFAULT — Phase 4; ~49% smaller hot VRAM, parity + PPL neutral
        // vs f16), 16 = f16 ring (kept selectable for A/B). Only 8 and 16 are valid:
        // the hot tier is the EXACT tier, so 4-bit belongs to the cold tier, not here
        // — any other value coerces to the 8-bit default.
        let hot_bits = match std::env::var("HIPFIRE_KV_HOT_BITS").ok().as_deref() {
            Some("16") => 16,
            Some("8") | None => 8,
            Some(other) => {
                eprintln!("[kv-hier] HIPFIRE_KV_HOT_BITS={other} invalid (want 8|16); using 8");
                8
            }
        };
        let hot_q8 = enabled && hot_bits == 8;
        // Phase 0 rotated-frame probe: rotate hot K on write + query on read (see
        // the `hot_rotate` field doc). q8 mode REQUIRES rotation (symmetric absmax
        // wants the FWHT-centered frame — codec probe: sym+FWHT ≈ affine+FWHT ≈
        // kvarn); otherwise it is the opt-in `HIPFIRE_KV_HOT_ROTATE` f16 A/B.
        let hot_rotate = enabled
            && (hot_q8 || std::env::var("HIPFIRE_KV_HOT_ROTATE").ok().as_deref() == Some("1"));
        let mut hot_k = Vec::with_capacity(n_layers);
        let mut hot_v = Vec::with_capacity(n_layers);
        let mut attn_mass = Vec::with_capacity(n_layers);
        let mut hot_cast = None;
        let mut rot_k = None;
        let mut q_rot = None;
        let mut hot_kq = Vec::new();
        let mut hot_vq = Vec::new();
        let mut hot_deq_k = None;
        let mut hot_deq_v = None;
        if enabled {
            // Hot ring is f16 (halves hot VRAM → a larger exact window fits the same
            // budget; f16 is far above the cold 2-bit floor, so it is near-lossless
            // for the exact tier). Read via attention_cold_slots slot-major-f16
            // (layout 2); migrate downloads raw f16 and widens to f32 for compaction.
            // Per-layer allocation: full rings for KV layers, [1] placeholders for
            // non-KV (linear-attention) layers (never appended/read — the hier hook
            // only fires on full-attention layers). Preserves absolute layer indexing.
            for &is_kv in is_kv_layer {
                attn_mass.push(gpu.zeros(&[if is_kv { hot_budget } else { 1 }], DType::F32)?);
            }
            if hot_q8 {
                // 8-bit hot ring: int8 codes + per-slot scale, plus a shared f16
                // dequant scratch (one ring's worth, reused across layers/reads).
                for &is_kv in is_kv_layer {
                    if is_kv {
                        hot_kq.push(Q8Ring::alloc(gpu, n_kv_heads, hot_budget, head_dim)?);
                        hot_vq.push(Q8Ring::alloc(gpu, n_kv_heads, hot_budget, head_dim)?);
                    } else {
                        hot_kq.push(Q8Ring::placeholder(gpu)?);
                        hot_vq.push(Q8Ring::placeholder(gpu)?);
                    }
                }
                hot_deq_k = Some(gpu.zeros(&[n_kv_heads * hot_budget * head_dim], DType::F16)?);
                hot_deq_v = Some(gpu.zeros(&[n_kv_heads * hot_budget * head_dim], DType::F16)?);
            } else {
                for &is_kv in is_kv_layer {
                    let n = if is_kv {
                        n_kv_heads * hot_budget * head_dim
                    } else {
                        1
                    };
                    hot_k.push(gpu.zeros(&[n], DType::F16)?);
                    hot_v.push(gpu.zeros(&[n], DType::F16)?);
                }
                hot_cast = Some(gpu.zeros(&[n_kv_heads * head_dim], DType::F16)?);
            }
            if hot_rotate {
                // f32 scratch for the FWHT-rotated K (before f16 cast/quant) and query.
                rot_k = Some(gpu.zeros(&[n_kv_heads * head_dim], DType::F32)?);
                q_rot = Some(gpu.zeros(&[n_heads * head_dim], DType::F32)?);
            }
            // Bit accounting (Phase 5): log the per-session hot-tier footprint ONCE
            // per process (not per session — the daemon batches thousands) so the
            // int8 win is visible in telemetry, with the multi-session projection.
            static LOG_ONCE: std::sync::Once = std::sync::Once::new();
            LOG_ONCE.call_once(|| {
                let per_ring = if hot_q8 {
                    n_kv_heads * hot_budget * head_dim + n_kv_heads * hot_budget * 4
                } else {
                    n_kv_heads * hot_budget * head_dim * 2
                };
                // Only KV-bearing layers hold real rings (hybrid arches skip the rest).
                let total = n_kv_layers * 2 * per_ring;
                let mb = total as f64 / 1e6;
                let gb_1k = total as f64 * 1000.0 / 1e9;
                if hot_q8 {
                    let f16_total = n_kv_layers * 2 * (n_kv_heads * hot_budget * head_dim * 2);
                    let saved = 100.0 * (1.0 - total as f64 / f16_total as f64);
                    eprintln!(
                        "[kv-hier] hot tier: int8, {mb:.1} MB/session ({n_kv_layers}/{n_layers} KV layers, nkv={n_kv_heads}, hot_budget={hot_budget}); f16 baseline {:.1} MB → {saved:.0}% saved; ~{gb_1k:.0} GB at 1000 sessions",
                        f16_total as f64 / 1e6
                    );
                } else {
                    eprintln!(
                        "[kv-hier] hot tier: f16, {mb:.1} MB/session ({n_kv_layers}/{n_layers} KV layers, nkv={n_kv_heads}, hot_budget={hot_budget}); ~{gb_1k:.0} GB at 1000 sessions (HIPFIRE_KV_HOT_BITS=8 halves it)"
                    );
                }
            });
        }
        Ok(Self {
            enabled,
            hot_budget,
            migrate_batch,
            core_frac,
            fold_m,
            importance_mode,
            position_local,
            similarity_merge,
            cold_qmax,
            cold_bits: cold_bits as usize,
            cold_v_qmax,
            cold_v_bits: cold_v_bits as usize,
            cold_v_perslot,
            defrag_segments,
            pyramid,
            pyramid_amp,
            n_heads,
            n_kv_heads,
            head_dim,
            hot_k,
            hot_v,
            attn_mass,
            hot_count: vec![0; n_layers],
            migrated: vec![0; n_layers],
            cold: (0..n_layers).map(|_| Vec::new()).collect(),
            scr: None,
            hot_cast,
            centers,
            hot_rotate,
            rot_k,
            q_rot,
            hot_q8,
            hot_kq,
            hot_vq,
            n_kv_layers,
            hot_deq_k,
            hot_deq_v,
        })
    }

    fn kv_dim(&self) -> usize {
        let hd = self.head_dim;
        self.n_kv_heads * hd
    }

    /// FWHT-rotate `k` floats of `x` into `x_rot` with the head_dim-matched
    /// orthonormal kernel: FWHT-256 (`rotate_x_mq`) or the FWHT-128 variant
    /// (`rotate_x_mq_128`). `head_dim ∈ {128, 256}`. Same transform for K (write)
    /// and the query (read), so `q_rot·K_rot = q·K` in either dimension.
    fn rotate_hd(
        &self,
        gpu: &mut Gpu,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        if self.head_dim == 128 {
            gpu.rotate_x_mq_128(x, x_rot, k)
        } else {
            gpu.rotate_x_mq(x, x_rot, k)
        }
    }

    /// Per-layer (fold_m, core_frac) under the PyramidKV schedule. Budget scale s(l) is
    /// `1+amp` at the bottom (l=0), `1−amp` at the top: lower layers get less fold + more
    /// core, upper layers more fold + less core. Identity when pyramid is off. (The
    /// paper's exact arithmetic is for a flat top-B budget; this maps the depth principle
    /// onto the hier cold-merge knobs.)
    fn layer_fold_core(&self, layer: usize, n_layers: usize) -> (usize, f32) {
        if !self.pyramid || n_layers <= 1 {
            return (self.fold_m, self.core_frac);
        }
        let t = layer as f32 / (n_layers - 1) as f32; // 0..1 bottom→top
        let s = 1.0 + self.pyramid_amp * (1.0 - 2.0 * t);
        let fold = ((self.fold_m as f32 / s).round() as usize).max(1);
        let core = (self.core_frac * s).clamp(0.0, 1.0);
        (fold, core)
    }

    /// Reset all per-layer tier state for a new sequence (pos==0). Hot ring buffers
    /// are kept (overwritten by `append_token`); cold segments are dropped; the
    /// attention-mass accumulators are zeroed. Call once at sequence start. NB:
    /// dropped segment GpuTensors are not pool-returned — a minor VRAM churn at the
    /// rare session boundary, not per-token.
    pub fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        for c in self.hot_count.iter_mut() {
            *c = 0;
        }
        for m in self.migrated.iter_mut() {
            *m = 0;
        }
        for segs in self.cold.iter_mut() {
            segs.clear();
        }
        if self.importance_mode == ImportanceMode::Attn {
            for mass in self.attn_mass.iter() {
                gpu.fill_f32(mass, 0.0)?;
            }
        }
        Ok(())
    }

    /// Append one token's K/V (`fa_k`/`fa_v` = [kv_dim] head-major) into the hot
    /// ring at the current tail slot. Migrates the oldest `migrate_batch` tokens
    /// to a cold segment first if the ring is full.
    pub fn append_token(
        &mut self,
        gpu: &mut Gpu,
        layer: usize,
        fa_k: &GpuTensor,
        fa_v: &GpuTensor,
    ) -> HipResult<()> {
        if self.hot_count[layer] >= self.hot_budget {
            // Overflow fallback (on the critical path): evict the oldest batch. The
            // idle/between-turns path (idle_compact) keeps this from firing often.
            self.migrate_n(gpu, layer, self.migrate_batch)?;
        }
        let slot = self.hot_count[layer];
        let hb = self.hot_budget;
        let nkv = self.n_kv_heads;
        let hd = self.head_dim;
        if self.hot_q8 {
            // 8-bit hot ring (Phase 1): rotate K (mandatory — symmetric absmax needs
            // the centered frame) then per-token symmetric-q8 quant into the ring; V
            // quantized un-rotated. No f16 cast / memcpy — the quant kernel writes the
            // slot directly.
            let rk = self.rot_k.as_ref().expect("rot_k present when hot_q8");
            self.rotate_hd(gpu, fa_k, rk, nkv * hd)?;
            let kq = &self.hot_kq[layer];
            gpu.kv_hot_quant_q8(&kq.codes, &kq.scale, rk, slot, hb, nkv, hd)?;
            let vq = &self.hot_vq[layer];
            gpu.kv_hot_quant_q8(&vq.codes, &vq.scale, fa_v, slot, hb, nkv, hd)?;
        } else {
            // f16 ring: cast the incoming f32 token [nkv×HD] into the reused f16
            // scratch, then place each head at hot slot (kv*hb+slot)*HD (2 bytes/elem).
            let cast = self
                .hot_cast
                .as_ref()
                .expect("hot_cast present when enabled");
            for (fa, ring, rotate) in [
                (fa_k, &self.hot_k[layer], self.hot_rotate),
                (fa_v, &self.hot_v[layer], false),
            ] {
                // Phase 0 (rotated frame): FWHT-rotate K into `rot_k` before the f16
                // cast so the hot tier is stored rotated; the query is rotated to match
                // on read, and the cold tier inherits the rotation via migrate
                // (rotate=false on already-rotated K). V is never rotated.
                let input = if rotate {
                    let rk = self.rot_k.as_ref().expect("rot_k present when hot_rotate");
                    self.rotate_hd(gpu, fa, rk, nkv * hd)?;
                    rk
                } else {
                    fa
                };
                gpu.cast_f32_to_f16(input, cast)?;
                for kv in 0..nkv {
                    let dst = ((kv * hb + slot) * hd) * 2;
                    let src = (kv * hd) * 2;
                    gpu.memcpy_dtod_at_auto(&ring.buf, dst, &cast.buf, src, hd * 2)?;
                }
            }
        }
        self.hot_count[layer] += 1;
        Ok(())
    }

    /// The hot tier for `layer` as f32, head-major slot-major
    /// `[nkv × hot_budget × HD]` (only the first `hot_count` slots are meaningful).
    /// f16 ring → download+widen; 8-bit ring → GPU dequant then widen. Used by the
    /// parity oracle and diagnostics so both codecs share one read-back path.
    pub fn hot_tier_f32(&mut self, gpu: &mut Gpu, layer: usize) -> HipResult<(Vec<f32>, Vec<f32>)> {
        let hb = self.hot_budget;
        let nkv = self.n_kv_heads;
        let hd = self.head_dim;
        let ring_elems = nkv * hb * hd;
        let widen = |bytes: &[u8]| -> Vec<f32> {
            (0..ring_elems)
                .map(|i| f16_to_f32(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])))
                .collect()
        };
        if self.hot_q8 {
            let hc = self.hot_count[layer];
            let dk = self
                .hot_deq_k
                .as_ref()
                .expect("hot_deq_k present when hot_q8");
            let dv = self
                .hot_deq_v
                .as_ref()
                .expect("hot_deq_v present when hot_q8");
            gpu.kv_hot_dequant_q8(
                &self.hot_kq[layer].codes,
                &self.hot_kq[layer].scale,
                dk,
                hc,
                hb,
                nkv,
                hd,
            )?;
            gpu.kv_hot_dequant_q8(
                &self.hot_vq[layer].codes,
                &self.hot_vq[layer].scale,
                dv,
                hc,
                hb,
                nkv,
                hd,
            )?;
            Ok((
                widen(&gpu.download_raw(dk, ring_elems * 2)?),
                widen(&gpu.download_raw(dv, ring_elems * 2)?),
            ))
        } else {
            Ok((
                widen(&gpu.download_raw(&self.hot_k[layer], ring_elems * 2)?),
                widen(&gpu.download_raw(&self.hot_v[layer], ring_elems * 2)?),
            ))
        }
    }

    /// Per-session resident hot-ring VRAM (bytes), summed over all layers (K+V).
    /// int8 ring = codes (`nkv·hb·HD`) + per-slot scale (`nkv·hb·4`); f16 ring =
    /// `2·nkv·hb·HD`. Excludes the shared dequant scratch (a fixed read-overhead of
    /// one ring's worth, not per-tier storage). Drives the bit-accounting telemetry
    /// so the multi-session int8 win is visible. 0 when disabled.
    pub fn hot_tier_bytes(&self) -> usize {
        if !self.enabled {
            return 0;
        }
        let hd = self.head_dim;
        let per_ring = if self.hot_q8 {
            self.n_kv_heads * self.hot_budget * hd + self.n_kv_heads * self.hot_budget * 4
        } else {
            self.n_kv_heads * self.hot_budget * hd * 2
        };
        self.n_kv_layers * 2 * per_ring // K + V rings, KV-bearing layers only
    }

    /// Migrate the oldest `n_req` hot tokens into ONE new cold segment, then shift
    /// the remaining hot tokens down to the front of the ring. Used both by the
    /// overflow fallback (n_req = migrate_batch) and idle_compact (n_req = drain).
    fn migrate_n(&mut self, gpu: &mut Gpu, layer: usize, n_req: usize) -> HipResult<()> {
        let mb = n_req.min(self.hot_count[layer]);
        if mb == 0 {
            return Ok(());
        }
        let hb = self.hot_budget;
        let nkv = self.n_kv_heads;
        let hd = self.head_dim;
        let kv_dim = self.kv_dim();
        // Download the f16 hot rings and widen to f32, then assemble the oldest `mb`
        // tokens as token-major [mb × kv_dim] for compact_cold_kv.
        let ring_elems = nkv * hb * hd;
        let widen = |bytes: &[u8]| -> Vec<f32> {
            (0..ring_elems)
                .map(|i| f16_to_f32(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])))
                .collect()
        };
        // 8-bit ring: dequant the live slots into the shared f16 scratch first, then
        // download+widen exactly as the f16 path (cold compaction stays f16→f32).
        let (hk, hv) = if self.hot_q8 {
            let dk = self
                .hot_deq_k
                .as_ref()
                .expect("hot_deq_k present when hot_q8");
            let dv = self
                .hot_deq_v
                .as_ref()
                .expect("hot_deq_v present when hot_q8");
            let hc = self.hot_count[layer];
            gpu.kv_hot_dequant_q8(
                &self.hot_kq[layer].codes,
                &self.hot_kq[layer].scale,
                dk,
                hc,
                hb,
                nkv,
                hd,
            )?;
            gpu.kv_hot_dequant_q8(
                &self.hot_vq[layer].codes,
                &self.hot_vq[layer].scale,
                dv,
                hc,
                hb,
                nkv,
                hd,
            )?;
            (
                widen(&gpu.download_raw(dk, ring_elems * 2)?),
                widen(&gpu.download_raw(dv, ring_elems * 2)?),
            )
        } else {
            (
                widen(&gpu.download_raw(&self.hot_k[layer], ring_elems * 2)?),
                widen(&gpu.download_raw(&self.hot_v[layer], ring_elems * 2)?),
            )
        };
        let mut ck = vec![0.0f32; mb * kv_dim];
        let mut cv = vec![0.0f32; mb * kv_dim];
        for t in 0..mb {
            for kv in 0..nkv {
                let src = (kv * hb + t) * hd;
                let dst = t * kv_dim + kv * hd;
                ck[dst..dst + hd].copy_from_slice(&hk[src..src + hd]);
                cv[dst..dst + hd].copy_from_slice(&hv[src..src + hd]);
            }
        }
        // Phase-2 (RoPE-dephased merge) ceiling capture. Debug-gated, no behavior
        // change when unset: append the post-RoPE K about to be merged, token-major
        // `[mb × kv_dim]`, with its absolute base position, so an offline analysis can
        // measure whether de-rotating a merge group collapses its intra-group variance
        // (blur is phase → lever has headroom) or not (blur is content → lever dead).
        // Record = [u32 base_pos][u32 mb][u32 nkv][u32 HD][f32 ck…]. See
        // docs/plans/2026-07-12-hot-cold-hierarchical-kv-implementation.md Phase 2.
        if let Ok(path) = std::env::var("HIPFIRE_KV_CAPTURE_K") {
            use std::io::Write;
            let mut buf = Vec::with_capacity(16 + ck.len() * 4);
            buf.extend_from_slice(&(self.migrated[layer] as u32).to_le_bytes());
            buf.extend_from_slice(&(mb as u32).to_le_bytes());
            buf.extend_from_slice(&(nkv as u32).to_le_bytes());
            buf.extend_from_slice(&(hd as u32).to_le_bytes());
            for &x in &ck {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                let _ = f.write_all(&buf);
            }
        }
        // Parallel V capture (`HIPFIRE_KV_CAPTURE_V`, same record format) for the
        // low-rank feasibility probe (Lever 5 gate): does static rank-r hold at HD=256?
        if let Ok(path) = std::env::var("HIPFIRE_KV_CAPTURE_V") {
            use std::io::Write;
            let mut buf = Vec::with_capacity(16 + cv.len() * 4);
            buf.extend_from_slice(&(self.migrated[layer] as u32).to_le_bytes());
            buf.extend_from_slice(&(mb as u32).to_le_bytes());
            buf.extend_from_slice(&(nkv as u32).to_le_bytes());
            buf.extend_from_slice(&(hd as u32).to_le_bytes());
            for &x in &cv {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                let _ = f.write_all(&buf);
            }
        }
        // Per-token importance for core selection + merge weighting. Norm proxies
        // pull the merged K toward the dominant token's RoPE phase (less blur) and
        // keep high-norm tokens exact; Attn = real accumulated attention mass
        // (CASK); uniform = the old average merge.
        let mass = if self.importance_mode == ImportanceMode::Attn {
            gpu.download_f32(&self.attn_mass[layer])?
        } else {
            Vec::new()
        };
        // TriAttn: per (kv-head, band) calibrated query-energy weight
        // W[kv*n_bands + b] = Σ_{query heads h in the GQA group of kv} ‖E[q_f]‖ from
        // the centers at this layer. Then score(token) = Σ_kv Σ_b W · ‖k_band‖ (the
        // key's RoPE-pair magnitude, rotation-invariant). None → vnorm fallback.
        let n_bands = hd / 2;
        let triattn_w: Option<Vec<f32>> = if self.importance_mode == ImportanceMode::TriAttn {
            self.centers
                .as_ref()
                .filter(|c| layer < c.n_layers && c.head_dim == hd)
                .map(|c| {
                    let group = (self.n_heads / nkv).max(1);
                    let mut w = vec![0.0f32; nkv * n_bands];
                    for kv in 0..nkv {
                        for h in (kv * group)..((kv + 1) * group).min(c.n_heads) {
                            for b in 0..n_bands {
                                w[kv * n_bands + b] += c.get(layer, h, b).magnitude();
                            }
                        }
                    }
                    w
                })
        } else {
            None
        };
        let importance: Vec<f32> = (0..mb)
            .map(|t| {
                let base = t * kv_dim;
                let kn = || {
                    (0..kv_dim)
                        .map(|d| ck[base + d] * ck[base + d])
                        .sum::<f32>()
                        .sqrt()
                };
                let vn = || {
                    (0..kv_dim)
                        .map(|d| cv[base + d] * cv[base + d])
                        .sum::<f32>()
                        .sqrt()
                };
                match self.importance_mode {
                    ImportanceMode::Uniform => 1.0,
                    ImportanceMode::VNorm => vn(),
                    ImportanceMode::KNorm => kn(),
                    ImportanceMode::KvNorm => kn() * vn(),
                    // Small floor so an unattended token still sorts/weights sanely.
                    ImportanceMode::Attn => mass[t] + 1e-6,
                    ImportanceMode::TriAttn => match &triattn_w {
                        Some(w) => {
                            let mut s = 0.0f32;
                            for kv in 0..nkv {
                                let kb = base + kv * hd;
                                for b in 0..n_bands {
                                    let kr = ck[kb + 2 * b];
                                    let ki = ck[kb + 2 * b + 1];
                                    s += w[kv * n_bands + b] * (kr * kr + ki * ki).sqrt();
                                }
                            }
                            s + 1e-6
                        }
                        None => vn(), // centers missing / layer out of range → vnorm
                    },
                }
            })
            .collect();
        let (layer_fold, layer_core) = self.layer_fold_core(layer, self.cold.len());
        let cold = compact_cold_kv(
            &ck,
            &cv,
            mb,
            nkv,
            hd,
            &importance,
            layer_core,
            layer_fold,
            false,
            self.position_local,
            self.similarity_merge,
            self.cold_qmax,
            self.cold_v_qmax,
            self.cold_v_perslot,
        );
        self.push_cold_segment(gpu, layer, &cold)?;
        self.migrated[layer] += mb;

        // Shift the remaining (hot_count - mb) tokens down to slots [0, ...).
        let rem = self.hot_count[layer] - mb;
        if rem > 0 {
            for kv in 0..nkv {
                if self.hot_q8 {
                    // int8 codes (1 B/elem) + f32 per-slot scale (4 B/slot).
                    let cdst = (kv * hb) * hd;
                    let csrc = (kv * hb + mb) * hd;
                    let sdst = (kv * hb) * 4;
                    let ssrc = (kv * hb + mb) * 4;
                    for ring in [&self.hot_kq[layer], &self.hot_vq[layer]] {
                        gpu.memcpy_dtod_at_auto(
                            &ring.codes.buf,
                            cdst,
                            &ring.codes.buf,
                            csrc,
                            rem * hd,
                        )?;
                        gpu.memcpy_dtod_at_auto(
                            &ring.scale.buf,
                            sdst,
                            &ring.scale.buf,
                            ssrc,
                            rem * 4,
                        )?;
                    }
                } else {
                    // f16 ring: 2 bytes/elem.
                    let dst = ((kv * hb) * hd) * 2;
                    let src = ((kv * hb + mb) * hd) * 2;
                    gpu.memcpy_dtod_at_auto(
                        &self.hot_k[layer].buf,
                        dst,
                        &self.hot_k[layer].buf,
                        src,
                        rem * hd * 2,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &self.hot_v[layer].buf,
                        dst,
                        &self.hot_v[layer].buf,
                        src,
                        rem * hd * 2,
                    )?;
                }
            }
        }
        // Mirror the shift for the attention-mass ring (slot s holds token s's mass),
        // then zero the vacated tail [rem, hot_budget) so reused slots start at 0.
        if self.importance_mode == ImportanceMode::Attn {
            if rem > 0 {
                gpu.memcpy_dtod_at_auto(
                    &self.attn_mass[layer].buf,
                    0,
                    &self.attn_mass[layer].buf,
                    mb * 4,
                    rem * 4,
                )?;
            }
            // Zero the vacated tail [rem, hot_budget) so reused slots start at 0
            // (the shift above already moved the surviving prefix down).
            let tail = hb - rem;
            if tail > 0 {
                let tail_view = self.attn_mass[layer].sub_offset(rem, tail);
                gpu.fill_f32(&tail_view, 0.0)?;
            }
        }
        self.hot_count[layer] = rem;
        Ok(())
    }

    /// Pack a freshly-produced `ColdTier` into GPU records and push it as one new
    /// cold segment for `layer`. Shared by `migrate_n` (hot→cold fold) and `defrag`
    /// (segment repack). K and V may pack at different bit widths → separate record
    /// strides; each is padded to a multiple of 4 for the f32-view upload. Per-slot
    /// V transposes the tile to `[n_slots × HD]`, so its record geometry differs
    /// from the K `[HD × n_slots]` layout.
    fn push_cold_segment(&mut self, gpu: &mut Gpu, layer: usize, cold: &ColdTier) -> HipResult<()> {
        let hd = self.head_dim;
        let nkv = self.n_kv_heads;
        let n_slots = cold.n_slots;
        let bits = self.cold_bits;
        let v_bits = self.cold_v_bits;
        let v_perslot = self.cold_v_perslot;
        let k_padded = kvarn_record_bytes_bits(hd, n_slots, bits).div_ceil(4) * 4;
        let v_padded = if v_perslot {
            kvarn_record_bytes_bits(n_slots, hd, v_bits).div_ceil(4) * 4
        } else {
            kvarn_record_bytes_bits(hd, n_slots, v_bits).div_ceil(4) * 4
        };
        let mut krecs = vec![0u8; nkv * k_padded];
        let mut vrecs = vec![0u8; nkv * v_padded];
        for h in 0..nkv {
            let kp = pack_kvarn_tile_bits(&cold.k_tiles[h], bits);
            let vp = pack_kvarn_tile_bits(&cold.v_tiles[h], v_bits);
            krecs[h * k_padded..h * k_padded + kp.len()].copy_from_slice(&kp);
            vrecs[h * v_padded..h * v_padded + vp.len()].copy_from_slice(&vp);
        }
        let k_recs = gpu.upload_raw(&krecs, &[nkv * k_padded / 4])?;
        let v_recs = gpu.upload_raw(&vrecs, &[nkv * v_padded / 4])?;
        self.cold[layer].push(ColdSegmentGpu {
            k_recs,
            v_recs,
            n_valid: cold.n_valid,
            n_slots,
            rec_bytes: k_padded,
            bits,
            v_rec_bytes: v_padded,
            v_bits,
            v_perslot,
        });
        Ok(())
    }

    /// Dequantize a resident cold segment back to token-major f32 `(K, V)`, each
    /// `[n_valid × kv_dim]`, in the original basis — the inverse of the migrate_n
    /// pack (mirrors `ColdTier::dequant_head`). Runtime cold segments are never
    /// FWHT-rotated (migrate_n / defrag pass `rotate=false`), so no inverse rotation
    /// is applied. Slots `[0, n_valid)` are the real ones; `[n_valid, n_slots)` are
    /// zero padding and are dropped here.
    fn dequant_segment_tokmajor(
        &self,
        gpu: &mut Gpu,
        seg: &ColdSegmentGpu,
    ) -> HipResult<(Vec<f32>, Vec<f32>)> {
        let hd = self.head_dim;
        let nkv = self.n_kv_heads;
        let kv_dim = self.kv_dim();
        let (nv, ns) = (seg.n_valid, seg.n_slots);
        // Records were uploaded as an f32-view (bytes/4); download and reinterpret
        // to the little-endian byte stream the unpack expects.
        let kf = gpu.download_f32(&seg.k_recs)?;
        let vf = gpu.download_f32(&seg.v_recs)?;
        let kbytes: Vec<u8> = kf.iter().flat_map(|x| x.to_le_bytes()).collect();
        let vbytes: Vec<u8> = vf.iter().flat_map(|x| x.to_le_bytes()).collect();
        let krb = kvarn_record_bytes_bits(hd, ns, seg.bits);
        let (vr, vc, vrb) = if seg.v_perslot {
            (ns, hd, kvarn_record_bytes_bits(ns, hd, seg.v_bits))
        } else {
            (hd, ns, kvarn_record_bytes_bits(hd, ns, seg.v_bits))
        };
        let mut k = vec![0.0f32; nv * kv_dim];
        let mut v = vec![0.0f32; nv * kv_dim];
        for h in 0..nkv {
            let kt = dequantize_tile(&unpack_kvarn_tile_bits(
                &kbytes[h * seg.rec_bytes..h * seg.rec_bytes + krb],
                hd,
                ns,
                seg.bits,
            )); // [HD × ns]
            let vt = dequantize_tile(&unpack_kvarn_tile_bits(
                &vbytes[h * seg.v_rec_bytes..h * seg.v_rec_bytes + vrb],
                vr,
                vc,
                seg.v_bits,
            )); // [vr × vc]
            for s in 0..nv {
                let dst = s * kv_dim + h * hd;
                for d in 0..hd {
                    k[dst + d] = kt[d * ns + s]; // channel-major → token-major
                    v[dst + d] = if seg.v_perslot {
                        vt[s * hd + d]
                    } else {
                        vt[d * ns + s]
                    };
                }
            }
        }
        Ok((k, v))
    }

    /// Idle-time cold-segment defragmentation (follow-up #2). `idle_compact` folds
    /// each turn's drain into ONE segment, so a layer accumulates ~1 segment/turn and
    /// the two-tier read pays one `attention_cold_slots`+`flash_tier_merge` per
    /// segment → read cost grows linearly with turn count. When a layer holds more
    /// than `max_segments` segments, dequant them all, concatenate their real slots,
    /// and re-pack into ONE wider tile via `compact_cold_kv(core_frac=1, fold_m=1)` —
    /// a pure repack (no further merge, so attention is unchanged up to one extra
    /// quant round on the oldest, least-important tokens). Win: bounded read cost +
    /// amortized per-channel scale overhead (fixed `r_dim*4 B/tile` now spans more
    /// slots). Attention over cold keys is permutation-invariant, so folding all
    /// segments and ignoring order is safe. Idle-path only (allocates); no-op unless
    /// enabled and a layer exceeds the threshold.
    pub fn defrag(&mut self, gpu: &mut Gpu, max_segments: usize) -> HipResult<()> {
        if !self.enabled || max_segments == 0 {
            return Ok(());
        }
        let hd = self.head_dim;
        let nkv = self.n_kv_heads;
        let kv_dim = self.kv_dim();
        let n_layers = self.cold.len();
        for layer in 0..n_layers {
            if self.cold[layer].len() <= max_segments {
                continue;
            }
            // ponytail: folds ALL segments (incl. a prior defrag result) into one, so
            // repeated idle_compact defrags re-quantize the whole cold history each
            // time → the oldest tokens accumulate requant coarsening (~1.6% output
            // error per fold-6→1, measured; parity_kv_hier defrag mode). Fine as an
            // opt-in (default off) that bounds read cost; if enabled by default, upgrade
            // to generational/LSM compaction (fold only same-generation small segments,
            // exempt the wide archive) to stop the compounding.
            let segs = std::mem::take(&mut self.cold[layer]);
            let total_valid: usize = segs.iter().map(|s| s.n_valid).sum();
            if total_valid == 0 {
                continue;
            }
            let mut k = vec![0.0f32; total_valid * kv_dim];
            let mut v = vec![0.0f32; total_valid * kv_dim];
            let mut off = 0usize;
            for seg in &segs {
                let (sk, sv) = self.dequant_segment_tokmajor(gpu, seg)?;
                let nv = seg.n_valid;
                k[off * kv_dim..(off + nv) * kv_dim].copy_from_slice(&sk);
                v[off * kv_dim..(off + nv) * kv_dim].copy_from_slice(&sv);
                off += nv;
            }
            // Repack the already-compacted slots into one wide tile: all-core
            // (core_frac=1) + fold_m=1 = every slot kept singleton, no re-merge.
            let importance = vec![1.0f32; total_valid];
            let cold = compact_cold_kv(
                &k,
                &v,
                total_valid,
                nkv,
                hd,
                &importance,
                1.0,
                1,
                false,
                false,
                false,
                self.cold_qmax,
                self.cold_v_qmax,
                self.cold_v_perslot,
            );
            self.push_cold_segment(gpu, layer, &cold)?;
        }
        Ok(())
    }

    /// Deferred between-turns compaction (the "deferred-hierarchical" thesis). Run
    /// in the idle gap after a turn ends, off the latency-critical path: drain each
    /// layer's hot ring down to `keep_recent` tokens, folding everything older into
    /// ONE cold segment per layer (big tile → better merge + amortized scale
    /// overhead). The next turn then starts with a near-empty hot ring but the full
    /// history present, compressed, in cold. No-op when a layer is already at/below
    /// `keep_recent`. Heavy compaction is justified here precisely because the user
    /// isn't waiting (single-user chat). Safe to call repeatedly.
    pub fn idle_compact(&mut self, gpu: &mut Gpu, keep_recent: usize) -> HipResult<()> {
        if !self.enabled {
            return Ok(());
        }
        let n_layers = self.hot_count.len();
        for layer in 0..n_layers {
            let hc = self.hot_count[layer];
            if hc > keep_recent {
                self.migrate_n(gpu, layer, hc - keep_recent)?;
            }
        }
        // Bound the accumulated per-turn segments (off the latency path).
        self.defrag(gpu, self.defrag_segments)?;
        Ok(())
    }

    fn ensure_scratch(&mut self, gpu: &mut Gpu, need_slots: usize) -> HipResult<()> {
        let hd = self.head_dim;
        let nh = self.n_heads;
        let nkv = self.n_kv_heads;
        let realloc = match &self.scr {
            None => true,
            Some(s) => need_slots > s.max_slots,
        };
        if realloc {
            let slots = need_slots.max(self.migrate_batch).max(1);
            self.scr = Some(HierScratch {
                acc_m: gpu.zeros(&[nh], DType::F32)?,
                acc_l: gpu.zeros(&[nh], DType::F32)?,
                out_c: gpu.zeros(&[nh * hd], DType::F32)?,
                m_c: gpu.zeros(&[nh], DType::F32)?,
                l_c: gpu.zeros(&[nh], DType::F32)?,
                // f16 dequant scratch: 2 bytes/elem.
                deq_k: gpu.upload_raw(&vec![0u8; nkv * hd * slots * 2], &[nkv * hd * slots])?,
                deq_v: gpu.upload_raw(&vec![0u8; nkv * hd * slots * 2], &[nkv * hd * slots])?,
                max_slots: slots,
            });
        }
        Ok(())
    }

    /// Two-tier decode read for one layer: hot (raw f32) ⊕ all cold segments, all
    /// folded by online-softmax merge into `out` ([n_heads × HD]). `q` = post-RoPE
    /// fa_q ([n_heads × HD]). The flash (m,l) accumulator is internal scratch.
    pub fn two_tier_read(
        &mut self,
        gpu: &mut Gpu,
        layer: usize,
        q: &GpuTensor,
        out: &GpuTensor,
    ) -> HipResult<()> {
        let hd = self.head_dim;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let nh = self.n_heads;
        let nkv = self.n_kv_heads;
        let max_seg = self.cold[layer]
            .iter()
            .map(|s| s.n_slots)
            .max()
            .unwrap_or(0);
        self.ensure_scratch(gpu, max_seg)?;
        // Take the scratch out to satisfy the borrow checker, then restore.
        let scr = self.scr.take().unwrap();

        // Phase 0 (rotated frame): hot K — and, via migrate, cold K — are stored
        // FWHT-rotated. Rotate the query once with the same orthonormal transform
        // and read BOTH tiers with it (q_rot·K_rot = q·K). V stays un-rotated, so
        // the attention output needs no inverse rotation.
        let q_read = if self.hot_rotate {
            let qr = self.q_rot.as_ref().expect("q_rot present when hot_rotate");
            self.rotate_hd(gpu, q, qr, nh * hd)?;
            qr
        } else {
            q
        };

        // Hot tier → accumulator (out/acc_m/acc_l). Slot-major f32, stride =
        // hot_budget so the live count reads from a fixed-width ring. When using
        // attention-mass importance, accumulate this query's per-token weight.
        let mass = if self.importance_mode == ImportanceMode::Attn {
            Some(&self.attn_mass[layer])
        } else {
            None
        };
        // 8-bit ring: dequant the live slots into the shared f16 scratch, then read
        // it with the existing layout-2 path (identical to the f16 ring downstream).
        let (hk, hv): (&GpuTensor, &GpuTensor) = if self.hot_q8 {
            let dk = self
                .hot_deq_k
                .as_ref()
                .expect("hot_deq_k present when hot_q8");
            let dv = self
                .hot_deq_v
                .as_ref()
                .expect("hot_deq_v present when hot_q8");
            let hc = self.hot_count[layer];
            gpu.kv_hot_dequant_q8(
                &self.hot_kq[layer].codes,
                &self.hot_kq[layer].scale,
                dk,
                hc,
                self.hot_budget,
                nkv,
                hd,
            )?;
            gpu.kv_hot_dequant_q8(
                &self.hot_vq[layer].codes,
                &self.hot_vq[layer].scale,
                dv,
                hc,
                self.hot_budget,
                nkv,
                hd,
            )?;
            (dk, dv)
        } else {
            (&self.hot_k[layer], &self.hot_v[layer])
        };
        gpu.attention_cold_slots(
            q_read,
            hk,
            hv,
            out,
            &scr.acc_m,
            &scr.acc_l,
            nh,
            nkv,
            self.hot_count[layer],
            scale,
            2, // k_layout: hot ring is slot-major f16 (dequant scratch when q8)
            2, // v_layout: hot ring is slot-major f16 (dequant scratch when q8)
            self.hot_budget,
            mass,
            hd,
        )?;

        // Fold each cold segment: dequant 4-bit → f16, channel-major attend, merge.
        for seg in &self.cold[layer] {
            gpu.kvarn_dequant_tile(
                &seg.k_recs,
                &scr.deq_k,
                nkv,
                hd,
                seg.n_slots,
                seg.rec_bytes,
                seg.bits,
            )?;
            // Per-slot V is a [n_slots × HD] tile (r=slot, c=channel) → swapped
            // r/c vs the K [HD × n_slots] tile; the dequant output is then
            // slot-major f16, read with v_layout=2 below.
            let (v_r, v_c) = if seg.v_perslot {
                (seg.n_slots, hd)
            } else {
                (hd, seg.n_slots)
            };
            gpu.kvarn_dequant_tile(
                &seg.v_recs,
                &scr.deq_v,
                nkv,
                v_r,
                v_c,
                seg.v_rec_bytes,
                seg.v_bits,
            )?;
            gpu.attention_cold_slots(
                q_read,
                &scr.deq_k,
                &scr.deq_v,
                &scr.out_c,
                &scr.m_c,
                &scr.l_c,
                nh,
                nkv,
                seg.n_valid,
                scale,
                1, // k_layout: per-channel var-norm dequant → channel-major f16
                if seg.v_perslot { 2 } else { 1 }, // v_layout: 2=slot-major f16 (per-slot V)
                seg.n_slots,
                None, // cold tier: no mass accumulation
                hd,
            )?;
            // Merge cold segment into the accumulator (in place — safe, see kernel).
            gpu.flash_tier_merge(
                out, &scr.acc_m, &scr.acc_l, &scr.out_c, &scr.m_c, &scr.l_c, out, &scr.acc_m,
                &scr.acc_l, nh, hd,
            )?;
        }
        self.scr = Some(scr);
        Ok(())
    }
}
