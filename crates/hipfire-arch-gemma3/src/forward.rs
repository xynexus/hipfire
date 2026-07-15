// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3 single-token decode forward. See LICENSE / NOTICE.

//! `Gemma3State` (per-decode GPU scratch + F32 KV cache) and the per-token
//! `forward_step`. Correctness-first bring-up: direct `gpu.*` kernel calls,
//! greedy/caller-sampled, one token at a time (prefill = N sequential calls),
//! modelled on `hipfire-arch-qwen2::forward_step`.
//!
//! Gemma3 layer body (4 norms; post-norms sit *inside* the residual, so the
//! fused gemv+residual can't be used — gemv → post-norm → explicit add):
//! ```text
//!   resid = x
//!   h = input_layernorm(x);  q,k,v = proj(h)
//!   q = q_norm(q); k = k_norm(k)              # per-head, q_norm carries the Q pre-scale
//!   rope(q,k, θ = global or local per layer)
//!   attn_out = GQA_attention(q,k,v, kv$);  o = o_proj(attn_out)
//!   x = resid + post_attention_layernorm(o)
//!   resid = x
//!   h = pre_feedforward_layernorm(x);  g,u = gate/up(h);  ffn = gelu_mul(g,u)
//!   o = down(ffn)
//!   x = resid + post_feedforward_layernorm(o)
//! ```
//! Embedding is scaled by √hidden_size before layer 0; norms are `(1+w)`-baked
//! at ingest so the plain rmsnorm kernel is correct. The attention kernel's
//! built-in `1/√head_dim` is corrected to `1/√query_pre_attn_scalar` by the Q
//! pre-scale baked into `q_norm` (see `load_weights`).

use hip_bridge::{DeviceBuffer, HipError, HipResult};
use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_runtime::kv::{KvCache, KvQuantMode};
use hipfire_runtime::layered_kv::LayeredKvArena;
use hipfire_runtime::llama::HiddenCaptureSink;
use hipfire_runtime::weights::{weight_gemm, weight_gemv, EmbeddingFormat, WeightTensor};

use crate::config::Gemma3Config;
use crate::weights::Gemma3Weights;

/// Default KV budget for bring-up validation (slots).
pub const DEFAULT_MAX_SEQ: usize = 4096;

/// Per-decode GPU scratch + F32 KV cache. `tmp` (size `hidden_size`) is reused
/// for every norm output; `o` (size `hidden_size`) holds both the attn and FFN
/// projection outputs before their post-norm + residual add.
pub struct Gemma3State {
    pub x: GpuTensor,          // residual stream [hidden]
    pub tmp: GpuTensor,        // norm-output scratch [hidden]
    pub q: GpuTensor,          // [n_heads*head_dim]
    pub k: GpuTensor,          // [n_kv*head_dim]
    pub v: GpuTensor,          // [n_kv*head_dim]
    pub attn_out: GpuTensor,   // [n_heads*head_dim]
    pub o: GpuTensor,          // projection output [hidden]
    pub gate: GpuTensor,       // [intermediate]
    pub up: GpuTensor,         // [intermediate]
    pub ffn_hidden: GpuTensor, // [intermediate]
    pub logits: GpuTensor,     // [vocab]
    pub pos_buf: DeviceBuffer,
    /// System KV cache (per-layer key/value GPU buffers + quant metadata). Its
    /// `physical_cap` is the per-layer slot stride and `quant_q8` selects the
    /// q8_0 (int8 + per-32-block fp16 scale) vs F32 storage — ~4× smaller KV,
    /// letting larger contexts fit (q8 requires `head_dim % 32 == 0`, true for
    /// gemma3: 128 @27b, 256 @4b). Shared with the llama/qwen backends; this is
    /// what unlocks the KVarN/asym quant modes and sliding-window KV sizing.
    pub kv_cache: KvCache,
    // ── Sliding-window attention (SWA) ────────────────────────────────
    // gemma3 interleaves 5 local (sliding_window) : 1 global layers. When SWA
    // is active (`swa_window > 0`), the KvCache above is FILTERED to the global
    // layers only, and each LOCAL layer keeps a small F32 ring buffer of the
    // last `swa_window` keys/values here instead of a full-context cache — the
    // memory win that lets gemma3 load at full context. Local-layer attention
    // reuses deepseek4's SWA primitives (per kv head): swa_visibility_stage →
    // attention_swa_gqa_batched → swa_ring_write, at batch size 1 (decode, and
    // per-token prefill). `swa_window == 0` disables SWA (all layers full — the
    // pre-SWA path).
    /// Per-layer F32 ring `[n_kv_heads, head_dim, swa_window]`; `Some` for local
    /// layers, `None` for global (which use `kv_cache`). Empty when SWA is off.
    pub swa_k: Vec<Option<GpuTensor>>,
    pub swa_v: Vec<Option<GpuTensor>>,
    /// B=1 head-major staging scratch `[n_kv_heads, head_dim, swa_window]` and
    /// the single `n_valid` (min(pos+1, window)) buffer for the windowed attn.
    pub swa_staged_k: GpuTensor,
    pub swa_staged_v: GpuTensor,
    pub swa_nvalid: GpuTensor,
    /// Sliding-window span (0 = SWA disabled).
    pub swa_window: usize,
    /// Next absolute KV write slot; bumped by `forward_step`.
    pub next_pos: usize,
    // ── KVarN scratch (Some only when `kv_cache.quant_kvarn`) ─────────────
    /// Reusable gather/quantize tile `[n_kv_heads × head_dim × GROUP]` for
    /// `kvarn_attend`; allocated once (per-call alloc would leak — GpuTensor has
    /// no pool-return Drop).
    pub kvarn_tiles: Option<GpuTensor>,
    /// FlashAttention partials for `attention_flash_kvarn_batched_masked`
    /// (`n_heads × ceil(max_seq/128) × (2 + head_dim)` F32).
    pub kvarn_flash_partials: Option<GpuTensor>,
}

impl Gemma3State {
    pub fn new(gpu: &mut Gpu, cfg: &Gemma3Config) -> Result<Self, String> {
        Self::new_with_max_seq(gpu, cfg, DEFAULT_MAX_SEQ, KvQuantMode::Unquantized, 4)
            .map_err(|e| format!("gemma3: Gemma3State::new failed: {e:?}"))
    }

    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &Gemma3Config,
        max_seq: usize,
        kv_mode: KvQuantMode,
        kvarn_bits: usize,
    ) -> HipResult<Self> {
        let dim = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let hidden_dim = cfg.intermediate_size;

        let n_layers = cfg.num_hidden_layers;
        let n_kv = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        // q8_0 KV requires head_dim divisible by the 32-element block; fall back
        // to F32 otherwise. KVarN (variance-normalized 4-bit K + Q8 V) requires
        // head_dim ∈ {128, 256} (the gemma3 shapes: 128 @27b, 256 @4b); the
        // optional Hadamard rotation engages only at 256. Anything else → F32.
        let kv_quant_q8 =
            matches!(kv_mode, KvQuantMode::Q8 | KvQuantMode::Int8) && head_dim.is_multiple_of(32);
        let kv_kvarn =
            matches!(kv_mode, KvQuantMode::Kvarn) && (head_dim == 128 || head_dim == 256);

        // Sliding-window attention: gemma3 interleaves 5 local : 1 global layers.
        // When SWA applies (window in (0, max_seq) and at least one local layer),
        // the KvCache carries only the GLOBAL layers (full context) and each
        // LOCAL layer keeps a small F32 ring of the last `swa_window` positions
        // — the memory win that lets gemma3 load at full context. Otherwise the
        // cache carries every layer (pre-SWA path).
        let has_local = (0..n_layers).any(|l| !cfg.is_global_layer(l));
        let swa_window = if cfg.sliding_window > 0 && cfg.sliding_window < max_seq && has_local {
            cfg.sliding_window
        } else {
            0
        };
        let swa = swa_window > 0;

        let kv_cache = if swa {
            let is_global: Vec<bool> = (0..n_layers).map(|l| cfg.is_global_layer(l)).collect();
            // KVarN/Q8 apply to the GLOBAL full-context layers (where the
            // long-context KV lives); local layers keep their own F32 rings.
            if kv_kvarn {
                KvCache::new_gpu_kvarn_filtered(
                    gpu, &is_global, n_kv, head_dim, max_seq, kvarn_bits,
                )?
            } else if kv_quant_q8 {
                KvCache::new_gpu_q8_filtered(gpu, &is_global, n_kv, head_dim, max_seq)?
            } else {
                KvCache::new_gpu_filtered(gpu, &is_global, n_kv, head_dim, max_seq)?
            }
        } else if kv_kvarn {
            KvCache::new_gpu_kvarn(gpu, n_layers, n_kv, head_dim, max_seq, kvarn_bits)?
        } else if kv_quant_q8 {
            KvCache::new_gpu_q8(gpu, n_layers, n_kv, head_dim, max_seq)?
        } else {
            LayeredKvArena::homogeneous_fp32_cache(
                gpu,
                n_layers,
                cfg.num_attention_heads,
                n_kv,
                head_dim,
                max_seq,
            )?
        };

        // KVarN needs two reusable scratch buffers (see field docs). Allocate
        // eagerly here so the single-token hot path never allocates. n=1 always
        // for gemma3 KVarN (decode + per-token prefill), so flash_partials needs
        // just one batch slot.
        let (kvarn_tiles, kvarn_flash_partials) = if kv_kvarn {
            let tiles = gpu.alloc_tensor(&[n_kv * head_dim * KvCache::KVARN_GROUP], DType::F32)?;
            let max_tiles = max_seq.div_ceil(KvCache::KVARN_GROUP);
            let partials = gpu.alloc_tensor(
                &[cfg.num_attention_heads * max_tiles * (2 + head_dim)],
                DType::F32,
            )?;
            (Some(tiles), Some(partials))
        } else {
            (None, None)
        };

        // Per-local-layer F32 rings + B=1 staging scratch. When SWA is off these
        // stay empty / 1-element dummies (never read).
        let ring_elems = n_kv * head_dim * swa_window.max(1);
        let mut swa_k: Vec<Option<GpuTensor>> = Vec::new();
        let mut swa_v: Vec<Option<GpuTensor>> = Vec::new();
        if swa {
            for l in 0..n_layers {
                if cfg.is_global_layer(l) {
                    swa_k.push(None);
                    swa_v.push(None);
                } else {
                    swa_k.push(Some(gpu.zeros(&[ring_elems], DType::F32)?));
                    swa_v.push(Some(gpu.zeros(&[ring_elems], DType::F32)?));
                }
            }
        }
        let staged_elems = if swa { ring_elems } else { 1 };
        let swa_staged_k = gpu.zeros(&[staged_elems], DType::F32)?;
        let swa_staged_v = gpu.zeros(&[staged_elems], DType::F32)?;
        let swa_nvalid = gpu.zeros(&[1], DType::F32)?;

        Ok(Self {
            x: gpu.alloc_tensor(&[dim], DType::F32)?,
            tmp: gpu.alloc_tensor(&[dim], DType::F32)?,
            q: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            k: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            v: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            attn_out: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            o: gpu.alloc_tensor(&[dim], DType::F32)?,
            gate: gpu.alloc_tensor(&[hidden_dim], DType::F32)?,
            up: gpu.alloc_tensor(&[hidden_dim], DType::F32)?,
            ffn_hidden: gpu.alloc_tensor(&[hidden_dim], DType::F32)?,
            logits: gpu.alloc_tensor(&[cfg.vocab_size], DType::F32)?,
            pos_buf: gpu.hip.malloc(4)?,
            kv_cache,
            swa_k,
            swa_v,
            swa_staged_k,
            swa_staged_v,
            swa_nvalid,
            swa_window,
            next_pos: 0,
            kvarn_tiles,
            kvarn_flash_partials,
        })
    }

    /// Rewind to position 0 (fresh conversation). KV slots are overwritten in
    /// place, so this is O(1).
    pub fn reset(&mut self) {
        self.next_pos = 0;
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.x,
            self.tmp,
            self.q,
            self.k,
            self.v,
            self.attn_out,
            self.o,
            self.gate,
            self.up,
            self.ffn_hidden,
            self.logits,
        ] {
            let _ = gpu.free_tensor(t);
        }
        self.kv_cache.free_gpu(gpu);
        for t in self.swa_k.into_iter().flatten() {
            let _ = gpu.free_tensor(t);
        }
        for t in self.swa_v.into_iter().flatten() {
            let _ = gpu.free_tensor(t);
        }
        for t in [self.swa_staged_k, self.swa_staged_v, self.swa_nvalid] {
            let _ = gpu.free_tensor(t);
        }
        for t in self
            .kvarn_tiles
            .into_iter()
            .chain(self.kvarn_flash_partials)
        {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.hip.free(self.pos_buf);
    }
}

fn prelude(gpu: &mut Gpu, state: &Gemma3State) -> HipResult<usize> {
    let pos = state.next_pos;
    if pos >= state.kv_cache.physical_cap {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "gemma3: forward_step pos={pos} >= max_seq={}; rebuild Gemma3State \
                 with a larger budget",
                state.kv_cache.physical_cap
            ),
        ));
    }
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(pos as i32).to_ne_bytes())?;
    Ok(pos)
}

/// Single-token decode: read `token` at `state.next_pos`, run the full stack,
/// write K/V at that position, leave logits in `state.logits`, bump `next_pos`.
pub fn forward_step(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    cfg: &Gemma3Config,
    state: &mut Gemma3State,
    token: u32,
) -> HipResult<()> {
    let pos = prelude(gpu, state)?;

    // Embedding lookup + Gemma √hidden scale → x.
    embed_token(gpu, weights, cfg, &state.x, token)?;

    forward_after_x(gpu, weights, cfg, state, pos)?;
    state.next_pos += 1;
    Ok(())
}

/// `forward_step` with an optional extract-layer hidden-capture sink. Runs the
/// same per-token stack (so it is bit-identical to `forward_step` and thus
/// greedy-equivalent to AR decode), but when `capture` is `Some` appends the
/// residual at the sink's extract layers to `sink.hidden`. Used by the gemma3
/// `SpecTarget` per-token verify/advance path and by DSpark/DFlash label-gen.
pub fn forward_step_capture(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    cfg: &Gemma3Config,
    state: &mut Gemma3State,
    token: u32,
    capture: Option<&mut HiddenCaptureSink>,
) -> HipResult<()> {
    let pos = prelude(gpu, state)?;
    embed_token(gpu, weights, cfg, &state.x, token)?;
    forward_after_x_capture(gpu, weights, cfg, state, pos, capture)?;
    state.next_pos += 1;
    Ok(())
}

/// Decode one position from a **prebuilt embedding** instead of an embedded
/// token — the image-token splice primitive for gemma3-vl. The multimodal
/// projector output already lives in the text embedding space and is inserted
/// into the (already-scaled) text stream **unscaled**, so this path does NOT
/// apply the `√hidden` embed scale. `embedding` is one row of `hidden_size`
/// F32s. Mirrors `hipfire-arch-qwen2::forward_step_with_embed`.
pub fn forward_step_with_embed(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    cfg: &Gemma3Config,
    state: &mut Gemma3State,
    embedding: &[f32],
) -> HipResult<()> {
    let dim = cfg.hidden_size;
    if embedding.len() != dim {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "gemma3: forward_step_with_embed expects {dim} F32s, got {}",
                embedding.len()
            ),
        ));
    }
    let pos = prelude(gpu, state)?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(embedding.as_ptr() as *const u8, embedding.len() * 4) };
    gpu.hip.memcpy_htod(&state.x.buf, bytes)?;
    // NB: no embed_scale here — the image embedding is inserted at its own
    // magnitude (only token embeddings get the √hidden normalizer).
    forward_after_x(gpu, weights, cfg, state, pos)?;
    state.next_pos += 1;
    Ok(())
}

/// Debug: when `HIPFIRE_LM_DUMP=<dir>` is set, write a decoder-stage tensor to
/// `<dir>/<name>.bin` (f32 LE) + `<dir>/<name>.json` for validation against an
/// HF reference (benchmarks/vision/diff_dumps.py). Called per forward, so for a
/// per-token prefill the LAST call's tensors (= last prompt position) survive.
/// No-op when unset.
fn maybe_dump_lm(gpu: &mut Gpu, t: &GpuTensor, name: &str, shape: &[usize]) {
    let dir = match std::env::var("HIPFIRE_LM_DUMP") {
        Ok(d) if !d.is_empty() => d,
        _ => return,
    };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    if let Ok(data) = gpu.download_f32(t) {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let _ = std::fs::write(format!("{dir}/{name}.bin"), bytes);
        let _ = std::fs::write(
            format!("{dir}/{name}.json"),
            format!("{{\"shape\":{shape:?}}}"),
        );
    }
}

fn forward_after_x(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    cfg: &Gemma3Config,
    state: &mut Gemma3State,
    pos: usize,
) -> HipResult<()> {
    forward_after_x_capture(gpu, weights, cfg, state, pos, None)
}

/// `forward_after_x` with an optional extract-layer hidden-capture sink.
///
/// The DSpark/DFlash drafter's `main_proj` ingests the target's residual hidden
/// at a set of extract layers. When `capture` is `Some`, this appends the settled
/// post-FFN residual (`state.x`, the fusion-proof block-boundary stream) at each
/// layer in `sink.extract_layers` to `sink.hidden`, in ascending layer order —
/// matching `hipfire_runtime::llama::forward_scratch_compute_capture`'s host-Vec
/// layout for a single position (`[num_extract × dim]` per call). Only the host
/// `hidden` Vec sink is supported on this per-token path; the GPU-resident
/// `hidden_gpu` sink is the batched (M1b) verify's job.
fn forward_after_x_capture(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    cfg: &Gemma3Config,
    state: &mut Gemma3State,
    pos: usize,
    mut capture: Option<&mut HiddenCaptureSink>,
) -> HipResult<()> {
    let n_heads = cfg.num_attention_heads;
    let n_kv_heads = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let eps = cfg.rms_norm_eps;

    for layer_idx in 0..cfg.num_hidden_layers {
        let layer = &weights.layers[layer_idx];

        // ── Attention block ──────────────────────────────────────────
        gpu.rmsnorm_f32(&state.x, &layer.input_norm, &state.tmp, eps)?;
        weight_gemv(gpu, &layer.wq, &state.tmp, &state.q)?;
        weight_gemv(gpu, &layer.wk, &state.tmp, &state.k)?;
        weight_gemv(gpu, &layer.wv, &state.tmp, &state.v)?;

        // Per-head QK-norm (q_norm carries the baked Q pre-scale).
        Gpu::rmsnorm_batched(
            gpu,
            &state.q,
            &layer.q_norm,
            &state.q,
            n_heads,
            head_dim,
            eps,
        )?;
        Gpu::rmsnorm_batched(
            gpu,
            &state.k,
            &layer.k_norm,
            &state.k,
            n_kv_heads,
            head_dim,
            eps,
        )?;

        // Dual-θ RoPE: global layers use rope_theta, local use rope_local_base_freq.
        gpu.rope_f32(
            &state.q,
            &state.k,
            &state.pos_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            cfg.rope_base_for_layer(layer_idx),
        )?;

        // Attention. Three routes:
        //   - SWA local layer: F32 ring + windowed staged attention (batch 1).
        //   - global layer (or SWA off): full-context KvCache attention.
        if state.swa_window > 0 && !cfg.is_global_layer(layer_idx) {
            let win = state.swa_window;
            let hdw = head_dim * win;
            let ring_k = state.swa_k[layer_idx].as_ref().unwrap();
            let ring_v = state.swa_v[layer_idx].as_ref().unwrap();
            // n_valid = min(pos+1, window) written into the [1] scalar buffer.
            let nv = ((pos + 1).min(win)) as i32;
            gpu.hip
                .memcpy_htod(&state.swa_nvalid.buf, &nv.to_ne_bytes())?;
            // Stage each kv head's visible window from its ring + this token's KV.
            for kvh in 0..n_kv_heads {
                gpu.swa_visibility_stage_batched(
                    &ring_k.sub_offset(kvh * hdw, hdw),
                    &state.k.sub_offset(kvh * head_dim, head_dim),
                    &state.swa_staged_k.sub_offset(kvh * hdw, hdw),
                    pos as i32,
                    win as i32,
                    head_dim as i32,
                    1,
                )?;
                gpu.swa_visibility_stage_batched(
                    &ring_v.sub_offset(kvh * hdw, hdw),
                    &state.v.sub_offset(kvh * head_dim, head_dim),
                    &state.swa_staged_v.sub_offset(kvh * hdw, hdw),
                    pos as i32,
                    win as i32,
                    head_dim as i32,
                    1,
                )?;
            }
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            gpu.attention_swa_gqa_batched(
                &state.q,
                &state.swa_staged_k,
                &state.swa_staged_v,
                &state.swa_nvalid,
                &state.attn_out,
                n_heads,
                n_kv_heads,
                head_dim,
                win,
                1,
                scale,
            )?;
            // Advance each kv head's ring with this token's KV (slot pos%window).
            for kvh in 0..n_kv_heads {
                gpu.swa_ring_write_batched_f32(
                    &state.k.sub_offset(kvh * head_dim, head_dim),
                    &ring_k.sub_offset(kvh * hdw, hdw),
                    1,
                    head_dim as i32,
                    win as i32,
                    pos as i32,
                    1,
                )?;
                gpu.swa_ring_write_batched_f32(
                    &state.v.sub_offset(kvh * head_dim, head_dim),
                    &ring_v.sub_offset(kvh * hdw, hdw),
                    1,
                    head_dim as i32,
                    win as i32,
                    pos as i32,
                    1,
                )?;
            }
        } else if state.kv_cache.quant_kvarn {
            // KVarN (variance-normalized 4-bit K + Q8 V), n=1. `kvarn_attend`
            // fuses V-write, K window-append + 128-token block flush, and the
            // fused flash read over [0, pos+1). Under SWA this branch is reached
            // only for GLOBAL layers (locals took the ring branch above); with
            // SWA off it serves every layer. Prefill routes here per-token too
            // (see forward_prefill's KVarN guard), so this one hook covers
            // prompt + decode.
            //
            // Optional Hadamard-incoherence rotation: rotate K and Q by the SAME
            // orthonormal per-head FWHT-256, so (RQ)·(RK)ᵀ = Q·Kᵀ exactly — scores
            // preserved, no un-rotation, no flash/dequant change. K lands in the
            // cache rotated (self-consistent); V (Q8) stays un-rotated so the
            // output basis and o_proj are unchanged. Requires head_dim==256; opt
            // out with HIPFIRE_KVARN_ROTATE=0.
            static KVARN_ROTATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let kvarn_rotate = *KVARN_ROTATE
                .get_or_init(|| std::env::var("HIPFIRE_KVARN_ROTATE").ok().as_deref() != Some("0"));
            if kvarn_rotate && head_dim == 256 {
                gpu.rotate_x_mq_batched(&state.k, &state.k, n_kv_heads * head_dim, 1)?;
                gpu.rotate_x_mq_batched(&state.q, &state.q, n_heads * head_dim, 1)?;
            }
            // The KV kernels read positions from a GpuTensor; wrap the raw 4-byte
            // i32 `pos_buf` as a non-owning [1] view (mirrors qwen35's KVarN hook).
            let pos_view = GpuTensor {
                buf: unsafe { DeviceBuffer::from_raw(state.pos_buf.as_ptr(), 4) },
                shape: vec![1],
                dtype: DType::F32,
            };
            gpu.kvarn_attend(
                &state.kv_cache.k_gpu[layer_idx],
                &state.kv_cache.k_window[layer_idx],
                &state.kv_cache.v_gpu[layer_idx],
                &state.q,
                &state.k,
                &state.v,
                &pos_view,
                &state.attn_out,
                state.kvarn_flash_partials.as_ref().unwrap(),
                state.kvarn_tiles.as_ref().unwrap(),
                1,
                pos,
                n_heads,
                n_kv_heads,
                head_dim,
                state.kv_cache.physical_cap,
                None,
                0,
                0,
                state.kv_cache.kvarn_bits,
            )?;
        } else if state.kv_cache.quant_q8 {
            gpu.kv_cache_write_q8_0(
                &state.kv_cache.k_gpu[layer_idx],
                &state.k,
                &state.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_q8_0(
                &state.kv_cache.v_gpu[layer_idx],
                &state.v,
                &state.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_q8_0_kv(
                &state.q,
                &state.kv_cache.k_gpu[layer_idx],
                &state.kv_cache.v_gpu[layer_idx],
                &state.attn_out,
                &state.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                state.kv_cache.physical_cap,
            )?;
        } else {
            gpu.kv_cache_write(
                &state.kv_cache.k_gpu[layer_idx],
                &state.k,
                &state.pos_buf,
                kv_dim,
            )?;
            gpu.kv_cache_write(
                &state.kv_cache.v_gpu[layer_idx],
                &state.v,
                &state.pos_buf,
                kv_dim,
            )?;
            Gpu::attention_f32(
                gpu,
                &state.q,
                &state.kv_cache.k_gpu[layer_idx],
                &state.kv_cache.v_gpu[layer_idx],
                &state.attn_out,
                &state.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                state.kv_cache.physical_cap,
            )?;
        }

        weight_gemv(gpu, &layer.wo, &state.attn_out, &state.o)?;
        // post_attention_layernorm sits INSIDE the residual: norm(o) then add.
        gpu.rmsnorm_f32(&state.o, &layer.post_attn_norm, &state.tmp, eps)?;
        gpu.add_f32(&state.x, &state.tmp, &state.x)?;

        // ── FFN block (GeGLU) ────────────────────────────────────────
        gpu.rmsnorm_f32(&state.x, &layer.pre_ffn_norm, &state.tmp, eps)?;
        weight_gemv(gpu, &layer.w_gate, &state.tmp, &state.gate)?;
        weight_gemv(gpu, &layer.w_up, &state.tmp, &state.up)?;
        gpu.gelu_mul_f32(&state.gate, &state.up, &state.ffn_hidden)?;
        // H-Neuron intervention gain (no-op unless a session is active): scale the
        // down_proj input in place before down. Decode is a single position.
        hipfire_hneurons::intervene::maybe_intervene_ffn(gpu, &state.ffn_hidden, layer_idx, 1)?;
        weight_gemv(gpu, &layer.w_down, &state.ffn_hidden, &state.o)?;
        // post_feedforward_layernorm, also inside the residual.
        gpu.rmsnorm_f32(&state.o, &layer.post_ffn_norm, &state.tmp, eps)?;
        gpu.add_f32(&state.x, &state.tmp, &state.x)?;
        // Block-boundary steering/abliteration hook (no-op unless a session is
        // active). `state.x` is the settled post-residual stream — fusion-proof.
        hipfire_steer::maybe_steer_block(gpu, &state.x, layer_idx)?;
        // DSpark/DFlash extract-layer capture: append the settled residual at the
        // requested layers (ascending) to the host sink. See fn doc.
        if let Some(sink) = capture.as_deref_mut() {
            if sink.extract_layers.contains(&layer_idx) {
                if sink.hidden_gpu.is_some() {
                    return Err(HipError::new(
                        0,
                        "gemma3 per-token capture: hidden_gpu sink unsupported \
                         (host Vec only; GPU-resident capture is the batched M1b path)",
                    ));
                }
                let row = gpu.download_f32(&state.x)?;
                sink.hidden.extend_from_slice(&row);
            }
        }
        maybe_dump_lm(
            gpu,
            &state.x,
            &format!("lm_block_{layer_idx:02}"),
            &[cfg.hidden_size],
        );
    }

    // Final norm + lm_head.
    gpu.rmsnorm_f32(&state.x, &weights.output_norm, &state.tmp, eps)?;
    weight_gemv(gpu, &weights.output, &state.tmp, &state.logits)?;
    maybe_dump_lm(gpu, &state.logits, "lm_logits", &[cfg.vocab_size]);
    Ok(())
}

/// Embed one text token (format-dispatched lookup + Gemma √hidden scale) into
/// `dest` (`[dim]`). The embedding step shared by `forward_step` and the
/// batched-prefill input builder. Image rows bypass this (spliced unscaled).
pub fn embed_token(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    cfg: &Gemma3Config,
    dest: &GpuTensor,
    token: u32,
) -> HipResult<()> {
    let dim = cfg.hidden_size;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, dest, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, dest, token, dim)?
        }
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, dest, token, dim)?,
        EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(&weights.token_embd, dest, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, dest, token, dim)?,
    }
    // Gemma scales the embedding by √hidden_size before the first layer; the
    // scale rides the residual stream (rmsnorm cancels it locally but each
    // residual add re-injects it).
    gpu.scale_f32(dest, cfg.embed_scale())?;
    Ok(())
}

/// Prefill linear for the batched gemma3 path. Routes Q8_0 weights through the
/// no-LDS WMMA kernel (`gemm_q8_0_wmma`, ~11–30× over the scalar
/// `gemm_q8_0_batched` that `weight_gemm`'s chunked driver otherwise selects on
/// non-RDNA4 arches) whenever the GPU has w32 WMMA (RDNA3/3.5/4) and `K % 32 ==
/// 0`; anything else falls back to the generic `weight_gemm` dispatch. Preserves
/// the imatrix/Hessian activation tap `weight_gemm` performs so batched
/// calibration still works on gemma3. Portability: RDNA2 (no WMMA) stays on the
/// scalar path via the fallback, and the kernel is register-tiled (no LDS), so
/// it is safe on the gfx1103 LDS-fault class.
fn prefill_linear(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
) -> HipResult<()> {
    if w.gpu_dtype == DType::Q8_0 && gpu.arch_caps.has_wmma_w32() && w.k % 32 == 0 {
        // Mirror `weight_gemm`'s batched calibration tap before dispatch.
        gpu.maybe_capture_activation(&w.buf, x, m, w.k);
        gpu.gemm_q8_0_wmma(&w.buf, x, y, w.m, w.k, m)
    } else {
        weight_gemm(gpu, w, x, y, m)
    }
}

/// Batched prefill of `m` already-embedded tokens. `x_batch` is `[m, dim]`
/// row-major — the caller embeds text tokens (×`embed_scale`) and splices image
/// rows (unscaled) just like the per-token path. Writes KV for absolute
/// positions `start_pos..start_pos+m`, advances `state.next_pos`, and leaves the
/// LAST position's logits in `state.logits`. Numerically equivalent to running
/// `m` sequential `forward_step`s (both full-causal) but reads each weight once
/// instead of `m` times — the multi-image / long-prompt prefill speedup.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    cfg: &Gemma3Config,
    state: &mut Gemma3State,
    x_batch: &GpuTensor,
    m: usize,
    start_pos: usize,
) -> HipResult<()> {
    let dim = cfg.hidden_size;
    let n_heads = cfg.num_attention_heads;
    let n_kv_heads = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let inter = cfg.intermediate_size;
    let eps = cfg.rms_norm_eps;
    let max_ctx = start_pos + m;

    // SWA safety net: the batched attention below assumes every layer carries a
    // full-context KvCache, which is false once SWA filters the cache to global
    // layers (local layers live in per-layer rings, correctly advanced only by
    // the single-token path). The inference callers already route per-token when
    // SWA is active; fall back per-token here too so any other caller (e.g.
    // calibration) stays correct — copy each row's embedding into `state.x`, run
    // the single-token stack, then emit the last position's logits.
    //
    // KVarN forces the same per-token route regardless of SWA: its window/block
    // write + fused flash (`kvarn_attend`) is a single-token (n=1) primitive with
    // no batched variant, so a batched prefill would corrupt the K records.
    if state.swa_window > 0 || state.kv_cache.quant_kvarn {
        for i in 0..m {
            gpu.memcpy_dtod_at_auto(&state.x.buf, 0, &x_batch.buf, i * dim * 4, dim * 4)?;
            gpu.hip
                .memcpy_htod(&state.pos_buf, &((start_pos + i) as i32).to_ne_bytes())?;
            forward_after_x(gpu, weights, cfg, state, start_pos + i)?;
        }
        gpu.rmsnorm_f32(&state.x, &weights.output_norm, &state.tmp, eps)?;
        weight_gemv(gpu, &weights.output, &state.tmp, &state.logits)?;
        state.next_pos = start_pos + m;
        return Ok(());
    }

    // Absolute positions start_pos..start_pos+m as an i32 device table (dtype is
    // cosmetic — the rope/attention/kv kernels read it as `const int*`).
    let pos_vals: Vec<i32> = (start_pos as i32..(start_pos + m) as i32).collect();
    let positions = gpu.alloc_owned(&[m], DType::F32)?;
    {
        let bytes: Vec<u8> = pos_vals.iter().flat_map(|p| p.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&positions.buf, &bytes)?;
    }

    // Batched scratch. `OwnedTensor` frees itself back to the pool on drop, on
    // EVERY exit path (`?` errors and panics included).
    let tmp = gpu.alloc_owned(&[m * dim], DType::F32)?;
    let q = gpu.alloc_owned(&[m * q_dim], DType::F32)?;
    let k = gpu.alloc_owned(&[m * kv_dim], DType::F32)?;
    let v = gpu.alloc_owned(&[m * kv_dim], DType::F32)?;
    let attn_out = gpu.alloc_owned(&[m * q_dim], DType::F32)?;
    let o = gpu.alloc_owned(&[m * dim], DType::F32)?;
    let gate = gpu.alloc_owned(&[m * inter], DType::F32)?;
    let up = gpu.alloc_owned(&[m * inter], DType::F32)?;
    let ffn = gpu.alloc_owned(&[m * inter], DType::F32)?;

    for layer_idx in 0..cfg.num_hidden_layers {
        let layer = &weights.layers[layer_idx];

        // ── Attention block ──
        gpu.rmsnorm_batched(x_batch, &layer.input_norm, &tmp, m, dim, eps)?;
        prefill_linear(gpu, &layer.wq, &tmp, &q, m)?;
        prefill_linear(gpu, &layer.wk, &tmp, &k, m)?;
        prefill_linear(gpu, &layer.wv, &tmp, &v, m)?;

        // Per-head QK-norm (q_norm carries the baked Q pre-scale): m*heads groups.
        gpu.rmsnorm_batched(&q, &layer.q_norm, &q, m * n_heads, head_dim, eps)?;
        gpu.rmsnorm_batched(&k, &layer.k_norm, &k, m * n_kv_heads, head_dim, eps)?;

        gpu.rope_batched_f32(
            &q,
            &k,
            &positions,
            n_heads,
            n_kv_heads,
            head_dim,
            cfg.rope_base_for_layer(layer_idx),
            m,
        )?;

        if state.kv_cache.quant_q8 {
            gpu.kv_cache_write_q8_0_batched(
                &state.kv_cache.k_gpu[layer_idx],
                &k,
                &positions,
                n_kv_heads,
                head_dim,
                m,
            )?;
            gpu.kv_cache_write_q8_0_batched(
                &state.kv_cache.v_gpu[layer_idx],
                &v,
                &positions,
                n_kv_heads,
                head_dim,
                m,
            )?;
            gpu.attention_q8_0_kv_batched(
                &q,
                &state.kv_cache.k_gpu[layer_idx],
                &state.kv_cache.v_gpu[layer_idx],
                &attn_out,
                &positions,
                n_heads,
                n_kv_heads,
                head_dim,
                state.kv_cache.physical_cap,
                max_ctx,
                m,
            )?;
        } else {
            gpu.kv_cache_write_f32_batched(
                &state.kv_cache.k_gpu[layer_idx],
                &k,
                &positions,
                kv_dim,
                m,
            )?;
            gpu.kv_cache_write_f32_batched(
                &state.kv_cache.v_gpu[layer_idx],
                &v,
                &positions,
                kv_dim,
                m,
            )?;
            gpu.attention_f32_batched(
                &q,
                &state.kv_cache.k_gpu[layer_idx],
                &state.kv_cache.v_gpu[layer_idx],
                &attn_out,
                &positions,
                n_heads,
                n_kv_heads,
                head_dim,
                state.kv_cache.physical_cap,
                max_ctx,
                m,
            )?;
        }

        prefill_linear(gpu, &layer.wo, &attn_out, &o, m)?;
        gpu.rmsnorm_batched(&o, &layer.post_attn_norm, &tmp, m, dim, eps)?;
        gpu.add_f32(x_batch, &tmp, x_batch)?;

        // ── FFN block (GeGLU) ──
        gpu.rmsnorm_batched(x_batch, &layer.pre_ffn_norm, &tmp, m, dim, eps)?;
        prefill_linear(gpu, &layer.w_gate, &tmp, &gate, m)?;
        prefill_linear(gpu, &layer.w_up, &tmp, &up, m)?;
        gpu.gelu_mul_f32(&gate, &up, &ffn)?;
        // H-Neuron intervention gain (no-op unless active): scale the down_proj
        // input in place before down, across all m positions.
        hipfire_hneurons::intervene::maybe_intervene_ffn(gpu, &ffn, layer_idx, m)?;
        prefill_linear(gpu, &layer.w_down, &ffn, &o, m)?;
        // H-Neurons CETT tap (no-op unless a capture session is active). `o` holds
        // the raw down_proj output here — the residual add below folds
        // post_ffn_norm(o), so both down_proj input (`ffn`) and output (`o`) stay
        // materialized. `start_pos` is the global position of this chunk's row 0.
        hipfire_hneurons::capture::maybe_capture_ffn(gpu, &ffn, &o, layer_idx, start_pos, m)?;
        gpu.rmsnorm_batched(&o, &layer.post_ffn_norm, &tmp, m, dim, eps)?;
        gpu.add_f32(x_batch, &tmp, x_batch)?;
        // Block-boundary steering/abliteration hook (no-op unless active).
        // Prefill convention: capture folds the last position, apply hits all.
        hipfire_steer::maybe_steer_block_batched(gpu, x_batch, layer_idx, m, dim)?;
    }

    // Final norm + lm_head on the LAST position only (the next-token logits).
    gpu.memcpy_dtod_at_auto(&state.x.buf, 0, &x_batch.buf, (m - 1) * dim * 4, dim * 4)?;
    gpu.rmsnorm_f32(&state.x, &weights.output_norm, &state.tmp, eps)?;
    weight_gemv(gpu, &weights.output, &state.tmp, &state.logits)?;

    // The per-call pooled scratch above (`OwnedTensor`) returned itself to the
    // deferred-free mailbox on drop; drain the mailbox at this forward boundary.
    gpu.reclaim_pending();
    state.next_pos = start_pos + m;
    Ok(())
}

/// Batched block VERIFY forward for gemma3 spec-decode (M1b). Runs the `m` block
/// positions `[start_pos, start_pos+m)` in ONE forward — **local (SWA) layers**
/// via the batched `swa_*_batched` primitives (per-head gather → stage →
/// `attention_swa_gqa_batched` → ring-write), **global layers** via the batched
/// full-context KvCache attention — and returns the per-position logits
/// (`[m, vocab]` host, row-major). Optionally scatters the settled residual at
/// `extract_layers` (ascending) into `hidden_gpu` (`[m, n_extract, dim]` F32) for
/// the DSpark/DFlash drafter's on-device `main_hidden`.
///
/// Prereq: the prior context KV `[0, start_pos)` must already be resident — global
/// layers in `state.kv_cache.k_gpu[l]`, local layers in the `state.swa_k/v[l]`
/// rings — exactly as the per-token prefill leaves them. Advances
/// `state.next_pos` to `start_pos+m`.
///
/// This is the spec-decode *speedup*: reads each weight once for the whole block
/// instead of `m` times. It is a verify parity of `m` sequential `forward_step`s
/// (same KV/ring writes at the same absolute positions, same per-layer math); the
/// only permissible divergence is float reduction order flipping an argmax on a
/// near-tie — the batched result is the verifier's truth, as in the llama path.
///
/// KVarN KV is unsupported (its window/block write is a strict n=1 fused
/// primitive); callers fall back to per-token verify for a KVarN state.
#[allow(clippy::too_many_arguments)]
pub fn forward_verify_batch(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    cfg: &Gemma3Config,
    state: &mut Gemma3State,
    x_batch: &GpuTensor,
    m: usize,
    start_pos: usize,
    extract_layers: &[usize],
    hidden_gpu: Option<&GpuTensor>,
) -> HipResult<Vec<f32>> {
    if state.kv_cache.quant_kvarn {
        return Err(HipError::new(
            0,
            "gemma3 forward_verify_batch: KVarN KV unsupported (use per-token verify)",
        ));
    }
    let dim = cfg.hidden_size;
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv * head_dim;
    let inter = cfg.intermediate_size;
    let eps = cfg.rms_norm_eps;
    let vocab = cfg.vocab_size;
    let max_ctx = start_pos + m;
    let win = state.swa_window; // 0 ⇒ SWA off (every layer full-context global)
    let swa = win > 0;

    // Absolute positions [start_pos, start_pos+m) as an i32-in-F32 device table.
    let positions = gpu.alloc_owned(&[m], DType::F32)?;
    {
        let bytes: Vec<u8> = (start_pos..start_pos + m)
            .flat_map(|p| (p as i32).to_ne_bytes())
            .collect();
        gpu.hip.memcpy_htod(&positions.buf, &bytes)?;
    }

    // Batched working scratch (pooled; returned to the pool on drop / reclaim).
    let tmp = gpu.alloc_owned(&[m * dim], DType::F32)?;
    let q = gpu.alloc_owned(&[m * q_dim], DType::F32)?;
    let k = gpu.alloc_owned(&[m * kv_dim], DType::F32)?;
    let v = gpu.alloc_owned(&[m * kv_dim], DType::F32)?;
    let attn_out = gpu.alloc_owned(&[m * q_dim], DType::F32)?;
    let o = gpu.alloc_owned(&[m * dim], DType::F32)?;
    let gate = gpu.alloc_owned(&[m * inter], DType::F32)?;
    let up = gpu.alloc_owned(&[m * inter], DType::F32)?;
    let ffn = gpu.alloc_owned(&[m * inter], DType::F32)?;

    // SWA-only scratch: head-major gather buffers, head-major staged windows
    // ([n_kv, m, head_dim, win]), and the per-position n_valid[m] table.
    let (k_hm, v_hm, staged_k, staged_v, nvalid) = if swa {
        let nvalid = gpu.alloc_owned(&[m], DType::F32)?;
        let nv: Vec<u8> = (0..m)
            .flat_map(|b| ((start_pos + b + 1).min(win) as i32).to_ne_bytes())
            .collect();
        gpu.hip.memcpy_htod(&nvalid.buf, &nv)?;
        (
            Some(gpu.alloc_owned(&[n_kv * m * head_dim], DType::F32)?),
            Some(gpu.alloc_owned(&[n_kv * m * head_dim], DType::F32)?),
            Some(gpu.alloc_owned(&[n_kv * m * head_dim * win], DType::F32)?),
            Some(gpu.alloc_owned(&[n_kv * m * head_dim * win], DType::F32)?),
            Some(nvalid),
        )
    } else {
        (None, None, None, None, None)
    };

    let n_extract = extract_layers.len();
    let mut e_next = 0usize; // cursor into ascending extract_layers

    for layer_idx in 0..cfg.num_hidden_layers {
        let layer = &weights.layers[layer_idx];

        // ── Attention: input norm, QKV, per-head QK-norm, dual-θ RoPE ──
        gpu.rmsnorm_batched(x_batch, &layer.input_norm, &tmp, m, dim, eps)?;
        prefill_linear(gpu, &layer.wq, &tmp, &q, m)?;
        prefill_linear(gpu, &layer.wk, &tmp, &k, m)?;
        prefill_linear(gpu, &layer.wv, &tmp, &v, m)?;
        gpu.rmsnorm_batched(&q, &layer.q_norm, &q, m * n_heads, head_dim, eps)?;
        gpu.rmsnorm_batched(&k, &layer.k_norm, &k, m * n_kv, head_dim, eps)?;
        gpu.rope_batched_f32(
            &q,
            &k,
            &positions,
            n_heads,
            n_kv,
            head_dim,
            cfg.rope_base_for_layer(layer_idx),
            m,
        )?;

        if swa && !cfg.is_global_layer(layer_idx) {
            // ── Local layer: batched sliding-window attention over the ring ──
            let k_hm = k_hm.as_ref().unwrap();
            let v_hm = v_hm.as_ref().unwrap();
            let staged_k = staged_k.as_ref().unwrap();
            let staged_v = staged_v.as_ref().unwrap();
            let nvalid = nvalid.as_ref().unwrap();
            let ring_k = state.swa_k[layer_idx].as_ref().unwrap();
            let ring_v = state.swa_v[layer_idx].as_ref().unwrap();
            let hdw = head_dim * win;

            // Gather position-major k/v [m, n_kv*head_dim] → head-major
            // [n_kv, m, head_dim] (the stage/ring-write kernels are per-head).
            for kvh in 0..n_kv {
                gpu.strided_copy_2d(
                    &k,
                    kvh * head_dim,
                    kv_dim,
                    k_hm,
                    kvh * m * head_dim,
                    head_dim,
                    m,
                    head_dim,
                    false,
                )?;
                gpu.strided_copy_2d(
                    &v,
                    kvh * head_dim,
                    kv_dim,
                    v_hm,
                    kvh * m * head_dim,
                    head_dim,
                    m,
                    head_dim,
                    false,
                )?;
            }
            // Stage each kv head's visible window (pre-chunk ring + within-chunk KV).
            for kvh in 0..n_kv {
                gpu.swa_visibility_stage_batched(
                    &ring_k.sub_offset(kvh * hdw, hdw),
                    &k_hm.sub_offset(kvh * m * head_dim, m * head_dim),
                    &staged_k.sub_offset(kvh * m * hdw, m * hdw),
                    start_pos as i32,
                    win as i32,
                    head_dim as i32,
                    m as i32,
                )?;
                gpu.swa_visibility_stage_batched(
                    &ring_v.sub_offset(kvh * hdw, hdw),
                    &v_hm.sub_offset(kvh * m * head_dim, m * head_dim),
                    &staged_v.sub_offset(kvh * m * hdw, m * hdw),
                    start_pos as i32,
                    win as i32,
                    head_dim as i32,
                    m as i32,
                )?;
            }
            // gemma3 bakes query_pre_attn_scalar into q_norm ⇒ plain 1/√head_dim.
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            gpu.attention_swa_gqa_batched(
                &q, staged_k, staged_v, nvalid, &attn_out, n_heads, n_kv, head_dim, win, m, scale,
            )?;
            // Advance the rings with this chunk's KV (slot = pos % window).
            for kvh in 0..n_kv {
                gpu.swa_ring_write_batched_f32(
                    &k_hm.sub_offset(kvh * m * head_dim, m * head_dim),
                    &ring_k.sub_offset(kvh * hdw, hdw),
                    1,
                    head_dim as i32,
                    win as i32,
                    start_pos as i32,
                    m as i32,
                )?;
                gpu.swa_ring_write_batched_f32(
                    &v_hm.sub_offset(kvh * m * head_dim, m * head_dim),
                    &ring_v.sub_offset(kvh * hdw, hdw),
                    1,
                    head_dim as i32,
                    win as i32,
                    start_pos as i32,
                    m as i32,
                )?;
            }
        } else if state.kv_cache.quant_q8 {
            // ── Global layer (or SWA off), Q8 KV ──
            gpu.kv_cache_write_q8_0_batched(
                &state.kv_cache.k_gpu[layer_idx],
                &k,
                &positions,
                n_kv,
                head_dim,
                m,
            )?;
            gpu.kv_cache_write_q8_0_batched(
                &state.kv_cache.v_gpu[layer_idx],
                &v,
                &positions,
                n_kv,
                head_dim,
                m,
            )?;
            gpu.attention_q8_0_kv_batched(
                &q,
                &state.kv_cache.k_gpu[layer_idx],
                &state.kv_cache.v_gpu[layer_idx],
                &attn_out,
                &positions,
                n_heads,
                n_kv,
                head_dim,
                state.kv_cache.physical_cap,
                max_ctx,
                m,
            )?;
        } else {
            // ── Global layer (or SWA off), F32 KV ──
            gpu.kv_cache_write_f32_batched(
                &state.kv_cache.k_gpu[layer_idx],
                &k,
                &positions,
                kv_dim,
                m,
            )?;
            gpu.kv_cache_write_f32_batched(
                &state.kv_cache.v_gpu[layer_idx],
                &v,
                &positions,
                kv_dim,
                m,
            )?;
            gpu.attention_f32_batched(
                &q,
                &state.kv_cache.k_gpu[layer_idx],
                &state.kv_cache.v_gpu[layer_idx],
                &attn_out,
                &positions,
                n_heads,
                n_kv,
                head_dim,
                state.kv_cache.physical_cap,
                max_ctx,
                m,
            )?;
        }

        // o_proj + post_attention_layernorm (inside the residual) + add.
        prefill_linear(gpu, &layer.wo, &attn_out, &o, m)?;
        gpu.rmsnorm_batched(&o, &layer.post_attn_norm, &tmp, m, dim, eps)?;
        gpu.add_f32(x_batch, &tmp, x_batch)?;

        // FFN (GeGLU) + post_feedforward_layernorm (inside the residual) + add.
        gpu.rmsnorm_batched(x_batch, &layer.pre_ffn_norm, &tmp, m, dim, eps)?;
        prefill_linear(gpu, &layer.w_gate, &tmp, &gate, m)?;
        prefill_linear(gpu, &layer.w_up, &tmp, &up, m)?;
        gpu.gelu_mul_f32(&gate, &up, &ffn)?;
        prefill_linear(gpu, &layer.w_down, &ffn, &o, m)?;
        gpu.rmsnorm_batched(&o, &layer.post_ffn_norm, &tmp, m, dim, eps)?;
        gpu.add_f32(x_batch, &tmp, x_batch)?;

        // Extract-layer capture: scatter the settled residual [m, dim] into the
        // GPU sink [m, n_extract, dim] at extract index e_next (position-major).
        if let Some(hg) = hidden_gpu {
            if e_next < n_extract && extract_layers[e_next] == layer_idx {
                gpu.strided_copy_2d(
                    x_batch,
                    0,
                    dim,
                    hg,
                    e_next * dim,
                    n_extract * dim,
                    m,
                    dim,
                    false,
                )?;
                e_next += 1;
            }
        }
    }

    // Per-position final norm + lm_head → [m, vocab].
    let normed = gpu.alloc_owned(&[m * dim], DType::F32)?;
    gpu.rmsnorm_batched(x_batch, &weights.output_norm, &normed, m, dim, eps)?;
    let logits_all = gpu.alloc_owned(&[m * vocab], DType::F32)?;
    prefill_linear(gpu, &weights.output, &normed, &logits_all, m)?;
    let host = gpu.download_f32(&logits_all)?;

    gpu.reclaim_pending();
    state.next_pos = start_pos + m;
    Ok(host)
}

/// Greedy variant: run a step, then return argmax of the resulting logits.
pub fn forward_step_greedy(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    cfg: &Gemma3Config,
    state: &mut Gemma3State,
    token: u32,
) -> HipResult<u32> {
    forward_step(gpu, weights, cfg, state, token)?;
    gpu.argmax_f32(&state.logits, cfg.vocab_size)
}
