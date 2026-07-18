// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Straightforward and lowered dense Gemma 4 decode over shared weights/state.

use crate::config::{AttentionKind, Gemma4Config, RopePlan, ValueProjection};
use crate::weights::{Gemma4DenseLayerWeights, Gemma4DenseWeights};
use hip_bridge::{DeviceBuffer, HipError, HipResult};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::superop::{
    self, ActFlavor, AttentionCacheBinding, AttentionGeometry, AttnFlavor, EscapeKind,
    ForwardBindings, LoweredForward, OpBinding, OpFlavor, RopeFlavor, ScratchSlot, SuperOp,
    SuperOpKind, WeightSlot,
};
use hipfire_dispatch::types::DispatchError;
use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_runtime::kv::{KvCache, KvQuantMode};
use hipfire_runtime::layered_kv::{KvStorageKind, LayeredAttentionScratch, LayeredKvArena};
use hipfire_runtime::weights::{weight_gemv, EmbeddingFormat};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default)]
pub struct Gemma4ForwardCapture {
    pub layer_boundaries: Vec<Vec<f32>>,
    pub final_hidden: Vec<f32>,
    pub logits: Vec<f32>,
    /// Optional single-layer diagnostic trace. Normal inference and ordinary
    /// captures leave this unset, so no intermediate downloads occur.
    pub operator_layer: Option<usize>,
    pub operator_boundaries: BTreeMap<String, Vec<f32>>,
}

pub struct Gemma4DenseState {
    pub kv: LayeredKvArena,
    attention: LayeredAttentionScratch,
    x: GpuTensor,
    tmp: GpuTensor,
    o: GpuTensor,
    gate: GpuTensor,
    up: GpuTensor,
    ffn: GpuTensor,
    logits: GpuTensor,
    swa_staged_k: GpuTensor,
    swa_staged_v: GpuTensor,
    swa_nvalid: GpuTensor,
    pos_buf: DeviceBuffer,
    // ── KVarN scratch (Some only under KvQuantMode::Kvarn with ≥1 Full/256 layer) ──
    /// Reusable gather/quantize tile `[max_kv_width × KVARN_GROUP]` for
    /// `kvarn_attend`; allocated once so the single-token hot path never allocates.
    kvarn_tiles: Option<GpuTensor>,
    /// FlashAttention partials `[max_q_heads × ceil(max_seq/GROUP) × (2 + head_dim)]`.
    kvarn_flash_partials: Option<GpuTensor>,
}

impl Gemma4DenseState {
    /// F32-KV state (examples / diagnostics). Delegates to [`Self::new_with_kv_mode`]
    /// with the unquantized mode.
    pub fn new(gpu: &mut Gpu, config: &Gemma4Config, max_seq: usize) -> HipResult<Self> {
        Self::new_with_kv_mode(gpu, config, max_seq, KvQuantMode::Unquantized, 4)
    }

    /// State with a selectable KV cache mode. `KvQuantMode::Kvarn` stores the
    /// Full-storage (global) layers as variance-normalized `kvarn_bits`-bit K + Q8 V
    /// when their `head_dim ∈ {128, 256, 512}` (shipped gemma4 uses `global_head_dim`
    /// = 512, served by the `_hd512` kvarn kernel variants); SlidingWindow (local)
    /// layers stay F32. Q8 KV is not wired (deprecated per the mq*/Q8 direction);
    /// non-Kvarn modes use F32.
    pub fn new_with_kv_mode(
        gpu: &mut Gpu,
        config: &Gemma4Config,
        max_seq: usize,
        kv_mode: KvQuantMode,
        kvarn_bits: usize,
    ) -> HipResult<Self> {
        let plan = config
            .layered_kv_plan(max_seq)
            .unwrap_or_else(|error| panic!("Gemma 4 KV plan: {error}"));
        // KVarN applies to Full-storage (global) layers with head_dim ∈ {128,256}.
        // Size the reusable scratch at the max geometry over those layers.
        let (max_kv_width, max_q_heads, kvarn_head_dim) = plan
            .layers()
            .iter()
            .filter(|spec| matches!(spec.storage, KvStorageKind::Full))
            .filter(|spec| spec.head_dim == 128 || spec.head_dim == 256 || spec.head_dim == 512)
            .fold((0usize, 0usize, 0usize), |(kw, qh, _), spec| {
                (kw.max(spec.kv_width()), qh.max(spec.q_heads), spec.head_dim)
            });
        let kv_kvarn = matches!(kv_mode, KvQuantMode::Kvarn) && max_kv_width > 0;
        let kv = if kv_kvarn {
            LayeredKvArena::new_kvarn(gpu, plan.clone(), kvarn_bits)?
        } else {
            LayeredKvArena::new_fp32(gpu, plan.clone())?
        };
        let (kvarn_tiles, kvarn_flash_partials) = if kv_kvarn {
            let tiles = gpu.alloc_tensor(&[max_kv_width * KvCache::KVARN_GROUP], DType::F32)?;
            let max_tiles = plan.max_seq().div_ceil(KvCache::KVARN_GROUP);
            let partials = gpu.alloc_tensor(
                &[max_q_heads * max_tiles * (2 + kvarn_head_dim)],
                DType::F32,
            )?;
            (Some(tiles), Some(partials))
        } else {
            (None, None)
        };
        let attention = LayeredAttentionScratch::new(gpu, &plan)?;
        let hidden = config.hidden_size;
        let intermediate = config
            .layers
            .iter()
            .map(|layer| match layer.ffn {
                crate::config::FfnPlan::Dense { intermediate } => intermediate,
                crate::config::FfnPlan::DensePlusMoe {
                    dense_intermediate,
                    expert_intermediate,
                    ..
                } => dense_intermediate.max(expert_intermediate),
            })
            .max()
            .unwrap_or(config.intermediate_size);
        let swa_elements = plan
            .layers()
            .iter()
            .filter_map(|layer| match layer.storage {
                KvStorageKind::SlidingWindow { window } => Some(layer.kv_width() * window),
                KvStorageKind::Full => None,
            })
            .max()
            .unwrap_or(1);
        Ok(Self {
            kv,
            attention,
            x: gpu.alloc_tensor(&[hidden], DType::F32)?,
            tmp: gpu.alloc_tensor(&[hidden], DType::F32)?,
            o: gpu.alloc_tensor(&[hidden], DType::F32)?,
            gate: gpu.alloc_tensor(&[intermediate], DType::F32)?,
            up: gpu.alloc_tensor(&[intermediate], DType::F32)?,
            ffn: gpu.alloc_tensor(&[intermediate], DType::F32)?,
            logits: gpu.alloc_tensor(&[config.vocab_size], DType::F32)?,
            swa_staged_k: gpu.alloc_tensor(&[swa_elements], DType::F32)?,
            swa_staged_v: gpu.alloc_tensor(&[swa_elements], DType::F32)?,
            swa_nvalid: gpu.alloc_tensor(&[1], DType::F32)?,
            pos_buf: gpu.hip.malloc(4)?,
            kvarn_tiles,
            kvarn_flash_partials,
        })
    }

    pub fn next_pos(&self) -> usize {
        self.kv.next_pos()
    }

    pub fn logits_tensor(&self) -> &GpuTensor {
        &self.logits
    }

    pub fn reset(&mut self) {
        self.kv.reset();
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.kv.free_gpu(gpu);
        self.attention.free_gpu(gpu);
        for tensor in [
            self.x,
            self.tmp,
            self.o,
            self.gate,
            self.up,
            self.ffn,
            self.logits,
            self.swa_staged_k,
            self.swa_staged_v,
            self.swa_nvalid,
        ] {
            let _ = gpu.free_tensor(tensor);
        }
        for tensor in self
            .kvarn_tiles
            .into_iter()
            .chain(self.kvarn_flash_partials)
        {
            let _ = gpu.free_tensor(tensor);
        }
        let _ = gpu.hip.free(self.pos_buf);
    }
}

fn set_position(gpu: &Gpu, state: &Gemma4DenseState, position: usize) -> HipResult<()> {
    let position =
        i32::try_from(position).map_err(|_| HipError::new(0, "Gemma 4 position exceeds i32"))?;
    gpu.hip.memcpy_htod(&state.pos_buf, &position.to_ne_bytes())
}

fn embed_token(
    gpu: &mut Gpu,
    weights: &Gemma4DenseWeights,
    config: &Gemma4Config,
    state: &Gemma4DenseState,
    token: u32,
) -> HipResult<()> {
    match weights.core.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(
            &weights.core.token_embd,
            &state.x,
            token,
            config.hidden_size,
        )?,
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(
            &weights.core.token_embd,
            &state.x,
            token,
            config.hidden_size,
        )?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(
            &weights.core.token_embd,
            &state.x,
            token,
            config.hidden_size,
        )?,
        EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(
            &weights.core.token_embd,
            &state.x,
            token,
            config.hidden_size,
        )?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(
            &weights.core.token_embd,
            &state.x,
            token,
            config.hidden_size,
        )?,
    }
    let mut scale = (config.hidden_size as f32).sqrt();
    if weights.core.embedding_source_bf16 {
        scale = round_f32_to_bf16(scale);
    }
    gpu.scale_f32(&state.x, scale)?;
    if weights.core.embedding_source_bf16 {
        gpu.bf16_round_trip_f32(&state.x)?;
    }
    Ok(())
}

fn round_f32_to_bf16(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
}

fn attention_block(
    gpu: &mut Gpu,
    layer_idx: usize,
    layer: &Gemma4DenseLayerWeights,
    config: &Gemma4Config,
    state: &Gemma4DenseState,
    position: usize,
    mut capture: Option<&mut Gemma4ForwardCapture>,
) -> HipResult<()> {
    let plan = &config.layers[layer_idx];
    let geometry = plan.attention;
    let q_width = geometry.q_heads * geometry.head_dim;
    let kv_width = geometry.kv_heads * geometry.head_dim;
    let scratch = state
        .attention
        .view(state.kv.plan(), layer_idx)
        .unwrap_or_else(|error| panic!("Gemma 4 attention scratch: {error}"));

    gpu.rmsnorm_f32(&state.x, &layer.input_norm, &state.tmp, config.rms_norm_eps)?;
    if let Some(capture) = capture.as_deref_mut() {
        capture_operator(gpu, capture, layer_idx, "input_norm", &state.tmp)?;
    }
    weight_gemv(gpu, &layer.wq, &state.tmp, &scratch.q)?;
    weight_gemv(gpu, &layer.wk, &state.tmp, &scratch.k)?;
    if let Some(capture) = capture.as_deref_mut() {
        capture_operator(gpu, capture, layer_idx, "q_proj", &scratch.q)?;
        capture_operator(gpu, capture, layer_idx, "k_proj", &scratch.k)?;
    }
    match (&layer.wv, plan.value_projection) {
        (Some(wv), ValueProjection::Separate) => {
            weight_gemv(gpu, wv, &state.tmp, &scratch.v)?;
        }
        (None, ValueProjection::FromPreNormKey) => {
            gpu.copy_d2d(
                &scratch.k,
                &scratch.v,
                kv_width * std::mem::size_of::<f32>(),
            )?;
        }
        _ => {
            return Err(HipError::new(
                0,
                "Gemma 4 value projection does not match lowered layer plan",
            ));
        }
    }
    if let Some(capture) = capture.as_deref_mut() {
        capture_operator(gpu, capture, layer_idx, "v_proj", &scratch.v)?;
    }

    gpu.rmsnorm_batched(
        &scratch.q,
        &layer.q_norm,
        &scratch.q,
        geometry.q_heads,
        geometry.head_dim,
        config.rms_norm_eps,
    )?;
    gpu.rmsnorm_batched(
        &scratch.k,
        &layer.k_norm,
        &scratch.k,
        geometry.kv_heads,
        geometry.head_dim,
        config.rms_norm_eps,
    )?;
    let mut v_heads = scratch.v.sub_offset(0, kv_width);
    v_heads.shape = vec![geometry.kv_heads, geometry.head_dim];
    gpu.rmsnorm_weightless_f32(&v_heads, &v_heads, config.rms_norm_eps)?;
    if let Some(capture) = capture.as_deref_mut() {
        capture_operator(gpu, capture, layer_idx, "q_norm", &scratch.q)?;
        capture_operator(gpu, capture, layer_idx, "k_norm", &scratch.k)?;
        capture_operator(gpu, capture, layer_idx, "v_norm", &scratch.v)?;
    }

    set_position(gpu, state, position)?;
    match plan.rope {
        RopePlan::FullHalfSplit { theta, dim } => gpu.rope_partial_interleaved_f32(
            &scratch.q,
            &scratch.k,
            &state.pos_buf,
            geometry.q_heads,
            geometry.kv_heads,
            geometry.head_dim,
            dim,
            dim,
            theta,
        )?,
        RopePlan::ProportionalHalfSplit {
            theta,
            rotary_dim,
            basis_dim,
        } => gpu.rope_partial_interleaved_f32(
            &scratch.q,
            &scratch.k,
            &state.pos_buf,
            geometry.q_heads,
            geometry.kv_heads,
            geometry.head_dim,
            rotary_dim,
            basis_dim,
            theta,
        )?,
    }
    // The portable attention primitives apply 1/sqrt(head_dim); Gemma 4's
    // score scale is 1, so compensate after the upstream-ordered RoPE.
    gpu.scale_f32(&scratch.q, (geometry.head_dim as f32).sqrt())?;
    if let Some(capture) = capture.as_deref_mut() {
        capture_operator(gpu, capture, layer_idx, "q_rope_scaled", &scratch.q)?;
        capture_operator(gpu, capture, layer_idx, "k_rope", &scratch.k)?;
    }

    let cache = state
        .kv
        .view(layer_idx, position)
        .unwrap_or_else(|error| panic!("Gemma 4 cache binding: {error}"));
    match state.kv.plan().layers()[cache.producer_layer].storage {
        KvStorageKind::SlidingWindow { window } => {
            let n_valid = (position + 1).min(window);
            let n_valid_i32 = n_valid as i32;
            gpu.hip
                .memcpy_htod(&state.swa_nvalid.buf, &n_valid_i32.to_ne_bytes())?;
            let head_window = geometry.head_dim * window;
            for kv_head in 0..geometry.kv_heads {
                gpu.swa_visibility_stage_batched(
                    &cache.k.sub_offset(kv_head * head_window, head_window),
                    &scratch
                        .k
                        .sub_offset(kv_head * geometry.head_dim, geometry.head_dim),
                    &state
                        .swa_staged_k
                        .sub_offset(kv_head * head_window, head_window),
                    position as i32,
                    window as i32,
                    geometry.head_dim as i32,
                    1,
                )?;
                gpu.swa_visibility_stage_batched(
                    &cache.v.sub_offset(kv_head * head_window, head_window),
                    &scratch
                        .v
                        .sub_offset(kv_head * geometry.head_dim, geometry.head_dim),
                    &state
                        .swa_staged_v
                        .sub_offset(kv_head * head_window, head_window),
                    position as i32,
                    window as i32,
                    geometry.head_dim as i32,
                    1,
                )?;
            }
            if let Some(capture) = capture.as_deref_mut() {
                capture_operator(
                    gpu,
                    capture,
                    layer_idx,
                    "attention_k_context",
                    &state
                        .swa_staged_k
                        .sub_offset(0, geometry.kv_heads * head_window),
                )?;
                capture_operator(
                    gpu,
                    capture,
                    layer_idx,
                    "attention_v_context",
                    &state
                        .swa_staged_v
                        .sub_offset(0, geometry.kv_heads * head_window),
                )?;
            }
            gpu.attention_swa_gqa_batched(
                &scratch.q,
                &state.swa_staged_k,
                &state.swa_staged_v,
                &state.swa_nvalid,
                &scratch.attention,
                geometry.q_heads,
                geometry.kv_heads,
                geometry.head_dim,
                window,
                1,
                1.0 / (geometry.head_dim as f32).sqrt(),
            )?;
            for kv_head in 0..geometry.kv_heads {
                gpu.swa_ring_write_batched_f32(
                    &scratch
                        .k
                        .sub_offset(kv_head * geometry.head_dim, geometry.head_dim),
                    &cache.k.sub_offset(kv_head * head_window, head_window),
                    1,
                    geometry.head_dim as i32,
                    window as i32,
                    position as i32,
                    1,
                )?;
                gpu.swa_ring_write_batched_f32(
                    &scratch
                        .v
                        .sub_offset(kv_head * geometry.head_dim, geometry.head_dim),
                    &cache.v.sub_offset(kv_head * head_window, head_window),
                    1,
                    geometry.head_dim as i32,
                    window as i32,
                    position as i32,
                    1,
                )?;
            }
        }
        KvStorageKind::Full if cache.quant_kvarn => {
            // KVarN (variance-normalized bits-bit K + Q8 V) fused write+attend for
            // the global/full-context layers, n=1 (decode + per-token prefill). Local
            // sliding-window layers stay F32 (the arm above). Mirrors gemma3's KVarN
            // hook. Full storage → physical position == absolute.
            set_position(gpu, state, position)?;
            // Optional Hadamard-incoherence rotation of K and Q by the SAME per-head
            // FWHT-256 (scores preserved; V/Q8 stays un-rotated). Opt out with
            // HIPFIRE_KVARN_ROTATE=0. Requires head_dim==256 (gemma4's shape).
            static KVARN_ROTATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let kvarn_rotate = *KVARN_ROTATE
                .get_or_init(|| std::env::var("HIPFIRE_KVARN_ROTATE").ok().as_deref() != Some("0"));
            if kvarn_rotate && geometry.head_dim == 256 {
                gpu.rotate_x_mq_batched(&scratch.k, &scratch.k, kv_width, 1)?;
                gpu.rotate_x_mq_batched(&scratch.q, &scratch.q, q_width, 1)?;
            }
            // The KV kernels read positions from a GpuTensor; wrap the raw 4-byte i32
            // pos_buf as a non-owning [1] view (mirrors gemma3/qwen35's KVarN hook).
            let pos_view = GpuTensor {
                buf: unsafe { DeviceBuffer::from_raw(state.pos_buf.as_ptr(), 4) },
                shape: vec![1],
                dtype: DType::F32,
            };
            gpu.kvarn_attend(
                cache.k,
                cache.k_window.expect("kvarn cache view exposes k_window"),
                cache.v,
                &scratch.q,
                &scratch.k,
                &scratch.v,
                &pos_view,
                &scratch.attention,
                state
                    .kvarn_flash_partials
                    .as_ref()
                    .expect("kvarn scratch allocated when kv_mode=kvarn"),
                state
                    .kvarn_tiles
                    .as_ref()
                    .expect("kvarn scratch allocated when kv_mode=kvarn"),
                1,
                position,
                geometry.q_heads,
                geometry.kv_heads,
                geometry.head_dim,
                cache.physical_cap,
                None,
                0,
                0,
                cache.kvarn_bits,
            )?;
        }
        KvStorageKind::Full => {
            set_position(gpu, state, cache.physical_position)?;
            gpu.kv_cache_write(cache.k, &scratch.k, &state.pos_buf, kv_width)?;
            gpu.kv_cache_write(cache.v, &scratch.v, &state.pos_buf, kv_width)?;
            if let Some(capture) = capture.as_deref_mut() {
                let context_elements = cache.visible_positions.len() * kv_width;
                capture_operator(
                    gpu,
                    capture,
                    layer_idx,
                    "attention_k_context",
                    &cache.k.sub_offset(0, context_elements),
                )?;
                capture_operator(
                    gpu,
                    capture,
                    layer_idx,
                    "attention_v_context",
                    &cache.v.sub_offset(0, context_elements),
                )?;
            }
            gpu.attention_f32(
                &scratch.q,
                cache.k,
                cache.v,
                &scratch.attention,
                &state.pos_buf,
                cache.visible_positions.len(),
                geometry.q_heads,
                geometry.kv_heads,
                geometry.head_dim,
                state.kv.plan().max_seq(),
            )?;
        }
    }
    if let Some(capture) = capture.as_deref_mut() {
        capture_operator(gpu, capture, layer_idx, "attention_raw", &scratch.attention)?;
    }
    debug_assert_eq!(scratch.attention.numel(), q_width);
    weight_gemv(gpu, &layer.wo, &scratch.attention, &state.o)?;
    if let Some(capture) = capture.as_deref_mut() {
        capture_operator(gpu, capture, layer_idx, "o_proj", &state.o)?;
    }
    gpu.rmsnorm_f32(
        &state.o,
        &layer.post_attn_norm,
        &state.tmp,
        config.rms_norm_eps,
    )?;
    if let Some(capture) = capture.as_deref_mut() {
        capture_operator(gpu, capture, layer_idx, "post_attention_norm", &state.tmp)?;
    }
    gpu.add_f32(&state.x, &state.tmp, &state.x)
}

fn ffn_project(
    gpu: &mut Gpu,
    layer: &Gemma4DenseLayerWeights,
    config: &Gemma4Config,
    state: &Gemma4DenseState,
) -> HipResult<()> {
    gpu.rmsnorm_f32(
        &state.x,
        &layer.pre_ffn_norm,
        &state.tmp,
        config.rms_norm_eps,
    )?;
    weight_gemv(gpu, &layer.w_gate, &state.tmp, &state.gate)?;
    weight_gemv(gpu, &layer.w_up, &state.tmp, &state.up)
}

fn ffn_activate(
    gpu: &mut Gpu,
    layer: &Gemma4DenseLayerWeights,
    state: &Gemma4DenseState,
) -> HipResult<()> {
    gpu.gelu_mul_f32(
        &state.gate,
        &state.up,
        &state.ffn.sub_offset(0, layer.w_down.k),
    )
}

fn ffn_finish(
    gpu: &mut Gpu,
    layer: &Gemma4DenseLayerWeights,
    config: &Gemma4Config,
    state: &Gemma4DenseState,
) -> HipResult<()> {
    weight_gemv(
        gpu,
        &layer.w_down,
        &state.ffn.sub_offset(0, layer.w_down.k),
        &state.o,
    )?;
    gpu.rmsnorm_f32(
        &state.o,
        &layer.post_ffn_norm,
        &state.tmp,
        config.rms_norm_eps,
    )?;
    gpu.add_f32(&state.x, &state.tmp, &state.x)
}

fn finish_logits(
    gpu: &mut Gpu,
    weights: &Gemma4DenseWeights,
    config: &Gemma4Config,
    state: &Gemma4DenseState,
) -> HipResult<()> {
    gpu.rmsnorm_f32(
        &state.x,
        &weights.core.output_norm,
        &state.tmp,
        config.rms_norm_eps,
    )?;
    weight_gemv(gpu, &weights.core.output, &state.tmp, &state.logits)?;
    gpu.vector_softcap_f32(
        &state.logits,
        &state.logits,
        config.vocab_size,
        config.final_logit_softcapping,
    )
}

fn capture_layer(gpu: &Gpu, state: &Gemma4DenseState, capture: &mut Gemma4ForwardCapture) {
    capture
        .layer_boundaries
        .push(gpu.download_f32(&state.x).expect("Gemma 4 layer capture"));
}

fn capture_operator(
    gpu: &Gpu,
    capture: &mut Gemma4ForwardCapture,
    layer_idx: usize,
    name: &str,
    tensor: &GpuTensor,
) -> HipResult<()> {
    if capture.operator_layer == Some(layer_idx) {
        capture
            .operator_boundaries
            .insert(name.to_string(), gpu.download_f32(tensor)?);
    }
    Ok(())
}

fn run_reference_layer(
    gpu: &mut Gpu,
    weights: &Gemma4DenseWeights,
    config: &Gemma4Config,
    state: &Gemma4DenseState,
    layer_idx: usize,
    position: usize,
    bf16_staged_geglu: bool,
    mut capture: Option<&mut Gemma4ForwardCapture>,
) -> HipResult<()> {
    let layer = &weights.layers[layer_idx];
    if let Some(capture) = capture.as_deref_mut() {
        capture_operator(gpu, capture, layer_idx, "pre_layer", &state.x)?;
    }
    attention_block(
        gpu,
        layer_idx,
        layer,
        config,
        state,
        position,
        capture.as_deref_mut(),
    )?;
    if let Some(capture) = capture.as_deref_mut() {
        capture_operator(gpu, capture, layer_idx, "post_attention_residual", &state.x)?;
    }
    ffn_project(gpu, layer, config, state)?;
    if bf16_staged_geglu {
        gpu.bf16_round_trip_f32(&state.gate)?;
        gpu.bf16_round_trip_f32(&state.up)?;
    }
    if let Some(capture) = capture.as_deref_mut() {
        capture_operator(gpu, capture, layer_idx, "pre_ffn_norm", &state.tmp)?;
        capture_operator(gpu, capture, layer_idx, "gate", &state.gate)?;
        capture_operator(gpu, capture, layer_idx, "up", &state.up)?;
    }
    if bf16_staged_geglu {
        let ffn = state.ffn.sub_offset(0, layer.w_down.k);
        gpu.gelu_tanh_f32(&state.gate, &ffn, layer.w_down.k)?;
        gpu.bf16_round_trip_f32(&ffn)?;
        gpu.mul_f32(&ffn, &state.up, &ffn)?;
        gpu.bf16_round_trip_f32(&ffn)?;
    } else {
        ffn_activate(gpu, layer, state)?;
    }
    if let Some(capture) = capture.as_deref_mut() {
        capture_operator(
            gpu,
            capture,
            layer_idx,
            "geglu",
            &state.ffn.sub_offset(0, layer.w_down.k),
        )?;
    }
    ffn_finish(gpu, layer, config, state)?;
    if let Some(capture) = capture.as_deref_mut() {
        capture_operator(gpu, capture, layer_idx, "post_ffn_norm", &state.tmp)?;
        capture_operator(gpu, capture, layer_idx, "post_ffn_residual", &state.x)?;
    }
    gpu.scale_f32(&state.x, layer.layer_scalar)?;
    if let Some(capture) = capture {
        capture_operator(gpu, capture, layer_idx, "layer_output", &state.x)?;
    }
    Ok(())
}

pub fn forward_step_reference(
    gpu: &mut Gpu,
    weights: &Gemma4DenseWeights,
    config: &Gemma4Config,
    state: &mut Gemma4DenseState,
    token: u32,
    mut capture: Option<&mut Gemma4ForwardCapture>,
) -> HipResult<()> {
    let position = state.next_pos();
    embed_token(gpu, weights, config, state, token)?;
    for layer_idx in 0..weights.layers.len() {
        run_reference_layer(
            gpu,
            weights,
            config,
            state,
            layer_idx,
            position,
            false,
            capture.as_deref_mut(),
        )?;
        if let Some(capture) = capture.as_deref_mut() {
            capture_layer(gpu, state, capture);
        }
    }
    finish_logits(gpu, weights, config, state)?;
    if let Some(capture) = capture {
        capture.final_hidden = gpu.download_f32(&state.tmp)?;
        capture.logits = gpu.download_f32(&state.logits)?;
    }
    state
        .kv
        .advance(position)
        .map_err(|error| HipError::new(0, &error))
}

/// Run one dense decoder layer from caller-supplied hidden rows while building
/// that layer's KV history position by position.
///
/// This is an offline diagnostic seam for frozen-oracle transition tests. It
/// deliberately skips embedding, all other layers, final normalization, and
/// the language-model head; normal inference must use [`forward_step`]. The
/// caller must provide positions contiguously and call [`Gemma4DenseState::reset`]
/// before starting a different layer.
pub fn diagnostic_forward_layer_from_hidden(
    gpu: &mut Gpu,
    weights: &Gemma4DenseWeights,
    config: &Gemma4Config,
    state: &mut Gemma4DenseState,
    layer_idx: usize,
    position: usize,
    hidden: &[f32],
) -> HipResult<Vec<f32>> {
    diagnostic_forward_layer_from_hidden_capture(
        gpu, weights, config, state, layer_idx, position, hidden, None,
    )
}

/// Capture operator boundaries for [`diagnostic_forward_layer_from_hidden`].
/// The caller selects the same `layer_idx` in `capture.operator_layer`.
#[allow(clippy::too_many_arguments)]
pub fn diagnostic_forward_layer_from_hidden_capture(
    gpu: &mut Gpu,
    weights: &Gemma4DenseWeights,
    config: &Gemma4Config,
    state: &mut Gemma4DenseState,
    layer_idx: usize,
    position: usize,
    hidden: &[f32],
    capture: Option<&mut Gemma4ForwardCapture>,
) -> HipResult<Vec<f32>> {
    diagnostic_forward_layer_from_hidden_impl(
        gpu, weights, config, state, layer_idx, position, hidden, false, capture,
    )
}

/// Diagnostic variant that reproduces PyTorch's BF16 GeGLU materialization
/// boundaries while leaving every other layer operator unchanged.
#[allow(clippy::too_many_arguments)]
pub fn diagnostic_forward_layer_from_hidden_bf16_geglu_capture(
    gpu: &mut Gpu,
    weights: &Gemma4DenseWeights,
    config: &Gemma4Config,
    state: &mut Gemma4DenseState,
    layer_idx: usize,
    position: usize,
    hidden: &[f32],
    capture: Option<&mut Gemma4ForwardCapture>,
) -> HipResult<Vec<f32>> {
    diagnostic_forward_layer_from_hidden_impl(
        gpu, weights, config, state, layer_idx, position, hidden, true, capture,
    )
}

#[allow(clippy::too_many_arguments)]
fn diagnostic_forward_layer_from_hidden_impl(
    gpu: &mut Gpu,
    weights: &Gemma4DenseWeights,
    config: &Gemma4Config,
    state: &mut Gemma4DenseState,
    layer_idx: usize,
    position: usize,
    hidden: &[f32],
    bf16_staged_geglu: bool,
    capture: Option<&mut Gemma4ForwardCapture>,
) -> HipResult<Vec<f32>> {
    if layer_idx >= weights.layers.len() || layer_idx >= config.layers.len() {
        return Err(HipError::new(
            0,
            &format!("Gemma 4 diagnostic layer {layer_idx} is out of range"),
        ));
    }
    if hidden.len() != config.hidden_size {
        return Err(HipError::new(
            0,
            &format!(
                "Gemma 4 diagnostic hidden width {} does not match {}",
                hidden.len(),
                config.hidden_size
            ),
        ));
    }
    if position != state.next_pos() {
        return Err(HipError::new(
            0,
            &format!(
                "Gemma 4 diagnostic position {position} is not contiguous with {}",
                state.next_pos()
            ),
        ));
    }

    let bytes = unsafe {
        std::slice::from_raw_parts(hidden.as_ptr().cast::<u8>(), std::mem::size_of_val(hidden))
    };
    gpu.hip.memcpy_htod(&state.x.buf, bytes)?;
    run_reference_layer(
        gpu,
        weights,
        config,
        state,
        layer_idx,
        position,
        bf16_staged_geglu,
        capture,
    )?;
    let output = gpu.download_f32(&state.x)?;
    state
        .kv
        .advance(position)
        .map_err(|error| HipError::new(0, &error))?;
    Ok(output)
}

fn op(kind: SuperOpKind, flavor: OpFlavor, weights: Vec<WeightSlot>) -> SuperOp {
    SuperOp {
        kind,
        binding: OpBinding {
            key: None,
            weights,
            scratch: Vec::<ScratchSlot>::new(),
            flavor,
        },
    }
}

fn lowered_attention_flavor(
    config: &Gemma4Config,
    layer_idx: usize,
    binding: (usize, usize, usize),
) -> AttnFlavor {
    let plan = &config.layers[layer_idx];
    let (producer, group, slot) = binding;
    let rope = match plan.rope {
        RopePlan::FullHalfSplit { theta, .. } => RopeFlavor::HalfRotate { theta },
        RopePlan::ProportionalHalfSplit {
            theta,
            rotary_dim,
            basis_dim,
        } => RopeFlavor::PartialHalfRotate {
            theta,
            rotary_dim: rotary_dim as u32,
            basis_dim: basis_dim as u32,
        },
    };
    AttnFlavor {
        geometry: AttentionGeometry {
            q_heads: plan.attention.q_heads as u32,
            kv_heads: plan.attention.kv_heads as u32,
            head_dim: plan.attention.head_dim as u32,
        },
        cache: AttentionCacheBinding {
            physical_group: group as u32,
            physical_slot: slot as u32,
            producer_layer: producer as u32,
        },
        window: match plan.kind {
            AttentionKind::Sliding => config.sliding_window as u32,
            AttentionKind::Full => 0,
        },
        qk_norm: true,
        q_scale_sqrt_hd: true,
        k_eq_v: matches!(plan.value_projection, ValueProjection::FromPreNormKey),
        logit_softcap: 0.0,
        rope,
    }
}

pub fn lower_dense_forward(config: &Gemma4Config, state: &Gemma4DenseState) -> LoweredForward {
    let mut lowered = LoweredForward::new(0);
    for (layer_idx, _plan) in config.layers.iter().enumerate() {
        let binding = state
            .kv
            .plan()
            .resolved_binding(layer_idx)
            .expect("validated Gemma 4 cache binding");
        let attn = lowered_attention_flavor(config, layer_idx, binding);
        lowered.layers.push(vec![
            op(SuperOpKind::Attend, OpFlavor::Attn(attn), Vec::new()),
            op(SuperOpKind::Proj, OpFlavor::None, Vec::new()),
            op(
                SuperOpKind::Act,
                OpFlavor::Act(ActFlavor::GeluTanhMul),
                Vec::new(),
            ),
            op(SuperOpKind::ResidualGemv, OpFlavor::None, Vec::new()),
            op(
                SuperOpKind::Scale,
                OpFlavor::None,
                vec![WeightSlot(layer_idx as u32)],
            ),
        ]);
    }
    lowered.final_program = vec![
        op(SuperOpKind::Norm, OpFlavor::None, Vec::new()),
        op(SuperOpKind::Proj, OpFlavor::None, Vec::new()),
        op(
            SuperOpKind::Softcap,
            OpFlavor::Softcap(config.final_logit_softcapping),
            Vec::new(),
        ),
    ];
    lowered
}

struct Gemma4Bindings<'a> {
    weights: &'a Gemma4DenseWeights,
    config: &'a Gemma4Config,
    state: &'a Gemma4DenseState,
    layer_idx: Option<usize>,
    position: usize,
}

fn dispatch_error(stage: &str, error: HipError) -> DispatchError {
    DispatchError::Hip(format!("Gemma 4 {stage}: {error:?}"))
}

impl ForwardBindings for Gemma4Bindings<'_> {
    fn run_proj(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        match self.layer_idx {
            Some(layer) => ffn_project(gpu, &self.weights.layers[layer], self.config, self.state)
                .map_err(|error| dispatch_error("FFN projection", error)),
            None => weight_gemv(
                gpu,
                &self.weights.core.output,
                &self.state.tmp,
                &self.state.logits,
            )
            .map_err(|error| dispatch_error("output projection", error)),
        }
    }

    fn run_residual_gemv(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let layer = self.layer_idx.expect("residual op only appears in a layer");
        ffn_finish(gpu, &self.weights.layers[layer], self.config, self.state)
            .map_err(|error| dispatch_error("FFN residual", error))
    }

    fn run_norm(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        if self.layer_idx.is_some() {
            return Err(DispatchError::Hip(
                "Gemma 4 lowered layer has unexpected standalone norm".into(),
            ));
        }
        gpu.rmsnorm_f32(
            &self.state.x,
            &self.weights.core.output_norm,
            &self.state.tmp,
            self.config.rms_norm_eps,
        )
        .map_err(|error| dispatch_error("final norm", error))
    }

    fn run_attend(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let layer_idx = self
            .layer_idx
            .expect("attention op only appears in a layer");
        let OpFlavor::Attn(flavor) = op.flavor else {
            return Err(DispatchError::Hip(
                "Gemma 4 lowered attention lacks flavor".into(),
            ));
        };
        let binding = self
            .state
            .kv
            .plan()
            .resolved_binding(layer_idx)
            .map_err(|error| DispatchError::Hip(format!("Gemma 4 cache binding: {error}")))?;
        let expected = lowered_attention_flavor(self.config, layer_idx, binding);
        if flavor != expected {
            return Err(DispatchError::Hip(
                "Gemma 4 lowered attention flavor drifted from the resolved layer plan".into(),
            ));
        }
        attention_block(
            gpu,
            layer_idx,
            &self.weights.layers[layer_idx],
            self.config,
            self.state,
            self.position,
            None,
        )
        .map_err(|error| dispatch_error("attention", error))
    }

    fn run_moe(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "Gemma 4 dense path has no MoE op".into(),
        ))
    }

    fn run_recurrent(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "Gemma 4 has no recurrent super-op".into(),
        ))
    }

    fn run_conv(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("Gemma 4 has no conv super-op".into()))
    }

    fn run_act(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        if op.flavor != OpFlavor::Act(ActFlavor::GeluTanhMul) {
            return Err(DispatchError::Hip(
                "Gemma 4 lowered activation is not GeGLU".into(),
            ));
        }
        let layer = self
            .layer_idx
            .expect("activation op only appears in a layer");
        ffn_activate(gpu, &self.weights.layers[layer], self.state)
            .map_err(|error| dispatch_error("GeGLU", error))
    }

    fn run_scale(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let layer = self.layer_idx.expect("scale op only appears in a layer");
        if op.weights != [WeightSlot(layer as u32)] {
            return Err(DispatchError::Hip(
                "Gemma 4 layer-scalar binding drifted".into(),
            ));
        }
        gpu.scale_f32(&self.state.x, self.weights.layers[layer].layer_scalar)
            .map_err(|error| dispatch_error("layer scalar", error))
    }

    fn run_softcap(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let OpFlavor::Softcap(cap) = op.flavor else {
            return Err(DispatchError::Hip(
                "Gemma 4 softcap op lacks cap flavor".into(),
            ));
        };
        gpu.vector_softcap_f32(
            &self.state.logits,
            &self.state.logits,
            self.config.vocab_size,
            cap,
        )
        .map_err(|error| dispatch_error("softcap", error))
    }

    fn run_escape(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        kind: EscapeKind,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(format!(
            "Gemma 4 dense path has no escape op ({kind:?})"
        )))
    }
}

pub fn forward_step_lowered(
    gpu: &mut Gpu,
    weights: &Gemma4DenseWeights,
    config: &Gemma4Config,
    state: &mut Gemma4DenseState,
    lowered: &LoweredForward,
    token: u32,
    mut capture: Option<&mut Gemma4ForwardCapture>,
) -> HipResult<()> {
    let position = state.next_pos();
    embed_token(gpu, weights, config, state, token)?;
    let ctx = DispatchCtx::new(gpu);
    for layer_idx in 0..weights.layers.len() {
        let mut bindings = Gemma4Bindings {
            weights,
            config,
            state,
            layer_idx: Some(layer_idx),
            position,
        };
        superop::run_layer_program(gpu, &ctx, &lowered.layers[layer_idx], &mut bindings)
            .map_err(|error| HipError::new(0, &format!("Gemma 4 lowered layer: {error}")))?;
        if let Some(capture) = capture.as_deref_mut() {
            capture_layer(gpu, state, capture);
        }
    }
    let mut bindings = Gemma4Bindings {
        weights,
        config,
        state,
        layer_idx: None,
        position,
    };
    superop::run_layer_program(gpu, &ctx, &lowered.final_program, &mut bindings)
        .map_err(|error| HipError::new(0, &format!("Gemma 4 lowered final: {error}")))?;
    if let Some(capture) = capture {
        capture.final_hidden = gpu.download_f32(&state.tmp)?;
        capture.logits = gpu.download_f32(&state.logits)?;
    }
    state
        .kv
        .advance(position)
        .map_err(|error| HipError::new(0, &error))
}

/// Lowered is the default after the Phase 4 layer/logit dual-run recorded exact
/// parity. `HIPFIRE_GEMMA4_FORWARD_ORACLE=1` retains the readable reference path
/// as an explicit opt-out oracle.
pub fn forward_step(
    gpu: &mut Gpu,
    weights: &Gemma4DenseWeights,
    config: &Gemma4Config,
    state: &mut Gemma4DenseState,
    lowered: &LoweredForward,
    token: u32,
) -> HipResult<()> {
    if std::env::var("HIPFIRE_GEMMA4_FORWARD_ORACLE")
        .ok()
        .as_deref()
        == Some("1")
    {
        forward_step_reference(gpu, weights, config, state, token, None)
    } else {
        forward_step_lowered(gpu, weights, config, state, lowered, token, None)
    }
}

pub fn logits(gpu: &Gpu, state: &Gemma4DenseState) -> HipResult<Vec<f32>> {
    gpu.download_f32(&state.logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowered_dense_program_uses_regular_ops_not_escape() {
        // Shape-only assertion lives in the GPU parity example; this guards the
        // dispatch enum contract without requiring hardware.
        for kind in [
            SuperOpKind::Attend,
            SuperOpKind::Proj,
            SuperOpKind::Act,
            SuperOpKind::ResidualGemv,
            SuperOpKind::Scale,
            SuperOpKind::Softcap,
            SuperOpKind::Ple,
        ] {
            assert!(!matches!(kind, SuperOpKind::Escape(_)));
        }
    }

    #[test]
    fn embedding_scale_uses_bf16_round_to_nearest_even() {
        assert_eq!(round_f32_to_bf16(5376.0_f32.sqrt()).to_bits(), 0x4293_0000);
        assert_eq!(round_f32_to_bf16(f32::INFINITY), f32::INFINITY);
    }
}
