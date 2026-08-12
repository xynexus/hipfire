// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Straightforward and lowered dense Gemma 4 decode over shared weights/state.

use crate::config::{AttentionKind, Gemma4Config, KvProducer, RopePlan, ValueProjection};
use crate::weights::{
    Gemma4DenseLayerWeights, Gemma4DenseWeights, Gemma4MoeLayerWeights, Gemma4PleWeights,
};
use hip_bridge::{DeviceBuffer, HipError, HipResult};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::superop::{
    self, ActFlavor, AttentionCacheBinding, AttentionGeometry, AttnFlavor, EscapeKind,
    ForwardBindings, LoweredForward, OpBinding, OpFlavor, RopeFlavor, ScratchSlot, SuperOp,
    SuperOpKind, WeightSlot,
};
use hipfire_dispatch::types::DispatchError;
use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_runtime::calibration::contracts::{
    CaptureAdmission, CaptureId, CaptureRegistry, ExpertCaptureRole, ExpertTelemetry,
    ProjectionRole, RoutedRowContext,
};
use hipfire_runtime::calibration::CalibCollector;
use hipfire_runtime::kv::{KvCache, KvQuantMode};
use hipfire_runtime::layered_kv::{KvStorageKind, LayeredAttentionScratch, LayeredKvArena};
use hipfire_runtime::triattn::{EvictionResult, LayeredEvictionCtx};
use hipfire_runtime::weights::{weight_gemv, EmbeddingFormat, WeightTensor};
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

/// Borrowed capture state for one streamed calibration layer. This is kept at
/// the production layer boundary so calibration cannot drift from Gemma 4's
/// attention, dense-plus-MoE, or layer-scalar math.
pub struct Gemma4CalibrationCapture<'a> {
    pub collector: &'a CalibCollector,
    pub registry: &'a CaptureRegistry,
    pub telemetry: Option<&'a mut ExpertTelemetry>,
    pub logical_layer: usize,
}

fn calibration_capture(
    gpu: &mut Gpu,
    calibration: &Gemma4CalibrationCapture<'_>,
    role: ProjectionRole,
    expert: Option<usize>,
    input: &GpuTensor,
    width: usize,
) -> HipResult<()> {
    calibration
        .collector
        .capture_by_id(
            gpu,
            calibration.registry,
            CaptureId::new(calibration.logical_layer, role, expert),
            input,
            1,
            width,
        )
        .map_err(|error| HipError::new(0, &error.to_string()))
}

pub struct Gemma4DenseState {
    pub kv: LayeredKvArena,
    attention: LayeredAttentionScratch,
    x: GpuTensor,
    tmp: GpuTensor,
    o: GpuTensor,
    ffn_norm: GpuTensor,
    router_logits: Option<GpuTensor>,
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
    // ── PLE scratch (Some only for E2B/E4B — hidden_size_per_layer_input != 0) ──
    /// Per-token per-layer inputs `[num_layers × ple_dim]`, precomputed at embed.
    per_layer_inputs: Option<GpuTensor>,
    /// Scratch `[num_layers × ple_dim]` for the model-projection branch at embed.
    ple_plmp: Option<GpuTensor>,
    /// Per-layer merge gate scratch `[ple_dim]`.
    ple_gate: Option<GpuTensor>,
    /// Per-layer merge projection scratch `[hidden]`.
    ple_proj: Option<GpuTensor>,
    // ── KV-sharing (E2B/E4B): producer layers save their post-RoPE K/V here so
    // later same-type shared layers reuse it. `Some` per producer (Own) layer when
    // the model has any shared layer; all `None` otherwise (no overhead). ──
    kv_share_saved_k: Vec<Option<GpuTensor>>,
    kv_share_saved_v: Vec<Option<GpuTensor>>,
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
        Self::new_with_kv_mode_capped(gpu, config, max_seq, max_seq, kv_mode, kvarn_bits)
    }

    /// CASK-aware constructor: the plan retains `max_seq` as the logical
    /// context while full-context groups allocate only `physical_cap` slots.
    /// Sliding layers retain their architecture-defined bounded rings.
    pub fn new_with_kv_mode_capped(
        gpu: &mut Gpu,
        config: &Gemma4Config,
        max_seq: usize,
        physical_cap: usize,
        kv_mode: KvQuantMode,
        kvarn_bits: usize,
    ) -> HipResult<Self> {
        if physical_cap == 0 || physical_cap > max_seq {
            return Err(HipError::new(
                0,
                "Gemma 4 physical KV cap must be in 1..=max_seq",
            ));
        }
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
        // KVarN applies to the producer (Own) global/Full layers; KV-shared global
        // layers attend read-only against the producer's kvarn cache (handled in
        // attention_block). E2B/E4B global head_dim is 512 → the `_hd512` kvarn kernels.
        let has_kv_share = config
            .layers
            .iter()
            .any(|l| !matches!(l.kv_producer, KvProducer::Own));
        let kv_kvarn = matches!(kv_mode, KvQuantMode::Kvarn) && max_kv_width > 0;
        if kv_kvarn && physical_cap != max_seq {
            return Err(HipError::new(
                0,
                "Gemma 4 capped full-context KV currently requires F32 mode",
            ));
        }
        let kv = if kv_kvarn {
            LayeredKvArena::new_kvarn(gpu, plan.clone(), kvarn_bits)?
        } else {
            LayeredKvArena::new_fp32_capped(gpu, plan.clone(), physical_cap)?
        };
        let (kvarn_tiles, kvarn_flash_partials) = if kv_kvarn {
            let tiles = gpu.alloc_tensor(&[max_kv_width * KvCache::KVARN_GROUP], DType::F32)?;
            let max_tiles = physical_cap.div_ceil(KvCache::KVARN_GROUP);
            let partials = gpu.alloc_tensor(
                &[max_q_heads * max_tiles * (2 + kvarn_head_dim)],
                DType::F32,
            )?;
            (Some(tiles), Some(partials))
        } else {
            (None, None)
        };
        // PLE scratch (E2B/E4B): per-layer inputs + the model-projection branch +
        // per-layer merge gate/projection scratch.
        let (per_layer_inputs, ple_plmp, ple_gate, ple_proj) =
            if config.hidden_size_per_layer_input != 0 {
                let ple_dim = config.hidden_size_per_layer_input;
                let n = config.layers.len() * ple_dim;
                (
                    Some(gpu.alloc_tensor(&[n], DType::F32)?),
                    Some(gpu.alloc_tensor(&[n], DType::F32)?),
                    Some(gpu.alloc_tensor(&[ple_dim], DType::F32)?),
                    Some(gpu.alloc_tensor(&[config.hidden_size], DType::F32)?),
                )
            } else {
                (None, None, None, None)
            };
        // KV-sharing save buffers: one post-RoPE K/V slot per producer (Own) layer,
        // sized to that layer's kv_width; only when the model actually has shared
        // layers (E2B/E4B) — plain-dense gemma4 allocates none.
        let (mut kv_share_saved_k, mut kv_share_saved_v) = (
            Vec::with_capacity(config.layers.len()),
            Vec::with_capacity(config.layers.len()),
        );
        for l in &config.layers {
            if has_kv_share && matches!(l.kv_producer, KvProducer::Own) {
                let w = l.attention.kv_heads * l.attention.head_dim;
                kv_share_saved_k.push(Some(gpu.alloc_tensor(&[w], DType::F32)?));
                kv_share_saved_v.push(Some(gpu.alloc_tensor(&[w], DType::F32)?));
            } else {
                kv_share_saved_k.push(None);
                kv_share_saved_v.push(None);
            }
        }
        let attention = LayeredAttentionScratch::new(gpu, &plan)?;
        let hidden = config.hidden_size;
        let max_experts = config
            .layers
            .iter()
            .filter_map(|layer| match layer.ffn {
                crate::config::FfnPlan::Dense { .. } => None,
                crate::config::FfnPlan::DensePlusMoe { experts, .. } => Some(experts),
            })
            .max();
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
            ffn_norm: gpu.alloc_tensor(&[hidden], DType::F32)?,
            router_logits: max_experts
                .map(|experts| gpu.alloc_tensor(&[experts], DType::F32))
                .transpose()?,
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
            per_layer_inputs,
            ple_plmp,
            ple_gate,
            ple_proj,
            kv_share_saved_k,
            kv_share_saved_v,
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

    pub fn maybe_evict(
        &mut self,
        gpu: &mut Gpu,
        eviction: &LayeredEvictionCtx,
    ) -> HipResult<Option<EvictionResult>> {
        eviction.maybe_evict(gpu, &mut self.kv)
    }

    pub fn build_eviction(
        &self,
        gpu: &mut Gpu,
        artifact: &hipfire_runtime::triattn::TriAttnArtifact,
        budget: usize,
        beta: usize,
    ) -> Result<LayeredEvictionCtx, String> {
        LayeredEvictionCtx::new(gpu, artifact, &self.kv, budget, beta)
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.kv.free_gpu(gpu);
        self.attention.free_gpu(gpu);
        for tensor in [
            self.x,
            self.tmp,
            self.o,
            self.ffn_norm,
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
            .chain(self.per_layer_inputs)
            .chain(self.ple_plmp)
            .chain(self.ple_gate)
            .chain(self.ple_proj)
            .chain(self.router_logits)
            .chain(self.kv_share_saved_k.into_iter().flatten())
            .chain(self.kv_share_saved_v.into_iter().flatten())
        {
            let _ = gpu.free_tensor(tensor);
        }
        let _ = gpu.hip.free(self.pos_buf);
    }
}

/// Debug: append `[i32 tag][f32 × n]` for the hidden state to
/// `HIPFIRE_GEMMA4_DUMP_HS` when `position == HIPFIRE_GEMMA4_DUMP_POS` (tag = layer
/// index, or -1 for the post-embed vector). For the HF per-layer cosine diff.
fn dump_hs(gpu: &mut Gpu, x: &GpuTensor, tag: i32, position: usize) {
    let (Ok(path), Ok(dp)) = (
        std::env::var("HIPFIRE_GEMMA4_DUMP_HS"),
        std::env::var("HIPFIRE_GEMMA4_DUMP_POS"),
    ) else {
        return;
    };
    if dp.parse::<usize>() != Ok(position) {
        return;
    }
    if let Ok(h) = gpu.download_f32(x) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = f.write_all(&tag.to_le_bytes());
            let mut buf = Vec::with_capacity(h.len() * 4);
            for v in &h {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            let _ = f.write_all(&buf);
        }
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
        EmbeddingFormat::BF16 => gpu.embedding_lookup_bf16(
            &weights.core.token_embd,
            &state.x,
            token,
            config.hidden_size,
        )?,
        EmbeddingFormat::F16 => gpu.embedding_lookup_f16(
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

fn cask_physical_layer(resident_layer: usize, calibration_layer: Option<usize>) -> usize {
    calibration_layer.unwrap_or(resident_layer)
}

fn attention_block(
    gpu: &mut Gpu,
    layer_idx: usize,
    layer: &Gemma4DenseLayerWeights,
    config: &Gemma4Config,
    state: &Gemma4DenseState,
    position: usize,
    mut capture: Option<&mut Gemma4ForwardCapture>,
    calibration: Option<&Gemma4CalibrationCapture<'_>>,
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
    if let Some(calibration) = calibration {
        calibration_capture(
            gpu,
            calibration,
            ProjectionRole::QueryInput,
            None,
            &state.tmp,
            config.hidden_size,
        )?;
    }
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
    // Architecture-owned CASK tap: normalized Q immediately before this
    // layer's RoPE. The common layer-stream producer installs the tap only
    // during calibration, so serving pays only the relaxed enabled check.
    if hipfire_runtime::triattn::tap_enabled() {
        let q = gpu.download_f32(&scratch.q)?;
        // A streamed calibration layer is represented by a one-layer resident
        // state, so `layer_idx` is zero there. CASK metadata is keyed by the
        // physical model layer and must use the adapter's logical index.
        let cask_layer =
            cask_physical_layer(layer_idx, calibration.map(|capture| capture.logical_layer));
        hipfire_runtime::triattn::record_prerope_q(cask_layer, &q[..q_width]);
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

    // KV-sharing (E2B/E4B): a producer (Own) layer saves its final post-RoPE K/V; a
    // shared layer discards its own (K/V projected from ITS hidden state, which is
    // wrong) and reuses the producer's saved K/V — matching upstream's
    // `key_states = shared_kv_states[layer_type]`. The Q path is unchanged.
    let kv_bytes = kv_width * std::mem::size_of::<f32>();
    let is_shared = !matches!(plan.kv_producer, KvProducer::Own);
    match plan.kv_producer {
        KvProducer::Own => {
            if let (Some(sk), Some(sv)) = (
                state.kv_share_saved_k[layer_idx].as_ref(),
                state.kv_share_saved_v[layer_idx].as_ref(),
            ) {
                gpu.copy_d2d(&scratch.k, sk, kv_bytes)?;
                gpu.copy_d2d(&scratch.v, sv, kv_bytes)?;
            }
        }
        KvProducer::SharedFrom { producer_layer } => {
            let sk = state.kv_share_saved_k[producer_layer]
                .as_ref()
                .expect("gemma4 KV-share: producer K not saved");
            let sv = state.kv_share_saved_v[producer_layer]
                .as_ref()
                .expect("gemma4 KV-share: producer V not saved");
            gpu.copy_d2d(sk, &scratch.k, kv_bytes)?;
            gpu.copy_d2d(sv, &scratch.v, kv_bytes)?;
        }
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
            // A shared SWA layer must NOT write into the producer's ring — the
            // producer already wrote this position; staging above re-injected the
            // producer's saved K/V, so the read is correct without a write.
            if !is_shared {
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
        }
        KvStorageKind::Full if cache.quant_kvarn => {
            // KVarN (variance-normalized bits-bit K + Q8 V) for global/full-context
            // layers, n=1 (decode + per-token prefill). A producer (Own) layer does the
            // fused write+attend; a KV-shared layer reuses the producer's kvarn cache
            // (already written this token) and attends READ-ONLY — no K/V write.
            set_position(gpu, state, position)?;
            // Optional Hadamard-incoherence rotation of K and Q by the SAME per-head
            // FWHT-256; skipped at gemma4's global head_dim 512 (kvarn runs unrotated).
            static KVARN_ROTATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let kvarn_rotate = *KVARN_ROTATE
                .get_or_init(|| std::env::var("HIPFIRE_KVARN_ROTATE").ok().as_deref() != Some("0"));
            if kvarn_rotate && geometry.head_dim == 256 {
                gpu.rotate_x_mq_batched(&scratch.q, &scratch.q, q_width, 1)?;
                if !is_shared {
                    gpu.rotate_x_mq_batched(&scratch.k, &scratch.k, kv_width, 1)?;
                }
            }
            // The KV kernels read positions from a GpuTensor; wrap the raw 4-byte i32
            // pos_buf as a non-owning [1] view (mirrors gemma3/qwen35's KVarN hook).
            let pos_view = GpuTensor {
                buf: unsafe { DeviceBuffer::from_raw(state.pos_buf.as_ptr(), 4) },
                shape: vec![1],
                dtype: DType::F32,
            };
            let flash_partials = state
                .kvarn_flash_partials
                .as_ref()
                .expect("kvarn scratch allocated when kv_mode=kvarn");
            if is_shared {
                // Read-only kvarn flash over the producer's records + window.
                let seq_len = position + 1;
                let n_full_blocks = seq_len / KvCache::KVARN_GROUP;
                let rec_bytes =
                    KvCache::kvarn_k_record_bytes_bits(geometry.head_dim, cache.kvarn_bits);
                gpu.attention_flash_kvarn_batched_masked(
                    &scratch.q,
                    cache.k,
                    cache.k_window.expect("kvarn cache view exposes k_window"),
                    cache.v,
                    &scratch.attention,
                    &pos_view,
                    geometry.q_heads,
                    geometry.kv_heads,
                    geometry.head_dim,
                    cache.physical_cap,
                    seq_len,
                    1,
                    flash_partials,
                    None,
                    0,
                    0,
                    n_full_blocks,
                    rec_bytes,
                    cache.kvarn_bits,
                )?;
            } else {
                gpu.kvarn_attend(
                    cache.k,
                    cache.k_window.expect("kvarn cache view exposes k_window"),
                    cache.v,
                    &scratch.q,
                    &scratch.k,
                    &scratch.v,
                    &pos_view,
                    &scratch.attention,
                    flash_partials,
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
        }
        KvStorageKind::Full => {
            set_position(gpu, state, cache.physical_position)?;
            // A shared global layer reads the producer's full cache (which already
            // holds this position) — no write of its own.
            if !is_shared {
                gpu.kv_cache_write(cache.k, &scratch.k, &state.pos_buf, kv_width)?;
                gpu.kv_cache_write(cache.v, &scratch.v, &state.pos_buf, kv_width)?;
            }
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
    if let Some(calibration) = calibration {
        calibration_capture(
            gpu,
            calibration,
            ProjectionRole::AttentionOutputInput,
            None,
            &scratch.attention,
            q_width,
        )?;
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
    calibration: Option<&Gemma4CalibrationCapture<'_>>,
) -> HipResult<()> {
    gpu.rmsnorm_f32(
        &state.x,
        &layer.pre_ffn_norm,
        &state.ffn_norm,
        config.rms_norm_eps,
    )?;
    if let Some(calibration) = calibration {
        calibration_capture(
            gpu,
            calibration,
            ProjectionRole::DenseMlpInput,
            None,
            &state.ffn_norm,
            config.hidden_size,
        )?;
    }
    weight_gemv(gpu, &layer.w_gate, &state.ffn_norm, &state.gate)?;
    weight_gemv(gpu, &layer.w_up, &state.ffn_norm, &state.up)
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

fn topk_router_weights(
    logits: &[f32],
    per_expert_scale: &[f32],
    top_k: usize,
) -> HipResult<Vec<(usize, f32)>> {
    if logits.len() != per_expert_scale.len() || top_k == 0 || top_k > logits.len() {
        return Err(HipError::new(0, "Gemma 4 MoE router shape is invalid"));
    }
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, &logit)| (idx, (logit - max).exp() * per_expert_scale[idx]))
        .collect();
    probs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    probs.truncate(top_k);
    let denom: f32 = probs.iter().map(|(_, p)| *p).sum();
    if denom <= 0.0 || !denom.is_finite() {
        return Err(HipError::new(
            0,
            "Gemma 4 MoE router produced invalid weights",
        ));
    }
    for (_, p) in &mut probs {
        *p /= denom;
    }
    Ok(probs)
}

fn moe_ffn_finish(
    gpu: &mut Gpu,
    layer: &Gemma4DenseLayerWeights,
    moe: &Gemma4MoeLayerWeights,
    config: &Gemma4Config,
    state: &Gemma4DenseState,
    mut calibration: Option<&mut Gemma4CalibrationCapture<'_>>,
) -> HipResult<()> {
    let router_logits = state
        .router_logits
        .as_ref()
        .ok_or_else(|| HipError::new(0, "Gemma 4 MoE state is missing router logits"))?;
    gpu.rmsnorm_f32(&state.x, &moe.router_scale, &state.tmp, config.rms_norm_eps)?;
    if let Some(calibration) = calibration.as_deref() {
        calibration_capture(
            gpu,
            calibration,
            ProjectionRole::RouterInput,
            None,
            &state.tmp,
            config.hidden_size,
        )?;
    }
    weight_gemv(gpu, &moe.router, &state.tmp, router_logits)?;
    let logits = gpu.download_f32(router_logits)?;
    let selected = topk_router_weights(&logits, &moe.per_expert_scale, moe.top_k)?;
    if let Some(calibration) = calibration.as_deref_mut() {
        if let Some(telemetry) = calibration.telemetry.as_deref_mut() {
            let indices = selected
                .iter()
                .map(|(expert, _)| *expert)
                .collect::<Vec<_>>();
            let weights = selected
                .iter()
                .map(|(_, weight)| *weight)
                .collect::<Vec<_>>();
            telemetry
                // No row provenance in scope here (this is the resident forward,
                // not the streamed calibration path that carries token/stratum),
                // so only load and gate statistics accrue — the same information
                // this call recorded before the context parameter existed.
                .record_router_selection(
                    calibration.logical_layer,
                    RoutedRowContext::unknown(),
                    &indices,
                    &weights,
                )
                .map_err(|error| HipError::new(0, &error.to_string()))?;
            telemetry
                .record_grouped_batch_shape(
                    calibration.logical_layer,
                    indices.len(),
                    indices.len(),
                    indices.len(),
                )
                .map_err(|error| HipError::new(0, &error.to_string()))?;
        }
    }

    weight_gemv(
        gpu,
        &layer.w_down,
        &state.ffn.sub_offset(0, layer.w_down.k),
        &state.o,
    )?;
    for (expert_idx, weight) in selected {
        let expert = &moe.experts[expert_idx];
        let capture_gate_up = if let Some(calibration) = calibration.as_deref_mut() {
            match calibration.telemetry.as_deref_mut() {
                Some(telemetry) => {
                    telemetry
                        .record_capture_route(
                            calibration.logical_layer,
                            expert_idx,
                            ExpertCaptureRole::GateUpInput,
                            weight,
                        )
                        .map_err(|error| HipError::new(0, &error.to_string()))?
                        == CaptureAdmission::Capture
                }
                None => true,
            }
        } else {
            false
        };
        if capture_gate_up {
            calibration_capture(
                gpu,
                calibration.as_deref().unwrap(),
                ProjectionRole::GateUpInput,
                Some(expert_idx),
                &state.ffn_norm,
                config.hidden_size,
            )?;
            if let Some(calibration) = calibration.as_deref_mut() {
                let logical_layer = calibration.logical_layer;
                if let Some(telemetry) = calibration.telemetry.as_deref_mut() {
                    telemetry
                        .record_direct_capture_launch(
                            logical_layer,
                            expert_idx,
                            ExpertCaptureRole::GateUpInput,
                        )
                        .map_err(|error| HipError::new(0, &error.to_string()))?;
                }
            }
        }
        weight_gemv(gpu, &expert.gate, &state.ffn_norm, &state.gate)?;
        weight_gemv(gpu, &expert.up, &state.ffn_norm, &state.up)?;
        gpu.gelu_mul_f32(
            &state.gate,
            &state.up,
            &state.ffn.sub_offset(0, expert.down.k),
        )?;
        let capture_down = if let Some(calibration) = calibration.as_deref_mut() {
            match calibration.telemetry.as_deref_mut() {
                Some(telemetry) => {
                    telemetry
                        .record_capture_route(
                            calibration.logical_layer,
                            expert_idx,
                            ExpertCaptureRole::DownInput,
                            weight,
                        )
                        .map_err(|error| HipError::new(0, &error.to_string()))?
                        == CaptureAdmission::Capture
                }
                None => true,
            }
        } else {
            false
        };
        if capture_down {
            calibration_capture(
                gpu,
                calibration.as_deref().unwrap(),
                ProjectionRole::DownInput,
                Some(expert_idx),
                &state.ffn.sub_offset(0, expert.down.k),
                expert.down.k,
            )?;
            if let Some(calibration) = calibration.as_deref_mut() {
                let logical_layer = calibration.logical_layer;
                if let Some(telemetry) = calibration.telemetry.as_deref_mut() {
                    telemetry
                        .record_direct_capture_launch(
                            logical_layer,
                            expert_idx,
                            ExpertCaptureRole::DownInput,
                        )
                        .map_err(|error| HipError::new(0, &error.to_string()))?;
                }
            }
        }
        weight_gemv(
            gpu,
            &expert.down,
            &state.ffn.sub_offset(0, expert.down.k),
            &state.tmp,
        )?;
        gpu.scaled_add_inplace_cpu_scalar_f32(&state.o, &state.tmp, weight)?;
    }
    gpu.rmsnorm_f32(
        &state.o,
        &layer.post_ffn_norm,
        &state.tmp,
        config.rms_norm_eps,
    )?;
    gpu.add_f32(&state.x, &state.tmp, &state.x)
}

/// Row-lookup a per-token embedding from a (possibly quantized) `[vocab, dim]`
/// `WeightTensor`, dispatched on its stored dtype (mirrors `embed_token`).
fn embed_lookup_weight(
    gpu: &mut Gpu,
    wt: &WeightTensor,
    out: &GpuTensor,
    token: u32,
    dim: usize,
) -> HipResult<()> {
    match wt.gpu_dtype {
        DType::Q8_0 => gpu.embedding_lookup_q8(&wt.buf, out, token, dim),
        DType::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&wt.buf, out, token, dim),
        DType::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&wt.buf, out, token, dim),
        DType::F32 => gpu.embedding_lookup(&wt.buf, out, token, dim),
        DType::BF16 => gpu.embedding_lookup_bf16(&wt.buf, out, token, dim),
        DType::F16 => gpu.embedding_lookup_f16(&wt.buf, out, token, dim),
        DType::Q4K => gpu.embedding_lookup_q4k(&wt.buf, out, token, dim),
        other => Err(HipError::new(
            0,
            &format!("Gemma 4 PLE embed table dtype {other:?} unsupported"),
        )),
    }
}

/// Precompute the per-token PLE inputs into `state.per_layer_inputs` [num_layers ×
/// ple_dim] (E2B/E4B). Matches upstream `get_per_layer_inputs` +
/// `project_per_layer_inputs`:
///   tokens = embed_tokens_per_layer[t] · sqrt(ple_dim)   (ScaledWordEmbedding)
///   proj   = per_layer_projection_norm( (per_layer_model_projection · x_embed)·H^-0.5 )
///   ple    = (proj + tokens) · 2^-0.5
/// `state.x` must hold the scaled token embedding (i.e. called right after embed_token).
fn ple_embed_precompute(
    gpu: &mut Gpu,
    ple: &Gemma4PleWeights,
    config: &Gemma4Config,
    state: &Gemma4DenseState,
    token: u32,
) -> HipResult<()> {
    let ple_dim = ple.ple_dim;
    let n = ple.num_layers * ple_dim;
    let pli = state.per_layer_inputs.as_ref().expect("PLE state buffers");
    let plmp = state.ple_plmp.as_ref().expect("PLE state buffers");
    embed_lookup_weight(gpu, &ple.embed_per_layer, pli, token, n)?;
    gpu.scale_f32(pli, (ple_dim as f32).sqrt())?;
    weight_gemv(gpu, &ple.model_projection, &state.x, plmp)?;
    gpu.scale_f32(plmp, (config.hidden_size as f32).powf(-0.5))?;
    gpu.rmsnorm_batched(
        plmp,
        &ple.projection_norm,
        plmp,
        ple.num_layers,
        ple_dim,
        config.rms_norm_eps,
    )?;
    gpu.add_f32(pli, plmp, pli)?;
    gpu.scale_f32(pli, 0.5f32.sqrt())?;
    Ok(())
}

/// PLE per-layer merge (E2B/E4B), applied after the FFN residual and before the
/// `layer_scalar` scale:
///   h += post_per_layer_input_norm( per_layer_projection( act(input_gate·h) ⊙ ple[L] ) )
fn ple_merge(
    gpu: &mut Gpu,
    ple: &crate::weights::Gemma4PleLayerWeights,
    config: &Gemma4Config,
    state: &Gemma4DenseState,
    layer_idx: usize,
) -> HipResult<()> {
    let ple_dim = config.hidden_size_per_layer_input;
    let gate = state.ple_gate.as_ref().expect("PLE state buffers");
    let proj = state.ple_proj.as_ref().expect("PLE state buffers");
    let pli = state.per_layer_inputs.as_ref().expect("PLE state buffers");
    // gate = gelu_tanh(input_gate · h)  → [ple_dim]
    weight_gemv(gpu, &ple.input_gate, &state.x, gate)?;
    gpu.gelu_tanh_f32(gate, gate, ple_dim)?;
    // gate ⊙ per_layer_inputs[layer_idx]
    let slice = pli.sub_offset(layer_idx * ple_dim, ple_dim);
    gpu.mul_f32(gate, &slice, gate)?;
    // proj = per_layer_projection · gate  → [hidden]; RMSNorm; residual add.
    weight_gemv(gpu, &ple.projection, gate, proj)?;
    gpu.rmsnorm_f32(proj, &ple.post_norm, proj, config.rms_norm_eps)?;
    gpu.add_f32(&state.x, proj, &state.x)
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
    mut calibration: Option<&mut Gemma4CalibrationCapture<'_>>,
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
        calibration.as_deref(),
    )?;
    if let Some(capture) = capture.as_deref_mut() {
        capture_operator(gpu, capture, layer_idx, "post_attention_residual", &state.x)?;
    }
    ffn_project(gpu, layer, config, state, calibration.as_deref())?;
    if bf16_staged_geglu {
        gpu.bf16_round_trip_f32(&state.gate)?;
        gpu.bf16_round_trip_f32(&state.up)?;
    }
    if let Some(capture) = capture.as_deref_mut() {
        capture_operator(gpu, capture, layer_idx, "pre_ffn_norm", &state.ffn_norm)?;
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
    if let Some(calibration) = calibration.as_deref() {
        calibration_capture(
            gpu,
            calibration,
            ProjectionRole::DownInput,
            None,
            &state.ffn.sub_offset(0, layer.w_down.k),
            layer.w_down.k,
        )?;
    }
    if let Some(moe) = &layer.moe {
        moe_ffn_finish(gpu, layer, moe, config, state, calibration.as_deref_mut())?;
    } else {
        ffn_finish(gpu, layer, config, state)?;
    }
    if let Some(capture) = capture.as_deref_mut() {
        capture_operator(gpu, capture, layer_idx, "post_ffn_norm", &state.tmp)?;
        capture_operator(gpu, capture, layer_idx, "post_ffn_residual", &state.x)?;
    }
    // PLE per-layer merge (E2B/E4B) — between the FFN residual and the layer_scalar
    // scale, matching Gemma4TextDecoderLayer.forward.
    if let Some(ple) = &layer.ple {
        ple_merge(gpu, ple, config, state, layer_idx)?;
        if let Some(capture) = capture.as_deref_mut() {
            capture_operator(gpu, capture, layer_idx, "post_ple", &state.x)?;
        }
    }
    gpu.scale_f32(&state.x, layer.layer_scalar)?;
    if std::env::var("HIPFIRE_GEMMA4_DEBUG_NORMS").is_ok() {
        if let Ok(h) = gpu.download_f32(&state.x) {
            let mut sum = 0.0f32;
            let mut amax = 0.0f32;
            let mut nan = false;
            for v in &h {
                sum += v * v;
                amax = amax.max(v.abs());
                nan |= !v.is_finite();
            }
            let lp = &config.layers[layer_idx];
            let shared = !matches!(lp.kv_producer, KvProducer::Own);
            eprintln!(
                "[g4norm] L{layer_idx:>2} shared={shared} kind={:?} |x|={:.3} amax={amax:.3} nan={nan}",
                lp.kind,
                sum.sqrt()
            );
        }
    }
    dump_hs(gpu, &state.x, layer_idx as i32, position);
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
    if std::env::var("HIPFIRE_GEMMA4_DEBUG_NORMS").is_ok() {
        eprintln!("[g4tok] pos={position} token={token}");
    }
    embed_token(gpu, weights, config, state, token)?;
    if let Some(ple) = &weights.ple {
        ple_embed_precompute(gpu, ple, config, state, token)?;
    }
    dump_hs(gpu, &state.x, -1, position);
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
            None,
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
        gpu, weights, config, state, layer_idx, position, hidden, false, capture, None,
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
        gpu, weights, config, state, layer_idx, position, hidden, true, capture, None,
    )
}

/// Stream one production Gemma 4 layer while accumulating its registered
/// dense/router/expert projection inputs.
#[allow(clippy::too_many_arguments)]
pub fn calibration_forward_layer_from_hidden(
    gpu: &mut Gpu,
    weights: &Gemma4DenseWeights,
    config: &Gemma4Config,
    state: &mut Gemma4DenseState,
    layer_idx: usize,
    position: usize,
    hidden: &[f32],
    calibration: &mut Gemma4CalibrationCapture<'_>,
) -> HipResult<Vec<f32>> {
    diagnostic_forward_layer_from_hidden_impl(
        gpu,
        weights,
        config,
        state,
        layer_idx,
        position,
        hidden,
        false,
        None,
        Some(calibration),
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
    calibration: Option<&mut Gemma4CalibrationCapture<'_>>,
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
        calibration,
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
            Some(layer) => ffn_project(
                gpu,
                &self.weights.layers[layer],
                self.config,
                self.state,
                None,
            )
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
    // PLE, KV-sharing, and dense-MoE are implemented only on the reference
    // forward so far, so those variants must use it regardless of the oracle
    // env; the lowered superop programs are follow-ups. Plain-dense gemma4
    // (31B) still defaults to lowered.
    let needs_reference = weights.ple.is_some()
        || config
            .layers
            .iter()
            .zip(weights.layers.iter())
            .any(|(plan, layer)| {
                !matches!(plan.kv_producer, KvProducer::Own) || layer.moe.is_some()
            });
    let oracle = std::env::var("HIPFIRE_GEMMA4_FORWARD_ORACLE")
        .ok()
        .as_deref()
        == Some("1");
    if oracle || needs_reference {
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

    #[test]
    fn streamed_cask_uses_physical_not_one_layer_resident_index() {
        assert_eq!(cask_physical_layer(0, Some(17)), 17);
        assert_eq!(cask_physical_layer(17, None), 17);
    }
}
