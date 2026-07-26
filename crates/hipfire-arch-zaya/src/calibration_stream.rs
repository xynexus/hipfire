// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! ZAYA1 source-layout adapter for the family-neutral layer-stream engine
//! (`hipfire-coexistence calibrate`), the resumable sibling of the single-load
//! [`crate::calibration`] collector.
//!
//! The engine is layer-major: one hybrid decoder block is resident at a time,
//! every corpus row is pushed through it, the block's Hessian/imatrix part is
//! spooled, and the block is freed before the next one loads. That is what makes
//! `--resume` possible — a run that dies at block 27 restarts at block 27.
//!
//! Two ZAYA specifics shape this adapter:
//!
//! **The boundary row is wider than the residual.** The engine hands each layer
//! one f32 row per token and stores it between layers. ZAYA's EDA router carries
//! a second cross-block quantity — `router_states` `[router_hidden_size]` — which
//! block `l` reads from block `l-1`. It has exactly the residual's lifetime, so
//! it rides in the same row: `[hidden_size | router_hidden_size]`, and
//! `ModelInspection::hidden_width` is the **boundary row width**, not the model's
//! hidden size. This is what makes a resumed run pick up the EDA state correctly
//! instead of silently restarting it at zero.
//!
//! **Rows are single tokens.** Like the Gemma3 adapter, each row runs the block's
//! decode-shaped math against per-sequence state (KV cache, CCA conv ring, the
//! 1-token delayed value), so time tiles can cross microbatches. The routing
//! decision (host softmax → top-1 over `probs + balancing_biases`, with the
//! trailing MoD slot meaning "skip the FFN") is byte-for-byte the same host
//! reduction the resident forward performs.
//!
//! Reads the **raw Megatron alternating-half-layer checkpoint** directly — even
//! half-layer `2l` is block `l`'s CCA attention, odd `2l+1` is its EDA/MoD MoE,
//! and residual scales sit one half-layer ahead of their weights (see
//! [`crate::ingest`], which encodes the same mapping for the offline `.hfq`
//! conversion). Captured tensor names are the canonical hipfire names, identical
//! to [`crate::gpu::build_capture_names`], so both calibration paths produce
//! artifacts the quantizer reads the same way.
//!
//! **Known gap vs the resident collector:** the tied `model.embed_tokens`
//! lm-head input is not captured. The engine has no capture seam in the
//! finalizer phase (neither the Gemma3 nor Qwen3.5 adapter captures it either),
//! so a streamed artifact carries no lm-head Hessian and that projection falls
//! back to RTN. ZAYA's embed is best left bf16 anyway, but it is a real
//! difference — use the resident `collect-artifacts` path if you need it.

use crate::{ZayaConfig, ARCH_ID_ZAYA};
use hipfire_model::ModelSource;
use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_runtime::calibration::contracts::{
    CalibError, CalibrationJob, CaptureAdmission, CaptureDescriptor, CaptureId, CapturePolicy,
    CaptureRegistry, ExpertCaptureQuota, ExpertCaptureRole, ExpertTelemetry, ProjectionRole,
    RoutedRowContext, SampleRow,
};
use hipfire_runtime::calibration::schedule::{LayerMicrobatch, MicrobatchGeometry};
use hipfire_runtime::calibration::source::{
    load_source_f32_tensor, load_source_matrix, source_payload_f32, validate_source_shape,
    PlannedTensorReader, TensorLoadRequest, TensorOwner,
};
use hipfire_runtime::calibration::stream::{
    CalibrationEmbedding, CalibrationFamilyAdapter, CalibrationFinalizer, CalibrationLayer,
    CalibrationResourceEstimate, LayerCapturePartSummary, ModelInspection, RmsNormLmHeadFinalizer,
};
use hipfire_runtime::calibration::CalibCollector;
use hipfire_runtime::weights::{weight_gemm, WeightTensor};
use std::sync::Arc;

const SOURCE_DTYPES: &[&str] = &["BF16", "F16", "F32"];
const ZAYA_CALIBRATION_ARCH_IDS: &[u32] = &[ARCH_ID_ZAYA];
// v2 = position-slice batching. The version is part of the run fingerprint, so
// bumping it deliberately invalidates v1 checkpoints rather than letting a
// per-token boundary be resumed by batched math.
const ADAPTER_VERSION: &str = "zaya-stream-v2";
const FAMILY: &str = "zaya";

/// Router-MLP capture roles. The three router linears all consume distinct
/// activations of the same width, so they need distinct role codes; `fc1` reuses
/// the generic dense-MLP role and the other two take custom codes (which the
/// contract requires to be above 10).
const ROLE_ROUTER_FC1: ProjectionRole = ProjectionRole::DenseMlpInput;
const ROLE_ROUTER_FC2: ProjectionRole = ProjectionRole::Other(11);
const ROLE_ROUTER_OUT: ProjectionRole = ProjectionRole::Other(12);

/// Cap on distinct expert co-occurrence pairs kept in router telemetry.
const MAX_COOCCURRENCE_PAIRS: usize = 4096;

#[derive(Default)]
pub struct ZayaCalibrationAdapter {
    config: Option<ZayaConfig>,
    source_dtypes: Vec<String>,
}

fn zaya_calibration_adapter_factory() -> Box<dyn CalibrationFamilyAdapter> {
    Box::new(ZayaCalibrationAdapter::default())
}

hipfire_runtime::register_calibration_adapter!(
    FAMILY,
    ADAPTER_VERSION,
    ZAYA_CALIBRATION_ARCH_IDS,
    zaya_calibration_adapter_factory
);

// ── source layout ────────────────────────────────────────────────────────────

/// Raw half-layer index of block `l`'s CCA attention side.
fn attn_half(block: usize) -> usize {
    2 * block
}

/// Raw half-layer index of block `l`'s EDA/MoD MoE side.
fn moe_half(block: usize) -> usize {
    2 * block + 1
}

/// The four residual-scale sub-tensors, in the order the affine kernel wants:
/// hidden scale, hidden bias, residual scale, residual bias.
const RESIDUAL_SCALE_PARTS: [&str; 4] = [
    "hidden_states_scale",
    "hidden_states_bias",
    "residual_scale",
    "residual_bias",
];

/// Raw prefix of the residual scale applied *after* block `l`'s attention. The
/// checkpoint stores residual scales one half-layer ahead of the weights they
/// belong to, so block `l`'s post-attention scale lives on raw half-layer `2l+1`.
fn post_attention_scale_prefix(block: usize) -> String {
    format!("model.layers.{}.res_scale", moe_half(block))
}

/// Raw prefix of the residual scale applied after block `l`'s MoE: raw half-layer
/// `2l+2`, except for the last block, whose scale is the model-level one.
fn post_mlp_scale_prefix(block: usize, num_blocks: usize) -> String {
    if block + 1 == num_blocks {
        "model.res_scale".to_string()
    } else {
        format!("model.layers.{}.res_scale", attn_half(block + 1))
    }
}

fn config_from_source(source: &dyn ModelSource) -> Result<ZayaConfig, CalibError> {
    let meta: serde_json::Value = serde_json::from_str(source.metadata_json())
        .map_err(|error| CalibError::InvalidSourcePlan(format!("zaya metadata json: {error}")))?;
    let config_value = meta.get("config").unwrap_or(&meta);
    ZayaConfig::from_json(config_value)
        .map_err(|error| CalibError::InvalidSourcePlan(format!("zaya config: {error}")))
}

pub fn inspect_zaya_stream_source(source: &dyn ModelSource) -> Result<ModelInspection, CalibError> {
    if source.arch_id() != ARCH_ID_ZAYA {
        return Err(CalibError::InvalidSourcePlan(format!(
            "ZAYA calibration requires arch {ARCH_ID_ZAYA}, got {}",
            source.arch_id()
        )));
    }
    let config = config_from_source(source)?;
    // This adapter reads the published Megatron checkpoint's raw alternating
    // half-layer names. A converted native checkpoint stores the same weights
    // under the canonical hybrid-block names and would need a second name map;
    // refuse it rather than fail tensor-by-tensor.
    if config.num_half_layers != 2 * config.num_blocks {
        return Err(CalibError::InvalidSourcePlan(format!(
            "streamed ZAYA calibration reads the alternating Megatron layout \
             (num_hidden_layers = 2 x blocks); this source declares {} half-layers \
             for {} blocks",
            config.num_half_layers, config.num_blocks
        )));
    }
    Ok(ModelInspection {
        family: FAMILY.into(),
        arch_id: source.arch_id(),
        hidden_width: boundary_row_width(&config),
        vocab_size: config.vocab_size,
        num_layers: config.num_blocks,
        tensor_requests: zaya_tensor_requests(source, &config)?,
    })
}

/// Boundary row width: the residual stream plus the EDA router state that rides
/// with it between blocks.
fn boundary_row_width(config: &ZayaConfig) -> usize {
    config.hidden_size + config.moe.router_hidden_size
}

fn push_required(
    source: &dyn ModelSource,
    requests: &mut Vec<TensorLoadRequest>,
    logical_name: &str,
    source_name: &str,
    owner: TensorOwner,
) -> Result<(), CalibError> {
    let info = source.tensor_info(source_name).ok_or_else(|| {
        CalibError::InvalidSourcePlan(format!("missing ZAYA tensor {source_name}"))
    })?;
    if !SOURCE_DTYPES.contains(&info.dtype.as_str()) {
        return Err(CalibError::InvalidSourcePlan(format!(
            "ZAYA source tensor {source_name} has unsupported dtype {}",
            info.dtype
        )));
    }
    requests.push(TensorLoadRequest::tensor(logical_name, source_name, owner));
    Ok(())
}

pub fn zaya_tensor_requests(
    source: &dyn ModelSource,
    config: &ZayaConfig,
) -> Result<Vec<TensorLoadRequest>, CalibError> {
    let mut requests = Vec::new();
    push_required(
        source,
        &mut requests,
        "embedding",
        "model.embed_tokens.weight",
        TensorOwner::Persistent,
    )?;
    // The model input affine is raw half-layer 0's residual scale (hidden part
    // only; layer 0 carries no residual sub-tensors).
    push_required(
        source,
        &mut requests,
        "input_hidden_scale",
        "model.layers.0.res_scale.hidden_states_scale",
        TensorOwner::Persistent,
    )?;
    push_required(
        source,
        &mut requests,
        "input_hidden_bias",
        "model.layers.0.res_scale.hidden_states_bias",
        TensorOwner::Persistent,
    )?;
    push_required(
        source,
        &mut requests,
        "final_norm",
        "model.final_norm.weight",
        TensorOwner::Persistent,
    )?;
    // ZAYA ties the lm-head to the embedding table.
    requests.push(TensorLoadRequest::alias(
        "lm_head",
        "model.embed_tokens.weight",
        TensorOwner::Persistent,
        "embedding",
    ));

    for block in 0..config.num_blocks {
        let owner = TensorOwner::Layer(block);
        let attn = format!("model.layers.{}", attn_half(block));
        let moe = format!("model.layers.{}", moe_half(block));
        let router = format!("{moe}.zaya_block.router");
        let mut push = |logical: &str, source_name: String| -> Result<(), CalibError> {
            push_required(
                source,
                &mut requests,
                &format!("b{block}.{logical}"),
                &source_name,
                owner,
            )
        };

        // CCA attention half-layer.
        push("input_ln", format!("{attn}.input_norm.weight"))?;
        push("q_proj", format!("{attn}.self_attn.qkv.linear_q.weight"))?;
        push("k_proj", format!("{attn}.self_attn.qkv.linear_k.weight"))?;
        push("v_cur", format!("{attn}.self_attn.qkv.val_proj1.weight"))?;
        push("v_del", format!("{attn}.self_attn.qkv.val_proj2.weight"))?;
        push(
            "conv_dw_w",
            format!("{attn}.self_attn.qkv.conv_qk.0.weight"),
        )?;
        push("conv_dw_b", format!("{attn}.self_attn.qkv.conv_qk.0.bias"))?;
        push(
            "conv_gr_w",
            format!("{attn}.self_attn.qkv.conv_qk.1.weight"),
        )?;
        push("conv_gr_b", format!("{attn}.self_attn.qkv.conv_qk.1.bias"))?;
        push("qk_temp", format!("{attn}.self_attn.qkv.temp"))?;
        push("o_proj", format!("{attn}.self_attn.o_proj.weight"))?;

        // EDA/MoD MoE half-layer.
        push("post_attn_ln", format!("{moe}.input_norm.weight"))?;
        push("down_proj_w", format!("{router}.down_proj.weight"))?;
        push("down_proj_b", format!("{router}.down_proj.bias"))?;
        push("rnorm_w", format!("{router}.rmsnorm_eda.weight"))?;
        push("fc1_w", format!("{router}.router_mlp.0.weight"))?;
        push("fc1_b", format!("{router}.router_mlp.0.bias"))?;
        push("fc2_w", format!("{router}.router_mlp.2.weight"))?;
        push("fc2_b", format!("{router}.router_mlp.2.bias"))?;
        push("out_proj_w", format!("{router}.router_mlp.4.weight"))?;
        push("balancing_biases", format!("{router}.balancing_biases"))?;
        // Block 0 has no previous router state to mix in, and the checkpoint
        // omits its scale.
        if block != 0 {
            push(
                "router_states_scale",
                format!("{router}.router_states_scale"),
            )?;
        }
        for expert in 0..config.moe.num_experts {
            let prefix = format!("{moe}.zaya_block.experts.local_experts.{expert}");
            push(
                &format!("expert{expert}.gate_up"),
                format!("{prefix}.linear_fc1.weight"),
            )?;
            push(
                &format!("expert{expert}.down"),
                format!("{prefix}.linear_fc2.weight"),
            )?;
        }

        // Residual scales, one half-layer ahead of the weights they scale.
        let pa = post_attention_scale_prefix(block);
        let pm = post_mlp_scale_prefix(block, config.num_blocks);
        for part in RESIDUAL_SCALE_PARTS {
            push(&format!("pa_rs.{part}"), format!("{pa}.{part}"))?;
        }
        for part in RESIDUAL_SCALE_PARTS {
            push(&format!("pm_rs.{part}"), format!("{pm}.{part}"))?;
        }
    }
    Ok(requests)
}

// ── capture plan ─────────────────────────────────────────────────────────────

fn register_dense(
    registry: &mut CaptureRegistry,
    block: usize,
    role: ProjectionRole,
    output_names: Vec<String>,
    input_width: usize,
) -> Result<(), CalibError> {
    registry.register(CaptureDescriptor {
        id: CaptureId::new(block, role, None),
        output_names,
        input_width,
        policy: CapturePolicy::HessianAndImatrix,
        layer: block,
        role,
        expert: None,
        expert_quota: None,
    })
}

fn register_expert(
    registry: &mut CaptureRegistry,
    block: usize,
    expert: usize,
    role: ProjectionRole,
    output_name: String,
    input_width: usize,
    quota: ExpertCaptureQuota,
) -> Result<(), CalibError> {
    registry.register(CaptureDescriptor {
        id: CaptureId::new(block, role, Some(expert)),
        output_names: vec![output_name],
        input_width,
        // Routed experts are sparse under top-1: imatrix only, matching the
        // resident collector, which skips full Hessians for `.experts.`.
        policy: CapturePolicy::ImatrixOnly,
        layer: block,
        role,
        expert: Some(expert),
        expert_quota: Some(quota),
    })
}

pub fn zaya_capture_registry(
    config: &ZayaConfig,
    quota: ExpertCaptureQuota,
) -> Result<CaptureRegistry, CalibError> {
    let mut registry = CaptureRegistry::default();
    let hidden = config.hidden_size;
    let q_dim = config.attn.num_heads * config.attn.head_dim;
    let rh = config.moe.router_hidden_size;
    for block in 0..config.num_blocks {
        let prefix = format!("model.layers.{block}");
        let qkv = format!("{prefix}.self_attn.qkv_proj");
        let rmlp = format!("{prefix}.mlp.gate.router_mlp");
        // All four in-projections read the post-input-norm hidden state.
        register_dense(
            &mut registry,
            block,
            ProjectionRole::QueryInput,
            vec![
                format!("{qkv}.q_proj"),
                format!("{qkv}.k_proj"),
                format!("{qkv}.v_proj_current"),
                format!("{qkv}.v_proj_delayed"),
            ],
            hidden,
        )?;
        register_dense(
            &mut registry,
            block,
            ProjectionRole::AttentionOutputInput,
            vec![format!("{prefix}.self_attn.o_proj")],
            q_dim,
        )?;
        register_dense(
            &mut registry,
            block,
            ProjectionRole::RouterInput,
            vec![format!("{prefix}.mlp.gate.down_proj")],
            hidden,
        )?;
        register_dense(
            &mut registry,
            block,
            ROLE_ROUTER_FC1,
            vec![format!("{rmlp}.fc1")],
            rh,
        )?;
        register_dense(
            &mut registry,
            block,
            ROLE_ROUTER_FC2,
            vec![format!("{rmlp}.fc2")],
            rh,
        )?;
        register_dense(
            &mut registry,
            block,
            ROLE_ROUTER_OUT,
            vec![format!("{rmlp}.out_proj")],
            rh,
        )?;
        for expert in 0..config.moe.num_experts {
            register_expert(
                &mut registry,
                block,
                expert,
                ProjectionRole::GateUpInput,
                format!("{prefix}.mlp.experts.{expert}.gate_up_proj"),
                hidden,
                quota,
            )?;
            register_expert(
                &mut registry,
                block,
                expert,
                ProjectionRole::DownInput,
                format!("{prefix}.mlp.experts.{expert}.down_proj"),
                config.moe.moe_intermediate_size,
                quota,
            )?;
        }
    }
    Ok(registry)
}

// ── embedding phase ──────────────────────────────────────────────────────────

struct ZayaCalibrationEmbedding {
    embedding: Option<GpuTensor>,
    in_scale: Option<GpuTensor>,
    in_bias: Option<GpuTensor>,
    row: Option<GpuTensor>,
    vocab_size: usize,
    hidden: usize,
    row_width: usize,
}

impl CalibrationEmbedding for ZayaCalibrationEmbedding {
    fn execute(
        &mut self,
        gpu: &mut Gpu,
        rows: &[SampleRow],
        output_f32: &mut [f32],
    ) -> Result<(), CalibError> {
        if rows.is_empty() || output_f32.len() != rows.len() * self.row_width {
            return Err(CalibError::InvalidOptions(
                "ZAYA embedding output does not match its row count and boundary width".into(),
            ));
        }
        let embedding = self.embedding.as_ref().unwrap();
        let row = self.row.as_ref().unwrap();
        for (index, sample_row) in rows.iter().enumerate() {
            if sample_row.token as usize >= self.vocab_size {
                return Err(CalibError::InvalidSamples(format!(
                    "token {} exceeds ZAYA vocabulary {}",
                    sample_row.token, self.vocab_size
                )));
            }
            // Gather one embedding row, then apply the model input affine.
            let source = embedding.sub_offset(sample_row.token as usize * self.hidden, self.hidden);
            gpu.memcpy_dtod_at_auto(&row.buf, 0, &source.buf, 0, self.hidden * 4)
                .map_err(|error| CalibError::Runtime(error.to_string()))?;
            gpu.zaya_affine_input_f32(
                row,
                row,
                self.in_scale.as_ref().unwrap(),
                self.in_bias.as_ref().unwrap(),
                self.hidden,
                self.hidden,
            )
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
            let values = gpu
                .download_f32(row)
                .map_err(|error| CalibError::Runtime(error.to_string()))?;
            let start = index * self.row_width;
            output_f32[start..start + self.hidden].copy_from_slice(&values[..self.hidden]);
            // The EDA router state starts at zero; block 0 does not mix it in.
            output_f32[start + self.hidden..start + self.row_width].fill(0.0);
        }
        Ok(())
    }

    fn finish(&mut self, gpu: &mut Gpu) -> Result<(), CalibError> {
        for tensor in [
            self.row.take(),
            self.in_bias.take(),
            self.in_scale.take(),
            self.embedding.take(),
        ]
        .into_iter()
        .flatten()
        {
            gpu.free_tensor(tensor)
                .map_err(|error| CalibError::Runtime(error.to_string()))?;
        }
        Ok(())
    }
}

// ── finalizer phase ──────────────────────────────────────────────────────────

/// Strips the EDA router tail off each boundary row before handing the residual
/// to the shared RMSNorm + lm-head KLD reducer.
struct ZayaFinalizer {
    inner: RmsNormLmHeadFinalizer,
    hidden: usize,
    row_width: usize,
    residual: Vec<f32>,
}

impl CalibrationFinalizer for ZayaFinalizer {
    fn execute_kld(
        &mut self,
        gpu: &mut Gpu,
        batch: &LayerMicrobatch,
        residual_f32: &[f32],
        top_k: usize,
        output: &mut Vec<hipfire_runtime::calibration::contracts::KldRefRow>,
    ) -> Result<(), CalibError> {
        let rows = batch.rows.len();
        if residual_f32.len() != rows * self.row_width {
            return Err(CalibError::InvalidOptions(
                "ZAYA finalizer input does not match its boundary row width".into(),
            ));
        }
        self.residual.clear();
        self.residual.reserve(rows * self.hidden);
        for row in 0..rows {
            let start = row * self.row_width;
            self.residual
                .extend_from_slice(&residual_f32[start..start + self.hidden]);
        }
        self.inner
            .execute_kld(gpu, batch, &self.residual, top_k, output)
    }

    fn finish(&mut self, gpu: &mut Gpu) -> Result<(), CalibError> {
        self.inner.finish(gpu)
    }
}

// ── per-sequence state ───────────────────────────────────────────────────────

/// One sequence's resident block state. Held across time tiles so a microbatch
/// boundary does not truncate attention, the CCA convolution history, or the
/// delayed value.
struct SeqState {
    k_cache: GpuTensor,
    v_cache: GpuTensor,
    conv_ring: GpuTensor,
    delayed_v: GpuTensor,
    pos_buf: hip_bridge::DeviceBuffer,
    max_seq: usize,
    next_pos: usize,
}

impl SeqState {
    fn new(gpu: &mut Gpu, config: &ZayaConfig, max_seq: usize) -> Result<Self, CalibError> {
        let kv_dim = config.attn.num_kv_heads * config.attn.head_dim;
        let conv_ch = config.attn.conv_channels();
        let pad = config.attn.conv_state_len();
        let v_half = kv_dim / 2;
        let mut alloc = |elements: usize| -> Result<GpuTensor, CalibError> {
            gpu.zeros(&[elements.max(1)], DType::F32)
                .map_err(|error| CalibError::Runtime(error.to_string()))
        };
        let k_cache = alloc(max_seq * kv_dim)?;
        let v_cache = alloc(max_seq * kv_dim)?;
        let conv_ring = alloc(conv_ch * pad)?;
        let delayed_v = alloc(v_half)?;
        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        Ok(Self {
            k_cache,
            v_cache,
            conv_ring,
            delayed_v,
            pos_buf,
            max_seq,
            next_pos: 0,
        })
    }

    /// Clear the state that a fresh sequence must not inherit. The KV rings need
    /// no clearing — attention only reads `0..=pos`.
    fn reset(&mut self, gpu: &mut Gpu) -> Result<(), CalibError> {
        gpu.fill_f32(&self.conv_ring, 0.0)
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        gpu.fill_f32(&self.delayed_v, 0.0)
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        self.next_pos = 0;
        Ok(())
    }

    fn free(self, gpu: &mut Gpu) {
        for tensor in [self.k_cache, self.v_cache, self.conv_ring, self.delayed_v] {
            let _ = gpu.free_tensor(tensor);
        }
        let _ = gpu.hip.free(self.pos_buf);
    }
}

// ── block weights + scratch ──────────────────────────────────────────────────

struct ExpertWeights {
    gate_up: WeightTensor,
    down: WeightTensor,
}

struct BlockWeights {
    input_ln: GpuTensor,
    post_attn_ln: GpuTensor,
    q_proj: WeightTensor,
    k_proj: WeightTensor,
    v_cur: WeightTensor,
    v_del: WeightTensor,
    conv_dw_w: GpuTensor,
    conv_dw_b: GpuTensor,
    conv_gr_w: GpuTensor,
    conv_gr_b: GpuTensor,
    qk_temp: GpuTensor,
    o_proj: WeightTensor,
    down_proj_w: WeightTensor,
    down_proj_b: GpuTensor,
    router_states_scale: Option<GpuTensor>,
    rnorm_w: GpuTensor,
    fc1_w: WeightTensor,
    fc1_b: GpuTensor,
    fc2_w: WeightTensor,
    fc2_b: GpuTensor,
    out_proj_w: WeightTensor,
    balancing_biases: Vec<f32>,
    experts: Vec<ExpertWeights>,
    pa_rs: [GpuTensor; 4],
    pm_rs: [GpuTensor; 4],
}

impl BlockWeights {
    fn free(self, gpu: &mut Gpu) {
        for weight in [
            self.q_proj,
            self.k_proj,
            self.v_cur,
            self.v_del,
            self.o_proj,
            self.down_proj_w,
            self.fc1_w,
            self.fc2_w,
            self.out_proj_w,
        ] {
            let _ = gpu.free_tensor(weight.buf);
        }
        for expert in self.experts {
            let _ = gpu.free_tensor(expert.gate_up.buf);
            let _ = gpu.free_tensor(expert.down.buf);
        }
        let mut tensors = vec![
            self.input_ln,
            self.post_attn_ln,
            self.conv_dw_w,
            self.conv_dw_b,
            self.conv_gr_w,
            self.conv_gr_b,
            self.qk_temp,
            self.down_proj_b,
            self.rnorm_w,
            self.fc1_b,
            self.fc2_b,
        ];
        tensors.extend(self.pa_rs);
        tensors.extend(self.pm_rs);
        tensors.extend(self.router_states_scale);
        for tensor in tensors {
            let _ = gpu.free_tensor(tensor);
        }
    }
}

/// Position-slice scratch, allocated once per resident block. Every buffer
/// carries a leading `max_rows` dimension so one slice of same-position tokens
/// (one per live sequence) flows through the block in a single set of launches.
///
/// The boundary row is split in two here rather than kept interleaved as
/// `[residual | router_state]`: `zaya_router_prep_f32` and
/// `zaya_affine_residual_f32` index their inputs flatly over `rows * width`, so
/// they need each field contiguous across rows. De-interleaving costs a host
/// memcpy per slice (`split_boundary_rows` / `join_boundary_rows`) and saves
/// writing strided variants of both kernels.
struct Scratch {
    /// `[max_rows, hidden]` — the residual half of the boundary row.
    hidden_rows: GpuTensor,
    /// `[max_rows, router_hidden]` — the EDA router-state half.
    router_rows: GpuTensor,
    /// `[max_rows, hidden]` — expert inputs gathered into per-expert runs.
    expert_in: GpuTensor,
    normed: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    v_cur: GpuTensor,
    v_del: GpuTensor,
    q_res: GpuTensor,
    k_res: GpuTensor,
    cur_qk: GpuTensor,
    window: GpuTensor,
    dw: GpuTensor,
    gw: GpuTensor,
    query: GpuTensor,
    key: GpuTensor,
    value: GpuTensor,
    ctx: GpuTensor,
    attn_out: GpuTensor,
    g_res2: GpuTensor,
    rhid: GpuTensor,
    rnormed: GpuTensor,
    a1: GpuTensor,
    a2: GpuTensor,
    rlogits: GpuTensor,
    moe_out: GpuTensor,
    gate_up: GpuTensor,
    act: GpuTensor,
    down_t: GpuTensor,
}

impl Scratch {
    fn new(gpu: &mut Gpu, config: &ZayaConfig, max_rows: usize) -> Result<Self, CalibError> {
        let hidden = config.hidden_size;
        let attn = &config.attn;
        let q_dim = attn.num_heads * attn.head_dim;
        let k_dim = attn.num_kv_heads * attn.head_dim;
        let v_half = k_dim / 2;
        let conv_ch = attn.conv_channels();
        let pad = attn.conv_state_len();
        let dw_len = pad + 1 - attn.conv_depthwise_kernel + 1;
        let rh = config.moe.router_hidden_size;
        let moe_int = config.moe.moe_intermediate_size;
        let rows = max_rows.max(1);
        // Every buffer is row-major `[rows, width]`; `slice_rows` below indexes
        // row `r` as `sub_offset(r * width, width)`. Allocating the per-sequence
        // conv scratch at full width too costs little and keeps one addressing
        // rule for the whole block instead of two.
        let mut alloc = |width: usize| -> Result<GpuTensor, CalibError> {
            gpu.zeros(&[(rows * width).max(1)], DType::F32)
                .map_err(|error| CalibError::Runtime(error.to_string()))
        };
        Ok(Self {
            hidden_rows: alloc(hidden)?,
            router_rows: alloc(rh)?,
            expert_in: alloc(hidden)?,
            normed: alloc(hidden)?,
            q: alloc(q_dim)?,
            k: alloc(k_dim)?,
            v_cur: alloc(v_half)?,
            v_del: alloc(v_half)?,
            q_res: alloc(q_dim)?,
            k_res: alloc(k_dim)?,
            cur_qk: alloc(conv_ch)?,
            window: alloc(conv_ch * (pad + 1))?,
            dw: alloc(conv_ch * dw_len)?,
            gw: alloc(conv_ch)?,
            query: alloc(q_dim)?,
            key: alloc(k_dim)?,
            value: alloc(k_dim)?,
            ctx: alloc(q_dim)?,
            attn_out: alloc(hidden)?,
            g_res2: alloc(hidden)?,
            rhid: alloc(rh)?,
            rnormed: alloc(rh)?,
            a1: alloc(rh)?,
            a2: alloc(rh)?,
            rlogits: alloc(config.moe.num_router_experts())?,
            moe_out: alloc(hidden)?,
            gate_up: alloc(2 * moe_int)?,
            act: alloc(moe_int)?,
            down_t: alloc(hidden)?,
        })
    }

    fn free(self, gpu: &mut Gpu) {
        for tensor in [
            self.hidden_rows,
            self.router_rows,
            self.expert_in,
            self.normed,
            self.q,
            self.k,
            self.v_cur,
            self.v_del,
            self.q_res,
            self.k_res,
            self.cur_qk,
            self.window,
            self.dw,
            self.gw,
            self.query,
            self.key,
            self.value,
            self.ctx,
            self.attn_out,
            self.g_res2,
            self.rhid,
            self.rnormed,
            self.a1,
            self.a2,
            self.rlogits,
            self.moe_out,
            self.gate_up,
            self.act,
            self.down_t,
        ] {
            let _ = gpu.free_tensor(tensor);
        }
    }
}

fn load_block_weights(
    reader: &mut PlannedTensorReader<'_, '_, '_>,
    gpu: &mut Gpu,
    config: &ZayaConfig,
    block: usize,
) -> Result<BlockWeights, CalibError> {
    let hidden = config.hidden_size;
    let attn = &config.attn;
    let q_dim = attn.num_heads * attn.head_dim;
    let k_dim = attn.num_kv_heads * attn.head_dim;
    let v_half = k_dim / 2;
    let conv_ch = attn.conv_channels();
    let rh = config.moe.router_hidden_size;
    let n_route = config.moe.num_router_experts();
    let moe_int = config.moe.moe_intermediate_size;
    let name = |suffix: &str| format!("b{block}.{suffix}");

    // Read in plan order so the read ledger stays a monotonic continuation.
    let input_ln = load_source_f32_tensor(reader, gpu, &name("input_ln"), hidden, false)?;
    let q_proj = load_source_matrix(reader, gpu, &name("q_proj"), q_dim, hidden)?;
    let k_proj = load_source_matrix(reader, gpu, &name("k_proj"), k_dim, hidden)?;
    let v_cur = load_source_matrix(reader, gpu, &name("v_cur"), v_half, hidden)?;
    let v_del = load_source_matrix(reader, gpu, &name("v_del"), v_half, hidden)?;
    let conv_dw_w = load_source_f32_tensor(
        reader,
        gpu,
        &name("conv_dw_w"),
        conv_ch * attn.conv_depthwise_kernel,
        false,
    )?;
    let conv_dw_b = load_source_f32_tensor(reader, gpu, &name("conv_dw_b"), conv_ch, false)?;
    // The grouped conv keeps `conv_ch / groups` input channels per output channel.
    let grouped_elements =
        conv_ch * (conv_ch / (attn.num_heads + attn.num_kv_heads)) * attn.conv_grouped_kernel;
    let conv_gr_w =
        load_source_f32_tensor(reader, gpu, &name("conv_gr_w"), grouped_elements, false)?;
    let conv_gr_b = load_source_f32_tensor(reader, gpu, &name("conv_gr_b"), conv_ch, false)?;
    let qk_temp = load_source_f32_tensor(reader, gpu, &name("qk_temp"), attn.num_kv_heads, false)?;
    let o_proj = load_source_matrix(reader, gpu, &name("o_proj"), hidden, q_dim)?;

    let post_attn_ln = load_source_f32_tensor(reader, gpu, &name("post_attn_ln"), hidden, false)?;
    let down_proj_w = load_source_matrix(reader, gpu, &name("down_proj_w"), rh, hidden)?;
    let down_proj_b = load_source_f32_tensor(reader, gpu, &name("down_proj_b"), rh, false)?;
    let rnorm_w = load_source_f32_tensor(reader, gpu, &name("rnorm_w"), rh, false)?;
    let fc1_w = load_source_matrix(reader, gpu, &name("fc1_w"), rh, rh)?;
    let fc1_b = load_source_f32_tensor(reader, gpu, &name("fc1_b"), rh, false)?;
    let fc2_w = load_source_matrix(reader, gpu, &name("fc2_w"), rh, rh)?;
    let fc2_b = load_source_f32_tensor(reader, gpu, &name("fc2_b"), rh, false)?;
    let out_proj_w = load_source_matrix(reader, gpu, &name("out_proj_w"), n_route, rh)?;

    // Balancing biases stay on the host: the top-1 argmax is a host reduction,
    // exactly as the resident collector performs it.
    let balancing_biases = {
        let view = reader.read(&name("balancing_biases"))?;
        let values = source_payload_f32(view.info.dtype.as_str(), view.bytes)?;
        if values.len() != n_route {
            return Err(CalibError::InvalidSourcePlan(format!(
                "ZAYA block {block} balancing_biases has {} values, expected {n_route}",
                values.len()
            )));
        }
        values
    };
    let router_states_scale = if block == 0 {
        None
    } else {
        Some(load_source_f32_tensor(
            reader,
            gpu,
            &name("router_states_scale"),
            rh,
            false,
        )?)
    };

    let mut experts = Vec::with_capacity(config.moe.num_experts);
    for expert in 0..config.moe.num_experts {
        let gate_up = load_source_matrix(
            reader,
            gpu,
            &name(&format!("expert{expert}.gate_up")),
            2 * moe_int,
            hidden,
        )?;
        let down = load_source_matrix(
            reader,
            gpu,
            &name(&format!("expert{expert}.down")),
            hidden,
            moe_int,
        )?;
        experts.push(ExpertWeights { gate_up, down });
    }

    let mut load_scales = |group: &str| -> Result<[GpuTensor; 4], CalibError> {
        let mut parts = Vec::with_capacity(4);
        for part in RESIDUAL_SCALE_PARTS {
            parts.push(load_source_f32_tensor(
                reader,
                gpu,
                &name(&format!("{group}.{part}")),
                hidden,
                false,
            )?);
        }
        let mut parts = parts.into_iter();
        Ok([
            parts.next().unwrap(),
            parts.next().unwrap(),
            parts.next().unwrap(),
            parts.next().unwrap(),
        ])
    };
    let pa_rs = load_scales("pa_rs")?;
    let pm_rs = load_scales("pm_rs")?;

    Ok(BlockWeights {
        input_ln,
        post_attn_ln,
        q_proj,
        k_proj,
        v_cur,
        v_del,
        conv_dw_w,
        conv_dw_b,
        conv_gr_w,
        conv_gr_b,
        qk_temp,
        o_proj,
        down_proj_w,
        down_proj_b,
        router_states_scale,
        rnorm_w,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        out_proj_w,
        balancing_biases,
        experts,
        pa_rs,
        pm_rs,
    })
}

/// Widest position slice a run can produce: one row per sequence in the group.
///
/// The planner walks `for position { for sample in sequence_batch }`, so a slice
/// never exceeds `sequence_batch` rows however large the row budget is — sizing
/// the scratch from `max_rows` (which is `sequence_batch * time_tile`) would
/// over-allocate by the whole time tile.
fn max_slice_rows(job: &CalibrationJob) -> usize {
    let sequences = job.samples.samples().len().max(1);
    job.options
        .sequence_batch
        .unwrap_or(1)
        .min(job.options.max_rows.max(1))
        .min(sequences)
        .max(1)
}

// ── streamed layer ───────────────────────────────────────────────────────────

pub struct ZayaStreamedCalibrationLayer {
    block: usize,
    config: ZayaConfig,
    weights: Option<BlockWeights>,
    scratch: Option<Scratch>,
    sample_lengths: Vec<usize>,
    /// Per-sample corpus label, indexed by `SampleRow::sample_index`. Feeds the
    /// per-expert stratum profile so semantically-routed layers can be read.
    sample_strata: Vec<String>,
    active_sequence_start: Option<usize>,
    states: Vec<SeqState>,
    capture_registry: Option<Arc<CaptureRegistry>>,
    collector: Option<Arc<CalibCollector>>,
    telemetry: Option<ExpertTelemetry>,
    /// Widest position slice the scratch was sized for: one row per sequence in
    /// the group, since a slice holds at most one token per sequence.
    max_slice_rows: usize,
    /// Reused host staging for the boundary de-interleave, so a slice costs no
    /// allocation per step.
    hidden_host: Vec<f32>,
    router_host: Vec<f32>,
}

impl ZayaStreamedCalibrationLayer {
    fn load(
        reader: &mut PlannedTensorReader<'_, '_, '_>,
        gpu: &mut Gpu,
        config: &ZayaConfig,
        block: usize,
        job: &CalibrationJob,
    ) -> Result<Self, CalibError> {
        if block >= config.num_blocks {
            return Err(CalibError::InvalidSourcePlan(format!(
                "ZAYA block {block} exceeds {} blocks",
                config.num_blocks
            )));
        }
        let weights = load_block_weights(reader, gpu, config, block)?;
        let max_slice_rows = max_slice_rows(job);
        let scratch = match Scratch::new(gpu, config, max_slice_rows) {
            Ok(scratch) => scratch,
            Err(error) => {
                weights.free(gpu);
                return Err(error);
            }
        };
        Ok(Self {
            block,
            config: config.clone(),
            weights: Some(weights),
            scratch: Some(scratch),
            max_slice_rows,
            hidden_host: Vec::with_capacity(max_slice_rows * config.hidden_size),
            router_host: Vec::with_capacity(max_slice_rows * config.moe.router_hidden_size),
            sample_lengths: job
                .samples
                .samples()
                .iter()
                .map(|sample| sample.tokens.len())
                .collect(),
            sample_strata: job
                .samples
                .samples()
                .iter()
                .map(|sample| sample.stratum.clone())
                .collect(),
            active_sequence_start: None,
            states: Vec::new(),
            capture_registry: None,
            collector: None,
            telemetry: None,
        })
    }

    fn prepare_capture(&mut self, registry: &CaptureRegistry) -> Result<(), CalibError> {
        if let Some(existing) = &self.capture_registry {
            if existing.as_ref() != registry {
                return Err(CalibError::InvalidCapture(
                    "capture registry changed while a ZAYA block was resident".into(),
                ));
            }
            return Ok(());
        }
        let quota = registry
            .get(CaptureId::new(
                self.block,
                ProjectionRole::GateUpInput,
                Some(0),
            ))
            .and_then(|descriptor| descriptor.expert_quota)
            .ok_or_else(|| {
                CalibError::InvalidCapture(format!(
                    "ZAYA block {} has no routed expert quota descriptor",
                    self.block
                ))
            })?;
        self.telemetry = Some(ExpertTelemetry::new(
            self.config.num_blocks,
            self.config.moe.num_experts,
            self.config.moe.top_k,
            quota,
            MAX_COOCCURRENCE_PAIRS,
        )?);
        self.capture_registry = Some(Arc::new(registry.clone()));
        self.collector = Some(Arc::new(CalibCollector::new()));
        Ok(())
    }

    fn release_states(&mut self, gpu: &mut Gpu) {
        for state in self.states.drain(..) {
            state.free(gpu);
        }
        self.active_sequence_start = None;
    }

    fn prepare_sequence_group(
        &mut self,
        gpu: &mut Gpu,
        batch: &LayerMicrobatch,
    ) -> Result<(), CalibError> {
        if self.active_sequence_start == Some(batch.sequence_start) {
            if self.states.len() != batch.sequence_end - batch.sequence_start {
                return Err(CalibError::InvalidOptions(
                    "ZAYA sequence group width changed across time tiles".into(),
                ));
            }
            return Ok(());
        }
        self.release_states(gpu);
        if batch.sequence_start >= batch.sequence_end
            || batch.sequence_end > self.sample_lengths.len()
        {
            return Err(CalibError::InvalidOptions(format!(
                "invalid ZAYA sequence group {}..{} for {} samples",
                batch.sequence_start,
                batch.sequence_end,
                self.sample_lengths.len()
            )));
        }
        let mut states = Vec::with_capacity(batch.sequence_end - batch.sequence_start);
        for &length in &self.sample_lengths[batch.sequence_start..batch.sequence_end] {
            match SeqState::new(gpu, &self.config, length.max(1)) {
                Ok(state) => states.push(state),
                Err(error) => {
                    for state in states {
                        state.free(gpu);
                    }
                    return Err(error);
                }
            }
        }
        self.states = states;
        self.active_sequence_start = Some(batch.sequence_start);
        Ok(())
    }

    /// One position-slice through the resident hybrid block: `rows` tokens that
    /// all sit at the SAME position, one per live sequence.
    ///
    /// `scratch.hidden_rows` / `scratch.router_rows` already hold the slice's
    /// boundary halves and are updated in place.
    ///
    /// The dense projections run as one `weight_gemm` over all `rows`, which is
    /// the entire point: at batch 1 every token re-read the whole block's
    /// weights (37.0 MB/token measured on ZAYA1-8B), and `expert_gate_up` +
    /// `expert_down` alone were 68% of that. Only the per-sequence state carriers
    /// stay in a row loop — the CCA convolution, the KV write, attention, and
    /// RoPE. Their weights are 0.65 MB of the 37, and each sequence must read its
    /// own history regardless, so looping them costs ~2% of the win.
    ///
    /// RoPE is in the row loop for a correctness reason, not a cost one:
    /// `zaya_rope_partial_qk_posbuf_f32` computes `t = row_index + pos_buf[0]`,
    /// i.e. it assumes consecutive positions of ONE sequence (prefill shape). A
    /// position slice is the transpose of that — one position across many
    /// sequences — so batching it would rotate row `r` by `pos + r`. Called at
    /// `s = 1` per row it is correct as written.
    #[allow(clippy::too_many_arguments)]
    fn forward_position_slice(
        gpu: &mut Gpu,
        config: &ZayaConfig,
        weights: &BlockWeights,
        scratch: &Scratch,
        states: &mut [SeqState],
        rows: &[SliceRow],
        block: usize,
        collector: &CalibCollector,
        registry: &CaptureRegistry,
        telemetry: &mut ExpertTelemetry,
    ) -> Result<(), CalibError> {
        let hidden_dim = config.hidden_size;
        let attn = &config.attn;
        let (nq, nkv, hd) = (attn.num_heads, attn.num_kv_heads, attn.head_dim);
        let q_dim = nq * hd;
        let k_dim = nkv * hd;
        let kv_dim = k_dim;
        let v_half = k_dim / 2;
        let conv_ch = attn.conv_channels();
        let pad = attn.conv_state_len();
        let dw_len = pad + 1 - attn.conv_depthwise_kernel + 1;
        let rh = config.moe.router_hidden_size;
        let n_route = config.moe.num_router_experts();
        let n_experts = config.moe.num_experts;
        let moe_int = config.moe.moe_intermediate_size;
        let eps = config.rms_norm_eps;
        let l2_scale = (hd as f32).sqrt();
        let s = rows.len();

        let run = |result: hipfire_rdna::HipResult<()>| -> Result<(), CalibError> {
            result.map_err(|error| CalibError::Runtime(format!("{error:?}")))
        };

        let hidden = scratch.hidden_rows.sub_offset(0, s * hidden_dim);
        let router_state = scratch.router_rows.sub_offset(0, s * rh);
        let row = |tensor: &GpuTensor, index: usize, width: usize| -> GpuTensor {
            tensor.sub_offset(index * width, width)
        };

        // ── CCA attention ────────────────────────────────────────────────────
        run(gpu.rmsnorm_batched(
            &hidden,
            &weights.input_ln,
            &scratch.normed,
            s,
            hidden_dim,
            eps,
        ))?;
        collector.capture_by_id(
            gpu,
            registry,
            CaptureId::new(block, ProjectionRole::QueryInput, None),
            &scratch.normed,
            s,
            hidden_dim,
        )?;
        let gemm = |gpu: &mut Gpu, w: &WeightTensor, x: &GpuTensor, y: &GpuTensor, n: usize| {
            weight_gemm(gpu, w, x, y, n).map_err(|error| CalibError::Runtime(format!("{error:?}")))
        };
        gemm(gpu, &weights.q_proj, &scratch.normed, &scratch.q, s)?;
        gemm(gpu, &weights.k_proj, &scratch.normed, &scratch.k, s)?;
        gemm(gpu, &weights.v_cur, &scratch.normed, &scratch.v_cur, s)?;
        gemm(gpu, &weights.v_del, &scratch.normed, &scratch.v_del, s)?;

        // Per-sequence region: everything that reads or advances a sequence's own
        // CCA ring, delayed value, KV cache, or position.
        for (index, slice_row) in rows.iter().enumerate() {
            let state = &mut states[slice_row.state_index];
            let pos = state.next_pos;
            gpu.hip
                .memcpy_htod(&state.pos_buf, &(pos as i32).to_ne_bytes())
                .map_err(|error| CalibError::Runtime(error.to_string()))?;
            let (q_row, k_row) = (row(&scratch.q, index, q_dim), row(&scratch.k, index, k_dim));
            let q_res = row(&scratch.q_res, index, q_dim);
            let k_res = row(&scratch.k_res, index, k_dim);
            let cur_qk = row(&scratch.cur_qk, index, conv_ch);
            let window = row(&scratch.window, index, conv_ch * (pad + 1));
            let dw = row(&scratch.dw, index, conv_ch * dw_len);
            let gw = row(&scratch.gw, index, conv_ch);
            let query = row(&scratch.query, index, q_dim);
            let key = row(&scratch.key, index, k_dim);
            let value = row(&scratch.value, index, k_dim);
            let ctx = row(&scratch.ctx, index, q_dim);
            run(gpu.zaya_qk_prep_decode_f32(
                &q_row, &k_row, &q_res, &k_res, &cur_qk, nq, nkv, hd, q_dim, k_dim,
            ))?;
            run(gpu.zaya_conv_window_f32(&window, &state.conv_ring, &cur_qk, conv_ch, pad))?;
            run(gpu.zaya_conv1d_valid_f32(
                &dw,
                &window,
                &weights.conv_dw_w,
                &weights.conv_dw_b,
                conv_ch,
                conv_ch,
                attn.conv_depthwise_kernel,
                pad + 1,
                dw_len,
            ))?;
            run(gpu.zaya_conv1d_valid_f32(
                &gw,
                &dw,
                &weights.conv_gr_w,
                &weights.conv_gr_b,
                conv_ch,
                nq + nkv,
                attn.conv_grouped_kernel,
                dw_len,
                1,
            ))?;
            // s = 1: `conv` is indexed `[channel * s + t]`, so a one-row call
            // reads the contiguous `gw` this sequence just produced.
            run(gpu.zaya_add_conv_residual_qk_f32(
                &query, &key, &gw, &q_res, &k_res, 1, nq, nkv, hd, q_dim,
            ))?;
            run(gpu.zaya_value_assemble_decode_f32(
                &value,
                &row(&scratch.v_cur, index, v_half),
                &state.delayed_v,
                &row(&scratch.v_del, index, v_half),
                v_half,
            ))?;
            run(gpu.zaya_qk_l2norm_qk_f32(
                &query,
                &key,
                &weights.qk_temp,
                1,
                nq,
                nkv,
                hd,
                l2_scale,
                f32::EPSILON,
            ))?;
            run(gpu.zaya_rope_partial_qk_posbuf_f32(
                &query,
                &key,
                &state.pos_buf,
                1,
                nq,
                nkv,
                hd,
                attn.n_rot,
                attn.rope_theta,
            ))?;
            run(gpu.kv_cache_write(&state.k_cache, &key, &state.pos_buf, kv_dim))?;
            run(gpu.kv_cache_write(&state.v_cache, &value, &state.pos_buf, kv_dim))?;
            run(gpu.attention_f32(
                &query,
                &state.k_cache,
                &state.v_cache,
                &ctx,
                &state.pos_buf,
                pos + 1,
                nq,
                nkv,
                hd,
                state.max_seq,
            ))?;
            state.next_pos += 1;
        }

        collector.capture_by_id(
            gpu,
            registry,
            CaptureId::new(block, ProjectionRole::AttentionOutputInput, None),
            &scratch.ctx,
            s,
            q_dim,
        )?;
        gemm(gpu, &weights.o_proj, &scratch.ctx, &scratch.attn_out, s)?;
        run(gpu.zaya_affine_residual_f32(
            &scratch.g_res2,
            &scratch.attn_out,
            &hidden,
            &weights.pa_rs[0],
            &weights.pa_rs[1],
            &weights.pa_rs[2],
            &weights.pa_rs[3],
            hidden_dim,
            s * hidden_dim,
        ))?;

        // ── EDA router ───────────────────────────────────────────────────────
        run(gpu.rmsnorm_batched(
            &scratch.g_res2,
            &weights.post_attn_ln,
            &scratch.normed,
            s,
            hidden_dim,
            eps,
        ))?;
        collector.capture_by_id(
            gpu,
            registry,
            CaptureId::new(block, ProjectionRole::RouterInput, None),
            &scratch.normed,
            s,
            hidden_dim,
        )?;
        gemm(gpu, &weights.down_proj_w, &scratch.normed, &scratch.rhid, s)?;
        // bias add, EDA mix from the previous block's router state, then publish
        // this block's state back into the slice's router rows.
        run(gpu.zaya_router_prep_f32(
            &scratch.rhid,
            &weights.down_proj_b,
            &router_state,
            weights.router_states_scale.as_ref(),
            rh,
            s * rh,
        ))?;
        run(gpu.rmsnorm_batched(
            &scratch.rhid,
            &weights.rnorm_w,
            &scratch.rnormed,
            s,
            rh,
            eps,
        ))?;
        collector.capture_by_id(
            gpu,
            registry,
            CaptureId::new(block, ROLE_ROUTER_FC1, None),
            &scratch.rnormed,
            s,
            rh,
        )?;
        gemm(gpu, &weights.fc1_w, &scratch.rnormed, &scratch.a1, s)?;
        run(gpu.zaya_bias_gelu_f32(&scratch.a1, &weights.fc1_b, rh, s * rh))?;
        collector.capture_by_id(
            gpu,
            registry,
            CaptureId::new(block, ROLE_ROUTER_FC2, None),
            &scratch.a1,
            s,
            rh,
        )?;
        gemm(gpu, &weights.fc2_w, &scratch.a1, &scratch.a2, s)?;
        run(gpu.zaya_bias_gelu_f32(&scratch.a2, &weights.fc2_b, rh, s * rh))?;
        collector.capture_by_id(
            gpu,
            registry,
            CaptureId::new(block, ROLE_ROUTER_OUT, None),
            &scratch.a2,
            s,
            rh,
        )?;
        gemm(gpu, &weights.out_proj_w, &scratch.a2, &scratch.rlogits, s)?;

        // ── top-1 route (host reduction, matching the resident collector) ────
        // One download for the whole slice, then the identical per-row softmax /
        // argmax the single-token path used — the routing decision stays bit-for-bit
        // what it was, only the arithmetic that produced the logits is batched.
        let logits = gpu
            .download_f32(&scratch.rlogits)
            .map_err(|error| CalibError::Runtime(format!("{error:?}")))?;
        let mut routes = Vec::with_capacity(s);
        for index in 0..s {
            let row_logits = &logits[index * n_route..index * n_route + n_route];
            let max_logit = row_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0.0f32; n_route];
            let mut denom = 0.0f32;
            for expert in 0..n_route {
                probs[expert] = (row_logits[expert] - max_logit).exp();
                denom += probs[expert];
            }
            for prob in probs.iter_mut() {
                *prob /= denom;
            }
            let mut best = 0usize;
            let mut best_value = f32::NEG_INFINITY;
            for expert in 0..n_route {
                let value = probs[expert] + weights.balancing_biases[expert];
                if value > best_value {
                    best_value = value;
                    best = expert;
                }
            }
            routes.push((best, probs[best]));
        }
        // Telemetry is recorded in row order, before any expert grouping, so the
        // routed-row sequence an artifact records does not depend on batching.
        for (index, slice_row) in rows.iter().enumerate() {
            let (best, weight) = routes[index];
            // The trailing router slot is the MoD skip route, not an expert; the
            // telemetry contract counts it as a dropped index.
            telemetry.record_router_selection(
                block,
                RoutedRowContext::new(slice_row.token, slice_row.stratum.as_str()),
                &[best],
                &[weight],
            )?;
        }

        run(gpu.fill_f32(&scratch.moe_out.sub_offset(0, s * hidden_dim), 0.0))?;
        // Group the slice by winning expert so each ACTIVE expert's weights are
        // read once for all its rows instead of once per row. With 16 experts and
        // s = 128 that is at most 16 GEMMs of ~8 rows in place of 128 GEMVs — an
        // 8x cut on the 68% of block traffic the experts own, improving with s.
        for expert_index in 0..n_experts {
            let members: Vec<usize> = (0..s)
                .filter(|&index| routes[index].0 == expert_index)
                .collect();
            if members.is_empty() {
                continue;
            }
            let expert = &weights.experts[expert_index];
            let count = members.len();
            // Gather this expert's rows into a contiguous run. Device-to-device,
            // so the activations never round-trip through the host.
            for (slot, &index) in members.iter().enumerate() {
                gpu.hip
                    .memcpy_dtod_at(
                        &scratch.expert_in.buf,
                        slot * hidden_dim * 4,
                        &scratch.normed.buf,
                        index * hidden_dim * 4,
                        hidden_dim * 4,
                    )
                    .map_err(|error| CalibError::Runtime(error.to_string()))?;
            }
            // Capture admission is per row and in row order, matching the
            // single-token path's call sequence for the same set of rows.
            for (slot, &index) in members.iter().enumerate() {
                if telemetry.record_capture_route(
                    block,
                    expert_index,
                    ExpertCaptureRole::GateUpInput,
                    routes[index].1,
                )? == CaptureAdmission::Capture
                {
                    collector.capture_by_id(
                        gpu,
                        registry,
                        CaptureId::new(block, ProjectionRole::GateUpInput, Some(expert_index)),
                        &row(&scratch.expert_in, slot, hidden_dim),
                        1,
                        hidden_dim,
                    )?;
                }
            }
            gemm(
                gpu,
                &expert.gate_up,
                &scratch.expert_in,
                &scratch.gate_up,
                count,
            )?;
            // `silu_mul` is elementwise with no weight traffic, and gate/up are
            // interleaved per row (`[count, 2 * moe_int]`), so it stays per row.
            for slot in 0..count {
                let base = slot * 2 * moe_int;
                let gate = scratch.gate_up.sub_offset(base, moe_int);
                let up = scratch.gate_up.sub_offset(base + moe_int, moe_int);
                run(gpu.silu_mul_f32(&gate, &up, &row(&scratch.act, slot, moe_int)))?;
            }
            for (slot, &index) in members.iter().enumerate() {
                if telemetry.record_capture_route(
                    block,
                    expert_index,
                    ExpertCaptureRole::DownInput,
                    routes[index].1,
                )? == CaptureAdmission::Capture
                {
                    collector.capture_by_id(
                        gpu,
                        registry,
                        CaptureId::new(block, ProjectionRole::DownInput, Some(expert_index)),
                        &row(&scratch.act, slot, moe_int),
                        1,
                        moe_int,
                    )?;
                }
            }
            gemm(gpu, &expert.down, &scratch.act, &scratch.down_t, count)?;
            for (slot, &index) in members.iter().enumerate() {
                run(gpu.scaled_add_inplace_cpu_scalar_f32(
                    &row(&scratch.moe_out, index, hidden_dim),
                    &row(&scratch.down_t, slot, hidden_dim),
                    routes[index].1,
                ))?;
            }
        }
        run(gpu.zaya_affine_residual_f32(
            &hidden,
            &scratch.moe_out,
            &scratch.g_res2,
            &weights.pm_rs[0],
            &weights.pm_rs[1],
            &weights.pm_rs[2],
            &weights.pm_rs[3],
            hidden_dim,
            s * hidden_dim,
        ))?;
        Ok(())
    }
}

impl CalibrationLayer for ZayaStreamedCalibrationLayer {
    fn execute(
        &mut self,
        gpu: &mut Gpu,
        batch: &LayerMicrobatch,
        input_f32: &[f32],
        output_f32: &mut [f32],
        capture: &CaptureRegistry,
    ) -> Result<(), CalibError> {
        let row_width = boundary_row_width(&self.config);
        let expected = batch
            .rows
            .len()
            .checked_mul(row_width)
            .ok_or_else(|| CalibError::InvalidOptions("ZAYA boundary size overflow".into()))?;
        if batch.rows.len() != batch.boundary_rows.len()
            || batch.rows.is_empty()
            || input_f32.len() != expected
            || output_f32.len() != expected
        {
            return Err(CalibError::InvalidOptions(format!(
                "ZAYA block boundary has input/output lengths {}/{}, expected {expected}",
                input_f32.len(),
                output_f32.len()
            )));
        }
        self.prepare_sequence_group(gpu, batch)?;
        self.prepare_capture(capture)?;
        let collector = Arc::clone(self.collector.as_ref().unwrap());
        let registry = Arc::clone(self.capture_registry.as_ref().unwrap());
        let mut telemetry = self.telemetry.take().ok_or_else(|| {
            CalibError::InvalidCapture(format!("ZAYA block {} has no telemetry", self.block))
        })?;
        let weights = self.weights.take().ok_or_else(|| {
            CalibError::Runtime("ZAYA calibration weights were already freed".into())
        })?;
        let scratch = self
            .scratch
            .take()
            .ok_or_else(|| CalibError::Runtime("ZAYA calibration scratch was freed".into()))?;

        let hidden_dim = self.config.hidden_size;
        let rh = self.config.moe.router_hidden_size;
        let max_slice = self.max_slice_rows;
        let result = (|| -> Result<(), CalibError> {
            // The planner emits rows position-major (`for position { for sample }`),
            // so each maximal run of equal-position rows is a set of tokens from
            // DIFFERENT sequences — mutually independent apart from the
            // within-sequence dependency that attention and the CCA convolution
            // carry in `SeqState`. That run is the natural batch, and because the
            // run is contiguous in `batch.rows` it is contiguous in
            // `input_f32`/`output_f32` too. Ragged tails need no special case: a
            // slice is however many sequences still have a token at that position.
            let positions: Vec<usize> = batch.rows.iter().map(|row| row.position).collect();
            for range in position_slices(&positions, max_slice) {
                let (start_index, end_index) = (range.start, range.end);
                let slice = &batch.rows[start_index..end_index];
                let mut slice_rows = Vec::with_capacity(slice.len());
                for row in slice {
                    if row.sample_index < batch.sequence_start
                        || row.sample_index >= batch.sequence_end
                    {
                        return Err(CalibError::InvalidOptions(format!(
                            "sample {} is outside ZAYA scheduler group {}..{}",
                            row.sample_index, batch.sequence_start, batch.sequence_end
                        )));
                    }
                    let local = row.sample_index - batch.sequence_start;
                    let state = &mut self.states[local];
                    if row.reset_state {
                        state.reset(gpu)?;
                    }
                    if state.next_pos != row.position {
                        return Err(CalibError::InvalidOptions(format!(
                            "ZAYA sample {} expected position {}, got {}",
                            row.sample_index, state.next_pos, row.position
                        )));
                    }
                    slice_rows.push(SliceRow {
                        state_index: local,
                        token: row.token,
                        stratum: self
                            .sample_strata
                            .get(row.sample_index)
                            .cloned()
                            .unwrap_or_default(),
                    });
                }

                let start = start_index * row_width;
                let end = end_index * row_width;
                split_boundary_rows(
                    &input_f32[start..end],
                    row_width,
                    hidden_dim,
                    &mut self.hidden_host,
                    &mut self.router_host,
                );
                gpu.hip
                    .memcpy_htod(
                        &scratch.hidden_rows.buf,
                        f32_slice_as_bytes(&self.hidden_host),
                    )
                    .map_err(|error| CalibError::Runtime(error.to_string()))?;
                gpu.hip
                    .memcpy_htod(
                        &scratch.router_rows.buf,
                        f32_slice_as_bytes(&self.router_host),
                    )
                    .map_err(|error| CalibError::Runtime(error.to_string()))?;

                Self::forward_position_slice(
                    gpu,
                    &self.config,
                    &weights,
                    &scratch,
                    &mut self.states,
                    &slice_rows,
                    self.block,
                    collector.as_ref(),
                    registry.as_ref(),
                    &mut telemetry,
                )?;

                let rows = slice_rows.len();
                let hidden_out = gpu
                    .download_f32(&scratch.hidden_rows.sub_offset(0, rows * hidden_dim))
                    .map_err(|error| CalibError::Runtime(format!("{error:?}")))?;
                let router_out = gpu
                    .download_f32(&scratch.router_rows.sub_offset(0, rows * rh))
                    .map_err(|error| CalibError::Runtime(format!("{error:?}")))?;
                join_boundary_rows(
                    &hidden_out,
                    &router_out,
                    row_width,
                    hidden_dim,
                    &mut output_f32[start..end],
                );
            }
            Ok(())
        })();

        self.weights = Some(weights);
        self.scratch = Some(scratch);
        self.telemetry = Some(telemetry);
        result
    }

    fn write_capture_part(
        &mut self,
        gpu: &mut Gpu,
        path: &std::path::Path,
        arch_id: u32,
        metadata_json: &str,
    ) -> Result<LayerCapturePartSummary, CalibError> {
        let expert_telemetry = self
            .telemetry
            .as_ref()
            .map(|telemetry| telemetry.layer_snapshot(self.block))
            .transpose()?;
        let collector = self.collector.take().ok_or_else(|| {
            CalibError::InvalidCapture(format!(
                "ZAYA block {} has no calibration collector",
                self.block
            ))
        })?;
        let descriptors = collector.tensor_descriptors();
        if descriptors.is_empty() {
            collector.free_gpu(gpu);
            return Err(CalibError::InvalidCapture(format!(
                "ZAYA block {} captured no calibration tensors",
                self.block
            )));
        }
        let write_result = collector.write_streaming(gpu, path, arch_id, metadata_json, &[]);
        collector.free_gpu(gpu);
        self.capture_registry = None;
        let max_consistency =
            write_result.map_err(|error| CalibError::Runtime(error.to_string()))?;
        Ok(LayerCapturePartSummary {
            descriptors,
            max_consistency,
            expert_telemetry,
        })
    }

    fn finish(&mut self, gpu: &mut Gpu) -> Result<(), CalibError> {
        if let Some(collector) = self.collector.take() {
            collector.free_gpu(gpu);
        }
        self.capture_registry = None;
        self.telemetry = None;
        self.release_states(gpu);
        if let Some(scratch) = self.scratch.take() {
            scratch.free(gpu);
        }
        if let Some(weights) = self.weights.take() {
            weights.free(gpu);
        }
        Ok(())
    }
}

/// Split a microbatch's rows into position slices: maximal runs of rows sharing
/// one position, capped at `max_slice`.
///
/// This encodes the premise the whole batching rests on. `MicrobatchPlanner::plan`
/// walks `for position { for sample }` and pushes a row only when that sample
/// still has a token at that position, so a run of equal-position rows is a set
/// of tokens from DIFFERENT sequences, and ragged tails shrink the run instead of
/// needing a mask. If that traversal ever became sample-major, runs would be
/// length 1 and the result would be correct but unbatched — never wrong.
fn position_slices(positions: &[usize], max_slice: usize) -> Vec<std::ops::Range<usize>> {
    let max_slice = max_slice.max(1);
    let mut slices = Vec::new();
    let mut start = 0usize;
    while start < positions.len() {
        let mut end = start;
        while end < positions.len() && positions[end] == positions[start] && end - start < max_slice
        {
            end += 1;
        }
        slices.push(start..end);
        start = end;
    }
    slices
}

/// One token of a position slice, resolved against the resident sequence group.
struct SliceRow {
    /// Index into `ZayaStreamedCalibrationLayer::states`.
    state_index: usize,
    token: u32,
    stratum: String,
}

/// De-interleave `[rows, hidden | router]` boundary rows into two contiguous
/// runs. The block's batched kernels index flatly over `rows * width`, so each
/// half has to be contiguous across rows; the boundary store keeps them
/// interleaved per row because that is the unit the engine checkpoints.
fn split_boundary_rows(
    boundary: &[f32],
    row_width: usize,
    hidden_dim: usize,
    hidden_out: &mut Vec<f32>,
    router_out: &mut Vec<f32>,
) {
    let rows = boundary.len() / row_width;
    hidden_out.clear();
    router_out.clear();
    for row in 0..rows {
        let base = row * row_width;
        hidden_out.extend_from_slice(&boundary[base..base + hidden_dim]);
        router_out.extend_from_slice(&boundary[base + hidden_dim..base + row_width]);
    }
}

/// Inverse of [`split_boundary_rows`], writing the slice back into the engine's
/// interleaved boundary layout.
fn join_boundary_rows(
    hidden: &[f32],
    router: &[f32],
    row_width: usize,
    hidden_dim: usize,
    boundary: &mut [f32],
) {
    let router_width = row_width - hidden_dim;
    let rows = boundary.len() / row_width;
    for row in 0..rows {
        let base = row * row_width;
        boundary[base..base + hidden_dim]
            .copy_from_slice(&hidden[row * hidden_dim..(row + 1) * hidden_dim]);
        boundary[base + hidden_dim..base + row_width]
            .copy_from_slice(&router[row * router_width..(row + 1) * router_width]);
    }
}

fn f32_slice_as_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

// ── adapter ──────────────────────────────────────────────────────────────────

impl ZayaCalibrationAdapter {
    fn config(&self) -> Result<&ZayaConfig, CalibError> {
        self.config.as_ref().ok_or_else(|| {
            CalibError::InvalidSourcePlan(
                "ZAYA adapter must inspect the source before loading phases".into(),
            )
        })
    }
}

impl CalibrationFamilyAdapter for ZayaCalibrationAdapter {
    fn family(&self) -> &'static str {
        FAMILY
    }

    fn adapter_version(&self) -> &'static str {
        ADAPTER_VERSION
    }

    fn resource_estimate(
        &self,
        _model: &ModelInspection,
        job: &CalibrationJob,
        geometry: MicrobatchGeometry,
    ) -> Result<Option<CalibrationResourceEstimate>, CalibError> {
        Ok(Some(zaya_resource_estimate(self.config()?, job, geometry)?))
    }

    fn effective_precision(&self, gpu: &Gpu) -> serde_json::Value {
        let weight_execution = if gpu.arch.starts_with("gfx11") || gpu.arch.starts_with("gfx12") {
            "bf16-native"
        } else {
            "f16-fallback"
        };
        serde_json::json!({
            "boundary": "f32",
            "boundary_row_layout": "hidden_size | router_hidden_size (EDA router state)",
            "lm_head_capture": "not captured on the streamed path (no finalizer capture seam)",
            "source_dtypes": self.source_dtypes,
            "bf16_weight_execution": weight_execution,
            "gpu_arch": gpu.arch,
        })
    }

    fn inspect(&mut self, source: &dyn ModelSource) -> Result<ModelInspection, CalibError> {
        let inspection = inspect_zaya_stream_source(source)?;
        self.source_dtypes = inspection
            .tensor_requests
            .iter()
            .filter_map(|request| source.tensor_info(&request.source_name))
            .map(|info| info.dtype.clone())
            .collect();
        self.source_dtypes.sort();
        self.source_dtypes.dedup();
        self.config = Some(config_from_source(source)?);
        Ok(inspection)
    }

    fn capture_plan(
        &self,
        model: &ModelInspection,
        job: &CalibrationJob,
    ) -> Result<CaptureRegistry, CalibError> {
        let config = self.config()?;
        if model.num_layers != config.num_blocks || model.hidden_width != boundary_row_width(config)
        {
            return Err(CalibError::InvalidSourcePlan(
                "ZAYA inspection geometry changed before capture planning".into(),
            ));
        }
        zaya_capture_registry(config, job.options.expert_quota)
    }

    fn load_embedding(
        &mut self,
        reader: &mut PlannedTensorReader<'_, '_, '_>,
        gpu: &mut Gpu,
        model: &ModelInspection,
        _job: &CalibrationJob,
    ) -> Result<Box<dyn CalibrationEmbedding>, CalibError> {
        let config = self.config()?.clone();
        let hidden = config.hidden_size;
        let values = {
            let view = reader.read("embedding")?;
            validate_source_shape(view.info, &[model.vocab_size, hidden], "embedding")?;
            source_payload_f32(view.info.dtype.as_str(), view.bytes)?
        };
        if values.len() != model.vocab_size * hidden {
            return Err(CalibError::InvalidSourcePlan(
                "ZAYA embedding payload length does not match its shape".into(),
            ));
        }
        let embedding = gpu
            .upload_f32(&values, &[model.vocab_size, hidden])
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        drop(values);
        // Each fallible step below frees everything already claimed, so a failed
        // load never strands a multi-gigabyte embedding table on the device.
        let in_scale =
            match load_source_f32_tensor(reader, gpu, "input_hidden_scale", hidden, false) {
                Ok(tensor) => tensor,
                Err(error) => {
                    let _ = gpu.free_tensor(embedding);
                    return Err(error);
                }
            };
        let in_bias = match load_source_f32_tensor(reader, gpu, "input_hidden_bias", hidden, false)
        {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(in_scale);
                let _ = gpu.free_tensor(embedding);
                return Err(error);
            }
        };
        let row = match gpu.zeros(&[hidden], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(in_bias);
                let _ = gpu.free_tensor(in_scale);
                let _ = gpu.free_tensor(embedding);
                return Err(CalibError::Runtime(error.to_string()));
            }
        };
        Ok(Box::new(ZayaCalibrationEmbedding {
            embedding: Some(embedding),
            in_scale: Some(in_scale),
            in_bias: Some(in_bias),
            row: Some(row),
            vocab_size: model.vocab_size,
            hidden,
            row_width: model.hidden_width,
        }))
    }

    fn load_layer(
        &mut self,
        reader: &mut PlannedTensorReader<'_, '_, '_>,
        gpu: &mut Gpu,
        _model: &ModelInspection,
        layer: usize,
        job: &CalibrationJob,
    ) -> Result<Box<dyn CalibrationLayer>, CalibError> {
        let config = self.config()?.clone();
        Ok(Box::new(ZayaStreamedCalibrationLayer::load(
            reader, gpu, &config, layer, job,
        )?))
    }

    fn load_finalizer(
        &mut self,
        reader: &mut PlannedTensorReader<'_, '_, '_>,
        gpu: &mut Gpu,
        model: &ModelInspection,
        job: &CalibrationJob,
    ) -> Result<Box<dyn CalibrationFinalizer>, CalibError> {
        let config = self.config()?.clone();
        let hidden = config.hidden_size;
        let norm = load_source_f32_tensor(reader, gpu, "final_norm", hidden, false)?;
        let lm_head = match load_source_matrix(reader, gpu, "lm_head", model.vocab_size, hidden) {
            Ok(weight) => weight,
            Err(error) => {
                let _ = gpu.free_tensor(norm);
                return Err(error);
            }
        };
        let inner = RmsNormLmHeadFinalizer::new(
            gpu,
            norm,
            lm_head,
            hidden,
            model.vocab_size,
            config.rms_norm_eps,
            job.options.max_rows,
        )?;
        Ok(Box::new(ZayaFinalizer {
            inner,
            hidden,
            row_width: model.hidden_width,
            residual: Vec::new(),
        }))
    }
}

fn zaya_resource_estimate(
    config: &ZayaConfig,
    job: &CalibrationJob,
    geometry: MicrobatchGeometry,
) -> Result<CalibrationResourceEstimate, CalibError> {
    let hidden = config.hidden_size as u128;
    let attn = &config.attn;
    let q_dim = (attn.num_heads * attn.head_dim) as u128;
    let kv_dim = (attn.num_kv_heads * attn.head_dim) as u128;
    let conv_ch = attn.conv_channels() as u128;
    let pad = attn.conv_state_len() as u128;
    let rh = config.moe.router_hidden_size as u128;
    let moe_int = config.moe.moe_intermediate_size as u128;
    let active_sequences = geometry
        .sequence_batch
        .min(job.samples.samples().len())
        .max(1) as u128;
    // Position-slice scratch. Every buffer carries a leading row dimension, and a
    // slice holds at most one token per sequence, so it scales with the sequence
    // batch — NOT with `max_rows`, which is the whole `sequence_batch * time_tile`
    // rectangle. Widths, in order: the two boundary halves + expert gather +
    // normed/attn_out/g_res2/moe_out/down_t; the router chain; q/k working sets;
    // the CCA conv staging; and one slice's expert gate/up/act.
    let scratch_values_per_row = 7 * hidden
        + 5 * rh
        + 4 * q_dim
        + 5 * kv_dim
        + conv_ch * (2 * pad + 5)
        + 3 * moe_int
        + config.moe.num_router_experts() as u128;
    let scratch_bytes_total = scratch_values_per_row * active_sequences * 4;
    let context = job.samples.context_len() as u128;
    // Per-sequence state: full-context KV plus the CCA conv ring and delayed value.
    let state_bytes_per_sequence = context * kv_dim * 8 + (conv_ch * pad + kv_dim / 2) * 4;
    let active_state_bytes = state_bytes_per_sequence * active_sequences + scratch_bytes_total;
    let to_u64 = |label: &str, value: u128| {
        u64::try_from(value).map_err(|_| {
            CalibError::InvalidOptions(format!("ZAYA {label} byte estimate overflows"))
        })
    };
    Ok(CalibrationResourceEstimate {
        scratch_bytes: to_u64("scratch", scratch_bytes_total)?,
        state_bytes_per_sequence: to_u64("per-sequence state", state_bytes_per_sequence)?,
        active_state_bytes: to_u64("active state", active_state_bytes)?,
        details: serde_json::json!({
            "active_sequences": to_u64("active sequence count", active_sequences)?,
            "context": job.samples.context_len(),
            "boundary_row_width": boundary_row_width(config),
            "hidden_size": config.hidden_size,
            "router_hidden_size": config.moe.router_hidden_size,
            "num_experts": config.moe.num_experts,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ZayaConfig {
        ZayaConfig::from_json(&serde_json::json!({
            "architectures": ["ZayaForCausalLM"],
            "bos_token_id": 2,
            "eos_token_id": 106,
            "ffn_hidden_size": 4096,
            "head_dim": 128,
            "hidden_size": 2048,
            "max_position_embeddings": 131072,
            "model_type": "zaya",
            "moe_router_topk": 1,
            "norm_epsilon": 1e-05,
            "num_attention_heads": 8,
            "num_experts": 16,
            "num_hidden_layers": 80,
            "num_key_value_heads": 2,
            "pad_token_id": 0,
            "partial_rotary_factor": 0.5,
            "rope_theta": 5000000,
            "sliding_window": null,
            "vocab_size": 262272,
            "zaya_mlp_expansion": 256
        }))
        .expect("parse")
    }

    fn quota() -> ExpertCaptureQuota {
        ExpertCaptureQuota {
            min_rows: 8,
            target_rows: 16,
            tile_rows: 8,
            sampling:
                hipfire_runtime::calibration::contracts::ExpertSamplingPolicy::DeterministicFirst {
                    seed: 1,
                },
        }
    }

    #[test]
    fn planner_rows_group_into_cross_sequence_position_slices() {
        use hipfire_runtime::calibration::contracts::{CalibrationSample, SampleSet};
        use hipfire_runtime::calibration::schedule::{MicrobatchGeometry, MicrobatchPlanner};

        // Ragged on purpose: 4, 2 and 3 tokens. The short sequences must drop out
        // of the later slices without any masking.
        let samples = SampleSet::new(
            vec![
                CalibrationSample::new("a", vec![1, 2, 3, 4], "text"),
                CalibrationSample::new("b", vec![5, 6], "text"),
                CalibrationSample::new("c", vec![7, 8, 9], "text"),
            ],
            8,
            1,
        )
        .unwrap();
        let planner = MicrobatchPlanner::new(MicrobatchGeometry {
            sequence_batch: 3,
            time_tile: 4,
            row_budget: 12,
        })
        .unwrap();
        let batches = planner.plan(&samples);
        assert_eq!(batches.len(), 1, "one group, one tile");
        let rows = &batches[0].rows;
        let positions: Vec<usize> = rows.iter().map(|row| row.position).collect();

        let slices = position_slices(&positions, 3);
        // Position 0: all three sequences. Position 1: all three. Position 2:
        // a and c (b ended). Position 3: a alone.
        let widths: Vec<usize> = slices.iter().map(|range| range.len()).collect();
        assert_eq!(widths, vec![3, 3, 2, 1]);

        for range in &slices {
            let slice = &rows[range.clone()];
            // A slice is one position across DISTINCT sequences — that is what
            // makes the rows independent enough to batch.
            let position = slice[0].position;
            assert!(slice.iter().all(|row| row.position == position));
            let mut seen: Vec<usize> = slice.iter().map(|row| row.sample_index).collect();
            let before = seen.len();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), before, "a slice repeats a sequence");
        }
    }

    #[test]
    fn a_narrow_scratch_caps_slice_width_without_dropping_rows() {
        // Every row still runs, just in narrower slices — the cap is a memory
        // bound, never a correctness one.
        let positions = vec![0, 0, 0, 0, 0, 1, 1];
        let slices = position_slices(&positions, 2);
        assert_eq!(
            slices,
            vec![0..2, 2..4, 4..5, 5..7],
            "runs split at the cap and never straddle a position change"
        );
        let covered: usize = slices.iter().map(|range| range.len()).sum();
        assert_eq!(covered, positions.len());
    }

    #[test]
    fn boundary_halves_round_trip_through_the_de_interleave() {
        // row_width = 5 = hidden 3 + router 2, three rows.
        let boundary: Vec<f32> = (0..15).map(|value| value as f32).collect();
        let (mut hidden, mut router) = (Vec::new(), Vec::new());
        split_boundary_rows(&boundary, 5, 3, &mut hidden, &mut router);
        assert_eq!(hidden, vec![0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 10.0, 11.0, 12.0]);
        assert_eq!(router, vec![3.0, 4.0, 8.0, 9.0, 13.0, 14.0]);

        let mut rejoined = vec![0.0f32; boundary.len()];
        join_boundary_rows(&hidden, &router, 5, 3, &mut rejoined);
        assert_eq!(rejoined, boundary, "split/join must be exactly inverse");
    }

    #[test]
    fn scratch_rows_follow_the_sequence_batch_not_the_row_budget() {
        use hipfire_runtime::calibration::contracts::{
            CalibrationOptions, CalibrationSample, SampleSet,
        };

        let build = |sequences: usize, options: CalibrationOptions| {
            let samples: Vec<CalibrationSample> = (0..sequences)
                .map(|index| CalibrationSample::new(format!("s{index}"), vec![1, 2, 3, 4], "text"))
                .collect();
            CalibrationJob::new(
                "source",
                "tokenizer",
                SampleSet::new(samples, 8, 1).unwrap(),
                options,
            )
            .unwrap()
        };

        // A slice holds at most one token per sequence, so the scratch is sized
        // by the sequence batch (8) — NOT by max_rows (8 * 16 = 128). Sizing it
        // from max_rows would over-allocate every buffer by the whole time tile.
        let options = CalibrationOptions {
            max_rows: 128,
            sequence_batch: Some(8),
            time_tile: Some(16),
            ..CalibrationOptions::default()
        };
        assert_eq!(max_slice_rows(&build(32, options)), 8);

        // Fewer sequences than the batch: the corpus clamps it, since a slice
        // cannot be wider than the number of live sequences.
        let options = CalibrationOptions {
            max_rows: 128,
            sequence_batch: Some(8),
            time_tile: Some(16),
            ..CalibrationOptions::default()
        };
        assert_eq!(max_slice_rows(&build(3, options)), 3);

        // Geometry not yet resolved: fall back to one row, never zero.
        let options = CalibrationOptions {
            max_rows: 16,
            sequence_batch: None,
            time_tile: None,
            ..CalibrationOptions::default()
        };
        assert_eq!(max_slice_rows(&build(4, options)), 1);
    }

    #[test]
    fn boundary_row_carries_the_eda_router_state() {
        let config = config();
        assert_eq!(boundary_row_width(&config), 2048 + 256);
    }

    #[test]
    fn half_layer_mapping_matches_the_ingest_name_map() {
        // Block 1's attention side is raw half-layer 2, its MoE side raw 3.
        assert_eq!(attn_half(1), 2);
        assert_eq!(moe_half(1), 3);
        // Residual scales sit one half-layer ahead of the weights they scale.
        assert_eq!(
            post_attention_scale_prefix(1),
            "model.layers.3.res_scale".to_string()
        );
        assert_eq!(
            post_mlp_scale_prefix(1, 40),
            "model.layers.4.res_scale".to_string()
        );
        // The last block's post-MLP scale is the model-level one.
        assert_eq!(post_mlp_scale_prefix(39, 40), "model.res_scale".to_string());
    }

    #[test]
    fn residual_scale_names_agree_with_ingest_canonicalization() {
        let config = config();
        for block in 0..config.num_blocks {
            for part in RESIDUAL_SCALE_PARTS {
                let raw = format!("{}.{part}", post_attention_scale_prefix(block));
                assert_eq!(
                    crate::ingest::canonical_name(&raw, config.num_blocks).as_deref(),
                    Some(
                        format!("model.layers.{block}.post_attention_residual_scale.{part}")
                            .as_str()
                    ),
                    "post-attention scale {raw}"
                );
                let raw = format!("{}.{part}", post_mlp_scale_prefix(block, config.num_blocks));
                assert_eq!(
                    crate::ingest::canonical_name(&raw, config.num_blocks).as_deref(),
                    Some(format!("model.layers.{block}.post_mlp_residual_scale.{part}").as_str()),
                    "post-MLP scale {raw}"
                );
            }
        }
    }

    #[test]
    fn capture_names_match_the_resident_collector() {
        let config = config();
        let registry = zaya_capture_registry(&config, quota()).expect("registry");
        // The resident collector keys Hessians by these canonical names
        // (`crate::gpu::build_capture_names`); the quantizer looks them up by
        // the same strings, so both paths must agree exactly.
        for name in [
            "model.layers.0.self_attn.qkv_proj.q_proj",
            "model.layers.0.self_attn.qkv_proj.v_proj_delayed",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate.down_proj",
            "model.layers.0.mlp.gate.router_mlp.fc1",
            "model.layers.0.mlp.gate.router_mlp.fc2",
            "model.layers.0.mlp.gate.router_mlp.out_proj",
            "model.layers.39.mlp.experts.15.gate_up_proj",
            "model.layers.39.mlp.experts.15.down_proj",
        ] {
            assert!(
                registry.resolve_output(name).is_some(),
                "capture registry is missing {name}"
            );
        }
    }

    #[test]
    fn routed_experts_are_imatrix_only_and_quota_capped() {
        let config = config();
        let registry = zaya_capture_registry(&config, quota()).expect("registry");
        let expert = registry
            .get(CaptureId::new(3, ProjectionRole::GateUpInput, Some(2)))
            .expect("expert descriptor");
        assert_eq!(expert.policy, CapturePolicy::ImatrixOnly);
        assert_eq!(expert.expert_quota, Some(quota()));
        let dense = registry
            .get(CaptureId::new(3, ProjectionRole::RouterInput, None))
            .expect("router descriptor");
        assert_eq!(dense.policy, CapturePolicy::HessianAndImatrix);
        assert_eq!(dense.expert_quota, None);
    }

    #[test]
    fn native_layout_checkpoints_are_refused_with_a_clear_reason() {
        let mut native = config();
        // A converted native checkpoint reports one half-layer per block.
        native.num_half_layers = native.num_blocks;
        assert_ne!(native.num_half_layers, 2 * native.num_blocks);
    }
}
