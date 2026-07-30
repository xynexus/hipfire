// SPDX-License-Identifier: Apache-2.0
//! Qwen3.5 source-layout adapter for the family-neutral layer-stream engine.

use crate::qwen35::{
    build_calibration_session_batch_execution_plan, config_from_safetensors,
    dense_prefill_session_batch_host_pointer_tables,
    dense_prefill_session_batch_pointer_table_plan,
    dense_prefill_session_batch_prefix_tokens_positions,
    expected_dense_prefill_session_state_route_shape, forward_streamed_dense_layer_batch,
    forward_streamed_grouped_moe_layer_batch, free_streamed_layer_weights,
    upload_dense_prefill_session_batch_pointer_tables, upload_prefill_batch_inputs_with_positions,
    DeltaNetLayerWeights, DeltaNetMoeLayerWeights, DeltaNetState, DensePrefillSessionBatchInput,
    DensePrefillSessionDeltaStateRoute, DensePrefillSessionKvStateRoute,
    DensePrefillSessionStateRoute, ExpertWeights, FullAttnLayerWeights, FullAttnMoeLayerWeights,
    LayerType, LayerWeights, MoeFfnWeights, PrefillBatchScratch, Qwen35Config, RawExpertStorage,
    SharedExpertWeights, StateQuant,
};
use hip_bridge::DeviceBuffer;
use hipfire_model::{ModelSource, ARCH_ID_QWEN35_DENSE, ARCH_ID_QWEN35_MOE};
use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_runtime::calibration::contracts::{
    CalibError, CalibrationJob, CaptureDescriptor, CaptureId, CapturePolicy, CaptureRegistry,
    ExpertCaptureQuota, ExpertTelemetry, ProjectionRole, SampleRow,
};
use hipfire_runtime::calibration::expert_capture::GroupedMoeCalibrationCapture;
use hipfire_runtime::calibration::schedule::{LayerMicrobatch, MicrobatchGeometry};
use hipfire_runtime::calibration::source::{
    load_source_f32_tensor as load_f32_tensor, load_source_matrix as load_matrix,
    load_source_tensor, source_payload_f32, validate_source_shape as validate_shape,
    PlannedTensorReader, TensorLoadRequest, TensorOwner,
};
use hipfire_runtime::calibration::stream::{
    CalibrationEmbedding, CalibrationFamilyAdapter, CalibrationFinalizer, CalibrationLayer,
    CalibrationResourceEstimate, LayerCapturePartSummary, ModelInspection, RmsNormLmHeadFinalizer,
};
use hipfire_runtime::calibration::CalibCollector;
use hipfire_runtime::kv::KvCache;
use hipfire_runtime::triattn::{
    TriAttnAttentionKind, TriAttnContextPolicy, TriAttnLayerRecord, TriAttnPackageMetadata,
    TriAttnRopeConvention, TRIATTN_ARTIFACT_KIND, TRIATTN_HFQM_SCHEMA,
};
use hipfire_runtime::weights::WeightTensor;
use std::ffi::c_void;
use std::sync::Arc;

const SOURCE_DTYPES: &[&str] = &["BF16", "F16", "F32"];

fn capture_registry_is_cask_only(registry: &CaptureRegistry) -> bool {
    !registry.is_empty()
        && registry
            .descriptors()
            .all(|descriptor| descriptor.policy == CapturePolicy::Skip)
}

#[derive(Default)]
pub struct Qwen35CalibrationAdapter {
    config: Option<Qwen35Config>,
    source_dtypes: Vec<String>,
}

const QWEN35_CALIBRATION_ARCH_IDS: &[u32] = &[ARCH_ID_QWEN35_DENSE, ARCH_ID_QWEN35_MOE];

fn qwen35_calibration_adapter_factory() -> Box<dyn CalibrationFamilyAdapter> {
    Box::new(Qwen35CalibrationAdapter::default())
}

hipfire_runtime::register_calibration_adapter!(
    "qwen3.5",
    "qwen3.5-stream-v2",
    QWEN35_CALIBRATION_ARCH_IDS,
    qwen35_calibration_adapter_factory
);

impl Qwen35CalibrationAdapter {
    fn config(&self) -> Result<&Qwen35Config, CalibError> {
        self.config.as_ref().ok_or_else(|| {
            CalibError::InvalidSourcePlan(
                "Qwen3.5 adapter must inspect the source before loading phases".into(),
            )
        })
    }
}

pub fn inspect_qwen35_stream_source(
    source: &dyn ModelSource,
) -> Result<ModelInspection, CalibError> {
    let config = config_from_safetensors(source).ok_or_else(|| {
        CalibError::InvalidSourcePlan("could not parse Qwen3.5 source config".into())
    })?;
    if config.num_experts > 0 && !matches!(config.num_experts_per_tok, 8 | 10) {
        return Err(CalibError::InvalidSourcePlan(format!(
            "Qwen3.5 streamed calibration currently admits K-top 8 or 10, got {}",
            config.num_experts_per_tok
        )));
    }
    let tensor_requests = qwen35_tensor_requests(source, &config)?;
    Ok(ModelInspection {
        family: "qwen3.5".into(),
        arch_id: source.arch_id(),
        hidden_width: config.dim,
        vocab_size: config.vocab_size,
        num_layers: config.n_layers,
        tensor_requests,
    })
}

pub fn qwen35_tensor_requests(
    source: &dyn ModelSource,
    config: &Qwen35Config,
) -> Result<Vec<TensorLoadRequest>, CalibError> {
    if config.layer_types.len() != config.n_layers {
        return Err(CalibError::InvalidSourcePlan(format!(
            "Qwen3.5 config has {} layer types for {} layers",
            config.layer_types.len(),
            config.n_layers
        )));
    }
    let mut requests = Vec::new();
    push_required(
        source,
        &mut requests,
        "embedding",
        "embed_tokens.weight",
        TensorOwner::Persistent,
    )?;
    let embedding_source = requests.last().unwrap().source_name.clone();
    push_required(
        source,
        &mut requests,
        "final_norm",
        "norm.weight",
        TensorOwner::Persistent,
    )?;
    if qwen35_source_candidates("lm_head.weight")
        .iter()
        .any(|candidate| source.tensor_info(candidate).is_some())
    {
        push_required(
            source,
            &mut requests,
            "lm_head",
            "lm_head.weight",
            TensorOwner::Persistent,
        )?;
    } else {
        requests.push(TensorLoadRequest::alias(
            "lm_head",
            embedding_source,
            TensorOwner::Persistent,
            "embedding",
        ));
    }

    for (layer, layer_type) in config.layer_types.iter().enumerate() {
        let owner = TensorOwner::Layer(layer);
        let layer_prefix = format!("layers.{layer}");
        let mut suffixes = vec!["input_layernorm.weight", "post_attention_layernorm.weight"];
        match layer_type {
            LayerType::LinearAttention => suffixes.extend([
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.A_log",
                "linear_attn.dt_bias",
                "linear_attn.conv1d.weight",
                "linear_attn.norm.weight",
                "linear_attn.out_proj.weight",
            ]),
            LayerType::FullAttention => suffixes.extend([
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            ]),
        }
        if config.num_experts > 0 {
            suffixes.extend([
                "mlp.gate.weight",
                "mlp.experts.gate_up_proj",
                "mlp.experts.down_proj",
                "mlp.shared_expert.gate_proj.weight",
                "mlp.shared_expert.up_proj.weight",
                "mlp.shared_expert.down_proj.weight",
                "mlp.shared_expert_gate.weight",
            ]);
        } else {
            suffixes.extend([
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ]);
        }
        for suffix in suffixes {
            let relative = format!("{layer_prefix}.{suffix}");
            push_required(source, &mut requests, &relative, &relative, owner)?;
        }
    }
    Ok(requests)
}

pub fn qwen35_capture_registry(
    config: &Qwen35Config,
    quota: ExpertCaptureQuota,
) -> Result<CaptureRegistry, CalibError> {
    quota.validate()?;
    let mut registry = CaptureRegistry::default();
    for (layer, layer_type) in config.layer_types.iter().copied().enumerate() {
        let prefix = format!("model.language_model.layers.{layer}");
        let (attention_names, attention_output_name, attention_output_width) = match layer_type {
            LayerType::LinearAttention => (
                vec![
                    format!("{prefix}.linear_attn.in_proj_qkv"),
                    format!("{prefix}.linear_attn.in_proj_z"),
                    format!("{prefix}.linear_attn.in_proj_a"),
                    format!("{prefix}.linear_attn.in_proj_b"),
                ],
                format!("{prefix}.linear_attn.out_proj"),
                config.linear_num_value_heads * config.linear_value_head_dim,
            ),
            LayerType::FullAttention => (
                vec![
                    format!("{prefix}.self_attn.q_proj"),
                    format!("{prefix}.self_attn.k_proj"),
                    format!("{prefix}.self_attn.v_proj"),
                ],
                format!("{prefix}.self_attn.o_proj"),
                config.n_heads * config.head_dim,
            ),
        };
        register_capture(
            &mut registry,
            layer,
            ProjectionRole::QueryInput,
            None,
            attention_names,
            config.dim,
            CapturePolicy::HessianAndImatrix,
            None,
        )?;
        register_capture(
            &mut registry,
            layer,
            ProjectionRole::AttentionOutputInput,
            None,
            vec![attention_output_name],
            attention_output_width,
            CapturePolicy::HessianAndImatrix,
            None,
        )?;
        if config.num_experts == 0 {
            register_capture(
                &mut registry,
                layer,
                ProjectionRole::GateUpInput,
                None,
                vec![
                    format!("{prefix}.mlp.gate_proj"),
                    format!("{prefix}.mlp.up_proj"),
                ],
                config.dim,
                CapturePolicy::HessianAndImatrix,
                None,
            )?;
            register_capture(
                &mut registry,
                layer,
                ProjectionRole::DownInput,
                None,
                vec![format!("{prefix}.mlp.down_proj")],
                config.hidden_dim,
                CapturePolicy::HessianAndImatrix,
                None,
            )?;
        } else {
            register_capture(
                &mut registry,
                layer,
                ProjectionRole::RouterInput,
                None,
                vec![
                    format!("{prefix}.mlp.gate"),
                    format!("{prefix}.mlp.shared_expert.gate_proj"),
                    format!("{prefix}.mlp.shared_expert.up_proj"),
                    format!("{prefix}.mlp.shared_expert_gate"),
                ],
                config.dim,
                CapturePolicy::HessianAndImatrix,
                None,
            )?;
            register_capture(
                &mut registry,
                layer,
                ProjectionRole::SharedExpertInput,
                None,
                vec![format!("{prefix}.mlp.shared_expert.down_proj")],
                config.shared_expert_intermediate_size,
                CapturePolicy::HessianAndImatrix,
                None,
            )?;
            for expert in 0..config.num_experts {
                register_capture(
                    &mut registry,
                    layer,
                    ProjectionRole::GateUpInput,
                    Some(expert),
                    vec![format!("{prefix}.mlp.experts.{expert}.gate_up_proj")],
                    config.dim,
                    CapturePolicy::ImatrixOnly,
                    Some(quota),
                )?;
                register_capture(
                    &mut registry,
                    layer,
                    ProjectionRole::DownInput,
                    Some(expert),
                    vec![format!("{prefix}.mlp.experts.{expert}.down_proj")],
                    config.moe_intermediate_size,
                    CapturePolicy::ImatrixOnly,
                    Some(quota),
                )?;
            }
        }
    }
    Ok(registry)
}

#[allow(clippy::too_many_arguments)]
fn register_capture(
    registry: &mut CaptureRegistry,
    layer: usize,
    role: ProjectionRole,
    expert: Option<usize>,
    output_names: Vec<String>,
    input_width: usize,
    policy: CapturePolicy,
    expert_quota: Option<ExpertCaptureQuota>,
) -> Result<(), CalibError> {
    registry.register(CaptureDescriptor {
        id: CaptureId::new(layer, role, expert),
        output_names,
        input_width,
        policy,
        layer,
        role,
        expert,
        expert_quota,
    })
}

fn push_required(
    source: &dyn ModelSource,
    requests: &mut Vec<TensorLoadRequest>,
    logical_name: &str,
    relative_name: &str,
    owner: TensorOwner,
) -> Result<(), CalibError> {
    let candidates = qwen35_source_candidates(relative_name);
    let source_name = candidates
        .iter()
        .find(|candidate| source.tensor_info(candidate).is_some())
        .ok_or_else(|| {
            CalibError::InvalidSourcePlan(format!(
                "missing Qwen3.5 tensor {relative_name}; tried {}",
                candidates.join(", ")
            ))
        })?;
    let info = source.tensor_info(source_name).unwrap();
    if !SOURCE_DTYPES.contains(&info.dtype.as_str()) {
        return Err(CalibError::InvalidSourcePlan(format!(
            "Qwen3.5 source tensor {source_name} has unsupported dtype {}",
            info.dtype
        )));
    }
    requests.push(TensorLoadRequest::tensor(
        logical_name,
        source_name.clone(),
        owner,
    ));
    Ok(())
}

fn qwen35_source_candidates(relative_name: &str) -> Vec<String> {
    if relative_name == "lm_head.weight" {
        return vec![
            "lm_head.weight".into(),
            "model.language_model.lm_head.weight".into(),
            "model.lm_head.weight".into(),
        ];
    }
    vec![
        format!("model.language_model.{relative_name}"),
        format!("model.{relative_name}"),
        relative_name.to_string(),
    ]
}

#[derive(Default)]
struct PendingGpuLoads {
    tensors: Vec<Option<GpuTensor>>,
    weights: Vec<Option<WeightTensor>>,
}

impl PendingGpuLoads {
    fn push_tensor(&mut self, tensor: GpuTensor) -> usize {
        self.tensors.push(Some(tensor));
        self.tensors.len() - 1
    }

    fn push_weight(&mut self, weight: WeightTensor) -> usize {
        self.weights.push(Some(weight));
        self.weights.len() - 1
    }

    fn tensor(&self, index: usize) -> &GpuTensor {
        self.tensors[index].as_ref().unwrap()
    }

    fn take_tensor(&mut self, index: usize) -> GpuTensor {
        self.tensors[index].take().unwrap()
    }

    fn take_weight(&mut self, index: usize) -> WeightTensor {
        self.weights[index].take().unwrap()
    }

    fn free(&mut self, gpu: &mut Gpu) {
        for tensor in self.tensors.iter_mut().filter_map(Option::take) {
            let _ = gpu.free_tensor(tensor);
        }
        for weight in self.weights.iter_mut().filter_map(Option::take) {
            let _ = gpu.free_tensor(weight.buf);
        }
    }
}

pub fn load_qwen35_streamed_layer(
    reader: &mut PlannedTensorReader<'_, '_, '_>,
    gpu: &mut Gpu,
    config: &Qwen35Config,
    layer: usize,
) -> Result<LayerWeights, CalibError> {
    let prefix = format!("layers.{layer}");
    let mut pending = PendingGpuLoads::default();
    let result = (|| match config.layer_types[layer] {
        LayerType::LinearAttention => {
            let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
                + config.linear_num_value_heads * config.linear_value_head_dim;
            let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;
            let attn_norm = pending.push_tensor(load_f32_tensor(
                reader,
                gpu,
                &format!("{prefix}.input_layernorm.weight"),
                config.dim,
                true,
            )?);
            let ffn_norm = pending.push_tensor(load_f32_tensor(
                reader,
                gpu,
                &format!("{prefix}.post_attention_layernorm.weight"),
                config.dim,
                true,
            )?);
            let wqkv = pending.push_weight(load_matrix(
                reader,
                gpu,
                &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                qkv_dim,
                config.dim,
            )?);
            let wz = pending.push_weight(load_matrix(
                reader,
                gpu,
                &format!("{prefix}.linear_attn.in_proj_z.weight"),
                d_inner,
                config.dim,
            )?);
            let w_alpha = pending.push_weight(load_matrix(
                reader,
                gpu,
                &format!("{prefix}.linear_attn.in_proj_a.weight"),
                config.linear_num_value_heads,
                config.dim,
            )?);
            let w_beta = pending.push_weight(load_matrix(
                reader,
                gpu,
                &format!("{prefix}.linear_attn.in_proj_b.weight"),
                config.linear_num_value_heads,
                config.dim,
            )?);
            let a_log = pending.push_tensor(load_f32_tensor(
                reader,
                gpu,
                &format!("{prefix}.linear_attn.A_log"),
                config.linear_num_value_heads,
                false,
            )?);
            let dt_bias = pending.push_tensor(load_f32_tensor(
                reader,
                gpu,
                &format!("{prefix}.linear_attn.dt_bias"),
                config.linear_num_value_heads,
                false,
            )?);
            let conv_weight = pending.push_tensor(load_f32_tensor(
                reader,
                gpu,
                &format!("{prefix}.linear_attn.conv1d.weight"),
                qkv_dim * config.conv_kernel_dim,
                false,
            )?);
            let norm_weight = pending.push_tensor(load_f32_tensor(
                reader,
                gpu,
                &format!("{prefix}.linear_attn.norm.weight"),
                config.linear_value_head_dim,
                false,
            )?);
            let wo = pending.push_weight(load_matrix(
                reader,
                gpu,
                &format!("{prefix}.linear_attn.out_proj.weight"),
                config.dim,
                d_inner,
            )?);
            if config.num_experts > 0 {
                // Load the large routed FFN last. Its loader has its own rollback;
                // after it succeeds, assembling the layer is infallible.
                let ffn = load_streamed_moe_ffn(reader, gpu, config, layer, &prefix)?;
                Ok(LayerWeights::DeltaNetMoe(DeltaNetMoeLayerWeights {
                    attn_norm: pending.take_tensor(attn_norm),
                    wqkv: pending.take_weight(wqkv),
                    wz: pending.take_weight(wz),
                    w_alpha: pending.take_weight(w_alpha),
                    w_beta: pending.take_weight(w_beta),
                    a_log: pending.take_tensor(a_log),
                    dt_bias: pending.take_tensor(dt_bias),
                    conv_weight: pending.take_tensor(conv_weight),
                    norm_weight: pending.take_tensor(norm_weight),
                    wo: pending.take_weight(wo),
                    ffn_norm: pending.take_tensor(ffn_norm),
                    ffn,
                }))
            } else {
                let w_gate = pending.push_weight(load_matrix(
                    reader,
                    gpu,
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    config.hidden_dim,
                    config.dim,
                )?);
                let w_up = pending.push_weight(load_matrix(
                    reader,
                    gpu,
                    &format!("{prefix}.mlp.up_proj.weight"),
                    config.hidden_dim,
                    config.dim,
                )?);
                let w_down = pending.push_weight(load_matrix(
                    reader,
                    gpu,
                    &format!("{prefix}.mlp.down_proj.weight"),
                    config.dim,
                    config.hidden_dim,
                )?);
                Ok(LayerWeights::DeltaNet(DeltaNetLayerWeights {
                    attn_norm: pending.take_tensor(attn_norm),
                    wqkv: pending.take_weight(wqkv),
                    wz: pending.take_weight(wz),
                    w_alpha: pending.take_weight(w_alpha),
                    w_beta: pending.take_weight(w_beta),
                    a_log: pending.take_tensor(a_log),
                    dt_bias: pending.take_tensor(dt_bias),
                    conv_weight: pending.take_tensor(conv_weight),
                    norm_weight: pending.take_tensor(norm_weight),
                    wo: pending.take_weight(wo),
                    ffn_norm: pending.take_tensor(ffn_norm),
                    w_gate: pending.take_weight(w_gate),
                    w_up: pending.take_weight(w_up),
                    w_down: pending.take_weight(w_down),
                    bf16_down_shadow: None,
                }))
            }
        }
        LayerType::FullAttention => {
            let q_dim = config.n_heads * config.head_dim;
            let q_out_dim = if config.attn_output_gate {
                q_dim * 2
            } else {
                q_dim
            };
            let kv_dim = config.n_kv_heads * config.head_dim;
            let attn_norm = pending.push_tensor(load_f32_tensor(
                reader,
                gpu,
                &format!("{prefix}.input_layernorm.weight"),
                config.dim,
                true,
            )?);
            let ffn_norm = pending.push_tensor(load_f32_tensor(
                reader,
                gpu,
                &format!("{prefix}.post_attention_layernorm.weight"),
                config.dim,
                true,
            )?);
            let wq = pending.push_weight(load_matrix(
                reader,
                gpu,
                &format!("{prefix}.self_attn.q_proj.weight"),
                q_out_dim,
                config.dim,
            )?);
            let wk = pending.push_weight(load_matrix(
                reader,
                gpu,
                &format!("{prefix}.self_attn.k_proj.weight"),
                kv_dim,
                config.dim,
            )?);
            let wv = pending.push_weight(load_matrix(
                reader,
                gpu,
                &format!("{prefix}.self_attn.v_proj.weight"),
                kv_dim,
                config.dim,
            )?);
            let wo = pending.push_weight(load_matrix(
                reader,
                gpu,
                &format!("{prefix}.self_attn.o_proj.weight"),
                config.dim,
                q_dim,
            )?);
            let q_norm = pending.push_tensor(load_f32_tensor(
                reader,
                gpu,
                &format!("{prefix}.self_attn.q_norm.weight"),
                config.head_dim,
                true,
            )?);
            let k_norm = pending.push_tensor(load_f32_tensor(
                reader,
                gpu,
                &format!("{prefix}.self_attn.k_norm.weight"),
                config.head_dim,
                true,
            )?);
            if config.num_experts > 0 {
                let ffn = load_streamed_moe_ffn(reader, gpu, config, layer, &prefix)?;
                Ok(LayerWeights::FullAttnMoe(FullAttnMoeLayerWeights {
                    attn_norm: pending.take_tensor(attn_norm),
                    wq: pending.take_weight(wq),
                    wk: pending.take_weight(wk),
                    wv: pending.take_weight(wv),
                    wo: pending.take_weight(wo),
                    q_norm: pending.take_tensor(q_norm),
                    k_norm: pending.take_tensor(k_norm),
                    ffn_norm: pending.take_tensor(ffn_norm),
                    ffn,
                }))
            } else {
                let w_gate = pending.push_weight(load_matrix(
                    reader,
                    gpu,
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    config.hidden_dim,
                    config.dim,
                )?);
                let w_up = pending.push_weight(load_matrix(
                    reader,
                    gpu,
                    &format!("{prefix}.mlp.up_proj.weight"),
                    config.hidden_dim,
                    config.dim,
                )?);
                let w_down = pending.push_weight(load_matrix(
                    reader,
                    gpu,
                    &format!("{prefix}.mlp.down_proj.weight"),
                    config.dim,
                    config.hidden_dim,
                )?);
                Ok(LayerWeights::FullAttn(FullAttnLayerWeights {
                    attn_norm: pending.take_tensor(attn_norm),
                    wq: pending.take_weight(wq),
                    wk: pending.take_weight(wk),
                    wv: pending.take_weight(wv),
                    wo: pending.take_weight(wo),
                    q_norm: pending.take_tensor(q_norm),
                    k_norm: pending.take_tensor(k_norm),
                    ffn_norm: pending.take_tensor(ffn_norm),
                    w_gate: pending.take_weight(w_gate),
                    w_up: pending.take_weight(w_up),
                    w_down: pending.take_weight(w_down),
                    bf16_down_shadow: None,
                }))
            }
        }
    })();
    if result.is_err() {
        pending.free(gpu);
    }
    result
}

pub fn free_qwen35_streamed_layer(gpu: &mut Gpu, layer: LayerWeights) {
    free_streamed_layer_weights(gpu, layer);
}

struct Qwen35CalibrationEmbedding {
    embedding: Option<GpuTensor>,
    token_ids: Option<GpuTensor>,
    output: Option<GpuTensor>,
    vocab_size: usize,
    dim: usize,
    max_rows: usize,
}

impl CalibrationEmbedding for Qwen35CalibrationEmbedding {
    fn execute(
        &mut self,
        gpu: &mut Gpu,
        rows: &[SampleRow],
        output_f32: &mut [f32],
    ) -> Result<(), CalibError> {
        if rows.is_empty() || rows.len() > self.max_rows {
            return Err(CalibError::InvalidOptions(format!(
                "embedding batch has {} rows, expected 1..={} rows",
                rows.len(),
                self.max_rows
            )));
        }
        if output_f32.len() != rows.len() * self.dim {
            return Err(CalibError::InvalidOptions(
                "embedding output does not match row count and hidden width".into(),
            ));
        }
        let tokens = rows
            .iter()
            .map(|row| {
                if row.token as usize >= self.vocab_size {
                    Err(CalibError::InvalidSamples(format!(
                        "token {} exceeds vocabulary {}",
                        row.token, self.vocab_size
                    )))
                } else {
                    Ok(row.token as i32)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        gpu.hip
            .memcpy_htod(
                &self.token_ids.as_ref().unwrap().buf,
                i32_slice_as_bytes(&tokens),
            )
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        gpu.embedding_lookup_f32_batched(
            self.embedding.as_ref().unwrap(),
            self.output.as_ref().unwrap(),
            self.token_ids.as_ref().unwrap(),
            rows.len(),
            self.dim,
        )
        .map_err(|error| CalibError::Runtime(error.to_string()))?;
        let live = self
            .output
            .as_ref()
            .unwrap()
            .sub_offset(0, output_f32.len());
        let values = gpu
            .download_f32(&live)
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        output_f32.copy_from_slice(&values);
        Ok(())
    }

    fn finish(&mut self, gpu: &mut Gpu) -> Result<(), CalibError> {
        for tensor in [
            self.output.take(),
            self.token_ids.take(),
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

fn qwen35_resource_estimate(
    config: &Qwen35Config,
    job: &CalibrationJob,
    geometry: MicrobatchGeometry,
) -> Result<CalibrationResourceEstimate, CalibError> {
    let scratch_rows = geometry
        .sequence_batch
        .checked_mul(geometry.time_tile)
        .ok_or_else(|| CalibError::InvalidOptions("Qwen3.5 scratch row overflow".into()))?;
    let n = scratch_rows as u128;
    let dim = config.dim as u128;
    let hidden = config.hidden_dim as u128;
    let k_top = config.num_experts_per_tok as u128;
    let experts = config.num_experts as u128;
    let moe_intermediate = config.moe_intermediate_size as u128;
    let shared_intermediate = config.shared_expert_intermediate_size as u128;
    let linear_heads = config.linear_num_value_heads as u128;
    let linear_head_dim = config.linear_value_head_dim as u128;
    let linear_k_dim = (config.linear_num_key_heads * config.linear_key_head_dim) as u128;
    let linear_v_dim = (config.linear_num_value_heads * config.linear_value_head_dim) as u128;
    let linear_qkv_dim = linear_k_dim * 2 + linear_v_dim;
    let q_dim = (config.n_heads * config.head_dim) as u128;
    let kv_dim = (config.n_kv_heads * config.head_dim) as u128;

    // Mirrors PrefillBatchScratch::new. Raw i32 arrays are accounted in the
    // separate byte terms; every other listed allocation is F32.
    let per_row_f32 = 3 * dim
        + linear_qkv_dim
        + 7 * linear_v_dim
        + 2 * linear_k_dim
        + 2 * linear_heads
        + 3 * hidden
        + 6 * q_dim
        + 2 * kv_dim
        + experts
        + 1
        + 3 * shared_intermediate
        + k_top
        + 3 * k_top * moe_intermediate
        + k_top * dim;
    let mut scratch_bytes = n * per_row_f32 * 4 + n * 2 * 4;
    scratch_bytes += n * k_top * 4; // routed indices, i32

    let m_total = if config.num_experts > 0 {
        hipfire_runtime::moe::grouped::grouped_m_total_max(
            scratch_rows,
            config.num_experts_per_tok,
            config.num_experts,
        )
        .map_err(|error| CalibError::InvalidOptions(error.to_string()))? as u128
    } else {
        0
    };
    if config.num_experts > 0 {
        let routed_slots = n * k_top;
        scratch_bytes += experts * 4
            + (experts + 1) * 4
            + m_total * 4
            + routed_slots * 4
            + (m_total / 16) * 4
            + m_total * (2 * moe_intermediate) * 4
            + m_total * dim * 4;
    }
    scratch_bytes += n * linear_heads * linear_head_dim * linear_head_dim;
    scratch_bytes += n * linear_heads * linear_head_dim * 4;

    let delta_state_bytes = linear_heads
        * (config.linear_key_head_dim as u128)
        * (config.linear_key_head_dim as u128)
        * 4
        + (2 * linear_k_dim + linear_v_dim)
            * (config.conv_kernel_dim.saturating_sub(1) as u128)
            * 4
        + kv_dim * 8;
    let full_attention_state_bytes = (job.samples.context_len() as u128) * kv_dim * 8;
    let state_bytes_per_sequence = delta_state_bytes.max(full_attention_state_bytes);
    let active_sequences = geometry.sequence_batch.min(job.samples.samples().len()) as u128;
    let active_state_bytes = state_bytes_per_sequence * active_sequences;
    let to_u64 = |name: &str, value: u128| {
        u64::try_from(value).map_err(|_| {
            CalibError::InvalidOptions(format!("Qwen3.5 {name} byte estimate overflows u64"))
        })
    };
    Ok(CalibrationResourceEstimate {
        scratch_bytes: to_u64("scratch", scratch_bytes)?,
        state_bytes_per_sequence: to_u64("per-sequence state", state_bytes_per_sequence)?,
        active_state_bytes: to_u64("active state", active_state_bytes)?,
        details: serde_json::json!({
            "num_experts": config.num_experts,
            "k_top": config.num_experts_per_tok,
            "active_sequences": to_u64("active sequence count", active_sequences)?,
            "scratch_rows": scratch_rows,
            "delta_state_bytes_per_sequence": to_u64("DeltaNet state", delta_state_bytes)?,
            "full_attention_state_bytes_per_sequence": to_u64("attention state", full_attention_state_bytes)?,
            "grouped_moe_padded_rows_bound": to_u64("grouped MoE rows", m_total)?,
        }),
    })
}

impl CalibrationFamilyAdapter for Qwen35CalibrationAdapter {
    fn family(&self) -> &'static str {
        "qwen3.5"
    }

    fn adapter_version(&self) -> &'static str {
        "qwen3.5-stream-v2"
    }

    fn resource_estimate(
        &self,
        _model: &ModelInspection,
        job: &CalibrationJob,
        geometry: MicrobatchGeometry,
    ) -> Result<Option<CalibrationResourceEstimate>, CalibError> {
        Ok(Some(qwen35_resource_estimate(
            self.config()?,
            job,
            geometry,
        )?))
    }

    fn effective_precision(&self, gpu: &Gpu) -> serde_json::Value {
        let bf16_execution = if gpu.arch.starts_with("gfx11") || gpu.arch.starts_with("gfx12") {
            "bf16-native"
        } else {
            "f16-fallback"
        };
        serde_json::json!({
            "boundary": "f32",
            "source_dtypes": self.source_dtypes,
            "bf16_weight_execution": bf16_execution,
            "gpu_arch": gpu.arch,
        })
    }

    fn inspect(&mut self, source: &dyn ModelSource) -> Result<ModelInspection, CalibError> {
        let inspection = inspect_qwen35_stream_source(source)?;
        self.source_dtypes = inspection
            .tensor_requests
            .iter()
            .filter_map(|request| source.tensor_info(&request.source_name))
            .map(|info| info.dtype.clone())
            .collect();
        self.source_dtypes.sort();
        self.source_dtypes.dedup();
        self.config = Some(config_from_safetensors(source).ok_or_else(|| {
            CalibError::InvalidSourcePlan("could not retain Qwen3.5 source config".into())
        })?);
        Ok(inspection)
    }

    fn capture_plan(
        &self,
        model: &ModelInspection,
        job: &CalibrationJob,
    ) -> Result<CaptureRegistry, CalibError> {
        let config = self.config()?;
        if model.num_layers != config.n_layers || model.hidden_width != config.dim {
            return Err(CalibError::InvalidSourcePlan(
                "Qwen3.5 inspection geometry changed before capture planning".into(),
            ));
        }
        qwen35_capture_registry(config, job.options.expert_quota)
    }

    fn cask_metadata(
        &self,
        model: &ModelInspection,
        job: &CalibrationJob,
    ) -> Result<Option<TriAttnPackageMetadata>, CalibError> {
        let config = self.config()?;
        let rotary_dim = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
        if rotary_dim == 0 || rotary_dim > config.head_dim || rotary_dim % 2 != 0 {
            return Err(CalibError::InvalidSourcePlan(format!(
                "Qwen3.5 has unsupported CASK rotary dimension {rotary_dim}"
            )));
        }
        let layers = config
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == LayerType::FullAttention)
            .map(|(layer, _)| TriAttnLayerRecord {
                physical_layer: layer as u32,
                attention_kind: TriAttnAttentionKind::Full,
                q_heads: config.n_heads as u32,
                kv_heads: config.n_kv_heads as u32,
                head_dim: config.head_dim as u32,
                rotary_dim: rotary_dim as u32,
                rope_theta: config.rope_theta,
                rope_convention: TriAttnRopeConvention::HalfSplit,
                context_policy: TriAttnContextPolicy::Full,
                sliding_window: None,
                kv_producer: None,
                center_tensor: format!("triattn.layers.{layer}.centers"),
                center_offset: 0,
                center_count: (config.n_heads * (config.head_dim / 2)) as u64,
                sample_count: 1,
            })
            .collect::<Vec<_>>();
        if layers.is_empty() {
            return Err(CalibError::InvalidSourcePlan(
                "Qwen3.5 model has no CASK-eligible full-attention layers".into(),
            ));
        }
        Ok(Some(TriAttnPackageMetadata {
            artifact_kind: TRIATTN_ARTIFACT_KIND.to_string(),
            package_schema: TRIATTN_HFQM_SCHEMA.to_string(),
            model_arch_id: model.arch_id,
            model_layers: model.num_layers as u32,
            model_fingerprint: job.source_fingerprint.clone(),
            corpus_fingerprint: job.corpus_fingerprint.clone(),
            adapter: self.adapter_version().to_string(),
            engine: "hipfire-cask-layer-stream-v1".to_string(),
            layers,
        }))
    }

    fn load_embedding(
        &mut self,
        reader: &mut PlannedTensorReader<'_, '_, '_>,
        gpu: &mut Gpu,
        model: &ModelInspection,
        job: &CalibrationJob,
    ) -> Result<Box<dyn CalibrationEmbedding>, CalibError> {
        let view = reader.read("embedding")?;
        validate_shape(
            view.info,
            &[model.vocab_size, model.hidden_width],
            "embedding",
        )?;
        let values = source_payload_f32(view.info.dtype.as_str(), view.bytes)?;
        if values.len() != model.vocab_size * model.hidden_width {
            return Err(CalibError::InvalidSourcePlan(
                "embedding payload length does not match its declared shape".into(),
            ));
        }
        let embedding = gpu
            .upload_f32(&values, &[model.vocab_size, model.hidden_width])
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        let token_ids = match gpu.alloc_tensor(&[job.options.max_rows], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(embedding);
                return Err(CalibError::Runtime(error.to_string()));
            }
        };
        let output =
            match gpu.alloc_tensor(&[job.options.max_rows * model.hidden_width], DType::F32) {
                Ok(tensor) => tensor,
                Err(error) => {
                    let _ = gpu.free_tensor(token_ids);
                    let _ = gpu.free_tensor(embedding);
                    return Err(CalibError::Runtime(error.to_string()));
                }
            };
        Ok(Box::new(Qwen35CalibrationEmbedding {
            embedding: Some(embedding),
            token_ids: Some(token_ids),
            output: Some(output),
            vocab_size: model.vocab_size,
            dim: model.hidden_width,
            max_rows: job.options.max_rows,
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
        Ok(Box::new(Qwen35StreamedCalibrationLayer::load(
            reader,
            gpu,
            self.config()?,
            layer,
            job,
        )?))
    }

    fn load_finalizer(
        &mut self,
        reader: &mut PlannedTensorReader<'_, '_, '_>,
        gpu: &mut Gpu,
        model: &ModelInspection,
        job: &CalibrationJob,
    ) -> Result<Box<dyn CalibrationFinalizer>, CalibError> {
        // Qwen3.5 uses GemmaRMSNorm semantics for the final norm too: the
        // safetensors value is an offset and the effective scale is 1 + w.
        // Match the resident loader and the per-layer streamed norm loads.
        let norm = load_f32_tensor(reader, gpu, "final_norm", model.hidden_width, true)?;
        let lm_head =
            match load_matrix(reader, gpu, "lm_head", model.vocab_size, model.hidden_width) {
                Ok(weight) => weight,
                Err(error) => {
                    let _ = gpu.free_tensor(norm);
                    return Err(error);
                }
            };
        Ok(Box::new(RmsNormLmHeadFinalizer::new(
            gpu,
            norm,
            lm_head,
            model.hidden_width,
            model.vocab_size,
            self.config()?.norm_eps,
            job.options.max_rows,
        )?))
    }
}

struct StreamedSessionState {
    kv: KvCache,
    delta: DeltaNetState,
}

/// Resident state for one source-loaded Qwen layer. Session state is allocated
/// only for the scheduler's current sequence group and reused across its time
/// tiles, bounding recurrent/KV memory by `sequence_batch` rather than the
/// total calibration corpus size.
pub struct Qwen35StreamedCalibrationLayer {
    logical_layer: usize,
    total_layers: usize,
    config: Qwen35Config,
    weights: Option<LayerWeights>,
    pbs: Option<PrefillBatchScratch>,
    dummy_logits: Option<GpuTensor>,
    sample_lengths: Vec<usize>,
    active_sequence_start: Option<usize>,
    states: Vec<StreamedSessionState>,
    started: Vec<bool>,
    capture_registry: Option<Arc<CaptureRegistry>>,
    collector: Option<Arc<CalibCollector>>,
    expert_capture: Option<GroupedMoeCalibrationCapture>,
}

impl Qwen35StreamedCalibrationLayer {
    pub fn load(
        reader: &mut PlannedTensorReader<'_, '_, '_>,
        gpu: &mut Gpu,
        full_config: &Qwen35Config,
        layer: usize,
        job: &CalibrationJob,
    ) -> Result<Self, CalibError> {
        if layer >= full_config.n_layers {
            return Err(CalibError::InvalidSourcePlan(format!(
                "Qwen3.5 calibration layer {layer} is outside 0..{}",
                full_config.n_layers
            )));
        }
        let weights = load_qwen35_streamed_layer(reader, gpu, full_config, layer)?;
        let mut config = full_config.clone();
        config.n_layers = 1;
        config.layer_types = vec![full_config.layer_types[layer]];
        config.paged_experts = false;
        let pbs = match PrefillBatchScratch::new(gpu, &config, job.options.max_rows) {
            Ok(pbs) => pbs,
            Err(error) => {
                free_qwen35_streamed_layer(gpu, weights);
                return Err(CalibError::Runtime(error.to_string()));
            }
        };
        let dummy_logits = match gpu.zeros(&[1], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                pbs.free_gpu(gpu);
                free_qwen35_streamed_layer(gpu, weights);
                return Err(CalibError::Runtime(error.to_string()));
            }
        };
        Ok(Self {
            logical_layer: layer,
            total_layers: full_config.n_layers,
            config,
            weights: Some(weights),
            pbs: Some(pbs),
            dummy_logits: Some(dummy_logits),
            sample_lengths: job
                .samples
                .samples()
                .iter()
                .map(|sample| sample.tokens.len())
                .collect(),
            active_sequence_start: None,
            states: Vec::new(),
            started: Vec::new(),
            capture_registry: None,
            collector: None,
            expert_capture: None,
        })
    }

    fn prepare_capture(&mut self, registry: &CaptureRegistry) -> Result<(), CalibError> {
        if registry.is_empty() {
            return Ok(());
        }
        if let Some(existing) = &self.capture_registry {
            if existing.as_ref() != registry {
                return Err(CalibError::InvalidCapture(
                    "capture registry changed while a streamed layer was resident".into(),
                ));
            }
            return Ok(());
        }
        let registry = Arc::new(registry.clone());
        let collector = Arc::new(CalibCollector::new());
        let expert_capture = if self.config.num_experts > 0 {
            let quota = registry
                .get(CaptureId::new(
                    self.logical_layer,
                    ProjectionRole::GateUpInput,
                    Some(0),
                ))
                .and_then(|descriptor| descriptor.expert_quota)
                .ok_or_else(|| {
                    CalibError::InvalidCapture(format!(
                        "layer {} has no routed expert quota descriptor",
                        self.logical_layer
                    ))
                })?;
            let telemetry = ExpertTelemetry::new(
                self.total_layers,
                self.config.num_experts,
                self.config.num_experts_per_tok,
                quota,
                4096,
            )?;
            Some(GroupedMoeCalibrationCapture::with_collector(
                Arc::clone(&registry),
                telemetry,
                Arc::clone(&collector),
            )?)
        } else {
            None
        };
        self.capture_registry = Some(registry);
        self.collector = Some(collector);
        self.expert_capture = expert_capture;
        Ok(())
    }

    fn release_states(&mut self, gpu: &mut Gpu) {
        for state in self.states.drain(..) {
            state.kv.free_gpu(gpu);
            state.delta.free_gpu(gpu);
        }
        self.started.clear();
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
                    "sequence group width changed across time tiles".into(),
                ));
            }
            return Ok(());
        }
        self.release_states(gpu);
        if batch.sequence_start >= batch.sequence_end
            || batch.sequence_end > self.sample_lengths.len()
        {
            return Err(CalibError::InvalidOptions(format!(
                "invalid calibration sequence group {}..{} for {} samples",
                batch.sequence_start,
                batch.sequence_end,
                self.sample_lengths.len()
            )));
        }
        let is_full_attention = self.config.layer_types[0] == LayerType::FullAttention;
        let mut states = Vec::with_capacity(batch.sequence_end - batch.sequence_start);
        for &sample_len in &self.sample_lengths[batch.sequence_start..batch.sequence_end] {
            let kv_len = if is_full_attention {
                sample_len.max(1)
            } else {
                1
            };
            let kv = match KvCache::new_gpu(
                gpu,
                1,
                self.config.n_kv_heads,
                self.config.head_dim,
                kv_len,
            ) {
                Ok(kv) => kv,
                Err(error) => {
                    for state in states {
                        let state: StreamedSessionState = state;
                        state.kv.free_gpu(gpu);
                        state.delta.free_gpu(gpu);
                    }
                    return Err(CalibError::Runtime(error.to_string()));
                }
            };
            let delta = match DeltaNetState::new_with_quant(gpu, &self.config, StateQuant::FP32) {
                Ok(delta) => delta,
                Err(error) => {
                    kv.free_gpu(gpu);
                    for state in states {
                        let state: StreamedSessionState = state;
                        state.kv.free_gpu(gpu);
                        state.delta.free_gpu(gpu);
                    }
                    return Err(CalibError::Runtime(error.to_string()));
                }
            };
            states.push(StreamedSessionState { kv, delta });
        }
        self.started = vec![false; states.len()];
        self.states = states;
        self.active_sequence_start = Some(batch.sequence_start);
        Ok(())
    }

    fn validate_and_group_rows(
        &mut self,
        batch: &LayerMicrobatch,
    ) -> Result<(Vec<Vec<u32>>, Vec<usize>, Vec<usize>), CalibError> {
        if batch.rows.len() != batch.boundary_rows.len() || batch.rows.is_empty() {
            return Err(CalibError::InvalidOptions(
                "calibration layer received empty or mismatched scheduler rows".into(),
            ));
        }
        let sessions = batch.sequence_end - batch.sequence_start;
        let mut tokens = vec![Vec::new(); sessions];
        let mut starts = vec![usize::MAX; sessions];
        for row in &batch.rows {
            if row.sample_index < batch.sequence_start || row.sample_index >= batch.sequence_end {
                return Err(CalibError::InvalidOptions(format!(
                    "sample {} is outside scheduler group {}..{}",
                    row.sample_index, batch.sequence_start, batch.sequence_end
                )));
            }
            let local = row.sample_index - batch.sequence_start;
            if row.position >= self.sample_lengths[row.sample_index] {
                return Err(CalibError::InvalidOptions(format!(
                    "sample {} position {} exceeds length {}",
                    row.sample_index, row.position, self.sample_lengths[row.sample_index]
                )));
            }
            if starts[local] == usize::MAX {
                starts[local] = row.position;
                if self.started[local] {
                    if row.reset_state {
                        return Err(CalibError::InvalidOptions(format!(
                            "sample {} reset recurrent state more than once",
                            row.sample_index
                        )));
                    }
                } else if !row.reset_state || row.position != 0 {
                    return Err(CalibError::InvalidOptions(format!(
                        "sample {} did not begin at a reset row",
                        row.sample_index
                    )));
                } else {
                    self.started[local] = true;
                }
            }
            let expected = starts[local] + tokens[local].len();
            if row.position != expected {
                return Err(CalibError::InvalidOptions(format!(
                    "sample {} positions are not contiguous: expected {expected}, got {}",
                    row.sample_index, row.position
                )));
            }
            tokens[local].push(row.token);
        }
        let mut compact_tokens = Vec::new();
        let mut compact_starts = Vec::new();
        let mut state_indices = Vec::new();
        for (state_index, (tokens, start)) in tokens.into_iter().zip(starts).enumerate() {
            if !tokens.is_empty() {
                compact_tokens.push(tokens);
                compact_starts.push(start);
                state_indices.push(state_index);
            }
        }
        Ok((compact_tokens, compact_starts, state_indices))
    }
}

impl CalibrationLayer for Qwen35StreamedCalibrationLayer {
    fn execute(
        &mut self,
        gpu: &mut Gpu,
        batch: &LayerMicrobatch,
        input_f32: &[f32],
        output_f32: &mut [f32],
        capture: &CaptureRegistry,
    ) -> Result<(), CalibError> {
        let expected = batch
            .rows
            .len()
            .checked_mul(self.config.dim)
            .ok_or_else(|| CalibError::InvalidOptions("boundary batch size overflow".into()))?;
        if input_f32.len() != expected || output_f32.len() != expected {
            return Err(CalibError::InvalidOptions(format!(
                "Qwen layer boundary has input/output lengths {}/{}, expected {expected}",
                input_f32.len(),
                output_f32.len()
            )));
        }
        self.prepare_sequence_group(gpu, batch)?;
        self.prepare_capture(capture)?;
        let (tokens, starts, state_indices) = self.validate_and_group_rows(batch)?;
        let inputs = tokens
            .iter()
            .zip(&starts)
            .map(|(tokens, &start_pos)| DensePrefillSessionBatchInput { tokens, start_pos })
            .collect::<Vec<_>>();
        let pbs = self.pbs.as_ref().ok_or_else(|| {
            CalibError::Runtime("Qwen calibration layer was already freed".into())
        })?;
        let plan = build_calibration_session_batch_execution_plan(&inputs, pbs.max_batch)
            .map_err(CalibError::InvalidOptions)?;
        if plan.total_rows != batch.rows.len() {
            return Err(CalibError::InvalidOptions(format!(
                "Qwen session plan produced {} rows for scheduler batch of {}",
                plan.total_rows,
                batch.rows.len()
            )));
        }
        let route_shape = expected_dense_prefill_session_state_route_shape(&self.config);
        let pointer_plan =
            dense_prefill_session_batch_pointer_table_plan(&plan, route_shape, inputs.len());
        let (flat_tokens, positions) =
            dense_prefill_session_batch_prefix_tokens_positions(&pointer_plan)
                .map_err(CalibError::InvalidOptions)?;
        if flat_tokens
            .iter()
            .zip(&positions)
            .zip(&batch.rows)
            .any(|((&token, &position), row)| token != row.token || position != row.position)
        {
            return Err(CalibError::InvalidOptions(
                "Qwen session plan row order differs from boundary row order".into(),
            ));
        }
        upload_prefill_batch_inputs_with_positions(gpu, pbs, &flat_tokens, &positions)
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        gpu.hip
            .memcpy_htod(&pbs.x_batch.buf, f32_slice_as_bytes(input_f32))
            .map_err(|error| CalibError::Runtime(error.to_string()))?;

        let dummy_logits = self.dummy_logits.as_ref().ok_or_else(|| {
            CalibError::Runtime("Qwen calibration dummy logits were already freed".into())
        })?;
        let state_offset = batch.sequence_start;
        let routes = inputs
            .iter()
            .zip(&state_indices)
            .map(|(_, &state_index)| {
                let state = &self.states[state_index];
                DensePrefillSessionStateRoute {
                    kv: DensePrefillSessionKvStateRoute {
                        k_gpu: &state.kv.k_gpu,
                        v_gpu: &state.kv.v_gpu,
                        physical_cap: state.kv.physical_cap,
                        compact_offset: state.kv.compact_offset,
                    },
                    delta: DensePrefillSessionDeltaStateRoute {
                        s_matrices: &state.delta.s_matrices,
                        s_scales: &state.delta.s_scales,
                        conv_states: &state.delta.conv_states,
                        quant: state.delta.quant,
                    },
                    logits: dummy_logits,
                }
            })
            .collect::<Vec<_>>();
        debug_assert_eq!(state_offset, self.active_sequence_start.unwrap());
        let host_tables = dense_prefill_session_batch_host_pointer_tables(&pointer_plan, &routes)
            .map_err(CalibError::InvalidOptions)?;
        let device_tables = upload_dense_prefill_session_batch_pointer_tables(
            gpu,
            pointer_plan.shape,
            &host_tables,
        )
        .map_err(|error| CalibError::Runtime(error.to_string()))?;
        let max_ctx_len = positions
            .iter()
            .copied()
            .max()
            .map(|pos| pos + 1)
            .unwrap_or(1);
        let weights = self.weights.as_ref().ok_or_else(|| {
            CalibError::Runtime("Qwen calibration weights were already freed".into())
        })?;
        let dense_capture = self
            .collector
            .as_deref()
            .zip(self.capture_registry.as_deref());
        let result = if self.config.num_experts > 0 {
            forward_streamed_grouped_moe_layer_batch(
                gpu,
                weights,
                self.logical_layer,
                &self.config,
                pbs,
                &device_tables,
                route_shape,
                batch.rows.len(),
                inputs.len(),
                max_ctx_len,
                self.expert_capture.as_ref().map(|capture| {
                    capture as &dyn hipfire_dispatch::families::moe::MoePrefillCapture
                }),
                dense_capture,
            )
        } else {
            forward_streamed_dense_layer_batch(
                gpu,
                weights,
                self.logical_layer,
                &self.config,
                pbs,
                &device_tables,
                route_shape,
                batch.rows.len(),
                inputs.len(),
                max_ctx_len,
                dense_capture,
            )
        };
        device_tables.free_gpu(gpu);
        result.map_err(|error| CalibError::Runtime(error.to_string()))?;
        let live_output = pbs.x_batch.sub_offset(0, expected);
        let downloaded = gpu
            .download_f32(&live_output)
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        output_f32.copy_from_slice(&downloaded);
        Ok(())
    }

    fn write_capture_part(
        &mut self,
        gpu: &mut Gpu,
        path: &std::path::Path,
        arch_id: u32,
        metadata_json: &str,
    ) -> Result<LayerCapturePartSummary, CalibError> {
        let capture_is_intentionally_empty = self
            .capture_registry
            .as_ref()
            .is_some_and(|registry| capture_registry_is_cask_only(registry));
        let expert_telemetry = if let Some(capture) = self.expert_capture.take() {
            capture.finalize()?;
            let snapshot = capture
                .telemetry_snapshot()
                .layer_snapshot(self.logical_layer)?;
            drop(capture);
            Some(snapshot)
        } else if self.config.num_experts > 0 {
            return Err(CalibError::InvalidCapture(format!(
                "layer {} has no routed capture state",
                self.logical_layer
            )));
        } else {
            None
        };
        let collector = self.collector.take().ok_or_else(|| {
            CalibError::InvalidCapture(format!(
                "layer {} has no calibration collector",
                self.logical_layer
            ))
        })?;
        let descriptors = collector.tensor_descriptors();
        if descriptors.is_empty() && !capture_is_intentionally_empty {
            collector.free_gpu(gpu);
            return Err(CalibError::InvalidCapture(format!(
                "layer {} captured no calibration tensors",
                self.logical_layer
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
        if let Some(capture) = self.expert_capture.take() {
            capture.finalize()?;
        }
        if let Some(collector) = self.collector.take() {
            collector.free_gpu(gpu);
        }
        self.capture_registry = None;
        self.release_states(gpu);
        if let Some(tensor) = self.dummy_logits.take() {
            gpu.free_tensor(tensor)
                .map_err(|error| CalibError::Runtime(error.to_string()))?;
        }
        if let Some(pbs) = self.pbs.take() {
            pbs.free_gpu(gpu);
        }
        if let Some(weights) = self.weights.take() {
            free_qwen35_streamed_layer(gpu, weights);
        }
        Ok(())
    }
}

fn f32_slice_as_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn i32_slice_as_bytes(values: &[i32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn load_streamed_moe_ffn(
    reader: &mut PlannedTensorReader<'_, '_, '_>,
    gpu: &mut Gpu,
    config: &Qwen35Config,
    layer: usize,
    prefix: &str,
) -> Result<MoeFfnWeights, CalibError> {
    let dim = config.dim;
    let experts = config.num_experts;
    let intermediate = config.moe_intermediate_size;
    let shared_intermediate = config.shared_expert_intermediate_size;
    let mut pending = PendingGpuLoads::default();
    let result = (|| {
        let router = pending.push_weight(load_matrix(
            reader,
            gpu,
            &format!("{prefix}.mlp.gate.weight"),
            experts,
            dim,
        )?);
        let shared_gate = pending.push_weight(load_matrix(
            reader,
            gpu,
            &format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
            shared_intermediate,
            dim,
        )?);
        let shared_up = pending.push_weight(load_matrix(
            reader,
            gpu,
            &format!("{prefix}.mlp.shared_expert.up_proj.weight"),
            shared_intermediate,
            dim,
        )?);
        let shared_down = pending.push_weight(load_matrix(
            reader,
            gpu,
            &format!("{prefix}.mlp.shared_expert.down_proj.weight"),
            dim,
            shared_intermediate,
        )?);
        let shared_expert_gate = pending.push_weight(load_matrix(
            reader,
            gpu,
            &format!("{prefix}.mlp.shared_expert_gate.weight"),
            1,
            dim,
        )?);
        let gate_up_storage = pending.push_tensor(load_stacked_matrix(
            reader,
            gpu,
            &format!("{prefix}.mlp.experts.gate_up_proj"),
            experts,
            2 * intermediate,
            dim,
        )?);
        let down_storage = pending.push_tensor(load_stacked_matrix(
            reader,
            gpu,
            &format!("{prefix}.mlp.experts.down_proj"),
            experts,
            dim,
            intermediate,
        )?);
        if pending.tensor(gate_up_storage).dtype != pending.tensor(down_storage).dtype {
            return Err(CalibError::InvalidSourcePlan(format!(
                "layer {layer} routed gate/up dtype {:?} differs from down dtype {:?}",
                pending.tensor(gate_up_storage).dtype,
                pending.tensor(down_storage).dtype
            )));
        }
        let dtype = pending.tensor(gate_up_storage).dtype;
        let gate_up_stride = 2 * intermediate * dim * dtype.size();
        let down_stride = dim * intermediate * dtype.size();
        let mut expert_weights = Vec::with_capacity(experts);
        let mut gate_up_ptrs = Vec::with_capacity(experts);
        let mut down_ptrs = Vec::with_capacity(experts);
        for expert in 0..experts {
            let gate_up_ptr = (pending.tensor(gate_up_storage).buf.as_ptr() as usize
                + expert * gate_up_stride) as *mut c_void;
            let down_ptr = (pending.tensor(down_storage).buf.as_ptr() as usize
                + expert * down_stride) as *mut c_void;
            gate_up_ptrs.push(gate_up_ptr as u64);
            down_ptrs.push(down_ptr as u64);
            expert_weights.push(ExpertWeights {
                gate_up: alias_weight(gate_up_ptr, gate_up_stride, dtype, 2 * intermediate, dim),
                down: alias_weight(down_ptr, down_stride, dtype, dim, intermediate),
            });
        }
        let expert_gate_up_ptrs = pending.push_tensor(upload_pointer_table(gpu, &gate_up_ptrs)?);
        let expert_down_ptrs = pending.push_tensor(upload_pointer_table(gpu, &down_ptrs)?);
        Ok(MoeFfnWeights {
            router: pending.take_weight(router),
            experts: expert_weights,
            shared_expert: SharedExpertWeights {
                gate: pending.take_weight(shared_gate),
                up: pending.take_weight(shared_up),
                down: pending.take_weight(shared_down),
            },
            shared_expert_gate: pending.take_weight(shared_expert_gate),
            expert_down_awq_ptrs: None,
            expert_gate_up_awq_ptrs: None,
            expert_gate_up_ptrs: pending.take_tensor(expert_gate_up_ptrs),
            expert_down_ptrs: pending.take_tensor(expert_down_ptrs),
            layer_idx: layer as u16,
            expert_shape: None,
            expert_gate_up_dtype: Some(dtype),
            expert_down_dtype: Some(dtype),
            expert_gate_up_dtypes: vec![dtype; experts],
            expert_down_dtypes: vec![dtype; experts],
            paro_shared: None,
            raw_expert_storage: Some(RawExpertStorage {
                gate_up: pending.take_tensor(gate_up_storage),
                down: pending.take_tensor(down_storage),
            }),
        })
    })();
    if result.is_err() {
        pending.free(gpu);
    }
    result
}

fn load_stacked_matrix(
    reader: &mut PlannedTensorReader<'_, '_, '_>,
    gpu: &Gpu,
    logical_name: &str,
    experts: usize,
    m: usize,
    k: usize,
) -> Result<GpuTensor, CalibError> {
    load_source_tensor(reader, gpu, logical_name, &[experts, m, k])
}

fn alias_weight(
    pointer: *mut c_void,
    bytes: usize,
    dtype: DType,
    m: usize,
    k: usize,
) -> WeightTensor {
    WeightTensor {
        buf: GpuTensor {
            buf: unsafe { DeviceBuffer::from_raw(pointer, bytes) },
            shape: vec![m, k],
            dtype,
        },
        gpu_dtype: dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    }
}

fn upload_pointer_table(gpu: &Gpu, pointers: &[u64]) -> Result<GpuTensor, CalibError> {
    let bytes: Vec<u8> = pointers
        .iter()
        .flat_map(|pointer| pointer.to_ne_bytes())
        .collect();
    gpu.upload_raw(&bytes, &[bytes.len()])
        .map_err(|error| CalibError::Runtime(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_model::{QuantConfig, TensorInfo, TensorStorageLocation};
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    struct FakeSource {
        root: PathBuf,
        metadata: String,
        infos: BTreeMap<String, TensorInfo>,
    }

    impl FakeSource {
        fn dense() -> Self {
            let layer_types = (0..4)
                .map(|layer| {
                    if layer % 4 == 3 {
                        serde_json::Value::String("full_attention".into())
                    } else {
                        serde_json::Value::String("linear_attention".into())
                    }
                })
                .collect::<Vec<_>>();
            let config = serde_json::json!({
                "model_type": "qwen3_5",
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "vocab_size": 248320,
                "layer_types": layer_types,
            });
            let metadata = serde_json::json!({"config": config}).to_string();
            let mut source = Self {
                root: PathBuf::from("/dense"),
                metadata,
                infos: BTreeMap::new(),
            };
            let config = config_from_safetensors(&source).unwrap();
            for request in expected_requests_without_lookup(&config) {
                let source_name = qwen35_source_candidates(&request)
                    .into_iter()
                    .next()
                    .unwrap();
                source.infos.insert(
                    source_name.clone(),
                    TensorInfo {
                        name: source_name,
                        dtype: "BF16".into(),
                        shape: vec![1],
                        quant_type: 0xff,
                        data_offset: source.infos.len() * 2,
                        data_size: 2,
                    },
                );
            }
            source
        }

        fn a17b() -> Self {
            let layer_types = (0..60)
                .map(|layer| {
                    if layer % 4 == 3 {
                        serde_json::Value::String("full_attention".into())
                    } else {
                        serde_json::Value::String("linear_attention".into())
                    }
                })
                .collect::<Vec<_>>();
            let config = serde_json::json!({
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "hidden_size": 4096,
                    "num_hidden_layers": 60,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "vocab_size": 248320,
                    "num_experts": 512,
                    "num_experts_per_tok": 10,
                    "moe_intermediate_size": 1024,
                    "shared_expert_intermediate_size": 1024,
                    "layer_types": layer_types,
                }
            });
            let metadata = serde_json::json!({"config": config}).to_string();
            let mut source = Self {
                root: PathBuf::from("/a17b"),
                metadata,
                infos: BTreeMap::new(),
            };
            let config = config_from_safetensors(&source).unwrap();
            for request in expected_requests_without_lookup(&config) {
                let source_name = qwen35_source_candidates(&request)
                    .into_iter()
                    .next()
                    .unwrap();
                source.infos.insert(
                    source_name.clone(),
                    TensorInfo {
                        name: source_name,
                        dtype: "BF16".into(),
                        shape: vec![1],
                        quant_type: 0xff,
                        data_offset: source.infos.len() * 2,
                        data_size: 2,
                    },
                );
            }
            source
        }
    }

    impl ModelSource for FakeSource {
        fn metadata_json(&self) -> &str {
            &self.metadata
        }
        fn arch_id(&self) -> u32 {
            6
        }
        fn quant_config(&self) -> Option<&QuantConfig> {
            None
        }
        fn tensor_data(&self, _name: &str) -> Option<(&TensorInfo, &[u8])> {
            None
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.infos.get(name)
        }
        fn tensor_names(&self) -> Vec<&str> {
            self.infos.keys().map(String::as_str).collect()
        }
        fn path(&self) -> &Path {
            &self.root
        }
        fn tensor_storage(&self, name: &str) -> Option<TensorStorageLocation> {
            let info = self.infos.get(name)?;
            Some(TensorStorageLocation {
                path: self.root.join("model.safetensors"),
                byte_offset: info.data_offset as u64,
                byte_len: info.data_size as u64,
            })
        }
    }

    fn expected_requests_without_lookup(config: &Qwen35Config) -> Vec<String> {
        let mut names = vec![
            "embed_tokens.weight".into(),
            "norm.weight".into(),
            "lm_head.weight".into(),
        ];
        for (layer, layer_type) in config.layer_types.iter().enumerate() {
            let prefix = format!("layers.{layer}");
            let mut suffixes = vec!["input_layernorm.weight", "post_attention_layernorm.weight"];
            match layer_type {
                LayerType::LinearAttention => suffixes.extend([
                    "linear_attn.in_proj_qkv.weight",
                    "linear_attn.in_proj_z.weight",
                    "linear_attn.in_proj_a.weight",
                    "linear_attn.in_proj_b.weight",
                    "linear_attn.A_log",
                    "linear_attn.dt_bias",
                    "linear_attn.conv1d.weight",
                    "linear_attn.norm.weight",
                    "linear_attn.out_proj.weight",
                ]),
                LayerType::FullAttention => suffixes.extend([
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "self_attn.q_norm.weight",
                    "self_attn.k_norm.weight",
                ]),
            }
            if config.num_experts > 0 {
                suffixes.extend([
                    "mlp.gate.weight",
                    "mlp.experts.gate_up_proj",
                    "mlp.experts.down_proj",
                    "mlp.shared_expert.gate_proj.weight",
                    "mlp.shared_expert.up_proj.weight",
                    "mlp.shared_expert.down_proj.weight",
                    "mlp.shared_expert_gate.weight",
                ]);
            } else {
                suffixes.extend([
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                ]);
            }
            names.extend(
                suffixes
                    .into_iter()
                    .map(|suffix| format!("{prefix}.{suffix}")),
            );
        }
        names
    }

    #[test]
    fn a17b_plan_covers_all_60_layers_and_k10_geometry() {
        let source = FakeSource::a17b();
        let model = inspect_qwen35_stream_source(&source).unwrap();
        assert_eq!(model.hidden_width, 4096);
        assert_eq!(model.num_layers, 60);
        assert_eq!(model.tensor_requests.len(), 1038);
        assert_eq!(
            model
                .tensor_requests
                .iter()
                .filter(|request| request.owner == TensorOwner::Layer(59))
                .count(),
            15
        );
    }

    #[test]
    fn dense_plan_covers_dense_ffn_tensors_without_expert_geometry() {
        let source = FakeSource::dense();
        let model = inspect_qwen35_stream_source(&source).unwrap();
        assert_eq!(model.hidden_width, 1024);
        assert_eq!(model.num_layers, 4);
        assert_eq!(model.tensor_requests.len(), 56);
        assert_eq!(
            model
                .tensor_requests
                .iter()
                .filter(|request| request.owner == TensorOwner::Layer(3))
                .count(),
            11
        );
        assert!(model
            .tensor_requests
            .iter()
            .any(|request| request.logical_name == "layers.0.mlp.gate_proj.weight"));
        assert!(!model
            .tensor_requests
            .iter()
            .any(|request| request.logical_name.contains("mlp.experts")));
    }

    #[test]
    fn dense_plan_aliases_a_missing_lm_head_to_tied_embeddings() {
        let mut source = FakeSource::dense();
        source.infos.remove("lm_head.weight");
        let model = inspect_qwen35_stream_source(&source).unwrap();
        let lm_head = model
            .tensor_requests
            .iter()
            .find(|request| request.logical_name == "lm_head")
            .unwrap();
        assert_eq!(lm_head.alias_of.as_deref(), Some("embedding"));
        assert_eq!(
            lm_head.source_name,
            "model.language_model.embed_tokens.weight"
        );
    }

    #[test]
    fn plan_rejects_missing_tensor_and_non_source_dtype() {
        let mut source = FakeSource::a17b();
        source
            .infos
            .remove("model.language_model.layers.0.mlp.gate.weight");
        assert!(inspect_qwen35_stream_source(&source).is_err());
        let mut source = FakeSource::a17b();
        source
            .infos
            .get_mut("model.language_model.layers.0.mlp.gate.weight")
            .unwrap()
            .dtype = "I8".into();
        assert!(inspect_qwen35_stream_source(&source).is_err());
    }

    #[test]
    fn capture_registry_aliases_shared_inputs_and_declares_every_expert_role() {
        let source = FakeSource::a17b();
        let config = config_from_safetensors(&source).unwrap();
        let quota = ExpertCaptureQuota::default();
        let registry = qwen35_capture_registry(&config, quota).unwrap();
        assert_eq!(registry.len(), 60 * (4 + 512 * 2));

        let qkv = registry
            .resolve_output("model.language_model.layers.0.linear_attn.in_proj_qkv")
            .unwrap();
        assert_eq!(
            Some(qkv),
            registry.resolve_output("model.language_model.layers.0.linear_attn.in_proj_z")
        );
        assert_eq!(
            registry.get(qkv).unwrap().output_names.len(),
            4,
            "DeltaNet qkv/z/a/b must share one activation accumulator"
        );
        let expert = registry
            .get(CaptureId::new(59, ProjectionRole::DownInput, Some(511)))
            .unwrap();
        assert_eq!(expert.policy, CapturePolicy::ImatrixOnly);
        assert_eq!(expert.input_width, 1024);
        assert_eq!(expert.expert_quota, Some(quota));
    }

    #[test]
    fn dense_capture_registry_aliases_gate_up_and_has_no_expert_quota() {
        let source = FakeSource::dense();
        let config = config_from_safetensors(&source).unwrap();
        let registry = qwen35_capture_registry(&config, ExpertCaptureQuota::default()).unwrap();
        assert_eq!(registry.len(), 4 * 4);
        assert!(!capture_registry_is_cask_only(&registry));
        assert!(capture_registry_is_cask_only(&registry.clone().skip_all()));

        let gate = registry
            .resolve_output("model.language_model.layers.0.mlp.gate_proj")
            .unwrap();
        assert_eq!(
            Some(gate),
            registry.resolve_output("model.language_model.layers.0.mlp.up_proj")
        );
        let gate = registry.get(gate).unwrap();
        assert_eq!(gate.role, ProjectionRole::GateUpInput);
        assert_eq!(gate.input_width, 1024);
        assert_eq!(gate.policy, CapturePolicy::HessianAndImatrix);
        assert_eq!(gate.expert_quota, None);

        let down = registry
            .get(CaptureId::new(3, ProjectionRole::DownInput, None))
            .unwrap();
        assert_eq!(down.input_width, 3072);
        assert_eq!(down.policy, CapturePolicy::HessianAndImatrix);
        assert_eq!(down.expert_quota, None);
        assert!(registry
            .get(CaptureId::new(0, ProjectionRole::RouterInput, None))
            .is_none());
        assert!(registry
            .get(CaptureId::new(0, ProjectionRole::GateUpInput, Some(0)))
            .is_none());
    }
}
