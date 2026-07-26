// SPDX-License-Identifier: Apache-2.0
//! Gemma3 source-layout adapter for the family-neutral layer-stream engine.

use crate::config::{config_from_metadata_json, Gemma3Config};
use crate::forward::{forward_single_layer_residual_capture, Gemma3State};
use crate::weights::{Gemma3LayerWeights, Gemma3Weights};
use hipfire_model::{ModelSource, ARCH_ID_GEMMA3_TEXT};
use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_runtime::calibration::contracts::{
    CalibError, CalibrationJob, CaptureDescriptor, CaptureId, CapturePolicy, CaptureRegistry,
    ProjectionRole, SampleRow,
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
use hipfire_runtime::kv::KvQuantMode;
use hipfire_runtime::weights::{EmbeddingFormat, WeightTensor};
use std::sync::Arc;

const SOURCE_DTYPES: &[&str] = &["BF16", "F16", "F32"];

#[derive(Default)]
pub struct Gemma3CalibrationAdapter {
    config: Option<Gemma3Config>,
    source_dtypes: Vec<String>,
}

const GEMMA3_CALIBRATION_ARCH_IDS: &[u32] = &[ARCH_ID_GEMMA3_TEXT];

fn gemma3_calibration_adapter_factory() -> Box<dyn CalibrationFamilyAdapter> {
    Box::new(Gemma3CalibrationAdapter::default())
}

hipfire_runtime::register_calibration_adapter!(
    "gemma3",
    "gemma3-stream-v1",
    GEMMA3_CALIBRATION_ARCH_IDS,
    gemma3_calibration_adapter_factory
);

impl Gemma3CalibrationAdapter {
    fn config(&self) -> Result<&Gemma3Config, CalibError> {
        self.config.as_ref().ok_or_else(|| {
            CalibError::InvalidSourcePlan(
                "Gemma3 adapter must inspect the source before loading phases".into(),
            )
        })
    }
}

pub fn inspect_gemma3_stream_source(
    source: &dyn ModelSource,
) -> Result<ModelInspection, CalibError> {
    if source.arch_id() != ARCH_ID_GEMMA3_TEXT {
        return Err(CalibError::InvalidSourcePlan(format!(
            "Gemma3 text calibration requires arch {ARCH_ID_GEMMA3_TEXT}, got {}",
            source.arch_id()
        )));
    }
    let config = config_from_metadata_json(source.metadata_json()).ok_or_else(|| {
        CalibError::InvalidSourcePlan("could not parse Gemma3 source config".into())
    })?;
    if config.hidden_activation != "gelu_pytorch_tanh" {
        return Err(CalibError::InvalidSourcePlan(format!(
            "Gemma3 streamed calibration requires gelu_pytorch_tanh, got {}",
            config.hidden_activation
        )));
    }
    Ok(ModelInspection {
        family: "gemma3".into(),
        arch_id: source.arch_id(),
        hidden_width: config.hidden_size,
        vocab_size: config.vocab_size,
        num_layers: config.num_hidden_layers,
        tensor_requests: gemma3_tensor_requests(source, &config)?,
    })
}

pub fn gemma3_tensor_requests(
    source: &dyn ModelSource,
    config: &Gemma3Config,
) -> Result<Vec<TensorLoadRequest>, CalibError> {
    let mut requests = Vec::with_capacity(3 + config.num_hidden_layers * 13);
    push_required(
        source,
        &mut requests,
        "embedding",
        "model.embed_tokens.weight",
        TensorOwner::Persistent,
    )?;
    push_required(
        source,
        &mut requests,
        "final_norm",
        "model.norm.weight",
        TensorOwner::Persistent,
    )?;
    if config.tie_word_embeddings || source.tensor_info("lm_head.weight").is_none() {
        requests.push(TensorLoadRequest::alias(
            "lm_head",
            "model.embed_tokens.weight",
            TensorOwner::Persistent,
            "embedding",
        ));
    } else {
        push_required(
            source,
            &mut requests,
            "lm_head",
            "lm_head.weight",
            TensorOwner::Persistent,
        )?;
    }

    for layer in 0..config.num_hidden_layers {
        let owner = TensorOwner::Layer(layer);
        let prefix = format!("model.layers.{layer}");
        for suffix in [
            "input_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            let name = format!("{prefix}.{suffix}");
            push_required(source, &mut requests, &name, &name, owner)?;
        }
    }
    Ok(requests)
}

pub fn gemma3_capture_registry(config: &Gemma3Config) -> Result<CaptureRegistry, CalibError> {
    let mut registry = CaptureRegistry::default();
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        register_capture(
            &mut registry,
            layer,
            ProjectionRole::QueryInput,
            vec![
                format!("{prefix}.self_attn.q_proj"),
                format!("{prefix}.self_attn.k_proj"),
                format!("{prefix}.self_attn.v_proj"),
            ],
            config.hidden_size,
        )?;
        register_capture(
            &mut registry,
            layer,
            ProjectionRole::AttentionOutputInput,
            vec![format!("{prefix}.self_attn.o_proj")],
            config.num_attention_heads * config.head_dim,
        )?;
        register_capture(
            &mut registry,
            layer,
            ProjectionRole::DenseMlpInput,
            vec![
                format!("{prefix}.mlp.gate_proj"),
                format!("{prefix}.mlp.up_proj"),
            ],
            config.hidden_size,
        )?;
        register_capture(
            &mut registry,
            layer,
            ProjectionRole::DownInput,
            vec![format!("{prefix}.mlp.down_proj")],
            config.intermediate_size,
        )?;
    }
    Ok(registry)
}

fn register_capture(
    registry: &mut CaptureRegistry,
    layer: usize,
    role: ProjectionRole,
    output_names: Vec<String>,
    input_width: usize,
) -> Result<(), CalibError> {
    registry.register(CaptureDescriptor {
        id: CaptureId::new(layer, role, None),
        output_names,
        input_width,
        policy: CapturePolicy::HessianAndImatrix,
        layer,
        role,
        expert: None,
        expert_quota: None,
    })
}

fn push_required(
    source: &dyn ModelSource,
    requests: &mut Vec<TensorLoadRequest>,
    logical_name: &str,
    source_name: &str,
    owner: TensorOwner,
) -> Result<(), CalibError> {
    let info = source.tensor_info(source_name).ok_or_else(|| {
        CalibError::InvalidSourcePlan(format!("missing Gemma3 tensor {source_name}"))
    })?;
    if !SOURCE_DTYPES.contains(&info.dtype.as_str()) {
        return Err(CalibError::InvalidSourcePlan(format!(
            "Gemma3 source tensor {source_name} has unsupported dtype {}",
            info.dtype
        )));
    }
    requests.push(TensorLoadRequest::tensor(logical_name, source_name, owner));
    Ok(())
}

struct Gemma3CalibrationEmbedding {
    embedding: Option<GpuTensor>,
    token_ids: Option<GpuTensor>,
    output: Option<GpuTensor>,
    vocab_size: usize,
    dim: usize,
    max_rows: usize,
    scale: f32,
}

impl CalibrationEmbedding for Gemma3CalibrationEmbedding {
    fn execute(
        &mut self,
        gpu: &mut Gpu,
        rows: &[SampleRow],
        output_f32: &mut [f32],
    ) -> Result<(), CalibError> {
        if rows.is_empty() || rows.len() > self.max_rows {
            return Err(CalibError::InvalidOptions(format!(
                "Gemma3 embedding batch has {} rows, expected 1..={} rows",
                rows.len(),
                self.max_rows
            )));
        }
        if output_f32.len() != rows.len() * self.dim {
            return Err(CalibError::InvalidOptions(
                "Gemma3 embedding output does not match row count and hidden width".into(),
            ));
        }
        let tokens = rows
            .iter()
            .map(|row| {
                if row.token as usize >= self.vocab_size {
                    Err(CalibError::InvalidSamples(format!(
                        "token {} exceeds Gemma3 vocabulary {}",
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
        gpu.scale_f32(&live, self.scale)
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
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

impl CalibrationFamilyAdapter for Gemma3CalibrationAdapter {
    fn family(&self) -> &'static str {
        "gemma3"
    }

    fn adapter_version(&self) -> &'static str {
        "gemma3-stream-v1"
    }

    fn resource_estimate(
        &self,
        _model: &ModelInspection,
        job: &CalibrationJob,
        geometry: MicrobatchGeometry,
    ) -> Result<Option<CalibrationResourceEstimate>, CalibError> {
        Ok(Some(gemma3_resource_estimate(
            self.config()?,
            job,
            geometry,
        )?))
    }

    fn effective_precision(&self, gpu: &Gpu) -> serde_json::Value {
        let weight_execution = if gpu.arch.starts_with("gfx11") || gpu.arch.starts_with("gfx12") {
            "bf16-native"
        } else {
            "f16-fallback"
        };
        serde_json::json!({
            "boundary": "f32",
            "source_dtypes": self.source_dtypes,
            "bf16_weight_execution": weight_execution,
            "gpu_arch": gpu.arch,
        })
    }

    fn inspect(&mut self, source: &dyn ModelSource) -> Result<ModelInspection, CalibError> {
        let inspection = inspect_gemma3_stream_source(source)?;
        self.source_dtypes = inspection
            .tensor_requests
            .iter()
            .filter_map(|request| source.tensor_info(&request.source_name))
            .map(|info| info.dtype.clone())
            .collect();
        self.source_dtypes.sort();
        self.source_dtypes.dedup();
        self.config = Some(
            config_from_metadata_json(source.metadata_json()).ok_or_else(|| {
                CalibError::InvalidSourcePlan("could not retain Gemma3 source config".into())
            })?,
        );
        Ok(inspection)
    }

    fn capture_plan(
        &self,
        model: &ModelInspection,
        _job: &CalibrationJob,
    ) -> Result<CaptureRegistry, CalibError> {
        let config = self.config()?;
        if model.num_layers != config.num_hidden_layers || model.hidden_width != config.hidden_size
        {
            return Err(CalibError::InvalidSourcePlan(
                "Gemma3 inspection geometry changed before capture planning".into(),
            ));
        }
        gemma3_capture_registry(config)
    }

    fn load_embedding(
        &mut self,
        reader: &mut PlannedTensorReader<'_, '_, '_>,
        gpu: &mut Gpu,
        model: &ModelInspection,
        job: &CalibrationJob,
    ) -> Result<Box<dyn CalibrationEmbedding>, CalibError> {
        let view = reader.read("embedding")?;
        validate_source_shape(
            view.info,
            &[model.vocab_size, model.hidden_width],
            "embedding",
        )?;
        let values = source_payload_f32(view.info.dtype.as_str(), view.bytes)?;
        if values.len() != model.vocab_size * model.hidden_width {
            return Err(CalibError::InvalidSourcePlan(
                "Gemma3 embedding payload length does not match its shape".into(),
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
        Ok(Box::new(Gemma3CalibrationEmbedding {
            embedding: Some(embedding),
            token_ids: Some(token_ids),
            output: Some(output),
            vocab_size: model.vocab_size,
            dim: model.hidden_width,
            max_rows: job.options.max_rows,
            scale: self.config()?.embed_scale(),
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
        Ok(Box::new(Gemma3StreamedCalibrationLayer::load(
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
        let norm = load_source_f32_tensor(reader, gpu, "final_norm", model.hidden_width, true)?;
        let lm_head = match load_source_matrix(
            reader,
            gpu,
            "lm_head",
            model.vocab_size,
            model.hidden_width,
        ) {
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
            self.config()?.rms_norm_eps,
            job.options.max_rows,
        )?))
    }
}

fn gemma3_resource_estimate(
    config: &Gemma3Config,
    job: &CalibrationJob,
    geometry: MicrobatchGeometry,
) -> Result<CalibrationResourceEstimate, CalibError> {
    let q_dim = config.num_attention_heads as u128 * config.head_dim as u128;
    let kv_dim = config.num_key_value_heads as u128 * config.head_dim as u128;
    let scratch_values = 3 * config.hidden_size as u128
        + 2 * q_dim
        + 2 * kv_dim
        + 3 * config.intermediate_size as u128
        + 1;
    let scratch_per_sequence = scratch_values * 4;
    let context = job.samples.context_len() as u128;
    let kv_bytes_per_sequence = context * kv_dim * 8;
    let active_sequences = geometry.sequence_batch.min(job.samples.samples().len()) as u128;
    let scratch_bytes = scratch_per_sequence * active_sequences;
    let active_state_bytes = (scratch_per_sequence + kv_bytes_per_sequence) * active_sequences;
    let to_u64 = |label: &str, value: u128| {
        u64::try_from(value).map_err(|_| {
            CalibError::InvalidOptions(format!("Gemma3 {label} byte estimate overflows u64"))
        })
    };
    Ok(CalibrationResourceEstimate {
        scratch_bytes: to_u64("scratch", scratch_bytes)?,
        state_bytes_per_sequence: to_u64("per-sequence state", kv_bytes_per_sequence)?,
        active_state_bytes: to_u64("active state", active_state_bytes)?,
        details: serde_json::json!({
            "active_sequences": to_u64("active sequence count", active_sequences)?,
            "context": job.samples.context_len(),
            "q_dim": to_u64("q dimension", q_dim)?,
            "kv_dim": to_u64("kv dimension", kv_dim)?,
            "scratch_bytes_per_sequence": to_u64("per-sequence scratch", scratch_per_sequence)?,
            "full_attention_kv_bytes_per_sequence": to_u64("per-sequence KV", kv_bytes_per_sequence)?,
        }),
    })
}

pub struct Gemma3StreamedCalibrationLayer {
    logical_layer: usize,
    config: Gemma3Config,
    weights: Option<Gemma3Weights>,
    sample_lengths: Vec<usize>,
    active_sequence_start: Option<usize>,
    states: Vec<Gemma3State>,
    capture_registry: Option<Arc<CaptureRegistry>>,
    collector: Option<Arc<CalibCollector>>,
}

impl Gemma3StreamedCalibrationLayer {
    pub fn load(
        reader: &mut PlannedTensorReader<'_, '_, '_>,
        gpu: &mut Gpu,
        full_config: &Gemma3Config,
        layer: usize,
        job: &CalibrationJob,
    ) -> Result<Self, CalibError> {
        if layer >= full_config.num_hidden_layers {
            return Err(CalibError::InvalidSourcePlan(format!(
                "Gemma3 layer {layer} exceeds {} layers",
                full_config.num_hidden_layers
            )));
        }
        let layer_weights = load_gemma3_streamed_layer(reader, gpu, full_config, layer)?;
        let mut config = full_config.clone();
        config.num_hidden_layers = 1;
        config.vocab_size = 1;
        config.tie_word_embeddings = false;
        config.sliding_window_pattern = if full_config.is_global_layer(layer) {
            1
        } else {
            0
        };
        let weights = match wrap_streamed_layer(gpu, layer_weights) {
            Ok(weights) => weights,
            Err((error, layer_weights)) => {
                free_gemma3_layer(gpu, layer_weights);
                return Err(error);
            }
        };
        Ok(Self {
            logical_layer: layer,
            config,
            weights: Some(weights),
            sample_lengths: job
                .samples
                .samples()
                .iter()
                .map(|sample| sample.tokens.len())
                .collect(),
            active_sequence_start: None,
            states: Vec::new(),
            capture_registry: None,
            collector: None,
        })
    }

    fn prepare_capture(&mut self, registry: &CaptureRegistry) -> Result<(), CalibError> {
        if let Some(existing) = &self.capture_registry {
            if existing.as_ref() != registry {
                return Err(CalibError::InvalidCapture(
                    "capture registry changed while a Gemma3 layer was resident".into(),
                ));
            }
            return Ok(());
        }
        self.capture_registry = Some(Arc::new(registry.clone()));
        self.collector = Some(Arc::new(CalibCollector::new()));
        Ok(())
    }

    fn release_states(&mut self, gpu: &mut Gpu) {
        for state in self.states.drain(..) {
            state.free_gpu(gpu);
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
                    "Gemma3 sequence group width changed across time tiles".into(),
                ));
            }
            return Ok(());
        }
        self.release_states(gpu);
        if batch.sequence_start >= batch.sequence_end
            || batch.sequence_end > self.sample_lengths.len()
        {
            return Err(CalibError::InvalidOptions(format!(
                "invalid Gemma3 sequence group {}..{} for {} samples",
                batch.sequence_start,
                batch.sequence_end,
                self.sample_lengths.len()
            )));
        }
        let mut states = Vec::with_capacity(batch.sequence_end - batch.sequence_start);
        for &sample_len in &self.sample_lengths[batch.sequence_start..batch.sequence_end] {
            match Gemma3State::new_with_max_seq(
                gpu,
                &self.config,
                sample_len.max(1),
                KvQuantMode::Unquantized,
                4,
            ) {
                Ok(state) => states.push(state),
                Err(error) => {
                    for state in states {
                        state.free_gpu(gpu);
                    }
                    return Err(CalibError::Runtime(error.to_string()));
                }
            }
        }
        self.states = states;
        self.active_sequence_start = Some(batch.sequence_start);
        Ok(())
    }
}

impl CalibrationLayer for Gemma3StreamedCalibrationLayer {
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
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| CalibError::InvalidOptions("Gemma3 boundary size overflow".into()))?;
        if batch.rows.len() != batch.boundary_rows.len()
            || batch.rows.is_empty()
            || input_f32.len() != expected
            || output_f32.len() != expected
        {
            return Err(CalibError::InvalidOptions(format!(
                "Gemma3 layer boundary has input/output lengths {}/{}, expected {expected}",
                input_f32.len(),
                output_f32.len()
            )));
        }
        self.prepare_sequence_group(gpu, batch)?;
        self.prepare_capture(capture)?;
        let collector = Arc::clone(self.collector.as_ref().unwrap());
        let registry = Arc::clone(self.capture_registry.as_ref().unwrap());
        let weights = self.weights.as_ref().ok_or_else(|| {
            CalibError::Runtime("Gemma3 calibration weights were already freed".into())
        })?;
        for (row_index, row) in batch.rows.iter().enumerate() {
            if row.sample_index < batch.sequence_start || row.sample_index >= batch.sequence_end {
                return Err(CalibError::InvalidOptions(format!(
                    "sample {} is outside Gemma3 scheduler group {}..{}",
                    row.sample_index, batch.sequence_start, batch.sequence_end
                )));
            }
            let local = row.sample_index - batch.sequence_start;
            let state = &mut self.states[local];
            if row.reset_state {
                state.reset();
            }
            if state.next_pos != row.position {
                return Err(CalibError::InvalidOptions(format!(
                    "Gemma3 sample {} expected position {}, got {}",
                    row.sample_index, state.next_pos, row.position
                )));
            }
            let start = row_index * self.config.hidden_size;
            let end = start + self.config.hidden_size;
            let output = forward_single_layer_residual_capture(
                gpu,
                weights,
                &self.config,
                state,
                &input_f32[start..end],
                self.logical_layer,
                Some((collector.as_ref(), registry.as_ref())),
            )
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
            output_f32[start..end].copy_from_slice(&output);
        }
        Ok(())
    }

    fn write_capture_part(
        &mut self,
        gpu: &mut Gpu,
        path: &std::path::Path,
        arch_id: u32,
        metadata_json: &str,
    ) -> Result<LayerCapturePartSummary, CalibError> {
        let collector = self.collector.take().ok_or_else(|| {
            CalibError::InvalidCapture(format!(
                "Gemma3 layer {} has no calibration collector",
                self.logical_layer
            ))
        })?;
        let descriptors = collector.tensor_descriptors();
        if descriptors.is_empty() {
            collector.free_gpu(gpu);
            return Err(CalibError::InvalidCapture(format!(
                "Gemma3 layer {} captured no calibration tensors",
                self.logical_layer
            )));
        }
        let result = collector.write_streaming(gpu, path, arch_id, metadata_json, &[]);
        collector.free_gpu(gpu);
        self.capture_registry = None;
        Ok(LayerCapturePartSummary {
            descriptors,
            max_consistency: result.map_err(|error| CalibError::Runtime(error.to_string()))?,
            expert_telemetry: None,
        })
    }

    fn finish(&mut self, gpu: &mut Gpu) -> Result<(), CalibError> {
        if let Some(collector) = self.collector.take() {
            collector.free_gpu(gpu);
        }
        self.capture_registry = None;
        self.release_states(gpu);
        if let Some(weights) = self.weights.take() {
            weights.free_gpu(gpu);
        }
        Ok(())
    }
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
            free_weight(gpu, weight);
        }
    }
}

fn load_gemma3_streamed_layer(
    reader: &mut PlannedTensorReader<'_, '_, '_>,
    gpu: &mut Gpu,
    config: &Gemma3Config,
    layer: usize,
) -> Result<Gemma3LayerWeights, CalibError> {
    let prefix = format!("model.layers.{layer}");
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let mut pending = PendingGpuLoads::default();
    let result = (|| {
        let input_norm = pending.push_tensor(load_source_f32_tensor(
            reader,
            gpu,
            &format!("{prefix}.input_layernorm.weight"),
            config.hidden_size,
            true,
        )?);
        let q_norm = pending.push_tensor(load_source_f32_tensor(
            reader,
            gpu,
            &format!("{prefix}.self_attn.q_norm.weight"),
            config.head_dim,
            true,
        )?);
        let prescale = config.q_prescale();
        if (prescale - 1.0).abs() > 1.0e-6 {
            gpu.scale_f32(pending.tensor(q_norm), prescale)
                .map_err(|error| CalibError::Runtime(error.to_string()))?;
        }
        let k_norm = pending.push_tensor(load_source_f32_tensor(
            reader,
            gpu,
            &format!("{prefix}.self_attn.k_norm.weight"),
            config.head_dim,
            true,
        )?);
        let wq = pending.push_weight(load_source_matrix(
            reader,
            gpu,
            &format!("{prefix}.self_attn.q_proj.weight"),
            q_dim,
            config.hidden_size,
        )?);
        let wk = pending.push_weight(load_source_matrix(
            reader,
            gpu,
            &format!("{prefix}.self_attn.k_proj.weight"),
            kv_dim,
            config.hidden_size,
        )?);
        let wv = pending.push_weight(load_source_matrix(
            reader,
            gpu,
            &format!("{prefix}.self_attn.v_proj.weight"),
            kv_dim,
            config.hidden_size,
        )?);
        let wo = pending.push_weight(load_source_matrix(
            reader,
            gpu,
            &format!("{prefix}.self_attn.o_proj.weight"),
            config.hidden_size,
            q_dim,
        )?);
        let post_attn_norm = pending.push_tensor(load_source_f32_tensor(
            reader,
            gpu,
            &format!("{prefix}.post_attention_layernorm.weight"),
            config.hidden_size,
            true,
        )?);
        let pre_ffn_norm = pending.push_tensor(load_source_f32_tensor(
            reader,
            gpu,
            &format!("{prefix}.pre_feedforward_layernorm.weight"),
            config.hidden_size,
            true,
        )?);
        let post_ffn_norm = pending.push_tensor(load_source_f32_tensor(
            reader,
            gpu,
            &format!("{prefix}.post_feedforward_layernorm.weight"),
            config.hidden_size,
            true,
        )?);
        let w_gate = pending.push_weight(load_source_matrix(
            reader,
            gpu,
            &format!("{prefix}.mlp.gate_proj.weight"),
            config.intermediate_size,
            config.hidden_size,
        )?);
        let w_up = pending.push_weight(load_source_matrix(
            reader,
            gpu,
            &format!("{prefix}.mlp.up_proj.weight"),
            config.intermediate_size,
            config.hidden_size,
        )?);
        let w_down = pending.push_weight(load_source_matrix(
            reader,
            gpu,
            &format!("{prefix}.mlp.down_proj.weight"),
            config.hidden_size,
            config.intermediate_size,
        )?);
        Ok(Gemma3LayerWeights {
            input_norm: pending.take_tensor(input_norm),
            q_norm: pending.take_tensor(q_norm),
            k_norm: pending.take_tensor(k_norm),
            wq: pending.take_weight(wq),
            wk: pending.take_weight(wk),
            wv: pending.take_weight(wv),
            wo: pending.take_weight(wo),
            post_attn_norm: pending.take_tensor(post_attn_norm),
            pre_ffn_norm: pending.take_tensor(pre_ffn_norm),
            post_ffn_norm: pending.take_tensor(post_ffn_norm),
            w_gate: pending.take_weight(w_gate),
            w_up: pending.take_weight(w_up),
            w_down: pending.take_weight(w_down),
        })
    })();
    if result.is_err() {
        pending.free(gpu);
    }
    result
}

fn wrap_streamed_layer(
    gpu: &mut Gpu,
    layer: Gemma3LayerWeights,
) -> Result<Gemma3Weights, (CalibError, Gemma3LayerWeights)> {
    let token_embd = match gpu.zeros(&[1], DType::F32) {
        Ok(tensor) => tensor,
        Err(error) => return Err((CalibError::Runtime(error.to_string()), layer)),
    };
    let output_norm = match gpu.zeros(&[1], DType::F32) {
        Ok(tensor) => tensor,
        Err(error) => {
            let _ = gpu.free_tensor(token_embd);
            return Err((CalibError::Runtime(error.to_string()), layer));
        }
    };
    let output_buf = match gpu.zeros(&[1], DType::F32) {
        Ok(tensor) => tensor,
        Err(error) => {
            let _ = gpu.free_tensor(output_norm);
            let _ = gpu.free_tensor(token_embd);
            return Err((CalibError::Runtime(error.to_string()), layer));
        }
    };
    Ok(Gemma3Weights {
        token_embd,
        embd_format: EmbeddingFormat::F32,
        output_norm,
        output: WeightTensor {
            buf: output_buf,
            gpu_dtype: DType::F32,
            m: 1,
            k: 1,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        },
        layers: vec![layer],
        tied_lm_head: false,
    })
}

fn free_gemma3_layer(gpu: &mut Gpu, layer: Gemma3LayerWeights) {
    for tensor in [
        layer.input_norm,
        layer.q_norm,
        layer.k_norm,
        layer.post_attn_norm,
        layer.pre_ffn_norm,
        layer.post_ffn_norm,
    ] {
        let _ = gpu.free_tensor(tensor);
    }
    for weight in [
        layer.wq,
        layer.wk,
        layer.wv,
        layer.wo,
        layer.w_gate,
        layer.w_up,
        layer.w_down,
    ] {
        free_weight(gpu, weight);
    }
}

fn free_weight(gpu: &mut Gpu, weight: WeightTensor) {
    let _ = gpu.free_tensor(weight.buf);
    if let Some(scale) = weight.awq_scale {
        let _ = gpu.free_tensor(scale);
    }
}

fn i32_slice_as_bytes(values: &[i32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Gemma3Config {
        Gemma3Config {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 1024,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            sliding_window: 128,
            sliding_window_pattern: 2,
            query_pre_attn_scalar: 16.0,
            hidden_activation: "gelu_pytorch_tanh".into(),
            tie_word_embeddings: true,
            gemma_norm_offset: 0.0,
            eos_token_id: 106,
        }
    }

    #[test]
    fn dense_capture_aliases_qkv_and_gate_up() {
        let registry = gemma3_capture_registry(&tiny_config()).unwrap();
        assert_eq!(registry.len(), 8);
        let q = registry
            .resolve_output("model.layers.1.self_attn.q_proj")
            .unwrap();
        assert_eq!(
            Some(q),
            registry.resolve_output("model.layers.1.self_attn.k_proj")
        );
        assert_eq!(
            Some(q),
            registry.resolve_output("model.layers.1.self_attn.v_proj")
        );
        let gate = registry
            .resolve_output("model.layers.0.mlp.gate_proj")
            .unwrap();
        assert_eq!(
            Some(gate),
            registry.resolve_output("model.layers.0.mlp.up_proj")
        );
        assert_eq!(registry.get(gate).unwrap().input_width, 64);
    }
}
