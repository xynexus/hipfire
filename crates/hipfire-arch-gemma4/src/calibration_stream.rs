// SPDX-License-Identifier: Apache-2.0
//! Gemma 4 source-layout adapter for the family-neutral layer-stream engine.

use crate::config::{AttentionKind, FfnPlan, Gemma4Config, KvProducer, RopePlan, ValueProjection};
use crate::forward::{
    calibration_forward_layer_from_hidden, Gemma4CalibrationCapture, Gemma4DenseState,
};
use crate::weights::{
    Gemma4CoreWeights, Gemma4DenseLayerWeights, Gemma4DenseWeights, Gemma4MoeExpertWeights,
    Gemma4MoeLayerWeights,
};
use hipfire_model::{ModelSource, ARCH_ID_GEMMA4};
use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_runtime::calibration::contracts::{
    CalibError, CalibrationJob, CaptureDescriptor, CaptureId, CapturePolicy, CaptureRegistry,
    ExpertTelemetry, ProjectionRole, SampleRow,
};
use hipfire_runtime::calibration::schedule::{LayerMicrobatch, MicrobatchGeometry};
use hipfire_runtime::calibration::source::{
    load_source_f32_tensor, load_source_matrix, load_source_tensor, PlannedTensorReader,
    TensorLoadRequest, TensorOwner,
};
use hipfire_runtime::calibration::stream::{
    CalibrationEmbedding, CalibrationFamilyAdapter, CalibrationFinalizer, CalibrationLayer,
    CalibrationResourceEstimate, LayerCapturePartSummary, ModelInspection, RmsNormLmHeadFinalizer,
};
use hipfire_runtime::calibration::CalibCollector;
use hipfire_runtime::triattn::{
    TriAttnAttentionKind, TriAttnContextPolicy, TriAttnLayerRecord, TriAttnPackageMetadata,
    TriAttnRopeConvention, TRIATTN_ARTIFACT_KIND, TRIATTN_HFQM_SCHEMA,
};
use hipfire_runtime::weights::{EmbeddingFormat, WeightTensor};
use std::sync::Arc;

const SOURCE_DTYPES: &[&str] = &["BF16", "F16", "F32"];
const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
const FINAL_NORM: &str = "model.language_model.norm.weight";
const LM_HEAD: &str = "lm_head.weight";

#[derive(Default)]
pub struct Gemma4CalibrationAdapter {
    config: Option<Gemma4Config>,
    source_dtypes: Vec<String>,
    embedding_bf16: bool,
}

const GEMMA4_CALIBRATION_ARCH_IDS: &[u32] = &[ARCH_ID_GEMMA4];

fn gemma4_calibration_adapter_factory() -> Box<dyn CalibrationFamilyAdapter> {
    Box::new(Gemma4CalibrationAdapter::default())
}

hipfire_runtime::register_calibration_adapter!(
    "gemma4",
    "gemma4-stream-v1",
    GEMMA4_CALIBRATION_ARCH_IDS,
    gemma4_calibration_adapter_factory
);

impl Gemma4CalibrationAdapter {
    fn config(&self) -> Result<&Gemma4Config, CalibError> {
        self.config.as_ref().ok_or_else(|| {
            CalibError::InvalidSourcePlan(
                "Gemma 4 adapter must inspect the source before loading phases".into(),
            )
        })
    }
}

pub fn inspect_gemma4_stream_source(
    source: &dyn ModelSource,
) -> Result<ModelInspection, CalibError> {
    if source.arch_id() != ARCH_ID_GEMMA4 {
        return Err(CalibError::InvalidSourcePlan(format!(
            "Gemma 4 calibration requires arch {ARCH_ID_GEMMA4}, got {}",
            source.arch_id()
        )));
    }
    let config = Gemma4Config::from_json_str(source.metadata_json())
        .map_err(CalibError::InvalidSourcePlan)?;
    if config.hidden_size_per_layer_input != 0 {
        return Err(CalibError::InvalidSourcePlan(
            "Gemma 4 streamed calibration does not yet admit PLE variants".into(),
        ));
    }
    if config
        .layers
        .iter()
        .any(|layer| !matches!(layer.kv_producer, KvProducer::Own))
    {
        return Err(CalibError::InvalidSourcePlan(
            "Gemma 4 streamed calibration requires owned per-layer KV producers".into(),
        ));
    }
    Ok(ModelInspection {
        family: "gemma4".into(),
        arch_id: source.arch_id(),
        hidden_width: config.hidden_size,
        vocab_size: config.vocab_size,
        num_layers: config.num_hidden_layers,
        tensor_requests: gemma4_tensor_requests(source, &config)?,
    })
}

pub fn gemma4_tensor_requests(
    source: &dyn ModelSource,
    config: &Gemma4Config,
) -> Result<Vec<TensorLoadRequest>, CalibError> {
    let mut requests = Vec::new();
    push_required(
        source,
        &mut requests,
        "embedding",
        EMBEDDING,
        TensorOwner::Persistent,
    )?;
    push_required(
        source,
        &mut requests,
        "final_norm",
        FINAL_NORM,
        TensorOwner::Persistent,
    )?;
    if config.tie_word_embeddings || source.tensor_info(LM_HEAD).is_none() {
        requests.push(TensorLoadRequest::alias(
            "lm_head",
            EMBEDDING,
            TensorOwner::Persistent,
            "embedding",
        ));
    } else {
        push_required(
            source,
            &mut requests,
            "lm_head",
            LM_HEAD,
            TensorOwner::Persistent,
        )?;
    }

    for (layer, plan) in config.layers.iter().enumerate() {
        let owner = TensorOwner::Layer(layer);
        let prefix = format!("model.language_model.layers.{layer}");
        let attn = format!("{prefix}.self_attn");
        for name in [
            format!("{prefix}.input_layernorm.weight"),
            format!("{prefix}.layer_scalar"),
            format!("{attn}.q_norm.weight"),
            format!("{attn}.k_norm.weight"),
            format!("{attn}.q_proj.weight"),
            format!("{attn}.k_proj.weight"),
            format!("{attn}.o_proj.weight"),
            format!("{prefix}.post_attention_layernorm.weight"),
            format!("{prefix}.pre_feedforward_layernorm.weight"),
            format!("{prefix}.post_feedforward_layernorm.weight"),
            format!("{prefix}.mlp.gate_proj.weight"),
            format!("{prefix}.mlp.up_proj.weight"),
            format!("{prefix}.mlp.down_proj.weight"),
        ] {
            push_required(source, &mut requests, &name, &name, owner)?;
        }
        if plan.value_projection == ValueProjection::Separate {
            let name = format!("{attn}.v_proj.weight");
            push_required(source, &mut requests, &name, &name, owner)?;
        }
        if matches!(plan.ffn, FfnPlan::DensePlusMoe { .. }) {
            for name in [
                format!("{prefix}.router.scale"),
                format!("{prefix}.router.proj.weight"),
                format!("{prefix}.router.per_expert_scale"),
                format!("{prefix}.experts.gate_up_proj"),
                format!("{prefix}.experts.down_proj"),
            ] {
                push_required(source, &mut requests, &name, &name, owner)?;
            }
        }
    }
    Ok(requests)
}

fn push_required(
    source: &dyn ModelSource,
    requests: &mut Vec<TensorLoadRequest>,
    logical: &str,
    source_name: &str,
    owner: TensorOwner,
) -> Result<(), CalibError> {
    let info = source.tensor_info(source_name).ok_or_else(|| {
        CalibError::InvalidSourcePlan(format!("Gemma 4 source tensor {source_name} is missing"))
    })?;
    if !SOURCE_DTYPES.contains(&info.dtype.as_str()) {
        return Err(CalibError::InvalidSourcePlan(format!(
            "Gemma 4 source tensor {source_name} has unsupported dtype {}",
            info.dtype
        )));
    }
    requests.push(TensorLoadRequest::tensor(logical, source_name, owner));
    Ok(())
}

pub fn gemma4_capture_registry(
    config: &Gemma4Config,
    job: &CalibrationJob,
) -> Result<CaptureRegistry, CalibError> {
    let mut registry = CaptureRegistry::default();
    for (layer, plan) in config.layers.iter().enumerate() {
        let prefix = format!("model.language_model.layers.{layer}");
        let mut qkv = vec![
            format!("{prefix}.self_attn.q_proj"),
            format!("{prefix}.self_attn.k_proj"),
        ];
        if plan.value_projection == ValueProjection::Separate {
            qkv.push(format!("{prefix}.self_attn.v_proj"));
        }
        register_capture(
            &mut registry,
            layer,
            ProjectionRole::QueryInput,
            None,
            qkv,
            config.hidden_size,
            CapturePolicy::HessianAndImatrix,
            None,
        )?;
        register_capture(
            &mut registry,
            layer,
            ProjectionRole::AttentionOutputInput,
            None,
            vec![format!("{prefix}.self_attn.o_proj")],
            plan.attention.q_heads * plan.attention.head_dim,
            CapturePolicy::HessianAndImatrix,
            None,
        )?;
        register_capture(
            &mut registry,
            layer,
            ProjectionRole::DenseMlpInput,
            None,
            vec![
                format!("{prefix}.mlp.gate_proj"),
                format!("{prefix}.mlp.up_proj"),
            ],
            config.hidden_size,
            CapturePolicy::HessianAndImatrix,
            None,
        )?;
        let dense_intermediate = match plan.ffn {
            FfnPlan::Dense { intermediate } => intermediate,
            FfnPlan::DensePlusMoe {
                dense_intermediate, ..
            } => dense_intermediate,
        };
        register_capture(
            &mut registry,
            layer,
            ProjectionRole::DownInput,
            None,
            vec![format!("{prefix}.mlp.down_proj")],
            dense_intermediate,
            CapturePolicy::HessianAndImatrix,
            None,
        )?;
        if let FfnPlan::DensePlusMoe {
            expert_intermediate,
            experts,
            ..
        } = plan.ffn
        {
            register_capture(
                &mut registry,
                layer,
                ProjectionRole::RouterInput,
                None,
                vec![format!("{prefix}.router.proj")],
                config.hidden_size,
                CapturePolicy::HessianAndImatrix,
                None,
            )?;
            for expert in 0..experts {
                let ep = format!("{prefix}.experts.{expert}");
                register_capture(
                    &mut registry,
                    layer,
                    ProjectionRole::GateUpInput,
                    Some(expert),
                    vec![format!("{ep}.gate_proj"), format!("{ep}.up_proj")],
                    config.hidden_size,
                    CapturePolicy::ImatrixOnly,
                    Some(job.options.expert_quota),
                )?;
                register_capture(
                    &mut registry,
                    layer,
                    ProjectionRole::DownInput,
                    Some(expert),
                    vec![format!("{ep}.down_proj")],
                    expert_intermediate,
                    CapturePolicy::ImatrixOnly,
                    Some(job.options.expert_quota),
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
    expert_quota: Option<hipfire_runtime::calibration::contracts::ExpertCaptureQuota>,
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

struct Gemma4CalibrationEmbedding {
    embedding: Option<GpuTensor>,
    token_ids: Option<GpuTensor>,
    output: Option<GpuTensor>,
    vocab_size: usize,
    dim: usize,
    max_rows: usize,
    source_bf16: bool,
}

impl CalibrationEmbedding for Gemma4CalibrationEmbedding {
    fn execute(
        &mut self,
        gpu: &mut Gpu,
        rows: &[SampleRow],
        output_f32: &mut [f32],
    ) -> Result<(), CalibError> {
        if rows.is_empty()
            || rows.len() > self.max_rows
            || output_f32.len() != rows.len() * self.dim
        {
            return Err(CalibError::InvalidOptions(
                "Gemma 4 embedding batch geometry is invalid".into(),
            ));
        }
        let tokens = rows
            .iter()
            .map(|row| {
                if row.token as usize >= self.vocab_size {
                    Err(CalibError::InvalidSamples(format!(
                        "token {} exceeds Gemma 4 vocabulary {}",
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
        let embedding = self.embedding.as_ref().unwrap();
        let output = self.output.as_ref().unwrap();
        match embedding.dtype {
            DType::BF16 => gpu.embedding_lookup_bf16_batched(
                embedding,
                output,
                self.token_ids.as_ref().unwrap(),
                rows.len(),
                self.dim,
            ),
            DType::F16 => gpu.embedding_lookup_f16_batched(
                embedding,
                output,
                self.token_ids.as_ref().unwrap(),
                rows.len(),
                self.dim,
            ),
            DType::F32 => gpu.embedding_lookup_f32_batched(
                embedding,
                output,
                self.token_ids.as_ref().unwrap(),
                rows.len(),
                self.dim,
            ),
            dtype => {
                return Err(CalibError::InvalidSourcePlan(format!(
                    "Gemma 4 embedding dtype {dtype:?} is unsupported"
                )))
            }
        }
        .map_err(|error| CalibError::Runtime(error.to_string()))?;
        let live = output.sub_offset(0, output_f32.len());
        let scale = if self.source_bf16 {
            round_f32_to_bf16((self.dim as f32).sqrt())
        } else {
            (self.dim as f32).sqrt()
        };
        gpu.scale_f32(&live, scale)
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        if self.source_bf16 {
            gpu.bf16_round_trip_f32(&live)
                .map_err(|error| CalibError::Runtime(error.to_string()))?;
        }
        output_f32.copy_from_slice(
            &gpu.download_f32(&live)
                .map_err(|error| CalibError::Runtime(error.to_string()))?,
        );
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

impl CalibrationFamilyAdapter for Gemma4CalibrationAdapter {
    fn family(&self) -> &'static str {
        "gemma4"
    }

    fn adapter_version(&self) -> &'static str {
        "gemma4-stream-v1"
    }

    fn resource_estimate(
        &self,
        _model: &ModelInspection,
        job: &CalibrationJob,
        geometry: MicrobatchGeometry,
    ) -> Result<Option<CalibrationResourceEstimate>, CalibError> {
        Ok(Some(gemma4_resource_estimate(
            self.config()?,
            job,
            geometry,
        )?))
    }

    fn effective_precision(&self, gpu: &Gpu) -> serde_json::Value {
        serde_json::json!({
            "boundary": "f32",
            "source_dtypes": self.source_dtypes,
            "bf16_weight_execution": if gpu.arch.starts_with("gfx11") || gpu.arch.starts_with("gfx12") { "bf16-native" } else { "f16-fallback" },
            "gpu_arch": gpu.arch,
        })
    }

    fn inspect(&mut self, source: &dyn ModelSource) -> Result<ModelInspection, CalibError> {
        let inspection = inspect_gemma4_stream_source(source)?;
        self.source_dtypes = inspection
            .tensor_requests
            .iter()
            .filter_map(|request| source.tensor_info(&request.source_name))
            .map(|info| info.dtype.clone())
            .collect();
        self.source_dtypes.sort();
        self.source_dtypes.dedup();
        self.embedding_bf16 = source
            .tensor_info(EMBEDDING)
            .is_some_and(|info| info.dtype == "BF16");
        self.config = Some(
            Gemma4Config::from_json_str(source.metadata_json())
                .map_err(CalibError::InvalidSourcePlan)?,
        );
        Ok(inspection)
    }

    fn capture_plan(
        &self,
        _model: &ModelInspection,
        job: &CalibrationJob,
    ) -> Result<CaptureRegistry, CalibError> {
        gemma4_capture_registry(self.config()?, job)
    }

    fn cask_metadata(
        &self,
        model: &ModelInspection,
        job: &CalibrationJob,
    ) -> Result<Option<TriAttnPackageMetadata>, CalibError> {
        let config = self.config()?;
        let layers = config
            .layers
            .iter()
            .enumerate()
            .map(|(layer, plan)| {
                let (rope_theta, rotary_dim) = match plan.rope {
                    RopePlan::FullHalfSplit { theta, dim } => (theta, dim),
                    RopePlan::ProportionalHalfSplit {
                        theta, rotary_dim, ..
                    } => (theta, rotary_dim),
                };
                let (attention_kind, context_policy, sliding_window) = match plan.kind {
                    AttentionKind::Full => {
                        (TriAttnAttentionKind::Full, TriAttnContextPolicy::Full, None)
                    }
                    AttentionKind::Sliding => (
                        TriAttnAttentionKind::Sliding,
                        TriAttnContextPolicy::Sliding,
                        Some(config.sliding_window as u32),
                    ),
                };
                TriAttnLayerRecord {
                    physical_layer: layer as u32,
                    attention_kind,
                    q_heads: plan.attention.q_heads as u32,
                    kv_heads: plan.attention.kv_heads as u32,
                    head_dim: plan.attention.head_dim as u32,
                    rotary_dim: rotary_dim as u32,
                    rope_theta,
                    rope_convention: TriAttnRopeConvention::HalfSplit,
                    context_policy,
                    sliding_window,
                    kv_producer: match plan.kv_producer {
                        KvProducer::Own => None,
                        KvProducer::SharedFrom { producer_layer } => Some(producer_layer as u32),
                    },
                    center_tensor: format!("triattn.layers.{layer}.centers"),
                    center_offset: 0,
                    center_count: (plan.attention.q_heads * (plan.attention.head_dim / 2)) as u64,
                    sample_count: 1,
                }
            })
            .collect();
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
        let embedding = load_source_tensor(
            reader,
            gpu,
            "embedding",
            &[model.vocab_size, model.hidden_width],
        )?;
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
        Ok(Box::new(Gemma4CalibrationEmbedding {
            embedding: Some(embedding),
            token_ids: Some(token_ids),
            output: Some(output),
            vocab_size: model.vocab_size,
            dim: model.hidden_width,
            max_rows: job.options.max_rows,
            source_bf16: self.embedding_bf16,
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
        Ok(Box::new(Gemma4StreamedCalibrationLayer::load(
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
        let norm = load_source_f32_tensor(reader, gpu, "final_norm", model.hidden_width, false)?;
        let lm_head =
            load_source_matrix(reader, gpu, "lm_head", model.vocab_size, model.hidden_width)?;
        Ok(Box::new(RmsNormLmHeadFinalizer::new_with_softcap(
            gpu,
            norm,
            lm_head,
            model.hidden_width,
            model.vocab_size,
            self.config()?.rms_norm_eps,
            job.options.max_rows,
            Some(self.config()?.final_logit_softcapping),
        )?))
    }
}

fn gemma4_resource_estimate(
    config: &Gemma4Config,
    job: &CalibrationJob,
    geometry: MicrobatchGeometry,
) -> Result<CalibrationResourceEstimate, CalibError> {
    let max_q = config
        .layers
        .iter()
        .map(|layer| layer.attention.q_heads * layer.attention.head_dim)
        .max()
        .unwrap_or(0) as u128;
    let max_kv = config
        .layers
        .iter()
        .map(|layer| layer.attention.kv_heads * layer.attention.head_dim)
        .max()
        .unwrap_or(0) as u128;
    let max_inter = config
        .layers
        .iter()
        .map(|layer| match layer.ffn {
            FfnPlan::Dense { intermediate } => intermediate,
            FfnPlan::DensePlusMoe {
                dense_intermediate,
                expert_intermediate,
                ..
            } => dense_intermediate.max(expert_intermediate),
        })
        .max()
        .unwrap_or(0) as u128;
    let state = job.samples.context_len() as u128 * max_kv * 8;
    let scratch = (4 * config.hidden_size as u128 + 2 * max_q + 2 * max_kv + 3 * max_inter) * 4;
    let active = geometry.sequence_batch.min(job.samples.samples().len()) as u128;
    let checked = |label: &str, value: u128| {
        u64::try_from(value).map_err(|_| {
            CalibError::InvalidOptions(format!("Gemma 4 {label} estimate overflows u64"))
        })
    };
    Ok(CalibrationResourceEstimate {
        scratch_bytes: checked("scratch", scratch * active)?,
        state_bytes_per_sequence: checked("state", state)?,
        active_state_bytes: checked("active state", (scratch + state) * active)?,
        details: serde_json::json!({
            "active_sequences": active,
            "max_q_width": max_q,
            "max_kv_width": max_kv,
            "max_intermediate": max_inter,
        }),
    })
}

pub struct Gemma4StreamedCalibrationLayer {
    logical_layer: usize,
    config: Gemma4Config,
    weights: Option<Gemma4DenseWeights>,
    sample_lengths: Vec<usize>,
    active_sequence_start: Option<usize>,
    states: Vec<Gemma4DenseState>,
    capture_registry: Option<Arc<CaptureRegistry>>,
    collector: Option<Arc<CalibCollector>>,
    telemetry: Option<ExpertTelemetry>,
}

impl Gemma4StreamedCalibrationLayer {
    fn load(
        reader: &mut PlannedTensorReader<'_, '_, '_>,
        gpu: &mut Gpu,
        full: &Gemma4Config,
        layer: usize,
        job: &CalibrationJob,
    ) -> Result<Self, CalibError> {
        if layer >= full.num_hidden_layers {
            return Err(CalibError::InvalidSourcePlan(format!(
                "Gemma 4 layer {layer} exceeds {} layers",
                full.num_hidden_layers
            )));
        }
        let loaded = load_gemma4_streamed_layer(reader, gpu, full, layer)?;
        let weights = wrap_streamed_layer(gpu, loaded)?;
        let mut config = full.clone();
        config.num_hidden_layers = 1;
        config.vocab_size = 1;
        config.tie_word_embeddings = false;
        config.layers = vec![full.layers[layer].clone()];
        let telemetry = match full.layers[layer].ffn {
            FfnPlan::Dense { .. } => None,
            FfnPlan::DensePlusMoe { experts, top_k, .. } => Some(ExpertTelemetry::new(
                full.num_hidden_layers,
                experts,
                top_k,
                job.options.expert_quota,
                4096,
            )?),
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
            telemetry,
        })
    }

    fn prepare_capture(&mut self, registry: &CaptureRegistry) -> Result<(), CalibError> {
        if let Some(existing) = &self.capture_registry {
            if existing.as_ref() != registry {
                return Err(CalibError::InvalidCapture(
                    "capture registry changed while a Gemma 4 layer was resident".into(),
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
            return Ok(());
        }
        if batch.sequence_start >= batch.sequence_end
            || batch.sequence_end > self.sample_lengths.len()
        {
            return Err(CalibError::InvalidOptions(format!(
                "Gemma 4 sequence range {}..{} is outside {} samples",
                batch.sequence_start,
                batch.sequence_end,
                self.sample_lengths.len()
            )));
        }
        self.release_states(gpu);
        let mut states = Vec::with_capacity(batch.sequence_end - batch.sequence_start);
        for &sample_len in &self.sample_lengths[batch.sequence_start..batch.sequence_end] {
            match Gemma4DenseState::new(gpu, &self.config, sample_len.max(1)) {
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

impl CalibrationLayer for Gemma4StreamedCalibrationLayer {
    fn execute(
        &mut self,
        gpu: &mut Gpu,
        batch: &LayerMicrobatch,
        input_f32: &[f32],
        output_f32: &mut [f32],
        capture: &CaptureRegistry,
    ) -> Result<(), CalibError> {
        let expected = batch.rows.len() * self.config.hidden_size;
        if batch.rows.is_empty() || input_f32.len() != expected || output_f32.len() != expected {
            return Err(CalibError::InvalidOptions(
                "Gemma 4 layer boundary geometry is invalid".into(),
            ));
        }
        self.prepare_sequence_group(gpu, batch)?;
        self.prepare_capture(capture)?;
        let collector = Arc::clone(self.collector.as_ref().unwrap());
        let registry = Arc::clone(self.capture_registry.as_ref().unwrap());
        let weights = self.weights.as_ref().ok_or_else(|| {
            CalibError::Runtime("Gemma 4 streamed weights were already freed".into())
        })?;
        for (row_index, row) in batch.rows.iter().enumerate() {
            let local = row
                .sample_index
                .checked_sub(batch.sequence_start)
                .ok_or_else(|| {
                    CalibError::InvalidOptions("Gemma 4 sample is outside its group".into())
                })?;
            let state = self.states.get_mut(local).ok_or_else(|| {
                CalibError::InvalidOptions("Gemma 4 sample is outside its group".into())
            })?;
            if row.reset_state {
                state.reset();
            }
            if state.next_pos() != row.position {
                return Err(CalibError::InvalidOptions(format!(
                    "Gemma 4 sample {} expected position {}, got {}",
                    row.sample_index,
                    state.next_pos(),
                    row.position
                )));
            }
            let start = row_index * self.config.hidden_size;
            let end = start + self.config.hidden_size;
            let mut calibration = Gemma4CalibrationCapture {
                collector: collector.as_ref(),
                registry: registry.as_ref(),
                telemetry: self.telemetry.as_mut(),
                logical_layer: self.logical_layer,
            };
            let output = calibration_forward_layer_from_hidden(
                gpu,
                weights,
                &self.config,
                state,
                0,
                row.position,
                &input_f32[start..end],
                &mut calibration,
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
                "Gemma 4 layer {} has no collector",
                self.logical_layer
            ))
        })?;
        let descriptors = collector.tensor_descriptors();
        if let Some(telemetry) = self.telemetry.as_mut() {
            telemetry.finalize_direct_capture_layer(self.logical_layer)?;
        }
        let expert_telemetry = self
            .telemetry
            .as_ref()
            .map(|telemetry| telemetry.layer_snapshot(self.logical_layer))
            .transpose()?;
        let max_consistency = collector
            .write_streaming(gpu, path, arch_id, metadata_json, &[])
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        collector.free_gpu(gpu);
        self.capture_registry = None;
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
        self.release_states(gpu);
        if let Some(weights) = self.weights.take() {
            weights.free_gpu(gpu);
        }
        Ok(())
    }
}

fn load_gemma4_streamed_layer(
    reader: &mut PlannedTensorReader<'_, '_, '_>,
    gpu: &mut Gpu,
    config: &Gemma4Config,
    layer: usize,
) -> Result<Gemma4DenseLayerWeights, CalibError> {
    let plan = &config.layers[layer];
    let prefix = format!("model.language_model.layers.{layer}");
    let attn = format!("{prefix}.self_attn");
    let q_dim = plan.attention.q_heads * plan.attention.head_dim;
    let kv_dim = plan.attention.kv_heads * plan.attention.head_dim;
    let dense_intermediate = match plan.ffn {
        FfnPlan::Dense { intermediate } => intermediate,
        FfnPlan::DensePlusMoe {
            dense_intermediate, ..
        } => dense_intermediate,
    };
    let scalar = load_source_f32_tensor(reader, gpu, &format!("{prefix}.layer_scalar"), 1, false)?;
    let layer_scalar = gpu
        .download_f32(&scalar)
        .map_err(|error| CalibError::Runtime(error.to_string()))?[0];
    gpu.free_tensor(scalar)
        .map_err(|error| CalibError::Runtime(error.to_string()))?;
    let input_norm = load_source_f32_tensor(
        reader,
        gpu,
        &format!("{prefix}.input_layernorm.weight"),
        config.hidden_size,
        false,
    )?;
    let q_norm = load_source_f32_tensor(
        reader,
        gpu,
        &format!("{attn}.q_norm.weight"),
        plan.attention.head_dim,
        false,
    )?;
    let k_norm = load_source_f32_tensor(
        reader,
        gpu,
        &format!("{attn}.k_norm.weight"),
        plan.attention.head_dim,
        false,
    )?;
    let wq = load_source_matrix(
        reader,
        gpu,
        &format!("{attn}.q_proj.weight"),
        q_dim,
        config.hidden_size,
    )?;
    let wk = load_source_matrix(
        reader,
        gpu,
        &format!("{attn}.k_proj.weight"),
        kv_dim,
        config.hidden_size,
    )?;
    let wv = if plan.value_projection == ValueProjection::Separate {
        Some(load_source_matrix(
            reader,
            gpu,
            &format!("{attn}.v_proj.weight"),
            kv_dim,
            config.hidden_size,
        )?)
    } else {
        None
    };
    let wo = load_source_matrix(
        reader,
        gpu,
        &format!("{attn}.o_proj.weight"),
        config.hidden_size,
        q_dim,
    )?;
    let post_attn_norm = load_source_f32_tensor(
        reader,
        gpu,
        &format!("{prefix}.post_attention_layernorm.weight"),
        config.hidden_size,
        false,
    )?;
    let pre_ffn_norm = load_source_f32_tensor(
        reader,
        gpu,
        &format!("{prefix}.pre_feedforward_layernorm.weight"),
        config.hidden_size,
        false,
    )?;
    let post_ffn_norm = load_source_f32_tensor(
        reader,
        gpu,
        &format!("{prefix}.post_feedforward_layernorm.weight"),
        config.hidden_size,
        false,
    )?;
    let w_gate = load_source_matrix(
        reader,
        gpu,
        &format!("{prefix}.mlp.gate_proj.weight"),
        dense_intermediate,
        config.hidden_size,
    )?;
    let w_up = load_source_matrix(
        reader,
        gpu,
        &format!("{prefix}.mlp.up_proj.weight"),
        dense_intermediate,
        config.hidden_size,
    )?;
    let w_down = load_source_matrix(
        reader,
        gpu,
        &format!("{prefix}.mlp.down_proj.weight"),
        config.hidden_size,
        dense_intermediate,
    )?;
    let moe = match plan.ffn {
        FfnPlan::Dense { .. } => None,
        FfnPlan::DensePlusMoe {
            expert_intermediate,
            experts,
            top_k,
            ..
        } => Some(load_stacked_moe(
            reader,
            gpu,
            &prefix,
            config.hidden_size,
            expert_intermediate,
            experts,
            top_k,
        )?),
    };
    Ok(Gemma4DenseLayerWeights {
        input_norm,
        q_norm,
        k_norm,
        wq,
        wk,
        wv,
        wo,
        post_attn_norm,
        pre_ffn_norm,
        post_ffn_norm,
        w_gate,
        w_up,
        w_down,
        layer_scalar,
        ple: None,
        moe,
    })
}

fn load_stacked_moe(
    reader: &mut PlannedTensorReader<'_, '_, '_>,
    gpu: &mut Gpu,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
    experts: usize,
    top_k: usize,
) -> Result<Gemma4MoeLayerWeights, CalibError> {
    let router_scale = load_source_f32_tensor(
        reader,
        gpu,
        &format!("{prefix}.router.scale"),
        hidden,
        false,
    )?;
    let router = load_source_matrix(
        reader,
        gpu,
        &format!("{prefix}.router.proj.weight"),
        experts,
        hidden,
    )?;
    let scale_tensor = load_source_f32_tensor(
        reader,
        gpu,
        &format!("{prefix}.router.per_expert_scale"),
        experts,
        false,
    )?;
    let per_expert_scale = gpu
        .download_f32(&scale_tensor)
        .map_err(|error| CalibError::Runtime(error.to_string()))?;
    gpu.free_tensor(scale_tensor)
        .map_err(|error| CalibError::Runtime(error.to_string()))?;
    let gate_up = load_source_tensor(
        reader,
        gpu,
        &format!("{prefix}.experts.gate_up_proj"),
        &[experts, 2 * intermediate, hidden],
    )?;
    let down = load_source_tensor(
        reader,
        gpu,
        &format!("{prefix}.experts.down_proj"),
        &[experts, hidden, intermediate],
    )?;
    let mut out = Vec::with_capacity(experts);
    for expert in 0..experts {
        let base = expert * 2 * intermediate * hidden;
        out.push(Gemma4MoeExpertWeights {
            gate: copy_weight(gpu, &gate_up, base, intermediate, hidden)?,
            up: copy_weight(
                gpu,
                &gate_up,
                base + intermediate * hidden,
                intermediate,
                hidden,
            )?,
            down: copy_weight(
                gpu,
                &down,
                expert * hidden * intermediate,
                hidden,
                intermediate,
            )?,
        });
    }
    gpu.free_tensor(gate_up)
        .map_err(|error| CalibError::Runtime(error.to_string()))?;
    gpu.free_tensor(down)
        .map_err(|error| CalibError::Runtime(error.to_string()))?;
    Ok(Gemma4MoeLayerWeights {
        router_scale,
        router,
        per_expert_scale,
        experts: out,
        top_k,
    })
}

fn copy_weight(
    gpu: &mut Gpu,
    source: &GpuTensor,
    element_offset: usize,
    m: usize,
    k: usize,
) -> Result<WeightTensor, CalibError> {
    let mut output = gpu
        .alloc_tensor(&[m * k], source.dtype)
        .map_err(|error| CalibError::Runtime(error.to_string()))?;
    output.dtype = source.dtype;
    gpu.copy_d2d(
        &source.sub_offset(element_offset, m * k),
        &output,
        m * k * source.dtype.size(),
    )
    .map_err(|error| CalibError::Runtime(error.to_string()))?;
    Ok(WeightTensor {
        buf: output,
        gpu_dtype: source.dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    })
}

fn wrap_streamed_layer(
    gpu: &mut Gpu,
    layer: Gemma4DenseLayerWeights,
) -> Result<Gemma4DenseWeights, CalibError> {
    let token_embd = gpu
        .zeros(&[1], DType::F32)
        .map_err(|error| CalibError::Runtime(error.to_string()))?;
    let output_norm = gpu
        .zeros(&[1], DType::F32)
        .map_err(|error| CalibError::Runtime(error.to_string()))?;
    let output_buf = gpu
        .zeros(&[1], DType::F32)
        .map_err(|error| CalibError::Runtime(error.to_string()))?;
    Ok(Gemma4DenseWeights {
        core: Gemma4CoreWeights {
            token_embd,
            embd_format: EmbeddingFormat::F32,
            embedding_source_bf16: false,
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
            tied_lm_head: false,
        },
        layers: vec![layer],
        ple: None,
    })
}

fn round_f32_to_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
}

fn i32_slice_as_bytes(values: &[i32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_scale_uses_bf16_round_to_nearest_even() {
        let value = 2816.0f32.sqrt();
        assert_eq!(round_f32_to_bf16(value).to_bits() & 0xffff, 0);
        assert_ne!(round_f32_to_bf16(value), value);
    }
}
