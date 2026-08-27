// SPDX-License-Identifier: Apache-2.0
//! Cohere2-MoE source contract and native layer-stream registration.

use crate::config::{Cohere2Config, Cohere2LayerKind, Cohere2MlpKind};
use hipfire_model::{ModelSource, ARCH_ID_COHERE2_MOE};
use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_runtime::calibration::contracts::{
    CalibError, CalibrationJob, CaptureAdmission, CaptureDescriptor, CaptureId, CapturePolicy,
    CaptureRegistry, ExpertCaptureRole, ExpertTelemetry, ProjectionRole, SampleRow,
};
use hipfire_runtime::calibration::schedule::LayerMicrobatch;
use hipfire_runtime::calibration::source::{
    load_source_f32_tensor, load_source_matrix, load_source_tensor, PlannedTensorReader,
    TensorLoadRequest, TensorOwner,
};
use hipfire_runtime::calibration::stream::{
    CalibrationEmbedding, CalibrationFamilyAdapter, CalibrationFinalizer, CalibrationLayer,
    LayerCapturePartSummary, ModelInspection, RmsNormLmHeadFinalizer,
};
use hipfire_runtime::calibration::CalibCollector;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv::KvCache;
use hipfire_runtime::layered_kv::{KvStorageKind, LayeredKvArena};
use hipfire_runtime::transformer_loader::TransformerLoader;
use hipfire_runtime::triattn::{
    EvictionResult, LayeredEvictionCtx, TriAttnArtifact, TriAttnAttentionKind,
    TriAttnContextPolicy, TriAttnLayerRecord, TriAttnPackageMetadata, TriAttnRopeConvention,
    TRIATTN_ARTIFACT_KIND, TRIATTN_HFQM_SCHEMA,
};
use hipfire_runtime::weights::{weight_gemv, EmbeddingFormat, WeightTensor};
use std::sync::Arc;

const SOURCE_DTYPES: &[&str] = &["BF16", "F16", "F32"];
const EMBEDDING: &str = "model.embed_tokens.weight";
const FINAL_NORM: &str = "model.norm.weight";
const LM_HEAD: &str = "lm_head.weight";
const ARCH_IDS: &[u32] = &[ARCH_ID_COHERE2_MOE];

#[derive(Default)]
pub struct Cohere2CalibrationAdapter {
    config: Option<Cohere2Config>,
    source_dtypes: Vec<String>,
}

fn factory() -> Box<dyn CalibrationFamilyAdapter> {
    Box::new(Cohere2CalibrationAdapter::default())
}

hipfire_runtime::register_calibration_adapter!(
    "cohere2-moe",
    "cohere2-moe-stream-v1",
    ARCH_IDS,
    factory
);

impl Cohere2CalibrationAdapter {
    fn config(&self) -> Result<&Cohere2Config, CalibError> {
        self.config.as_ref().ok_or_else(|| {
            CalibError::InvalidSourcePlan(
                "Cohere2 adapter must inspect the source before loading phases".into(),
            )
        })
    }
}

pub fn inspect_cohere2_stream_source(
    source: &dyn ModelSource,
) -> Result<ModelInspection, CalibError> {
    if source.arch_id() != ARCH_ID_COHERE2_MOE {
        return Err(CalibError::InvalidSourcePlan(format!(
            "Cohere2 calibration requires arch {ARCH_ID_COHERE2_MOE}, got {}",
            source.arch_id()
        )));
    }
    let config = Cohere2Config::from_json_str(source.metadata_json())
        .map_err(CalibError::InvalidSourcePlan)?;
    Ok(ModelInspection {
        family: "cohere2-moe".into(),
        arch_id: source.arch_id(),
        hidden_width: config.hidden_size,
        vocab_size: config.vocab_size,
        num_layers: config.num_hidden_layers,
        tensor_requests: cohere2_tensor_requests(source, &config)?,
    })
}

pub fn cohere2_tensor_requests(
    source: &dyn ModelSource,
    config: &Cohere2Config,
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
    for (layer, mlp) in config.mlp_kinds.iter().enumerate() {
        let owner = TensorOwner::Layer(layer);
        let prefix = format!("model.layers.{layer}");
        for name in [
            format!("{prefix}.input_layernorm.weight"),
            format!("{prefix}.self_attn.q_proj.weight"),
            format!("{prefix}.self_attn.k_proj.weight"),
            format!("{prefix}.self_attn.v_proj.weight"),
            format!("{prefix}.self_attn.o_proj.weight"),
        ] {
            push_required(source, &mut requests, &name, &name, owner)?;
        }
        match mlp {
            Cohere2MlpKind::Dense => {
                for name in ["gate_proj", "up_proj", "down_proj"] {
                    let tensor = format!("{prefix}.mlp.{name}.weight");
                    push_required(source, &mut requests, &tensor, &tensor, owner)?;
                }
            }
            Cohere2MlpKind::Sparse => {
                let router = format!("{prefix}.mlp.gate.weight");
                push_required(source, &mut requests, &router, &router, owner)?;
                for expert in 0..config.num_experts {
                    for name in ["gate_proj", "up_proj", "down_proj"] {
                        let tensor = format!("{prefix}.mlp.experts.{expert}.{name}.weight");
                        push_required(source, &mut requests, &tensor, &tensor, owner)?;
                    }
                }
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
        CalibError::InvalidSourcePlan(format!("Cohere2 source tensor {source_name} is missing"))
    })?;
    if !SOURCE_DTYPES.contains(&info.dtype.as_str()) {
        return Err(CalibError::InvalidSourcePlan(format!(
            "Cohere2 source tensor {source_name} has unsupported dtype {}",
            info.dtype
        )));
    }
    requests.push(TensorLoadRequest::tensor(logical, source_name, owner));
    Ok(())
}

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

fn capture_registry(
    config: &Cohere2Config,
    job: &CalibrationJob,
) -> Result<CaptureRegistry, CalibError> {
    let mut registry = CaptureRegistry::default();
    for (layer, mlp) in config.mlp_kinds.iter().enumerate() {
        let prefix = format!("model.layers.{layer}");
        register_capture(
            &mut registry,
            layer,
            ProjectionRole::QueryInput,
            None,
            vec![
                format!("{prefix}.self_attn.q_proj"),
                format!("{prefix}.self_attn.k_proj"),
                format!("{prefix}.self_attn.v_proj"),
            ],
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
            config.q_heads * config.head_dim,
            CapturePolicy::HessianAndImatrix,
            None,
        )?;
        match mlp {
            Cohere2MlpKind::Dense => {
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
                register_capture(
                    &mut registry,
                    layer,
                    ProjectionRole::DownInput,
                    None,
                    vec![format!("{prefix}.mlp.down_proj")],
                    config.dense_intermediate,
                    CapturePolicy::HessianAndImatrix,
                    None,
                )?;
            }
            Cohere2MlpKind::Sparse => {
                register_capture(
                    &mut registry,
                    layer,
                    ProjectionRole::RouterInput,
                    None,
                    vec![format!("{prefix}.mlp.gate")],
                    config.hidden_size,
                    CapturePolicy::HessianAndImatrix,
                    None,
                )?;
                for expert in 0..config.num_experts {
                    register_capture(
                        &mut registry,
                        layer,
                        ProjectionRole::GateUpInput,
                        Some(expert),
                        vec![
                            format!("{prefix}.mlp.experts.{expert}.gate_proj"),
                            format!("{prefix}.mlp.experts.{expert}.up_proj"),
                        ],
                        config.hidden_size,
                        CapturePolicy::ImatrixOnly,
                        Some(job.options.expert_quota),
                    )?;
                    register_capture(
                        &mut registry,
                        layer,
                        ProjectionRole::DownInput,
                        Some(expert),
                        vec![format!("{prefix}.mlp.experts.{expert}.down_proj")],
                        config.expert_intermediate,
                        CapturePolicy::ImatrixOnly,
                        Some(job.options.expert_quota),
                    )?;
                }
            }
        }
    }
    Ok(registry)
}

fn i32_slice_as_bytes(values: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
}

fn f32_slice_as_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

struct Cohere2CalibrationEmbedding {
    embedding: Option<GpuTensor>,
    token_ids: Option<GpuTensor>,
    output: Option<GpuTensor>,
    vocab_size: usize,
    dim: usize,
    max_rows: usize,
}

impl CalibrationEmbedding for Cohere2CalibrationEmbedding {
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
                "Cohere2 embedding batch geometry is invalid".into(),
            ));
        }
        let tokens = rows
            .iter()
            .map(|row| {
                if row.token as usize >= self.vocab_size {
                    Err(CalibError::InvalidSamples(format!(
                        "token {} exceeds Cohere2 vocabulary {}",
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
                    "Cohere2 embedding dtype {dtype:?} is unsupported"
                )))
            }
        }
        .map_err(|error| CalibError::Runtime(error.to_string()))?;
        let live = output.sub_offset(0, output_f32.len());
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

pub(crate) struct Cohere2AttentionWeights {
    norm: GpuTensor,
    q: WeightTensor,
    k: WeightTensor,
    v: WeightTensor,
    o: WeightTensor,
}

pub(crate) struct Cohere2ExpertWeights {
    gate: WeightTensor,
    up: WeightTensor,
    down: WeightTensor,
}

pub(crate) enum Cohere2MlpWeights {
    Dense(Cohere2ExpertWeights),
    Sparse {
        router: WeightTensor,
        experts: Vec<Cohere2ExpertWeights>,
    },
}

pub(crate) struct Cohere2LayerWeights {
    attention: Cohere2AttentionWeights,
    mlp: Cohere2MlpWeights,
}

impl Cohere2LayerWeights {
    pub(crate) fn free_gpu(self, gpu: &mut Gpu) {
        let Cohere2AttentionWeights { norm, q, k, v, o } = self.attention;
        let _ = gpu.free_tensor(norm);
        for weight in [q, k, v, o] {
            weight.free_all(gpu);
        }
        match self.mlp {
            Cohere2MlpWeights::Dense(expert) => expert.free_gpu(gpu),
            Cohere2MlpWeights::Sparse { router, experts } => {
                router.free_all(gpu);
                for expert in experts {
                    expert.free_gpu(gpu);
                }
            }
        }
    }
}

impl Cohere2ExpertWeights {
    fn free_gpu(self, gpu: &mut Gpu) {
        self.gate.free_all(gpu);
        self.up.free_all(gpu);
        self.down.free_all(gpu);
    }
}

pub(crate) struct Cohere2LayerState {
    kv: Option<KvCache>,
    next_pos: usize,
    pos: GpuTensor,
    x: GpuTensor,
    norm: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    attention: GpuTensor,
    attention_out: GpuTensor,
    gate: GpuTensor,
    up: GpuTensor,
    activation: GpuTensor,
    mlp: GpuTensor,
    expert_out: GpuTensor,
    output: GpuTensor,
    router_logits: Option<GpuTensor>,
    swa_staged_k: Option<GpuTensor>,
    swa_staged_v: Option<GpuTensor>,
    swa_nvalid: Option<GpuTensor>,
}

impl Cohere2LayerState {
    pub(crate) fn new(
        gpu: &mut Gpu,
        config: &Cohere2Config,
        mlp_kind: Cohere2MlpKind,
        max_seq: usize,
    ) -> Result<Self, CalibError> {
        Self::new_inner(gpu, config, mlp_kind, Some(max_seq.max(1)), None)
    }

    fn new_layered_scratch(
        gpu: &mut Gpu,
        config: &Cohere2Config,
        mlp_kind: Cohere2MlpKind,
        sliding_window: usize,
    ) -> Result<Self, CalibError> {
        Self::new_inner(gpu, config, mlp_kind, None, Some(sliding_window))
    }

    fn new_inner(
        gpu: &mut Gpu,
        config: &Cohere2Config,
        mlp_kind: Cohere2MlpKind,
        linear_max_seq: Option<usize>,
        sliding_window: Option<usize>,
    ) -> Result<Self, CalibError> {
        let q_width = config.q_heads * config.head_dim;
        let kv_width = config.kv_heads * config.head_dim;
        let intermediate = match mlp_kind {
            Cohere2MlpKind::Dense => config.dense_intermediate,
            Cohere2MlpKind::Sparse => config.expert_intermediate,
        };
        Ok(Self {
            kv: linear_max_seq
                .map(|max_seq| {
                    KvCache::new_gpu(gpu, 1, config.kv_heads, config.head_dim, max_seq)
                        .map_err(|error| CalibError::Runtime(error.to_string()))
                })
                .transpose()?,
            next_pos: 0,
            pos: gpu
                .alloc_tensor(&[1], DType::F32)
                .map_err(|error| CalibError::Runtime(error.to_string()))?,
            x: alloc_f32(gpu, config.hidden_size)?,
            norm: alloc_f32(gpu, config.hidden_size)?,
            q: alloc_f32(gpu, q_width)?,
            k: alloc_f32(gpu, kv_width)?,
            v: alloc_f32(gpu, kv_width)?,
            attention: alloc_f32(gpu, q_width)?,
            attention_out: alloc_f32(gpu, config.hidden_size)?,
            gate: alloc_f32(gpu, intermediate)?,
            up: alloc_f32(gpu, intermediate)?,
            activation: alloc_f32(gpu, intermediate)?,
            mlp: alloc_f32(gpu, config.hidden_size)?,
            expert_out: alloc_f32(gpu, config.hidden_size)?,
            output: alloc_f32(gpu, config.hidden_size)?,
            router_logits: (mlp_kind == Cohere2MlpKind::Sparse)
                .then(|| alloc_f32(gpu, config.num_experts))
                .transpose()?,
            swa_staged_k: sliding_window
                .map(|window| alloc_f32(gpu, kv_width * window))
                .transpose()?,
            swa_staged_v: sliding_window
                .map(|window| alloc_f32(gpu, kv_width * window))
                .transpose()?,
            swa_nvalid: sliding_window.map(|_| alloc_f32(gpu, 1)).transpose()?,
        })
    }

    pub(crate) fn reset(&mut self) {
        self.next_pos = 0;
    }

    pub(crate) fn free_gpu(self, gpu: &mut Gpu) {
        if let Some(kv) = self.kv {
            kv.free_gpu(gpu);
        }
        for tensor in [
            self.pos,
            self.x,
            self.norm,
            self.q,
            self.k,
            self.v,
            self.attention,
            self.attention_out,
            self.gate,
            self.up,
            self.activation,
            self.mlp,
            self.expert_out,
            self.output,
        ] {
            let _ = gpu.free_tensor(tensor);
        }
        for tensor in self
            .router_logits
            .into_iter()
            .chain(self.swa_staged_k)
            .chain(self.swa_staged_v)
            .chain(self.swa_nvalid)
        {
            let _ = gpu.free_tensor(tensor);
        }
    }

    pub(crate) fn output(&self) -> &GpuTensor {
        &self.output
    }
}

fn alloc_f32(gpu: &mut Gpu, elements: usize) -> Result<GpuTensor, CalibError> {
    gpu.alloc_tensor(&[elements], DType::F32)
        .map_err(|error| CalibError::Runtime(error.to_string()))
}

fn capture_activation(
    gpu: &mut Gpu,
    capture: Option<(&CalibCollector, &CaptureRegistry)>,
    layer: usize,
    role: ProjectionRole,
    expert: Option<usize>,
    input: &GpuTensor,
    width: usize,
) -> Result<(), CalibError> {
    let Some((collector, registry)) = capture else {
        return Ok(());
    };
    collector.capture_by_id(
        gpu,
        registry,
        CaptureId::new(layer, role, expert),
        input,
        1,
        width,
    )
}

fn topk_sigmoid(logits: &[f32], top_k: usize) -> Result<Vec<(usize, f32)>, CalibError> {
    if top_k == 0 || top_k > logits.len() || logits.iter().any(|value| !value.is_finite()) {
        return Err(CalibError::InvalidRouting(
            "Cohere2 router produced invalid logits or K-top geometry".into(),
        ));
    }
    let mut order = logits.iter().copied().enumerate().collect::<Vec<_>>();
    order.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    order.truncate(top_k);
    Ok(order
        .into_iter()
        .map(|(expert, logit)| (expert, 1.0 / (1.0 + (-logit).exp())))
        .collect())
}

pub struct Cohere2StreamedCalibrationLayer {
    logical_layer: usize,
    config: Cohere2Config,
    weights: Option<Cohere2LayerWeights>,
    sample_lengths: Vec<usize>,
    active_sequence_start: Option<usize>,
    states: Vec<Cohere2LayerState>,
    capture_registry: Option<Arc<CaptureRegistry>>,
    collector: Option<Arc<CalibCollector>>,
    telemetry: Option<ExpertTelemetry>,
}

impl Cohere2StreamedCalibrationLayer {
    fn load(
        reader: &mut PlannedTensorReader<'_, '_, '_>,
        gpu: &mut Gpu,
        config: &Cohere2Config,
        layer: usize,
        job: &CalibrationJob,
    ) -> Result<Self, CalibError> {
        if layer >= config.num_hidden_layers {
            return Err(CalibError::InvalidSourcePlan(format!(
                "Cohere2 layer {layer} exceeds {} layers",
                config.num_hidden_layers
            )));
        }
        if config.layer_kinds[layer] == Cohere2LayerKind::Sliding
            && job.samples.context_len() > config.sliding_window
        {
            return Err(CalibError::InvalidOptions(format!(
                "Cohere2 sliding layer {layer} requires calibration context <= {}, got {}",
                config.sliding_window,
                job.samples.context_len()
            )));
        }
        let weights = load_cohere2_streamed_layer(reader, gpu, config, layer)?;
        let telemetry = (config.mlp_kinds[layer] == Cohere2MlpKind::Sparse)
            .then(|| {
                ExpertTelemetry::new(
                    config.num_hidden_layers,
                    config.num_experts,
                    config.top_k,
                    job.options.expert_quota,
                    4096,
                )
            })
            .transpose()?;
        Ok(Self {
            logical_layer: layer,
            config: config.clone(),
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
                    "capture registry changed while a Cohere2 layer was resident".into(),
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
                "Cohere2 sequence range {}..{} is outside {} samples",
                batch.sequence_start,
                batch.sequence_end,
                self.sample_lengths.len()
            )));
        }
        self.release_states(gpu);
        let mlp_kind = self.config.mlp_kinds[self.logical_layer];
        let mut states = Vec::with_capacity(batch.sequence_end - batch.sequence_start);
        for &sample_len in &self.sample_lengths[batch.sequence_start..batch.sequence_end] {
            match Cohere2LayerState::new(gpu, &self.config, mlp_kind, sample_len) {
                Ok(state) => states.push(state),
                Err(error) => {
                    for state in states {
                        state.free_gpu(gpu);
                    }
                    return Err(error);
                }
            }
        }
        self.states = states;
        self.active_sequence_start = Some(batch.sequence_start);
        Ok(())
    }
}

impl CalibrationLayer for Cohere2StreamedCalibrationLayer {
    fn execute(
        &mut self,
        gpu: &mut Gpu,
        batch: &LayerMicrobatch,
        input_f32: &[f32],
        output_f32: &mut [f32],
        capture: &CaptureRegistry,
    ) -> Result<(), CalibError> {
        let hidden = self.config.hidden_size;
        let expected = batch.rows.len() * hidden;
        if batch.rows.is_empty() || input_f32.len() != expected || output_f32.len() != expected {
            return Err(CalibError::InvalidOptions(
                "Cohere2 layer boundary geometry is invalid".into(),
            ));
        }
        self.prepare_sequence_group(gpu, batch)?;
        self.prepare_capture(capture)?;
        let collector = Arc::clone(self.collector.as_ref().unwrap());
        let registry = Arc::clone(self.capture_registry.as_ref().unwrap());
        let weights = self.weights.as_ref().ok_or_else(|| {
            CalibError::Runtime("Cohere2 streamed weights were already freed".into())
        })?;
        for (row_index, row) in batch.rows.iter().enumerate() {
            let local = row
                .sample_index
                .checked_sub(batch.sequence_start)
                .ok_or_else(|| {
                    CalibError::InvalidOptions("Cohere2 sample is outside its group".into())
                })?;
            let state = self.states.get_mut(local).ok_or_else(|| {
                CalibError::InvalidOptions("Cohere2 sample is outside its group".into())
            })?;
            if row.reset_state {
                state.reset();
            }
            if state.next_pos != row.position {
                return Err(CalibError::InvalidOptions(format!(
                    "Cohere2 sample {} expected position {}, got {}",
                    row.sample_index, state.next_pos, row.position
                )));
            }
            let start = row_index * hidden;
            let end = start + hidden;
            execute_cohere2_row(
                gpu,
                &self.config,
                self.logical_layer,
                weights,
                state,
                None,
                &input_f32[start..end],
                Some((collector.as_ref(), registry.as_ref())),
                self.telemetry.as_mut(),
            )?;
            output_f32[start..end].copy_from_slice(
                &gpu.download_f32(&state.output)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            );
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
                "Cohere2 layer {} has no collector",
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

fn load_expert(
    reader: &mut PlannedTensorReader<'_, '_, '_>,
    gpu: &mut Gpu,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
) -> Result<Cohere2ExpertWeights, CalibError> {
    Ok(Cohere2ExpertWeights {
        gate: load_source_matrix(
            reader,
            gpu,
            &format!("{prefix}.gate_proj.weight"),
            intermediate,
            hidden,
        )?,
        up: load_source_matrix(
            reader,
            gpu,
            &format!("{prefix}.up_proj.weight"),
            intermediate,
            hidden,
        )?,
        down: load_source_matrix(
            reader,
            gpu,
            &format!("{prefix}.down_proj.weight"),
            hidden,
            intermediate,
        )?,
    })
}

fn load_cohere2_streamed_layer(
    reader: &mut PlannedTensorReader<'_, '_, '_>,
    gpu: &mut Gpu,
    config: &Cohere2Config,
    layer: usize,
) -> Result<Cohere2LayerWeights, CalibError> {
    let prefix = format!("model.layers.{layer}");
    let attention_prefix = format!("{prefix}.self_attn");
    let q_width = config.q_heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    let attention = Cohere2AttentionWeights {
        norm: load_source_f32_tensor(
            reader,
            gpu,
            &format!("{prefix}.input_layernorm.weight"),
            config.hidden_size,
            false,
        )?,
        q: load_source_matrix(
            reader,
            gpu,
            &format!("{attention_prefix}.q_proj.weight"),
            q_width,
            config.hidden_size,
        )?,
        k: load_source_matrix(
            reader,
            gpu,
            &format!("{attention_prefix}.k_proj.weight"),
            kv_width,
            config.hidden_size,
        )?,
        v: load_source_matrix(
            reader,
            gpu,
            &format!("{attention_prefix}.v_proj.weight"),
            kv_width,
            config.hidden_size,
        )?,
        o: load_source_matrix(
            reader,
            gpu,
            &format!("{attention_prefix}.o_proj.weight"),
            config.hidden_size,
            q_width,
        )?,
    };
    let mlp = match config.mlp_kinds[layer] {
        Cohere2MlpKind::Dense => Cohere2MlpWeights::Dense(load_expert(
            reader,
            gpu,
            &format!("{prefix}.mlp"),
            config.hidden_size,
            config.dense_intermediate,
        )?),
        Cohere2MlpKind::Sparse => {
            let router = load_source_matrix(
                reader,
                gpu,
                &format!("{prefix}.mlp.gate.weight"),
                config.num_experts,
                config.hidden_size,
            )?;
            let mut experts = Vec::with_capacity(config.num_experts);
            for expert in 0..config.num_experts {
                experts.push(load_expert(
                    reader,
                    gpu,
                    &format!("{prefix}.mlp.experts.{expert}"),
                    config.hidden_size,
                    config.expert_intermediate,
                )?);
            }
            Cohere2MlpWeights::Sparse { router, experts }
        }
    };
    Ok(Cohere2LayerWeights { attention, mlp })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_cohere2_row(
    gpu: &mut Gpu,
    config: &Cohere2Config,
    layer: usize,
    weights: &Cohere2LayerWeights,
    state: &mut Cohere2LayerState,
    layered_kv: Option<&LayeredKvArena>,
    input: &[f32],
    capture: Option<(&CalibCollector, &CaptureRegistry)>,
    mut telemetry: Option<&mut ExpertTelemetry>,
) -> Result<(), CalibError> {
    let hip = |error: hip_bridge::HipError| CalibError::Runtime(error.to_string());
    let q_width = config.q_heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    gpu.hip
        .memcpy_htod(&state.x.buf, f32_slice_as_bytes(input))
        .map_err(hip)?;
    gpu.rmsnorm_f32(
        &state.x,
        &weights.attention.norm,
        &state.norm,
        config.norm_eps,
    )
    .map_err(hip)?;
    capture_activation(
        gpu,
        capture,
        layer,
        ProjectionRole::QueryInput,
        None,
        &state.norm,
        config.hidden_size,
    )?;
    weight_gemv(gpu, &weights.attention.q, &state.norm, &state.q).map_err(hip)?;
    weight_gemv(gpu, &weights.attention.k, &state.norm, &state.k).map_err(hip)?;
    weight_gemv(gpu, &weights.attention.v, &state.norm, &state.v).map_err(hip)?;

    if hipfire_runtime::triattn::tap_enabled() {
        let q = gpu.download_f32(&state.q).map_err(hip)?;
        hipfire_runtime::triattn::record_prerope_q(layer, &q[..q_width]);
    }
    let position = i32::try_from(state.next_pos).map_err(|_| {
        CalibError::InvalidOptions("Cohere2 calibration position exceeds i32".into())
    })?;
    gpu.hip
        .memcpy_htod(&state.pos.buf, &position.to_ne_bytes())
        .map_err(hip)?;
    if config.force_rope[layer] {
        gpu.rope_tail_interleaved(
            &state.q,
            &state.k,
            &state.pos,
            config.q_heads as i32,
            config.kv_heads as i32,
            config.head_dim as i32,
            config.head_dim as i32,
            config.rope_theta,
        )
        .map_err(hip)?;
    }
    if let Some(arena) = layered_kv {
        let cache = arena
            .view(layer, state.next_pos)
            .map_err(CalibError::Runtime)?;
        match arena.plan().layers()[cache.producer_layer].storage {
            KvStorageKind::SlidingWindow { window } => {
                let staged_k = state.swa_staged_k.as_ref().ok_or_else(|| {
                    CalibError::Runtime("Cohere2 layered state lacks staged K".into())
                })?;
                let staged_v = state.swa_staged_v.as_ref().ok_or_else(|| {
                    CalibError::Runtime("Cohere2 layered state lacks staged V".into())
                })?;
                let nvalid = state.swa_nvalid.as_ref().ok_or_else(|| {
                    CalibError::Runtime("Cohere2 layered state lacks SWA length".into())
                })?;
                let visible = (state.next_pos + 1).min(window) as i32;
                gpu.hip
                    .memcpy_htod(&nvalid.buf, &visible.to_ne_bytes())
                    .map_err(hip)?;
                let head_window = config.head_dim * window;
                for head in 0..config.kv_heads {
                    gpu.swa_visibility_stage_batched(
                        &cache.k.sub_offset(head * head_window, head_window),
                        &state.k.sub_offset(head * config.head_dim, config.head_dim),
                        &staged_k.sub_offset(head * head_window, head_window),
                        state.next_pos as i32,
                        window as i32,
                        config.head_dim as i32,
                        1,
                    )
                    .map_err(hip)?;
                    gpu.swa_visibility_stage_batched(
                        &cache.v.sub_offset(head * head_window, head_window),
                        &state.v.sub_offset(head * config.head_dim, config.head_dim),
                        &staged_v.sub_offset(head * head_window, head_window),
                        state.next_pos as i32,
                        window as i32,
                        config.head_dim as i32,
                        1,
                    )
                    .map_err(hip)?;
                }
                gpu.attention_swa_gqa_batched(
                    &state.q,
                    staged_k,
                    staged_v,
                    nvalid,
                    &state.attention,
                    config.q_heads,
                    config.kv_heads,
                    config.head_dim,
                    window,
                    1,
                    1.0 / (config.head_dim as f32).sqrt(),
                )
                .map_err(hip)?;
                for head in 0..config.kv_heads {
                    gpu.swa_ring_write_batched_f32(
                        &state.k.sub_offset(head * config.head_dim, config.head_dim),
                        &cache.k.sub_offset(head * head_window, head_window),
                        1,
                        config.head_dim as i32,
                        window as i32,
                        state.next_pos as i32,
                        1,
                    )
                    .map_err(hip)?;
                    gpu.swa_ring_write_batched_f32(
                        &state.v.sub_offset(head * config.head_dim, config.head_dim),
                        &cache.v.sub_offset(head * head_window, head_window),
                        1,
                        config.head_dim as i32,
                        window as i32,
                        state.next_pos as i32,
                        1,
                    )
                    .map_err(hip)?;
                }
            }
            KvStorageKind::Full => {
                let physical = i32::try_from(cache.physical_position).map_err(|_| {
                    CalibError::InvalidOptions("Cohere2 physical position exceeds i32".into())
                })?;
                gpu.hip
                    .memcpy_htod(&state.pos.buf, &physical.to_ne_bytes())
                    .map_err(hip)?;
                gpu.kv_cache_write(cache.k, &state.k, &state.pos.buf, kv_width)
                    .map_err(hip)?;
                gpu.kv_cache_write(cache.v, &state.v, &state.pos.buf, kv_width)
                    .map_err(hip)?;
                gpu.attention_f32(
                    &state.q,
                    cache.k,
                    cache.v,
                    &state.attention,
                    &state.pos.buf,
                    cache.visible_positions.len(),
                    config.q_heads,
                    config.kv_heads,
                    config.head_dim,
                    arena.plan().max_seq(),
                )
                .map_err(hip)?;
            }
        }
    } else {
        let kv = state
            .kv
            .as_ref()
            .ok_or_else(|| CalibError::Runtime("Cohere2 linear state lacks KV cache".into()))?;
        gpu.kv_cache_write(&kv.k_gpu[0], &state.k, &state.pos.buf, kv_width)
            .map_err(hip)?;
        gpu.kv_cache_write(&kv.v_gpu[0], &state.v, &state.pos.buf, kv_width)
            .map_err(hip)?;
        gpu.attention_f32(
            &state.q,
            &kv.k_gpu[0],
            &kv.v_gpu[0],
            &state.attention,
            &state.pos.buf,
            state.next_pos + 1,
            config.q_heads,
            config.kv_heads,
            config.head_dim,
            kv.max_seq,
        )
        .map_err(hip)?;
    }
    capture_activation(
        gpu,
        capture,
        layer,
        ProjectionRole::AttentionOutputInput,
        None,
        &state.attention,
        q_width,
    )?;
    weight_gemv(
        gpu,
        &weights.attention.o,
        &state.attention,
        &state.attention_out,
    )
    .map_err(hip)?;

    match &weights.mlp {
        Cohere2MlpWeights::Dense(expert) => {
            capture_activation(
                gpu,
                capture,
                layer,
                ProjectionRole::DenseMlpInput,
                None,
                &state.norm,
                config.hidden_size,
            )?;
            weight_gemv(gpu, &expert.gate, &state.norm, &state.gate).map_err(hip)?;
            weight_gemv(gpu, &expert.up, &state.norm, &state.up).map_err(hip)?;
            gpu.silu_mul_f32(&state.gate, &state.up, &state.activation)
                .map_err(hip)?;
            capture_activation(
                gpu,
                capture,
                layer,
                ProjectionRole::DownInput,
                None,
                &state.activation,
                config.dense_intermediate,
            )?;
            weight_gemv(gpu, &expert.down, &state.activation, &state.mlp).map_err(hip)?;
        }
        Cohere2MlpWeights::Sparse { router, experts } => {
            capture_activation(
                gpu,
                capture,
                layer,
                ProjectionRole::RouterInput,
                None,
                &state.norm,
                config.hidden_size,
            )?;
            let router_logits = state.router_logits.as_ref().ok_or_else(|| {
                CalibError::Runtime("Cohere2 sparse state lacks router logits".into())
            })?;
            weight_gemv(gpu, router, &state.norm, router_logits).map_err(hip)?;
            let logits = gpu.download_f32(router_logits).map_err(hip)?;
            let selected = topk_sigmoid(&logits, config.top_k)?;
            if let Some(telemetry) = telemetry.as_deref_mut() {
                let indices = selected
                    .iter()
                    .map(|(expert, _)| *expert)
                    .collect::<Vec<_>>();
                let route_weights = selected
                    .iter()
                    .map(|(_, weight)| *weight)
                    .collect::<Vec<_>>();
                telemetry.record_router_selection(layer, &indices, &route_weights)?;
                telemetry.record_grouped_batch_shape(
                    layer,
                    indices.len(),
                    indices.len(),
                    indices.len(),
                )?;
            }
            gpu.hip
                .memset(&state.mlp.buf, 0, state.mlp.byte_size())
                .map_err(hip)?;
            for (expert_index, route_weight) in selected {
                let expert = &experts[expert_index];
                let capture_gate_up = match telemetry.as_deref_mut() {
                    Some(telemetry) => {
                        telemetry.record_capture_route(
                            layer,
                            expert_index,
                            ExpertCaptureRole::GateUpInput,
                            route_weight,
                        )? == CaptureAdmission::Capture
                    }
                    None => capture.is_some(),
                };
                if capture_gate_up {
                    capture_activation(
                        gpu,
                        capture,
                        layer,
                        ProjectionRole::GateUpInput,
                        Some(expert_index),
                        &state.norm,
                        config.hidden_size,
                    )?;
                    if let Some(telemetry) = telemetry.as_deref_mut() {
                        telemetry.record_direct_capture_launch(
                            layer,
                            expert_index,
                            ExpertCaptureRole::GateUpInput,
                        )?;
                    }
                }
                weight_gemv(gpu, &expert.gate, &state.norm, &state.gate).map_err(hip)?;
                weight_gemv(gpu, &expert.up, &state.norm, &state.up).map_err(hip)?;
                gpu.silu_mul_f32(&state.gate, &state.up, &state.activation)
                    .map_err(hip)?;
                let capture_down = match telemetry.as_deref_mut() {
                    Some(telemetry) => {
                        telemetry.record_capture_route(
                            layer,
                            expert_index,
                            ExpertCaptureRole::DownInput,
                            route_weight,
                        )? == CaptureAdmission::Capture
                    }
                    None => capture.is_some(),
                };
                if capture_down {
                    capture_activation(
                        gpu,
                        capture,
                        layer,
                        ProjectionRole::DownInput,
                        Some(expert_index),
                        &state.activation,
                        config.expert_intermediate,
                    )?;
                    if let Some(telemetry) = telemetry.as_deref_mut() {
                        telemetry.record_direct_capture_launch(
                            layer,
                            expert_index,
                            ExpertCaptureRole::DownInput,
                        )?;
                    }
                }
                weight_gemv(gpu, &expert.down, &state.activation, &state.expert_out)
                    .map_err(hip)?;
                gpu.scaled_add_inplace_cpu_scalar_f32(&state.mlp, &state.expert_out, route_weight)
                    .map_err(hip)?;
            }
        }
    }
    gpu.add_f32(&state.x, &state.attention_out, &state.output)
        .map_err(hip)?;
    gpu.add_f32(&state.output, &state.mlp, &state.output)
        .map_err(hip)?;
    state.next_pos += 1;
    Ok(())
}

pub(crate) struct Cohere2ResidentWeights {
    token_embedding: GpuTensor,
    embedding_format: EmbeddingFormat,
    final_norm: GpuTensor,
    lm_head: WeightTensor,
    layers: Vec<Cohere2LayerWeights>,
}

impl Cohere2ResidentWeights {
    pub(crate) fn load(
        hfq: &HfqFile,
        gpu: &mut Gpu,
        config: &Cohere2Config,
    ) -> Result<Self, CalibError> {
        let loader = TransformerLoader::new(hfq, "cohere2-moe");
        let (token_embedding, embedding_format) = loader
            .load_embedding(gpu, EMBEDDING, config.vocab_size, config.hidden_size)
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        let final_norm = loader
            .load_direct_f32(gpu, FINAL_NORM, &[config.hidden_size])
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        let (lm_head, _) = loader
            .load_lm_head(
                gpu,
                EMBEDDING,
                LM_HEAD,
                config.tie_word_embeddings,
                config.vocab_size,
                config.hidden_size,
            )
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            let attention = Cohere2AttentionWeights {
                norm: loader
                    .load_direct_f32(
                        gpu,
                        &format!("{prefix}.input_layernorm.weight"),
                        &[config.hidden_size],
                    )
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
                q: loader
                    .load_weight(
                        gpu,
                        &format!("{prefix}.self_attn.q_proj.weight"),
                        config.q_heads * config.head_dim,
                        config.hidden_size,
                    )
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
                k: loader
                    .load_weight(
                        gpu,
                        &format!("{prefix}.self_attn.k_proj.weight"),
                        config.kv_heads * config.head_dim,
                        config.hidden_size,
                    )
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
                v: loader
                    .load_weight(
                        gpu,
                        &format!("{prefix}.self_attn.v_proj.weight"),
                        config.kv_heads * config.head_dim,
                        config.hidden_size,
                    )
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
                o: loader
                    .load_weight(
                        gpu,
                        &format!("{prefix}.self_attn.o_proj.weight"),
                        config.hidden_size,
                        config.q_heads * config.head_dim,
                    )
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            };
            let load_expert = |gpu: &mut Gpu,
                               expert_prefix: &str,
                               intermediate: usize|
             -> Result<Cohere2ExpertWeights, CalibError> {
                Ok(Cohere2ExpertWeights {
                    gate: loader
                        .load_weight(
                            gpu,
                            &format!("{expert_prefix}.gate_proj.weight"),
                            intermediate,
                            config.hidden_size,
                        )
                        .map_err(|error| CalibError::Runtime(error.to_string()))?,
                    up: loader
                        .load_weight(
                            gpu,
                            &format!("{expert_prefix}.up_proj.weight"),
                            intermediate,
                            config.hidden_size,
                        )
                        .map_err(|error| CalibError::Runtime(error.to_string()))?,
                    down: loader
                        .load_weight(
                            gpu,
                            &format!("{expert_prefix}.down_proj.weight"),
                            config.hidden_size,
                            intermediate,
                        )
                        .map_err(|error| CalibError::Runtime(error.to_string()))?,
                })
            };
            let mlp = match config.mlp_kinds[layer] {
                Cohere2MlpKind::Dense => Cohere2MlpWeights::Dense(load_expert(
                    gpu,
                    &format!("{prefix}.mlp"),
                    config.dense_intermediate,
                )?),
                Cohere2MlpKind::Sparse => {
                    let router = loader
                        .load_weight(
                            gpu,
                            &format!("{prefix}.mlp.gate.weight"),
                            config.num_experts,
                            config.hidden_size,
                        )
                        .map_err(|error| CalibError::Runtime(error.to_string()))?;
                    let mut experts = Vec::with_capacity(config.num_experts);
                    for expert in 0..config.num_experts {
                        experts.push(load_expert(
                            gpu,
                            &format!("{prefix}.mlp.experts.{expert}"),
                            config.expert_intermediate,
                        )?);
                    }
                    Cohere2MlpWeights::Sparse { router, experts }
                }
            };
            layers.push(Cohere2LayerWeights { attention, mlp });
        }
        Ok(Self {
            token_embedding,
            embedding_format,
            final_norm,
            lm_head,
            layers,
        })
    }

    pub(crate) fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.token_embedding);
        let _ = gpu.free_tensor(self.final_norm);
        self.lm_head.free_all(gpu);
        for layer in self.layers {
            layer.free_gpu(gpu);
        }
    }
}

pub(crate) struct Cohere2ResidentState {
    layers: Vec<Cohere2LayerState>,
    kv: LayeredKvArena,
    embedding: GpuTensor,
    final_hidden: GpuTensor,
    normalized: GpuTensor,
    logits: GpuTensor,
}

impl Cohere2ResidentState {
    pub(crate) fn new(
        gpu: &mut Gpu,
        config: &Cohere2Config,
        max_seq: usize,
        physical_cap: usize,
    ) -> Result<Self, CalibError> {
        let plan = config
            .layered_kv_plan(max_seq)
            .map_err(CalibError::InvalidOptions)?;
        let kv = LayeredKvArena::new_fp32_capped(gpu, plan, physical_cap)
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for &kind in &config.mlp_kinds {
            match Cohere2LayerState::new_layered_scratch(
                gpu,
                config,
                kind,
                config.sliding_window.min(max_seq),
            ) {
                Ok(state) => layers.push(state),
                Err(error) => {
                    for state in layers {
                        state.free_gpu(gpu);
                    }
                    kv.free_gpu(gpu);
                    return Err(error);
                }
            }
        }
        let mut globals = Vec::with_capacity(4);
        for elements in [
            config.hidden_size,
            config.hidden_size,
            config.hidden_size,
            config.vocab_size,
        ] {
            match alloc_f32(gpu, elements) {
                Ok(tensor) => globals.push(tensor),
                Err(error) => {
                    for tensor in globals {
                        let _ = gpu.free_tensor(tensor);
                    }
                    for state in layers {
                        state.free_gpu(gpu);
                    }
                    kv.free_gpu(gpu);
                    return Err(error);
                }
            }
        }
        let mut globals = globals.into_iter();
        Ok(Self {
            layers,
            kv,
            embedding: globals.next().unwrap(),
            final_hidden: globals.next().unwrap(),
            normalized: globals.next().unwrap(),
            logits: globals.next().unwrap(),
        })
    }

    pub(crate) fn reset(&mut self) {
        self.kv.reset();
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    pub(crate) fn next_pos(&self) -> usize {
        self.kv.next_pos()
    }

    pub(crate) fn logits(&self) -> &GpuTensor {
        &self.logits
    }

    pub(crate) fn build_eviction(
        &self,
        gpu: &mut Gpu,
        artifact: &TriAttnArtifact,
        budget: usize,
        beta: usize,
    ) -> Result<LayeredEvictionCtx, String> {
        LayeredEvictionCtx::new(gpu, artifact, &self.kv, budget, beta)
    }

    pub(crate) fn maybe_evict(
        &mut self,
        gpu: &mut Gpu,
        eviction: &LayeredEvictionCtx,
    ) -> Result<Option<EvictionResult>, CalibError> {
        eviction
            .maybe_evict(gpu, &mut self.kv)
            .map_err(|error| CalibError::Runtime(error.to_string()))
    }

    pub(crate) fn free_gpu(self, gpu: &mut Gpu) {
        self.kv.free_gpu(gpu);
        for layer in self.layers {
            layer.free_gpu(gpu);
        }
        for tensor in [
            self.embedding,
            self.final_hidden,
            self.normalized,
            self.logits,
        ] {
            let _ = gpu.free_tensor(tensor);
        }
    }
}

pub(crate) fn forward_resident_token(
    gpu: &mut Gpu,
    config: &Cohere2Config,
    weights: &Cohere2ResidentWeights,
    state: &mut Cohere2ResidentState,
    token: u32,
) -> Result<(), CalibError> {
    if token as usize >= config.vocab_size {
        return Err(CalibError::InvalidSamples(format!(
            "token {token} exceeds Cohere2 vocabulary {}",
            config.vocab_size
        )));
    }
    let hip = |error: hip_bridge::HipError| CalibError::Runtime(error.to_string());
    match weights.embedding_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(
            &weights.token_embedding,
            &state.embedding,
            token,
            config.hidden_size,
        ),
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(
            &weights.token_embedding,
            &state.embedding,
            token,
            config.hidden_size,
        ),
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(
            &weights.token_embedding,
            &state.embedding,
            token,
            config.hidden_size,
        ),
        EmbeddingFormat::BF16 => gpu.embedding_lookup_bf16(
            &weights.token_embedding,
            &state.embedding,
            token,
            config.hidden_size,
        ),
        EmbeddingFormat::F16 => gpu.embedding_lookup_f16(
            &weights.token_embedding,
            &state.embedding,
            token,
            config.hidden_size,
        ),
        EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(
            &weights.token_embedding,
            &state.embedding,
            token,
            config.hidden_size,
        ),
        EmbeddingFormat::F32 => gpu.embedding_lookup(
            &weights.token_embedding,
            &state.embedding,
            token,
            config.hidden_size,
        ),
    }
    .map_err(hip)?;
    let position = state.kv.next_pos();
    let mut hidden = gpu.download_f32(&state.embedding).map_err(hip)?;
    for (layer, (layer_weights, layer_state)) in
        weights.layers.iter().zip(&mut state.layers).enumerate()
    {
        execute_cohere2_row(
            gpu,
            config,
            layer,
            layer_weights,
            layer_state,
            Some(&state.kv),
            &hidden,
            None,
            None,
        )?;
        hidden = gpu.download_f32(layer_state.output()).map_err(hip)?;
    }
    state
        .kv
        .advance(position)
        .map_err(CalibError::InvalidOptions)?;
    gpu.hip
        .memcpy_htod(&state.final_hidden.buf, f32_slice_as_bytes(&hidden))
        .map_err(hip)?;
    gpu.rmsnorm_f32(
        &state.final_hidden,
        &weights.final_norm,
        &state.normalized,
        config.norm_eps,
    )
    .map_err(hip)?;
    weight_gemv(gpu, &weights.lm_head, &state.normalized, &state.logits).map_err(hip)?;
    Ok(())
}

impl CalibrationFamilyAdapter for Cohere2CalibrationAdapter {
    fn family(&self) -> &'static str {
        "cohere2-moe"
    }

    fn adapter_version(&self) -> &'static str {
        "cohere2-moe-stream-v1"
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
        let inspection = inspect_cohere2_stream_source(source)?;
        self.source_dtypes = inspection
            .tensor_requests
            .iter()
            .filter_map(|request| source.tensor_info(&request.source_name))
            .map(|info| info.dtype.clone())
            .collect();
        self.source_dtypes.sort();
        self.source_dtypes.dedup();
        self.config = Some(
            Cohere2Config::from_json_str(source.metadata_json())
                .map_err(CalibError::InvalidSourcePlan)?,
        );
        Ok(inspection)
    }

    fn capture_plan(
        &self,
        _model: &ModelInspection,
        job: &CalibrationJob,
    ) -> Result<CaptureRegistry, CalibError> {
        capture_registry(self.config()?, job)
    }

    fn cask_metadata(
        &self,
        model: &ModelInspection,
        job: &CalibrationJob,
    ) -> Result<Option<TriAttnPackageMetadata>, CalibError> {
        let config = self.config()?;
        let layers = config
            .layer_kinds
            .iter()
            .enumerate()
            .map(|(layer, kind)| {
                let roped = config.force_rope[layer];
                TriAttnLayerRecord {
                    physical_layer: layer as u32,
                    attention_kind: match kind {
                        Cohere2LayerKind::Full => TriAttnAttentionKind::Full,
                        Cohere2LayerKind::Sliding => TriAttnAttentionKind::Sliding,
                    },
                    q_heads: config.q_heads as u32,
                    kv_heads: config.kv_heads as u32,
                    head_dim: config.head_dim as u32,
                    rotary_dim: if roped { config.head_dim as u32 } else { 0 },
                    rope_theta: config.rope_theta,
                    rope_convention: if roped {
                        TriAttnRopeConvention::Interleaved
                    } else {
                        TriAttnRopeConvention::None
                    },
                    context_policy: match kind {
                        Cohere2LayerKind::Full => TriAttnContextPolicy::Full,
                        Cohere2LayerKind::Sliding => TriAttnContextPolicy::Sliding,
                    },
                    sliding_window: (*kind == Cohere2LayerKind::Sliding)
                        .then_some(config.sliding_window as u32),
                    kv_producer: None,
                    center_tensor: format!("triattn.layers.{layer}.centers"),
                    center_offset: 0,
                    center_count: (config.q_heads * (config.head_dim / 2)) as u64,
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
        let token_ids = gpu
            .alloc_tensor(&[job.options.max_rows], DType::F32)
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        let output = gpu
            .alloc_tensor(&[job.options.max_rows * model.hidden_width], DType::F32)
            .map_err(|error| CalibError::Runtime(error.to_string()))?;
        Ok(Box::new(Cohere2CalibrationEmbedding {
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
        Ok(Box::new(Cohere2StreamedCalibrationLayer::load(
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
            self.config()?.norm_eps,
            job.options.max_rows,
            None,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_router_selects_raw_topk_without_renormalizing() {
        let selected = topk_sigmoid(&[-2.0, 2.0, 0.0, 1.0], 2).unwrap();
        assert_eq!(
            selected.iter().map(|item| item.0).collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert!((selected[0].1 - 0.880_797).abs() < 1e-6);
        assert!((selected[1].1 - 0.731_058_6).abs() < 1e-6);
        assert!((selected.iter().map(|item| item.1).sum::<f32>() - 1.0).abs() > 0.5);
    }

    #[test]
    fn sigmoid_router_rejects_nonfinite_logits() {
        assert!(topk_sigmoid(&[0.0, f32::NAN], 1).is_err());
    }
}
