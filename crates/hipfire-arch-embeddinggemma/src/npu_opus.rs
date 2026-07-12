use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use hip_bridge::{HipError, HipResult};
use hipfire_rdna::{
    DType, Gpu, GpuTensor, ImportedTensor, OpusNpuIoLayout, OwnedTensor, SharedGttBuffer,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::quant::f16_to_f32;
use hipfire_runtime::weights::{weight_gemm, WeightTensor};
use hipfire_xdna::{
    NpuAttentionOutputBf16, NpuAttentionOutputBf16Weights, NpuEmbeddingFinalNormMean,
    NpuEmbeddingFinalNormMeanParams, NpuEmbeddingLayerAttentionDenseW8,
    NpuEmbeddingLayerAttentionDenseW8Weights, NpuEmbeddingNextLayerPrepW8,
    NpuEmbeddingNextLayerPrepW8Params, NpuEmbeddingPostFfnDirectTailBf16x2,
    NpuEmbeddingPostFfnDirectTailBf16x2Params, NpuEmbeddingPreFfnException,
    NpuEmbeddingResidualPrep, NpuOpusExecutor, NpuResidentAttentionDenseW8,
    NpuResidentAttentionDenseW8Weights, NpuResidentFfnDenseW8, NpuResidentFfnDenseW8Weights,
    NpuResidentFfnW4, NpuResidentFfnW4Weights, NpuWholeMode, NpuWholeScaledIoLayout,
    OpusMatrixEncoding, OpusPackedMatrix, OpusResidentMode,
};

use crate::config::{EmbeddingGemmaConfig, PoolingMode};
use crate::forward::{AttentionBoundary, LinearProjector, Projection};

struct LayerMatrices {
    qkv: Option<OpusPackedMatrix>,
    query: OpusPackedMatrix,
    key: OpusPackedMatrix,
    value: OpusPackedMatrix,
    attention_output: OpusPackedMatrix,
    gate: OpusPackedMatrix,
    up: OpusPackedMatrix,
    gate_up: Option<OpusPackedMatrix>,
    down: Option<OpusPackedMatrix>,
}

/// Format-generic EmbeddingGemma projector for W4, mixed, and W8 Opus matrices.
/// All q/k/v/o/gate/up and Opus-encoded down projections execute on XDNA while
/// the surrounding attention, norms, residuals, and activations remain on the
/// GPU bridge.
pub struct NpuOpusProjector {
    executors: HashMap<usize, NpuOpusExecutor>,
    layers: Vec<LayerMatrices>,
    shared_io: HashMap<SharedIoKey, SharedProjectionIo>,
    awq_gpu: HashMap<MatrixGpuKey, OwnedTensor>,
    resident_attention: Option<ResidentAttentionState>,
    resident_attention_selected: bool,
    resident_ffn: Option<ResidentFfnState>,
    resident_ffn_selected: bool,
    resident_layer: Option<ResidentLayerState>,
    debug_resident_hidden: Option<Vec<f32>>,
    debug_resident_residual: Option<Vec<f32>>,
    debug_resident_exception: Option<(usize, Vec<f32>)>,
    debug_resident_ffn: Option<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SharedIoKey {
    mode: NpuWholeMode,
    k: usize,
    n: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum MatrixRole {
    Qkv,
    Query,
    Key,
    Value,
    AttentionOutput,
    Gate,
    Up,
    GateUp,
    Down,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MatrixGpuKey {
    layer: usize,
    role: MatrixRole,
}

struct SharedProjectionIo {
    input_gpu: ImportedTensor,
    output_gpu: ImportedTensor,
    _input_buffer: SharedGttBuffer,
    _output_buffer: SharedGttBuffer,
}

struct SharedAttentionInput {
    input_gpu: ImportedTensor,
    _input_buffer: SharedGttBuffer,
}

struct ResidentAttentionState {
    executor: NpuResidentAttentionDenseW8,
    weights: Vec<NpuResidentAttentionDenseW8Weights>,
    input: Option<SharedAttentionInput>,
    output: Option<ResidentAttentionOutputState>,
}

struct ResidentAttentionOutputState {
    executor: NpuAttentionOutputBf16,
    weights: Vec<NpuAttentionOutputBf16Weights>,
    staging: Vec<Option<SharedGttBuffer>>,
    io: Option<SharedAttentionOutput>,
}

struct SharedAttentionOutput {
    output_gpu: ImportedTensor,
    _output_buffer: SharedGttBuffer,
}

struct ResidentLayerState {
    attention: ResidentLayerAttentionExecutors,
    attention_weights: Vec<ResidentLayerAttentionWeights>,
    ffn: NpuResidentFfnDenseW8,
    ffn_weights: Vec<NpuResidentFfnDenseW8Weights>,
    tail: NpuEmbeddingPostFfnDirectTailBf16x2,
    tail_params: Vec<NpuEmbeddingPostFfnDirectTailBf16x2Params>,
    next_prep: Option<NpuEmbeddingNextLayerPrepW8>,
    residual_prep: Option<NpuEmbeddingResidualPrep>,
    final_norm_mean: Option<NpuEmbeddingFinalNormMean>,
    final_norm_mean_params: Option<NpuEmbeddingFinalNormMeanParams>,
    next_prep_params: Vec<NpuEmbeddingNextLayerPrepW8Params>,
    prepared_input_layer: Option<usize>,
    tail_pre_norms: Vec<Vec<u16>>,
    tail_post_norms: Vec<Vec<u16>>,
    io: Option<ResidentLayerIo>,
}

struct ResidentLayerAttentionExecutors {
    standard: NpuEmbeddingLayerAttentionDenseW8,
    exception_39: Option<NpuEmbeddingLayerAttentionDenseW8>,
    exception_731: Option<NpuEmbeddingLayerAttentionDenseW8>,
}

enum ResidentLayerAttentionWeights {
    Standard(NpuEmbeddingLayerAttentionDenseW8Weights),
    Exception39(NpuEmbeddingLayerAttentionDenseW8Weights),
    Exception731(NpuEmbeddingLayerAttentionDenseW8Weights),
    Unavailable,
}

impl ResidentLayerAttentionWeights {
    fn awq_scale(&self) -> Option<&[f32]> {
        match self {
            Self::Standard(weights) | Self::Exception39(weights) | Self::Exception731(weights) => {
                weights.awq_scale()
            }
            Self::Unavailable => None,
        }
    }

    fn is_available(&self) -> bool {
        !matches!(self, Self::Unavailable)
    }
}

impl ResidentLayerAttentionExecutors {
    fn input_bytes(&self) -> usize {
        self.standard.input_bytes()
    }

    fn uses_external_residual(&self) -> bool {
        self.standard.uses_external_residual()
    }

    fn attach_shared_input(
        &mut self,
        fd: i32,
        bytes: usize,
    ) -> Result<(), hipfire_xdna::XdnaError> {
        self.standard.attach_shared_input(fd, bytes)?;
        if let Some(executor) = &mut self.exception_39 {
            executor.attach_shared_input(fd, bytes)?;
        }
        if let Some(executor) = &mut self.exception_731 {
            executor.attach_shared_input(fd, bytes)?;
        }
        Ok(())
    }

    fn attach_shared_hidden(
        &mut self,
        fd: i32,
        bytes: usize,
    ) -> Result<(), hipfire_xdna::XdnaError> {
        self.standard.attach_shared_hidden(fd, bytes)?;
        if let Some(executor) = &mut self.exception_39 {
            executor.attach_shared_hidden(fd, bytes)?;
        }
        if let Some(executor) = &mut self.exception_731 {
            executor.attach_shared_hidden(fd, bytes)?;
        }
        Ok(())
    }

    fn set_residual_bf16(
        &self,
        weights: &mut ResidentLayerAttentionWeights,
        residual: &[u16],
    ) -> Result<(), hipfire_xdna::XdnaError> {
        match weights {
            ResidentLayerAttentionWeights::Standard(weights) => {
                self.standard.set_residual_bf16(weights, residual)
            }
            ResidentLayerAttentionWeights::Exception39(weights) => self
                .exception_39
                .as_ref()
                .expect("exception-39 weights require executor")
                .set_residual_bf16(weights, residual),
            ResidentLayerAttentionWeights::Exception731(weights) => self
                .exception_731
                .as_ref()
                .expect("exception-731 weights require executor")
                .set_residual_bf16(weights, residual),
            ResidentLayerAttentionWeights::Unavailable => {
                unreachable!("unavailable resident layer rejected before dispatch")
            }
        }
    }

    fn prepare_layer(
        &mut self,
        weights: &ResidentLayerAttentionWeights,
    ) -> Result<(), hipfire_xdna::XdnaError> {
        match weights {
            ResidentLayerAttentionWeights::Standard(weights) => {
                self.standard.prepare_layer(weights)
            }
            ResidentLayerAttentionWeights::Exception39(weights) => self
                .exception_39
                .as_mut()
                .expect("exception-39 weights require executor")
                .prepare_layer(weights),
            ResidentLayerAttentionWeights::Exception731(weights) => self
                .exception_731
                .as_mut()
                .expect("exception-731 weights require executor")
                .prepare_layer(weights),
            ResidentLayerAttentionWeights::Unavailable => {
                unreachable!("unavailable resident layer rejected before dispatch")
            }
        }
    }

    fn run_shared(
        &mut self,
        weights: &ResidentLayerAttentionWeights,
    ) -> Result<(), hipfire_xdna::XdnaError> {
        match weights {
            ResidentLayerAttentionWeights::Standard(weights) => self.standard.run_shared(weights),
            ResidentLayerAttentionWeights::Exception39(weights) => self
                .exception_39
                .as_mut()
                .expect("exception-39 weights require executor")
                .run_shared(weights),
            ResidentLayerAttentionWeights::Exception731(weights) => self
                .exception_731
                .as_mut()
                .expect("exception-731 weights require executor")
                .run_shared(weights),
            ResidentLayerAttentionWeights::Unavailable => {
                unreachable!("unavailable resident layer rejected before dispatch")
            }
        }
    }

    fn read_hidden_f32(
        &self,
        weights: &ResidentLayerAttentionWeights,
    ) -> Result<Vec<f32>, hipfire_xdna::XdnaError> {
        self.executor(weights).read_hidden_f32()
    }

    fn read_pre_ffn_state(
        &self,
        weights: &ResidentLayerAttentionWeights,
    ) -> Result<hipfire_xdna::NpuEmbeddingPreFfnState, hipfire_xdna::XdnaError> {
        self.executor(weights).read_pre_ffn_state()
    }

    fn outputs_direct_x(&self, weights: &ResidentLayerAttentionWeights) -> bool {
        self.executor(weights).outputs_direct_x()
    }

    fn executor(
        &self,
        weights: &ResidentLayerAttentionWeights,
    ) -> &NpuEmbeddingLayerAttentionDenseW8 {
        match weights {
            ResidentLayerAttentionWeights::Standard(_) => &self.standard,
            ResidentLayerAttentionWeights::Exception39(_) => self
                .exception_39
                .as_ref()
                .expect("exception-39 weights require executor"),
            ResidentLayerAttentionWeights::Exception731(_) => self
                .exception_731
                .as_ref()
                .expect("exception-731 weights require executor"),
            ResidentLayerAttentionWeights::Unavailable => {
                unreachable!("unavailable resident layer rejected before dispatch")
            }
        }
    }
}

struct ResidentLayerIo {
    input_gpu: ImportedTensor,
    residual_gpu: ImportedTensor,
    _input: SharedGttBuffer,
    residual: SharedGttBuffer,
    hidden: SharedGttBuffer,
    ffn: SharedGttBuffer,
}

enum ResidentFfnState {
    W4 {
        executor: NpuResidentFfnW4,
        weights: Vec<NpuResidentFfnW4Weights>,
        io: Option<SharedProjectionIo>,
    },
    DenseW8 {
        executor: NpuResidentFfnDenseW8,
        weights: Vec<NpuResidentFfnDenseW8Weights>,
        io: Option<SharedProjectionIo>,
    },
}

impl ResidentFfnState {
    fn rows(&self) -> usize {
        match self {
            Self::W4 { .. } => NpuResidentFfnW4::rows(),
            Self::DenseW8 { .. } => NpuResidentFfnDenseW8::rows(),
        }
    }

    fn layout(&self) -> ResidentFfnLayout {
        match self {
            Self::W4 { .. } => resident_ffn_w4_layout(),
            Self::DenseW8 { .. } => resident_ffn_dense_w8_layout(),
        }
    }

    fn io(&self) -> Option<&SharedProjectionIo> {
        match self {
            Self::W4 { io, .. } | Self::DenseW8 { io, .. } => io.as_ref(),
        }
    }

    fn attach_io(
        &mut self,
        input_fd: i32,
        input_bytes: usize,
        output_fd: i32,
        output_bytes: usize,
        io: SharedProjectionIo,
    ) -> Result<(), hipfire_xdna::XdnaError> {
        match self {
            Self::W4 {
                executor,
                io: state_io,
                ..
            } => {
                executor.attach_shared_io(input_fd, input_bytes, output_fd, output_bytes)?;
                *state_io = Some(io);
            }
            Self::DenseW8 {
                executor,
                io: state_io,
                ..
            } => {
                executor.attach_shared_io(input_fd, input_bytes, output_fd, output_bytes)?;
                *state_io = Some(io);
            }
        }
        Ok(())
    }

    fn run_layer(&mut self, layer: usize) -> Result<(), hipfire_xdna::XdnaError> {
        match self {
            Self::W4 {
                executor, weights, ..
            } => executor.run_shared(&weights[layer]),
            Self::DenseW8 {
                executor, weights, ..
            } => executor.run_shared(&weights[layer]),
        }
    }
}

impl NpuOpusProjector {
    pub fn load_cached(
        hfq: &HfqFile,
        cfg: &EmbeddingGemmaConfig,
        cache_root: &Path,
    ) -> Result<Self, String> {
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let requirements = fullk_requirements(hfq, cfg)?;
        let fullk_paths: BTreeMap<usize, Vec<PathBuf>> = requirements
            .iter()
            .map(|(&width, shapes)| {
                let paths = shapes
                    .iter()
                    .map(|(mode, padded_k)| {
                        cache_root.join(format!(
                            "embgemma_aie2p_fullk_submit_{mode}_m256_kg{}_n{width}",
                            padded_k / 256
                        ))
                    })
                    .collect();
                (width, paths)
            })
            .collect();
        let whole_paths: BTreeMap<usize, Vec<PathBuf>> = requirements
            .iter()
            .map(|(&width, shapes)| {
                let paths = shapes
                    .iter()
                    .filter(|(mode, _)| matches!(*mode, "w4" | "w8"))
                    .map(|(mode, padded_k)| {
                        cache_root.join(format!(
                            "embgemma_aie2p_whole_{mode}_m256_kg{}_n{width}",
                            padded_k / 256
                        ))
                    })
                    .collect();
                (width, paths)
            })
            .collect();
        let whole_scaled_paths: BTreeMap<usize, Vec<PathBuf>> = requirements
            .iter()
            .map(|(&width, shapes)| {
                let paths = shapes
                    .iter()
                    .filter(|(mode, _)| matches!(*mode, "w4" | "w8"))
                    .map(|(mode, padded_k)| {
                        let resident = cache_root.join(format!(
                            "embgemma_aie2p_rowmajor_whole8_{mode}-scaled_m256_kg{}_n{width}",
                            padded_k / 256
                        ));
                        let fast = cache_root.join(format!(
                            "embgemma_aie2p_whole8_{mode}-scaled_m256_kg{}_n{width}",
                            padded_k / 256
                        ));
                        if resident.join("final.xclbin").is_file() {
                            resident
                        } else if fast.join("final.xclbin").is_file() {
                            fast
                        } else {
                            cache_root.join(format!(
                                "embgemma_aie2p_whole_{mode}-scaled_m256_kg{}_n{width}",
                                padded_k / 256
                            ))
                        }
                    })
                    .collect();
                (width, paths)
            })
            .collect();
        let scaled_compatible = requirements
            .values()
            .flatten()
            .all(|(mode, _)| matches!(*mode, "w4" | "w8"));
        let use_whole_scaled = scaled_compatible
            && whole_scaled_paths.values().all(|paths| !paths.is_empty())
            && whole_scaled_paths.values().flatten().all(|path| {
                path.join("final.xclbin").is_file() && path.join("insts.bin").is_file()
            });
        let whole_compatible = requirements
            .values()
            .flatten()
            .all(|(mode, _)| matches!(*mode, "w4" | "w8"));
        let use_whole = !use_whole_scaled
            && whole_compatible
            && whole_paths.values().all(|paths| !paths.is_empty())
            && whole_paths.values().flatten().all(|path| {
                path.join("final.xclbin").is_file() && path.join("insts.bin").is_file()
            });
        let use_fullk = !use_whole
            && fullk_paths.values().flatten().all(|path| {
                path.join("final.xclbin").is_file() && path.join("insts.bin").is_file()
            });
        let widths = BTreeSet::from([q_dim, kv_dim, cfg.hidden_size, cfg.intermediate_size]);
        let mut executors = HashMap::with_capacity(widths.len());
        for width in widths {
            if width == 0 || width % 64 != 0 {
                return Err(format!(
                    "embeddinggemma NPU: unsupported output width {width}"
                ));
            }
            let executor = if use_whole_scaled {
                let paths = whole_scaled_paths.get(&width).ok_or_else(|| {
                    format!("embeddinggemma NPU: no scaled whole-array N={width} shapes")
                })?;
                let caches: Vec<&str> = paths
                    .iter()
                    .map(|path| path.to_str().expect("UTF-8 cache path"))
                    .collect();
                NpuOpusExecutor::load_whole_scaled_cached(&caches, width)
            } else if use_whole {
                let paths = whole_paths.get(&width).ok_or_else(|| {
                    format!("embeddinggemma NPU: no whole-array N={width} shapes")
                })?;
                let caches: Vec<&str> = paths
                    .iter()
                    .map(|path| path.to_str().expect("UTF-8 cache path"))
                    .collect();
                NpuOpusExecutor::load_whole_cached(&caches, width)
            } else if use_fullk {
                let paths = fullk_paths
                    .get(&width)
                    .ok_or_else(|| format!("embeddinggemma NPU: no full-K N={width} shapes"))?;
                let caches: Vec<(&str, usize)> = paths
                    .iter()
                    .map(|path| (path.to_str().expect("UTF-8 cache path"), 8))
                    .collect();
                NpuOpusExecutor::load_fullk_cached(&caches, width)
            } else {
                let blocks = width / 64;
                let w4 = cache_root.join(format!("embgemma_aie2p_w4_4x4x16_c8_nb{blocks}"));
                let w8 = cache_root.join(format!("embgemma_aie2p_w8_4x4x32_c8_nb{blocks}_m8k8_w8"));
                let sparse3 = cache_root.join(format!(
                    "embgemma_aie2p_sparse3_4x4x16_c8_nb{blocks}_sparse3"
                ));
                NpuOpusExecutor::load_cached(
                    &w4.to_string_lossy(),
                    &w8.to_string_lossy(),
                    &sparse3.to_string_lossy(),
                    width,
                )
            }
            .map_err(|error| format!("embeddinggemma NPU: load N={width} caches: {error}"))?;
            executors.insert(width, executor);
        }
        if use_whole_scaled {
            let mode = requirements
                .values()
                .flatten()
                .next()
                .map(|(mode, _)| *mode)
                .ok_or_else(|| "embeddinggemma NPU: empty projection requirements".to_string())?;
            for width in [q_dim + 2 * kv_dim, 2 * cfg.intermediate_size] {
                let resident = cache_root.join(format!(
                    "embgemma_aie2p_rowmajor_whole8_{mode}-scaled_m256_kg{}_n{width}",
                    cfg.hidden_size.div_ceil(256)
                ));
                let fast = cache_root.join(format!(
                    "embgemma_aie2p_whole8_{mode}-scaled_m256_kg{}_n{width}",
                    cfg.hidden_size.div_ceil(256)
                ));
                let path = if resident.join("final.xclbin").is_file() {
                    resident
                } else if fast.join("final.xclbin").is_file() {
                    fast
                } else {
                    cache_root.join(format!(
                        "embgemma_aie2p_whole_{mode}-scaled_m256_kg{}_n{width}",
                        cfg.hidden_size.div_ceil(256)
                    ))
                };
                if path.join("final.xclbin").is_file() && path.join("insts.bin").is_file() {
                    let cache = path.to_str().expect("UTF-8 cache path");
                    let executor = NpuOpusExecutor::load_whole_scaled_cached(&[cache], width)
                        .map_err(|error| {
                            format!("embeddinggemma NPU: load combined N={width}: {error}")
                        })?;
                    executors.insert(width, executor);
                }
            }
        }

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let prefix = format!("model.layers.{layer_idx}");
            layers.push(LayerMatrices {
                qkv: load_concat_matrix(
                    hfq,
                    executors.get(&(q_dim + 2 * kv_dim)),
                    &[
                        format!("{prefix}.self_attn.q_proj.weight"),
                        format!("{prefix}.self_attn.k_proj.weight"),
                        format!("{prefix}.self_attn.v_proj.weight"),
                    ],
                )?,
                query: load_matrix(
                    hfq,
                    executor(&executors, q_dim)?,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                )?,
                key: load_matrix(
                    hfq,
                    executor(&executors, kv_dim)?,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                )?,
                value: load_matrix(
                    hfq,
                    executor(&executors, kv_dim)?,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                )?,
                attention_output: load_matrix(
                    hfq,
                    executor(&executors, cfg.hidden_size)?,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                )?,
                gate: load_matrix(
                    hfq,
                    executor(&executors, cfg.intermediate_size)?,
                    &format!("{prefix}.mlp.gate_proj.weight"),
                )?,
                up: load_matrix(
                    hfq,
                    executor(&executors, cfg.intermediate_size)?,
                    &format!("{prefix}.mlp.up_proj.weight"),
                )?,
                gate_up: load_concat_matrix(
                    hfq,
                    executors.get(&(2 * cfg.intermediate_size)),
                    &[
                        format!("{prefix}.mlp.gate_proj.weight"),
                        format!("{prefix}.mlp.up_proj.weight"),
                    ],
                )?,
                down: load_optional_matrix(
                    hfq,
                    executor(&executors, cfg.hidden_size)?,
                    &format!("{prefix}.mlp.down_proj.weight"),
                )?,
            });
        }
        let resident_attention_path =
            cache_root.join("embgemma_aie2p_resident_w8_qkv_attention_m256_k768_n1280");
        let packed_attention_path =
            cache_root.join("embgemma_aie2p_resident_w8_qkv_attention_packed_m256_k768_n1280");
        let attention_output_path =
            cache_root.join("embgemma_aie2p_attention_o_bf16_m256_k768_n768");
        let use_packed_attention = packed_attention_path.join("final.xclbin").is_file()
            && packed_attention_path.join("insts.bin").is_file()
            && attention_output_path.join("final.xclbin").is_file()
            && attention_output_path.join("insts.bin").is_file();
        let selected_attention_path = if use_packed_attention {
            &packed_attention_path
        } else {
            &resident_attention_path
        };
        let resident_attention = if cfg.hidden_size == 768
            && q_dim == 768
            && kv_dim == 256
            && selected_attention_path.join("final.xclbin").is_file()
            && selected_attention_path.join("insts.bin").is_file()
            && layers
                .iter()
                .all(|layer| resident_attention_dense_groups(layer).is_ok())
        {
            let cache = selected_attention_path
                .to_str()
                .expect("UTF-8 resident attention cache path");
            let executor = if use_packed_attention {
                NpuResidentAttentionDenseW8::load_packed_cached(cache)
            } else {
                NpuResidentAttentionDenseW8::load_cached(cache)
            }
            .map_err(|error| format!("embeddinggemma NPU: load resident attention: {error}"))?;
            let mut weights = Vec::with_capacity(layers.len());
            for (layer_idx, layer) in layers.iter().enumerate() {
                let (dense_groups, dense_scales, awq_scale) =
                    resident_attention_dense_groups(layer)?;
                let group_refs = dense_groups.iter().map(Vec::as_slice).collect::<Vec<_>>();
                let scale_refs = dense_scales.iter().map(Vec::as_slice).collect::<Vec<_>>();
                let prefix = format!("model.layers.{layer_idx}.self_attn");
                let mut qnorm =
                    load_vector_f32(hfq, &format!("{prefix}.q_norm.weight"), cfg.head_dim)?;
                let prescale = cfg.q_prescale();
                if (prescale - 1.0).abs() > 1.0e-6 {
                    qnorm.iter_mut().for_each(|value| *value *= prescale);
                }
                let knorm = load_vector_f32(hfq, &format!("{prefix}.k_norm.weight"), cfg.head_dim)?;
                weights.push(
                    executor
                        .upload_dense_groups(
                            &group_refs,
                            &scale_refs,
                            awq_scale.as_deref(),
                            &qnorm,
                            &knorm,
                            cfg.rms_norm_eps,
                            cfg.rope_base_for_layer(layer_idx),
                        )
                        .map_err(|error| {
                            format!(
                                "embeddinggemma NPU: upload resident attention layer {layer_idx}: {error}"
                            )
                        })?,
                );
            }
            let output = if use_packed_attention {
                let output_cache = attention_output_path
                    .to_str()
                    .expect("UTF-8 attention output cache path");
                let output_executor =
                    NpuAttentionOutputBf16::load_cached(output_cache).map_err(|error| {
                        format!("embeddinggemma NPU: load attention output: {error}")
                    })?;
                let output_weights = layers
                    .iter()
                    .enumerate()
                    .map(|(layer_idx, layer)| {
                        output_executor
                            .upload_weights(&layer.attention_output)
                            .map_err(|error| {
                                format!(
                                    "embeddinggemma NPU: upload attention output layer {layer_idx}: {error}"
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Some(ResidentAttentionOutputState {
                    executor: output_executor,
                    weights: output_weights,
                    staging: (0..layers.len()).map(|_| None).collect(),
                    io: None,
                })
            } else {
                None
            };
            Some(ResidentAttentionState {
                executor,
                weights,
                input: None,
                output,
            })
        } else {
            None
        };
        let resident_attention_selected = resident_attention.is_some();
        let resident_ffn_w4_path =
            cache_root.join("embgemma_aie2p_resident_ffn_w4_m256_k768_i1152_o768");
        let resident_ffn_dense_w8_path =
            cache_root.join("embgemma_aie2p_resident_ffn_dense_w8_m256_k768_i1152_o768");
        let resident_layer_ffn_direct_x_path = cache_root
            .join("embgemma_aie2p_resident_ffn_dense_w8_direct_x_bf16x2_m256_k768_i1152_o768");
        let resident_layer_ffn_canonical_path = cache_root
            .join("embgemma_aie2p_resident_ffn_dense_w8_canonical_bf16x2_m256_k768_i1152_o768");
        let resident_layer_ffn_path = if resident_layer_ffn_direct_x_path
            .join("final.xclbin")
            .is_file()
            && resident_layer_ffn_direct_x_path.join("insts.bin").is_file()
            && cache_root
                .join("embgemma_aie2p_resident_w8_qkv_paired_attention_o_norm_x_bf16_m256_k768_n1280/final.xclbin")
                .is_file()
            && cache_root
                .join("embgemma_aie2p_resident_w8_qkv_paired_attention_o_norm_x_bf16_m256_k768_n1280/insts.bin")
                .is_file()
        {
            resident_layer_ffn_direct_x_path
        } else {
            resident_layer_ffn_canonical_path
        };
        let resident_mode = resident_ffn_mode(&layers);
        if let Err(reason) = &resident_mode {
            eprintln!("embeddinggemma NPU: resident FFN unavailable: {reason}");
        }
        let resident_ffn = match resident_mode {
            Ok(OpusResidentMode::W4)
                if resident_ffn_w4_path.join("final.xclbin").is_file()
                    && resident_ffn_w4_path.join("insts.bin").is_file() =>
            {
                let executor = NpuResidentFfnW4::load_cached(
                    resident_ffn_w4_path
                        .to_str()
                        .expect("UTF-8 resident FFN cache path"),
                )
                .map_err(|error| format!("embeddinggemma NPU: load resident W4 FFN: {error}"))?;
                let weights = layers
                    .iter()
                    .map(|layer| {
                        executor.upload_weights(
                            &layer.gate,
                            &layer.up,
                            layer.down.as_ref().expect("resident FFN down matrix"),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| {
                        format!("embeddinggemma NPU: pack resident W4 FFN: {error}")
                    })?;
                Some(ResidentFfnState::W4 {
                    executor,
                    weights,
                    io: None,
                })
            }
            Ok(OpusResidentMode::DenseW8)
                if resident_ffn_dense_w8_path.join("final.xclbin").is_file()
                    && resident_ffn_dense_w8_path.join("insts.bin").is_file() =>
            {
                let executor = NpuResidentFfnDenseW8::load_cached(
                    resident_ffn_dense_w8_path
                        .to_str()
                        .expect("UTF-8 resident dense-W8 FFN cache path"),
                )
                .map_err(|error| {
                    format!("embeddinggemma NPU: load resident dense-W8 FFN: {error}")
                })?;
                let weights = layers
                    .iter()
                    .map(|layer| {
                        executor.upload_weights(
                            &layer.gate,
                            &layer.up,
                            layer.down.as_ref().expect("resident FFN down matrix"),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| {
                        format!("embeddinggemma NPU: pack resident dense-W8 FFN: {error}")
                    })?;
                Some(ResidentFfnState::DenseW8 {
                    executor,
                    weights,
                    io: None,
                })
            }
            Ok(mode) => {
                eprintln!(
                    "embeddinggemma NPU: resident FFN unavailable: {:?} cache is not built",
                    mode
                );
                None
            }
            Err(_) => None,
        };
        let resident_ffn_selected = resident_ffn.is_some();
        let resident_layer_attention_external_path = cache_root.join(
            "embgemma_aie2p_resident_w8_qkv_paired_attention_o_norm_external_x_bf16x2_m256_k768_n1280",
        );
        let resident_layer_attention_direct_x_path = cache_root
            .join("embgemma_aie2p_resident_w8_qkv_paired_attention_o_norm_x_bf16_m256_k768_n1280");
        let resident_layer_attention_h_path = cache_root
            .join("embgemma_aie2p_resident_w8_qkv_paired_attention_o_norm_m256_k768_n1280");
        let resident_layer_external_selected = resident_layer_attention_external_path
            .join("final.xclbin")
            .is_file()
            && resident_layer_attention_external_path
                .join("insts.bin")
                .is_file();
        let resident_layer_attention_path = if resident_layer_external_selected {
            resident_layer_attention_external_path
        } else if resident_layer_attention_direct_x_path
            .join("final.xclbin")
            .is_file()
            && resident_layer_attention_direct_x_path
                .join("insts.bin")
                .is_file()
        {
            resident_layer_attention_direct_x_path
        } else {
            resident_layer_attention_h_path
        };
        let resident_layer_exception_39_path = cache_root.join(
            "embgemma_aie2p_resident_w8_qkv_paired_attention_o_norm_x_exception_c39_m256_k768_n1280",
        );
        let resident_layer_exception_731_path = cache_root.join(
            "embgemma_aie2p_resident_w8_qkv_paired_attention_o_norm_x_exception_c731_m256_k768_n1280",
        );
        let resident_layer_tail_split_x_path = cache_root
            .join("embgemma_aie2p_post_ffn_direct_tail_bf16x2_split_x_completed_bf16x2_m256_k768");
        let resident_layer_tail_bf16x2_path = cache_root
            .join("embgemma_aie2p_post_ffn_direct_tail_bf16x2_completed_bf16x2_m256_k768");
        let resident_layer_tail_bf16_path =
            cache_root.join("embgemma_aie2p_post_ffn_direct_tail_bf16x2_m256_k768");
        let resident_layer_tail_path = if resident_layer_tail_split_x_path
            .join("final.xclbin")
            .is_file()
            && resident_layer_tail_split_x_path.join("insts.bin").is_file()
        {
            resident_layer_tail_split_x_path
        } else if resident_layer_tail_bf16x2_path
            .join("final.xclbin")
            .is_file()
            && resident_layer_tail_bf16x2_path.join("insts.bin").is_file()
        {
            resident_layer_tail_bf16x2_path
        } else {
            resident_layer_tail_bf16_path
        };
        let resident_next_prep_path =
            cache_root.join("embgemma_aie2p_next_layer_prep_w8_bf16x2_m256_k768");
        let resident_residual_prep_path =
            cache_root.join("embgemma_aie2p_residual_prep_bf16x2_to_r34_records_m256_k768");
        let resident_final_norm_mean_path =
            cache_root.join("embgemma_aie2p_final_norm_mean_bf16x2_m256_k768");
        let resident_layer_requested =
            std::env::var("HIPFIRE_EMBED_RESIDENT_LAYER").is_ok_and(|value| value != "0");
        let resident_layer = if resident_layer_requested
            && cfg.hidden_size == 768
            && resident_ffn_mode(&layers).is_ok()
            && resident_layer_ffn_path.join("final.xclbin").is_file()
            && resident_layer_ffn_path.join("insts.bin").is_file()
            && resident_layer_attention_path.join("final.xclbin").is_file()
            && resident_layer_attention_path.join("insts.bin").is_file()
            && resident_layer_tail_path.join("final.xclbin").is_file()
            && resident_layer_tail_path.join("insts.bin").is_file()
            && (!resident_layer_external_selected
                || (resident_next_prep_path.join("final.xclbin").is_file()
                    && resident_next_prep_path.join("insts.bin").is_file()
                    && resident_residual_prep_path.join("final.xclbin").is_file()
                    && resident_residual_prep_path.join("insts.bin").is_file()))
        {
            let attention_cache = resident_layer_attention_path
                .to_str()
                .expect("UTF-8 resident layer attention cache path");
            let attention = NpuEmbeddingLayerAttentionDenseW8::load_cached(attention_cache)
                .map_err(|error| {
                    format!("embeddinggemma NPU: load completed-layer attention: {error}")
                })?;
            let load_exception = |path: &Path, column: usize| {
                if !path.join("final.xclbin").is_file() || !path.join("insts.bin").is_file() {
                    return Ok(None);
                }
                NpuEmbeddingLayerAttentionDenseW8::load_cached(
                    path.to_str().expect("UTF-8 exception cache path"),
                )
                .map(Some)
                .map_err(|error| {
                    format!("embeddinggemma NPU: load completed-layer exception-{column}: {error}")
                })
            };
            let direct_x_attention = attention.outputs_direct_x();
            let exception_39 = if direct_x_attention {
                None
            } else {
                load_exception(&resident_layer_exception_39_path, 39)?
            };
            let exception_731 = if direct_x_attention {
                None
            } else {
                load_exception(&resident_layer_exception_731_path, 731)?
            };
            let tail_cache = resident_layer_tail_path
                .to_str()
                .expect("UTF-8 resident layer tail cache path");
            let tail =
                NpuEmbeddingPostFfnDirectTailBf16x2::load_cached(tail_cache).map_err(|error| {
                    format!("embeddinggemma NPU: load completed-layer tail: {error}")
                })?;
            let next_prep = if resident_next_prep_path.join("final.xclbin").is_file()
                && resident_next_prep_path.join("insts.bin").is_file()
                && tail.output_bytes() == NpuEmbeddingNextLayerPrepW8::completed_bytes()
            {
                Some(
                    NpuEmbeddingNextLayerPrepW8::load_cached(
                        resident_next_prep_path
                            .to_str()
                            .expect("UTF-8 next-layer prep cache path"),
                    )
                    .map_err(|error| {
                        format!("embeddinggemma NPU: load next-layer prep: {error}")
                    })?,
                )
            } else {
                None
            };
            let residual_prep = if attention.uses_external_residual() {
                Some(
                    NpuEmbeddingResidualPrep::load_cached(
                        resident_residual_prep_path
                            .to_str()
                            .expect("UTF-8 residual prep cache path"),
                    )
                    .map_err(|error| format!("embeddinggemma NPU: load residual prep: {error}"))?,
                )
            } else {
                None
            };
            let (final_norm_mean, final_norm_mean_params) = if tail.output_bytes()
                == NpuEmbeddingFinalNormMean::completed_bytes()
                && resident_final_norm_mean_path.join("final.xclbin").is_file()
                && resident_final_norm_mean_path.join("insts.bin").is_file()
            {
                let kernel = NpuEmbeddingFinalNormMean::load_cached(
                    resident_final_norm_mean_path
                        .to_str()
                        .expect("UTF-8 final norm/mean cache path"),
                )
                .map_err(|error| format!("embeddinggemma NPU: load final norm/mean: {error}"))?;
                let output_norm = load_vector_f32(hfq, "model.norm.weight", cfg.hidden_size)?;
                let params = kernel
                    .upload_params(&output_norm, cfg.rms_norm_eps)
                    .map_err(|error| {
                        format!("embeddinggemma NPU: upload final norm/mean: {error}")
                    })?;
                (Some(kernel), Some(params))
            } else {
                (None, None)
            };
            let ffn_cache = resident_layer_ffn_path
                .to_str()
                .expect("UTF-8 completed-layer FFN cache path");
            let ffn = NpuResidentFfnDenseW8::load_cached(ffn_cache).map_err(|error| {
                format!("embeddinggemma NPU: load completed-layer FFN: {error}")
            })?;
            if ffn.consumes_direct_x() && !direct_x_attention {
                return Err(format!(
                    "embeddinggemma NPU: completed-layer attention/FFN handoff mismatch: attention direct-X={direct_x_attention}, FFN direct-X={}",
                    ffn.consumes_direct_x()
                ));
            }
            if tail.consumes_split_x() && (!direct_x_attention || !ffn.consumes_direct_x()) {
                return Err(
                    "embeddinggemma NPU: split-X tail requires direct-X attention and FFN"
                        .to_string(),
                );
            }
            let mut ffn_weights = Vec::with_capacity(layers.len());
            let mut attention_weights = Vec::with_capacity(layers.len());
            let mut tail_params = Vec::with_capacity(layers.len());
            let mut next_prep_params = Vec::with_capacity(layers.len());
            let mut tail_pre_norms = Vec::with_capacity(layers.len());
            let mut tail_post_norms = Vec::with_capacity(layers.len());
            let zero_residual = vec![0u16; 256 * cfg.hidden_size];
            for (layer_idx, layer) in layers.iter().enumerate() {
                let prefix = format!("model.layers.{layer_idx}");
                let (dense_groups, dense_scales, awq_scale) =
                    resident_attention_dense_groups(layer)?;
                if let Some(prep) = next_prep.as_ref() {
                    let input_norm = load_vector_f32(
                        hfq,
                        &format!("{prefix}.input_layernorm.weight"),
                        cfg.hidden_size,
                    )?;
                    next_prep_params.push(
                        prep.upload_params(&input_norm, awq_scale.as_deref())
                            .map_err(|error| {
                                format!(
                                    "embeddinggemma NPU: upload next-layer prep {layer_idx}: {error}"
                                )
                            })?,
                    );
                }
                let group_refs = dense_groups.iter().map(Vec::as_slice).collect::<Vec<_>>();
                let scale_refs = dense_scales.iter().map(Vec::as_slice).collect::<Vec<_>>();
                let mut qnorm = load_vector_f32(
                    hfq,
                    &format!("{prefix}.self_attn.q_norm.weight"),
                    cfg.head_dim,
                )?;
                let prescale = cfg.q_prescale();
                if (prescale - 1.0).abs() > 1.0e-6 {
                    qnorm.iter_mut().for_each(|value| *value *= prescale);
                }
                let knorm = load_vector_f32(
                    hfq,
                    &format!("{prefix}.self_attn.k_norm.weight"),
                    cfg.head_dim,
                )?;
                let post_attention_norm = load_vector_bf16(
                    hfq,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    cfg.hidden_size,
                )?;
                let pre_ffn_norm = load_vector_bf16(
                    hfq,
                    &format!("{prefix}.pre_feedforward_layernorm.weight"),
                    cfg.hidden_size,
                )?;
                ffn_weights.push(
                    ffn.upload_weights_with_pre_ffn_norm(
                        &layer.gate,
                        &layer.up,
                        layer.down.as_ref().expect("completed-layer down matrix"),
                        &pre_ffn_norm,
                    )
                    .map_err(|error| {
                        format!(
                            "embeddinggemma NPU: upload completed-layer FFN {layer_idx}: {error}"
                        )
                    })?,
                );
                let post_ffn_norm = load_vector_bf16(
                    hfq,
                    &format!("{prefix}.post_feedforward_layernorm.weight"),
                    cfg.hidden_size,
                )?;
                let zero_columns = pre_ffn_norm
                    .iter()
                    .enumerate()
                    .filter_map(|(column, &bits)| (bits & 0x7fff == 0).then_some(column))
                    .collect::<Vec<_>>();
                let selected = if direct_x_attention {
                    Some((0usize, &attention))
                } else {
                    match zero_columns.as_slice() {
                        [] => Some((0usize, &attention)),
                        [39] => exception_39.as_ref().map(|executor| (39, executor)),
                        [731] => exception_731.as_ref().map(|executor| (731, executor)),
                        _ => None,
                    }
                };
                let attention_weight = if let Some((variant, executor)) = selected {
                    let weights = executor
                        .upload_dense_groups(
                            &group_refs,
                            &scale_refs,
                            awq_scale.as_deref(),
                            &layer.attention_output,
                            &zero_residual,
                            &qnorm,
                            &knorm,
                            &post_attention_norm,
                            &pre_ffn_norm,
                            cfg.rms_norm_eps,
                            cfg.rope_base_for_layer(layer_idx),
                        )
                        .map_err(|error| {
                            format!(
                                "embeddinggemma NPU: upload completed-layer attention {layer_idx}: {error}"
                            )
                        })?;
                    match variant {
                        0 => ResidentLayerAttentionWeights::Standard(weights),
                        39 => ResidentLayerAttentionWeights::Exception39(weights),
                        731 => ResidentLayerAttentionWeights::Exception731(weights),
                        _ => unreachable!("selected resident exception variant"),
                    }
                } else {
                    ResidentLayerAttentionWeights::Unavailable
                };
                attention_weights.push(attention_weight);
                tail_params.push(
                    tail.upload_params(&post_ffn_norm, cfg.rms_norm_eps)
                        .map_err(|error| {
                            format!(
                                "embeddinggemma NPU: upload completed-layer tail {layer_idx}: {error}"
                            )
                        })?,
                );
                tail_pre_norms.push(pre_ffn_norm);
                tail_post_norms.push(post_ffn_norm);
            }
            Some(ResidentLayerState {
                attention: ResidentLayerAttentionExecutors {
                    standard: attention,
                    exception_39,
                    exception_731,
                },
                attention_weights,
                ffn,
                ffn_weights,
                tail,
                tail_params,
                next_prep,
                residual_prep,
                final_norm_mean,
                final_norm_mean_params,
                next_prep_params,
                prepared_input_layer: None,
                tail_pre_norms,
                tail_post_norms,
                io: None,
            })
        } else {
            None
        };
        Ok(Self {
            executors,
            layers,
            shared_io: HashMap::new(),
            awq_gpu: HashMap::new(),
            resident_attention,
            resident_attention_selected,
            resident_ffn,
            resident_ffn_selected,
            resident_layer,
            debug_resident_hidden: None,
            debug_resident_residual: None,
            debug_resident_exception: None,
            debug_resident_ffn: None,
        })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn executor_count(&self) -> usize {
        self.executors.len()
    }

    pub fn resident_ffn_enabled(&self) -> bool {
        self.resident_ffn.is_some() && self.resident_ffn_selected
    }

    pub fn resident_attention_enabled(&self) -> bool {
        self.resident_attention.is_some() && self.resident_attention_selected
    }

    pub fn resident_layer_enabled(&self) -> bool {
        self.resident_layer.is_some() && self.resident_ffn_selected
    }

    pub fn select_resident_attention(&mut self, selected: bool) -> Result<(), String> {
        if selected && self.resident_attention.is_none() {
            return Err("embeddinggemma NPU: resident attention is unavailable".to_string());
        }
        self.resident_attention_selected = selected;
        Ok(())
    }

    /// Select or bypass the complete resident FFN while retaining all resident
    /// buffers. This is intended for same-process correctness comparisons
    /// against the established per-projection Opus path.
    pub fn select_resident_ffn(&mut self, selected: bool) -> Result<(), String> {
        if selected && self.resident_ffn.is_none() {
            return Err("embeddinggemma NPU: resident FFN is unavailable".to_string());
        }
        self.resident_ffn_selected = selected;
        Ok(())
    }
}

fn resident_ffn_mode(layers: &[LayerMatrices]) -> Result<OpusResidentMode, String> {
    let mut selected = None;
    for (index, layer) in layers.iter().enumerate() {
        let Some(down) = layer.down.as_ref() else {
            return Err(format!("layer {index} down projection is not Opus"));
        };
        let modes = [
            layer.gate.resident_mode(),
            layer.up.resident_mode(),
            down.resident_mode(),
        ];
        if modes[1..].iter().any(|mode| *mode != modes[0]) {
            return Err(format!(
                "layer {index} resident modes differ: gate={:?} up={:?} down={:?}",
                modes[0], modes[1], modes[2]
            ));
        }
        if selected.is_some_and(|mode| mode != modes[0]) {
            return Err(format!(
                "layer {index} resident mode {:?} differs from earlier layers {:?}",
                modes[0], selected
            ));
        }
        selected = Some(modes[0]);
        if layer.gate.awq_scale() != layer.up.awq_scale() {
            return Err(format!(
                "layer {index} gate/up AWQ activation scales differ"
            ));
        }
    }
    selected.ok_or_else(|| "model has no FFN layers".to_string())
}

type ResidentAttentionDenseGroups = (Vec<Vec<i8>>, Vec<Vec<f32>>, Option<Vec<f32>>);

fn resident_attention_dense_groups(
    layer: &LayerMatrices,
) -> Result<ResidentAttentionDenseGroups, String> {
    let roles = [
        (&layer.query, 768usize),
        (&layer.key, 256),
        (&layer.value, 256),
    ];
    for (matrix, width) in roles {
        // R30 is a dense-W8 execution contract, not a source-format
        // restriction. `group_dense_i8()` expands compact mixed matrices and
        // exposes native W4/W8 groups through the same upload representation.
        if matrix.k() != 768 || matrix.n() != width || matrix.group_count() != 3 {
            return Err(format!(
                "resident attention role wants Opus K=768 N={width} groups=3, got {:?} K={} N={} groups={}",
                matrix.encoding(),
                matrix.k(),
                matrix.n(),
                matrix.group_count()
            ));
        }
    }
    let awq = layer.query.awq_scale();
    if layer.key.awq_scale() != awq || layer.value.awq_scale() != awq {
        return Err("resident attention Q/K/V AWQ activation scales differ".to_string());
    }
    let mut groups = Vec::with_capacity(3);
    let mut scales = Vec::with_capacity(3);
    for group in 0..3 {
        let query = layer.query.group_dense_i8(group);
        let key = layer.key.group_dense_i8(group);
        let value = layer.value.group_dense_i8(group);
        let mut combined = vec![0i8; 256 * 1280];
        for inner in 0..256 {
            let target = inner * 1280;
            combined[target..target + 768].copy_from_slice(&query[inner * 768..(inner + 1) * 768]);
            combined[target + 768..target + 1024]
                .copy_from_slice(&key[inner * 256..(inner + 1) * 256]);
            combined[target + 1024..target + 1280]
                .copy_from_slice(&value[inner * 256..(inner + 1) * 256]);
        }
        let mut combined_scales = Vec::with_capacity(1280);
        combined_scales.extend_from_slice(layer.query.group_scales(group));
        combined_scales.extend_from_slice(layer.key.group_scales(group));
        combined_scales.extend_from_slice(layer.value.group_scales(group));
        groups.push(combined);
        scales.push(combined_scales);
    }
    Ok((groups, scales, awq.map(<[f32]>::to_vec)))
}

fn fullk_requirements(
    hfq: &HfqFile,
    cfg: &EmbeddingGemmaConfig,
) -> Result<BTreeMap<usize, BTreeSet<(&'static str, usize)>>, String> {
    let mut requirements: BTreeMap<usize, BTreeSet<(&'static str, usize)>> = BTreeMap::new();
    for layer_idx in 0..cfg.num_hidden_layers {
        let prefix = format!("model.layers.{layer_idx}");
        for suffix in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            let name = format!("{prefix}.{suffix}");
            let info = hfq
                .find_tensor_info(&name)
                .ok_or_else(|| format!("embeddinggemma NPU: missing tensor {name}"))?;
            let Some(mode) = fullk_mode_tag(info.quant_type) else {
                if suffix == "mlp.down_proj.weight" {
                    continue;
                }
                return Err(format!(
                    "embeddinggemma NPU: {name} qt={} is not Opus",
                    info.quant_type
                ));
            };
            if info.shape.len() != 2 {
                return Err(format!("embeddinggemma NPU: {name} must be rank-2"));
            }
            let n = info.shape[0] as usize;
            let k = info.shape[1] as usize;
            OpusMatrixEncoding::classify(info.quant_type, info.data_size, k, n)
                .map_err(|error| format!("embeddinggemma NPU: classify {name}: {error}"))?;
            requirements
                .entry(n)
                .or_default()
                .insert((mode, k.div_ceil(256) * 256));
        }
    }
    Ok(requirements)
}

fn fullk_mode_tag(quant_type: u8) -> Option<&'static str> {
    match quant_type {
        33 | 34 => Some("w4"),
        35 => Some("w8"),
        36 => Some("mixed"),
        _ => None,
    }
}

impl SharedProjectionIo {
    fn allocate(
        gpu: &mut Gpu,
        executor: &mut NpuOpusExecutor,
        matrix: &OpusPackedMatrix,
        layout: NpuWholeScaledIoLayout,
    ) -> HipResult<Self> {
        let mut input_buffer = gpu.alloc_shared_gtt(layout.input_bytes())?;
        let mut output_buffer = gpu.alloc_shared_gtt(layout.output_bytes())?;
        input_buffer.as_mut_slice().fill(0);
        output_buffer.as_mut_slice().fill(0);
        let input_gpu = gpu.import_dmabuf(
            input_buffer.dmabuf_fd(),
            layout.input_bytes(),
            &[layout.input_bytes()],
            DType::Raw,
        )?;
        let output_gpu = gpu.import_dmabuf(
            output_buffer.dmabuf_fd(),
            layout.output_bytes(),
            &[layout.output_bytes()],
            DType::Raw,
        )?;
        executor
            .attach_whole_scaled_shared_io(
                matrix,
                input_buffer.dmabuf_fd(),
                layout.input_bytes(),
                output_buffer.dmabuf_fd(),
                layout.output_bytes(),
            )
            .map_err(|error| hip_error(format!("attach shared Opus I/O: {error}")))?;
        Ok(Self {
            input_gpu,
            output_gpu,
            _input_buffer: input_buffer,
            _output_buffer: output_buffer,
        })
    }
}

fn rdna_io_layout(layout: NpuWholeScaledIoLayout) -> OpusNpuIoLayout {
    OpusNpuIoLayout::new(
        layout.mode() == NpuWholeMode::W8,
        layout.cols(),
        layout.rows(),
        layout.groups(),
        layout.n(),
        layout.n_macros(),
        layout.outblocks(),
        8192,
        layout.input_bytes(),
        layout.output_bytes(),
        layout.row_major_output(),
        layout.padded_n(),
    )
}

#[allow(clippy::too_many_arguments)]
fn try_shared_projection(
    gpu: &mut Gpu,
    executors: &mut HashMap<usize, NpuOpusExecutor>,
    shared_io: &mut HashMap<SharedIoKey, SharedProjectionIo>,
    awq_gpu: &mut HashMap<MatrixGpuKey, OwnedTensor>,
    matrix_key: MatrixGpuKey,
    matrix: &OpusPackedMatrix,
    input: &GpuTensor,
    outputs: &[(&GpuTensor, usize)],
    rows: usize,
) -> HipResult<bool> {
    let Some(layout) = executors
        .get(&matrix.n())
        .and_then(|executor| executor.whole_scaled_io_layout(matrix))
    else {
        return Ok(false);
    };
    if rows > layout.rows() || outputs.is_empty() || outputs.len() > 3 {
        return Ok(false);
    }
    let key = SharedIoKey {
        mode: layout.mode(),
        k: layout.k(),
        n: layout.n(),
    };
    if !shared_io.contains_key(&key) {
        let io = SharedProjectionIo::allocate(
            gpu,
            executors
                .get_mut(&matrix.n())
                .ok_or_else(|| hip_error("missing shared Opus executor"))?,
            matrix,
            layout,
        )?;
        shared_io.insert(key, io);
    }
    if !awq_gpu.contains_key(&matrix_key) {
        if let Some(scale) = matrix.awq_scale() {
            awq_gpu.insert(matrix_key, gpu.upload_owned_f32(scale, &[scale.len()])?);
        }
    }
    let io = shared_io
        .get(&key)
        .ok_or_else(|| hip_error("missing allocated shared Opus I/O"))?;
    let input_view = io.input_gpu.view();
    let output_view = io.output_gpu.view();
    gpu.pack_opus_npu_activations(
        input,
        awq_gpu.get(&matrix_key).map(OwnedTensor::view),
        &input_view,
        rows,
        matrix.k(),
        rdna_io_layout(layout),
    )?;
    // The NPU cannot observe an in-flight HIP stream; make the shared producer
    // completion explicit before the XDNA cache reconciliation and submit.
    gpu.device_synchronize()?;
    executors
        .get_mut(&matrix.n())
        .ok_or_else(|| hip_error("missing shared Opus executor"))?
        .run_whole_scaled_shared(matrix)
        .map_err(|error| hip_error(format!("shared NPU Opus projection failed: {error}")))?;
    let (out0, width0) = outputs[0];
    gpu.unpack_opus_npu_output(
        &output_view,
        out0,
        width0,
        outputs.get(1).copied(),
        outputs.get(2).copied(),
        rows,
        rdna_io_layout(layout),
    )?;
    Ok(true)
}

impl LinearProjector for NpuOpusProjector {
    fn project_layer(
        &mut self,
        gpu: &mut Gpu,
        layer_idx: usize,
        normalized_input: &GpuTensor,
        residual_and_output: &GpuTensor,
        rows: usize,
    ) -> HipResult<bool> {
        let resident_layer_limit = std::env::var("HIPFIRE_EMBED_RESIDENT_LAYER_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(usize::MAX);
        if rows != NpuEmbeddingLayerAttentionDenseW8::rows()
            || self.resident_layer.is_none()
            || !self.resident_ffn_selected
            || layer_idx >= resident_layer_limit
        {
            return Ok(false);
        }
        if layer_idx >= self.layers.len() {
            return Err(hip_error(format!(
                "missing completed resident layer {layer_idx}"
            )));
        }
        if self
            .resident_layer
            .as_ref()
            .and_then(|state| state.attention_weights.get(layer_idx))
            .is_none_or(|weights| !weights.is_available())
        {
            return Ok(false);
        }
        let needs_io = self
            .resident_layer
            .as_ref()
            .is_some_and(|state| state.io.is_none());
        if needs_io {
            let input_bytes = self
                .resident_layer
                .as_ref()
                .expect("checked resident layer")
                .attention
                .input_bytes();
            let mut input = gpu.alloc_shared_gtt(input_bytes)?;
            let mut hidden =
                gpu.alloc_shared_gtt(NpuEmbeddingLayerAttentionDenseW8::hidden_backing_bytes())?;
            let tail_output_bytes = self
                .resident_layer
                .as_ref()
                .expect("checked resident layer")
                .tail
                .output_bytes();
            let mut residual = gpu.alloc_shared_gtt(tail_output_bytes)?;
            let mut ffn =
                gpu.alloc_shared_gtt(NpuEmbeddingPostFfnDirectTailBf16x2::combined_bytes())?;
            let activation_bytes = NpuEmbeddingLayerAttentionDenseW8::activation_bytes();
            input.as_mut_slice()[..activation_bytes].fill(0);
            hidden.as_mut_slice().fill(0);
            residual.as_mut_slice().fill(0);
            ffn.as_mut_slice().fill(0);
            let input_gpu = gpu.import_dmabuf(
                input.dmabuf_fd(),
                activation_bytes,
                &[activation_bytes],
                DType::Raw,
            )?;
            let residual_gpu = gpu.import_dmabuf(
                residual.dmabuf_fd(),
                residual.len(),
                &[rows * 768],
                DType::BF16,
            )?;
            let state = self
                .resident_layer
                .as_mut()
                .expect("checked resident layer");
            state
                .attention
                .attach_shared_input(input.dmabuf_fd(), input.len())
                .map_err(|error| hip_error(format!("attach completed-layer input: {error}")))?;
            state
                .attention
                .attach_shared_hidden(hidden.dmabuf_fd(), hidden.len())
                .map_err(|error| hip_error(format!("attach completed-layer hidden: {error}")))?;
            state
                .ffn
                .attach_shared_input(hidden.dmabuf_fd(), hidden.len())
                .map_err(|error| hip_error(format!("attach completed-layer FFN: {error}")))?;
            state
                .ffn
                .attach_shared_output(ffn.dmabuf_fd(), ffn.len())
                .map_err(|error| {
                    hip_error(format!("attach completed-layer FFN output: {error}"))
                })?;
            if state.tail.consumes_split_x() {
                state
                    .tail
                    .attach_shared_split_state(
                        ffn.dmabuf_fd(),
                        ffn.len(),
                        hidden.dmabuf_fd(),
                        hidden.len(),
                        residual.dmabuf_fd(),
                        residual.len(),
                    )
                    .map_err(|error| {
                        hip_error(format!("attach split-X completed-layer tail: {error}"))
                    })?;
            } else {
                state
                    .tail
                    .attach_shared_state(
                        ffn.dmabuf_fd(),
                        ffn.len(),
                        residual.dmabuf_fd(),
                        residual.len(),
                    )
                    .map_err(|error| hip_error(format!("attach completed-layer tail: {error}")))?;
            }
            if let Some(prep) = state.next_prep.as_mut() {
                prep.attach_shared(
                    residual.dmabuf_fd(),
                    residual.len(),
                    input.dmabuf_fd(),
                    NpuEmbeddingNextLayerPrepW8::output_bytes(),
                )
                .map_err(|error| hip_error(format!("attach next-layer prep: {error}")))?;
            }
            if let Some(prep) = state.residual_prep.as_mut() {
                prep.attach_shared(
                    residual.dmabuf_fd(),
                    residual.len(),
                    input.dmabuf_fd(),
                    input.len(),
                )
                .map_err(|error| hip_error(format!("attach residual prep: {error}")))?;
                prep.fill_output(0).map_err(|error| {
                    hip_error(format!("initialize residual prep output: {error}"))
                })?;
            }
            if let Some(final_norm_mean) = state.final_norm_mean.as_mut() {
                final_norm_mean
                    .attach_shared_completed(residual.dmabuf_fd(), residual.len())
                    .map_err(|error| hip_error(format!("attach final norm/mean input: {error}")))?;
            }
            state.io = Some(ResidentLayerIo {
                input_gpu,
                residual_gpu,
                _input: input,
                residual,
                hidden,
                ffn,
            });
        }

        let matrix_key = MatrixGpuKey {
            layer: layer_idx,
            role: MatrixRole::Qkv,
        };
        if !self.awq_gpu.contains_key(&matrix_key) {
            let scale = self
                .resident_layer
                .as_ref()
                .and_then(|state| state.attention_weights.get(layer_idx))
                .and_then(ResidentLayerAttentionWeights::awq_scale)
                .map(<[f32]>::to_vec);
            if let Some(scale) = scale {
                self.awq_gpu
                    .insert(matrix_key, gpu.upload_owned_f32(&scale, &[scale.len()])?);
            }
        }

        let compare_layer = compare_this_layer_enabled(layer_idx);
        let (external_residual, input_prepared) = {
            let state = self
                .resident_layer
                .as_mut()
                .expect("checked resident layer");
            (
                state.attention.uses_external_residual(),
                state.prepared_input_layer.take() == Some(layer_idx),
            )
        };
        let skip_host_residual = external_residual && input_prepared && !compare_layer;
        let residual = if skip_host_residual {
            Vec::new()
        } else {
            let residual_view = self
                .resident_layer
                .as_ref()
                .and_then(|state| state.io.as_ref())
                .expect("completed resident layer I/O")
                .residual_gpu
                .view();
            gpu.cast_f32_to_bf16(residual_and_output, &residual_view)?;
            gpu.device_synchronize()?;
            self.resident_layer
                .as_ref()
                .and_then(|state| state.io.as_ref())
                .expect("completed resident layer I/O")
                .residual
                .as_slice()[..rows * 768 * size_of::<u16>()]
                .chunks_exact(size_of::<u16>())
                .map(|word| u16::from_le_bytes([word[0], word[1]]))
                .collect::<Vec<_>>()
        };
        if compare_this_layer_enabled(layer_idx) {
            let original = gpu.download_f32(residual_and_output)?;
            let rounded = residual
                .iter()
                .map(|&bits| f32::from_bits((bits as u32) << 16))
                .collect::<Vec<_>>();
            let (cosine, max_abs) = host_metrics(&rounded, &original);
            eprintln!(
                "embeddinggemma_resident_residual_compare layer={layer_idx} cosine={cosine:.8} max_abs={max_abs:.7}"
            );
        }
        let state = self
            .resident_layer
            .as_mut()
            .expect("checked resident layer");
        if layer_idx >= state.attention_weights.len() || layer_idx >= state.tail_params.len() {
            return Err(hip_error(format!(
                "missing completed resident layer payload {layer_idx}"
            )));
        }
        if external_residual {
            if !input_prepared {
                let completed = completed_high_bf16x2(&residual, rows, 768)?;
                let prep = state
                    .residual_prep
                    .as_mut()
                    .ok_or_else(|| hip_error("missing resident residual prep"))?;
                prep.write_bootstrap_bf16x2(&completed).map_err(|error| {
                    hip_error(format!("stage initial layer residual {layer_idx}: {error}"))
                })?;
                prep.run_bootstrap().map_err(|error| {
                    hip_error(format!(
                        "prepare initial layer residual {layer_idx}: {error}"
                    ))
                })?;
                if compare_layer {
                    let mismatches = external_residual_record_mismatches(prep.output(), &residual)?;
                    eprintln!(
                        "embeddinggemma_external_residual_producer_compare layer={layer_idx} mismatches={mismatches}"
                    );
                }
            }
        } else {
            state
                .attention
                .set_residual_bf16(&mut state.attention_weights[layer_idx], &residual)
                .map_err(|error| hip_error(format!("stage layer {layer_idx} residual: {error}")))?;
        }
        state
            .attention
            .prepare_layer(&state.attention_weights[layer_idx])
            .map_err(|error| hip_error(format!("prepare layer {layer_idx}: {error}")))?;
        if !input_prepared {
            let input_view = state
                .io
                .as_ref()
                .expect("completed resident layer I/O")
                .input_gpu
                .view();
            gpu.pack_opus_npu_activations(
                normalized_input,
                self.awq_gpu.get(&matrix_key).map(OwnedTensor::view),
                &input_view,
                rows,
                768,
                resident_attention_layout(),
            )?;
            gpu.device_synchronize()?;
        }
        state
            .attention
            .run_shared(&state.attention_weights[layer_idx])
            .map_err(|error| {
                hip_error(format!("completed-layer attention {layer_idx}: {error}"))
            })?;
        let split_x_tail = state.tail.consumes_split_x();
        let resident_output_state = if split_x_tail && !compare_layer {
            Vec::new()
        } else {
            state
                .attention
                .read_hidden_f32(&state.attention_weights[layer_idx])
                .map_err(|error| hip_error(format!("read resident state {layer_idx}: {error}")))?
        };
        let direct_x_consumer = state.ffn.consumes_direct_x();
        let pre_ffn_state = if direct_x_consumer && !compare_layer {
            None
        } else {
            Some(
                state
                    .attention
                    .read_pre_ffn_state(&state.attention_weights[layer_idx])
                    .map_err(|error| {
                        hip_error(format!("read resident pre-FFN state {layer_idx}: {error}"))
                    })?,
            )
        };
        if compare_layer {
            let pre_ffn_state = pre_ffn_state
                .as_ref()
                .expect("comparison reads resident pre-FFN state");
            let finite_inverse = pre_ffn_state
                .inverse
                .iter()
                .filter(|value| value.is_finite())
                .count();
            let inverse_min = pre_ffn_state
                .inverse
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .fold(f32::INFINITY, f32::min);
            let inverse_max = pre_ffn_state
                .inverse
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .fold(f32::NEG_INFINITY, f32::max);
            let exception = pre_ffn_state.exception.as_ref().map(|exception| {
                let values = exception
                    .x
                    .iter()
                    .map(|&bits| f32::from_bits((bits as u32) << 16))
                    .collect::<Vec<_>>();
                let finite = values.iter().filter(|value| value.is_finite()).count();
                let min = values.iter().copied().fold(f32::INFINITY, f32::min);
                let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                (exception.column, finite, min, max)
            });
            eprintln!(
                "embeddinggemma_resident_pre_ffn_state layer={layer_idx} finite_inverse={finite_inverse}/{rows} inverse_min={inverse_min:.7} inverse_max={inverse_max:.7} exception={exception:?}"
            );
        }
        let (resident_hidden, reconstructed_x) = if state
            .attention
            .outputs_direct_x(&state.attention_weights[layer_idx])
        {
            if split_x_tail && !compare_layer {
                (Vec::new(), Vec::new())
            } else {
                let direct_x = resident_output_state
                    .iter()
                    .map(|value| (value.to_bits() >> 16) as u16)
                    .collect::<Vec<_>>();
                let normalized_h_f32 = if direct_x_consumer && !compare_layer {
                    Vec::new()
                } else {
                    let normalized_h = normalize_direct_x_bf16(
                        &direct_x,
                        &pre_ffn_state
                            .as_ref()
                            .expect("host normalization reads pre-FFN state")
                            .inverse,
                        &state.tail_pre_norms[layer_idx],
                        rows,
                        768,
                    )?;
                    if !direct_x_consumer {
                        write_bf16_prefix(
                            state
                                .io
                                .as_mut()
                                .expect("completed resident layer I/O")
                                .hidden
                                .as_mut_slice(),
                            &normalized_h,
                        )?;
                        state.ffn.sync_shared_input().map_err(|error| {
                            hip_error(format!("sync direct-X normalized H {layer_idx}: {error}"))
                        })?;
                    }
                    normalized_h
                        .iter()
                        .map(|&bits| f32::from_bits((bits as u32) << 16))
                        .collect()
                };
                (normalized_h_f32, direct_x)
            }
        } else {
            let pre_ffn_state = pre_ffn_state
                .as_ref()
                .expect("reconstructible H reads pre-FFN state");
            let reconstructed_x = reconstruct_attention_residual_bf16(
                &resident_output_state,
                &pre_ffn_state.inverse,
                pre_ffn_state.exception.as_ref(),
                &state.tail_pre_norms[layer_idx],
                &residual,
                rows,
                768,
            )?;
            (resident_output_state, reconstructed_x)
        };
        if compare_layer {
            self.debug_resident_hidden = Some(resident_hidden);
            self.debug_resident_residual = Some(
                reconstructed_x
                    .iter()
                    .map(|&bits| f32::from_bits((bits as u32) << 16))
                    .collect(),
            );
            self.debug_resident_exception = pre_ffn_state
                .as_ref()
                .and_then(|state| state.exception.as_ref())
                .map(|exception| {
                    (
                        exception.column,
                        exception
                            .x
                            .iter()
                            .map(|&bits| f32::from_bits((bits as u32) << 16))
                            .collect(),
                    )
                });
        }
        state
            .ffn
            .run_shared(&state.ffn_weights[layer_idx])
            .map_err(|error| hip_error(format!("completed-layer FFN {layer_idx}: {error}")))?;
        if !state.tail.consumes_split_x() {
            write_residual_component(
                state
                    .io
                    .as_mut()
                    .expect("completed resident layer I/O")
                    .ffn
                    .as_mut_slice(),
                &reconstructed_x,
                rows,
                768,
            )?;
        }
        if compare_this_layer_enabled(layer_idx) {
            self.debug_resident_ffn =
                Some(state.ffn.read_canonical_output_f32().map_err(|error| {
                    hip_error(format!("read resident FFN {layer_idx}: {error}"))
                })?);
        }
        if !state.tail.consumes_split_x() {
            state.tail.sync_shared_inputs().map_err(|error| {
                hip_error(format!("sync completed-layer tail {layer_idx}: {error}"))
            })?;
        }
        state
            .tail
            .run_shared(&state.tail_params[layer_idx])
            .map_err(|error| hip_error(format!("completed-layer tail {layer_idx}: {error}")))?;
        let next_layer = layer_idx + 1;
        let has_resident_next = next_layer < state.attention_weights.len()
            && next_layer < resident_layer_limit
            && state.attention_weights[next_layer].is_available();
        let mut prepared_activation = false;
        let mut prepared_residual = false;
        if has_resident_next {
            if let (Some(prep), Some(params)) = (
                state.next_prep.as_mut(),
                state.next_prep_params.get(next_layer),
            ) {
                prep.run_shared(params).map_err(|error| {
                    hip_error(format!(
                        "prepare completed-layer input {layer_idx}->{next_layer}: {error}"
                    ))
                })?;
                prepared_activation = true;
            }
            if let Some(prep) = state.residual_prep.as_mut() {
                prep.run_shared().map_err(|error| {
                    hip_error(format!(
                        "prepare completed-layer residual {layer_idx}->{next_layer}: {error}"
                    ))
                })?;
                prepared_residual = true;
            }
            if prepared_activation {
                state.prepared_input_layer = Some(next_layer);
            }
        }
        let compare_layer = compare_this_layer_enabled(layer_idx);
        let final_norm_mean_ready = next_layer == state.attention_weights.len()
            && state.final_norm_mean.is_some()
            && state.final_norm_mean_params.is_some();
        if should_materialize_completed_output(
            !has_resident_next && !final_norm_mean_ready,
            prepared_activation,
            prepared_residual,
            compare_layer,
        ) {
            let completed = state
                .tail
                .read_output_f32()
                .map_err(|error| hip_error(format!("read completed layer {layer_idx}: {error}")))?;
            if compare_layer {
                let expected = direct_tail_reference(
                    &reconstructed_x,
                    self.debug_resident_ffn
                        .as_ref()
                        .expect("resident FFN debug output"),
                    &state.tail_post_norms[layer_idx],
                    1.0e-6,
                );
                let (cosine, max_abs) = host_metrics(&completed, &expected);
                eprintln!(
                "embeddinggemma_resident_tail_compare layer={layer_idx} cosine={cosine:.8} max_abs={max_abs:.7}"
            );
            }
            copy_host_output(gpu, residual_and_output, &completed)?;
        }
        Ok(true)
    }

    fn finalize_pooled_hidden(
        &mut self,
        rows: usize,
        hidden: usize,
        mode: PoolingMode,
    ) -> HipResult<Option<Vec<f32>>> {
        if rows != NpuEmbeddingFinalNormMean::rows() || hidden != 768 || mode != PoolingMode::Mean {
            return Ok(None);
        }
        let Some(state) = self.resident_layer.as_mut() else {
            return Ok(None);
        };
        let (Some(kernel), Some(params)) = (
            state.final_norm_mean.as_mut(),
            state.final_norm_mean_params.as_ref(),
        ) else {
            return Ok(None);
        };
        kernel
            .run_shared(params)
            .map_err(|error| hip_error(format!("final norm/mean: {error}")))?;
        Ok(Some(kernel.read_pooled_f32()))
    }

    fn has_prepared_layer_input(&self, layer_idx: usize, rows: usize) -> bool {
        let resident_layer_limit = std::env::var("HIPFIRE_EMBED_RESIDENT_LAYER_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(usize::MAX);
        rows == NpuEmbeddingNextLayerPrepW8::rows()
            && layer_idx < resident_layer_limit
            && self.resident_layer.as_ref().is_some_and(|state| {
                state.next_prep.is_some() && state.prepared_input_layer == Some(layer_idx)
            })
    }

    fn take_layer_debug_hidden(&mut self) -> Option<Vec<f32>> {
        self.debug_resident_hidden.take()
    }

    fn take_layer_debug_residual(&mut self) -> Option<Vec<f32>> {
        self.debug_resident_residual.take()
    }

    fn take_layer_debug_exception(&mut self) -> Option<(usize, Vec<f32>)> {
        self.debug_resident_exception.take()
    }

    fn take_layer_debug_ffn(&mut self) -> Option<Vec<f32>> {
        self.debug_resident_ffn.take()
    }

    fn project_attention(
        &mut self,
        gpu: &mut Gpu,
        layer_idx: usize,
        input: &GpuTensor,
        attention_output: &GpuTensor,
        projected_output: &GpuTensor,
        rows: usize,
    ) -> HipResult<AttentionBoundary> {
        if rows != NpuResidentAttentionDenseW8::rows()
            || self.resident_attention.is_none()
            || !self.resident_attention_selected
        {
            return Ok(AttentionBoundary::Fallback);
        }
        let matrix_key = MatrixGpuKey {
            layer: layer_idx,
            role: MatrixRole::Qkv,
        };
        if !self.awq_gpu.contains_key(&matrix_key) {
            let scale = self
                .resident_attention
                .as_ref()
                .and_then(|state| state.weights.get(layer_idx))
                .and_then(NpuResidentAttentionDenseW8Weights::awq_scale)
                .map(<[f32]>::to_vec);
            if let Some(scale) = scale {
                self.awq_gpu
                    .insert(matrix_key, gpu.upload_owned_f32(&scale, &[scale.len()])?);
            }
        }
        let state = self
            .resident_attention
            .as_mut()
            .expect("checked resident attention");
        if layer_idx >= state.weights.len() {
            return Err(hip_error(format!(
                "missing resident attention layer {layer_idx}"
            )));
        }
        if state.input.is_none() {
            let mut input_buffer =
                gpu.alloc_shared_gtt(NpuResidentAttentionDenseW8::input_bytes())?;
            input_buffer.as_mut_slice().fill(0);
            let input_gpu = gpu.import_dmabuf(
                input_buffer.dmabuf_fd(),
                NpuResidentAttentionDenseW8::input_bytes(),
                &[NpuResidentAttentionDenseW8::input_bytes()],
                DType::Raw,
            )?;
            state
                .executor
                .attach_shared_input(
                    input_buffer.dmabuf_fd(),
                    NpuResidentAttentionDenseW8::input_bytes(),
                )
                .map_err(|error| hip_error(format!("attach resident attention input: {error}")))?;
            state.input = Some(SharedAttentionInput {
                input_gpu,
                _input_buffer: input_buffer,
            });
        }
        let layout = resident_attention_layout();
        gpu.pack_opus_npu_activations(
            input,
            self.awq_gpu.get(&matrix_key).map(OwnedTensor::view),
            &state
                .input
                .as_ref()
                .expect("resident attention input")
                .input_gpu
                .view(),
            rows,
            768,
            layout,
        )?;
        gpu.device_synchronize()?;
        let needs_staging = state
            .output
            .as_ref()
            .is_some_and(|output| output.staging[layer_idx].is_none());
        if needs_staging {
            let mut staging = gpu.alloc_shared_gtt(NpuResidentAttentionDenseW8::staging_bytes())?;
            staging.as_mut_slice().fill(0);
            state
                .executor
                .attach_shared_staging(
                    &mut state.weights[layer_idx],
                    staging.dmabuf_fd(),
                    staging.len(),
                )
                .map_err(|error| {
                    hip_error(format!("attach resident attention staging: {error}"))
                })?;
            let output_state = state.output.as_mut().expect("checked output state");
            output_state
                .executor
                .attach_shared_layer_input(
                    &mut output_state.weights[layer_idx],
                    staging.dmabuf_fd(),
                    staging.len(),
                )
                .map_err(|error| hip_error(format!("attach attention output input: {error}")))?;
            output_state.staging[layer_idx] = Some(staging);
        }
        let needs_output = state
            .output
            .as_ref()
            .is_some_and(|output| output.io.is_none());
        if needs_output {
            let mut output_buffer = gpu.alloc_shared_gtt(NpuAttentionOutputBf16::output_bytes())?;
            output_buffer.as_mut_slice().fill(0);
            let output_gpu = gpu.import_dmabuf(
                output_buffer.dmabuf_fd(),
                output_buffer.len(),
                &[rows * 768],
                DType::F32,
            )?;
            let output_state = state.output.as_mut().expect("checked output state");
            output_state
                .executor
                .attach_shared_output(output_buffer.dmabuf_fd(), output_buffer.len())
                .map_err(|error| {
                    hip_error(format!("attach attention output destination: {error}"))
                })?;
            output_state.io = Some(SharedAttentionOutput {
                output_gpu,
                _output_buffer: output_buffer,
            });
        }
        state
            .executor
            .run_shared_to_device(&state.weights[layer_idx])
            .map_err(|error| hip_error(format!("resident NPU attention failed: {error}")))?;

        if let Some(output_state) = state.output.as_mut() {
            output_state
                .executor
                .run(&output_state.weights[layer_idx])
                .map_err(|error| hip_error(format!("resident NPU output projection: {error}")))?;
            let output_view = output_state
                .io
                .as_ref()
                .expect("resident attention output I/O")
                .output_gpu
                .view();
            gpu.memcpy_dtod_at_auto(
                &projected_output.buf,
                0,
                &output_view.buf,
                0,
                NpuAttentionOutputBf16::output_bytes(),
            )?;
            return Ok(AttentionBoundary::OutputProjected);
        }
        let head_major = state
            .executor
            .read_output_f32(&state.weights[layer_idx])
            .map_err(|error| hip_error(format!("read resident NPU attention: {error}")))?;
        let mut token_major = vec![0.0f32; head_major.len()];
        for token in 0..256 {
            for head in 0..3 {
                let source = (head * 256 + token) * 256;
                let target = token * 768 + head * 256;
                token_major[target..target + 256]
                    .copy_from_slice(&head_major[source..source + 256]);
            }
        }
        copy_host_output(gpu, attention_output, &token_major)?;
        Ok(AttentionBoundary::AttentionOnly)
    }

    fn project(
        &mut self,
        gpu: &mut Gpu,
        layer_idx: usize,
        projection: Projection,
        weight: &WeightTensor,
        input: &GpuTensor,
        output: &GpuTensor,
        rows: usize,
    ) -> HipResult<()> {
        let Self {
            executors,
            layers,
            shared_io,
            awq_gpu,
            ..
        } = self;
        let layer = layers
            .get(layer_idx)
            .ok_or_else(|| hip_error(format!("missing packed layer {layer_idx}")))?;
        let matrix = match projection {
            Projection::Query => &layer.query,
            Projection::Key => &layer.key,
            Projection::Value => &layer.value,
            Projection::AttentionOutput => &layer.attention_output,
            Projection::Gate => &layer.gate,
            Projection::Up => &layer.up,
            Projection::Down => match &layer.down {
                Some(matrix) => matrix,
                None => return weight_gemm(gpu, weight, input, output, rows),
            },
        };
        if matrix.k() != weight.k || matrix.n() != weight.m {
            return Err(hip_error(format!(
                "projection shape mismatch: packed={}x{} GPU={}x{}",
                matrix.n(),
                matrix.k(),
                weight.m,
                weight.k
            )));
        }
        let width = matrix.n();
        let role = match projection {
            Projection::Query => MatrixRole::Query,
            Projection::Key => MatrixRole::Key,
            Projection::Value => MatrixRole::Value,
            Projection::AttentionOutput => MatrixRole::AttentionOutput,
            Projection::Gate => MatrixRole::Gate,
            Projection::Up => MatrixRole::Up,
            Projection::Down => MatrixRole::Down,
        };
        if try_shared_projection(
            gpu,
            executors,
            shared_io,
            awq_gpu,
            MatrixGpuKey {
                layer: layer_idx,
                role,
            },
            matrix,
            input,
            &[(output, width)],
            rows,
        )? {
            return Ok(());
        }
        let input_host = gpu.download_f32(input)?;
        let mut output_host = vec![0.0f32; rows * width];
        let executor = executors
            .get_mut(&width)
            .ok_or_else(|| hip_error(format!("missing N={width} executor")))?;
        executor
            .run_f32(matrix, rows, &input_host, &mut output_host)
            .map_err(|error| hip_error(format!("NPU Opus projection failed: {error}")))?;
        let uploaded = gpu.upload_f32(&output_host, &[rows * width])?;
        gpu.memcpy_dtod_at_auto(&output.buf, 0, &uploaded.buf, 0, output_host.len() * 4)?;
        gpu.free_tensor(uploaded)?;
        Ok(())
    }

    fn project_qkv(
        &mut self,
        gpu: &mut Gpu,
        layer_idx: usize,
        wq: &WeightTensor,
        wk: &WeightTensor,
        wv: &WeightTensor,
        input: &GpuTensor,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        rows: usize,
    ) -> HipResult<()> {
        if self
            .layers
            .get(layer_idx)
            .and_then(|layer| layer.qkv.as_ref())
            .is_none()
        {
            self.project(gpu, layer_idx, Projection::Query, wq, input, q, rows)?;
            self.project(gpu, layer_idx, Projection::Key, wk, input, k, rows)?;
            return self.project(gpu, layer_idx, Projection::Value, wv, input, v, rows);
        }
        let Self {
            executors,
            layers,
            shared_io,
            awq_gpu,
            ..
        } = self;
        let matrix = layers[layer_idx].qkv.as_ref().unwrap();
        let widths = [wq.m, wk.m, wv.m];
        if wq.k != matrix.k()
            || wk.k != matrix.k()
            || wv.k != matrix.k()
            || widths.iter().sum::<usize>() != matrix.n()
        {
            return Err(hip_error("combined q/k/v projection geometry mismatch"));
        }
        if try_shared_projection(
            gpu,
            executors,
            shared_io,
            awq_gpu,
            MatrixGpuKey {
                layer: layer_idx,
                role: MatrixRole::Qkv,
            },
            matrix,
            input,
            &[(q, widths[0]), (k, widths[1]), (v, widths[2])],
            rows,
        )? {
            return Ok(());
        }
        let input_host = gpu.download_f32(input)?;
        let mut combined = vec![0.0f32; rows * matrix.n()];
        executors
            .get_mut(&matrix.n())
            .ok_or_else(|| hip_error("missing combined q/k/v executor"))?
            .run_f32(matrix, rows, &input_host, &mut combined)
            .map_err(|error| hip_error(format!("combined q/k/v failed: {error}")))?;
        let outputs = [q, k, v];
        let mut offset = 0usize;
        for (output, width) in outputs.into_iter().zip(widths) {
            let mut role = Vec::with_capacity(rows * width);
            for row in 0..rows {
                let start = row * matrix.n() + offset;
                role.extend_from_slice(&combined[start..start + width]);
            }
            copy_host_output(gpu, output, &role)?;
            offset += width;
        }
        Ok(())
    }

    fn project_gate_up(
        &mut self,
        gpu: &mut Gpu,
        layer_idx: usize,
        gate_weight: &WeightTensor,
        up_weight: &WeightTensor,
        input: &GpuTensor,
        gate: &GpuTensor,
        up: &GpuTensor,
        rows: usize,
    ) -> HipResult<()> {
        if self
            .layers
            .get(layer_idx)
            .and_then(|layer| layer.gate_up.as_ref())
            .is_none()
        {
            self.project(
                gpu,
                layer_idx,
                Projection::Gate,
                gate_weight,
                input,
                gate,
                rows,
            )?;
            return self.project(gpu, layer_idx, Projection::Up, up_weight, input, up, rows);
        }
        let Self {
            executors,
            layers,
            shared_io,
            awq_gpu,
            ..
        } = self;
        let matrix = layers[layer_idx].gate_up.as_ref().unwrap();
        if gate_weight.k != matrix.k()
            || up_weight.k != matrix.k()
            || gate_weight.m + up_weight.m != matrix.n()
        {
            return Err(hip_error("combined gate/up projection geometry mismatch"));
        }
        if try_shared_projection(
            gpu,
            executors,
            shared_io,
            awq_gpu,
            MatrixGpuKey {
                layer: layer_idx,
                role: MatrixRole::GateUp,
            },
            matrix,
            input,
            &[(gate, gate_weight.m), (up, up_weight.m)],
            rows,
        )? {
            return Ok(());
        }
        let input_host = gpu.download_f32(input)?;
        let mut combined = vec![0.0f32; rows * matrix.n()];
        executors
            .get_mut(&matrix.n())
            .ok_or_else(|| hip_error("missing combined gate/up executor"))?
            .run_f32(matrix, rows, &input_host, &mut combined)
            .map_err(|error| hip_error(format!("combined gate/up failed: {error}")))?;
        let width = gate_weight.m;
        let mut gate_host = Vec::with_capacity(rows * width);
        let mut up_host = Vec::with_capacity(rows * up_weight.m);
        for row in 0..rows {
            let start = row * matrix.n();
            gate_host.extend_from_slice(&combined[start..start + width]);
            up_host.extend_from_slice(&combined[start + width..start + matrix.n()]);
        }
        copy_host_output(gpu, gate, &gate_host)?;
        copy_host_output(gpu, up, &up_host)
    }

    #[allow(clippy::too_many_arguments)]
    fn project_ffn(
        &mut self,
        gpu: &mut Gpu,
        layer_idx: usize,
        gate_weight: &WeightTensor,
        up_weight: &WeightTensor,
        down_weight: &WeightTensor,
        input: &GpuTensor,
        gate: &GpuTensor,
        up: &GpuTensor,
        activated: &GpuTensor,
        output: &GpuTensor,
        rows: usize,
    ) -> HipResult<()> {
        if self
            .resident_ffn
            .as_ref()
            .is_some_and(|state| rows <= state.rows())
            && self.resident_ffn_selected
        {
            let layer = self
                .layers
                .get(layer_idx)
                .ok_or_else(|| hip_error(format!("missing packed layer {layer_idx}")))?;
            if gate_weight.k == 768
                && gate_weight.m == 1152
                && up_weight.k == 768
                && up_weight.m == 1152
                && down_weight.k == 1152
                && down_weight.m == 768
            {
                let state = self.resident_ffn.as_mut().expect("checked resident FFN");
                let layout = state.layout();
                if state.io().is_none() {
                    let mut input_buffer = gpu.alloc_shared_gtt(layout.input_bytes)?;
                    let mut output_buffer = gpu.alloc_shared_gtt(layout.output_bytes)?;
                    input_buffer.as_mut_slice().fill(0);
                    output_buffer.as_mut_slice().fill(0);
                    let input_gpu = gpu.import_dmabuf(
                        input_buffer.dmabuf_fd(),
                        layout.input_bytes,
                        &[layout.input_bytes],
                        DType::Raw,
                    )?;
                    let output_gpu = gpu.import_dmabuf(
                        output_buffer.dmabuf_fd(),
                        layout.output_bytes,
                        &[layout.output_bytes],
                        DType::Raw,
                    )?;
                    state
                        .attach_io(
                            input_buffer.dmabuf_fd(),
                            layout.input_bytes,
                            output_buffer.dmabuf_fd(),
                            layout.output_bytes,
                            SharedProjectionIo {
                                input_gpu,
                                output_gpu,
                                _input_buffer: input_buffer,
                                _output_buffer: output_buffer,
                            },
                        )
                        .map_err(|error| hip_error(format!("attach resident FFN I/O: {error}")))?;
                }
                let matrix_key = MatrixGpuKey {
                    layer: layer_idx,
                    role: MatrixRole::GateUp,
                };
                if !self.awq_gpu.contains_key(&matrix_key) {
                    if let Some(scale) = layer.gate.awq_scale() {
                        self.awq_gpu
                            .insert(matrix_key, gpu.upload_owned_f32(scale, &[scale.len()])?);
                    }
                }
                let io = state.io().expect("resident FFN I/O allocated");
                gpu.pack_opus_npu_activations(
                    input,
                    self.awq_gpu.get(&matrix_key).map(OwnedTensor::view),
                    &io.input_gpu.view(),
                    rows,
                    768,
                    layout.rdna,
                )?;
                gpu.device_synchronize()?;
                state
                    .run_layer(layer_idx)
                    .map_err(|error| hip_error(format!("resident NPU FFN failed: {error}")))?;
                let io = state.io().expect("resident FFN I/O allocated");
                gpu.unpack_opus_npu_output(
                    &io.output_gpu.view(),
                    output,
                    768,
                    None,
                    None,
                    rows,
                    layout.rdna,
                )?;
                return Ok(());
            }
        }
        self.project_gate_up(
            gpu,
            layer_idx,
            gate_weight,
            up_weight,
            input,
            gate,
            up,
            rows,
        )?;
        gpu.gelu_mul_f32(gate, up, activated)?;
        self.project(
            gpu,
            layer_idx,
            Projection::Down,
            down_weight,
            activated,
            output,
            rows,
        )
    }
}

#[derive(Clone, Copy)]
struct ResidentFfnLayout {
    rdna: OpusNpuIoLayout,
    input_bytes: usize,
    output_bytes: usize,
}

fn resident_attention_layout() -> OpusNpuIoLayout {
    OpusNpuIoLayout::new(
        true,
        8,
        256,
        3,
        1280,
        5,
        15,
        16384,
        NpuResidentAttentionDenseW8::input_bytes(),
        NpuResidentAttentionDenseW8::output_bytes(),
        false,
        1280,
    )
}

fn resident_ffn_w4_layout() -> ResidentFfnLayout {
    let input_bytes = NpuResidentFfnW4::input_bytes();
    let output_bytes = NpuResidentFfnW4::output_bytes();
    ResidentFfnLayout {
        rdna: OpusNpuIoLayout::new(
            false,
            8,
            256,
            3,
            768,
            3,
            9,
            NpuResidentFfnW4::input_block_bytes(),
            input_bytes,
            output_bytes,
            true,
            768,
        ),
        input_bytes,
        output_bytes,
    }
}

fn resident_ffn_dense_w8_layout() -> ResidentFfnLayout {
    let input_bytes = NpuResidentFfnDenseW8::input_bytes();
    let output_bytes = NpuResidentFfnDenseW8::output_bytes();
    ResidentFfnLayout {
        rdna: OpusNpuIoLayout::new(
            true,
            8,
            256,
            3,
            768,
            6,
            18,
            NpuResidentFfnDenseW8::input_block_bytes(),
            input_bytes,
            output_bytes,
            true,
            768,
        )
        .with_input_repetition(
            NpuResidentFfnDenseW8::input_repeats(),
            NpuResidentFfnDenseW8::input_repeat_stride(),
        ),
        input_bytes,
        output_bytes,
    }
}

fn copy_host_output(gpu: &mut Gpu, output: &GpuTensor, values: &[f32]) -> HipResult<()> {
    let uploaded = gpu.upload_f32(values, &[values.len()])?;
    gpu.memcpy_dtod_at_auto(&output.buf, 0, &uploaded.buf, 0, values.len() * 4)?;
    gpu.free_tensor(uploaded)
}

fn executor(
    executors: &HashMap<usize, NpuOpusExecutor>,
    width: usize,
) -> Result<&NpuOpusExecutor, String> {
    executors
        .get(&width)
        .ok_or_else(|| format!("embeddinggemma NPU: missing N={width} executor"))
}

fn load_matrix(
    hfq: &HfqFile,
    executor: &NpuOpusExecutor,
    name: &str,
) -> Result<OpusPackedMatrix, String> {
    let (info, payload) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("embeddinggemma NPU: missing tensor {name}"))?;
    if info.shape.len() != 2 {
        return Err(format!(
            "embeddinggemma NPU: {name} must be a rank-2 Opus matrix, got qt={} shape={:?}",
            info.quant_type, info.shape
        ));
    }
    let n = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    OpusMatrixEncoding::classify(info.quant_type, payload.len(), k, n)
        .map_err(|error| format!("embeddinggemma NPU: classify {name}: {error}"))?;
    executor
        .pack_matrix(
            info.quant_type,
            k,
            n,
            &payload,
            load_awq_scale(hfq, name, k)?,
        )
        .map_err(|error| format!("embeddinggemma NPU: pack {name}: {error}"))
}

fn load_concat_matrix(
    hfq: &HfqFile,
    executor: Option<&NpuOpusExecutor>,
    names: &[String],
) -> Result<Option<OpusPackedMatrix>, String> {
    let Some(executor) = executor else {
        return Ok(None);
    };
    let mut quant_type = None;
    let mut k = None;
    let mut n = 0usize;
    let mut payload = Vec::new();
    let mut shared_awq: Option<Option<Vec<f32>>> = None;
    for name in names {
        let (info, bytes) = hfq
            .tensor_data_vec(name)
            .ok_or_else(|| format!("embeddinggemma NPU: missing tensor {name}"))?;
        if info.shape.len() != 2 {
            return Ok(None);
        }
        let matrix_n = info.shape[0] as usize;
        let matrix_k = info.shape[1] as usize;
        if quant_type.is_some_and(|value| value != info.quant_type)
            || k.is_some_and(|value| value != matrix_k)
        {
            return Ok(None);
        }
        OpusMatrixEncoding::classify(info.quant_type, bytes.len(), matrix_k, matrix_n)
            .map_err(|error| format!("embeddinggemma NPU: classify {name}: {error}"))?;
        let awq = load_awq_scale(hfq, name, matrix_k)?;
        if shared_awq.as_ref().is_some_and(|value| value != &awq) {
            return Ok(None);
        }
        shared_awq.get_or_insert_with(|| awq.clone());
        quant_type = Some(info.quant_type);
        k = Some(matrix_k);
        n += matrix_n;
        // Opus payloads are column-major blocks, so role concatenation is byte
        // concatenation when quant type and K agree.
        payload.extend_from_slice(&bytes);
    }
    let Some(quant_type) = quant_type else {
        return Ok(None);
    };
    let k = k.expect("non-empty concatenated matrix has K");
    executor
        .pack_matrix(quant_type, k, n, &payload, shared_awq.unwrap_or(None))
        .map(Some)
        .map_err(|error| format!("embeddinggemma NPU: pack concatenated roles: {error}"))
}

fn load_optional_matrix(
    hfq: &HfqFile,
    executor: &NpuOpusExecutor,
    name: &str,
) -> Result<Option<OpusPackedMatrix>, String> {
    let (info, payload) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("embeddinggemma NPU: missing tensor {name}"))?;
    if !matches!(info.quant_type, 33..=36) {
        return Ok(None);
    }
    if info.shape.len() != 2 {
        return Err(format!(
            "embeddinggemma NPU: {name} must be rank-2, got shape={:?}",
            info.shape
        ));
    }
    let n = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    OpusMatrixEncoding::classify(info.quant_type, payload.len(), k, n)
        .map_err(|error| format!("embeddinggemma NPU: classify {name}: {error}"))?;
    executor
        .pack_matrix(
            info.quant_type,
            k,
            n,
            &payload,
            load_awq_scale(hfq, name, k)?,
        )
        .map(Some)
        .map_err(|error| format!("embeddinggemma NPU: pack {name}: {error}"))
}

fn load_awq_scale(hfq: &HfqFile, name: &str, k: usize) -> Result<Option<Vec<f32>>, String> {
    let sidecar = name.strip_suffix(".weight").map_or_else(
        || format!("{name}.awq_scale.weight"),
        |stem| format!("{stem}.awq_scale.weight"),
    );
    let Some((info, data)) = hfq.tensor_data_vec(&sidecar) else {
        return Ok(None);
    };
    if info.quant_type != 1 || data.len() != k * 2 {
        return Err(format!("embeddinggemma NPU: {sidecar} must be f16[{k}]"));
    }
    Ok(Some(
        data.chunks_exact(2)
            .map(|bytes| f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect(),
    ))
}

fn load_vector_f32(hfq: &HfqFile, name: &str, length: usize) -> Result<Vec<f32>, String> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("embeddinggemma NPU: missing tensor {name}"))?;
    let values = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|bytes| f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect::<Vec<_>>(),
        2 => data
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect(),
        16 => data
            .chunks_exact(2)
            .map(|bytes| f32::from_bits((u16::from_le_bytes([bytes[0], bytes[1]]) as u32) << 16))
            .collect(),
        quant_type => {
            return Err(format!(
                "embeddinggemma NPU: {name} must be f16/f32/bf16, got qt={quant_type}"
            ));
        }
    };
    if values.len() != length {
        return Err(format!(
            "embeddinggemma NPU: {name} has {} values, expected {length}",
            values.len()
        ));
    }
    Ok(values)
}

fn load_vector_bf16(hfq: &HfqFile, name: &str, length: usize) -> Result<Vec<u16>, String> {
    Ok(load_vector_f32(hfq, name, length)?
        .into_iter()
        .map(f32_to_bf16_bits)
        .collect())
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn completed_high_bf16x2(residual: &[u16], rows: usize, hidden: usize) -> HipResult<Vec<u8>> {
    const PAD_ROWS: usize = 288;
    if rows != 256 || hidden != 768 || residual.len() != rows * hidden {
        return Err(hip_error(
            "initial completed-state BF16x2 geometry mismatch",
        ));
    }
    let mut output = vec![0u8; PAD_ROWS * 2 * hidden * size_of::<u16>()];
    for row in 0..rows {
        let target = row * 2 * hidden * size_of::<u16>();
        for (word, &bits) in output[target..target + hidden * 2]
            .chunks_exact_mut(2)
            .zip(&residual[row * hidden..(row + 1) * hidden])
        {
            word.copy_from_slice(&bits.to_le_bytes());
        }
    }
    Ok(output)
}

fn external_residual_record_mismatches(input: &[u8], residual: &[u16]) -> HipResult<usize> {
    const ROWS: usize = 256;
    const HIDDEN: usize = 768;
    const ACTIVATION_BYTES: usize = 4 * 45 * 16_384;
    const RECORD_BYTES: usize = 16_384;
    if residual.len() != ROWS * HIDDEN || input.len() < ACTIVATION_BYTES + 32 * RECORD_BYTES {
        return Err(hip_error("external residual record geometry mismatch"));
    }
    let records = &input[ACTIVATION_BYTES..];
    let mut mismatches = 0usize;
    for wave in 0..2 {
        for active_col in 0..4 {
            for core_row in 0..4 {
                let record = ((wave * 4 + active_col) * 4 + core_row) * RECORD_BYTES;
                let token_base = wave * 128 + core_row * 32 + active_col * 8;
                for row in 0..8 {
                    for hidden in 0..HIDDEN {
                        let offset = record + (row * HIDDEN + hidden) * 2;
                        let got = u16::from_le_bytes([records[offset], records[offset + 1]]);
                        mismatches +=
                            usize::from(got != residual[(token_base + row) * HIDDEN + hidden]);
                    }
                }
            }
        }
    }
    Ok(mismatches)
}

fn compare_this_layer_enabled(layer_idx: usize) -> bool {
    std::env::var("HIPFIRE_EMBED_COMPARE_RESIDENT_LAYER").is_ok_and(|value| value != "0")
        && std::env::var("HIPFIRE_EMBED_COMPARE_RESIDENT_LAYER_INDEX")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0)
            == layer_idx
}

fn should_materialize_completed_output(
    terminal_or_fallback: bool,
    prepared_activation: bool,
    prepared_residual: bool,
    compare_layer: bool,
) -> bool {
    terminal_or_fallback || !prepared_activation || !prepared_residual || compare_layer
}

fn host_metrics(left: &[f32], right: &[f32]) -> (f64, f32) {
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&left, &right) in left.iter().zip(right) {
        dot += left as f64 * right as f64;
        left_norm += (left as f64).powi(2);
        right_norm += (right as f64).powi(2);
        max_abs = max_abs.max((left - right).abs());
    }
    (dot / (left_norm.sqrt() * right_norm.sqrt()), max_abs)
}

fn direct_tail_reference(
    residual: &[u16],
    ffn: &[f32],
    post_norm: &[u16],
    epsilon: f32,
) -> Vec<f32> {
    const HIDDEN: usize = 768;
    let mut output = vec![0.0f32; ffn.len()];
    for token in 0..ffn.len() / HIDDEN {
        let base = token * HIDDEN;
        let sum = ffn[base..base + HIDDEN]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let inverse = (sum / HIDDEN as f32 + epsilon).sqrt().recip();
        for hidden in 0..HIDDEN {
            let index = base + hidden;
            let value = f32::from_bits((residual[index] as u32) << 16)
                + ffn[index] * f32::from_bits((post_norm[hidden] as u32) << 16) * inverse;
            output[index] = f32::from_bits((f32_to_bf16_bits(value) as u32) << 16);
        }
    }
    output
}

fn write_residual_component(
    destination: &mut [u8],
    residual: &[u16],
    rows: usize,
    hidden: usize,
) -> HipResult<()> {
    if residual.len() != rows * hidden || destination.len() < rows * hidden * 3 * 2 {
        return Err(hip_error(
            "completed-layer BF16x2 residual component geometry mismatch",
        ));
    }
    for row in 0..rows {
        let source = &residual[row * hidden..(row + 1) * hidden];
        let start = (row * 3 * hidden + 2 * hidden) * size_of::<u16>();
        for (encoded, &value) in destination[start..start + hidden * 2]
            .chunks_exact_mut(2)
            .zip(source)
        {
            encoded.copy_from_slice(&value.to_le_bytes());
        }
    }
    Ok(())
}

fn write_bf16_prefix(destination: &mut [u8], values: &[u16]) -> HipResult<()> {
    if destination.len() < values.len() * size_of::<u16>() {
        return Err(hip_error("direct-X normalized H backing is too small"));
    }
    for (encoded, &value) in destination[..values.len() * size_of::<u16>()]
        .chunks_exact_mut(size_of::<u16>())
        .zip(values)
    {
        encoded.copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

fn normalize_direct_x_bf16(
    direct_x: &[u16],
    pre_inverse: &[f32],
    pre_norm: &[u16],
    rows: usize,
    width: usize,
) -> HipResult<Vec<u16>> {
    if direct_x.len() != rows * width || pre_inverse.len() != rows || pre_norm.len() != width {
        return Err(hip_error("direct-X pre-FFN norm geometry mismatch"));
    }
    let mut output = Vec::with_capacity(direct_x.len());
    for row in 0..rows {
        for column in 0..width {
            let x = f32::from_bits((direct_x[row * width + column] as u32) << 16);
            let weight = f32::from_bits((pre_norm[column] as u32) << 16);
            output.push(f32_to_bf16_bits(x * weight * pre_inverse[row]));
        }
    }
    Ok(output)
}

fn reconstruct_attention_residual_bf16(
    hidden: &[f32],
    pre_inverse: &[f32],
    exception: Option<&NpuEmbeddingPreFfnException>,
    pre_norm: &[u16],
    fallback: &[u16],
    rows: usize,
    width: usize,
) -> HipResult<Vec<u16>> {
    if hidden.len() != rows * width
        || pre_inverse.len() != rows
        || pre_norm.len() != width
        || fallback.len() != rows * width
        || exception.is_some_and(|exception| exception.column >= width || exception.x.len() != rows)
    {
        return Err(hip_error(
            "completed-layer reconstructible residual geometry mismatch",
        ));
    }
    let mut output = Vec::with_capacity(hidden.len());
    for row in 0..rows {
        let inverse = pre_inverse[row];
        for column in 0..width {
            let index = row * width + column;
            if let Some(exception) = exception {
                if column == exception.column {
                    output.push(exception.x[row]);
                    continue;
                }
            }
            let weight = f32::from_bits((pre_norm[column] as u32) << 16);
            let scale = weight * inverse;
            if scale.is_finite() && scale != 0.0 {
                output.push(f32_to_bf16_bits(hidden[index] / scale));
            } else {
                output.push(fallback[index]);
            }
        }
    }
    Ok(output)
}

fn hip_error(message: impl AsRef<str>) -> HipError {
    HipError::new(0, message.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_chain_skips_only_fully_prepared_intermediate_outputs() {
        assert!(!should_materialize_completed_output(
            false, true, true, false
        ));
        assert!(should_materialize_completed_output(
            false, true, false, false
        ));
        assert!(should_materialize_completed_output(
            false, false, true, false
        ));
        assert!(should_materialize_completed_output(true, true, true, false));
        assert!(should_materialize_completed_output(false, true, true, true));
    }

    #[test]
    fn packed_exception_reconstructs_direct_x() {
        let x = f32_to_bf16_bits(-0.625);
        let exception = NpuEmbeddingPreFfnException {
            column: 0,
            x: vec![x],
        };
        let reconstructed = reconstruct_attention_residual_bf16(
            &[0.0],
            &[1.375],
            Some(&exception),
            &[f32_to_bf16_bits(2.0)],
            &[f32_to_bf16_bits(4.0)],
            1,
            1,
        )
        .expect("packed exception reconstruction");
        assert_eq!(reconstructed, vec![x]);
    }

    #[test]
    fn direct_x_normalization_matches_resident_equation() {
        let normalized = normalize_direct_x_bf16(
            &[f32_to_bf16_bits(2.0), f32_to_bf16_bits(-4.0)],
            &[0.5],
            &[f32_to_bf16_bits(3.0), f32_to_bf16_bits(0.25)],
            1,
            2,
        )
        .expect("direct X normalization");
        assert_eq!(
            normalized,
            vec![f32_to_bf16_bits(3.0), f32_to_bf16_bits(-0.5)]
        );
    }
}
