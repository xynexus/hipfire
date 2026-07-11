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
    NpuOpusExecutor, NpuResidentFfnW4, NpuResidentFfnW4Weights, NpuWholeMode,
    NpuWholeScaledIoLayout, OpusMatrixEncoding, OpusPackedMatrix,
};

use crate::config::EmbeddingGemmaConfig;
use crate::forward::{LinearProjector, Projection};

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
    resident_ffn: Option<ResidentFfnState>,
    resident_ffn_selected: bool,
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

struct ResidentFfnState {
    executor: NpuResidentFfnW4,
    weights: Vec<NpuResidentFfnW4Weights>,
    io: Option<SharedProjectionIo>,
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
        let resident_ffn_path =
            cache_root.join("embgemma_aie2p_resident_ffn_w4_m256_k768_i1152_o768");
        let resident_ffn_rejection = resident_ffn_rejection(&layers);
        if resident_ffn_path.join("final.xclbin").is_file() && resident_ffn_rejection.is_some() {
            eprintln!(
                "embeddinggemma NPU: resident FFN unavailable: {}",
                resident_ffn_rejection.as_deref().unwrap()
            );
        }
        let resident_ffn = if resident_ffn_path.join("final.xclbin").is_file()
            && resident_ffn_path.join("insts.bin").is_file()
            && resident_ffn_rejection.is_none()
        {
            let executor = NpuResidentFfnW4::load_cached(
                resident_ffn_path
                    .to_str()
                    .expect("UTF-8 resident FFN cache path"),
            )
            .map_err(|error| format!("embeddinggemma NPU: load resident FFN: {error}"))?;
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
                .map_err(|error| format!("embeddinggemma NPU: pack resident FFN: {error}"))?;
            Some(ResidentFfnState {
                executor,
                weights,
                io: None,
            })
        } else {
            None
        };
        let resident_ffn_selected = resident_ffn.is_some();
        Ok(Self {
            executors,
            layers,
            shared_io: HashMap::new(),
            awq_gpu: HashMap::new(),
            resident_ffn,
            resident_ffn_selected,
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

fn resident_ffn_rejection(layers: &[LayerMatrices]) -> Option<String> {
    for (index, layer) in layers.iter().enumerate() {
        let Some(down) = layer.down.as_ref() else {
            return Some(format!("layer {index} down projection is not Opus"));
        };
        if layer.gate.encoding() != OpusMatrixEncoding::W4
            || layer.up.encoding() != OpusMatrixEncoding::W4
            || down.encoding() != OpusMatrixEncoding::W4
        {
            return Some(format!(
                "layer {index} encodings are gate={:?} up={:?} down={:?}",
                layer.gate.encoding(),
                layer.up.encoding(),
                down.encoding()
            ));
        }
        if layer.gate.awq_scale() != layer.up.awq_scale() {
            return Some(format!(
                "layer {index} gate/up AWQ activation scales differ"
            ));
        }
    }
    None
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
        if rows <= NpuResidentFfnW4::rows()
            && self.resident_ffn_selected
            && self.resident_ffn.is_some()
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
                if state.io.is_none() {
                    let layout = resident_ffn_layout();
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
                        .executor
                        .attach_shared_io(
                            input_buffer.dmabuf_fd(),
                            layout.input_bytes,
                            output_buffer.dmabuf_fd(),
                            layout.output_bytes,
                        )
                        .map_err(|error| hip_error(format!("attach resident FFN I/O: {error}")))?;
                    state.io = Some(SharedProjectionIo {
                        input_gpu,
                        output_gpu,
                        _input_buffer: input_buffer,
                        _output_buffer: output_buffer,
                    });
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
                let io = state.io.as_ref().expect("resident FFN I/O allocated");
                gpu.pack_opus_npu_activations(
                    input,
                    self.awq_gpu.get(&matrix_key).map(OwnedTensor::view),
                    &io.input_gpu.view(),
                    rows,
                    768,
                    resident_ffn_layout().rdna,
                )?;
                gpu.device_synchronize()?;
                state
                    .executor
                    .run_shared(&state.weights[layer_idx])
                    .map_err(|error| hip_error(format!("resident NPU FFN failed: {error}")))?;
                gpu.unpack_opus_npu_output(
                    &io.output_gpu.view(),
                    output,
                    768,
                    None,
                    None,
                    rows,
                    resident_ffn_layout().rdna,
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

fn resident_ffn_layout() -> ResidentFfnLayout {
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

fn hip_error(message: impl AsRef<str>) -> HipError {
    HipError::new(0, message.as_ref())
}
