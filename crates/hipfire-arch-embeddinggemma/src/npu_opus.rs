use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use hip_bridge::{HipError, HipResult};
use hipfire_rdna::{Gpu, GpuTensor};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::quant::f16_to_f32;
use hipfire_runtime::weights::{weight_gemm, WeightTensor};
use hipfire_xdna::{NpuOpusExecutor, OpusMatrixEncoding, OpusPackedMatrix};

use crate::config::EmbeddingGemmaConfig;
use crate::forward::{LinearProjector, Projection};

struct LayerMatrices {
    query: OpusPackedMatrix,
    key: OpusPackedMatrix,
    value: OpusPackedMatrix,
    attention_output: OpusPackedMatrix,
    gate: OpusPackedMatrix,
    up: OpusPackedMatrix,
    down: Option<OpusPackedMatrix>,
}

/// Format-generic EmbeddingGemma projector for W4, mixed, and W8 Opus matrices.
/// All q/k/v/o/gate/up and Opus-encoded down projections execute on XDNA while
/// the surrounding attention, norms, residuals, and activations remain on the
/// GPU bridge.
pub struct NpuOpusProjector {
    executors: HashMap<usize, NpuOpusExecutor>,
    layers: Vec<LayerMatrices>,
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
        let use_fullk = fullk_paths
            .values()
            .flatten()
            .all(|path| path.join("final.xclbin").is_file() && path.join("insts.bin").is_file());
        let widths = BTreeSet::from([q_dim, kv_dim, cfg.hidden_size, cfg.intermediate_size]);
        let mut executors = HashMap::with_capacity(widths.len());
        for width in widths {
            if width == 0 || width % 64 != 0 {
                return Err(format!(
                    "embeddinggemma NPU: unsupported output width {width}"
                ));
            }
            let executor = if use_fullk {
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

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let prefix = format!("model.layers.{layer_idx}");
            layers.push(LayerMatrices {
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
                down: load_optional_matrix(
                    hfq,
                    executor(&executors, cfg.hidden_size)?,
                    &format!("{prefix}.mlp.down_proj.weight"),
                )?,
            });
        }
        Ok(Self { executors, layers })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn executor_count(&self) -> usize {
        self.executors.len()
    }
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
        let Self { executors, layers } = self;
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
