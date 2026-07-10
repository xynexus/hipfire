use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use hip_bridge::{HipError, HipResult};
use hipfire_rdna::{Gpu, GpuTensor};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::quant::f16_to_f32;
use hipfire_runtime::weights::{weight_gemm, WeightTensor};
use hipfire_xdna::{NpuOpusMixedExecutor, OpusMixedPackedMatrix};

use crate::config::EmbeddingGemmaConfig;
use crate::forward::{LinearProjector, Projection};

struct LayerMatrices {
    query: OpusMixedPackedMatrix,
    key: OpusMixedPackedMatrix,
    value: OpusMixedPackedMatrix,
    attention_output: OpusMixedPackedMatrix,
    gate: OpusMixedPackedMatrix,
    up: OpusMixedPackedMatrix,
}

/// Hybrid EmbeddingGemma projector for compact mixed Opus matrices.
/// Q/K/V/O/gate/up execute on XDNA while attention, norms, residuals, and the
/// Q8 fallback down projection remain on the GPU.
pub struct NpuOpusMixedProjector {
    executors: HashMap<usize, NpuOpusMixedExecutor>,
    layers: Vec<LayerMatrices>,
}

impl NpuOpusMixedProjector {
    pub fn load_cached(
        hfq: &HfqFile,
        cfg: &EmbeddingGemmaConfig,
        cache_root: &Path,
    ) -> Result<Self, String> {
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let widths = BTreeSet::from([q_dim, kv_dim, cfg.hidden_size, cfg.intermediate_size]);
        let mut executors = HashMap::with_capacity(widths.len());
        for width in widths {
            if width == 0 || width % 64 != 0 {
                return Err(format!(
                    "embeddinggemma NPU: unsupported output width {width}"
                ));
            }
            let blocks = width / 64;
            let w4 = cache_root.join(format!("embgemma_aie2p_w4_4x4x16_c8_nb{blocks}"));
            let sparse3 = cache_root.join(format!(
                "embgemma_aie2p_sparse3_4x4x16_c8_nb{blocks}_sparse3"
            ));
            let executor = NpuOpusMixedExecutor::load_cached(
                &w4.to_string_lossy(),
                &sparse3.to_string_lossy(),
                width,
            )
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

impl LinearProjector for NpuOpusMixedProjector {
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
        if projection == Projection::Down {
            return weight_gemm(gpu, weight, input, output, rows);
        }
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
            Projection::Down => unreachable!(),
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
            .map_err(|error| hip_error(format!("NPU mixed Opus projection failed: {error}")))?;
        let uploaded = gpu.upload_f32(&output_host, &[rows * width])?;
        gpu.memcpy_dtod_at_auto(&output.buf, 0, &uploaded.buf, 0, output_host.len() * 4)?;
        gpu.free_tensor(uploaded)?;
        Ok(())
    }
}

fn executor(
    executors: &HashMap<usize, NpuOpusMixedExecutor>,
    width: usize,
) -> Result<&NpuOpusMixedExecutor, String> {
    executors
        .get(&width)
        .ok_or_else(|| format!("embeddinggemma NPU: missing N={width} executor"))
}

fn load_matrix(
    hfq: &HfqFile,
    executor: &NpuOpusMixedExecutor,
    name: &str,
) -> Result<OpusMixedPackedMatrix, String> {
    let (info, compact) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("embeddinggemma NPU: missing tensor {name}"))?;
    if info.quant_type != 36 || info.shape.len() != 2 {
        return Err(format!(
            "embeddinggemma NPU: {name} must use the supported compact mixed Opus qt=36 rank-2 layout, got qt={} shape={:?}",
            info.quant_type, info.shape
        ));
    }
    let n = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    executor
        .pack_matrix(k, n, &compact, load_awq_scale(hfq, name, k)?)
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
