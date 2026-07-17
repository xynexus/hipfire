// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Qwen3 embedding-only state and the full-encoder XDNA dispatch boundary.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::PathBuf;

use hipfire_model::embedding::EmbeddingMetadata;
use hipfire_npu::{resolve_embedding_image, EmbeddingImageCacheKey, EmbeddingModelGeometry};
use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use hipfire_runtime::llama::LlamaConfig;

use crate::embedding_batch::{plan_embedding_dispatches, EmbeddingDispatchChunk};
use crate::embedding_runtime::npu_embedding_image_key;

const ENCODER_BLOB_MAGIC: &[u8; 8] = b"HFENCB01";
const ENCODER_BLOB_VERSION: u32 = 1;
const ENCODER_BLOB_HEADER_BYTES: usize = 64;
const ENCODER_BLOB_DESCRIPTOR_BYTES: usize = 48;

#[cfg(target_os = "linux")]
type ResidentExecutor = (
    EmbeddingImageCacheKey,
    hipfire_xdna::NpuFullEmbeddingEncoder,
);

pub struct Qwen3EmbeddingState {
    pub config: LlamaConfig,
    pub metadata: EmbeddingMetadata,
    token_embeddings_bf16: Vec<u16>,
    encoder_weight_blob: Vec<u8>,
    image_cache_root: PathBuf,
    #[cfg(target_os = "linux")]
    executor: RefCell<Option<ResidentExecutor>>,
}

#[derive(Debug)]
pub struct Qwen3EmbeddingTrace {
    pub layer_last_token_residuals: Vec<Vec<f32>>,
    pub stage_layer: usize,
    pub sequence_bucket: usize,
    pub dispatch_batch: usize,
    pub last_layer_stages: BTreeMap<String, Vec<f32>>,
    pub stage_token_major: BTreeMap<String, Vec<f32>>,
}

impl Qwen3EmbeddingState {
    pub fn load(
        hfq: &HfqFile,
        config: LlamaConfig,
        metadata: EmbeddingMetadata,
    ) -> Result<Self, String> {
        if metadata.output.native_dimensions != config.dim {
            return Err(format!(
                "Qwen3 embedding output dimensions {} do not match hidden size {}",
                metadata.output.native_dimensions, config.dim
            ));
        }
        if metadata.sequence != hipfire_model::embedding::EmbeddingSequenceMetadata::npu_default() {
            return Err(
                "Qwen3 embedding requires the fixed 128/256/512/1024/2048 bucket contract and 4096-row ceiling"
                    .into(),
            );
        }
        npu_embedding_image_key(
            &metadata,
            EmbeddingModelGeometry {
                architecture: "qwen3".into(),
                hidden_size: config.dim,
                num_hidden_layers: config.n_layers,
                num_attention_heads: config.n_heads,
                num_key_value_heads: config.n_kv_heads,
                head_dim: config.head_dim,
                intermediate_size: config.hidden_dim,
            },
            128,
            1,
        )?;
        let artifact_quant = serde_json::from_str::<serde_json::Value>(&hfq.metadata_json)
            .ok()
            .and_then(|root| {
                root.get("quant_format")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            })
            .ok_or_else(|| "Qwen3 embedding artifact has no quant_format metadata".to_string())?;
        let npu_quant = &metadata
            .npu
            .as_ref()
            .expect("NPU contract validated")
            .quant_format;
        if &artifact_quant != npu_quant {
            return Err(format!(
                "Qwen3 embedding artifact quant_format={artifact_quant:?} does not match embedding NPU contract {npu_quant:?}"
            ));
        }
        if hfq.find_tensor_info("lm_head.weight").is_some() {
            return Err(
                "Qwen3 embedding artifact contains lm_head.weight; requantize with --npu-embedding"
                    .into(),
            );
        }
        let token_embeddings_bf16 = load_token_embeddings_bf16(hfq, &config)?;
        let encoder_weight_blob = build_qwen3_encoder_weight_blob(hfq, &config)?;
        let image_cache_root = std::env::var_os("HIPFIRE_NPU_IMAGE_CACHE")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .map(|home| home.join(".cache/hipfire/npu"))
            })
            .ok_or_else(|| {
                "Qwen3 embedding requires HIPFIRE_NPU_IMAGE_CACHE or HOME to resolve images"
                    .to_string()
            })?;
        Ok(Self {
            config,
            metadata,
            token_embeddings_bf16,
            encoder_weight_blob,
            image_cache_root,
            #[cfg(target_os = "linux")]
            executor: RefCell::new(None),
        })
    }

    pub fn encode_token_batches(&self, tokenized: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, String> {
        let lengths = tokenized.iter().map(Vec::len).collect::<Vec<_>>();
        let dispatches = plan_embedding_dispatches(&lengths, &self.metadata.sequence)?;
        let mut output = vec![Vec::new(); tokenized.len()];
        for dispatch in dispatches {
            let embeddings = self.run_dispatch(&dispatch, tokenized)?;
            for (embedding, &request_index) in embeddings.into_iter().zip(&dispatch.request_indices)
            {
                output[request_index] = embedding;
            }
        }
        Ok(output)
    }

    /// Diagnostic full-encoder run that also returns the last-real-token
    /// residual stream after each transformer layer. All documents must fit in
    /// one dispatch; this keeps the layer ordering unambiguous for oracle
    /// comparison tools.
    #[cfg(target_os = "linux")]
    pub fn encode_token_batches_with_layer_trace(
        &self,
        tokenized: &[Vec<u32>],
    ) -> Result<(Vec<Vec<f32>>, Qwen3EmbeddingTrace), String> {
        let lengths = tokenized.iter().map(Vec::len).collect::<Vec<_>>();
        let dispatches = plan_embedding_dispatches(&lengths, &self.metadata.sequence)?;
        if dispatches.len() != 1 {
            return Err(format!(
                "Qwen3 layer trace requires one NPU dispatch; planner produced {}",
                dispatches.len()
            ));
        }
        let dispatch = &dispatches[0];
        let (mut embeddings, trace) = self.run_dispatch_inner(dispatch, tokenized, true)?;
        embeddings.truncate(dispatch.request_indices.len());
        Ok((
            embeddings,
            trace.expect("layer trace requested from Qwen3 encoder"),
        ))
    }

    #[cfg(target_os = "linux")]
    fn run_dispatch(
        &self,
        dispatch: &EmbeddingDispatchChunk,
        tokenized: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>, String> {
        self.run_dispatch_inner(dispatch, tokenized, false)
            .map(|(embeddings, _)| embeddings)
    }

    #[cfg(target_os = "linux")]
    fn run_dispatch_inner(
        &self,
        dispatch: &EmbeddingDispatchChunk,
        tokenized: &[Vec<u32>],
        capture_layer_trace: bool,
    ) -> Result<(Vec<Vec<f32>>, Option<Qwen3EmbeddingTrace>), String> {
        let key = npu_embedding_image_key(
            &self.metadata,
            EmbeddingModelGeometry {
                architecture: "qwen3".into(),
                hidden_size: self.config.dim,
                num_hidden_layers: self.config.n_layers,
                num_attention_heads: self.config.n_heads,
                num_key_value_heads: self.config.n_kv_heads,
                head_dim: self.config.head_dim,
                intermediate_size: self.config.hidden_dim,
            },
            dispatch.bucket,
            dispatch.dispatch_batch,
        )?;
        let mut executor = self.executor.borrow_mut();
        if executor
            .as_ref()
            .is_none_or(|(resident_key, _)| resident_key != &key)
        {
            // A full encoder owns thirteen XDNA hardware contexts. Drop the
            // previous geometry before loading its replacement; constructing
            // the new bundle first transiently doubles that footprint and can
            // exhaust AIE resources when a request crosses sequence buckets.
            *executor = None;
            let image = resolve_embedding_image(&self.image_cache_root, &key)?;
            let loaded = hipfire_xdna::NpuFullEmbeddingEncoder::load_cached(
                &image.directory,
                hipfire_xdna::FullEmbeddingIoGeometry {
                    sequence_bucket: dispatch.bucket,
                    dispatch_batch: dispatch.dispatch_batch,
                    hidden_size: self.config.dim,
                    output_dimensions: self.metadata.output.native_dimensions,
                },
                &self.encoder_weight_blob,
            )
            .map_err(|error| format!("load Qwen3 embedding NPU image: {error}"))?;
            *executor = Some((key, loaded));
        }
        let mut padded = vec![0u16; dispatch.padded_rows * self.config.dim];
        for (document, &request_index) in dispatch.request_indices.iter().enumerate() {
            for (position, &token) in tokenized[request_index].iter().enumerate() {
                let token = token as usize;
                if token >= self.config.vocab_size {
                    return Err(format!(
                        "embedding token id {token} exceeds vocabulary size {}",
                        self.config.vocab_size
                    ));
                }
                let source = &self.token_embeddings_bf16
                    [token * self.config.dim..(token + 1) * self.config.dim];
                let row = document * dispatch.bucket + position;
                padded[row * self.config.dim..(row + 1) * self.config.dim].copy_from_slice(source);
            }
        }
        let lengths = dispatch
            .token_lengths
            .iter()
            .map(|&length| length as u32)
            .chain(std::iter::repeat_n(
                1,
                dispatch.dispatch_batch - dispatch.token_lengths.len(),
            ))
            .collect::<Vec<_>>();
        let encoder = &mut executor.as_mut().expect("executor loaded").1;
        if capture_layer_trace {
            let (embeddings, trace) = encoder
                .run_with_layer_trace(&padded, &lengths)
                .map_err(|error| format!("Qwen3 embedding NPU dispatch: {error}"))?;
            let layer_last_token_residuals = select_last_real_token_layers(
                &trace.completed_layers_bf16,
                &dispatch.token_lengths,
                dispatch.bucket,
                dispatch.dispatch_batch,
                self.config.dim,
            )?;
            let stage_layer = trace.stage_layer;
            let mut last_layer_stages = BTreeMap::new();
            let mut stage_token_major = BTreeMap::new();
            for stage in trace.last_layer_stages {
                stage_token_major.insert(
                    stage.name.to_string(),
                    stage
                        .token_major_bf16
                        .iter()
                        .map(|&bits| f32::from_bits((bits as u32) << 16))
                        .collect(),
                );
                last_layer_stages.insert(
                    stage.name.to_string(),
                    select_last_real_tokens(
                        &stage.token_major_bf16,
                        &dispatch.token_lengths,
                        dispatch.bucket,
                        dispatch.dispatch_batch,
                        stage.columns,
                        stage.name,
                    )?,
                );
            }
            Ok((
                embeddings,
                Some(Qwen3EmbeddingTrace {
                    layer_last_token_residuals,
                    stage_layer,
                    sequence_bucket: dispatch.bucket,
                    dispatch_batch: dispatch.dispatch_batch,
                    last_layer_stages,
                    stage_token_major,
                }),
            ))
        } else {
            encoder
                .run(&padded, &lengths)
                .map(|embeddings| (embeddings, None))
                .map_err(|error| format!("Qwen3 embedding NPU dispatch: {error}"))
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn run_dispatch(
        &self,
        _dispatch: &EmbeddingDispatchChunk,
        _tokenized: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>, String> {
        Err("Qwen3 NPU embedding requires the Linux XDNA backend".into())
    }
}

fn select_last_real_token_layers(
    layers: &[Vec<u16>],
    lengths: &[usize],
    sequence_bucket: usize,
    dispatch_batch: usize,
    hidden_size: usize,
) -> Result<Vec<Vec<f32>>, String> {
    layers
        .iter()
        .enumerate()
        .map(|(layer, values)| {
            select_last_real_tokens(
                values,
                lengths,
                sequence_bucket,
                dispatch_batch,
                hidden_size,
                &format!("layer {layer}"),
            )
        })
        .collect()
}

fn select_last_real_tokens(
    values: &[u16],
    lengths: &[usize],
    sequence_bucket: usize,
    dispatch_batch: usize,
    columns: usize,
    label: &str,
) -> Result<Vec<f32>, String> {
    let expected = dispatch_batch
        .checked_mul(sequence_bucket)
        .and_then(|rows| rows.checked_mul(columns))
        .ok_or_else(|| "Qwen3 layer trace geometry overflow".to_string())?;
    if values.len() != expected {
        return Err(format!(
            "Qwen3 layer trace {label} has {} values; expected {expected}",
            values.len()
        ));
    }
    let mut selected = Vec::with_capacity(lengths.len() * columns);
    for (document, &length) in lengths.iter().enumerate() {
        if length == 0 || length > sequence_bucket {
            return Err(format!(
                "Qwen3 layer trace length[{document}]={length} is outside 1..={sequence_bucket}"
            ));
        }
        let row = document * sequence_bucket + length - 1;
        selected.extend(
            values[row * columns..(row + 1) * columns]
                .iter()
                .map(|&bits| f32::from_bits((bits as u32) << 16)),
        );
    }
    Ok(selected)
}

#[derive(Clone, Copy)]
struct BlobEntry<'a> {
    role: u16,
    layer: u32,
    info: &'a HfqTensorInfo,
    data: &'a [u8],
}

fn build_qwen3_encoder_weight_blob(hfq: &HfqFile, config: &LlamaConfig) -> Result<Vec<u8>, String> {
    let mut entries = Vec::with_capacity(1 + config.n_layers * 16);
    push_entry(hfq, &mut entries, 1, u32::MAX, "model.norm.weight", false)?;
    for layer in 0..config.n_layers {
        let prefix = format!("model.layers.{layer}");
        for (role, suffix, matrix) in [
            (2, "input_layernorm.weight", false),
            (3, "self_attn.q_proj.weight", true),
            (4, "self_attn.k_proj.weight", true),
            (5, "self_attn.v_proj.weight", true),
            (6, "self_attn.o_proj.weight", true),
            (7, "self_attn.q_norm.weight", false),
            (8, "self_attn.k_norm.weight", false),
            (9, "post_attention_layernorm.weight", false),
            (10, "mlp.gate_proj.weight", true),
            (11, "mlp.up_proj.weight", true),
            (12, "mlp.down_proj.weight", true),
        ] {
            let name = format!("{prefix}.{suffix}");
            push_entry(hfq, &mut entries, role, layer as u32, &name, matrix)?;
            if matrix {
                let sidecar = name
                    .strip_suffix(".weight")
                    .map(|stem| format!("{stem}.awq_scale.weight"))
                    .expect("matrix name ends in .weight");
                if hfq.find_tensor_info(&sidecar).is_some() {
                    push_entry(
                        hfq,
                        &mut entries,
                        role | 0x8000,
                        layer as u32,
                        &sidecar,
                        false,
                    )?;
                }
            }
        }
    }
    encode_blob(config, &entries)
}

fn push_entry<'a>(
    hfq: &'a HfqFile,
    entries: &mut Vec<BlobEntry<'a>>,
    role: u16,
    layer: u32,
    name: &str,
    matrix: bool,
) -> Result<(), String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("Qwen3 embedding encoder tensor {name} is missing"))?;
    if matrix && !matches!(info.quant_type, 35 | 43) {
        return Err(format!(
            "Qwen3 embedding encoder matrix {name} has quant_type={}; expected OQ8 qt=35/43",
            info.quant_type
        ));
    }
    if !matrix && !matches!(info.quant_type, 1 | 2 | 16) {
        return Err(format!(
            "Qwen3 embedding encoder vector {name} has quant_type={}; expected F16/F32/BF16",
            info.quant_type
        ));
    }
    entries.push(BlobEntry {
        role,
        layer,
        info,
        data,
    });
    Ok(())
}

fn encode_blob(config: &LlamaConfig, entries: &[BlobEntry<'_>]) -> Result<Vec<u8>, String> {
    let descriptors_end = ENCODER_BLOB_HEADER_BYTES
        .checked_add(entries.len() * ENCODER_BLOB_DESCRIPTOR_BYTES)
        .ok_or_else(|| "Qwen3 embedding encoder descriptor size overflow".to_string())?;
    let payload_start = descriptors_end.div_ceil(64) * 64;
    let payload_bytes = entries.iter().try_fold(0usize, |total, entry| {
        total
            .checked_add(entry.data.len().div_ceil(64) * 64)
            .ok_or_else(|| "Qwen3 embedding encoder payload size overflow".to_string())
    })?;
    let mut blob = vec![0u8; payload_start + payload_bytes];
    blob[..8].copy_from_slice(ENCODER_BLOB_MAGIC);
    put_u32(&mut blob, 8, ENCODER_BLOB_VERSION);
    put_u32(&mut blob, 12, entries.len() as u32);
    for (offset, value) in [
        (16, config.dim),
        (20, config.n_layers),
        (24, config.n_heads),
        (28, config.n_kv_heads),
        (32, config.head_dim),
        (36, config.hidden_dim),
    ] {
        put_u32(&mut blob, offset, value as u32);
    }
    put_f32(&mut blob, 40, config.norm_eps);
    put_f32(&mut blob, 44, config.rope_freq_base);
    let mut payload_offset = payload_start;
    for (index, entry) in entries.iter().enumerate() {
        let descriptor = ENCODER_BLOB_HEADER_BYTES + index * ENCODER_BLOB_DESCRIPTOR_BYTES;
        put_u16(&mut blob, descriptor, entry.role);
        put_u32(&mut blob, descriptor + 4, entry.layer);
        put_u32(&mut blob, descriptor + 8, entry.info.quant_type as u32);
        put_u32(&mut blob, descriptor + 12, entry.info.shape.len() as u32);
        put_u32(
            &mut blob,
            descriptor + 16,
            entry.info.shape.first().copied().unwrap_or(1),
        );
        put_u32(
            &mut blob,
            descriptor + 20,
            entry.info.shape.get(1).copied().unwrap_or(1),
        );
        put_u32(&mut blob, descriptor + 24, entry.info.group_size);
        put_u64(&mut blob, descriptor + 32, payload_offset as u64);
        put_u64(&mut blob, descriptor + 40, entry.data.len() as u64);
        blob[payload_offset..payload_offset + entry.data.len()].copy_from_slice(entry.data);
        payload_offset += entry.data.len().div_ceil(64) * 64;
    }
    Ok(blob)
}

fn load_token_embeddings_bf16(hfq: &HfqFile, config: &LlamaConfig) -> Result<Vec<u16>, String> {
    let (info, data) = hfq
        .tensor_data("model.embed_tokens.weight")
        .ok_or_else(|| "Qwen3 embedding artifact has no model.embed_tokens.weight".to_string())?;
    if info.shape != [config.vocab_size as u32, config.dim as u32] {
        return Err(format!(
            "Qwen3 embedding table shape {:?} does not match [{}, {}]",
            info.shape, config.vocab_size, config.dim
        ));
    }
    if hfq
        .find_tensor_info("model.embed_tokens.awq_scale.weight")
        .is_some()
    {
        return Err("Qwen3 embedding table must not carry an AWQ activation sidecar".into());
    }
    let mut result = Vec::with_capacity(config.vocab_size * config.dim);
    match info.quant_type {
        16 => result.extend(
            data.chunks_exact(2)
                .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])),
        ),
        1 => result.extend(data.chunks_exact(2).map(|bytes| {
            f32_to_bf16(hipfire_runtime::quant::f16_to_f32(u16::from_le_bytes([
                bytes[0], bytes[1],
            ])))
        })),
        3 => result.extend(
            hipfire_runtime::quant::dequant_q8f16(data, config.vocab_size * config.dim)
                .into_iter()
                .map(f32_to_bf16),
        ),
        35 | 43 => {
            let row_bytes = config.dim.div_ceil(256) * 258;
            if data.len() != config.vocab_size * row_bytes {
                return Err(format!(
                    "Qwen3 embedding OQ8 table has {} bytes; expected {}",
                    data.len(),
                    config.vocab_size * row_bytes
                ));
            }
            for row in data.chunks_exact(row_bytes) {
                result.extend(
                    hipfire_runtime::quant::dequant_oq8g256(row, config.dim)
                        .into_iter()
                        .map(f32_to_bf16),
                );
            }
        }
        quant_type => {
            return Err(format!(
            "Qwen3 embedding table quant_type={quant_type} is unsupported; expected BF16/F16/Q8F16/OQ8"
        ))
        }
    }
    Ok(result)
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn put_f32(bytes: &mut [u8], offset: usize, value: f32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
    use serde_json::json;

    #[test]
    fn bf16_rounding_is_ties_to_even() {
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
    }

    #[test]
    fn layer_trace_selects_each_documents_last_real_token() {
        let layers = vec![
            (0..24).map(|value| f32_to_bf16(value as f32)).collect(),
            (24..48).map(|value| f32_to_bf16(value as f32)).collect(),
        ];
        let selected = select_last_real_token_layers(&layers, &[2, 3], 4, 2, 3).unwrap();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0], vec![3.0, 4.0, 5.0, 18.0, 19.0, 20.0]);
        assert_eq!(selected[1], vec![27.0, 28.0, 29.0, 42.0, 43.0, 44.0]);
    }

    fn bf16(name: impl Into<String>, shape: Vec<u32>) -> HfqMemTensor {
        let elements = shape.iter().map(|&value| value as usize).product::<usize>();
        HfqMemTensor {
            name: name.into(),
            quant_type: 16,
            shape,
            group_size: 0,
            data: vec![0; elements * 2],
        }
    }

    fn oq8(name: impl Into<String>, rows: usize, cols: usize) -> HfqMemTensor {
        HfqMemTensor {
            name: name.into(),
            quant_type: 35,
            shape: vec![rows as u32, cols as u32],
            group_size: 256,
            data: vec![0; rows * cols.div_ceil(256) * 258],
        }
    }

    #[test]
    fn embedding_state_loads_encoder_tensors_without_lm_head_or_kv() {
        let root = std::env::temp_dir().join(format!(
            "hipfire-qwen3-embedding-state-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("Qwen3-Embedding-0.6B.npu.oq8+.hfq");
        let mut tensors = vec![
            oq8("model.embed_tokens.weight", 2, 256),
            bf16("model.norm.weight", vec![256]),
        ];
        let prefix = "model.layers.0";
        tensors.extend([
            bf16(format!("{prefix}.input_layernorm.weight"), vec![256]),
            oq8(format!("{prefix}.self_attn.q_proj.weight"), 256, 256),
            oq8(format!("{prefix}.self_attn.k_proj.weight"), 128, 256),
            oq8(format!("{prefix}.self_attn.v_proj.weight"), 128, 256),
            oq8(format!("{prefix}.self_attn.o_proj.weight"), 256, 256),
            bf16(format!("{prefix}.self_attn.q_norm.weight"), vec![128]),
            bf16(format!("{prefix}.self_attn.k_norm.weight"), vec![128]),
            bf16(
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![256],
            ),
            oq8(format!("{prefix}.mlp.gate_proj.weight"), 256, 256),
            oq8(format!("{prefix}.mlp.up_proj.weight"), 256, 256),
            oq8(format!("{prefix}.mlp.down_proj.weight"), 256, 256),
        ]);
        let metadata = json!({
            "quant_format": "oq8+",
            "config": {
                "model_type": "qwen3",
                "hidden_size": 256,
                "intermediate_size": 256,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 128,
                "vocab_size": 2,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 32768,
                "rope_theta": 1000000.0
            },
            "embedding": {
                "schema": "hipfire.embedding.v1",
                "workload": "embedding",
                "attention": "causal",
                "pooling": {"mode":"last_token", "normalize":true, "include_prompt":true},
                "prompts": {"query":"Instruct: test\nQuery:", "document":""},
                "output": {"native_dimensions":256, "matryoshka_dimensions":[256,128,32]},
                "sequence": {
                    "max_tokens":2048,
                    "buckets":[128,256,512,1024,2048],
                    "max_padded_rows_per_dispatch":4096
                },
                "npu": {
                    "required":true,
                    "architecture":"aie2p",
                    "quant_format":"oq8+",
                    "storage_layout":"opus_oq8_g256_per_row"
                }
            }
        });
        write_hfqm_package_mem(&path, 1, &metadata.to_string(), &tensors).unwrap();
        let hfq = HfqFile::open(&path).unwrap();
        let config = hipfire_runtime::hfq::config_from_hfq(&hfq).unwrap();
        let metadata = EmbeddingMetadata::from_hfq_metadata_json(&hfq.metadata_json)
            .unwrap()
            .unwrap();
        let state = Qwen3EmbeddingState::load(&hfq, config, metadata).unwrap();

        assert_eq!(&state.encoder_weight_blob[..8], ENCODER_BLOB_MAGIC);
        assert_eq!(
            u32::from_le_bytes(state.encoder_weight_blob[12..16].try_into().unwrap()),
            12
        );
        assert_eq!(state.token_embeddings_bf16.len(), 2 * 256);
        assert!(hfq.find_tensor_info("lm_head.weight").is_none());

        drop(state);
        drop(hfq);
        std::fs::remove_dir_all(root).unwrap();
    }
}
