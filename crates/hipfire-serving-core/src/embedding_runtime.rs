// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Architecture-preserving admission for typed embedding workloads.

use hipfire_model::embedding::{EmbeddingAttentionMode, EmbeddingMetadata, EmbeddingPoolingMode};
use hipfire_model::{ARCH_ID_EMBEDDINGGEMMA, ARCH_ID_QWEN3_QWEN2_LEGACY};
use hipfire_npu::{EmbeddingImageCacheKey, EmbeddingModelGeometry};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingRuntimeKind {
    Qwen3,
    EmbeddingGemma,
}

/// Classify an explicit embedding workload without changing the model's HFQ
/// architecture id. `None` means the artifact is a normal generation model.
pub fn classify_embedding_workload(
    arch_id: u32,
    metadata: Option<&EmbeddingMetadata>,
) -> Result<Option<EmbeddingRuntimeKind>, String> {
    let Some(metadata) = metadata else {
        return Ok(None);
    };
    metadata.validate()?;
    let kind = match arch_id {
        ARCH_ID_QWEN3_QWEN2_LEGACY => EmbeddingRuntimeKind::Qwen3,
        ARCH_ID_EMBEDDINGGEMMA => EmbeddingRuntimeKind::EmbeddingGemma,
        _ => {
            return Err(format!(
                "embedding workload metadata is not supported on arch_id={arch_id}"
            ))
        }
    };
    match kind {
        EmbeddingRuntimeKind::Qwen3 => {
            if metadata.attention != EmbeddingAttentionMode::Causal
                || metadata.pooling.mode != EmbeddingPoolingMode::LastToken
                || !metadata.pooling.normalize
            {
                return Err(
                    "Qwen3 embedding requires causal attention, last-token pooling, and L2 normalization"
                        .into(),
                );
            }
        }
        EmbeddingRuntimeKind::EmbeddingGemma => {
            if !matches!(
                metadata.attention,
                EmbeddingAttentionMode::Bidirectional
                    | EmbeddingAttentionMode::BidirectionalSliding
            ) || metadata.pooling.mode != EmbeddingPoolingMode::Mean
                || !metadata.pooling.normalize
            {
                return Err(
                    "EmbeddingGemma requires bidirectional attention, mean pooling, and L2 normalization"
                        .into(),
                );
            }
        }
    }
    Ok(Some(kind))
}

/// Enforce the first production NPU contract and construct an exact image key.
/// There is no format, architecture, geometry, bucket, or batch fallback.
pub fn npu_embedding_image_key(
    metadata: &EmbeddingMetadata,
    geometry: EmbeddingModelGeometry,
    sequence_bucket: usize,
    dispatch_batch: usize,
) -> Result<EmbeddingImageCacheKey, String> {
    let npu = metadata
        .npu
        .as_ref()
        .filter(|npu| npu.required)
        .ok_or_else(|| "embedding artifact is not marked NPU-only".to_string())?;
    if npu.architecture != "aie2p" {
        return Err(format!(
            "unsupported embedding NPU architecture {:?}; expected aie2p",
            npu.architecture
        ));
    }
    if npu.quant_format != "oq8+" || npu.storage_layout != "opus_oq8_g256_per_row" {
        return Err(format!(
            "unsupported embedding NPU weight contract quant={:?}, layout={:?}; expected oq8+ / opus_oq8_g256_per_row",
            npu.quant_format, npu.storage_layout
        ));
    }
    if !metadata.sequence.buckets.contains(&sequence_bucket) {
        return Err(format!(
            "sequence bucket {sequence_bucket} is not declared by the embedding artifact"
        ));
    }
    if dispatch_batch == 0
        || sequence_bucket
            .checked_mul(dispatch_batch)
            .is_none_or(|rows| rows > metadata.sequence.max_padded_rows_per_dispatch)
    {
        return Err(format!(
            "embedding dispatch {dispatch_batch}x{sequence_bucket} exceeds the {} padded-row limit",
            metadata.sequence.max_padded_rows_per_dispatch
        ));
    }
    Ok(EmbeddingImageCacheKey {
        npu_architecture: npu.architecture.clone(),
        model_geometry: geometry,
        quant_format: npu.quant_format.clone(),
        sequence_bucket,
        dispatch_batch,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_model::embedding::{
        EmbeddingNpuMetadata, EmbeddingOutputMetadata, EmbeddingPoolingMetadata,
        EmbeddingSequenceMetadata, EMBEDDING_METADATA_SCHEMA,
    };
    use std::collections::BTreeMap;

    fn qwen_metadata() -> EmbeddingMetadata {
        EmbeddingMetadata {
            schema: EMBEDDING_METADATA_SCHEMA.into(),
            workload: "embedding".into(),
            attention: EmbeddingAttentionMode::Causal,
            pooling: EmbeddingPoolingMetadata {
                mode: EmbeddingPoolingMode::LastToken,
                normalize: true,
                include_prompt: true,
            },
            prompts: BTreeMap::new(),
            output: EmbeddingOutputMetadata {
                native_dimensions: 1024,
                matryoshka_dimensions: vec![],
            },
            sequence: EmbeddingSequenceMetadata::npu_default(),
            npu: Some(EmbeddingNpuMetadata {
                required: true,
                architecture: "aie2p".into(),
                quant_format: "oq8+".into(),
                storage_layout: "opus_oq8_g256_per_row".into(),
            }),
        }
    }

    fn qwen_geometry() -> EmbeddingModelGeometry {
        EmbeddingModelGeometry {
            architecture: "qwen3".into(),
            hidden_size: 1024,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 3072,
        }
    }

    #[test]
    fn routing_uses_architecture_and_workload_metadata() {
        let metadata = qwen_metadata();
        assert_eq!(
            classify_embedding_workload(ARCH_ID_QWEN3_QWEN2_LEGACY, Some(&metadata)).unwrap(),
            Some(EmbeddingRuntimeKind::Qwen3)
        );
        assert_eq!(
            classify_embedding_workload(ARCH_ID_QWEN3_QWEN2_LEGACY, None).unwrap(),
            None
        );
        assert!(classify_embedding_workload(99, Some(&metadata)).is_err());
    }

    #[test]
    fn image_key_rejects_format_and_padded_row_fallbacks() {
        let metadata = qwen_metadata();
        let key = npu_embedding_image_key(&metadata, qwen_geometry(), 512, 8).unwrap();
        assert_eq!(key.quant_format, "oq8+");
        assert_eq!(key.sequence_bucket, 512);
        assert_eq!(key.dispatch_batch, 8);

        assert!(npu_embedding_image_key(&metadata, qwen_geometry(), 512, 9).is_err());
        let mut wrong_format = metadata;
        wrong_format.npu.as_mut().unwrap().quant_format = "oq8++".into();
        assert!(npu_embedding_image_key(&wrong_format, qwen_geometry(), 512, 8).is_err());
    }
}
