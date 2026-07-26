// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Typed HFQ metadata for non-autoregressive embedding workloads.
//!
//! The HFQ header keeps the underlying architecture id (Qwen3 is 1,
//! EmbeddingGemma is 19). This block describes how that architecture is used:
//! prompt roles, pooling, output geometry, and the compiled NPU contract.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const EMBEDDING_METADATA_SCHEMA: &str = "hipfire.embedding.v1";
pub const NPU_EMBEDDING_SEQUENCE_BUCKETS: [usize; 5] = [128, 256, 512, 1024, 2048];
pub const NPU_EMBEDDING_MAX_PADDED_ROWS: usize = 4096;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingInputType {
    Query,
    #[default]
    Document,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingAttentionMode {
    Causal,
    Bidirectional,
    BidirectionalSliding,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingPoolingMode {
    Mean,
    LastToken,
    Cls,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EmbeddingPoolingMetadata {
    pub mode: EmbeddingPoolingMode,
    pub normalize: bool,
    pub include_prompt: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EmbeddingOutputMetadata {
    pub native_dimensions: usize,
    #[serde(default)]
    pub matryoshka_dimensions: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EmbeddingSequenceMetadata {
    pub max_tokens: usize,
    pub buckets: Vec<usize>,
    pub max_padded_rows_per_dispatch: usize,
}

impl EmbeddingSequenceMetadata {
    pub fn npu_default() -> Self {
        Self {
            max_tokens: 2048,
            buckets: NPU_EMBEDDING_SEQUENCE_BUCKETS.to_vec(),
            max_padded_rows_per_dispatch: NPU_EMBEDDING_MAX_PADDED_ROWS,
        }
    }

    pub fn bucket_for_len(&self, tokens: usize) -> Result<usize, String> {
        if tokens == 0 {
            return Err("embedding input token sequence must be non-empty".to_string());
        }
        if tokens > self.max_tokens {
            return Err(format!(
                "embedding input has {tokens} tokens; maximum supported length is {}",
                self.max_tokens
            ));
        }
        self.buckets
            .iter()
            .copied()
            .find(|bucket| *bucket >= tokens)
            .ok_or_else(|| {
                format!(
                    "embedding input has {tokens} tokens but no compiled sequence bucket covers it"
                )
            })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EmbeddingNpuMetadata {
    pub required: bool,
    /// NPU tile ISA, for example `aie2p`.
    pub architecture: String,
    /// Canonical quant token used to compile/cache the image, for example `oq8+`.
    pub quant_format: String,
    /// Weight/image storage contract, for example `opus_oq8_g256_per_row`.
    pub storage_layout: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EmbeddingMetadata {
    pub schema: String,
    pub workload: String,
    pub attention: EmbeddingAttentionMode,
    pub pooling: EmbeddingPoolingMetadata,
    #[serde(default)]
    pub prompts: BTreeMap<String, String>,
    pub output: EmbeddingOutputMetadata,
    pub sequence: EmbeddingSequenceMetadata,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub npu: Option<EmbeddingNpuMetadata>,
}

impl EmbeddingMetadata {
    pub fn prompt(&self, input_type: EmbeddingInputType) -> &str {
        let key = match input_type {
            EmbeddingInputType::Query => "query",
            EmbeddingInputType::Document => "document",
        };
        self.prompts.get(key).map(String::as_str).unwrap_or("")
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != EMBEDDING_METADATA_SCHEMA {
            return Err(format!(
                "unsupported embedding metadata schema {:?}",
                self.schema
            ));
        }
        if self.workload != "embedding" {
            return Err(format!(
                "unsupported embedding workload {:?}",
                self.workload
            ));
        }
        if self.output.native_dimensions == 0 {
            return Err("embedding output dimensions must be positive".to_string());
        }
        if self.sequence.max_tokens == 0
            || self.sequence.max_padded_rows_per_dispatch == 0
            || self.sequence.buckets.is_empty()
        {
            return Err("embedding sequence geometry must be non-empty".to_string());
        }
        if !self
            .sequence
            .buckets
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        {
            return Err("embedding sequence buckets must be strictly increasing".to_string());
        }
        if self.sequence.buckets.last().copied() != Some(self.sequence.max_tokens) {
            return Err("embedding sequence buckets must end at max_tokens".to_string());
        }
        if let Some(npu) = &self.npu {
            if npu.architecture.trim().is_empty()
                || npu.quant_format.trim().is_empty()
                || npu.storage_layout.trim().is_empty()
            {
                return Err(
                    "embedding NPU architecture, quant format, and storage layout must be non-empty"
                        .into(),
                );
            }
        }
        Ok(())
    }

    pub fn from_hfq_metadata_json(metadata_json: &str) -> Result<Option<Self>, String> {
        let root: Value = serde_json::from_str(metadata_json)
            .map_err(|error| format!("parse HFQ metadata: {error}"))?;
        let Some(value) = root.get("embedding") else {
            return Ok(None);
        };
        let metadata: Self = serde_json::from_value(value.clone())
            .map_err(|error| format!("parse embedding metadata: {error}"))?;
        metadata.validate()?;
        Ok(Some(metadata))
    }

    /// Convert SentenceTransformers module configs into the canonical HFQ block.
    /// The caller reads the sidecars and keys `module_configs` by each module's
    /// `path` from `modules.json`.
    pub fn from_sentence_transformers(
        model_config: &Value,
        modules: &Value,
        module_configs: &BTreeMap<String, Value>,
        sentence_config: Option<&Value>,
        npu: Option<EmbeddingNpuMetadata>,
    ) -> Result<Self, String> {
        let modules = modules
            .as_array()
            .ok_or_else(|| "SentenceTransformers modules.json must be an array".to_string())?;

        let mut pooling = None;
        let mut normalize = false;
        let mut final_dimensions = None;
        for module in modules {
            let ty = module
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| "SentenceTransformers module is missing type".to_string())?;
            let path = module.get("path").and_then(Value::as_str).unwrap_or("");
            if ty.ends_with(".Pooling") {
                let config = module_configs.get(path).ok_or_else(|| {
                    format!("SentenceTransformers pooling module {path:?} has no config.json")
                })?;
                pooling = Some(parse_pooling(config)?);
                final_dimensions = config
                    .get("word_embedding_dimension")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize);
            } else if ty.ends_with(".Dense") {
                let config = module_configs.get(path).ok_or_else(|| {
                    format!("SentenceTransformers Dense module {path:?} has no config.json")
                })?;
                final_dimensions = Some(
                    config
                        .get("out_features")
                        .and_then(Value::as_u64)
                        .ok_or_else(|| format!("Dense module {path:?} is missing out_features"))?
                        as usize,
                );
            } else if ty.ends_with(".Normalize") {
                normalize = true;
            } else if !ty.ends_with(".Transformer") {
                return Err(format!(
                    "unsupported SentenceTransformers module type {ty:?}"
                ));
            }
        }

        let mut pooling = pooling
            .ok_or_else(|| "SentenceTransformers modules.json has no pooling module".to_string())?;
        pooling.normalize = normalize;

        let hidden = model_config
            .get("hidden_size")
            .or_else(|| {
                model_config
                    .get("text_config")
                    .and_then(|value| value.get("hidden_size"))
            })
            .and_then(Value::as_u64)
            .ok_or_else(|| "model config is missing hidden_size".to_string())?
            as usize;
        let native_dimensions = final_dimensions.unwrap_or(hidden);

        let bidirectional = model_config
            .get("use_bidirectional_attention")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let sliding = model_config
            .get("sliding_window")
            .and_then(Value::as_u64)
            .is_some_and(|window| window > 0);
        let attention = match (bidirectional, sliding) {
            (true, true) => EmbeddingAttentionMode::BidirectionalSliding,
            (true, false) => EmbeddingAttentionMode::Bidirectional,
            (false, _) => EmbeddingAttentionMode::Causal,
        };

        let prompts = sentence_config
            .and_then(|value| value.get("prompts"))
            .and_then(Value::as_object)
            .map(|prompts| {
                prompts
                    .iter()
                    .filter_map(|(name, value)| {
                        value
                            .as_str()
                            .map(|prompt| (name.clone(), prompt.to_string()))
                    })
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();

        let mut matryoshka_dimensions = model_config
            .get("sentence_transformers")
            .and_then(|value| value.get("matryoshka_dims"))
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_u64)
                    .map(|value| value as usize)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| {
                if model_config.get("model_type").and_then(Value::as_str) == Some("qwen3") {
                    (32..=native_dimensions).collect()
                } else {
                    vec![native_dimensions]
                }
            });
        matryoshka_dimensions.sort_unstable_by(|a, b| b.cmp(a));
        matryoshka_dimensions.dedup();

        let metadata = Self {
            schema: EMBEDDING_METADATA_SCHEMA.to_string(),
            workload: "embedding".to_string(),
            attention,
            pooling,
            prompts,
            output: EmbeddingOutputMetadata {
                native_dimensions,
                matryoshka_dimensions,
            },
            sequence: EmbeddingSequenceMetadata::npu_default(),
            npu,
        };
        metadata.validate()?;
        Ok(metadata)
    }
}

fn parse_pooling(config: &Value) -> Result<EmbeddingPoolingMetadata, String> {
    let modes = [
        ("pooling_mode_mean_tokens", EmbeddingPoolingMode::Mean),
        ("pooling_mode_lasttoken", EmbeddingPoolingMode::LastToken),
        ("pooling_mode_cls_token", EmbeddingPoolingMode::Cls),
    ];
    let enabled = modes
        .iter()
        .filter_map(|(key, mode)| {
            config
                .get(*key)
                .and_then(Value::as_bool)
                .unwrap_or(false)
                .then_some(*mode)
        })
        .collect::<Vec<_>>();
    if enabled.len() != 1 {
        return Err(format!(
            "SentenceTransformers pooling must enable exactly one of mean, lasttoken, or cls; got {}",
            enabled.len()
        ));
    }
    for unsupported in [
        "pooling_mode_max_tokens",
        "pooling_mode_mean_sqrt_len_tokens",
        "pooling_mode_weightedmean_tokens",
    ] {
        if config
            .get(unsupported)
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(format!(
                "unsupported SentenceTransformers pooling mode {unsupported}"
            ));
        }
    }
    Ok(EmbeddingPoolingMetadata {
        mode: enabled[0],
        normalize: false,
        include_prompt: config
            .get("include_prompt")
            .and_then(Value::as_bool)
            .unwrap_or(true),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn qwen_modules() -> (Value, BTreeMap<String, Value>) {
        let modules = json!([
            {"path":"", "type":"sentence_transformers.models.Transformer"},
            {"path":"1_Pooling", "type":"sentence_transformers.models.Pooling"},
            {"path":"2_Normalize", "type":"sentence_transformers.models.Normalize"}
        ]);
        let configs = BTreeMap::from([(
            "1_Pooling".to_string(),
            json!({
                "word_embedding_dimension": 1024,
                "pooling_mode_cls_token": false,
                "pooling_mode_mean_tokens": false,
                "pooling_mode_max_tokens": false,
                "pooling_mode_mean_sqrt_len_tokens": false,
                "pooling_mode_weightedmean_tokens": false,
                "pooling_mode_lasttoken": true,
                "include_prompt": true
            }),
        )]);
        (modules, configs)
    }

    #[test]
    fn qwen_sentence_transformers_metadata_preserves_prompts_and_geometry() {
        let (modules, configs) = qwen_modules();
        let metadata = EmbeddingMetadata::from_sentence_transformers(
            &json!({"model_type":"qwen3", "hidden_size":1024}),
            &modules,
            &configs,
            Some(&json!({"prompts":{"query":"Instruct: q\nQuery:", "document":""}})),
            Some(EmbeddingNpuMetadata {
                required: true,
                architecture: "aie2p".into(),
                quant_format: "oq8+".into(),
                storage_layout: "opus_oq8_g256_per_row".into(),
            }),
        )
        .unwrap();
        assert_eq!(metadata.attention, EmbeddingAttentionMode::Causal);
        assert_eq!(metadata.pooling.mode, EmbeddingPoolingMode::LastToken);
        assert!(metadata.pooling.normalize);
        assert_eq!(
            metadata.prompt(EmbeddingInputType::Query),
            "Instruct: q\nQuery:"
        );
        assert_eq!(metadata.output.native_dimensions, 1024);
        assert!(metadata.output.matryoshka_dimensions.contains(&32));
        assert!(metadata.output.matryoshka_dimensions.contains(&256));
        assert!(metadata.output.matryoshka_dimensions.contains(&1024));
        assert_eq!(metadata.sequence.bucket_for_len(257).unwrap(), 512);
        assert!(metadata
            .sequence
            .bucket_for_len(2049)
            .unwrap_err()
            .contains("maximum"));
        assert!(metadata.npu.as_ref().unwrap().required);
    }

    #[test]
    fn invalid_or_ambiguous_pooling_is_rejected() {
        let (modules, mut configs) = qwen_modules();
        configs.get_mut("1_Pooling").unwrap()["pooling_mode_mean_tokens"] = json!(true);
        let error = EmbeddingMetadata::from_sentence_transformers(
            &json!({"hidden_size":1024}),
            &modules,
            &configs,
            None,
            None,
        )
        .unwrap_err();
        assert!(error.contains("exactly one"));
    }
}
