// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Small shared HFQ loader for transformer tensor mechanics.
//!
//! Architecture crates retain tensor-name construction, required/optional
//! policy, prefixes, paging, and layer assembly. This module owns the mechanical
//! part: exact logical-shape checks, source-precision widening, raw/quant GPU
//! upload, embeddings, direct norms, and tied/untied language-model heads.

use crate::hfq::{
    load_awq_scale, oq4_arch_load, oq8_arch_load, HfqFile, HfqTensorInfo, OQ4_ARCH_PACKED_QT,
    OQ4_CANONICAL_QT,
};
use crate::quant::f16_to_f32;
use crate::weights::{EmbeddingFormat, WeightTensor};
use hip_bridge::HipResult;
use hipfire_rdna::{DType, Gpu, GpuTensor};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorLookupError {
    Missing {
        name: String,
    },
    WrongRank {
        name: String,
        actual: usize,
        expected: usize,
    },
    WrongShape {
        name: String,
        actual: Vec<u32>,
        expected: Vec<u32>,
    },
    UnsupportedSourceType {
        name: String,
        quant_type: u8,
    },
    InvalidByteLength {
        name: String,
        actual: usize,
        expected: usize,
    },
}

impl fmt::Display for TensorLookupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { name } => write!(f, "required tensor `{name}` is missing"),
            Self::WrongRank {
                name,
                actual,
                expected,
            } => write!(
                f,
                "tensor `{name}` has rank {actual}, expected rank {expected}"
            ),
            Self::WrongShape {
                name,
                actual,
                expected,
            } => write!(
                f,
                "tensor `{name}` has shape {actual:?}, expected {expected:?}"
            ),
            Self::UnsupportedSourceType { name, quant_type } => write!(
                f,
                "tensor `{name}` has quant type {quant_type}; expected F16/F32/BF16"
            ),
            Self::InvalidByteLength {
                name,
                actual,
                expected,
            } => write!(
                f,
                "tensor `{name}` has {actual} data bytes, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for TensorLookupError {}

pub fn validate_required_info<'a>(
    info: Option<&'a HfqTensorInfo>,
    name: &str,
    expected_shape: &[usize],
) -> Result<&'a HfqTensorInfo, TensorLookupError> {
    let info = info.ok_or_else(|| TensorLookupError::Missing {
        name: name.to_string(),
    })?;
    if info.shape.len() != expected_shape.len() {
        return Err(TensorLookupError::WrongRank {
            name: name.to_string(),
            actual: info.shape.len(),
            expected: expected_shape.len(),
        });
    }
    let expected: Vec<u32> = expected_shape.iter().map(|&dim| dim as u32).collect();
    if info.shape != expected {
        return Err(TensorLookupError::WrongShape {
            name: name.to_string(),
            actual: info.shape.clone(),
            expected,
        });
    }
    Ok(info)
}

pub fn validate_optional_info<'a>(
    info: Option<&'a HfqTensorInfo>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Option<&'a HfqTensorInfo>, TensorLookupError> {
    info.map(|info| validate_required_info(Some(info), name, expected_shape))
        .transpose()
}

/// Widen an exact-shape source-precision tensor without applying a norm offset.
pub fn decode_direct_f32(info: &HfqTensorInfo, data: &[u8]) -> Result<Vec<f32>, TensorLookupError> {
    let elements = info
        .shape
        .iter()
        .fold(1usize, |count, &dim| count.saturating_mul(dim as usize));
    let bytes_per_element = match info.quant_type {
        1 | 16 => 2,
        2 => 4,
        quant_type => {
            return Err(TensorLookupError::UnsupportedSourceType {
                name: info.name.clone(),
                quant_type,
            });
        }
    };
    let expected_bytes = elements * bytes_per_element;
    if data.len() != expected_bytes {
        return Err(TensorLookupError::InvalidByteLength {
            name: info.name.clone(),
            actual: data.len(),
            expected: expected_bytes,
        });
    }
    Ok(match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect(),
        16 => data
            .chunks_exact(2)
            .map(|chunk| f32::from_bits((u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16))
            .collect(),
        _ => unreachable!("source type checked above"),
    })
}

pub fn lm_head_source_name<'a>(
    tied: bool,
    embedding_name: &'a str,
    separate_head_name: &'a str,
) -> (&'a str, bool) {
    if tied {
        (embedding_name, true)
    } else {
        (separate_head_name, false)
    }
}

pub struct TransformerLoader<'a> {
    hfq: &'a HfqFile,
    family: &'a str,
}

impl<'a> TransformerLoader<'a> {
    pub fn new(hfq: &'a HfqFile, family: &'a str) -> Self {
        Self { hfq, family }
    }

    fn required_data(&self, name: &str, shape: &[usize]) -> (&HfqTensorInfo, Vec<u8>) {
        validate_required_info(self.hfq.find_tensor_info(name), name, shape)
            .unwrap_or_else(|error| panic!("{}: {error}", self.family));
        self.hfq
            .tensor_data_vec(name)
            .unwrap_or_else(|| panic!("{}: required tensor `{name}` disappeared", self.family))
    }

    pub fn has_exact(&self, name: &str, shape: &[usize]) -> Result<bool, TensorLookupError> {
        Ok(validate_optional_info(self.hfq.find_tensor_info(name), name, shape)?.is_some())
    }

    pub fn load_direct_f32(
        &self,
        gpu: &mut Gpu,
        name: &str,
        shape: &[usize],
    ) -> HipResult<GpuTensor> {
        let (info, data) = self.required_data(name, shape);
        let values = decode_direct_f32(info, &data)
            .unwrap_or_else(|error| panic!("{}: {error}", self.family));
        gpu.upload_f32(&values, shape)
    }

    pub fn load_optional_direct_f32(
        &self,
        gpu: &mut Gpu,
        name: &str,
        shape: &[usize],
    ) -> HipResult<Option<GpuTensor>> {
        if !self
            .has_exact(name, shape)
            .unwrap_or_else(|error| panic!("{}: {error}", self.family))
        {
            return Ok(None);
        }
        self.load_direct_f32(gpu, name, shape).map(Some)
    }

    pub fn load_embedding(
        &self,
        gpu: &mut Gpu,
        name: &str,
        vocab_size: usize,
        hidden_size: usize,
    ) -> HipResult<(GpuTensor, EmbeddingFormat)> {
        let shape = [vocab_size, hidden_size];
        let (info, data) = self.required_data(name, &shape);
        match info.quant_type {
            6 => Ok((
                gpu.upload_raw(&data, &[data.len()])?,
                EmbeddingFormat::HFQ4G256,
            )),
            7 => Ok((
                gpu.upload_raw(&data, &[data.len()])?,
                EmbeddingFormat::HFQ4G128,
            )),
            3 => Ok((gpu.upload_raw(&data, &[data.len()])?, EmbeddingFormat::Q8_0)),
            1 | 16 | 2 => {
                let values = decode_direct_f32(info, &data)
                    .unwrap_or_else(|error| panic!("{}: {error}", self.family));
                Ok((gpu.upload_f32(&values, &shape)?, EmbeddingFormat::F32))
            }
            quant_type => panic!(
                "{}: unsupported embedding quant type {quant_type} for {name}",
                self.family
            ),
        }
    }

    pub fn load_weight(
        &self,
        gpu: &Gpu,
        name: &str,
        m: usize,
        k: usize,
    ) -> HipResult<WeightTensor> {
        let (info, data) = self.required_data(name, &[m, k]);
        let mut weight = match info.quant_type {
            // OQ int8-activation family (33 = OQ+ W4A8, 35 = OQ8 W8A8, 36 = OQ+
            // compact) via the shared `oq8_arch_load`, parallel to the OQ4 arm
            // below — the single arch-agnostic OQ8 dispatch every loader routes
            // through.
            qt @ (33 | 35 | 36) => {
                let (bytes, dtype) = oq8_arch_load(qt, &data, m, k)
                    .expect("oq8_arch_load resolves the OQ8-family codes 33/35/36");
                self.upload_weight_bytes(gpu, bytes, dtype, m, k)?
            }
            OQ4_CANONICAL_QT | OQ4_ARCH_PACKED_QT => {
                let (bytes, dtype) = oq4_arch_load(info.quant_type, &data, m, k)
                    .expect("OQ4 quant type handled by oq4_arch_load");
                self.upload_weight_bytes(gpu, bytes.into_owned(), dtype, m, k)?
            }
            16 => {
                let mut buf = gpu.upload_raw(&data, &[data.len()])?;
                buf.dtype = DType::BF16;
                weight_tensor(buf, DType::BF16, m, k)
            }
            quant_type => {
                let dtype =
                    crate::quant::dtype_for_quant_type(quant_type, k).unwrap_or_else(|| {
                        panic!(
                            "{}: unsupported linear quant type {quant_type} for {name}",
                            self.family
                        )
                    });
                self.upload_weight_bytes(gpu, data, dtype, m, k)?
            }
        };
        if weight.gpu_dtype.supports_awq_sidecar() {
            weight.awq_scale = load_awq_scale(self.hfq, gpu, name, k);
        }
        Ok(weight)
    }

    pub fn load_optional_weight(
        &self,
        gpu: &Gpu,
        name: &str,
        m: usize,
        k: usize,
    ) -> HipResult<Option<WeightTensor>> {
        if !self
            .has_exact(name, &[m, k])
            .unwrap_or_else(|error| panic!("{}: {error}", self.family))
        {
            return Ok(None);
        }
        self.load_weight(gpu, name, m, k).map(Some)
    }

    pub fn load_lm_head(
        &self,
        gpu: &mut Gpu,
        embedding_name: &str,
        separate_head_name: &str,
        tied: bool,
        vocab_size: usize,
        hidden_size: usize,
    ) -> HipResult<(WeightTensor, bool)> {
        let (name, tied) = lm_head_source_name(tied, embedding_name, separate_head_name);
        if !tied {
            return self
                .load_weight(gpu, name, vocab_size, hidden_size)
                .map(|weight| (weight, false));
        }

        let (info, data) = self.required_data(name, &[vocab_size, hidden_size]);
        let weight = match info.quant_type {
            1 | 2 | 16 => {
                let values = decode_direct_f32(info, &data)
                    .unwrap_or_else(|error| panic!("{}: {error}", self.family));
                weight_tensor(
                    gpu.upload_f32(&values, &[vocab_size, hidden_size])?,
                    DType::F32,
                    vocab_size,
                    hidden_size,
                )
            }
            _ => self.load_weight(gpu, name, vocab_size, hidden_size)?,
        };
        Ok((weight, true))
    }

    fn upload_weight_bytes(
        &self,
        gpu: &Gpu,
        data: Vec<u8>,
        dtype: DType,
        m: usize,
        k: usize,
    ) -> HipResult<WeightTensor> {
        Ok(weight_tensor(
            gpu.upload_raw(&data, &[data.len()])?,
            dtype,
            m,
            k,
        ))
    }
}

fn weight_tensor(buf: GpuTensor, gpu_dtype: DType, m: usize, k: usize) -> WeightTensor {
    WeightTensor {
        buf,
        gpu_dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(name: &str, quant_type: u8, shape: &[u32], data_size: usize) -> HfqTensorInfo {
        HfqTensorInfo {
            name: name.to_string(),
            quant_type,
            shape: shape.to_vec(),
            group_size: 0,
            data_offset: 0,
            data_size,
        }
    }

    #[test]
    fn required_reports_missing_tensor() {
        assert!(matches!(
            validate_required_info(None, "model.norm.weight", &[4]),
            Err(TensorLookupError::Missing { .. })
        ));
    }

    #[test]
    fn required_distinguishes_wrong_rank_and_wrong_shape() {
        let rank = info("w", 16, &[2, 2], 8);
        assert!(matches!(
            validate_required_info(Some(&rank), "w", &[4]),
            Err(TensorLookupError::WrongRank { .. })
        ));
        let shape = info("w", 16, &[2, 3], 12);
        assert!(matches!(
            validate_required_info(Some(&shape), "w", &[3, 2]),
            Err(TensorLookupError::WrongShape { .. })
        ));
    }

    #[test]
    fn optional_absence_is_not_an_error_but_present_shape_is_checked() {
        assert!(validate_optional_info(None, "optional.weight", &[2, 2])
            .unwrap()
            .is_none());
        let wrong = info("optional.weight", 16, &[4], 8);
        assert!(matches!(
            validate_optional_info(Some(&wrong), "optional.weight", &[2, 2]),
            Err(TensorLookupError::WrongRank { .. })
        ));
    }

    #[test]
    fn direct_norm_widens_bf16_without_offset() {
        let norm = info("model.norm.weight", 16, &[2], 4);
        let one = (1.0f32.to_bits() >> 16) as u16;
        let half = (0.5f32.to_bits() >> 16) as u16;
        let data = [one.to_le_bytes(), half.to_le_bytes()].concat();
        assert_eq!(decode_direct_f32(&norm, &data).unwrap(), vec![1.0, 0.5]);
    }

    #[test]
    fn tied_head_selects_embedding_and_untied_selects_separate_tensor() {
        assert_eq!(
            lm_head_source_name(true, "model.embed_tokens.weight", "lm_head.weight"),
            ("model.embed_tokens.weight", true)
        );
        assert_eq!(
            lm_head_source_name(false, "model.embed_tokens.weight", "lm_head.weight"),
            ("lm_head.weight", false)
        );
    }
}
