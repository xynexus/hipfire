// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Validated decoder for the canonical `HFENCB01` Qwen3 encoder blob.

use hipfire_primitives::conv::{bf16_bits_to_f32, f16_bits_to_f32};

use crate::{OpusPackedMatrix, XdnaError};

const MAGIC: &[u8; 8] = b"HFENCB01";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 64;
const DESCRIPTOR_BYTES: usize = 48;
const SIDECAR_BIT: u16 = 0x8000;

pub(crate) struct Qwen3EncoderBlobWeights {
    pub hidden_size: usize,
    pub layer_count: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub norm_epsilon: f32,
    pub rope_theta: f32,
    pub final_norm: Vec<f32>,
    pub layers: Vec<Qwen3EncoderLayerWeights>,
}

pub(crate) struct Qwen3EncoderLayerWeights {
    pub input_norm: Vec<f32>,
    pub query: OpusPackedMatrix,
    pub key: OpusPackedMatrix,
    pub value: OpusPackedMatrix,
    pub attention_output: OpusPackedMatrix,
    pub query_norm: Vec<f32>,
    pub key_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub gate: OpusPackedMatrix,
    pub up: OpusPackedMatrix,
    pub down: OpusPackedMatrix,
}

#[derive(Clone, Copy)]
struct Entry<'a> {
    role: u16,
    layer: u32,
    quant_type: u8,
    shape: [usize; 2],
    rank: usize,
    group_size: usize,
    payload: &'a [u8],
}

impl Qwen3EncoderBlobWeights {
    pub fn parse(blob: &[u8]) -> Result<Self, XdnaError> {
        if blob.len() < HEADER_BYTES || &blob[..8] != MAGIC || u32_at(blob, 8)? != VERSION {
            return Err(invalid("invalid HFENCB01 header"));
        }
        let entry_count = usize_at(blob, 12)?;
        let descriptor_end = HEADER_BYTES
            .checked_add(entry_count * DESCRIPTOR_BYTES)
            .ok_or_else(|| invalid("HFENCB01 descriptor overflow"))?;
        if descriptor_end > blob.len() {
            return Err(invalid("truncated HFENCB01 descriptor table"));
        }
        let hidden_size = usize_at(blob, 16)?;
        let layer_count = usize_at(blob, 20)?;
        let query_heads = usize_at(blob, 24)?;
        let kv_heads = usize_at(blob, 28)?;
        let head_dim = usize_at(blob, 32)?;
        let intermediate_size = usize_at(blob, 36)?;
        let norm_epsilon = f32_at(blob, 40)?;
        let rope_theta = f32_at(blob, 44)?;
        if hidden_size == 0
            || layer_count == 0
            || query_heads == 0
            || kv_heads == 0
            || head_dim == 0
            || intermediate_size == 0
            || !norm_epsilon.is_finite()
            || norm_epsilon <= 0.0
            || !rope_theta.is_finite()
            || rope_theta <= 0.0
        {
            return Err(invalid("HFENCB01 geometry must be positive"));
        }
        let mut entries = Vec::with_capacity(entry_count);
        for index in 0..entry_count {
            let descriptor = HEADER_BYTES + index * DESCRIPTOR_BYTES;
            let role = u16_at(blob, descriptor)?;
            let layer = u32_at(blob, descriptor + 4)?;
            let quant_type = u32_at(blob, descriptor + 8)? as u8;
            let rank = usize_at(blob, descriptor + 12)?;
            let shape = [
                usize_at(blob, descriptor + 16)?,
                usize_at(blob, descriptor + 20)?,
            ];
            let group_size = usize_at(blob, descriptor + 24)?;
            let offset = u64_at(blob, descriptor + 32)? as usize;
            let bytes = u64_at(blob, descriptor + 40)? as usize;
            let end = offset
                .checked_add(bytes)
                .ok_or_else(|| invalid("HFENCB01 payload range overflow"))?;
            if offset < descriptor_end || end > blob.len() {
                return Err(invalid(format!(
                    "HFENCB01 entry {index} payload is outside the blob"
                )));
            }
            entries.push(Entry {
                role,
                layer,
                quant_type,
                shape,
                rank,
                group_size,
                payload: &blob[offset..end],
            });
        }
        let final_norm = vector(required(&entries, 1, u32::MAX)?, hidden_size)?;
        let mut layers = Vec::with_capacity(layer_count);
        for layer in 0..layer_count {
            let layer = layer as u32;
            layers.push(Qwen3EncoderLayerWeights {
                input_norm: vector(required(&entries, 2, layer)?, hidden_size)?,
                query: matrix(&entries, 3, layer)?,
                key: matrix(&entries, 4, layer)?,
                value: matrix(&entries, 5, layer)?,
                attention_output: matrix(&entries, 6, layer)?,
                query_norm: vector(required(&entries, 7, layer)?, head_dim)?,
                key_norm: vector(required(&entries, 8, layer)?, head_dim)?,
                post_attention_norm: vector(required(&entries, 9, layer)?, hidden_size)?,
                gate: matrix(&entries, 10, layer)?,
                up: matrix(&entries, 11, layer)?,
                down: matrix(&entries, 12, layer)?,
            });
        }
        let parsed = Self {
            hidden_size,
            layer_count,
            query_heads,
            kv_heads,
            head_dim,
            intermediate_size,
            norm_epsilon,
            rope_theta,
            final_norm,
            layers,
        };
        parsed.validate_shapes()?;
        Ok(parsed)
    }

    fn validate_shapes(&self) -> Result<(), XdnaError> {
        let q_width = self.query_heads * self.head_dim;
        let kv_width = self.kv_heads * self.head_dim;
        for (index, layer) in self.layers.iter().enumerate() {
            for (name, matrix, k, n) in [
                ("query", &layer.query, self.hidden_size, q_width),
                ("key", &layer.key, self.hidden_size, kv_width),
                ("value", &layer.value, self.hidden_size, kv_width),
                (
                    "attention output",
                    &layer.attention_output,
                    q_width,
                    self.hidden_size,
                ),
                (
                    "gate",
                    &layer.gate,
                    self.hidden_size,
                    self.intermediate_size,
                ),
                ("up", &layer.up, self.hidden_size, self.intermediate_size),
                (
                    "down",
                    &layer.down,
                    self.intermediate_size,
                    self.hidden_size,
                ),
            ] {
                if matrix.k() != k || matrix.n() != n {
                    return Err(invalid(format!(
                        "HFENCB01 layer {index} {name} is K={} N={}; expected K={k} N={n}",
                        matrix.k(),
                        matrix.n()
                    )));
                }
            }
        }
        Ok(())
    }
}

fn required<'a>(entries: &'a [Entry<'a>], role: u16, layer: u32) -> Result<Entry<'a>, XdnaError> {
    let mut matches = entries
        .iter()
        .copied()
        .filter(|entry| entry.role == role && entry.layer == layer);
    let entry = matches
        .next()
        .ok_or_else(|| invalid(format!("HFENCB01 is missing role={role} layer={layer}")))?;
    if matches.next().is_some() {
        return Err(invalid(format!(
            "HFENCB01 duplicates role={role} layer={layer}"
        )));
    }
    Ok(entry)
}

fn matrix(entries: &[Entry<'_>], role: u16, layer: u32) -> Result<OpusPackedMatrix, XdnaError> {
    let entry = required(entries, role, layer)?;
    if entry.rank != 2 || !matches!(entry.quant_type, 35 | 43) || entry.group_size != 256 {
        return Err(invalid(format!(
            "HFENCB01 role={role} layer={layer} is not an OQ8 G256 matrix"
        )));
    }
    let awq = entries
        .iter()
        .find(|sidecar| sidecar.role == (role | SIDECAR_BIT) && sidecar.layer == layer)
        .map(|sidecar| vector(*sidecar, entry.shape[1]))
        .transpose()?;
    OpusPackedMatrix::from_payload(
        entry.quant_type,
        entry.shape[1],
        entry.shape[0],
        entry.payload,
        awq,
    )
}

fn vector(entry: Entry<'_>, expected: usize) -> Result<Vec<f32>, XdnaError> {
    if entry.rank != 1 || entry.shape[0] != expected {
        return Err(invalid(format!(
            "HFENCB01 vector has rank={} shape={:?}; expected [{expected}]",
            entry.rank, entry.shape
        )));
    }
    let values: Vec<f32> = match entry.quant_type {
        1 => entry
            .payload
            .chunks_exact(2)
            .map(|bytes| f16_bits_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect(),
        2 => entry
            .payload
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte f32")))
            .collect(),
        16 => entry
            .payload
            .chunks_exact(2)
            .map(|bytes| bf16_bits_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect(),
        quant => {
            return Err(invalid(format!(
                "HFENCB01 vector quant_type={quant} is unsupported"
            )))
        }
    };
    if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("HFENCB01 vector payload is invalid"));
    }
    Ok(values)
}

fn u16_at(bytes: &[u8], offset: usize) -> Result<u16, XdnaError> {
    bytes
        .get(offset..offset + 2)
        .map(|bytes| u16::from_le_bytes(bytes.try_into().expect("two bytes")))
        .ok_or_else(|| invalid("truncated HFENCB01 integer"))
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, XdnaError> {
    bytes
        .get(offset..offset + 4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four bytes")))
        .ok_or_else(|| invalid("truncated HFENCB01 integer"))
}

fn usize_at(bytes: &[u8], offset: usize) -> Result<usize, XdnaError> {
    Ok(u32_at(bytes, offset)? as usize)
}

fn u64_at(bytes: &[u8], offset: usize) -> Result<u64, XdnaError> {
    bytes
        .get(offset..offset + 8)
        .map(|bytes| u64::from_le_bytes(bytes.try_into().expect("eight bytes")))
        .ok_or_else(|| invalid("truncated HFENCB01 integer"))
}

fn f32_at(bytes: &[u8], offset: usize) -> Result<f32, XdnaError> {
    bytes
        .get(offset..offset + 4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four bytes")))
        .ok_or_else(|| invalid("truncated HFENCB01 float"))
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_truncated_and_wrong_magic() {
        assert!(Qwen3EncoderBlobWeights::parse(&[]).is_err());
        let mut header = vec![0u8; HEADER_BYTES];
        header[..8].copy_from_slice(b"BADMAGIC");
        assert!(Qwen3EncoderBlobWeights::parse(&header).is_err());
    }
}
