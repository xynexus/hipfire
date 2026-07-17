// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Segmented Qwen3 Q/K head RMSNorm and full rotary embedding on AIE2P.

use hipfire_primitives::conv::f32_to_bf16_bits;

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const PARAM_ALIGNMENT: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3HeadNormRopeGeometry {
    pub sequence_bucket: usize,
    pub dispatch_batch: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl Qwen3HeadNormRopeGeometry {
    pub fn validate(self) -> Result<Self, XdnaError> {
        if !matches!(self.sequence_bucket, 128 | 256 | 512 | 1024 | 2048)
            || self.dispatch_batch == 0
            || self.sequence_bucket * self.dispatch_batch > 4096
            || !matches!(self.query_heads, 16 | 32)
            || self.kv_heads != 8
            || self.head_dim != 128
        {
            return Err(invalid("invalid Qwen3 headnorm/RoPE geometry"));
        }
        Ok(self)
    }

    fn rows(self) -> usize {
        self.sequence_bucket * self.dispatch_batch
    }
}

pub struct NpuQwen3HeadNormRope {
    kernel: NpuKernel,
    geometry: Qwen3HeadNormRopeGeometry,
    query: DeviceBuffer,
    key: DeviceBuffer,
    parameters: Vec<DeviceBuffer>,
    output_query: DeviceBuffer,
    output_key: DeviceBuffer,
}

impl NpuQwen3HeadNormRope {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        xclbin: &[u8],
        instructions: &[u8],
        geometry: Qwen3HeadNormRopeGeometry,
        query_weight: &[f32],
        key_weight: &[f32],
        rope_theta: f32,
        epsilon: f32,
    ) -> Result<Self, XdnaError> {
        Self::load_bank(
            xclbin,
            instructions,
            geometry,
            &[query_weight],
            &[key_weight],
            rope_theta,
            epsilon,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_bank(
        xclbin: &[u8],
        instructions: &[u8],
        geometry: Qwen3HeadNormRopeGeometry,
        query_weights: &[&[f32]],
        key_weights: &[&[f32]],
        rope_theta: f32,
        epsilon: f32,
    ) -> Result<Self, XdnaError> {
        let geometry = geometry.validate()?;
        if query_weights.is_empty()
            || query_weights.len() != key_weights.len()
            || query_weights.iter().chain(key_weights).any(|weight| {
                weight.len() != geometry.head_dim || weight.iter().any(|value| !value.is_finite())
            })
            || !rope_theta.is_finite()
            || rope_theta <= 0.0
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(invalid("invalid Qwen3 headnorm/RoPE parameters"));
        }
        let kernel = NpuKernel::load(xclbin, instructions)?;
        let rows = geometry.rows();
        let query_bytes = rows * geometry.query_heads * geometry.head_dim * 2;
        let key_bytes = rows * geometry.kv_heads * geometry.head_dim * 2;
        let query = kernel.alloc_arg(query_bytes)?;
        let key = kernel.alloc_arg(key_bytes)?;
        let mut parameters = Vec::with_capacity(query_weights.len());
        for (&query_weight, &key_weight) in query_weights.iter().zip(key_weights) {
            let packed = pack_parameters(geometry, query_weight, key_weight, rope_theta, epsilon);
            let mut buffer = kernel.alloc_arg(packed.len())?;
            buffer.as_mut_slice().copy_from_slice(&packed);
            kernel.sync_to_device(&buffer)?;
            parameters.push(buffer);
        }
        let output_query = kernel.alloc_arg(query_bytes)?;
        let output_key = kernel.alloc_arg(key_bytes)?;
        Ok(Self {
            kernel,
            geometry,
            query,
            key,
            parameters,
            output_query,
            output_key,
        })
    }

    pub fn geometry(&self) -> Qwen3HeadNormRopeGeometry {
        self.geometry
    }

    pub fn run(&mut self, query: &[u16], key: &[u16]) -> Result<(Vec<u16>, Vec<u16>), XdnaError> {
        self.run_index(0, query, key)
    }

    pub fn run_index(
        &mut self,
        parameter_index: usize,
        query: &[u16],
        key: &[u16],
    ) -> Result<(Vec<u16>, Vec<u16>), XdnaError> {
        let parameters = self.parameters.get(parameter_index).ok_or_else(|| {
            invalid(format!(
                "Qwen3 headnorm/RoPE parameter index {parameter_index} is outside bank size {}",
                self.parameters.len()
            ))
        })?;
        let q_elements = self.geometry.rows() * self.geometry.query_heads * self.geometry.head_dim;
        let k_elements = self.geometry.rows() * self.geometry.kv_heads * self.geometry.head_dim;
        if query.len() != q_elements || key.len() != k_elements {
            return Err(invalid(format!(
                "Qwen3 headnorm/RoPE inputs have Q={} K={}; expected Q={q_elements} K={k_elements}",
                query.len(),
                key.len()
            )));
        }
        encode_bf16(self.query.as_mut_slice(), query);
        encode_bf16(self.key.as_mut_slice(), key);
        self.kernel.recreate_hwctx()?;
        self.kernel.dispatch_synced(
            &[
                &self.query,
                &self.key,
                parameters,
                &self.output_query,
                &self.output_key,
            ],
            &[true, true, false, true, true],
        )?;
        self.kernel.sync_output(&self.output_query)?;
        self.kernel.sync_output(&self.output_key)?;
        Ok((
            decode_bf16(self.output_query.as_slice()),
            decode_bf16(self.output_key.as_slice()),
        ))
    }
}

fn parameter_bytes(head_dim: usize) -> usize {
    (2 * head_dim * 2 + 4).div_ceil(PARAM_ALIGNMENT) * PARAM_ALIGNMENT
}

fn pack_parameters(
    geometry: Qwen3HeadNormRopeGeometry,
    query_weight: &[f32],
    key_weight: &[f32],
    rope_theta: f32,
    epsilon: f32,
) -> Vec<u8> {
    let record_bytes = parameter_bytes(geometry.head_dim);
    let rows = geometry.rows();
    let mut packed = vec![0u8; 2 * rows * record_bytes];
    let half = geometry.head_dim / 2;
    for kind in 0..2 {
        let weight = if kind == 0 { query_weight } else { key_weight };
        for row in 0..rows {
            let position = row % geometry.sequence_bucket;
            let record = &mut packed[(kind * rows + row) * record_bytes..][..record_bytes];
            for (inner, &value) in weight.iter().enumerate() {
                record[inner * 2..inner * 2 + 2]
                    .copy_from_slice(&f32_to_bf16_bits(value).to_le_bytes());
            }
            for inner in 0..half {
                let frequency = rope_theta.powf(-((2 * inner) as f32) / geometry.head_dim as f32);
                let angle = position as f32 * frequency;
                let cosine_offset = (geometry.head_dim + inner) * 2;
                let sine_offset = (geometry.head_dim + half + inner) * 2;
                record[cosine_offset..cosine_offset + 2]
                    .copy_from_slice(&f32_to_bf16_bits(angle.cos()).to_le_bytes());
                record[sine_offset..sine_offset + 2]
                    .copy_from_slice(&f32_to_bf16_bits(angle.sin()).to_le_bytes());
            }
            let epsilon_offset = 2 * geometry.head_dim * 2;
            record[epsilon_offset..epsilon_offset + 4].copy_from_slice(&epsilon.to_le_bytes());
        }
    }
    packed
}

fn encode_bf16(destination: &mut [u8], values: &[u16]) {
    for (bytes, value) in destination.chunks_exact_mut(2).zip(values) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
}

fn decode_bf16(source: &[u8]) -> Vec<u16> {
    source
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect()
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameters_reset_positions_per_document() {
        let geometry = Qwen3HeadNormRopeGeometry {
            sequence_bucket: 128,
            dispatch_batch: 2,
            query_heads: 16,
            kv_heads: 8,
            head_dim: 128,
        };
        let packed = pack_parameters(geometry, &[1.0; 128], &[2.0; 128], 1_000_000.0, 1e-6);
        let record = parameter_bytes(128);
        assert_eq!(&packed[..record], &packed[128 * record..129 * record]);
    }
}
