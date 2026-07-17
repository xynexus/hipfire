// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Qwen3 residual-add plus weighted RMSNorm on AIE2P.

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const PARAM_PADDING: usize = 16;

pub struct NpuQwen3ResidualRmsNorm {
    kernel: NpuKernel,
    rows: usize,
    hidden_size: usize,
    input: DeviceBuffer,
    parameters: Vec<DeviceBuffer>,
    output: DeviceBuffer,
}

impl NpuQwen3ResidualRmsNorm {
    pub fn load(
        xclbin: &[u8],
        instructions: &[u8],
        rows: usize,
        hidden_size: usize,
        weight: &[f32],
        epsilon: f32,
    ) -> Result<Self, XdnaError> {
        Self::load_bank(xclbin, instructions, rows, hidden_size, &[weight], epsilon)
    }

    pub fn load_bank(
        xclbin: &[u8],
        instructions: &[u8],
        rows: usize,
        hidden_size: usize,
        weights: &[&[f32]],
        epsilon: f32,
    ) -> Result<Self, XdnaError> {
        validate_geometry(rows, hidden_size)?;
        if weights.is_empty()
            || weights.iter().any(|weight| {
                weight.len() != hidden_size || weight.iter().any(|value| !value.is_finite())
            })
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(invalid("invalid Qwen3 residual RMSNorm parameters"));
        }
        let kernel = NpuKernel::load(xclbin, instructions)?;
        let io_bytes = rows * 2 * hidden_size * size_of::<u16>();
        let input = kernel.alloc_arg(io_bytes)?;
        let mut parameters = Vec::with_capacity(weights.len());
        for weight in weights {
            let mut buffer = kernel.alloc_arg((hidden_size + PARAM_PADDING) * size_of::<f32>())?;
            for (bytes, value) in buffer
                .as_mut_slice()
                .chunks_exact_mut(size_of::<f32>())
                .zip(weight.iter().copied().chain(std::iter::once(epsilon)))
            {
                bytes.copy_from_slice(&value.to_le_bytes());
            }
            kernel.sync_to_device(&buffer)?;
            parameters.push(buffer);
        }
        let output = kernel.alloc_arg(io_bytes)?;
        Ok(Self {
            kernel,
            rows,
            hidden_size,
            input,
            parameters,
            output,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn run(
        &mut self,
        residual_bf16: &[u16],
        delta_bf16: &[u16],
    ) -> Result<(Vec<u16>, Vec<u16>), XdnaError> {
        self.run_index(0, residual_bf16, delta_bf16)
    }

    pub fn run_index(
        &mut self,
        parameter_index: usize,
        residual_bf16: &[u16],
        delta_bf16: &[u16],
    ) -> Result<(Vec<u16>, Vec<u16>), XdnaError> {
        let parameters = self.parameters.get(parameter_index).ok_or_else(|| {
            invalid(format!(
                "Qwen3 residual RMSNorm parameter index {parameter_index} is outside bank size {}",
                self.parameters.len()
            ))
        })?;
        let elements = self.rows * self.hidden_size;
        if residual_bf16.len() != elements || delta_bf16.len() != elements {
            return Err(invalid(format!(
                "Qwen3 residual RMSNorm inputs must each contain {elements} BF16 values"
            )));
        }
        let record_bytes = 2 * self.hidden_size * size_of::<u16>();
        for row in 0..self.rows {
            let target =
                &mut self.input.as_mut_slice()[row * record_bytes..(row + 1) * record_bytes];
            encode_bf16(
                &mut target[..record_bytes / 2],
                &residual_bf16[row * self.hidden_size..(row + 1) * self.hidden_size],
            );
            encode_bf16(
                &mut target[record_bytes / 2..],
                &delta_bf16[row * self.hidden_size..(row + 1) * self.hidden_size],
            );
        }
        // Reset array-local FIFO/core state between public request sequences;
        // the full encoder will compose repeated layer phases inside one image.
        self.kernel.recreate_hwctx()?;
        self.kernel.dispatch_synced(
            &[&self.input, parameters, &self.output],
            &[true, false, true],
        )?;
        self.kernel.sync_output(&self.output)?;
        let mut completed = vec![0u16; elements];
        let mut normalized = vec![0u16; elements];
        for row in 0..self.rows {
            let source = &self.output.as_slice()[row * record_bytes..(row + 1) * record_bytes];
            decode_bf16(
                &source[..record_bytes / 2],
                &mut completed[row * self.hidden_size..(row + 1) * self.hidden_size],
            );
            decode_bf16(
                &source[record_bytes / 2..],
                &mut normalized[row * self.hidden_size..(row + 1) * self.hidden_size],
            );
        }
        Ok((completed, normalized))
    }
}

fn validate_geometry(rows: usize, hidden_size: usize) -> Result<(), XdnaError> {
    if rows == 0 || rows > 4096 || !rows.is_multiple_of(256) {
        return Err(invalid(
            "Qwen3 residual RMSNorm rows must be a multiple of 256 in 256..=4096",
        ));
    }
    if hidden_size == 0 || hidden_size > 4096 || !hidden_size.is_multiple_of(256) {
        return Err(invalid(
            "Qwen3 residual RMSNorm hidden size must be a multiple of 256 in 256..=4096",
        ));
    }
    Ok(())
}

fn encode_bf16(destination: &mut [u8], values: &[u16]) {
    for (bytes, value) in destination.chunks_exact_mut(2).zip(values) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
}

fn decode_bf16(source: &[u8], destination: &mut [u16]) {
    for (value, bytes) in destination.iter_mut().zip(source.chunks_exact(2)) {
        *value = u16::from_le_bytes([bytes[0], bytes[1]]);
    }
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_qwen3_embedding_models() {
        for hidden_size in [1024, 2560, 4096] {
            validate_geometry(256, hidden_size).unwrap();
        }
        assert!(validate_geometry(257, 1024).is_err());
    }
}
