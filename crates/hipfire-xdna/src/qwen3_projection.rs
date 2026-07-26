// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Resident Qwen3 OQ8+ projection with load-time BF16 dequantization.

use crate::{DeviceBuffer, NpuKernel, OpusPackedMatrix, OpusResidentMode, XdnaError};

const GROUP: usize = 256;
const OUTPUT_TILE: usize = 16;
const MMUL_K: usize = 8;
const MMUL_N: usize = 8;
const W_TILE: usize = GROUP * OUTPUT_TILE * size_of::<u16>();

pub struct NpuQwen3Oq8Projection {
    kernel: NpuKernel,
    rows: usize,
    input_columns: usize,
    output_columns: usize,
    input: DeviceBuffer,
    weights: Vec<DeviceBuffer>,
    output: DeviceBuffer,
}

impl NpuQwen3Oq8Projection {
    pub fn load(
        xclbin: &[u8],
        instructions: &[u8],
        rows: usize,
        input_columns: usize,
        output_columns: usize,
        matrix: &OpusPackedMatrix,
    ) -> Result<Self, XdnaError> {
        Self::load_bank(
            xclbin,
            instructions,
            rows,
            input_columns,
            output_columns,
            &[matrix],
        )
    }

    pub fn load_bank(
        xclbin: &[u8],
        instructions: &[u8],
        rows: usize,
        input_columns: usize,
        output_columns: usize,
        matrices: &[&OpusPackedMatrix],
    ) -> Result<Self, XdnaError> {
        if matrices.is_empty() {
            return Err(invalid("Qwen3 OQ8 projection bank is empty"));
        }
        for matrix in matrices {
            validate_geometry(rows, input_columns, output_columns, matrix)?;
        }
        let kernel = NpuKernel::load(xclbin, instructions)?;
        let input = kernel.alloc_arg(rows * input_columns * size_of::<u16>())?;
        let mut weights = Vec::with_capacity(matrices.len());
        for matrix in matrices {
            let packed = pack_weights(matrix);
            let mut buffer = kernel.alloc_arg(packed.len())?;
            buffer.as_mut_slice().copy_from_slice(&packed);
            kernel.sync_to_device(&buffer)?;
            weights.push(buffer);
        }
        let output = kernel.alloc_arg(rows * output_columns * size_of::<u16>())?;
        Ok(Self {
            kernel,
            rows,
            input_columns,
            output_columns,
            input,
            weights,
            output,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn input_columns(&self) -> usize {
        self.input_columns
    }

    pub fn output_columns(&self) -> usize {
        self.output_columns
    }

    pub fn run(&mut self, token_major_bf16: &[u16]) -> Result<Vec<u16>, XdnaError> {
        self.run_index(0, token_major_bf16)
    }

    pub fn run_index(
        &mut self,
        matrix_index: usize,
        token_major_bf16: &[u16],
    ) -> Result<Vec<u16>, XdnaError> {
        let weights = self.weights.get(matrix_index).ok_or_else(|| {
            invalid(format!(
                "Qwen3 OQ8 projection matrix index {matrix_index} is outside bank size {}",
                self.weights.len()
            ))
        })?;
        let expected = self.rows * self.input_columns;
        if token_major_bf16.len() != expected {
            return Err(invalid(format!(
                "Qwen3 OQ8 projection has {} BF16 values; expected {expected}",
                token_major_bf16.len()
            )));
        }
        for (bytes, &value) in self
            .input
            .as_mut_slice()
            .chunks_exact_mut(size_of::<u16>())
            .zip(token_major_bf16)
        {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        // The 32-core projection graph does not reliably re-arm every broadcast
        // consumer after a completed runtime sequence on the current amdxdna
        // stack. Recreating only the hwctx resets FIFO/core state while keeping
        // the PDI, instructions, resident weights, and argument BOs intact.
        self.kernel.recreate_hwctx()?;
        self.kernel
            .dispatch_synced(&[&self.input, weights, &self.output], &[true, false, true])?;
        self.kernel.sync_output(&self.output)?;
        Ok(self
            .output
            .as_slice()
            .chunks_exact(size_of::<u16>())
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect())
    }
}

fn validate_geometry(
    rows: usize,
    input_columns: usize,
    output_columns: usize,
    matrix: &OpusPackedMatrix,
) -> Result<(), XdnaError> {
    if rows == 0 || rows > 4096 || !rows.is_multiple_of(256) {
        return Err(invalid(
            "Qwen3 OQ8 projection rows must be a multiple of 256 in 256..=4096",
        ));
    }
    if input_columns == 0
        || !input_columns.is_multiple_of(GROUP)
        || output_columns == 0
        || !output_columns.is_multiple_of(OUTPUT_TILE)
    {
        return Err(invalid("Qwen3 OQ8 projection needs K%256=0 and N%16=0"));
    }
    if matrix.resident_mode() != OpusResidentMode::DenseW8
        || matrix.k() != input_columns
        || matrix.n() != output_columns
    {
        return Err(invalid(format!(
            "Qwen3 OQ8 projection matrix is {:?} K={} N={}; expected dense W8 K={input_columns} N={output_columns}",
            matrix.resident_mode(),
            matrix.k(),
            matrix.n()
        )));
    }
    Ok(())
}

fn pack_weights(matrix: &OpusPackedMatrix) -> Vec<u8> {
    let groups = matrix.group_count();
    let output_tiles = matrix.n() / OUTPUT_TILE;
    let mut packed = vec![0u8; output_tiles * groups * W_TILE];
    let dequantized = matrix.dequantized_bf16();
    for output_tile in 0..output_tiles {
        for group in 0..groups {
            let base = (output_tile * groups + group) * W_TILE;
            for k_tile in 0..GROUP / MMUL_K {
                for output_half in 0..2 {
                    for k_lane in 0..MMUL_K {
                        for output_lane in 0..MMUL_N {
                            let inner = k_tile * MMUL_K + k_lane;
                            let lane = output_half * MMUL_N + output_lane;
                            let row = group * GROUP + inner;
                            let output = output_tile * OUTPUT_TILE + lane;
                            let value = dequantized[row * matrix.n() + output];
                            let destination = base
                                + (k_tile * 2 + output_half) * MMUL_K * MMUL_N * 2
                                + (k_lane * MMUL_N + output_lane) * 2;
                            packed[destination..destination + 2]
                                .copy_from_slice(&value.to_le_bytes());
                        }
                    }
                }
            }
        }
    }
    packed
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_weight_tile_geometry_is_cacheline_aligned() {
        assert_eq!(W_TILE, 8_192);
        assert!(W_TILE.is_multiple_of(64));
    }
}
