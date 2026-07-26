// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Qwen3 final weighted RMSNorm, last-real-token pooling, and L2 on AIE2P.

use crate::{DeviceBuffer, NpuKernel, XdnaError};

pub struct NpuQwen3FinalPoolL2 {
    kernel: NpuKernel,
    bucket: usize,
    batch: usize,
    hidden_size: usize,
    physical_batch: usize,
    hidden: DeviceBuffer,
    lengths: DeviceBuffer,
    parameters: DeviceBuffer,
    output: DeviceBuffer,
}

impl NpuQwen3FinalPoolL2 {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        xclbin: &[u8],
        instructions: &[u8],
        bucket: usize,
        batch: usize,
        hidden_size: usize,
        weight: &[f32],
        epsilon: f32,
    ) -> Result<Self, XdnaError> {
        if !matches!(bucket, 128 | 256 | 512 | 1024 | 2048)
            || batch == 0
            || batch > 32
            || hidden_size == 0
            || hidden_size > 4096
            || !hidden_size.is_multiple_of(256)
            || weight.len() != hidden_size
            || weight.iter().any(|value| !value.is_finite())
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(invalid(
                "invalid Qwen3 final pool/L2 geometry or parameters",
            ));
        }
        let physical_batch = batch.next_multiple_of(2);
        let pairs = physical_batch / 2;
        let kernel = NpuKernel::load(xclbin, instructions)?;
        let hidden = kernel.alloc_arg(physical_batch * bucket * hidden_size * 2)?;
        let lengths = kernel.alloc_arg(pairs * bucket * 8)?;
        let mut parameters = kernel.alloc_arg((hidden_size + 16) * 4)?;
        for (bytes, value) in parameters
            .as_mut_slice()
            .chunks_exact_mut(4)
            .zip(weight.iter().copied().chain(std::iter::once(epsilon)))
        {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        kernel.sync_to_device(&parameters)?;
        let output = kernel.alloc_arg(physical_batch * hidden_size * 4)?;
        Ok(Self {
            kernel,
            bucket,
            batch,
            hidden_size,
            physical_batch,
            hidden,
            lengths,
            parameters,
            output,
        })
    }

    pub fn run(&mut self, hidden: &[u16], lengths: &[u32]) -> Result<Vec<Vec<f32>>, XdnaError> {
        let elements = self.batch * self.bucket * self.hidden_size;
        if hidden.len() != elements || lengths.len() != self.batch {
            return Err(invalid(format!(
                "Qwen3 final pool/L2 received hidden={} lengths={}; expected hidden={elements} lengths={}",
                hidden.len(), lengths.len(), self.batch
            )));
        }
        for (index, &length) in lengths.iter().enumerate() {
            if length == 0 || length as usize > self.bucket {
                return Err(invalid(format!(
                    "Qwen3 final pool/L2 length[{index}]={length} is outside 1..={}",
                    self.bucket
                )));
            }
        }
        self.hidden.as_mut_slice().fill(0);
        for (bytes, value) in self.hidden.as_mut_slice().chunks_exact_mut(2).zip(hidden) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        for pair in 0..self.physical_batch / 2 {
            let pair_lengths = [
                lengths.get(2 * pair).copied().unwrap_or(0),
                lengths.get(2 * pair + 1).copied().unwrap_or(0),
            ];
            for token in 0..self.bucket {
                let offset = (pair * self.bucket + token) * 8;
                self.lengths.as_mut_slice()[offset..offset + 4]
                    .copy_from_slice(&pair_lengths[0].to_le_bytes());
                self.lengths.as_mut_slice()[offset + 4..offset + 8]
                    .copy_from_slice(&pair_lengths[1].to_le_bytes());
            }
        }
        self.kernel.recreate_hwctx()?;
        self.kernel.dispatch_synced(
            &[&self.hidden, &self.lengths, &self.parameters, &self.output],
            &[true, true, false, true],
        )?;
        self.kernel.sync_output(&self.output)?;
        let flat = self
            .output
            .as_slice()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte f32")))
            .collect::<Vec<_>>();
        Ok(flat[..self.batch * self.hidden_size]
            .chunks_exact(self.hidden_size)
            .map(<[f32]>::to_vec)
            .collect())
    }
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}
