// SPDX-License-Identifier: Apache-2.0

//! R105 canonical direct-X to canonical unit-RMS BF16.

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const M: usize = 256;
const PAD_M: usize = 288;
const K: usize = 768;
const DIRECT_X_BYTES: usize = M * K * size_of::<u16>();
const R_STAGE_BYTES: usize = 5 * 48 * 16_384;
const HIDDEN_BACKING_BYTES: usize = R_STAGE_BYTES + 3 * DIRECT_X_BYTES;
const OUTPUT_BYTES: usize = PAD_M * K * size_of::<u16>();

pub struct NpuEmbeddingPreFfnUnitRms {
    kernel: NpuKernel,
    input: DeviceBuffer,
    output: DeviceBuffer,
}

impl NpuEmbeddingPreFfnUnitRms {
    pub fn load_cached(cache: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{cache}/shape.txt")).map_err(XdnaError::Open)?;
        for field in [
            "op=embeddinggemma-direct-x-unit-rms",
            "mode=bf16",
            "m=256",
            "k=768",
            "input=r44-canonical-direct-x",
            "output=canonical-bf16-unit-rms",
            "output-bytes=442368",
            "immutable-pre-ffn-norm=loader-folded",
        ] {
            if !manifest.lines().any(|line| line == field) {
                return Err(invalid(format!("pre-FFN unit-RMS cache missing {field}")));
            }
        }
        let kernel = NpuKernel::load(
            &std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?,
            &std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?,
        )?;
        let mut input = kernel.alloc_arg(HIDDEN_BACKING_BYTES)?;
        input.as_mut_slice().fill(0);
        kernel.sync_to_device(&input)?;
        let mut output = kernel.alloc_arg(OUTPUT_BYTES)?;
        output.as_mut_slice().fill(0);
        kernel.sync_to_device(&output)?;
        Ok(Self {
            kernel,
            input,
            output,
        })
    }

    pub const fn rows() -> usize {
        M
    }

    pub const fn direct_x_bytes() -> usize {
        DIRECT_X_BYTES
    }

    pub const fn input_bytes() -> usize {
        HIDDEN_BACKING_BYTES
    }

    pub const fn output_bytes() -> usize {
        OUTPUT_BYTES
    }

    pub fn attach_shared_input(&mut self, fd: i32, bytes: usize) -> Result<(), XdnaError> {
        if bytes < HIDDEN_BACKING_BYTES {
            return Err(invalid("pre-FFN unit-RMS shared input is too small"));
        }
        self.input = self.kernel.import_dmabuf(fd, bytes, true)?;
        Ok(())
    }

    pub fn attach_shared_output(&mut self, fd: i32, bytes: usize) -> Result<(), XdnaError> {
        if bytes != OUTPUT_BYTES {
            return Err(invalid("pre-FFN unit-RMS shared output size mismatch"));
        }
        self.output = self.kernel.import_dmabuf(fd, bytes, true)?;
        Ok(())
    }

    pub fn sync_shared_input(&self) -> Result<(), XdnaError> {
        self.kernel.sync_to_device(&self.input)
    }

    pub fn write_direct_x_bf16(&mut self, values: &[u16]) -> Result<(), XdnaError> {
        if values.len() != M * K {
            return Err(invalid("pre-FFN unit-RMS direct-X shape mismatch"));
        }
        self.input.as_mut_slice().fill(0);
        for (bytes, &value) in self.input.as_mut_slice()[..DIRECT_X_BYTES]
            .chunks_exact_mut(size_of::<u16>())
            .zip(values)
        {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        self.kernel.sync_to_device(&self.input)
    }

    /// Dispatch without host output synchronization. This is the production
    /// NPU-to-NPU boundary used for timing and shared-buffer composition.
    pub fn run_shared(&self) -> Result<(), XdnaError> {
        self.kernel
            .dispatch_synced(&[&self.input, &self.output], &[false, false])
    }

    pub fn sync_shared_output(&self) -> Result<(), XdnaError> {
        self.kernel.sync_output(&self.output)
    }

    pub fn read_output_bf16(&self) -> Result<Vec<u16>, XdnaError> {
        self.kernel.sync_output(&self.output)?;
        Ok(self.output.as_slice()[..DIRECT_X_BYTES]
            .chunks_exact(size_of::<u16>())
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect())
    }
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_r44_and_r99() {
        assert_eq!(NpuEmbeddingPreFfnUnitRms::rows(), 256);
        assert_eq!(NpuEmbeddingPreFfnUnitRms::direct_x_bytes(), 393_216);
        assert_eq!(NpuEmbeddingPreFfnUnitRms::input_bytes(), 5_111_808);
        assert_eq!(NpuEmbeddingPreFfnUnitRms::output_bytes(), 442_368);
    }
}
