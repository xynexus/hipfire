// SPDX-License-Identifier: Apache-2.0

//! Resident R46 BF16x2 to R48 residual-record preparation.

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const PAD_M: usize = 288;
const K: usize = 768;
const COMPLETED_BYTES: usize = PAD_M * K * 2 * size_of::<u16>();
const R34_INPUT_BYTES: usize = 4 * 45 * 16_384;
const RESIDUAL_RECORD_BYTES: usize = 32 * 16_384;
const OUTPUT_BYTES: usize = R34_INPUT_BYTES + RESIDUAL_RECORD_BYTES;

pub struct NpuEmbeddingResidualPrep {
    kernel: NpuKernel,
    bootstrap_completed: DeviceBuffer,
    completed: DeviceBuffer,
    output: DeviceBuffer,
}

impl NpuEmbeddingResidualPrep {
    pub fn load_cached(cache: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{cache}/shape.txt")).map_err(XdnaError::Open)?;
        for field in [
            "op=embeddinggemma-residual-prep",
            "m=256",
            "k=768",
            "input=shared-completed-bf16x2",
            "output=shared-activation-tail-r34-bf16-records",
            "residual-records=32",
            "residual-record-bytes=16384",
        ] {
            if !manifest.lines().any(|line| line == field) {
                return Err(invalid(format!("residual prep cache missing {field}")));
            }
        }
        let kernel = NpuKernel::load(
            &std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?,
            &std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?,
        )?;
        Ok(Self {
            bootstrap_completed: kernel.alloc_arg(COMPLETED_BYTES)?,
            completed: kernel.alloc_arg(COMPLETED_BYTES)?,
            output: kernel.alloc_arg(OUTPUT_BYTES)?,
            kernel,
        })
    }

    pub const fn completed_bytes() -> usize {
        COMPLETED_BYTES
    }

    pub const fn activation_bytes() -> usize {
        R34_INPUT_BYTES
    }

    pub const fn output_bytes() -> usize {
        OUTPUT_BYTES
    }

    pub fn attach_shared(
        &mut self,
        completed_fd: i32,
        completed_bytes: usize,
        output_fd: i32,
        output_bytes: usize,
    ) -> Result<(), XdnaError> {
        if completed_bytes != COMPLETED_BYTES || output_bytes != OUTPUT_BYTES {
            return Err(invalid("residual prep shared dma-buf size mismatch"));
        }
        self.completed = self
            .kernel
            .import_dmabuf(completed_fd, completed_bytes, true)?;
        self.output = self.kernel.import_dmabuf(output_fd, output_bytes, true)?;
        Ok(())
    }

    pub fn write_completed_bf16x2(&mut self, bytes: &[u8]) -> Result<(), XdnaError> {
        if bytes.len() != COMPLETED_BYTES {
            return Err(invalid("residual prep completed-state size mismatch"));
        }
        self.completed.as_mut_slice().copy_from_slice(bytes);
        self.kernel.sync_to_device(&self.completed)
    }

    pub fn write_bootstrap_bf16x2(&mut self, bytes: &[u8]) -> Result<(), XdnaError> {
        if bytes.len() != COMPLETED_BYTES {
            return Err(invalid("residual prep bootstrap-state size mismatch"));
        }
        self.bootstrap_completed
            .as_mut_slice()
            .copy_from_slice(bytes);
        self.kernel.sync_to_device(&self.bootstrap_completed)
    }

    pub fn fill_output(&mut self, value: u8) -> Result<(), XdnaError> {
        self.output.as_mut_slice().fill(value);
        self.kernel.sync_to_device(&self.output)
    }

    pub fn run_shared(&mut self) -> Result<(), XdnaError> {
        self.kernel
            .dispatch_synced(&[&self.completed, &self.output], &[false, false])?;
        self.kernel.sync_output(&self.output)
    }

    pub fn run_bootstrap(&mut self) -> Result<(), XdnaError> {
        self.kernel
            .dispatch_synced(&[&self.bootstrap_completed, &self.output], &[false, false])?;
        self.kernel.sync_output(&self.output)
    }

    pub fn output(&self) -> &[u8] {
        self.output.as_slice()
    }
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_r46_r47_and_r48() {
        assert_eq!(NpuEmbeddingResidualPrep::completed_bytes(), 884_736);
        assert_eq!(NpuEmbeddingResidualPrep::activation_bytes(), 2_949_120);
        assert_eq!(NpuEmbeddingResidualPrep::output_bytes(), 3_473_408);
    }
}
