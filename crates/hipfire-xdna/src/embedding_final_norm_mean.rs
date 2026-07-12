// SPDX-License-Identifier: Apache-2.0

//! Resident EmbeddingGemma final RMSNorm and mean pooling over R46 BF16x2.

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const M: usize = 256;
const PAD_M: usize = 288;
const K: usize = 768;
const COMPLETED_BYTES: usize = PAD_M * K * 2 * size_of::<u16>();
const PARAM_BYTES: usize = 4096;
const EPSILON_OFFSET: usize = K * size_of::<f32>();
const OUTPUT_BYTES: usize = K * size_of::<f32>();

pub struct NpuEmbeddingFinalNormMeanParams {
    buffer: DeviceBuffer,
}

pub struct NpuEmbeddingFinalNormMean {
    kernel: NpuKernel,
    completed: DeviceBuffer,
    output: DeviceBuffer,
    primed: bool,
}

impl NpuEmbeddingFinalNormMean {
    pub fn load_cached(cache: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{cache}/shape.txt")).map_err(XdnaError::Open)?;
        for field in [
            "op=embeddinggemma-final-norm-pool",
            "mode=bf16x2-resident",
            "m=256",
            "k=768",
            "pool=mean",
            "input=shared-completed-bf16x2",
            "output=pooled-f32",
        ] {
            if !manifest.lines().any(|line| line == field) {
                return Err(invalid(format!("final norm/mean cache missing {field}")));
            }
        }
        let kernel = NpuKernel::load(
            &std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?,
            &std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?,
        )?;
        Ok(Self {
            completed: kernel.alloc_arg(COMPLETED_BYTES)?,
            output: kernel.alloc_arg(OUTPUT_BYTES)?,
            kernel,
            primed: false,
        })
    }

    pub const fn rows() -> usize {
        M
    }

    pub const fn completed_bytes() -> usize {
        COMPLETED_BYTES
    }

    pub const fn output_bytes() -> usize {
        OUTPUT_BYTES
    }

    pub fn attach_shared_completed(&mut self, fd: i32, bytes: usize) -> Result<(), XdnaError> {
        if bytes != COMPLETED_BYTES {
            return Err(invalid("final norm/mean shared input size mismatch"));
        }
        self.completed = self.kernel.import_dmabuf(fd, bytes, true)?;
        self.primed = false;
        Ok(())
    }

    pub fn attach_shared_output(&mut self, fd: i32, bytes: usize) -> Result<(), XdnaError> {
        if bytes != OUTPUT_BYTES {
            return Err(invalid("final norm/mean shared output size mismatch"));
        }
        self.output = self.kernel.import_dmabuf(fd, bytes, true)?;
        self.primed = false;
        Ok(())
    }

    pub fn upload_params(
        &self,
        output_norm: &[f32],
        epsilon: f32,
    ) -> Result<NpuEmbeddingFinalNormMeanParams, XdnaError> {
        if output_norm.len() != K
            || output_norm.iter().any(|value| !value.is_finite())
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(invalid("invalid final norm/mean parameters"));
        }
        let mut packed = vec![0u8; PARAM_BYTES];
        for (bytes, value) in packed[..EPSILON_OFFSET]
            .chunks_exact_mut(size_of::<f32>())
            .zip(output_norm)
        {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        packed[EPSILON_OFFSET..EPSILON_OFFSET + size_of::<f32>()]
            .copy_from_slice(&epsilon.to_le_bytes());
        let mut buffer = self.kernel.alloc_arg(PARAM_BYTES)?;
        buffer.as_mut_slice().copy_from_slice(&packed);
        self.kernel.sync_to_device(&buffer)?;
        Ok(NpuEmbeddingFinalNormMeanParams { buffer })
    }

    pub fn write_completed_bf16x2(&mut self, bytes: &[u8]) -> Result<(), XdnaError> {
        if bytes.len() != COMPLETED_BYTES {
            return Err(invalid("final norm/mean completed-state size mismatch"));
        }
        self.completed.as_mut_slice().copy_from_slice(bytes);
        self.kernel.sync_to_device(&self.completed)
    }

    pub fn run_shared(
        &mut self,
        params: &NpuEmbeddingFinalNormMeanParams,
    ) -> Result<(), XdnaError> {
        if params.buffer.len() != PARAM_BYTES {
            return Err(invalid("final norm/mean parameter size mismatch"));
        }
        if !self.primed {
            self.dispatch(params)?;
            self.kernel.sync_output(&self.output)?;
            self.primed = true;
        }
        self.dispatch(params)?;
        self.kernel.sync_output(&self.output)
    }

    fn dispatch(&self, params: &NpuEmbeddingFinalNormMeanParams) -> Result<(), XdnaError> {
        self.kernel.dispatch_synced(
            &[&self.completed, &params.buffer, &self.output],
            &[false, false, false],
        )
    }

    pub fn read_pooled_f32(&self) -> Vec<f32> {
        self.output
            .as_slice()
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte f32")))
            .collect()
    }
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_r46_and_embeddinggemma() {
        assert_eq!(NpuEmbeddingFinalNormMean::rows(), 256);
        assert_eq!(NpuEmbeddingFinalNormMean::completed_bytes(), 884_736);
        assert_eq!(NpuEmbeddingFinalNormMean::output_bytes(), 3_072);
        assert!(EPSILON_OFFSET + size_of::<f32>() <= PARAM_BYTES);
    }
}
