// SPDX-License-Identifier: Apache-2.0

//! Direct-residual EmbeddingGemma post-FFN RMSNorm tail (R40).

use hipfire_primitives::conv::bf16_bits_to_f32;

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const M: usize = 256;
const PAD_M: usize = 288;
const HIDDEN: usize = 768;
const CORES: usize = 32;
const CORE_TILE_BYTES: usize = 8 * HIDDEN * size_of::<u16>();
const POST_NORM_BYTES: usize = HIDDEN * size_of::<u16>();
const EPSILON_OFFSET: usize = POST_NORM_BYTES;
const PARAM_BYTES: usize = CORES * CORE_TILE_BYTES;
const STATE_BYTES: usize = PAD_M * HIDDEN * size_of::<u16>();

pub struct NpuEmbeddingPostFfnDirectTailParams {
    buffer: DeviceBuffer,
}

pub struct NpuEmbeddingPostFfnDirectTail {
    kernel: NpuKernel,
    residual: DeviceBuffer,
    ffn: DeviceBuffer,
}

impl NpuEmbeddingPostFfnDirectTail {
    pub fn load_cached(cache: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{cache}/shape.txt")).map_err(XdnaError::Open)?;
        for field in [
            "op=embeddinggemma-post-ffn-direct-tail",
            "mode=bf16-resident",
            "m=256",
            "k=768",
            "input=shared-residual-and-y",
            "output=shared-y-overwrite",
        ] {
            if !manifest.lines().any(|line| line == field) {
                return Err(invalid(format!(
                    "direct post-FFN tail cache missing {field}"
                )));
            }
        }
        let kernel = NpuKernel::load(
            &std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?,
            &std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?,
        )?;
        Ok(Self {
            residual: kernel.alloc_arg(STATE_BYTES)?,
            ffn: kernel.alloc_arg(STATE_BYTES)?,
            kernel,
        })
    }

    pub const fn shared_state_bytes() -> usize {
        STATE_BYTES
    }

    pub const fn params_bytes() -> usize {
        PARAM_BYTES
    }

    pub fn attach_shared_state(
        &mut self,
        residual_fd: i32,
        residual_bytes: usize,
        ffn_fd: i32,
        ffn_bytes: usize,
    ) -> Result<(), XdnaError> {
        if residual_bytes != STATE_BYTES || ffn_bytes != STATE_BYTES {
            return Err(invalid("direct post-FFN tail shared dma-buf size mismatch"));
        }
        self.residual = self
            .kernel
            .import_dmabuf(residual_fd, residual_bytes, true)?;
        self.ffn = self.kernel.import_dmabuf(ffn_fd, ffn_bytes, true)?;
        self.kernel.sync_to_device(&self.residual)?;
        self.kernel.sync_to_device(&self.ffn)
    }

    pub fn sync_shared_inputs(&self) -> Result<(), XdnaError> {
        self.kernel.sync_to_device(&self.residual)?;
        self.kernel.sync_to_device(&self.ffn)
    }

    pub fn upload_params(
        &self,
        post_ffn_norm: &[u16],
        epsilon: f32,
    ) -> Result<NpuEmbeddingPostFfnDirectTailParams, XdnaError> {
        if post_ffn_norm.len() != HIDDEN {
            return Err(invalid("direct post-FFN norm vector must have 768 values"));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(invalid(
                "direct post-FFN epsilon must be positive and finite",
            ));
        }
        if post_ffn_norm
            .iter()
            .any(|&bits| !bf16_bits_to_f32(bits).is_finite())
        {
            return Err(invalid("direct post-FFN norm weights must be finite"));
        }
        let mut packed = vec![0u8; PARAM_BYTES];
        for core in 0..CORES {
            let record = &mut packed[core * CORE_TILE_BYTES..(core + 1) * CORE_TILE_BYTES];
            for (hidden, &bits) in post_ffn_norm.iter().enumerate() {
                let offset = hidden * size_of::<u16>();
                record[offset..offset + size_of::<u16>()].copy_from_slice(&bits.to_le_bytes());
            }
            record[EPSILON_OFFSET..EPSILON_OFFSET + size_of::<f32>()]
                .copy_from_slice(&epsilon.to_le_bytes());
        }
        let mut buffer = self.kernel.alloc_arg(PARAM_BYTES)?;
        buffer.as_mut_slice().copy_from_slice(&packed);
        self.kernel.sync_to_device(&buffer)?;
        Ok(NpuEmbeddingPostFfnDirectTailParams { buffer })
    }

    pub fn run_shared(
        &self,
        params: &NpuEmbeddingPostFfnDirectTailParams,
    ) -> Result<(), XdnaError> {
        if params.buffer.len() != PARAM_BYTES {
            return Err(invalid("direct post-FFN tail parameter size mismatch"));
        }
        self.kernel.dispatch_synced(
            &[&self.residual, &self.ffn, &params.buffer, &self.ffn],
            &[false, false, false, false],
        )?;
        self.kernel.sync_output(&self.ffn)
    }

    pub fn read_output_f32(&self) -> Result<Vec<f32>, XdnaError> {
        Ok(self.ffn.as_slice()[..M * HIDDEN * size_of::<u16>()]
            .chunks_exact(size_of::<u16>())
            .map(|word| bf16_bits_to_f32(u16::from_le_bytes([word[0], word[1]])))
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
    fn direct_tail_geometry_matches_r40() {
        assert_eq!(NpuEmbeddingPostFfnDirectTail::shared_state_bytes(), 442_368);
        assert_eq!(NpuEmbeddingPostFfnDirectTail::params_bytes(), 393_216);
        assert!(EPSILON_OFFSET + size_of::<f32>() <= CORE_TILE_BYTES);
    }
}
