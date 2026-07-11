//! Resident EmbeddingGemma post-FFN RMSNorm and residual tail.

use hipfire_primitives::conv::bf16_bits_to_f32;

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const M: usize = 256;
const PAD_M: usize = 288;
const HIDDEN: usize = 768;
const CORES: usize = 32;
const CORE_TILE_BYTES: usize = 8 * HIDDEN * size_of::<u16>();
const PRE_RECIP_BYTES: usize = HIDDEN * size_of::<f32>();
const POST_NORM_BYTES: usize = HIDDEN * size_of::<u16>();
const EPSILON_OFFSET: usize = PRE_RECIP_BYTES + POST_NORM_BYTES;
const PARAM_BYTES: usize = CORES * CORE_TILE_BYTES;
const H_BACKING_BYTES: usize = 5_111_808;
const Y_BACKING_BYTES: usize = PAD_M * HIDDEN * size_of::<u16>();

pub struct NpuEmbeddingPostFfnTailParams {
    buffer: DeviceBuffer,
}

pub struct NpuEmbeddingPostFfnTail {
    kernel: NpuKernel,
    hidden: DeviceBuffer,
    ffn: DeviceBuffer,
}

impl NpuEmbeddingPostFfnTail {
    pub fn load_cached(cache: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{cache}/shape.txt")).map_err(XdnaError::Open)?;
        for field in [
            "op=embeddinggemma-post-ffn-tail",
            "mode=bf16-resident",
            "m=256",
            "k=768",
            "input=shared-h-and-y",
            "output=shared-y-overwrite",
            "state=pre-ffn-inverse-f32",
        ] {
            if !manifest.lines().any(|line| line == field) {
                return Err(invalid(format!("post-FFN tail cache missing {field}")));
            }
        }
        let kernel = NpuKernel::load(
            &std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?,
            &std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?,
        )?;
        let hidden = kernel.alloc_arg(H_BACKING_BYTES)?;
        let ffn = kernel.alloc_arg(Y_BACKING_BYTES)?;
        Ok(Self {
            kernel,
            hidden,
            ffn,
        })
    }

    pub const fn hidden_backing_bytes() -> usize {
        H_BACKING_BYTES
    }

    pub const fn shared_state_bytes() -> usize {
        Y_BACKING_BYTES
    }

    pub const fn params_bytes() -> usize {
        PARAM_BYTES
    }

    pub fn attach_shared_state(
        &mut self,
        hidden_fd: i32,
        hidden_bytes: usize,
        ffn_fd: i32,
        ffn_bytes: usize,
    ) -> Result<(), XdnaError> {
        if hidden_bytes < H_BACKING_BYTES || ffn_bytes != Y_BACKING_BYTES {
            return Err(invalid("post-FFN tail shared dma-buf size mismatch"));
        }
        self.hidden = self.kernel.import_dmabuf(hidden_fd, hidden_bytes, true)?;
        self.ffn = self.kernel.import_dmabuf(ffn_fd, ffn_bytes, true)?;
        self.sync_shared_inputs()?;
        Ok(())
    }

    pub fn sync_shared_inputs(&self) -> Result<(), XdnaError> {
        self.kernel.sync_to_device(&self.hidden)?;
        self.kernel.sync_to_device(&self.ffn)
    }

    pub fn upload_params(
        &self,
        pre_ffn_norm: &[u16],
        post_ffn_norm: &[u16],
        epsilon: f32,
    ) -> Result<NpuEmbeddingPostFfnTailParams, XdnaError> {
        if pre_ffn_norm.len() != HIDDEN || post_ffn_norm.len() != HIDDEN {
            return Err(invalid("post-FFN tail norm vectors must have 768 values"));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(invalid("post-FFN tail epsilon must be positive and finite"));
        }
        let mut packed = vec![0u8; PARAM_BYTES];
        for core in 0..CORES {
            let record = &mut packed[core * CORE_TILE_BYTES..(core + 1) * CORE_TILE_BYTES];
            for (hidden, &bits) in pre_ffn_norm.iter().enumerate() {
                let value = bf16_bits_to_f32(bits);
                if !value.is_finite() || value == 0.0 {
                    return Err(invalid(
                        "post-FFN tail pre-FFN norm weights must be finite and nonzero",
                    ));
                }
                write_f32(record, hidden * size_of::<f32>(), value.recip());
            }
            for (hidden, &bits) in post_ffn_norm.iter().enumerate() {
                let offset = PRE_RECIP_BYTES + hidden * size_of::<u16>();
                record[offset..offset + size_of::<u16>()].copy_from_slice(&bits.to_le_bytes());
            }
            write_f32(record, EPSILON_OFFSET, epsilon);
        }
        let mut buffer = self.kernel.alloc_arg(packed.len())?;
        buffer.as_mut_slice().copy_from_slice(&packed);
        self.kernel.sync_to_device(&buffer)?;
        Ok(NpuEmbeddingPostFfnTailParams { buffer })
    }

    pub fn run_shared(&self, params: &NpuEmbeddingPostFfnTailParams) -> Result<(), XdnaError> {
        if params.buffer.len() != PARAM_BYTES {
            return Err(invalid("post-FFN tail parameter buffer size mismatch"));
        }
        self.kernel.dispatch_synced(
            &[&self.hidden, &self.ffn, &params.buffer, &self.ffn],
            &[false, false, false, false],
        )?;
        self.kernel.sync_output(&self.ffn)
    }

    pub fn read_output_f32(&self) -> Result<Vec<f32>, XdnaError> {
        let bytes = self.ffn.as_slice();
        if bytes.len() < M * HIDDEN * size_of::<u16>() {
            return Err(invalid("post-FFN tail output buffer is too small"));
        }
        Ok(bytes[..M * HIDDEN * size_of::<u16>()]
            .chunks_exact(size_of::<u16>())
            .map(|word| bf16_bits_to_f32(u16::from_le_bytes([word[0], word[1]])))
            .collect())
    }
}

fn write_f32(destination: &mut [u8], offset: usize, value: f32) {
    destination[offset..offset + size_of::<f32>()].copy_from_slice(&value.to_le_bytes());
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_tail_geometry_matches_r39() {
        assert_eq!(NpuEmbeddingPostFfnTail::hidden_backing_bytes(), 5_111_808);
        assert_eq!(NpuEmbeddingPostFfnTail::shared_state_bytes(), 442_368);
        assert_eq!(NpuEmbeddingPostFfnTail::params_bytes(), 393_216);
        assert!(EPSILON_OFFSET + size_of::<f32>() <= CORE_TILE_BYTES);
    }
}
