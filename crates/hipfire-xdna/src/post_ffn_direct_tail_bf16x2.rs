// SPDX-License-Identifier: Apache-2.0

//! Precision-preserving EmbeddingGemma post-FFN RMSNorm tail.
//!
//! The FFN input is compensated BF16x2 (`high + low`) in token-major rows,
//! while residual input remains canonical BF16. Cached kernels may return the
//! completed state as either canonical BF16 or compensated token-major BF16x2.

use hipfire_primitives::conv::bf16_bits_to_f32;

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const M: usize = 256;
const PAD_M: usize = 288;
const HIDDEN: usize = 768;
const CORES: usize = 32;
const PARAM_RECORD_BYTES: usize = 2 * 3 * HIDDEN * size_of::<u16>();
const SPLIT_PARAM_RECORD_BYTES: usize = 2 * 2 * HIDDEN * size_of::<u16>();
const POST_NORM_BYTES: usize = HIDDEN * size_of::<u16>();
const EPSILON_OFFSET: usize = POST_NORM_BYTES;
const RESIDUAL_BYTES: usize = PAD_M * HIDDEN * size_of::<u16>();
const COMBINED_BYTES: usize = PAD_M * HIDDEN * 3 * size_of::<u16>();
const PARAM_BYTES: usize = CORES * PARAM_RECORD_BYTES;
const SPLIT_PARAM_BYTES: usize = CORES * SPLIT_PARAM_RECORD_BYTES;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompletedOutputEncoding {
    Bf16,
    Bf16x2,
}

impl CompletedOutputEncoding {
    const fn bytes(self) -> usize {
        match self {
            Self::Bf16 => RESIDUAL_BYTES,
            Self::Bf16x2 => 2 * RESIDUAL_BYTES,
        }
    }
}

pub struct NpuEmbeddingPostFfnDirectTailBf16x2Params {
    buffer: DeviceBuffer,
}

pub struct NpuEmbeddingPostFfnDirectTailBf16x2 {
    kernel: NpuKernel,
    combined: DeviceBuffer,
    residual: Option<DeviceBuffer>,
    output: DeviceBuffer,
    output_encoding: CompletedOutputEncoding,
    split_residual: bool,
    param_record_bytes: usize,
    param_bytes: usize,
}

impl NpuEmbeddingPostFfnDirectTailBf16x2 {
    pub fn load_cached(cache: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{cache}/shape.txt")).map_err(XdnaError::Open)?;
        for field in [
            "op=embeddinggemma-post-ffn-direct-tail",
            "mode=bf16x2-resident",
            "m=256",
            "k=768",
        ] {
            if !manifest.lines().any(|line| line == field) {
                return Err(invalid(format!(
                    "compensated post-FFN tail cache missing {field}"
                )));
            }
        }
        let split_residual = if manifest
            .lines()
            .any(|line| line == "input=shared-y-bf16x2-and-split-x-bf16")
        {
            true
        } else if manifest
            .lines()
            .any(|line| line == "input=shared-y-bf16x2-and-residual-bf16")
        {
            false
        } else {
            return Err(invalid(
                "compensated post-FFN tail cache missing supported input encoding",
            ));
        };
        let output_encoding = if manifest
            .lines()
            .any(|line| line == "output=shared-completed-bf16x2")
        {
            CompletedOutputEncoding::Bf16x2
        } else if manifest
            .lines()
            .any(|line| line == "output=shared-completed-bf16")
        {
            CompletedOutputEncoding::Bf16
        } else {
            return Err(invalid(
                "compensated post-FFN tail cache missing supported output encoding",
            ));
        };
        let kernel = NpuKernel::load(
            &std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?,
            &std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?,
        )?;
        Ok(Self {
            combined: kernel.alloc_arg(COMBINED_BYTES)?,
            residual: split_residual
                .then(|| kernel.alloc_arg(RESIDUAL_BYTES))
                .transpose()?,
            output: kernel.alloc_arg(output_encoding.bytes())?,
            kernel,
            output_encoding,
            split_residual,
            param_record_bytes: if split_residual {
                SPLIT_PARAM_RECORD_BYTES
            } else {
                PARAM_RECORD_BYTES
            },
            param_bytes: if split_residual {
                SPLIT_PARAM_BYTES
            } else {
                PARAM_BYTES
            },
        })
    }

    pub const fn residual_bytes() -> usize {
        RESIDUAL_BYTES
    }

    pub const fn completed_bf16x2_bytes() -> usize {
        CompletedOutputEncoding::Bf16x2.bytes()
    }

    pub const fn output_bytes(&self) -> usize {
        self.output_encoding.bytes()
    }

    pub const fn combined_bytes() -> usize {
        COMBINED_BYTES
    }

    pub const fn params_bytes() -> usize {
        PARAM_BYTES
    }

    pub const fn consumes_split_x(&self) -> bool {
        self.split_residual
    }

    pub fn attach_shared_state(
        &mut self,
        combined_fd: i32,
        combined_bytes: usize,
        output_fd: i32,
        output_bytes: usize,
    ) -> Result<(), XdnaError> {
        if self.split_residual {
            return Err(invalid(
                "split-X post-FFN tail requires a separate residual dma-buf",
            ));
        }
        if combined_bytes != COMBINED_BYTES || output_bytes != self.output_bytes() {
            return Err(invalid(
                "compensated post-FFN tail shared dma-buf size mismatch",
            ));
        }
        self.combined = self
            .kernel
            .import_dmabuf(combined_fd, combined_bytes, true)?;
        self.output = self.kernel.import_dmabuf(output_fd, output_bytes, true)?;
        self.kernel.sync_to_device(&self.combined)?;
        self.kernel.sync_to_device(&self.output)
    }

    pub fn attach_shared_split_state(
        &mut self,
        combined_fd: i32,
        combined_bytes: usize,
        residual_fd: i32,
        residual_bytes: usize,
        output_fd: i32,
        output_bytes: usize,
    ) -> Result<(), XdnaError> {
        if !self.split_residual
            || combined_bytes != COMBINED_BYTES
            || residual_bytes < RESIDUAL_BYTES
            || output_bytes != self.output_bytes()
        {
            return Err(invalid(
                "split-X post-FFN tail shared dma-buf size/mode mismatch",
            ));
        }
        self.combined = self
            .kernel
            .import_dmabuf(combined_fd, combined_bytes, true)?;
        self.residual = Some(
            self.kernel
                .import_dmabuf(residual_fd, residual_bytes, true)?,
        );
        self.output = self.kernel.import_dmabuf(output_fd, output_bytes, true)?;
        self.kernel.sync_to_device(&self.output)
    }

    pub fn sync_shared_inputs(&self) -> Result<(), XdnaError> {
        self.kernel.sync_to_device(&self.combined)
    }

    pub fn upload_params(
        &self,
        post_ffn_norm: &[u16],
        epsilon: f32,
    ) -> Result<NpuEmbeddingPostFfnDirectTailBf16x2Params, XdnaError> {
        if post_ffn_norm.len() != HIDDEN {
            return Err(invalid("compensated post-FFN norm must have 768 values"));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(invalid(
                "compensated post-FFN epsilon must be positive and finite",
            ));
        }
        if post_ffn_norm
            .iter()
            .any(|&bits| !bf16_bits_to_f32(bits).is_finite())
        {
            return Err(invalid("compensated post-FFN norm weights must be finite"));
        }
        let mut packed = vec![0u8; self.param_bytes];
        for core in 0..CORES {
            let record =
                &mut packed[core * self.param_record_bytes..(core + 1) * self.param_record_bytes];
            for (hidden, &bits) in post_ffn_norm.iter().enumerate() {
                let offset = hidden * size_of::<u16>();
                record[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
            }
            record[EPSILON_OFFSET..EPSILON_OFFSET + size_of::<f32>()]
                .copy_from_slice(&epsilon.to_le_bytes());
        }
        let mut buffer = self.kernel.alloc_arg(self.param_bytes)?;
        buffer.as_mut_slice().copy_from_slice(&packed);
        self.kernel.sync_to_device(&buffer)?;
        Ok(NpuEmbeddingPostFfnDirectTailBf16x2Params { buffer })
    }

    pub fn run_shared(
        &self,
        params: &NpuEmbeddingPostFfnDirectTailBf16x2Params,
    ) -> Result<(), XdnaError> {
        if params.buffer.len() != self.param_bytes {
            return Err(invalid("compensated post-FFN tail parameter size mismatch"));
        }
        if self.split_residual {
            self.kernel.dispatch_synced(
                &[
                    &self.combined,
                    self.residual
                        .as_ref()
                        .expect("split residual allocated with split mode"),
                    &params.buffer,
                    &self.output,
                ],
                &[false, false, false, false],
            )?;
        } else {
            self.kernel.dispatch_synced(
                &[&self.combined, &params.buffer, &self.output],
                &[false, false, false],
            )?;
        }
        self.kernel.sync_output(&self.output)
    }

    pub fn read_output_f32(&self) -> Result<Vec<f32>, XdnaError> {
        let output = self.output.as_slice();
        match self.output_encoding {
            CompletedOutputEncoding::Bf16 => Ok(decode_bf16(output, M * HIDDEN)),
            CompletedOutputEncoding::Bf16x2 => Ok(decode_token_major_bf16x2(output, M, HIDDEN)),
        }
    }
}

fn decode_bf16(bytes: &[u8], elements: usize) -> Vec<f32> {
    bytes[..elements * size_of::<u16>()]
        .chunks_exact(size_of::<u16>())
        .map(|word| bf16_bits_to_f32(u16::from_le_bytes([word[0], word[1]])))
        .collect()
}

fn decode_token_major_bf16x2(bytes: &[u8], rows: usize, width: usize) -> Vec<f32> {
    let words = bytes
        .chunks_exact(size_of::<u16>())
        .map(|word| bf16_bits_to_f32(u16::from_le_bytes([word[0], word[1]])))
        .collect::<Vec<_>>();
    let mut output = Vec::with_capacity(rows * width);
    for row in 0..rows {
        let base = row * 2 * width;
        output.extend(
            words[base..base + width]
                .iter()
                .zip(&words[base + width..base + 2 * width])
                .map(|(&high, &low)| high + low),
        );
    }
    output
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compensated_tail_geometry_matches_r41() {
        assert_eq!(
            NpuEmbeddingPostFfnDirectTailBf16x2::residual_bytes(),
            442_368
        );
        assert_eq!(
            NpuEmbeddingPostFfnDirectTailBf16x2::combined_bytes(),
            1_327_104
        );
        assert_eq!(NpuEmbeddingPostFfnDirectTailBf16x2::params_bytes(), 294_912);
        assert_eq!(
            NpuEmbeddingPostFfnDirectTailBf16x2::completed_bf16x2_bytes(),
            884_736
        );
        assert!(EPSILON_OFFSET + size_of::<f32>() <= PARAM_RECORD_BYTES);
    }

    #[test]
    fn decodes_token_major_completed_bf16x2() {
        let values = [1.0f32, -2.0, 0.25, -0.5, 3.0, -4.0, 0.75, -1.0];
        let words = values
            .into_iter()
            .map(|value| (value.to_bits() >> 16) as u16)
            .collect::<Vec<_>>();
        let bytes = words
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            decode_token_major_bf16x2(&bytes, 2, 2),
            vec![1.25, -2.5, 3.75, -5.0]
        );
    }
}
