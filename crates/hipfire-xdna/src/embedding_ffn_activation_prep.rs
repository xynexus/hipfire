// SPDX-License-Identifier: Apache-2.0

//! R93 canonical BF16 pre-FFN state to the resident R25 W4 activation ABI.

use hipfire_primitives::fwht::gen_fwht_signs;

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const M: usize = 256;
const PAD_M: usize = 288;
const K: usize = 768;
const GROUP: usize = 256;
const GROUPS: usize = K / GROUP;
const CORE_ROWS: usize = 4;
const INPUT_BYTES: usize = PAD_M * K * size_of::<u16>();
const PARAM_RECORD_BYTES: usize = 2 * K * size_of::<u16>();
const PARAM_TOTAL_BYTES: usize = GROUPS * CORE_ROWS * PARAM_RECORD_BYTES;
const OUTPUT_BYTES: usize = 4 * 27 * 6_656;

pub struct NpuEmbeddingFfnActivationPrepW4Params {
    buffer: DeviceBuffer,
}

pub struct NpuEmbeddingFfnActivationPrepW4 {
    kernel: NpuKernel,
    input: DeviceBuffer,
    output: DeviceBuffer,
    r25_vector_params: bool,
}

impl NpuEmbeddingFfnActivationPrepW4 {
    pub fn load_cached(cache: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{cache}/shape.txt")).map_err(XdnaError::Open)?;
        for field in [
            "op=embeddinggemma-ffn-activation-prep",
            "mode=w4-scaled",
            "m=256",
            "k=768",
            "input=canonical-bf16-pre-ffn-norm",
            "output=resident-r25-w4-activation",
            "block-bytes=6656",
            "prefix-bytes=6240",
            "replicas=3",
        ] {
            if !manifest.lines().any(|line| line == field) {
                return Err(invalid(format!(
                    "FFN activation-prep cache missing {field}"
                )));
            }
        }
        let kernel = NpuKernel::load(
            &std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?,
            &std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?,
        )?;
        let mut input = kernel.alloc_arg(INPUT_BYTES)?;
        input.as_mut_slice().fill(0);
        kernel.sync_to_device(&input)?;
        let mut output = kernel.alloc_arg(OUTPUT_BYTES)?;
        output.as_mut_slice().fill(0);
        kernel.sync_to_device(&output)?;
        Ok(Self {
            kernel,
            input,
            output,
            r25_vector_params: manifest
                .lines()
                .any(|line| line == "prep=vector-r25-params"),
        })
    }

    pub const fn rows() -> usize {
        M
    }

    pub const fn input_bytes() -> usize {
        INPUT_BYTES
    }

    pub const fn output_bytes() -> usize {
        OUTPUT_BYTES
    }

    pub fn attach_shared(
        &mut self,
        input_fd: i32,
        input_bytes: usize,
        output_fd: i32,
        output_bytes: usize,
    ) -> Result<(), XdnaError> {
        if input_bytes != INPUT_BYTES || output_bytes != OUTPUT_BYTES {
            return Err(invalid("FFN activation-prep shared dma-buf size mismatch"));
        }
        self.input = self.kernel.import_dmabuf(input_fd, input_bytes, true)?;
        self.output = self.kernel.import_dmabuf(output_fd, output_bytes, true)?;
        Ok(())
    }

    pub fn upload_params(
        &self,
        awq_scale: Option<&[f32]>,
    ) -> Result<NpuEmbeddingFfnActivationPrepW4Params, XdnaError> {
        if awq_scale.is_some_and(|scale| {
            scale.len() != K
                || scale
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
        }) {
            return Err(invalid("invalid FFN activation-prep AWQ parameters"));
        }
        let signs1 = gen_fwht_signs(42, GROUP);
        let signs2 = gen_fwht_signs(1042, GROUP);
        let mut packed = vec![0u8; PARAM_TOTAL_BYTES];
        for group in 0..GROUPS {
            let mut record = vec![0u8; PARAM_RECORD_BYTES];
            let group_base = group * GROUP;
            for inner in 0..GROUP {
                let awq = awq_scale.map_or(1.0, |scale| scale[group_base + inner]);
                if self.r25_vector_params {
                    let awq_offset = inner * size_of::<f32>();
                    record[awq_offset..awq_offset + 4].copy_from_slice(&awq.to_le_bytes());
                    let sign1_offset = (GROUP + inner) * size_of::<f32>();
                    record[sign1_offset..sign1_offset + 4]
                        .copy_from_slice(&signs1[inner].to_le_bytes());
                    let sign2_offset = (2 * GROUP + inner) * size_of::<f32>();
                    record[sign2_offset..sign2_offset + 4]
                        .copy_from_slice(&signs2[inner].to_le_bytes());
                } else {
                    let norm_offset = inner * size_of::<f32>();
                    record[norm_offset..norm_offset + 4].copy_from_slice(&1.0f32.to_le_bytes());
                    let awq_offset = GROUP * size_of::<f32>() + inner * size_of::<f32>();
                    record[awq_offset..awq_offset + 4].copy_from_slice(&awq.to_le_bytes());
                    let sign1 = if signs1[inner] > 0.0 {
                        0x3f80u16
                    } else {
                        0xbf80u16
                    };
                    let sign2 = if signs2[inner] > 0.0 {
                        0x3f80u16
                    } else {
                        0xbf80u16
                    };
                    let sign1_offset = 2 * GROUP * size_of::<f32>() + inner * 2;
                    record[sign1_offset..sign1_offset + 2].copy_from_slice(&sign1.to_le_bytes());
                    let sign2_offset = 2 * GROUP * size_of::<f32>() + GROUP * 2 + inner * 2;
                    record[sign2_offset..sign2_offset + 2].copy_from_slice(&sign2.to_le_bytes());
                }
            }
            for core_row in 0..CORE_ROWS {
                let target = (group * CORE_ROWS + core_row) * PARAM_RECORD_BYTES;
                packed[target..target + PARAM_RECORD_BYTES].copy_from_slice(&record);
            }
        }
        let mut buffer = self.kernel.alloc_arg(PARAM_TOTAL_BYTES)?;
        buffer.as_mut_slice().copy_from_slice(&packed);
        self.kernel.sync_to_device(&buffer)?;
        Ok(NpuEmbeddingFfnActivationPrepW4Params { buffer })
    }

    pub fn write_input_bf16(&mut self, values: &[u16]) -> Result<(), XdnaError> {
        if values.len() != M * K {
            return Err(invalid("FFN activation-prep BF16 input size mismatch"));
        }
        for (target, value) in self
            .input
            .as_mut_slice()
            .chunks_exact_mut(size_of::<u16>())
            .zip(values)
        {
            target.copy_from_slice(&value.to_le_bytes());
        }
        self.kernel.sync_to_device(&self.input)
    }

    pub fn run_shared(
        &mut self,
        params: &NpuEmbeddingFfnActivationPrepW4Params,
    ) -> Result<(), XdnaError> {
        if params.buffer.len() != PARAM_TOTAL_BYTES {
            return Err(invalid("FFN activation-prep parameter size mismatch"));
        }
        self.kernel.dispatch_synced(
            &[&self.input, &params.buffer, &self.output],
            &[false, false, false],
        )?;
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
    fn geometry_matches_r25() {
        assert_eq!(NpuEmbeddingFfnActivationPrepW4::rows(), 256);
        assert_eq!(NpuEmbeddingFfnActivationPrepW4::input_bytes(), 442_368);
        assert_eq!(NpuEmbeddingFfnActivationPrepW4::output_bytes(), 718_848);
        assert_eq!(PARAM_RECORD_BYTES, 3_072);
        assert_eq!(PARAM_TOTAL_BYTES, 36_864);
    }
}
