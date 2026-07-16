// SPDX-License-Identifier: Apache-2.0

//! R47 resident EmbeddingGemma cross-layer activation preparation.
//!
//! The kernel consumes R46's compensated BF16x2 completed state and writes
//! only the dynamic 6,240-byte prefix of each R34 activation block.  QKV
//! weight scales in the remainder of the shared input buffer stay resident.

use hipfire_primitives::fwht::gen_fwht_signs;

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const M: usize = 256;
const PAD_M: usize = 288;
const K: usize = 768;
const GROUP: usize = 256;
const GROUPS: usize = K / GROUP;
const CORE_ROWS: usize = 4;
const COMPLETED_BYTES: usize = PAD_M * K * 2 * size_of::<u16>();
const PARAM_BYTES: usize = 2 * GROUP * size_of::<f32>() + 2 * GROUP * size_of::<u16>();
const PARAM_RECORD_BYTES: usize = 2 * K * size_of::<u16>();
const PARAM_TOTAL_BYTES: usize = GROUPS * CORE_ROWS * PARAM_RECORD_BYTES;
const R34_INPUT_BYTES: usize = 4 * 45 * 16_384;

pub struct NpuEmbeddingNextLayerPrepW8Params {
    buffer: DeviceBuffer,
}

pub struct NpuEmbeddingNextLayerPrepW8 {
    kernel: NpuKernel,
    batch: usize,
    completed: Option<DeviceBuffer>,
    output: DeviceBuffer,
    output_prefix_offset: usize,
    in_place: bool,
}

impl NpuEmbeddingNextLayerPrepW8 {
    pub fn load_cached(cache: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{cache}/shape.txt")).map_err(XdnaError::Open)?;
        for field in [
            "op=embeddinggemma-next-layer-prep",
            "mode=w8-scaled",
            "k=768",
            "input=shared-completed-bf16x2",
            "output=shared-r34-activation-prefix",
            "prefix-bytes=6240",
        ] {
            if !manifest.lines().any(|line| line == field) {
                return Err(invalid(format!("next-layer prep cache missing {field}")));
            }
        }
        let rows = manifest
            .lines()
            .find_map(|line| line.strip_prefix("m="))
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| invalid("next-layer prep cache missing m="))?;
        if rows == 0 || rows % M != 0 {
            return Err(invalid(format!(
                "next-layer prep m={rows} must be a positive multiple of {M}"
            )));
        }
        let batch = rows / M;
        let kernel = NpuKernel::load(
            &std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?,
            &std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?,
        )?;
        let output_prefix_offset = manifest
            .lines()
            .find_map(|line| line.strip_prefix("output-prefix-offset="))
            .map(str::parse::<usize>)
            .transpose()
            .map_err(|_| invalid("invalid next-layer output prefix offset"))?
            .unwrap_or(0);
        let in_place = manifest
            .lines()
            .any(|line| line == "buffer-mode=in-place-disjoint-prefix-suffix");
        let output_bytes = output_prefix_offset + batch * R34_INPUT_BYTES;
        let output = kernel.alloc_arg(output_bytes)?;
        Ok(Self {
            completed: (!in_place)
                .then(|| kernel.alloc_arg(batch * COMPLETED_BYTES))
                .transpose()?,
            output,
            output_prefix_offset,
            in_place,
            batch,
            kernel,
        })
    }

    pub const fn rows() -> usize {
        M
    }

    pub const fn completed_bytes() -> usize {
        COMPLETED_BYTES
    }

    pub fn output_bytes(&self) -> usize {
        self.output_prefix_offset + self.batch * R34_INPUT_BYTES
    }

    pub const fn batch(&self) -> usize {
        self.batch
    }

    pub const fn loaded_completed_bytes(&self) -> usize {
        self.batch * COMPLETED_BYTES
    }

    pub const fn loaded_canonical_output_bytes(&self) -> usize {
        self.batch * R34_INPUT_BYTES
    }

    pub const fn canonical_output_bytes() -> usize {
        R34_INPUT_BYTES
    }

    pub fn attach_shared(
        &mut self,
        completed_fd: i32,
        completed_bytes: usize,
        output_fd: i32,
        output_bytes: usize,
    ) -> Result<(), XdnaError> {
        if completed_bytes != self.loaded_completed_bytes() || output_bytes != self.output_bytes() {
            return Err(invalid("next-layer prep shared dma-buf size mismatch"));
        }
        if self.in_place {
            if completed_fd != output_fd {
                return Err(invalid(
                    "in-place next-layer prep requires one shared dma-buf",
                ));
            }
            self.output = self.kernel.import_dmabuf(output_fd, output_bytes, true)?;
        } else {
            self.completed = Some(self.kernel.import_dmabuf(
                completed_fd,
                completed_bytes,
                true,
            )?);
            self.output = self.kernel.import_dmabuf(output_fd, output_bytes, true)?;
        }
        Ok(())
    }

    pub fn upload_params(
        &self,
        input_norm: &[f32],
        awq_scale: Option<&[f32]>,
    ) -> Result<NpuEmbeddingNextLayerPrepW8Params, XdnaError> {
        if input_norm.len() != K
            || awq_scale.is_some_and(|scale| scale.len() != K)
            || input_norm.iter().any(|value| !value.is_finite())
            || awq_scale.is_some_and(|scale| {
                scale
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
            })
        {
            return Err(invalid("invalid next-layer norm or AWQ parameters"));
        }
        let signs1 = gen_fwht_signs(42, GROUP);
        let signs2 = gen_fwht_signs(1042, GROUP);
        let mut packed = vec![0u8; PARAM_TOTAL_BYTES];
        for group in 0..GROUPS {
            let mut record = vec![0u8; PARAM_RECORD_BYTES];
            let group_base = group * GROUP;
            for inner in 0..GROUP {
                let norm_offset = inner * size_of::<f32>();
                record[norm_offset..norm_offset + 4]
                    .copy_from_slice(&input_norm[group_base + inner].to_le_bytes());
                let awq_offset = GROUP * size_of::<f32>() + inner * size_of::<f32>();
                record[awq_offset..awq_offset + 4].copy_from_slice(
                    &awq_scale
                        .map_or(1.0, |scale| scale[group_base + inner])
                        .to_le_bytes(),
                );
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
            debug_assert!(PARAM_BYTES <= PARAM_RECORD_BYTES);
            for core_row in 0..CORE_ROWS {
                let target = (group * CORE_ROWS + core_row) * PARAM_RECORD_BYTES;
                packed[target..target + PARAM_RECORD_BYTES].copy_from_slice(&record);
            }
        }
        let mut buffer = self.kernel.alloc_arg(PARAM_TOTAL_BYTES)?;
        buffer.as_mut_slice().copy_from_slice(&packed);
        self.kernel.sync_to_device(&buffer)?;
        Ok(NpuEmbeddingNextLayerPrepW8Params { buffer })
    }

    pub fn run_shared(
        &mut self,
        params: &NpuEmbeddingNextLayerPrepW8Params,
    ) -> Result<(), XdnaError> {
        if params.buffer.len() != PARAM_TOTAL_BYTES {
            return Err(invalid("next-layer prep parameter size mismatch"));
        }
        if self.in_place {
            self.kernel
                .dispatch_synced(&[&self.output, &params.buffer], &[false, false])?;
        } else {
            self.kernel.dispatch_synced(
                &[
                    self.completed
                        .as_ref()
                        .expect("non-in-place prep has completed input"),
                    &params.buffer,
                    &self.output,
                ],
                &[false, false, false],
            )?;
        }
        self.kernel.sync_output(&self.output)
    }

    pub fn write_completed_bf16x2(&mut self, bytes: &[u8]) -> Result<(), XdnaError> {
        if bytes.len() != self.loaded_completed_bytes() {
            return Err(invalid("next-layer completed-state size mismatch"));
        }
        if self.in_place {
            let completed_bytes = self.loaded_completed_bytes();
            self.output.as_mut_slice()[..completed_bytes].copy_from_slice(bytes);
            self.kernel.sync_to_device(&self.output)
        } else {
            let completed = self
                .completed
                .as_mut()
                .expect("non-in-place prep has completed input");
            completed.as_mut_slice().copy_from_slice(bytes);
            self.kernel.sync_to_device(completed)
        }
    }

    pub fn output_prefixes(&self) -> &[u8] {
        &self.output.as_slice()[self.output_prefix_offset
            ..self.output_prefix_offset + self.loaded_canonical_output_bytes()]
    }
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_r34_and_r46() {
        assert_eq!(NpuEmbeddingNextLayerPrepW8::rows(), 256);
        assert_eq!(NpuEmbeddingNextLayerPrepW8::completed_bytes(), 884_736);
        assert_eq!(
            NpuEmbeddingNextLayerPrepW8::canonical_output_bytes(),
            2_949_120
        );
        assert_eq!(PARAM_BYTES, 3_072);
        assert_eq!(PARAM_TOTAL_BYTES, 36_864);
    }

    #[test]
    fn batched_geometry_scales_completed_and_prefix_buffers() {
        assert_eq!(2 * COMPLETED_BYTES, 1_769_472);
        assert_eq!(2 * R34_INPUT_BYTES, 5_898_240);
    }
}
