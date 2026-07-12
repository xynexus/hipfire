// SPDX-License-Identifier: Apache-2.0

//! Resident EmbeddingGemma Dense(768->3072->768) heads and final L2 norm.

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const INPUT: usize = 768;
const INTERMEDIATE: usize = 3072;
const OUTPUT: usize = 768;
const INPUT_BYTES: usize = INPUT * size_of::<f32>();
const W0_WEIGHT_BYTES: usize = INTERMEDIATE * INPUT * size_of::<u16>();
const W0_BYTES: usize = INPUT_BYTES + W0_WEIGHT_BYTES;
const W1_BYTES: usize = OUTPUT * INTERMEDIATE * size_of::<u16>();
const OUTPUT_BYTES: usize = OUTPUT * size_of::<f32>();

pub struct NpuEmbeddingDenseL2 {
    kernel: NpuKernel,
    input_and_w0: DeviceBuffer,
    w1: DeviceBuffer,
    output: DeviceBuffer,
    primed: bool,
}

impl NpuEmbeddingDenseL2 {
    pub fn load_cached(cache: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{cache}/shape.txt")).map_err(XdnaError::Open)?;
        for field in [
            "op=embeddinggemma-dense-l2",
            "input=768",
            "intermediate=3072",
            "output=768",
            "weights=bf16",
            "activation=identity",
            "normalize=l2",
        ] {
            if !manifest.lines().any(|line| line == field) {
                return Err(invalid(format!("Dense/L2 cache missing {field}")));
            }
        }
        let kernel = NpuKernel::load(
            &std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?,
            &std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?,
        )?;
        Ok(Self {
            input_and_w0: kernel.alloc_arg(W0_BYTES)?,
            w1: kernel.alloc_arg(W1_BYTES)?,
            output: kernel.alloc_arg(OUTPUT_BYTES)?,
            kernel,
            primed: false,
        })
    }

    pub const fn input_bytes() -> usize {
        INPUT_BYTES
    }

    pub const fn input_and_w0_bytes() -> usize {
        W0_BYTES
    }

    pub fn attach_shared_input_and_w0(&mut self, fd: i32, bytes: usize) -> Result<(), XdnaError> {
        if bytes != W0_BYTES {
            return Err(invalid("Dense/L2 shared input/weight size mismatch"));
        }
        self.input_and_w0 = self.kernel.import_dmabuf(fd, bytes, true)?;
        self.primed = false;
        Ok(())
    }

    pub fn upload_weights(&mut self, head0: &[f32], head1: &[f32]) -> Result<(), XdnaError> {
        if head0.len() != INTERMEDIATE * INPUT || head1.len() != OUTPUT * INTERMEDIATE {
            return Err(invalid("Dense/L2 weight geometry mismatch"));
        }
        encode_bf16(&mut self.input_and_w0.as_mut_slice()[INPUT_BYTES..], head0);
        encode_bf16(self.w1.as_mut_slice(), head1);
        self.kernel.sync_to_device(&self.input_and_w0)?;
        self.kernel.sync_to_device(&self.w1)
    }

    pub fn write_input(&mut self, input: &[f32]) -> Result<(), XdnaError> {
        if input.len() != INPUT {
            return Err(invalid("Dense/L2 input geometry mismatch"));
        }
        for (bytes, value) in self.input_and_w0.as_mut_slice()[..INPUT_BYTES]
            .chunks_exact_mut(size_of::<f32>())
            .zip(input)
        {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        self.kernel
            .sync_to_device_prefix(&self.input_and_w0, INPUT_BYTES)
    }

    pub fn run_shared(&mut self) -> Result<(), XdnaError> {
        if !self.primed {
            self.dispatch()?;
            self.kernel.sync_output(&self.output)?;
            self.primed = true;
        }
        self.dispatch()?;
        self.kernel.sync_output(&self.output)
    }

    pub fn sync_shared_input(&self) -> Result<(), XdnaError> {
        self.kernel
            .sync_to_device_prefix(&self.input_and_w0, INPUT_BYTES)
    }

    pub fn read_embedding_f32(&self) -> Vec<f32> {
        self.output
            .as_slice()
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte f32")))
            .collect()
    }

    fn dispatch(&self) -> Result<(), XdnaError> {
        self.kernel.dispatch_synced(
            &[&self.input_and_w0, &self.w1, &self.output],
            &[false, false, false],
        )
    }
}

fn encode_bf16(destination: &mut [u8], values: &[f32]) {
    for (bytes, &value) in destination.chunks_exact_mut(size_of::<u16>()).zip(values) {
        let bits = value.to_bits();
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
        bytes.copy_from_slice(&((rounded >> 16) as u16).to_le_bytes());
    }
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_embeddinggemma_dense_heads() {
        assert_eq!(NpuEmbeddingDenseL2::input_bytes(), 3_072);
        assert_eq!(NpuEmbeddingDenseL2::input_and_w0_bytes(), 4_721_664);
        assert_eq!(W1_BYTES, 4_718_592);
        assert_eq!(OUTPUT_BYTES, 3_072);
    }
}
