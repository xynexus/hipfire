// SPDX-License-Identifier: Apache-2.0

//! Full-array BF16 output projection consuming R31's packed attention layout.

use hipfire_primitives::conv::f32_to_bf16_bits;
use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};

use crate::{DeviceBuffer, NpuKernel, OpusPackedMatrix, XdnaError};

const M: usize = 256;
const K: usize = 768;
const N: usize = 768;
const GROUP: usize = 256;
const GROUPS: usize = 3;
const COLS: usize = 8;
const SLICES: usize = 3;
const BLOCK: usize = 16384;
const INPUT_BYTES: usize = M * K * size_of::<u16>();
const WEIGHT_BYTES: usize = COLS * SLICES * GROUPS * BLOCK;
const OUTPUT_BYTES: usize = M * N * size_of::<f32>();
const MAX_CONTEXT_COMMANDS: usize = 1_000;

pub struct NpuAttentionOutputBf16Weights {
    weights: DeviceBuffer,
    input: Option<DeviceBuffer>,
}

pub struct NpuAttentionOutputBf16 {
    kernel: NpuKernel,
    input: DeviceBuffer,
    output: DeviceBuffer,
    primed: bool,
    context_commands: usize,
}

impl NpuAttentionOutputBf16 {
    pub fn load_cached(cache: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{cache}/shape.txt")).map_err(XdnaError::Open)?;
        for required in [
            "op=attention-output-projection",
            "mode=bf16",
            "m=256",
            "k=768",
            "n=768",
            "input=projection-packed-bf16",
            "output=token-major-f32",
        ] {
            if !manifest.lines().any(|line| line == required) {
                return Err(invalid(format!(
                    "attention output cache missing {required}"
                )));
            }
        }
        let kernel = NpuKernel::load(
            &std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?,
            &std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?,
        )?;
        let input = kernel.alloc_arg(INPUT_BYTES)?;
        let output = kernel.alloc_arg(OUTPUT_BYTES)?;
        Ok(Self {
            kernel,
            input,
            output,
            primed: false,
            context_commands: 0,
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

    pub const fn weight_bytes() -> usize {
        WEIGHT_BYTES
    }

    pub fn attach_shared_input(&mut self, fd: i32, bytes: usize) -> Result<(), XdnaError> {
        if bytes < INPUT_BYTES {
            return Err(invalid("shared attention output input is too small"));
        }
        self.input = self.kernel.import_dmabuf(fd, bytes, true)?;
        self.primed = false;
        self.context_commands = 0;
        Ok(())
    }

    pub fn attach_shared_output(&mut self, fd: i32, bytes: usize) -> Result<(), XdnaError> {
        if bytes < OUTPUT_BYTES {
            return Err(invalid("shared attention output destination is too small"));
        }
        self.output = self.kernel.import_dmabuf(fd, bytes, true)?;
        self.primed = false;
        self.context_commands = 0;
        Ok(())
    }

    pub fn set_input(&mut self, packed: &[u8]) -> Result<(), XdnaError> {
        if packed.len() != INPUT_BYTES {
            return Err(invalid("packed attention output input size mismatch"));
        }
        self.input.as_mut_slice()[..INPUT_BYTES].copy_from_slice(packed);
        Ok(())
    }

    pub fn upload_weights(
        &self,
        matrix: &OpusPackedMatrix,
    ) -> Result<NpuAttentionOutputBf16Weights, XdnaError> {
        validate_matrix(matrix)?;
        self.upload_bf16(&dense_effective_bf16(matrix))
    }

    pub fn upload_bf16(&self, matrix: &[u16]) -> Result<NpuAttentionOutputBf16Weights, XdnaError> {
        if matrix.len() != K * N {
            return Err(invalid("BF16 attention output matrix size mismatch"));
        }
        let packed = pack_weights(matrix);
        let mut weights = self.kernel.alloc_arg(packed.len())?;
        weights.as_mut_slice().copy_from_slice(&packed);
        self.kernel.sync_to_device(&weights)?;
        Ok(NpuAttentionOutputBf16Weights {
            weights,
            input: None,
        })
    }

    /// Bind a layer's projection-packed attention prefix as this layer's input.
    pub fn attach_shared_layer_input(
        &self,
        weights: &mut NpuAttentionOutputBf16Weights,
        fd: i32,
        bytes: usize,
    ) -> Result<(), XdnaError> {
        if bytes < INPUT_BYTES {
            return Err(invalid("shared attention output layer input is too small"));
        }
        weights.input = Some(self.kernel.import_dmabuf(fd, bytes, true)?);
        Ok(())
    }

    pub fn run(&mut self, weights: &NpuAttentionOutputBf16Weights) -> Result<(), XdnaError> {
        let input = weights.input.as_ref().unwrap_or(&self.input);
        if weights.weights.len() != WEIGHT_BYTES
            || input.len() < INPUT_BYTES
            || self.output.len() < OUTPUT_BYTES
        {
            return Err(invalid("attention output argument geometry mismatch"));
        }
        if self.context_commands >= MAX_CONTEXT_COMMANDS {
            self.kernel.recreate_hwctx()?;
            self.primed = false;
            self.context_commands = 0;
        }
        if !self.primed {
            self.kernel.dispatch_synced(
                &[input, &weights.weights, &self.output],
                &[true, false, false],
            )?;
            self.kernel.sync_output(&self.output)?;
            self.output.as_mut_slice()[..OUTPUT_BYTES].fill(0);
            self.kernel.sync_to_device(&self.output)?;
            self.context_commands += 1;
            self.primed = true;
        }
        self.kernel.dispatch_synced(
            &[input, &weights.weights, &self.output],
            &[false, false, false],
        )?;
        self.context_commands += 1;
        Ok(())
    }

    pub fn read_output_f32(&self) -> Result<Vec<f32>, XdnaError> {
        self.kernel.sync_output(&self.output)?;
        Ok(self.output.as_slice()[..OUTPUT_BYTES]
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect())
    }
}

fn validate_matrix(matrix: &OpusPackedMatrix) -> Result<(), XdnaError> {
    if matrix.k() != K || matrix.n() != N || matrix.group_count() != GROUPS {
        return Err(invalid(format!(
            "attention output wants K={K} N={N} groups={GROUPS}, got {:?} K={} N={} groups={}",
            matrix.encoding(),
            matrix.k(),
            matrix.n(),
            matrix.group_count()
        )));
    }
    Ok(())
}

pub(crate) fn dense_effective_bf16(matrix: &OpusPackedMatrix) -> Vec<u16> {
    let mut dense = vec![0u16; K * N];
    let signs1 = gen_fwht_signs(42, GROUP);
    let signs2 = gen_fwht_signs(1042, GROUP);
    for group in 0..GROUPS {
        let values = matrix.group_dense_i8(group);
        let scales = matrix.group_scales(group);
        for n in 0..N {
            let mut column = [0.0f32; GROUP];
            for inner in 0..GROUP {
                column[inner] = values[inner * N + n] as f32 * scales[n];
            }
            // Opus stores weights in the signed-FWHT basis. The attention
            // boundary is canonical BF16, so invert that transform here once
            // at upload. AWQ stores W*s and normally consumes x/s; dividing
            // the recovered weight by s restores W for the canonical input.
            cpu_fwht_256(&mut column, &signs2, &signs1);
            for inner in 0..GROUP {
                let k = group * GROUP + inner;
                let awq = matrix.awq_scale().map_or(1.0, |scale| scale[k]);
                dense[k * N + n] = f32_to_bf16_bits(column[inner] / awq);
            }
        }
    }
    dense
}

fn pack_weights(matrix: &[u16]) -> Vec<u8> {
    let mut packed = vec![0u8; WEIGHT_BYTES];
    for col in 0..COLS {
        for slice in 0..SLICES {
            let column_base = col * SLICES * 32 + slice * 32;
            for group in 0..GROUPS {
                let block = (col * SLICES * GROUPS + slice * GROUPS + group) * BLOCK;
                for nt in 0..4 {
                    for kt in 0..32 {
                        for kk in 0..8 {
                            for nn in 0..8 {
                                let k = group * GROUP + kt * 8 + kk;
                                let n = column_base + nt * 8 + nn;
                                let target = block + ((nt * 32 + kt) * 64 + kk * 8 + nn) * 2;
                                packed[target..target + 2]
                                    .copy_from_slice(&matrix[k * N + n].to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
    }
    packed
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_full_array_output_projection() {
        assert_eq!(NpuAttentionOutputBf16::input_bytes(), 393_216);
        assert_eq!(NpuAttentionOutputBf16::output_bytes(), 786_432);
        assert_eq!(NpuAttentionOutputBf16::weight_bytes(), 1_179_648);
    }

    #[test]
    fn weight_packer_covers_every_bf16_once() {
        let source = (0..K * N).map(|index| index as u16).collect::<Vec<_>>();
        let packed = pack_weights(&source);
        assert_eq!(packed.len(), WEIGHT_BYTES);
        let nonzero = packed
            .chunks_exact(2)
            .filter(|pair| pair[0] != 0 || pair[1] != 0)
            .count();
        assert_eq!(nonzero, source.iter().filter(|&&value| value != 0).count());
    }
}
