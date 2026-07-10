//! Group-retaining AIE2P whole-array W4A8 GEMM with fused f32 scaling.
#![cfg(target_os = "linux")]

use std::collections::HashMap;

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const ARRAY: usize = 4;
const LM: usize = 6;
const LN: usize = 6;
const GROUP_K: usize = 256;
const ROWS_STRIPE: usize = 24;
const COLS_STRIPE: usize = 96;
const MACRO_M: usize = 96;
const MACRO_N: usize = 384;
const A_DATA: usize = 6144;
const W_DATA: usize = 12288;
const A_BLOCK: usize = 8192;
const W_BLOCK: usize = 16384;
const C_CORE: usize = 2304;
const C_JOIN: usize = 9216;

pub struct NpuWholeScaledResidentWeights {
    buffer: DeviceBuffer,
}

pub struct NpuGemmWholeScaled {
    kernel: NpuKernel,
    rows: usize,
    groups: usize,
    n: usize,
    m_macros: usize,
    n_macros: usize,
    outblocks: usize,
    input: DeviceBuffer,
    output: DeviceBuffer,
}

impl NpuGemmWholeScaled {
    pub fn load_cached(dir: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{dir}/shape.txt")).map_err(XdnaError::Open)?;
        if !manifest.lines().any(|line| line == "mode=w4-scaled") {
            return Err(invalid("scaled whole-array cache must be mode=w4-scaled"));
        }
        let shape = parse_shape(&manifest);
        let rows = required(&shape, "m")?;
        let k = required(&shape, "k")?;
        let n = required(&shape, "n")?;
        let m_macros = required(&shape, "mm")?;
        let n_macros = required(&shape, "nm")?;
        let groups = required(&shape, "kg")?;
        let outblocks = required(&shape, "outblocks")?;
        if rows == 0
            || n == 0
            || k != groups * GROUP_K
            || m_macros != rows.div_ceil(MACRO_M)
            || n_macros != n.div_ceil(MACRO_N)
            || outblocks != m_macros * n_macros
        {
            return Err(invalid("invalid scaled whole-array cache geometry"));
        }
        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).map_err(XdnaError::Open)?;
        let insts = std::fs::read(format!("{dir}/insts.bin")).map_err(XdnaError::Open)?;
        let kernel = NpuKernel::load(&xclbin, &insts)?;
        let inblocks = outblocks * groups;
        let input = kernel.alloc_arg(ARRAY * inblocks * A_BLOCK)?;
        let output = kernel.alloc_arg(ARRAY * outblocks * C_JOIN * size_of::<f32>())?;
        Ok(Self {
            kernel,
            rows,
            groups,
            n,
            m_macros,
            n_macros,
            outblocks,
            input,
            output,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn k(&self) -> usize {
        self.groups * GROUP_K
    }

    pub fn n(&self) -> usize {
        self.n
    }

    fn inblocks(&self) -> usize {
        self.outblocks * self.groups
    }

    pub fn packed_weight_bytes(&self) -> usize {
        ARRAY * self.inblocks() * W_BLOCK
    }

    /// Pack W4 `[256,N]` groups and their per-column scales into persistent,
    /// power-of-two-padded streams. Scale tails begin after the exact R14 W4
    /// prefix; padding avoids the AIE2P DMA corruption seen with odd payloads.
    pub fn prepack_weights(
        &self,
        weights: &[&[i8]],
        scales: &[&[f32]],
    ) -> Result<Vec<u8>, XdnaError> {
        if weights.len() != self.groups
            || scales.len() != self.groups
            || weights.iter().any(|group| group.len() != GROUP_K * self.n)
            || scales.iter().any(|group| group.len() != self.n)
        {
            return Err(invalid("scaled whole-array weight geometry mismatch"));
        }
        let mut packed = vec![0u8; self.packed_weight_bytes()];
        for stripe in 0..ARRAY {
            for m_macro in 0..self.m_macros {
                for n_macro in 0..self.n_macros {
                    let outblock = m_macro * self.n_macros + n_macro;
                    for group in 0..self.groups {
                        let block = outblock * self.groups + group;
                        let base = (stripe * self.inblocks() + block) * W_BLOCK;
                        for ln in 0..LN {
                            for kt in 0..16 {
                                for kk in 0..16 {
                                    for nn in 0..16 {
                                        let col =
                                            n_macro * MACRO_N + stripe * COLS_STRIPE + ln * 16 + nn;
                                        let value = if col < self.n {
                                            weights[group][(kt * 16 + kk) * self.n + col]
                                        } else {
                                            0
                                        };
                                        if !(-8..=7).contains(&value) {
                                            return Err(invalid(format!(
                                                "W4 value {value} outside -8..=7"
                                            )));
                                        }
                                        let index = (ln * 16 + kt) * 256 + kk * 16 + nn;
                                        let nibble = (value & 0x0f) as u8;
                                        packed[base + index / 2] |=
                                            if index % 2 == 0 { nibble } else { nibble << 4 };
                                    }
                                }
                            }
                        }
                        for local_col in 0..COLS_STRIPE {
                            let col = n_macro * MACRO_N + stripe * COLS_STRIPE + local_col;
                            let scale = if col < self.n {
                                scales[group][col]
                            } else {
                                0.0
                            };
                            let offset = base + W_DATA + local_col * size_of::<f32>();
                            packed[offset..offset + size_of::<f32>()]
                                .copy_from_slice(&scale.to_ne_bytes());
                        }
                    }
                }
            }
        }
        Ok(packed)
    }

    pub fn upload_resident_weights(
        &self,
        packed: &[u8],
    ) -> Result<NpuWholeScaledResidentWeights, XdnaError> {
        if packed.len() != self.packed_weight_bytes() {
            return Err(invalid("scaled whole-array packed weight size mismatch"));
        }
        let mut buffer = self.kernel.alloc_arg(packed.len())?;
        buffer.as_mut_slice().copy_from_slice(packed);
        self.kernel.sync_to_device(&buffer)?;
        Ok(NpuWholeScaledResidentWeights { buffer })
    }

    pub fn run_resident(
        &mut self,
        weights: &NpuWholeScaledResidentWeights,
        activations: &[i8],
        activation_scales: &[f32],
        output: &mut [f32],
    ) -> Result<(), XdnaError> {
        if activations.len() != self.rows * self.k()
            || activation_scales.len() != self.groups * self.rows
            || output.len() != self.rows * self.n
            || weights.buffer.len() != self.packed_weight_bytes()
        {
            return Err(invalid(
                "scaled whole-array activation/output geometry mismatch",
            ));
        }
        self.pack_activations(activations, activation_scales);
        self.kernel.dispatch_synced(
            &[&self.input, &weights.buffer, &self.output],
            &[true, false, true],
        )?;
        self.kernel.sync_output(&self.output)?;
        self.unpack_output(output);
        Ok(())
    }

    fn pack_activations(&mut self, activations: &[i8], scales: &[f32]) {
        self.input.as_mut_slice().fill(0);
        for stripe in 0..ARRAY {
            for m_macro in 0..self.m_macros {
                for n_macro in 0..self.n_macros {
                    let outblock = m_macro * self.n_macros + n_macro;
                    for group in 0..self.groups {
                        let block = outblock * self.groups + group;
                        let base = (stripe * self.inblocks() + block) * A_BLOCK;
                        for lm in 0..LM {
                            for kt in 0..16 {
                                for local_row in 0..4 {
                                    let row = m_macro * MACRO_M
                                        + stripe * ROWS_STRIPE
                                        + lm * 4
                                        + local_row;
                                    if row < self.rows {
                                        let source = row * self.k() + group * GROUP_K + kt * 16;
                                        let target = base + (lm * 16 + kt) * 64 + local_row * 16;
                                        self.input.as_mut_slice()[target..target + 16]
                                            .copy_from_slice(as_bytes(
                                                &activations[source..source + 16],
                                            ));
                                    }
                                }
                            }
                        }
                        for local_row in 0..ROWS_STRIPE {
                            let row = m_macro * MACRO_M + stripe * ROWS_STRIPE + local_row;
                            let scale = if row < self.rows {
                                scales[group * self.rows + row]
                            } else {
                                0.0
                            };
                            let offset = base + A_DATA + local_row * size_of::<f32>();
                            self.input.as_mut_slice()[offset..offset + size_of::<f32>()]
                                .copy_from_slice(&scale.to_ne_bytes());
                        }
                    }
                }
            }
        }
    }

    fn unpack_output(&self, output: &mut [f32]) {
        let physical = as_f32(self.output.as_slice());
        for col_stripe in 0..ARRAY {
            for m_macro in 0..self.m_macros {
                for n_macro in 0..self.n_macros {
                    let outblock = m_macro * self.n_macros + n_macro;
                    for row_stripe in 0..ARRAY {
                        let core =
                            (col_stripe * self.outblocks + outblock) * C_JOIN + row_stripe * C_CORE;
                        for lm in 0..LM {
                            for ln in 0..LN {
                                for local_row in 0..4 {
                                    let row = m_macro * MACRO_M
                                        + row_stripe * ROWS_STRIPE
                                        + lm * 4
                                        + local_row;
                                    if row >= self.rows {
                                        continue;
                                    }
                                    for local_col in 0..16 {
                                        let col = n_macro * MACRO_N
                                            + col_stripe * COLS_STRIPE
                                            + ln * 16
                                            + local_col;
                                        if col < self.n {
                                            let source = core
                                                + (lm * LN + ln) * 64
                                                + local_row * 16
                                                + local_col;
                                            output[row * self.n + col] = physical[source];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn parse_shape(contents: &str) -> HashMap<&str, usize> {
    contents
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once('=')?;
            Some((key, value.parse().ok()?))
        })
        .collect()
}

fn required(shape: &HashMap<&str, usize>, key: &str) -> Result<usize, XdnaError> {
    shape
        .get(key)
        .copied()
        .ok_or_else(|| invalid(format!("missing {key} in scaled cache manifest")))
}

fn as_bytes(values: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
}

fn as_f32(values: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len() / size_of::<f32>()) }
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_streams_preserve_exact_prefix_and_scale_tails() {
        assert_eq!((A_DATA, A_BLOCK), (6144, 8192));
        assert_eq!((W_DATA, W_BLOCK), (12288, 16384));
        assert!(A_DATA + ROWS_STRIPE * 4 <= A_BLOCK);
        assert!(W_DATA + COLS_STRIPE * 4 <= W_BLOCK);
    }
}
