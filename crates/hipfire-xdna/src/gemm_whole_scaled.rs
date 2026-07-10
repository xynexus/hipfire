//! Group-retaining AIE2P whole-array W4A8 GEMM with fused f32 scaling.
#![cfg(target_os = "linux")]

use std::collections::HashMap;

use crate::{DeviceBuffer, NpuKernel, NpuWholeMode, XdnaError};

const ARRAY: usize = 4;
const GROUP_K: usize = 256;
const MACRO_M: usize = 96;
const A_DATA: usize = 6144;
const W_DATA: usize = 12288;
const A_BLOCK: usize = 8192;
const W_BLOCK: usize = 16384;

#[derive(Clone, Copy)]
struct Layout {
    mode: NpuWholeMode,
    lm: usize,
    ln: usize,
    mr: usize,
    inner_k: usize,
    cols_stripe: usize,
    c_core: usize,
}

impl Layout {
    fn for_mode(mode: NpuWholeMode) -> Self {
        match mode {
            NpuWholeMode::W4 => Self {
                mode,
                lm: 6,
                ln: 6,
                mr: 4,
                inner_k: 16,
                cols_stripe: 96,
                c_core: 2304,
            },
            NpuWholeMode::W8 => Self {
                mode,
                lm: 3,
                ln: 3,
                mr: 8,
                inner_k: 8,
                cols_stripe: 48,
                c_core: 1152,
            },
        }
    }

    fn rows_stripe(self) -> usize {
        self.lm * self.mr
    }

    fn macro_n(self) -> usize {
        ARRAY * self.cols_stripe
    }

    fn c_join(self) -> usize {
        ARRAY * self.c_core
    }
}

pub struct NpuWholeScaledResidentWeights {
    buffer: DeviceBuffer,
}

pub struct NpuGemmWholeScaled {
    kernel: NpuKernel,
    layout: Layout,
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
        let mode = if manifest.lines().any(|line| line == "mode=w4-scaled") {
            NpuWholeMode::W4
        } else if manifest.lines().any(|line| line == "mode=w8-scaled") {
            NpuWholeMode::W8
        } else {
            return Err(invalid("scaled whole-array cache must be W4 or W8"));
        };
        let layout = Layout::for_mode(mode);
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
            || n_macros != n.div_ceil(layout.macro_n())
            || outblocks != m_macros * n_macros
        {
            return Err(invalid("invalid scaled whole-array cache geometry"));
        }
        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).map_err(XdnaError::Open)?;
        let insts = std::fs::read(format!("{dir}/insts.bin")).map_err(XdnaError::Open)?;
        let kernel = NpuKernel::load(&xclbin, &insts)?;
        let inblocks = outblocks * groups;
        let input = kernel.alloc_arg(ARRAY * inblocks * A_BLOCK)?;
        let output = kernel.alloc_arg(ARRAY * outblocks * layout.c_join() * size_of::<f32>())?;
        Ok(Self {
            kernel,
            layout,
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

    pub fn mode(&self) -> NpuWholeMode {
        self.layout.mode
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
                        for ln in 0..self.layout.ln {
                            for kt in 0..GROUP_K / self.layout.inner_k {
                                for kk in 0..self.layout.inner_k {
                                    for nn in 0..16 {
                                        let col = n_macro * self.layout.macro_n()
                                            + stripe * self.layout.cols_stripe
                                            + ln * 16
                                            + nn;
                                        let value = if col < self.n {
                                            weights[group]
                                                [(kt * self.layout.inner_k + kk) * self.n + col]
                                        } else {
                                            0
                                        };
                                        match self.layout.mode {
                                            NpuWholeMode::W4 => {
                                                if !(-8..=7).contains(&value) {
                                                    return Err(invalid(format!(
                                                        "W4 value {value} outside -8..=7"
                                                    )));
                                                }
                                                let index = (ln * 16 + kt) * 256 + kk * 16 + nn;
                                                let nibble = (value & 0x0f) as u8;
                                                packed[base + index / 2] |= if index % 2 == 0 {
                                                    nibble
                                                } else {
                                                    nibble << 4
                                                };
                                            }
                                            NpuWholeMode::W8 => {
                                                let index = (ln * 32 + kt) * 128
                                                    + (nn / 8) * 64
                                                    + kk * 8
                                                    + nn % 8;
                                                packed[base + index] = value as u8;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        for local_col in 0..self.layout.cols_stripe {
                            let col = n_macro * self.layout.macro_n()
                                + stripe * self.layout.cols_stripe
                                + local_col;
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
                        for lm in 0..self.layout.lm {
                            for kt in 0..GROUP_K / self.layout.inner_k {
                                for local_row in 0..self.layout.mr {
                                    let row = m_macro * MACRO_M
                                        + stripe * self.layout.rows_stripe()
                                        + lm * self.layout.mr
                                        + local_row;
                                    if row < self.rows {
                                        let source = row * self.k()
                                            + group * GROUP_K
                                            + kt * self.layout.inner_k;
                                        let target = base
                                            + (lm * (GROUP_K / self.layout.inner_k) + kt) * 64
                                            + local_row * self.layout.inner_k;
                                        self.input.as_mut_slice()
                                            [target..target + self.layout.inner_k]
                                            .copy_from_slice(as_bytes(
                                                &activations[source..source + self.layout.inner_k],
                                            ));
                                    }
                                }
                            }
                        }
                        for local_row in 0..self.layout.rows_stripe() {
                            let row =
                                m_macro * MACRO_M + stripe * self.layout.rows_stripe() + local_row;
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
                        let core = (col_stripe * self.outblocks + outblock) * self.layout.c_join()
                            + row_stripe * self.layout.c_core;
                        for lm in 0..self.layout.lm {
                            for ln in 0..self.layout.ln {
                                for local_row in 0..self.layout.mr {
                                    let row = m_macro * MACRO_M
                                        + row_stripe * self.layout.rows_stripe()
                                        + lm * self.layout.mr
                                        + local_row;
                                    if row >= self.rows {
                                        continue;
                                    }
                                    for local_col in 0..16 {
                                        let col = n_macro * self.layout.macro_n()
                                            + col_stripe * self.layout.cols_stripe
                                            + ln * 16
                                            + local_col;
                                        if col < self.n {
                                            let source = core
                                                + (lm * self.layout.ln + ln) * self.layout.mr * 16
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
        for mode in [NpuWholeMode::W4, NpuWholeMode::W8] {
            let layout = Layout::for_mode(mode);
            assert!(A_DATA + layout.rows_stripe() * 4 <= A_BLOCK);
            assert!(W_DATA + layout.cols_stripe * 4 <= W_BLOCK);
        }
    }
}
