//! Complete resident EmbeddingGemma dense-W8 FFN over the R26 AIE2P schedule.
//!
//! Native OQ8 and arbitrary compact mixed Opus matrices share this executor:
//! mixed groups are expanded exactly once while their resident weights are
//! uploaded, leaving one format-independent dense-int8 dispatch contract.

use hipfire_primitives::fwht::gen_fwht_signs;

use crate::{DeviceBuffer, NpuKernel, OpusPackedMatrix, OpusResidentMode, XdnaError};

const M: usize = 256;
const PAD_M: usize = 288;
const K: usize = 768;
const INTERMEDIATE: usize = 1152;
const OUTPUT: usize = 768;
const GROUP: usize = 256;
const GATE_GROUPS: usize = 3;
const DOWN_GROUPS: usize = 5;
const COLS: usize = 8;
const ROW_STRIPES: usize = 4;
const GATE_N_BLOCKS: usize = 6;
const GATE_BLOCKS: usize = 3 * GATE_N_BLOCKS * GATE_GROUPS;
const DOWN_BLOCKS: usize = 3 * DOWN_GROUPS * 2;
const WEIGHT_BLOCKS: usize = GATE_BLOCKS + DOWN_BLOCKS;
const DATA_PAIR: usize = 9216;
const DATA_REPEATS: usize = 4;
const DATA_JOIN: usize = DATA_REPEATS * DATA_PAIR;
const W_BLOCK: usize = 16384;
const W_DATA: usize = 12288;
const W_COLS: usize = 48;
const PARAM_OFFSET: usize = W_DATA + W_COLS * size_of::<f32>();
const T_ROWS: usize = 296;
const T_STRIDE: usize = 5376;

// R26 is repeatable for six commands in one context. Count the mandatory
// context-prime command in that allowance and recreate before command seven.
const MAX_CONTEXT_COMMANDS: usize = 6;

pub struct NpuResidentFfnDenseW8Weights {
    buffer: DeviceBuffer,
}

pub struct NpuResidentFfnDenseW8 {
    kernel: NpuKernel,
    input: DeviceBuffer,
    scratch: DeviceBuffer,
    output: DeviceBuffer,
    primed: bool,
    context_commands: usize,
}

impl NpuResidentFfnDenseW8 {
    pub fn load_cached(cache: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{cache}/shape.txt")).map_err(XdnaError::Open)?;
        for required in [
            "op=resident_ffn",
            "mode=dense-w8",
            "m=256",
            "k=768",
            "intermediate=1152",
            "out=768",
        ] {
            if !manifest.lines().any(|line| line == required) {
                return Err(invalid(format!(
                    "resident dense-W8 FFN cache missing shape field {required}"
                )));
            }
        }
        let xclbin = std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?;
        let insts = std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?;
        let kernel = NpuKernel::load(&xclbin, &insts)?;
        let input = kernel.alloc_arg(Self::input_bytes())?;
        let scratch = kernel.alloc_arg(Self::scratch_bytes())?;
        let output = kernel.alloc_arg(Self::output_bytes())?;
        Ok(Self {
            kernel,
            input,
            scratch,
            output,
            primed: false,
            context_commands: 0,
        })
    }

    pub const fn rows() -> usize {
        M
    }

    pub const fn input_bytes() -> usize {
        ROW_STRIPES * GATE_BLOCKS * DATA_JOIN
    }

    pub const fn output_bytes() -> usize {
        PAD_M * OUTPUT * size_of::<f32>()
    }

    pub const fn scratch_bytes() -> usize {
        T_ROWS * T_STRIDE * size_of::<f32>()
    }

    pub const fn input_block_bytes() -> usize {
        DATA_JOIN
    }

    pub const fn input_repeats() -> usize {
        DATA_REPEATS
    }

    pub const fn input_repeat_stride() -> usize {
        DATA_PAIR
    }

    pub fn attach_shared_io(
        &mut self,
        input_fd: i32,
        input_bytes: usize,
        output_fd: i32,
        output_bytes: usize,
    ) -> Result<(), XdnaError> {
        if input_bytes != Self::input_bytes() || output_bytes != Self::output_bytes() {
            return Err(invalid(
                "resident dense-W8 FFN shared dma-buf size mismatch",
            ));
        }
        self.input = self.kernel.import_dmabuf(input_fd, input_bytes, true)?;
        self.output = self.kernel.import_dmabuf(output_fd, output_bytes, true)?;
        self.kernel.sync_to_device(&self.output)?;
        self.primed = false;
        self.context_commands = 0;
        Ok(())
    }

    pub fn upload_weights(
        &self,
        gate: &OpusPackedMatrix,
        up: &OpusPackedMatrix,
        down: &OpusPackedMatrix,
    ) -> Result<NpuResidentFfnDenseW8Weights, XdnaError> {
        validate_matrix(gate, K, INTERMEDIATE, GATE_GROUPS, "gate")?;
        validate_matrix(up, K, INTERMEDIATE, GATE_GROUPS, "up")?;
        validate_matrix(down, INTERMEDIATE, OUTPUT, DOWN_GROUPS, "down")?;
        if gate.awq_scale() != up.awq_scale() {
            return Err(invalid(
                "resident dense-W8 FFN gate/up matrices must share one AWQ activation scale",
            ));
        }

        let gate_groups = dense_groups(gate);
        let up_groups = dense_groups(up);
        let down_groups = dense_groups(down);
        let down_awq = padded_awq(down.awq_scale(), DOWN_GROUPS * GROUP);
        let signs1 = gen_fwht_signs(42, GROUP);
        let signs2 = gen_fwht_signs(1042, GROUP);
        let mut packed = vec![0u8; COLS * WEIGHT_BLOCKS * W_BLOCK];
        for stripe in 0..COLS {
            for mblock in 0..3 {
                for nblock in 0..GATE_N_BLOCKS {
                    for group in 0..GATE_GROUPS {
                        let block = (mblock * GATE_N_BLOCKS + nblock) * GATE_GROUPS + group;
                        let start = (stripe * WEIGHT_BLOCKS + block) * W_BLOCK;
                        pack_gate_block(
                            &mut packed[start..start + W_BLOCK],
                            stripe,
                            nblock,
                            &gate_groups[group],
                            gate.group_scales(group),
                            &up_groups[group],
                            up.group_scales(group),
                        );
                    }
                }
            }
            for mblock in 0..3 {
                for group in 0..DOWN_GROUPS {
                    for nmacro in 0..2 {
                        let block = GATE_BLOCKS + (mblock * DOWN_GROUPS + group) * 2 + nmacro;
                        let start = (stripe * WEIGHT_BLOCKS + block) * W_BLOCK;
                        pack_down_block(
                            &mut packed[start..start + W_BLOCK],
                            stripe,
                            nmacro,
                            group,
                            &down_groups[group],
                            down.group_scales(group),
                            &down_awq,
                            &signs1,
                            &signs2,
                        );
                    }
                }
            }
        }
        let mut buffer = self.kernel.alloc_arg(packed.len())?;
        buffer.as_mut_slice().copy_from_slice(&packed);
        self.kernel.sync_to_device(&buffer)?;
        Ok(NpuResidentFfnDenseW8Weights { buffer })
    }

    pub fn run_shared(&mut self, weights: &NpuResidentFfnDenseW8Weights) -> Result<(), XdnaError> {
        if weights.buffer.len() != COLS * WEIGHT_BLOCKS * W_BLOCK {
            return Err(invalid("resident dense-W8 FFN packed weight size mismatch"));
        }
        if self.context_commands >= MAX_CONTEXT_COMMANDS {
            self.kernel.recreate_hwctx()?;
            self.primed = false;
            self.context_commands = 0;
        }
        if !self.primed {
            self.clear_intermediates();
            self.kernel.dispatch_synced(
                &[&self.input, &weights.buffer, &self.scratch, &self.output],
                &[true, false, true, true],
            )?;
            self.context_commands += 1;
            self.clear_intermediates();
            self.kernel.sync_to_device(&self.scratch)?;
            self.kernel.sync_to_device(&self.output)?;
            self.primed = true;
        }
        self.kernel.dispatch_synced(
            &[&self.input, &weights.buffer, &self.scratch, &self.output],
            &[false, false, false, false],
        )?;
        self.kernel.sync_output(&self.output)?;
        self.context_commands += 1;
        Ok(())
    }

    fn clear_intermediates(&mut self) {
        self.scratch.as_mut_slice().fill(0);
        self.output.as_mut_slice().fill(0);
    }
}

fn validate_matrix(
    matrix: &OpusPackedMatrix,
    k: usize,
    n: usize,
    groups: usize,
    role: &str,
) -> Result<(), XdnaError> {
    if matrix.resident_mode() != OpusResidentMode::DenseW8
        || matrix.k() != k
        || matrix.n() != n
        || matrix.group_count() != groups
    {
        return Err(invalid(format!(
            "resident dense-W8 FFN {role} wants dense-W8 K={k} N={n} groups={groups}, got {:?} K={} N={} groups={}",
            matrix.resident_mode(),
            matrix.k(),
            matrix.n(),
            matrix.group_count()
        )));
    }
    Ok(())
}

fn dense_groups(matrix: &OpusPackedMatrix) -> Vec<std::borrow::Cow<'_, [i8]>> {
    (0..matrix.group_count())
        .map(|group| matrix.group_dense_i8(group))
        .collect()
}

fn padded_awq(scale: Option<&[f32]>, padded_k: usize) -> Vec<f32> {
    let mut padded = vec![1.0f32; padded_k];
    if let Some(scale) = scale {
        padded[..scale.len()].copy_from_slice(scale);
    }
    padded
}

#[allow(clippy::too_many_arguments)]
fn pack_gate_block(
    block: &mut [u8],
    stripe: usize,
    nblock: usize,
    gate_weights: &[i8],
    gate_scales: &[f32],
    up_weights: &[i8],
    up_scales: &[f32],
) {
    for ln in 0..3 {
        for kt in 0..32 {
            for kk in 0..8 {
                for nn in 0..16 {
                    let local = ln * 16 + nn;
                    let (up, tail) = gate_up_local(local);
                    let col = (nblock * COLS + stripe) * 24 + tail;
                    let weights = if up { up_weights } else { gate_weights };
                    let index = (ln * 32 + kt) * 128 + (nn / 8) * 64 + kk * 8 + nn % 8;
                    block[index] = weights[(kt * 8 + kk) * INTERMEDIATE + col] as u8;
                }
            }
        }
    }
    for local in 0..W_COLS {
        let (up, tail) = gate_up_local(local);
        let col = (nblock * COLS + stripe) * 24 + tail;
        let scale = if up { up_scales[col] } else { gate_scales[col] };
        let offset = W_DATA + local * size_of::<f32>();
        block[offset..offset + size_of::<f32>()].copy_from_slice(&scale.to_ne_bytes());
    }
}

fn gate_up_local(local: usize) -> (bool, usize) {
    match local {
        0..16 => (false, local),
        16..32 => (true, local - 16),
        32..40 => (false, local - 16),
        40..48 => (true, local - 24),
        _ => unreachable!("gate/up physical column is outside 0..48"),
    }
}

#[allow(clippy::too_many_arguments)]
fn pack_down_block(
    block: &mut [u8],
    stripe: usize,
    nmacro: usize,
    group: usize,
    weights: &[i8],
    scales: &[f32],
    awq: &[f32],
    signs1: &[f32],
    signs2: &[f32],
) {
    for ln in 0..3 {
        for kt in 0..32 {
            for kk in 0..8 {
                for nn in 0..16 {
                    let col = nmacro * 384 + stripe * W_COLS + ln * 16 + nn;
                    let index = (ln * 32 + kt) * 128 + (nn / 8) * 64 + kk * 8 + nn % 8;
                    block[index] = weights[(kt * 8 + kk) * OUTPUT + col] as u8;
                }
            }
        }
    }
    for local in 0..W_COLS {
        let col = nmacro * 384 + stripe * W_COLS + local;
        let offset = W_DATA + local * size_of::<f32>();
        block[offset..offset + size_of::<f32>()].copy_from_slice(&scales[col].to_ne_bytes());
    }
    let mut params = Vec::with_capacity(3 * GROUP);
    params.extend_from_slice(&awq[group * GROUP..(group + 1) * GROUP]);
    params.extend_from_slice(signs1);
    params.extend_from_slice(signs2);
    block[PARAM_OFFSET..PARAM_OFFSET + params.len() * size_of::<f32>()]
        .copy_from_slice(unsafe { as_bytes(&params) });
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

unsafe fn as_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_up_physical_columns_cover_both_roles_without_aliases() {
        let mapped = (0..W_COLS).map(gate_up_local).collect::<Vec<_>>();
        let gate = mapped.iter().filter(|(up, _)| !up).count();
        let up = mapped.iter().filter(|(up, _)| *up).count();
        assert_eq!((gate, up), (24, 24));
        for role in [false, true] {
            let mut tails = mapped
                .iter()
                .filter_map(|&(up, tail)| (up == role).then_some(tail))
                .collect::<Vec<_>>();
            tails.sort_unstable();
            assert_eq!(tails, (0..24).collect::<Vec<_>>());
        }
    }

    #[test]
    fn dense_w8_argument_geometry_matches_r26() {
        assert_eq!(NpuResidentFfnDenseW8::input_bytes(), 7_962_624);
        assert_eq!(NpuResidentFfnDenseW8::scratch_bytes(), 6_365_184);
        assert_eq!(NpuResidentFfnDenseW8::output_bytes(), 884_736);
        assert_eq!(COLS * WEIGHT_BLOCKS * W_BLOCK, 11_010_048);
    }
}
