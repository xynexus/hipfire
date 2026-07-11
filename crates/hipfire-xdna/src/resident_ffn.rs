//! Complete resident EmbeddingGemma W4 FFN over the R25 AIE2P schedule.

use hipfire_primitives::fwht::gen_fwht_signs;

use crate::{DeviceBuffer, NpuKernel, OpusMatrixEncoding, OpusPackedMatrix, XdnaError};

const M: usize = 256;
const PAD_M: usize = 288;
const K: usize = 768;
const INTERMEDIATE: usize = 1152;
const OUTPUT: usize = 768;
const GROUP: usize = 256;
const GATE_GROUPS: usize = 3;
const DOWN_GROUPS: usize = 5;
const COLS: usize = 8;
const A_BLOCK: usize = 6656;
const A_BLOCKS: usize = 27;
const W_BLOCK: usize = 15872;
const W_BLOCKS: usize = 42;
const W_DATA: usize = 12288;
const W_COLS: usize = 96;
const PARAM_OFFSET: usize = W_DATA + W_COLS * size_of::<f32>();
const RECYCLE_DISPATCHES: usize = 7;

pub struct NpuResidentFfnW4Weights {
    buffer: DeviceBuffer,
}

pub struct NpuResidentFfnW4 {
    kernel: NpuKernel,
    input: DeviceBuffer,
    output: DeviceBuffer,
    dispatches: usize,
}

impl NpuResidentFfnW4 {
    pub fn load_cached(cache: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{cache}/shape.txt")).map_err(XdnaError::Open)?;
        for required in [
            "op=resident_ffn",
            "mode=w4",
            "m=256",
            "k=768",
            "intermediate=1152",
            "out=768",
        ] {
            if !manifest.lines().any(|line| line == required) {
                return Err(invalid(format!(
                    "resident FFN cache missing shape field {required}"
                )));
            }
        }
        let xclbin = std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?;
        let insts = std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?;
        let kernel = NpuKernel::load(&xclbin, &insts)?;
        let input = kernel.alloc_arg(Self::input_bytes())?;
        let output = kernel.alloc_arg(Self::output_bytes())?;
        Ok(Self {
            kernel,
            input,
            output,
            dispatches: 0,
        })
    }

    pub const fn rows() -> usize {
        M
    }

    pub const fn input_bytes() -> usize {
        4 * A_BLOCKS * A_BLOCK
    }

    pub const fn output_bytes() -> usize {
        PAD_M * OUTPUT * size_of::<f32>()
    }

    pub const fn input_block_bytes() -> usize {
        A_BLOCK
    }

    pub fn attach_shared_io(
        &mut self,
        input_fd: i32,
        input_bytes: usize,
        output_fd: i32,
        output_bytes: usize,
    ) -> Result<(), XdnaError> {
        if input_bytes != Self::input_bytes() || output_bytes != Self::output_bytes() {
            return Err(invalid("resident FFN shared dma-buf size mismatch"));
        }
        self.input = self.kernel.import_dmabuf(input_fd, input_bytes, true)?;
        self.output = self.kernel.import_dmabuf(output_fd, output_bytes, true)?;
        self.kernel.sync_to_device(&self.output)
    }

    pub fn upload_weights(
        &self,
        gate: &OpusPackedMatrix,
        up: &OpusPackedMatrix,
        down: &OpusPackedMatrix,
    ) -> Result<NpuResidentFfnW4Weights, XdnaError> {
        validate_matrix(gate, K, INTERMEDIATE, GATE_GROUPS, "gate")?;
        validate_matrix(up, K, INTERMEDIATE, GATE_GROUPS, "up")?;
        validate_matrix(down, INTERMEDIATE, OUTPUT, DOWN_GROUPS, "down")?;
        if gate.awq_scale() != up.awq_scale() {
            return Err(invalid(
                "resident FFN gate/up matrices must share one AWQ activation scale",
            ));
        }

        let mut packed = vec![0u8; COLS * W_BLOCKS * W_BLOCK];
        let down_awq = padded_awq(down.awq_scale(), DOWN_GROUPS * GROUP);
        let signs1 = gen_fwht_signs(42, GROUP);
        let signs2 = gen_fwht_signs(1042, GROUP);
        for stripe in 0..COLS {
            let mut destination_block = 0;
            for _m_macro in 0..3 {
                for n_macro in 0..3 {
                    for group in 0..GATE_GROUPS {
                        let destination = (stripe * W_BLOCKS + destination_block) * W_BLOCK;
                        pack_gate_block(
                            &mut packed[destination..destination + W_BLOCK],
                            stripe,
                            n_macro,
                            group,
                            gate,
                            up,
                        );
                        destination_block += 1;
                    }
                    let ready: &[usize] = match n_macro {
                        0 => &[0],
                        1 => &[1, 2],
                        _ => &[3, 4],
                    };
                    for &group in ready {
                        let destination = (stripe * W_BLOCKS + destination_block) * W_BLOCK;
                        pack_down_block(
                            &mut packed[destination..destination + W_BLOCK],
                            stripe,
                            group,
                            down,
                            &down_awq,
                            &signs1,
                            &signs2,
                        );
                        destination_block += 1;
                    }
                }
            }
            debug_assert_eq!(destination_block, W_BLOCKS);
        }
        let mut buffer = self.kernel.alloc_arg(packed.len())?;
        buffer.as_mut_slice().copy_from_slice(&packed);
        self.kernel.sync_to_device(&buffer)?;
        Ok(NpuResidentFfnW4Weights { buffer })
    }

    pub fn run_shared(&mut self, weights: &NpuResidentFfnW4Weights) -> Result<(), XdnaError> {
        if weights.buffer.len() != COLS * W_BLOCKS * W_BLOCK {
            return Err(invalid("resident FFN packed weight size mismatch"));
        }
        if self.dispatches == RECYCLE_DISPATCHES {
            self.kernel.recreate_hwctx()?;
            self.dispatches = 0;
        }
        self.kernel.dispatch_synced(
            &[&self.input, &weights.buffer, &self.output],
            &[true, false, true],
        )?;
        self.kernel.sync_output(&self.output)?;
        self.dispatches += 1;
        Ok(())
    }
}

fn validate_matrix(
    matrix: &OpusPackedMatrix,
    k: usize,
    n: usize,
    groups: usize,
    role: &str,
) -> Result<(), XdnaError> {
    if matrix.encoding() != OpusMatrixEncoding::W4
        || matrix.k() != k
        || matrix.n() != n
        || matrix.group_count() != groups
    {
        return Err(invalid(format!(
            "resident FFN {role} wants W4 K={k} N={n} groups={groups}, got {:?} K={} N={} groups={}",
            matrix.encoding(),
            matrix.k(),
            matrix.n(),
            matrix.group_count()
        )));
    }
    Ok(())
}

fn padded_awq(scale: Option<&[f32]>, padded_k: usize) -> Vec<f32> {
    let mut padded = vec![1.0f32; padded_k];
    if let Some(scale) = scale {
        padded[..scale.len()].copy_from_slice(scale);
    }
    padded
}

fn pack_gate_block(
    block: &mut [u8],
    stripe: usize,
    n_macro: usize,
    group: usize,
    gate: &OpusPackedMatrix,
    up: &OpusPackedMatrix,
) {
    let gate_weights = gate.group_base(group);
    let up_weights = up.group_base(group);
    let gate_scales = gate.group_scales(group);
    let up_scales = up.group_scales(group);
    for ln in 0..6 {
        for kt in 0..16 {
            for kk in 0..16 {
                for nn in 0..16 {
                    let local_col = ln * 16 + nn;
                    let logical_col = (n_macro * COLS + stripe) * 48 + local_col % 48;
                    let weights = if local_col < 48 {
                        gate_weights
                    } else {
                        up_weights
                    };
                    let value = weights[(kt * 16 + kk) * INTERMEDIATE + logical_col];
                    let index = (ln * 16 + kt) * 256 + kk * 16 + nn;
                    let nibble = (value & 0x0f) as u8;
                    block[index / 2] |= if index % 2 == 0 { nibble } else { nibble << 4 };
                }
            }
        }
    }
    for local_col in 0..W_COLS {
        let logical_col = (n_macro * COLS + stripe) * 48 + local_col % 48;
        let scales = if local_col < 48 {
            gate_scales
        } else {
            up_scales
        };
        let offset = W_DATA + local_col * size_of::<f32>();
        block[offset..offset + size_of::<f32>()]
            .copy_from_slice(&scales[logical_col].to_ne_bytes());
    }
}

#[allow(clippy::too_many_arguments)]
fn pack_down_block(
    block: &mut [u8],
    stripe: usize,
    group: usize,
    down: &OpusPackedMatrix,
    awq: &[f32],
    signs1: &[f32],
    signs2: &[f32],
) {
    let weights = down.group_base(group);
    let scales = down.group_scales(group);
    for ln in 0..6 {
        for kt in 0..16 {
            for kk in 0..16 {
                for nn in 0..16 {
                    let col = stripe * 96 + ln * 16 + nn;
                    let value = weights[(kt * 16 + kk) * OUTPUT + col];
                    let index = (ln * 16 + kt) * 256 + kk * 16 + nn;
                    let nibble = (value & 0x0f) as u8;
                    block[index / 2] |= if index % 2 == 0 { nibble } else { nibble << 4 };
                }
            }
        }
    }
    for local_col in 0..W_COLS {
        let col = stripe * W_COLS + local_col;
        let offset = W_DATA + local_col * size_of::<f32>();
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
