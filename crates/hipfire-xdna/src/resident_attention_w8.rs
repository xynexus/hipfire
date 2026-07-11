// SPDX-License-Identifier: Apache-2.0

//! Resident EmbeddingGemma dense-W8 QKV projection and bidirectional attention.
//!
//! Native OQ8 and arbitrary compact mixed Opus share this executor. Mixed
//! groups are expanded once during upload through [`OpusPackedMatrix::group_dense_i8`];
//! `+` and `++` remain the same runtime encoding and retain their AWQ sidecar.

use std::borrow::Cow;

use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};

use crate::{
    DeviceBuffer, EmbeddingGemmaAttentionLayout as AttentionLayout, NpuKernel, OpusPackedMatrix,
    OpusResidentMode, XdnaError,
};

const M: usize = 256;
const K: usize = 768;
const N: usize = 1280;
const GROUP: usize = 256;
const GROUPS: usize = 3;
const COLS: usize = 8;
const ROW_STRIPES: usize = 4;
const M_MACROS: usize = 3;
const N_MACROS: usize = 5;
const OUTBLOCKS: usize = M_MACROS * N_MACROS;
const INBLOCKS: usize = GROUPS * OUTBLOCKS;
const A_BLOCK: usize = 16384;
const W_BLOCK: usize = 16384;
const W_DATA: usize = 8192;
const R_PAIR: usize = 16384;
const PAIRS_PER_ROLE: usize = 48;
const ROLES: usize = 5;
const R_STAGE_BYTES: usize = ROLES * PAIRS_PER_ROLE * R_PAIR;
const PARAM_OFFSET: usize = 8192;
const MAX_CONTEXT_COMMANDS: usize = 1_000;

pub struct NpuResidentAttentionDenseW8Weights {
    weights: DeviceBuffer,
    staging: DeviceBuffer,
    awq_scale: Option<Vec<f32>>,
}

impl NpuResidentAttentionDenseW8Weights {
    pub fn awq_scale(&self) -> Option<&[f32]> {
        self.awq_scale.as_deref()
    }
}

pub struct NpuResidentAttentionDenseW8 {
    kernel: NpuKernel,
    input: DeviceBuffer,
    queries: DeviceBuffer,
    key_values: DeviceBuffer,
    primed: bool,
    context_commands: usize,
}

impl NpuResidentAttentionDenseW8 {
    pub fn load_cached(cache: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{cache}/shape.txt")).map_err(XdnaError::Open)?;
        for required in [
            "op=resident-qkv-attention",
            "mode=w8-scaled",
            "m=256",
            "k=768",
            "n=1280",
            "roles=q0,q1,q2,k,v",
        ] {
            if !manifest.lines().any(|line| line == required) {
                return Err(invalid(format!(
                    "resident dense-W8 attention cache missing {required}"
                )));
            }
        }
        let kernel = NpuKernel::load(
            &std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?,
            &std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?,
        )?;
        let input = kernel.alloc_arg(Self::input_bytes())?;
        let queries = kernel.alloc_arg(AttentionLayout::Q_BYTES)?;
        let key_values = kernel.alloc_arg(AttentionLayout::KV_BYTES)?;
        Ok(Self {
            kernel,
            input,
            queries,
            key_values,
            primed: false,
            context_commands: 0,
        })
    }

    pub const fn rows() -> usize {
        M
    }

    pub const fn input_bytes() -> usize {
        ROW_STRIPES * INBLOCKS * A_BLOCK
    }

    pub const fn staging_bytes() -> usize {
        R_STAGE_BYTES + AttentionLayout::OUTPUT_BYTES
    }

    pub const fn output_bytes() -> usize {
        AttentionLayout::OUTPUT_BYTES
    }

    pub fn attach_shared_input(
        &mut self,
        input_fd: i32,
        input_bytes: usize,
    ) -> Result<(), XdnaError> {
        if input_bytes != Self::input_bytes() {
            return Err(invalid("resident attention shared input size mismatch"));
        }
        self.input = self.kernel.import_dmabuf(input_fd, input_bytes, true)?;
        self.primed = false;
        self.context_commands = 0;
        Ok(())
    }

    pub fn upload_weights(
        &self,
        qkv: &OpusPackedMatrix,
        qnorm: &[f32],
        knorm: &[f32],
        epsilon: f32,
        rope_base: f32,
    ) -> Result<NpuResidentAttentionDenseW8Weights, XdnaError> {
        validate_qkv(qkv)?;
        if qnorm.len() != AttentionLayout::HEAD_DIM
            || knorm.len() != AttentionLayout::HEAD_DIM
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || !rope_base.is_finite()
            || rope_base <= 0.0
        {
            return Err(invalid("invalid resident attention norm/RoPE parameters"));
        }
        let groups = dense_groups(qkv);
        let group_refs = groups.iter().map(Cow::as_ref).collect::<Vec<_>>();
        let scales = (0..GROUPS)
            .map(|group| qkv.group_scales(group))
            .collect::<Vec<_>>();
        self.upload_dense_groups(
            &group_refs,
            &scales,
            qkv.awq_scale(),
            qnorm,
            knorm,
            epsilon,
            rope_base,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upload_dense_groups(
        &self,
        groups: &[&[i8]],
        scales: &[&[f32]],
        awq_scale: Option<&[f32]>,
        qnorm: &[f32],
        knorm: &[f32],
        epsilon: f32,
        rope_base: f32,
    ) -> Result<NpuResidentAttentionDenseW8Weights, XdnaError> {
        if groups.len() != GROUPS
            || scales.len() != GROUPS
            || groups.iter().any(|group| group.len() != GROUP * N)
            || scales.iter().any(|scale| scale.len() != N)
            || awq_scale.is_some_and(|scale| scale.len() != K)
            || qnorm.len() != AttentionLayout::HEAD_DIM
            || knorm.len() != AttentionLayout::HEAD_DIM
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || !rope_base.is_finite()
            || rope_base <= 0.0
        {
            return Err(invalid(
                "invalid resident attention dense-group geometry or parameters",
            ));
        }
        let packed = pack_dense_weights(groups, scales);
        let mut weights = self.kernel.alloc_arg(packed.len())?;
        weights.as_mut_slice().copy_from_slice(&packed);
        self.kernel.sync_to_device(&weights)?;

        let staged = stage_positions_and_params(qnorm, knorm, epsilon, rope_base);
        let mut staging = self.kernel.alloc_arg(staged.len())?;
        staging.as_mut_slice().copy_from_slice(&staged);
        self.kernel.sync_to_device(&staging)?;
        Ok(NpuResidentAttentionDenseW8Weights {
            weights,
            staging,
            awq_scale: awq_scale.map(<[f32]>::to_vec),
        })
    }

    /// Pack canonical int8/FWHT activations and per-group row scales into R30's
    /// physical input. AWQ, signs, FWHT and quantization are producer concerns;
    /// this method is the CPU oracle for the GPU/shared-buffer producer.
    pub fn prepack_activations(activations: &[i8], scales: &[f32]) -> Result<Vec<u8>, XdnaError> {
        if activations.len() != M * K || scales.len() != GROUPS * M {
            return Err(invalid("resident attention activation geometry mismatch"));
        }
        let mut packed = vec![0u8; Self::input_bytes()];
        for stripe in 0..ROW_STRIPES {
            for m_macro in 0..M_MACROS {
                for n_macro in 0..N_MACROS {
                    let outblock = m_macro * N_MACROS + n_macro;
                    for group in 0..GROUPS {
                        let block = outblock * GROUPS + group;
                        let base = (stripe * INBLOCKS + block) * A_BLOCK;
                        for lm in 0..3 {
                            for kt in 0..32 {
                                for local_row in 0..8 {
                                    let row = m_macro * 96 + stripe * 24 + lm * 8 + local_row;
                                    if row < M {
                                        let source = row * K + group * GROUP + kt * 8;
                                        let target = base + (lm * 32 + kt) * 64 + local_row * 8;
                                        packed[target..target + 8].copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(
                                                activations[source..source + 8].as_ptr().cast(),
                                                8,
                                            )
                                        });
                                    }
                                }
                            }
                        }
                        for local_row in 0..24 {
                            let row = m_macro * 96 + stripe * 24 + local_row;
                            let scale = if row < M {
                                scales[group * M + row]
                            } else {
                                0.0
                            };
                            let offset = base + 6144 + local_row * size_of::<f32>();
                            packed[offset..offset + size_of::<f32>()]
                                .copy_from_slice(&scale.to_ne_bytes());
                        }
                    }
                }
            }
        }
        Ok(packed)
    }

    pub fn set_prepacked_input(&mut self, packed: &[u8]) -> Result<(), XdnaError> {
        if packed.len() != Self::input_bytes() {
            return Err(invalid("resident attention packed input size mismatch"));
        }
        self.input.as_mut_slice().copy_from_slice(packed);
        Ok(())
    }

    pub fn run_shared_to_device(
        &mut self,
        weights: &NpuResidentAttentionDenseW8Weights,
    ) -> Result<(), XdnaError> {
        if weights.weights.len() != COLS * INBLOCKS * W_BLOCK
            || weights.staging.len() != Self::staging_bytes()
        {
            return Err(invalid("resident attention argument geometry mismatch"));
        }
        if self.context_commands >= MAX_CONTEXT_COMMANDS {
            self.kernel.recreate_hwctx()?;
            self.primed = false;
            self.context_commands = 0;
        }
        if !self.primed {
            self.kernel.dispatch_synced(
                &[
                    &self.input,
                    &weights.weights,
                    &weights.staging,
                    &self.queries,
                    &self.key_values,
                ],
                &[true, false, false, false, false],
            )?;
            self.context_commands += 1;
            self.primed = true;
        }
        self.kernel.dispatch_synced(
            &[
                &self.input,
                &weights.weights,
                &weights.staging,
                &self.queries,
                &self.key_values,
            ],
            &[false, false, false, false, false],
        )?;
        self.context_commands += 1;
        Ok(())
    }

    pub fn read_output_bf16(
        &self,
        weights: &NpuResidentAttentionDenseW8Weights,
    ) -> Result<Vec<u16>, XdnaError> {
        if weights.staging.len() != Self::staging_bytes() {
            return Err(invalid("resident attention staging size mismatch"));
        }
        self.kernel.sync_output(&weights.staging)?;
        AttentionLayout::unpack_output_bf16(&weights.staging.as_slice()[R_STAGE_BYTES..])
            .ok_or_else(|| invalid("invalid resident attention physical output"))
    }

    pub fn read_output_f32(
        &self,
        weights: &NpuResidentAttentionDenseW8Weights,
    ) -> Result<Vec<f32>, XdnaError> {
        Ok(self
            .read_output_bf16(weights)?
            .into_iter()
            .map(bf16_bits_to_f32)
            .collect())
    }
}

fn validate_qkv(matrix: &OpusPackedMatrix) -> Result<(), XdnaError> {
    if matrix.resident_mode() != OpusResidentMode::DenseW8
        || matrix.k() != K
        || matrix.n() != N
        || matrix.group_count() != GROUPS
    {
        return Err(invalid(format!(
            "resident dense-W8 attention wants K={K} N={N} groups={GROUPS}, got {:?} K={} N={} groups={}",
            matrix.resident_mode(),
            matrix.k(),
            matrix.n(),
            matrix.group_count()
        )));
    }
    Ok(())
}

fn dense_groups(matrix: &OpusPackedMatrix) -> Vec<Cow<'_, [i8]>> {
    (0..matrix.group_count())
        .map(|group| matrix.group_dense_i8(group))
        .collect()
}

fn pack_dense_weights(groups: &[&[i8]], scales: &[&[f32]]) -> Vec<u8> {
    let mut packed = vec![0u8; COLS * INBLOCKS * W_BLOCK];
    for stripe in 0..COLS {
        for m_macro in 0..M_MACROS {
            for n_macro in 0..N_MACROS {
                let outblock = m_macro * N_MACROS + n_macro;
                for group in 0..GROUPS {
                    let block = outblock * GROUPS + group;
                    let base = (stripe * INBLOCKS + block) * W_BLOCK;
                    for ln in 0..2 {
                        for kt in 0..32 {
                            for kk in 0..8 {
                                for nn in 0..16 {
                                    let col = n_macro * 256 + stripe * 32 + ln * 16 + nn;
                                    let index =
                                        (ln * 32 + kt) * 128 + (nn / 8) * 64 + kk * 8 + nn % 8;
                                    packed[base + index] =
                                        groups[group][(kt * 8 + kk) * N + col] as u8;
                                }
                            }
                        }
                    }
                    for local_col in 0..32 {
                        let col = n_macro * 256 + stripe * 32 + local_col;
                        let offset = base + W_DATA + local_col * size_of::<f32>();
                        packed[offset..offset + size_of::<f32>()]
                            .copy_from_slice(&scales[group][col].to_ne_bytes());
                    }
                }
            }
        }
    }
    packed
}

fn stage_positions_and_params(
    qnorm: &[f32],
    knorm: &[f32],
    epsilon: f32,
    rope_base: f32,
) -> Vec<u8> {
    let mut staged = vec![0u8; R_STAGE_BYTES + AttentionLayout::OUTPUT_BYTES];
    for role in 0..ROLES {
        for physical_pair in 0..PAIRS_PER_ROLE {
            let base = (role * PAIRS_PER_ROLE + physical_pair) * R_PAIR;
            let m_macro = physical_pair / 16;
            let within = physical_pair % 16;
            let core_row = within / 4;
            let subpair = within % 4;
            if subpair < 3 {
                let token0 = m_macro * 96 + core_row * 24 + subpair * 8;
                for row in 0..8 {
                    let token = token0 + row;
                    if token < M {
                        for dim in 0..AttentionLayout::HEAD_DIM / 2 {
                            let frequency = 1.0
                                / rope_base
                                    .powf((2 * dim) as f32 / AttentionLayout::HEAD_DIM as f32);
                            let angle = token as f32 * frequency;
                            write_u16(
                                &mut staged,
                                base + 4096 + (row * AttentionLayout::HEAD_DIM + dim) * 2,
                                f32_to_bf16_bits(angle.cos()),
                            );
                            write_u16(
                                &mut staged,
                                base + 4096
                                    + (row * AttentionLayout::HEAD_DIM
                                        + AttentionLayout::HEAD_DIM / 2
                                        + dim)
                                        * 2,
                                f32_to_bf16_bits(angle.sin()),
                            );
                        }
                    }
                }
            }
            for (index, &value) in qnorm.iter().enumerate() {
                write_u16(
                    &mut staged,
                    base + PARAM_OFFSET + index * 2,
                    f32_to_bf16_bits(value),
                );
            }
            for (index, &value) in knorm.iter().enumerate() {
                write_u16(
                    &mut staged,
                    base + PARAM_OFFSET + 512 + index * 2,
                    f32_to_bf16_bits(value),
                );
            }
            staged[base + PARAM_OFFSET + 1024..base + PARAM_OFFSET + 1028]
                .copy_from_slice(&epsilon.to_le_bytes());
        }
    }
    staged
}

fn write_u16(destination: &mut [u8], offset: usize, value: u16) {
    destination[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_attention_argument_geometry_matches_r30() {
        assert_eq!(NpuResidentAttentionDenseW8::input_bytes(), 2_949_120);
        assert_eq!(R_STAGE_BYTES, 3_932_160);
        assert_eq!(NpuResidentAttentionDenseW8::staging_bytes(), 4_325_376);
        assert_eq!(COLS * INBLOCKS * W_BLOCK, 5_898_240);
    }

    #[test]
    fn activation_packer_repeats_each_projection_input_over_n_macros() {
        let activations = (0..M * K)
            .map(|index| (index % 127) as i8)
            .collect::<Vec<_>>();
        let scales = (0..GROUPS * M)
            .map(|index| index as f32 * 0.001)
            .collect::<Vec<_>>();
        let packed =
            NpuResidentAttentionDenseW8::prepack_activations(&activations, &scales).unwrap();
        for stripe in 0..ROW_STRIPES {
            for m_macro in 0..M_MACROS {
                for group in 0..GROUPS {
                    let first =
                        ((stripe * INBLOCKS + (m_macro * N_MACROS) * GROUPS + group) * A_BLOCK)..;
                    let reference = &packed[first.start..first.start + A_BLOCK];
                    for n_macro in 1..N_MACROS {
                        let start =
                            (stripe * INBLOCKS + (m_macro * N_MACROS + n_macro) * GROUPS + group)
                                * A_BLOCK;
                        assert_eq!(reference, &packed[start..start + A_BLOCK]);
                    }
                }
            }
        }
    }

    #[test]
    fn staging_keeps_position_and_parameter_tails_outside_projection_prefixes() {
        let qnorm = vec![0.75; AttentionLayout::HEAD_DIM];
        let knorm = vec![1.25; AttentionLayout::HEAD_DIM];
        let staged = stage_positions_and_params(&qnorm, &knorm, 1.0e-6, 10_000.0);
        assert_eq!(staged.len(), NpuResidentAttentionDenseW8::staging_bytes());
        for role in 0..ROLES {
            for pair in 0..PAIRS_PER_ROLE {
                let base = (role * PAIRS_PER_ROLE + pair) * R_PAIR;
                assert!(staged[base..base + 4096].iter().all(|&byte| byte == 0));
                assert_eq!(
                    u16::from_le_bytes([
                        staged[base + PARAM_OFFSET],
                        staged[base + PARAM_OFFSET + 1]
                    ]),
                    f32_to_bf16_bits(0.75)
                );
                assert_eq!(
                    u16::from_le_bytes([
                        staged[base + PARAM_OFFSET + 512],
                        staged[base + PARAM_OFFSET + 513]
                    ]),
                    f32_to_bf16_bits(1.25)
                );
            }
        }
    }
}
