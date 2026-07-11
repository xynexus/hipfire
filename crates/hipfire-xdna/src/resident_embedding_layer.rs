// SPDX-License-Identifier: Apache-2.0

//! R34 resident EmbeddingGemma attention, output projection, residual, and norms.
//!
//! The source Opus encoding is intentionally absent from this type's name. OQ4,
//! compact mixed OQ, and OQ8 all enter through dense signed-byte groups expanded
//! once at upload; `+` and `++` retain their shared AWQ activation sidecar.

use hipfire_primitives::conv::bf16_bits_to_f32;
#[cfg(test)]
use hipfire_primitives::conv::f32_to_bf16_bits;

use crate::attention_output_bf16::dense_effective_bf16;
use crate::resident_attention_w8::{pack_dense_weights, stage_positions_and_params};
use crate::{DeviceBuffer, NpuKernel, OpusPackedMatrix, XdnaError};

const M: usize = 256;
const K: usize = 768;
const QKV_N: usize = 1280;
const GROUPS: usize = 3;
const COLS: usize = 8;
const ACTIVE_COLS: usize = COLS / 2;
const CORE_ROWS: usize = 4;
const BLOCK: usize = 16_384;
const DATA: usize = 8_192;
const QKV_BLOCKS_PER_STRIPE: usize = 45;
const O_BLOCKS_PER_ACTIVE_COL: usize = 72;
const R_STAGE_BYTES: usize = 5 * 48 * BLOCK;
const ATTENTION_BYTES: usize = M * K * size_of::<u16>();
const INPUT_BYTES: usize = ACTIVE_COLS * QKV_BLOCKS_PER_STRIPE * BLOCK;
const QKV_WEIGHT_BYTES: usize = INPUT_BYTES;
const OUTPUT_WEIGHT_BYTES: usize = ACTIVE_COLS * O_BLOCKS_PER_ACTIVE_COL * BLOCK;
const NORM_PARAM_BYTES: usize = ACTIVE_COLS * 2 * CORE_ROWS * BLOCK;
const WEIGHT_BYTES: usize = QKV_WEIGHT_BYTES + OUTPUT_WEIGHT_BYTES + NORM_PARAM_BYTES;
const HIDDEN_BACKING_BYTES: usize = R_STAGE_BYTES + 3 * ATTENTION_BYTES;
const Q_BYTES: usize = 3 * ATTENTION_BYTES;
const KV_BYTES: usize = 2 * ATTENTION_BYTES;
const PAIRED_SCALE_BASE: usize = 6_272;
const SCALE_OFFSET: usize = 8_192;
const SCALE_BYTES: usize = 128;
const ROWS_PER_CORE: usize = 8;
const POST_NORM_OFFSET: usize = ROWS_PER_CORE * K * size_of::<u16>();
const PRE_NORM_OFFSET: usize = POST_NORM_OFFSET + K * size_of::<u16>();
const EPSILON_OFFSET: usize = PRE_NORM_OFFSET + K * size_of::<u16>();
const PRE_INVERSE_BASE: usize = M * K * size_of::<u16>();
const PRE_INVERSE_RECORD_BYTES: usize = ROWS_PER_CORE * K * size_of::<u16>();
const MAX_CONTEXT_COMMANDS: usize = 1_000;

/// Per-layer immutable R34 payload plus the input-scale template that must be
/// restored before packing that layer's dynamic activations.
pub struct NpuEmbeddingLayerAttentionDenseW8Weights {
    weights: DeviceBuffer,
    input_template: Vec<u8>,
    hidden_template: Vec<u8>,
    awq_scale: Option<Vec<f32>>,
}

impl NpuEmbeddingLayerAttentionDenseW8Weights {
    pub fn awq_scale(&self) -> Option<&[f32]> {
        self.awq_scale.as_deref()
    }
}

/// One long-lived R34 hardware context. Input and completed attention state may
/// be replaced with shared dma-bufs so RDNA producers and R35/R39 consumers can
/// use the same pages without host-visible activation copies.
pub struct NpuEmbeddingLayerAttentionDenseW8 {
    kernel: NpuKernel,
    input: DeviceBuffer,
    hidden: DeviceBuffer,
    queries: DeviceBuffer,
    key_values: DeviceBuffer,
    primed: bool,
    context_commands: usize,
}

impl NpuEmbeddingLayerAttentionDenseW8 {
    pub fn load_cached(cache: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{cache}/shape.txt")).map_err(XdnaError::Open)?;
        for field in [
            "op=resident-qkv-paired-attention-output-norm",
            "mode=w8-scaled",
            "m=256",
            "k=768",
            "n=1280",
            "roles=q0,q1,q2,k,v,o",
            "tails=post-attn-norm,residual,pre-ffn-norm",
            "output=canonical-token-major-bf16",
            "handoff=staging-prefix-dmabuf",
            "state=pre-ffn-inverse-f32",
        ] {
            if !manifest.lines().any(|line| line == field) {
                return Err(invalid(format!(
                    "resident layer attention cache missing {field}"
                )));
            }
        }
        let kernel = NpuKernel::load(
            &std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?,
            &std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?,
        )?;
        Ok(Self {
            input: kernel.alloc_arg(INPUT_BYTES)?,
            hidden: kernel.alloc_arg(HIDDEN_BACKING_BYTES)?,
            queries: kernel.alloc_arg(Q_BYTES)?,
            key_values: kernel.alloc_arg(KV_BYTES)?,
            kernel,
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

    pub const fn weight_bytes() -> usize {
        WEIGHT_BYTES
    }

    pub const fn hidden_backing_bytes() -> usize {
        HIDDEN_BACKING_BYTES
    }

    pub fn attach_shared_input(&mut self, fd: i32, bytes: usize) -> Result<(), XdnaError> {
        if bytes != INPUT_BYTES {
            return Err(invalid(
                "resident layer attention shared input size mismatch",
            ));
        }
        self.input = self.kernel.import_dmabuf(fd, bytes, true)?;
        self.reset_context_state();
        Ok(())
    }

    pub fn attach_shared_hidden(&mut self, fd: i32, bytes: usize) -> Result<(), XdnaError> {
        if bytes < HIDDEN_BACKING_BYTES {
            return Err(invalid(
                "resident layer attention shared hidden buffer is too small",
            ));
        }
        self.hidden = self.kernel.import_dmabuf(fd, bytes, true)?;
        self.reset_context_state();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upload_dense_groups(
        &self,
        groups: &[&[i8]],
        scales: &[&[f32]],
        awq_scale: Option<&[f32]>,
        output: &OpusPackedMatrix,
        residual: &[u16],
        qnorm: &[f32],
        knorm: &[f32],
        post_attention_norm: &[u16],
        pre_ffn_norm: &[u16],
        epsilon: f32,
        rope_base: f32,
    ) -> Result<NpuEmbeddingLayerAttentionDenseW8Weights, XdnaError> {
        if groups.len() != GROUPS
            || scales.len() != GROUPS
            || groups.iter().any(|group| group.len() != 256 * QKV_N)
            || scales.iter().any(|scale| scale.len() != QKV_N)
            || awq_scale.is_some_and(|scale| scale.len() != K)
            || residual.len() != M * K
            || qnorm.len() != 256
            || knorm.len() != 256
            || post_attention_norm.len() != K
            || pre_ffn_norm.len() != K
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || !rope_base.is_finite()
            || rope_base <= 0.0
        {
            return Err(invalid(
                "invalid resident layer attention geometry or parameters",
            ));
        }
        if output.k() != K || output.n() != K || output.group_count() != GROUPS {
            return Err(invalid(
                "resident layer output projection must be K=N=768 with 3 groups",
            ));
        }
        let unpacked_qkv = pack_dense_weights(groups, scales);
        let mut input_template = vec![0u8; INPUT_BYTES];
        inject_paired_weight_scales(&mut input_template, &unpacked_qkv);

        let mut packed = pack_paired_weights(&unpacked_qkv);
        packed.extend_from_slice(&pack_output_projection_direct(&dense_effective_bf16(
            output,
        )));
        packed.extend_from_slice(&pack_residual_norm_params(
            residual,
            post_attention_norm,
            pre_ffn_norm,
            epsilon,
        ));
        debug_assert_eq!(packed.len(), WEIGHT_BYTES);
        let mut weights = self.kernel.alloc_arg(WEIGHT_BYTES)?;
        weights.as_mut_slice().copy_from_slice(&packed);
        self.kernel.sync_to_device(&weights)?;

        let mut hidden_template =
            stage_positions_and_params(qnorm, knorm, epsilon, rope_base, false);
        hidden_template.resize(HIDDEN_BACKING_BYTES, 0);

        Ok(NpuEmbeddingLayerAttentionDenseW8Weights {
            weights,
            input_template,
            hidden_template,
            awq_scale: awq_scale.map(<[f32]>::to_vec),
        })
    }

    /// Restore immutable per-layer QKV scales and norm/RoPE staging before the
    /// RDNA producer writes dynamic activations and the R34 command executes.
    pub fn prepare_layer(
        &mut self,
        weights: &NpuEmbeddingLayerAttentionDenseW8Weights,
    ) -> Result<(), XdnaError> {
        if weights.input_template.len() != INPUT_BYTES
            || weights.hidden_template.len() != HIDDEN_BACKING_BYTES
        {
            return Err(invalid("resident layer template size mismatch"));
        }
        copy_paired_weight_scales(self.input.as_mut_slice(), &weights.input_template);
        self.restore_hidden(weights)?;
        self.kernel.sync_to_device(&self.input)?;
        Ok(())
    }

    pub fn set_prepacked_input(
        &mut self,
        weights: &NpuEmbeddingLayerAttentionDenseW8Weights,
        packed: &[u8],
    ) -> Result<(), XdnaError> {
        if packed.len() != INPUT_BYTES {
            return Err(invalid("resident layer prepacked input size mismatch"));
        }
        self.input.as_mut_slice().copy_from_slice(packed);
        copy_paired_weight_scales(self.input.as_mut_slice(), &weights.input_template);
        self.kernel.sync_to_device(&self.input)
    }

    /// Refresh only the dynamic residual rows in the layer-owned R34 argument.
    /// Norm vectors and all projection weights remain resident.
    pub fn set_residual_bf16(
        &self,
        weights: &mut NpuEmbeddingLayerAttentionDenseW8Weights,
        residual: &[u16],
    ) -> Result<(), XdnaError> {
        if residual.len() != M * K || weights.weights.len() != WEIGHT_BYTES {
            return Err(invalid("resident layer residual geometry mismatch"));
        }
        let norm_base = QKV_WEIGHT_BYTES + OUTPUT_WEIGHT_BYTES;
        write_residual_rows(&mut weights.weights.as_mut_slice()[norm_base..], residual);
        self.kernel.sync_to_device(&weights.weights)
    }

    pub fn run_shared(
        &mut self,
        weights: &NpuEmbeddingLayerAttentionDenseW8Weights,
    ) -> Result<(), XdnaError> {
        if weights.weights.len() != WEIGHT_BYTES {
            return Err(invalid("resident layer attention weight size mismatch"));
        }
        if self.context_commands >= MAX_CONTEXT_COMMANDS {
            self.kernel.recreate_hwctx()?;
            self.reset_context_state();
        }
        if !self.primed {
            self.dispatch(weights, true)?;
            self.context_commands += 1;
            // R34 writes its canonical H output over the staging prefix and
            // uses Q/KV scratch. The warm-up command therefore needs the same
            // reset performed by the admitted hardware oracle before the real
            // command can consume valid RoPE/norm parameters.
            self.restore_hidden(weights)?;
            self.queries.as_mut_slice().fill(0);
            self.key_values.as_mut_slice().fill(0);
            self.kernel.sync_to_device(&self.queries)?;
            self.kernel.sync_to_device(&self.key_values)?;
            self.primed = true;
        }
        self.dispatch(weights, false)?;
        self.context_commands += 1;
        Ok(())
    }

    pub fn read_hidden_f32(&self) -> Result<Vec<f32>, XdnaError> {
        self.kernel.sync_output(&self.hidden)?;
        Ok(self.hidden.as_slice()[..M * K * size_of::<u16>()]
            .chunks_exact(size_of::<u16>())
            .map(|word| bf16_bits_to_f32(u16::from_le_bytes([word[0], word[1]])))
            .collect())
    }

    /// Read the per-token inverse RMS exported by R38/R34 immediately after
    /// the canonical H prefix. Physical records follow core-row then column;
    /// map them back to canonical token order for host-side boundary probes.
    pub fn read_pre_inverse_f32(&self) -> Result<Vec<f32>, XdnaError> {
        self.kernel.sync_output(&self.hidden)?;
        let bytes = self.hidden.as_slice();
        let required = PRE_INVERSE_BASE + COLS * CORE_ROWS * PRE_INVERSE_RECORD_BYTES;
        if bytes.len() < required {
            return Err(invalid("resident layer pre-inverse backing is too small"));
        }
        let mut inverse = vec![0.0f32; M];
        for core_row in 0..CORE_ROWS {
            for column in 0..COLS {
                let record = core_row * COLS + column;
                let token_base = (column / 4) * 128 + core_row * 32 + (column % 4) * 8;
                let start = PRE_INVERSE_BASE + record * PRE_INVERSE_RECORD_BYTES;
                for row in 0..ROWS_PER_CORE {
                    let offset = start + row * size_of::<f32>();
                    inverse[token_base + row] = f32::from_le_bytes(
                        bytes[offset..offset + size_of::<f32>()]
                            .try_into()
                            .expect("four-byte inverse"),
                    );
                }
            }
        }
        Ok(inverse)
    }

    fn dispatch(
        &self,
        weights: &NpuEmbeddingLayerAttentionDenseW8Weights,
        sync: bool,
    ) -> Result<(), XdnaError> {
        self.kernel.dispatch_synced(
            &[
                &self.input,
                &weights.weights,
                &self.hidden,
                &self.queries,
                &self.key_values,
            ],
            &[sync, sync, sync, sync, sync],
        )
    }

    fn restore_hidden(
        &mut self,
        weights: &NpuEmbeddingLayerAttentionDenseW8Weights,
    ) -> Result<(), XdnaError> {
        self.hidden
            .as_mut_slice()
            .copy_from_slice(&weights.hidden_template);
        self.kernel.sync_to_device(&self.hidden)
    }

    fn reset_context_state(&mut self) {
        self.primed = false;
        self.context_commands = 0;
    }
}

fn pack_paired_weights(unpaired: &[u8]) -> Vec<u8> {
    debug_assert_eq!(unpaired.len(), COLS * QKV_BLOCKS_PER_STRIPE * BLOCK);
    let mut paired = vec![0u8; QKV_WEIGHT_BYTES];
    for pair in 0..ACTIVE_COLS {
        for block in 0..QKV_BLOCKS_PER_STRIPE {
            let target = (pair * QKV_BLOCKS_PER_STRIPE + block) * BLOCK;
            for lane in 0..2 {
                let source = ((pair * 2 + lane) * QKV_BLOCKS_PER_STRIPE + block) * BLOCK;
                paired[target + lane * DATA..target + (lane + 1) * DATA]
                    .copy_from_slice(&unpaired[source..source + DATA]);
            }
        }
    }
    paired
}

fn inject_paired_weight_scales(input: &mut [u8], unpaired: &[u8]) {
    for row_stripe in 0..CORE_ROWS {
        for block in 0..QKV_BLOCKS_PER_STRIPE {
            let activation = (row_stripe * QKV_BLOCKS_PER_STRIPE + block) * BLOCK;
            for pair in 0..ACTIVE_COLS {
                for lane in 0..2 {
                    let source =
                        ((pair * 2 + lane) * QKV_BLOCKS_PER_STRIPE + block) * BLOCK + SCALE_OFFSET;
                    let target = activation + PAIRED_SCALE_BASE + (pair * 2 + lane) * SCALE_BYTES;
                    input[target..target + SCALE_BYTES]
                        .copy_from_slice(&unpaired[source..source + SCALE_BYTES]);
                }
            }
        }
    }
}

fn copy_paired_weight_scales(destination: &mut [u8], template: &[u8]) {
    for row_stripe in 0..CORE_ROWS {
        for block in 0..QKV_BLOCKS_PER_STRIPE {
            let base = (row_stripe * QKV_BLOCKS_PER_STRIPE + block) * BLOCK + PAIRED_SCALE_BASE;
            destination[base..base + COLS * SCALE_BYTES]
                .copy_from_slice(&template[base..base + COLS * SCALE_BYTES]);
        }
    }
}

fn pack_output_projection_direct(matrix: &[u16]) -> Vec<u8> {
    debug_assert_eq!(matrix.len(), K * K);
    let mut packed = vec![0u8; OUTPUT_WEIGHT_BYTES];
    for active_col in 0..ACTIVE_COLS {
        for slice in 0..24 {
            let column_base = slice * 32;
            for group in 0..GROUPS {
                let block = (active_col * O_BLOCKS_PER_ACTIVE_COL + slice * GROUPS + group) * BLOCK;
                for nt in 0..4 {
                    for kt in 0..32 {
                        for kk in 0..8 {
                            for nn in 0..8 {
                                let k = group * 256 + kt * 8 + kk;
                                let n = column_base + nt * 8 + nn;
                                let target = block + ((nt * 32 + kt) * 64 + kk * 8 + nn) * 2;
                                packed[target..target + 2]
                                    .copy_from_slice(&matrix[k * K + n].to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
    }
    packed
}

fn pack_residual_norm_params(
    residual: &[u16],
    post_attention_norm: &[u16],
    pre_ffn_norm: &[u16],
    epsilon: f32,
) -> Vec<u8> {
    let mut packed = vec![0u8; NORM_PARAM_BYTES];
    write_residual_rows(&mut packed, residual);
    for active_col in 0..ACTIVE_COLS {
        for wave in 0..2 {
            for core_row in 0..CORE_ROWS {
                let block = ((active_col * 2 + wave) * CORE_ROWS + core_row) * BLOCK;
                for hidden in 0..K {
                    write_u16(
                        &mut packed,
                        block + POST_NORM_OFFSET + hidden * 2,
                        post_attention_norm[hidden],
                    );
                    write_u16(
                        &mut packed,
                        block + PRE_NORM_OFFSET + hidden * 2,
                        pre_ffn_norm[hidden],
                    );
                }
                packed[block + EPSILON_OFFSET..block + EPSILON_OFFSET + 4]
                    .copy_from_slice(&epsilon.to_le_bytes());
            }
        }
    }
    packed
}

fn write_residual_rows(destination: &mut [u8], residual: &[u16]) {
    debug_assert_eq!(destination.len(), NORM_PARAM_BYTES);
    debug_assert_eq!(residual.len(), M * K);
    for active_col in 0..ACTIVE_COLS {
        for wave in 0..2 {
            for core_row in 0..CORE_ROWS {
                let block = ((active_col * 2 + wave) * CORE_ROWS + core_row) * BLOCK;
                let token_base = wave * 128 + core_row * 32 + active_col * ROWS_PER_CORE;
                for row in 0..ROWS_PER_CORE {
                    let source = (token_base + row) * K;
                    let target = block + row * K * size_of::<u16>();
                    for hidden in 0..K {
                        write_u16(destination, target + hidden * 2, residual[source + hidden]);
                    }
                }
            }
        }
    }
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
    fn r34_argument_geometry_matches_the_admitted_graph() {
        assert_eq!(INPUT_BYTES, 2_949_120);
        assert_eq!(WEIGHT_BYTES, 8_192_000);
        assert_eq!(HIDDEN_BACKING_BYTES, 5_111_808);
        assert_eq!(NORM_PARAM_BYTES, 524_288);
    }

    #[test]
    fn paired_weight_scales_are_recoverable_without_overwriting_dynamic_input() {
        let mut unpacked = vec![0u8; COLS * QKV_BLOCKS_PER_STRIPE * BLOCK];
        for stripe in 0..COLS {
            for block in 0..QKV_BLOCKS_PER_STRIPE {
                let base = (stripe * QKV_BLOCKS_PER_STRIPE + block) * BLOCK + SCALE_OFFSET;
                unpacked[base..base + SCALE_BYTES].fill((stripe * 17 + block) as u8);
            }
        }
        let mut template = vec![0u8; INPUT_BYTES];
        inject_paired_weight_scales(&mut template, &unpacked);
        let mut dynamic = vec![0x5au8; INPUT_BYTES];
        copy_paired_weight_scales(&mut dynamic, &template);
        for stripe in 0..CORE_ROWS {
            for block in 0..QKV_BLOCKS_PER_STRIPE {
                let base = (stripe * QKV_BLOCKS_PER_STRIPE + block) * BLOCK;
                assert_eq!(dynamic[base], 0x5a);
                for source_col in 0..COLS {
                    let scale = base + PAIRED_SCALE_BASE + source_col * SCALE_BYTES;
                    assert!(dynamic[scale..scale + SCALE_BYTES]
                        .iter()
                        .all(|&value| value == (source_col * 17 + block) as u8));
                }
            }
        }
    }

    #[test]
    fn residual_records_cover_every_token_once_per_norm_copy() {
        let residual = (0..M * K)
            .map(|index| f32_to_bf16_bits(index as f32))
            .collect::<Vec<_>>();
        let post = vec![f32_to_bf16_bits(0.75); K];
        let pre = vec![f32_to_bf16_bits(1.25); K];
        let packed = pack_residual_norm_params(&residual, &post, &pre, 1.0e-6);
        for active_col in 0..ACTIVE_COLS {
            for wave in 0..2 {
                for core_row in 0..CORE_ROWS {
                    let block = ((active_col * 2 + wave) * CORE_ROWS + core_row) * BLOCK;
                    let token_base = wave * 128 + core_row * 32 + active_col * ROWS_PER_CORE;
                    for row in 0..ROWS_PER_CORE {
                        let got = u16::from_le_bytes(
                            packed[block + row * K * 2..block + row * K * 2 + 2]
                                .try_into()
                                .unwrap(),
                        );
                        assert_eq!(got, residual[(token_base + row) * K]);
                    }
                    assert_eq!(
                        &packed[block + EPSILON_OFFSET..block + EPSILON_OFFSET + 4],
                        &1.0e-6f32.to_le_bytes()
                    );
                }
            }
        }
    }

    #[test]
    fn shared_stage_prefix_matches_r39_contract() {
        let staged =
            stage_positions_and_params(&vec![1.0; 256], &vec![1.0; 256], 1.0e-6, 10_000.0, false);
        assert_eq!(staged.len(), R_STAGE_BYTES + ATTENTION_BYTES);
        assert!(HIDDEN_BACKING_BYTES >= staged.len());
        assert_eq!(ATTENTION_BYTES, M * K * 2);
    }

    #[test]
    fn inverse_records_cover_each_token_once() {
        let mut seen = vec![false; M];
        for core_row in 0..CORE_ROWS {
            for column in 0..COLS {
                let token_base = (column / 4) * 128 + core_row * 32 + (column % 4) * 8;
                for row in 0..ROWS_PER_CORE {
                    assert!(!std::mem::replace(&mut seen[token_base + row], true));
                }
            }
        }
        assert!(seen.into_iter().all(|value| value));
    }
}
