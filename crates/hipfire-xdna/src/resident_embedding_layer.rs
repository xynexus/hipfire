// SPDX-License-Identifier: Apache-2.0

//! R34 resident EmbeddingGemma attention, output projection, residual, and norms.
//!
//! The source Opus encoding is intentionally absent from this type's name. OQ4,
//! compact mixed OQ, and OQ8 all enter through dense signed-byte groups expanded
//! once at upload; `+` and `++` retain their shared AWQ activation sidecar.

use std::path::Path;

use hipfire_primitives::conv::bf16_bits_to_f32;
#[cfg(test)]
use hipfire_primitives::conv::f32_to_bf16_bits;

use crate::attention_output_bf16::dense_effective_bf16;
use crate::r34_prepacked;
use crate::resident_attention_w8::{
    pack_dense_weights, stage_positions_and_params, NpuResidentAttentionDenseW8,
};
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
const EXTERNAL_RESIDUAL_BYTES: usize = COLS * CORE_ROWS * BLOCK;
const COMPLETED_BF16X2_BYTES: usize = 288 * 2 * K * size_of::<u16>();
const QKV_WEIGHT_BYTES: usize = INPUT_BYTES;
const OUTPUT_WEIGHT_BYTES: usize = ACTIVE_COLS * O_BLOCKS_PER_ACTIVE_COL * BLOCK;
const NORM_PARAM_BYTES: usize = ACTIVE_COLS * 2 * CORE_ROWS * BLOCK;
const WEIGHT_BYTES: usize = QKV_WEIGHT_BYTES + OUTPUT_WEIGHT_BYTES + NORM_PARAM_BYTES;
const HIDDEN_BACKING_BYTES: usize = R_STAGE_BYTES + 3 * ATTENTION_BYTES;
const Q_BYTES: usize = 3 * ATTENTION_BYTES;
const KV_BYTES: usize = 2 * ATTENTION_BYTES;
const PAIRED_SCALE_BASE: usize = 6_272;
#[cfg(test)]
const SCALE_OFFSET: usize = 8_192;
const SCALE_BYTES: usize = 128;
const ROWS_PER_CORE: usize = 8;
const POST_NORM_OFFSET: usize = ROWS_PER_CORE * K * size_of::<u16>();
const PRE_NORM_OFFSET: usize = POST_NORM_OFFSET + K * size_of::<u16>();
const EPSILON_OFFSET: usize = PRE_NORM_OFFSET + K * size_of::<u16>();
const PRE_INVERSE_BASE: usize = M * K * size_of::<u16>();
const PRE_INVERSE_RECORD_BYTES: usize = ROWS_PER_CORE * K * size_of::<u16>();
const PRE_INVERSE_PLANE_BYTES: usize = COLS * CORE_ROWS * PRE_INVERSE_RECORD_BYTES;
const DIRECT_X_DOCUMENT_BYTES: usize = 288 * K * size_of::<u16>();
const PRE_EXCEPTION_OFFSET: usize = ROWS_PER_CORE * size_of::<f32>();
const ROW_STATE_BYTES: usize = 1_664;
const ROW_STATE_OFFSET: usize = K * size_of::<u16>();
const MAX_CONTEXT_COMMANDS: usize = 1_000;

/// Per-layer immutable R34 payload plus the input-scale template that must be
/// restored before packing that layer's dynamic activations.
pub struct NpuEmbeddingLayerAttentionDenseW8Weights {
    weights: DeviceBuffer,
    input_template: Vec<u8>,
    hidden_template: Vec<u8>,
    awq_scale: Option<Vec<f32>>,
}

/// Per-token state exported after the resident attention tail. Exception
/// images retain the exact F32 inverse and append the one BF16 X component
/// hidden by a zero weight in otherwise unused record space.
pub struct NpuEmbeddingPreFfnState {
    pub inverse: Vec<f32>,
    pub exception: Option<NpuEmbeddingPreFfnException>,
}

pub struct NpuEmbeddingPreFfnException {
    pub column: usize,
    pub x: Vec<u16>,
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
    exception_column: Option<usize>,
    outputs_direct_x: bool,
    output_row_bytes: usize,
    row_state_output: bool,
    external_residual: bool,
    direct_completed_residual: bool,
    activation_offset: usize,
    input_bytes: usize,
    batch: usize,
    primed: bool,
    context_commands: usize,
}

impl NpuEmbeddingLayerAttentionDenseW8 {
    /// Pack canonical int8/FWHT activations and per-group row scales into the
    /// shared R34 input layout. The dynamic 8-KiB activation record is byte
    /// identical to R30 and occupies the prefix of each 16-KiB R34 block; the
    /// remaining tail is reserved for immutable per-layer metadata.
    pub fn prepack_activations(activations: &[i8], scales: &[f32]) -> Result<Vec<u8>, XdnaError> {
        let packed = NpuResidentAttentionDenseW8::prepack_activations(activations, scales)?;
        if packed.len() != INPUT_BYTES {
            return Err(invalid(
                "resident layer compact activation geometry mismatch",
            ));
        }
        Ok(packed)
    }

    pub fn load_cached(cache: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{cache}/shape.txt")).map_err(XdnaError::Open)?;
        for field in [
            "op=resident-qkv-paired-attention-output-norm",
            "mode=w8-scaled",
            "k=768",
            "n=1280",
            "roles=q0,q1,q2,k,v,o",
            "handoff=staging-prefix-dmabuf",
        ] {
            if !manifest.lines().any(|line| line == field) {
                return Err(invalid(format!(
                    "resident layer attention cache missing {field}"
                )));
            }
        }
        let rows = manifest
            .lines()
            .find_map(|line| line.strip_prefix("m="))
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| invalid("resident layer attention cache missing m="))?;
        if rows == 0 || rows % M != 0 {
            return Err(invalid(format!(
                "resident layer attention cache m={rows} must be a positive multiple of {M}"
            )));
        }
        let batch = rows / M;
        let direct_completed_residual = manifest_uses_direct_completed_residual(&manifest);
        if batch > 1
            && (!direct_completed_residual
                || !manifest
                    .lines()
                    .any(|line| line == "attention=block-diagonal-documents")
                || !manifest.lines().any(|line| line == "segment-rows=256"))
        {
            return Err(invalid(
                "batched resident attention requires direct completed residual and 256-row segmentation",
            ));
        }
        let external_residual = direct_completed_residual
            || manifest
                .lines()
                .any(|line| line == "residual-input=shared-activation-tail-r34-bf16-records");
        let row_state_output = manifest
            .lines()
            .any(|line| line == "output=canonical-token-major-x-bf16-row-state")
            && manifest.lines().any(|line| line == "output-row-bytes=1664")
            && manifest
                .lines()
                .any(|line| line == "state=pre-ffn-inverse-f32-row-tail");
        let outputs_direct_x = if row_state_output {
            true
        } else if manifest
            .lines()
            .any(|line| line == "tails=post-attn-norm,residual")
            && manifest
                .lines()
                .any(|line| line == "output=canonical-token-major-x-bf16")
        {
            true
        } else if external_residual
            && manifest
                .lines()
                .any(|line| line == "tails=post-attn-norm,external-residual")
            && manifest
                .lines()
                .any(|line| line == "output=canonical-token-major-x-bf16")
        {
            true
        } else if manifest
            .lines()
            .any(|line| line == "tails=post-attn-norm,residual,pre-ffn-norm")
            && manifest
                .lines()
                .any(|line| line == "output=canonical-token-major-bf16")
        {
            false
        } else {
            return Err(invalid(
                "resident layer attention cache has unsupported output state",
            ));
        };
        let exception_column = if row_state_output {
            None
        } else if manifest
            .lines()
            .any(|line| line == "state=pre-ffn-inverse-f32")
        {
            None
        } else if manifest
            .lines()
            .any(|line| line == "state=pre-ffn-inverse-f32-x-bf16")
        {
            let column = manifest
                .lines()
                .find_map(|line| line.strip_prefix("exception-column="))
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|&column| column < K)
                .ok_or_else(|| invalid("resident layer exception cache has invalid column"))?;
            Some(column)
        } else {
            return Err(invalid("resident layer attention cache has unknown state"));
        };
        let kernel = NpuKernel::load(
            &std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?,
            &std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?,
        )?;
        let activation_offset = if direct_completed_residual {
            batch * COMPLETED_BF16X2_BYTES
        } else {
            0
        };
        let input_bytes = activation_offset
            + batch * INPUT_BYTES
            + if external_residual && !direct_completed_residual {
                batch * EXTERNAL_RESIDUAL_BYTES
            } else {
                0
            };
        Ok(Self {
            input: kernel.alloc_arg(input_bytes)?,
            hidden: kernel.alloc_arg(hidden_backing_bytes_for(batch))?,
            queries: kernel.alloc_arg(batch * Q_BYTES)?,
            key_values: kernel.alloc_arg(batch * KV_BYTES)?,
            kernel,
            exception_column,
            outputs_direct_x,
            output_row_bytes: if row_state_output {
                ROW_STATE_BYTES
            } else {
                K * size_of::<u16>()
            },
            row_state_output,
            external_residual,
            direct_completed_residual,
            activation_offset,
            input_bytes,
            batch,
            primed: false,
            context_commands: 0,
        })
    }

    pub const fn rows() -> usize {
        M
    }

    pub const fn loaded_rows(&self) -> usize {
        M * self.batch
    }

    pub fn input_bytes(&self) -> usize {
        self.input_bytes
    }

    pub const fn activation_bytes() -> usize {
        INPUT_BYTES
    }

    pub const fn loaded_activation_bytes(&self) -> usize {
        INPUT_BYTES * self.batch
    }

    pub const fn external_input_bytes() -> usize {
        INPUT_BYTES + EXTERNAL_RESIDUAL_BYTES
    }

    pub const fn uses_external_residual(&self) -> bool {
        self.external_residual
    }

    pub fn uses_direct_completed_residual(&self) -> bool {
        self.direct_completed_residual
    }

    pub const fn weight_bytes() -> usize {
        WEIGHT_BYTES
    }

    pub const fn hidden_backing_bytes() -> usize {
        HIDDEN_BACKING_BYTES
    }

    pub const fn loaded_hidden_backing_bytes(&self) -> usize {
        hidden_backing_bytes_for(self.batch)
    }

    pub const fn outputs_direct_x(&self) -> bool {
        self.outputs_direct_x
    }

    pub const fn outputs_row_state(&self) -> bool {
        self.row_state_output
    }

    pub fn sync_shared_hidden(&self) -> Result<(), XdnaError> {
        self.kernel.sync_output(&self.hidden)
    }

    pub fn attach_shared_input(&mut self, fd: i32, bytes: usize) -> Result<(), XdnaError> {
        if bytes != self.input_bytes {
            return Err(invalid(
                "resident layer attention shared input size mismatch",
            ));
        }
        self.input = self.kernel.import_dmabuf(fd, bytes, true)?;
        self.reset_context_state();
        Ok(())
    }

    pub fn attach_shared_hidden(&mut self, fd: i32, bytes: usize) -> Result<(), XdnaError> {
        if bytes < self.loaded_hidden_backing_bytes() {
            return Err(invalid(
                "resident layer attention shared hidden buffer is too small",
            ));
        }
        self.hidden = self.kernel.import_dmabuf(fd, bytes, true)?;
        self.reset_context_state();
        Ok(())
    }

    pub fn sync_shared_completed_residual(&self) -> Result<(), XdnaError> {
        if !self.direct_completed_residual {
            return Err(invalid("resident layer has no direct completed residual"));
        }
        self.kernel.sync_to_device(&self.input)
    }

    /// Stage document-padded BF16x2 completed rows ahead of the packed
    /// activation suffix. Documents use independent 288-row physical slots.
    pub fn set_completed_bf16x2(&mut self, completed: &[u8]) -> Result<(), XdnaError> {
        if !self.direct_completed_residual {
            return Err(invalid("resident layer has no direct completed residual"));
        }
        if completed.len() != self.batch * COMPLETED_BF16X2_BYTES {
            return Err(invalid("resident layer completed BF16x2 size mismatch"));
        }
        self.input.as_mut_slice()[..completed.len()].copy_from_slice(completed);
        self.kernel
            .sync_to_device_prefix(&self.input, completed.len())
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
        self.upload_dense_groups_impl(
            None,
            groups,
            scales,
            awq_scale,
            output,
            residual,
            qnorm,
            knorm,
            post_attention_norm,
            pre_ffn_norm,
            epsilon,
            rope_base,
        )
    }

    /// Upload one R34 layer from an architecture-packed derivative. On the
    /// first load the loader writes `<name>.rdna2.hfp`; subsequent loads verify
    /// the source and payload SHA-256 values and skip tensor-block reordering.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_dense_groups_prepacked(
        &self,
        prepacked_path: &Path,
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
        self.upload_dense_groups_impl(
            Some(prepacked_path),
            groups,
            scales,
            awq_scale,
            output,
            residual,
            qnorm,
            knorm,
            post_attention_norm,
            pre_ffn_norm,
            epsilon,
            rope_base,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_dense_groups_impl(
        &self,
        prepacked_path: Option<&Path>,
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
        let input_template = paired_weight_scale_template(scales);
        let output_dense = dense_effective_bf16(output);
        let source_sha256 = prepacked_source_sha256(
            groups,
            &output_dense,
            residual,
            post_attention_norm,
            pre_ffn_norm,
            epsilon,
        );
        let pack = || {
            let unpacked_qkv = pack_dense_weights(groups, scales);
            let mut packed = pack_paired_weights(&unpacked_qkv);
            packed.extend_from_slice(&pack_output_projection_direct(&output_dense));
            packed.extend_from_slice(&pack_residual_norm_params(
                residual,
                post_attention_norm,
                pre_ffn_norm,
                epsilon,
            ));
            packed
        };
        let packed = if let Some(path) = prepacked_path {
            match r34_prepacked::read(path, source_sha256).map_err(invalid)? {
                Some(packed) => packed,
                None => {
                    let packed = pack();
                    r34_prepacked::write(path, source_sha256, &packed).map_err(invalid)?;
                    packed
                }
            }
        } else {
            pack()
        };
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
        for document in 0..self.batch {
            let start = self.activation_offset + document * INPUT_BYTES;
            copy_paired_weight_scales(
                &mut self.input.as_mut_slice()[start..start + INPUT_BYTES],
                &weights.input_template,
            );
        }
        self.restore_hidden(weights)?;
        if self.external_residual {
            self.kernel
                .sync_to_device_prefix(&self.input, self.input_bytes)?;
        } else {
            self.kernel.sync_to_device(&self.input)?;
        }
        Ok(())
    }

    pub fn set_prepacked_input(
        &mut self,
        weights: &NpuEmbeddingLayerAttentionDenseW8Weights,
        packed: &[u8],
    ) -> Result<(), XdnaError> {
        if packed.len() != self.loaded_activation_bytes() {
            return Err(invalid("resident layer prepacked input size mismatch"));
        }
        self.input.as_mut_slice()[self.activation_offset..self.activation_offset + packed.len()]
            .copy_from_slice(packed);
        for document in 0..self.batch {
            let start = self.activation_offset + document * INPUT_BYTES;
            copy_paired_weight_scales(
                &mut self.input.as_mut_slice()[start..start + INPUT_BYTES],
                &weights.input_template,
            );
        }
        if self.external_residual {
            self.kernel
                .sync_to_device_prefix(&self.input, self.input_bytes)
        } else {
            self.kernel.sync_to_device(&self.input)
        }
    }

    /// Refresh only the dynamic residual rows in the layer-owned R34 argument.
    /// Norm vectors and all projection weights remain resident.
    pub fn set_residual_bf16(
        &self,
        weights: &mut NpuEmbeddingLayerAttentionDenseW8Weights,
        residual: &[u16],
    ) -> Result<(), XdnaError> {
        if self.batch != 1 || residual.len() != M * K || weights.weights.len() != WEIGHT_BYTES {
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
            // R34/R44 writes its canonical H or X output over the staging prefix and
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
        let bytes = self.hidden.as_slice();
        let mut output = Vec::with_capacity(self.loaded_rows() * K);
        for document in 0..self.batch {
            let document_base = if self.batch == 1 {
                0
            } else {
                document * DIRECT_X_DOCUMENT_BYTES
            };
            for row in 0..M {
                let start = document_base + row * self.output_row_bytes;
                output.extend(
                    bytes[start..start + K * size_of::<u16>()]
                        .chunks_exact(size_of::<u16>())
                        .map(|word| bf16_bits_to_f32(u16::from_le_bytes([word[0], word[1]]))),
                );
            }
        }
        Ok(output)
    }

    /// Read the per-token inverse RMS exported by R38/R34 immediately after
    /// the canonical H prefix. Physical records follow core-row then column;
    /// map them back to canonical token order for host-side boundary probes.
    pub fn read_pre_inverse_f32(&self) -> Result<Vec<f32>, XdnaError> {
        Ok(self.read_pre_ffn_state()?.inverse)
    }

    /// Read the inverse RMS and, for a fixed-column R42 image, the direct BF16 X
    /// exception packed into the same metadata word.
    pub fn read_pre_ffn_state(&self) -> Result<NpuEmbeddingPreFfnState, XdnaError> {
        self.kernel.sync_output(&self.hidden)?;
        let bytes = self.hidden.as_slice();
        if self.row_state_output {
            let mut inverse = Vec::with_capacity(self.loaded_rows());
            for document in 0..self.batch {
                let document_base = if self.batch == 1 {
                    0
                } else {
                    document * DIRECT_X_DOCUMENT_BYTES
                };
                for row in 0..M {
                    let offset = document_base + row * ROW_STATE_BYTES + ROW_STATE_OFFSET;
                    inverse.push(f32::from_le_bytes(
                        bytes[offset..offset + size_of::<f32>()]
                            .try_into()
                            .expect("four-byte row-state inverse"),
                    ));
                }
            }
            return Ok(NpuEmbeddingPreFfnState {
                inverse,
                exception: None,
            });
        }
        let required = if self.batch == 1 {
            PRE_INVERSE_BASE + PRE_INVERSE_PLANE_BYTES
        } else {
            self.batch * (DIRECT_X_DOCUMENT_BYTES + PRE_INVERSE_PLANE_BYTES)
        };
        if bytes.len() < required {
            return Err(invalid("resident layer pre-inverse backing is too small"));
        }
        let mut inverse = vec![0.0f32; self.loaded_rows()];
        let mut exception_x = self
            .exception_column
            .map(|_| vec![0u16; self.loaded_rows()]);
        for document in 0..self.batch {
            let inverse_base = if self.batch == 1 {
                PRE_INVERSE_BASE
            } else {
                self.batch * DIRECT_X_DOCUMENT_BYTES + document * PRE_INVERSE_PLANE_BYTES
            };
            for core_row in 0..CORE_ROWS {
                for column in 0..COLS {
                    let record = core_row * COLS + column;
                    let token_base =
                        document * M + (column / 4) * 128 + core_row * 32 + (column % 4) * 8;
                    let start = inverse_base + record * PRE_INVERSE_RECORD_BYTES;
                    for row in 0..ROWS_PER_CORE {
                        let (decoded_inverse, decoded_exception) = decode_pre_ffn_record(
                            &bytes[start..start + PRE_INVERSE_RECORD_BYTES],
                            row,
                            self.exception_column.is_some(),
                        );
                        inverse[token_base + row] = decoded_inverse;
                        if let (Some(x), Some(value)) = (&mut exception_x, decoded_exception) {
                            x[token_base + row] = value;
                        }
                    }
                }
            }
        }
        Ok(NpuEmbeddingPreFfnState {
            inverse,
            exception: self
                .exception_column
                .map(|column| NpuEmbeddingPreFfnException {
                    column,
                    x: exception_x.expect("exception X allocated with column"),
                }),
        })
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
        if self.batch == 1 {
            self.hidden
                .as_mut_slice()
                .copy_from_slice(&weights.hidden_template);
        } else {
            self.hidden.as_mut_slice().fill(0);
            let scratch_base = self.batch * (DIRECT_X_DOCUMENT_BYTES + PRE_INVERSE_PLANE_BYTES);
            for document in 0..self.batch {
                let start = scratch_base + document * HIDDEN_BACKING_BYTES;
                self.hidden.as_mut_slice()[start..start + HIDDEN_BACKING_BYTES]
                    .copy_from_slice(&weights.hidden_template);
            }
        }
        self.kernel.sync_to_device(&self.hidden)
    }

    fn reset_context_state(&mut self) {
        self.primed = false;
        self.context_commands = 0;
    }
}

fn decode_pre_ffn_record(record: &[u8], row: usize, has_exception: bool) -> (f32, Option<u16>) {
    let inverse_offset = row * size_of::<f32>();
    let inverse = f32::from_le_bytes(
        record[inverse_offset..inverse_offset + size_of::<f32>()]
            .try_into()
            .expect("four-byte pre-FFN inverse"),
    );
    let exception = has_exception.then(|| {
        let offset = PRE_EXCEPTION_OFFSET + row * size_of::<u32>();
        let word = u32::from_le_bytes(
            record[offset..offset + size_of::<u32>()]
                .try_into()
                .expect("four-byte pre-FFN exception state"),
        );
        word as u16
    });
    (inverse, exception)
}

fn manifest_uses_direct_completed_residual(manifest: &str) -> bool {
    manifest
        .lines()
        .any(|line| line == "residual-input=shared-completed-bf16x2-high")
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

#[cfg(test)]
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

fn paired_weight_scale_template(scales: &[&[f32]]) -> Vec<u8> {
    let mut template = vec![0u8; INPUT_BYTES];
    for row_stripe in 0..CORE_ROWS {
        for block in 0..QKV_BLOCKS_PER_STRIPE {
            let group = block % GROUPS;
            let n_macro = (block / GROUPS) % 5;
            let target_block = (row_stripe * QKV_BLOCKS_PER_STRIPE + block) * BLOCK;
            for source_col in 0..COLS {
                for local_col in 0..32 {
                    let canonical_col = n_macro * 256 + source_col * 32 + local_col;
                    let target = target_block
                        + PAIRED_SCALE_BASE
                        + source_col * SCALE_BYTES
                        + local_col * size_of::<f32>();
                    template[target..target + size_of::<f32>()]
                        .copy_from_slice(&scales[group][canonical_col].to_ne_bytes());
                }
            }
        }
    }
    template
}

fn prepacked_source_sha256(
    groups: &[&[i8]],
    output: &[u16],
    residual: &[u16],
    post_attention_norm: &[u16],
    pre_ffn_norm: &[u16],
    epsilon: f32,
) -> [u8; 32] {
    let capacity = groups.iter().map(|group| group.len()).sum::<usize>()
        + (output.len() + residual.len() + post_attention_norm.len() + pre_ffn_norm.len()) * 2
        + size_of::<f32>();
    let mut source = Vec::with_capacity(capacity);
    for group in groups {
        source.extend(group.iter().map(|&value| value as u8));
    }
    for values in [output, residual, post_attention_norm, pre_ffn_norm] {
        for &value in values {
            source.extend_from_slice(&value.to_le_bytes());
        }
    }
    source.extend_from_slice(&epsilon.to_le_bytes());
    r34_prepacked::sha256_parts(&[&source])
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

const fn hidden_backing_bytes_for(batch: usize) -> usize {
    if batch == 1 {
        HIDDEN_BACKING_BYTES
    } else {
        batch * (DIRECT_X_DOCUMENT_BYTES + PRE_INVERSE_PLANE_BYTES + HIDDEN_BACKING_BYTES)
    }
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r34_argument_geometry_matches_the_admitted_graph() {
        assert_eq!(
            NpuEmbeddingLayerAttentionDenseW8::activation_bytes(),
            2_949_120
        );
        assert_eq!(
            NpuEmbeddingLayerAttentionDenseW8::external_input_bytes(),
            3_473_408
        );
        assert_eq!(WEIGHT_BYTES, 8_192_000);
        assert_eq!(HIDDEN_BACKING_BYTES, 5_111_808);
        assert_eq!(hidden_backing_bytes_for(2), 11_894_784);
        assert_eq!(DIRECT_X_DOCUMENT_BYTES, 442_368);
        assert_eq!(PRE_INVERSE_PLANE_BYTES, 393_216);
        assert_eq!(NORM_PARAM_BYTES, 524_288);
        assert_eq!(COMPLETED_BF16X2_BYTES, 884_736);
    }

    #[test]
    fn r108_manifest_selects_direct_completed_residual() {
        let manifest = "tails=post-attn-norm,external-residual\n\
                        residual-input=shared-completed-bf16x2-high\n\
                        residual-row-stride-bytes=3072\n";
        assert!(manifest_uses_direct_completed_residual(manifest));
    }

    #[test]
    fn shared_activation_packer_preserves_the_r30_block_prefix() {
        let activations = (0..M * K)
            .map(|index| (index as u8).wrapping_mul(29) as i8)
            .collect::<Vec<_>>();
        let scales = (0..GROUPS * M)
            .map(|index| index as f32 * 0.000_125 + 0.25)
            .collect::<Vec<_>>();
        let compact =
            NpuResidentAttentionDenseW8::prepack_activations(&activations, &scales).unwrap();
        let packed =
            NpuEmbeddingLayerAttentionDenseW8::prepack_activations(&activations, &scales).unwrap();
        assert_eq!(packed.len(), INPUT_BYTES);
        assert_eq!(packed, compact);
        for block in 0..CORE_ROWS * QKV_BLOCKS_PER_STRIPE {
            let base = block * BLOCK;
            assert!(packed[base + BLOCK / 2..base + BLOCK]
                .iter()
                .all(|&value| value == 0));
        }
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
    fn direct_scale_template_matches_the_prior_unpaired_layout_oracle() {
        let groups = (0..GROUPS)
            .map(|_| vec![0i8; 256 * QKV_N])
            .collect::<Vec<_>>();
        let scales = (0..GROUPS)
            .map(|group| {
                (0..QKV_N)
                    .map(|column| group as f32 * 10_000.0 + column as f32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let group_refs = groups.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let scale_refs = scales.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let unpaired = pack_dense_weights(&group_refs, &scale_refs);
        let mut prior = vec![0u8; INPUT_BYTES];
        inject_paired_weight_scales(&mut prior, &unpaired);
        assert_eq!(paired_weight_scale_template(&scale_refs), prior);
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

    #[test]
    fn expanded_exception_record_preserves_inverse_and_x() {
        let inverse = 1.375f32;
        let x = f32_to_bf16_bits(-0.625);
        let mut record = vec![0u8; PRE_EXCEPTION_OFFSET + ROWS_PER_CORE * size_of::<u32>()];
        record[3 * size_of::<f32>()..4 * size_of::<f32>()].copy_from_slice(&inverse.to_le_bytes());
        let exception_offset = PRE_EXCEPTION_OFFSET + 3 * size_of::<u32>();
        let exception_word = x as u32;
        record[exception_offset..exception_offset + size_of::<u32>()]
            .copy_from_slice(&exception_word.to_le_bytes());
        let (decoded_inverse, decoded_exception) = decode_pre_ffn_record(&record, 3, true);
        assert_eq!(decoded_inverse, inverse);
        assert_eq!(decoded_exception, Some(x));
        let (plain, no_exception) = decode_pre_ffn_record(&record, 3, false);
        assert_eq!(plain, inverse);
        assert_eq!(no_exception, None);
    }
}
