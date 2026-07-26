//! Complete resident EmbeddingGemma dense-W8 FFN over the R26 AIE2P schedule.
//!
//! Native OQ8 and arbitrary compact mixed Opus matrices share this executor:
//! mixed groups are expanded exactly once while their resident weights are
//! uploaded, leaving one format-independent dense-int8 dispatch contract.

use hipfire_primitives::{conv::bf16_bits_to_f32, fwht::gen_fwht_signs};

use crate::{DeviceBuffer, NpuKernel, OpusPackedMatrix, XdnaError};

const M: usize = 256;
const PAD_M: usize = 288;
const K: usize = 768;
const INTERMEDIATE: usize = 1152;
const PAD_INTERMEDIATE: usize = 1280;
const OUTPUT: usize = 768;
const GROUP: usize = 256;
const GATE_GROUPS: usize = 3;
const DOWN_GROUPS: usize = 5;
const COLS: usize = 8;
const ROW_STRIPES: usize = 4;
const GATE_N_BLOCKS: usize = 6;
const GATE_BLOCKS: usize = 3 * GATE_N_BLOCKS * GATE_GROUPS;
const GATE_REUSE_PARAM_BLOCKS: usize = 3 * GATE_GROUPS;
const DOWN_BLOCKS: usize = 3 * DOWN_GROUPS * 2;
const WEIGHT_BLOCKS: usize = GATE_BLOCKS + DOWN_BLOCKS;
const REUSE_WEIGHT_BLOCKS: usize = GATE_REUSE_PARAM_BLOCKS + GATE_BLOCKS + DOWN_BLOCKS;
const WEIGHT_REPLAY_BLOCKS: usize = GATE_N_BLOCKS * GATE_GROUPS + 2 * DOWN_GROUPS;
const DATA_PAIR: usize = 9216;
const DATA_REPEATS: usize = 4;
const DATA_JOIN: usize = DATA_REPEATS * DATA_PAIR;
const W_BLOCK: usize = 16384;
const W_DATA: usize = 12288;
const W_COLS: usize = 48;
const PARAM_OFFSET: usize = W_DATA + W_COLS * size_of::<f32>();
const PRE_NORM_OFFSET: usize = PARAM_OFFSET + 3 * GROUP * size_of::<f32>();
const STATIONARY_W_BLOCK: usize = PRE_NORM_OFFSET;
const T_ROWS: usize = 296;
const T_STRIDE: usize = 5376;
const CANONICAL_INPUT_BYTES: usize = PAD_M * K * size_of::<u16>();
const PRE_INVERSE_RECORD_BYTES: usize = 12_288;
const PRE_INVERSE_RECORDS: usize = 32;
const PRE_INVERSE_PLANE_BYTES: usize = PRE_INVERSE_RECORDS * PRE_INVERSE_RECORD_BYTES;
const CANONICAL_SCRATCH_BYTES: usize = PAD_M * PAD_INTERMEDIATE * size_of::<u16>();
const CANONICAL_OUTPUT_BYTES: usize = PAD_M * OUTPUT * size_of::<u16>();

// A primed R26 context completed 1,000 measured commands with unchanged final
// parity. Keep a finite evidence-backed bound, counting the prime command, so
// long-lived servers still periodically reset array-local state.
const MAX_CONTEXT_COMMANDS: usize = 1_000;

pub struct NpuResidentFfnDenseW8Weights {
    buffer: DeviceBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuResidentFfnDenseW8IoMode {
    PackedF32,
    CanonicalBf16,
    /// Token-major BF16 input with compensated token-major BF16x2 output.
    /// This preserves the accumulator precision required by a following
    /// post-FFN RMSNorm while retaining the compact canonical input ABI.
    CanonicalBf16Bf16x2Output,
    /// Direct architectural X plus the producer's physical inverse-RMS record
    /// plane. The AIE consumer applies pre-FFN RMSNorm before activation pack.
    DirectXBf16Bf16x2Output,
}

pub struct NpuResidentFfnDenseW8 {
    kernel: NpuKernel,
    io_mode: NpuResidentFfnDenseW8IoMode,
    reuse_gate_activation: bool,
    reuse_weights_across_macros: bool,
    // Number of 256-row documents packed into one dispatch (M = 256 * batch).
    // batch == 1 is the original M256 kernel. The FFN is per-row independent, so
    // batching concatenates documents' rows; every macro slot gets identical
    // weights, buffers scale by `batch`, and the block-index formulas are
    // batch-agnostic (linear in the macro index).
    batch: usize,
    input: DeviceBuffer,
    scratch: DeviceBuffer,
    output: DeviceBuffer,
    primed: bool,
    context_commands: usize,
}

impl NpuResidentFfnDenseW8 {
    pub fn load_cached(cache: &str) -> Result<Self, XdnaError> {
        let xclbin = std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?;
        let insts = std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?;
        let kernel = NpuKernel::load(&xclbin, &insts)?;
        Self::load_cached_on_kernel(cache, kernel)
    }

    /// Load the FFN on a peer hardware context sharing the producer's DRM file
    /// and device heap. This keeps a zero-copy producer/consumer chain in one
    /// GEM-handle namespace while retaining separate near-capacity AIE images.
    pub fn load_cached_peer(cache: &str, peer: &NpuKernel) -> Result<Self, XdnaError> {
        let xclbin = std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?;
        let insts = std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?;
        let kernel = NpuKernel::load_peer(peer, &xclbin, &insts)?;
        Self::load_cached_on_kernel(cache, kernel)
    }

    fn load_cached_on_kernel(cache: &str, kernel: NpuKernel) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{cache}/shape.txt")).map_err(XdnaError::Open)?;
        let io_mode = parse_io_mode(&manifest)?;
        let reuse_gate_activation = manifest
            .lines()
            .any(|line| line == "gate-activation=reuse-quantized-local-fragments");
        let reuse_weights_across_macros = manifest
            .lines()
            .any(|line| line == "weight-reuse=core-weight-stationary");
        if reuse_gate_activation && io_mode != NpuResidentFfnDenseW8IoMode::DirectXBf16Bf16x2Output
        {
            return Err(invalid(
                "resident dense-W8 gate reuse requires the direct-X BF16x2 ABI",
            ));
        }
        if reuse_weights_across_macros
            && (reuse_gate_activation || io_mode != NpuResidentFfnDenseW8IoMode::CanonicalBf16)
        {
            return Err(invalid(
                "resident dense-W8 stationary weights currently require canonical BF16 without gate-activation reuse",
            ));
        }
        for required in ["op=resident_ffn", "k=768", "intermediate=1152", "out=768"] {
            if !manifest.lines().any(|line| line == required) {
                return Err(invalid(format!(
                    "resident dense-W8 FFN cache missing shape field {required}"
                )));
            }
        }
        // Row count M = 256 * batch. Accept any positive multiple of 256 so a
        // cache built with r26_gen.py --batch=N (m=256*N) drives the batched path;
        // m=256 is the original single-document kernel (batch == 1).
        let m_value = manifest
            .lines()
            .find_map(|line| line.strip_prefix("m="))
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| invalid("resident dense-W8 FFN cache missing shape field m="))?;
        if m_value == 0 || m_value % M != 0 {
            return Err(invalid(format!(
                "resident dense-W8 FFN cache m={m_value} must be a positive multiple of {M}"
            )));
        }
        let batch = m_value / M;
        let input = kernel.alloc_arg(input_bytes_for(io_mode, batch))?;
        let scratch = kernel.alloc_arg(scratch_bytes_for(io_mode) * batch)?;
        let output = kernel.alloc_arg(output_bytes_for(io_mode) * batch)?;
        Ok(Self {
            kernel,
            io_mode,
            reuse_gate_activation,
            reuse_weights_across_macros,
            batch,
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

    pub const fn loaded_rows(&self) -> usize {
        M * self.batch
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

    pub const fn canonical_input_bytes() -> usize {
        CANONICAL_INPUT_BYTES
    }

    pub const fn canonical_scratch_bytes() -> usize {
        CANONICAL_SCRATCH_BYTES
    }

    pub const fn canonical_output_bytes() -> usize {
        CANONICAL_OUTPUT_BYTES
    }

    pub const fn io_mode(&self) -> NpuResidentFfnDenseW8IoMode {
        self.io_mode
    }

    pub const fn reuses_gate_activation(&self) -> bool {
        self.reuse_gate_activation
    }

    pub const fn reuses_weights_across_macros(&self) -> bool {
        self.reuse_weights_across_macros
    }

    pub const fn consumes_direct_x(&self) -> bool {
        matches!(
            self.io_mode,
            NpuResidentFfnDenseW8IoMode::DirectXBf16Bf16x2Output
        )
    }

    pub const fn loaded_input_bytes(&self) -> usize {
        input_bytes_for(self.io_mode, self.batch)
    }

    pub const fn loaded_scratch_bytes(&self) -> usize {
        scratch_bytes_for(self.io_mode) * self.batch
    }

    pub const fn loaded_output_bytes(&self) -> usize {
        output_bytes_for(self.io_mode) * self.batch
    }

    pub fn attach_shared_io(
        &mut self,
        input_fd: i32,
        input_bytes: usize,
        output_fd: i32,
        output_bytes: usize,
    ) -> Result<(), XdnaError> {
        if input_bytes != self.loaded_input_bytes() || output_bytes != self.loaded_output_bytes() {
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

    /// Replace the canonical input with a caller-owned dma-buf. A preceding
    /// NPU context can write the same physical pages and this context consumes
    /// them directly without a host-visible copy.
    pub fn attach_shared_input(
        &mut self,
        input_fd: i32,
        input_bytes: usize,
    ) -> Result<(), XdnaError> {
        if input_bytes < self.loaded_input_bytes() {
            return Err(invalid(
                "resident dense-W8 FFN shared input dma-buf is too small",
            ));
        }
        self.input = self.kernel.import_dmabuf(input_fd, input_bytes, true)?;
        self.primed = false;
        self.context_commands = 0;
        Ok(())
    }

    /// Publish caller writes to the shared input before a dispatch. NPU-to-NPU
    /// handoffs do not need this, but a host correctness bridge that rewrites
    /// direct architectural X into normalized H does.
    pub fn sync_shared_input(&self) -> Result<(), XdnaError> {
        self.kernel.sync_to_device(&self.input)
    }

    /// Populate the direct-X ABI without a preceding resident attention image.
    ///
    /// The inverse plane follows the producer's physical 32-record layout.
    /// Each batch item occupies one M288 physical slot: 256 logical rows plus
    /// 32 padding rows. This avoids aliasing rows 256..287 onto the inverse
    /// records used by rows 32..63.
    pub fn write_direct_x_bf16_with_inverse(
        &mut self,
        x: &[u16],
        inverse: &[f32],
    ) -> Result<(), XdnaError> {
        if self.io_mode != NpuResidentFfnDenseW8IoMode::DirectXBf16Bf16x2Output {
            return Err(invalid(
                "resident dense-W8 FFN cache does not accept direct-X state",
            ));
        }
        let rows = M * self.batch;
        if x.len() != rows * K || inverse.len() != rows {
            return Err(invalid(format!(
                "direct-X resident dense-W8 FFN wants {} BF16 values and {rows} inverses, got {} and {}",
                rows * K,
                x.len(),
                inverse.len()
            )));
        }
        let x_bytes = direct_x_storage_bytes(self.batch);
        self.input.as_mut_slice().fill(0);
        for document in 0..self.batch {
            let source = &x[document * M * K..(document + 1) * M * K];
            let destination_base = if self.batch == 1 {
                0
            } else {
                document * PAD_M * K * size_of::<u16>()
            };
            for (destination, value) in self.input.as_mut_slice()
                [destination_base..destination_base + M * K * size_of::<u16>()]
                .chunks_exact_mut(size_of::<u16>())
                .zip(source.iter().copied())
            {
                destination.copy_from_slice(&value.to_le_bytes());
            }
        }
        pack_pre_inverse_plane(
            &mut self.input.as_mut_slice()[x_bytes..x_bytes + self.batch * PRE_INVERSE_PLANE_BYTES],
            inverse,
            self.batch,
        );
        self.kernel.sync_to_device(&self.input)
    }

    /// Replace the canonical output with caller-owned shared pages. The
    /// post-FFN tail can consume and overwrite those pages without a host copy.
    pub fn attach_shared_output(
        &mut self,
        output_fd: i32,
        output_bytes: usize,
    ) -> Result<(), XdnaError> {
        if output_bytes != self.loaded_output_bytes() {
            return Err(invalid(
                "resident dense-W8 FFN shared output dma-buf size mismatch",
            ));
        }
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
        self.upload_weights_inner(gate, up, down, None)
    }

    pub fn upload_weights_with_pre_ffn_norm(
        &self,
        gate: &OpusPackedMatrix,
        up: &OpusPackedMatrix,
        down: &OpusPackedMatrix,
        pre_ffn_norm: &[u16],
    ) -> Result<NpuResidentFfnDenseW8Weights, XdnaError> {
        self.upload_weights_inner(gate, up, down, Some(pre_ffn_norm))
    }

    fn upload_weights_inner(
        &self,
        gate: &OpusPackedMatrix,
        up: &OpusPackedMatrix,
        down: &OpusPackedMatrix,
        pre_ffn_norm: Option<&[u16]>,
    ) -> Result<NpuResidentFfnDenseW8Weights, XdnaError> {
        validate_matrix(gate, K, INTERMEDIATE, GATE_GROUPS, "gate")?;
        validate_matrix(up, K, INTERMEDIATE, GATE_GROUPS, "up")?;
        validate_matrix(down, INTERMEDIATE, OUTPUT, DOWN_GROUPS, "down")?;
        if gate.awq_scale() != up.awq_scale() {
            return Err(invalid(
                "resident dense-W8 FFN gate/up matrices must share one AWQ activation scale",
            ));
        }
        let pre_ffn_norm = match self.io_mode {
            NpuResidentFfnDenseW8IoMode::DirectXBf16Bf16x2Output => {
                let norm = pre_ffn_norm.ok_or_else(|| {
                    invalid("direct-X resident dense-W8 FFN requires pre-FFN norm weights")
                })?;
                if norm.len() != K {
                    return Err(invalid(format!(
                        "direct-X resident dense-W8 FFN wants {K} pre-norm values, got {}",
                        norm.len()
                    )));
                }
                Some(norm)
            }
            _ => None,
        };

        let gate_groups = dense_groups(gate);
        let up_groups = dense_groups(up);
        let down_groups = dense_groups(down);
        let gate_awq = padded_awq(gate.awq_scale(), GATE_GROUPS * GROUP);
        let down_awq = padded_awq(down.awq_scale(), DOWN_GROUPS * GROUP);
        let signs1 = gen_fwht_signs(42, GROUP);
        let signs2 = gen_fwht_signs(1042, GROUP);
        let weight_blocks = packed_weight_blocks(
            self.batch,
            self.reuse_gate_activation,
            self.reuse_weights_across_macros,
        );
        let weight_block_bytes = packed_weight_block_bytes(self.reuse_weights_across_macros);
        let macros = 3 * self.batch;
        let mut packed = vec![0u8; COLS * weight_blocks * weight_block_bytes];
        for stripe in 0..COLS {
            let packed_macros = if self.reuse_weights_across_macros {
                1
            } else {
                macros
            };
            for mblock in 0..packed_macros {
                if self.reuse_gate_activation {
                    for group in 0..GATE_GROUPS {
                        let block = gate_param_block_index(mblock, group);
                        let start = (stripe * weight_blocks + block) * weight_block_bytes;
                        pack_gate_block(
                            &mut packed[start..start + weight_block_bytes],
                            stripe,
                            0,
                            &gate_groups[group],
                            gate.group_scales(group),
                            &up_groups[group],
                            up.group_scales(group),
                            group,
                            &gate_awq,
                            &signs1,
                            &signs2,
                            pre_ffn_norm,
                        );
                    }
                }
                for nblock in 0..GATE_N_BLOCKS {
                    for group in 0..GATE_GROUPS {
                        let block =
                            gate_block_index(self.reuse_gate_activation, mblock, nblock, group);
                        let start = (stripe * weight_blocks + block) * weight_block_bytes;
                        pack_gate_block(
                            &mut packed[start..start + weight_block_bytes],
                            stripe,
                            nblock,
                            &gate_groups[group],
                            gate.group_scales(group),
                            &up_groups[group],
                            up.group_scales(group),
                            group,
                            &gate_awq,
                            &signs1,
                            &signs2,
                            pre_ffn_norm,
                        );
                    }
                }
            }
            for mblock in 0..packed_macros {
                for group in 0..DOWN_GROUPS {
                    for nmacro in 0..2 {
                        let block = if self.reuse_weights_across_macros {
                            replay_down_block_index(group, nmacro)
                        } else {
                            down_block_index(
                                self.io_mode,
                                self.reuse_gate_activation,
                                mblock,
                                group,
                                nmacro,
                                self.batch,
                            )
                        };
                        let start = (stripe * weight_blocks + block) * weight_block_bytes;
                        pack_down_block(
                            &mut packed[start..start + weight_block_bytes],
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

    pub fn run_canonical_bf16(
        &mut self,
        weights: &NpuResidentFfnDenseW8Weights,
        input: &[u16],
    ) -> Result<Vec<f32>, XdnaError> {
        if !matches!(
            self.io_mode,
            NpuResidentFfnDenseW8IoMode::CanonicalBf16
                | NpuResidentFfnDenseW8IoMode::CanonicalBf16Bf16x2Output
        ) {
            return Err(invalid(
                "resident dense-W8 FFN cache does not accept canonical BF16 input",
            ));
        }
        let rows = M * self.batch;
        if input.len() != rows * K {
            return Err(invalid(format!(
                "resident dense-W8 FFN canonical input wants {} BF16 values, got {}",
                rows * K,
                input.len()
            )));
        }
        self.input.as_mut_slice().fill(0);
        for (destination, value) in self.input.as_mut_slice()[..rows * K * size_of::<u16>()]
            .chunks_exact_mut(size_of::<u16>())
            .zip(input.iter().copied())
        {
            destination.copy_from_slice(&value.to_le_bytes());
        }
        self.kernel.sync_to_device(&self.input)?;
        self.run_shared(weights)?;
        self.read_canonical_output_f32()
    }

    pub fn read_canonical_output_f32(&self) -> Result<Vec<f32>, XdnaError> {
        match self.io_mode {
            NpuResidentFfnDenseW8IoMode::CanonicalBf16 => {
                decode_canonical_bf16_rows(self.output.as_slice(), M * self.batch, OUTPUT)
            }
            NpuResidentFfnDenseW8IoMode::CanonicalBf16Bf16x2Output => {
                decode_canonical_bf16x2_rows(self.output.as_slice(), M * self.batch, OUTPUT)
            }
            NpuResidentFfnDenseW8IoMode::DirectXBf16Bf16x2Output => {
                decode_canonical_bf16x2_padded_documents(self.output.as_slice(), self.batch, OUTPUT)
            }
            NpuResidentFfnDenseW8IoMode::PackedF32 => Err(invalid(
                "resident dense-W8 FFN cache does not use canonical row-major output",
            )),
        }
    }

    pub fn read_canonical_intermediate_f32(&self) -> Result<Vec<f32>, XdnaError> {
        if !matches!(
            self.io_mode,
            NpuResidentFfnDenseW8IoMode::CanonicalBf16
                | NpuResidentFfnDenseW8IoMode::CanonicalBf16Bf16x2Output
                | NpuResidentFfnDenseW8IoMode::DirectXBf16Bf16x2Output
        ) {
            return Err(invalid(
                "resident dense-W8 FFN cache has no canonical BF16 intermediate",
            ));
        }
        self.kernel.sync_output(&self.scratch)?;
        if self.io_mode == NpuResidentFfnDenseW8IoMode::DirectXBf16Bf16x2Output && self.batch > 1 {
            decode_canonical_bf16_padded_documents_strided(
                self.scratch.as_slice(),
                self.batch,
                INTERMEDIATE,
                PAD_INTERMEDIATE,
            )
        } else {
            decode_canonical_bf16_rows_strided(
                self.scratch.as_slice(),
                M * self.batch,
                INTERMEDIATE,
                PAD_INTERMEDIATE,
            )
        }
    }

    pub fn run_shared(&mut self, weights: &NpuResidentFfnDenseW8Weights) -> Result<(), XdnaError> {
        let weight_blocks = packed_weight_blocks(
            self.batch,
            self.reuse_gate_activation,
            self.reuse_weights_across_macros,
        );
        let weight_block_bytes = packed_weight_block_bytes(self.reuse_weights_across_macros);
        if weights.buffer.len() != COLS * weight_blocks * weight_block_bytes {
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

fn decode_canonical_bf16_rows(
    bytes: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, XdnaError> {
    let required = rows * columns * size_of::<u16>();
    if bytes.len() < required {
        return Err(invalid(format!(
            "canonical BF16 buffer wants at least {required} bytes, got {}",
            bytes.len()
        )));
    }
    Ok(bytes[..required]
        .chunks_exact(size_of::<u16>())
        .map(|encoded| bf16_bits_to_f32(u16::from_le_bytes([encoded[0], encoded[1]])))
        .collect())
}

fn decode_canonical_bf16x2_rows(
    bytes: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, XdnaError> {
    let required = rows * columns * 3 * size_of::<u16>();
    if bytes.len() < required {
        return Err(invalid(format!(
            "canonical BF16x2 buffer wants at least {required} bytes, got {}",
            bytes.len()
        )));
    }
    let mut output = Vec::with_capacity(rows * columns);
    let row_bytes = columns * 3 * size_of::<u16>();
    for row in bytes[..required].chunks_exact(row_bytes) {
        let (high, low) = row.split_at(columns * size_of::<u16>());
        for (high, low) in high.chunks_exact(2).zip(low.chunks_exact(2)) {
            output.push(
                bf16_bits_to_f32(u16::from_le_bytes([high[0], high[1]]))
                    + bf16_bits_to_f32(u16::from_le_bytes([low[0], low[1]])),
            );
        }
    }
    Ok(output)
}

fn decode_canonical_bf16_rows_strided(
    bytes: &[u8],
    rows: usize,
    columns: usize,
    stride: usize,
) -> Result<Vec<f32>, XdnaError> {
    let required = rows * stride * size_of::<u16>();
    if columns > stride || bytes.len() < required {
        return Err(invalid(format!(
            "strided canonical BF16 buffer wants columns <= {stride} and at least {required} bytes"
        )));
    }
    let mut output = Vec::with_capacity(rows * columns);
    for row in 0..rows {
        let start = row * stride * size_of::<u16>();
        for encoded in bytes[start..start + columns * size_of::<u16>()].chunks_exact(2) {
            output.push(bf16_bits_to_f32(u16::from_le_bytes([
                encoded[0], encoded[1],
            ])));
        }
    }
    Ok(output)
}

fn decode_canonical_bf16x2_padded_documents(
    bytes: &[u8],
    batch: usize,
    columns: usize,
) -> Result<Vec<f32>, XdnaError> {
    let row_bytes = columns * 3 * size_of::<u16>();
    let required = batch * PAD_M * row_bytes;
    if bytes.len() < required {
        return Err(invalid(format!(
            "document-padded canonical BF16x2 buffer wants at least {required} bytes, got {}",
            bytes.len()
        )));
    }
    let mut output = Vec::with_capacity(batch * M * columns);
    for document in 0..batch {
        for token in 0..M {
            let start = (document * PAD_M + token) * row_bytes;
            let row = &bytes[start..start + row_bytes];
            let (high, low) = row.split_at(columns * size_of::<u16>());
            for (high, low) in high.chunks_exact(2).zip(low.chunks_exact(2)) {
                output.push(
                    bf16_bits_to_f32(u16::from_le_bytes([high[0], high[1]]))
                        + bf16_bits_to_f32(u16::from_le_bytes([low[0], low[1]])),
                );
            }
        }
    }
    Ok(output)
}

fn decode_canonical_bf16_padded_documents_strided(
    bytes: &[u8],
    batch: usize,
    columns: usize,
    stride: usize,
) -> Result<Vec<f32>, XdnaError> {
    let row_bytes = stride * size_of::<u16>();
    let required = batch * PAD_M * row_bytes;
    if columns > stride || bytes.len() < required {
        return Err(invalid(format!(
            "document-padded strided canonical BF16 buffer wants columns <= {stride} and at least {required} bytes"
        )));
    }
    let mut output = Vec::with_capacity(batch * M * columns);
    for document in 0..batch {
        for token in 0..M {
            let start = (document * PAD_M + token) * row_bytes;
            for encoded in bytes[start..start + columns * size_of::<u16>()].chunks_exact(2) {
                output.push(bf16_bits_to_f32(u16::from_le_bytes([
                    encoded[0], encoded[1],
                ])));
            }
        }
    }
    Ok(output)
}

const fn direct_x_storage_bytes(batch: usize) -> usize {
    if batch == 1 {
        M * K * size_of::<u16>()
    } else {
        batch * PAD_M * K * size_of::<u16>()
    }
}

const fn direct_x_input_bytes(batch: usize) -> usize {
    direct_x_storage_bytes(batch) + batch * PRE_INVERSE_PLANE_BYTES
}

const fn input_bytes_for(mode: NpuResidentFfnDenseW8IoMode, batch: usize) -> usize {
    match mode {
        NpuResidentFfnDenseW8IoMode::PackedF32 => ROW_STRIPES * GATE_BLOCKS * DATA_JOIN,
        NpuResidentFfnDenseW8IoMode::CanonicalBf16
        | NpuResidentFfnDenseW8IoMode::CanonicalBf16Bf16x2Output => CANONICAL_INPUT_BYTES * batch,
        NpuResidentFfnDenseW8IoMode::DirectXBf16Bf16x2Output => direct_x_input_bytes(batch),
    }
}

const fn scratch_bytes_for(mode: NpuResidentFfnDenseW8IoMode) -> usize {
    match mode {
        NpuResidentFfnDenseW8IoMode::PackedF32 => T_ROWS * T_STRIDE * size_of::<f32>(),
        NpuResidentFfnDenseW8IoMode::CanonicalBf16
        | NpuResidentFfnDenseW8IoMode::CanonicalBf16Bf16x2Output
        | NpuResidentFfnDenseW8IoMode::DirectXBf16Bf16x2Output => CANONICAL_SCRATCH_BYTES,
    }
}

const fn output_bytes_for(mode: NpuResidentFfnDenseW8IoMode) -> usize {
    match mode {
        NpuResidentFfnDenseW8IoMode::PackedF32 => PAD_M * OUTPUT * size_of::<f32>(),
        NpuResidentFfnDenseW8IoMode::CanonicalBf16 => CANONICAL_OUTPUT_BYTES,
        NpuResidentFfnDenseW8IoMode::CanonicalBf16Bf16x2Output
        | NpuResidentFfnDenseW8IoMode::DirectXBf16Bf16x2Output => {
            PAD_M * OUTPUT * 3 * size_of::<u16>()
        }
    }
}

fn parse_io_mode(manifest: &str) -> Result<NpuResidentFfnDenseW8IoMode, XdnaError> {
    if manifest.lines().any(|line| line == "mode=dense-w8") {
        return Ok(NpuResidentFfnDenseW8IoMode::PackedF32);
    }
    if manifest
        .lines()
        .any(|line| line == "mode=dense-w8-canonical-bf16")
        && manifest
            .lines()
            .any(|line| line == "input=token-major-bf16")
    {
        return Ok(NpuResidentFfnDenseW8IoMode::CanonicalBf16);
    }
    if manifest
        .lines()
        .any(|line| line == "mode=dense-w8-canonical-bf16-bf16x2-output")
        && manifest
            .lines()
            .any(|line| line == "input=token-major-bf16")
        && manifest
            .lines()
            .any(|line| line == "output=token-major-bf16x2")
    {
        return Ok(NpuResidentFfnDenseW8IoMode::CanonicalBf16Bf16x2Output);
    }
    if manifest
        .lines()
        .any(|line| line == "mode=dense-w8-direct-x-pre-norm-bf16x2-output")
        && manifest
            .lines()
            .any(|line| line == "input=token-major-x-bf16")
        && manifest
            .lines()
            .any(|line| line == "input-state=pre-ffn-inverse-f32-physical-records")
        && manifest
            .lines()
            .any(|line| line == "output=token-major-bf16x2")
    {
        return Ok(NpuResidentFfnDenseW8IoMode::DirectXBf16Bf16x2Output);
    }
    Err(invalid(
        "resident dense-W8 FFN cache has no supported mode/input contract",
    ))
}

const fn packed_weight_blocks(
    batch: usize,
    reuse_gate_activation: bool,
    reuse_weights_across_macros: bool,
) -> usize {
    if reuse_weights_across_macros {
        WEIGHT_REPLAY_BLOCKS
    } else {
        batch
            * if reuse_gate_activation {
                REUSE_WEIGHT_BLOCKS
            } else {
                WEIGHT_BLOCKS
            }
    }
}

const fn packed_weight_block_bytes(reuse_weights_across_macros: bool) -> usize {
    if reuse_weights_across_macros {
        STATIONARY_W_BLOCK
    } else {
        W_BLOCK
    }
}

fn pack_pre_inverse_plane(destination: &mut [u8], inverse: &[f32], batch: usize) {
    debug_assert_eq!(destination.len(), batch * PRE_INVERSE_PLANE_BYTES);
    debug_assert_eq!(inverse.len(), batch * M);
    for document in 0..batch {
        for token in 0..M {
            let logical_row = document * M + token;
            let wave = token >> 7;
            let within_wave = token & 127;
            let source_core_row = within_wave >> 5;
            let within_core = within_wave & 31;
            let column = wave * 4 + (within_core >> 3);
            let lane = within_core & 7;
            let record = source_core_row * 8 + column;
            let offset = document * PRE_INVERSE_PLANE_BYTES
                + record * PRE_INVERSE_RECORD_BYTES
                + lane * size_of::<f32>();
            destination[offset..offset + size_of::<f32>()]
                .copy_from_slice(&inverse[logical_row].to_le_bytes());
        }
    }
}

const fn replay_down_block_index(group: usize, nmacro: usize) -> usize {
    GATE_N_BLOCKS * GATE_GROUPS + nmacro * DOWN_GROUPS + group
}

const fn down_block_index(
    mode: NpuResidentFfnDenseW8IoMode,
    reuse_gate_activation: bool,
    mblock: usize,
    group: usize,
    nmacro: usize,
    batch: usize,
) -> usize {
    // Down blocks follow every macro's gate blocks; the gate-block count scales
    // with batch (3*batch macros), so does this offset.
    let gate_blocks = batch
        * if reuse_gate_activation {
            GATE_REUSE_PARAM_BLOCKS + GATE_BLOCKS
        } else {
            GATE_BLOCKS
        };
    gate_blocks
        + match mode {
            NpuResidentFfnDenseW8IoMode::PackedF32 => (mblock * DOWN_GROUPS + group) * 2 + nmacro,
            NpuResidentFfnDenseW8IoMode::CanonicalBf16
            | NpuResidentFfnDenseW8IoMode::CanonicalBf16Bf16x2Output
            | NpuResidentFfnDenseW8IoMode::DirectXBf16Bf16x2Output => {
                (mblock * 2 + nmacro) * DOWN_GROUPS + group
            }
        }
}

const fn gate_param_block_index(mblock: usize, group: usize) -> usize {
    mblock * (GATE_GROUPS + GATE_N_BLOCKS * GATE_GROUPS) + group
}

const fn gate_block_index(
    reuse_gate_activation: bool,
    mblock: usize,
    nblock: usize,
    group: usize,
) -> usize {
    if reuse_gate_activation {
        mblock * (GATE_GROUPS + GATE_N_BLOCKS * GATE_GROUPS)
            + GATE_GROUPS
            + nblock * GATE_GROUPS
            + group
    } else {
        (mblock * GATE_N_BLOCKS + nblock) * GATE_GROUPS + group
    }
}

fn validate_matrix(
    matrix: &OpusPackedMatrix,
    k: usize,
    n: usize,
    groups: usize,
    role: &str,
) -> Result<(), XdnaError> {
    if matrix.k() != k || matrix.n() != n || matrix.group_count() != groups {
        return Err(invalid(format!(
            "resident dense execution FFN {role} wants K={k} N={n} groups={groups}, got {:?} K={} N={} groups={}",
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
    group: usize,
    awq: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    pre_ffn_norm: Option<&[u16]>,
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
    let mut params = Vec::with_capacity(3 * GROUP);
    params.extend_from_slice(&awq[group * GROUP..(group + 1) * GROUP]);
    params.extend_from_slice(signs1);
    params.extend_from_slice(signs2);
    block[PARAM_OFFSET..PARAM_OFFSET + params.len() * size_of::<f32>()]
        .copy_from_slice(unsafe { as_bytes(&params) });
    if let Some(pre_ffn_norm) = pre_ffn_norm {
        let start = group * GROUP;
        for (destination, &value) in block
            [PRE_NORM_OFFSET..PRE_NORM_OFFSET + GROUP * size_of::<u16>()]
            .chunks_exact_mut(size_of::<u16>())
            .zip(&pre_ffn_norm[start..start + GROUP])
        {
            destination.copy_from_slice(&value.to_le_bytes());
        }
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
    fn dense_execution_contract_accepts_native_w4_source_groups() {
        let payload = vec![0u8; INTERMEDIATE * GATE_GROUPS * 130];
        let matrix = OpusPackedMatrix::from_payload(33, K, INTERMEDIATE, &payload, None)
            .expect("native W4 matrix");
        validate_matrix(&matrix, K, INTERMEDIATE, GATE_GROUPS, "gate")
            .expect("dense execution expands native W4 groups");
    }

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

    #[test]
    fn canonical_bf16_argument_geometry_matches_r35() {
        assert_eq!(NpuResidentFfnDenseW8::canonical_input_bytes(), 442_368);
        assert_eq!(NpuResidentFfnDenseW8::canonical_scratch_bytes(), 737_280);
        assert_eq!(NpuResidentFfnDenseW8::canonical_output_bytes(), 442_368);
        assert_eq!(direct_x_input_bytes(1), 786_432);
    }

    #[test]
    fn manifest_selects_packed_bf16_or_precision_preserving_contract() {
        assert_eq!(
            parse_io_mode("op=resident_ffn\nmode=dense-w8\n").unwrap(),
            NpuResidentFfnDenseW8IoMode::PackedF32
        );
        assert_eq!(
            parse_io_mode(
                "op=resident_ffn\nmode=dense-w8-canonical-bf16\ninput=token-major-bf16\n"
            )
            .unwrap(),
            NpuResidentFfnDenseW8IoMode::CanonicalBf16
        );
        assert_eq!(
            parse_io_mode(
                "op=resident_ffn\nmode=dense-w8-canonical-bf16-bf16x2-output\ninput=token-major-bf16\noutput=token-major-bf16x2\n"
            )
            .unwrap(),
            NpuResidentFfnDenseW8IoMode::CanonicalBf16Bf16x2Output
        );
        assert_eq!(
            parse_io_mode(
                "op=resident_ffn\nmode=dense-w8-direct-x-pre-norm-bf16x2-output\ninput=token-major-x-bf16\ninput-state=pre-ffn-inverse-f32-physical-records\noutput=token-major-bf16x2\n"
            )
            .unwrap(),
            NpuResidentFfnDenseW8IoMode::DirectXBf16Bf16x2Output
        );
        assert!(parse_io_mode("mode=dense-w8-canonical-bf16\n").is_err());
        assert!(parse_io_mode(
            "mode=dense-w8-canonical-bf16-bf16x2-output\ninput=token-major-bf16\n"
        )
        .is_err());
    }

    #[test]
    fn canonical_down_weights_follow_nmacro_before_group() {
        assert_eq!(
            down_block_index(NpuResidentFfnDenseW8IoMode::PackedF32, false, 1, 3, 0, 1),
            GATE_BLOCKS + 16
        );
        assert_eq!(
            down_block_index(
                NpuResidentFfnDenseW8IoMode::CanonicalBf16,
                false,
                1,
                3,
                0,
                1
            ),
            GATE_BLOCKS + 13
        );
        assert_eq!(
            down_block_index(
                NpuResidentFfnDenseW8IoMode::CanonicalBf16,
                false,
                1,
                3,
                1,
                1
            ),
            GATE_BLOCKS + 18
        );
        assert_eq!(
            down_block_index(
                NpuResidentFfnDenseW8IoMode::CanonicalBf16Bf16x2Output,
                false,
                1,
                3,
                1,
                1
            ),
            GATE_BLOCKS + 18
        );
        assert_eq!(gate_param_block_index(1, 2), 23);
        assert_eq!(gate_block_index(true, 1, 0, 0), 24);
        assert_eq!(gate_block_index(true, 2, 5, 2), 62);
        assert_eq!(
            down_block_index(
                NpuResidentFfnDenseW8IoMode::DirectXBf16Bf16x2Output,
                true,
                0,
                0,
                0,
                1
            ),
            63
        );
        assert_eq!(COLS * REUSE_WEIGHT_BLOCKS * W_BLOCK, 12_189_696);
    }

    #[test]
    fn stationary_weights_pack_one_gate_down_sequence() {
        assert_eq!(WEIGHT_REPLAY_BLOCKS, 28);
        assert_eq!(packed_weight_blocks(1, false, true), 28);
        assert_eq!(packed_weight_blocks(2, false, true), 28);
        assert_eq!(replay_down_block_index(0, 0), 18);
        assert_eq!(replay_down_block_index(4, 0), 22);
        assert_eq!(replay_down_block_index(0, 1), 23);
        assert_eq!(replay_down_block_index(4, 1), 27);
        assert_eq!(STATIONARY_W_BLOCK, 15_552);
        assert_eq!(COLS * WEIGHT_REPLAY_BLOCKS * STATIONARY_W_BLOCK, 3_483_648);
    }

    #[test]
    fn direct_x_inverse_plane_uses_one_m256_record_set_per_document() {
        let inverse = (0..2 * M).map(|row| row as f32 + 0.25).collect::<Vec<_>>();
        let mut plane = vec![0u8; 2 * PRE_INVERSE_PLANE_BYTES];
        pack_pre_inverse_plane(&mut plane, &inverse, 2);
        for document in 0..2 {
            for token in 0..M {
                let logical_row = document * M + token;
                let wave = token >> 7;
                let within_wave = token & 127;
                let record = (within_wave >> 5) * 8 + wave * 4 + ((within_wave & 31) >> 3);
                let lane = within_wave & 7;
                let offset = document * PRE_INVERSE_PLANE_BYTES
                    + record * PRE_INVERSE_RECORD_BYTES
                    + lane * size_of::<f32>();
                assert_eq!(
                    f32::from_le_bytes(plane[offset..offset + 4].try_into().unwrap()),
                    inverse[logical_row]
                );
            }
        }
        assert_ne!(
            &plane[..PRE_INVERSE_PLANE_BYTES],
            &plane[PRE_INVERSE_PLANE_BYTES..]
        );
    }

    #[test]
    fn batched_canonical_and_direct_x_argument_geometry_scale_by_document() {
        assert_eq!(
            input_bytes_for(NpuResidentFfnDenseW8IoMode::CanonicalBf16, 2),
            884_736
        );
        assert_eq!(
            input_bytes_for(NpuResidentFfnDenseW8IoMode::CanonicalBf16Bf16x2Output, 2),
            884_736
        );
        assert_eq!(
            input_bytes_for(NpuResidentFfnDenseW8IoMode::DirectXBf16Bf16x2Output, 2),
            1_671_168
        );
    }

    #[test]
    fn gate_block_carries_activation_transform_parameters() {
        let mut block = vec![0u8; W_BLOCK];
        let weights = vec![0i8; GROUP * INTERMEDIATE];
        let scales = vec![1.0f32; INTERMEDIATE];
        let awq = (0..GATE_GROUPS * GROUP)
            .map(|index| index as f32 + 0.25)
            .collect::<Vec<_>>();
        let signs1 = gen_fwht_signs(42, GROUP);
        let signs2 = gen_fwht_signs(1042, GROUP);
        pack_gate_block(
            &mut block, 0, 0, &weights, &scales, &weights, &scales, 2, &awq, &signs1, &signs2, None,
        );
        let params = block[PARAM_OFFSET..PARAM_OFFSET + 3 * GROUP * size_of::<f32>()]
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(&params[..GROUP], &awq[2 * GROUP..3 * GROUP]);
        assert_eq!(&params[GROUP..2 * GROUP], signs1.as_slice());
        assert_eq!(&params[2 * GROUP..3 * GROUP], signs2.as_slice());
    }

    #[test]
    fn direct_x_gate_block_carries_grouped_pre_ffn_norm() {
        let mut block = vec![0u8; W_BLOCK];
        let weights = vec![0i8; GROUP * INTERMEDIATE];
        let scales = vec![1.0f32; INTERMEDIATE];
        let awq = vec![1.0f32; GATE_GROUPS * GROUP];
        let signs1 = gen_fwht_signs(42, GROUP);
        let signs2 = gen_fwht_signs(1042, GROUP);
        let pre_norm = (0..K).map(|index| index as u16).collect::<Vec<_>>();
        pack_gate_block(
            &mut block,
            0,
            0,
            &weights,
            &scales,
            &weights,
            &scales,
            1,
            &awq,
            &signs1,
            &signs2,
            Some(&pre_norm),
        );
        let encoded = block[PRE_NORM_OFFSET..PRE_NORM_OFFSET + GROUP * size_of::<u16>()]
            .chunks_exact(size_of::<u16>())
            .map(|bytes| u16::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(encoded, pre_norm[GROUP..2 * GROUP]);
    }
}
