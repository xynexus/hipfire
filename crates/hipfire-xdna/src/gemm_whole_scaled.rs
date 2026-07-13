//! Group-retaining AIE2P whole-array W4A8 GEMM with fused f32 scaling.
#![cfg(target_os = "linux")]

use std::collections::HashMap;
use std::path::Path;

use crate::opus_hfp::{self, OpusHfpDescriptor, OpusHfpEncoding, OpusHfpLayout};
use crate::{DeviceBuffer, NpuKernel, NpuWholeMode, XdnaError};

const ROW_STRIPES: usize = 4;
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

    fn macro_n(self, cols: usize) -> usize {
        cols * self.cols_stripe
    }

    fn c_join(self) -> usize {
        ROW_STRIPES * self.c_core
    }
}

pub struct NpuWholeScaledResidentWeights {
    buffer: DeviceBuffer,
}

/// Physical argument-buffer contract for one group-retaining whole-array
/// projection. GPU producers/consumers use this metadata to address the AIE
/// block layout directly in imported dma-bufs without a host pack/unpack pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NpuWholeScaledIoLayout {
    mode: NpuWholeMode,
    cols: usize,
    rows: usize,
    groups: usize,
    n: usize,
    n_macros: usize,
    outblocks: usize,
    row_major_output: bool,
    padded_rows: usize,
    padded_n: usize,
    input_bytes: usize,
    output_bytes: usize,
}

impl NpuWholeScaledIoLayout {
    fn new(
        mode: NpuWholeMode,
        cols: usize,
        rows: usize,
        groups: usize,
        n: usize,
        n_macros: usize,
        outblocks: usize,
        row_major_output: bool,
    ) -> Self {
        let layout = Layout::for_mode(mode);
        let padded_rows = outblocks / n_macros * MACRO_M;
        let padded_n = n_macros * layout.macro_n(cols);
        Self {
            mode,
            cols,
            rows,
            groups,
            n,
            n_macros,
            outblocks,
            row_major_output,
            padded_rows,
            padded_n,
            input_bytes: ROW_STRIPES * outblocks * groups * A_BLOCK,
            output_bytes: if row_major_output {
                padded_rows * padded_n * size_of::<f32>()
            } else {
                cols * outblocks * layout.c_join() * size_of::<f32>()
            },
        }
    }

    pub fn mode(self) -> NpuWholeMode {
        self.mode
    }

    pub fn cols(self) -> usize {
        self.cols
    }

    pub fn rows(self) -> usize {
        self.rows
    }

    pub fn groups(self) -> usize {
        self.groups
    }

    pub fn k(self) -> usize {
        self.groups * GROUP_K
    }

    pub fn n(self) -> usize {
        self.n
    }

    pub fn n_macros(self) -> usize {
        self.n_macros
    }

    pub fn outblocks(self) -> usize {
        self.outblocks
    }

    pub fn row_major_output(self) -> bool {
        self.row_major_output
    }

    pub fn padded_rows(self) -> usize {
        self.padded_rows
    }

    pub fn padded_n(self) -> usize {
        self.padded_n
    }

    pub fn input_bytes(self) -> usize {
        self.input_bytes
    }

    pub fn output_bytes(self) -> usize {
        self.output_bytes
    }
}

pub struct NpuGemmWholeScaled {
    kernel: NpuKernel,
    layout: Layout,
    cols: usize,
    rows: usize,
    groups: usize,
    n: usize,
    m_macros: usize,
    n_macros: usize,
    outblocks: usize,
    row_major_output: bool,
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
        let cols = shape.get("cols").copied().unwrap_or(4);
        let m_macros = required(&shape, "mm")?;
        let n_macros = required(&shape, "nm")?;
        let groups = required(&shape, "kg")?;
        let outblocks = required(&shape, "outblocks")?;
        let row_major_output = manifest.lines().any(|line| line == "output=rowmajor");
        let padded_rows = m_macros * MACRO_M;
        let padded_n = n_macros * layout.macro_n(cols);
        if rows == 0
            || n == 0
            || !matches!(cols, 4 | 8)
            || k != groups * GROUP_K
            || m_macros != rows.div_ceil(MACRO_M)
            || n_macros != n.div_ceil(layout.macro_n(cols))
            || outblocks != m_macros * n_macros
            || (row_major_output
                && (shape.get("pm").copied() != Some(padded_rows)
                    || shape.get("pn").copied() != Some(padded_n)))
        {
            return Err(invalid("invalid scaled whole-array cache geometry"));
        }
        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).map_err(XdnaError::Open)?;
        let insts = std::fs::read(format!("{dir}/insts.bin")).map_err(XdnaError::Open)?;
        let kernel = NpuKernel::load(&xclbin, &insts)?;
        let inblocks = outblocks * groups;
        let input = kernel.alloc_arg(ROW_STRIPES * inblocks * A_BLOCK)?;
        let output_bytes = if row_major_output {
            padded_rows * padded_n * size_of::<f32>()
        } else {
            cols * outblocks * layout.c_join() * size_of::<f32>()
        };
        let output = kernel.alloc_arg(output_bytes)?;
        Ok(Self {
            kernel,
            layout,
            cols,
            rows,
            groups,
            n,
            m_macros,
            n_macros,
            outblocks,
            row_major_output,
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

    pub fn io_layout(&self) -> NpuWholeScaledIoLayout {
        NpuWholeScaledIoLayout::new(
            self.layout.mode,
            self.cols,
            self.rows,
            self.groups,
            self.n,
            self.n_macros,
            self.outblocks,
            self.row_major_output,
        )
    }

    fn inblocks(&self) -> usize {
        self.outblocks * self.groups
    }

    pub fn packed_weight_bytes(&self) -> usize {
        self.cols * self.inblocks() * W_BLOCK
    }

    fn hfp_descriptor(&self, quant_type: u8) -> OpusHfpDescriptor {
        OpusHfpDescriptor {
            encoding: match self.layout.mode {
                NpuWholeMode::W4 => OpusHfpEncoding::W4,
                NpuWholeMode::W8 => OpusHfpEncoding::W8,
            },
            layout: OpusHfpLayout::WholeScaledV1,
            quant_type: quant_type.into(),
            flags: 0,
            m: self.rows as u32,
            k: self.k() as u32,
            n: self.n as u32,
            columns: self.cols as u32,
            groups: self.groups as u32,
            m_macros: self.m_macros as u32,
            n_macros: self.n_macros as u32,
            outblocks: self.outblocks as u32,
            tile_bytes: W_BLOCK as u32,
            data_bytes: W_DATA as u32,
            scale_offset: W_DATA as u32,
            scale_values: self.layout.cols_stripe as u32,
            payload_bytes: self.packed_weight_bytes() as u64,
            segment_bytes: [0; 4],
        }
    }

    /// Load an already converted `.rdna2.hfp` weight stream, or create it once
    /// from the source Opus tensor. Only global block/tile ordering happens
    /// here; W4 nibbles remain packed for the AIE kernel to decode and swizzle.
    pub(crate) fn prepack_weights_cached(
        &self,
        path: &Path,
        quant_type: u8,
        source_payload: &[u8],
        weights: &[&[i8]],
        scales: &[&[f32]],
    ) -> Result<Vec<u8>, XdnaError> {
        let descriptor = self.hfp_descriptor(quant_type);
        let source_sha = opus_hfp::source_sha256(&[source_payload]);
        if let Some(packed) = opus_hfp::read(path, descriptor, source_sha).map_err(invalid)? {
            return Ok(packed);
        }
        let packed = self.prepack_weights(weights, scales)?;
        opus_hfp::write(path, descriptor, source_sha, &packed).map_err(invalid)?;
        Ok(packed)
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
        for stripe in 0..self.cols {
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
                                        let col = n_macro * self.layout.macro_n(self.cols)
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
                            let col = n_macro * self.layout.macro_n(self.cols)
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

    /// Marshal already-quantized `[rows,K]` activations and per-group row
    /// scales into the shared whole-array physical input contract.
    pub fn prepack_activations(
        &self,
        activations: &[i8],
        scales: &[f32],
    ) -> Result<Vec<u8>, XdnaError> {
        if activations.len() != self.rows * self.k() || scales.len() != self.groups * self.rows {
            return Err(invalid("scaled whole-array activation geometry mismatch"));
        }
        let mut packed = vec![0u8; self.io_layout().input_bytes()];
        Self::pack_activations_into(
            self.layout,
            self.rows,
            self.groups,
            self.m_macros,
            self.n_macros,
            self.outblocks,
            activations,
            scales,
            &mut packed,
        );
        Ok(packed)
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

    /// Replace the private argument BOs with GPU-exported dma-bufs. The input
    /// buffer must already use the AIE activation layout and the output remains
    /// in the AIE physical tile layout until a GPU consumer deblocks it.
    pub fn attach_shared_io(
        &mut self,
        input_fd: i32,
        input_bytes: usize,
        output_fd: i32,
        output_bytes: usize,
    ) -> Result<(), XdnaError> {
        let layout = self.io_layout();
        if input_bytes != layout.input_bytes() || output_bytes != layout.output_bytes() {
            return Err(invalid("scaled whole-array shared dma-buf size mismatch"));
        }
        self.input = self.kernel.import_dmabuf(input_fd, input_bytes, true)?;
        self.output = self.kernel.import_dmabuf(output_fd, output_bytes, true)?;
        // Establish the pure-output imported BO once. Subsequent projection
        // dispatches do not need to clean the entire output before overwriting
        // it; device-chain consumers reconcile the produced pages themselves.
        self.kernel.sync_to_device(&self.output)?;
        Ok(())
    }

    /// Dispatch using prepacked activations and physical output pages shared
    /// with the GPU. No CPU-side activation copy or output deblocking occurs.
    pub fn run_resident_shared(
        &mut self,
        weights: &NpuWholeScaledResidentWeights,
    ) -> Result<(), XdnaError> {
        if weights.buffer.len() != self.packed_weight_bytes() {
            return Err(invalid("scaled whole-array resident weight size mismatch"));
        }
        self.kernel.dispatch_synced(
            &[&self.input, &weights.buffer, &self.output],
            &[true, false, true],
        )?;
        self.kernel.sync_output(&self.output)
    }

    /// Device-chain variant: wait for the projection to complete but leave the
    /// shared output device-resident for the next imported-buffer consumer.
    /// The consumer performs the cross-context cache reconciliation.
    pub fn run_resident_shared_to_device(
        &mut self,
        weights: &NpuWholeScaledResidentWeights,
    ) -> Result<(), XdnaError> {
        if weights.buffer.len() != self.packed_weight_bytes() {
            return Err(invalid("scaled whole-array resident weight size mismatch"));
        }
        self.kernel.dispatch_synced(
            &[&self.input, &weights.buffer, &self.output],
            &[true, false, false],
        )
    }

    fn pack_activations(&mut self, activations: &[i8], scales: &[f32]) {
        Self::pack_activations_into(
            self.layout,
            self.rows,
            self.groups,
            self.m_macros,
            self.n_macros,
            self.outblocks,
            activations,
            scales,
            self.input.as_mut_slice(),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn pack_activations_into(
        layout: Layout,
        rows: usize,
        groups: usize,
        m_macros: usize,
        n_macros: usize,
        outblocks: usize,
        activations: &[i8],
        scales: &[f32],
        packed: &mut [u8],
    ) {
        packed.fill(0);
        for stripe in 0..ROW_STRIPES {
            for m_macro in 0..m_macros {
                for n_macro in 0..n_macros {
                    let outblock = m_macro * n_macros + n_macro;
                    for group in 0..groups {
                        let block = outblock * groups + group;
                        let base = (stripe * outblocks * groups + block) * A_BLOCK;
                        for lm in 0..layout.lm {
                            for kt in 0..GROUP_K / layout.inner_k {
                                for local_row in 0..layout.mr {
                                    let row = m_macro * MACRO_M
                                        + stripe * layout.rows_stripe()
                                        + lm * layout.mr
                                        + local_row;
                                    if row < rows {
                                        let source = row * groups * GROUP_K
                                            + group * GROUP_K
                                            + kt * layout.inner_k;
                                        let target = base
                                            + (lm * (GROUP_K / layout.inner_k) + kt) * 64
                                            + local_row * layout.inner_k;
                                        packed[target..target + layout.inner_k].copy_from_slice(
                                            as_bytes(&activations[source..source + layout.inner_k]),
                                        );
                                    }
                                }
                            }
                        }
                        for local_row in 0..layout.rows_stripe() {
                            let row = m_macro * MACRO_M + stripe * layout.rows_stripe() + local_row;
                            let scale = if row < rows {
                                scales[group * rows + row]
                            } else {
                                0.0
                            };
                            let offset = base + A_DATA + local_row * size_of::<f32>();
                            packed[offset..offset + size_of::<f32>()]
                                .copy_from_slice(&scale.to_ne_bytes());
                        }
                    }
                }
            }
        }
    }

    fn unpack_output(&self, output: &mut [f32]) {
        let physical = as_f32(self.output.as_slice());
        if self.row_major_output {
            let padded_n = self.n_macros * self.layout.macro_n(self.cols);
            for row in 0..self.rows {
                output[row * self.n..(row + 1) * self.n]
                    .copy_from_slice(&physical[row * padded_n..row * padded_n + self.n]);
            }
            return;
        }
        for col_stripe in 0..self.cols {
            for m_macro in 0..self.m_macros {
                for n_macro in 0..self.n_macros {
                    let outblock = m_macro * self.n_macros + n_macro;
                    for row_stripe in 0..ROW_STRIPES {
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
                                        let col = n_macro * self.layout.macro_n(self.cols)
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

    #[test]
    fn shared_io_layout_describes_the_exact_whole_array_buffers() {
        let w4 = NpuWholeScaledIoLayout::new(NpuWholeMode::W4, 8, 256, 3, 768, 1, 3, false);
        assert_eq!(w4.input_bytes(), 4 * 3 * 3 * A_BLOCK);
        assert_eq!(w4.output_bytes(), 8 * 3 * 4 * 2304 * size_of::<f32>());
        assert_eq!(w4.rows(), 256);
        assert_eq!(w4.k(), 768);
        assert_eq!(w4.n(), 768);
        assert_eq!(w4.cols(), 8);
        assert_eq!(w4.groups(), 3);
        assert_eq!(w4.n_macros(), 1);
        assert_eq!(w4.outblocks(), 3);

        let w8 = NpuWholeScaledIoLayout::new(NpuWholeMode::W8, 8, 256, 5, 768, 2, 6, false);
        assert_eq!(w8.input_bytes(), 4 * 30 * A_BLOCK);
        assert_eq!(w8.output_bytes(), 8 * 6 * 4 * 1152 * size_of::<f32>());
        assert_eq!(w8.k(), 1280);

        let rowmajor = NpuWholeScaledIoLayout::new(NpuWholeMode::W4, 8, 256, 3, 1280, 2, 6, true);
        assert!(rowmajor.row_major_output());
        assert_eq!((rowmajor.padded_rows(), rowmajor.padded_n()), (288, 1536));
        assert_eq!(rowmajor.output_bytes(), 288 * 1536 * size_of::<f32>());
    }
}
