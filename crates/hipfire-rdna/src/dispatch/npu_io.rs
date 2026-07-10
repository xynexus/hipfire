//! GPU producer/consumer kernels for the XDNA whole-array Opus dma-buf layout.

use super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

/// Backend-neutral copy of the physical whole-array dma-buf geometry. The
/// EmbeddingGemma bridge constructs it from XDNA metadata, keeping the RDNA
/// crate independently buildable on systems without the XDNA runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpusNpuIoLayout {
    mode_w8: bool,
    cols: usize,
    rows: usize,
    groups: usize,
    n: usize,
    n_macros: usize,
    outblocks: usize,
    input_bytes: usize,
    output_bytes: usize,
}

impl OpusNpuIoLayout {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mode_w8: bool,
        cols: usize,
        rows: usize,
        groups: usize,
        n: usize,
        n_macros: usize,
        outblocks: usize,
        input_bytes: usize,
        output_bytes: usize,
    ) -> Self {
        Self {
            mode_w8,
            cols,
            rows,
            groups,
            n,
            n_macros,
            outblocks,
            input_bytes,
            output_bytes,
        }
    }
}

impl Gpu {
    /// Apply optional AWQ scaling, the canonical Opus FWHT-256, and per-row
    /// int8 quantization, then scatter directly into the AIE input dma-buf.
    pub fn pack_opus_npu_activations(
        &mut self,
        input: &GpuTensor,
        awq_scale: Option<&GpuTensor>,
        packed: &GpuTensor,
        rows: usize,
        k: usize,
        layout: OpusNpuIoLayout,
    ) -> HipResult<()> {
        if rows > layout.rows
            || k > layout.groups * 256
            || input.numel() < rows * k
            || packed.buf.size() != layout.input_bytes
            || awq_scale.is_some_and(|scale| scale.numel() < k)
        {
            return Err(HipError::new(
                0,
                "invalid Opus NPU activation-pack geometry",
            ));
        }
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "opus_npu_io",
            kernels::OPUS_NPU_IO_SRC,
            "opus_npu_pack_activations",
        )?;
        let input_ptr = input.buf.as_ptr();
        let packed_ptr = packed.buf.as_ptr();
        let awq_ptr = awq_scale
            .map(|scale| scale.buf.as_ptr())
            .unwrap_or(std::ptr::null_mut::<c_void>());
        let signs1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let signs2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let mode = i32::from(layout.mode_w8);
        self.launch_kernargs(
            "opus_npu_pack_activations",
            [layout.groups as u32, layout.rows as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![
                ptr input_ptr,
                ptr packed_ptr,
                ptr awq_ptr,
                ptr signs1_ptr,
                ptr signs2_ptr,
                i32 k as i32,
                i32 rows as i32,
                i32 layout.rows as i32,
                i32 layout.groups as i32,
                i32 layout.n_macros as i32,
                i32 layout.outblocks as i32,
                i32 mode,
            ],
        )
    }

    /// Deblock the AIE physical f32 output dma-buf directly into one to three
    /// regular row-major GPU tensors. Widths must concatenate to `layout.n()`.
    #[allow(clippy::too_many_arguments)]
    pub fn unpack_opus_npu_output(
        &mut self,
        packed: &GpuTensor,
        out0: &GpuTensor,
        width0: usize,
        out1: Option<(&GpuTensor, usize)>,
        out2: Option<(&GpuTensor, usize)>,
        rows: usize,
        layout: OpusNpuIoLayout,
    ) -> HipResult<()> {
        let width1 = out1.map_or(0, |(_, width)| width);
        let width2 = out2.map_or(0, |(_, width)| width);
        if rows > layout.rows
            || width0 + width1 + width2 != layout.n
            || packed.buf.size() != layout.output_bytes
            || out0.numel() < rows * width0
            || out1.is_some_and(|(output, width)| output.numel() < rows * width)
            || out2.is_some_and(|(output, width)| output.numel() < rows * width)
        {
            return Err(HipError::new(0, "invalid Opus NPU output-unpack geometry"));
        }
        self.bind_thread()?;
        self.ensure_kernel(
            "opus_npu_io",
            kernels::OPUS_NPU_IO_SRC,
            "opus_npu_unpack_output",
        )?;
        let packed_ptr = packed.buf.as_ptr();
        let out0_ptr = out0.buf.as_ptr();
        let out1_ptr = out1
            .map(|(output, _)| output.buf.as_ptr())
            .unwrap_or(std::ptr::null_mut::<c_void>());
        let out2_ptr = out2
            .map(|(output, _)| output.buf.as_ptr())
            .unwrap_or(std::ptr::null_mut::<c_void>());
        let mode = i32::from(layout.mode_w8);
        let elements = rows * layout.n;
        let block = 256u32;
        self.launch_kernargs(
            "opus_npu_unpack_output",
            [((elements as u32) + block - 1) / block, 1, 1],
            [block, 1, 1],
            0,
            &kernargs![
                ptr packed_ptr,
                ptr out0_ptr,
                ptr out1_ptr,
                ptr out2_ptr,
                i32 rows as i32,
                i32 layout.n as i32,
                i32 width0 as i32,
                i32 width1 as i32,
                i32 width2 as i32,
                i32 layout.cols as i32,
                i32 layout.n_macros as i32,
                i32 layout.outblocks as i32,
                i32 mode,
            ],
        )
    }
}
