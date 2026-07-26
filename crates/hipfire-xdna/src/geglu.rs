//! Resident AIE2P GeGLU stage for the EmbeddingGemma FFN handoff.
#![cfg(target_os = "linux")]

use std::path::Path;

use crate::{DeviceBuffer, NpuKernel, XdnaError};

/// One fixed-shape row-major `GeGLU([gate, up])` AIE2P program.
///
/// The input is `[rows, 2 * intermediate]` f32 with gate and up contiguous in
/// each row. The output is `[rows, intermediate]` f32. Both argument BOs can be
/// replaced by imported dma-bufs, allowing a projection kernel to feed this
/// stage without CPU or GPU materialization.
pub struct NpuGeGlu {
    kernel: NpuKernel,
    input: DeviceBuffer,
    output: DeviceBuffer,
    rows: usize,
    intermediate: usize,
}

impl NpuGeGlu {
    pub fn load_cached(
        cache: impl AsRef<Path>,
        rows: usize,
        intermediate: usize,
    ) -> Result<Self, XdnaError> {
        if rows == 0 || intermediate == 0 {
            return Err(invalid("GeGLU rows and intermediate must be non-zero"));
        }
        let cache = cache.as_ref();
        let xclbin = read(cache.join("final.xclbin"))?;
        let insts = read(cache.join("insts.bin"))?;
        let kernel = NpuKernel::load(&xclbin, &insts)?;
        let input = kernel.alloc_arg(rows * 2 * intermediate * size_of::<f32>())?;
        let output = kernel.alloc_arg(rows * intermediate * size_of::<f32>())?;
        Ok(Self {
            kernel,
            input,
            output,
            rows,
            intermediate,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn intermediate(&self) -> usize {
        self.intermediate
    }

    pub fn input_bytes(&self) -> usize {
        self.rows * 2 * self.intermediate * size_of::<f32>()
    }

    pub fn output_bytes(&self) -> usize {
        self.rows * self.intermediate * size_of::<f32>()
    }

    /// Import a producer's row-major output and a consumer-visible output.
    /// Backing allocations may be larger than the logical views (for example,
    /// a projection's padded row-major output), but never smaller.
    pub fn attach_shared_io(
        &mut self,
        input_fd: i32,
        input_backing_bytes: usize,
        output_fd: i32,
        output_backing_bytes: usize,
    ) -> Result<(), XdnaError> {
        if input_backing_bytes < self.input_bytes() || output_backing_bytes < self.output_bytes() {
            return Err(invalid("GeGLU shared dma-buf backing is too small"));
        }
        self.attach_shared_input(input_fd, input_backing_bytes)?;
        self.attach_shared_output(output_fd, output_backing_bytes)?;
        Ok(())
    }

    pub fn attach_shared_input(
        &mut self,
        input_fd: i32,
        input_backing_bytes: usize,
    ) -> Result<(), XdnaError> {
        if input_backing_bytes < self.input_bytes() {
            return Err(invalid("GeGLU shared input dma-buf backing is too small"));
        }
        // Import the exporter-reported backing size. amdxdna keys cache and
        // sharing state to the complete dma-buf object; importing only the
        // logical prefix can create a distinct, non-coherent view.
        self.input = self
            .kernel
            .import_dmabuf(input_fd, input_backing_bytes, true)?;
        self.kernel.sync_to_device(&self.input)?;
        Ok(())
    }

    pub fn attach_shared_output(
        &mut self,
        output_fd: i32,
        output_backing_bytes: usize,
    ) -> Result<(), XdnaError> {
        if output_backing_bytes < self.output_bytes() {
            return Err(invalid("GeGLU shared output dma-buf backing is too small"));
        }
        self.output = self
            .kernel
            .import_dmabuf(output_fd, output_backing_bytes, true)?;
        // Prime the imported mapping once. Repeating this full-buffer
        // host-to-device reconciliation before every pure device write costs
        // milliseconds and is unnecessary after the BO is established.
        self.kernel.sync_to_device(&self.output)?;
        Ok(())
    }

    /// Run over already-device-produced shared pages. The input cache operation
    /// reconciles the dma-buf across hardware contexts; no host-side copy or
    /// output unpack is performed.
    pub fn run_shared(&self) -> Result<(), XdnaError> {
        self.kernel
            .dispatch_synced(&[&self.input, &self.output], &[false, false])?;
        self.kernel.sync_output(&self.output)
    }

    /// Diagnostic/host-consumer variant retaining the same shared input path.
    pub fn run_shared_to_host(&self, output: &mut [f32]) -> Result<(), XdnaError> {
        if output.len() * size_of::<f32>() != self.output_bytes() {
            return Err(invalid("GeGLU host output geometry mismatch"));
        }
        self.run_shared()?;
        output.copy_from_slice(unsafe { as_f32(&self.output.as_slice()[..self.output_bytes()]) });
        Ok(())
    }

    /// Host verification path for a row-major combined gate/up matrix.
    pub fn run(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), XdnaError> {
        if input.len() * size_of::<f32>() != self.input_bytes()
            || output.len() * size_of::<f32>() != self.output_bytes()
        {
            return Err(invalid("GeGLU host input/output geometry mismatch"));
        }
        self.input
            .as_mut_slice()
            .copy_from_slice(unsafe { as_bytes(input) });
        self.kernel
            .dispatch_synced(&[&self.input, &self.output], &[true, false])?;
        self.kernel.sync_output(&self.output)?;
        output.copy_from_slice(unsafe { as_f32(self.output.as_slice()) });
        Ok(())
    }
}

fn read(path: impl AsRef<Path>) -> Result<Vec<u8>, XdnaError> {
    std::fs::read(path).map_err(|error| XdnaError::Open(error))
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

unsafe fn as_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

unsafe fn as_f32(values: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len() / size_of::<f32>()) }
}
