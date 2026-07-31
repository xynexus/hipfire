// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Base-dtype GEMM dispatch (f16/bf16/f32, incl. WMMA + train). Pure move (Phase 1 M6).

use super::{DType, Gpu, GpuTensor};
use crate::arch_caps::ArchCaps;
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RawXf32Backend {
    Wmma,
    Portable,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_flags::FeatureFlags;
    use std::sync::Arc;

    fn caps(arch: &str) -> ArchCaps {
        ArchCaps::new(arch, Arc::new(FeatureFlags::from_env_for_test(arch)))
    }

    #[test]
    fn raw_x_f32_backend_is_wmma_only_where_the_instruction_is_available() {
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            assert_eq!(raw_x_f32_backend(&caps(arch)), RawXf32Backend::Wmma);
        }
        for arch in ["gfx1030", "gfx906", "gfx908", "gfx942"] {
            assert_eq!(raw_x_f32_backend(&caps(arch)), RawXf32Backend::Portable);
        }
    }
}

fn raw_x_f32_backend(arch: &ArchCaps) -> RawXf32Backend {
    if arch.has_wmma() {
        RawXf32Backend::Wmma
    } else {
        RawXf32Backend::Portable
    }
}

impl Gpu {
    /// Raw F16/BF16 weight x F32 activation GEMM with an architecture-honest
    /// backend. Wave32 WMMA is retained on RDNA3/4; RDNA2/CDNA/older targets
    /// use the scalar correctness kernel instead of attempting an unsupported
    /// WMMA intrinsic.
    pub fn gemm_raw_x_f32_auto(
        &mut self,
        weight: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        match (weight.dtype, raw_x_f32_backend(&self.arch_caps)) {
            (DType::F16 | DType::Raw, RawXf32Backend::Wmma) => {
                self.gemm_f16_x_f32_wmma(weight, x, y, m, k, batch_size)
            }
            (DType::BF16, RawXf32Backend::Wmma) => {
                self.gemm_bf16_x_bf16_wmma(weight, x, y, m, k, batch_size)
            }
            (DType::F16 | DType::Raw | DType::BF16, RawXf32Backend::Portable) => {
                self.gemm_raw_x_f32_portable(weight, x, y, m, k, batch_size)
            }
            (dtype, _) => Err(hip_bridge::HipError::new(
                0,
                &format!("raw F16/BF16 GEMM does not support weight dtype {dtype:?}"),
            )),
        }
    }

    /// Scalar raw matrix fallback used by [`Self::gemm_raw_x_f32_auto`].
    pub fn gemm_raw_x_f32_portable(
        &mut self,
        weight: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(matches!(
            weight.dtype,
            DType::F16 | DType::Raw | DType::BF16
        ));
        assert_eq!(x.dtype, DType::F32);
        assert_eq!(y.dtype, DType::F32);
        assert!(m <= i32::MAX as usize && k <= i32::MAX as usize);
        assert!(batch_size <= i32::MAX as usize);
        let kernel_name = match weight.dtype {
            DType::F16 | DType::Raw => "gemm_f16_x_f32_portable",
            DType::BF16 => "gemm_bf16_x_f32_portable",
            _ => unreachable!(),
        };
        self.ensure_kernel(
            kernel_name,
            kernels::GEMM_RAW_X_F32_PORTABLE_SRC,
            kernel_name,
        )?;
        let wp = weight.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let bi = batch_size as i32;
        let block = 256u32;
        let grid_x = (m as u32).div_ceil(block);
        let bytes = m
            .saturating_mul(k)
            .saturating_mul(2)
            .saturating_add(batch_size.saturating_mul(k).saturating_mul(4))
            .saturating_add(batch_size.saturating_mul(m).saturating_mul(4));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_kernargs(
            kernel_name,
            [grid_x, batch_size as u32, 1],
            [block, 1, 1],
            0,
            &kernargs![ptr wp, ptr xp, ptr yp, i32 mi, i32 ki, i32 bi],
        );
        if let Some(timer) = timer {
            timer.finish(&self.hip);
        }
        result
    }

    /// FP16-weight lm_head fast path for DFlash drafts that ship F16 (not
    /// quantized) weights. Routes through `gemm_mw16_residual_wmma` with the
    /// usual memset-then-atomicAdd residual pattern.
    ///
    /// Shape requirements: K must be a multiple of 32 (mw16 processes 32 K
    /// elements per WMMA iteration). All 27B/9B draft shapes satisfy this
    /// (hidden=5120, intermediate=17408, q_dim=4096, kv_dim=1024, fc-K=25600).
    ///
    /// Non-gfx11 falls through to row-by-row F16 GEMM so lm_head output keeps
    /// the `[batch, vocab]` layout expected by callers. Set
    /// `HIPFIRE_LM_HEAD_F16=f32` at load time to use the legacy F32-expanded
    /// storage and bypass this path entirely.
    pub fn gemm_f16_batched_lmhead(
        &mut self,
        w_f16: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Calibration capture: this is the F16/BF16 linear chokepoint the lowered
        // super-op path routes through (GemvFamily K::GemvF16), so capturing here
        // fires for the resident bf16 calibration forward. No-op when unarmed.
        self.maybe_capture_activation(w_f16, x, batch_size, k);
        if !self.arch_caps.has_wmma_w32() {
            // No mw16 WMMA on non-RDNA3. The generic F16 GEMM writes [M,N],
            // while lm_head consumers expect [N,M], so preserve layout by
            // launching one row at a time.
            for b in 0..batch_size {
                let x_row = x.sub_offset(b * k, k);
                let y_row = y.sub_offset(b * m, m);
                self.gemm_f16_tiled(w_f16, &x_row, &y_row, m, k, 1)?;
            }
            return Ok(());
        }
        self.ensure_kernel(
            "gemm_mw16_residual_wmma",
            kernels::GEMM_MW16_RESIDUAL_WMMA_SRC,
            "gemm_mw16_residual_wmma",
        )?;
        // Pre-zero Y (residual WMMA does y += acc) and force FP16-X reconversion
        // (the draft reuses the same scratch pointer every cycle with new data).
        self.fp16_x_source_ptr = std::ptr::null_mut();
        match self.active_stream.as_ref() {
            Some(stream) => self
                .hip
                .memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
            None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
        }
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let wp = w_f16.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let ni = batch_size as i32;
        let rows = ((m + 15) / 16) as u32;
        let batches = ((batch_size + 15) / 16) as u32;
        // Bytes: FP16 weight + FP16 x + FP32 y (read+write).
        let bytes = m * k * 2 + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_mw16_residual_wmma", bytes);
        let result = self.launch_kernargs(
            "gemm_mw16_residual_wmma",
            [rows, batches, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr wp, ptr xp, ptr yp, i32 mi, i32 ki, i32 ni],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Batched GEMV (GEMM) for F16 weights: Y[M,N] = W_f16[M,K] @ X_f32[N,K]^T
    pub fn gemm_f16(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_f16", kernels::GEMM_F16_SRC, "gemm_f16")?;
        let func = &self.functions["gemm_f16"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, n as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// WMMA-accelerated batched GEMM for F16 weights × F32 activations (gfx1100+).
    /// Y[M,N] = W_f16[M,K] @ X_f32[N,K]^T.  Tiled 16×16 WMMA matrix multiply.
    /// Grid=[ceil(M/16), ceil(N/16)], Block=[32].  Replaces naive gemm_f16 for vision encoder.
    pub fn gemm_f16_wmma(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_f16_wmma", kernels::GEMM_F16_WMMA_SRC, "gemm_f16_wmma")?;
        let func = &self.functions["gemm_f16_wmma"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_n = ((n + 15) / 16) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_n, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Tiled F16 GEMM — 4-way ILP unrolled, no shared memory (high occupancy).
    /// Grid=[M, N], Block=[32], LDS=0.
    pub fn gemm_f16_tiled(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_f16_tiled",
            kernels::GEMM_F16_TILED_SRC,
            "gemm_f16_tiled",
        )?;
        let func = &self.functions["gemm_f16_tiled"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        // Same grid as naive: [M, N], block [32], no LDS
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, n as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Fused GEMM + bias: Y[N,M] = X[N,K] @ W_f16[M,K]^T + bias[M].
    /// Replaces gemm_f16 + transpose_f32 + bias_add_f32 (3 ops → 1).
    /// Grid=[N, 1], Block=[256].
    pub fn gemm_f16_bias(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        bias: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_f16_bias", kernels::GEMM_F16_BIAS_SRC, "gemm_f16_bias")?;
        let func = &self.functions["gemm_f16_bias"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut bp = bias.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        // One block per row of X, 256 threads, no LDS
        unsafe {
            self.hip.launch_kernel(
                func,
                [n as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Batched GEMM for F32: Y[M,N] = A[M,K] @ B[N,K]^T
    pub fn gemm_f32_batched(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_f32_batched",
            kernels::GEMM_F32_SRC,
            "gemm_f32_batched",
        )?;
        let func = &self.functions["gemm_f32_batched"];
        let mut ap = a.buf.as_ptr();
        let mut bp = b.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, n as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// General fp32 GEMM with per-operand transpose flags (training path).
    ///
    /// Computes `C[M,N] = op(A)·op(B)`, C row-major. `op(A)` is `[M,K]`, `op(B)`
    /// is `[K,N]`. `lda`/`ldb` are the leading dims of the *stored* operands;
    /// `trans_a`/`trans_b` select whether the stored matrix is read transposed.
    /// One general kernel covers the forward and both backward matmuls of a
    /// linear — see `kernels/src/gemm_f32_train.hip`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f32_train(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_f32_train",
            kernels::GEMM_F32_TRAIN_SRC,
            "gemm_f32_train",
        )?;
        let func = &self.functions["gemm_f32_train"];
        let mut ap = a.buf.as_ptr();
        let mut bp = b.buf.as_ptr();
        let mut cp = c.buf.as_ptr();
        let mut mi = m as i32;
        let mut ni = n as i32;
        let mut ki = k as i32;
        let mut ldai = lda as i32;
        let mut ldbi = ldb as i32;
        let mut tai = trans_a as i32;
        let mut tbi = trans_b as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut cp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ldai as *mut _ as *mut c_void,
            &mut ldbi as *mut _ as *mut c_void,
            &mut tai as *mut _ as *mut c_void,
            &mut tbi as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [((n + 63) / 64) as u32, ((m + 63) / 64) as u32, 1],
                [16, 16, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// `gemm_f32_train` variant that accumulates: `C = beta*C + op(A)·op(B)`.
    /// Used where a gradient lands on a buffer that already holds a partial.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f32_train_accum(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        trans_a: bool,
        trans_b: bool,
        beta: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_f32_train_accum",
            kernels::GEMM_F32_TRAIN_SRC,
            "gemm_f32_train_accum",
        )?;
        let func = &self.functions["gemm_f32_train_accum"];
        let mut ap = a.buf.as_ptr();
        let mut bp = b.buf.as_ptr();
        let mut cp = c.buf.as_ptr();
        let mut mi = m as i32;
        let mut ni = n as i32;
        let mut ki = k as i32;
        let mut ldai = lda as i32;
        let mut ldbi = ldb as i32;
        let mut tai = trans_a as i32;
        let mut tbi = trans_b as i32;
        let mut betaf = beta;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut cp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ldai as *mut _ as *mut c_void,
            &mut ldbi as *mut _ as *mut c_void,
            &mut tai as *mut _ as *mut c_void,
            &mut tbi as *mut _ as *mut c_void,
            &mut betaf as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [((n + 63) / 64) as u32, ((m + 63) / 64) as u32, 1],
                [16, 16, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// BF16-compute forward of the training linear op:
    /// `y[m,n] = x[m,k] · w[n,k]^T` (NT). Reads `x`/`w` as F32, casts to BF16
    /// in-register for a WMMA f32-accumulate multiply, writes F32 `y`. Same
    /// numerics contract as `gemm_f32_train(x, w, y, m, n, k, k, k, false, true)`
    /// but on the matrix cores; leaves the F32 master weights/activations and the
    /// F32 backward untouched. Args mirror `linear_forward`'s `(x, w, y, m, k, n)`.
    pub fn gemm_bf16c_train_nt(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_bf16c_train_nt",
            kernels::GEMM_BF16C_TRAIN_NT_SRC,
            "gemm_bf16c_train_nt",
        )?;
        // Kernel computes Y[B,M] = X[B,K]·A[M,K]^T. Map A=w (kernel M=n_out),
        // X=x (kernel B=m_tok), Y=y stored row-major [B=m, M=n].
        let ap = w.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let mi = n as i32; // kernel M = output cols = n
        let ki = k as i32;
        let bi = m as i32; // kernel B = tokens = m
        let grid_m = ((n + 15) / 16) as u32;
        let grid_b = ((m + 15) / 16) as u32;
        self.launch_kernargs(
            "gemm_bf16c_train_nt",
            [grid_m, grid_b, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr ap, ptr xp, ptr yp, i32 mi, i32 ki, i32 bi],
        )
    }
    /// Split-precision ("2xbf16") forward of the training linear op:
    /// `y[m,n] = x[m,k] · w[n,k]^T` (NT) at near-f32 accuracy on the WMMA cores.
    /// Splits each f32 operand into bf16 hi+lo and accumulates 3 WMMA passes
    /// (~16 mantissa bits vs bf16's 8). Same args as `gemm_bf16c_train_nt`; used
    /// for the precision-sensitive vocab heads in place of the scalar f32 path.
    pub fn gemm_bf16x2_train_nt(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let ap = w.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let mi = n as i32; // kernel M = output cols = n
        let ki = k as i32;
        let bi = m as i32; // kernel B = tokens = m
                           // gfx1151 M-heavy path (e.g. the vocab head: huge n, small batch): stage
                           // the split activation in LDS and reuse across warps. Falls back to the
                           // LDS-free kernel otherwise.
        if self.arch == "gfx1151" && n >= 128 && m >= 16 && k % 16 == 0 {
            self.ensure_kernel(
                "gemm_bf16x2_train_nt_gfx1151_m128",
                kernels::GEMM_BF16X2_TRAIN_NT_SRC,
                "gemm_bf16x2_train_nt_gfx1151_m128",
            )?;
            let grid_m = ((n + 127) / 128) as u32;
            let grid_b = ((m + 15) / 16) as u32;
            return self.launch_kernargs(
                "gemm_bf16x2_train_nt_gfx1151_m128",
                [grid_m, grid_b, 1],
                [256, 1, 1],
                0,
                &kernargs![ptr ap, ptr xp, ptr yp, i32 mi, i32 ki, i32 bi],
            );
        }
        self.ensure_kernel(
            "gemm_bf16x2_train_nt",
            kernels::GEMM_BF16X2_TRAIN_NT_SRC,
            "gemm_bf16x2_train_nt",
        )?;
        let grid_m = ((n + 15) / 16) as u32;
        let grid_b = ((m + 15) / 16) as u32;
        self.launch_kernargs(
            "gemm_bf16x2_train_nt",
            [grid_m, grid_b, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr ap, ptr xp, ptr yp, i32 mi, i32 ki, i32 bi],
        )
    }
    /// F16-compute forward of the training linear op (see `gemm_bf16c_train_nt`);
    /// casts to `_Float16` for higher forward precision at the same WMMA speed.
    pub fn gemm_f16c_train_nt(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_f16c_train_nt",
            kernels::GEMM_F16C_TRAIN_NT_SRC,
            "gemm_f16c_train_nt",
        )?;
        let ap = w.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let mi = n as i32;
        let ki = k as i32;
        let bi = m as i32;
        let grid_m = ((n + 15) / 16) as u32;
        let grid_b = ((m + 15) / 16) as u32;
        self.launch_kernargs(
            "gemm_f16c_train_nt",
            [grid_m, grid_b, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr ap, ptr xp, ptr yp, i32 mi, i32 ki, i32 bi],
        )
    }
    /// Deterministic per-tensor abs-max `max_i |x[i]|`, read back as a scalar.
    /// Used to pick the per-tensor scale for [`Self::gemm_f16s_train_nt`]. `max`
    /// is order-independent, so this stays bit-reproducible.
    pub fn abs_max_f32(&mut self, x: &GpuTensor) -> HipResult<f32> {
        self.bind_thread()?;
        self.ensure_kernel(
            "abs_max_f32",
            kernels::GEMM_F16S_TRAIN_NT_SRC,
            "abs_max_f32",
        )?;
        let n = x.numel();
        // Holds float_as_uint(amax); reinterpreted as f32 on download IS amax.
        let out = self.zeros(&[1], DType::F32)?;
        let xptr = x.buf.as_ptr();
        let outptr = out.buf.as_ptr();
        let ni = n as i32;
        let blocks = (n.div_ceil(256)).clamp(1, 1024) as u32;
        self.launch_kernargs(
            "abs_max_f32",
            [blocks, 1, 1],
            [256, 1, 1],
            0,
            &kernargs![ptr xptr, ptr outptr, i32 ni],
        )?;
        let v = self.download_f32(&out)?;
        self.free_tensor(out)?;
        Ok(v[0])
    }
    /// Scaled f16-compute training GEMM (NT): `y[m,n] = x[m,k]·w[n,k]^T` on the
    /// f16 WMMA cores, but each operand is multiplied by a per-tensor scale
    /// (`sx`/`sw`) before the f16 cast and the f32 accumulator is unscaled by
    /// `1/(sx·sw)` — f16's 10-bit mantissa without its range trap. Same args as
    /// `gemm_f16c_train_nt` plus the scales (pass powers of two for exactness).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16s_train_nt(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
        sx: f32,
        sw: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_f16s_train_nt",
            kernels::GEMM_F16S_TRAIN_NT_SRC,
            "gemm_f16s_train_nt",
        )?;
        let func = &self.functions["gemm_f16s_train_nt"];
        // Kernel: Y[B,M] = X[B,K]·A[M,K]^T with A=w (M=n), X=x (B=m), scale A by
        // sw, X by sx.
        let mut ap = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = n as i32;
        let mut ki = k as i32;
        let mut bi = m as i32;
        let mut saf = sw;
        let mut sxf = sx;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut saf as *mut _ as *mut c_void,
            &mut sxf as *mut _ as *mut c_void,
        ];
        let grid_m = n.div_ceil(16) as u32;
        let grid_b = m.div_ceil(16) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Phase C2 scaled-f16 backward dX: `dX[m,k] = Σ_n dY[m,n]·W[n,k]` (contract
    /// N) on the f16 WMMA cores, reading W strided (no transpose pass). `dy`/`w`
    /// scaled by `s_dy`/`s_w` before the f16 cast; output unscaled by 1/(s_dy·s_w).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16s_nn_train(
        &mut self,
        dy: &GpuTensor,
        w: &GpuTensor,
        dx: &GpuTensor,
        m: usize,
        n: usize,
        k: usize,
        s_dy: f32,
        s_w: f32,
    ) -> HipResult<()> {
        self.gemm_f16s_backward_launch("gemm_f16s_nn_train", dy, w, dx, m, n, k, s_dy, s_w)
    }
    /// Phase C2 scaled-f16 backward dW: `dW[n,k] = Σ_m dY[m,n]·X[m,k]` (contract
    /// M) on the f16 WMMA cores, reading both operands strided (no transposes).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16s_tn_train(
        &mut self,
        dy: &GpuTensor,
        x: &GpuTensor,
        dw: &GpuTensor,
        m: usize,
        n: usize,
        k: usize,
        s_dy: f32,
        s_x: f32,
    ) -> HipResult<()> {
        self.gemm_f16s_backward_launch("gemm_f16s_tn_train", dy, x, dw, m, n, k, s_dy, s_x)
    }
    /// Shared launcher for the C2 backward kernels: both take
    /// `(op_a[.,.], op_b[.,.], out, M, N, K, sa, sb)` and grid over (K, out-rows).
    /// NN out-rows = M; TN out-rows = N; the kernel's blockIdx.y bound handles it.
    #[allow(clippy::too_many_arguments)]
    fn gemm_f16s_backward_launch(
        &mut self,
        kernel: &'static str,
        a: &GpuTensor,
        b: &GpuTensor,
        out: &GpuTensor,
        m: usize,
        n: usize,
        k: usize,
        sa: f32,
        sb: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(kernel, kernels::GEMM_F16S_BACKWARD_SRC, kernel)?;
        let func = &self.functions[kernel];
        let mut ap = a.buf.as_ptr();
        let mut bp = b.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut mi = m as i32;
        let mut ni = n as i32;
        let mut ki = k as i32;
        let mut saf = sa;
        let mut sbf = sb;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut saf as *mut _ as *mut c_void,
            &mut sbf as *mut _ as *mut c_void,
        ];
        // grid.x tiles K (b-rows); grid.y tiles the a-rows (M for NN, N for TN).
        let out_rows = if kernel == "gemm_f16s_tn_train" { n } else { m };
        let grid_k = k.div_ceil(16) as u32;
        let grid_r = out_rows.div_ceil(16) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_k, grid_r, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn gemm_f16_wmma_mb4(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to gemm_f16 (which binds)
        self.gemm_f16(w, x, y, m, k, n)
    }
    pub fn gemm_f16_wmma_mb8(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to gemm_f16 (which binds)
        self.gemm_f16(w, x, y, m, k, n)
    }
    /// WMMA F16 weight × F16 input → F32 output GEMM with (B, M)
    /// output layout. Drop-in for `gemm_f32_register_tiled` once the
    /// weight has been kept on device as F16 and the input has been
    /// staged through `convert_f32_to_f16`.
    pub fn gemm_f16_x_f16_wmma(
        &mut self,
        a_f16: &GpuTensor,
        x_f16: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_f16_x_f16_wmma",
            kernels::GEMM_F16_X_F16_WMMA_SRC,
            "gemm_f16_x_f16_wmma",
        )?;
        let ap = a_f16.buf.as_ptr();
        let xp = x_f16.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let bi = batch_size as i32;
        let grid_m = ((m + 15) / 16) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        self.launch_kernargs(
            "gemm_f16_x_f16_wmma",
            [grid_m, grid_b, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr ap, ptr xp, ptr yp, i32 mi, i32 ki, i32 bi],
        )
    }
    /// WMMA F16 weight × FP32 input → F32 output GEMM with (B, M)
    /// output layout. This stages `x_f32` through the cached FP16 scratch
    /// before launching `gemm_f16_x_f16_wmma`.
    pub fn gemm_f16_x_f32_wmma(
        &mut self,
        a_f16: &GpuTensor,
        x_f32: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            matches!(a_f16.dtype, DType::F16 | DType::Raw),
            "gemm_f16_x_f32_wmma: weights must be F16 or raw F16 payload"
        );
        assert_eq!(
            x_f32.dtype,
            DType::F32,
            "gemm_f16_x_f32_wmma: input must be F32 before FP16 staging"
        );
        assert_eq!(
            y_f32.dtype,
            DType::F32,
            "gemm_f16_x_f32_wmma: output must be F32"
        );
        self.ensure_kernel(
            "gemm_f16_x_f16_wmma",
            kernels::GEMM_F16_X_F16_WMMA_SRC,
            "gemm_f16_x_f16_wmma",
        )?;
        let ap = a_f16.buf.as_ptr();
        let xp = self.ensure_fp16_x(x_f32, batch_size * k)?;
        let yp = y_f32.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let bi = batch_size as i32;
        let grid_m = ((m + 15) / 16) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        self.launch_kernargs(
            "gemm_f16_x_f16_wmma",
            [grid_m, grid_b, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr ap, ptr xp, ptr yp, i32 mi, i32 ki, i32 bi],
        )
    }
    /// WMMA BF16 weight × BF16-staged input → F32 output GEMM with
    /// (B, M) output layout. `a_bf16` is raw BF16 row-major [M, K].
    /// `x_f32` is staged through the cached BF16 scratch once per source
    /// pointer, then consumed by the RDNA wave32 BF16 WMMA kernel.
    /// Generic kernel library: WMMA GEMM, BF16 inputs → BF16 output.
    /// `a_bf16` [M,K], `x_bf16` [B,K], `y_bf16` [B,M], all raw BF16 (u16)
    /// payloads. gfx1103/RDNA3 wave32, zero LDS. Accumulation is in bf16
    /// precision (16-bit-output WMMA), so this trades accuracy for a bf16
    /// output; use the `→F32` sibling when long-K accuracy matters.
    /// Requires `k % 16 == 0` and wave32 WMMA.
    pub fn gemm_bf16_bf16_wmma(
        &mut self,
        a_bf16: &GpuTensor,
        x_bf16: &GpuTensor,
        y_bf16: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(k % 16, 0, "gemm_bf16_bf16_wmma: K must be a multiple of 16");
        self.ensure_kernel(
            "gemm_bf16_bf16_wmma",
            kernels::GEMM_BF16_BF16_WMMA_SRC,
            "gemm_bf16_bf16_wmma",
        )?;
        let ap = a_bf16.buf.as_ptr();
        let xp = x_bf16.buf.as_ptr();
        let yp = y_bf16.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        let func = &self.functions["gemm_bf16_bf16_wmma"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn gemm_f16_f16_wmma(
        &mut self,
        a_f16: &GpuTensor,
        x_f16: &GpuTensor,
        y_f16: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(k % 16, 0, "gemm_f16_f16_wmma: K must be a multiple of 16");
        self.ensure_kernel(
            "gemm_f16_f16_wmma",
            kernels::GEMM_F16_F16_WMMA_SRC,
            "gemm_f16_f16_wmma",
        )?;
        let ap = a_f16.buf.as_ptr();
        let xp = x_f16.buf.as_ptr();
        let yp = y_f16.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        let func = &self.functions["gemm_f16_f16_wmma"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn gemm_bf16_x_bf16_wmma(
        &mut self,
        a_bf16: &GpuTensor,
        x_f32: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.gemm_bf16_x_bf16_wmma_labeled(
            a_bf16,
            x_f32,
            y_f32,
            m,
            k,
            batch_size,
            "gemm_bf16_x_bf16_wmma",
        )
    }
    pub fn gemm_bf16_x_bf16_wmma_labeled(
        &mut self,
        a_bf16: &GpuTensor,
        x_f32: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        profile_label: &'static str,
    ) -> HipResult<()> {
        // Calibration capture: the BF16 linear chokepoint (the lowered super-op
        // path's bf16 gemv funnels here via gemm_bf16_x_bf16_wmma). Captures the
        // input activation before the compute. No-op when unarmed.
        self.maybe_capture_activation(a_bf16, x_f32, batch_size, k);
        if self.arch == "gfx1151"
            && m >= 128
            && batch_size >= 16
            && k % 16 == 0
            && std::env::var("HIPFIRE_BF16_DENSE_M128").ok().as_deref() != Some("0")
        {
            return self.gemm_bf16_x_bf16_wmma_gfx1151_m128_labeled(
                a_bf16,
                x_f32,
                y_f32,
                m,
                k,
                batch_size,
                profile_label,
            );
        }
        self.bind_thread()?;
        assert_eq!(
            a_bf16.dtype,
            DType::BF16,
            "gemm_bf16_x_bf16_wmma: weights must be BF16"
        );
        assert_eq!(
            x_f32.dtype,
            DType::F32,
            "gemm_bf16_x_bf16_wmma: input must be F32 before BF16 staging"
        );
        assert_eq!(
            y_f32.dtype,
            DType::F32,
            "gemm_bf16_x_bf16_wmma: output must be F32"
        );
        self.ensure_kernel(
            "gemm_bf16_x_bf16_wmma",
            kernels::GEMM_BF16_X_BF16_WMMA_SRC,
            "gemm_bf16_x_bf16_wmma",
        )?;
        let ap = a_bf16.buf.as_ptr();
        let xp = self.ensure_bf16_x(x_f32, batch_size * k)?;
        let yp = y_f32.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let bi = batch_size as i32;
        let grid_m = ((m + 15) / 16) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        let bytes = m * k * 2 + batch_size * k * 2 + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", profile_label, bytes);
        let result = self.launch_kernargs(
            "gemm_bf16_x_bf16_wmma",
            [grid_m, grid_b, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr ap, ptr xp, ptr yp, i32 mi, i32 ki, i32 bi],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Register-tiled, zero-LDS BF16×BF16→F32 WMMA GEMM. Same I/O contract as
    /// `gemm_bf16_x_bf16_wmma` (A[M,K] BF16, X[B,K] F32 staged to BF16, Y[B,M]
    /// F32) but each wave computes an MB×NB grid of 16×16 output subtiles with
    /// MB×NB independent accumulators — ILP hides the WMMA latency the baseline's
    /// single dependent chain cannot. `(mb, nb)` selects the compiled tiling
    /// entrypoint: (2,2), (4,2) or (4,4).
    pub fn gemm_bf16_tiled_wmma(
        &mut self,
        a_bf16: &GpuTensor,
        x_f32: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        mb: usize,
        nb: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            a_bf16.dtype,
            DType::BF16,
            "gemm_bf16_tiled_wmma: weights BF16"
        );
        let kname = match (mb, nb) {
            (2, 2) => "gemm_bf16_tiled_wmma_2x2",
            (4, 2) => "gemm_bf16_tiled_wmma_4x2",
            (4, 4) => "gemm_bf16_tiled_wmma_4x4",
            _ => panic!("gemm_bf16_tiled_wmma: unsupported tiling {mb}x{nb}"),
        };
        self.ensure_kernel(kname, kernels::GEMM_BF16_TILED_WMMA_SRC, kname)?;
        let ap = a_bf16.buf.as_ptr();
        let xp = self.ensure_bf16_x(x_f32, batch_size * k)?;
        let yp = y_f32.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let bi = batch_size as i32;
        let grid_m = m.div_ceil(16 * mb) as u32;
        let grid_b = batch_size.div_ceil(16 * nb) as u32;
        let bytes = m * k * 2 + batch_size * k * 2 + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kname, bytes);
        let result = self.launch_kernargs(
            kname,
            [grid_m, grid_b, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr ap, ptr xp, ptr yp, i32 mi, i32 ki, i32 bi],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// LDS-staged, double-buffered, register-super-tiled bf16 GEMM (gfx1103 wave32)
    /// — the DiT throughput kernel. Same contract as [`Self::gemm_bf16_tiled_wmma`]
    /// (bf16 weight × f32 activation staged to bf16, F32 [B,M] output) and bit-exact
    /// to it, but LDS-staged for far higher occupancy. Requires `k % 64 == 0`.
    /// Parity: `parity_gemm_bf16_tiled_wmma_lds`.
    pub fn gemm_bf16_tiled_wmma_lds(
        &mut self,
        a_bf16: &GpuTensor,
        x_f32: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(a_bf16.dtype, DType::BF16, "gemm_bf16_tiled_wmma_lds: weights BF16");
        assert_eq!(k % 64, 0, "gemm_bf16_tiled_wmma_lds: K must be a multiple of 64");
        self.ensure_kernel(
            "gemm_bf16_tiled_wmma_lds",
            kernels::GEMM_BF16_TILED_WMMA_LDS_SRC,
            "gemm_bf16_tiled_wmma_lds",
        )?;
        let ap = a_bf16.buf.as_ptr();
        let xp = self.ensure_bf16_x(x_f32, batch_size * k)?;
        let yp = y_f32.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let bi = batch_size as i32;
        let grid_m = m.div_ceil(64) as u32; // BM = 64
        let grid_b = batch_size.div_ceil(128) as u32; // BN = 128
        let bytes = m * k * 2 + batch_size * k * 2 + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_bf16_tiled_wmma_lds", bytes);
        let result = self.launch_kernargs(
            "gemm_bf16_tiled_wmma_lds",
            [grid_m, grid_b, 1],
            [256, 1, 1],
            0,
            &kernargs![ptr ap, ptr xp, ptr yp, i32 mi, i32 ki, i32 bi],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Register-tiled F32 batched GEMM. Y[batch, M] = A[M,K] @ x[batch,K]^T.
    /// Each block holds BATCH_TILE=8 accumulators in registers and
    /// reuses each loaded weight element across them — amortizing
    /// weight bandwidth, which is the prefill bottleneck on unified-
    /// memory Strix Halo. Replaces the fake-batched `gemm_f32_batched`
    /// for prefill (where weight reads dominate).
    pub fn gemm_f32_register_tiled(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (kname, src, batch_tile, block_x) = (
            "gemm_f32_register_tiled",
            kernels::GEMM_F32_REGISTER_TILED_SRC,
            8u32,
            32u32,
        );
        self.ensure_kernel(kname, src, kname)?;
        let ap = a.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let bs = batch_size as i32;
        let grid_y = (batch_size as u32 + batch_tile - 1) / batch_tile;
        let bytes = m * k * 4 + batch_size * k * 4 + batch_size * m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_f32_register_tiled", bytes);
        let result = self.launch_kernargs(
            kname,
            [m as u32, grid_y, 1],
            [block_x, 1, 1],
            0,
            &kernargs![ptr ap, ptr xp, ptr yp, i32 mi, i32 ki, i32 bs],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// BATCH_TILE=16 sibling of `gemm_f32_register_tiled`. This is useful for
    /// large-vocab lm_head/reference paths where the M dimension is enormous
    /// and reducing CTA count matters more than preserving the BATCH_TILE=8
    /// register budget used by production prefill paths.
    pub fn gemm_f32_register_tiled_b16(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (kname, src, batch_tile, block_x) = (
            "gemm_f32_register_tiled_b16",
            kernels::GEMM_F32_REGISTER_TILED_B16_SRC,
            16u32,
            32u32,
        );
        self.ensure_kernel(kname, src, kname)?;
        let ap = a.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let bs = batch_size as i32;
        let grid_y = (batch_size as u32 + batch_tile - 1) / batch_tile;
        let bytes = m * k * 4 + batch_size * k * 4 + batch_size * m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_f32_register_tiled_b16", bytes);
        let result = self.launch_kernargs(
            kname,
            [m as u32, grid_y, 1],
            [block_x, 1, 1],
            0,
            &kernargs![ptr ap, ptr xp, ptr yp, i32 mi, i32 ki, i32 bs],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// BATCH_TILE=32 sibling for large-vocab lm_head/reference paths.
    pub fn gemm_f32_register_tiled_b32(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (kname, src, batch_tile, block_x) = (
            "gemm_f32_register_tiled_b32",
            kernels::GEMM_F32_REGISTER_TILED_B32_SRC,
            32u32,
            32u32,
        );
        self.ensure_kernel(kname, src, kname)?;
        let ap = a.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let bs = batch_size as i32;
        let grid_y = (batch_size as u32 + batch_tile - 1) / batch_tile;
        let bytes = m * k * 4 + batch_size * k * 4 + batch_size * m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_f32_register_tiled_b32", bytes);
        let result = self.launch_kernargs(
            kname,
            [m as u32, grid_y, 1],
            [block_x, 1, 1],
            0,
            &kernargs![ptr ap, ptr xp, ptr yp, i32 mi, i32 ki, i32 bs],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// BATCH_TILE=64 sibling for large-vocab lm_head/reference paths.
    pub fn gemm_f32_register_tiled_b64(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (kname, src, batch_tile, block_x) = (
            "gemm_f32_register_tiled_b64",
            kernels::GEMM_F32_REGISTER_TILED_B64_SRC,
            64u32,
            32u32,
        );
        self.ensure_kernel(kname, src, kname)?;
        let ap = a.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let bs = batch_size as i32;
        let grid_y = (batch_size as u32 + batch_tile - 1) / batch_tile;
        let bytes = m * k * 4 + batch_size * k * 4 + batch_size * m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_f32_register_tiled_b64", bytes);
        let result = self.launch_kernargs(
            kname,
            [m as u32, grid_y, 1],
            [block_x, 1, 1],
            0,
            &kernargs![ptr ap, ptr xp, ptr yp, i32 mi, i32 ki, i32 bs],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
