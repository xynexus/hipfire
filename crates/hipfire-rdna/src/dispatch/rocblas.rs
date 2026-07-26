// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! rocBLAS GEMM fallback wrappers + arch-eligibility helpers. Pure move (Phase 1 M1).

use super::Gpu;
use hip_bridge::{DeviceBuffer, HipResult};
use std::ffi::c_void;
use std::sync::OnceLock;

impl Gpu {
    /// BF16 row-major `Y[N,M] = X[N,K] * W[M,K]^T` with F32 accumulation and
    /// BF16 output. Keeping the destination type inside `rocblas_gemm_ex` is
    /// part of the library's solution-selection contract and can differ from
    /// an F32 destination followed by a separate cast.
    pub fn rocblas_gemm_bf16_nt_bf16(
        &self,
        w_bf16: &super::GpuTensor,
        x_bf16: &super::GpuTensor,
        y_bf16: &super::GpuTensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        use hip_bridge::{RocblasDatatype, RocblasOperation};
        self.bind_thread()?;
        let rb = self
            .rocblas
            .as_ref()
            .expect("rocblas_gemm_bf16_nt_bf16: rocBLAS not initialized");
        let alpha = 1.0f32;
        let beta = 0.0f32;
        unsafe {
            rb.gemm_ex(
                RocblasOperation::Transpose,
                RocblasOperation::None,
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const f32 as *const c_void,
                w_bf16.buf.as_ptr(),
                RocblasDatatype::Bf16,
                k as i32,
                x_bf16.buf.as_ptr(),
                RocblasDatatype::Bf16,
                k as i32,
                &beta as *const f32 as *const c_void,
                y_bf16.buf.as_ptr(),
                RocblasDatatype::Bf16,
                m as i32,
                y_bf16.buf.as_ptr(),
                RocblasDatatype::Bf16,
                m as i32,
                RocblasDatatype::F32,
            )
            .map_err(|error| {
                hip_bridge::HipError::new(
                    error.status,
                    &format!("rocblas BF16 NT GEMM BF16 output: {}", error.context),
                )
            })
        }
    }

    /// BF16 row-major `Y[N,M] = X[N,K] * W[M,K]^T` with F32 accumulation and
    /// output. Used by numerical parity probes for ROCm SDPA's math backend.
    pub fn rocblas_gemm_bf16_nt_f32(
        &self,
        w_bf16: &super::GpuTensor,
        x_bf16: &super::GpuTensor,
        y_f32: &super::GpuTensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        use hip_bridge::{RocblasDatatype, RocblasOperation};
        self.bind_thread()?;
        let rb = self
            .rocblas
            .as_ref()
            .expect("rocblas_gemm_bf16_nt_f32: rocBLAS not initialized");
        let alpha = 1.0f32;
        let beta = 0.0f32;
        unsafe {
            rb.gemm_ex(
                RocblasOperation::Transpose,
                RocblasOperation::None,
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const f32 as *const c_void,
                w_bf16.buf.as_ptr(),
                RocblasDatatype::Bf16,
                k as i32,
                x_bf16.buf.as_ptr(),
                RocblasDatatype::Bf16,
                k as i32,
                &beta as *const f32 as *const c_void,
                y_f32.buf.as_ptr(),
                RocblasDatatype::F32,
                m as i32,
                y_f32.buf.as_ptr(),
                RocblasDatatype::F32,
                m as i32,
                RocblasDatatype::F32,
            )
            .map_err(|error| {
                hip_bridge::HipError::new(
                    error.status,
                    &format!("rocblas BF16 NT GEMM: {}", error.context),
                )
            })
        }
    }

    /// F32 row-major `Y[N,M] = A[N,K] * B[K,M]`. `B[K,M]` is byte-viewed as
    /// column-major `[M,K]`. SDPA math promotes BF16 values to F32 for this
    /// intermediate GEMM.
    pub fn rocblas_gemm_f32_nn_f32(
        &self,
        b_f32: &super::GpuTensor,
        a_f32: &super::GpuTensor,
        y_f32: &super::GpuTensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        use hip_bridge::{RocblasDatatype, RocblasOperation};
        self.bind_thread()?;
        let rb = self
            .rocblas
            .as_ref()
            .expect("rocblas_gemm_f32_nn_f32: rocBLAS not initialized");
        let alpha = 1.0f32;
        let beta = 0.0f32;
        unsafe {
            rb.gemm_ex(
                RocblasOperation::None,
                RocblasOperation::None,
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const f32 as *const c_void,
                b_f32.buf.as_ptr(),
                RocblasDatatype::F32,
                m as i32,
                a_f32.buf.as_ptr(),
                RocblasDatatype::F32,
                k as i32,
                &beta as *const f32 as *const c_void,
                y_f32.buf.as_ptr(),
                RocblasDatatype::F32,
                m as i32,
                y_f32.buf.as_ptr(),
                RocblasDatatype::F32,
                m as i32,
                RocblasDatatype::F32,
            )
            .map_err(|error| {
                hip_bridge::HipError::new(
                    error.status,
                    &format!("rocblas BF16 NN GEMM: {}", error.context),
                )
            })
        }
    }

    /// One Gemma 4 math-SDPA QK row over a strided, position-major K tensor.
    ///
    /// `K` is `[positions, kv_width]`, with `k_bf16` already offset to one KV
    /// head. `Q` is one contiguous head. The output is `Q * K^T` in F32.
    pub fn rocblas_sdpa_qk_strided_bf16_f32(
        &self,
        k_bf16: &super::GpuTensor,
        q_bf16: &super::GpuTensor,
        scores_f32: &super::GpuTensor,
        n_valid: usize,
        head_dim: usize,
        kv_width: usize,
    ) -> HipResult<()> {
        use hip_bridge::{RocblasDatatype, RocblasOperation};
        self.bind_thread()?;
        let rb = self
            .rocblas
            .as_ref()
            .expect("rocblas_sdpa_qk_strided_bf16_f32: rocBLAS not initialized");
        let alpha = 1.0f32;
        let beta = 0.0f32;
        unsafe {
            rb.gemm_ex(
                RocblasOperation::Transpose,
                RocblasOperation::None,
                n_valid as i32,
                1,
                head_dim as i32,
                &alpha as *const f32 as *const c_void,
                k_bf16.buf.as_ptr(),
                RocblasDatatype::Bf16,
                kv_width as i32,
                q_bf16.buf.as_ptr(),
                RocblasDatatype::Bf16,
                head_dim as i32,
                &beta as *const f32 as *const c_void,
                scores_f32.buf.as_ptr(),
                RocblasDatatype::F32,
                n_valid as i32,
                scores_f32.buf.as_ptr(),
                RocblasDatatype::F32,
                n_valid as i32,
                RocblasDatatype::F32,
            )
            .map_err(|error| {
                hip_bridge::HipError::new(
                    error.status,
                    &format!("rocBLAS strided BF16 QK: {}", error.context),
                )
            })
        }
    }

    /// One Gemma 4 math-SDPA PV row over a strided, position-major V tensor.
    ///
    /// `V` is `[positions, kv_width]`, with `v_f32` already offset to one KV
    /// head. Probabilities are a contiguous F32 row of length `n_valid`.
    pub fn rocblas_sdpa_pv_strided_f32(
        &self,
        v_f32: &super::GpuTensor,
        probabilities_f32: &super::GpuTensor,
        output_f32: &super::GpuTensor,
        n_valid: usize,
        head_dim: usize,
        kv_width: usize,
    ) -> HipResult<()> {
        use hip_bridge::{RocblasDatatype, RocblasOperation};
        self.bind_thread()?;
        let rb = self
            .rocblas
            .as_ref()
            .expect("rocblas_sdpa_pv_strided_f32: rocBLAS not initialized");
        let alpha = 1.0f32;
        let beta = 0.0f32;
        unsafe {
            rb.gemm_ex(
                RocblasOperation::None,
                RocblasOperation::None,
                head_dim as i32,
                1,
                n_valid as i32,
                &alpha as *const f32 as *const c_void,
                v_f32.buf.as_ptr(),
                RocblasDatatype::F32,
                kv_width as i32,
                probabilities_f32.buf.as_ptr(),
                RocblasDatatype::F32,
                n_valid as i32,
                &beta as *const f32 as *const c_void,
                output_f32.buf.as_ptr(),
                RocblasDatatype::F32,
                head_dim as i32,
                output_f32.buf.as_ptr(),
                RocblasDatatype::F32,
                head_dim as i32,
                RocblasDatatype::F32,
            )
            .map_err(|error| {
                hip_bridge::HipError::new(
                    error.status,
                    &format!("rocBLAS strided F32 PV: {}", error.context),
                )
            })
        }
    }

    /// CDNA3-only: prefill GEMM used by `gemm_hfq4g256` rocBLAS path.
    ///
    /// Computes Y_rowmajor[N × M] = X_rowmajor[N × K] · W_transposed, where
    /// the weight is stored row-major [M × K] but the operation needs W^T.
    /// This matches the engine's convention (weight dotted with each row of X
    /// produces one output column per batch row).
    ///
    /// rocBLAS is column-major. A row-major [M × K] matrix is byte-identical
    /// to a column-major [K × M] matrix. So the call is:
    ///   col-major C[M × N] = op_A(W) · X_col[K × N]
    /// with op_A = T (transpose the col-major [K × M] view of W to get [M × K]).
    /// X_row[N × K] viewed col-major is [K × N] with ld=K. Y_row[N × M] viewed
    /// col-major is [M × N] with ld=M — so pointer+ld match C directly.
    pub fn rocblas_gemm_hfq4_prefill(
        &self,
        w_fp16: &DeviceBuffer, // row-major [M × K]
        x_fp16: &DeviceBuffer, // row-major [N × K]
        y_fp32: &DeviceBuffer, // row-major [N × M]
        m: usize,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rocblas_gemm_hfq4_generic(w_fp16, x_fp16, y_fp32, m, n, k, 1.0, 0.0)
    }
    /// Same op as `rocblas_gemm_hfq4_prefill` but with Y += alpha·(X·W^T) +
    /// beta·Y. Covers the residual-GEMM pattern (w_down on LA path, wo on
    /// attention path) where the existing hand-rolled kernels fuse the add.
    pub fn rocblas_gemm_hfq4_prefill_residual(
        &self,
        w_fp16: &DeviceBuffer,
        x_fp16: &DeviceBuffer,
        y_fp32: &DeviceBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rocblas_gemm_hfq4_generic(w_fp16, x_fp16, y_fp32, m, n, k, 1.0, 1.0)
    }
    fn rocblas_gemm_hfq4_generic(
        &self,
        w_fp16: &DeviceBuffer,
        x_fp16: &DeviceBuffer,
        y_fp32: &DeviceBuffer,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> HipResult<()> {
        use hip_bridge::{RocblasDatatype, RocblasOperation};
        let rb = self
            .rocblas
            .as_ref()
            .expect("rocblas_gemm_hfq4: rocBLAS not initialized");
        unsafe {
            rb.gemm_ex(
                RocblasOperation::Transpose,
                RocblasOperation::None,
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const f32 as *const c_void,
                w_fp16.as_ptr(),
                RocblasDatatype::F16,
                k as i32,
                x_fp16.as_ptr(),
                RocblasDatatype::F16,
                k as i32,
                &beta as *const f32 as *const c_void,
                y_fp32.as_ptr(),
                RocblasDatatype::F32,
                m as i32,
                y_fp32.as_ptr(),
                RocblasDatatype::F32,
                m as i32,
                RocblasDatatype::F32,
            )
            .map_err(|e| {
                hip_bridge::HipError::new(e.status, &format!("rocblas_gemm: {}", e.context))
            })
        }
    }
    /// Whether the arch is eligible for the rocBLAS/MFMA batched-prefill
    /// path. Default: CDNA3 only (MI300-series, gfx94x). Override with
    /// `HIPFIRE_ROCBLAS_ALL_ARCHS=1` for local testing on RDNA3+ — rocBLAS
    /// runs fine there (uses WMMA backends on RDNA3, not MFMA) so this is
    /// a useful smoke-path in the absence of an MI300.
    pub(crate) fn rocblas_arch_eligible(&self) -> bool {
        static CACHE: OnceLock<bool> = OnceLock::new();
        let all_archs = *CACHE.get_or_init(|| self.flags.rocblas_all_archs);
        if all_archs {
            return self.rocblas.is_some();
        }
        self.arch_caps.is_cdna3()
    }
    /// Configurable batch threshold for MFMA dispatch. Below this we stay on
    /// the hand-rolled GEMV — rocBLAS launch overhead eats the compute win
    /// at tiny batches. Overridable via `HIPFIRE_ROCBLAS_MIN_BATCH` env var.
    ///
    /// Kill-switch: `HIPFIRE_ROCBLAS_OFF=1` forces the threshold to usize::MAX,
    /// which disables the rocBLAS path entirely for A/B benchmarking against
    /// the hand-rolled GEMV baseline.
    pub(crate) fn rocblas_min_batch(&self) -> usize {
        static CACHE: OnceLock<usize> = OnceLock::new();
        *CACHE.get_or_init(|| {
            if self.flags.rocblas_off {
                return usize::MAX;
            }
            self.flags.rocblas_min_batch.unwrap_or(4)
        })
    }
}
