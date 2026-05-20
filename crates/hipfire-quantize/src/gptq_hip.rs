//! GPU Cholesky for GPTQ via rocSOLVER (FP64).
//!
//! See `docs/plans/gptq-hip-port.md` for design rationale. Replaces the
//! dominant O(K^3) CPU Cholesky in `compute_damped_inv_cholesky_upper`
//! (gptq.rs:193-335). At K=12288, CPU is ~6 min/tensor; rocSOLVER targets ~12 s.
//!
//! Opt-in via `gptq-hip` Cargo feature. Linked at runtime via libloading.
//! On failure to load rocSOLVER + rocBLAS + libamdhip64, callers fall
//! through to CPU silently. CDNA-gating is enforced at the main.rs layer.

#![cfg(feature = "gptq-hip")]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::c_void;
use std::os::raw::{c_double, c_int};

use faer::Mat;
use libloading::{Library, Symbol};

use crate::gptq::CholeskyError;

const ROCBLAS_FILL_LOWER: c_int = 121;
const ROCBLAS_FILL_UPPER: c_int = 122;
const ROCBLAS_DIAG_NON_UNIT: c_int = 131;
const ROCBLAS_OPERATION_NONE: c_int = 111;
const ROCBLAS_OPERATION_TRANSPOSE: c_int = 112;
const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
const HIP_SUCCESS: c_int = 0;
const ROCBLAS_STATUS_SUCCESS: c_int = 0;

pub type rocblas_handle = *mut c_void;
type hipError_t = c_int;
type rocblas_status = c_int;

type FnHipMalloc = unsafe extern "C" fn(ptr: *mut *mut c_void, size: usize) -> hipError_t;
type FnHipFree = unsafe extern "C" fn(ptr: *mut c_void) -> hipError_t;
type FnHipMemcpy = unsafe extern "C" fn(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> hipError_t;
type FnHipDeviceSynchronize = unsafe extern "C" fn() -> hipError_t;
type FnRocblasCreateHandle = unsafe extern "C" fn(handle: *mut rocblas_handle) -> rocblas_status;
type FnRocblasDestroyHandle = unsafe extern "C" fn(handle: rocblas_handle) -> rocblas_status;
type FnRocsolverDpotrf = unsafe extern "C" fn(handle: rocblas_handle, uplo: c_int, n: c_int, A: *mut c_double, lda: c_int, info: *mut c_int) -> rocblas_status;
type FnRocsolverDtrtri = unsafe extern "C" fn(handle: rocblas_handle, uplo: c_int, diag: c_int, n: c_int, A: *mut c_double, lda: c_int, info: *mut c_int) -> rocblas_status;
type FnRocblasDsyrk = unsafe extern "C" fn(handle: rocblas_handle, uplo: c_int, transA: c_int, n: c_int, k: c_int, alpha: *const c_double, A: *const c_double, lda: c_int, beta: *const c_double, C: *mut c_double, ldc: c_int) -> rocblas_status;
type FnRocblasDgeam = unsafe extern "C" fn(handle: rocblas_handle, transA: c_int, transB: c_int, m: c_int, n: c_int, alpha: *const c_double, A: *const c_double, lda: c_int, beta: *const c_double, B: *const c_double, ldb: c_int, C: *mut c_double, ldc: c_int) -> rocblas_status;

pub struct RocSolver {
    handle: rocblas_handle,
    fn_hip_malloc: FnHipMalloc,
    fn_hip_free: FnHipFree,
    fn_hip_memcpy: FnHipMemcpy,
    fn_hip_device_sync: FnHipDeviceSynchronize,
    fn_rocblas_destroy_handle: FnRocblasDestroyHandle,
    fn_rocsolver_dpotrf: FnRocsolverDpotrf,
    fn_rocsolver_dtrtri: FnRocsolverDtrtri,
    fn_rocblas_dsyrk: FnRocblasDsyrk,
    fn_rocblas_dgeam: FnRocblasDgeam,
    _libhip: Library,
    _librocblas: Library,
    _librocsolver: Library,
}

unsafe impl Send for RocSolver {}
unsafe impl Sync for RocSolver {}

impl RocSolver {
    pub fn load() -> Result<Self, String> {
        unsafe {
            let libhip = open_versioned(&["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so.5"]).map_err(|e| format!("libamdhip64: {e}"))?;
            let librocblas = open_versioned(&["librocblas.so", "librocblas.so.4", "librocblas.so.3"]).map_err(|e| format!("librocblas: {e}"))?;
            let librocsolver = open_versioned(&["librocsolver.so", "librocsolver.so.4", "librocsolver.so.3", "librocsolver.so.2", "librocsolver.so.1"]).map_err(|e| format!("librocsolver: {e}"))?;

            let fn_hip_malloc = *(libhip.get::<FnHipMalloc>(b"hipMalloc").map_err(|e| format!("hipMalloc: {e}"))?);
            let fn_hip_free = *(libhip.get::<FnHipFree>(b"hipFree").map_err(|e| format!("hipFree: {e}"))?);
            let fn_hip_memcpy = *(libhip.get::<FnHipMemcpy>(b"hipMemcpy").map_err(|e| format!("hipMemcpy: {e}"))?);
            let fn_hip_device_sync = *(libhip.get::<FnHipDeviceSynchronize>(b"hipDeviceSynchronize").map_err(|e| format!("hipDeviceSynchronize: {e}"))?);
            let fn_create_handle: Symbol<FnRocblasCreateHandle> = librocblas.get(b"rocblas_create_handle").map_err(|e| format!("rocblas_create_handle: {e}"))?;
            let fn_rocblas_destroy_handle = *(librocblas.get::<FnRocblasDestroyHandle>(b"rocblas_destroy_handle").map_err(|e| format!("rocblas_destroy_handle: {e}"))?);
            let fn_rocsolver_dpotrf = *(librocsolver.get::<FnRocsolverDpotrf>(b"rocsolver_dpotrf").map_err(|e| format!("rocsolver_dpotrf: {e}"))?);
            let fn_rocsolver_dtrtri = *(librocsolver.get::<FnRocsolverDtrtri>(b"rocsolver_dtrtri").map_err(|e| format!("rocsolver_dtrtri: {e}"))?);
            let fn_rocblas_dsyrk = *(librocblas.get::<FnRocblasDsyrk>(b"rocblas_dsyrk").map_err(|e| format!("rocblas_dsyrk: {e}"))?);
            let fn_rocblas_dgeam = *(librocblas.get::<FnRocblasDgeam>(b"rocblas_dgeam").map_err(|e| format!("rocblas_dgeam: {e}"))?);

            let mut handle: rocblas_handle = std::ptr::null_mut();
            let status = (*fn_create_handle)(&mut handle as *mut rocblas_handle);
            if status != ROCBLAS_STATUS_SUCCESS {
                return Err(format!("rocblas_create_handle returned status {status}"));
            }

            Ok(Self {
                handle, fn_hip_malloc, fn_hip_free, fn_hip_memcpy, fn_hip_device_sync,
                fn_rocblas_destroy_handle, fn_rocsolver_dpotrf, fn_rocsolver_dtrtri,
                fn_rocblas_dsyrk, fn_rocblas_dgeam,
                _libhip: libhip, _librocblas: librocblas, _librocsolver: librocsolver,
            })
        }
    }

    fn handle(&self) -> rocblas_handle { self.handle }
}

impl Drop for RocSolver {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                let _ = (self.fn_rocblas_destroy_handle)(self.handle);
                self.handle = std::ptr::null_mut();
            }
        }
    }
}

fn open_versioned(candidates: &[&str]) -> Result<Library, String> {
    let mut last_err = String::from("no candidates tried");
    for name in candidates {
        match unsafe { Library::new(name) } {
            Ok(lib) => return Ok(lib),
            Err(e) => last_err = format!("{name}: {e}"),
        }
    }
    Err(last_err)
}

pub fn compute_damped_inv_cholesky_upper_hip(
    solver: &RocSolver,
    h: &Mat<f64>,
    perm: Option<&[usize]>,
    initial_damp: f64,
    max_damp_multiplier: f64,
) -> Result<(Mat<f64>, f64), CholeskyError> {
    let k = h.nrows();
    assert_eq!(k, h.ncols(), "H must be square");
    if let Some(p) = perm { assert_eq!(p.len(), k, "perm length must equal K"); }

    let mut host_a: Vec<f64> = vec![0.0; k * k];
    let mut diag_mean = 0.0_f64;
    for col in 0..k {
        let src_col = perm.map(|p| p[col]).unwrap_or(col);
        for row in 0..k {
            let src_row = perm.map(|p| p[row]).unwrap_or(row);
            host_a[col * k + row] = h[(src_row, src_col)];
        }
        diag_mean += host_a[col * k + col];
    }
    diag_mean /= k as f64;

    if !diag_mean.is_finite() || diag_mean <= 0.0 {
        return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: initial_damp, k, diag_mean });
    }
    let damp_cap = max_damp_multiplier * diag_mean;
    let mut damp = initial_damp * diag_mean;

    let bytes = k * k * std::mem::size_of::<f64>();
    let mut d_a: *mut c_void = std::ptr::null_mut();
    let mut d_h_inv: *mut c_void = std::ptr::null_mut();
    let mut d_u: *mut c_void = std::ptr::null_mut();
    let mut d_info: *mut c_void = std::ptr::null_mut();
    let mut effective_damp = damp;

    unsafe {
        let oom = || CholeskyError::SingularEvenWithMaxDamp { max_damp: damp, k, diag_mean };
        if (solver.fn_hip_malloc)(&mut d_a as *mut *mut c_void, bytes) != HIP_SUCCESS { return Err(oom()); }
        if (solver.fn_hip_malloc)(&mut d_h_inv as *mut *mut c_void, bytes) != HIP_SUCCESS {
            let _ = (solver.fn_hip_free)(d_a); return Err(oom());
        }
        if (solver.fn_hip_malloc)(&mut d_u as *mut *mut c_void, bytes) != HIP_SUCCESS {
            let _ = (solver.fn_hip_free)(d_a); let _ = (solver.fn_hip_free)(d_h_inv); return Err(oom());
        }
        if (solver.fn_hip_malloc)(&mut d_info as *mut *mut c_void, std::mem::size_of::<c_int>()) != HIP_SUCCESS {
            let _ = (solver.fn_hip_free)(d_a); let _ = (solver.fn_hip_free)(d_h_inv); let _ = (solver.fn_hip_free)(d_u); return Err(oom());
        }

        let cleanup = |s: &RocSolver| {
            let _ = (s.fn_hip_free)(d_a);
            let _ = (s.fn_hip_free)(d_h_inv);
            let _ = (s.fn_hip_free)(d_u);
            let _ = (s.fn_hip_free)(d_info);
        };

        // Adaptive damp retry loop on first dpotrf.
        loop {
            for i in 0..k {
                let src_i = perm.map(|p| p[i]).unwrap_or(i);
                host_a[i * k + i] = h[(src_i, src_i)] + damp;
            }
            if (solver.fn_hip_memcpy)(d_a, host_a.as_ptr() as *const c_void, bytes, HIP_MEMCPY_HOST_TO_DEVICE) != HIP_SUCCESS {
                cleanup(solver);
                return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: damp, k, diag_mean });
            }
            let status = (solver.fn_rocsolver_dpotrf)(solver.handle(), ROCBLAS_FILL_LOWER, k as c_int, d_a as *mut c_double, k as c_int, d_info as *mut c_int);
            if status != ROCBLAS_STATUS_SUCCESS {
                cleanup(solver);
                return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: damp, k, diag_mean });
            }
            let mut info_host: c_int = 0;
            if (solver.fn_hip_memcpy)(&mut info_host as *mut c_int as *mut c_void, d_info, std::mem::size_of::<c_int>(), HIP_MEMCPY_DEVICE_TO_HOST) != HIP_SUCCESS {
                cleanup(solver);
                return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: damp, k, diag_mean });
            }
            if info_host == 0 {
                effective_damp = damp;
                break;
            }
            damp *= 10.0;
            if damp > damp_cap {
                cleanup(solver);
                return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: damp_cap, k, diag_mean });
            }
        }

        // dtrtri: L -> L^-1 in lower triangle.
        let mut info_host: c_int = 0;
        let status = (solver.fn_rocsolver_dtrtri)(solver.handle(), ROCBLAS_FILL_LOWER, ROCBLAS_DIAG_NON_UNIT, k as c_int, d_a as *mut c_double, k as c_int, d_info as *mut c_int);
        if status != ROCBLAS_STATUS_SUCCESS {
            cleanup(solver);
            return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: effective_damp, k, diag_mean });
        }
        let _ = (solver.fn_hip_memcpy)(&mut info_host as *mut c_int as *mut c_void, d_info, std::mem::size_of::<c_int>(), HIP_MEMCPY_DEVICE_TO_HOST);
        if info_host != 0 {
            cleanup(solver);
            return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: effective_damp, k, diag_mean });
        }

        // dsyrk: H_inv (upper) = L_inv^T * L_inv.
        let alpha: c_double = 1.0;
        let beta: c_double = 0.0;
        let status = (solver.fn_rocblas_dsyrk)(solver.handle(), ROCBLAS_FILL_UPPER, ROCBLAS_OPERATION_TRANSPOSE, k as c_int, k as c_int, &alpha, d_a as *const c_double, k as c_int, &beta, d_h_inv as *mut c_double, k as c_int);
        if status != ROCBLAS_STATUS_SUCCESS {
            cleanup(solver);
            return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: effective_damp, k, diag_mean });
        }

        // Second dpotrf on H_inv (UPLO=Upper).
        let status = (solver.fn_rocsolver_dpotrf)(solver.handle(), ROCBLAS_FILL_UPPER, k as c_int, d_h_inv as *mut c_double, k as c_int, d_info as *mut c_int);
        if status != ROCBLAS_STATUS_SUCCESS {
            cleanup(solver);
            return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: effective_damp, k, diag_mean });
        }
        let _ = (solver.fn_hip_memcpy)(&mut info_host as *mut c_int as *mut c_void, d_info, std::mem::size_of::<c_int>(), HIP_MEMCPY_DEVICE_TO_HOST);
        if info_host != 0 {
            cleanup(solver);
            return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: effective_damp, k, diag_mean });
        }

        // dgeam: U = (L_HI)^T transposed into d_u.
        let status = (solver.fn_rocblas_dgeam)(solver.handle(), ROCBLAS_OPERATION_TRANSPOSE, ROCBLAS_OPERATION_NONE, k as c_int, k as c_int, &alpha, d_h_inv as *const c_double, k as c_int, &beta, std::ptr::null(), k as c_int, d_u as *mut c_double, k as c_int);
        if status != ROCBLAS_STATUS_SUCCESS {
            cleanup(solver);
            return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: effective_damp, k, diag_mean });
        }

        let _ = (solver.fn_hip_device_sync)();

        let mut host_u: Vec<f64> = vec![0.0; k * k];
        if (solver.fn_hip_memcpy)(host_u.as_mut_ptr() as *mut c_void, d_u, bytes, HIP_MEMCPY_DEVICE_TO_HOST) != HIP_SUCCESS {
            cleanup(solver);
            return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: effective_damp, k, diag_mean });
        }
        cleanup(solver);

        let mut u_mat: Mat<f64> = Mat::zeros(k, k);
        for col in 0..k {
            for row in 0..=col {
                u_mat[(row, col)] = host_u[col * k + row];
            }
        }
        Ok((u_mat, effective_damp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocsolver_link_smoke() {
        if std::env::var("HIPFIRE_SKIP_GPU_TESTS").is_ok() {
            eprintln!("skip: HIPFIRE_SKIP_GPU_TESTS set");
            return;
        }
        match RocSolver::load() {
            Ok(s) => { drop(s); }
            Err(e) => { eprintln!("skip: RocSolver::load failed ({e})"); }
        }
    }
}

#[cfg(test)]
mod parity_tests {
    use super::*;
    use crate::gptq::compute_damped_inv_cholesky_upper;

    fn random_spd(k: usize, seed: u64) -> Mat<f64> {
        let mut state = seed.wrapping_mul(0xdeadbeefcafebabe);
        let mut next_f64 = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = (state >> 11) as f64;
            bits / (1u64 << 53) as f64 - 0.5
        };
        let mut a: Mat<f64> = Mat::zeros(k, k);
        for i in 0..k { for j in 0..k { a[(i, j)] = next_f64(); } }
        let mut h: Mat<f64> = Mat::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0_f64;
                for l in 0..k { s += a[(l, i)] * a[(l, j)]; }
                h[(i, j)] = s;
            }
            h[(i, i)] += 0.01;
        }
        h
    }

    fn max_upper_err(u_cpu: &Mat<f64>, u_hip: &Mat<f64>) -> f64 {
        let k = u_cpu.nrows();
        let mut max_err = 0.0_f64;
        for i in 0..k { for j in i..k {
            let e = (u_cpu[(i, j)] - u_hip[(i, j)]).abs();
            if e > max_err { max_err = e; }
        }}
        max_err
    }

    #[test]
    fn parity_gate_k256() {
        if std::env::var("HIPFIRE_SKIP_GPU_TESTS").is_ok() { return; }
        let solver = match RocSolver::load() { Ok(s) => s, Err(e) => { eprintln!("skip: {e}"); return; } };
        let h = random_spd(256, 42);
        let (u_cpu, _) = compute_damped_inv_cholesky_upper(&h, None, 0.01, 1.0).expect("CPU Cholesky K=256");
        let (u_hip, _) = compute_damped_inv_cholesky_upper_hip(&solver, &h, None, 0.01, 1.0).expect("GPU Cholesky K=256");
        let err = max_upper_err(&u_cpu, &u_hip);
        assert!(err < 1e-10, "max_upper_err {err} >= 1e-10");
    }

    #[test]
    fn parity_gate_k4096() {
        if std::env::var("HIPFIRE_SKIP_GPU_TESTS").is_ok() { return; }
        let solver = match RocSolver::load() { Ok(s) => s, Err(e) => { eprintln!("skip: {e}"); return; } };
        let h = random_spd(4096, 1042);
        let (u_cpu, _) = compute_damped_inv_cholesky_upper(&h, None, 0.01, 1.0).expect("CPU Cholesky K=4096");
        let (u_hip, _) = compute_damped_inv_cholesky_upper_hip(&solver, &h, None, 0.01, 1.0).expect("GPU Cholesky K=4096");
        let err = max_upper_err(&u_cpu, &u_hip);
        assert!(err < 1e-8, "max_upper_err {err} >= 1e-8");
    }
}
