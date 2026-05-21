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
// Phase 2A lands the infra (module load + helpers + the first kernel).
// Helpers wired to the orchestrator in Phase 2D/2E; silence dead-code
// warnings on intermediate phases. Each helper has an explicit doc comment
// describing its eventual call site.
#![allow(dead_code)]

use std::ffi::{c_void, CString};
use std::os::raw::{c_double, c_int, c_uint};
use std::path::PathBuf;

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
const HIP_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
const HIP_SUCCESS: c_int = 0;
const ROCBLAS_STATUS_SUCCESS: c_int = 0;

// Phase 2 block size B=128 (paper default per Frantar et al., no
// experimentation). The OBS column-sequential GPU loop processes K columns
// in blocks of B and emits one BLAS-3 GEMM per block after B serial
// quantize-and-rank-1-within-block steps.
pub const PHASE2_BLOCK_SIZE: usize = 128;

pub type rocblas_handle = *mut c_void;
type hipError_t = c_int;
type rocblas_status = c_int;
type hipModule_t = *mut c_void;
type hipFunction_t = *mut c_void;
type hipStream_t = *mut c_void;

type FnHipMalloc = unsafe extern "C" fn(ptr: *mut *mut c_void, size: usize) -> hipError_t;
type FnHipFree = unsafe extern "C" fn(ptr: *mut c_void) -> hipError_t;
type FnHipMemcpy = unsafe extern "C" fn(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> hipError_t;
type FnHipMemset = unsafe extern "C" fn(dst: *mut c_void, value: c_int, size: usize) -> hipError_t;
type FnHipDeviceSynchronize = unsafe extern "C" fn() -> hipError_t;
type FnHipModuleLoad = unsafe extern "C" fn(module: *mut hipModule_t, fname: *const i8) -> hipError_t;
type FnHipModuleUnload = unsafe extern "C" fn(module: hipModule_t) -> hipError_t;
type FnHipModuleGetFunction = unsafe extern "C" fn(function: *mut hipFunction_t, module: hipModule_t, kname: *const i8) -> hipError_t;
type FnHipModuleLaunchKernel = unsafe extern "C" fn(
    f: hipFunction_t,
    grid_x: c_uint, grid_y: c_uint, grid_z: c_uint,
    block_x: c_uint, block_y: c_uint, block_z: c_uint,
    shared_mem_bytes: c_uint,
    stream: hipStream_t,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> hipError_t;
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
    fn_hip_memset: FnHipMemset,
    fn_hip_device_sync: FnHipDeviceSynchronize,
    fn_hip_module_load: FnHipModuleLoad,
    fn_hip_module_unload: FnHipModuleUnload,
    fn_hip_module_get_function: FnHipModuleGetFunction,
    fn_hip_module_launch_kernel: FnHipModuleLaunchKernel,
    fn_rocblas_destroy_handle: FnRocblasDestroyHandle,
    fn_rocsolver_dpotrf: FnRocsolverDpotrf,
    fn_rocsolver_dtrtri: FnRocsolverDtrtri,
    fn_rocblas_dsyrk: FnRocblasDsyrk,
    fn_rocblas_dgeam: FnRocblasDgeam,
    // Phase 2 OBS kernels (loaded lazily on first dispatch).
    obs_modules: std::sync::Mutex<ObsModules>,
    _libhip: Library,
    _librocblas: Library,
    _librocsolver: Library,
}

#[derive(Default)]
struct ObsModules {
    quantize_mq4_column: Option<KernelHandle>,
    within_block_rank1: Option<KernelHandle>,
    block_apply_mfma: Option<KernelHandle>,
    block_apply_mfma_bf16: Option<KernelHandle>,
}

struct KernelHandle {
    module: hipModule_t,
    function: hipFunction_t,
}

// SAFETY: HIP module + function handles are opaque pointers managed by
// libamdhip64. They are not exposed via thread-local state on the host side;
// hipModuleLaunchKernel is documented thread-safe.
unsafe impl Send for KernelHandle {}
unsafe impl Sync for KernelHandle {}

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
            let fn_hip_memset = *(libhip.get::<FnHipMemset>(b"hipMemset").map_err(|e| format!("hipMemset: {e}"))?);
            let fn_hip_device_sync = *(libhip.get::<FnHipDeviceSynchronize>(b"hipDeviceSynchronize").map_err(|e| format!("hipDeviceSynchronize: {e}"))?);
            let fn_hip_module_load = *(libhip.get::<FnHipModuleLoad>(b"hipModuleLoad").map_err(|e| format!("hipModuleLoad: {e}"))?);
            let fn_hip_module_unload = *(libhip.get::<FnHipModuleUnload>(b"hipModuleUnload").map_err(|e| format!("hipModuleUnload: {e}"))?);
            let fn_hip_module_get_function = *(libhip.get::<FnHipModuleGetFunction>(b"hipModuleGetFunction").map_err(|e| format!("hipModuleGetFunction: {e}"))?);
            let fn_hip_module_launch_kernel = *(libhip.get::<FnHipModuleLaunchKernel>(b"hipModuleLaunchKernel").map_err(|e| format!("hipModuleLaunchKernel: {e}"))?);
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
                handle, fn_hip_malloc, fn_hip_free, fn_hip_memcpy, fn_hip_memset, fn_hip_device_sync,
                fn_hip_module_load, fn_hip_module_unload, fn_hip_module_get_function,
                fn_hip_module_launch_kernel,
                fn_rocblas_destroy_handle, fn_rocsolver_dpotrf, fn_rocsolver_dtrtri,
                fn_rocblas_dsyrk, fn_rocblas_dgeam,
                obs_modules: std::sync::Mutex::new(ObsModules::default()),
                _libhip: libhip, _librocblas: librocblas, _librocsolver: librocsolver,
            })
        }
    }

    fn handle(&self) -> rocblas_handle { self.handle }
}

impl Drop for RocSolver {
    fn drop(&mut self) {
        unsafe {
            // Unload Phase 2 OBS kernel modules if any are resident.
            if let Ok(mut mods) = self.obs_modules.lock() {
                for slot in [
                    mods.quantize_mq4_column.take(),
                    mods.within_block_rank1.take(),
                    mods.block_apply_mfma.take(),
                    mods.block_apply_mfma_bf16.take(),
                ] {
                    if let Some(KernelHandle { module, .. }) = slot {
                        if !module.is_null() {
                            let _ = (self.fn_hip_module_unload)(module);
                        }
                    }
                }
            }
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

/// Phase 2 keep-on-device variant: same algorithm as
/// `compute_damped_inv_cholesky_upper_hip` but transfers ownership of the
/// device U buffer to the caller via `GpuU` instead of D2H-copying to a
/// host `Mat<f64>`.
///
/// Used by `gptq_column_sequential_hip` so the OBS column loop can read U
/// directly from device memory across all K serial steps + K/B GEMM
/// flushes. Saves one round-trip 2× K²×8-byte H2D copy.
///
/// Layout: `GpuU::d_u` points to a K×K FP64 buffer in COLUMN-major (post-
/// dgeam, upper-tri stored at d_u[col*K + row] for row<=col). Lower
/// triangle is undefined (rocSOLVER does not zero it). Kernels must
/// only read upper-tri entries.
pub fn compute_damped_inv_cholesky_upper_hip_keep(
    solver: &RocSolver,
    h: &Mat<f64>,
    perm: Option<&[usize]>,
    initial_damp: f64,
    max_damp_multiplier: f64,
) -> Result<(GpuU, f64), CholeskyError> {
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

        // ── Cleanup helpers ──
        // On any error we free d_a, d_h_inv, d_u, d_info.
        // On success we free d_a, d_h_inv, d_info but transfer d_u to GpuU.
        let cleanup_all = |s: &RocSolver, du: *mut c_void| {
            let _ = (s.fn_hip_free)(d_a);
            let _ = (s.fn_hip_free)(d_h_inv);
            let _ = (s.fn_hip_free)(du);
            let _ = (s.fn_hip_free)(d_info);
        };
        let cleanup_keep_u = |s: &RocSolver| {
            let _ = (s.fn_hip_free)(d_a);
            let _ = (s.fn_hip_free)(d_h_inv);
            let _ = (s.fn_hip_free)(d_info);
        };

        // Adaptive damp retry loop on first dpotrf.
        loop {
            for i in 0..k {
                let src_i = perm.map(|p| p[i]).unwrap_or(i);
                host_a[i * k + i] = h[(src_i, src_i)] + damp;
            }
            if (solver.fn_hip_memcpy)(d_a, host_a.as_ptr() as *const c_void, bytes, HIP_MEMCPY_HOST_TO_DEVICE) != HIP_SUCCESS {
                cleanup_all(solver, d_u);
                eprintln!(
                    "[gptq-hip-diag] H2D memcpy of damped H_eff failed at K={k} damp={damp:.6e}"
                );
                return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: damp, k, diag_mean });
            }
            let status = (solver.fn_rocsolver_dpotrf)(solver.handle(), ROCBLAS_FILL_LOWER, k as c_int, d_a as *mut c_double, k as c_int, d_info as *mut c_int);
            if status != ROCBLAS_STATUS_SUCCESS {
                // RETRY on API-level error: bump damp, try again. K=12288
                // path empirically hits this at first-iter damp=initial*
                // diag_mean. Higher damp may resolve numerical sensitivity.
                eprintln!(
                    "[gptq-hip-diag] first dpotrf API error: status={status} K={k} damp={damp:.6e}; retrying with 10x damp"
                );
                damp *= 10.0;
                if damp > damp_cap {
                    cleanup_all(solver, d_u);
                    eprintln!(
                        "[gptq-hip-diag] first dpotrf exhausted retries: damp_cap={damp_cap:.6e} K={k} diag_mean={diag_mean:.6e}"
                    );
                    return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: damp_cap, k, diag_mean });
                }
                continue;
            }
            let mut info_host: c_int = 0;
            if (solver.fn_hip_memcpy)(&mut info_host as *mut c_int as *mut c_void, d_info, std::mem::size_of::<c_int>(), HIP_MEMCPY_DEVICE_TO_HOST) != HIP_SUCCESS {
                cleanup_all(solver, d_u);
                eprintln!(
                    "[gptq-hip-diag] info-D2H memcpy after first dpotrf failed at K={k} damp={damp:.6e}"
                );
                return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: damp, k, diag_mean });
            }
            if info_host == 0 {
                effective_damp = damp;
                break;
            }
            damp *= 10.0;
            if damp > damp_cap {
                cleanup_all(solver, d_u);
                eprintln!(
                    "[gptq-hip-diag] first dpotrf info_host={info_host} at K={k} damp_cap={damp_cap:.6e} diag_mean={diag_mean:.6e}"
                );
                return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: damp_cap, k, diag_mean });
            }
        }

        let mut info_host: c_int = 0;
        let status = (solver.fn_rocsolver_dtrtri)(solver.handle(), ROCBLAS_FILL_LOWER, ROCBLAS_DIAG_NON_UNIT, k as c_int, d_a as *mut c_double, k as c_int, d_info as *mut c_int);
        if status != ROCBLAS_STATUS_SUCCESS {
            cleanup_all(solver, d_u);
            eprintln!(
                "[gptq-hip-diag] dtrtri API error: status={status} K={k} effective_damp={effective_damp:.6e} diag_mean={diag_mean:.6e}"
            );
            return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: effective_damp, k, diag_mean });
        }
        let _ = (solver.fn_hip_memcpy)(&mut info_host as *mut c_int as *mut c_void, d_info, std::mem::size_of::<c_int>(), HIP_MEMCPY_DEVICE_TO_HOST);
        if info_host != 0 {
            cleanup_all(solver, d_u);
            eprintln!(
                "[gptq-hip-diag] dtrtri info_host={info_host} K={k} effective_damp={effective_damp:.6e}"
            );
            return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: effective_damp, k, diag_mean });
        }

        let alpha: c_double = 1.0;
        let beta: c_double = 0.0;
        let status = (solver.fn_rocblas_dsyrk)(solver.handle(), ROCBLAS_FILL_UPPER, ROCBLAS_OPERATION_TRANSPOSE, k as c_int, k as c_int, &alpha, d_a as *const c_double, k as c_int, &beta, d_h_inv as *mut c_double, k as c_int);
        if status != ROCBLAS_STATUS_SUCCESS {
            cleanup_all(solver, d_u);
            eprintln!(
                "[gptq-hip-diag] dsyrk API error: status={status} K={k} effective_damp={effective_damp:.6e}"
            );
            return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: effective_damp, k, diag_mean });
        }

        let status = (solver.fn_rocsolver_dpotrf)(solver.handle(), ROCBLAS_FILL_UPPER, k as c_int, d_h_inv as *mut c_double, k as c_int, d_info as *mut c_int);
        if status != ROCBLAS_STATUS_SUCCESS {
            cleanup_all(solver, d_u);
            eprintln!(
                "[gptq-hip-diag] second dpotrf (upper, H_inv) API error: status={status} K={k} effective_damp={effective_damp:.6e}"
            );
            return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: effective_damp, k, diag_mean });
        }
        let _ = (solver.fn_hip_memcpy)(&mut info_host as *mut c_int as *mut c_void, d_info, std::mem::size_of::<c_int>(), HIP_MEMCPY_DEVICE_TO_HOST);
        if info_host != 0 {
            cleanup_all(solver, d_u);
            eprintln!(
                "[gptq-hip-diag] second dpotrf info_host={info_host} K={k} effective_damp={effective_damp:.6e}"
            );
            return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: effective_damp, k, diag_mean });
        }

        let status = (solver.fn_rocblas_dgeam)(solver.handle(), ROCBLAS_OPERATION_TRANSPOSE, ROCBLAS_OPERATION_NONE, k as c_int, k as c_int, &alpha, d_h_inv as *const c_double, k as c_int, &beta, std::ptr::null(), k as c_int, d_u as *mut c_double, k as c_int);
        if status != ROCBLAS_STATUS_SUCCESS {
            cleanup_all(solver, d_u);
            eprintln!(
                "[gptq-hip-diag] dgeam (U = H_inv^T) API error: status={status} K={k} effective_damp={effective_damp:.6e}"
            );
            return Err(CholeskyError::SingularEvenWithMaxDamp { max_damp: effective_damp, k, diag_mean });
        }

        let _ = (solver.fn_hip_device_sync)();

        // Caller takes ownership of d_u via GpuU.
        cleanup_keep_u(solver);

        // Build the fn_hip_free pointer as a plain unsafe extern "C" fn
        // value (no captured state) so GpuU's Drop can free without needing
        // a reference to RocSolver.
        let gpu_u = GpuU {
            d_u,
            k,
            effective_damp,
            fn_hip_free: solver.fn_hip_free,
        };
        Ok((gpu_u, effective_damp))
    }
}

/// Owning handle for the device-resident upper-Cholesky factor U.
///
/// Lifetime contract: created by `compute_damped_inv_cholesky_upper_hip_keep`,
/// freed automatically by `Drop`. Callers (Phase 2 OBS orchestrator) hold
/// `GpuU` for the duration of the column-sequential loop; on either
/// success or panic, U is freed deterministically.
pub struct GpuU {
    /// Device pointer to a K×K FP64 buffer in column-major layout (upper-tri
    /// stored at d_u[col*K + row] for row<=col). Lower triangle undefined.
    pub d_u: *mut c_void,
    pub k: usize,
    pub effective_damp: f64,
    /// Bound at construction so Drop doesn't need a live RocSolver borrow.
    fn_hip_free: FnHipFree,
}

unsafe impl Send for GpuU {}

impl Drop for GpuU {
    fn drop(&mut self) {
        if !self.d_u.is_null() {
            unsafe { let _ = (self.fn_hip_free)(self.d_u); }
            self.d_u = std::ptr::null_mut();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2 OBS kernel infrastructure
//
// Sequence:
//   * Each kernel source is embedded with `include_str!()` from `kernels/src/`.
//   * On first dispatch we compile to a `.hsaco` via the system `hipcc`,
//     cache by source-hash under `$HIPFIRE_GPTQ_HIP_KERNEL_CACHE` (default
//     `$TMPDIR/hipfire-gptq-hip-kernels`), then `hipModuleLoad` it.
//   * Subsequent dispatches reuse the module/function handles.
//
// CDNA gate: callers (main.rs) only construct a RocSolver on gfx942-class
// arches per `detect_is_cdna()`. The kernel source files are tagged
// `.gfx942.hip` and use FP64 MFMA intrinsics — they will not compile for
// RDNA. Source-string compile errors propagate as Err strings.
// ─────────────────────────────────────────────────────────────────────────────

/// Per-256-block grid entry in the device H2D layout. Matches the C struct
/// `packed_grid_t` in `kernels/src/gptq_quantize_mq4_column_f64.gfx942.hip`:
/// f32 scale, f32 min_val, 8 bytes of padding (total 16 bytes).
///
/// Rust-side `BlockGrid` stores FP64 scale/min_val for CPU numerical
/// fidelity. The H2D narrows to FP32 — kernel constants are intentionally
/// FP32 to match the on-disk MQ4G256 layout (4-byte scale + 4-byte
/// min_val per block) that production kernels read back. FP64 dequant in
/// the GPU OBS path would be a divergence from the on-disk layout.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PackedGrid {
    scale: f32,
    min_val: f32,
    _pad: [u8; 8],
}

#[inline]
fn pack_grids(grids: &[crate::gptq::BlockGrid]) -> Vec<PackedGrid> {
    grids.iter().map(|g| PackedGrid {
        scale: g.scale as f32,
        min_val: g.min_val as f32,
        _pad: [0u8; 8],
    }).collect()
}

#[inline]
fn kernel_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("HIPFIRE_GPTQ_HIP_KERNEL_CACHE") {
        return PathBuf::from(dir);
    }
    let tmp = std::env::var_os("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    tmp.join("hipfire-gptq-hip-kernels")
}

/// Hash a source string + arch deterministically. Stable across processes
/// (uses std's `DefaultHasher` which is deterministic within a given
/// libstd version — the cache invalidates harmlessly across upgrades).
fn hash_source(name: &str, source: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut h);
    source.hash(&mut h);
    "gfx942".hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Compile a HIP kernel source string to a `.hsaco` on disk, caching by
/// source hash. Returns the absolute path to the compiled blob.
fn compile_kernel(name: &str, source: &str) -> Result<PathBuf, String> {
    let dir = kernel_cache_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("create cache dir: {e}"))?;
    let src_hash = hash_source(name, source);
    let stem = format!("{name}.{src_hash}");
    let hsaco_path = dir.join(format!("{stem}.hsaco"));
    if hsaco_path.exists() {
        return Ok(hsaco_path);
    }

    // Write source to a temp .hip file.
    let src_path = dir.join(format!("{stem}.hip"));
    std::fs::write(&src_path, source).map_err(|e| format!("write src: {e}"))?;

    // Locate HIP include dir for `hip_runtime.h`.
    let hip_path = std::env::var("HIP_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    let include_args: Vec<String> = ["include"].iter()
        .map(|sub| format!("-I{hip_path}/{sub}"))
        .collect();

    let output = std::process::Command::new("hipcc")
        .arg("--genco")
        .arg("--offload-arch=gfx942")
        .arg("-O3")
        .args(&include_args)
        .arg("-o").arg(&hsaco_path)
        .arg(&src_path)
        .output()
        .map_err(|e| format!("hipcc spawn: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "hipcc failed for {name} (status {:?}):\n--- stderr ---\n{stderr}\n--- stdout ---\n{stdout}",
            output.status.code(),
        ));
    }
    if !hsaco_path.exists() {
        return Err(format!("hipcc reported success but {} missing", hsaco_path.display()));
    }
    Ok(hsaco_path)
}

unsafe fn load_kernel(
    solver: &RocSolver,
    name: &str,
    source: &str,
    fn_name: &str,
) -> Result<KernelHandle, String> {
    let hsaco = compile_kernel(name, source)?;
    let cpath = CString::new(hsaco.to_string_lossy().as_bytes())
        .map_err(|e| format!("CString hsaco: {e}"))?;
    let mut module: hipModule_t = std::ptr::null_mut();
    let code = (solver.fn_hip_module_load)(&mut module as *mut hipModule_t, cpath.as_ptr());
    if code != HIP_SUCCESS || module.is_null() {
        return Err(format!("hipModuleLoad {} returned {code}", hsaco.display()));
    }
    let cfname = CString::new(fn_name).map_err(|e| format!("CString fn: {e}"))?;
    let mut function: hipFunction_t = std::ptr::null_mut();
    let code = (solver.fn_hip_module_get_function)(&mut function as *mut hipFunction_t, module, cfname.as_ptr());
    if code != HIP_SUCCESS || function.is_null() {
        let _ = (solver.fn_hip_module_unload)(module);
        return Err(format!("hipModuleGetFunction {} returned {code}", fn_name));
    }
    Ok(KernelHandle { module, function })
}

// Kernel source strings — embedded at compile time.
const KSRC_QUANTIZE_MQ4_COLUMN: &str = include_str!("../../../kernels/src/gptq_quantize_mq4_column_f64.gfx942.hip");
const KSRC_OBS_WITHIN_BLOCK_RANK1: &str = include_str!("../../../kernels/src/gptq_obs_within_block_rank1_f64.gfx942.hip");
const KSRC_OBS_BLOCK_APPLY_MFMA: &str = include_str!("../../../kernels/src/gptq_obs_block_apply_mfma_f64.gfx942.hip");
const KSRC_OBS_BLOCK_APPLY_MFMA_BF16: &str = include_str!("../../../kernels/src/gptq_obs_block_apply_mfma_bf16.gfx942.hip");

/// Launch the Phase 2A quantize-column kernel.
///
/// One thread per M-row. Quantizes `d_w_residual[:, col_orig]` to MQ4
/// using the frozen per-256-block grid, writes the quantized values into
/// `d_w_quant[:, col_orig]`, and emits the per-row error
/// `(w - q) / u_ss` into `d_err_block[:, err_col_slot]`.
///
/// `d_err_block` is M × B row-major (column-stride = B, row-stride = 1).
///
/// # Safety
/// All pointers must be valid device buffers of the documented sizes.
/// `col_orig` must be in `[0, k_dim)`. `err_col_slot` must be in `[0, b_dim)`.
pub unsafe fn quantize_mq4_column_hip(
    solver: &RocSolver,
    d_w_residual: *const c_void,
    d_w_quant: *mut c_void,
    d_grids: *const c_void,
    d_err_block: *mut c_void,
    col_orig: i32,
    u_ss: f64,
    k_dim: i32,
    m_dim: i32,
    err_col_slot: i32,
    b_dim: i32,
) -> Result<(), String> {
    // Lazy-load module.
    {
        let mut guard = solver.obs_modules.lock().map_err(|e| format!("module lock: {e}"))?;
        if guard.quantize_mq4_column.is_none() {
            let h = load_kernel(solver, "gptq_quantize_mq4_column_f64", KSRC_QUANTIZE_MQ4_COLUMN, "gptq_quantize_mq4_column_f64")?;
            guard.quantize_mq4_column = Some(h);
        }
    }

    let grid_x = ((m_dim as u32) + 255) / 256;
    if grid_x == 0 { return Ok(()); }

    // Build kernel params (each entry is a pointer to the value).
    // Layout matches kernel signature exactly.
    let mut p_d_w_res = d_w_residual;
    let mut p_d_w_q   = d_w_quant;
    let mut p_d_grids = d_grids;
    let mut p_d_err   = d_err_block;
    let mut p_col     = col_orig;
    let mut p_u_ss    = u_ss;
    let mut p_kdim    = k_dim;
    let mut p_mdim    = m_dim;
    let mut p_eslot   = err_col_slot;
    let mut p_bdim    = b_dim;

    let mut params: [*mut c_void; 10] = [
        &mut p_d_w_res as *mut _ as *mut c_void,
        &mut p_d_w_q   as *mut _ as *mut c_void,
        &mut p_d_grids as *mut _ as *mut c_void,
        &mut p_d_err   as *mut _ as *mut c_void,
        &mut p_col     as *mut _ as *mut c_void,
        &mut p_u_ss    as *mut _ as *mut c_void,
        &mut p_kdim    as *mut _ as *mut c_void,
        &mut p_mdim    as *mut _ as *mut c_void,
        &mut p_eslot   as *mut _ as *mut c_void,
        &mut p_bdim    as *mut _ as *mut c_void,
    ];

    let guard = solver.obs_modules.lock().map_err(|e| format!("module lock: {e}"))?;
    let function = guard.quantize_mq4_column.as_ref().unwrap().function;
    drop(guard);

    let code = (solver.fn_hip_module_launch_kernel)(
        function,
        grid_x, 1, 1,
        256, 1, 1,
        0,
        std::ptr::null_mut(),
        params.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    if code != HIP_SUCCESS {
        return Err(format!("hipModuleLaunchKernel quantize_mq4_column_f64 returned {code}"));
    }
    Ok(())
}

/// Launch the Phase 2B within-block rank-1 update kernel.
///
/// Applies the OBS rank-1 update for the current step to the remaining
/// columns inside the current block. See kernel doc + plan §3 Kernel 2.
///
/// `d_u` is the K×K upper-Cholesky factor in COLUMN-major device layout
/// (post-dgeam from Phase D). `d_perm` is the K-length WEIGHT-mode
/// actorder permutation as i32 (K=12288 fits comfortably in i32).
///
/// No-op when `step + 1 >= block_end` (last step in the block has no
/// within-block tail).
///
/// # Safety
/// All pointers must be valid device buffers of the documented shapes.
pub unsafe fn gptq_obs_within_block_hip(
    solver: &RocSolver,
    d_w_residual: *mut c_void,
    d_err_block: *const c_void,
    d_u: *const c_void,
    d_perm: *const c_void,
    step: i32,
    block_start: i32,
    block_end: i32,
    k_dim: i32,
    m_dim: i32,
    b_dim: i32,
) -> Result<(), String> {
    {
        let mut guard = solver.obs_modules.lock().map_err(|e| format!("module lock: {e}"))?;
        if guard.within_block_rank1.is_none() {
            let h = load_kernel(
                solver,
                "gptq_obs_within_block_rank1_f64",
                KSRC_OBS_WITHIN_BLOCK_RANK1,
                "gptq_obs_within_block_rank1_f64",
            )?;
            guard.within_block_rank1 = Some(h);
        }
    }

    let remaining = block_end - step - 1;
    if remaining <= 0 { return Ok(()); }
    let grid_x = ((m_dim as u32) + 63) / 64;
    if grid_x == 0 { return Ok(()); }

    let mut p_wres = d_w_residual;
    let mut p_err  = d_err_block;
    let mut p_u    = d_u;
    let mut p_perm = d_perm;
    let mut p_step = step;
    let mut p_bs   = block_start;
    let mut p_be   = block_end;
    let mut p_kdim = k_dim;
    let mut p_mdim = m_dim;
    let mut p_bdim = b_dim;

    let mut params: [*mut c_void; 10] = [
        &mut p_wres as *mut _ as *mut c_void,
        &mut p_err  as *mut _ as *mut c_void,
        &mut p_u    as *mut _ as *mut c_void,
        &mut p_perm as *mut _ as *mut c_void,
        &mut p_step as *mut _ as *mut c_void,
        &mut p_bs   as *mut _ as *mut c_void,
        &mut p_be   as *mut _ as *mut c_void,
        &mut p_kdim as *mut _ as *mut c_void,
        &mut p_mdim as *mut _ as *mut c_void,
        &mut p_bdim as *mut _ as *mut c_void,
    ];

    let guard = solver.obs_modules.lock().map_err(|e| format!("module lock: {e}"))?;
    let function = guard.within_block_rank1.as_ref().unwrap().function;
    drop(guard);

    let code = (solver.fn_hip_module_launch_kernel)(
        function,
        grid_x, remaining as u32, 1,
        64, 1, 1,
        0,
        std::ptr::null_mut(),
        params.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    if code != HIP_SUCCESS {
        return Err(format!("hipModuleLaunchKernel gptq_obs_within_block_rank1_f64 returned {code}"));
    }
    Ok(())
}

/// Launch the Phase 2C block-apply MFMA F64 GEMM kernel.
///
/// Computes the cross-block rank-B propagation:
///     W_residual[:, perm[block_end..K]] -= err_block @ U_submatrix
/// via 16×16 FP64 MFMA tiles. See kernel doc + plan §3 Kernel 3.
///
/// `b_dim` is the actual block size for this block (= block_end -
/// block_start), bounded by B=128. The last block of K may be smaller
/// when K is not a multiple of B; in that case the kernel masks out
/// out-of-range k_inner indices.
///
/// `d_u` is the K×K upper-Cholesky factor in column-major layout
/// (post-dgeam, same as Phase 2B).
///
/// No-op when `block_end >= k_dim` (no remaining columns to propagate to).
///
/// # Safety
/// All pointers must be valid device buffers of the documented shapes.
pub unsafe fn gptq_obs_block_apply_mfma_hip(
    solver: &RocSolver,
    d_w_residual: *mut c_void,
    d_err_block: *const c_void,
    d_u: *const c_void,
    d_perm: *const c_void,
    block_start: i32,
    block_end: i32,
    k_dim: i32,
    m_dim: i32,
    b_dim: i32,
) -> Result<(), String> {
    {
        let mut guard = solver.obs_modules.lock().map_err(|e| format!("module lock: {e}"))?;
        if guard.block_apply_mfma.is_none() {
            let h = load_kernel(
                solver,
                "gptq_obs_block_apply_mfma_f64",
                KSRC_OBS_BLOCK_APPLY_MFMA,
                "gptq_obs_block_apply_mfma_f64",
            )?;
            guard.block_apply_mfma = Some(h);
        }
    }

    let n_remaining = k_dim - block_end;
    if n_remaining <= 0 || m_dim <= 0 { return Ok(()); }

    let grid_x = ((m_dim as u32) + 31) / 32;
    let grid_y = ((n_remaining as u32) + 15) / 16;

    let mut p_w     = d_w_residual;
    let mut p_err   = d_err_block;
    let mut p_u     = d_u;
    let mut p_perm  = d_perm;
    let mut p_bs    = block_start;
    let mut p_be    = block_end;
    let mut p_kdim  = k_dim;
    let mut p_mdim  = m_dim;
    let mut p_bdim  = b_dim;
    let mut p_nrem  = n_remaining;

    let mut params: [*mut c_void; 10] = [
        &mut p_w    as *mut _ as *mut c_void,
        &mut p_err  as *mut _ as *mut c_void,
        &mut p_u    as *mut _ as *mut c_void,
        &mut p_perm as *mut _ as *mut c_void,
        &mut p_bs   as *mut _ as *mut c_void,
        &mut p_be   as *mut _ as *mut c_void,
        &mut p_kdim as *mut _ as *mut c_void,
        &mut p_mdim as *mut _ as *mut c_void,
        &mut p_bdim as *mut _ as *mut c_void,
        &mut p_nrem as *mut _ as *mut c_void,
    ];

    let guard = solver.obs_modules.lock().map_err(|e| format!("module lock: {e}"))?;
    let function = guard.block_apply_mfma.as_ref().unwrap().function;
    drop(guard);

    let code = (solver.fn_hip_module_launch_kernel)(
        function,
        grid_x, grid_y, 1,
        128, 1, 1,
        0,
        std::ptr::null_mut(),
        params.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    if code != HIP_SUCCESS {
        return Err(format!("hipModuleLaunchKernel gptq_obs_block_apply_mfma_f64 returned {code}"));
    }
    Ok(())
}

/// Launch the Phase 2C-BF16 block-apply MFMA GEMM kernel (cast-trick variant).
///
/// Same call surface as [`gptq_obs_block_apply_mfma_hip`] — buffers are F64
/// in host memory and HBM. The kernel casts F64 -> F32 -> BF16 at MFMA-feed
/// time and casts the F32 accumulator back to F64 at the subtract-store.
///
/// Per plan §6 the numerical contract is: codeword divergence rate < 0.1%
/// vs the F64 sibling on real Hessians. The throughput win is the BF16/F64
/// MFMA throughput ratio on gfx942 (~7x).
///
/// Wired in `gptq_column_sequential_hip` via the `HIPFIRE_GPTQ_HIP_OBS_BF16`
/// env var (see that function's doc-comment).
///
/// # Safety
/// All pointers must be valid device buffers of the documented shapes.
pub unsafe fn gptq_obs_block_apply_mfma_bf16_hip(
    solver: &RocSolver,
    d_w_residual: *mut c_void,
    d_err_block: *const c_void,
    d_u: *const c_void,
    d_perm: *const c_void,
    block_start: i32,
    block_end: i32,
    k_dim: i32,
    m_dim: i32,
    b_dim: i32,
) -> Result<(), String> {
    {
        let mut guard = solver.obs_modules.lock().map_err(|e| format!("module lock: {e}"))?;
        if guard.block_apply_mfma_bf16.is_none() {
            let h = load_kernel(
                solver,
                "gptq_obs_block_apply_mfma_bf16",
                KSRC_OBS_BLOCK_APPLY_MFMA_BF16,
                "gptq_obs_block_apply_mfma_bf16",
            )?;
            guard.block_apply_mfma_bf16 = Some(h);
        }
    }

    let n_remaining = k_dim - block_end;
    if n_remaining <= 0 || m_dim <= 0 { return Ok(()); }

    let grid_x = ((m_dim as u32) + 31) / 32;
    let grid_y = ((n_remaining as u32) + 15) / 16;

    let mut p_w     = d_w_residual;
    let mut p_err   = d_err_block;
    let mut p_u     = d_u;
    let mut p_perm  = d_perm;
    let mut p_bs    = block_start;
    let mut p_be    = block_end;
    let mut p_kdim  = k_dim;
    let mut p_mdim  = m_dim;
    let mut p_bdim  = b_dim;
    let mut p_nrem  = n_remaining;

    let mut params: [*mut c_void; 10] = [
        &mut p_w    as *mut _ as *mut c_void,
        &mut p_err  as *mut _ as *mut c_void,
        &mut p_u    as *mut _ as *mut c_void,
        &mut p_perm as *mut _ as *mut c_void,
        &mut p_bs   as *mut _ as *mut c_void,
        &mut p_be   as *mut _ as *mut c_void,
        &mut p_kdim as *mut _ as *mut c_void,
        &mut p_mdim as *mut _ as *mut c_void,
        &mut p_bdim as *mut _ as *mut c_void,
        &mut p_nrem as *mut _ as *mut c_void,
    ];

    let guard = solver.obs_modules.lock().map_err(|e| format!("module lock: {e}"))?;
    let function = guard.block_apply_mfma_bf16.as_ref().unwrap().function;
    drop(guard);

    let code = (solver.fn_hip_module_launch_kernel)(
        function,
        grid_x, grid_y, 1,
        128, 1, 1,
        0,
        std::ptr::null_mut(),
        params.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    if code != HIP_SUCCESS {
        return Err(format!("hipModuleLaunchKernel gptq_obs_block_apply_mfma_bf16 returned {code}"));
    }
    Ok(())
}

/// Helper: allocate `size` bytes on device. Returns the device pointer or
/// a stringified error.
#[inline]
unsafe fn dev_alloc(solver: &RocSolver, size: usize) -> Result<*mut c_void, String> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let code = (solver.fn_hip_malloc)(&mut ptr as *mut *mut c_void, size);
    if code != HIP_SUCCESS || ptr.is_null() {
        return Err(format!("hipMalloc({size}) returned {code}"));
    }
    Ok(ptr)
}

#[inline]
unsafe fn dev_free(solver: &RocSolver, ptr: *mut c_void) {
    if !ptr.is_null() {
        let _ = (solver.fn_hip_free)(ptr);
    }
}

#[inline]
unsafe fn h2d(solver: &RocSolver, dst: *mut c_void, src: *const u8, size: usize) -> Result<(), String> {
    let code = (solver.fn_hip_memcpy)(dst, src as *const c_void, size, HIP_MEMCPY_HOST_TO_DEVICE);
    if code != HIP_SUCCESS { return Err(format!("H2D({size}) returned {code}")); }
    Ok(())
}

#[inline]
unsafe fn d2h(solver: &RocSolver, dst: *mut u8, src: *const c_void, size: usize) -> Result<(), String> {
    let code = (solver.fn_hip_memcpy)(dst as *mut c_void, src, size, HIP_MEMCPY_DEVICE_TO_HOST);
    if code != HIP_SUCCESS { return Err(format!("D2H({size}) returned {code}")); }
    Ok(())
}

#[inline]
unsafe fn dev_memset_zero(solver: &RocSolver, dst: *mut c_void, size: usize) -> Result<(), String> {
    let code = (solver.fn_hip_memset)(dst, 0, size);
    if code != HIP_SUCCESS { return Err(format!("hipMemset({size}) returned {code}")); }
    Ok(())
}

#[inline]
unsafe fn dev_sync(solver: &RocSolver) -> Result<(), String> {
    let code = (solver.fn_hip_device_sync)();
    if code != HIP_SUCCESS { return Err(format!("hipDeviceSynchronize returned {code}")); }
    Ok(())
}

/// Phase 2D — GPU OBS column-sequential orchestrator.
///
/// Replaces the CPU loop in `gptq_column_sequential` (gptq.rs:860-915)
/// with the paper §3.2 block-128 variant executed on gfx942:
///   * WEIGHT-mode actorder on CPU (~50µs)
///   * GPU Cholesky via rocSOLVER (`_keep` variant, U stays on device)
///   * H2D copy of W + frozen grids + perm
///   * Block loop:
///       - For step in [block_start..block_end):
///           * Phase 2A kernel: quantize column → emit err_col into d_err_block
///           * Phase 2B kernel: within-block rank-1 update for remaining cols
///       - Phase 2C kernel: cross-block rank-B MFMA GEMM
///   * D2H copy of W_quant back to `weights_flat`
///   * Auto-free of d_u (via GpuU::Drop) and all per-tensor buffers
///
/// Diagnostic [gptq-clamp] line is emitted as a post-D2H scan to keep
/// pipeline log compatibility with the CPU path's line format.
///
/// Returns the effective damping value (after adaptive escalation in the
/// Cholesky step).
///
/// ── Env-var gating ──
///
///   `HIPFIRE_GPTQ_HIP_OBS=1`       (outer, in gptq.rs)  — enables this
///                                  GPU OBS orchestrator at all (default
///                                  off → CPU path).
///   `HIPFIRE_GPTQ_HIP_OBS_BF16=1`  (inner, this function) — switches the
///                                  Phase 2C rank-B GEMM (Kernel 3) from
///                                  F64 MFMA to the BF16 cast-trick MFMA
///                                  variant. Kernel 1 (quantize_mq4_column)
///                                  and Kernel 2 (within_block_rank1) stay
///                                  F64; only the BLAS-3 cross-block GEMM
///                                  switches precision.
///
///   Matrix of behaviors:
///     outer off, inner off  →  CPU path (default)
///     outer ON,  inner off  →  GPU OBS, F64 MFMA Kernel 3 (Phase 2D shipped)
///     outer ON,  inner ON   →  GPU OBS, BF16 cast-trick Kernel 3 (this commit)
///     outer off, inner ON   →  CPU path (inner is a no-op without outer)
///
///   Per plan §6 the BF16 path is opt-in pending real-tensor codeword
///   divergence validation on droplet hardware. Default-flip discussion
///   deferred per feedback_pr_gating_policy.md.
pub fn gptq_column_sequential_hip(
    solver: &RocSolver,
    weights_flat: &mut [f64],
    h_target: &Mat<f64>,
    m: usize,
    k_dim: usize,
    frozen_grids: &[crate::gptq::BlockGrid],
    initial_damp: f64,
    max_damp_multiplier: f64,
    tensor_name: &str,
) -> Result<f64, CholeskyError> {
    assert_eq!(weights_flat.len(), m * k_dim, "weight shape mismatch");
    assert_eq!(h_target.nrows(), k_dim);
    assert_eq!(h_target.ncols(), k_dim);
    assert_eq!(frozen_grids.len(), (m * k_dim) / 256, "frozen_grids count mismatch");

    // Inner opt-in: BF16 cast-trick variant of the Phase 2C rank-B GEMM.
    // Captured once at the start of the call so the per-block branch is a
    // plain bool check instead of a syscall. See the doc-comment above for
    // the full env-var matrix.
    let use_bf16_block_apply = std::env::var("HIPFIRE_GPTQ_HIP_OBS_BF16").is_ok();

    // Pre-compute the real Hessian-diagonal mean for error diagnostics.
    // Used in every `CholeskyError::SingularEvenWithMaxDamp` constructed
    // below so failure logs print a useful sentinel instead of `0.0` —
    // see docs/investigations/2026-05-20-gptq-hip-obs-9b-failure/README.md.
    let diag_mean_for_err: f64 =
        (0..k_dim).map(|i| h_target[(i, i)]).sum::<f64>() / k_dim as f64;

    // ── Step 1: CPU actorder (same as gptq.rs:803-804) ──
    let h_diag: Vec<f64> = (0..k_dim).map(|i| h_target[(i, i)]).collect();
    let perm = crate::gptq::weight_mode_actorder(&h_diag);

    // ── Step 2: GPU Cholesky (keep U on device) ──
    let (gpu_u, effective_damp) = compute_damped_inv_cholesky_upper_hip_keep(
        solver, h_target, Some(&perm), initial_damp, max_damp_multiplier,
    )?;

    // ── Step 3: Diagonal D2H (ONE bulk D2H of the full K×K F64 buffer) ──
    // U is column-major in d_u: U[i, i] at offset i*K + i (= (K+1)*i).
    // The Phase 2 v1 shortcut was K sync hipMemcpy calls of 8 bytes each —
    // K=4096 × ~5µs ≈ 20ms; K=12288 × ~5µs ≈ 60ms — and any one of those
    // returning non-SUCCESS aborted the tensor with a misleading
    // `diag_mean=0.0` warning (see docs/investigations/
    // 2026-05-20-gptq-hip-obs-9b-failure/README.md). Replace with a single
    // bulk D2H of the entire K×K U buffer (which we have to bring back for
    // the host-side diagonal extract anyway), then index the diagonal
    // host-side. ONE syscall, faster wall-clock, no K-fold per-call
    // failure amplification.
    let b_dim = PHASE2_BLOCK_SIZE;
    let bytes_per_f64 = std::mem::size_of::<f64>();
    let u_diag: Vec<f64> = unsafe {
        let u_bytes = k_dim * k_dim * bytes_per_f64;
        let mut host_u: Vec<f64> = vec![0.0; k_dim * k_dim];
        let code = (solver.fn_hip_memcpy)(
            host_u.as_mut_ptr() as *mut c_void,
            gpu_u.d_u as *const c_void,
            u_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
        );
        if code != HIP_SUCCESS {
            eprintln!(
                "[gptq-hip-diag] step3 bulk D2H of d_u failed K={k_dim} u_bytes={u_bytes} effective_damp={:.6e} code={}",
                effective_damp, code
            );
            return Err(CholeskyError::SingularEvenWithMaxDamp {
                max_damp: effective_damp, k: k_dim, diag_mean: diag_mean_for_err,
            });
        }
        let mut diag = vec![0.0_f64; k_dim];
        for i in 0..k_dim {
            diag[i] = host_u[i * k_dim + i];
        }
        diag
    };

    // ── Step 4: H2D — W (twice, into residual + quant), grids, perm ──
    let w_bytes    = m * k_dim * bytes_per_f64;
    let err_bytes  = m * b_dim * bytes_per_f64;
    let grid_bytes = frozen_grids.len() * std::mem::size_of::<PackedGrid>();
    let perm_bytes = k_dim * std::mem::size_of::<i32>();

    let to_chol_err = |damp: f64| CholeskyError::SingularEvenWithMaxDamp {
        max_damp: damp, k: k_dim, diag_mean: diag_mean_for_err,
    };

    unsafe {
        let d_w_residual = match dev_alloc(solver, w_bytes) {
            Ok(p) => p, Err(_) => return Err(to_chol_err(effective_damp)),
        };
        let d_w_quant = match dev_alloc(solver, w_bytes) {
            Ok(p) => p,
            Err(_) => { dev_free(solver, d_w_residual); { eprintln!("[gptq-hip-diag] S1 K={k_dim} effective_damp={:.6e}", effective_damp); return Err(to_chol_err(effective_damp)); } }
        };
        let d_grids = match dev_alloc(solver, grid_bytes) {
            Ok(p) => p,
            Err(_) => {
                dev_free(solver, d_w_residual); dev_free(solver, d_w_quant);
                { eprintln!("[gptq-hip-diag] S2 K={k_dim} effective_damp={:.6e}", effective_damp); return Err(to_chol_err(effective_damp)); }
            }
        };
        let d_err_block = match dev_alloc(solver, err_bytes) {
            Ok(p) => p,
            Err(_) => {
                dev_free(solver, d_w_residual); dev_free(solver, d_w_quant); dev_free(solver, d_grids);
                { eprintln!("[gptq-hip-diag] S3 K={k_dim} effective_damp={:.6e}", effective_damp); return Err(to_chol_err(effective_damp)); }
            }
        };
        let d_perm = match dev_alloc(solver, perm_bytes) {
            Ok(p) => p,
            Err(_) => {
                dev_free(solver, d_w_residual); dev_free(solver, d_w_quant);
                dev_free(solver, d_grids); dev_free(solver, d_err_block);
                { eprintln!("[gptq-hip-diag] S4 K={k_dim} effective_damp={:.6e}", effective_damp); return Err(to_chol_err(effective_damp)); }
            }
        };

        // Wrap cleanup in a helper for the error paths below.
        let cleanup = |s: &RocSolver| {
            dev_free(s, d_w_residual);
            dev_free(s, d_w_quant);
            dev_free(s, d_grids);
            dev_free(s, d_err_block);
            dev_free(s, d_perm);
        };

        if h2d(solver, d_w_residual, weights_flat.as_ptr() as *const u8, w_bytes).is_err() {
            cleanup(solver); { eprintln!("[gptq-hip-diag] S5 K={k_dim} effective_damp={:.6e}", effective_damp); return Err(to_chol_err(effective_damp)); }
        }
        if h2d(solver, d_w_quant, weights_flat.as_ptr() as *const u8, w_bytes).is_err() {
            cleanup(solver); { eprintln!("[gptq-hip-diag] S6 K={k_dim} effective_damp={:.6e}", effective_damp); return Err(to_chol_err(effective_damp)); }
        }
        let packed = pack_grids(frozen_grids);
        if h2d(solver, d_grids, packed.as_ptr() as *const u8, grid_bytes).is_err() {
            cleanup(solver); { eprintln!("[gptq-hip-diag] S7 K={k_dim} effective_damp={:.6e}", effective_damp); return Err(to_chol_err(effective_damp)); }
        }
        let perm_i32: Vec<i32> = perm.iter().map(|&v| v as i32).collect();
        if h2d(solver, d_perm, perm_i32.as_ptr() as *const u8, perm_bytes).is_err() {
            cleanup(solver); { eprintln!("[gptq-hip-diag] S8 K={k_dim} effective_damp={:.6e}", effective_damp); return Err(to_chol_err(effective_damp)); }
        }

        // ── Step 5: Block loop ──
        let mut block_start = 0usize;
        while block_start < k_dim {
            let block_end = (block_start + b_dim).min(k_dim);
            let cur_b = block_end - block_start;

            // Zero d_err_block at the start of each block.
            if dev_memset_zero(solver, d_err_block, err_bytes).is_err() {
                cleanup(solver); { eprintln!("[gptq-hip-diag] S9 K={k_dim} effective_damp={:.6e}", effective_damp); return Err(to_chol_err(effective_damp)); }
            }

            // Phase A: B serial within-block steps.
            for step in block_start..block_end {
                let u_ss = u_diag[step];
                if u_ss <= 0.0 {
                    // Defensive skip — mirrors the CPU path's gptq.rs:863-868
                    // "U's diagonal entries are reciprocals of L's diagonal"
                    // guard.
                    continue;
                }
                let j_orig = perm[step] as i32;
                let err_col_slot = (step - block_start) as i32;

                // Phase 2A kernel: quantize column + emit err.
                if let Err(_) = quantize_mq4_column_hip(
                    solver,
                    d_w_residual, d_w_quant, d_grids, d_err_block,
                    j_orig, u_ss, k_dim as i32, m as i32, err_col_slot, b_dim as i32,
                ) {
                    cleanup(solver); { eprintln!("[gptq-hip-diag] S10 K={k_dim} effective_damp={:.6e}", effective_damp); return Err(to_chol_err(effective_damp)); }
                }

                // Phase 2B kernel: within-block rank-1 update.
                if step + 1 < block_end {
                    if let Err(_) = gptq_obs_within_block_hip(
                        solver,
                        d_w_residual, d_err_block, gpu_u.d_u, d_perm,
                        step as i32, block_start as i32, block_end as i32,
                        k_dim as i32, m as i32, b_dim as i32,
                    ) {
                        cleanup(solver); { eprintln!("[gptq-hip-diag] S11 K={k_dim} effective_damp={:.6e}", effective_damp); return Err(to_chol_err(effective_damp)); }
                    }
                }
            }

            // Phase B: cross-block rank-B MFMA GEMM (Phase 2C kernel).
            // F64 path is the default; HIPFIRE_GPTQ_HIP_OBS_BF16=1 swaps in
            // the BF16 cast-trick variant for the ~7x MFMA throughput gain.
            if block_end < k_dim {
                let res = if use_bf16_block_apply {
                    gptq_obs_block_apply_mfma_bf16_hip(
                        solver,
                        d_w_residual, d_err_block, gpu_u.d_u, d_perm,
                        block_start as i32, block_end as i32,
                        k_dim as i32, m as i32, cur_b as i32,
                    )
                } else {
                    gptq_obs_block_apply_mfma_hip(
                        solver,
                        d_w_residual, d_err_block, gpu_u.d_u, d_perm,
                        block_start as i32, block_end as i32,
                        k_dim as i32, m as i32, cur_b as i32,
                    )
                };
                if let Err(__e) = res {
                    eprintln!("[gptq-hip-diag] S12 block_apply_mfma kernel-err: {__e} K={k_dim} m={m} bs={block_start} be={block_end} b_dim={b_dim} effective_damp={:.6e}", effective_damp);
                    cleanup(solver); return Err(to_chol_err(effective_damp));
                }
            }

            block_start = block_end;
        }

        // Sync before D2H readback.
        if dev_sync(solver).is_err() {
            cleanup(solver); { eprintln!("[gptq-hip-diag] S13 K={k_dim} effective_damp={:.6e}", effective_damp); return Err(to_chol_err(effective_damp)); }
        }

        // ── Step 6: D2H W_quant ──
        if d2h(solver, weights_flat.as_mut_ptr() as *mut u8, d_w_quant, w_bytes).is_err() {
            cleanup(solver); { eprintln!("[gptq-hip-diag] S14 K={k_dim} effective_damp={:.6e}", effective_damp); return Err(to_chol_err(effective_damp)); }
        }
        cleanup(solver);
    }

    // gpu_u dropped here → d_u freed automatically.

    // ── Step 7: Clamp diagnostic (post-D2H scan) ──
    // The CPU path at gptq.rs:917-928 counts PRE-clamp indices (i.e., it
    // can distinguish "value at min bin because of clamp-below" from "value
    // at min bin because the input was within range"). The post-quant scan
    // here cannot reproduce that distinction without re-running the OBS
    // residual on the CPU. Emit a sentinel line so pipeline log readers see
    // a consistent format; precise per-tensor clamp counts on the GPU path
    // are deferred to Phase 3 (would require adding two atomic counters
    // into the quantize_mq4_column_f64 kernel — straightforward but not
    // wired in this commit).
    let total = m * k_dim;
    eprintln!(
        "[gptq-clamp] {tensor_name} M={m} K={k_dim} elements={total} \
         clamps=N/A (gpu-path; per-element clamp counters deferred to Phase 3)",
    );

    Ok(effective_damp)
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

    /// Process-wide lock for tests that mutate `HIPFIRE_GPTQ_HIP_OBS*`
    /// env vars. Rust's test harness runs in threads by default, and
    /// `std::env::set_var` is process-global. Serializing across the two
    /// env-var-dependent tests (F64 + BF16 OBS) prevents flakes when
    /// `cargo test` runs without `--test-threads=1`.
    static OBS_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Recover the MQ4 codeword (0..15) from a dequantized weight value
    /// using the FP32-cast scale + min_val (mirrors the kernel's path).
    /// `q * scale + min_val` round-trips back to the integer codeword
    /// exactly when scale > 0; clamp + round are applied for safety.
    #[inline]
    fn recover_mq4_codeword(w_quant: f64, scale_f32: f32, min_val_f32: f32) -> u8 {
        if scale_f32 == 0.0 { return 0; }
        let q = ((w_quant as f32 - min_val_f32) / scale_f32 + 0.5_f32).floor();
        q.clamp(0.0, 15.0) as u8
    }

    /// Phase 2C-BF16 — codeword-divergence parity gate.
    ///
    /// Per plan §6, the BF16 cast-trick variant of the Phase 2C rank-B
    /// GEMM is allowed to diverge from the F64 sibling on up to 0.1% of
    /// the output MQ4 codewords. Most codewords land far from a bin
    /// boundary so the BF16 mantissa truncation is absorbed by the
    /// round-to-nearest at `(w - min_val) / scale + 0.5`; only weights
    /// pre-quantize that sit within ~1 BF16-ULP of a bin boundary can
    /// flip.
    ///
    /// Test setup:
    ///   M=128, K=512  (4 blocks of B=128, 3 cross-block GEMM dispatches)
    ///   Well-conditioned SPD H from `random_spd`
    ///   Deterministic synthetic W with mixed signs + magnitudes
    ///
    /// Runs `gptq_column_sequential_hip` twice — once with the BF16
    /// inner-opt-in OFF (F64 reference), once with it ON — and counts
    /// the fraction of MQ4 codewords that differ.
    ///
    /// NOTE: Synthetic SPD H is uniform-ish and not representative of
    /// real LLM Hessians (which are spikier). The 0.1% gate is plan §6's
    /// ship-decision threshold for REAL Hessians; the synthetic case is
    /// expected to come in well under that. Real-tensor data point is
    /// produced by a separate user-side workstream on droplet hardware
    /// (out of scope for this commit).
    #[test]
    fn parity_gate_obs_bf16_codeword_divergence() {
        if std::env::var("HIPFIRE_SKIP_GPU_TESTS").is_ok() {
            eprintln!("skip: HIPFIRE_SKIP_GPU_TESTS set");
            return;
        }
        let solver = match RocSolver::load() {
            Ok(s) => s,
            Err(e) => { eprintln!("skip: RocSolver::load failed ({e})"); return; }
        };

        let m: usize = 128;
        let k: usize = 512;

        // SPD H (well-conditioned).
        let h = random_spd(k, 12025);

        // Synthetic W with mixed signs and a useful dynamic range so multiple
        // grid bins are exercised within each 256-block.
        let mut w_orig = vec![0.0_f64; m * k];
        for r in 0..m {
            for c in 0..k {
                let v = ((r as f64) * 0.013 - (c as f64) * 0.027).sin() * 1.5
                      + (((r as f64) + (c as f64)) * 0.0021).cos() * 0.5;
                w_orig[r * k + c] = v;
            }
        }

        // Frozen MQ4G256 grids (same pre-loop computation the production
        // pipeline does).
        let grids = crate::gptq::compute_frozen_block_grids(&w_orig);

        // ── Run F64 OBS path (inner env var OFF) ──
        let mut w_f64 = w_orig.clone();
        let damp_f64 = {
            let _guard = OBS_ENV_LOCK.lock().expect("env lock poisoned");
            std::env::remove_var("HIPFIRE_GPTQ_HIP_OBS_BF16");
            gptq_column_sequential_hip(
                &solver, &mut w_f64, &h, m, k, &grids, 0.01, 1.0,
                "test:bf16-codeword-divergence:f64",
            ).expect("GPU OBS F64 path")
        };

        // ── Run BF16 OBS path (inner env var ON) ──
        let mut w_bf16 = w_orig.clone();
        let damp_bf16 = {
            let _guard = OBS_ENV_LOCK.lock().expect("env lock poisoned");
            std::env::set_var("HIPFIRE_GPTQ_HIP_OBS_BF16", "1");
            let r = gptq_column_sequential_hip(
                &solver, &mut w_bf16, &h, m, k, &grids, 0.01, 1.0,
                "test:bf16-codeword-divergence:bf16",
            );
            std::env::remove_var("HIPFIRE_GPTQ_HIP_OBS_BF16");
            r.expect("GPU OBS BF16 path")
        };

        // Damp value sanity (the BF16 swap only affects Kernel 3, not the
        // Cholesky step that sets damp).
        let damp_rel = (damp_f64 - damp_bf16).abs() / damp_f64.max(1e-30);
        assert!(damp_rel < 1e-12, "damp mismatch: f64={damp_f64} bf16={damp_bf16}");

        // ── Codeword-divergence count ──
        let mut diff_count: usize = 0;
        let mut max_abs_w: f64 = 0.0;
        let total = m * k;
        for r in 0..m {
            for c in 0..k {
                let idx = r * k + c;
                let blk = idx / 256;
                let g = grids[blk];
                // The kernel narrows scale + min_val to FP32 to match the
                // on-disk MQ4G256 layout; mirror that cast for codeword
                // recovery from the F64 W_quant outputs.
                let scale_f32 = g.scale as f32;
                let min_val_f32 = g.min_val as f32;
                let cw_f64 = recover_mq4_codeword(w_f64[idx], scale_f32, min_val_f32);
                let cw_bf16 = recover_mq4_codeword(w_bf16[idx], scale_f32, min_val_f32);
                if cw_f64 != cw_bf16 { diff_count += 1; }
                let aw = w_f64[idx].abs();
                if aw > max_abs_w { max_abs_w = aw; }
            }
        }
        let divergence_rate = (diff_count as f64) / (total as f64);
        eprintln!(
            "[parity_gate_obs_bf16] M={m} K={k} total={total} diff={diff_count} \
             divergence_rate={:.6e} max_abs_w={:.3e}",
            divergence_rate, max_abs_w,
        );

        // Plan §6 contract: < 0.1% codeword divergence rate.
        assert!(
            divergence_rate < 0.001,
            "BF16 codeword divergence {divergence_rate:.6e} >= 1e-3 (diff={diff_count}/{total})",
        );
    }
}

#[cfg(test)]
mod phase2_tests {
    //! Phase 2 OBS kernel unit tests. All tests skip cleanly when
    //! `HIPFIRE_SKIP_GPU_TESTS=1` or when `RocSolver::load()` fails
    //! (no ROCm or no GPU available).

    use super::*;
    use crate::gptq::{quantize_mq4_element_with_clamp, BlockGrid};

    /// Synthetic FP64 weights M×K with simple deterministic values, K
    /// divisible by 256 so the per-256-block grid math is clean.
    fn synth_weights(m: usize, k: usize) -> Vec<f64> {
        let mut w = vec![0.0_f64; m * k];
        for r in 0..m {
            for c in 0..k {
                // Mix of small floats with both signs and a wide range so
                // multiple grid bins are exercised within each 256-block.
                let v = ((r as f64) * 0.013 - (c as f64) * 0.027).sin() * 1.5
                      + (((r as f64) + (c as f64)) * 0.0021).cos() * 0.5;
                w[r * k + c] = v;
            }
        }
        w
    }

    /// CPU reference for the Phase 2A kernel: quantize one column +
    /// compute (w - q) / u_ss per row. Mirrors gptq.rs:872-890 restricted
    /// to a single column.
    fn cpu_quantize_column_reference(
        weights_residual: &[f64],
        weights_flat: &mut [f64],
        grids: &[BlockGrid],
        col_orig: usize,
        u_ss: f64,
        m: usize,
        k: usize,
    ) -> Vec<f64> {
        let mut err = vec![0.0_f64; m];
        for row in 0..m {
            let blk = (row * k + col_orig) / 256;
            let g = grids[blk];
            let w = weights_residual[row * k + col_orig];
            let (q, _) = quantize_mq4_element_with_clamp(w, g.scale, g.min_val);
            weights_flat[row * k + col_orig] = q;
            err[row] = if u_ss == 0.0 { 0.0 } else { (w - q) / u_ss };
        }
        err
    }

    #[test]
    fn phase2a_quantize_column_parity() {
        if std::env::var("HIPFIRE_SKIP_GPU_TESTS").is_ok() {
            eprintln!("skip: HIPFIRE_SKIP_GPU_TESTS set");
            return;
        }
        let solver = match RocSolver::load() {
            Ok(s) => s,
            Err(e) => { eprintln!("skip: RocSolver::load failed ({e})"); return; }
        };

        let m = 64;
        let k = 256;
        let b = 128;
        let u_ss = 0.37_f64;
        let col_orig: usize = 17;
        let err_slot: usize = 5;

        // Build synthetic W + grids (frozen pre-loop, per CPU pipeline).
        let weights = synth_weights(m, k);
        let mut grids: Vec<BlockGrid> = Vec::with_capacity((m * k) / 256);
        for b_i in 0..(m * k) / 256 {
            let block = &weights[b_i * 256..(b_i + 1) * 256];
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            for &v in block { if v < mn { mn = v; } if v > mx { mx = v; } }
            let range = mx - mn;
            let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
            grids.push(BlockGrid { scale, min_val: mn });
        }

        // CPU reference.
        let mut cpu_w_quant = weights.clone();
        let cpu_err = cpu_quantize_column_reference(
            &weights, &mut cpu_w_quant, &grids, col_orig, u_ss, m, k,
        );

        // GPU path.
        let total_w_bytes = m * k * std::mem::size_of::<f64>();
        let grid_bytes = grids.len() * std::mem::size_of::<PackedGrid>();
        let err_bytes = m * b * std::mem::size_of::<f64>();
        unsafe {
            let d_w_res = dev_alloc(&solver, total_w_bytes).expect("alloc w_res");
            let d_w_q = dev_alloc(&solver, total_w_bytes).expect("alloc w_q");
            let d_grids = dev_alloc(&solver, grid_bytes).expect("alloc grids");
            let d_err = dev_alloc(&solver, err_bytes).expect("alloc err");
            h2d(&solver, d_w_res, weights.as_ptr() as *const u8, total_w_bytes).expect("h2d w_res");
            h2d(&solver, d_w_q, weights.as_ptr() as *const u8, total_w_bytes).expect("h2d w_q init");
            let packed = pack_grids(&grids);
            h2d(&solver, d_grids, packed.as_ptr() as *const u8, grid_bytes).expect("h2d grids");
            dev_memset_zero(&solver, d_err, err_bytes).expect("memset err");

            quantize_mq4_column_hip(
                &solver,
                d_w_res, d_w_q, d_grids, d_err,
                col_orig as i32, u_ss, k as i32, m as i32, err_slot as i32, b as i32,
            ).expect("launch quantize_mq4_column");
            dev_sync(&solver).expect("sync");

            // D2H W_quant and err.
            let mut gpu_w_quant = vec![0.0_f64; m * k];
            let mut gpu_err = vec![0.0_f64; m * b];
            d2h(&solver, gpu_w_quant.as_mut_ptr() as *mut u8, d_w_q, total_w_bytes).expect("d2h w_q");
            d2h(&solver, gpu_err.as_mut_ptr() as *mut u8, d_err, err_bytes).expect("d2h err");

            dev_free(&solver, d_w_res);
            dev_free(&solver, d_w_q);
            dev_free(&solver, d_grids);
            dev_free(&solver, d_err);

            // Compare. Each row's W_quant byte exactness depends on FP32 vs
            // FP64 of scale/min_val — the kernel uses FP32 (matching on-disk
            // layout). The CPU reference uses the same FP32-cast for parity.
            let mut max_q_err = 0.0_f64;
            let mut max_e_err = 0.0_f64;
            for row in 0..m {
                let q_cpu = cpu_w_quant[row * k + col_orig];
                let q_gpu = gpu_w_quant[row * k + col_orig];
                let e_cpu = cpu_err[row];
                let e_gpu = gpu_err[row * b + err_slot];
                let dq = (q_cpu - q_gpu).abs();
                let de = (e_cpu - e_gpu).abs();
                if dq > max_q_err { max_q_err = dq; }
                if de > max_e_err { max_e_err = de; }
            }
            // FP32 scale/min_val truncation introduces ~1e-7 absolute error
            // relative to FP64 CPU reference. The contract here is
            // "byte-identical W on the kernel's own grid"; we allow a small
            // numerical tolerance for the FP32 cast in the kernel.
            // Phase 2F is the end-to-end byte-compare; this is unit-level
            // sanity.
            assert!(max_q_err < 1e-6, "max_q_err {max_q_err} >= 1e-6");
            assert!(max_e_err < 1e-6, "max_e_err {max_e_err} >= 1e-6");
        }
    }

    /// Pack U (faer Mat<f64>) into the COLUMN-major device layout produced
    /// by Phase D's dgeam. flat[col*K + row] = U[row, col]. Off-diagonal
    /// (row > col) entries are zero (upper-tri).
    fn pack_u_column_major(u: &Mat<f64>) -> Vec<f64> {
        let k = u.nrows();
        let mut out = vec![0.0_f64; k * k];
        for col in 0..k {
            for row in 0..=col {
                out[col * k + row] = u[(row, col)];
            }
        }
        out
    }

    /// CPU reference for the Phase 2B kernel: apply within-block rank-1
    /// updates for ONE step to all next_step in (step+1)..block_end.
    fn cpu_within_block_rank1_reference(
        w_res: &mut [f64],
        err_block: &[f64],
        u: &Mat<f64>,
        perm: &[usize],
        step: usize,
        block_start: usize,
        block_end: usize,
        m: usize,
        k: usize,
        b: usize,
    ) {
        for row in 0..m {
            let err = err_block[row * b + (step - block_start)];
            if err == 0.0 { continue; }
            for next_step in (step + 1)..block_end {
                let u_sn = u[(step, next_step)];
                if u_sn == 0.0 { continue; }
                let kk_orig = perm[next_step];
                w_res[row * k + kk_orig] -= err * u_sn;
            }
        }
    }

    #[test]
    fn phase2b_within_block_rank1_parity() {
        if std::env::var("HIPFIRE_SKIP_GPU_TESTS").is_ok() {
            eprintln!("skip: HIPFIRE_SKIP_GPU_TESTS set");
            return;
        }
        let solver = match RocSolver::load() {
            Ok(s) => s,
            Err(e) => { eprintln!("skip: RocSolver::load failed ({e})"); return; }
        };

        // Use a small block size (B=8) for testability; the kernel does
        // not care about the value of B beyond pointer arithmetic.
        let m = 16;
        let k = 256;
        let b = 8;
        let block_start = 0;
        let block_end = b;
        let step = 2;

        // Build deterministic synthetic state.
        let mut w_res = vec![0.0_f64; m * k];
        for r in 0..m { for c in 0..k {
            w_res[r * k + c] = ((r * 7 + c * 11) as f64).sin() * 0.3;
        }}

        // Random-ish but deterministic err_block (only column `step - block_start` matters).
        let mut err_block = vec![0.0_f64; m * b];
        for r in 0..m {
            err_block[r * b + (step - block_start)] = ((r as f64) * 0.13 + 0.5).cos() * 0.05;
        }

        // Identity permutation for simplicity.
        let perm: Vec<usize> = (0..k).collect();

        // Synthetic upper-tri U with non-zero off-diagonals for next_step in (step+1)..block_end.
        let mut u_mat: Mat<f64> = Mat::zeros(k, k);
        for row in 0..k { for col in row..k {
            u_mat[(row, col)] = ((row * 3 + col) as f64).cos() * 0.07 + 0.001;
        }}

        // CPU reference.
        let mut cpu_w_res = w_res.clone();
        cpu_within_block_rank1_reference(
            &mut cpu_w_res, &err_block, &u_mat, &perm,
            step, block_start, block_end, m, k, b,
        );

        // GPU path.
        let w_bytes   = m * k * std::mem::size_of::<f64>();
        let err_bytes = m * b * std::mem::size_of::<f64>();
        let u_bytes   = k * k * std::mem::size_of::<f64>();
        let perm_bytes = k * std::mem::size_of::<i32>();

        unsafe {
            let d_w = dev_alloc(&solver, w_bytes).expect("alloc w");
            let d_err = dev_alloc(&solver, err_bytes).expect("alloc err");
            let d_u = dev_alloc(&solver, u_bytes).expect("alloc u");
            let d_perm = dev_alloc(&solver, perm_bytes).expect("alloc perm");

            h2d(&solver, d_w, w_res.as_ptr() as *const u8, w_bytes).expect("h2d w");
            h2d(&solver, d_err, err_block.as_ptr() as *const u8, err_bytes).expect("h2d err");
            let u_col_major = pack_u_column_major(&u_mat);
            h2d(&solver, d_u, u_col_major.as_ptr() as *const u8, u_bytes).expect("h2d u");
            let perm_i32: Vec<i32> = perm.iter().map(|&v| v as i32).collect();
            h2d(&solver, d_perm, perm_i32.as_ptr() as *const u8, perm_bytes).expect("h2d perm");

            gptq_obs_within_block_hip(
                &solver,
                d_w, d_err, d_u, d_perm,
                step as i32, block_start as i32, block_end as i32,
                k as i32, m as i32, b as i32,
            ).expect("launch within_block_rank1");
            dev_sync(&solver).expect("sync");

            let mut gpu_w_res = vec![0.0_f64; m * k];
            d2h(&solver, gpu_w_res.as_mut_ptr() as *mut u8, d_w, w_bytes).expect("d2h w");

            dev_free(&solver, d_w);
            dev_free(&solver, d_err);
            dev_free(&solver, d_u);
            dev_free(&solver, d_perm);

            let mut max_err = 0.0_f64;
            for i in 0..(m * k) {
                let e = (cpu_w_res[i] - gpu_w_res[i]).abs();
                if e > max_err { max_err = e; }
            }
            assert!(max_err < 1e-12, "max_err {max_err} >= 1e-12");
        }
    }

    /// Edge case: step == block_end - 1 (last in block) — the kernel
    /// should be a no-op since there are no remaining columns in-block.
    #[test]
    fn phase2b_last_in_block_is_noop() {
        if std::env::var("HIPFIRE_SKIP_GPU_TESTS").is_ok() {
            eprintln!("skip: HIPFIRE_SKIP_GPU_TESTS set");
            return;
        }
        let solver = match RocSolver::load() {
            Ok(s) => s,
            Err(e) => { eprintln!("skip: RocSolver::load failed ({e})"); return; }
        };

        let m = 8;
        let k = 256;
        let b = 8;
        let block_start = 0;
        let block_end = b;
        let step = block_end - 1; // last step in block

        let w_res = vec![1.5_f64; m * k];
        let err_block = vec![0.1_f64; m * b];
        let perm_i32: Vec<i32> = (0..k as i32).collect();
        let u_col_major = vec![0.5_f64; k * k];

        let w_bytes = m * k * std::mem::size_of::<f64>();
        let err_bytes = m * b * std::mem::size_of::<f64>();
        let u_bytes = k * k * std::mem::size_of::<f64>();
        let perm_bytes = k * std::mem::size_of::<i32>();

        unsafe {
            let d_w = dev_alloc(&solver, w_bytes).expect("alloc w");
            let d_err = dev_alloc(&solver, err_bytes).expect("alloc err");
            let d_u = dev_alloc(&solver, u_bytes).expect("alloc u");
            let d_perm = dev_alloc(&solver, perm_bytes).expect("alloc perm");

            h2d(&solver, d_w, w_res.as_ptr() as *const u8, w_bytes).expect("h2d w");
            h2d(&solver, d_err, err_block.as_ptr() as *const u8, err_bytes).expect("h2d err");
            h2d(&solver, d_u, u_col_major.as_ptr() as *const u8, u_bytes).expect("h2d u");
            h2d(&solver, d_perm, perm_i32.as_ptr() as *const u8, perm_bytes).expect("h2d perm");

            gptq_obs_within_block_hip(
                &solver,
                d_w, d_err, d_u, d_perm,
                step as i32, block_start as i32, block_end as i32,
                k as i32, m as i32, b as i32,
            ).expect("launch within_block_rank1");
            dev_sync(&solver).expect("sync");

            let mut gpu_w_res = vec![0.0_f64; m * k];
            d2h(&solver, gpu_w_res.as_mut_ptr() as *mut u8, d_w, w_bytes).expect("d2h w");

            dev_free(&solver, d_w);
            dev_free(&solver, d_err);
            dev_free(&solver, d_u);
            dev_free(&solver, d_perm);

            for i in 0..(m * k) {
                assert_eq!(gpu_w_res[i], 1.5, "noop violation at {i}: got {}", gpu_w_res[i]);
            }
        }
    }

    /// CPU reference for the Phase 2C kernel: apply the rank-B GEMM
    /// W[:, perm[block_end..K]] -= err_block @ U_submatrix.
    fn cpu_block_apply_reference(
        w_res: &mut [f64],
        err_block: &[f64],
        u: &Mat<f64>,
        perm: &[usize],
        block_start: usize,
        block_end: usize,
        m: usize,
        k: usize,
        b_dim: usize,
    ) {
        for row in 0..m {
            for j in 0..(k - block_end) {
                let mut sum = 0.0_f64;
                for i in 0..b_dim {
                    let kk = block_start + i;
                    let ll = block_end + j;
                    if kk > ll { continue; }              // upper-tri: U[kk, ll] valid only when kk <= ll
                    sum += err_block[row * b_dim + i] * u[(kk, ll)];
                }
                let col_orig = perm[block_end + j];
                w_res[row * k + col_orig] -= sum;
            }
        }
    }

    /// Phase 2F — end-to-end CPU vs GPU-OBS parity gate.
    ///
    /// Synthetic small case: M=64, K=256 (so K/B=2 blocks). Builds a
    /// well-conditioned SPD Hessian, frozen MQ4 grids, and runs both
    /// gptq_column_sequential (CPU reference) and
    /// gptq_column_sequential_hip (GPU full-OBS path). Asserts:
    ///   max_i |W_gpu[i] - W_cpu[i]| / max_i |W_cpu[i]| < 1e-6
    ///
    /// This is a loose tolerance for v1: the kernel uses FP32 scale +
    /// min_val to match the on-disk MQ4G256 layout (the same FP32 cast
    /// that gemm_hfq4g256 family kernels read at inference). Any
    /// codeword that lands on a quantization-bin boundary will flip
    /// between FP32 (GPU) and FP64 (CPU) grids — this is expected and
    /// intentional. The 1e-6 tolerance allows for the FP32 cast plus
    /// minor MFMA reordering effects.
    ///
    /// The Phase 3 byte-identity gate (per plan §6) requires the CPU
    /// path to use FP32 grids too; that pre-quant alignment is out of
    /// scope here (would require a CPU-side `quantize_mq4_element_f32_cast`
    /// helper). For Phase 2F shipping the relative-1e-6 gate is the right
    /// tradeoff.
    #[test]
    fn phase2f_end_to_end_parity() {
        if std::env::var("HIPFIRE_SKIP_GPU_TESTS").is_ok() {
            eprintln!("skip: HIPFIRE_SKIP_GPU_TESTS set");
            return;
        }
        let solver = match RocSolver::load() {
            Ok(s) => s,
            Err(e) => { eprintln!("skip: RocSolver::load failed ({e})"); return; }
        };

        let m = 64;
        let k = 256;

        // Synthetic SPD H (well-conditioned).
        let mut state: u64 = 0xdeadbeef_cafebabe;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let mut a: Mat<f64> = Mat::zeros(k, k);
        for i in 0..k { for j in 0..k { a[(i, j)] = next(); } }
        let mut h: Mat<f64> = Mat::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0_f64;
                for l in 0..k { s += a[(l, i)] * a[(l, j)]; }
                h[(i, j)] = s;
            }
            h[(i, i)] += 0.01;
        }

        // Synthetic W (deterministic).
        let mut w_orig = vec![0.0_f64; m * k];
        for r in 0..m { for c in 0..k {
            w_orig[r * k + c] = ((r as f64 * 0.013).sin() + (c as f64 * 0.027).cos()) * 0.5;
        }}

        // Frozen grids — compute from pre-loop weights using the standard helper.
        let grids = crate::gptq::compute_frozen_block_grids(&w_orig);

        // CPU reference (gptq-hip env unset, no solver passed → CPU OBS).
        let mut w_cpu = w_orig.clone();
        let damp_cpu = crate::gptq::gptq_column_sequential(
            &mut w_cpu, &h, m, k, &grids, 0.01, 1.0,
            "test:phase2f:cpu",
            None::<&crate::gptq::GptqHipSolver>,
        ).expect("CPU OBS");

        // GPU full-OBS path.
        let mut w_gpu = w_orig.clone();
        let damp_gpu = gptq_column_sequential_hip(
            &solver, &mut w_gpu, &h, m, k, &grids, 0.01, 1.0,
            "test:phase2f:gpu",
        ).expect("GPU OBS");

        // Damp value sanity (both should pick the same initial damp since
        // the Hessian is well-conditioned).
        let damp_rel = (damp_cpu - damp_gpu).abs() / damp_cpu.max(1e-30);
        assert!(damp_rel < 1e-12, "damp mismatch: cpu={damp_cpu} gpu={damp_gpu}");

        // Element-wise comparison.
        let mut max_abs = 0.0_f64;
        let mut max_cpu_abs = 0.0_f64;
        let mut max_at = (0usize, 0usize);
        for r in 0..m {
            for c in 0..k {
                let cpu = w_cpu[r * k + c];
                let gpu = w_gpu[r * k + c];
                let d = (cpu - gpu).abs();
                if d > max_abs { max_abs = d; max_at = (r, c); }
                if cpu.abs() > max_cpu_abs { max_cpu_abs = cpu.abs(); }
            }
        }
        let rel = if max_cpu_abs > 0.0 { max_abs / max_cpu_abs } else { max_abs };
        eprintln!(
            "[phase2f] max_abs={max_abs:.3e} max_cpu={max_cpu_abs:.3e} rel={rel:.3e} at (r={}, c={})",
            max_at.0, max_at.1,
        );
        // Loose gate per docstring rationale (FP32 scale cast in kernel).
        assert!(rel < 1e-6, "phase2f rel-err {rel} >= 1e-6 (max_abs {max_abs})");
    }

    #[test]
    fn phase2c_block_apply_mfma_parity() {
        if std::env::var("HIPFIRE_SKIP_GPU_TESTS").is_ok() {
            eprintln!("skip: HIPFIRE_SKIP_GPU_TESTS set");
            return;
        }
        let solver = match RocSolver::load() {
            Ok(s) => s,
            Err(e) => { eprintln!("skip: RocSolver::load failed ({e})"); return; }
        };

        // Synthetic small case: M=128, K=256, B=128.
        // block_start=0, block_end=128 → n_remaining=128 (cols [128..256]).
        let m = 128;
        let k = 256;
        let b_dim = 128;
        let block_start = 0;
        let block_end = 128;

        // Synthetic W (deterministic).
        let mut w_res = vec![0.0_f64; m * k];
        for r in 0..m { for c in 0..k {
            w_res[r * k + c] = ((r as f64) * 0.011 - (c as f64) * 0.0073).sin();
        }}

        // Synthetic err_block.
        let mut err_block = vec![0.0_f64; m * b_dim];
        for r in 0..m { for i in 0..b_dim {
            err_block[r * b_dim + i] = (((r + i * 3) as f64) * 0.017).cos() * 0.02;
        }}

        // Identity perm.
        let perm: Vec<usize> = (0..k).collect();

        // Synthetic upper-tri U.
        let mut u_mat: Mat<f64> = Mat::zeros(k, k);
        for row in 0..k { for col in row..k {
            u_mat[(row, col)] = ((row * 5 + col * 7) as f64).sin() * 0.03 + 0.0005;
        }}

        // CPU reference.
        let mut cpu_w_res = w_res.clone();
        cpu_block_apply_reference(
            &mut cpu_w_res, &err_block, &u_mat, &perm,
            block_start, block_end, m, k, b_dim,
        );

        // GPU path.
        let w_bytes   = m * k * std::mem::size_of::<f64>();
        let err_bytes = m * b_dim * std::mem::size_of::<f64>();
        let u_bytes   = k * k * std::mem::size_of::<f64>();
        let perm_bytes = k * std::mem::size_of::<i32>();

        unsafe {
            let d_w = dev_alloc(&solver, w_bytes).expect("alloc w");
            let d_err = dev_alloc(&solver, err_bytes).expect("alloc err");
            let d_u = dev_alloc(&solver, u_bytes).expect("alloc u");
            let d_perm = dev_alloc(&solver, perm_bytes).expect("alloc perm");

            h2d(&solver, d_w, w_res.as_ptr() as *const u8, w_bytes).expect("h2d w");
            h2d(&solver, d_err, err_block.as_ptr() as *const u8, err_bytes).expect("h2d err");
            let u_col_major = pack_u_column_major(&u_mat);
            h2d(&solver, d_u, u_col_major.as_ptr() as *const u8, u_bytes).expect("h2d u");
            let perm_i32: Vec<i32> = perm.iter().map(|&v| v as i32).collect();
            h2d(&solver, d_perm, perm_i32.as_ptr() as *const u8, perm_bytes).expect("h2d perm");

            gptq_obs_block_apply_mfma_hip(
                &solver,
                d_w, d_err, d_u, d_perm,
                block_start as i32, block_end as i32,
                k as i32, m as i32, b_dim as i32,
            ).expect("launch block_apply_mfma");
            dev_sync(&solver).expect("sync");

            let mut gpu_w_res = vec![0.0_f64; m * k];
            d2h(&solver, gpu_w_res.as_mut_ptr() as *mut u8, d_w, w_bytes).expect("d2h w");

            dev_free(&solver, d_w);
            dev_free(&solver, d_err);
            dev_free(&solver, d_u);
            dev_free(&solver, d_perm);

            let mut max_err = 0.0_f64;
            let mut max_at = (0usize, 0usize);
            for r in 0..m {
                for c in 0..k {
                    let e = (cpu_w_res[r * k + c] - gpu_w_res[r * k + c]).abs();
                    if e > max_err { max_err = e; max_at = (r, c); }
                }
            }
            // FP64 MFMA on gfx942 is bit-exact for compatible operands;
            // tolerance 1e-10 absolute is the same gate Phase D uses.
            assert!(
                max_err < 1e-10,
                "max_err {max_err} >= 1e-10 at (r={}, c={})", max_at.0, max_at.1,
            );
        }
    }
}
