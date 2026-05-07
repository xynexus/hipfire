//! hip-bridge: Safe Rust FFI to AMD HIP runtime via dlopen.
//! Modeled after rustane's ane-bridge — no link-time dependency on libamdhip64.

mod ffi;
mod error;
mod kernarg;
mod rocblas;

pub use error::{
    HipError, HipResult, HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED,
    HIP_ERROR_PEER_ACCESS_NOT_ENABLED, HIP_ERROR_PEER_ACCESS_UNSUPPORTED,
};
pub use ffi::{Event, Function, Graph, GraphExec, HipPointerAttribute, HipRuntime, Module, Stream};
pub use ffi::launch_counters;
pub use kernarg::KernargBlob;
pub use rocblas::{Rocblas, RocblasDatatype, RocblasError, RocblasOperation, RocblasResult};

/// Re-export memory copy direction for callers.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

/// Mirrors `hipMemoryType`. FFI stores raw `u32`; use `from_raw` to convert.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    Unregistered = 0,
    Host = 1,
    Device = 2,
    Managed = 3,
    Array = 10,
    Unified = 11,
}

impl MemoryType {
    pub fn from_raw(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Unregistered),
            1 => Some(Self::Host),
            2 => Some(Self::Device),
            3 => Some(Self::Managed),
            10 => Some(Self::Array),
            11 => Some(Self::Unified),
            _ => None,
        }
    }
}

/// Opaque GPU buffer handle. Tracks pointer + size for safety.
pub struct DeviceBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
}

impl DeviceBuffer {
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Create a non-owning DeviceBuffer from a raw pointer and size.
    /// The caller must ensure the pointer is valid GPU memory.
    /// The resulting buffer must NOT be freed (it doesn't own the memory).
    pub unsafe fn from_raw(ptr: *mut std::ffi::c_void, size: usize) -> DeviceBuffer {
        DeviceBuffer { ptr, size }
    }

    /// Create a non-owning alias to the same GPU memory.
    /// The alias must not outlive the original buffer.
    /// Used for reshaping tensors without reallocating.
    /// # Safety
    /// Caller must ensure the alias doesn't outlive the original.
    pub unsafe fn alias(&self) -> DeviceBuffer {
        DeviceBuffer { ptr: self.ptr, size: self.size }
    }
}

// DeviceBuffer is Send — GPU pointers can be sent between threads.
// They are NOT Sync — concurrent access requires stream synchronization.
unsafe impl Send for DeviceBuffer {}

/// Pinned (page-locked) host memory allocated via `hipHostMalloc`.
/// Used as a fast-path staging buffer for cross-card transfers in the
/// hetero PP+DFlash path: writes from one GPU + reads by another GPU
/// over `hipMemcpyHtoDAsync` / `hipMemcpyDtoHAsync` go through the
/// driver's direct-DMA path instead of HIP's bounce-buffer fallback
/// (the slow ~64 MB/s path on iGPU↔eGPU pairs over Thunderbolt). On
/// Strix Halo (gfx1151 iGPU + system DDR5), the pinned host buffer
/// physically lives in the same memory the iGPU's "VRAM" lives in,
/// so an iGPU `memcpy_dtoh_async` to this buffer is effectively a
/// system-RAM-to-system-RAM copy.
///
/// Owns the allocation. `HipRuntime::host_free` must be called
/// explicitly to surface errors; Drop does not auto-free (mirrors the
/// `DeviceBuffer` pattern).
pub struct PinnedHostBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
}

impl PinnedHostBuffer {
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.ptr as *const _
    }

    pub fn as_mut_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Borrow the buffer as a host byte slice. Safe because pinned host
    /// memory is CPU-addressable.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.size) }
    }

    /// Borrow the buffer as a mutable host byte slice. The caller is
    /// responsible for ensuring no concurrent GPU async memcpy is in
    /// flight against this buffer (host-side Rust borrow checker can't
    /// see GPU access).
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.size) }
    }

    /// # Safety
    /// Caller must ensure `ptr` was returned from `hipHostMalloc` and
    /// `size` matches the allocation. The resulting buffer takes ownership.
    pub unsafe fn from_raw(ptr: *mut std::ffi::c_void, size: usize) -> PinnedHostBuffer {
        PinnedHostBuffer { ptr, size }
    }
}

unsafe impl Send for PinnedHostBuffer {}
