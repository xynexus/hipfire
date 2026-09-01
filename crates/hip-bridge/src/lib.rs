// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 alpineq
// hipfire — see LICENSE and NOTICE in the project root.

#![allow(
    clippy::manual_dangling_ptr,
    clippy::manual_pattern_char_comparison,
    clippy::missing_safety_doc,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::type_complexity
)]

//! hip-bridge: Safe Rust FFI to AMD HIP runtime via dlopen.
//! Modeled after rustane's ane-bridge — no link-time dependency on libamdhip64.

mod error;
mod ffi;
mod kernarg;
mod rccl;
mod rocblas;

pub use error::{
    HipError, HipResult, HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED, HIP_ERROR_PEER_ACCESS_NOT_ENABLED,
    HIP_ERROR_PEER_ACCESS_UNSUPPORTED,
};
pub use ffi::alloc_stats;
pub use ffi::launch_counters;
pub use ffi::{
    Event, Function, Graph, GraphExec, HipPointerAttribute, HipRuntime, HostBuffer, ImportedBuffer,
    Module, Stream,
};
pub use kernarg::{KernArg, KernargBlob};
pub use rccl::{RcclComms, RcclDataType, RcclError, RcclRedOp, RcclResult, NCCL_SUCCESS};
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

/// How a [`DeviceBuffer`] was allocated, and therefore how it must be disposed of.
///
/// `DeviceBuffer` used to be `{ ptr, size }` with no provenance, so
/// `Gpu::free_tensor` returned every buffer to `GpuPool` regardless of where it
/// came from. Two of the three cases were wrong, and both have cost real bugs:
///
/// * a **non-owning** slab alias returned to the pool was re-handed out as
///   scratch and written over another layer's live weights — silent corruption
///   (#262, `shard_moe_experts`);
/// * a **direct** `hipMalloc` buffer freed into a pool nothing drew from leaked
///   ~9.6 MB per page-in and OOM'd a 122B mid-generation (#253, expert pager).
///
/// The tag makes both unrepresentable: disposal is chosen by the value, not
/// remembered by the caller. See `docs/todo/free-tensor-provenance.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferOrigin {
    /// Handed out by `GpuPool::alloc`. Must go back to `GpuPool::free`.
    Pooled,
    /// Allocated by [`HipRuntime::malloc`] outside the pool. Must go to
    /// [`HipRuntime::free`].
    Direct,
    /// A view into memory owned by someone else — [`DeviceBuffer::from_raw`],
    /// [`DeviceBuffer::alias`], `GpuTensor::sub_offset`. Must be freed by
    /// NEITHER path.
    NonOwning,
}

/// Per-call-site copy census for `HIPFIRE_COPY_REPORT=1`.
///
/// `__amd_rocclr_copyBuffer` measured **7.4% of decode GPU time** on
/// Qwen3.6-35B-A3B — 25771 dispatches, grid 512 / workgroup 512, ~2.9 us each,
/// roughly 537 per token. That is launch LATENCY, not bandwidth: the copies are
/// tiny and serialised into the decode stream.
///
/// A kernel trace gives dispatches but not call sites, and reading the source
/// eliminated the two obvious candidates without finding the real one. This sits
/// at the HIP boundary rather than on the `*_auto` wrappers because 266 call
/// sites use `hip.memcpy_htod` directly against 60 that use the wrapper — the
/// wrapper is not the chokepoint. The wrappers carry `#[track_caller]` so
/// attribution passes through them to the real origin.
///
/// Off by default; one relaxed atomic load per copy when unset.
pub mod copy_census {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    static ON: AtomicBool = AtomicBool::new(false);
    static INIT: std::sync::Once = std::sync::Once::new();
    static SITES: Mutex<Option<BTreeMap<String, (u64, u64)>>> = Mutex::new(None);

    pub fn enabled() -> bool {
        INIT.call_once(|| {
            let on = std::env::var("HIPFIRE_COPY_REPORT")
                .map(|v| matches!(v.as_str(), "1" | "true" | "on" | "yes"))
                .unwrap_or(false);
            ON.store(on, Ordering::Relaxed);
        });
        ON.load(Ordering::Relaxed)
    }

    pub fn record(site: &'static std::panic::Location<'static>, bytes: usize) {
        if !enabled() {
            return;
        }
        let key = format!("{}:{}", site.file(), site.line());
        if let Ok(mut g) = SITES.lock() {
            let m = g.get_or_insert_with(BTreeMap::new);
            let e = m.entry(key).or_insert((0, 0));
            e.0 += 1;
            e.1 += bytes as u64;
        }
    }

    /// Print the census, busiest site first.
    pub fn report() {
        if !enabled() {
            return;
        }
        let Ok(g) = SITES.lock() else { return };
        let Some(m) = g.as_ref() else { return };
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by_key(|(_, (c, _))| std::cmp::Reverse(*c));
        let total: u64 = v.iter().map(|(_, (c, _))| *c).sum();
        eprintln!("\n=== copy census ({total} copies) ===");
        for (site, (calls, bytes)) in v.iter().take(24) {
            eprintln!(
                "  {calls:>8} calls  {:>10.2} MB  {:>5.1}%  {site}",
                *bytes as f64 / 1.048576e6,
                100.0 * *calls as f64 / total.max(1) as f64
            );
        }
    }
}

/// Opaque GPU buffer handle. Tracks pointer + size + how to dispose of it.
pub struct DeviceBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
    origin: BufferOrigin,
}

impl DeviceBuffer {
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// How this buffer must be disposed of. See [`BufferOrigin`].
    pub fn origin(&self) -> BufferOrigin {
        self.origin
    }

    /// Re-stamp provenance as ownership transfers between allocators.
    ///
    /// ONLY an allocator taking or releasing responsibility for a buffer may
    /// call this: `GpuPool::alloc` stamps [`BufferOrigin::Pooled`] on a buffer
    /// it begins managing, and stamps [`BufferOrigin::Direct`] back on one it
    /// hands to HIP. Stamping anywhere else reintroduces exactly the mispairing
    /// the tag exists to prevent.
    pub fn with_origin(mut self, origin: BufferOrigin) -> DeviceBuffer {
        self.origin = origin;
        self
    }

    /// Create a non-owning DeviceBuffer from a raw pointer and size.
    /// The caller must ensure the pointer is valid GPU memory.
    /// The resulting buffer must NOT be freed (it doesn't own the memory).
    pub unsafe fn from_raw(ptr: *mut std::ffi::c_void, size: usize) -> DeviceBuffer {
        DeviceBuffer {
            ptr,
            size,
            origin: BufferOrigin::NonOwning,
        }
    }

    /// Create a non-owning alias to the same GPU memory.
    /// The alias must not outlive the original buffer.
    /// Used for reshaping tensors without reallocating.
    /// # Safety
    /// Caller must ensure the alias doesn't outlive the original.
    pub unsafe fn alias(&self) -> DeviceBuffer {
        DeviceBuffer {
            ptr: self.ptr,
            size: self.size,
            origin: BufferOrigin::NonOwning,
        }
    }
}

// DeviceBuffer is Send — GPU pointers can be sent between threads.
// They are NOT Sync — concurrent access requires stream synchronization.
unsafe impl Send for DeviceBuffer {}

#[cfg(test)]
mod buffer_origin_tests {
    use super::*;

    // These construct buffers over bogus pointers and never free them. That is
    // sound here: `DeviceBuffer` has no `Drop`, so nothing is dereferenced and
    // nothing is released — the tests read tags only, never memory.

    /// An alias must NOT inherit the original's ownership.
    ///
    /// This is #262 in miniature: the slab was pooled, `alias`/`sub_offset`
    /// handed out an interior view by value, and freeing that view returned
    /// mid-slab memory to the pool, which re-issued it as scratch over live
    /// weights. If this assertion ever flips, that bug is representable again.
    #[test]
    fn an_alias_never_inherits_ownership() {
        let owned = unsafe { DeviceBuffer::from_raw(0x1000 as *mut std::ffi::c_void, 4096) }
            .with_origin(BufferOrigin::Pooled);
        assert_eq!(owned.origin(), BufferOrigin::Pooled);

        let view = unsafe { owned.alias() };
        assert_eq!(view.origin(), BufferOrigin::NonOwning);
        assert_eq!(
            view.as_ptr(),
            owned.as_ptr(),
            "the alias still points at it"
        );
    }

    /// Every constructor that does not allocate yields a non-owning buffer, so a
    /// new call site cannot acquire ownership by forgetting to think about it.
    #[test]
    fn non_allocating_constructors_are_non_owning() {
        let raw = unsafe { DeviceBuffer::from_raw(0x2000 as *mut std::ffi::c_void, 16) };
        assert_eq!(raw.origin(), BufferOrigin::NonOwning);
    }

    /// Re-stamping transfers responsibility between allocators and touches
    /// nothing else — `GpuPool` relies on this to adopt a fresh `hipMalloc`
    /// buffer and to release one back to HIP on drain.
    #[test]
    fn restamping_moves_responsibility_not_memory() {
        let b = unsafe { DeviceBuffer::from_raw(0x3000 as *mut std::ffi::c_void, 64) };
        let (ptr, size) = (b.as_ptr(), b.size());

        let b = b.with_origin(BufferOrigin::Pooled);
        assert_eq!(b.origin(), BufferOrigin::Pooled);

        let b = b.with_origin(BufferOrigin::Direct);
        assert_eq!(b.origin(), BufferOrigin::Direct);
        assert_eq!((b.as_ptr(), b.size()), (ptr, size));
    }
}
