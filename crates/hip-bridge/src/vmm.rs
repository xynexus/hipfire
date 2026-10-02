//! Device buffers built from separately mapped physical pages behind one reserved
//! virtual range.
//!
//! A kernel sees an ordinary contiguous pointer; behind it every page is its own
//! physical allocation. Two things follow that a plain `hipMalloc` buffer cannot do:
//!
//! - **Grow in place.** Mapping more pages at the end extends the buffer without
//!   moving it, so nothing that holds the pointer has to be told.
//! - **Share a prefix.** Another region can map the same physical pages at the same
//!   offsets, so two buffers read identical leading bytes from one copy in memory.
//!   The caller must only share bytes neither side will write again.
//!
//! No reference counting: a page's memory lives until its last mapping is unmapped
//! (the handle is released as soon as it is mapped, and recovered from the address
//! when a second region aliases it).

use crate::{DeviceBuffer, HipResult, HipRuntime};
use std::ffi::c_void;

pub struct VmmRegion {
    base: *mut c_void,
    reserved: usize,
    page: usize,
    mapped: usize,
}

// Device addresses, not host memory; the region is moved between threads with the
// session that owns it, exactly as a `DeviceBuffer` is.
unsafe impl Send for VmmRegion {}
unsafe impl Sync for VmmRegion {}

impl VmmRegion {
    /// Reserve address space for `capacity` bytes (rounded up to whole pages). No
    /// memory is committed until [`Self::ensure_mapped`] or [`Self::alias_prefix_from`].
    /// `page` must be a multiple of [`HipRuntime::vmm_granularity`].
    pub fn reserve(hip: &HipRuntime, capacity: usize, page: usize) -> HipResult<Self> {
        let reserved = capacity.div_ceil(page).max(1) * page;
        Ok(Self {
            base: hip.vmm_reserve(reserved)?,
            reserved,
            page,
            mapped: 0,
        })
    }

    pub fn page_size(&self) -> usize {
        self.page
    }

    /// Bytes currently backed by memory, from the start of the range.
    pub fn mapped_bytes(&self) -> usize {
        self.mapped
    }

    pub fn reserved_bytes(&self) -> usize {
        self.reserved
    }

    /// A non-owning view of the mapped bytes. Stays valid across growth: pages are
    /// only ever added after it.
    pub fn buffer(&self) -> DeviceBuffer {
        unsafe { DeviceBuffer::from_raw(self.base, self.mapped) }
    }

    /// Back at least `bytes` from the start with fresh pages; the new bytes are
    /// uninitialised. Returns the offset the new pages start at.
    pub fn ensure_mapped(&mut self, hip: &HipRuntime, bytes: usize) -> HipResult<usize> {
        let target = bytes.div_ceil(self.page) * self.page;
        if target > self.reserved {
            return Err(crate::HipError::new(
                0,
                &format!(
                    "VmmRegion: {bytes} bytes exceeds the {} reserved",
                    self.reserved
                ),
            ));
        }
        let start = self.mapped;
        while self.mapped < target {
            let at = unsafe { (self.base as *mut u8).add(self.mapped) } as *mut c_void;
            hip.vmm_map_new(at, self.page)?;
            self.mapped += self.page;
        }
        if self.mapped > start {
            let at = unsafe { (self.base as *mut u8).add(start) } as *mut c_void;
            hip.vmm_set_access(at, self.mapped - start)?;
        }
        Ok(start)
    }

    /// Map `src`'s physical pages covering its first `bytes` (whole pages only —
    /// rounded DOWN) at the same offsets here, so both read the same memory. Must be
    /// called on an empty region. Returns how many leading bytes are now shared.
    pub fn alias_prefix_from(
        &mut self,
        hip: &HipRuntime,
        src: &VmmRegion,
        bytes: usize,
    ) -> HipResult<usize> {
        assert_eq!(
            self.mapped, 0,
            "alias_prefix_from on a region that already has pages"
        );
        assert_eq!(self.page, src.page, "alias_prefix_from across page sizes");
        let shared = (bytes.min(src.mapped).min(self.reserved) / self.page) * self.page;
        while self.mapped < shared {
            let off = self.mapped;
            let dst = unsafe { (self.base as *mut u8).add(off) } as *mut c_void;
            let from = unsafe { (src.base as *mut u8).add(off) } as *mut c_void;
            hip.vmm_map_alias(dst, from, self.page)?;
            self.mapped += self.page;
        }
        if shared > 0 {
            hip.vmm_set_access(self.base, shared)?;
        }
        Ok(shared)
    }

    /// Unmap everything and give the address range back. Memory another region
    /// still maps stays alive for it.
    pub fn free(self, hip: &HipRuntime) -> HipResult<()> {
        if self.mapped > 0 {
            hip.vmm_unmap(self.base, self.mapped)?;
        }
        hip.vmm_free_reservation(self.base, self.reserved)
    }
}
