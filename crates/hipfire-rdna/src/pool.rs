// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU memory pool — eliminates hipMalloc/hipFree overhead in the hot loop.
//! Pre-allocates buffers of common sizes and reuses them via a free list.

use hip_bridge::{
    BufferOrigin, DeviceBuffer, HipError, HipResult, HipRuntime, HIP_ERROR_OUT_OF_MEMORY,
};

/// Device memory the pool leaves free for the runtime, `HIPFIRE_POOL_HEADROOM_MB`
/// (default 2048; 0 disables the check). See `GpuPool::alloc`.
pub fn pool_headroom_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("HIPFIRE_POOL_HEADROOM_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2048)
            * 1024
            * 1024
    })
}
use std::collections::HashMap;

/// Host `MemAvailable` in bytes; `None` where `/proc/meminfo` is absent.
///
/// `MemAvailable`, not `MemFree`: reclaimable page cache is genuinely available,
/// and on this box the cache is routinely tens of GiB. `MemFree` would refuse
/// allocations (and loads) that fit comfortably.
pub fn mem_available_bytes() -> Option<usize> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    meminfo.lines().find_map(|line| {
        let kib: usize = line
            .strip_prefix("MemAvailable:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()?;
        Some(kib * 1024)
    })
}

/// What an allocation may draw on. On an integrated GPU, GTT is carved from the
/// same RAM as everything else on the host, so GTT free alone (`hipMemGetInfo`)
/// overstates it whenever something else -- the swarm's own builds -- holds RAM
/// GTT still counts as free: hipfire would map pages the kernel then has to reap
/// processes for (load.rs records this host losing dbus, pipewire and both agents
/// that way). A discrete GPU's VRAM is its own pool.
pub fn admissible(gtt_free: usize, host_available: Option<usize>, integrated: bool) -> usize {
    match host_available {
        Some(host) if integrated => gtt_free.min(host),
        _ => gtt_free,
    }
}

/// [`admissible`] for this device, now. `None` if HIP cannot report.
pub fn admissible_free(hip: &HipRuntime, integrated: bool) -> Option<usize> {
    let (gtt, _) = hip.get_vram_info().ok()?;
    let host = if integrated {
        mem_available_bytes()
    } else {
        None
    };
    Some(admissible(gtt, host, integrated))
}

/// Bytes the pool may keep parked on its free lists, `HIPFIRE_POOL_CACHE_MAX_MB`
/// (default 4096). Past it, a returned buffer goes straight back to HIP.
///
/// Uncapped, a freed buffer stayed cached until device memory itself ran short.
/// On an APU "device memory" is host RAM (gfx1151: a 116 GiB GTT cap of 125 GiB),
/// so the cache of freed per-request KV caches and state snapshots -- every one a
/// different size, so rarely reused -- grew GTT toward all of RAM before the
/// headroom check saw anything: 24 -> 94 GiB over one Corrode swarm turn, with
/// the host down to 23 GB free.
pub fn pool_cache_max_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("HIPFIRE_POOL_CACHE_MAX_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4096)
            * 1024
            * 1024
    })
}

/// A buffer this large is never cached: it is a KV cache or a state snapshot
/// sized to one request, not decode scratch.
const POOL_CACHE_MAX_BUFFER: usize = 256 * 1024 * 1024;

const MIN_ALLOC: usize = 256;

/// A pool of GPU buffers, bucketed by size.
/// Requesting a buffer returns one from the pool (if available) or allocates new.
/// Returning a buffer puts it back in the pool for reuse.
pub struct GpuPool {
    /// Free buffers bucketed by size (rounded up to power of 2)
    free_lists: HashMap<usize, Vec<DeviceBuffer>>,
    /// Total bytes currently allocated (for diagnostics)
    pub total_allocated: usize,
    pub total_reused: usize,
    pub total_new: usize,
    /// Bytes currently parked on `free_lists`.
    cached_bytes: usize,
    /// Integrated GPU: allocations also come out of host RAM (see [`admissible`]).
    pub integrated: bool,
}

/// A snapshot of pool accounting, for leak hunting.
///
/// The discriminating pair is `total_new` against `free_buffers`: a workload
/// that allocates and returns the same shapes every iteration should see
/// `total_new` go flat once warm. `total_new` climbing while `free_buffers`
/// stays put means buffers are being allocated and never handed back — a
/// `GpuTensor` dropped without `free_tensor` leaks exactly that way, since it
/// has no `Drop`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuPoolStats {
    /// Buffers served from a free list.
    pub total_reused: usize,
    /// Buffers that required a fresh `hipMalloc`.
    pub total_new: usize,
    /// Bytes ever freshly allocated (cumulative, never decremented).
    pub total_allocated: usize,
    /// Buffers currently parked on the free lists, and their bytes.
    pub free_buffers: usize,
    pub free_bytes: usize,
    /// How many distinct power-of-2 buckets have free lists.
    pub buckets: usize,
}

impl GpuPool {
    /// Snapshot the pool counters plus the live free-list occupancy.
    pub fn stats(&self) -> GpuPoolStats {
        let mut free_buffers = 0usize;
        let mut free_bytes = 0usize;
        for list in self.free_lists.values() {
            free_buffers += list.len();
            free_bytes += list.iter().map(|b| b.size()).sum::<usize>();
        }
        GpuPoolStats {
            total_reused: self.total_reused,
            total_new: self.total_new,
            total_allocated: self.total_allocated,
            free_buffers,
            free_bytes,
            buckets: self.free_lists.len(),
        }
    }

    pub fn new() -> Self {
        Self {
            free_lists: HashMap::new(),
            total_allocated: 0,
            total_reused: 0,
            total_new: 0,
            cached_bytes: 0,
            integrated: false,
        }
    }

    /// Free-list bucket key. Buffers group by power-of-2 bucket so a
    /// decode-hot scratch of size X reliably finds a reusable slot from
    /// a previous step. The bucket is ONLY a reuse key — the actual HIP
    /// allocation uses the exact requested size (see `alloc`), so there
    /// is no VRAM padding waste.
    fn bucket_key(size: usize) -> usize {
        const MIN: usize = 256;
        if size <= MIN {
            MIN
        } else {
            size.next_power_of_two()
        }
    }

    /// Get a buffer of at least `size` bytes. Reuses from the free-list
    /// if a pooled buffer in the same bucket is large enough; otherwise
    /// allocates from HIP at the EXACT requested size.
    ///
    /// Exact HIP allocation matters for large buffers: previously,
    /// target's 15 GB of per-layer weights on 27B sprawled into
    /// ~100–500 MB power-of-2 buckets that each padded up to 2×,
    /// leaving no contiguous room for the ~3.5 GB draft to load on
    /// 24 GB cards. With exact sizing the padding is zero, all
    /// intended bytes are used, and the draft fits.
    pub fn alloc(&mut self, hip: &HipRuntime, size: usize) -> HipResult<DeviceBuffer> {
        let bucket = Self::bucket_key(size);
        if let Some(list) = self.free_lists.get_mut(&bucket) {
            // Pop buffers until we find one with enough capacity. Smaller
            // pooled buffers (from prior smaller requests) are returned
            // to HIP — better to re-allocate at the right size than to
            // carry undersized buffers around.
            while let Some(buf) = list.pop() {
                self.cached_bytes -= buf.size();
                if buf.size() >= size {
                    self.total_reused += 1;
                    return Ok(buf);
                }
                let _ = hip.free(buf.with_origin(BufferOrigin::Direct));
            }
        }
        // No suitable buffer — allocate at exact requested size. Round up
        // to the nearest 256 B for alignment; HIP may round further but
        // this keeps our accounting honest and avoids tiny-alloc churn.
        let actual = if size < MIN_ALLOC { MIN_ALLOC } else { size };
        self.total_new += 1;
        self.total_allocated += actual;
        // `hip.malloc` stamps Direct. The pool is taking responsibility for this
        // buffer from here on, so re-stamp it Pooled: that is what routes it back
        // to `GpuPool::free` instead of `hipFree` when the caller is done.
        // Keep headroom for the runtime's own allocations (kernel scratch, signals,
        // staging): drain the cache, or fail, BEFORE the driver runs dry. Draining
        // after it had deadlocked: `hipFree` synchronises the device, while queued
        // work waited on runtime memory that only a free could release — the GPU sat
        // idle and the daemon spun in `hipFree` until killed. Failing here instead is
        // a clean `hipError=2` the batch runner splits and retries on.
        let headroom = pool_headroom_bytes();
        if headroom > 0 {
            let integrated = self.integrated;
            let short = |hip: &HipRuntime| {
                admissible_free(hip, integrated).filter(|&free| free < actual + headroom)
            };
            if short(hip).is_some() && self.free_lists.values().any(|list| !list.is_empty()) {
                self.drain(hip);
            }
            if let Some(free) = short(hip) {
                return Err(HipError::new(
                    HIP_ERROR_OUT_OF_MEMORY,
                    &format!(
                        "hipMalloc({actual} bytes) would leave {:.1} MiB free (GTT, or host MemAvailable on an integrated GPU), under the {} MiB headroom kept for the runtime",
                        free.saturating_sub(actual) as f64 / 1048576.0,
                        headroom / 1048576
                    ),
                ));
            }
        }
        let buf = hip.malloc(actual)?;
        Ok(buf.with_origin(BufferOrigin::Pooled))
    }

    /// Return a buffer to the pool for reuse. The buffer's ACTUAL
    /// capacity is what gets reused — we key the free-list by the
    /// power-of-2 bucket so same-size-shaped requests hit the same
    /// slot.
    pub fn free(&mut self, hip: &HipRuntime, buf: DeviceBuffer) {
        // `Gpu::dispose` routes on the tag, so only pooled buffers should arrive.
        // A Direct one here is the #253 leak (a hipMalloc buffer piling into a
        // list nothing draws from); a NonOwning one is the #262 corruption.
        debug_assert!(
            buf.origin() == BufferOrigin::Pooled,
            "GpuPool::free on a {:?} buffer ({:p}, {} B) — allocation and free disagree",
            buf.origin(),
            buf.as_ptr(),
            buf.size()
        );
        let size = buf.size();
        if size >= POOL_CACHE_MAX_BUFFER || self.cached_bytes + size > pool_cache_max_bytes() {
            // Ownership leaves the pool for HIP; stamp to match.
            let _ = hip.free(buf.with_origin(BufferOrigin::Direct));
            return;
        }
        self.cached_bytes += size;
        self.free_lists
            .entry(Self::bucket_key(size))
            .or_default()
            .push(buf);
    }

    /// Actually free all pooled buffers (call on cleanup).
    pub fn drain(&mut self, hip: &HipRuntime) {
        self.cached_bytes = 0;
        for (_, list) in self.free_lists.drain() {
            for buf in list {
                // Ownership leaves the pool for HIP; stamp to match.
                let _ = hip.free(buf.with_origin(BufferOrigin::Direct));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::admissible;

    // An integrated GPU's GTT is host RAM: whichever is lower bounds it. A
    // discrete GPU's VRAM is not, and a missing /proc leaves GTT alone.
    #[test]
    fn host_memory_bounds_only_an_integrated_gpu() {
        const GIB: usize = 1 << 30;
        assert_eq!(admissible(40 * GIB, Some(3 * GIB), true), 3 * GIB);
        assert_eq!(admissible(2 * GIB, Some(30 * GIB), true), 2 * GIB);
        assert_eq!(admissible(40 * GIB, Some(3 * GIB), false), 40 * GIB);
        assert_eq!(admissible(40 * GIB, None, true), 40 * GIB);
    }
}
