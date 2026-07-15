//! W5 — reusable NPU kernel dispatch: the `run_smoke` flow behind a Rust API the
//! runtime can call. An [`NpuKernel`] is one compiled mlir-aie kernel (one shape);
//! it opens the device, allocates the heap, creates a hwctx, loads the tile
//! program (PDI) and the instruction stream once, and then dispatches repeatedly.
//!
//! xclbins are built offline by the mlir-aie toolchain (Python is not in the
//! inference hot path); the runtime loads the cached bytes and dispatches through
//! this HIP-direct-adjacent amdxdna path.
//!
//! Linux-only (amdxdna DRM ioctls).
#![cfg(target_os = "linux")]

use crate::submit::{self, QosInfo, AMDXDNA_BO_CMD, AMDXDNA_BO_SHMEM};
use crate::xclbin::Axlf;
use crate::{DeviceBuffer, XdnaDevice, XdnaError};
use std::cell::RefCell;
use std::os::fd::OwnedFd;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

const DEV_HEAP_BASE_VA: usize = 0x7000_0000_0000;
const DEV_HEAP_STRIDE: usize = 128 * 1024 * 1024;
static NEXT_HEAP_SLOT: AtomicUsize = AtomicUsize::new(0);

// Route-A step-1 instrumentation: split every dispatch into submit-side (host
// cache flush + EXEC_CMD ioctl) vs wait-side (NPU execution + weight/activation
// DMA + syncobj signal). Env-gated (HIPFIRE_XDNA_TRACE), cumulative across the
// process; the last printed line holds the totals. Cheap enough for the hot path.
static XDNA_TRACE_SUBMIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static XDNA_TRACE_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static XDNA_TRACE_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn xdna_trace_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("HIPFIRE_XDNA_TRACE").is_ok_and(|v| v != "0"))
}

fn xdna_trace_record(submit: std::time::Duration, wait: std::time::Duration) {
    let s = XDNA_TRACE_SUBMIT_NS.fetch_add(submit.as_nanos() as u64, Ordering::Relaxed)
        + submit.as_nanos() as u64;
    let w = XDNA_TRACE_WAIT_NS.fetch_add(wait.as_nanos() as u64, Ordering::Relaxed)
        + wait.as_nanos() as u64;
    let c = XDNA_TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    eprintln!(
        "xdna_dispatch_cumulative count={c} submit_ms={:.3} wait_ms={:.3}",
        s as f64 / 1e6,
        w as f64 / 1e6
    );
}

/// A prepared ERT command BO for a fixed set of argument buffers. Building it costs
/// a CREATE_BO + GET_BO_INFO + mmap (~tens of µs); caching it across dispatches with
/// the same buffers removes that from the per-dispatch path (measured ~100µs → far
/// less), which matters for the runtime offload seam.
struct CachedCmd {
    arg_handles: Vec<u32>,
    exec_handles: Vec<u32>, // arg_handles + instr_bo
    cmd_bo: DeviceBuffer,
}

/// A single compiled NPU kernel with its hwctx and loaded program. Bind argument
/// buffers with [`Self::alloc_arg`], fill inputs, then [`Self::dispatch`].
pub struct NpuKernel {
    // Backing heap for the PDI + instruction DEV BOs; must outlive the hwctx.
    _heap: Rc<RefCell<DeviceBuffer>>,
    hwctx: u32,
    syncobj: u32,
    pdi_bo: u32,
    num_tiles: u32,
    instr_bo: u32,
    instr_addr: u64,
    instr_size: usize,
    // Reused across dispatches; one entry per distinct argument set (e.g. the two
    // C-buffers of a pipelined loop), so alternating arg sets don't thrash the cache.
    cmd_cache: RefCell<Vec<CachedCmd>>,
    // Declared last so internal BOs release their GEM handles before the last
    // shared device reference closes. Peer kernels share this DRM file.
    dev: Rc<XdnaDevice>,
}

impl NpuKernel {
    /// Heap size backing the PDI + instruction streams. The AIE2 dev-mem window is
    /// 64 MiB; PDIs/instructions are a few KiB, so one 64 MiB heap is ample.
    const HEAP_BYTES: usize = 64 * 1024 * 1024;

    /// Load a compiled kernel: `xclbin` bytes (for the PDI) and its `insts`
    /// instruction stream. Sets up the hwctx and loads the program on hardware.
    pub fn load(xclbin: &[u8], insts: &[u8]) -> Result<Self, XdnaError> {
        let dev = Rc::new(XdnaDevice::open_default()?);
        let heap_slot = NEXT_HEAP_SLOT.fetch_add(1, Ordering::Relaxed);
        let heap_va = DEV_HEAP_BASE_VA
            .checked_add(heap_slot * DEV_HEAP_STRIDE)
            .ok_or_else(|| {
                XdnaError::Ioctl(std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "NPU device heap VA slots exhausted",
                ))
            })?;
        let heap = Rc::new(RefCell::new(
            dev.alloc_dev_heap_at(Self::HEAP_BYTES, heap_va)?,
        ));
        Self::load_on_device(dev, heap, xclbin, insts)
    }

    /// Load a second kernel on a new hardware context backed by the same DRM
    /// file description as `peer`. Both kernels then share one GEM-handle
    /// namespace, allowing direct SHMEM producer/consumer boundaries without
    /// PRIME export/import.
    pub fn load_peer(peer: &Self, xclbin: &[u8], insts: &[u8]) -> Result<Self, XdnaError> {
        Self::load_on_device(Rc::clone(&peer.dev), Rc::clone(&peer._heap), xclbin, insts)
    }

    fn load_on_device(
        dev: Rc<XdnaDevice>,
        heap: Rc<RefCell<DeviceBuffer>>,
        xclbin: &[u8],
        insts: &[u8],
    ) -> Result<Self, XdnaError> {
        let axlf = Axlf::parse(xclbin)?;
        let part = axlf.aie_partition().ok_or(XdnaError::NoAiePartition)?;
        let num_tiles = part.column_width as u32 * 4; // aie2p: 4 core rows/column

        let (hwctx, syncobj) = dev.create_hwctx(num_tiles, 0, 0x800, &QosInfo::default())?;
        let (pdi_bo, _) = dev.alloc_dev_bo(&mut heap.borrow_mut(), part.pdi)?;
        if let Err(e) = dev.config_hwctx_cu(hwctx, pdi_bo) {
            let _ = dev.destroy_hwctx(hwctx);
            return Err(e);
        }
        let (instr_bo, instr_addr) = match dev.alloc_dev_bo(&mut heap.borrow_mut(), insts) {
            Ok(v) => v,
            Err(e) => {
                let _ = dev.destroy_hwctx(hwctx);
                return Err(e);
            }
        };

        Ok(Self {
            _heap: heap,
            hwctx,
            syncobj,
            pdi_bo,
            num_tiles,
            instr_bo,
            instr_addr,
            instr_size: insts.len(),
            cmd_cache: RefCell::new(Vec::new()),
            dev,
        })
    }

    /// Allocate a SHMEM argument buffer (host-visible, NPU-accessible via PASID).
    /// The caller fills inputs and reads outputs directly through its slices.
    pub fn alloc_arg(&self, size: usize) -> Result<DeviceBuffer, XdnaError> {
        self.dev.alloc_buffer(size, AMDXDNA_BO_SHMEM)
    }

    /// Make host-written argument bytes visible to the NPU. Resident weights
    /// call this once at upload time and skip repeated flushes during dispatch.
    pub fn sync_to_device(&self, buffer: &DeviceBuffer) -> Result<(), XdnaError> {
        self.dev
            .sync_bo(buffer.handle(), submit::SYNC_DIRECT_TO_DEVICE, buffer.len())
    }

    /// Make only the prefix of a shared argument visible to the NPU. This is
    /// required when another hardware context owns an appended region of the
    /// same dma-buf and this context's host mapping may still cache stale tail
    /// lines.
    pub fn sync_to_device_prefix(
        &self,
        buffer: &DeviceBuffer,
        bytes: usize,
    ) -> Result<(), XdnaError> {
        if bytes > buffer.len() {
            return Err(XdnaError::InvalidOpus(
                "sync prefix exceeds argument buffer".to_string(),
            ));
        }
        self.dev
            .sync_bo(buffer.handle(), submit::SYNC_DIRECT_TO_DEVICE, bytes)
    }

    /// Recreate the hardware context while retaining the DRM device, PDI,
    /// instruction, argument, and imported dma-buf BOs. This resets array-local
    /// core/FIFO state without changing any command-packet argument addresses.
    pub fn recreate_hwctx(&mut self) -> Result<(), XdnaError> {
        self.dev.destroy_hwctx(self.hwctx)?;
        let (hwctx, syncobj) =
            self.dev
                .create_hwctx(self.num_tiles, 0, 0x800, &QosInfo::default())?;
        if let Err(error) = self.dev.config_hwctx_cu(hwctx, self.pdi_bo) {
            let _ = self.dev.destroy_hwctx(hwctx);
            return Err(error);
        }
        self.hwctx = hwctx;
        self.syncobj = syncobj;
        Ok(())
    }

    /// Import an external dma-buf (e.g. an amdgpu GTT BO) as an argument buffer, zero-copy:
    /// the kernel then reads/writes the *same physical pages* as the exporting engine (the
    /// GPU) — no host round-trip. See [`crate::XdnaDevice::import_dmabuf`].
    pub fn import_dmabuf(
        &self,
        fd: i32,
        size: usize,
        map: bool,
    ) -> Result<DeviceBuffer, XdnaError> {
        self.dev.import_dmabuf(fd, size, map)
    }

    /// Export an argument BO allocated by this kernel's XDNA device as a
    /// dma-buf. Other NPU contexts can import it for a physical zero-copy
    /// producer/consumer boundary.
    pub fn export_dmabuf(&self, buffer: &DeviceBuffer) -> Result<OwnedFd, XdnaError> {
        self.dev.export_dmabuf(buffer)
    }

    /// Run the kernel over `args` in kernel-signature order (e.g. A, W, C). Flushes
    /// the argument buffers to the device, submits the command, and blocks until it
    /// completes; on return the output buffers are readable directly (SHMEM is
    /// coherent once the timeline signals).
    pub fn dispatch(&self, args: &[&DeviceBuffer]) -> Result<(), XdnaError> {
        if xdna_trace_enabled() {
            let t0 = std::time::Instant::now();
            let seq = self.submit(args)?;
            let t1 = std::time::Instant::now();
            let r = self.wait(seq);
            xdna_trace_record(t1 - t0, t1.elapsed());
            return r;
        }
        let seq = self.submit(args)?;
        self.wait(seq)
    }

    /// Blocking dispatch with explicit host-to-device synchronization flags.
    pub fn dispatch_synced(&self, args: &[&DeviceBuffer], sync: &[bool]) -> Result<(), XdnaError> {
        if xdna_trace_enabled() {
            let t0 = std::time::Instant::now();
            let seq = self.submit_synced(args, Some(sync))?;
            let t1 = std::time::Instant::now();
            let r = self.wait(seq);
            xdna_trace_record(t1 - t0, t1.elapsed());
            return r;
        }
        let seq = self.submit_synced(args, Some(sync))?;
        self.wait(seq)
    }

    /// Enqueue several fixed-buffer invocations and wait once for the final
    /// timeline point. The installed amdxdna driver accepts one command BO per
    /// submit ioctl, but commands on a hardware context execute in order, so
    /// waiting for the last submission also completes every earlier one.
    pub fn dispatch_batch(&self, arg_sets: &[Vec<&DeviceBuffer>]) -> Result<(), XdnaError> {
        self.dispatch_batch_synced(arg_sets, None)
    }

    /// Batched counterpart to [`Self::dispatch_synced`]. The same sync mask
    /// applies to every argument set.
    pub fn dispatch_batch_synced(
        &self,
        arg_sets: &[Vec<&DeviceBuffer>],
        sync: Option<&[bool]>,
    ) -> Result<(), XdnaError> {
        let mut final_sequence = None;
        for args in arg_sets {
            final_sequence = Some(self.submit_synced(args, sync)?);
        }
        final_sequence.map_or(Ok(()), |sequence| self.wait(sequence))
    }

    /// Non-blocking submit: flush inputs and enqueue the command, returning the
    /// timeline sequence to [`Self::wait`] on. Lets the caller overlap host work (e.g.
    /// reading a previous dispatch's output) with this dispatch's execution — commands
    /// on the hwctx run in submit order, so a later [`Self::wait`] still sees this one
    /// complete. Pair each `submit` with exactly one `wait`, and double-buffer any
    /// output the next submit would overwrite before you read it.
    pub fn submit(&self, args: &[&DeviceBuffer]) -> Result<u64, XdnaError> {
        self.submit_synced(args, None)
    }

    /// Like [`Self::submit`], but only flush the args whose `sync[i]` is true. `None`
    /// flushes all (same as `submit`). Use this to skip the `sync_bo` on inputs the host
    /// did not change since their last dispatch, and on pure outputs (the kernel writes
    /// them, so they never need a host→device flush; SHMEM is coherent for read-back once
    /// the timeline signals). Each unnecessary flush is a full-buffer cache op.
    pub fn submit_synced(
        &self,
        args: &[&DeviceBuffer],
        sync: Option<&[bool]>,
    ) -> Result<u64, XdnaError> {
        for (i, a) in args.iter().enumerate() {
            if sync.is_none_or(|s| s[i]) {
                self.dev
                    .sync_bo(a.handle(), submit::SYNC_DIRECT_TO_DEVICE, a.len())?;
            }
        }

        // Reuse the command BO per argument set — the packet's device addresses are
        // fixed per buffer, so only the first submit of a given set pays CREATE_BO +
        // mmap. One cache entry per set so alternating (pipelined) sets don't thrash.
        let arg_handles: Vec<u32> = args.iter().map(|b| b.handle()).collect();
        let mut cache = self.cmd_cache.borrow_mut();
        if !cache.iter().any(|c| c.arg_handles == arg_handles) {
            let addrs: Vec<u64> = args.iter().map(|b| b.host_addr()).collect();
            let packet = submit::dpu_cmd_packet(self.instr_addr, self.instr_size, &addrs);
            let mut cmd_bo = self.dev.alloc_buffer(4096, AMDXDNA_BO_CMD)?;
            cmd_bo.as_mut_slice()[..packet.len()].copy_from_slice(&packet);
            // One physical BO may intentionally back multiple DPU arguments
            // (for example an in-place input/output tail). The command packet
            // keeps every address slot, but the residency handle list must not
            // repeat a GEM handle: amdxdna rejects duplicates with EALREADY.
            let mut exec_handles = Vec::with_capacity(arg_handles.len() + 1);
            for &handle in &arg_handles {
                if !exec_handles.contains(&handle) {
                    exec_handles.push(handle);
                }
            }
            if !exec_handles.contains(&self.instr_bo) {
                exec_handles.push(self.instr_bo); // instruction BO residency
            }
            cache.push(CachedCmd {
                arg_handles: arg_handles.clone(),
                exec_handles,
                cmd_bo,
            });
        }
        let cmd = cache.iter().find(|c| c.arg_handles == arg_handles).unwrap();
        self.dev
            .exec_cmd(self.hwctx, cmd.cmd_bo.handle(), &cmd.exec_handles)
    }

    /// Block until the submitted command at timeline point `seq` completes; its output
    /// buffers are then readable directly (SHMEM is coherent once the timeline signals).
    pub fn wait(&self, seq: u64) -> Result<(), XdnaError> {
        self.dev.syncobj_wait(self.syncobj, seq)
    }

    /// Reconcile the host cache for an output buffer before a CPU read-back. Call after
    /// [`Self::wait`], before reading. A blocking dispatch+read is coherent without this,
    /// but a *pipelined* loop overlaps the read-back of one buffer with a concurrent DMA
    /// write to another; the hardware prefetcher can pull stale lines of the in-flight
    /// buffer into cache, and there is no invalidate before its later read. `TO_DEVICE`
    /// clean+invalidates on this driver (`FROM_DEVICE` EINVALs on data BOs), which clears
    /// those stale lines so the read sees the kernel's writes. Verified: without this,
    /// pipelined read-back fails intermittently (~1 run in 3); with it, 0/16 across the
    /// single- and multi-K-chunk paths.
    pub fn sync_output(&self, buf: &DeviceBuffer) -> Result<(), XdnaError> {
        self.dev
            .sync_bo(buf.handle(), submit::SYNC_DIRECT_TO_DEVICE, buf.len())
    }

    /// Submit `args` WITHOUT blocking, returning an owning in-flight handle. Where
    /// [`Self::dispatch`] fuses submit + wait, this lets the caller drive another engine
    /// (the GPU) while the NPU runs, then [`Self::poll`] / [`Self::wait_inflight`] for
    /// completion — the basis for a GPU‖NPU microbatch pipeline (e.g. GPU verifies step N
    /// while the NPU drafts N+1). The argument buffers must outlive the handle (the caller
    /// owns them), and the handle must be waited (or polled to completion) before drop.
    ///
    /// Async counterpart to the blocking [`Self::submit`] (which returns a timeline `seq`);
    /// this returns an owning [`NpuInFlight`] so multiple dispatches can be in flight at once.
    pub fn submit_inflight(&self, args: &[&DeviceBuffer]) -> Result<NpuInFlight, XdnaError> {
        self.submit_tagged(args, 0)
    }

    /// [`Self::submit_inflight`] with a caller-defined `tag` carried on the handle. The
    /// scheduler stamps the microbatch / layer / expert id so it can correlate NPU
    /// completions with the per-token grouping it is pipelining across the GPU without a
    /// side table — the explicit shared state between dispatcher and scheduler.
    ///
    /// Each submit builds its OWN command BO, owned by the returned handle so it stays
    /// resident until the dispatch completes. (The blocking `submit_synced` path's shared
    /// command-BO cache cannot back multiple in-flight dispatches: a second submit with
    /// different buffers would free the first's command BO mid-flight.)
    pub fn submit_tagged(
        &self,
        args: &[&DeviceBuffer],
        tag: u64,
    ) -> Result<NpuInFlight, XdnaError> {
        for a in args {
            self.dev
                .sync_bo(a.handle(), submit::SYNC_DIRECT_TO_DEVICE, a.len())?;
        }
        let addrs: Vec<u64> = args.iter().map(|b| b.host_addr()).collect();
        let packet = submit::dpu_cmd_packet(self.instr_addr, self.instr_size, &addrs);
        let mut cmd_bo = self.dev.alloc_buffer(4096, AMDXDNA_BO_CMD)?;
        cmd_bo.as_mut_slice()[..packet.len()].copy_from_slice(&packet);
        let mut exec_handles: Vec<u32> = args.iter().map(|b| b.handle()).collect();
        exec_handles.push(self.instr_bo); // instruction BO is an EXEC arg (residency)
        let seq = self
            .dev
            .exec_cmd(self.hwctx, cmd_bo.handle(), &exec_handles)?;
        Ok(NpuInFlight {
            seq,
            tag,
            _cmd_bo: cmd_bo,
        })
    }

    /// Non-blocking completion check for an in-flight dispatch (`true` = done). Lets the
    /// scheduler reap finished NPU work between GPU steps without parking the GPU thread.
    pub fn poll(&self, f: &NpuInFlight) -> Result<bool, XdnaError> {
        self.dev.syncobj_poll(self.syncobj, f.seq)
    }

    /// Block until an in-flight dispatch completes; on return its output buffers are
    /// readable (SHMEM is coherent once the timeline signals). Consumes the handle,
    /// freeing its command BO. Dispatches on one kernel complete in submission order.
    /// Async counterpart to the blocking [`Self::wait`], which takes a timeline `seq`.
    pub fn wait_inflight(&self, f: NpuInFlight) -> Result<(), XdnaError> {
        self.dev.syncobj_wait(self.syncobj, f.seq)
    }
}

/// An in-flight NPU dispatch from [`NpuKernel::submit_inflight`]. Owns the command BO
/// backing the submission, keeping it resident until [`NpuKernel::wait_inflight`] (or
/// drop). Carries the timeline `seq` (submission order on the kernel's hwctx) and a caller
/// `tag` (the scheduler's microbatch / layer / expert id) so the dispatcher and scheduler
/// share in-flight state explicitly: poll by handle, correlate by tag, order by seq.
pub struct NpuInFlight {
    seq: u64,
    tag: u64,
    _cmd_bo: DeviceBuffer,
}

impl NpuInFlight {
    /// Timeline point for this dispatch — monotonic per kernel; the scheduler's ordering hint.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Caller-defined correlation id (microbatch / layer / expert).
    pub fn tag(&self) -> u64 {
        self.tag
    }

    /// Re-tag in place, e.g. when the scheduler regroups tokens across a layer boundary.
    pub fn set_tag(&mut self, tag: u64) {
        self.tag = tag;
    }
}

impl Drop for NpuKernel {
    fn drop(&mut self) {
        let _ = self.dev.destroy_hwctx(self.hwctx);
    }
}
