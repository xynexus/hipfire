//! Per-(kernel, M-tuple, K) hipGraph cache for decode-path GEMVs.
//!
//! See `docs/plans/gemv-graph-cache.prd` for the full design.
//!
//! **PR2 (this file):** capture-on-second-call / replay-on-third-call for
//! `gemv_hfq4g256` (plain) only. The cache stores Box-allocated
//! `kernarg`, `kernarg_len`, and `extra[5]` arrays per shape so HIP graph
//! capture can record stable pointers (see PoC at
//! `crates/hsa-bridge/examples/hip_graph_gemv_poc.rs:524-555`).
//!
//! **PR3:** extend to fused_qkv / fused_qkvza / fused_gate_up.
//! **PR4:** stream-affinity invalidation + LRU bound.

use hip_bridge::{Function, HipResult, HipRuntime, Stream};
use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;

// HIP `extra` mode sentinel constants (from hip_runtime.h):
//   HIP_LAUNCH_PARAM_BUFFER_POINTER = 0x01
//   HIP_LAUNCH_PARAM_BUFFER_SIZE    = 0x02
//   HIP_LAUNCH_PARAM_END            = 0x03
const HIP_LAUNCH_PARAM_BUFFER_POINTER: *mut c_void = 0x01 as *mut c_void;
const HIP_LAUNCH_PARAM_BUFFER_SIZE: *mut c_void = 0x02 as *mut c_void;
const HIP_LAUNCH_PARAM_END: *mut c_void = 0x03 as *mut c_void;

/// Fixed-capacity M-tuple. Matches the worst case (`fused_qkvza_hfq4g256`,
/// 4 entries: qkv_m, z_m, beta_m, alpha_m). Plain GEMV uses 1 slot,
/// fused_qkv 3, fused_gate_up 2.
///
/// Inline storage avoids pulling in `smallvec` for a single use site.
#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub struct MTuple {
    data: [u32; 4],
    len: u8,
}

impl MTuple {
    pub fn from_slice(s: &[u32]) -> Self {
        let mut data = [0u32; 4];
        let len = s.len().min(4);
        data[..len].copy_from_slice(&s[..len]);
        Self { data, len: len as u8 }
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.data[..self.len as usize]
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl fmt::Display for MTuple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, m) in self.as_slice().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{m}")?;
        }
        write!(f, ")")
    }
}

/// Cache key. Block/grid is a pure function of `(kernel, m_tuple, k)`,
/// so these three uniquely identify a hipGraph entry.
///
/// `kernel` is the **family name**, normalized across per-arch variants
/// (e.g. `gemv_hfq4g256`, `gemv_hfq4g256_wide`, `gemv_hfq4g256_multirow_r2`
/// all classify to family `"gemv_hfq4g256"`). The captured graph itself
/// records the function handle, so cache replay is variant-correct
/// without the variant being part of the key — but the function-handle
/// address is captured into `GemvGraphEntry` and used as a tiebreaker
/// to invalidate after recompile (see PR4).
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct GemvShape {
    /// Family name, one of the 6 keys returned by `classify()`.
    pub kernel: &'static str,
    /// Output sizes per kernel family:
    /// - `gemv_hfq4g256`: `[m]`
    /// - `fused_qkv_hfq4g256`: `[q_m, k_m, v_m]`
    /// - `fused_qkvza_hfq4g256`: `[qkv_m, z_m, beta_m, alpha_m]`
    /// - `fused_gate_up_hfq4g256`: `[gate_m, up_m]`
    pub m_tuple: MTuple,
    /// Inner-dim K. Required: a single `gemv_hfq4g256` family runs at
    /// multiple K values within one decode (wo at 256/1024, w_down at 2816,
    /// lm_head at 1024 — see PRD section 1).
    pub k: u32,
}

impl fmt::Display for GemvShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}{} K={}", self.kernel, self.m_tuple, "", self.k)
    }
}

/// Scalar tail in the kernarg blob, used to mutate the right bytes during
/// per-replay rewrite-in-place.
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub enum ScalarTy {
    I32,
    U32,
    I64,
    U64,
    F32,
}

/// Per-shape cache entry.
///
/// **Heap-stable invariant.** All three of `kernarg`, `kernarg_len`, and
/// `extra` are `Box`-allocated so their addresses stay stable for the
/// entry's lifetime. HIP graph capture records the *pointer* to the
/// kernarg buffer, the *pointer* to the kernarg-size cell, and the
/// *pointer* to the `extra[5]` array. All three must outlive the
/// captured graph or replay dereferences stack-gone memory and crashes
/// with `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` (PoC at
/// `crates/hsa-bridge/examples/hip_graph_gemv_poc.rs:524-555`).
///
/// Drop order: `exec` first, then `graph`, then the Boxes.
pub struct GemvGraphEntry {
    /// Instantiated executable graph, ready for replay. Destroyed via
    /// `HipRuntime::graph_exec_destroy` in `GemvGraphCache::clear()`.
    pub(crate) exec: Option<hip_bridge::GraphExec>,
    /// Underlying captured graph, retained so `exec` stays valid.
    pub(crate) graph: Option<hip_bridge::Graph>,
    /// Kernarg blob, heap-stable. Mutated in place per replay.
    pub(crate) kernarg: Box<[u8]>,
    /// Kernarg-size cell, heap-stable. HIP records `&usize` at capture
    /// and dereferences it at every replay — must stay alive even though
    /// no Rust code reads it after construction.
    #[allow(dead_code)]
    pub(crate) kernarg_len: Box<usize>,
    /// 5-slot `extra` array. HIP records the array pointer at capture
    /// and dereferences it at every replay — must stay alive even though
    /// no Rust code reads it after construction.
    /// Layout per `hip_graph_gemv_poc.rs:544-550`:
    ///   [HIP_LAUNCH_PARAM_BUFFER_POINTER,
    ///    kernarg.as_mut_ptr(),
    ///    HIP_LAUNCH_PARAM_BUFFER_SIZE,
    ///    &mut *kernarg_len,
    ///    HIP_LAUNCH_PARAM_END]
    #[allow(dead_code)]
    pub(crate) extra: Box<[*mut c_void; 5]>,
    /// Byte offsets of pointer fields in `kernarg` (3 for plain GEMV).
    pub(crate) ptr_offsets: Vec<usize>,
    /// Byte offsets of scalar fields + their types (m, k for plain GEMV).
    pub(crate) scalar_offsets: Vec<(usize, ScalarTy)>,
    /// Captured `hipFunction_t` address. Recompile invalidation key (PR4).
    #[allow(dead_code)]
    pub(crate) func_handle_addr: usize,
    /// Number of times this entry has been replayed.
    pub(crate) replays: u32,
}

// SAFETY: pointers in `extra` reference `kernarg` / `kernarg_len` Boxes
// owned by this same struct. Same Send pattern as Stream/Graph in the
// hip-bridge — the cache is single-threaded per Gpu (mut-borrow).
unsafe impl Send for GemvGraphEntry {}

impl GemvGraphEntry {
    /// Capture a hipGraph for the given launch shape.
    ///
    /// Box-allocates kernarg / kernarg_len / extra, builds the launch
    /// args, runs `hipStreamBeginCapture` → `hipModuleLaunchKernel` →
    /// `hipStreamEndCapture` → `hipGraphInstantiate`. The first launch
    /// of this graph happens during capture (capture is a real launch
    /// on this stream), so the caller does NOT need to run a sequential
    /// launch after `capture()` returns — the captured side-effect IS
    /// this call's compute.
    ///
    /// Drop order on subsequent invalidate: `exec` first, then `graph`,
    /// then drop the Boxes — see `GemvGraphCache::clear()`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture(
        hip: &HipRuntime,
        func: &Function,
        func_handle_addr: usize,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernarg_bytes: &[u8],
        ptr_offsets: Vec<usize>,
        scalar_offsets: Vec<(usize, ScalarTy)>,
        stream: &Stream,
    ) -> HipResult<Self> {
        // Box-allocate the kernarg blob so its address is stable for the
        // life of the captured graph.
        let mut kernarg: Box<[u8]> = kernarg_bytes.to_vec().into_boxed_slice();
        let mut kernarg_len: Box<usize> = Box::new(kernarg.len());

        // Build the `extra` array referencing the kernarg/kernarg_len
        // Boxes. HIP records this array pointer at capture and reads it
        // at every replay — so the array itself must also be heap-stable.
        let kernarg_ptr = kernarg.as_mut_ptr() as *mut c_void;
        let kernarg_len_ptr = (&mut *kernarg_len) as *mut usize as *mut c_void;
        let extra: Box<[*mut c_void; 5]> = Box::new([
            HIP_LAUNCH_PARAM_BUFFER_POINTER,
            kernarg_ptr,
            HIP_LAUNCH_PARAM_BUFFER_SIZE,
            kernarg_len_ptr,
            HIP_LAUNCH_PARAM_END,
        ]);

        // Begin capture, do the launch (the captured side-effect IS this
        // call's compute), end capture.
        hip.stream_begin_capture(stream, 0)?; // hipStreamCaptureModeGlobal
        let extra_ptr = extra.as_ptr() as *mut *mut c_void;
        // SAFETY: extra is Box-stable for the entry's lifetime; kernarg /
        // kernarg_len likewise. Function and stream are valid for the
        // duration of this call. Layout matches the kernarg ABI for the
        // caller's kernel signature.
        unsafe {
            hip.launch_kernel_with_extra(
                func, grid, block, shared_mem, Some(stream), extra_ptr,
            )?;
        }
        let graph = hip.stream_end_capture(stream)?;
        let exec = hip.graph_instantiate(&graph)?;

        Ok(Self {
            exec: Some(exec),
            graph: Some(graph),
            kernarg,
            kernarg_len,
            extra,
            ptr_offsets,
            scalar_offsets,
            func_handle_addr,
            replays: 0,
        })
    }

    /// Replay the cached graph on `stream` with new pointer / scalar
    /// values rewritten into the kernarg blob.
    ///
    /// `ptrs.len()` must match `self.ptr_offsets.len()`; same for
    /// `scalars` and `self.scalar_offsets`.
    pub(crate) fn replay(
        &mut self,
        hip: &HipRuntime,
        ptrs: &[*mut c_void],
        scalars: &[u64],
        stream: &Stream,
    ) -> HipResult<()> {
        debug_assert_eq!(ptrs.len(), self.ptr_offsets.len(),
            "ptr count mismatch");
        debug_assert_eq!(scalars.len(), self.scalar_offsets.len(),
            "scalar count mismatch");

        // Rewrite pointer slots
        for (i, &off) in self.ptr_offsets.iter().enumerate() {
            let bytes = (ptrs[i] as usize).to_ne_bytes();
            self.kernarg[off..off + 8].copy_from_slice(&bytes);
        }

        // Rewrite scalar slots according to type
        for (i, &(off, ty)) in self.scalar_offsets.iter().enumerate() {
            let v = scalars[i];
            match ty {
                ScalarTy::I32 => {
                    let bytes = (v as i32).to_ne_bytes();
                    self.kernarg[off..off + 4].copy_from_slice(&bytes);
                }
                ScalarTy::U32 => {
                    let bytes = (v as u32).to_ne_bytes();
                    self.kernarg[off..off + 4].copy_from_slice(&bytes);
                }
                ScalarTy::F32 => {
                    let bytes = f32::from_bits(v as u32).to_ne_bytes();
                    self.kernarg[off..off + 4].copy_from_slice(&bytes);
                }
                ScalarTy::I64 => {
                    let bytes = (v as i64).to_ne_bytes();
                    self.kernarg[off..off + 8].copy_from_slice(&bytes);
                }
                ScalarTy::U64 => {
                    let bytes = v.to_ne_bytes();
                    self.kernarg[off..off + 8].copy_from_slice(&bytes);
                }
            }
        }

        let exec = self.exec.as_ref().expect("entry exec dropped");
        hip.graph_launch(exec, stream)?;
        self.replays = self.replays.saturating_add(1);
        Ok(())
    }
}

/// Diagnostic counters. Display impl renders a single-line summary
/// suitable for periodic logging.
#[derive(Default, Clone, Debug)]
pub struct GemvGraphStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub captures: u64,
}

impl fmt::Display for GemvGraphStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            (self.hits as f64) / (total as f64) * 100.0
        } else {
            0.0
        };
        write!(
            f,
            "GemvGraphStats {{ hits: {}, misses: {}, captures: {}, evictions: {}, hit_rate: {:.1}% }}",
            self.hits, self.misses, self.captures, self.evictions, hit_rate
        )
    }
}

/// Per-shape miss counter, used to gate "first miss runs sequential,
/// second miss captures on the next call". Stored per-shape so independent
/// shapes amortize independently.
type MissCounter = u32;

/// Per-`Gpu` per-shape hipGraph cache.
///
/// **Construction is gated by `HIPFIRE_GEMV_GRAPH=1`.** When the env
/// var is unset, `Gpu::gemv_graph_cache` is `None` and the entire cache
/// path is dead — zero overhead in the default config.
///
/// Cache lookups are O(1) HashMap. Per-shape graph capture is O(N_kernel)
/// once per shape (~1-3 ms on RDNA1 per the PoC); replay is the same
/// O(N_kernel) but elides per-launch driver overhead, which is the win.
pub struct GemvGraphCache {
    /// One entry per (kernel family, M-tuple, K) shape.
    pub(crate) entries: HashMap<GemvShape, GemvGraphEntry>,

    /// Per-shape miss counters. Incremented every time a shape is seen
    /// without a captured entry; capture happens once a counter reaches
    /// `min_amortize_replays`.
    pub(crate) miss_counters: HashMap<GemvShape, MissCounter>,

    /// Stream pointer the cache's graphs are bound to. Set on first
    /// capture; if a subsequent dispatch comes in on a different stream,
    /// the entire cache is invalidated (PRD §6 — simpler than per-stream
    /// keying for now).
    pub(crate) captured_stream: Option<*mut c_void>,

    /// Minimum miss count on a shape before we instantiate a graph.
    /// Default 2 (first miss = sequential; counter == 2 → next call
    /// captures; thereafter replays). Configurable via env later.
    pub(crate) min_amortize_replays: u32,

    /// Diagnostic counters.
    pub stats: GemvGraphStats,
}

// SAFETY: `*mut c_void` doesn't impl Send by default, but stream pointers
// are owned-by-value HIP handles and we serialize all access through the
// owning `Gpu` (which is single-threaded per-device). Same pattern as
// `Gpu::active_stream` itself.
unsafe impl Send for GemvGraphCache {}

/// Marker returned by `dispatch()`.
#[derive(Eq, PartialEq, Debug)]
pub enum DispatchOutcome {
    /// Caller must execute the legacy sequential-launch path.
    Fallthrough,
    /// Graph was replayed for this call; caller can skip the sequential
    /// launch. Counts as `stats.hits += 1`.
    Replayed,
    /// Graph was captured for this call. The captured side-effect IS
    /// this call's compute, so the caller MUST NOT also run the legacy
    /// sequential launch — that would double-execute. Counts as
    /// `stats.captures += 1`.
    Captured,
}

impl GemvGraphCache {
    /// Construct a fresh cache. Caller is responsible for the env-var
    /// gate (see `Gpu::init`).
    pub fn new() -> Self {
        Self {
            entries: HashMap::with_capacity(16),
            miss_counters: HashMap::with_capacity(16),
            captured_stream: None,
            min_amortize_replays: 2,
            stats: GemvGraphStats::default(),
        }
    }

    /// Map a kernel function name (as passed to `launch_maybe_blob`)
    /// to the cache's family name. Returns `None` for kernels outside
    /// the 6 GEMV families we cache.
    ///
    /// Variant collapse rules:
    /// - `gemv_hfq4g256` / `_wide` / `_multirow_r{2,4,8}` → `"gemv_hfq4g256"`
    /// - `fused_qkv_hfq4g256` / `_wave64` / `_wave64_dp4a` → `"fused_qkv_hfq4g256"`
    /// - `fused_qkvza_hfq4g256` / `_wave64` / `_wave64_dp4a` → `"fused_qkvza_hfq4g256"`
    /// - `fused_gate_up_hfq4g256` / `_wave64` / `_wave64_dp4a` → `"fused_gate_up_hfq4g256"`
    pub fn family_of(func_name: &str) -> Option<&'static str> {
        if func_name.starts_with("fused_qkv_hfq4g256") {
            Some("fused_qkv_hfq4g256")
        } else if func_name.starts_with("fused_qkvza_hfq4g256") {
            Some("fused_qkvza_hfq4g256")
        } else if func_name.starts_with("fused_gate_up_hfq4g256") {
            Some("fused_gate_up_hfq4g256")
        } else if func_name == "gemv_hfq4g256"
            || func_name == "gemv_hfq4g256_wide"
            || func_name == "gemv_hfq4g256_multirow_r2"
            || func_name == "gemv_hfq4g256_multirow_r4"
            || func_name == "gemv_hfq4g256_multirow_r8"
        {
            Some("gemv_hfq4g256")
        } else {
            None
        }
    }

    /// Build a `GemvShape` from a (func_name, m_tuple, k) triple, or
    /// return `None` for kernels we don't cache.
    pub fn classify(func_name: &str, m_tuple: &[u32], k: u32) -> Option<GemvShape> {
        let family = Self::family_of(func_name)?;
        let expected_arity = match family {
            "gemv_hfq4g256" => 1,
            "fused_gate_up_hfq4g256" => 2,
            "fused_qkv_hfq4g256" => 3,
            "fused_qkvza_hfq4g256" => 4,
            _ => return None,
        };
        if m_tuple.len() != expected_arity {
            return None;
        }
        Some(GemvShape {
            kernel: family,
            m_tuple: MTuple::from_slice(m_tuple),
            k,
        })
    }

    /// Drop and destroy all cached graphs. Idempotent.
    pub fn clear(&mut self, hip: &HipRuntime) {
        let evicted = self.entries.len();
        for (_, mut entry) in self.entries.drain() {
            // Drop order: exec → graph → Boxes (Boxes auto-drop).
            if let Some(exec) = entry.exec.take() {
                let _ = hip.graph_exec_destroy(exec);
            }
            if let Some(graph) = entry.graph.take() {
                let _ = hip.graph_destroy(graph);
            }
        }
        self.miss_counters.clear();
        self.captured_stream = None;
        self.stats.evictions = self.stats.evictions.saturating_add(evicted as u64);
    }

    /// Look up + dispatch.
    ///
    /// **Hit path:** rewrite pointer/scalar slots in the cached kernarg
    /// blob, hipGraphLaunch, return `Replayed`.
    ///
    /// **Miss path:**
    /// 1. If miss-count < `min_amortize_replays`: increment counter,
    ///    return `Fallthrough` so the caller runs the legacy sequential
    ///    launch.
    /// 2. If miss-count >= `min_amortize_replays`: capture the graph
    ///    (the capture runs this call's compute as its side-effect),
    ///    install the entry, return `Captured` so the caller skips the
    ///    legacy launch.
    ///
    /// **Stream affinity:** if `stream_raw` differs from the cache's
    /// `captured_stream`, the entire cache is invalidated and this call
    /// becomes a fresh miss.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch(
        &mut self,
        hip: &HipRuntime,
        shape: &GemvShape,
        func: &Function,
        func_handle_addr: usize,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernarg_bytes: &[u8],
        ptr_offsets: &[usize],
        scalar_offsets: &[(usize, ScalarTy)],
        ptrs: &[*mut c_void],
        scalars: &[u64],
        stream: &Stream,
        stream_raw: *mut c_void,
    ) -> HipResult<DispatchOutcome> {
        // Stream-affinity check — if stream changed, drop everything.
        match self.captured_stream {
            Some(s) if s != stream_raw => {
                // Stream changed under us. Invalidate the whole cache.
                self.clear(hip);
            }
            _ => {}
        }

        // Hit path
        if let Some(entry) = self.entries.get_mut(shape) {
            entry.replay(hip, ptrs, scalars, stream)?;
            self.stats.hits = self.stats.hits.saturating_add(1);
            return Ok(DispatchOutcome::Replayed);
        }

        // Miss path
        self.stats.misses = self.stats.misses.saturating_add(1);
        let counter = self.miss_counters.entry(shape.clone()).or_insert(0);
        *counter = counter.saturating_add(1);
        let counter_val = *counter;

        if counter_val < self.min_amortize_replays {
            // Not yet ready to amortize — caller runs sequential.
            return Ok(DispatchOutcome::Fallthrough);
        }

        // Counter reached threshold: capture on this call. The captured
        // launch IS this call's compute, so the caller MUST NOT also run
        // sequential.
        let entry = GemvGraphEntry::capture(
            hip,
            func,
            func_handle_addr,
            grid,
            block,
            shared_mem,
            kernarg_bytes,
            ptr_offsets.to_vec(),
            scalar_offsets.to_vec(),
            stream,
        )?;
        self.entries.insert(shape.clone(), entry);
        self.miss_counters.remove(shape);
        if self.captured_stream.is_none() {
            self.captured_stream = Some(stream_raw);
        }
        self.stats.captures = self.stats.captures.saturating_add(1);
        Ok(DispatchOutcome::Captured)
    }

    /// Total entry count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for GemvGraphCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_of_collapses_variants() {
        assert_eq!(GemvGraphCache::family_of("gemv_hfq4g256"), Some("gemv_hfq4g256"));
        assert_eq!(GemvGraphCache::family_of("gemv_hfq4g256_wide"), Some("gemv_hfq4g256"));
        assert_eq!(
            GemvGraphCache::family_of("gemv_hfq4g256_multirow_r2"),
            Some("gemv_hfq4g256")
        );
        assert_eq!(
            GemvGraphCache::family_of("gemv_hfq4g256_multirow_r4"),
            Some("gemv_hfq4g256")
        );
        assert_eq!(
            GemvGraphCache::family_of("gemv_hfq4g256_multirow_r8"),
            Some("gemv_hfq4g256")
        );

        assert_eq!(
            GemvGraphCache::family_of("fused_qkv_hfq4g256"),
            Some("fused_qkv_hfq4g256")
        );
        assert_eq!(
            GemvGraphCache::family_of("fused_qkv_hfq4g256_wave64"),
            Some("fused_qkv_hfq4g256")
        );
        assert_eq!(
            GemvGraphCache::family_of("fused_qkv_hfq4g256_wave64_dp4a"),
            Some("fused_qkv_hfq4g256")
        );

        assert_eq!(
            GemvGraphCache::family_of("fused_qkvza_hfq4g256"),
            Some("fused_qkvza_hfq4g256")
        );
        assert_eq!(
            GemvGraphCache::family_of("fused_qkvza_hfq4g256_wave64"),
            Some("fused_qkvza_hfq4g256")
        );
        assert_eq!(
            GemvGraphCache::family_of("fused_qkvza_hfq4g256_wave64_dp4a"),
            Some("fused_qkvza_hfq4g256")
        );

        assert_eq!(
            GemvGraphCache::family_of("fused_gate_up_hfq4g256"),
            Some("fused_gate_up_hfq4g256")
        );
        assert_eq!(
            GemvGraphCache::family_of("fused_gate_up_hfq4g256_wave64"),
            Some("fused_gate_up_hfq4g256")
        );
        assert_eq!(
            GemvGraphCache::family_of("fused_gate_up_hfq4g256_wave64_dp4a"),
            Some("fused_gate_up_hfq4g256")
        );
    }

    #[test]
    fn family_of_rejects_unrelated_kernels() {
        assert_eq!(GemvGraphCache::family_of("rmsnorm"), None);
        assert_eq!(GemvGraphCache::family_of("rope"), None);
        assert_eq!(GemvGraphCache::family_of("flash_attn_fwd"), None);
        assert_eq!(GemvGraphCache::family_of(""), None);
        assert_eq!(GemvGraphCache::family_of("gemv_q4k"), None);
    }

    #[test]
    fn classify_arity_check() {
        // gemv_hfq4g256: arity 1
        assert!(GemvGraphCache::classify("gemv_hfq4g256", &[1024], 256).is_some());
        // wrong arity → None
        assert!(GemvGraphCache::classify("gemv_hfq4g256", &[1024, 256], 256).is_none());

        // fused_qkv: arity 3
        assert!(GemvGraphCache::classify("fused_qkv_hfq4g256", &[1024, 256, 256], 1024).is_some());
        assert!(GemvGraphCache::classify("fused_qkv_hfq4g256", &[1024], 1024).is_none());

        // fused_qkvza: arity 4
        assert!(
            GemvGraphCache::classify("fused_qkvza_hfq4g256", &[768, 256, 32, 32], 1024).is_some()
        );
        assert!(GemvGraphCache::classify("fused_qkvza_hfq4g256", &[768, 256], 1024).is_none());

        // fused_gate_up: arity 2
        assert!(
            GemvGraphCache::classify("fused_gate_up_hfq4g256", &[2816, 2816], 1024).is_some()
        );

        // unrelated kernel → None
        assert!(GemvGraphCache::classify("rmsnorm", &[1024], 1024).is_none());
    }

    #[test]
    fn shape_eq_and_hash() {
        let a = GemvGraphCache::classify("gemv_hfq4g256", &[1024], 256).unwrap();
        let b = GemvGraphCache::classify("gemv_hfq4g256_wide", &[1024], 256).unwrap();
        // Same family, same M, same K → equal shapes (variants collapse).
        assert_eq!(a, b);

        let c = GemvGraphCache::classify("gemv_hfq4g256", &[1024], 1024).unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn stats_display_contains_hit_rate() {
        let mut s = GemvGraphStats::default();
        s.hits = 95;
        s.misses = 5;
        let rendered = format!("{s}");
        assert!(rendered.contains("hits: 95"));
        assert!(rendered.contains("misses: 5"));
        assert!(rendered.contains("hit_rate: 95.0%"));
    }
}
