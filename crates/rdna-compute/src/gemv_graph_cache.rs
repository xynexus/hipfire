//! Per-(kernel, M-tuple, K) hipGraph cache for decode-path GEMVs.
//!
//! See `docs/plans/gemv-graph-cache.prd` for the full design.
//!
//! **PR1 (this file): skeleton only.** The cache type, classifier, and
//! `HIPFIRE_GEMV_GRAPH=1` env-var gate are wired in but inert — every
//! `dispatch()` call increments `stats.misses` and returns the
//! `Fallthrough` marker so the caller can run the legacy launch path.
//! No graph capture, no graph replay, no kernarg-blob storage yet.
//!
//! **PR2 (next):** capture-on-second-call / replay-on-third-call for
//! `gemv_hfq4g256` (plain) only. The `GemvGraphEntry` cells defined
//! below are placeholders for the kernarg / kernarg-len / extra Boxes
//! that PR2 will allocate per shape. They must be `Box`-allocated so
//! their addresses stay stable across `Vec` re-allocs — see
//! `crates/hsa-bridge/examples/hip_graph_gemv_poc.rs:524-555` for the
//! invariant: HIP graph capture records the *pointer* to the kernarg
//! buffer, the *pointer* to the kernarg-size cell, and the *pointer*
//! to the `extra[5]` array. All three must outlive the captured graph.
//!
//! **PR3:** extend to fused_qkv / fused_qkvza / fused_gate_up.
//! **PR4:** stream-affinity invalidation + LRU bound.

use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;

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

/// Scalar tail in the kernarg blob, used by PR2 to mutate the right
/// bytes during per-replay rewrite-in-place.
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
#[allow(dead_code)] // PR1: defined for PR2's offset table.
pub enum ScalarTy {
    I32,
    U32,
    I64,
    U64,
    F32,
}

/// Per-shape cache entry.
///
/// **PR1: empty placeholder.** PR2 will populate:
///
/// - `kernarg: Box<[u8]>` — the kernarg blob, mutated in place per replay.
///   `Box`-allocated so its address is stable for the entry's lifetime
///   (HIP graph capture records this pointer; a Vec resize would dangle it).
/// - `kernarg_len: Box<usize>` — HIP records `&usize` via the
///   `HIP_LAUNCH_PARAM_BUFFER_SIZE` extra slot. Must be heap-stable.
/// - `extra: Box<[*mut c_void; 5]>` — the `extra` array passed to
///   `hipModuleLaunchKernel`. HIP records this array pointer at capture
///   and dereferences it at every replay.
/// - `exec` / `graph` — the instantiated `hipGraphExec_t` and the
///   underlying `hipGraph_t`. Drop order matters: `exec` first, then
///   `graph`, then the Boxes above.
/// - `ptr_offsets` — byte offsets in `kernarg` of device-pointer slots
///   (3 slots for plain GEMV: a, x, y; 7 slots for fused_qkv).
/// - `scalar_offsets` — byte offsets and types of scalar slots (m, k for
///   plain GEMV; q_m, k_m, v_m, k for fused_qkv; etc.).
/// - `func_handle_addr: usize` — captured `hipFunction_t` address. On
///   recompile (kernel cache invalidation), the address changes and
///   the entry is invalidated. PR4 wires this in.
#[allow(dead_code)] // PR1: fields exist as type carriers; populated in PR2.
pub struct GemvGraphEntry {
    /// PR2: `Some(hip_bridge::GraphExec)` after instantiation.
    /// Boxed-Option keeps the size stable; can switch to plain `Option<GraphExec>`
    /// once PR2 lands the type.
    pub(crate) exec: Option<()>,
    /// PR2: `Some(hip_bridge::Graph)`.
    pub(crate) graph: Option<()>,
    /// PR2: kernarg blob, heap-stable. Will be `Some(Box<[u8]>)`.
    pub(crate) kernarg: Option<Box<[u8]>>,
    /// PR2: kernarg-size cell, heap-stable. HIP records `&usize`.
    pub(crate) kernarg_len: Option<Box<usize>>,
    /// PR2: 5-slot extra array. HIP records the array pointer at capture.
    /// Layout per `hip_graph_gemv_poc.rs:544-550`:
    ///   [HIP_LAUNCH_PARAM_BUFFER_POINTER,
    ///    kernarg.as_mut_ptr(),
    ///    HIP_LAUNCH_PARAM_BUFFER_SIZE,
    ///    &mut *kernarg_len,
    ///    HIP_LAUNCH_PARAM_END]
    pub(crate) extra: Option<Box<[*mut c_void; 5]>>,
    /// PR2: byte offsets of pointer fields in `kernarg`.
    pub(crate) ptr_offsets: Vec<usize>,
    /// PR2: byte offsets of scalar fields + their types.
    pub(crate) scalar_offsets: Vec<(usize, ScalarTy)>,
    /// PR4: captured `hipFunction_t` address. Recompile invalidation key.
    pub(crate) func_handle_addr: usize,
    /// Number of times this entry has been replayed. PR2 uses this to
    /// gate "first miss runs sequential, second miss captures, third+
    /// replays" — so an entry that's only seen 1-2 calls won't pay
    /// instantiation cost yet.
    pub(crate) replays: u32,
}

impl GemvGraphEntry {
    /// PR1: empty placeholder. PR2 will rename to `new_uninstantiated`
    /// or merge into a `capture_for_shape` constructor.
    #[allow(dead_code)]
    pub(crate) fn empty() -> Self {
        Self {
            exec: None,
            graph: None,
            kernarg: None,
            kernarg_len: None,
            extra: None,
            ptr_offsets: Vec::new(),
            scalar_offsets: Vec::new(),
            func_handle_addr: 0,
            replays: 0,
        }
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
    /// One entry per (kernel family, M-tuple, K) shape. Sized to the
    /// cap at ~7 shapes for Qwen3.5 0.8B (PRD section 1) — initial
    /// capacity 16 covers 0.8B and 9B with headroom.
    pub(crate) entries: HashMap<GemvShape, GemvGraphEntry>,

    /// Stream pointer the cache's graphs are bound to. PR4 will key
    /// off this for invalidate-on-stream-change. PR1 records None
    /// because no captures have happened yet.
    #[allow(dead_code)] // PR1: read by PR2's stream-affinity check.
    pub(crate) captured_stream: Option<*mut c_void>,

    /// Minimum miss count on a shape before we bother instantiating
    /// a graph for it. PR2 default is 2 (first call sequential, second
    /// captures, third+ replays). Configurable via env in PR4.
    #[allow(dead_code)] // PR1: read by PR2's dispatch.
    pub(crate) min_amortize_replays: u32,

    /// Diagnostic counters.
    pub stats: GemvGraphStats,
}

// SAFETY: `*mut c_void` doesn't impl Send by default, but stream pointers
// are owned-by-value HIP handles and we serialize all access through the
// owning `Gpu` (which is single-threaded per-device). Same pattern as
// `Gpu::active_stream` itself.
unsafe impl Send for GemvGraphCache {}

/// Marker returned by `dispatch()`. PR1 always returns `Fallthrough`;
/// the caller treats this as "do the legacy launch path".
#[derive(Eq, PartialEq, Debug)]
pub enum DispatchOutcome {
    /// Caller must execute the legacy sequential-launch path. PR1 always
    /// returns this. PR2 returns this on first/second observed call to a
    /// new shape (capture happens lazily) and on cache miss after eviction.
    Fallthrough,
    /// PR2: graph was replayed for this call; caller can skip the
    /// sequential launch. Always counts as `stats.hits += 1`.
    #[allow(dead_code)]
    Replayed,
}

impl GemvGraphCache {
    /// Construct a fresh cache. Caller is responsible for the env-var
    /// gate (see `Gpu::init`).
    pub fn new() -> Self {
        Self {
            entries: HashMap::with_capacity(16),
            captured_stream: None,
            // PR2: first call runs sequential (warmup), second captures,
            // third+ replays. So we need at least 2 misses on a shape
            // before instantiation pays off.
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
    ///
    /// Per-arch variants are collapsed because their dispatch shapes are
    /// pure functions of `(family, m_tuple, k)`. The captured graph
    /// records the actual `hipFunction_t`, so replay routes to the right
    /// variant; only the *cache key* needs to be variant-agnostic.
    pub fn family_of(func_name: &str) -> Option<&'static str> {
        // Order matters: fused_* must match before the bare gemv_hfq4g256
        // family (which is a substring of none of them, but be defensive).
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
    ///
    /// PR1 caller (in `launch_maybe_blob`) passes a best-effort
    /// `m_tuple` derived from the launch grid; for plain GEMV variants
    /// the `m_tuple` is `[m]` and the K dimension isn't directly
    /// recoverable from grid+block alone. PR2 plumbs M and K through
    /// from the per-kernel dispatch sites (`gemv_hfq4g256`,
    /// `fused_qkv_hfq4g256`, etc.) where both are available.
    pub fn classify(func_name: &str, m_tuple: &[u32], k: u32) -> Option<GemvShape> {
        let family = Self::family_of(func_name)?;
        // M-tuple arity must match the family. PR1 tolerates a slack-1
        // best-effort tuple from launch_maybe_blob (always 1 element);
        // PR2's per-kernel call sites provide the full tuple.
        let expected_arity = match family {
            "gemv_hfq4g256" => 1,
            "fused_gate_up_hfq4g256" => 2,
            "fused_qkv_hfq4g256" => 3,
            "fused_qkvza_hfq4g256" => 4,
            _ => return None,
        };
        // Reject obviously-wrong arity. For PR1's launch_maybe_blob
        // best-effort plumbing (always 1-elem tuple), only plain
        // gemv_hfq4g256 will pass; the fused families fall through
        // to legacy until PR2 plumbs full tuples. This is intentional.
        if m_tuple.len() != expected_arity {
            return None;
        }
        Some(GemvShape {
            kernel: family,
            m_tuple: MTuple::from_slice(m_tuple),
            k,
        })
    }

    /// PR1: always falls through to legacy launch.
    ///
    /// Increments `stats.misses` (every call is a miss in PR1) and
    /// returns `DispatchOutcome::Fallthrough`. The shape parameter is
    /// kept in the signature so the `Gpu::launch_maybe_blob` plug-in
    /// is symbol-stable across PR1→PR2.
    ///
    /// PR2 will:
    /// 1. Look up `shape` in `entries`.
    /// 2. On hit: rewrite ptr/scalar slots in place from `params`,
    ///    `hipGraphLaunch(exec, stream)`, return `Replayed`.
    /// 3. On miss with `replays >= min_amortize_replays`: begin_capture,
    ///    sequential launch, end_capture, instantiate, store entry,
    ///    return `Replayed` (the captured launch IS this call).
    /// 4. On miss with `replays < min_amortize_replays`: increment
    ///    `replays`, return `Fallthrough`.
    #[allow(dead_code)] // PR1: cache is gated and dispatch isn't called yet.
    pub fn dispatch(&mut self, shape: &GemvShape) -> DispatchOutcome {
        // Touch the entries map so PR2's lookup path is alive in PR1
        // for unit-test reachability, but never act on the result.
        let _ = self.entries.get(shape);
        self.stats.misses = self.stats.misses.saturating_add(1);
        DispatchOutcome::Fallthrough
    }

    /// Total entry count. Useful for diagnostics + the LRU bound in PR4.
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
    fn dispatch_pr1_always_falls_through() {
        let mut cache = GemvGraphCache::new();
        let shape = GemvGraphCache::classify("gemv_hfq4g256", &[1024], 256).unwrap();
        for _ in 0..10 {
            assert_eq!(cache.dispatch(&shape), DispatchOutcome::Fallthrough);
        }
        assert_eq!(cache.stats.misses, 10);
        assert_eq!(cache.stats.hits, 0);
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
