// SPDX-License-Identifier: Apache-2.0
// hipfire-steer — refusal-direction steering / abliteration.
//
// See docs/plans/2026-06-29-refusal-direction-steering.md.
//
// The whole technique reduces to one runtime op on the *residual stream* at the
// **block boundary** (after a transformer block's residual has settled), where
// the residual is uniformly an addressable f32 buffer across every hipfire arch
// — so MoE/attention kernel fusion is irrelevant (we read/inject *after* the
// block, never inside a fused kernel).
//
// Two phases share the same block-boundary hook:
//   * CAPTURE  — read the residual to accumulate per-block means for a +set and
//                a -set, from which a contrastive direction is derived.
//   * APPLY    — mutate the residual with that direction:
//                  Steer (additive):   x += alpha * v
//                  Ablate (projective): x -= lambda * (v . x) * v   (v unit-norm)
//
// Algebraic note: projective ablation of the *activation* equals directional
// ablation of the *weight* (`W·a - λ v (vᵀW·a) = o - λ v (vᵀo)`), so we get
// Heretic-style abliteration with NO weight edit and NO re-quantization.
//
// Phase-1 STUB BOUNDARY:
//   * Session model, control API, capture accumulation, direction derivation,
//     and the pure-Rust apply math are complete and unit-tested.
//   * APPLY currently uses a host round-trip (download → compute → upload) as a
//     correct-but-slow reference path. Replacing it with on-GPU ops
//     (`upload_f32` the direction once + `scaled_add_inplace_gpu_scalar_f32` /
//     a fused projective-subtract kernel) is the first Phase-1 follow-up.
//   * Granularity is block-boundary only (uniform, MoE-agnostic). Per-component
//     (attn-out vs mlp-out) is deferred — see the plan.

use std::cell::RefCell;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

use hip_bridge::HipResult;
use hipfire_rdna::{DType, Gpu, GpuTensor};

pub mod driver;
pub mod lora;

/// How a direction is applied to the residual stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SteerMode {
    /// `x += strength * v` — push the residual along the direction (steering).
    Steer,
    /// `x -= strength * (v·x) * v` — remove the component along the direction
    /// (abliteration). Assumes `v` is unit-norm (the derivation guarantees it).
    Ablate,
}

/// A fully-derived steering configuration ready to apply.
///
/// `directions[layer_idx]` is the unit-norm direction for that block. Blocks
/// outside `layer_range` are left untouched.
#[derive(Clone, Debug)]
pub struct SteerSpec {
    pub directions: Vec<Vec<f32>>,
    pub mode: SteerMode,
    pub strength: f32,
    pub layer_range: Range<usize>,
}

/// Per-block running sum of residuals (host f64 for accumulation precision),
/// used during a CAPTURE session.
struct CaptureAcc {
    /// `sums[layer_idx]` has length `hidden`.
    sums: Vec<Vec<f64>>,
    /// Number of prompts folded in (shared across layers).
    count: u64,
    /// Last residual seen this prompt, per block — overwritten each forward,
    /// folded into `sums` by `commit`.
    current: Vec<Vec<f32>>,
    hidden: usize,
}

impl CaptureAcc {
    fn new(num_layers: usize, hidden: usize) -> Self {
        Self {
            sums: vec![vec![0.0; hidden]; num_layers],
            count: 0,
            current: vec![vec![0.0; hidden]; num_layers],
            hidden,
        }
    }

    /// Record the latest residual at `layer_idx`, overwriting any prior value
    /// this prompt. After a prompt's full prefill, `current[layer]` holds the
    /// LAST prompt-token residual — Heretic's capture position.
    fn observe(&mut self, layer_idx: usize, x: &[f32]) {
        debug_assert_eq!(x.len(), self.hidden);
        self.current[layer_idx].copy_from_slice(x);
    }

    /// Fold the current (last-token) residuals into the running means and count
    /// one prompt. Called by the harness after each prompt's forward.
    fn commit(&mut self) {
        for (sum, cur) in self.sums.iter_mut().zip(self.current.iter()) {
            for (s, &v) in sum.iter_mut().zip(cur.iter()) {
                *s += v as f64;
            }
        }
        self.count += 1;
    }

    /// Per-block means as f32. Panics-free: empty capture yields zeros.
    fn means(&self) -> CaptureMeans {
        let n = self.count.max(1) as f64;
        CaptureMeans(
            self.sums
                .iter()
                .map(|row| row.iter().map(|&s| (s / n) as f32).collect())
                .collect(),
        )
    }
}

/// Per-block mean residual for one prompt set. `0[layer_idx]` has length `hidden`.
#[derive(Clone, Debug)]
pub struct CaptureMeans(pub Vec<Vec<f32>>);

/// One loaded adapter resident in an APPLY session. `directions` are per-block
/// unit vectors; `scale` is the live intensity (the steer `strength`, adjustable
/// via [`set_adapter_scale`]). A `SteerSpec` is the single-adapter case.
#[derive(Clone, Debug)]
struct ResidentAdapter {
    id: String,
    directions: Vec<Vec<f32>>,
    mode: SteerMode,
    scale: f32,
    layer_range: Range<usize>,
}

impl ResidentAdapter {
    /// Whether this adapter mutates block `layer_idx` (in range and not disabled).
    fn touches(&self, layer_idx: usize) -> bool {
        self.scale != 0.0 && self.layer_range.contains(&layer_idx)
    }
}

/// An ordered set of adapters applied at each in-range block boundary. Their
/// deltas SUM (each read from the same pre-apply residual), so the stack is
/// order-independent — see [`apply_stack_host`] / [`apply_on_gpu`].
struct LoraStack {
    adapters: Vec<ResidentAdapter>,
}

enum Session {
    Inactive,
    Capturing(CaptureAcc),
    Applying(LoraStack),
}

/// Which steering session an operation addresses.
///
/// [`SteerKey::default()`] is the **unscoped** session: every function in this
/// crate that does not take a key operates on it, so all existing callers keep
/// exactly today's behaviour. A keyed session is addressed by a stream's wire
/// `session_id` (see the daemon's `SessionKey`), which is what lets two streams
/// decoding in one batched step each carry their own spec.
///
/// Deliberately a plain string rather than the daemon's `StreamId`: this crate
/// sits below the daemon and must not depend on the executor's internal
/// bookkeeping.
#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SteerKey(String);

impl SteerKey {
    /// A session scoped to a stream, named by its wire `session_id`.
    pub fn session(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// True for the unscoped session that the un-keyed API operates on.
    pub fn is_unscoped(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

thread_local! {
    /// The session the CURRENT thread's forward belongs to.
    ///
    /// The forward calls `maybe_steer_block(gpu, x, layer_idx)` from deep inside
    /// a layer loop and has no idea which stream it is serving. Threading a key
    /// explicitly would mean a parameter on `forward_scratch` and its three
    /// siblings — **155 external call sites**, nearly all of which never steer.
    ///
    /// So the key is thread-scoped instead, exactly as `load_progress`'s sink is
    /// (§M1d treats that as a legitimate answer to this same shape of problem).
    /// The justification is the same: the forward runs synchronously on the
    /// calling thread, so "the thread doing the forward" and "the stream being
    /// forwarded" are the same thing, and the executor is serial by design — one
    /// stream holds the GPU per quantum.
    ///
    /// This is scoped ambient state, not a process global: it is installed for a
    /// stream's quantum and restored after, including on panic. The failure the
    /// M1d globals had — a second installer silently redirecting the first
    /// stream's work — cannot happen, because there is no second installer on
    /// this thread while a guard is live.
    static CURRENT_KEY: RefCell<SteerKey> = RefCell::new(SteerKey::default());
}

/// Install the current thread's steering session, restoring the previous one on
/// drop — including on an early return or a panic, so a forward that fails
/// partway cannot leave another stream's key installed for whatever runs next.
pub struct SteerKeyGuard {
    prev: Option<SteerKey>,
    /// Pins the guard to the thread that installed it.
    ///
    /// Without this the guard was `Send` — it wraps only a `String` — and its
    /// `Drop` restores into WHICHEVER thread drops it, not the one that
    /// installed it. A guard moved to a worker, or stored in a struct that
    /// crosses threads, would therefore leave the installing thread pinned to
    /// another stream's key: the same wrong-stream-spec failure the keyed
    /// registry exists to prevent, reached by a different route.
    ///
    /// A raw pointer is `!Send`, so moving the guard across a thread boundary is
    /// now a compile error rather than a silent runtime hazard. `PhantomData`
    /// means it costs nothing at runtime.
    _not_send: std::marker::PhantomData<*const ()>,
}

impl SteerKeyGuard {
    pub fn install(key: SteerKey) -> Self {
        Self {
            prev: Some(CURRENT_KEY.with(|c| std::mem::replace(&mut *c.borrow_mut(), key))),
            _not_send: std::marker::PhantomData,
        }
    }
}

impl Drop for SteerKeyGuard {
    fn drop(&mut self) {
        if let Some(prev) = self.prev.take() {
            CURRENT_KEY.with(|c| *c.borrow_mut() = prev);
        }
    }
}

/// Run `f` against this thread's current key.
///
/// Borrow failure falls back to the unscoped session rather than panicking:
/// steering is an intervention on inference, and it must not be able to take
/// down a forward. Same reasoning as `load_progress::report`'s `try_borrow`.
fn with_current_key<R>(f: impl FnOnce(&SteerKey) -> R) -> R {
    CURRENT_KEY.with(|c| match c.try_borrow() {
        Ok(k) => f(&k),
        Err(_) => f(&SteerKey::default()),
    })
}

/// The key whose session actually applies to this thread's forward.
///
/// This thread's own key when it HAS a session; otherwise the default key.
///
/// The fallback is what makes per-stream steering additive rather than a
/// breaking change. Every steer op registers under the default key unless a
/// request names a session, and the forward path had no key installed at all —
/// so once the daemon started installing a per-stream `SteerKeyGuard`, a
/// process-wide steer op would have stopped applying to every stream at once,
/// silently. With the fallback the semantics are:
///
/// * a stream WITH its own session is steered by that session, and by nothing
///   else — which is the per-stream isolation §M1d is for;
/// * a stream WITHOUT one falls back to the process-wide session, i.e. exactly
///   today's behaviour.
///
/// Costs one extra map lookup, and only while someone is steering: every hot
/// path checks the global `ACTIVE` gate first and returns before reaching here.
fn effective_key() -> SteerKey {
    let own = current_key();
    // ACTIVE, not merely present. `clear_for` leaves an `Inactive` entry behind,
    // so "this key has a map entry" is true for a stream that has explicitly
    // cleared its steering — which must still fall back, not resolve to a
    // session that does nothing.
    if own != SteerKey::default() && is_active_for(&own) {
        own
    } else {
        SteerKey::default()
    }
}

/// This thread's current steering session.
pub fn current_key() -> SteerKey {
    with_current_key(|k| k.clone())
}

type SessionMap = std::collections::HashMap<SteerKey, Session>;

static SESSIONS: OnceLock<RwLock<SessionMap>> = OnceLock::new();
/// Fast-path gate so the hot forward path pays only one relaxed atomic load when
/// steering is inactive (the common case during normal serving).
///
/// Now means "ANY session is active", which keeps the hot path at exactly one
/// relaxed load regardless of how many streams carry specs. A stream with no spec
/// still costs one load plus, only when some other stream is steering, a map
/// lookup — the gate cannot be per-key without the hot path knowing the key
/// before it checks the gate.
static ACTIVE: AtomicBool = AtomicBool::new(false);
/// Bumped on every session change so the per-thread GPU apply cache
/// (`APPLY_CACHE`, which can't live in the `Sync` static because `GpuTensor` is
/// `!Sync`) knows when to refresh its uploaded directions.
///
/// One counter across all keys, which is sound ONLY because `ApplyCache` now
/// also compares its `key`.
///
/// An earlier version of this comment argued the shared counter was the safe
/// direction of error because it could only over-invalidate. That was wrong in
/// the case that matters: EPOCH moves on session MUTATION, and switching which
/// stream is applying mutates nothing, so across keys the shared counter
/// UNDER-invalidated — the cache stayed "valid" and the next stream reused the
/// previous one's uploaded directions. The key comparison in
/// `ensure_apply_cache` is what closes that; the epoch alone does not.
static EPOCH: AtomicU64 = AtomicU64::new(0);

fn sessions() -> &'static RwLock<SessionMap> {
    SESSIONS.get_or_init(|| RwLock::new(SessionMap::new()))
}

/// Run `f` against `key`'s session. `None` when that key has no session.
///
/// Deliberately does NOT create the entry. It used to be
/// `entry(key.clone()).or_insert(Session::Inactive)`, which ran on the per-layer,
/// per-token hot path and therefore (a) heap-cloned the key string on every call
/// and (b) left a permanent map entry for every key ever *queried* — so the
/// registry grew without bound from lookups alone, never mind sessions. A key
/// with no session has nothing to do, which is exactly what `None` says.
///
/// Does not touch the activity gate: callers that only READ or fold state cannot
/// change whether anyone is steering. Callers that can must use
/// [`with_session_changing`], which recomputes it without releasing the lock.
fn with_session<R>(key: &SteerKey, f: impl FnOnce(&mut Session) -> R) -> Option<R> {
    let mut guard = sessions().write().unwrap();
    guard.get_mut(key).map(f)
}

/// Like [`with_session`], but for mutations that can change whether ANY session
/// is active — and it recomputes the gate while still holding the write lock.
///
/// The atomicity is the point. The gate was previously refreshed after the guard
/// dropped, re-acquiring a read lock, so `clear_for(A)` racing `begin_apply_for(B)`
/// could interleave as: A clears, B applies, B computes `any = true`, A computes
/// `any = false`, A stores last. `ACTIVE` then reads false while B is Applying and
/// B's steering silently does nothing — a lost intervention with no error.
fn with_session_changing<R>(key: &SteerKey, f: impl FnOnce(&mut Session) -> R) -> Option<R> {
    let mut guard = sessions().write().unwrap();
    let out = guard.get_mut(key).map(f);
    refresh_active_gate_locked(&guard);
    out
}

/// Read-only view of `key`'s session. `None` when that key has none.
fn read_session<R>(key: &SteerKey, f: impl FnOnce(&Session) -> R) -> Option<R> {
    let guard = sessions().read().unwrap();
    guard.get(key).map(f)
}

/// Recompute the `ACTIVE` gate from the whole map, WHILE THE CALLER HOLDS THE
/// WRITE LOCK.
///
/// Takes the map by reference rather than re-locking, so the mutation, the
/// recomputation and the store are one critical section. Recomputed from the
/// whole map rather than from the mutated entry, so one session going inactive
/// cannot clear the gate while another is still steering.
fn refresh_active_gate_locked(map: &SessionMap) {
    let any = map.values().any(|s| !matches!(s, Session::Inactive));
    EPOCH.fetch_add(1, Ordering::Release);
    ACTIVE.store(any, Ordering::Release);
}

/// Number of live session entries. Test-only: the registry is an implementation
/// detail, but "querying must not create entries" is a property worth asserting.
#[cfg(test)]
pub(crate) fn session_count() -> usize {
    sessions().read().unwrap().len()
}

/// Serializes tests that mutate the process-global session (the apply control API
/// + the driver orchestration test), which would otherwise race under the default
/// parallel test runner.
#[cfg(test)]
pub(crate) static SESSION_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn set_session_for(key: &SteerKey, s: Session) {
    let mut guard = sessions().write().unwrap();
    guard.insert(key.clone(), s);
    // Under the same guard: see `refresh_active_gate_locked`.
    refresh_active_gate_locked(&guard);
}

// ── Control API ─────────────────────────────────────────────────────────────

/// Begin a CAPTURE session: subsequent forwards accumulate per-block residual
/// means. Run the +set, call [`finish_capture`], then the -set similarly.
pub fn begin_capture(num_layers: usize, hidden: usize) {
    begin_capture_for(&SteerKey::default(), num_layers, hidden)
}

/// [`begin_capture`] against a specific session.
pub fn begin_capture_for(key: &SteerKey, num_layers: usize, hidden: usize) {
    set_session_for(key, Session::Capturing(CaptureAcc::new(num_layers, hidden)));
}

/// Fold the current prompt's last-token residuals into the capture means and
/// count it. Call once after each prompt's forward during a CAPTURE session.
pub fn commit_capture() {
    commit_capture_for(&SteerKey::default())
}

/// [`commit_capture`] against a specific session.
pub fn commit_capture_for(key: &SteerKey) {
    with_session(key, |s| {
        if let Session::Capturing(acc) = s {
            acc.commit();
        }
    });
}

/// End a CAPTURE session and return the accumulated per-block means (`None` if
/// no capture was active).
pub fn finish_capture() -> Option<CaptureMeans> {
    finish_capture_for(&SteerKey::default())
}

/// [`finish_capture`] against a specific session.
pub fn finish_capture_for(key: &SteerKey) -> Option<CaptureMeans> {
    let means = read_session(key, |s| match s {
        Session::Capturing(acc) => Some(acc.means()),
        _ => None,
    })
    .flatten();
    if means.is_some() {
        set_session_for(key, Session::Inactive);
    }
    means
}

/// Begin an APPLY session from a single spec: replaces any current stack with one
/// adapter (`id = "default"`, `scale = spec.strength`). The back-compat entry the
/// search loop uses; [`load_adapter`] composes a multi-adapter stack instead.
pub fn begin_apply(spec: SteerSpec) {
    begin_apply_for(&SteerKey::default(), spec)
}

/// [`begin_apply`] against a specific session. This is the entry a per-stream
/// spec uses: two streams may hold different specs simultaneously.
pub fn begin_apply_for(key: &SteerKey, spec: SteerSpec) {
    let adapter = ResidentAdapter {
        id: "default".to_string(),
        directions: spec.directions,
        mode: spec.mode,
        scale: spec.strength,
        layer_range: spec.layer_range,
    };
    set_session_for(
        key,
        Session::Applying(LoraStack {
            adapters: vec![adapter],
        }),
    );
}

/// Push (or replace, by `id`) an adapter onto the APPLY stack, starting a session
/// if none is active. `directions` are per-block unit vectors; `scale` is the live
/// intensity. Stacked adapters' deltas sum at each block boundary.
pub fn load_adapter(
    id: impl Into<String>,
    directions: Vec<Vec<f32>>,
    mode: SteerMode,
    scale: f32,
    layer_range: Range<usize>,
) {
    load_adapter_for(
        &SteerKey::default(),
        id,
        directions,
        mode,
        scale,
        layer_range,
    )
}

/// [`load_adapter`] against a specific session.
#[allow(clippy::too_many_arguments)]
pub fn load_adapter_for(
    key: &SteerKey,
    id: impl Into<String>,
    directions: Vec<Vec<f32>>,
    mode: SteerMode,
    scale: f32,
    layer_range: Range<usize>,
) {
    let adapter = ResidentAdapter {
        id: id.into(),
        directions,
        mode,
        scale,
        layer_range,
    };
    // This one CREATES: its contract is "starting a session if none is active",
    // so it cannot use `with_session`, which deliberately no longer inserts.
    // Creating here is fine — it is a control-plane op, not the hot path, so the
    // key clone and the insert happen once per load rather than per token.
    let mut guard = sessions().write().unwrap();
    match guard.entry(key.clone()).or_insert(Session::Inactive) {
        Session::Applying(stack) => {
            stack.adapters.retain(|a| a.id != adapter.id);
            stack.adapters.push(adapter);
        }
        other => {
            *other = Session::Applying(LoraStack {
                adapters: vec![adapter],
            });
        }
    }
    refresh_active_gate_locked(&guard);
}

/// Materialize and load a rank-1 residual [`lora::LoraAdapter`] (ablate-only) onto
/// the APPLY stack — the bridge from the serialized artifact to the runtime.
pub fn load_lora_adapter(adapter: &lora::LoraAdapter) -> Result<(), String> {
    load_lora_adapter_for(&SteerKey::default(), adapter)
}

/// [`load_lora_adapter`] against a specific session.
pub fn load_lora_adapter_for(key: &SteerKey, adapter: &lora::LoraAdapter) -> Result<(), String> {
    use lora::LoraTarget;
    if adapter.deltas.is_empty() {
        return Err("load_lora_adapter: adapter has no deltas".to_string());
    }
    let layer = |t: &LoraTarget| match t {
        LoraTarget::Residual { layer } => *layer,
    };
    let max_layer = adapter
        .deltas
        .iter()
        .map(|d| layer(&d.target))
        .max()
        .unwrap();
    let min_layer = adapter
        .deltas
        .iter()
        .map(|d| layer(&d.target))
        .min()
        .unwrap();
    let hidden = adapter.meta.hidden;
    let mut directions = vec![vec![0.0f32; hidden]; max_layer + 1];
    for d in &adapter.deltas {
        if d.rank() != 1 {
            return Err(format!(
                "load_lora_adapter: layer {} delta is rank {}, only rank-1 supported",
                layer(&d.target),
                d.rank()
            ));
        }
        // Residual form: A row = vᵀ = the unit direction.
        directions[layer(&d.target)] = d.a[0].clone();
    }
    load_adapter_for(
        key,
        adapter.id.clone(),
        directions,
        SteerMode::Ablate,
        adapter.scale,
        min_layer..max_layer + 1,
    );
    Ok(())
}

/// Set a loaded adapter's live `scale` (intensity). Returns `false` if no adapter
/// with `id` is loaded. Cheap — bumps the epoch so the GPU cache refreshes.
pub fn set_adapter_scale(id: &str, scale: f32) -> bool {
    set_adapter_scale_for(&SteerKey::default(), id, scale)
}

/// [`set_adapter_scale`] against a specific session.
pub fn set_adapter_scale_for(key: &SteerKey, id: &str, scale: f32) -> bool {
    // `_changing` because a scale of 0 disables an adapter; the gate is
    // recomputed under the same lock. `None` = no session for this key.
    with_session_changing(key, |slot| match slot {
        Session::Applying(stack) => stack
            .adapters
            .iter_mut()
            .find(|a| a.id == id)
            .map(|a| a.scale = scale)
            .is_some(),
        _ => false,
    })
    .unwrap_or(false)
}

/// Remove an adapter by `id`. Returns `false` if absent. The session goes
/// `Inactive` when the last adapter is unloaded.
pub fn unload_adapter(id: &str) -> bool {
    unload_adapter_for(&SteerKey::default(), id)
}

/// [`unload_adapter`] against a specific session.
pub fn unload_adapter_for(key: &SteerKey, id: &str) -> bool {
    let (found, _empty) = with_session_changing(key, |slot| match slot {
        Session::Applying(stack) => {
            let before = stack.adapters.len();
            stack.adapters.retain(|a| a.id != id);
            let found = stack.adapters.len() < before;
            let empty = stack.adapters.is_empty();
            if empty {
                *slot = Session::Inactive;
            }
            (found, empty)
        }
        _ => (false, false),
    })
    .unwrap_or((false, false));
    found
}

/// `(id, scale)` for each loaded adapter (apply session only). Drives a
/// `lora_list`-style introspection.
pub fn loaded_adapters() -> Vec<(String, f32)> {
    loaded_adapters_for(&SteerKey::default())
}

/// [`loaded_adapters`] for a specific session. Without this, a keyed session's
/// adapters could not be listed, rescaled or unloaded — `lora_list` reported
/// nothing and the daemon's lora handlers silently addressed the wrong session.
pub fn loaded_adapters_for(key: &SteerKey) -> Vec<(String, f32)> {
    read_session(key, |slot| match slot {
        Session::Applying(stack) => stack
            .adapters
            .iter()
            .map(|a| (a.id.clone(), a.scale))
            .collect(),
        _ => Vec::new(),
    })
    .unwrap_or_default()
}

/// Tear down any active session.
pub fn clear() {
    clear_for(&SteerKey::default())
}

/// [`clear`] for one session. Other sessions are untouched — tearing down one
/// stream's steering must not disarm another's.
pub fn clear_for(key: &SteerKey) {
    set_session_for(key, Session::Inactive);
}

/// Drop every session, keyed and unscoped.
///
/// Wired into `handlers/lifecycle.rs` at both model-swap sites (load and unload).
/// [`clear`] would drop only the unscoped session, so a keyed spec would survive
/// the swap and silently apply to the next model — meaningless against a
/// different model's residual geometry.
pub fn clear_all() {
    let mut guard = sessions().write().unwrap();
    guard.clear();
    refresh_active_gate_locked(&guard);
}

/// Whether a capture or apply session is currently active.
pub fn is_active() -> bool {
    ACTIVE.load(Ordering::Acquire)
}

/// Whether THIS session is active. The un-keyed [`is_active`] stays the hot-path
/// gate ("is anyone steering"); this answers the per-stream question.
pub fn is_active_for(key: &SteerKey) -> bool {
    read_session(key, |s| !matches!(s, Session::Inactive)).unwrap_or(false)
}

/// Whether the session THIS THREAD is currently forwarding is active.
///
/// Use this for any PER-REQUEST decision. [`is_active`] answers "is anyone
/// steering" — correct as a hot-path gate, wrong as a predicate about one
/// request, because once sessions are keyed it is true whenever ANY stream holds
/// a spec. A routing decision built on it would send every unsteered request
/// down whatever path a single steered stream requires.
///
/// Cheap in the common case: the global gate is checked first, so a process with
/// nobody steering pays one atomic load and never touches the thread-local or
/// the map.
pub fn current_is_active() -> bool {
    if !is_active() {
        return false;
    }
    is_active_for(&effective_key())
}

// ── Direction derivation ────────────────────────────────────────────────────

/// Derive per-block unit-norm contrastive directions:
/// `dir_L = normalize(mean_bad_L - mean_good_L)`. When `orthogonalize` is set,
/// the component along the "good" direction is projected out first (projected
/// abliteration), reducing collateral damage to benign behaviour.
pub fn derive_directions(
    good: &CaptureMeans,
    bad: &CaptureMeans,
    orthogonalize: bool,
) -> Vec<Vec<f32>> {
    good.0
        .iter()
        .zip(bad.0.iter())
        .map(|(g, b)| {
            let mut dir: Vec<f32> = b.iter().zip(g.iter()).map(|(&bi, &gi)| bi - gi).collect();
            normalize(&mut dir);
            if orthogonalize {
                let mut good_dir = g.clone();
                normalize(&mut good_dir);
                let proj = dot(&dir, &good_dir);
                for (d, &gd) in dir.iter_mut().zip(good_dir.iter()) {
                    *d -= proj * gd;
                }
                normalize(&mut dir);
            }
            dir
        })
        .collect()
}

// ── Hook entry points (called from arch forwards at the block boundary) ──────

/// Single-vector block-boundary hook for the decode/AR path. `x` is the
/// `[hidden]` residual after block `layer_idx`.
pub fn maybe_steer_block(gpu: &mut Gpu, x: &GpuTensor, layer_idx: usize) -> HipResult<()> {
    // Gate BEFORE touching the thread-local: on the unsteered path this must stay
    // one relaxed atomic load and nothing else. Reading the key first would put a
    // TLS access on every layer of every token for no reason.
    if !is_active() {
        return Ok(());
    }
    maybe_steer_block_for(&effective_key(), gpu, x, layer_idx)
}

/// [`maybe_steer_block`] against a specific session — the per-stream hook.
///
/// The `is_active()` gate stays FIRST and stays global: it answers "is anyone
/// steering" with one relaxed load, so a stream with no spec pays nothing extra
/// while nobody is steering, and pays only a map lookup when someone is.
pub fn maybe_steer_block_for(
    key: &SteerKey,
    gpu: &mut Gpu,
    x: &GpuTensor,
    layer_idx: usize,
) -> HipResult<()> {
    if !is_active() {
        return Ok(());
    }
    let epoch = EPOCH.load(Ordering::Acquire);
    // `None` (no session for this key) is the common case once keys exist and
    // means there is nothing to apply — NOT an error, and no entry is created.
    with_session(key, |slot| -> HipResult<()> {
        match slot {
            Session::Inactive => {}
            Session::Capturing(acc) => {
                let host = gpu.download_f32(x)?;
                acc.observe(layer_idx, &host);
            }
            Session::Applying(stack) => {
                if stack.adapters.iter().any(|a| a.touches(layer_idx)) {
                    apply_on_gpu(gpu, x, layer_idx, stack, epoch, key)?;
                }
            }
        }
        Ok(())
    })
    .unwrap_or(Ok(()))
}

/// Batched block-boundary hook for the prefill path. `x_batch` is the
/// `[num_positions * hidden]` residual after block `layer_idx`.
///
/// Convention (matches Heretic / the plan's open question): CAPTURE folds in the
/// LAST position only (the next-token residual); APPLY mutates ALL positions.
pub fn maybe_steer_block_batched(
    gpu: &mut Gpu,
    x_batch: &GpuTensor,
    layer_idx: usize,
    num_positions: usize,
    hidden: usize,
) -> HipResult<()> {
    if !is_active() {
        return Ok(());
    }
    maybe_steer_block_batched_for(
        &effective_key(),
        gpu,
        x_batch,
        layer_idx,
        num_positions,
        hidden,
    )
}

/// [`maybe_steer_block_batched`] against a specific session.
#[allow(clippy::too_many_arguments)]
pub fn maybe_steer_block_batched_for(
    key: &SteerKey,
    gpu: &mut Gpu,
    x_batch: &GpuTensor,
    layer_idx: usize,
    num_positions: usize,
    hidden: usize,
) -> HipResult<()> {
    if !is_active() {
        return Ok(());
    }
    with_session(key, |slot| -> HipResult<()> {
        match slot {
            Session::Inactive => {}
            Session::Capturing(acc) => {
                let host = gpu.download_f32(x_batch)?;
                let last = (num_positions - 1) * hidden;
                acc.observe(layer_idx, &host[last..last + hidden]);
            }
            Session::Applying(stack) => {
                if stack.adapters.iter().any(|a| a.touches(layer_idx)) {
                    // Prefill is one-shot per request, and the search loop scores via
                    // single-token decode forwards, so this host round-trip is amortized
                    // — the per-token decode path is the one moved on-GPU. The whole
                    // stack is summed per position (read from the pre-apply residual).
                    let mut host = gpu.download_f32(x_batch)?;
                    for p in 0..num_positions {
                        let off = p * hidden;
                        apply_stack_host(&stack.adapters, layer_idx, &mut host[off..off + hidden]);
                    }
                    write_back(gpu, x_batch, &host)?;
                }
            }
        }
        Ok(())
    })
    .unwrap_or(Ok(()))
}

// ── On-GPU apply (decode/AR path) ───────────────────────────────────────────

thread_local! {
    /// Per-thread GPU resources for the apply path. Lives here rather than in the
    /// `Sync` SESSION static because `GpuTensor` is `!Sync`. Refreshed when EPOCH
    /// moves; buffers are reused across epochs when dims match (no per-trial leak).
    static APPLY_CACHE: RefCell<Option<ApplyCache>> = const { RefCell::new(None) };
}

struct ApplyCache {
    /// The session whose directions are currently uploaded.
    ///
    /// Load-bearing. Validity was once (epoch, hidden, adapter count, dirs
    /// count) with no key, and EPOCH moves on session MUTATION, not on a key
    /// switch — so two keyed sessions of identical shape alternating on one
    /// thread both saw `shape_matches && cache.epoch == epoch` and the second
    /// was steered with the FIRST one's directions, scale and mode. Silent, and
    /// exactly the wrong-vector failure the epoch comment claimed to prevent.
    key: SteerKey,
    epoch: u64,
    hidden: usize,
    /// GPU-resident mirror of the stack (per-adapter uploaded directions + scale).
    adapters: Vec<CachedAdapter>,
    /// `[1]` scratch for the ablate dot product `v·x`.
    proj_buf: GpuTensor,
    /// `[1]` scratch for the data-dependent per-adapter coefficient.
    coef_buf: GpuTensor,
}

struct CachedAdapter {
    mode: SteerMode,
    scale: f32,
    layer_range: Range<usize>,
    /// One `[1, hidden]` unit direction per block (2-D so `gemv_f32` reads m=1, k=hidden).
    dirs: Vec<GpuTensor>,
}

/// Apply the whole stack to one `[hidden]` residual fully on-GPU — no full-vector
/// host bounce. Reuses register-tiled `gemv_f32` (dot) +
/// `scaled_add_inplace_gpu_scalar_f32` (axpy). Stacking is the additive sum: every
/// adapter's coefficient is read from the PRE-apply residual (all `gemv` reads run
/// before any `scaled_add` write), so the result is order-independent. For ablate
/// only a 4-byte scalar round-trips per adapter.
fn apply_on_gpu(
    gpu: &mut Gpu,
    x: &GpuTensor,
    layer_idx: usize,
    stack: &LoraStack,
    epoch: u64,
    key: &SteerKey,
) -> HipResult<()> {
    APPLY_CACHE.with(|cell| -> HipResult<()> {
        let mut slot = cell.borrow_mut();
        ensure_apply_cache(&mut slot, gpu, stack, epoch, key)?;
        let cache = slot.as_ref().unwrap();

        // Phase 1 (reads): per-adapter coefficient from the pre-apply residual.
        // `gemv_f32` reads `x` without writing, so collecting all coefficients
        // before any write makes the stack a sum from the original `x`.
        let mut writes: Vec<(usize, f32)> = Vec::new();
        for (ai, a) in cache.adapters.iter().enumerate() {
            if a.scale == 0.0 || !a.layer_range.contains(&layer_idx) {
                continue;
            }
            let coef = match a.mode {
                SteerMode::Steer => a.scale,
                SteerMode::Ablate => {
                    gpu.gemv_f32(&a.dirs[layer_idx], x, &cache.proj_buf)?;
                    let proj = gpu.download_f32(&cache.proj_buf)?[0];
                    -a.scale * proj
                }
            };
            writes.push((ai, coef));
        }

        // Phase 2 (writes): x += Σ coef · v.
        for (ai, coef) in writes {
            let dir = &cache.adapters[ai].dirs[layer_idx];
            gpu.memcpy_htod_auto(&cache.coef_buf.buf, &coef.to_le_bytes())?;
            gpu.scaled_add_inplace_gpu_scalar_f32(x, dir, &cache.coef_buf)?;
        }
        Ok(())
    })
}

/// Build (first use / shape change) or refresh (epoch change) the per-thread GPU
/// apply cache to mirror the resident stack. Direction buffers are reused across
/// epochs when the stack shape is unchanged, so the search loop neither
/// reallocates nor leaks; scales/modes/ranges are cheap host fields refreshed
/// every epoch (a `set_adapter_scale` re-uploads dirs too — fine, small, and
/// avoids tracking a separate scale epoch).
fn ensure_apply_cache(
    slot: &mut Option<ApplyCache>,
    gpu: &mut Gpu,
    stack: &LoraStack,
    epoch: u64,
    key: &SteerKey,
) -> HipResult<()> {
    let hidden = stack
        .adapters
        .first()
        .and_then(|a| a.directions.first())
        .map_or(0, |d| d.len());

    let shape_matches = slot.as_ref().is_some_and(|c| {
        c.hidden == hidden
            && c.adapters.len() == stack.adapters.len()
            && c.adapters
                .iter()
                .zip(stack.adapters.iter())
                .all(|(ca, a)| ca.dirs.len() == a.directions.len())
    });

    if !shape_matches {
        let mut adapters = Vec::with_capacity(stack.adapters.len());
        for a in &stack.adapters {
            let mut dirs = Vec::with_capacity(a.directions.len());
            for d in &a.directions {
                dirs.push(gpu.upload_f32(d, &[1, hidden])?);
            }
            adapters.push(CachedAdapter {
                mode: a.mode,
                scale: a.scale,
                layer_range: a.layer_range.clone(),
                dirs,
            });
        }
        *slot = Some(ApplyCache {
            key: key.clone(),
            epoch,
            hidden,
            adapters,
            proj_buf: gpu.alloc_tensor(&[1], DType::F32)?,
            coef_buf: gpu.alloc_tensor(&[1], DType::F32)?,
        });
        return Ok(());
    }

    let cache = slot.as_mut().unwrap();
    // Key change re-uploads even at an unchanged epoch: switching streams does
    // not mutate any session, so EPOCH does not move, and without this the
    // shape-compatible cache of the PREVIOUS stream would be reused verbatim.
    if cache.epoch != epoch || cache.key != *key {
        for (ca, a) in cache.adapters.iter_mut().zip(stack.adapters.iter()) {
            for (buf, d) in ca.dirs.iter().zip(a.directions.iter()) {
                gpu.memcpy_htod_auto(&buf.buf, &f32_bytes(d))?;
            }
            ca.mode = a.mode;
            ca.scale = a.scale;
            ca.layer_range = a.layer_range.clone();
        }
        cache.epoch = epoch;
        cache.key = key.clone();
    }
    Ok(())
}

/// Host reference for [`apply_on_gpu`]: sum every in-range adapter's delta into
/// `x`, each read from the pre-apply residual (so the stack is order-independent).
/// Drives the batched prefill apply and is the unit-tested correctness anchor.
fn apply_stack_host(adapters: &[ResidentAdapter], layer_idx: usize, x: &mut [f32]) {
    let mut acc = vec![0.0f32; x.len()];
    for a in adapters {
        if !a.touches(layer_idx) {
            continue;
        }
        let v = &a.directions[layer_idx];
        match a.mode {
            SteerMode::Steer => {
                for (ai, &vi) in acc.iter_mut().zip(v.iter()) {
                    *ai += a.scale * vi;
                }
            }
            SteerMode::Ablate => {
                let proj = dot(x, v) * a.scale;
                for (ai, &vi) in acc.iter_mut().zip(v.iter()) {
                    *ai -= proj * vi;
                }
            }
        }
    }
    for (xi, &ai) in x.iter_mut().zip(acc.iter()) {
        *xi += ai;
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// ── Pure math (unit-tested; no GPU) ─────────────────────────────────────────

/// Apply a direction to one residual vector in place.
pub fn apply_direction(x: &mut [f32], v: &[f32], mode: SteerMode, strength: f32) {
    debug_assert_eq!(x.len(), v.len());
    match mode {
        SteerMode::Steer => {
            for (xi, &vi) in x.iter_mut().zip(v.iter()) {
                *xi += strength * vi;
            }
        }
        SteerMode::Ablate => {
            let proj = dot(x, v) * strength;
            for (xi, &vi) in x.iter_mut().zip(v.iter()) {
                *xi -= proj * vi;
            }
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Host → device writeback for the reference apply path.
fn write_back(gpu: &mut Gpu, x: &GpuTensor, host: &[f32]) -> HipResult<()> {
    let bytes: Vec<u8> = host.iter().flat_map(|f| f.to_le_bytes()).collect();
    gpu.memcpy_htod_auto(&x.buf, &bytes)
}

#[cfg(test)]
mod stack_tests {
    use super::*;
    use crate::lora;

    fn spec_of(strength: f32) -> SteerSpec {
        SteerSpec {
            directions: vec![vec![1.0, 0.0, 0.0]],
            mode: SteerMode::Steer,
            strength,
            layer_range: 0..1,
        }
    }

    #[test]
    fn no_guard_means_the_unscoped_session() {
        assert!(
            current_key().is_unscoped(),
            "a thread that never installed a key must behave exactly as before"
        );
    }

    #[test]
    fn the_guard_installs_and_restores() {
        assert!(current_key().is_unscoped());
        {
            let _g = SteerKeyGuard::install(SteerKey::session("outer"));
            assert_eq!(current_key().as_str(), "outer");
            {
                let _inner = SteerKeyGuard::install(SteerKey::session("inner"));
                assert_eq!(current_key().as_str(), "inner");
            }
            assert_eq!(
                current_key().as_str(),
                "outer",
                "the inner guard must restore the outer key, not clear it"
            );
        }
        assert!(
            current_key().is_unscoped(),
            "leaving every guard returns the thread to the unscoped session"
        );
    }

    /// Compile-time proof that a `SteerKeyGuard` cannot cross a thread boundary.
    ///
    /// Two blanket impls, one gated on `Send`. If `SteerKeyGuard` were `Send`
    /// BOTH would apply and the call below would be ambiguous — a compile error.
    /// The fact that this resolves IS the assertion; there is no runtime part.
    ///
    /// It matters because the guard's `Drop` restores into whichever thread drops
    /// it. A `Send` guard moved to a worker would leave the INSTALLING thread
    /// pinned to another stream's key — steering the wrong stream, silently.
    #[test]
    fn the_guard_cannot_cross_a_thread_boundary() {
        trait AmbiguousIfSend<A> {
            fn assert_not_send() {}
        }
        impl<T: ?Sized> AmbiguousIfSend<()> for T {}
        impl<T: ?Sized + Send> AmbiguousIfSend<u8> for T {}
        let _ = <SteerKeyGuard as AmbiguousIfSend<_>>::assert_not_send;
    }

    /// A guard that is dropped by unwinding must still restore, or a forward that
    /// failed partway would leave another stream's key installed for whatever the
    /// thread ran next.
    #[test]
    fn the_guard_restores_on_panic() {
        let _ = std::panic::catch_unwind(|| {
            let _g = SteerKeyGuard::install(SteerKey::session("doomed"));
            assert_eq!(current_key().as_str(), "doomed");
            panic!("forward failed partway");
        });
        assert!(
            current_key().is_unscoped(),
            "unwinding past a guard must not leave its key installed"
        );
    }

    /// The key is per THREAD, so two threads forwarding different streams cannot
    /// see each other's — the property that makes this scoped ambient state
    /// rather than another process global.
    #[test]
    fn current_key_does_not_leak_between_threads() {
        let _g = SteerKeyGuard::install(SteerKey::session("main-thread"));
        let seen = std::thread::spawn(|| {
            let before = current_key();
            let _g = SteerKeyGuard::install(SteerKey::session("other-thread"));
            (before, current_key())
        })
        .join()
        .unwrap();

        assert!(
            seen.0.is_unscoped(),
            "a fresh thread starts unscoped, not on this thread's key"
        );
        assert_eq!(seen.1.as_str(), "other-thread");
        assert_eq!(
            current_key().as_str(),
            "main-thread",
            "the other thread's guard must not disturb this one"
        );
    }

    /// A keyed session's adapters must be listable, rescalable and unloadable.
    ///
    /// Before this, the whole adapter half of the API was hard-wired to the
    /// default key with no `_for` variant, so a keyed session's stack was
    /// invisible: `lora_list` reported nothing for it, and a scale or unload
    /// aimed at it silently addressed the UNSCOPED session instead — mutating a
    /// stack the caller never named.
    #[test]
    fn a_keyed_sessions_adapters_are_addressable() {
        let _lock = SESSION_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_all();
        let k = SteerKey::session("stream-k");
        let v = vec![1.0f32, 0.0, 0.0];

        load_adapter_for(&k, "a1", vec![v.clone()], SteerMode::Steer, 0.5, 0..1);
        load_adapter_for(&k, "a2", vec![v.clone()], SteerMode::Ablate, 1.0, 0..1);

        let listed = loaded_adapters_for(&k);
        assert_eq!(listed.len(), 2, "a keyed session's stack must be listable");
        assert!(
            loaded_adapters().is_empty(),
            "and must NOT appear in the unscoped session"
        );

        assert!(set_adapter_scale_for(&k, "a1", 0.25));
        assert_eq!(
            loaded_adapters_for(&k)
                .iter()
                .find(|(id, _)| id == "a1")
                .map(|(_, s)| *s),
            Some(0.25),
            "rescaling a keyed adapter must take effect on THAT session"
        );
        assert!(
            !set_adapter_scale_for(&SteerKey::default(), "a1", 9.0),
            "the unscoped session does not hold this adapter"
        );

        assert!(unload_adapter_for(&k, "a1"));
        assert_eq!(loaded_adapters_for(&k).len(), 1);
        assert!(unload_adapter_for(&k, "a2"));
        assert!(
            !is_active_for(&k),
            "the session goes Inactive with its last adapter"
        );

        clear_all();
    }

    /// Querying a key must not CREATE it.
    ///
    /// `with_session` used to be `entry(key.clone()).or_insert(Inactive)` on the
    /// per-layer/per-token hot path, so every key ever *looked at* left a
    /// permanent entry — the registry grew from lookups, not from sessions.
    #[test]
    fn querying_an_absent_session_leaves_no_tombstone() {
        let _lock = SESSION_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_all();
        assert_eq!(session_count(), 0);

        for i in 0..50 {
            let k = SteerKey::session(format!("ephemeral-{i}"));
            assert!(!is_active_for(&k));
            assert!(commit_capture_for(&k) == () || true); // exercises with_session
            assert!(finish_capture_for(&k).is_none());
        }
        assert_eq!(
            session_count(),
            0,
            "querying absent keys grew the registry — every lookup left a tombstone"
        );

        // A real session still lands, and clearing it is not a tombstone either.
        let live = SteerKey::session("live");
        begin_apply_for(&live, spec_of(1.0));
        assert_eq!(session_count(), 1);
        clear_all();
        assert_eq!(session_count(), 0);
    }

    /// The gate must stay consistent with the map under concurrent control ops.
    ///
    /// ACTIVE was stored after the write guard dropped, so clear_for(A) racing
    /// begin_apply_for(B) could land a stale `false` while B was still Applying —
    /// B's steering silently a no-op.
    #[test]
    fn the_gate_stays_consistent_under_concurrent_control_ops() {
        let _lock = SESSION_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_all();
        let b = SteerKey::session("B");
        begin_apply_for(&b, spec_of(1.0));

        // Hammer an unrelated session while B stays Applying throughout.
        let t = std::thread::spawn(|| {
            let a = SteerKey::session("A");
            for _ in 0..500 {
                begin_apply_for(&a, spec_of(0.5));
                clear_for(&a);
            }
        });
        for _ in 0..500 {
            assert!(
                is_active(),
                "the gate went false while B was still Applying — a lost intervention"
            );
        }
        t.join().unwrap();

        assert!(is_active_for(&b), "B must still be applying");
        assert!(is_active(), "and the gate must agree with the map");
        clear_all();
        assert!(!is_active());
    }

    /// The routing predicate must be about THIS request, not about the process.
    ///
    /// `is_active()` is true whenever any session holds a spec, so a routing
    /// decision built on it sends unsteered requests down the steered path the
    /// moment one stream is steering. `current_is_active()` is the per-request
    /// question.
    #[test]
    fn current_is_active_is_per_request_not_global() {
        let _lock = SESSION_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_all();

        let a = SteerKey::session("steered");
        begin_apply_for(&a, spec_of(1.0));

        // Global gate: someone is steering.
        assert!(
            is_active(),
            "the hot-path gate sees the process-wide answer"
        );

        // This thread is forwarding an UNSTEERED request.
        assert!(
            !current_is_active(),
            "an unsteered request must not be told it is steering just because \
             another stream is — that is the routing regression"
        );

        // Now this thread is forwarding the steered stream.
        {
            let _g = SteerKeyGuard::install(a.clone());
            assert!(
                current_is_active(),
                "the steered stream must see its own session"
            );
        }
        assert!(
            !current_is_active(),
            "and the answer reverts with the guard"
        );

        clear_all();
        assert!(!current_is_active());
        assert!(!is_active());
    }

    /// The M3 property: two streams hold their own specs simultaneously, and
    /// neither can see or disturb the other's.
    #[test]
    fn keyed_sessions_do_not_alias() {
        let _lock = SESSION_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_all();

        let a = SteerKey::session("stream-a");
        let b = SteerKey::session("stream-b");
        begin_apply_for(&a, spec_of(0.5));
        begin_apply_for(&b, spec_of(2.0));

        assert!(is_active_for(&a) && is_active_for(&b));
        assert!(
            is_active(),
            "the hot-path gate reports that someone is steering"
        );
        assert!(
            !is_active_for(&SteerKey::default()),
            "keyed sessions must not activate the unscoped one"
        );

        // Tearing down one must not disarm the other — the failure mode that
        // makes a single global session unusable under interleaving.
        clear_for(&a);
        assert!(!is_active_for(&a));
        assert!(is_active_for(&b), "clearing stream A disarmed stream B");
        assert!(
            is_active(),
            "the gate must stay set while B is still steering"
        );

        clear_for(&b);
        assert!(!is_active(), "the gate must clear once nobody is steering");
        clear_all();
    }

    /// The un-keyed API is exactly the keyed API on the default key, which is
    /// what makes this migration invisible to every existing caller.
    #[test]
    fn the_unscoped_session_is_the_default_key() {
        let _lock = SESSION_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_all();

        begin_apply(spec_of(1.0));
        assert!(
            is_active_for(&SteerKey::default()),
            "begin_apply targets the default key"
        );
        assert!(is_active());

        let other = SteerKey::session("s1");
        assert!(!is_active_for(&other), "an unrelated stream sees nothing");

        clear();
        assert!(!is_active_for(&SteerKey::default()));
        assert!(!is_active());
        clear_all();
    }

    /// Capture is per-session too: folding one stream's residuals must not land
    /// in another's means.
    #[test]
    fn keyed_capture_accumulates_independently() {
        let _lock = SESSION_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_all();

        let a = SteerKey::session("cap-a");
        let b = SteerKey::session("cap-b");
        begin_capture_for(&a, 1, 3);
        begin_capture_for(&b, 1, 3);

        with_session(&a, |s| {
            if let Session::Capturing(acc) = s {
                acc.observe(0, &[1.0, 2.0, 3.0]);
            }
        });
        commit_capture_for(&a);

        let means_a = finish_capture_for(&a).expect("A captured");
        assert_eq!(means_a.0[0], vec![1.0, 2.0, 3.0]);

        let means_b = finish_capture_for(&b).expect("B had a session");
        assert_eq!(
            means_b.0[0],
            vec![0.0, 0.0, 0.0],
            "stream A's observation must not fold into stream B's means"
        );
        clear_all();
    }

    #[test]
    fn clear_all_drops_every_session() {
        let _lock = SESSION_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_all();
        begin_apply(spec_of(1.0));
        begin_apply_for(&SteerKey::session("x"), spec_of(1.0));
        assert!(is_active());
        clear_all();
        assert!(!is_active(), "model load/unload must leave nothing armed");
        assert!(!is_active_for(&SteerKey::session("x")));
    }

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    fn adapter(id: &str, dir: Vec<f32>, mode: SteerMode, scale: f32) -> ResidentAdapter {
        ResidentAdapter {
            id: id.to_string(),
            directions: vec![dir],
            mode,
            scale,
            layer_range: 0..1,
        }
    }

    #[test]
    fn stack_host_single_ablate_equals_apply_direction() {
        let v = unit(&[0.3, -0.4, 0.5, 0.2]);
        let stack = vec![adapter("a", v.clone(), SteerMode::Ablate, 0.8)];
        let x0 = vec![1.0, 2.0, -1.5, 0.7];
        let mut xs = x0.clone();
        apply_stack_host(&stack, 0, &mut xs);
        let mut xr = x0.clone();
        apply_direction(&mut xr, &v, SteerMode::Ablate, 0.8);
        for (a, b) in xs.iter().zip(&xr) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn stack_host_orthogonal_sum_equals_sequential() {
        let v1 = unit(&[1.0, 0.0, 0.0, 0.0]);
        let v2 = unit(&[0.0, 1.0, 0.0, 0.0]);
        let stack = vec![
            adapter("a", v1.clone(), SteerMode::Ablate, 0.6),
            adapter("b", v2.clone(), SteerMode::Ablate, 0.9),
        ];
        let x0 = vec![2.0, -3.0, 1.0, 0.5];
        let mut xs = x0.clone();
        apply_stack_host(&stack, 0, &mut xs);
        let mut xseq = x0.clone();
        apply_direction(&mut xseq, &v1, SteerMode::Ablate, 0.6);
        apply_direction(&mut xseq, &v2, SteerMode::Ablate, 0.9);
        for (a, b) in xs.iter().zip(&xseq) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn stack_host_skips_zero_scale_and_out_of_range() {
        let v = unit(&[1.0, 1.0, 0.0, 0.0]);
        let mut a = adapter("z", v, SteerMode::Ablate, 0.0);
        let x0 = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = x0.clone();
        apply_stack_host(std::slice::from_ref(&a), 0, &mut x);
        assert_eq!(x, x0); // scale 0 → no-op
        a.scale = 1.0;
        a.layer_range = 1..2;
        apply_stack_host(std::slice::from_ref(&a), 0, &mut x);
        assert_eq!(x, x0); // out of range at layer 0 → no-op
    }

    #[test]
    fn stack_host_matches_lora_module_reference() {
        // The resident-stack apply == the increment-1 LoraAdapter host apply.
        let v = unit(&[0.2, 0.9, -0.1, 0.3]);
        let lad =
            lora::abliteration_adapter("x", &[v.clone()], SteerMode::Ablate, 0.7, 0..1).unwrap();
        let resident = vec![adapter("x", v, SteerMode::Ablate, 0.7)];
        let x0 = vec![1.0, -1.0, 2.0, 0.5];
        let mut xa = x0.clone();
        apply_stack_host(&resident, 0, &mut xa);
        let mut xb = x0.clone();
        lora::apply_residual_stack(std::slice::from_ref(&lad), 0, &mut xb);
        for (a, b) in xa.iter().zip(&xb) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn control_api_load_scale_unload() {
        let _g = crate::SESSION_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        clear();
        assert!(!is_active());
        let v = unit(&[1.0, 0.0, 0.0, 0.0]);
        // begin_apply seeds a single "default" adapter.
        begin_apply(SteerSpec {
            directions: vec![v.clone()],
            mode: SteerMode::Ablate,
            strength: 0.5,
            layer_range: 0..1,
        });
        assert!(is_active());
        assert_eq!(loaded_adapters(), vec![("default".to_string(), 0.5)]);
        // load_adapter stacks a second.
        load_adapter("ablit", vec![v], SteerMode::Ablate, 1.0, 0..1);
        assert_eq!(loaded_adapters().len(), 2);
        // set_adapter_scale dials intensity; unknown id is a no-op.
        assert!(set_adapter_scale("ablit", 0.25));
        assert!(!set_adapter_scale("missing", 1.0));
        let ablit_scale = loaded_adapters()
            .into_iter()
            .find(|(id, _)| id == "ablit")
            .unwrap()
            .1;
        assert_eq!(ablit_scale, 0.25);
        // Unloading one keeps the session active; the last unload clears it.
        assert!(unload_adapter("default"));
        assert!(is_active());
        assert!(unload_adapter("ablit"));
        assert!(!is_active());
        assert!(loaded_adapters().is_empty());
        clear();
    }

    #[test]
    fn load_lora_adapter_populates_stack() {
        let _g = crate::SESSION_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        clear();
        let v = unit(&[0.1, 0.2, 0.3, 0.9]);
        let lad = lora::abliteration_adapter("med", &[v], SteerMode::Ablate, 0.6, 0..1).unwrap();
        load_lora_adapter(&lad).unwrap();
        assert_eq!(loaded_adapters(), vec![("med".to_string(), 0.6)]);
        clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steer_adds_scaled_direction() {
        let mut x = vec![1.0, 2.0, 3.0];
        apply_direction(&mut x, &[1.0, 0.0, 0.0], SteerMode::Steer, 2.0);
        assert_eq!(x, vec![3.0, 2.0, 3.0]);
    }

    #[test]
    fn ablate_removes_component_along_unit_direction() {
        // v is unit-norm along axis 0; full ablation (strength 1) zeros that axis.
        let mut x = vec![5.0, 2.0, 3.0];
        apply_direction(&mut x, &[1.0, 0.0, 0.0], SteerMode::Ablate, 1.0);
        assert!((x[0]).abs() < 1e-6);
        assert_eq!(&x[1..], &[2.0, 3.0]);
    }

    #[test]
    fn derive_is_unit_norm_and_points_bad_minus_good() {
        let good = CaptureMeans(vec![vec![0.0, 0.0]]);
        let bad = CaptureMeans(vec![vec![3.0, 4.0]]);
        let dirs = derive_directions(&good, &bad, false);
        let n = dot(&dirs[0], &dirs[0]).sqrt();
        assert!((n - 1.0).abs() < 1e-6);
        assert!((dirs[0][0] - 0.6).abs() < 1e-6);
        assert!((dirs[0][1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn orthogonalize_removes_good_component() {
        // good points along axis 0; raw refusal dir has a component along it that
        // must be projected out, leaving a pure axis-1 direction.
        let good = CaptureMeans(vec![vec![1.0, 0.0]]);
        let bad = CaptureMeans(vec![vec![1.0, 1.0]]);
        let dirs = derive_directions(&good, &bad, true);
        assert!(dirs[0][0].abs() < 1e-6);
        assert!((dirs[0][1].abs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn capture_means_average_over_prompts() {
        let mut acc = CaptureAcc::new(1, 2);
        acc.observe(0, &[2.0, 4.0]);
        acc.commit();
        acc.observe(0, &[4.0, 8.0]);
        acc.commit();
        let m = acc.means();
        assert_eq!(m.0[0], vec![3.0, 6.0]);
    }
    /// §M1d: a stream with its own session is steered by THAT session, and a
    /// stream without one falls back to the process-wide session.
    ///
    /// The fallback half is what makes per-stream steering additive. Every steer
    /// op registers under the default key unless a request names a session, and
    /// until now the forward path installed no key at all — so the moment the
    /// daemon began installing a per-stream guard, a process-wide steer op would
    /// have stopped applying to every stream at once, silently.
    #[test]
    fn effective_key_prefers_the_streams_own_session_and_falls_back() {
        let _lock = SESSION_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_for(&SteerKey::default());
        let mine = SteerKey::session("stream-a");
        clear_for(&mine);

        // No key installed at all: the unscoped session, as before.
        assert_eq!(effective_key(), SteerKey::default());

        // A key installed but with NO session of its own must fall back, or a
        // process-wide steer op silently stops applying to it.
        let guard = SteerKeyGuard::install(mine.clone());
        assert_eq!(
            effective_key(),
            SteerKey::default(),
            "a stream with no session of its own must fall back to the unscoped one"
        );

        // Once that stream HAS a session, it is the one that applies.
        begin_capture_for(&mine, 1, 4);
        assert_eq!(
            effective_key(),
            mine,
            "a stream with its own session must not be resolved to the unscoped one"
        );

        drop(guard);
        // Off that stream's thread scope, resolution is unscoped again.
        assert_eq!(effective_key(), SteerKey::default());
        clear_for(&mine);
    }
}
