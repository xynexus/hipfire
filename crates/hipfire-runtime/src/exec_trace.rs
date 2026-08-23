//! The executor trace — M0 of
//! `docs/plans/2026-08-09-v2-daemon-module-major-multistream.md`.
//!
//! Every later stage of the v2 plan exits on a latency measurement, and none of
//! them can be taken from outside the process. Replies to different connections
//! race each other through separate sockets, so client-side arrival order
//! reports the client's thread wake order, not what the daemon chose to run —
//! a trap that has already produced two false failures in this repo's history
//! (see the M4b note in the 2026-07-25 daemon-merge plan). The trace is the
//! ground truth those criteria read.
//!
//! What the daemon had before this was `HIPFIRE_DAEMON_SCHED_DEBUG`, one
//! `eprintln!` per chosen frame, plus `SchedulerStats`, which counts frames.
//! Neither records *time*, so neither can answer "what was the p99 gap between
//! this stream's tokens while a training job ran".
//!
//! ## Shape, and why
//!
//! A fixed-capacity ring of POD records, allocated once at init. The hot path is
//! a timestamp, a lock, and a slot write — **no allocation and no IO**, because
//! this must be cheap enough to leave on. M0's exit criterion holds it to <1%
//! throughput cost, measured A/B within one daemon lifetime.
//!
//! The ring wraps rather than growing: an unbounded trace on a long-running
//! daemon is a memory leak with a friendly name. Wrapping loses the *oldest*
//! records, which is the right end to lose — every measurement here is about
//! recent behaviour — and [`TraceSnapshot::dropped`] reports how many went, so a
//! reader can tell a complete window from a truncated one instead of quietly
//! computing a percentile over a partial sample.
//!
//! ## Why a process global, and why it lives here rather than in the daemon
//!
//! Tokens do not reach the wire from the daemon. They are emitted deep inside
//! `hipfire-serving-core`'s generate paths, and `generate` already takes 28
//! positional parameters — the same argument that made [`crate::cancel`] a
//! global rather than a threaded token.
//!
//! This was found the expensive way. The first version of this module lived in
//! `hipfire-daemon` and hooked `Responder::emit`, on the reasoning that every
//! frame the daemon sends funnels through it. An end-to-end run then recorded
//! 17 records and **zero tokens** for a 256-token generation: `Responder::emit`
//! carries the daemon's *own* frames, while token frames are written straight to
//! the sink by [`crate::super::events::emit_text_bytes`]-shaped helpers in
//! serving-core, which never touch it. Unit tests could not have caught that —
//! they exercised the ring, and the ring was correct. Only a real generation
//! showed the hook was in the wrong place.
//!
//! Hence the ring sits in `hipfire-runtime`, which both the daemon and
//! `hipfire-serving-core` already depend on, and the token hook sits at the
//! actual choke point in serving-core. The arch crates depend on this crate too,
//! which is what M3/M4 will need when modules start recording their own spans.
//!
//! This is not the class of global the v2 plan wants dead. Those
//! (`RAW_OVERRIDE`, the sampler RNG, the steer session) hold **per-stream**
//! state, and a serial executor interleaving streams silently applies one
//! stream's value to another. A trace ring is process-wide observability, like a
//! logger: records carry their stream id as *data*, so interleaving is the thing
//! it is built to record rather than a bug it can suffer.

use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Ring capacity in records, unless `HIPFIRE_DAEMON_TRACE_CAPACITY` overrides.
///
/// 1 MiB at 32 B/record. At M3's per-module rates (a 48-layer march is a few
/// hundred records per token) this holds roughly a minute of single-stream
/// decode, which comfortably covers the 30-second measurement windows M3's
/// procedure calls for.
const DEFAULT_CAPACITY: usize = 32_768;

/// What happened. `u8`-tagged and `Copy` so a record stays POD and the ring is a
/// flat array rather than a structure the writer has to walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TraceEvent {
    /// A frame was taken off the queue and is about to run. `aux` is the queue
    /// depth remaining behind it — the scheduler's contention signal.
    DispatchBegin = 0,
    /// The handler for that frame returned.
    DispatchEnd = 1,
    /// One token frame reached the wire. This is what inter-token gaps are
    /// computed from.
    TokenEmitted = 2,
    /// A generation finished (terminal frame emitted).
    Completed = 3,
    /// VRAM sample. `aux` is bytes *in use* (total - free).
    ///
    /// Sampled per frame rather than per record: it is a driver call, and risk 1
    /// of the v2 plan is a slow leak visible over minutes, not microseconds.
    /// Carrying it in the same artifact as the latency numbers is deliberate —
    /// a paging executor's failure mode reads as "the model got slower" long
    /// before it reads as OOM, and that is only diagnosable if the memory slope
    /// and the latency series share a timeline.
    VramSample = 4,
    // A `Yielded` variant belongs here when M3 gives the executor something to
    // yield *between*. Left out deliberately: today nothing can construct it, and
    // a variant that only ever appears in a `match` arm is API that looks
    // supported without being reachable.
}

/// One trace entry. 32 bytes, `Copy`, no indirection.
#[derive(Clone, Copy, Debug)]
pub struct TraceRecord {
    /// Nanoseconds since the trace was initialised. Monotonic — `Instant`, not
    /// wall clock, so NTP adjustments cannot produce a negative gap.
    pub t_ns: u64,
    pub event: TraceEvent,
    /// Which stream/session this belongs to. `u32::MAX` means "not stream-scoped".
    pub stream: u32,
    /// Which module (M4 onward). 0 until the module graph exists.
    pub module: u32,
    /// Event-specific payload; see each [`TraceEvent`] variant.
    pub aux: u64,
}

/// Not stream-scoped.
pub const NO_STREAM: u32 = u32::MAX;

/// Map a request/session id string onto a numeric stream id.
///
/// FNV-1a, folded to 32 bits. A hash rather than a registry because the
/// alternative is a `HashMap<String, u32>` consulted on every token — an
/// allocation-free lookup is worth more here than perfect injectivity.
///
/// Collisions merge two streams' series, which is precisely the trap
/// [`inter_token_gaps_ns`] exists to avoid, so it is worth being explicit: at
/// 32 bits the birthday bound puts the probability below 1e-5 for a hundred
/// concurrent streams. If a future stage runs enough streams for that to matter,
/// this becomes an index assigned at stream admission — which is where the v2
/// executor will already be minting `StreamId`s anyway.
pub fn stream_id_of(id: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in id.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    // Never collide with the "not stream-scoped" sentinel.
    if hash == NO_STREAM {
        0
    } else {
        hash
    }
}

struct TraceRing {
    slots: Box<[TraceRecord]>,
    /// Total records ever written. The live window is the last
    /// `min(written, capacity)` of them, so this doubles as the drop count.
    written: u64,
    origin: Instant,
}

impl TraceRing {
    fn with_capacity(capacity: usize) -> Self {
        let seed = TraceRecord {
            t_ns: 0,
            event: TraceEvent::DispatchBegin,
            stream: NO_STREAM,
            module: 0,
            aux: 0,
        };
        Self {
            slots: vec![seed; capacity.max(1)].into_boxed_slice(),
            written: 0,
            origin: Instant::now(),
        }
    }

    fn push(&mut self, event: TraceEvent, stream: u32, module: u32, aux: u64) {
        let idx = (self.written as usize) % self.slots.len();
        self.slots[idx] = TraceRecord {
            t_ns: self.origin.elapsed().as_nanos() as u64,
            event,
            stream,
            module,
            aux,
        };
        self.written += 1;
    }

    /// Live records, oldest first.
    fn snapshot(&self) -> TraceSnapshot {
        let capacity = self.slots.len();
        let live = (self.written as usize).min(capacity);
        let start = (self.written as usize).saturating_sub(live);
        let records = (0..live)
            .map(|i| self.slots[(start + i) % capacity])
            .collect();
        TraceSnapshot {
            records,
            dropped: (self.written as usize).saturating_sub(live) as u64,
            capacity: capacity as u64,
        }
    }
}

/// A consistent read of the ring.
pub struct TraceSnapshot {
    pub records: Vec<TraceRecord>,
    /// Records evicted by wraparound. Non-zero means the window is truncated and
    /// any percentile over it describes a partial sample.
    pub dropped: u64,
    pub capacity: u64,
}

static TRACE: OnceLock<Option<Mutex<TraceRing>>> = OnceLock::new();

/// The ring, or `None` when tracing is off.
///
/// Resolved once. `HIPFIRE_DAEMON_TRACE=1` enables; `HIPFIRE_DAEMON_TRACE_CAPACITY`
/// sets the record count. Off by default so the cost is opt-in until M3's
/// measurement says what it actually is.
fn ring() -> Option<&'static Mutex<TraceRing>> {
    TRACE
        .get_or_init(|| {
            if std::env::var("HIPFIRE_DAEMON_TRACE").as_deref() != Ok("1") {
                return None;
            }
            let capacity = std::env::var("HIPFIRE_DAEMON_TRACE_CAPACITY")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(DEFAULT_CAPACITY);
            Some(Mutex::new(TraceRing::with_capacity(capacity)))
        })
        .as_ref()
}

/// Whether tracing is on. Callers use this to skip building `aux` values that
/// would otherwise cost something to compute.
pub fn enabled() -> bool {
    ring().is_some()
}

/// Record one event. A no-op — one `OnceLock` read and a branch — when tracing
/// is off, which is why this can be called unconditionally from the hot path.
pub fn record(event: TraceEvent, stream: u32, module: u32, aux: u64) {
    let Some(ring) = ring() else { return };
    // A poisoned trace lock must never take down the executor: this is
    // observability, and losing a record is strictly better than losing the
    // daemon. Nothing here can leave the ring inconsistent anyway — the only
    // panic-capable step is the slot write, which is infallible.
    if let Ok(mut ring) = ring.lock() {
        ring.push(event, stream, module, aux);
    }
}

/// Records [`TraceEvent::DispatchBegin`] on construction and
/// [`TraceEvent::DispatchEnd`] on drop.
///
/// A guard rather than a matched pair of calls because the executor loop leaves
/// its body through several `continue` paths (malformed frame, unparseable
/// request, empty pick) as well as the normal one. A hand-placed `DispatchEnd`
/// would be forgotten on one of them — and the resulting missing record does not
/// fail loudly, it silently stretches the *next* frame's measured duration to
/// cover both, which is the kind of error that survives review because the
/// numbers still look plausible.
pub struct DispatchGuard {
    queue_depth: u64,
}

impl DispatchGuard {
    pub fn begin(queue_depth: u64) -> Self {
        record(TraceEvent::DispatchBegin, NO_STREAM, 0, queue_depth);
        Self { queue_depth }
    }
}

impl Drop for DispatchGuard {
    fn drop(&mut self) {
        record(TraceEvent::DispatchEnd, NO_STREAM, 0, self.queue_depth);
    }
}

pub fn snapshot() -> Option<TraceSnapshot> {
    let ring = ring()?;
    ring.lock().ok().map(|ring| ring.snapshot())
}

/// Gaps between consecutive [`TraceEvent::TokenEmitted`] records, nanoseconds.
///
/// Restricted to one stream because interleaved streams produce interleaved
/// token events, and differencing across them measures the *executor's*
/// alternation rather than any one stream's latency. Pass [`NO_STREAM`] to take
/// every token event regardless of stream, which is what a single-stream
/// baseline run wants.
pub fn inter_token_gaps_ns(snapshot: &TraceSnapshot, stream: u32) -> Vec<u64> {
    let mut previous: Option<u64> = None;
    let mut gaps = Vec::new();
    for record in &snapshot.records {
        if record.event != TraceEvent::TokenEmitted {
            continue;
        }
        if stream != NO_STREAM && record.stream != stream {
            continue;
        }
        if let Some(previous) = previous {
            gaps.push(record.t_ns.saturating_sub(previous));
        }
        previous = Some(record.t_ns);
    }
    gaps
}

/// `(p50, p99, max)` of `values`, or `None` when empty.
///
/// Nearest-rank on a sorted copy: exact for the sample, no interpolation to
/// argue about, and the sample sizes here (hundreds to thousands) make the sort
/// irrelevant next to a GPU decode step.
pub fn percentiles(values: &[u64]) -> Option<(u64, u64, u64)> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = |q: f64| -> u64 {
        let idx = ((sorted.len() as f64) * q).ceil() as usize;
        sorted[idx.saturating_sub(1).min(sorted.len() - 1)]
    };
    Some((rank(0.50), rank(0.99), sorted[sorted.len() - 1]))
}

/// The `executor_trace` reply body.
///
/// Carries the derived statistics *and* the raw records. The statistics are what
/// the exit criteria assert on; the raw records are what makes a surprising
/// statistic diagnosable rather than merely disputable.
pub fn snapshot_json() -> serde_json::Value {
    let Some(snapshot) = snapshot() else {
        return serde_json::json!({
            "type": "executor_trace",
            "enabled": false,
            "reason": "set HIPFIRE_DAEMON_TRACE=1 before starting the daemon",
        });
    };

    let gaps = inter_token_gaps_ns(&snapshot, NO_STREAM);
    let gap_stats = percentiles(&gaps)
        .map(|(p50, p99, max)| serde_json::json!({ "p50_ns": p50, "p99_ns": p99, "max_ns": max }));

    let span_ns = match (snapshot.records.first(), snapshot.records.last()) {
        (Some(first), Some(last)) => last.t_ns.saturating_sub(first.t_ns),
        _ => 0,
    };

    let records: Vec<serde_json::Value> = snapshot
        .records
        .iter()
        .map(|r| {
            serde_json::json!({
                "t_ns": r.t_ns,
                "event": event_name(r.event),
                "stream": if r.stream == NO_STREAM { serde_json::Value::Null } else { r.stream.into() },
                "module": r.module,
                "aux": r.aux,
            })
        })
        .collect();

    serde_json::json!({
        "type": "executor_trace",
        "enabled": true,
        "capacity": snapshot.capacity,
        "dropped": snapshot.dropped,
        "record_count": records.len(),
        "span_ns": span_ns,
        "token_count": gaps.len() + usize::from(!gaps.is_empty()),
        "inter_token_gap": gap_stats,
        "records": records,
    })
}

fn event_name(event: TraceEvent) -> &'static str {
    match event {
        TraceEvent::DispatchBegin => "dispatch_begin",
        TraceEvent::DispatchEnd => "dispatch_end",
        TraceEvent::TokenEmitted => "token_emitted",
        TraceEvent::Completed => "completed",
        TraceEvent::VramSample => "vram_sample",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ring is tested directly rather than through the global, because the
    /// global resolves its enabled-ness exactly once per process and tests share
    /// one. That is the right behaviour for the daemon (an env read per record
    /// would be absurd) and it simply means the unit tests own a ring.
    fn ring_of(capacity: usize) -> TraceRing {
        TraceRing::with_capacity(capacity)
    }

    #[test]
    fn snapshot_returns_records_oldest_first() {
        let mut ring = ring_of(8);
        for i in 0..5 {
            ring.push(TraceEvent::TokenEmitted, 1, 0, i);
        }
        let snapshot = ring.snapshot();
        assert_eq!(snapshot.records.len(), 5);
        assert_eq!(snapshot.dropped, 0);
        let aux: Vec<u64> = snapshot.records.iter().map(|r| r.aux).collect();
        assert_eq!(aux, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn wraparound_drops_the_oldest_and_says_how_many() {
        // The drop count is the whole point: a percentile over a silently
        // truncated window is a wrong number that looks like a right one.
        let mut ring = ring_of(4);
        for i in 0..10 {
            ring.push(TraceEvent::TokenEmitted, 1, 0, i);
        }
        let snapshot = ring.snapshot();
        assert_eq!(snapshot.records.len(), 4, "window is capped at capacity");
        assert_eq!(snapshot.dropped, 6);
        let aux: Vec<u64> = snapshot.records.iter().map(|r| r.aux).collect();
        assert_eq!(aux, vec![6, 7, 8, 9], "kept the newest, oldest first");
    }

    #[test]
    fn timestamps_are_monotonic_across_a_wrap() {
        let mut ring = ring_of(4);
        for _ in 0..9 {
            ring.push(TraceEvent::DispatchBegin, NO_STREAM, 0, 0);
        }
        let snapshot = ring.snapshot();
        let times: Vec<u64> = snapshot.records.iter().map(|r| r.t_ns).collect();
        assert!(
            times.windows(2).all(|w| w[0] <= w[1]),
            "reordered across the wrap: {times:?}"
        );
    }

    #[test]
    fn n_tokens_yield_n_minus_one_gaps() {
        // This is M0's exit criterion in miniature: a 256-token generation must
        // produce exactly 255 gaps. Off-by-one here would quietly bias every
        // latency percentile the later stages report.
        let mut ring = ring_of(1024);
        for i in 0..256 {
            ring.push(TraceEvent::TokenEmitted, 7, 0, i);
        }
        let snapshot = ring.snapshot();
        assert_eq!(inter_token_gaps_ns(&snapshot, 7).len(), 255);
        assert_eq!(inter_token_gaps_ns(&snapshot, NO_STREAM).len(), 255);
    }

    #[test]
    fn gaps_are_per_stream_not_across_interleaved_streams() {
        // Two streams alternating. Differencing the merged series would measure
        // the executor's alternation rate and report it as each stream's latency
        // — roughly half the true gap, and wrong in the flattering direction.
        let mut ring = ring_of(64);
        for i in 0..6u64 {
            ring.push(TraceEvent::TokenEmitted, (i % 2) as u32, 0, i);
        }
        let snapshot = ring.snapshot();
        assert_eq!(inter_token_gaps_ns(&snapshot, 0).len(), 2);
        assert_eq!(inter_token_gaps_ns(&snapshot, 1).len(), 2);
        assert_eq!(
            inter_token_gaps_ns(&snapshot, NO_STREAM).len(),
            5,
            "the merged series is longer, which is exactly the trap"
        );
    }

    #[test]
    fn non_token_events_do_not_contribute_gaps() {
        let mut ring = ring_of(64);
        ring.push(TraceEvent::TokenEmitted, 1, 0, 0);
        ring.push(TraceEvent::DispatchBegin, 1, 0, 0);
        ring.push(TraceEvent::VramSample, NO_STREAM, 0, 1 << 30);
        ring.push(TraceEvent::TokenEmitted, 1, 0, 0);
        let snapshot = ring.snapshot();
        assert_eq!(inter_token_gaps_ns(&snapshot, 1).len(), 1);
    }

    #[test]
    fn percentiles_are_nearest_rank_and_bounded_by_the_sample() {
        let values: Vec<u64> = (1..=100).collect();
        let (p50, p99, max) = percentiles(&values).expect("non-empty");
        assert_eq!(p50, 50);
        assert_eq!(p99, 99);
        assert_eq!(max, 100);
        assert_eq!(percentiles(&[]), None);
        assert_eq!(percentiles(&[42]), Some((42, 42, 42)));
    }

    #[test]
    fn an_empty_ring_snapshots_cleanly() {
        let snapshot = ring_of(8).snapshot();
        assert!(snapshot.records.is_empty());
        assert_eq!(snapshot.dropped, 0);
        assert!(inter_token_gaps_ns(&snapshot, NO_STREAM).is_empty());
    }
}
