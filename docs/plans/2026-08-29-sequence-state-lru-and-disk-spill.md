# GOAL: bound resident sequence state, then decide whether to spill it to disk

Scoped 2026-08-29 off the leak in #385. Three tracks. **A and B are the fix**;
**C is deferred** and this document exists so the decision to defer it is
reviewable rather than forgotten.

Measured on nix2 / gfx1103 / 30 GB UMA.

---

## The measurement that defines the problem

Every plain `/v1/chat/completions` against a **qwen3.5-family** model (arch 5
and 6) mints a resident session that is never released. `/health`'s
`resident_sessions` tracks the request count exactly, and `MemAvailable` falls
by the session's size each time, until allocation fails and the host is at risk
of OOM-reaping unrelated processes.

Per-session cost is the KV allocation, so it scales with `max_seq` and
`kv_cache`:

| model | arch | KV config | per session |
|---|---|---|---|
| `Qwen3.8-27B--oq4.25++` | 5 (qwen3.5) | `max_seq 8192`, fp32 | **~1156 MB** |
| `Qwen3.6-35B-A3B--oq4.25++` | 6 (qwen3.5-moe) | `max_seq 2048`, kvarn | **~107 MB** |
| `zaya1-8b-native.oq4++` | zaya | — | **0 — no sessions, no drift** |

The 27B exhausts a 21 GB-free host in **two requests**. zaya is unaffected: it
creates no resident sessions at all. This is not an arch-independent leak; it is
confined to the one family that has a state arena
(`SequenceStateArenaBackend::Qwen35Wrapped`, the only implementation — see #383).

## What already exists — do not rebuild it

- **The LRU selection algorithm.** `select_lru_sequence_state_eviction_candidates`
  (`crates/hipfire-state/src/lib.rs:581`) sorts oldest-`last_touched_ms` first,
  tie-breaking on size then handle id/generation, and treats `target_bytes == 0`
  as "select everything". Complete and unit-tested. Its only caller today is a
  `#[test]` at `:2136`.
- **The release primitive.** `SessionServingBackend::release_sessions(&mut self,
  gpu, session_ids) -> Result<usize, String>` (`crates/hipfire-runtime/src/arch.rs:664`),
  already driven end-to-end by a daemon handler
  (`crates/hipfire-daemon/src/handlers/sessions.rs`) and used by the file-batch
  runner (`crates/hipfire-server/src/batch_runner.rs:1173`).
- **Per-session accounting.** `SequenceStatePageDescriptor`
  (`crates/hipfire-state/src/lib.rs:469`) carries `session_id`, `handle`,
  `resident_bytes`, `allocation_epoch`, `owns_pages`, `kind`, `shape`,
  `placement`, `role`. `request_session_count()` and `state_page_descriptors()`
  are on the trait.
- **The policy enums.** `SequenceStateEvictionPolicy::{ManualReleaseOnly,
  LruCheckpoint}` (`:501`) and `SequenceStateSpillTarget::{Disabled, Disk}`
  (`:516`).
- **The knobs.** `resident_state_limit` (8), `resident_checkpoint_max` (4),
  `state_cache_disk`, `disk_spill_allowed`, `disk_spill_min_priority` (128) are
  already parsed into the scheduler's `controls`/`policy`.

The gap is not machinery. It is that `SequenceStateAllocatorPolicy::for_backend`
(`:539`) hardcodes `ManualReleaseOnly` and `Disabled` for **every** backend,
`LruCheckpoint` is never constructed anywhere, and no code consults
`resident_state_limit` to trigger anything.

---

## Track A — release the session a one-shot request created

A request that carries no client `session_id` is a one-shot; the daemon already
treats it as one and refuses to let it inherit the previous conversation
(`crates/hipfire-daemon/src/handlers/generate.rs:271`). Nothing reuses the state
it leaves behind, so it should be released when the request completes, exactly
as the documented protocol chain already prescribes:
`reserve_session_state` → `generate_batch_prefill` →
`generate_batch_decode_step` → `release_sessions`
(`crates/hipfire-daemon/src/queue.rs:11`).

**Precondition, and the reason this is Track A rather than the whole fix:** the
retained state handle advertises `checkpoint_id`, `prefix_hash`,
`cached_prefix_tokens` and `checkpoint_runtime_state: "attachable"`
(`crates/hipfire-generate/src/lib.rs:1153`). If a live path looks a prior
session up by prefix hash to reuse its KV, then these sessions are a deliberate
prefix cache and eager release would delete a feature rather than fix a bug.
Confirm that lookup is reachable in production before landing A; if it is,
A collapses into B and the bound becomes the entire fix.

## Track B — bound resident sessions with an LRU

Four changes, one of which is real work:

1. **Recency does not exist yet.** Descriptors carry `allocation_epoch`
   (creation order), but `SequenceStateEvictionCandidate` needs
   `last_touched_ms`. Without a touch timestamp updated on `activate_session`,
   the result is FIFO, not LRU — which for one-shots is nearly the same thing,
   but is wrong the moment multi-turn sessions interleave. This is the only new
   state the track introduces.
2. **Nothing triggers.** Consult `resident_state_limit` (or a byte budget) after
   a request completes, or before admitting a new session.
3. **Policy.** `for_backend` should return `LruCheckpoint` for backends that
   implement release, keeping `ManualReleaseOnly` for `Unsupported`.
4. **Glue.** Group descriptors by `session_id`, sum `resident_bytes`, build
   candidates, select, call `release_sessions`.

**Never evict** the active session, nor any session with a client-supplied id
that is still in flight — that is the multi-turn correctness boundary.

**Observability is part of the work, not a follow-up.** `state_cache_evictions_total`
is a hardcoded `0` in the health payload (#384), so without wiring it the
feature cannot be observed in production at all.

---

## Track C — spill evicted state to disk (DEFERRED)

Track B drops evicted state; C would write it to disk and restore it on the next
reference, turning an eviction into a cache miss instead of a recompute.

### Why it is deferred

The arithmetic favours spilling **only for sessions that are actually reused**:

- Writing a 27B session (~1156 MB) at the ~0.74 GiB/s streamed during model load
  is ~1.5 s out and a similar cost back.
- Re-prefilling the same 8192-token context at the 25–130 tok/s measured on this
  host is 60–300 s.

So a restore is one to two orders of magnitude cheaper than a recompute — but a
one-shot request is never reused, and one-shots are precisely the traffic that
caused the leak. C buys nothing for the workload that motivated it.

**The reuse rate cannot currently be measured.**
`responses_previous_response_hits` is exactly the counter that would justify C,
and it is one of the hardcoded zeros in #384. Fix that first; the decision then
makes itself from data instead of intuition.

### What C requires beyond B

1. **A serialization format, which does not exist.** Needs GPU→host readback per
   page. The metadata to describe a page is already in
   `SequenceStatePageDescriptor` (`kind`, `shape`, `placement`, `role`,
   `logical_position`), so the descriptor is a viable header; the payload path
   is new.
2. **Versioning, or silent corruption.** A spill file must be keyed by model
   fingerprint, arch id, KV mode and `max_seq`. Restoring a spill taken under
   `kv_cache: fp32` into a `kvarn` runtime would not error — it would produce
   wrong tokens. This is the single highest-risk part of C.
3. **A rehydrate path.** `activate_session` returns a "newly created" bool; it
   needs a third state, restored-from-disk, and every arch implementing
   `SessionServingBackend` has to honour it.
4. **Lifecycle.** Disk quota and spill-file eviction, cleanup on model unload,
   and reaping leftovers from a crashed process.
5. **Policy wiring.** `SequenceStateSpillTarget::Disk` is never constructed;
   `state_cache_disk`, `disk_spill_allowed` and `disk_spill_min_priority` are
   parsed but unused.

### Revisit C when

- `responses_previous_response_hits` is real and shows meaningful session reuse; or
- multi-turn `/v1/responses` traffic becomes a primary workload; or
- a deployment needs more concurrent sessions than host memory holds, where
  spilling is the only way to admit them.

Until one of those is true, B's bound is sufficient and C is carrying cost —
a serialization format and a versioning key — for no measured benefit.
