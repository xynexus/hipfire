# v2 daemon: outstanding changes

Running ledger of what is in flight, what is next, and what was deliberately
deferred. Companion to `2026-08-09-v2-daemon-module-major-multistream.md` (the
destination) and `2026-08-20-v2-prerequisites-autonomous.md` (the queue).

**Keep this current.** It exists because four PRs are open at once with a merge
order that matters, and because several items below were discovered by tripping
over them rather than by reading a plan — those are exactly the ones that get
rediscovered expensively.

Last updated 2026-08-21, against `master` at `7b30c7209`. Six PRs open.

---

## In flight — four open PRs

| PR | What | State |
|---|---|---|
| [#266](https://github.com/xynexus/hipfire/pull/266) | P2/M1 — steer hook on the lowered path, default-off bypass; M2's exit rewritten and satisfied | green, mergeable |
| [#267](https://github.com/xynexus/hipfire/pull/267) | P1 — load-progress sink scoped to the loading thread | green, mergeable |
| [#268](https://github.com/xynexus/hipfire/pull/268) | `RunningStream`, `StreamId`, `SessionKey`, `StreamTable` | green, mergeable |
| [#269](https://github.com/xynexus/hipfire/pull/269) | `hipfire-steer` per-session state (`SteerKey`, keyed registry) | green, mergeable |
| [#270](https://github.com/xynexus/hipfire/pull/270) | this ledger | docs only |
| [#271](https://github.com/xynexus/hipfire/pull/271) | P2/M3 forward-side thread-scoped key — **stacks on #269** | green |

"Green" means every **required** check passes. The `rustfmt (advisory)` job fails
on all four; it fails on `master` too (see *Pre-existing noise*).

### Merge order

`#267` and `#269` are independent — any order, any time.

`#268` is independent to merge.

**Correction (2026-08-21).** This section previously said `#266` must land before
the forward-side hook threading. That was true only of the explicit-parameter
approach, which was measured and rejected: a `SteerKey` parameter on
`forward_scratch` and its three siblings reaches **155 external call sites**,
nearly all of which never steer. The thread-scoped guard that landed instead
(`#271`) changes no forward signature, so it touches no call site in `#266` and
is independent of it.

`#271` stacks on `#269` (it needs `SteerKey`), so merge `#269` first.

---

## Next — what finishes P2/M3

M3's exit: *two streams with different specs decode in one batched step and each
gets its own steering, asserted on output; a third stream with no spec is
unaffected.*

Both remaining steps were blocked on state having nowhere to live. `#268` and
`#269` remove that.

1. ~~**Thread the stream key to the hook.**~~ **DONE — `#271`.** A thread-scoped
   `SteerKeyGuard`, installed for a stream's quantum and restored on drop
   (including on panic). No forward signature changes.

   This entry originally called it "plumbing, not design". **It was design**, and
   the measurement is why: the explicit-parameter alternative reaches 155
   external call sites. The lesson generalises — size the call graph before
   calling a threading change mechanical.
2. **Daemon steer handler accepts `session_id`.** `SteerBeginApply` /
   `SteerBeginCapture` carry no stream identifier today. Add an optional
   `session_id`, resolve it through `StreamTable`, and route to
   `begin_apply_for`. Absent ⇒ the unscoped session, which is exactly today's
   behaviour, so this is backward compatible.

   Do **not** add the protocol field before step 1 lands: a field that is parsed
   and ignored is worse than no field.

3. **M4 — retire the steer escape** in `decode_layers.rs`, and the
   `HIPFIRE_STEER_LOWERED` flag with it. Only after M3. This deletes a fallback,
   which is a stop-line in the prerequisite plan — authorised by the P2
   sequencing decision, but only in this order.

---

## Executor v2 — the rest of parent §M3

`#268` lands the stream object only. Still to build:

- **The march loop** — advance runnable streams a module at a time.
- **Admission** — `PendingQueue` narrows from *running* a `Generate` frame to
  *admitting* a stream and returning. `transport.rs` survives verbatim;
  `batch_runner.rs` is deleted, not moved.
- **The suspension boundary** — park/resume across a real forward, with output
  byte-identical to an uninterrupted run.
- **Remove `#![allow(dead_code)]` from `stream.rs`.** It is there because the
  type landed ahead of its driver. Once the executor exists, a dead item in that
  file is a real signal and the attribute hides it.
- **`MarchFront`** — named in the parent plan (`MarchFront::admit` enforces a
  per-stream bit), does not exist in the tree.

Parent §M3's exit is three measurements read from the M0 trace, not a build
milestone: p99/max module duration and which `SuperOpKind` owns the max; realtime
admission→first-dispatch under saturating bulk load; and bulk throughput loaded
vs solo ≥ 0.6×. The third is what stops the second from being satisfied by
refusing to run the bulk job.

---

## Deferred with cause

**Per-key `EPOCH`, and keying `APPLY_CACHE`.** `#269` leaves one epoch counter
for all keys. That is the safe direction — over-invalidating costs a wasted
upload, whereas a per-key counter that missed a bump would reuse another stream's
uploaded directions and silently steer with the wrong vector. Do both together or
neither.

**`Capturing` cannot be asserted over the daemon protocol.** Only the
prefill-only `steer_capture` op reaches `CaptureAcc::commit`, so a capture session
driven through `generate` returns all-zero means and comparing them is a vacuous
pass. Assert it in-process against `maybe_steer_block` (the `gpu_validate.rs`
route), or add a commit reachable from decode. Related: an active `Capturing`
session currently runs a discarded `download_f32` per layer per decode token.

**P3 — `upload_raw` fixed-frame slabs.** ~757 call sites and a borrow constraint,
not an oversight. Its own project with its own plan; the queue says so
explicitly. The doc-comment warning has landed, which addresses the live risk
(*a new caller reintroduces the leak*) without touching allocation.

**Prefill lowering (parent §M2a).** ~14.2k lines. Not in the prerequisite queue.
It is the critical-path blocker for "the lowered substrate becomes the sole path".

---

## Gaps this session opened, and should close

**No gate covers the hand path.** The tiny gates exercise only the default
(lowered) path and produced byte-identical state hashes before and after the
`ffn_norm` fix that made the hand path coherent again. A cell pinning
`HIPFIRE_FORWARD_LOWERED=0` against the lowered path, for one dense hybrid model,
would close it. That absence is *how* the hand path rotted unobserved — see
`docs/experiments/2026-08-20-qwen35-hand-dense-ffn-norm-fix.md`.

**RoughQuant's mechanism numbers need re-taking.** KLD 0.598 → 0.571, and the
0.598-vs-lowered-0.158 divergence, were all measured through the broken forward.
`docs/roughquant/phase3-real-format-scope.md` now carries the correction; the
numbers themselves have not been re-measured.

**bf16 hand path unverified.** The `ffn_norm` fix was confirmed on two Opus
artifacts (`oq8`, `oq4++`). No qwen3.5 bf16 `.hfq` exists on this box, so the
recorded bf16 symptom (self-KLD 13.89) was never re-measured post-fix. Expected
to be fixed — the defect was dtype-independent — but expected is not measured.

**Daemon-level crosstalk is unasserted.** P1's property is tested on the
*primitive*; "two concurrent daemon loads reach their own clients" is verified by
construction plus a frame-identical real-load check, not end to end.

---

## Pre-existing noise — do not attribute these to new work

Each was confirmed to reproduce on clean `origin/master` with identical numbers.

- **`tiny-quant-gate`**: `qwen3_5_moe/kld:oq8+(calib)` and `oq8++` fail with KLD
  drift `0.005677` vs baseline `0.008147`.
- **`tiny-state-gate`**: `qwen3_5/fp16`, `qwen3_5_moe/fp16`, `qwen3_5_vl/fp16`
  drift, with byte-identical observed hashes across branches.
- **`rustfmt (advisory)`** fails workspace-wide on NINE files: `hip-bridge/src/ffi.rs`,
  `hipfire-arch-qwen35/src/qwen35/layout.rs`, `hipfire-eval`, `hipfire-quantize`,
  `hipfire-runtime/src/{exec_trace,load_progress,sampler}.rs`, plus
  `hipfire-scheduler/src/lib.rs` and `hipfire-serving-core/src/qwen35_decode.rs`.
  (This list previously omitted the last two — a section whose whole job is
  preventing misattribution was itself sending people hunting.)

### Two ways the tiny gate reports a false green

Both cost time this session.

1. It diffs **staged** paths. An unstaged tree reports `no changed paths` and
   exits **0 even under `--require-coverage`** — coverage was silently zero.
2. Piping it to `tail` masks its exit code. It does exit non-zero, and a quant
   failure **short-circuits the state gate**, which then never runs. Run
   `tiny-state-gate.sh` directly when the quant gate is red.

### `docs/env-vars.md` is a papercut

Generated but **untracked**, so it goes stale on every branch switch between a
branch that declares a `HIPFIRE_*` var and one that does not — and a stale copy
fails `no-gpu-ci`. Four false alarms in one session. Either track it, or have the
gate regenerate instead of erroring.

---

## Wording in the plans that does not match the code

Three cases, all found by grepping before building. Worth the habit.

- **`StreamState`** — parent §M1b says "move the state into `StreamState`". No
  such type exists. What landed twice was explicit threading (`RAW_OVERRIDE` → a
  request parameter, `SAMPLER_STATE` → a `SamplerRng` value).
- **"extend `gpu_validate.rs` to the lowered call site"** — there is no separate
  lowered call site to extend to. Both decode paths call the same
  `maybe_steer_block`; they differ in *where* the hook fires, not in the apply
  math. Running the existing oracle is the whole assertion.
- **"fix the hand path = resurrect dead code"** — it was one missing block, not
  dead code. `docs/roughquant/phase3-real-format-scope.md` carries the
  correction.


---

## Found by the pre-merge audit (2026-08-21)

A parallel audit before merging found one blocker and a set of accuracy defects,
most of them in claims these very PRs made. Fixed before merge: the
`ApplyCache`/`SteerKey` aliasing (proven on GPU — without the key check
`gpu_validate` fails 4 cases at 1.08e-1, every failure on the second stream), the
backend trace reporting a predicate rather than the decision, `run()` silently
un-parking, `admit` checking presence rather than liveness, and three classes of
doc that contradicted their own code.

Still open, and each verified:

**~~`is_active()` is a per-request predicate over global state.~~ FIXED (#272)** —
`current_is_active()` answers the per-request question; `is_active()` stays the
hot-path gate. Original description: `decode_layers.rs`
routes with `steer_forces_hand = is_active() && !steer_lowered_enabled()`, while
the keyed registry defines `ACTIVE` as "ANY session is active". Composed, one
stream holding a keyed spec forces EVERY request onto the hand path. Not a
miscompute — both paths are correct since `1f7c2eeba` — but it is a routing
regression the moment keyed sessions exist. It must become per-key, or M4 must
delete the escape first.

**Five hook call sites, not three.** `hipfire-arch-gemma3/src/forward.rs:781` and
`:1050` also call the steer hooks. Earlier text in this file and in the P2 plan
says qwen35-only; the thread-scoped key therefore changes gemma3's routing too,
which no PR mentioned or tested.

**No gate runs the daemon's tests.** `hipfire-daemon` is bin-only (no `src/lib.rs`,
no `[lib]`), so `ci.yml`'s `cargo test --lib --workspace` selects zero targets
from it, and `no-gpu-ci.sh`'s `cargo check --workspace --examples` does not even
type-check `#[cfg(test)]` code. `stream.rs`'s tests — offered as the justification
for its file-scope `#![allow(dead_code)]` — are run by nothing in CI. Add
`hipfire-daemon` (and `hipfire-steer`) to the gate's explicit test list.

**Half the steer API is keyed.** `load_adapter`, `load_lora_adapter`,
`set_adapter_scale`, `unload_adapter` and `loaded_adapters` are hard-wired to the
default key, so a keyed session's adapters cannot be listed, rescaled or unloaded.

**`with_session` takes a write lock and inserts.** It is `entry(key.clone())
.or_insert(...)` under the exclusive lock, per layer per token — a String clone
and a permanent map entry, not the "map lookup" the prose claims. Every distinct
key ever passed leaves a tombstone.

**`ACTIVE` is stored outside the lock.** `refresh_active_gate` drops its read
guard before storing, so `clear_for(A)` racing `begin_apply_for(B)` can land a
stale `false` and silently disable B's steering.

**`SteerKeyGuard` is `Send`.** Its `Drop` restores into whatever thread drops it,
so a guard that crosses threads pins the installing thread to another stream's key.
