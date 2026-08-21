# v2 daemon: outstanding changes

Running ledger of what is in flight, what is next, and what was deliberately
deferred. Companion to `2026-08-09-v2-daemon-module-major-multistream.md` (the
destination) and `2026-08-20-v2-prerequisites-autonomous.md` (the queue).

**Keep this current.** It exists because four PRs are open at once with a merge
order that matters, and because several items below were discovered by tripping
over them rather than by reading a plan — those are exactly the ones that get
rediscovered expensively.

Last updated 2026-08-21, against `master` at `5bc7fa788`. Three PRs open.

---

## In flight — three open PRs

| PR | What | Base | State |
|---|---|---|---|
| [#281](https://github.com/xynexus/hipfire/pull/281) | executor v2 **M3a** — a `Generate` frame admits a `RunningStream` | `master` | green |
| [#282](https://github.com/xynexus/hipfire/pull/282) | eleven plan corrections for M3b/M3c/M3d — **stacks on #281** | `#281` | docs only |
| [#283](https://github.com/xynexus/hipfire/pull/283) | **M3b0** — `qwen35_decode_one`, the quantum M3b marches | `master` | green |

"Green" means every **required** check passes. `rustfmt (advisory)` fails on all
of them; it fails on `master` too (see *Pre-existing noise*).

### Merge order

`#281` and `#283` are independent — any order, any time. `#283` touches only
`hipfire-serving-core`; `#281` only `hipfire-daemon`.

`#282` stacks on `#281` (it is based on that branch), so merge `#281` first.

**The previous contents of this table were stale, and that is the failure this
section exists to prevent.** It listed `#266`–`#271` as open with a merge order
that mattered; all seven of `#266`–`#272` had long since merged. A stale
in-flight table is worse than no table — it is read as current by exactly the
person who has not been following along. **Re-check `gh pr list` before trusting
this section**, and update it when you open or merge anything.

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

- **The march loop** — advance runnable streams a module at a time. Its
  prerequisite, **M3b0, is in review as #283**: there was no quantum to march.
  Both per-token loops held every cross-token variable in stack locals, so a
  march loop built first would have marched whole requests. `qwen35_decode_one`
  makes advancing a token a call. The loop itself is the small part
  (~150–250 lines); see the M3b section of the march-loop plan for the four
  things that must move with it.
- ~~**Admission**~~ **DONE — M3a.** A `Generate` frame admits a `RunningStream`
  into a `StreamTable` on `DaemonState` and is retired out of it at the dispatch
  site. `transport.rs` untouched, as the plan requires.

  The seam is the **dispatch site** (`main.rs`'s `DaemonRequest::Generate` arm),
  not inside `handlers::generate::text`. That handler has many early returns, and
  a retire per exit path is the leak shape `ThreadSinkGuard` exists to prevent —
  measured: the two frames that early-return on an unsupported `session_id`
  retire correctly, and the table reads 0 after every request including those.

  Two things this does NOT do. Admission is **unconditional**, not flag-gated:
  gating it would leave the shape untested by every default run, and the flag's
  job is selecting who *runs* an admitted stream. And `AdmitError` is currently
  **unreachable** — the daemon is serial and retires before the next frame, so no
  session can hold a live stream across requests until the march loop lands. Its
  caller runs the frame anyway rather than refusing, because a refusal would be
  user-visible.

  `batch_runner.rs` is in `hipfire-server`, not the daemon; not touched. See the
  correction in the M3 plan.
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

## The trace and the table partition by different things (live, from M3a)

The M3 plan flagged this as "reconcile explicitly, do not assume they agree."
They do not. `exec_trace::stream_id_of` derives a trace stream from the **request
id**; admission keys on the **session id, falling back to the request id**. For
every frame that carries an explicit `session_id`, the two disagree — the trace
splits one conversation across a stream per turn.

Harmless today (nothing reads the table yet), and it is M3d that pays: its exit
is three numbers read off that trace, per stream. Reconcile before measuring, or
the numbers partition by turn rather than by stream.

## Wording in the plans that does not match the code

Three cases below, all found by grepping before building — plus **eleven more**
found by a recon pass after M3a landed, now corrected in place in
`2026-08-21-executor-v2-march-loop.md`. The load-bearing ones, so they are not
rediscovered from the ledger alone:

* **"three per-token hook sites"** — there are **six**, across two primitives:
  id-keyed `cancel::is_cancelled` (`arch.rs:1206`, `generate_arch.rs:1073`,
  `events.rs:174`) and unkeyed `take_generation_cancel()` (`arch.rs:1215`,
  `generate.rs:1557`, `generate.rs:3132`). The unkeyed one is a process-global
  `AtomicBool` whose take is a consuming `swap(false)`, so two marching streams
  race for one cancel. And the pick step does not hold the key it is stored
  under — `PENDING` is keyed by request id, `RunningStream` carries a
  `SessionKey`.
* **"`hipGraph` capture is off on the v2 path"** — it is default **ON**
  (deepseek4's whole decode body, plus qwen35's DFlash verify graph). Turning it
  off is an action item across three capture families, not an observation.
* **"the executor may march whatever quanta exist today"** contradicts the same
  plan's exit ("interleave at quantum granularity"). On the `Generate` path the
  only quantum that exists is the whole request: both per-token loops
  (`generate.rs:3124`, `arch.rs:1192`) hold every cross-token variable in stack
  locals and are not re-enterable. This is M3b's real cost, and the plan costed
  only the loop.
* **"each stage lands behind `HIPFIRE_DAEMON_EXECUTOR=v2`"** — no stage does.
  `executor_v2_enabled()` has exactly one caller in the workspace and its whole
  effect is a one-shot `tracing::warn!`; M3a admits, runs inline and retires
  unconditionally, by design. M3b must *build* the selection, not inherit it.
* **M3d's first measurement is not obtainable** from the M0 trace:
  `TraceRecord.module` is documented as "0 until the module graph exists", every
  `record()` site passes 0, and `SuperOpKind` appears in `exec_trace.rs` zero
  times.
* **`--listen` is never started by anything in-tree.** It exists only in
  `main.rs` (help text, parser, dispatch); `scripts/` and `tests/` return zero,
  and `daemon-adapter`'s `attach_or_spawn` always falls through to a stdio spawn.
  So "two concurrent `Generate` requests" has no harness today.

**Not a plan doc, same hazard class**, found while checking the quantum:
`hipfire-dispatch/src/pipeline/superop.rs` asserts in its own doc comment that
"qwen35, qwen2, deepseek4, minimax and lfm2moe all build a `LoweredForward` at
load". Grepping for that type returns `hipfire-dispatch`, `hipfire-serving-core`
and `hipfire-arch-gemma4` — **zero** in any of the five arches it names. An
in-tree doc that is factually false about five arches, and load-bearing for
anyone reasoning about module granularity.

The original three:

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

## Next: executor v2 (parent §M3)

Planned in `2026-08-21-executor-v2-march-loop.md`. Staged M3a admit → M3b march →
M3c suspend → M3d measure, all behind `HIPFIRE_DAEMON_EXECUTOR=v2`, default off.
M3a alone unblocks half of P2/M3 below.

## M3 is BLOCKED on two prerequisites (found 2026-08-21, before writing code)

Wiring the daemon steer handler to route per session was attempted and stopped.
Both blockers were found by grepping first; building would have shipped a feature
that looks complete and routes nowhere.

**1. The batched decode path never consults steer.** `hipfire-serving-core`
contains **zero** references to `hipfire_steer` — the fused multi-session decode
backends do not call the hook at all. M3's exit is "two streams with different
specs decode in ONE BATCHED STEP and each gets its own steering". That path
cannot satisfy it, because it never asks. Reaching the exit needs per-row
steering inside the fused batched decode, which is a materially larger change
than the handler wiring and touches the fused kernels' row structure.

**2. `StreamTable` is instantiated nowhere.** Not in `DaemonState`, not in any
handler — grep returns zero. Nothing admits a stream, because the daemon still
executes `Generate` inline; admission arrives with the executor (parent §M3).
Resolving `session_id` through it today would resolve against a permanently empty
table.

So the ordering in the ledger was still too optimistic. The steer subsystem is
ready — keyed sessions, keyed adapters, the thread guard, the keyed apply cache
— but its two consumers are not. **M3 waits on the executor** (admission, so
there are streams) **and on the batched decode path calling the hook** (so per-row
specs are consulted). Neither is steer work.

M4 (deleting the escape) waits on M3 and is unchanged.

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

**~~Five hook call sites, not three.~~ FIXED** — inventoried in the P2 plan's M3
section and named at both gemma3 call sites. Investigating it corrected the
concern as stated: gemma3 has **no routing predicate** — no lowered/hand split,
so no escape and no M1-M4 staging — and was therefore never exposed to the
`is_active()` regression. It does participate in per-stream steering (its
un-keyed hooks resolve the calling thread's key), which nothing had said.

**~~No gate runs the daemon's tests.~~ FIXED** — `no-gpu-ci.sh` now runs
`cargo test -p hipfire-daemon --bin hipfire-daemon` (95 tests). `--bin` is required:
`-p` alone inherits the same empty `--lib` selection. `hipfire-steer` needed no
change — it is a lib crate, so `ci.yml`'s `--lib --workspace` already covered it.
Original description: `hipfire-daemon` is bin-only (no `src/lib.rs`,
no `[lib]`), so `ci.yml`'s `cargo test --lib --workspace` selects zero targets
from it, and `no-gpu-ci.sh`'s `cargo check --workspace --examples` does not even
type-check `#[cfg(test)]` code. `stream.rs`'s tests — offered as the justification
for its file-scope `#![allow(dead_code)]` — are run by nothing in CI. Add
`hipfire-daemon` (and `hipfire-steer`) to the gate's explicit test list.

**~~Half the steer API is keyed.~~ FIXED** — `load_adapter_for`,
`load_lora_adapter_for`, `set_adapter_scale_for`, `unload_adapter_for` and
`loaded_adapters_for` added; the un-keyed forms delegate to the default key, so no
caller changed. Original description: `load_adapter`, `load_lora_adapter`,
`set_adapter_scale`, `unload_adapter` and `loaded_adapters` are hard-wired to the
default key, so a keyed session's adapters cannot be listed, rescaled or unloaded.

**~~`with_session` takes a write lock and inserts.~~ FIXED** — it no longer
creates: absent key returns `None`, so no key clone and no tombstone on the hot
path. `load_adapter` keeps an explicit creating path (its contract is to start a
session) but that is control-plane, once per load. Original description: It is `entry(key.clone())
.or_insert(...)` under the exclusive lock, per layer per token — a String clone
and a permanent map entry, not the "map lookup" the prose claims. Every distinct
key ever passed leaves a tombstone.

**~~`ACTIVE` is stored outside the lock.~~ FIXED** — `refresh_active_gate_locked`
takes the map by reference, so mutation + recompute + store are one critical
section. Original description: `refresh_active_gate` drops its read
guard before storing, so `clear_for(A)` racing `begin_apply_for(B)` can land a
stale `false` and silently disable B's steering.

**~~`SteerKeyGuard` is `Send`.~~ FIXED** — a `PhantomData<*const ()>` field pins
it to its installing thread, so a cross-thread move is now a compile error. A
compile-time assertion (two blanket impls, one gated on `Send`) proves it: the
call resolving IS the assertion. Original description: Its `Drop` restores into whatever thread drops it,
so a guard that crosses threads pins the installing thread to another stream's key.
