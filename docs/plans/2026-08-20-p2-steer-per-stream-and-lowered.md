# Plan: P2 — steer per-stream, and onto the lowered path

Implementation plan for the decision recorded in
`2026-08-20-v2-prerequisites-autonomous.md` (§"P2 sequencing decision"):
**option (b) — do the per-stream state and the lowered-path hook together, not
separately.** That decision is settled; this plan is how.

## The shape of the work, measured

**The hook is trivial** — and as of M1 it is in. Both files live under
`crates/hipfire-arch-qwen35/src/` (*not* `hipfire-runtime`; the shorthand below
has misdirected at least one reader). The hand path calls
`hipfire_steer::maybe_steer_block(gpu, &s.x, layer_idx)` — one line at
`qwen35/decode_layers.rs:2787`. The lowered executor had exactly that shape
available at its layer boundary, where `hidden_rb`'s
`extract_slot` / `write_at_head` already runs on the same `&s.x` and `layer_idx`
(`qwen35/lowered.rs:718-724`); M1 put the same call there, at
`qwen35/lowered.rs:732`, after `dump_hidden_localize` so the ordering mirrors the
hand path exactly. `hidden_rb` is the worked precedent: it used to force the hand
path for the same reason steer does, and was retired by giving the lowered path
the hook.

**The per-stream half is a subsystem change.** `maybe_steer_block` reads
`SESSION` / `ACTIVE` / `EPOCH` *internally*, so per-stream means changing its
signature, which reaches ~18 public functions in `hipfire-steer`, the daemon
handlers that install sessions (`handlers/steer.rs`, `handlers/lora.rs`,
`handlers/lifecycle.rs`), and `APPLY_CACHE`'s epoch semantics.

**The knot that shapes the staging.** While the escape stands, an active steer
session forces the hand path — so the lowered hook never fires and nothing about
it can be measured. **Nothing is assertable until the escape can be bypassed**,
which means there is no "safe first slice" that avoids touching the forward path.
Plan accordingly rather than discovering it at M2.

M1 delivered that bypass, and the first thing it measured was that the hand path
is not a correct reference — which is why M2's exit below no longer compares
against it. The knot was worth untying early for exactly this reason: the
original exit was unsatisfiable, and only running the comparison revealed it.

## Staging

### M1 — the hook, plus a bypass that makes it observable — **DONE** (`edc13da5c`)

Add the `maybe_steer_block` call at the lowered layer boundary, mirroring
`hidden_rb`. Then make the escape's steer condition bypassable —
`HIPFIRE_STEER_LOWERED=1`, default off — so the lowered path can run WITH steer
active for comparison.

Default-off is what keeps this stage non-breaking: production behaviour is
byte-identical because the escape still fires.

*Exit:* with the flag off, a steering session still takes the hand path (assert
via `HIPFIRE_DECODE_BACKEND_TRACE`); with it on, the lowered path runs and the
hook fires. No parity claim yet — only that both paths are now reachable.

### M2 — correctness, and it is the whole point

**This exit was rewritten on 2026-08-20, because at the time the hand path could
not be the reference.** Measured while probing M1's hook
(`docs/experiments/2026-08-20-p2-m2-hand-path-is-a-broken-reference.md`): the
qwen35 hand decode forward was broken *independently of steering*. At
`strength = 0.0`, where steering is the identity `x += 0·v`, it did not reproduce
the unsteered baseline — its output there was byte-identical to what it emitted
with no steer session at all.

**That cause is now FIXED on master** (`1f7c2eeba`): the dense DeltaNet arm never
applied `ffn_norm`, so its FFN consumed the attention-normalized, pre-attention
residual. Hand and lowered now agree — self-KLD 5.3e-10, see
`docs/experiments/2026-08-20-qwen35-hand-dense-ffn-norm-fix.md`.

**The rewritten exit below still stands, and should not be reverted to the
original.** Parity with the hand path is now *available*, but it remains the
weaker assertion: it is defined as "whatever the other path did", whereas the
identity anchor has an independently known correct answer. Parity is worth
keeping as a corroborating check, not as the exit. The trap documented further
down — the two paths agreeing at high strength *on garbage* — is a property of
steering, not of the bug that was fixed, and is undiminished by the fix.

*Measured status (2026-08-20, post-fix, `Qwen3.5-0.8B-Base--oq8`, 24L/dim 1024):*

| assertion | result |
|---|---|
| 1 — identity anchor | **PASSES** both paths: `strength 0.0` byte-identical to the unsteered baseline over 126 tokens |
| 2 — oracle | **PASSES**: `gpu_validate` ALL PASS on gfx1103 — Steer and Ablate × 4 layers × 2 strengths, worst `max_abs_err` 1.79e-7 against 1e-4 / 2e-3 tolerances, plus a 256× Ablate sync-stress (6.11e-7) and the capture→derive round-trip |
| 3 — graceful degradation | **PASSES**: smooth across `{0, 0.1, 0.25, 0.5, 1.0}`, 3/3 identical md5 per point on both paths |
| parity (corroborating) | **PASSES** 5/5 strengths, same md5 across paths |

**M2 is satisfied.** M3 is unblocked.

*Correction to assertion 2 as first written.* It said to "extend `gpu_validate.rs`
to the lowered call site". **There is no such thing to extend to**, and following
that instruction would have been wasted work. `gpu_validate` calls
`maybe_steer_block` directly on a synthetic tensor, and the hand and lowered paths
call that *same function* — the paths differ in WHERE the hook is invoked within
the layer, not in the apply math it performs. So the existing oracle already
covers both; running it is the whole of assertion 2. The boundary difference is
what assertions 1 and 3 cover, which is the right division.

The `Capturing` caveat below still applies and is unaffected by the hand-path fix
— note `gpu_validate` does exercise capture→derive, but not the decode-path
capture boundary.

*Exit:* all three hold, same model, greedy.

1. **Identity anchor.** `strength = 0.0`, ≥128 tokens: byte-identical to the
   unsteered baseline (same prompt, no session). This is the load-bearing one —
   it is the only assertion whose correct answer is known independently rather
   than defined as "whatever the other path did". Both paths pass it as of
   `1f7c2eeba`; before that fix the lowered path passed and the hand path failed,
   which is what exposed the bug.
2. **Oracle.** The on-GPU apply inside `maybe_steer_block` matches
   `hipfire-steer`'s host reference within f32 tolerance, per layer, for both
   `steer` and `ablate`. Run `hipfire-steer/examples/gpu_validate.rs` — it already
   performs exactly this cross-check, and because both decode paths call the same
   `maybe_steer_block`, it covers both. Do not author a second oracle, and do not
   try to "extend it to the lowered call site": there isn't one.

   ```sh
   hipfire lock acquire steer-validate && \
     cargo run --release -p hipfire-steer --example gpu_validate; hipfire lock release
   ```
3. **Graceful degradation.** Sweep `strength` over `{0, 0.1, 0.25, 0.5, 1.0}`:
   output degrades smoothly, and each point is deterministic across ≥3 runs. No
   cliff, no instant EOS.

**Do not assert parity at a single high strength.** Measured: the two paths *do*
agree at 1.0 and 2.0 — on garbage, having converged on the same degenerate
attractor (`>\n\n>\n\n`, `用电用电`). A run that picked one "clearly steering hard
enough to see an effect" strength would have reported parity and been wrong. A
longer token budget is no defence: 128 tokens of identical garbage is still
identical. Anchoring at `strength = 0.0` is the defence.

**`Capturing` is not assertable over the daemon protocol.** The original exit
asked for matching `CaptureMeans`, on the reasoning that a boundary placed one op
early or late shows up there and nowhere else. It cannot show up there at all:
`CaptureAcc::observe` only overwrites `current[layer]`; only `commit()` folds
`current` into `sums` and bumps `count`; `means()` divides by `count.max(1)`; and
`commit()` is reachable only from the `steer_capture` op, which is **prefill-only**
and runs `maybe_steer_block_batched` in `prefill_chunk.rs` — a call site this plan
never touches. A capture session driven through `generate` therefore returns all
zeros, and comparing them across paths is a **vacuous pass** (verified:
24576/24576 zero elements on both sides). Assert the `Capturing` boundary
in-process against `maybe_steer_block` (the `gpu_validate.rs` route), or add a
commit reachable from decode. Do not assert it through the daemon.

*Falsified by:* assertion 1 failing. That is the accept-and-miscompute class —
steering a slightly different residual produces plausible output, so only exact
comparison against the known-correct baseline catches it.

**If assertion 1 fails, stop.** It means the boundary is not at the right point in
the layer, and the fix is the boundary, not the test. Replacing the *reference*
(this rewrite) was a plan-level decision forced by that reference being
independently broken; loosening an *assertion* to accommodate a mismatch is a
different act and is not authorised by it.

### M3 — per-stream state

**Unblocked: M2 is satisfied as of 2026-08-20.** Move `SESSION`/`ACTIVE`/`EPOCH`
into per-stream state and thread a handle to the hook.

**Follow the two landed precedents; do NOT invent a container.** M3 is one third
of the parent plan's M1d ("the remaining process globals become per-stream:
`RAW_OVERRIDE`, `hipfire_steer::{SESSION, ACTIVE, EPOCH}`, `load_progress::SINK`").
The other two thirds are already done, and both used the *same* shape — replace
the global with a value the caller owns and passes explicitly:

| global | became | where |
|---|---|---|
| `RAW_OVERRIDE` (`thread_local`) | an `override_: Option<bool>` parameter | `serving-core/src/model.rs::effective_raw` |
| `SAMPLER_STATE: AtomicU32` | a `SamplerRng` value, `from_seed`, owned per stream | `runtime/src/sampler.rs` |
| `load_progress::SINK` | a thread-scoped sink + `ThreadSinkGuard` | P1, `handlers/lifecycle.rs` |

The parent plan's M1b text says to "move the state into `StreamState`". **There is
no `StreamState` type in this tree** — grep before building one. What landed was
explicit threading, which is what M3's "thread a handle to the hook" already
describes.

Note `load_progress` took the *thread-scoped* variant rather than a parameter,
because its call sites are six arch loaders whose signatures were not worth
touching. **Both sentences that once followed here were wrong and are retracted.**
They said steer had "only two real forward-path call sites" so an explicit
parameter was the stronger shape, and that a thread-local "would re-create the
same 'who owns this' ambiguity". There are FIVE call sites (inventory in M3), and
the parameter shape was measured at 155 external call sites and rejected; the
thread-scoped guard is what landed, in #271. A scoped guard installed for one
quantum and restored on drop is not the ambient state that argument feared.

One transferable detail from the sampler work: its tests "used to share a
`static SAMPLER_STATE` and cargo runs them on parallel threads", which is the same
test-isolation hazard P1 hit. Whatever M3 lands, its tests must not share process
state or they will flake under `cargo test`.

`APPLY_CACHE` stays a `thread_local` holding `GpuTensor` — it is `!Sync` and
cannot move into shared state. What changes is what the epoch invalidates
*against*: today one global counter, after this a per-stream one, so a cache
entry uploaded for stream A is not reused for stream B.

*Exit:* two streams with different specs decode in one batched step and each gets
its own steering, asserted on output. A third stream with no spec is unaffected.

*Falsified by:* stream B's output changing when only stream A's spec changes.

#### BLOCKED — there is no "stream" to be per, yet (found 2026-08-20)

M3 says "move the state into per-stream state". **The per-stream home does not
exist.** The parent plan creates it in *its* M3 ("Replace the request `match` ...
with `RunningStream` cursors", `2026-08-09-v2-daemon-module-major-multistream.md`
§M3) — and `RunningStream` has no definition in the tree today. The two landed
M1d conversions did not need one: `RAW_OVERRIDE` became a request parameter and
the sampler RNG became a value the caller already owned. Steer has no such
existing owner.

The second half of the problem is the wire protocol. `SteerBeginCapture` and
`SteerBeginApply` carry **no stream, session, or worker identifier** —
`handlers/steer.rs` says as much ("the session is process-global ... two steer ops
must never interleave"). So even with a per-stream container, a client currently
has no way to say *which* stream a spec applies to. Nor is "stream" the same as
"worker": the exit above wants two streams inside one batched step, i.e. two
sessions against one resident model.

Three shapes, and this plan does not pick — it is a protocol/ownership decision:

1. **Wait for the parent plan's M3.** Build steer per-stream directly into
   `RunningStream` when it lands. No double work, but blocks on a larger milestone.
2. **Attach at admission.** Add the spec to the generate request and snapshot it
   into per-request state; steer control ops stay global but become a *default*
   a request overrides. Smallest protocol change; no new identifier.
3. **Identify streams on the steer ops.** Add an explicit id to
   `SteerBeginApply`/`SteerBeginCapture`. Most direct, but invents a naming scheme
   ahead of the executor that will own it — likely to be re-done at parent-M3.

Recommendation: **(1)**, with (2) as the cheap interim if per-stream steering is
needed before the executor lands. What must NOT happen is picking (3) and
threading a handle through ~10 `hipfire-steer` functions and ~14 daemon call sites
against an identifier the executor then replaces.

Measured surface for whenever it proceeds: **10** public functions in
`hipfire-steer` touch `SESSION`/`ACTIVE`/`EPOCH` (the queue's "eighteen" counts
doc mentions, as it notes); ~14 daemon call sites across `handlers/steer.rs`,
`handlers/lora.rs`, `handlers/lifecycle.rs`; and **five** forward call sites,
across TWO arch crates:

| crate | file | hook |
|---|---|---|
| `hipfire-arch-qwen35` | `qwen35/decode_layers.rs` | `maybe_steer_block` |
| `hipfire-arch-qwen35` | `qwen35/lowered.rs` (as of M1) | `maybe_steer_block` |
| `hipfire-arch-qwen35` | `qwen35/prefill_chunk.rs` | `maybe_steer_block_batched` |
| `hipfire-arch-gemma3` | `src/forward.rs:781` | `maybe_steer_block` |
| `hipfire-arch-gemma3` | `src/forward.rs:1057` | `maybe_steer_block_batched` |

gemma3 was omitted from every earlier count in this file, despite the daemon's own
comment naming it as an arch the hook is compiled into. It carries no lowered/hand
split, so it has no routing escape and the M1-M4 staging does not apply to it —
but it IS steered, and any change to the hook's contract must count it.

### M4 — retire the escape

Remove the steer condition from `decode_layers.rs`, and the M1 flag with it.

*Exit:* M2's three assertions still hold with no flag set; a steering session
now runs the lowered path by default. `HIPFIRE_FORWARD_LOWERED=0` remains the
general opt-out — though note it opts into a forward that is currently broken
(see M2), so it is a debugging lever, not a safety net.

**Do not do M4 before M2 and M3 both hold.** It deletes the fallback, which is a
stop-line in the parent plan — authorised by the sequencing decision, but only in
this order.

## Verification

Every stage: `./tests/no-gpu-ci.sh`, and `./tests/tiny-affected-gate.sh
--require-coverage` — reporting honestly if it selects no coverage, which it may,
since the tiny fixtures do not steer.

Non-steering numerics must not move at any stage. If a greedy generation without
a steer session changes by one token, something is wrong regardless of what the
steering tests say.

## Stop and report

* M2's identity anchor (`strength = 0.0`) fails — report it, do not loosen the
  assertion; the boundary is wrong;
* the per-stream change would require touching the sampler or KV paths — that
  means the boundary is wrong;
* any non-steering output changes;
* `APPLY_CACHE` appears to need to become `Sync` — it holds `GpuTensor`; if the
  design demands that, the design is wrong.

## Note for whoever picks this up

Two of the four escapes in `decode_layers.rs` are now retired (`hidden_rb`, and
RoughQuant behind an opt-in). Steer is the third, and GDN tape capture is the
fourth. The pattern in both retirements was the same: give the lowered path the
capability, prove parity, then delete the escape — never the reverse.

**Steer breaks that pattern in one place, and the fourth escape probably will
too.** "Prove parity" assumes the hand path is a correct reference. For steer it
is not (M2), so the middle step became "prove correctness against an independent
anchor" instead. Before planning GDN tape capture the same way, check whether its
hand arm still computes correctly — the two earlier retirements predate the hand
path's regression, so their playbook silently assumed something that is no longer
true.
