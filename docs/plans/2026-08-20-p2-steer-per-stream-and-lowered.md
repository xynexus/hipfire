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

**This exit was rewritten on 2026-08-20. The hand path cannot be the reference.**
Measured while probing M1's hook
(`docs/experiments/2026-08-20-p2-m2-hand-path-is-a-broken-reference.md`): the
qwen35 hand decode forward is broken *independently of steering*. At
`strength = 0.0`, where steering is the identity `x += 0·v`, it does not
reproduce the unsteered baseline — and its output there is byte-identical to what
it emits with no steer session at all. Reproduced on both Opus artifacts on disk
(`oq8`, `oq4++`); bf16 is untested for want of a qwen3.5 bf16 artifact, though the
in-tree comment in `decode_layers.rs` reports bf16 self-KLD 13.89 vs lowered
0.000.

So the original exit — byte-identical token streams between the two paths — is
unsatisfiable, and **meeting it would be a regression**, because it would mean
making the lowered path reproduce a broken forward. The replacement asserts that
lowered steering is *correct*, not that it is *equal to the hand path*. No
assertion below references the hand path.

*Exit:* all three hold, same model, greedy.

1. **Identity anchor.** `strength = 0.0`, ≥128 tokens: byte-identical to the
   unsteered baseline (same prompt, no session). This is the load-bearing one —
   it is the only assertion whose correct answer is known independently rather
   than defined as "whatever the other path did". The lowered path passes it
   today; the hand path fails it.
2. **Oracle.** The on-GPU decode apply matches `hipfire-steer`'s host-side
   `apply_stack_host` within f32 tolerance, per layer, for both `steer` and
   `ablate`. `hipfire-steer/examples/gpu_validate.rs` already performs exactly
   this cross-check — extend it to the lowered call site rather than authoring a
   second oracle.
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

Only after M2. Move `SESSION`/`ACTIVE`/`EPOCH` into per-stream state and thread a
handle to the hook.

`APPLY_CACHE` stays a `thread_local` holding `GpuTensor` — it is `!Sync` and
cannot move into shared state. What changes is what the epoch invalidates
*against*: today one global counter, after this a per-stream one, so a cache
entry uploaded for stream A is not reused for stream B.

*Exit:* two streams with different specs decode in one batched step and each gets
its own steering, asserted on output. A third stream with no spec is unaffected.

*Falsified by:* stream B's output changing when only stream A's spec changes.

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
