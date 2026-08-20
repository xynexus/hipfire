# Plan: P2 — steer per-stream, and onto the lowered path

Implementation plan for the decision recorded in
`2026-08-20-v2-prerequisites-autonomous.md` (§"P2 sequencing decision"):
**option (b) — do the per-stream state and the lowered-path hook together, not
separately.** That decision is settled; this plan is how.

## The shape of the work, measured

**The hook is trivial.** The hand path calls
`hipfire_steer::maybe_steer_block(gpu, &s.x, layer_idx)` — one line at
`qwen35/decode_layers.rs:2768`. The lowered executor has exactly that shape
available at its layer boundary, where `hidden_rb`'s
`extract_slot` / `write_at_head` already runs on the same `&s.x` and
`layer_idx` (`qwen35/lowered.rs:690-699`). `hidden_rb` is the worked precedent:
it used to force the hand path for the same reason steer does, and was retired by
giving the lowered path the hook.

**The per-stream half is a subsystem change.** `maybe_steer_block` reads
`SESSION` / `ACTIVE` / `EPOCH` *internally*, so per-stream means changing its
signature, which reaches ~18 public functions in `hipfire-steer`, the daemon
handlers that install sessions (`handlers/steer.rs`, `handlers/lora.rs`,
`handlers/lifecycle.rs`), and `APPLY_CACHE`'s epoch semantics.

**The knot that shapes the staging.** The exit for this work is "lowered-path
steering matches hand-path steering, asserted on output". But while the escape
stands, an active steer session forces the hand path — so the lowered hook never
fires and the two cannot be compared. **Parity cannot be measured until the
escape can be bypassed**, which means there is no "safe first slice" that avoids
touching the forward path. Plan accordingly rather than discovering it at M2.

## Staging

### M1 — the hook, plus a bypass that makes it observable

Add the `maybe_steer_block` call at the lowered layer boundary, mirroring
`hidden_rb`. Then make the escape's steer condition bypassable —
`HIPFIRE_STEER_LOWERED=1`, default off — so the lowered path can run WITH steer
active for comparison.

Default-off is what keeps this stage non-breaking: production behaviour is
byte-identical because the escape still fires.

*Exit:* with the flag off, a steering session still takes the hand path (assert
via `HIPFIRE_DECODE_BACKEND_TRACE`); with it on, the lowered path runs and the
hook fires. No parity claim yet — only that both paths are now reachable.

### M2 — parity, and it is the whole point

Same model, same prompt, same steer spec, greedy. Run hand-path and lowered-path
steering; compare **output tokens**, not activations.

*Exit:* byte-identical token streams over ≥128 tokens, for both `Applying` and
`Capturing` sessions. `Capturing` additionally must produce the same
`CaptureMeans` — it downloads and folds `x`, so a boundary placed one op early or
late shows up there and nowhere else.

**If parity fails, stop.** A mismatch means the boundary is not the same point in
the two paths, and the fix is the boundary, not the test.

*Falsified by:* any token difference. This is the accept-and-miscompute class —
steering a slightly different residual produces plausible output, so only exact
comparison catches it.

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

*Exit:* M2's parity assertions still hold with no flag set; a steering session
now runs the lowered path by default. `HIPFIRE_FORWARD_LOWERED=0` remains the
general opt-out.

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

* M2 parity fails — report the mismatch, do not adjust the test;
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
