# The FP16 DeltaNet recurrence is not chunk-invariant — batched prefill and speculation advance the state differently from decode

Status: **FIXED 2026-09-02**, after one reverted attempt. The residual FP32-state spec/AR question is still open and is NOT closed by this. This is the origin the batched/per-token and spec/AR
divergences were pointing at. It is not the KV cache.

## The result

`crates/hipfire-runtime/examples/gdn_chunk_seq_parity.rs` — no model. Identical
synthetic q/k/v/gate/beta, one GDN state, two schedules:

- **A**: `gated_delta_net_*_batch_seq` with `chunk` tokens per launch — what
  batched prefill and the speculative verify do.
- **B**: one token per launch — what plain decode does. (For FP16 this mirrors
  `speculative.rs` exactly, which loops `f16_batch_seq` with `n_steps = 1`.)

n = 64 tokens, 16 heads, head_dim 128:

| state dtype | chunk | state elements differing | output elements differing | worst \|rel\| |
|---|---|---|---|---|
| FP32 | 7 / 16 / 17 / 32 / 64 | **0 / 262144** | **0 / 131072** | 0.000e0 |
| FP16 | 17 | 0 / 262144 | 33565 / 131072 | **1.989** |
| FP16 | 64 | **262144 / 262144** | 129022 / 131072 | **1.989** |

FP32 is bit-exact at every chunk size tested. FP16 is not: at a single 64-token
chunk **every element of the recurrent state differs** from stepping the same 64
tokens one at a time, and the worst relative error is ~2.0 — sign disagreement,
not rounding noise.

## Why this is the explanation

- The prefill probe reports `FIRST DIVERGING LAYER: 0`, and layer 0 is
  `linear_attn` — a DeltaNet layer that **carries no KV at all** on this model
  (full-attention layers are 3, 7, 11, 15, 19, 23). The divergence appears before
  any KV layer is reached.
- It is bit-width independent across KVarN 2/4/8, because it is not a KV effect.
- `kvarn_attend`'s write AND read paths are bit-exact batch-invariant
  (`2026-09-02-kvarn-write-path-is-batch-invariant.md`), so the cache is not it.
- The daemon's own prefill warning says as much: "the per-token fallback rounds
  the FP16 DeltaNet state once per token where the batched path rounds once per
  chunk ... **the KV mode is not the lever**."

## The code already knows — for one of the two paths

`speculative.rs` handles this correctly on the ROLLBACK REPLAY path, and says
why:

    // Same FP32-batched / FP16-per-token split as `replay_gdn_inner`, and for
    // the same reason: f16 narrows the state once per launch, so replaying an
    // accepted prefix must step token by token to match decode.

So FP16 replay steps token by token, deliberately. But **batched prefill still
calls `batch_seq` with FP16 state**, so a prompt processed in chunks lands in a
different state than the same prompt decoded token by token — which is exactly
what the prefill probe measures, and it switches on at `n = 257`, the first
prompt needing a second `PREFILL_MAX_BATCH` chunk.

The workaround was applied where someone hit the symptom, not where the
non-equivalence lives.

## Consequences

1. **Prefill is not equivalent to decode** for any FP16-state model whose prompt
   exceeds one 256-token chunk.
2. **Speculation is not equivalent to AR**: the verify advances the state over
   `b` rows in one launch, AR over one.
3. It is the mechanism behind
   `2026-09-02-ngram-speculation-drives-prompt-echo.md`, where n-gram
   speculation plus FP16 state deterministically drives the model into echoing
   87% of its prompt while reporting a 3.4x speedup.

## Mitigation and its limit

`2026-09-02-deltanet-redundancy-gate-had-no-caller.md` restores the guard that
forces FP32 state below a redundancy floor, and FP32 is chunk-invariant here, so
models under the floor are covered. `Qwen3.6-35B-A3B` (redundancy 4096) is above
it and still runs FP16 — untested, and this is the reason to test it.

## What this does NOT explain

At decode with FP32 state, speculative output still differs from AR under
quantised KV and matches only at `fp32` KV. If both the GDN recurrence (at FP32)
and the KVarN paths are chunk/batch invariant, that result needs a separate
cause. It is measured on real generations, so it is not an artifact of the
prefill probe. Do not treat this document as closing that question.

## Fix: reconcile the dither index, THEN narrow per token

The first attempt narrowed per token in `gated_delta_net_f16.hip` only. That
fixed chunk-invariance and broke `test_gated_delta_net_routed_f16`
(1089/3072 byte-exact), which replays each session through the LINEAR kernel and
requires ROUTED to reproduce it byte-for-byte. Patching routed as well made it
worse (1024/3072). Reverted.

The reason is that the three kernels did not derive the dither index the same
way:

    linear / tree:  (row_start * HD + tile_flat) ^ (h << 19)
    routed:         tile_flat ^ (blockIdx.x << 19) ^ (blockIdx.y << 7)

With `h == blockIdx.x` and `tile == blockIdx.y`, routed's is
`tile_flat ^ (h << 19) ^ (tile << 7)`. Linear ADDS `row_start * HD`
(= `tile * 512`) into the index where routed XORed `tile << 7` (= `tile * 128`).
**They coincide only at `tile == 0`.** Narrowing once per call, that mismatch is
invisible — the index is consulted once, at the end, and the tests compare
OUTPUTS, which are computed before the narrowing. Narrowing per token puts the
index inside the state trajectory, where the mismatch becomes visible on the very
next token.

So the fix is two steps, in order:

1. **Unify the index.** Routed now uses `(row_start * HD + tile_flat) ^ (h << 19)`,
   the linear/tree derivation. The session (`blockIdx.z`) is deliberately absent:
   each session has its own state buffer, and the per-session linear replay that
   routed must reproduce has no session term to match. Verified on its own, with
   no other change: all three kernel tests still pass.
2. **Narrow once per token** in both `batch_seq` kernels, using that shared
   derivation, and make the final store an exact conversion so the last token is
   not double-rounded. `n_tokens == 1` is therefore unchanged, and an N-token call
   lands where N single-token calls land.

`gated_delta_net_f16_tree.hip` needed no change — it already narrowed per token on
the persist-write, and already used the linear derivation. It was the reference
all along.

**Verified after:**

    gdn_chunk_seq_parity        FP32 chunk 7/17/32/64            PASS bit-exact
    gdn_chunk_seq_parity --f16  chunk 7/17/32/64, n 64 and 128   PASS bit-exact (was FAIL)
    tiny-deltanet-gate          PASS (5 cells)
      parity_gated_delta_net_f64acc, ..._f64acc_routed,
      test_gated_delta_net_tree_f32, ..._routed_f32, ..._routed_f16
    tiny-state-gate             PASS 18/18, no baseline movement

### The trade

More roundings is a worse approximation of an FP32 reference, which is what the
old comment optimised for ("a multi-token batch pays a single rounding rather
than one per token"). Sound in isolation, wrong in context: prefill and decode
must agree with EACH OTHER, and `speculative.rs` already pays exactly this price
on the rollback replay path for exactly this reason.

### What the fix does and does not buy, measured

**Does**: the pathological prompt-echo collapses. On the harness in
`2026-09-02-ngram-speculation-drives-prompt-echo.md`, FP16 state with n-gram
speculation went from `echo_frac 0.872` (87% of the output copied verbatim from
the prompt, at an inflated 263 tok/s) to **0.195 / 0.243**. The feedback loop is
broken.

**Does not**: speculative output still differs from AR, at BOTH state precisions,
under quantised KV:

    state=FP16   AR 4257e0a8a3c1   spec 1b030299490f   differs
    state=FP32   AR 77ab36b1ad5a   spec 675ebc8a5988   differs

That matches the earlier finding that only `fp32` KV makes them byte-identical.
So there are (at least) two independent contributors to spec/AR non-equivalence,
and this fixes one of them. The KV-axis contributor is still open and is NOT
explained by anything measured so far — `kvarn_attend`'s write and read paths are
bit-exact batch-invariant, and the residual FP16-vs-FP32 echo gap (0.24 vs 0.00)
is the ordinary cost of half-precision state, not a chunk-invariance defect.

### Coverage gap this leaves — CLOSED

`tiny-state-gate` passing means little here: the redundancy guard forces FP32
state on the tiny fixtures, so they never exercise multi-token f16 GDN. No gate
in the tree did, which is why this survived.

`tests/tiny-deltanet-gate.sh` now runs `gdn_chunk_seq_parity` in four
configurations — chunk 17 and 64 at FP32, chunk 17 and 32 at FP16, n=64 and 128 —
taking it from 5 cells to 9.

The gate was verified to actually catch the regression, not merely to pass:
reverting `gated_delta_net_f16.hip` to its pre-fix version makes it FAIL both
`--f16` cells with "chunking changes the recurrence", while the four f32/agreement
cells stay green. Restoring the fix returns it to PASS (9 cells). A gate that has
never been shown to fail is not cover.

Note what the existing trio could NOT see: it proves the three kernels agree with
EACH OTHER, which says nothing about whether one launch of N tokens equals N
launches of one. Both properties are needed and only one was gated.

 The FP16 DeltaNet recurrence is not chunk-invariant — batched prefill and speculation advance the state differently from decode

Status: **FIXED 2026-09-02**, after one reverted attempt. The residual FP32-state spec/AR question is still open and is NOT closed by this. This is the origin the batched/per-token and spec/AR
divergences were pointing at. It is not the KV cache.

## The result

`crates/hipfire-runtime/examples/gdn_chunk_seq_parity.rs` — no model. Identical
synthetic q/k/v/gate/beta, one GDN state, two schedules:

- **A**: `gated_delta_net_*_batch_seq` with `chunk` tokens per launch — what
  batched prefill and the speculative verify do.
- **B**: one token per launch — what plain decode does. (For FP16 this mirrors
  `speculative.rs` exactly, which loops `f16_batch_seq` with `n_steps = 1`.)

n = 64 tokens, 16 heads, head_dim 128:

| state dtype | chunk | state elements differing | output elements differing | worst \|rel\| |
|---|---|---|---|---|
| FP32 | 7 / 16 / 17 / 32 / 64 | **0 / 262144** | **0 / 131072** | 0.000e0 |
| FP16 | 17 | 0 / 262144 | 33565 / 131072 | **1.989** |
| FP16 | 64 | **262144 / 262144** | 129022 / 131072 | **1.989** |

FP32 is bit-exact at every chunk size tested. FP16 is not: at a single 64-token
chunk **every element of the recurrent state differs** from stepping the same 64
tokens one at a time, and the worst relative error is ~2.0 — sign disagreement,
not rounding noise.

## Why this is the explanation

- The prefill probe reports `FIRST DIVERGING LAYER: 0`, and layer 0 is
  `linear_attn` — a DeltaNet layer that **carries no KV at all** on this model
  (full-attention layers are 3, 7, 11, 15, 19, 23). The divergence appears before
  any KV layer is reached.
- It is bit-width independent across KVarN 2/4/8, because it is not a KV effect.
- `kvarn_attend`'s write AND read paths are bit-exact batch-invariant
  (`2026-09-02-kvarn-write-path-is-batch-invariant.md`), so the cache is not it.
- The daemon's own prefill warning says as much: "the per-token fallback rounds
  the FP16 DeltaNet state once per token where the batched path rounds once per
  chunk ... **the KV mode is not the lever**."

## The code already knows — for one of the two paths

`speculative.rs` handles this correctly on the ROLLBACK REPLAY path, and says
why:

    // Same FP32-batched / FP16-per-token split as `replay_gdn_inner`, and for
    // the same reason: f16 narrows the state once per launch, so replaying an
    // accepted prefix must step token by token to match decode.

So FP16 replay steps token by token, deliberately. But **batched prefill still
calls `batch_seq` with FP16 state**, so a prompt processed in chunks lands in a
different state than the same prompt decoded token by token — which is exactly
what the prefill probe measures, and it switches on at `n = 257`, the first
prompt needing a second `PREFILL_MAX_BATCH` chunk.

The workaround was applied where someone hit the symptom, not where the
non-equivalence lives.

## Consequences

1. **Prefill is not equivalent to decode** for any FP16-state model whose prompt
   exceeds one 256-token chunk.
2. **Speculation is not equivalent to AR**: the verify advances the state over
   `b` rows in one launch, AR over one.
3. It is the mechanism behind
   `2026-09-02-ngram-speculation-drives-prompt-echo.md`, where n-gram
   speculation plus FP16 state deterministically drives the model into echoing
   87% of its prompt while reporting a 3.4x speedup.

## Mitigation and its limit

`2026-09-02-deltanet-redundancy-gate-had-no-caller.md` restores the guard that
forces FP32 state below a redundancy floor, and FP32 is chunk-invariant here, so
models under the floor are covered. `Qwen3.6-35B-A3B` (redundancy 4096) is above
it and still runs FP16 — untested, and this is the reason to test it.

## What this does NOT explain

At decode with FP32 state, speculative output still differs from AR under
quantised KV and matches only at `fp32` KV. If both the GDN recurrence (at FP32)
and the KVarN paths are chunk/batch invariant, that result needs a separate
cause. It is measured on real generations, so it is not an artifact of the
prefill probe. Do not treat this document as closing that question.

## Fix: reconcile the dither index, THEN narrow per token

The first attempt narrowed per token in `gated_delta_net_f16.hip` only. That
fixed chunk-invariance and broke `test_gated_delta_net_routed_f16`
(1089/3072 byte-exact), which replays each session through the LINEAR kernel and
requires ROUTED to reproduce it byte-for-byte. Patching routed as well made it
worse (1024/3072). Reverted.

The reason is that the three kernels did not derive the dither index the same
way:

    linear / tree:  (row_start * HD + tile_flat) ^ (h << 19)
    routed:         tile_flat ^ (blockIdx.x << 19) ^ (blockIdx.y << 7)

With `h == blockIdx.x` and `tile == blockIdx.y`, routed's is
`tile_flat ^ (h << 19) ^ (tile << 7)`. Linear ADDS `row_start * HD`
(= `tile * 512`) into the index where routed XORed `tile << 7` (= `tile * 128`).
**They coincide only at `tile == 0`.** Narrowing once per call, that mismatch is
invisible — the index is consulted once, at the end, and the tests compare
OUTPUTS, which are computed before the narrowing. Narrowing per token puts the
index inside the state trajectory, where the mismatch becomes visible on the very
next token.

So the fix is two steps, in order:

1. **Unify the index.** Routed now uses `(row_start * HD + tile_flat) ^ (h << 19)`,
   the linear/tree derivation. The session (`blockIdx.z`) is deliberately absent:
   each session has its own state buffer, and the per-session linear replay that
   routed must reproduce has no session term to match. Verified on its own, with
   no other change: all three kernel tests still pass.
2. **Narrow once per token** in both `batch_seq` kernels, using that shared
   derivation, and make the final store an exact conversion so the last token is
   not double-rounded. `n_tokens == 1` is therefore unchanged, and an N-token call
   lands where N single-token calls land.

`gated_delta_net_f16_tree.hip` needed no change — it already narrowed per token on
the persist-write, and already used the linear derivation. It was the reference
all along.

**Verified after:**

    gdn_chunk_seq_parity        FP32 chunk 7/17/32/64            PASS bit-exact
    gdn_chunk_seq_parity --f16  chunk 7/17/32/64, n 64 and 128   PASS bit-exact (was FAIL)
    tiny-deltanet-gate          PASS (5 cells)
      parity_gated_delta_net_f64acc, ..._f64acc_routed,
      test_gated_delta_net_tree_f32, ..._routed_f32, ..._routed_f16
    tiny-state-gate             PASS 18/18, no baseline movement

### The trade

More roundings is a worse approximation of an FP32 reference, which is what the
old comment optimised for ("a multi-token batch pays a single rounding rather
than one per token"). Sound in isolation, wrong in context: prefill and decode
must agree with EACH OTHER, and `speculative.rs` already pays exactly this price
on the rollback replay path for exactly this reason.

### What the fix does and does not buy, measured

**Does**: the pathological prompt-echo collapses. On the harness in
`2026-09-02-ngram-speculation-drives-prompt-echo.md`, FP16 state with n-gram
speculation went from `echo_frac 0.872` (87% of the output copied verbatim from
the prompt, at an inflated 263 tok/s) to **0.195 / 0.243**. The feedback loop is
broken.

**Does not**: speculative output still differs from AR, at BOTH state precisions,
under quantised KV:

    state=FP16   AR 4257e0a8a3c1   spec 1b030299490f   differs
    state=FP32   AR 77ab36b1ad5a   spec 675ebc8a5988   differs

That matches the earlier finding that only `fp32` KV makes them byte-identical.
So there are (at least) two independent contributors to spec/AR non-equivalence,
and this fixes one of them. The KV-axis contributor is still open and is NOT
explained by anything measured so far — `kvarn_attend`'s write and read paths are
bit-exact batch-invariant, and the residual FP16-vs-FP32 echo gap (0.24 vs 0.00)
is the ordinary cost of half-precision state, not a chunk-invariance defect.

### Coverage gap this leaves

`tiny-state-gate` passing means little here — the restored redundancy guard forces
FP32 state on the tiny fixtures, so they never exercise multi-token f16 GDN. No
gate in the tree did, which is why this survived. `gdn_chunk_seq_parity --f16` is
the check that does, and it is not yet wired into any gate.

## Fix directions## Fix directions

- Make `f16_batch_seq` chunk-invariant (accumulate in f32 within the launch and
  narrow once per token, matching the per-token path), or
- route batched prefill through the per-token path whenever the state is FP16,
  as the replay path already does, or
- make FP32 state the default for every model rather than only below a
  redundancy floor, and treat FP16 state as an explicitly-opted-in approximation
  that breaks prefill/decode equivalence.
