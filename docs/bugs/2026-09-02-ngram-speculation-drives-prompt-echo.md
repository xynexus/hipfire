# n-gram speculation can drive the model into echoing its prompt — and the "speedup" is that corruption

Status: **OPEN, mechanism measured.** Found 2026-09-02 while auditing why an
earlier throughput figure would not reproduce. It is the practical consequence of
`2026-09-01-spec-decode-not-output-equivalent-to-ar.md`, and it upgrades that bug
from "output differs" to "output degrades".

## What happens

`qwen3.5-0.8b--oq4++.hfq`, greedy, prompt = 6151 chars of a Rust source file,
two 400-token generations per run, three runs per configuration. `echo_frac` is
the fraction of 40-char windows of the OUTPUT that appear verbatim in the prompt.

| configuration | echoed | mean coverage | mean tok/s |
|---|---|---|---|
| n-gram ON, FP16 DeltaNet state | **3 of 6** (every g2 at 87.2%) | 0.323 | 161.4 |
| n-gram ON, FP32 DeltaNet state | 0 of 6 | 0.388 | 76.5 |
| n-gram OFF, FP16 DeltaNet state | 0 of 2 | — | 71.9 |

Byte-reproducible: all three reps of each cell agree to the digit.

Two facts do the work:

- **Neither ingredient degenerates alone.** FP16 state without speculation never
  echoes. Speculation with FP32 state never echoes. Only the combination does.
- **The echoing runs are the fast ones.** The 87.2%-echo generations run at
  263 tok/s; the non-echo FP16 generations run at 59. The throughput is bought
  with the corruption.

## Mechanism

1. Speculative decode is not bit-exact on this stack — the batched verify and the
   per-token AR path disagree under quantised KV / narrow state (see the
   companion doc; fp32 KV is byte-identical to AR, everything else is not).
2. The n-gram tier drafts from what it has observed, and it observes the PROMPT
   (`ng.observe(&prompt_tokens)` at request start).
3. A verify that is not exact accepts some drafted tokens the target would not
   have chosen on its own.
4. Each accepted prompt-derived token pulls the continuation further toward the
   prompt.
5. The more the output resembles the prompt, the better the n-gram predicts it —
   coverage rises toward 1.0, draft lengths grow, and throughput climbs.

It is a feedback loop, and the verify is exactly the component that is supposed
to break it. It cannot, because it does not reproduce the target's own argmax.

## How the n-gram reaches the logits: through the KV cache

The n-gram never touches the logits. The path is the KV cache, and it is
measurable — holding FP16 state and speculation fixed and varying only the KV:

    kv=kvarn4   g1 echo=0.000   g2 echo=0.872   (264 tok/s)
    kv=fp32     g1 echo=0.026   g2 echo=0.000   (14 tok/s, oracle path)

An exact KV removes the loop entirely, with everything else unchanged.

The route: drafted tokens are INPUT POSITIONS of the verify forward. The model
runs over `[seed, d1, d2, ...]`, computes hidden states for all of them, and
writes their K/V into the cache. Positions whose drafts are rejected must have
that K/V undone. If the undo is not exact, later attention reads keys for tokens
the model never emitted — which, for an n-gram seeded on the prompt, are prompt
tokens. That biases subsequent logits toward the prompt, which is the backfeed.

KVarN makes the undo structurally hard, which is the likely reason fp32 is the
only clean mode. Its K records are **block-granular**: one var-norm record per
128-token block. Writing a speculative token into a block and sealing it computes
that block's scales over data that includes the draft. Removing the token
afterwards does not restore the previous scales unless the whole record is
restored, so a rejected draft can perturb the stored K of tokens that were
already committed.

That one mechanism is consistent with every other observation in this
investigation:

- KV **bit width** barely matters (2/4/8-bit all ~6.18e-1) — the error is in the
  scales, not the mantissa.
- **Batched vs per-token prefill** diverges from the first multi-chunk prompt —
  the two paths seal blocks on different schedules.
- **spec != AR** at every quantised width, and byte-identical at fp32.
- Swapping the two **rollback replay implementations** changed nothing, which
  earlier reads as "rollback is not involved". It does not: both replays restore
  the same thing, and if neither restores the block's prior var-norm record, the
  choice between them is irrelevant. That elimination was wrong and is retracted
  here.

### A single speculative token is enough

Holding FP16 state and kvarn4 fixed and sweeping only the verify width:

    b=2    g1 echo=0.000   g2 echo=0.872    99 tok/s
    b=4    g1 echo=0.000   g2 echo=0.897   168 tok/s
    b=16   g1 echo=0.000   g2 echo=0.872   309 tok/s

`b=2` is one drafted token per step, and it already corrupts as thoroughly as
`b=16`. The corruption level is flat across widths; only the throughput scales
with them, because a wider block emits the echoed text faster once the loop is
running.

This rules out the narrower version of the block-scale theory — that the damage
needs a speculative write to CROSS a 128-token boundary and seal a block early.
One token does it, so whatever fails to restore is touched by any speculative
write, not only by a block completion. The open block's var-norm statistics
being recomputed on every write fits; a boundary-crossing requirement does not.

## Measured with a cache probe: the RECORDS are clean, the WINDOW is not

`HIPFIRE_KVARN_ROLLBACK_PROBE=1` (added with this work) hashes the KVarN sealed
records and the trailing window across a speculative step — before the verify
writes, and after rollback — and counts how many window rows changed. 137
speculating steps on the echo prompt:

**The block-scale theory is REFUTED.** Records changed in 5 steps, and every one
of those completed a 128-token block:

    record changes with no block completion: 0

So sealing behaves correctly and rejected drafts do not perturb an already-sealed
record. (An earlier version of this probe reported 2 spurious changes; its
completion test was `(pos/group) != ((pos+b-1)/group)`, which for `b=1` can never
fire even when the written position IS the last of its block. A block completes
when `p % group == group - 1`. The two "spurious" cases were positions 1663 and
1791 — both exactly that.)

**The window retains rejected rows.** Rows changed per step is `b` in every
configuration, independent of how many tokens were committed:

| b | accept | rejected | committed | window rows written |
|---|---|---|---|---|
| 2 | 0 | 1 | 1 | 2 |
| 2 | 1 | 0 | 2 | 2 |
| 3 | 0 | 2 | 1 | 3 |
| 17 | 0 | 16 | 1 | **17** |
| 17 | 16 | 0 | 17 | 17 |

(The raw counter reads `2b` because the window is F16 held in an F32-typed
buffer, so one real row occupies half the f32 slots the probe strides by.)

The bottom row is the point: a step that committed ONE token wrote seventeen
window rows and left sixteen of them — the rejected drafts' K — staged in the
buffer. Nothing clears them.

That is a concrete, reproducible asymmetry, and it is the only KV carrier left
standing: the records are clean, and an exact (fp32) KV — which has neither
records nor window — removes the corruption entirely.

**Still not proven**: that those stale rows are READ. The window is a staging
ring, so if attention is masked to the committed length they are inert. Proving
the loop end-to-end means showing a read past the committed length — which is
the next step, and is now a much narrower question than when this started.

## Why this matters more than the losslessness bug reads on its own

"Speculative output differs from AR output" sounds like a numerical footnote.
This is what it cashes out as: a 400-token answer that is 87% copied from the
question, produced deterministically, while the throughput counter reports a
3.4x speedup. Both the quality metric and the performance metric mislead in the
same direction at the same time.

It also invalidates a benchmark. The 328/397 tok/s recorded in
`2026-09-01-ngram-spine-discarded-by-block-fallback.md` was measured under FP16
state with n-gram speculation on — the exact configuration above. That figure is
marked indicative in its own document; this is why.

## Mitigation in place, and its limit

`2026-09-02-deltanet-redundancy-gate-had-no-caller.md` restores the guard that
forces FP32 state below a redundancy floor of 3000. The 0.8B measures 2048, so it
now gets FP32 and does not degenerate (0 of 6 above).

**That is a mitigation, not a fix, and it does not cover every model.**
`Qwen3.6-35B-A3B` measures 4096, above the floor, so it still runs FP16 state. It
was not tested here — the loop requires a model, a prompt long enough to seed the
n-gram, and enough tokens to fall in. Whether models above the floor exhibit the
same loop is **open and worth checking before trusting n-gram speculation on
them**.

The real fix is upstream: make the verify reproduce AR. While it does not, any
draft source that can propose prompt text can bias the output toward it.

## Reproduction

    # degenerates (g2 echoes 87%)
    HIPFIRE_NGRAM_SPEC=1 HIPFIRE_DN_STATE_FP32_BELOW=0 hipfire daemon < echo.jsonl
    # does not
    HIPFIRE_NGRAM_SPEC=1 hipfire daemon < echo.jsonl
    # does not (speculation off)
    HIPFIRE_NGRAM_SPEC=0 HIPFIRE_DN_STATE_FP32_BELOW=0 hipfire daemon < echo.jsonl

`echo.jsonl` loads the model and issues two identical 400-token greedy generates
whose prompt is `"Here is a Rust module:\n\n"` plus 6151 chars of
`crates/hipfire-specdecode-ngram/src/hot.rs`. The first generation never echoes;
the second always does. Something carried between requests is required to enter
the loop — identifying it is the next step.
