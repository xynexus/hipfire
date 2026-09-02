# Speculative decode does not reproduce autoregressive output, and block width changes the result

Status: **OPEN — diagnosed, not fixed.** Found 2026-09-01 while trying to
establish whether the adaptive block controller beats a fixed block. It does not
just affect that question; it invalidates every throughput comparison taken
across different block widths, including ones already published in this repo.

## The invariant that is violated

Speculative decode's defining guarantee is that the verify makes it *lossless*:
the emitted sequence is whatever the target would have produced on its own,
regardless of how wide the draft block is. That is the entire reason a rejected
draft is safe. Here it does not hold.

## Evidence

gfx1103/nix2, `qwen3.5-0.8b--oq4++.hfq`, greedy (temperature 0), 2000 tokens,
same prompt every run.

**Fixed block width alone changes the output**, and each width is internally
deterministic:

| run | tokens | sha256 (first 16) |
|---|---|---|
| `spec_block=16`, run 1 | 1999 | `a4b97025a9676edc` |
| `spec_block=16`, run 2 | 1999 | `a4b97025a9676edc` ← reproducible |
| `spec_block=8` | 1999 | `3e86c3ab6361ff60` |
| `spec_block=4` | 1802 | `e0d59b9ac9c41cdf` |

**And no width reproduces plain AR.** With speculation off the model emits 1688
tokens, identically on both requests (`627ee38b76f44c28`). Every speculative
width diverges from it at the *same* character offset:

| arm | matches AR | first divergence |
|---|---|---|
| b=16 | no | char 1297 (req 1), 1707 (req 2) |
| b=8 | no | char 1297, 1707 |
| b=4 | no | char 1297, 1707 |

## What it is NOT

Ruled out by measurement, because the divergence offset does not move:

- **DeltaNet state precision.** `HIPFIRE_DN_STATE_FP16=0` (confirmed
  `dn_quant=FP32` in the run log) changes the sequences but not the divergence
  offset. Same 1297/1707.
- **KV-cache quantization.** `kv_cache=q8` — the highest-precision mode the
  schema offers — same 1297/1707. (There is no unquantized KV option; every
  value in the enum is quantized.)
- **Numeric drift generally.** Drift moves when you change precision. This
  offset is stable across every precision knob and every block width, which is
  the signature of a deterministic logic difference, not rounding.

## Ruled out: repeat_penalty (was the leading hypothesis — WRONG)

The speculative call site passes `1.0_f32, // repeat_penalty (off)` while the
schema's `repeat_penalty` default is 1.05, which looked like the answer: two
paths decoding different objectives would separate deterministically at a fixed
offset, invariant to precision, exactly as observed.

**It is not the cause.** The daemon does not use the schema default here. It
picks a per-ARCH default (`handlers/generate.rs:511`):

    let default_repeat_penalty = if m.arch_id == ARCH_ID_LFM2_MOE { 1.05 }
        else if m.arch_id == ARCH_ID_GEMMA3_VL { 1.3 } else { 1.0 };

`qwen3.5` is arch 5, so it takes the `else` branch: **1.0**, the same value the
speculative path hardcodes. Both arms decoded with the penalty off, and the
divergence is something else. (That the config field's 1.05 default reaches
almost no model is a separate inconsistency, noted below.)

## Refined diagnosis: the DRAFT CONTENT changes committed tokens

Re-measured 2026-09-01 after a reboot, 800 tokens, two identical requests per
run, `qwen3.5-0.8b--oq4++.hfq`, greedy.

AR is deterministic and request-order independent: g1 and g2 are byte-identical
(`dbd086d69db0`, 800 tokens each). Speculation is not:

| block | g1 first divergence from AR | g2 first divergence from AR |
|---|---|---|
| b=2 | token 49 | **token 1** |
| b=4 | token 49 | **token 1** |
| b=16 | token 49 | **token 1** |

Two facts narrow this sharply:

1. **`b=2` diverges at exactly the same token as `b=16`.** `b=2` is the
   narrowest possible speculation — a single drafted token. So this is not a
   batch-width effect and not accumulated drift; a one-token draft is enough.
2. **g1 and g2 differ from each other**, on an identical prompt, with AR
   identical across both. The only thing that changed between them is the
   n-gram store, which is warm by g2. So the DRAFT CONTENT is changing what
   gets COMMITTED.

The text confirms it is not a near-tie numeric flip:

    AR   : 'The' ' module'   ' implements' ' a' ' **' 'RAM' ' staging'
    b=16 : 'The' ' provided' ' code'       ' defines' ' a' ' **' 'RAM' ' staging'

Different words, then the sequences re-converge.

That combination is diagnostic. A correct verify commits either the drafted
token (only when it equals the target's own argmax) or the target's argmax — so
the committed sequence cannot depend on what was drafted. Here it does.

## Also ruled out: the rollback replay path

`spec_step_dflash` has two rollback replay implementations and an env lever
between them. The batched one carries a strong precedent — measured on
Qwen3.8-27B/DFlash2 over a five-prompt greedy mix, it is **byte-identical to
plain AR** while the serial one diverges (see the doc comment on
`dflash_rollback_batched_prefill_from_env`). So the machinery demonstrably CAN
be exact.

Both settings were tried on the n-gram path. They diverge from AR **identically**
— same token, 49 in request 1 and 1 in request 2:

    HIPFIRE_DFLASH_ROLLBACK_BATCHED_PREFILL   (default, batched) -> diverges
    HIPFIRE_DFLASH_ROLLBACK_BATCHED_PREFILL=0 (serial)           -> diverges, same tokens

Since the choice of replay makes no difference, the divergence is upstream of
the rollback, not in it.

Note this run rejects heavily — `accepted=119` of `proposed=992`,
`mean_accept_len=0.19` — so the rollback path is exercised constantly and is
still not the discriminator. (`seed_oracle.rej_rate=0.0` in the same JSON is a
different metric — seed-position matching — and must not be read as "no
rejections".)

## Why this was undiscoverable until now

Worth stating, because it explains why a losslessness violation sat in a shipped
feature. Before `2026-09-01-ngram-spine-discarded-by-block-fallback.md`, the
drafter-free n-gram path never speculated at all: the spine was discarded every
step and `b` collapsed to 1, a plain AR step. A path that only ever performs AR
steps is trivially AR-equivalent. Fixing the spine turned on real speculation
for the first time and exposed this immediately.

The DFlash drafter path, which DID speculate, is the one with the AR-parity
measurement quoted above. So the guarantee was verified where it was exercised
and unverified where it was not.

## REFUTED: a causal-mask leak

The earlier hypothesis was that row *i* of the batched verify could attend to
drafted position *j > i*, so committed tokens depended on guesses. **Measurement
refutes it.**

`HIPFIRE_DFLASH_VERIFY_DEBUG=1` already dumps each verify block's inputs and its
per-position argmax, and its own comment states the invariant: "Slot 0's argmax
is the target's OWN next token for the already-committed prefix — it is what
plain decode would emit at `start_pos`". At one position, varying only the block
width:

    b=2   in=[760, 4536]                        argmax=[3766, 369]
    b=4   in=[760, 4536, 5004, 264]             argmax=[3766, 369, 264, 2972]
    b=8   in=[760, 4536, 5004, 264, ...]        argmax=[3766, 369, 264, 2972, ...]
    b=16  in=[760, 4536, 5004, 264, ...]        argmax=[3766, 369, 264, 2972, ...]

Slot 0 is `3766` in every case. At `b=2` the block holds only two tokens, and
slot 0 is already wrong; adding fourteen more changes nothing. **Later tokens do
not leak backward.** The mask is fine.

## Cleanest statement of the bug

`tests/spec-ar-equivalence-gate.sh` (added with this write-up) reduces it to one
line. 200 greedy tokens, same prompt:

    AR      69b9d6568590b620   <- reference
    b=2     45e403a7b86ebae3   FAIL
    b=4     45e403a7b86ebae3   FAIL
    b=16    45e403a7b86ebae3   FAIL

**Every speculative width produces the SAME sequence as every other, and all of
them differ from AR.** So the defect is not width-dependent — it is
speculation-versus-not, which is precisely the `b=1` / `b>=2` split the slot-0
probe shows. (Over longer runs the widths eventually drift apart from each other
too, but that is a secondary effect downstream of this one.)

## PROVEN: the batched forward disagrees with the single-token forward

The split is `b == 1` versus `b >= 2`, not block contents. At the same
`start_pos`, with the same committed prefix and the same slot-0 input token:

    b=1  (no speculation)  in=[760]   argmax=[4536]
    b>=2 (any width)       in=[760,…] argmax=[3766, …]

Slot 0 depends only on the prefix and its own input. It must be `4536`. Batching
the forward changes it. Per the probe's own comment, that "localizes the
miscompute to the verify forward itself rather than to acceptance or rollback",
which is consistent with rollback having already been eliminated above.

Prefill-cache reuse is not the explanation: with speculation off, two identical
back-to-back requests are byte-identical, so re-running a prompt does not by
itself change output.

## ROOT CAUSE: quantised KV makes the batched and per-token forwards disagree

`crates/hipfire-runtime/examples/compare_prefill_hidden_paths` runs the BATCHED
and PER-TOKEN forwards in one process against the same `HiddenStateRingBuffer`
and diffs them layer by layer. On the real model, above the prefill chunk size
so attention actually reads the KV cache:

    --n 512, qwen3.5-0.8b--oq4++.hfq
      kv-mode fp32          IDENTICAL across all layers (worst 0.00e0)
      kv-mode q8            FIRST DIVERGING LAYER: 0   (worst overall 5.24e-1)
      kv-mode kvarn 2-bit   FIRST DIVERGING LAYER: 0   (worst overall 6.18e-1)
      kv-mode kvarn 4-bit   FIRST DIVERGING LAYER: 0   (worst overall 6.18e-1)
      kv-mode kvarn 8-bit   FIRST DIVERGING LAYER: 0   (worst overall 6.18e-1)

With an unquantised KV the two paths are bit-identical. With any quantised mode
they diverge **at layer 0** by 0.5-0.6 relative — an order of magnitude over the
5e-2 ceiling this repo's own prefill gate applies.

### It is NOT quantisation error

The KVarN width is swept via `HIPFIRE_KVARN_BITS` (2/4/8; the probe reads
`KvCache::kvarn_bits_from_env()`, and bare `--kv-mode kvarn` means **4-bit**, the
shipping default — an earlier revision of that probe hardcoded 8 and silently
compared 8-bit against 8-bit, so the width must always be stated).

Quantisation error would fall by roughly 64x from 2-bit to 8-bit. It does not
fall at all:

| layer | KVarN 2-bit | KVarN 8-bit |
|---|---|---|
| 0 | 3.68e-1 | 3.68e-1 |
| 1 | 3.11e-1 | 3.11e-1 |
| 2 | 3.32e-1 | 3.32e-1 |
| 3 | 4.89e-1 | **5.46e-1** (worse) |
| 4 | 3.33e-1 @row 494 | **3.79e-1 @row 220** |
| 5 | 6.18e-1 | 6.18e-1 |
| 6 | 4.96e-1 | 4.96e-1 |

The widths are genuinely taking effect — layers 3 and 4 move — so this is not a
stuck env var; the identical headline is just the maximum landing on layer 5
either way. But eight bits is no better than two, and layer 3 is *worse* at 8.

So the batched and per-token paths are computing genuinely DIFFERENT QUANTITIES
whenever the KV goes through a quantised cache, not the same quantity to
different precision. That rules out the obvious workaround: **widening the KV
does not restore losslessness.** Only `fp32` does, and `fp32` is not in the
`kv_cache` enum.

That is the mechanism, end to end:

1. A speculative verify is a BATCHED forward; plain AR decode is PER-TOKEN.
2. Under quantised KV those two forwards do not agree.
3. So the token committed at slot 0 differs between `b=1` and `b>=2` — exactly
   what the verify probe shows (4536 vs 3766).
4. So speculative output diverges from AR output.

**And every KV mode the daemon can select is quantised.** The `kv_cache` schema
enum is `auto, q8, asym2, asym3, asym4, kvarn2, kvarn, kvarn4, kvarn8` — there is
no fp32 entry, so the one configuration where the invariant holds is not
reachable from config. The probe reaches it only through its own `--kv-mode`
argument.

This reframes the FP16-DeltaNet finding below: state precision changes how often
the disagreement flips a token, but the KV path is where the disagreement comes
from.

## The gate that should have caught this was measuring nothing

`tests/tiny-prefill-gate.sh` asserts exactly the right invariant — "the prefill
paths must be exactly equivalent absent KV quantisation", with a 5e-2 ceiling for
quantised modes. It ran the probe at `--n 32`.

The prefill chunk size is 256. The probe prints, on its own initiative:

    WARNING: --n 32 <= the prefill chunk size (256), so prefill runs as ONE
    chunk, attention never reads the quantised KV cache, and every --kv-mode /
    HIPFIRE_KVARN_BITS will return IDENTICAL numbers.

So every per-KV-mode row was measuring a configuration in which the KV cache is
never read. `hidden q8: worst 0.00e0` meant "not measured", not "no divergence",
and the ceiling could never fire. This is the same failure the file warns about
three comments earlier — an unmeasured invariant reported as a pass — reached by
a different route.

Fixed here: `--n` is now 300 (above the chunk, below the tiny fixtures' context —
512 panics them), a degenerate run FAILS instead of reporting zeros, and the size
is overridable via `HIPFIRE_TINYPREFILL_HID_N`. The gate immediately found two
real violations that had been invisible:

    qwen3_5              fp32 0.00e0   q8 5.75e-3   kvarn 3.35e-2    (pass)
    qwen3_5_moe_indexed  fp32 0.00e0   q8 7.16e-2   kvarn 7.40e-2    (FAIL, ceiling 5e-2)

Total failures went 6 -> 2, and the 2 are real findings rather than gate defects.

## Primary contributor: FP16 DeltaNet state

Re-run with `HIPFIRE_DN_STATE_FP16=0` (log confirms `dn_quant=FP32`), same
position, same probe:

    b=1  in=[1919]                    argmax=[4536]
    b=4  in=[1919, 4536, 5004, 264]   argmax=[4536, 5004, 264, 2972]

Slot 0 now **agrees** with the single-token forward, and the whole block
verifies cleanly (full acceptance). This is the mechanism the prefill path
already warns about: "the per-token fallback rounds the FP16 DeltaNet state once
per token where the batched path rounds once per chunk". A speculative block is
a chunk; `b=1` is a token.

**But FP32 does not fully close it.** End to end at 800 tokens, spec still
diverges from AR — first difference moves from token 49 to 74 (request 1) and
from token 1 to 18 (request 2). So FP16 state is a major contributor and there
is at least one more source of batched/single-token disagreement still
unidentified.

## How much FP32 actually buys, and the residual case

Method: run two identical requests with the n-gram on. The first is cold, so
most windows miss and run at `b=1`; the second is warm and runs wide. Where both
requests verify the SAME `start_pos` with the SAME slot-0 input token but
different `b`, slot 0's argmax must agree.

| DeltaNet state | comparable positions | slot-0 mismatches |
|---|---|---|
| FP16 (shipped default) | 2 | **2 (100%)** |
| FP32 | 7 | **1 (14%)** |

FP32 cuts the rate hard but does not reach zero, which matches the end-to-end
result (divergence moves later, from token 49 to 74 and 1 to 18, rather than
disappearing).

The residual case, reproducible and specific — a starting point for the
remaining work:

    start_pos=1690   in slot0 = 2972
      b=1  -> slot0 argmax 36514
      b=4  -> slot0 argmax   608

Sample sizes are small (2 and 7 comparable positions in a 100-token run) because
a position only becomes comparable when the cold and warm requests happen to
verify it at different widths. A longer run, or a probe that forces both widths
at every position, would tighten these numbers; the FP16 result is unambiguous
regardless.

## The FP16 default contradicts its own documentation

Worth fixing regardless of the above, because it decides which path users get:

| source | claim |
|---|---|
| `qwen35/state.rs:88` | "State precision is FP32 unless `HIPFIRE_DN_STATE_FP16` opts in." |
| `default_state_quant` doc comment | "**FP32 is the DEFAULT and the numerical reference; FP16 is opt-in.**" |
| `config/schema.rs:656` | `deltanet_state_precision` default is `Some("fp16")` |

`dn_state_precedence` resolves `!cfg.eq_ignore_ascii_case("fp32")`, so the
shipped default is FP16 and FP32 is the opt-in — the inverse of what both
comments state. The comments also record the measurements that were used to
choose FP32 ("This is the reading that kept the default at FP32"), so the
default appears to have moved without those comments following.

Given FP16 state demonstrably breaks slot-0 equivalence under speculation, the
default deserves re-deciding on purpose rather than by drift.

## Why this matters beyond correctness

Any benchmark comparing block widths is comparing different generated text.
Sequences differ in how predictable they are, so tok/s differences of tens of
percent can be an artifact of which text was produced rather than how fast it
was produced.

Concretely, this repo published such a comparison earlier the same day: the
claim that the adaptive controller cost 10-30% versus a fixed block came from
runs whose outputs differed (one arm emitted 1943 tokens, the other 1999). That
claim is **withdrawn** in
`docs/bugs/2026-09-01-spec-block-controller-and-naming.md`. On the single
comparison that was valid — a warm request where both arms emitted a
byte-identical sequence — the controller matched the fixed block, 338.7 vs 343.9
tok/s, at better verify efficiency (0.78 vs 0.58).

The reasoning error is worth naming: a self-optimizing search whose range
CONTAINS the fixed value cannot legitimately lose to it. When it appears to,
either the search is broken or the measurement is. Here both were.

## Fix direction

1. **Find the remaining batched/single-token disagreement.** FP32 state moves the
   divergence later but does not remove it. Bisect with the existing probe: run
   at FP32 and find the first `start_pos` where a `b>=2` slot-0 argmax differs
   from what `b=1` gives at that position, then narrow to the layer whose output
   differs between a width-1 and width-N forward at the same position.
2. **Re-decide `deltanet_state_precision`'s default deliberately**, and make the
   comments and the schema agree whichever way it goes.
3. **Add a gate.** `tests/tiny-state-gate.sh` already hashes decode output, but
   it does not vary the block width, so it cannot see this. The missing
   assertion is: for a fixed prompt, output is identical at b = 1, 2, 4, 16.
   That single check would have caught this the day the drafter-free path
   started speculating.
4. Only then re-run any block-width throughput comparison.

## Tooling note

No new instrumentation was needed: `HIPFIRE_DFLASH_VERIFY_DEBUG=1` already
existed and already documented the exact invariant this violates. It had simply
never been pointed at the drafter-free path, because until the spine-discard fix
that path never produced a block wider than 1.

## Reproduction hazard: this wedges the GPU

The last run left the daemon unkillable:

    tid 2448403 state=Z   (defunct)
    tid 2448404 state=D   wchan=__flush_workqueue

    dmesg: Workqueue: kfd_process_wq kfd_process_wq_release [amdgpu]
             __flush_workqueue -> kfd_process_wq_release   (hung task)

The amdgpu KFD teardown hangs, so the process cannot finish exiting and never
drops its `flock` — `hipfire lock status` reports the GPU busy and `lock kill`
cannot help, because the holder is genuinely alive in D state, not a stale
holder line. No signal reaches a D-state task; this needs a GPU reset or a
reboot. Whether the wedge is caused by the same defect or is the independent
gfx1103 hazard (README §"gfx1103 / Phoenix LDS status") is unknown.
