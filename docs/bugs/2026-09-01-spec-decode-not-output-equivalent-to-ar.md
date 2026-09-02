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

## Most likely cause: the drafted positions are visible to earlier rows

The batched verify evaluates seed + drafted positions in one forward. If the
causal mask is wrong, row *i* can attend to drafted position *j > i*, and then
the logits at *i* — and hence the token committed there — depend on tokens that
were only guesses. That reproduces every observation above, including a
single-token draft being sufficient and the dependence on store warmth.

**This exact bug class has occurred in this repo before**: "Routed KVarN prefill:
rows in a wrapped block attend to FUTURE tokens"
(`docs/bugs/2026-08-29-kvarn-routed-prefill-window-wrap.md`, fixed 2026-08-29) —
same shape, different code path.

Not yet confirmed; the next step is to dump the target's logits at the first
committed position of one window with two different drafted suffixes. If they
differ, the mask is leaking and this is proven.

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

1. Confirm the `repeat_penalty` mismatch and make the speculative path use the
   same sampling parameters as the AR path.
2. Add a gate asserting AR/spec output equivalence at several block widths on a
   tiny fixture. `tests/tiny-state-gate.sh` hashes decode output already, but it
   does not vary the block width, so it cannot see this.
3. Only then re-run any block-width throughput comparison.

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
