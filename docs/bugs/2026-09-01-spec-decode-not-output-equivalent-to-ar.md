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

## Leading hypothesis (not yet confirmed)

The speculative call site passes sampling parameters the AR path does not:

    crates/hipfire-serving-core/src/generate.rs:1153
        1.0_f32, // repeat_penalty (off)
        0,       // repeat_window

while `repeat_penalty` is a schema field defaulting to **1.05**, and the other
generate paths forward `cfg.repeat_penalty`. If the AR path applies 1.05 and the
speculative path hardcodes 1.0, the two are not decoding the same objective, and
they would separate at the first token where the penalty flips an argmax —
deterministically, at a fixed offset, invariant to precision. That matches every
observation.

This would explain **spec vs AR**. It does not by itself explain **b=16 vs b=8**,
since both disable the penalty; that difference is probably ordinary numeric
path-dependence in the batched verify, which only becomes visible once the runs
have already separated. Both need confirming.

The confirming experiment — rerun AR and spec with `repeat_penalty` forced to
1.0 in the request — is written but **did not complete**: the run wedged the GPU
(see below).

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
