# BUG: `mamba2/fp16` logit-hash drift — a fragile cell asserted exactly

**Status:** DIAGNOSED 2026-09-01 on halo (gfx1151). Not a behaviour change — the
token stream is identical and only the logit hash moved. The cell is a known-
fragile one and the gate cannot express that. See the recommendation below.

    observed  0x41484800e3bd1e1f / 0xea870c172ccfeef5
    baseline  0x1eab04c5f2756c24 / 0xea870c172ccfeef5   [exact]

Note the **second hash matches**; only the first differs.

## Why this is worth a file rather than a re-record

The cell was **GREEN several times earlier the same session**, on the same
commit, with the same command — `tiny-state-gate` reported
`OK fp16: matches baseline (0x1eab04c5f2756c24/...)` repeatedly while landing
unrelated work. It then went red and has stayed red. **17 of 18 cells still
match**, so the harness and the other families are fine.

Re-recording the baseline would bury that. A cell that changes value with no
input change is either a latent race or an environmental dependency nobody has
named, and both matter more than the number.

## Eliminated, with the test that eliminated each

| hypothesis | test | result |
|---|---|---|
| my working-tree changes | A/B: reverted `hfq_cli.rs`, rebuilt, re-ran | **identical drift both ways** |
| a commit in the 38 pulled from origin | checked out `7f739be21` (pre-batch master, my branch's own base) and rebuilt | **still drifts** |
| PR #396 (`fix/bug-sweep`) — the only one touching rdna/runtime | checked out `07498cb31^1` | **drift already present** |
| PR #398 | changed files are CLI/coexistence/docs only | cannot reach mamba2 |
| stale/corrupt JIT kernel cache | moved `~/.hipfire/kernels/gfx1151` aside; 673 kernels recompiled from scratch | **still drifts** |
| build-layout-dependent race | forced two relinks, re-ran each time | **same value all three times** |
| nondeterminism per run | ran the cell 3x on one binary | **identical every run** |
| stale cached fixture | gate emits into a `mktemp` dir with `--seed 42`, cleaned on exit | no caching to blame |
| HIP/ROCm upgrade | baseline records `hip=7.14.60850-d34cbb6409`; current is the same string | **identical** |
| baseline edited underneath us | `git log -- tests/tiny-state-baselines.txt` — untouched by the pulled range | unchanged |

So: same source, same HIP, same box, same seed, deterministic across runs AND
across rebuilds — and different from what the identical configuration produced
hours earlier.

## RESOLVED as far as it can be: it is logit-only, on a known-fragile cell

Three facts close this out, none of which needed a reboot.

### 1. The token stream is IDENTICAL — only the logits moved

The gate stores `logit_hash token_hash`. The mismatch is the FIRST field;
`token_hash` matches exactly. So greedy decode produced the same tokens and the
difference is sub-threshold float noise that never crossed an argmax. This is a
numerical-stability observation, **not a behaviour change**, which is a very
different severity from how the red line reads.

### 2. This cell has a documented history of exactly this

From the gate's own header:

> mamba2/fp16 took three different values in 15 days with no code change
> between them.

That was attributed to two gfx1103 hosts (nix1/nix2) sharing one baseline row,
and fixed by keying rows on `hip=`. **That explanation cannot apply here**:
single host, gfx1151, and the row's `hip=7.14.60850-d34cbb6409` matches what the
box reports. So this is a fourth instance with the previous cause excluded.

### 3. mamba2 is the only cell whose state is a recurrent fp16 accumulator

Twelve families share the `fp16` anchor (qwen2, dots_ocr, gemma3, gemma4_*,
qwen3_5*, …) and all twelve are stable. What distinguishes mamba2 is not the
anchor format but that Mamba-2's SSM state IS the accumulator, held in fp16 and
re-rounded on every kernel invocation. `BUGS.md` measures that shape directly for
the DeltaNet analogue: round-to-nearest with no error feedback, and divergence
that grows **superlinearly** (13x more error for 2.5x more tokens).

A quantity like that does not have a stable low bit. Pinning its hash with
`[exact]` asserts a property the arithmetic does not have.

## The gate cannot express "compare this cell loosely" — the knob is dead

The baseline format advertises a tolerance:

    # gpu_arch family format logit_hash token_hash rel_tol

Every row carries `rel_tol=0`, and **`$6` is read nowhere in the gate** (`grep -c
'\$6'` returns 0). The parser takes `$4" "$5` and drops the rest, then compares
both hashes with string equality. The column looks like a configured tolerance
and is inert — the same shape as the two vacuous gates already fixed this session.

### Recommended fix, in preference order

1. **Split the two hashes by severity.** `token_hash` is the behavioural
   assertion and should stay exact and blocking. `logit_hash` is a numerical
   tripwire; on families whose state is a recurrent low-precision accumulator it
   should warn, not fail. That is one `if` and it makes the gate say what it
   means.
2. Failing that, wire `rel_tol` so it does something, or delete the column so it
   stops implying a knob that does not exist.
3. **Do not simply re-record.** The value has moved at least four times across
   two architectures; the next re-record buys days, not correctness.

## Reproduce

```sh
./tests/tiny-state-gate.sh 2>&1 | grep -A1 "tiny-state: mamba2"
```

## Note on the gate's behaviour

The failure **does not block a commit**. `tiny-affected-gate` escalates to the
coherence battery, that battery reports "no hard errors" (it only fails on hard
errors, not output changes), and the pre-commit hook then prints "Both gates
passed. Proceeding with commit."

So a red state cell is advisory in practice. That is the same fail-open shape as
the `tiny-spec-gate` bug fixed in `f212ae076`, and it is why this drift reached
`master` without anyone being stopped.
