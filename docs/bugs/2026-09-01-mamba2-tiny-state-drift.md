# BUG: `mamba2/fp16` tiny-state drifts, and nothing in the source explains it

**Status:** OPEN, reproducible, cause NOT found. Filed 2026-09-01 on halo (gfx1151).

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

## What that leaves

Something outside the source tree and outside the kernel cache. Candidates not
yet tested, roughly in order of cheapness:

1. **Driver/GPU state.** The session ran many hours of GPU work, including two
   full kernel-cache rebuilds and repeated large allocations. A reboot is the
   cheap discriminator and has not been tried.
2. **An uninitialised read** in the Mamba-2 path whose value happens to be stable
   while the allocator hands back the same pages, and changed when the heap
   shape changed. This fits "stable now, different from before" better than a
   scheduling race does, and the class has precedent here —
   `2026-08-30-calibration-capture-nondeterminism.md` was a real data race in
   `zaya_value_compose_f32` found the same way, by a hash that would not sit
   still.
3. Something in `~/.hipfire` the gate reads that is not the kernel cache.

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
