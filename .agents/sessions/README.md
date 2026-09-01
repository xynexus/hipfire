# `.agents/sessions/` — one directory per startable piece of work

Each subdirectory holds a `START.md`: a brief scoped so a fresh agent session can
begin without re-deriving context. One session, one objective.

These are **not** a backlog to work through in order. Pick one, do it, and either
delete the directory or leave a `RESULT.md` beside the brief saying what happened
— including if the answer was "this item does not survive contact with
measurement", which is a real outcome and worth as much as a fix.

## State as of 2026-09-02

Every brief here now carries a `RESULT.md`. Four were done, two were **refuted**,
two remain open with corrected premises. Read the `RESULT.md` before the
`START.md` — three of the briefs are wrong in ways that matter.

| session | outcome |
|---|---|
| `tiny-state-hash-severity` | **DONE** — token_hash blocks, mamba2 logit_hash advisory; dead `rel_tol` deleted |
| `repair-dflash-drafters` | **DONE, and narrower than briefed** — 6 repaired, 2 verified working (τ 3.0 / 2.0 vs a 2.1 control); gemma-4 has NO DFlash runtime path |
| `gate-vacuity-sweep` | **DONE** — found a live one: a tripwire FAIL was escalating into a gate that cannot fail on drift |
| `moe-expert-allocation` | **DONE** (earlier) — −3.96 GiB, not the predicted −35.4 |
| `gtt-slab-suballocator` | **REFUTED** — step 2 doubles GTT while scoring a perfect ratio; step 3's prize is <0.2 GiB |
| `guidedquant-backward-capture` | **REFUTED AS SCOPED** — largely already built; the gap is a measurement, not a feature |
| `qwen4exp-batched-forward` | **OPEN** — premise re-measured and holds (0.114 s/token marginal) |
| `speculator-seam` | **PARTIALLY DONE** — first implementor landed and verified; the registry + daemon routing remain |

The pattern from the house rules held again: of the eight, **three briefs were
materially wrong about the state of the code**, and one proposed a change that
measurement showed would make things worse.

## What every brief must contain

- **Objective** — one paragraph, and what "done" means.
- **Why now** — with numbers, not adjectives.
- **Blocked on** — explicitly, or "nothing".
- **First moves** — concrete enough to run.
- **The verification bar** — how you will know it worked, chosen so it can FAIL.
- **Traps** — what has already burned someone here.

## House rules earned the hard way (2026-09-01)

A twenty-item ranked list was worked overnight. **Nine items did not survive**:
already done, already fixed, or wrong about what the thing was. One would have
made quality worse. The list was built from memory, and nothing invalidates a
memory when the code moves.

1. **Check the repo before analysing.** `git log --grep=<topic>` plus a sweep of
   `docs/todo`, `docs/plans` and `BUGS.md`. Two minutes per item would have
   caught all nine.
2. **Discard the first run after any rebuild, kernel edit, or new arm.** Kernels
   JIT-compile inside the timed window: measured 6.41 vs 22.13 tok/s — 3.45x —
   with tau BIT-IDENTICAL, so every quality metric looked healthy while
   throughput read a third of real. `dflash_spec_demo` now warns.
3. **Profile the phase you mean.** A whole-run profile divided by decode tokens
   attributed model LOAD to decode and produced a confident wrong lead.
   `HIPFIRE_COPY_REPORT=1` gives per-call-site attribution.
4. **Ask whether your check can fail.** Three gates this session looked
   configured and were not: `tiny-spec-gate` asserted a path it had itself
   disabled, the baselines' `rel_tol` column is read nowhere, and a new test
   sized its input from the constant under test so it held for any value.
5. **Successive benchmark runs are not independent.** A 48 GiB arm left 45 GiB
   resident and made the next arm read 40% slow with identical counters.
