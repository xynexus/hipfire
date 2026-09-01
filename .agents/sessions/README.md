# `.agents/sessions/` — one directory per startable piece of work

Each subdirectory holds a `START.md`: a brief scoped so a fresh agent session can
begin without re-deriving context. One session, one objective.

These are **not** a backlog to work through in order. Pick one, do it, and either
delete the directory or leave a `RESULT.md` beside the brief saying what happened
— including if the answer was "this item does not survive contact with
measurement", which is a real outcome and worth as much as a fix.

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
