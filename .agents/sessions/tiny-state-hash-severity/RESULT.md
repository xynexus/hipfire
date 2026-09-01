# RESULT: tiny-state hashes split by severity — DONE 2026-09-01

## What changed

`tests/tiny-state-gate.sh`:

- **`token_hash` is now the blocking assertion for every family.** A token-stream
  difference FAILs unconditionally — that is the behavioural claim the gate exists
  to make.
- **`logit_hash` is advisory for families listed in `logit_is_advisory()`**, and
  only when `token_hash` still matches. Today that list is `mamba2` alone. For
  every other family a logit difference still FAILs, exactly as before.
- The dead `rel_tol` column was **deleted**, not wired. It was `$6`, every row set
  it to `0`, and `grep -c '\$6'` on the gate returned 0. The env-field scan now
  starts at `$6`, which parses both the old rows (`... 0 hip=`) and the new ones
  (`... hip=`), so nothing was invalidated; `tests/tiny-state-baselines.txt` had
  the column stripped from all 36 rows and the header updated.
- Summary lines report the advisory count so a downgraded cell is visible rather
  than silent.

## Why mamba2 and nothing else

Mamba-2's SSM state IS the accumulator: fp16, re-rounded every kernel invocation,
no error feedback. `logit_hash [exact]` asserts a stability property that
arithmetic does not have — the cell has taken four values across two
architectures with no code change. The other twelve `fp16`-anchor families are
stable and stay exact. The predicate carries that criterion as a comment: add a
family only with a documented instance, never on suspicion.

## Verification — including the half that had to fail

Full 18-cell run: **PASS (17 cells, 1 advisory logit drift)**. mamba2 reports

    ADVISORY logit drift (token stream identical): observed 0x41484800e3bd1e1f baseline 0x1eab04c5f2756c24 [exact]

Two negative controls, run by corrupting baseline rows and confirming red:

| corruption | expected | observed |
|---|---|---|
| mamba2 `token_hash` -> `0xdeadbeef...` | FAIL (advisory family, token still blocking) | `FAIL token drift` |
| qwen2 `logit_hash` -> `0xcafebabe...` | FAIL (non-advisory family) | `FAIL logit drift` |

`tiny-state-gate: FAIL (2 cell(s))`. Both halves of the bar met: the downgrade
does not disable the gate for the family it applies to.

## Not done

`docs/bugs/2026-09-01-mamba2-tiny-state-drift.md` stays open. This changes how the
gate REPORTS the cell; it does not explain why the value moved. The drift is still
unexplained and still worth explaining — it is now advisory, not invisible.
