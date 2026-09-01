# RESULT: gate vacuity sweep — 2026-09-02

33 gates in `tests/`, ~9k lines. Swept mechanically for the shapes the brief
names, then **demonstrated red** on everything I changed. One live, serious
instance found; the rest of the surface is in better shape than three findings
in one night suggested.

## The finding: a FAIL was being laundered into a pass, in the hook itself

`.githooks/pre-commit` routed the tiny-affected tripwire's exit codes like this:

    1) echo "tiny affected: FAIL → escalating to the full coherence battery." ;;

And `coherence-gate.sh` says of itself, in its own header:

> The gate itself only fails on hard daemon/error signals (panics, non-zero
> exit, zero tokens emitted); correctness is assessed qualitatively on the
> report.

So a tripwire FAIL — **KLD drift / crash / non-finite**, per the hook's own
comment — escalated into a gate that *structurally cannot go red on drift*. The
battery came back clean, the hook printed a passing summary, the commit
proceeded. This is the same mechanism the brief credits with letting
`tiny-spec-gate` stay broken on `master` for its whole life.

Exit 1 is a **verdict**, not an absence of one. Exits 2 and 3 (couldn't run /
no coverage) are absences, and escalating those is correct — that half is
unchanged.

**Fix:** the battery still runs, because its report is useful evidence for
telling a regression from an intended numerical improvement. It no longer gets
to overrule the verdict. The commit is blocked, with a message saying how to
accept a deliberate change (re-record the cell's baseline in the same commit).

**Proof it can fail** — `tests/precommit-escalation-selftest.sh`, no GPU, now in
`no-gpu-ci.sh`. It extracts the hook's REAL verdict region between two markers
(so it cannot rot into a copy of itself), runs it against stub gates where the
battery always exits 0, and asserts all four routes:

    OK  tiny exit=0 -> hook exit=0  (pass skips the battery)
    OK  tiny exit=1 -> hook exit=1  (FAIL BLOCKS even with a clean battery)
    OK  tiny exit=3 -> hook exit=0  (inconclusive escalates, does not block)
    OK  tiny exit=2 -> hook exit=0  (could-not-run escalates, does not block)
    OK  negative control: without the fix, tiny exit=1 -> hook exit=0 (the bug)

The last line is the important one: the test reverts the single line carrying
the fix and requires the original bug to reappear. If it ever stops
reproducing, the test fails loudly rather than passing vacuously.

## Also fixed

- **`tiny-state-baselines.txt` `rel_tol` was dead** — `$6`, every row `0`, read
  nowhere. **Deleted**, not wired (see below), column stripped from all 36 rows,
  env-field scan moved to `$6` so old and new rows both parse. Shipped
  separately with the tiny-state severity split.
- **`dflash_spec_demo` had no arch check.** It bypassed
  `load.rs::require_arch_feature`, so an unsupported target died as `tensor not
  found: layers.0.mlp.gate.weight` from inside the *qwen35* loader. Now refuses
  by name. Tested both ways.
- **`SOFT_WARN_TOTAL` in `agentic-gate.sh`** — initialised, never incremented,
  never read; the real count is `soft_count` from a `grep -c` at the end.
  Cosmetic, but it is the "looks configured" smell. Deleted.

## Why `rel_tol` was deleted rather than wired

`tiny-quant-baselines.txt` advertises the same-looking column — and **it is
live**: `executor_tinyquant.rs:628` computes `budget = (tol * b).max(ABS_FLOOR)`
and all 334 rows carry `0.25`, a real 25% tolerance. Two files that look alike,
one knob real and one inert. Wiring the inert one would have invented a
tolerance policy for hashes, which cannot have one — a hash is equal or it is
not. Deleting says so.

## Checked and found healthy — with what convinced me

| gate / assertion | would it go red? |
|---|---|
| `tiny-spec-gate` (instance #1, fixed `f212ae076`) | **demonstrated** — ran it: `checkpoint engaged 31x, output == AR` |
| `tiny-state-gate` | **demonstrated** — corrupted both hash kinds, both went red |
| `qwen4exp-gate` paged arm | already carries the guard: *"paged matches resident proves nothing if nothing was paged"* — asserts cold loads and evictions |
| `tiny-affected-gate` | hardened, with a comment naming the hole it closed (exit 3 used to skip later gates) and its own `-selftest.sh` |
| `coherence-gate-dflash` | `"ok": checked > 0 and not mismatch_counts` — explicit vacuity guard |
| `calibration.rs` `DEFAULT_CALIB_SEQ_LEN` test (instance #3) | fixed: expectation pinned to the literal `2048` |
| `tiny-quant` `rel_tol` | live, `0.25`, used in the budget |

## Mechanical sweeps that came back clean

- **Assigned-never-read variables**, all 33 gates. After excluding exported env
  consumed by children, Python kwargs in heredocs, and `IFS=` for `read`: only
  `SOFT_WARN_TOTAL`. The four `gfx1103-lds-tail-snop-repro.sh` "knobs"
  (`VARIANT`/`MODE`/`N_LAUNCH`/`K_LIMIT`) and `mi300x`'s `allow_patterns` are
  false positives — env prefixes to `$RUNNER` and a `snapshot_download` kwarg.
- **Every baseline file's advertised columns vs the fields actually parsed.**
  Four files; only tiny-state's `rel_tol` was inert.
- **Gates with no `exit 1`** — eight matched, all false positives (`exit
  "$status"` / `exit "$fail"`).
- **Env-gated counter assertions** (the instance-#1 shape). One match:
  `tiny-spec-gate:137`, and it is an acknowledged gap that the gate *prints*
  (`note: accept_len==0 on fixtures; slot indexing NOT covered here`). Honest,
  not vacuous.

## Known gap, left as-is

When the tiny tier returns exit 3 (no coverage for a touched family), the only
thing standing between the change and a commit is the coherence battery — which
still only detects hard errors. That is by design, not a defect, but it means
**"gates passed" on an uncovered family means "did not crash"**. Closing it is
coverage work (`docs/model-support.toml` families with no tiny cell), not gate
work, so it is not in this session's scope.

## Coverage, stated honestly

All 33 gates were swept mechanically. Red was **demonstrated** for the three
things I changed and for `tiny-spec-gate`. I did not break-test the other ~29
individually — that is days of GPU time — so this narrows the surface, it does
not certify it.
