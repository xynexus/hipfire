# AGENTS.md - tests and gates

This subtree owns enforcement wrappers, smoke tests, and CI gates. Keep shell
gates focused on orchestration and evidence collection; put reusable runtime
admission logic in `hipfire-eval` first.

## Gate Policy

- `./tests/tiny-affected-gate.sh --require-coverage` is the automatic GPU
  correctness front tier for covered runtime and quantization changes.
- `./tests/coherence-gate-dflash.sh` is a manual DFlash/DDTree diagnostic. Keep
  it runnable, but do not wire it into commit hooks or mandatory gate chains.
- `./tests/no-gpu-ci.sh` is the default no-GPU handoff check for workflow-only
  changes.
- GPU gates must acquire `hipfire lock` (`gpu-lock` is only an alias) unless they
  exclusively drive a daemon path that acquires resource leases itself.
  `hipfire-eval` is that exception and must **not** be wrapped: it loads through
  the daemon, which takes the lease itself, so wrapping deadlocks against your own
  holder — and the error names *your* label as the blocker, which reads like an
  unrelated job.
- Preserve byte-identical prompts by reading committed files from
  `benchmarks/prompts/`; do not inline new benchmark prompts in shell heredocs
  when the exact text matters.
- When adding or repairing model/runtime admission coverage, update
  `crates/hipfire-eval/` first and keep shell scripts as wrappers where useful.

## Measurement Traps

Standing warnings. Each of these has produced a *confident wrong conclusion* in
this repo, not merely a nuisance — they are recorded because the failure mode is
a measurement that looks clean.

- **A green gate answers "did the selected tests pass", not "did the tests pass".**
  `tiny-affected-gate.sh` derives a family allowlist from the touched paths, so two
  runs with different `--files-from` are running *different tests*. Comparing them
  as before/after is invalid. State which families a run selected before citing it.
- **Client-side reply order does not report service order.** Anything about
  scheduling, batching, or preemption must be read from the in-daemon trace; the
  order replies arrive at a client reflects the client. This has produced two
  separate false failures.
- **A/B must alternate within one daemon lifetime.** gfx1103 shows an ~8.6 %
  first-run position effect, so run-A-then-run-B attributes position to the change.
  Pair the comparisons per repetition too: scoring one rep against the *median of
  all* reps manufactured a 25 % phantom drift in the M0 gate that vanished on
  pairing.
- **Capture a gate's exit status; never pipe it away.** `gate.sh | grep | tail`
  reports `tail`'s status, so a failing gate reads as "completed exit 0". Redirect
  to a log and echo `$?`.
- **`pkill -f` / `pgrep -f` match the pattern against your own command line.**
  They will kill the invoking shell or count their own waiter loop as a live job.
  Match on the executable (`pgrep -x`) or exclude self.
- **Confirm the binary under test is the one you just built.** An A/B against a
  stale `target/release` binary silently measures the old code.
