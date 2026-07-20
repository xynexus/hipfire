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
- GPU gates must acquire `hipfire gpu-lock` unless they exclusively drive a
  daemon path that acquires resource leases itself.
- Preserve byte-identical prompts by reading committed files from
  `benchmarks/prompts/`; do not inline new benchmark prompts in shell heredocs
  when the exact text matters.
- When adding or repairing model/runtime admission coverage, update
  `crates/hipfire-eval/` first and keep shell scripts as wrappers where useful.
