# Validation procedure for arch ports

Three gates. All three must pass before merging an arch port. Two
of them require **real hardware for the target arch** — there is no
emulator/simulator path. If you don't have hardware, you'll need a
contributor who does (`contributor-onboarding.md`).

## Gate 1: Channel-test (correctness)

This is the load-bearing one. WMMA / MFMA kernels have arch-specific
per-lane mappings, and getting the C-mapping wrong silently
corrupts all matrix outputs. The 6-week WMMA bug
(`memory/project_wmma_correctness_fix.md`) is a documented case of
this passing the speed-gate AND coherence-gate while silently
producing garbage.

### What it tests

`crates/engine/examples/test_kernels.rs` (and `test_kernelsQA.rs`)
runs each registered GPU kernel through a battery of small, golden-
output cases. Inputs are deterministic; outputs are compared
element-wise against a CPU reference within a tight tolerance.

### How to run

On the target hardware, with a fresh hipfire checkout:

```bash
cd hipfire
cargo build --release --features deltanet -p engine \
  --example test_kernels --example test_kernelsQA

./target/release/examples/test_kernels        # smoke battery
./target/release/examples/test_kernelsQA      # full QA matrix
```

### What "pass" looks like

Every dispatched kernel emits `OK` (or `PASS`) at the end of its
case. Any `FAIL` / `MISMATCH` is a hard block — the arch port is
not merge-ready.

### Hot tip

If `test_kernels` doesn't currently exercise the new arch's WMMA
path (because dispatch is gated on the arch and the arch isn't
listed), **add a test case first** that targets the new kernel
explicitly, then run the suite. Don't merge a port that has no
correctness coverage on its specific kernel.

## Gate 2: Coherence-gate (output sanity)

`scripts/coherence-gate.sh` runs a small fixed prompt matrix through
the daemon and writes a markdown report.

### What it tests

- Daemon doesn't panic on the arch.
- Generation emits non-zero tokens within timeout.
- Outputs are coherent (English, on-topic, not stuck in a loop).

The hard-fail conditions are panic / zero-tokens / timeout. Soft
output diffs (different but still correct answers) are not block-
ing — the human / agent reviewer reads the report.

### How to run

```bash
./scripts/coherence-gate.sh           # short — 0.8b/4b/9b dense
./scripts/coherence-gate.sh --full    # adds A3B MoE
```

### Important note for arch ports

A *new* arch may not have its WMMA path exercised by the
coherence-gate matrix if the gate is running models that fall
through to the dispatch fallback. Check the daemon's stderr — if
you don't see `[verify-graph] captured...` for prefill or the
arch-specific dispatch print, you're not actually testing the
ported path. Either:

- Force the arch's path via env (`HIPFIRE_FORCE_ARCH=gfx1201` if
  such a knob exists; otherwise temporarily edit dispatch.rs to
  remove the fallback for the gate-run only), OR
- Add a model + prompt to the gate matrix that's known to require
  the arch-specific kernel.

## Gate 3: Speed-gate (no regression on the baseline arch)

`scripts/speed-gate.sh --fast` benches 4b prefill + decode against
the committed ground-floor baselines in `tests/speed-baselines/`.

### Why this matters even for an arch port

A "no-op" arch-conditional refactor of `dispatch.rs` regressed 4b
prefill on gfx1100 by 50% in this session
(`master a048544 → reverted in 1f3bad3`). The mechanism is not
yet root-caused — possibly inlining / register-allocator
interactions in the dispatch hot loop. **Until that's diagnosed,
any change to `dispatch.rs` MUST run the speed-gate and pass on
the baseline arch.**

### How to run

```bash
./scripts/speed-gate.sh --fast        # 4b only, ~30s
./scripts/speed-gate.sh                # all sizes (0.8b/4b/9b/27b)
./scripts/speed-gate.sh --verbose      # full bench output
```

Tolerance is ±5% from the committed baseline. The pre-commit hook
runs `--fast` automatically when the staged diff touches kernel,
dispatch, forward-pass, or engine code.

### If you legitimately trade speed for something else

If your arch port intentionally regresses gfx1100 perf (highly
unusual but possible — e.g., a refactor that enables a much faster
gfx12 path at the cost of a small gfx11 slowdown), justify it AND
re-baseline:

```bash
./scripts/speed-gate.sh --update-baselines
git add tests/speed-baselines/
git commit                            # runs hook again, now passes
```

The baseline change must be in the SAME commit as the regressing
diff so reviewers see the trade-off explicitly.

## What you do NOT do

- **Do not bypass the speed-gate with `--no-verify`.** The repo's
  contract: this is a red line unless explicitly authorized by the
  maintainer in writing. (Documented this session: `--no-verify`
  bypass on commit `a048544` was a contract violation; the bypass
  hid that the change actually DID regress perf, leading to a
  revert.)

- **Do not assume "my change is functionally identical so the gate
  doesn't apply."** It might not be. Run the gate.

- **Do not merge without all three gates green** unless you have
  the maintainer's written sign-off on a known limitation (e.g.
  "channel-test deferred to follow-up because hardware not
  available").

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Channel-test FAIL on a new arch | Per-lane C-mapping wrong | Add `eprintln!` of `(tid, acc[j])` for first warp, compare to CPU reference, derive correct mapping. See `memory/project_wmma_correctness_fix.md` for the gfx11 case. |
| Coherence-gate panic | Codegen failure, missing kernel file, bad dispatch | Read the panic message; usually a stack trace from `dispatch.rs` or `kernels.rs` |
| Coherence-gate zero-tokens | Daemon stops at EOS immediately, often a tokenizer / chat-template / KV-init bug | Check `m.seq_pos` and `prompt_tokens` — see `feedback_dflash_chatml_and_drift.md` |
| Speed-gate regress on gfx1100 from "no-op" arch refactor | The predicate-vs-inline trap | Revert the refactor; add new arch via separate inline branch above the existing gfx11 inline check |
| Speed-gate regress with system in known-good state | Firmware shadowing | `sudo mv /lib/firmware/updates/amdgpu .bak && reboot` (`feedback_firmware_shadowing_perf_trap.md`) |

## Last verified

This procedure was used to validate gfx1100 (RDNA3) and gfx1030
(RDNA2) ports. gfx1201 (RDNA4) port is in progress as of 2026-04-27
(tracking issue #54).
