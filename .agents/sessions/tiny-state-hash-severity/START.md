# Session: split the tiny-state hashes by severity

**Blocked on:** nothing. **Est:** small — one `if`, plus the judgement about which
families qualify. **Value:** retires a gate that currently fails open.

## Objective

`tests/tiny-state-gate.sh` compares `logit_hash` and `token_hash` with equal
severity, both exact. Make `token_hash` the blocking behavioural assertion and
`logit_hash` advisory for families whose state is a recurrent low-precision
accumulator.

## Why

`mamba2/fp16` drifts on gfx1151: `logit_hash` moved, **`token_hash` did not**.
The token stream is identical, so it is sub-threshold float noise that never
crossed an argmax — reported as a hard FAIL.

That cell has a documented history. From the gate's own header: *"mamba2/fp16
took three different values in 15 days with no code change between them."* That
was blamed on two gfx1103 hosts sharing a baseline row and fixed by keying on
`hip=`; the current instance is a single host with a matching `hip=`, so the
previous cause is excluded. Fourth instance.

Twelve families share the `fp16` anchor and all twelve are stable. What
distinguishes mamba2 is that Mamba-2's SSM state IS the accumulator, held in fp16
and re-rounded every kernel invocation. `BUGS.md` measures that shape for the
DeltaNet analogue: no error feedback, divergence growing **superlinearly** (13x
error for 2.5x tokens). Asserting its hash `[exact]` asserts a property the
arithmetic does not have.

Full diagnosis: `docs/bugs/2026-09-01-mamba2-tiny-state-drift.md`.

## The dead knob to fix or delete

The baseline format advertises a tolerance:

    # gpu_arch family format logit_hash token_hash rel_tol

Every row sets `rel_tol=0`, and **`$6` is read nowhere** — `grep -c '\$6'`
returns 0. The parser takes `$4" "$5` and string-compares. A column that looks
configured and is inert. Either wire it or delete it; do not leave it implying a
knob that does not exist.

## The verification bar

- With the split in place, `mamba2` reports advisory-not-failing while its
  `token_hash` still matches — and **flips to failing if you corrupt the token
  stream**. Verify that second half; a gate that cannot fail is the thing being
  fixed.
- The other 17 cells are unaffected.

## Do NOT

Re-record the mamba2 baseline as the fix. The value has moved at least four times
across two architectures; a re-record buys days, not correctness.

## Related

Third gate this session that looked configured and was not, after `tiny-spec-gate`
asserting a path it had itself disabled (fixed, `f212ae076`) and a test that sized
its input from the constant under test. Worth a wider sweep — see
`.agents/sessions/gate-vacuity-sweep/`.
