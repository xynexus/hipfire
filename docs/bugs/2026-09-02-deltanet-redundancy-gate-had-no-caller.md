# The DeltaNet low-redundancy FP32 guard took the model config and threw it away

Status: **FIXED 2026-09-02.** Found while chasing
`2026-09-01-spec-decode-not-output-equivalent-to-ar.md`, after the observation
that KVarN-8 was not behaving like KVarN-8.

## The defect

`qwen35::state::default_state_quant` opened with:

    pub fn default_state_quant(config: &Qwen35Config) -> StateQuant {
        let _ = config;

It takes the model config and immediately discards it, then decides purely from
`deltanet_state_precision()` (config field, ships `fp16`) and the
`HIPFIRE_DN_STATE_FP16` env var.

So the size-based guard the file documents at length had **no production
caller**:

- `deltanet_state_redundancy(config)` — `key_head_dim x num_value_heads` — was
  referenced only by a unit test.
- `deltanet_state_fp32_below()` was referenced only from doc comments. It also
  ignored `HIPFIRE_DN_STATE_FP32_BELOW`, the env var its own doc comment
  advertises, and simply returned `usize::MAX`.

The policy note directly above the function is emphatic that low-redundancy
models are the numerical anchor — "long-decode attractors on low-redundancy
models (2026-06-15)", "on the low-redundancy 2B — where Q8 broke first". The
guard for exactly those models was inert, and `deltanet_state_precision` had
meanwhile been defaulted to `fp16`, so they silently got the narrow state with
no warning.

## How it got that way

Q8 state was removed on 2026-08-09, and the threshold that selected Q8 above a
size cutoff was removed with it — reasonably, since it no longer selected
anything. What went with it was the *low end* of the same test: the rule that
below a redundancy floor the state must be FP32 regardless of what anyone asked
for. The comment even says so — "it no longer SELECTS anything" — which is true
of the Q8 arm and was quietly assumed of the FP32 arm.

`deltanet_state_precision` then landed defaulting to `fp16` (2026-08-27), and
nothing was left to object for a small model.

## Fix

`default_state_quant` uses its `config` again. When the requested precision is
FP16 and `deltanet_state_redundancy(config) < deltanet_state_fp32_below()`, the
state is forced to FP32 and says so once:

    deltanet state: forcing FP32 — redundancy 2048 (key_head_dim x value_heads)
    is below 3000. Low-redundancy models are where narrow state breaks first,
    and state is only ~1-3% of per-token bandwidth. Override with
    HIPFIRE_DN_STATE_FP32_BELOW=0.

`deltanet_state_fp32_below()` now reads `HIPFIRE_DN_STATE_FP32_BELOW` as
documented, defaulting to `DN_STATE_FP32_BELOW_DEFAULT = 3000`; `0` disables the
guard. The `#[deprecated]` marker is gone, because the function is load-bearing
again.

**Threshold basis.** `qwen3.5-0.8b` measures 2048 (`key_head_dim 128 x 16 value
heads`), and the policy note records 3000 as the boundary that would have put
9B/27B on the other side. So 3000 catches the 0.8B/2B class the note names and
leaves larger models on whatever the config asks for. It is a documented
boundary rather than a fresh measurement, and is env-tunable for that reason.

Verified on `qwen3.5-0.8b--oq4++.hfq`: the warning fires and the daemon then logs
`DeltaNet state: FP32` where it previously logged FP16.

## What this does NOT fix

It does not fix the spec/AR divergence that led here.
`tests/spec-ar-equivalence-gate.sh` still fails with the guard active and the
state at FP32 — every speculative width emits one sequence, AR emits another.
The guard is correct on its own terms; it is not that bug's cause.

The unit test at `qwen35/mod.rs` that asserted FP16 for a redundancy-8 fixture
now asserts FP32. Its expectation encoded the window in which the guard had no
caller.
