# RESULT: qwen4_exp batched forward — PREMISE RE-CONFIRMED, not implemented

**Status: still open, still worth doing.** Unlike the other briefs worked this
session, this one survives contact with measurement — the numbers changed
slightly, the conclusion did not.

## Re-measured 2026-09-02, `Qwen3.8-Flash-Next-180B--oq4`, this box

`serve_real <model> 4 <prompt_len>`, one process per arm:

| prompt tokens | prefill | s/token | note |
|---|---|---|---|
| 8 | **11.97 s** | 1.496 | ⚠️ **DISCARD — cold kernel cache** |
| 16 | 3.96 s | 0.248 | |
| 32 | 5.94 s | 0.186 | |
| 64 | 9.42 s | 0.147 | |

The first arm is the documented trap firing in plain sight: 8 tokens "cost"
three times what 16 tokens cost, because that process JIT-compiled kernels
inside the timed window. **Read the 8-token row and you would conclude prefill
is superlinear and chase a phantom.** The brief's own trap list says discard it;
this is what it looks like when you don't.

Warm arms 16→64 are a clean straight line:

    marginal = (9.42 − 3.96) / (64 − 16) = 0.114 s/token
    fixed    ≈ 2.14 s

Linear, as a per-token forward must be — the premise holds. Extrapolated:
**512 tokens ≈ 60 s, 2048 tokens ≈ 3.9 min before the first output token.**
(The brief said ~88 s and ~6 min; the gap is likely the 1.39x paged-expert
decode win in `322324721`, which landed after those numbers were taken. Same
conclusion, smaller constant.)

`crates/hipfire-arch-qwen4exp/src/serving.rs:8` still says it outright:
*"**Prefill is per-token.** The trunk has no batched prefill."*

## Why it was not implemented here

It is genuinely multi-session, and the two sequential halves are the reason:

- **Gated DeltaNet** — the recurrent state. `decode_step_into`
  (`trunk_gpu.rs:671`) advances exactly one position per call, and the state is
  carried across layers in `TrunkState.gdn`. Batching needs the chunked-scan
  treatment qwen35 already has (`gated_delta_net_f16` + the chunked-SSD prefill
  work) ported onto this trunk's layout.
- **The PLE conv ring** — `ple_step` is additive on the WIDE stream before the
  residual read (`trunk_gpu.rs:699-717`), with per-layer `PleScratch`. A conv
  over a known window should batch more easily than the SSM, and the brief is
  right that the two can land independently.

Starting either without being able to finish and verify it would leave the trunk
half-batched, which is worse than per-token: the bar
(`tests/qwen4exp-gate.sh` PASS, paged arm bit-identical to resident, decode
argmax **1892 (13.9764)** unchanged) is all-or-nothing per half.

## For the next session

1. The measurement above is fresh — do not re-derive it, and **do not re-take it
   on a cold cache**. Run any arm twice and use the second.
2. `qwen4exp-gate.sh` already carries the guard that matters: it asserts the
   paged arm did real cold loads and evictions, because *"paged matches resident
   proves nothing if nothing was paged"*. Keep that shape when adding a batched
   arm — a batched-vs-per-token parity check must likewise prove the batched path
   actually batched.
3. Land PLE and DeltaNet independently, measuring the slope after each. The slope
   bending is the deliverable; τ, tok/s and argmax identity are the guards.
