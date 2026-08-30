# Calibration capture is nondeterministic; inference is not

Status: **localized, NOT fixed** — 2026-08-30, nix2 (gfx1103), master `5f6bb9dd6`.

Chasing the flaky `zaya/kld:oq4.25++` tiny-quant cell led to something larger
than a flaky cell. `tiny_quant_probe collect` — the instrumented forward that
captures Hessians and imatrices — **does not produce the same artifact twice**.

This matters beyond the gate: that same capture path is what produces the
`.calib.hfq` behind every `oq4+` / `oq4++` / `oq8+` / `oq8++` artifact. Two
calibration runs over identical inputs can yield different quantized weights.

## What was measured

Each stage was isolated on a fixed zaya fixture (seed 42), GPU lock held:

| stage | runs | result |
|---|---|---|
| `--emit-fixture zaya --seed 42` | 5 | **deterministic** (content-identical) |
| `--format q8f16` anchor from one fixture | 5 | **deterministic** |
| `--format oq4.25++` from one calib | 5 | **deterministic** (also with `HIPFIRE_OQ_RAGGED_Q8=1`) |
| `ar-hash` (plain greedy decode, exact logit hash) | 10 | **deterministic**, `0x136dd20f13e71b86` ×10 |
| `collect` (calibration capture) | 10 | **9 identical, 1 different** |
| `collect` with `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` | 10 | **6 distinct** |

Inference is bit-exact. Only the capture path moves.

## What the divergence looks like

When `collect` diverges, **all 54 captured tensors differ** — Hessians and
imatrices, every layer, including `model.embed_tokens.hessian`, which precedes
routing. The differences are not last-bit: on the embed Hessian, 60% of stored
bytes differ, mean byte delta ~50, only 15% of those are ±1.

The metadata narrows it to a single decision. Both runs record
`n_calib_tokens: 24`, `n_hessian: 19`, `n_imatrix: 35` — identical. But
`per_tensor_tokens` differs in exactly two entries:

    model.layers.0.mlp.experts.2.down_proj:     10 vs 9
    model.layers.0.mlp.experts.2.gate_up_proj:  10 vs 9

One token's routed-expert assignment moves. Everything else follows.

## Ruled out

- **Not an async race.** `AMD_SERIALIZE_KERNEL=3` + `AMD_SERIALIZE_COPY=3` made
  it strictly WORSE (6 distinct results instead of 2). A missing stream sync
  would have been suppressed by serialization, not amplified. Amplification
  under changed allocation/scheduling timing is the signature of reading memory
  whose contents depend on what was there before.
- **Not the router tie-break.** `zaya_router_select_f32` is single-threaded
  (`blockIdx.x != 0 || threadIdx.x != 0` returns) and uses strict `>`, so equal
  scores deterministically pick the lowest expert index.
- **Not unzeroed accumulators.** `allocate_acc` takes every buffer from
  `gpu.zeros`, which is a real `memset`/`memset_async`, not a bare alloc.
- **Not the corpus.** `collect` defaults to `--seed 42` and the battery never
  overrides it; 9 of 10 runs agree exactly, which a varying corpus would not do.
- **Not the crest atomics.** `calib_group_crest_reduce_f32` does use a float
  `atomicAdd`, but it feeds crest telemetry, not the Hessian or imatrix, and the
  Hessian kernel (`calib_hessian_outer_f32`) is a fixed-schedule tiled GEMM with
  one thread per output and no atomics.

## Where to look next

The staging tile in `calibration.rs` is zeroed **once at allocation** and reused
across flushes (`flush_result` sets `buf_rows = 0` without re-zeroing). The
gather that fills it, `calib_gather_rows_f32`, *skips* writing any row whose
slot index is negative:

    int flat = sorted_slot_index[sorted_start + row];
    if (flat < 0) return;

A skipped row inside `0..rows` is still inside `buf_rows` and is still reduced —
so it would contribute the PREVIOUS tile's values rather than nothing. That is
the one place found where the capture can read a row nobody wrote this tile, and
it fits "depends on what was there before". It has not been confirmed as the
live cause: the admission planner in `calibration/expert_capture.rs` is supposed
to exclude padding, so proving it needs instrumentation of whether a negative
slot is ever admitted, not just inspection.

## Impact today

The tiny-quant gate is green (`5f6bb9dd6`) because the affected baselines were
re-recorded and the ±0.25 relative tolerance absorbs the variation on every cell
except the smallest. `zaya/kld:oq4.25++` at ~2e-5 KLD was the only cell where the
spread crossed its budget, which is why it presented as a "bimodal" flake. It is
inside tolerance now, not fixed.

Production calibration is affected in principle — the same code path, without the
tolerance. Whether the resulting weight differences matter at real model scale is
unmeasured; the fixture is random-init and tiny, where a single re-routed token
is a large fraction of the evidence.
