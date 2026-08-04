# Scope: per-layer / per-expert gamma collection

Written 2026-08-04. Scoping only.

## Why the current collector cannot be the answer

`calib_gamma` loads the WHOLE model as f32 and runs a full forward+backward. For
Llama-3.2-1B that is ~4 GB and fine. For the actual target it is not:

    model                  params    f32 weights
    Llama-3.2-1B             1.24B        ~5 GB
    Qwen3.6-35B-A3B            35B      ~140 GB     <- exceeds 128 GB UMA

So the per-model collector is a 1B-only instrument. It was worth building to
settle whether gamma is the missing factor (it is — see
`investigate-blockwise-objective.md`), but it cannot produce the artifact the
35B needs.

Widening precision does not rescue it either. bf16 storage halves it to ~70 GB,
which still leaves no room for activations, optimizer-free though this is. The
structural fix is to never hold the whole model.

## The forward half already exists

`crates/hipfire-runtime/src/calibration/` is a layer-streamed calibration
engine, and it already carries every concept this needs:

* `layer_stream.rs` — the orchestration, family-neutral.
* `boundary.rs` — `BoundaryStore` / `BoundaryCheckpoint`, which durably commit
  the activation at each layer boundary (`completed_layers`, `total_layers`,
  double-buffered `boundary-a/b.f32`, `sample_fingerprint`). This is the piece
  that matters: it is exactly the checkpointing a layer-by-layer backward needs.
* `source.rs` — `LayerPrefetch`, `TensorLoadPlan`, `PlannedTensorReader`,
  `ReadLedger`. Weights are paged in per layer rather than held.
* `contracts.rs` — `LayerExpert`, `ExpertCaptureQuota`, `ExpertLayerTelemetry`,
  `ExpertSamplingPolicy`, `ExpertCoveragePolicy`. **Per-expert capture is
  already a first-class concept**, including quotas and coverage policy for the
  under-activated experts a routed MoE inevitably has.

None of that has to be invented. What is missing is the reverse walk.

## CORRECTION 2026-08-04 — there are TWO gaps, not one

This document said "the forward half already exists", which is true of the
CALIBRATION engine in `hipfire-runtime` and misleading about the thing that
actually has to run. The backward lives in `hipfire-train`, and:

    $ grep -rin "expert|moe|router" crates/hipfire-train/src/
    (nothing)

**`hipfire-train` has no MoE support whatsoever.** `block.rs` is a dense
pre-norm LLaMA block: rmsnorm, GQA attention, SwiGLU MLP. No router, no routed
experts, no top-k dispatch, and therefore no backward for any of it.

So per-expert gamma is not an extension of the streaming work, it is a second
project of comparable or larger size — implementing MoE forward AND backward in
the training crate. The per-expert plumbing in `hipfire-runtime`
(`LayerExpert`, `ExpertCaptureQuota`, `ExpertLayerTelemetry`) exists on the
FORWARD/capture side and does not supply any of it.

Splitting the work accordingly:

    A. Layer-streamed reverse walk, DENSE.   Removes the f32 depth ceiling.
                                             Validates against the known 1B
                                             per-model gamma table.
    B. MoE block forward + backward.         New. Required for the 35B, and
                                             independent of A.

A is worth doing on its own: `Llama-3.1-8B-Instruct--bf16.hfq` is ~32 GB in f32,
which fits but painfully, and A makes depth free. It is also the foundation B
would plug into, and it can be validated against an answer we already have —
the streamed walk must reproduce the per-model gamma table on 1B.

## What has to be built

**A backward that streams layers in reverse.** The forward commits boundary
activations going up; the backward walks down, and for each layer needs only:

1. its own weights (paged in, then dropped),
2. the boundary activation that was its INPUT (already checkpointed),
3. the incoming `d_x` from the layer above (one activation-sized buffer).

Recompute the layer's internal activations from the saved boundary input rather
than storing them — standard gradient checkpointing, and the boundary store
already provides the checkpoint granularity. Peak residency becomes one layer's
weights plus two activation buffers, independent of depth.

`block_backward_capture` already returns the per-projection adjoints
(`BlockAdjoints`), so per-layer gamma falls out of the walk directly.

**Per-expert gamma for MoE.** A routed expert only sees the tokens routed to it,
so its adjoint rows are a subset. Two things follow, and both have existing
handles:

* Accumulate gamma keyed by `(layer, expert)`, matching `LayerExpert`.
* Normalise by the number of tokens that actually reached the expert, not by
  sequence length. Otherwise a rarely-routed expert looks insensitive purely
  because it was rarely routed — which is a routing fact, not a sensitivity
  fact. `ExpertLayerTelemetry` already tracks activation counts, and
  `ExpertCaptureQuota` / `ExpertCoveragePolicy` already exist because the
  forward side hit this same problem.

**Emission.** One f32 per `(tensor)` or `(layer, expert, tensor)` into the
`.calib.hfq`, beside `<name>.imatrix` / `<name>.hessian`. Storage is nil — the
35B has ~1.5k routed-expert tensors, so kilobytes.

## What this does NOT need

* **Not an optimizer.** No steps, no state, no convergence concern. This is why
  the usual objection to bf16 in the training path ("7 mantissa bits degrades
  convergence", per `ops/linear.rs`) does not apply: we take one backward and
  read an energy off it.
* **Not weight gradients.** Only activation adjoints. `block_backward` already
  computes those to propagate; the base weights are frozen.
* **Not full precision.** A ranking statistic tolerates bf16 accumulation. That
  matters because it is what lets the backward run in the same precision the
  layer-stream forward already uses.

## Sequencing

The 1B result should decide this, not the other way round. Current state: gamma
moves `o_proj` from 79..113 to 26..66 but still ranks `k_proj` above it, so the
objective is **not yet correct even on 1B**. Building the streamed collector now
would scale an objective we know to be wrong.

    1. Settle the 1B objective first — the CE-vs-Fisher question and the
       scalar-vs-matrix G question in investigate-blockwise-objective.md.
    2. Fold in down_proj (BlockAdjoints has no d_down; the existing
       down_guided_capture path takes it from d_x) so all 113 candidates are
       covered rather than 96.
    3. Only then build the reverse walk, against a known-good objective.

The one exception worth doing early: **decide the artifact schema now**, because
it is cheap to agree and expensive to change once calibration artifacts exist in
the wild. Per-expert keying should be in the schema from the start even if the
dense path fills only the degenerate case, so a 35B artifact does not need a
schema bump.

## Open question for the MoE case

Whether gamma is even the right granularity for routed experts. The dense
argument is that a tensor's error reaches the loss in proportion to its output
gradient. For an expert, the error reaches the loss only on the tokens routed to
it, weighted by the router's gate value. Whether that is captured by accumulating
the adjoint over routed tokens, or needs the gate weight folded in explicitly, is
not obvious from the dense derivation and should be checked against a measured
per-expert promotion before trusting it.
