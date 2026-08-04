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


## Part B progress 2026-08-04 — MLP half done, block assembly is the next unit

Landed (hipfire `de033b4fe`): `ops/moe.rs` with routed-MoE forward and backward,
gradchecked at 4.178e-4 worst relative error against central differences, with
one expert receiving zero tokens so the empty-expert path is exercised. Plus
`accumulate_gamma_moe`, normalising by ROUTED tokens rather than sequence
length.

That was the novel part — the derivation. What remains is assembly, and there is
one design choice worth settling before starting rather than discovering
halfway.

### The block cannot be composed around the dense one

The obvious shortcut does not work. Zeroing the dense block's MLP weights makes
`x_out = x_mid` in the forward, but in the BACKWARD it also makes the dense
path's `d_xn2` zero, so the MoE's contribution to `d_xn2` never reaches
`rmsnorm_backward` and the attention half receives a wrong gradient. The MLP is
not a separable addend on the backward; it feeds the norm2 path.

### Two viable routes, and the trade

**(a) Refactor `block.rs` into attention/MLP halves.** Add
`block_forward_upto_xn2` and `block_backward_from_dxn2(d_x_out, d_xn2)`, and
have both the dense and MoE blocks call them. Clean, no duplication, and
`gradcheck_block` guards the dense path against regression.

**(b) A separate `block_moe.rs` duplicating the ~120-line attention sequence.**
Touches nothing already tested; risk is confined to new code, gradchecked
independently. Costs ~240 lines of duplicated forward+backward that will drift.

**(a) is the better end state** — the duplication in (b) is exactly the kind
that silently diverges when someone fixes a rope or GQA detail in one copy. The
argument for (b) is only that the dense path is load-bearing and already
gradchecked, so leaving it untouched has value. Given `gradcheck_block` exists
and would catch a bad extraction immediately, that argument is weak: the guard
that makes (b) safe is the same guard that makes (a) safe.

Recommend (a), with `gradcheck_block` run before and after the extraction as the
regression check, then `gradcheck_moe_block` added for the new path.

### Then

    * layer loader: recognise routed-expert tensor names
      (`model.layers.N.mlp.experts.E.{gate,up,down}_proj`) and the router
    * streamed walk: dispatch dense vs MoE per layer
    * emit gamma keyed by (layer, expert) into the .calib.hfq

None of that is derivation; it is wiring against pieces that now exist and are
tested.


## CORRECTION 2026-08-05 — the MoE MLP was not the whole gap

Part B's MLP half is built and gradchecked (hipfire `de033b4fe`, `9839ab130`,
`d26af85da`). Then it turned out no locally available MoE model matches the
topology it assumes. Surveying what is actually on this box:

    model                    arch          blocker
    BLS-Mini-Code-1.0        cohere2-moe   PARALLEL block: input_layernorm only,
                                           no post_attention_layernorm. Topology
                                           is x + attn(norm(x)) + mlp(norm(x)),
                                           not LLaMA's sequential pre-norm.
    zaya1-8b                 zaya          fused gate_up_proj, AND the router is
                                           an MLP (fc1/fc2/norm/out_proj plus
                                           balancing_biases), not a linear.
    Qwen3.6-35B-A3B          qwen3.6       30 of 40 layers are `linear_attn` —
                                           DeltaNet/SSM (A_log, conv1d, dt_bias,
                                           in_proj_{a,b,qkv,z}). Only 10 are
                                           self_attn. Also: no unquantized source
                                           exists here, fused gate_up_proj, and a
                                           shared expert.

**The MoE MLP targets a Mixtral-style topology that none of these use.** The
math is correct and gradchecked; what is missing is a compatible model.

### What this means for the 35B specifically

`hipfire-train`'s block is GQA softmax attention. The 35B is a HYBRID: three
quarters of its layers are linear attention. Running gamma on it needs a
DeltaNet forward AND backward — SSM recurrence, conv1d, gating — which is a
larger piece of work than the MoE MLP was, and none of it exists in that crate.

So "add MoE to hipfire-train" was never sufficient for this target, and this
document said it was. The error was scoping from the MLP naming without checking
the ATTENTION the target uses. A topology survey costs minutes and should come
before any block-level work, not after.

### Where that leaves things

    done and verified   layer-streamed DENSE gamma (bit-identical to whole-model)
                        MoE MLP fwd/bwd (gradchecked, no compatible model yet)
                        MoE block assembly (gradchecked)
                        loader + dispatch (dense path bit-identical)

    open fork           (1) implement DeltaNet fwd/bwd — unblocks the real target
                        (2) fetch a Mixtral-style MoE to validate what exists
                        (3) stop; the dense path already serves 8B-class models

The dense work stands on its own regardless: it removed the f32 depth ceiling and
is verified bit-identical, and the gamma objective it feeds is what took auto
mixed precision from losing to hand-picking to matching it.


## MiniMax-M2 IS the topological match (2026-08-05)

Surveying further: `hipfire-arch-minimax` is MiniMax-M2, described in its own
header as a Mixtral-style MoE, and its block is

    h += attn(input_layernorm(h));  h += moe(post_attention_layernorm(h))

which is EXACTLY the sequential pre-norm topology `moe_block_forward`
implements. Not parallel (BLS), not an MLP router (zaya), not linear attention
(Qwen3.6). The deltas are small and known:

    experts live under `.block_sparse_moe.experts.N.`, not `.mlp.experts.N.`
        -> one change to the probe and the name format
    per-layer QK-norm on the attention half
        -> a contained addition to block_forward/backward, not a rewrite

So the MoE work already built is validatable against a supported architecture.
The cost is the artifact: `/srv/hipfire/archives/models--MiniMaxAI--MiniMax-M2.7.hfa`
is 133 GB, and whether an unquantized form is extractable at a workable size has
not been checked.

### And the tiny fixture is a DELTANET target, not a MoE-block one

`/srv/hipfire/fixtures/qwen3_5_moe-tiny` (config.json + 11.7 MB safetensors) is
NOT a Mixtral-style MoE. Its config says

    layer_types: [linear_attention, linear_attention, linear_attention, full_attention]
    attn_output_gate: true

i.e. it mirrors Qwen3.6-35B-A3B's hybrid architecture in miniature — three
quarters linear attention, one quarter full. That makes it the right
DEVELOPMENT fixture for the DeltaNet work: small, fast, safetensors-loadable,
and structurally identical to the real target. It cannot validate the MoE block.

### The fork, sharpened

    (1) DeltaNet fwd/bwd     the only path to Qwen3.6-35B-A3B. Largest piece.
                             qwen3_5_moe-tiny makes development tractable.
    (2) MiniMax-M2           proves the MoE work already built. Topology matches;
                             needs QK-norm + the block_sparse_moe naming. 133 GB
                             archive is the open cost.
    (3) Stop                 the dense path is done, verified, and is what fed
                             the auto mixed-precision result that landed.
