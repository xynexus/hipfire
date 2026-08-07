# Routed-expert Hessians — what they would cost, and whether they would pay

Status 2026-08-07. Two studies, both reproducible:
`moe_expert_saliency_study` (hipfire-runtime, zero GPU) and
`hessian_generalisation_study` (hipfire-quantize, CPU only).

## The problem

Routed MoE experts are captured **imatrix-only**. `calibration_stream.rs:311`
pins them to `CapturePolicy::ImatrixOnly`, `expert_capture.rs:338` asserts it,
and the resident collector blocks them with a `vec![".experts."]` substring
list (`arch-zaya/src/calibration.rs:53`). `--ldlq` does not fail on a missing
Hessian — it logs `ldlq: skip <t>` and falls back to RTN. So **an `oq4++` on a
MoE model is `++` on the dense path and plain RTN on every routed expert**,
while reporting success. Verify on any artefact with
`hipfire inspect <x>.calib.hfq`.

Storing the missing Hessians is infeasible: for qwen3.6-35b-a3b, 9778 captured
expert-tensors at the compact `4K + K(K−1)` size (gate_up K=2048 → 4.20 MB,
down K=512 → 0.26 MB) is **43.6 GB**, against 1.8 GB for the whole current
calib.

## Q1 — experts do not share an input basis, and the two projections differ

Measured from the per-expert imatrix already in every MoE calib. The imatrix is
`diag(XᵀX)`, the diagonal of the Hessian in question, so cosine between
L1-normalised per-expert profiles bounds what a layer-POOLED Hessian would
lose. (L1-normalise first, or a hot and a cold expert differ purely by row
count, which is not specialisation.)

Mean pairwise cosine between experts, per layer:

| projection | zaya1-8b (16 experts) | qwen3.6-35b-a3b (256 experts) |
|---|---|---|
| `gate_up_proj` | 0.82 – 0.90 | 0.62 – 0.77 |
| `down_proj` | 0.04 – 0.23 | 0.09 – 0.35 (min ≈ 0.0000) |

Mechanically obvious in hindsight, which is a good sign. `gate_up` reads the
**shared residual stream** — every expert sees the same distribution, filtered
by routing. `down_proj` reads that expert's **own SwiGLU intermediate**,
produced by its own gate/up weights, so each expert's input lives in a private
basis. Near-orthogonality is exactly what that predicts.

**Do not build one mechanism for both.**

## Q2 — at a routed expert's sample ratio, does `XᵀX` still help?

A per-expert Hessian sees only the rows routed to that expert: `n/K ≈ 2` for a
top-1 16-expert layer at a realistic corpus. Measured on the DENSE path
(qwen3.5-0.8b layer 12, where K spans 3.5× so one capture sweeps the ratio),
oq4 LDLQ vs RTN, fitted on one corpus half and evaluated on a **disjoint**
half. Metric is the H-weighted proxy loss LDLQ minimises, normalised by the
same quantity for the unquantized weights.

| tensor | K | n | n/K | in-sample | held-out | vs RTN |
|---|---|---|---|---|---|---|
| gate_proj | 1024 | 512 | 0.50 | 0.001714 | 0.003180 | 0.806× |
| gate_proj | 1024 | 2048 | 2.00 | 0.001958 | 0.002820 | **0.715×** |
| gate_proj | 1024 | 8192 | 8.00 | 0.002344 | 0.002787 | 0.706× |
| out_proj | 2048 | 512 | 0.25 | 0.018815 | 0.027755 | 0.962× |
| out_proj | 2048 | 2048 | 1.00 | 0.021616 | 0.026677 | 0.925× |
| out_proj | 2048 | 8192 | 4.00 | 0.022242 | 0.024684 | 0.855× |
| down_proj | 3584 | 512 | 0.14 | 0.002321 | 0.011324 | 0.840× |
| down_proj | 3584 | 2048 | 0.57 | 0.004124 | 0.010230 | 0.759× |
| down_proj | 3584 | 8192 | 2.29 | 0.006061 | 0.009510 | **0.706×** |

Three things, and the first one surprised us:

1. **LDLQ beats RTN on held-out data at every ratio tested**, down to n/K =
   0.14, where the Hessian has rank 512 for a 3584-wide problem. The damping
   (`0.01 × mean diag(H)`, matching production) absorbs the rank deficiency.
   The sample-starvation objection to per-expert Hessians is therefore **not**
   fatal.
2. **The benefit saturates early.** gate_proj: 0.806 → 0.715 → 0.706 across
   n/K 0.5 → 2 → 8. Going from n/K 2 to 8 buys ~1%. A routed expert at n/K ≈ 2
   already captures nearly all of the available gain.
3. **Overfitting is real but shrinking**, and visible in the right signature:
   the in-sample/held-out gap is 4.9× at n=512 and 1.57× at n=8192 for
   down_proj, and in-sample error *rises* with n (0.0023 → 0.0061) because a
   better-sampled Hessian is a harder, more honest target.

### The caveat that governs the decision

This measures the **proxy loss LDLQ itself optimises**, so it is biased toward
showing benefit. `opus-quant.md` §7 already records that **on held-out data,
plain `XᵀX` LDLQ ≈ no calibration** in end-loss terms, and
`opus_outlier_budget_study` records weight-space SSE and KLD disagreeing in
this codebase before. Both can be true at once: LDLQ reliably reduces
H-weighted *weight* error out of sample (this table) while that reduction fails
to become *KLD*. Nothing here contradicts that; this table ranks a mechanism,
it does not certify a format.

## What to build, if anything

- **`gate_up`: layer-pooled Hessian.** Q1 says pooling costs little
  (expert-vs-pooled cosine 0.78–0.95) and it multiplies the sample count by E,
  moving n/K from ~2 to ~500. One K×K per layer ≈ **168 MB for a whole model**,
  storable in the existing artefact, no fused pass, no new pipeline.
- **`down_proj`: per-expert or nothing.** Pooling near-orthogonal bases is
  meaningless. Per-expert is affordable if it is never stored: hold one
  **layer's** experts and discard after quantizing that layer — ~1.1 GB peak
  for qwen3.6-35b-a3b, against 43.6 GB stored. Finalising each expert as it
  hits its `target_rows` quota drops the peak to one expert (~4.5 MB).
- Compute is not the constraint: ~357 TFLOP to accumulate every expert Hessian
  (2·K²·4096 per expert) plus ~28 TFLOP of LDL factorisation — minutes on
  gfx1151, against the 43.6 GB of storage that was the actual blocker.

The plumbing largely exists. `expert_capture.rs` already gathers rows per
expert from the routing permutation, quota-capped and tiled, "carrying partial
reduction tiles across model microbatches", and `CapturePolicy` already has a
`HessianAndImatrix` variant. The gate is the assertion at
`expert_capture.rs:338`, not missing machinery.

**Do not start with the fused pass.** Start with the pooled `gate_up` Hessian,
which is cheap and needs no new pipeline, and measure **KLD** — not proxy loss
— on one MoE model. If pooled `gate_up` moves KLD, the per-expert `down_proj`
pass is worth building. If it does not, this whole line is the `XᵀX`-doesn't-
generalise result again and the effort belongs on GuidedQuant instead, which
`opus-quant.md` §7 names as the only robust winner so far.

## Reproduce

```sh
# Q1 — zero GPU, uses any existing MoE calib
cargo run --release -p hipfire-runtime --example moe_expert_saliency_study -- \
  --calib ~/.hipfire/calib/zaya1-8b-resident-2ktok.calib.hfq

# Q2 — needs Hessians on two disjoint corpus halves (~10 min of capture)
cargo run --release -p hipfire-quantize --example hessian_generalisation_study -- \
  --fits fit.512.calib.hfq,fit.2048.calib.hfq,fit.8192.calib.hfq \
  --test test.8192.calib.hfq
```

Capture note: `collect_artifacts` sizes the sequence to `--max-tokens`, and
16384 fails on qwen3.5-0.8b with `hipModuleLaunchKernel: invalid argument`.
8192 is the working ceiling for that path today.
