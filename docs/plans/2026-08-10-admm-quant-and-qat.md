# Plan: replace the Hessian+Cholesky quantizer with layer-local ADMM, and fuse it with QAT

Opened 2026-08-10 after profiling the Qwen3.5-122B-A10B `oq4.25++` conversion.
This is a research direction, not a tuning item — it proposes changing the
quantization ALGORITHM, so every phase below is gated on quality evidence, not
speed.

## Acceptance criterion — the only one that counts

**None of this replaces the existing quantizer unless it produces an equal or
better KLD against the reference.** Not a lower proxy loss, not a faster run, not
a smaller artifact. Teacher-forced KLD on the same weights, against a `kld_eval`
reference built from the current `oq4.25++` pipeline, is the gate.

This is stated first because P0 already produced a 5.75× lower proxy loss, and a
proxy-loss win is precisely the kind of number that can coexist with worse model
quality. `tr((W−Ŵ)H(W−Ŵ)ᵀ)` is what the algorithm optimises; it is NOT what the
model is judged on. Every optimisation in this repo that was adopted on a proxy
and retracted later followed that pattern.

Concretely, to land any phase:

1. Build both artifacts from the SAME calib and the SAME source weights.
2. `hipfire-daemon` `kld_eval build_ref` on the incumbent, `score` on the
   candidate — same corpus, same n_ctx, same chunk count, so the delta is the
   quantizer alone.
3. Candidate KLD <= incumbent KLD. If it is worse, the change does not land,
   however much faster it is.
4. Report what got WORSE alongside what got better.

## What it is trying to fix

Measured on the 122B conversion (see
`docs/todo/2026-08-10-calibration-kernel-tuning.md` for the traces):

| stage | cost | dominated by |
|---|---|---|
| calibrate | ~1.5 h | `calib_hessian_outer_f32` at **32%** of GPU time — forming `H = XᵀX` |
| quantize | ~7 h | **12,288** Cholesky factorizations, K=3072, ~9.7 GFLOP each |
| artifact | 7.18 GB | 468 pooled `K×K` Hessians (38 MB each at K=3072) |

All three costs exist to produce ONE thing: the triangular factor of
`(H_rot + λI)⁻¹`, which LDLQ walks column by column.

Three escape routes are already closed and must not be re-derived — factor
caching across experts (reverted, 2× pessimization: AWQ rebases `H` per tensor),
the algebraic shortcut (`(DHD + λI)⁻¹ ≠ D(H+λ'I)⁻¹D`), and GPU Cholesky
(measured 34× slower; ~4500 block iterations each paying two device syncs).

## The two proposals

### 1. ADMM — keep the exact Hessian, change the optimizer

Treat `min ‖(W − Ŵ)X‖²` as constrained optimization and split it: a continuous
`W` (gradient step), a quantized `Z` (projection onto the quant grid), and a
dual `U` enforcing consensus. Iterate.

The decisive property is that ADMM touches `H` **only as an operator**:

    H·v = Xᵀ(X·v)

so there is no factorization AND no need to form `H` at all. That removes the
32% `calib_hessian_outer_f32` cost, the 38 MB-per-Hessian storage, and the K³
Cholesky in one move. `O(K³)` sequential becomes `O(iters × K²)` fully parallel —
the right shape for a GPU.

**The honest cost.** GPTQ/LDLQ's back-to-front sweep is *where its quality comes
from*: each column's error is propagated exactly into the columns not yet
quantized. Relaxing that to simultaneous block updates trades exact error
propagation for parallelism, and ADMM over a discrete constraint set has no
convergence guarantee — quality depends on `ρ` and iteration count. This is the
risk of the whole plan and it is not small.

### 2. Kronecker-factored Hessian (YAQA / K-FAC shaped)

Approximate `H ≈ A ⊗ B`. At K=3072 factored 48×64:

| | full | Kronecker |
|---|---|---|
| Cholesky | 9.66 GFLOP | 0.124 MFLOP (**~78,000×**) |
| storage | 38 MB | 25.6 KB (**~1,475×**) |

`chol(A⊗B) = chol(A) ⊗ chol(B)` and `(A⊗B)⁻¹ = A⁻¹⊗B⁻¹`, so LDLQ still gets a
real triangular factor and the column sweep is UNCHANGED — this is the
lower-behaviour-risk of the two, because the algorithm keeps its exact form and
only the curvature estimate is approximated.

Two catches specific to this codebase:

- **AWQ rebasing.** `H' = diag(1/s)·H·diag(1/s)` with per-input-channel `s`; `D`
  is Kronecker only if `s` factorises as an outer product, which it generally
  does not. Fit the Kronecker approximation AFTER rebasing, per expert — the fit
  is `O(K²)` moment matching, still trivial next to `O(K³)`.
- **Damping.** `A⊗B + λI` is not a Kronecker product. K-FAC's factored damping
  `(A + π√λ·I) ⊗ (B + √λ/π·I)` is an approximation, so it changes numerics.

The storage win is arguably the bigger prize: it is what makes a 397B calibration
artifact tractable at all.

## Why they belong with QAT

ADMM's split IS quantization-aware training: the `W`-update is a QAT gradient
step and the `Z`-update is "quantize". So the marginal cost of quantization on
top of QAT approaches zero — you stop paying for a separate 7-hour quantize
stage and get it as a by-product of training you were doing anyway.

The repo already has the slot. `docs/plans/2026-06-17-hipfire-train-phase0.md`
line 240:

> Full QAT-on-quanta (STE + periodic Viterbi re-projection) is a separate, later
> increment built on this same loop.

**STE + periodic re-projection is ADMM without the dual variable.** Adding `U` is
the upgrade, and it is what makes the consensus between `W` and `Z` converge
rather than oscillate.

### The incompatibility to resolve first

Current recovery-FT (`phase3-hfq-export.md:48`) tunes **LoRA + layernorms with
the codes FROZEN**. ADMM requires `W` itself to be updatable, so the two designs
do not compose as written. This is a real fork:

- LoRA-with-frozen-codes is cheap and already designed, but cannot host ADMM.
- ADMM-QAT needs full-weight updates, which at 122B means optimizer state for
  122B parameters — ~1 TB for Adam fp32 moments. **Infeasible on a 128 GB box.**

### What makes it feasible anyway: go layer-local

The streamed calibration engine ALREADY visits one layer at a time with that
layer's activations resident (`max_layer_source_bytes` was 5.03 GB on the 122B,
against a 34 GB host reserve). A layer-local ADMM needs only `X` and that layer's
`W`, and its objective `‖(W − Ŵ)X‖²` is exactly the proxy loss the calibration
already targets. So:

- optimizer state is bounded by ONE layer, not the model;
- no backward pass through the rest of the network is required for the basic
  form — the local proxy loss is enough;
- it drops into `LayerStreamEngine` where the Hessian capture currently sits.

That is the version to build. Full end-loss curvature (GuidedQuant/YAQA proper,
which wants gradients of the actual loss rather than forward-only `XᵀX`) needs
`hipfire-train`'s backward through the model and is a much larger programme.

## Phasing

Each phase must beat or match the CURRENT artifact on KLD before the next starts.

### P0 RESULT (2026-08-10): ADMM reaches a much lower objective — with caveats

`crates/hipfire-quantize/examples/admm_probe.rs`. Controlled comparison: same
objective, same int4 grid, same domain, synthetic `W` and `H = XᵀX`.

| shape | LDLQ sweep vs RTN | ADMM vs RTN |
|---|---|---|
| m=16 k=1024 | −55.4% | **−92.3%** |
| m=32 k=512 | −25.5% | **−92.3%** |
| m=16 k=256 (ONE group) | −0.0% | **−93.9%** |

Output verified representable: 0/16384 values off-grid, max \|code\| = 7.00.
A lower objective would be meaningless otherwise.

**Why ADMM wins is partly a defect in the current packer, not only algorithmic
superiority.** `oq4_ldlq_pack` propagates error only to columns `>= c0 + 256` —
i.e. ACROSS 256-column blocks, never WITHIN one. All 256 columns of a block are
quantized from the same block-entry residual. With a single block it therefore
degenerates EXACTLY to RTN, which the k=256 row shows (−0.0%). ADMM optimises
intra-block as well, and that is where most of its margin comes from.

**That implies a cheaper first move than any of this**: add intra-block error
feedback to the existing sweep (standard GPTQ is column-by-column). Small diff,
same algorithm, no new risk. Do that before adopting ADMM.

**ρ is the whole ballgame and the default is badly wrong.** At `ρ = λ` (the
damping) ADMM DIVERGES — it gets worse with more iterations (+74.8% at 15 iters).
It only works at `ρ ≈ 100–1000 · λ`, i.e. on the order of the Hessian's own
scale. This is exactly the tuning sensitivity flagged as the risk of the
approach, now confirmed rather than hypothesised.

**What P0 does NOT establish.** Synthetic weights and synthetic activations, not
a real tensor. Proxy loss is not KLD — a lower `tr((W−Ŵ)H(W−Ŵ)ᵀ)` does not
guarantee a better model, and this repo has been burned by exactly that gap. The
timings are unoptimised CPU and say nothing about the GPU shape. P1 must rerun
this against a REAL `W` and a REAL pooled Hessian from the 122B calib artifact
before anything moves into the pipeline.

- **P0 — offline harness.** Reimplement the existing LDLQ objective as an ADMM
  loop on ONE tensor, offline, and reproduce the current packed output within
  tolerance. No pipeline changes. This is where `ρ`/iteration count get tuned and
  where the quality risk is actually measured, cheaply.
- **P1 — layer-local ADMM in `layer_stream`**, replacing Hessian capture for a
  single arch, behind a flag. Gate: `compare-calibration` on the artifact plus a
  teacher-forced KLD run against a `oq4.25++` build of the same model.
- **P2 — Kronecker curvature** as an independent switch (it helps LDLQ *or*
  ADMM). Gate: Hessian reconstruction error, then the same KLD comparison.
- **P3 — fuse with QAT.** Requires resolving the frozen-codes fork above. Only
  worth starting once P1 shows the quantization quality holds.

## Validation, non-negotiable (see the acceptance criterion at the top)

This repo has repeatedly been bitten by quantization changes that were faster and
quietly worse — the retracted "budget is the biggest lever" result, the
train-on-test calibration, the LDLQ cache that reported success while skipping.
Every phase needs:

1. `hipfire-coexistence artifact compare-calibration` against a pre-change run.
2. Teacher-forced KLD (`kld_eval build_ref` / `score`) on the SAME weights, so
   the delta is the algorithm and nothing else.
3. An explicit statement of what got WORSE, not only what got faster.

Speed alone is never sufficient evidence to land any phase here.
