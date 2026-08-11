# Plan: replace the Hessian+Cholesky quantizer with layer-local ADMM, and fuse it with QAT

Opened 2026-08-10 after profiling the Qwen3.5-122B-A10B `oq4.25++` conversion.
This is a research direction, not a tuning item — it proposes changing the
quantization ALGORITHM, so every phase below is gated on quality evidence, not
speed.

## AMDAHL (2026-08-11): this plan optimized the MINORITY cost. Read this first.

Measured cost split of one LDLQ tensor (`ADMM_PROBE_TIMING=1`, m=2048):

| | k=3072 (122B) | k=4096 (397B) | scaling |
|---|---|---|---|
| Cholesky | 1.465s (**20.9%**) | 2.891s (**22.1%**) | 1.97x |
| column sweep | 5.552s (79.1%) | 10.162s (77.9%) | 1.83x |
| total/tensor | 7.017s | 13.053s | 1.86x |

**The Cholesky is ~21% of quantize time. The sweep is ~78%.**

Both headline proposals in this document target the Cholesky. ADMM's pitch was
that it needs no factorization at all; Kronecker's was a ~78,000x cheaper
factorization. Driving the Cholesky to *literally zero* caps the total speedup
at 1/0.79 = **1.27x** — 41 h becomes 32 h. Neither lever could ever have made
the 397B "practical", which was the entire premise for doing this work before
the conversion.

This was knowable from a ten-minute measurement and was never taken. The plan
reasoned from FLOP counts (`O(K^3)` Cholesky looks dominant on paper) instead of
from a profile. The `O(m*K^2)` sweep wins in practice because m=2048 makes it
the bigger term, and because the Cholesky is one well-optimized LAPACK-shaped
call while the sweep is a long serial dependency chain.

Consequences, all of which drop out of that one number:

- The AWQ-sharing idea (share the rebasing across a layer's experts so one
  factorization serves all 512) is capped at 21% and was dropped WITHOUT being
  tested — the ceiling does not justify the quality risk.
- Any future speed work here must target the SWEEP, which is irreducibly
  per-tensor (each expert has its own W) but is a column-serial loop over
  256-wide blocks — the shape most likely to have real headroom on a GPU.
- Measure before optimizing. Every failed lever in this document was chosen
  from an asymptotic argument, not a profile.

### Corrected 397B estimate

The "~41 h = 5.9x the 122B" figure came from `2.6x tensors * 2.37x K^3`. The K^3
term overweights a component that is only 21% of the work. Measured per-tensor
scaling 3072->4096 is **1.86x** at fixed m; tensor count is 2.5x (60x512 vs
48x256), giving **4.65x**, or up to ~6x if the expert width also grows with
hidden. So **33-42 h** — the original number is inside the range, but its stated
derivation was wrong.

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

### P0.5 RESULT (2026-08-11): intra-block feedback CLEARS the KLD gate on a 2B

The cheap first move from P0 was implemented and gated. `HIPFIRE_LDLQ_INTRA_BLOCK`
on `qwen3.5-2b--bf16.hfq` → `oq4.25++`, both arms from the SAME calib, SAME
source weights, SAME bf16 reference, 8 chunks / 8184 tokens:

| metric | A (incumbent) | B (intra-block) | delta |
|---|---|---|---|
| mean_kld | 0.032972854 | **0.030189654** | **−8.44%** |
| p99_kld | 0.042991739 | **0.039273560** | −8.65% |
| ppl | 13.03449 | 12.99762 | −0.28% |

Candidate KLD < incumbent KLD, so the gate PASSES. Flag stays OFF by default
until the MoE confirmation below.

**Two false results were reported by the harness before this one, and both
looked like clean passes.** Recording them because the failure mode is the
point, not the fix:

1. **Stale binary.** The gate ran a `hipfire-quantize` built before the flag
   existed. Both arms scored *identically to 17 significant figures*. A "no
   regression" reading would have accepted that.
2. **Wrong packer.** The intra-block arm was added to `oq4_ldlq_pack`
   (main.rs:4670), but `oq4.25++` dispatches to `oqplus_compact_ldlq_pack`
   (main.rs:4808). The flag was live in the binary and still changed nothing.

The defect is real in the production packer: `oqplus_compact_ldlq_pack`
quantizes all 256 columns of a block from the block-entry residual `grp[i]` and
propagates `err` only to `f in c1..k` — later blocks, never within one. Same
flaw as the plain packer.

**The check that makes the result trustworthy is the HASH, not the KLD.**
Flag-OFF must reproduce the incumbent artifact byte-for-byte
(`ede9b8ee…`, confirmed) and flag-ON must differ (`189be13b…`, confirmed).
Without both, the KLD numbers describe whatever the binary happened to contain.
Any future gate here must assert artifact hashes before reading a metric.

**Proxy massively overstated the win.** P0 measured −93.0% vs −55.4% proxy loss;
the end metric moved −8.44%. Directionally right, an order of magnitude off in
size — exactly the proxy/KLD gap this plan warns about, now quantified.

**What this does NOT establish.** One 2B DENSE model, one corpus, 8 chunks. The
production targets (122B/397B) are MoE with different tensor shapes and expert
structure, and the compact packer's outlier-tier selection interacts with the
sweep in ways a dense 2B does not exercise. Do NOT flip the default on this
evidence — confirm on an MoE artifact first. `oq3_ldlq_pack` and
`oqplus_tiered_ldlq_pack` still carry the same defect, untouched.

### P1 PARTIAL (2026-08-11): real data deflates the synthetic result by ~4x

`admm_probe` now loads a REAL weight and a REAL pooled Hessian (set
`ADMM_PROBE_CALIB` / `ADMM_PROBE_HESSIAN` / `ADMM_PROBE_HFA` /
`ADMM_PROBE_TENSOR` / `ADMM_PROBE_EXPERT`; with only `ADMM_PROBE_CALIB` set it
lists the artifact's Hessian names and exits). Run on the 122B:
W = `layers.5.mlp.experts.gate_up_proj` expert 3, [2048, 3072]; H = the pooled
donor `layers.5.mlp.gate`, k=3072, damp=0.0021.

| method | synthetic k=1024 | **real k=3072** |
|---|---|---|
| LDLQ sweep (current) | −55.4% | **−32.1%** |
| LDLQ + intra-block | −93.0% | **−42.6%** |

**The synthetic harness overstated the win roughly 4x.** Intra-block's margin
over the current sweep is −10.5 pp on real data, not −37.6 pp. That is
consistent with the −8.44% the KLD gate measured, giving a clean ordering:
synthetic proxy >> real proxy > real KLD. Treat P0's headline "5.75x lower proxy
loss" as the least trustworthy number in this document.

**Intra-block does NOT help the 397B time problem — it costs ~9%** (25.8 s vs
23.7 s on this tensor). The stated reason for doing algorithm work before the
397B was that ~41 h might be impractical; the cheapest lever makes that
marginally worse while improving quality. ADMM and Kronecker were the speed
plays, and only they can change that number.

Two facts about the real artifacts, discovered rather than assumed, now encoded
in the probe so nobody re-derives them:

- Routed experts are stored **STACKED** in the `.hfa` as
  `model.language_model.layers.N.mlp.experts.gate_up_proj` with shape
  `[n_experts, a, b]`. There is no per-expert `...experts.3.gate_up_proj.weight`
  tensor, even though that IS the quantizer-side name.
- Calib Hessian names carry **no `.weight` suffix** (`...mlp.gate`, not
  `...mlp.gate.weight`).

ADMM's rho sweep on real data is still running (single-core-bound at K=3072,
m=2048) and is NOT yet reported here.

> **RETRACTED 2026-08-11 — see "Statistical correction" below.** The MoE result
> in this section is NOT significant (paired t=+0.98 on 8 chunks, B worse in
> 4/8). Intra-block is UNRESOLVED on MoE, not failed, and the dense/MoE split
> described here is not established. The mechanism proposed below is retracted.

### VERDICT (2026-08-11): intra-block FAILS the gate on MoE. It does not land.

The 2B dense gate passed (−8.44%). The MoE confirmation on Qwen3.5-35B-A3B
(arch 6 `qwen3_5_moe`, the same family as the 122B/397B targets) FAILS:

| metric | A (incumbent) | B (intra-block) | delta |
|---|---|---|---|
| **mean_kld** | 0.030367332 | 0.031042689 | **+2.22% WORSE** |
| p99_kld | 0.038573902 | 0.038090829 | −1.25% better |
| ppl | 7.462186 | 7.490597 | +0.38% worse |

Both arms: same calib, same source, same bf16 reference, 8 chunks / 8184 tokens,
`HIPFIRE_POOLED_EXPERT_HESSIAN=1` on both, 10238 routed experts LDLQ'd in each,
artifact hashes `7a2e2c7c…` vs `26be5edf…` (differing, so the flag was live).

**Per the acceptance criterion, this is disqualifying** — candidate KLD must be
<= incumbent. `HIPFIRE_LDLQ_INTRA_BLOCK` stays OFF and is NOT applied to the
122B or 397B.

**The dense/MoE split is the whole finding.** Intra-block helps a dense 2B by
−8.44% and hurts a MoE 35B by +2.22%. The plausible mechanism is the pooled
Hessian: routed-expert `gate_up_proj` does not have its own Hessian, it borrows
the router's (`mlp.gate`) as a donor. Intra-block feedback leans much harder on
`L`'s within-block off-diagonal structure than the block-entry scheme does, and
that structure is only trustworthy when `H` is the tensor's OWN curvature. Feed
it a proxy and stronger feedback drives error along directions the real
curvature does not have.

That predicts a per-tensor policy rather than a global flag: intra-block ON for
tensors with their own captured Hessian, OFF for pooled-donor tensors. Worth
testing, and cheap to implement (`pooled_hessian_donor()` already knows which is
which at the call site). Untested as of this writing — do not assume it works.

**What got worse alongside what got better:** p99_kld improved 1.25%, so the
change does help the worst-case chunk even as it hurts the mean. Mean is the
gate metric; a p99 win does not rescue it.

### Statistical correction (2026-08-11): the gate had no power to decide this

Per-chunk paired comparison of the two arms, positive = intra-block worse:

| gate | B worse in | mean delta | paired t (df=7) | verdict |
|---|---|---|---|---|
| MoE 35B-A3B | **4/8 chunks** | +2.22% | **+0.98** | no detectable effect |
| dense 2B | 1/8 chunks | −8.44% | −2.08 | suggestive, p~0.08 |

Two-tailed critical value at df=7 is 2.365. Neither gate clears it. The MoE
per-chunk deltas run from −6.93% to +12.51% and land worse in exactly half the
chunks — a coin flip. **The "+2.22% WORSE" headline is a point estimate on 8
chunks, not a regression.**

What actually went wrong: the acceptance criterion ("candidate KLD <= incumbent")
was applied to a mean without asking whether the measurement could resolve the
difference. It could not. A mechanism was then invented to explain the split
(pooled Hessians degrading intra-block feedback), and that mechanism directly
contradicted P1, where intra-block cleanly beat the plain sweep on a
pooled-donor expert tensor (−42.6% vs −32.1%). A story that contradicts your own
measurement is a signal the effect it explains is not there.

#### Resolved at 32 chunks (dense): the effect is REAL

Same two artifacts, same corpus, 32-chunk reference instead of 8:

| | 8 chunks | **32 chunks** |
|---|---|---|
| delta | −8.44% | **−11.84%** |
| B worse in | 1/8 | 6/32 |
| paired t | −2.08 (n.s.) | **−2.824** |
| critical \|t\| | 2.365 | 2.040 |

95% CI on the delta is [−0.0082, −0.0013], entirely below zero. The effect held
and strengthened with more power — what a real effect does and noise does not.
**Intra-block genuinely improves dense KLD.**

Caveat worth keeping: ppl moved the OTHER way, 12.0335 → 12.0589. KLD is the
stated gate and it passes, but the two metrics disagree in sign and that is
recorded rather than filtered out.

#### What that implies for the MoE null — it is informative after all

With a real dense-sized effect now measured, the MoE run can be re-read. Its
per-chunk sd of the paired delta is 0.001948, so se at n=8 is 0.000689. An
effect of the dense magnitude (−11.84% of 0.030367 = −0.0036) would have
produced **t = −5.22** against a critical 2.365 — i.e. the 8-chunk run had
ample power to see it. It measured t = +0.98.

**So a dense-sized benefit on MoE is excluded, not merely undetected.** The
minimum detectable effect at n=8 was 0.001929, or 6.4% of A. The honest
three-way statement:

- dense: real improvement, −11.84%, significant
- MoE: any benefit ≥6.4% is EXCLUDED
- MoE: whether a benefit smaller than ~6.4% exists is unresolved (~64 chunks,
  roughly 7 h of GPU, would settle it)

This is the third position this document has taken on the same question. First
"MoE fails" (wrong — that read a noise point estimate as a regression), then "no
evidence of a split" (also too strong — it ignored that the null had real power).
The data supports a difference in MAGNITUDE between dense and MoE; it does not
support the original claim that intra-block makes MoE worse, and the
pooled-Hessian mechanism remains retracted since P1 contradicts it directly.

The flag stays OFF for MoE targets (122B/397B), where no benefit is demonstrated
and the largest plausible one is excluded. For dense models it now has a real
result behind it.

This does not touch the ADMM or Kronecker conclusions below: those are
order-of-magnitude effects (−11.0% vs −32.1%; 52–64% reconstruction error), not
marginal ones, and no plausible amount of noise closes them.

### ADMM on real data: DEAD (2026-08-11)

Full real-data rho sweep, same tensor/Hessian as P1:

| rho | proxy loss | vs RTN |
|---|---|---|
| 1λ | 1.87e9 | +2.5e11% (explodes) |
| 100λ | 0.6547 | −11.0% |
| 300λ | 0.8299 | +12.8% (worse than RTN) |
| 1000λ | 1.7440 | +137.0% |

The incumbent LDLQ sweep is −32.1% and intra-block −42.6%, so **no tested rho
comes close to the algorithm ADMM was meant to replace**, and only one beats
doing nothing at all. On synthetic data ADMM scored −92.3%, statistically tied
with intra-block: the synthetic harness did not merely exaggerate, it INVERTED
the ranking.

The bracket that was the one open hole has since closed. Full curve, rho as a
multiple of the damping:

| rho | 1λ | 10λ | 30λ | **100λ** | 300λ | 1000λ |
|---|---|---|---|---|---|---|
| vs RTN | +2.5e11% | +154.1% | +42.4% | **−11.0%** | +12.8% | +137.0% |

A clean U with its minimum AT 100λ. Note this refuted the intermediate guess
that the optimum lay below 100λ — 10λ and 30λ are far worse, not better. So
−11.0% is ADMM's best case on real curvature, against −32.1% for the incumbent
sweep it was meant to replace. No tuning rescues it.

Consequence for the plan: ADMM was the SPEED play that would decide whether the
397B's ~41 h is avoidable. It is dead on quality before speed ever mattered. The
41 h stands, and **P2 (Kronecker) is the only remaining lever** — with the
storage win (~1475x) arguably the bigger prize anyway.

### P2 Kronecker: DEAD on reconstruction error (2026-08-11)

`ADMM_PROBE_KRON=48,64,32,96,16` on the same real pooled Hessian
(`layers.5.mlp.gate`, k=3072). Nearest-Kronecker-product is a rank-1 problem on
the rearrangement R(H), so this is exact, not fitted — there is no knob:

| k1 x k2 | rel err | storage |
|---|---|---|
| 48 x 64 | **59.50%** | 1475x |
| 64 x 48 | 63.96% | 1475x |
| 32 x 96 | 55.17% | 922x |
| 96 x 32 | 64.25% | 922x |
| 16 x 192 | **51.85%** | 254x |

`||H - A(x)B||_F` is 52–64% of `||H||_F`. The plan projected a 1475x storage win;
that factorization reproduces the Hessian to within 59.5% error, and the least
bad shape (16x192) still misses half the norm while giving only 254x. LDLQ's
quality comes from the off-diagonal structure of `H`, which is precisely what a
Kronecker factorization at this error is discarding. **Not usable.**

Caveat: one tensor's Hessian (the layer-5 router donor). Five factorizations all
landing in 52–64% is a strong signal but not a survey; a different layer or a
non-pooled Hessian could differ. It would have to differ by a lot to matter.

### Investigation closed: every lever is negative

| lever | result | verdict |
|---|---|---|
| intra-block feedback | dense 2B −8.44%, **MoE 35B +2.22%** | fails the gate |
| ADMM | best −11.0% at rho=100λ vs incumbent −32.1% | dead |
| Kronecker curvature | 52–64% reconstruction error | dead |
| per-tensor intra-block | 97% of MoE LDLQ tensors are pooled | ~3% coverage, not worth testing |

**No quality or speed improvement is available from this direction.** The 397B's
~41 h stands. The premise that opened this plan — that a better algorithm might
make the 397B practical — is answered NO.

What the exercise did produce, which is worth more than the failed levers: the
existing LDLQ sweep is a stronger baseline than the synthetic harness implied
(−32.1% on real curvature), and the synthetic harness itself is now known to
mislead in three distinct ways — exaggerating magnitude (~4x), inverting a
ranking (ADMM vs intra-block), and understating a failure mode (rho=1λ at
+74.8% synthetic vs +2.5e11% real). Any future proxy work here starts from real
Hessians.

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
