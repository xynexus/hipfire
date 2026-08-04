# Scope: auto mixed precision

Written 2026-08-04. Scoping only — no implementation decision taken.

## The gap we are trying to close

`mixed_precision.rs` is wired behind `--mixed-bpw` and works, but loses to
hand-picking:

    hand-picked v_proj + o_proj -> oq8++    4.578 b/w   KLD 0.025025
    auto --mixed-bpw                       ~4.88 b/w   KLD 0.030071

**Worse KLD for ~0.30 b/w MORE.** (The 4.88 is the corrected figure; it was
recorded as 4.72 before the floor bug in hipfire `6d8a840fd`.)

Two independent things could cause that — the RANKING (which tensors it picks)
or the SEARCH (how many bits it spends). They need different fixes, and we have
not yet established which is at fault. That ordering matters more than anything
else below.

## Step 0 — the diagnostic that might end this project (~1 hour)

Dump the current ranking and see where `v_proj` / `o_proj` actually land.

* If they are already at the top, **the objective is not the problem** — the
  greedy fill or the budget is, and the expensive full-Hessian work below is
  unnecessary.
* If they rank low, the objective is confirmed as the culprit and the rest of
  this document applies.

This is a `--mixed-bpw` run with the promotion list printed (already emitted:
`promote: <name>` lines) plus the raw sensitivity per tensor, which is a few
lines of `eprintln!`. **Do this before anything else.** It is cheap and it is
decisive in one direction.

## A defect that is independent of the objective

`assign_tiers` models the gain of promoting oq4 -> oq8 as `c.err_oq4` — i.e. it
assumes oq8 has ZERO error:

    Tier::Oq4 => (c.err_oq4, (OQ8_BPW - c.bpw_at(...)) * numel, true)

oq8++ is near-lossless but not lossless (3.5e-4 KLD from bf16 per
`layer_sensitivity_hessian`'s header), so the true gain is
`err_oq4 - err_oq8`. Tensors whose oq4 error is mostly irreducible get
over-credited. This is cheap to fix and worth fixing regardless of which
objective wins — it is a modelling error, not an approximation.

## What the objective currently is

Both `oq4_sensitivity` and the standalone `layer_sensitivity_hessian` example
compute the SAME thing, and that example's own header states the framing
exactly:

> For a linear layer `y = W x`, replacing `W` by `Ŵ` costs output error
> `E‖δy‖² = tr(ΔW · H · ΔWᵀ)` with `H = E[x xᵀ]`. Using only the diagonal of `H`
> (the captured imatrix `d_j = E[x_j²]`) gives the standard GPTQ/imatrix proxy.

So we are already computing the right quantity's **diagonal approximation**. The
upgrade is the off-diagonal — the cross-channel terms. This project's own
history suggests those matter: FWHT rotation (which exists precisely to
decorrelate channels) moved KLD substantially, and LDLQ already uses the full H
rather than its diagonal.

## Options, with costs

Model is Llama-3.2-1B: **112 candidate tensors, 973.1 M weights**, shapes
`32x[8192,2048]`, `32x[512,2048]`, `32x[2048,2048]`, `16x[2048,8192]`.

### A. Full Hessian, `tr(ΔW H ΔWᵀ)`

    compute      3.64 T MAC = 7.28 TFLOP   (E·H then elementwise-sum with E)
    Hessian I/O  ~5.9 GB read
    hot spot     down_proj (k=8192): H is 268 MB EACH, 137 G MAC each,
                 16 of them = 60% of the total cost

Needs a blocked GEMM (faer is already a dependency); the rayon-over-rows idiom
in `ldlq.rs` would re-stream a 268 MB Hessian per row and is not viable here.

Plumbing is NOT a blocker: `OQ4_LDLQ_HESSIAN` is a global and
`ldlq_hessian_for_tensor(idx, name, k)` already returns the k x k matrix, so the
`--mixed-bpw` site can reach it without threading anything.

**The basis complication is the real cost.** Weights are stored AWQ-scaled then
FWHT-rotated, so a faithful ΔW must be simulated in that basis — which means H
must be rebased the same way: `H' = R·S⁻¹·H·S⁻¹·Rᵀ`. For the diagonal that is
the cheap `d_j/s_j²` + group-mean the example describes. For the full matrix it
is a two-sided congruence at O(k³) — 550 G MAC per down_proj, ~8.8 T MAC
model-wide, which **exceeds the sensitivity computation itself**. `ldlq.rs`
already has both halves (`rotate_hessian`, and the AWQ rebase inside the LDLQ
arm), so it is reuse rather than new math, but it must be budgeted.

Fidelity ceiling: still a linearization of ONE layer's output error. It ignores
error propagation through the rest of the network, which is the thing that has
repeatedly surprised us.

### B. Block-wise output error (OmniQuant-style)

The objective `docs/2308.13137.md` argues for: optimize the actual block output,
not a per-tensor proxy. Strictly more faithful than A because it captures the
nonlinearity and the interaction between tensors in a block.

Cost is a forward pass per evaluation, so it needs the runtime in the loop —
architecturally the largest change here. Note `hipfire-runtime`'s calibration
already runs layer-stream forwards, so the machinery is closer than it looks.

### C. Real KLD per candidate

Gold standard, and what hand-picking effectively did. ~10 min per arm
(6.5 min quantize + eval), so a greedy search over 112 tensors is out of
reach; usable only to VALIDATE a cheaper objective on a handful of arms.

## How we would know it worked

The bar is already set and is not a proxy: **beat KLD 0.025025 at or below
4.578 b/w**, on the same slice, same reference. Anything that does not beat
hand-picking on both axes has not earned its complexity — the point of auto
allocation is to find configurations a human would not, not to reproduce one
more expensively.

Secondary check: the ranking should place `v_proj`/`o_proj` near the top, since
we have independent evidence those are the sensitive ones (v alone -13.4%,
o alone -15.1%, together -28.4%).

## Recommendation

1. **Step 0 first.** It is an hour and it can eliminate the rest.
2. **Fix the oq8-residual gain model** regardless — it is a real defect and
   nearly free.
3. Only then decide between A and B, and note the expected prize is modest:
   `docs/2308.13137.md` measures OmniQuant's advantage over AWQ collapsing from
   3.6% to 0.7% once grouping is used, and we quantize at g256. The reason to do
   this is that hand-picking does not scale to the 35B MoE, not that 4-bit dense
   accuracy is currently short.

## Why this matters beyond mixed precision

The same reconstruction-vs-output objective mismatch is behind the clip-search
dead end (`next-phase-goals.md`: the MSE search picks c = 1.0 because MSE says
so). If A or B lands, that machinery applies there too, which changes the
cost-benefit — a shared objective fix rather than a single-feature one.
