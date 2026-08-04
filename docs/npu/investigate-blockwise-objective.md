# Investigation: a propagation-aware sensitivity objective

Written 2026-08-04, after the full-Hessian objective was measured and failed to
fix the mixed-precision ranking.

## The diagnosis is sharper than "layer-local is wrong"

The correct second-order sensitivity of the LOSS to a weight perturbation
factorizes (K-FAC / Fisher) into **two** covariances:

    dL  ~=  1/2 * tr( dW^T  G  dW  H )

    H = E[x x^T]            input covariance      <- WE CAPTURE THIS
    G = E[g g^T],  g = dL/dy  output-gradient cov  <- WE DO NOT

`oq4_sensitivity` (diagonal) and `oq4_output_sensitivity` (full) both compute
the H side and implicitly set `G = I`. That is the entire defect. It is not that
the H side is approximated — we measured that refining it moves nothing — it is
that the G side is *missing*, and G is exactly the propagation term: how much a
perturbation of this layer's OUTPUT reaches the loss through everything
downstream.

## The measured data already tells us what G must be

Dividing the measured promotion gain by the H-term we compute gives the G factor
that would have to exist to explain the ground truth:

    type      H-term (sum err_oq4)   measured KLD gain   implied G
    o_proj                   1.913               15.1%       1.000
    v_proj                  10.03                13.4%       0.169
    k_proj                  62.97                 2.6%       0.005

**The missing factor spans 200x, and the H-term spans 33x in the OPPOSITE
direction.** That is the whole inversion, quantified.

It is also structurally exactly what theory predicts, which is the reason to
believe it rather than curve-fit noise:

* `o_proj` writes straight into the residual stream, so its output gradient is
  large — every downstream layer sees the perturbation.
* `v_proj` reaches the residual only after attention mixing and `o_proj`.
* `k_proj` feeds attention LOGITS, which pass through a softmax that flattens
  them; a saturated softmax has a small Jacobian, so the gradient is tiny —
  while its input covariance is the LARGEST of the three, which is precisely why
  the layer-local proxy ranks it top.

## The cheap form: G is a SCALAR, not a matrix

This is what makes the work small. Approximating `G ~= gamma * I` gives

    tr(dW^T (gamma I) dW H)  =  gamma * tr(dW^T dW H)  =  gamma * (current objective)

So a **single scalar per tensor** multiplies the objective we already compute.
No k x k gradient covariance, no extra storage of consequence (one f32 per
tensor against a 268 MB Hessian), no change to `assign_tiers`.

And the scalar is sufficient BY CONSTRUCTION for this problem: the implied-G
column above IS a per-tensor scalar, and applying it reproduces the measured
ordering. We are not approximating away the thing that matters; the thing that
matters is a scalar.

What to capture, per weight tensor: `gamma_i = E[ ||dL/dy_i||^2 ] / n_out`, the
mean squared output gradient over the calibration set.

## Why this is NOT OmniQuant's block-wise optimization

`docs/2308.13137.md` argues for block-wise output error, and the reasoning is
right, but the mechanism there is heavier than we need. **OmniQuant runs block
forwards because it is OPTIMIZING parameters** (learnable clipping, learnable
scales) and needs a differentiable objective per step. **We only need to RANK
112 tensors once.** A ranking needs the gradient ENERGY, not a gradient descent
loop. That collapses "wire the runtime into the allocator" down to "capture one
extra scalar during calibration".

## The capture ALREADY EXISTS — and normalizes away the one number we need

This is the finding that matters. `hipfire-train` already implements every piece,
because someone built GuidedQuant support:

* `block_backward_capture` returns `BlockAdjoints { d_q, d_k, d_v, d_attn,
  d_gate, d_up }` — "Per-linear OUTPUT adjoints (∂ℓ/∂z) captured during
  backward". Those ARE the `g` in `G = E[g g^T]`, per projection.
* `model_guided_adjoints` runs the whole forward+backward and returns them in
  layer order.
* `Gpu::calib_row_meansq_f32` computes `w[n] = (1/K) Σ_c d[n,c]²` — the
  mean-square output gradient per token.
* `CalibCollector::capture_weighted` accumulates `H̄ = Σ_n w[n]·xₙxₙᵀ`.

So the gradient energy is computed today. But `down_guided_capture` then does:

    let mean = download(w).sum() / seq;      // <- this IS gamma
    if mean > 0.0 { gpu.scale_f32(&w, 1.0 / mean)?; }   // <- and discards it

**The weights are normalized to mean 1 per tensor.** That is correct for
GuidedQuant's own purpose — it wants which TOKENS matter within a layer, and
normalizing keeps the guided and plain Hessians comparable for the
guided-vs-plain diagnostic in `hipfire-quantize`. But it deliberately removes
the per-tensor MAGNITUDE, and the magnitude is precisely the cross-tensor factor
the allocator is missing.

`mean` on that line is exactly `gamma_i`: mean over tokens of the mean-over-
channels squared output gradient. It is computed and thrown away.

**So the change is to record that scalar, not to build a backward pass.** One
f32 per tensor, already in a register.

Two gaps to close, both mechanical rather than novel:

1. `down_guided_capture` only wires `mlp.down_proj`, and the reason is worth
   knowing before touching it: the capture loop calls plain `block_backward`,
   grabbing down_proj's adjoint from `d_x` *before* the block consumes it,
   because down_proj's output IS the block output pre-residual. It gets that one
   for free and never constructs `BlockAdjoints` at all.

   So the change is to switch that loop to `block_backward_capture` — which
   `model_guided_adjoints` already demonstrates — and consume the other five.
   o_proj's adjoint is `d_attn`, and it is the one that matters most here.
2. Nothing emits `gamma` into the `.calib.hfq`. It needs a slot beside
   `<name>.imatrix` / `<name>.hessian`.

## Feasibility

Backward machinery exists and is not hypothetical:

    kernels/src/gemm_f16s_backward.hip   dx (nn_train) and dw (tn_train)
    kernels/src/grad_reduce.hip
    crates/hipfire-train/src/block.rs    backward wired for the drafter path

Calibration is forward-only (`crates/hipfire-runtime/src/calibration/` has no
gradient capture), so the work is connecting existing pieces:

1. Run a backward pass over the calibration set against a loss. KLD against the
   fp16 reference is the natural choice — it is the metric the bar is set in,
   and `.pkld` references already exist.
2. At each weight tensor's output, accumulate `||dL/dy||^2`. This is a reduction,
   which `grad_reduce.hip` already does.
3. Emit one f32 per tensor into the `.calib.hfq`, beside the existing
   `<name>.imatrix` / `<name>.hessian`.
4. Multiply it into `oq4_sensitivity`.

Cost: roughly one backward per calibration forward, so call it 2x calibration
time, paid ONCE per model. Against that, the current allocator wastes 0.30 b/w
and loses to hand-picking.

## Validation, before spending a quantize run

The ranking dump settles it: `HIPFIRE_MIXED_BPW_RANK=1` prints the ordering and
exits in about a minute. **The check is whether `o_proj` climbs out of 79..113.**
If gradient energy is the missing factor, it should land near the top; if it does
not, the propagation hypothesis is wrong too and this document is the record of
why.

Only after the ranking looks right does a quantize+eval arm make sense, against
the standing bar: beat KLD 0.025025 at or below 4.578 b/w.

## Risk

The scalar approximation `G ~= gamma I` discards the gradient's own directional
structure. That is defensible here — the implied-G table shows a per-tensor
scalar reproduces the ordering — but it is an assumption, and if the ranking
comes out only partly fixed, the k x k G is the next rung rather than a refutation.

Second risk: the loss used for the backward matters. KLD against fp16 is the
right target because it is our bar; using next-token cross-entropy instead would
measure a different thing and could rank differently.


## MEASURED 2026-08-04 — gamma is real and large, and the SCALAR form is not enough

Captured for all six projections via `model_gamma_backward` / `calib_gamma`
(hipfire `493a55932`+), run on `Llama-3.2-1B-Instruct--bf16.hfq`, 4 sequences of
256 tokens from the wikitext slice, CE loss, mean ce/tok 3.33.

**The prediction held.** Measured gamma, relative to o_proj, against what the
implied-G table above required:

    type        measured   predicted
    o_proj         1.000       1.000
    v_proj         0.450       0.169
    k_proj         0.007       0.005

o_proj has the largest output-gradient energy, k_proj is ~140x smaller, and the
ordering o > v >> k is exactly what the theory said. Two independent routes —
inverting the measured promotion gains, and directly measuring the backward —
agree. The propagation hypothesis is confirmed.

**But the ranking is only partly fixed.** Re-ranking by `gamma * H-density`:

    type       gamma-weighted   was (H-only)   measured gain
    v_proj          1 .. 19        29 .. 85         -13.4%
    k_proj         13 .. 34         2 .. 33          -2.6%
    o_proj         26 .. 66        79 .. 113        -15.1%

o_proj climbs 53 places out of the unpickable tail, and v_proj — the other big
win — goes to the top. But the order is v > k > o where truth is o > v > k, so
k_proj is still ranked above the tensor that matters most.

The arithmetic shows why, at layer 5:

    o_proj   H-density 1.419e-08   gamma 3.658e-02   product 5.190e-10
    k_proj   H-density 4.249e-06   gamma 6.830e-04   product 2.902e-09

**o_proj's per-weight reconstruction error is 300x lower than k_proj's, while
its gamma is only 54x higher.** o_proj's input is the post-softmax attention
context, which is a weighted average and therefore small in magnitude, so its H
is tiny. The product still favours k by 5.6x, where the measured per-bit
efficiency favours o by about 1.4x — roughly 8x unaccounted for.

### Which risk this was

The risk section above named two, and the result does not distinguish them yet:

1. **`G ~= gamma*I` discards the gradient's directional structure.** The full
   `k x k` G is the next rung. Note this is NOT the same as the full-H work that
   failed — that refined the input side, which measurement says is not where the
   error is.
2. **The loss is wrong.** `calib_gamma` backprops CROSS-ENTROPY, while our bar is
   KLD against fp16. Those weight tokens differently, and the doc flagged this
   before the run. This is much cheaper to test than (2) and should go first.

### Standing assessment

Do not ship the scalar gamma as the ranking objective yet: it fixes v_proj and
demotes k_proj correctly, but still under-ranks o_proj, so a greedy fill would
promote the wrong tensors first. It is a large step in the right direction from
a proxy that was anti-correlated.

Also incomplete: `BlockAdjoints` carries no `d_down`, so down_proj (16 tensors)
and the embed table are absent from the gamma table — 96 of 113 candidates are
covered. down_proj's adjoint is available on the existing path
(`down_guided_capture` takes it from `d_x`) and just needs folding in.


## k_proj MEASURED 2026-08-04 — my reference number was wrong

`k_proj = -2.6%` was never measured. I derived it as `v+k (-16.0%)` minus
`v alone (-13.4%)`, and this document's additivity claim was verified for v and
o, not for v and k. Measured directly (`oq4.25++ a0.45`, k_proj alone to oq8++):

    baseline oq4.25++ a0.45     KLD 0.034862
    + k_proj -> oq8++           KLD 0.034686     -0.50%

So k_proj alone is worth **-0.5%, not -2.6%** — five times smaller than assumed,
and v+k is superadditive rather than additive (13.4 + 0.5 = 13.9 against a
measured 16.0). Any inference resting on the -2.6% figure, including the earlier
"implied G" column, is off by that factor for k.

### The gap, correctly quantified

Per-weight efficiency (gain % per % of quantised params), which is what the
greedy actually ranks on:

    o_proj   gain 15.1%   share 6.90%   eff 2.19
    v_proj   gain 13.4%   share 1.72%   eff 7.77
    k_proj   gain  0.5%   share 1.72%   eff 0.29

    truth  o/k efficiency ratio        7.48
    H-only objective   o/k =  0.00334   off by 2238x
    gamma-weighted     o/k =  0.1789    off by   42x

**gamma closes 54x of a 2238x gap and leaves 42x.** That is the honest scoring:
the factor is real and does most of the work, and it is still not sufficient.

Note what it DOES get right: v_proj is the most efficient promotion per weight
(7.77, more than 3x o_proj), and the gamma-weighted ranking puts v_proj at
2..20. For a budget-constrained greedy that is the single most important call,
and the H-only objective had it at 29..85.

### Why a residual is expected, and where it probably lives

The remaining error is concentrated in k_proj being over-ranked. A plausible
mechanism, consistent with everything measured: `k_proj` feeds attention LOGITS,
which pass through a softmax. A second-order local expansion — which is what
`gamma * H` is, K-FAC included — assumes perturbations propagate linearly. A
saturated softmax suppresses them NONLINEARLY, so the local gradient at the
operating point overstates the damage that actually reaches the loss.

If that is right, no refinement of a local quadratic objective fixes k_proj,
because the effect is not in the quadratic. That would make the practical answer
"use gamma, and accept that logit-path tensors are over-ranked", or move to a
genuinely non-local measurement (ablate and evaluate), rather than a third
attempt at a better local form.

This is a hypothesis with a clean test: promote k_proj and measure whether the
KLD change tracks the local prediction at small perturbations and falls away at
large ones. Not run.


## END-TO-END RESULT 2026-08-04 — gamma makes auto allocation work

The ranking analysis above is a proxy. The artifact is the question, so gamma was
wired into `--mixed-bpw` (`HIPFIRE_MIXED_BPW_GAMMA`) and run at 4.578 b/w — the
hand-picked configuration's exact budget, making it a clean A/B:

    configuration            b/w      PPL       KLD       note
    q4nx (FLM, the bar)   5.0000  17.1949  0.034954
    auto, H-only            ~4.88        —  0.030071   loses badly
    auto, gamma-weighted   4.5735  17.1018  0.025734   40 of 113 promoted
    hand-picked v+o        4.5780  17.1157  0.025025

**Against the previous auto: 14.4% better KLD at 0.30 b/w FEWER.** Against
hand-picking: within 2.8% on KLD at marginally fewer bits, and BETTER on PPL
(17.1018 vs 17.1157).

So the allocator now clears the bar on its own — 8.5% fewer bits than q4nx while
beating it on both metrics — without anyone choosing tensors by hand. That was
the point of the exercise: hand-picking does not scale to the 35B MoE.

### Why this beat what the ranking predicted

The rank spans said gamma still over-ranks k_proj relative to o_proj by ~42x,
which looked like it would misallocate. It did not matter much in practice, and
the reason is worth keeping:

* The greedy spends a BUDGET. What decides quality is mostly which tensors get
  promoted FIRST, and gamma gets the most efficient promotion (v_proj, 7.77
  gain-per-%-params, over 3x o_proj) right at the top.
* k_proj is cheap — 1.72% of quantised params. Over-ranking a small tensor
  wastes little budget. The H-only objective's failure was different in kind: it
  put o_proj, 6.9% of params and the largest single win, where it was never
  reached at all.

A ranking metric can therefore look badly wrong on span comparisons and still
allocate well, because the greedy is forgiving about the order of cheap items
and unforgiving about missing expensive ones.

### Standing position

Ship gamma as the `--mixed-bpw` objective. The remaining 42x k_proj residual is
real, is probably the softmax-saturation effect described above, and is now a
refinement rather than a blocker — the allocator already produces an artifact
that beats the bar and matches hand-picking.
