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
