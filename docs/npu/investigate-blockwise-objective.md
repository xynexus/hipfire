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
