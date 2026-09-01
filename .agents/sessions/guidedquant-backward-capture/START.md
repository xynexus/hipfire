# Session: end-loss gradient capture, and GuidedQuant on top of it

**Blocked on:** nothing technical. **Est:** multi-session / research-scale.
**Value:** the highest quality ceiling on the list — and the keystone already exists.

## Objective

Capture end-loss gradients during calibration, then implement GuidedQuant against
them. Everything downstream (YAQA, PV-Tuning) needs the same capture.

## Why now

Held-out finding, and it is the uncomfortable one: **plain XᵀX LDLQ scores about
the same as no calibration at all.** GuidedQuant was the only robust winner in
that comparison. Today's calibration is **forward-only**, which is the mechanical
reason most of the calibration surface measures as noise.

The keystone is not missing: **`hipfire-train` is real fp32 GPU autograd**, and
step 1 was already done on Llama-3.2-1B. What is missing is extending gradient
capture to a real target and wiring it into the quantizer's objective.

## First moves

1. Confirm the current state of `hipfire-train`'s autograd against a real target
   — the Llama-3.2-1B result is the existing foothold.
2. Decide where captured gradients live. Calibration already produces
   `.calib.hfq` (Hessians + imatrix); a gradient plane is the natural sibling and
   should follow the same artifact discipline.
3. Only then touch the quantizer objective.

## The verification bar

**Downstream KLD on a held-out corpus, never reconstruction error.** Two codecs
that reconstruct at per-row cosine 0.99999 differed by ~0.06 KLD downstream —
reconstruction MSE is not a valid proxy and has misled this project before.

## Traps

- **Never calibrate and evaluate on the same corpus.** A "-13.6% from more
  calibration tokens" result had to be retracted as train-on-test; the damage is
  invisible in the numbers and shows up as a fake improvement plus an inverted-U
  budget curve. `reject_eval_corpus` guards the obvious case — do not defeat it.
- Calibration sequence length now defaults to 2048 (`DEFAULT_CALIB_SEQ_LEN`).
  That matches the n_ctx KLD references are built at; changing it changes what
  you are comparing against.
- Calibration capture WAS nondeterministic — a data race in
  `zaya_value_compose_f32`, root-caused and fixed 2026-08-30. If two runs over
  identical inputs disagree again, suspect the same class before the algorithm.

## Reference

`docs/plans/` and `./Quantization-research/` carry the GuidedQuant / YAQA
material. Check `git log --grep` and `docs/todo` before starting — nine of twenty
items on the last ranked list turned out to be already done or void.
