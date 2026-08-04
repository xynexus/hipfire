# linear_attn on a real model: what works and what does not

**Status: the assembly runs at 35B scale and is NOT yet numerically correct.**

Earlier commits in this session said the hybrid assembly was "validated on a
real model". That was too strong, and this note is the correction. What was
validated was PLUMBING — layer dispatch, loaders, streaming, shapes. Running a
real model over real text shows the forward does not reproduce the model.

## The measurement

`gamma_hybrid` now tokenizes with the artifact's own embedded tokenizer over
`benchmarks/calib/calib-1m.txt`, so the loss is meaningful for the first time.
A working Qwen3.5-0.8B should land around 2-3.

| state | mean loss |
|---|---|
| before the q/k L2 norm | 10.27, plus NaN in gamma from layer 20 down |
| after the q/k L2 norm | 13.33, no NaN |
| ln(vocab) = uniform | 12.42 |

So one real bug is fixed and the model is still wrong.

## Fixed: the missing q/k L2 norm

`fused_qk_l2_norm_scale_f32` sits between the conv split and the recurrence —
`q *= rsqrt(sum(q^2)+eps)` then `*= 1/sqrt(hd)`, and `k *= rsqrt(sum(k^2)+eps)`
(`qwen35/lowered.rs:464`). This core omitted it entirely.

Without it the delta-rule state is unbounded: `S_t = alpha*S + k (x) delta` with
un-normalised `k` grows until the backward overflows. That is exactly the
observed signature — clean at seq 8, NaN at seq 64, first appearing in the
middle of a layer's core rather than at its edges.

Nothing self-consistent could have caught this. The layer gradchecked at 5e-4,
stayed exactly causal, matched the conv/activation/gated-norm kernels to 1e-7,
and ran at 35B. A missing normalisation is still a differentiable function of
its inputs.

## Unresolved: the AWQ fold, and a claim to retract

A previous commit asserted the fold direction was "measured, not assumed":
divide 11.02 < no fold 11.70 < multiply 12.21. Those numbers came from
SYNTHETIC token ids, where every variant sits near ln(vocab) and the spread is
noise. Retract that reasoning.

Re-measured on real text, the ordering inverts: no fold 11.76, divide 13.33,
multiply 13.43.

The divide direction is what the contract states — `supports_awq_sidecar`
includes `Oq8G256`, and the comment is explicit that "SmoothQuant migrates a
per-channel scale into the weight offline (W·s); the forward divides x by the
awq_scale sidecar, completing (W·s)·(x/s) = W·x". The code still divides,
because a documented contract outranks a loss comparison taken on a forward
that is known to be broken.

The general lesson, which cost two wrong conclusions here: **do not tune
sub-decisions by loss while the forward is broken.** Every variant is being
scored through the same unknown defect.

## Next

The remaining defect is not yet located. Untested candidates, roughly ordered:

- rope on the full-attention layers. This is a VL checkpoint; `head_dim` is
  256 and mrope/partial-rope handling has not been checked against the arch.
- `linear_num_key_heads` vs `linear_num_value_heads`. The inference path calls
  `repeat_interleave_qk_f32` when key heads < value heads; this loader derives
  one head count for both. Equal on the 0.8B, NOT verified on the 35B.
- the embedding/lm_head tie, and whether any scale applies on either side.
