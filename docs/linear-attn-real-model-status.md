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

## Bisected: the defect is in the FULL-ATTENTION path

The bf16 source turned out to be on disk after all —
`/srv/hipfire/archives/models--Qwen--Qwen3.5-0.8B.hfa`, which
`hipfire-coexistence repack` restores losslessly to an HF directory. That
removes quantization from the picture entirely, and the first thing it says is
that quantization was never the problem:

| source | mean loss, real text |
|---|---|
| oq8++ artifact (dequantized, AWQ folded) | 13.331 |
| bf16 source, no quantization anywhere | 13.338 |

Identical. So the AWQ fold question is moot for this defect, and the earlier
retraction stands for a second reason: the fold was never what was wrong.

Skipping whole layer types localizes it:

| configuration | mean loss |
|---|---|
| everything on | 13.34 |
| linear_attn layers skipped (pass-through) | 22.33 |
| **full-attention layers skipped** | **11.44** |
| uniform, ln(248320) | 12.42 |

The linear_attn layers are working — removing them costs 9 nats. The
full-attention layers are ACTIVELY CORRUPTING: the model is better off without
them. That is where the remaining defect lives.

Inside that block, narrowing further:

| variant | mean loss |
|---|---|
| attn_output_gate applied, per-head interleaved (current) | 13.34 |
| gate not applied at all | 10.21 |
| gate applied, Q/gate halves swapped | 10.23 |
| gate applied, block layout `[all Q | all gate]` | 10.71 |
| rope disabled entirely (theta 1e30) | 13.35 |

Two readings. Rope is NOT the dominant issue — disabling it moves the loss by
0.01. And every gate variant is worse than not applying the gate at all, so the
gate handling is wrong in a way that guessing has not resolved.

The code still applies the gate per-head interleaved, because that is what
`kernels/src/deinterleave.hip` documents and `qwen35/lowered.rs` calls, and the
lesson from the AWQ episode is not to overturn a documented contract on a loss
comparison taken through a forward that is still broken.

## Still open

- **The attention block defect itself.** Not located, and variant-guessing has
  now failed twice (rope, then the gate table). Stop guessing and use the
  oracle: `hipfire-runtime/examples/dump_qwen35_hidden_states.rs` dumps
  per-layer hidden states from the runtime's own forward in a documented binary
  format. Comparing this walk's per-layer output against it pinpoints the first
  divergent layer — and, within a layer, whether it diverges before or after
  the attention block — in one run instead of a variant at a time.

  That rope contributes almost nothing (0.01 nats either way) is itself
  diagnostic: if attention were working, positional information would matter a
  great deal. An attention output that is indifferent to position is one whose
  context is already wrong.
- ~~`partial_rotary_factor` = 0.25 is not honored.~~ FIXED: `BlockDims::
  rotary_dim` (0 = full head), gathered/rotated/scattered around the rotary
  slice, gradchecked, and falsified (a wrong scatter stride shows as dAq 2.6e-1
  while dAv, which never crosses rope, stays 3e-4). It did NOT move the loss:
  13.34 -> 13.39. Correct, and not the defect — as the rope-disable measurement
  had already predicted.
- **mrope.** `rope_parameters` has `mrope_interleaved: true` and
  `mrope_section: [11, 11, 10]`. Unexamined.
- **`linear_num_key_heads` vs `linear_num_value_heads`.** Equal (16/16) on the
  0.8B, so not implicated here; still unverified on the 35B, where the
  inference path calls `repeat_interleave_qk_f32` when they differ.
