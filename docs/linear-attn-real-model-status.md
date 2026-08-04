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

## The oracle says layer 0 — and that supersedes the bisect above

`dump_qwen35_hidden_states` now runs (it needed two fixes, below) and dumps the
runtime's own per-layer states on the SAME 64 tokens this walk uses. Comparing
cosine similarity, aligning this walk's layer INPUTS against the runtime's
layer outputs:

| layer | ref rms | mine rms | cos |
|---|---|---|---|
| 0 | 0.0582 | 0.0258 | +0.46 |
| 1 | 0.0745 | 0.0302 | +0.34 |
| 2 | 0.0798 | 0.0282 | +0.26 |
| 9 | 0.0719 | 0.0452 | +0.03 |
| 23 | 0.4063 | — | — |

Divergence is immediate. Layer 0 — a `linear_attn` + dense-MLP layer — already
produces the wrong output, and the residual stream never recovers.

The embedding is NOT the cause and is now positively verified: the walk's
layer-0 input matches the raw bf16 `embed_tokens` row for the same token id at
cosine 0.99999 (the residual is the oq8 quantization of the embed table). So
the divergence is entirely inside layer 0's own computation.

This supersedes the layer-type bisect above. That measurement said attention
layers hurt, and they may well, but "removing a wrong layer helps" is not the
same as "the wrong layer is the only wrong one" — the linear_attn path is
also wrong, from the very first layer, and it is the one to fix first because
everything downstream inherits it.

What is NOT implicated, each positively checked: the embedding lookup, the
conv/activation/gated-norm/recurrence math (all match kernels to ~1e-7), and
quantization (bf16 and oq8 score within 0.007). What has never been checked
against anything is the ASSEMBLY around the verified core — norm1, the four
input projections, out_proj, and the two residual joins.

## A third implementation says the UNDERSTANDING is wrong, not the code

`tools/qwen35_layer0_oracle.py` recomputes layer 0 in numpy straight from the
bf16 safetensors, following the documented formulas. Comparing all three:

| pair | cosine |
|---|---|
| numpy vs the Rust walk | **0.9953** |
| numpy vs the runtime | 0.4515 |
| the Rust walk vs the runtime | 0.4609 |

The Rust faithfully implements what I understand the layer to be. What I
understand it to be is wrong. Those need different fixes, and every hypothesis
is now a seconds-long numpy edit instead of a rebuild.

Ruled out by this, on top of what was already cleared:

- The gated norm. Five variants tried; the current one (`rmsnorm * w * silu(z)`)
  is the best by a wide margin — 0.4515 against 0.0254 ungated, 0.1294 with
  `sigmoid(z)`, 0.3162 with an l2 norm, 0.3564 with a `sqrt(hv)` rescale. That
  agrees with the kernel cross-check.
- Position ordering in the reference dump. Checked all 64 rotations and
  per-row best matches; there is no offset that aligns them, so the two are
  genuinely different states rather than misaligned ones.
- The reference's meaning. `decode_layers.rs:949` writes `s.x` into the ring
  buffer immediately after the "LinearAttention residual" trace point, so it
  IS the post-layer residual stream, as assumed.

### Position 0 diverges, which narrows it hard

`cos(numpy, ref)` at position 0 is 0.489 — and at position 0 the recurrence has
no history and the conv sees a single token, so the whole layer collapses to a
closed form. The defect is in the PER-TOKEN math, not in sequential state
handling. That eliminates the conv ring buffer, the state carry, and every
ordering question about time.

### Variant sweep at layer 0

Each is one edit in the numpy oracle (`VAR` argument), scored as mean cosine
against the runtime over 64 positions:

| variant | pos0 | mean | last |
|---|---|---|---|
| base (current) | 0.489 | 0.452 | 0.298 |
| **`input_layernorm` as `1 + w`** | **0.552** | **0.619** | **0.633** |
| no `input_layernorm` | 0.438 | 0.284 | 0.086 |
| conv taps reversed | 0.440 | 0.430 | 0.305 |
| q/k swapped in the split | 0.489 | 0.439 | 0.229 |
| `[V|Q|K]` split order | 0.435 | 0.247 | 0.036 |
| no conv at all | 0.282 | 0.165 | -0.001 |
| no silu after conv | 0.510 | 0.454 | 0.290 |

Everything except the unit-offset norm is worse, which is further confirmation
for the conv, the split order, and silu.

### The `1 + w` lead is NOT acted on

It is the one variant that improves, and substantially. But:

- No unit-offset appears anywhere in hipfire's qwen35 rmsnorm path, and the
  runtime ships working Qwen3.5 inference, so plain `w` is what actually runs.
- The weight statistics are ambiguous rather than supporting: layer-0
  `input_layernorm` centres on +0.24 and layer 5 on +0.43 — neither the ~0 a
  `1 + w` convention implies nor the ~1 of a plain weight. The final norm
  centres on +3.31, which fits neither.

So this is recorded as a lead, not a change. Acting on a metric improvement
that contradicts the implementation is exactly the mistake the AWQ retraction
above was about, and one improving number is not worth repeating it. What
would settle it: find where the runtime applies these two norms and read
whether anything is added to the weight.

One unexplained observation worth chasing: in this implementation the MLP
branch contributes almost nothing — rms 0.0011 against the attention branch's
0.0135 and the residual's 0.0206. A dense SwiGLU MLP contributing 5% of the
stream is not normal.

## A whole-model numpy forward fails the same way — and the oracle is suspect

`tools/qwen35_full_forward_oracle.py` implements all 24 layers independently in
numpy: linear_attn and full attention, QK-norm, partial rope, GQA, the output
gate. Next-token NLL **11.70** against a uniform 12.42 — the same failure as the
Rust walk's 13.34. Three implementations now agree with each other and all fail,
so the defect is in the reading of the architecture, not in either codebase.

Also settled this round, by reading rather than measuring: `kernels/src/rmsnorm.hip`
computes `out[idx] = x[idx] * weight[i] * rms`. Plain weight, no unit offset. The
`1 + w` improvement recorded above is an artifact, and not acting on it was right.

**But the oracle itself does not pass a basic sanity check.** Putting the
runtime's own final hidden state through the final norm and tied lm_head:

| predicts | NLL | acc |
|---|---|---|
| token[i-1] | 7.14 | 0.06 |
| token[i] | 6.08 | 0.22 |
| token[i+1] (next-token) | 8.16 | 0.06 |

A working Qwen3.5-0.8B should score ~2-3. No shift rescues it, and hidden-state
alignment confirms no offset (shift 0 is the best match at every layer). So the
reference states cannot predict text either.

Cleared while establishing that: tokenization round-trips exactly (ids decode
back to the corpus text), and the embedding matches raw weights at cos 0.99999.

That leaves two things needing explanation rather than one:

1. This implementation is wrong — established independently of the oracle, since
   the walk and the numpy forward both sit near uniform on their own.
2. The oracle is wrong, or `dump_qwen35_hidden_states` is. Two bugs have already
   been found in that example this session, and it drives an unusual per-token
   decode path. The runtime ships working Qwen3.5 inference with KLD baselines,
   so the dumper is the more likely culprit than the engine.

**Resolved: the engine is fine, the dumper is not.** Generating from the same
`qwen3.5-0.8b.oq8++.hfq` through the normal path (`hipfire chat`) produces
correct, coherent output — "The capital of France is Paris", with a sensible
reasoning trace, 138 tokens at 68 tok/s. The forward that ships is right.

So `dump_qwen35_hidden_states` is producing bad hidden states: its NLL 6.08 is a
property of that path, not of the model or the artifact. Three bugs have now
turned up in that example (the ring-buffer constructor, the stale kldref magic,
and now the states themselves), which is a fair characterisation of how much it
had rotted.

That retires the second unexplained thing. The first stands unchanged and on its
own evidence: this implementation is wrong, with the walk at 13.34 and an
independent numpy forward at 11.70 against a uniform 12.42 — neither of which
involves the oracle at all.

Note for anyone running the CLI here: the user config sets `dflash_mode: on`
globally, and a model without a DFlash sidecar fails to load with
"dflash_mode=on but no explicit, embedded, or sibling DFLASH component". Run
with a shadow `HOME` whose `.hipfire/config.json` sets it off rather than
editing the real config.

## A trustworthy oracle exists now, and it sharpens the symptom

`dump_logits_qwen35` runs the real prefill path — the one generation proved
correct — on the deterministic prompt `0,1,2,...`, so the tokens are trivially
reproducible. Comparing last-position logits after `0..63`:

| | top-5 ids | rms |
|---|---|---|
| runtime | 44576, **64**, 91, 61, 93 | 3.28 |
| this implementation | 220, **63**, 96110, 17, 271 | 3.11 |

cos 0.566. The runtime continues the count — 64 after 0..63. This
implementation's second choice is 63: the CURRENT token.

### Correction: "the branches are four times too weak" was wrong

That reading came from comparing branch magnitudes against the hidden-state
dumper's per-layer rms — and that dumper is the component just shown to be
broken. Reasoning from its numbers was a mistake; measurements taken with only
trustworthy sources say something different:

| measurement | value |
|---|---|
| `cos(final h, embedding of the current token)` | **0.14** |
| `\|final h\|` vs `\|embedding\|` | 5.37 vs 0.71 |
| `logit[64]` (the correct next token) — mine / runtime | **+2.78** / +16.15 |
| `logit[63]` (the current token) — mine / runtime | +9.84 / +13.63 |

The residual stream grows 7.5x through the network and ends up nearly
orthogonal to the current token's embedding, so the layers are emphatically not
inert. The real symptom is narrower: **both models rank 63 highly, and only the
runtime also promotes 64.** This implementation fails to produce the increment,
not to produce a signal.

Logit magnitudes are comparable (rms 3.11 vs 3.28), so this is a ranking
failure rather than a scale failure. That rules out a uniform magnitude bug —
which is what the retracted reading would have sent me looking for.

## Single token already diverges

`dump_logits_qwen35 --prefill 1` against a one-token numpy forward: **cos 0.79**.

That is the sharpest narrowing yet. With one token there is no cross-token flow
at all — the conv sees a single tap, the recurrence is one step
(`S = (v*beta) (x) k`, `out = S.q`), and attention's softmax is over ONE key, so
it reduces to `ctx = v * sigmoid(gate)` with rope and QK-norm irrelevant
(position 0 rotates by nothing; q and k cannot affect a one-element softmax).

So the defect is in per-token math, and it iterates in seconds rather than six
minutes.

Variant sweep at one token, cos against the runtime:

| variant | cos |
|---|---|
| base | 0.7896 |
| drop q's 1/sqrt(hd) scale | 0.8349 |
| attention gate: Q/gate halves swapped | 0.8085 |
| attention gate: block layout | 0.7907 |
| attention gate: off | 0.7883 |
| MLP: gate/up swapped | 0.7511 |
| MLP: off entirely | 0.7532 |

Nothing is decisive — the whole spread is 0.75-0.83, and the best (dropping the
q scale) contradicts a kernel this was verified against at 1e-7. No single
structural flip explains it, which suggests either a small error compounding
across 24 layers, or something about this checkpoint not visible in tensor names
and shapes.

Worth noting what the MLP rows say: turning the MLP off entirely costs only
0.036 cos. At one token, 24 SwiGLU blocks are apparently near-irrelevant to the
output direction — which is itself odd and may be a clue rather than noise.

### Next

The dumper's per-layer states are untrustworthy for SEQUENCES, but its bug may
well be sequence-related (ring buffer, state carry, decode-from-cold). At
`n_ctx = 1` there is no ring to wrap and no state to carry. If its layer-0
state at one token matches this implementation's, it is usable for exactly the
narrow case that now matters — and one layer at one token is a small enough
system to check exhaustively.

## Superseded: the old next-step

Per-layer states are what localise a defect, and that source is now known-bad.
Two options, in order of cost:

1. `dump_logits_qwen35` — if the runtime's LOGITS score a sane NLL on these
   tokens, that is a valid end-to-end oracle immediately, though it only says
   "wrong", not "wrong from layer N".
2. Fix the hidden-state dumper. More work, but per-layer comparison is what
   actually finds this.

Either way the target is the same: something in the architecture reading is
wrong in a way three implementations share, and the runtime — which is correct
— is the only thing that can say what.

## Two fixes the oracle itself needed

- `dump_qwen35_hidden_states` asked `HiddenStateRingBuffer::new` for all 24
  layers. That derives ids via `dflash_extract_layer_ids`, which spaces
  `num_extract` picks across `1..n_layers-3` — for all-layers the rounding
  collides and the constructor rejects duplicate ids. Switched to
  `new_for_layers` with an explicit list.
- It expects a `HFKLDR` kldref, while every kldref on disk is now `HFQM` or
  `HFKREF`. Rather than guess at those layouts, `gamma_hybrid` writes a minimal
  token shim in the format the dumper does read — which also guarantees both
  sides run the identical token sequence.

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
