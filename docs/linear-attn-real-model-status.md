# linear_attn on a real model — FIXED

**Two separate defects, both resolved. Read this first; the rest of the file is
the investigation log, in the order it happened, retractions included.**

| # | defect | where | symptom | section |
|---|---|---|---|---|
| 1 | GemmaRMSNorm — norms are `rmsnorm(x)*(1+w)`, applied plainly | the walk's loader | loss 13.34, cos 0.7896 | immediately below |
| 2 | untied `lm_head` scored through the embedding | the HARNESS (`gamma_hybrid`) | 35B cos ~0 at one token | *SOLVED: the harness scored the 35B through the wrong head* |

Current state — every testbed reproduces the runtime's own prefill logits:

| testbed | head | cos |
|---|---|---|
| Qwen3.5-0.8B, 1 and 4 layers | tied | 0.999993 |
| qwen3_5_moe-tiny, 1 layer | tied | 0.999978 |
| GQA fixture, `n_k=1 < n_v=2` | tied | 0.999982 |
| Qwen3.6-35B-A3B, 1 layer | **untied** | 0.999990 |

**End-to-end on the real artifact.** `Qwen3.6-35B-A3B--oq4++.hfq`, all 40
layers, real tokens from the artifact's own tokenizer over `calib-1m.txt` at
seq 64:

| model | mean loss, real text | ln(vocab) |
|---|---|---|
| Qwen3.5-0.8B | 3.52 | 12.42 |
| **Qwen3.6-35B-A3B (oq4++)** | **2.502** | 12.42 |

The 35B lands below the 0.8B, which is the ordering a working 35B should
produce. On the synthetic `0,1,2,...` prompt the same stack gives 5.036 — that
number says the stack is coherent, NOT that it models language, and the harness
now refuses to call it "predicts text".

Defect 2 was in the measuring instrument, not the model. Everything the log
below attributes to "the 35B's layer math" was the walk's output projected
through the wrong matrix — the 35B's `lm_head` correlates with its own
embedding at 0.025. The layer math was never wrong.

**Regression fixture.** `/srv/hipfire/fixtures/qwen3_5_moe-tiny-untied` is
`qwen3_5_moe-tiny` (already GQA, `n_k=1 < n_v=2`) plus a deliberately unrelated
`lm_head.weight` and `tie_word_embeddings: false`. Its head correlates with its
own embedding at **-0.0004**, so scoring through the embedding cannot give a
right answer by luck — the counterfactual is guaranteed by construction. Built
through the same pipeline it reproduces at **cos 0.999985**, in seconds rather
than the ten minutes the 19 GB artifact takes. Any regression of the tied/untied
handling shows up here first:

    hipfire-quantize --input <fixture> --output bf16.hfq --format bf16
    dump_logits_qwen35 bf16.hfq ref_4.f32 --prefill 4
    HIPFIRE_GH_ORACLE_DIR=<dir> gamma_hybrid bf16.hfq arange 4

What the hunt did leave behind, all of it real coverage that did not exist
before: the linear_attn cross-checks now run at the model's true geometry
(32 value / 16 key heads, h=2048) instead of a 2-head toy; the recurrence is
checked at 32 heads out to seq 64; the MoE router is checked at 256 experts and
top-8; and the 1-layer truncation harness is validated against a known-good
4-layer result.

The methodological lesson, since it recurred six times: **every wrong
conclusion in this file came from an inference that was never tested
directly.** The GQA hypothesis was eliminated-by-exhaustion and died in one
run against a purpose-built GQA fixture. The seq-1 variant sweep measured a
quantity structurally blind to the variable. The fix itself came from reading
what the failing artifact actually contained rather than reasoning about what
distinguished it. Build the direct test first; it is almost always cheaper
than the argument.

---

**The first defect was GemmaRMSNorm.** Qwen3.5/3.6 store their block and final norm
weights as deviations from 1, so the norm is `rmsnorm(x) * (1 + w)`. This
implementation applied `w` plainly.

| | mean loss, real text |
|---|---|
| before | 13.34 (uniform is 12.42) |
| after, block + final norms | 4.03 |
| **after, also q/k norms** | **3.52** |

At one token, against the runtime's own prefill logits: cos 0.7896 -> **0.9994**.

The runtime settles exactly which norms take it, and it is not uniform:

| tensor | offset? | how the runtime loads it |
|---|---|---|
| `input_layernorm`, `post_attention_layernorm` | yes | `load_norm_weight` (`loading.rs:4419`) |
| final `norm.weight` | yes | `load_norm_weight` (`loading.rs:4210`) |
| `q_norm`, `k_norm` | yes | `load_norm_weight` (`loading.rs:4571`) |
| `linear_attn.norm` | **no** | `load_any_as_f32` (`loading.rs:4475`) |

The q/k row is worth dwelling on: a one-token measurement CANNOT see it, because
softmax over a single key ignores q and k entirely. The measurement that found
the main bug was structurally blind to this part of it, and only the runtime's
loader settles it. Adding it took the loss from 4.03 to 3.52.

The stored weights corroborate the split — layer-0 `input_layernorm` centres at
+0.24 where a plain weight would centre at 1, while `linear_attn.norm` centres
at +0.96.

The runtime also records that skipping the offset on MoE final norms was tried
and reverted: it "under-scaled the MoE final norm by ~38% ... 1.63 instead of
the correct 2.63 = 1 + 1.63 that vLLM/llama.cpp produce", and the symptom it was
masking had a different root cause entirely.

## Dense path: verified. MoE path: still broken.

With GemmaRMSNorm, the q/k unit offset, and the 16/32 key-value head split all
in, the DENSE 0.8B is healthy and several long-standing questions are settled by
measurement rather than argument:

| configuration | mean loss, real text |
|---|---|
| bf16 source | **3.5187** |
| oq8++ artifact (AWQ folded) | **3.5147** |
| oq4 (quantized here from the bf16 source) | 3.7960 |
| uniform | 12.42 |

- **The AWQ fold is correct.** bf16 carries no AWQ sidecars at all and the
  folded oq8 artifact matches it to 0.004 nats. This closes a question that was
  answered wrongly twice earlier in the session — first tuned on synthetic
  tokens, then re-litigated through a broken forward. With a working model it
  takes one measurement.
- **Oq4G256 decode is correct.** 3.796 against 3.519 is a plausible 4-bit cost,
  not a defect.

The 35B, with every one of those fixes, still sits at **12.10** against a
uniform 12.42. The dense path is exonerated end to end, so what remains is the
part the 0.8B does not exercise: the routed MoE — 256 experts, top-8, a shared
expert, and the per-expert-FUSED `gate_up` layout, which is the one expert
layout never checked against anything but its own shapes.

### The numpy oracle has its own bug

Its accuracy decays with sequence length — cos 0.9994 at one token, 0.9741 at
four, 0.207 at 64 — which is characteristic of an error that grows with the rope
angle. One cause found: it paired rope dims interleaved `(2i, 2i+1)` while
`kernels/src/rope.hip` pairs half-split `(i, i + n_rot/2)`. Both conventions
ship in this repo (`rope_tail_interleaved.hip` is the DeepSeek one), and the
difference is invisible at position 0.

Fixing that moved four tokens 0.9689 -> 0.9741, so it was not the whole story
and the oracle still has a length-dependent defect. This does not implicate the
Rust, which uses hipfire's rope kernel directly and scores 3.52 on real text;
it does mean the numpy oracle is only trustworthy at short lengths until the
rest is found.

## The 35B is wrong at ONE token, and nothing found yet explains it

| model | seq 1 | seq 4 |
|---|---|---|
| Qwen3.5-0.8B | 0.9996 | 0.9998 |
| **Qwen3.6-35B-A3B** | **0.0038** | **-0.1854** |

Orthogonal, not merely degraded. At seq 1 there is no cross-token flow and the
MoE is fully exercised, so this is per-token math specific to this model.

Ruled out for the 35B specifically:

- **Tokenization** — its embedded tokenizer emits the same ids as the 0.8B's
  for the same text, and the comparison above uses the `arange` prompt anyway.
- **Config/tensor agreement** — 16 heads, 2 KV heads, head_dim 256, n_rot 64,
  q_proj 8192 = 2*4096 so the output gate is detected; 32 value / 16 key linear
  heads with qkv 2*16*128 + 32*128 = 8192 exactly.
- **The repeat-interleave convention** — `repeat_interleave_qk.hip` documents
  `dst[(kh*ratio + r)] = src[kh]`, so value head j reads key head j/ratio, which
  is what this implements.
- **The expert AWQ fold** — disabling it for expert tensors moves cos from
  0.0038 to -0.0906. Both are noise around zero; it is not the cause.

One anomaly found and not yet explained: `experts.0.down_proj` dequantizes to
raw rms 0.151 where `gate_up_proj` is 0.0073 and the shared expert's own
`down_proj` is 0.0118 — 13-20x its siblings. Its AWQ sidecar spans 0.083 to
44.6 (mean 9.86), against 0.32-5.5 for gate_up and 0.55-2.4 for the shared
expert. A 500x dynamic range on one tensor's scales is unusual enough to note
even though removing the fold did not fix anything.

### The missing fixture was built — and n_k < n_v is NOT the cause

Synthesised from `qwen3_5_moe-tiny` by halving `linear_num_key_heads` (2 -> 1)
and narrowing `in_proj_qkv` and `conv1d` to match (768 -> 512 rows), then
quantised to bf16 and compared at seq 1:

| fixture | seq 1 |
|---|---|
| qwen3_5_moe-tiny (n_k == n_v) | 1.0000 |
| synthesised n_k < n_v | **0.9915** |

So the GQA linear-attention path carries a small real discrepancy worth fixing
— 0.9915 where its sibling is exact — but nothing like the 35B's 0.0038. It is
not the cause.

Also cleared for the 35B this round:

- **The artifact is good.** `hipfire chat` on
  `Qwen3.6-35B-A3B--oq4++.hfq` answers "The capital of France is Paris" with a
  coherent reasoning trace. The file and the engine are fine; this
  implementation is what is wrong.
- **The embedding.** BF16, rms ~0.009 per row, loads correctly.
- **The final norm.** Mean +1.6279, and the unit offset applied gives 2.63 —
  exactly what the runtime's own history says is correct ("1.63 instead of the
  correct 2.63 = 1 + 1.63 that vLLM/llama.cpp produce").

### oq4++ is not the cause either

Quantised the 0.8B bf16 source to **oq4++** — the 35B's exact format, full
Hessian/LDLQ feedback and all — using `/srv/hipfire/calib/qwen3.5-0.8b.calib.hfq`
as the Hessian, and compared at seq 1:

| 0.8B format | seq 1 |
|---|---|
| oq8++ | 0.9996 |
| plain oq4 | (loss 3.796, healthy) |
| **oq4++** | **0.9998** |

So the oq4++ decode, its AWQ sidecars and the LDLQ feedback are all handled
correctly. Combined with the earlier results, every quantisation this
implementation reads is now verified on a real model.

### The routed experts are not the cause, and the residual stream is pathological

Ablating the routed experts entirely on the 35B (shared branch only) moves cos
from 0.0038 to -0.0025 — no change, both noise. The output is already corrupt
before the routed contribution, so the defect is upstream of them.

Per-layer residual rms at seq 1 tells a clearer story. Layer INPUTS, so index i
is the output of layer i-1:

```
35B   0.009 0.029 0.035 0.056 0.047 0.050 0.056 0.286 0.171 0.154 0.153 0.935
      0.941 0.947 0.947 0.952 0.954 0.953 0.953 0.955 0.954 0.953 0.952 0.952
      0.951 0.948 0.947 0.948 0.941 0.936 0.933 0.929 0.098 0.093 0.117 0.144 ...
0.8B  0.023 0.038 0.051 0.057 0.089 0.088 0.088 0.384 0.211 0.208 0.204 0.223
      0.105 0.086 0.087 0.551 0.108 0.111 0.117 0.140 0.136 0.140 0.154 0.193
```

The 0.8B spikes at layers 7 and 15 — both full-attention layers — and comes
back down, so spikes there are normal and not in themselves a defect.

The 35B pins at ~0.95 from layer 11 to 31 and then collapses — which I first
read as mechanical. **That reading was wrong.** Decomposing the energy:

| model | layer | rms | max abs | dim | share of energy in that dim |
|---|---|---|---|---|---|
| 35B | 11 | 0.935 | 34.2 | 1108 | **0.65** |
| 35B | 20 | 0.954 | 34.4 | 1108 | 0.63 |
| 35B | 31 | 0.929 | 33.3 | 1108 | 0.63 |
| 0.8B | 7 | 0.384 | 3.9 | 119 | 0.10 |

A single dimension carrying 65% of the residual energy is the well-known
massive-activation / attention-sink phenomenon, and it explains the flat rms
completely: once dim 1108 reaches 34, every later layer's contribution is small
against it, so the norm stops moving. The 0.8B shows a milder version of the
same thing.

So the per-layer rms profile is NOT evidence of where the bug is, and layer 11
is not implicated by it. What the profile does show is that the magnitude of
that one dimension dominates everything downstream of it — the final norm and
the tied lm_head see mostly dim 1108 — so if its value is wrong, the logits are
orthogonal no matter what else is right.

### Ablation table: no single layer type is responsible

All at seq 1 against `dump_logits_qwen35`, which the 0.8B matches at 0.9996:

| configuration | cos |
|---|---|
| base | 0.0038 |
| routed experts ablated (shared branch still on) | -0.0025 |
| linear_attn layers ablated (30 of 40) | -0.0406 |
| full-attention layers ablated (10 of 40) | -0.0953 |
| **all layers ablated (head only)** | **argmax = the input token, logit +43.09** |

The last row is a reference-free check and it passes: with tied embeddings, a
model that computes nothing should rank the current token first, and it does.
So embedding lookup, final norm and the tied head are all correct at this scale.
The embedding is exact too — the walk's layer-0 input is bit-identical to
`embed_tokens[0]`.

What the other rows say is that removing ANY single layer type still leaves the
output orthogonal. No one type is the culprit; either several are wrong, or
something common to every layer is.

Two further ablations completed the sweep, and neither isolates it:

| configuration | cos |
|---|---|
| whole MoE MLP ablated (routed AND shared) | 0.1077 |
| norm unit offset disabled | 0.1179 |

Nothing moves the 35B off roughly 0.1. Reading the table as a whole: with the
MLP gone, the linear_attn and attention halves alone — fed a verified-correct
embedding — still produce an orthogonal result over 40 layers. With the layers
gone, the head is provably right. So the wrongness is in the layer math itself
and is not confined to one component, one layer type, or the norm convention.

## A 1-layer 35B testbed, and what it shows

`export_truncated_model` writes the first N layers of an `.hfq` as bf16
safetensors — dequantised and AWQ-folded by this crate's loader, so both sides
of a comparison read IDENTICAL weights. Validated on the 0.8B cut to 4 layers:
cos 1.0001.

Applied to the 35B at ONE layer: **cos 0.0144**. Same weights, same token, one
linear_attn+MoE layer, and the outputs are orthogonal. Quantisation, the AWQ
fold and the decode path are all eliminated in a single measurement — the defect
is in the layer math at 35B dimensions. The testbed is 2.6 GB and runs in
seconds against ~20 minutes for the full model.

### RETRACTED: "v and z are 8-10x the 0.8B's"

That comparison was invalid. The 0.8B figures came from
`tools/qwen35_layer0_oracle.py`'s stage dump, which was recorded BEFORE the
GemmaRMSNorm fix and therefore used a plain `w` norm instead of `1 + w` — a
factor of 3.8 on `xn1` and everything downstream of it. Comparing a pre-fix
number against a post-fix one produced a difference that is an artifact of the
fix, not a property of the model.

Recomputed with the same convention on both:

| quantity | 35B | 0.8B |
|---|---|---|
| xn1 | 1.0478 | 1.2287 |
| z rms | 2.2882 | 0.8781 |
| z min / max | -14.60 / +4.86 | -5.21 / +3.11 |
| fraction of z negative | 0.75 | 0.60 |
| **silu(z) rms** | **0.8941** | 0.4848 |

`silu(z)` is 1.8x LARGER for the 35B, not collapsing. The z ratio is 2.6x
against a 2x wider hidden — unremarkable. So the 35B's linear_attn internals
look normal relative to the 0.8B's, and the "z carries large negative outliers
that kill the gate" lead is dead.

This is the fifth time this session a conclusion has come from a number whose
provenance was not checked (after the AWQ fold on synthetic tokens, the branch
magnitudes from the broken dumper, the flat-rms "pathology", and the random-init
fixture at seq 64). The rule that keeps proving itself: **before comparing two
measurements, confirm they were produced by the same code at the same time.**
Stale numbers in a doc are not evidence.

**Those seq-1 variant results were artifacts.** At sequence length 1 the
recurrence collapses to `out = (v * beta) * (k . q)`, and `k . q` is SYMMETRIC —
so seq 1 cannot distinguish q from k at all. Confirmed directly: swapping the Q
and K blocks in the channel split gives cos identical to base to four decimals.
Re-run at seq 4, where q and k are distinguishable, the "improvements" vanish:

| variant | seq 1 | seq 4 |
|---|---|---|
| base | 0.0144 | -0.0655 |
| GQA repeat as `hh % nk` | 0.2052 | +0.0216 |
| `[K\|Q\|V]` split | 0.0144 (blind) | -0.0748 |

Declining to adopt the seq-1 winners on the grounds that they contradicted
kernel-verified conventions turned out to be right for a reason I had not
identified at the time: the measurement itself could not see the thing it
appeared to be measuring.

**Every linear_attn stage is now verified at the REAL 35B geometry**, not the
toy 2-head one every earlier run of these files used. `verify_la_core_vs_kernels`
and `verify_deltanet_vs_kernel` both take the geometry from argv now:

| stage | 2 heads | 32v/16k heads, h=2048 | 32v/32k |
|---|---|---|---|
| conv1d + SiLU + split | 1.6e-7 | 2.0e-7 | 2.0e-7 |
| l2norm(q)/sqrt(hd), l2norm(k) | 1.7e-7 | 2.5e-7 | 3.1e-7 |
| alpha / beta | 1.1e-7 | 1.8e-7 | 2.0e-7 |
| gated norm | 1.3e-7 | 1.9e-7 | 2.5e-7 |

and the recurrence, at the real head count and out to seq 64:

| seq x heads | 6x2 | 6x32 | 16x32 | 64x32 |
|---|---|---|---|---|
| rel vs `gated_delta_net_f32` | 5.0e-7 | 6.5e-7 | 9.9e-7 | 2.0e-6 |

The GQA repeat convention is confirmed by reading
`repeat_interleave_qk_batched.hip` directly: `dst[b, kh*ratio+r, d] =
src[b, kh, d]`, which is what the host core does. (Note the l2norm check *uses*
that convention to gather post-repeat values back, so it corroborates rather
than independently proves it — the kernel source is the proof.)

Also verified at 35B dimensions, against a host matmul inside the walk: every
projection is exact — `out_dim` 8192 worst 3.8e-6 against magnitude 26.8, 4096
worst 2.9e-6, the two 32-wide ones ~1e-6. So the GEMM path is not
dimension-sensitive and is eliminated.

## SOLVED: the harness scored the 35B through the wrong head

The 35B ships an **untied `lm_head.weight`**. `gamma_hybrid` hardcoded
`None, // tie_word_embeddings: the head IS the embedding` and scored every
model through its embedding matrix.

The 35B's `lm_head` correlates with its own embedding at **0.025** — they are
unrelated matrices. That number is the answer: every "orthogonal at one token"
measurement in this document, cos 0.0144 / -0.0382 / -0.0655, was the walk's
layer output projected through the wrong matrix. The layer math was never
wrong.

With `load_lm_head_f32` loading the untied head when the artifact has one:

| testbed | before | after |
|---|---|---|
| Qwen3.5-0.8B, 1 layer (tied) | +1.0000 | +0.999993 |
| qwen3_5_moe-tiny, 1 layer (tied) | +1.0000 | +0.999978 |
| GQA fixture n_k=1<n_v=2 (tied) | +1.0000 | +0.999982 |
| **35B, 1 layer (UNTIED)** | **-0.0382** | **+0.999990** |

‖mine‖ 883.60 against ‖ref‖ 884.18.

Why every control passed and only the 35B failed: the 0.8B and both tiny
fixtures tie their heads, so for them "the head IS the embedding" was true and
the hardcoded `None` was correct. The 35B is the only artifact in reach with an
untied head, and the assumption was written as a comment asserting a fact
rather than as a lookup.

Note also that the name needs trying with AND without the model prefix — the
35B stores `model.language_model.embed_tokens.weight` but a bare
`lm_head.weight`.

## RETRACTED: GQA is not the defect

The section below concluded, by elimination, that `n_k < n_v` was the defect,
because it was the only structural feature neither passing control had. Tested
directly, it is not.

A GQA fixture built by slicing `qwen3_5_moe-tiny`'s layer-0 QKV projection down
to a single key head — Q head 0, K head 0, both V heads, so `n_k=1 < n_v=2`,
with `conv1d.weight` sliced to the same rows and `linear_num_key_heads` set to
1 — gives **cos +1.0000**, ‖mine‖ 59.01 against ‖ref‖ 59.01. GQA linear
attention reproduces exactly.

That was an argument from elimination, and it was wrong. The list of
"structural features the failing case has and the passing ones don't" was only
as good as my inventory of features, and scale is not on it. Note also that
both the GQA fixture and the 35B run repeat ratio 2, so even the ratio is
shared.

With structure now excluded — the 35B's layer-0 tensor names and shapes are
identical to the fixture's, modulo size — what is left between a passing
testbed and the failing one is **pure scale**: h 256 -> 2048, value heads
2 -> 32, qkv 512 -> 8192, experts 8 -> 256, top-k 2 -> 8, intermediate
128 -> 512, vocab 1024 -> 248320.

The next step is a bisect over those, one dimension at a time, using a
synthetic 1-layer model generated at chosen dims. Both sides read identical
weights, so random init is fine here — it is what the two cos-1.0000 controls
already are.

## Superseded: the elimination argument for GQA

Two controls settle it. The 1-layer testbed methodology had only ever been
validated at FOUR layers on the 0.8B, so the harness itself was unproven at the
size everything was being measured at. Running it at one layer:

| testbed | layer 0 | n_k / n_v | cos | ‖mine‖ | ‖ref‖ |
|---|---|---|---|---|---|
| Qwen3.5-0.8B, 1 layer | linear_attn + dense | 16 / 16 | **+1.0000** | 1296.98 | 1297.17 |
| Qwen3.5-0.8B, 4 layers | linear_attn + dense | 16 / 16 | +1.0000 | 1159.65 | 1159.64 |
| qwen3_5_moe-tiny, 1 layer | linear_attn + **MoE** | 2 / 2 | **+1.0000** | 58.67 | 58.68 |
| qwen3_5_moe-tiny, 4 layers | linear_attn + MoE | 2 / 2 | +0.9990 | 58.54 | 58.52 |
| **35B, 1 layer** | linear_attn + MoE | **16 / 32** | **-0.0382** | 689.11 | 884.18 |

The first row proves the harness is sound at one layer, so the 35B failure is
real. The third proves a full MoE layer — router, 8 experts, shared branch,
top-2 — reproduces exactly through the same harness, so the MoE is not the
defect.

What is left is the one structural feature neither passing testbed has:
**n_k < n_v**. Both controls run n_k == n_v, the 35B is the only GQA linear-attn
model in reach, and it is the only failure. Every earlier component check ran
GQA geometry through MY OWN call, with my own k_dim/v_dim and my own repeat —
which is why they all passed. The runtime's own k_dim agrees
(`linear_num_key_heads * linear_key_head_dim`, prefill_batch.rs:2058), so the
split offsets are not it; what remains untested is the PLACEMENT of the repeat
within the runtime's pipeline relative to mine.

(Superseded in scope by the section above: linear_attn's COMPONENTS are each
the inference path's at 35B dimensions, but the GQA repeat's placement in the
pipeline is an integration question no component check reaches.) Projections,
conv, the norms, the activations, and the recurrence are all the inference
path's, at the inference path's dimensions. Since the embedding and the head
were already exact at 35B scale, what remains inside the layer is the MoE
branch and the residual/norm plumbing around it — and the MoE has only ever
been checked for LOADING (`verify_moe_per_expert`, bit-exact) and for FORWARD
math on a tiny fixture at seq 1. The 35B routes top-k over its full expert
count at h=2048; that forward has never been cross-checked against the
kernels at that scale, which is now the same gap this tick just closed for
linear_attn.

The original three variants, for the record (all at seq 1, so all suspect):

| variant | cos |
|---|---|
| base | 0.0144 |
| MoE MLP ablated | 0.0226 |
| GQA repeat as `hh % nk` instead of `hh / rep` | 0.2052 |
| gated norm using `sigmoid(z)` instead of `silu(z)` | 0.2306 |
| norm unit offset disabled (full model) | 0.1179 |

None is adopted. `silu` is confirmed against `gated_norm.hip` at 1e-7 and
`hh / rep` against `repeat_interleave_qk.hip`'s documented
`dst[kh*ratio + r] = src[kh]`; both are also exact on the 0.8B and the tiny
fixtures. Several unrelated changes each buying 10x, none reaching 1, is the
signature of a systematic error upstream of all of them that these variants
partially mask — most likely whatever makes z carry those outliers. Adopting any
one would be the AWQ mistake again, on better-looking numbers.

## Stopping the bisect

This is the stated stopping point from the previous round: if ablating the MLP
did not isolate the defect, further 20-minute-per-hypothesis guessing is poor
value. Ten such runs have now eliminated every cheap hypothesis without
localising anything.

What would actually settle it is a per-layer reference, and there are exactly
two ways to get one:

1. **Settle `dump_qwen35_hidden_states`' capture semantics.** Its forward is now
   correct (it captures during prefill) but its states still disagree with a
   verified forward for reasons documented in
   `dump-qwen35-hidden-states-status.md`. Whoever knows what `pbs.x_batch` holds
   at that point can resolve it in minutes.
2. **Build a truncated-model oracle.** Cut a model to N layers, dump its logits
   through the trusted prefill path, and compare against this walk truncated the
   same way. Every piece is verified; it is bounded work, but it means rewriting
   a 20 GB artifact per depth for the 35B, or doing it on the 0.8B first to
   validate the technique.

Meanwhile the dense hybrid path is complete and verified — Qwen3.5-0.8B at
cos 0.9999 against the runtime and mean loss 3.52 on real text — so gamma
capture on dense hybrids is available today.

### What is left

With oq4++ cleared, the untested combination narrows to **MoE together with
4-bit quantisation**, and to raw scale — 40 layers at h=2048, 256 experts at
top-8. Every witness so far is either dense-and-quantised (the 0.8B) or
MoE-and-bf16 (the tiny fixtures); nothing on disk is both MoE and 4-bit except
the 35B itself.

An attempt to synthesise one — padding `qwen3_5_moe-tiny`'s experts from
`moe_intermediate` 128 to 256 so they clear the G256 group boundary — hit a
size assertion inside the quantizer (expected 65536 per expert, got 262144).
Worth another try with the padding done to match what the quantizer computes,
since it would give a seconds-per-iteration witness for the last untested
combination.

The cheaper alternative, at ~20 minutes a run: ablate the routed experts in the
35B walk (shared branch only) and see whether cos moves off zero. That splits
"the routed-expert path is wrong at 4-bit" from "something else at this scale"
in one measurement.

## The dense hybrid is correct

Exactly one structural feature is exercised by the 35B and by nothing that
works: **fewer key heads than value heads** in linear attention. The 0.8B is
16/16 and both tiny fixtures are 2/2. The gradcheck covers n_k < n_v
synthetically, which proves internal consistency, not correctness — the same
gap that hid GemmaRMSNorm for a dozen commits.

The fix is to stop relying on the 35B as the only witness: synthesize a tiny
fixture with n_k < n_v (halve `linear_num_key_heads` and narrow `in_proj_qkv`
and `conv1d` to match), and compare it against the runtime at seq 1. Seconds
per iteration instead of twenty minutes, and a definitive answer either way.

## The dense hybrid is correct — 0.9999 against the runtime

Sweeping sequence length against `dump_logits_qwen35` settles the last doubts
about the shared path:

| model | seq 1 | seq 4 | seq 16 | seq 64 |
|---|---|---|---|---|
| **Qwen3.5-0.8B (trained)** | 0.9996 | 0.9998 | 0.9998 | **0.9999** |
| qwen3_5-tiny (random init) | 1.0000 | 0.9995 | 0.9347 | 0.9441 |
| qwen3_5_moe-tiny (random init) | 1.0000 | 0.9990 | 0.8143 | — |

Two conclusions, and the second corrects the previous section:

1. **The implementation is essentially exact** on a real model, at every length
   tested. Per-token math and sequence handling are both right.
2. **The random-init fixtures' decay with length is chaos amplification, not a
   bug.** A randomly initialised network has no trained attractor, so f32-level
   differences between two implementations diverge as they propagate. The
   non-monotonicity (0.9347 at 16, 0.9441 at 64) is the tell — a systematic
   error accumulates, it does not wander.

**The MoE math is correct too.** Routing, expert SwiGLU and the shared branch
are entirely per-token — they depend only on the current hidden state — so
`cos 1.0000` at seq 1 on the MoE fixture is a complete test of them. The
earlier reading of 0.6470 at seq 64 as "the MoE gap" was measuring chaos in a
random model, not a defect.

That leaves the 35B's near-uniform loss unexplained by anything found so far,
and the useful lesson is about the oracle rather than the code: **a random-init
fixture is a valid oracle only at short sequence length.** Comparisons on one
at seq 64 measure divergence rate, not correctness.

## A fast MoE oracle, and the MoE gap confirmed

Every 35B check costs ~20 minutes and there is no small real MoE on disk — but
the tiny fixture IS a real MoE (8 experts, top-2, shared branch), and random
init does not matter for a comparison: both sides compute the same function.
`gamma_hybrid` now exports last-position logits, and takes `arange` as its
corpus argument to force the deterministic `0,1,2,...` prompt that
`dump_logits_qwen35` uses, so the two are directly comparable on any artifact.
Runs take seconds.

| fixture | cos(mine, runtime) |
|---|---|
| qwen3_5-tiny (DENSE) | 0.9441 |
| qwen3_5_moe-tiny (MoE) | 0.6470 |

That is the MoE gap isolated on a model that iterates fast. Two routing
hypotheses are already dead — both score WORSE than base, so both are right as
implemented:

| variant | cos |
|---|---|
| base | 0.6470 |
| top-k gates not renormalised | 0.4918 |
| shared-expert scalar gate not applied | 0.4523 |

The dense fixture at 0.944 is worth its own look: the 0.8B scores 3.52 on real
text and matched the runtime at cos 0.9994 for one token, so 0.944 on a
64-token random-init model suggests a smaller second effect rather than a clean
bill of health for the dense path.

## Why it took so long

The failure was silent in every way a failure can be. Shapes match. Everything
stays differentiable, so all five gradchecks passed throughout. Every individual
stage matched its kernel to ~1e-7, because each stage WAS right — the norm
weights feeding them were wrong. Three independent implementations agreed with
each other, because all three were written from the same misreading.

Two earlier detours are worth keeping as cautions, both recorded in the git
history:

- **The AWQ fold.** Tuned by loss on synthetic tokens, where every variant sits
  near uniform and the spread is noise. Retracted.
- **"The branches are four times too weak."** Derived from the hidden-state
  dumper's per-layer rms one commit after establishing that the dumper is
  broken. Retracted.

Both share a shape: a number was trusted without checking what produced it. The
thing that finally worked was insisting on an oracle whose correctness was
independently demonstrated — `hipfire chat` generating coherent text from the
same artifact — and then reducing to the smallest case that still failed (one
token, where rope and QK-norm are provably irrelevant).

The `1 + w` variant had in fact been measured 6 commits earlier and set aside,
because no unit offset appears in `rmsnorm.hip` and the runtime ships working
inference. The reasoning was sound; the conclusion was wrong. hipfire bakes the
offset at LOAD time (`qwen35/loading.rs:717`), not in the kernel — so searching
the kernels for `1.0 +` was searching the wrong layer of the stack. Searching
for the CONCEPT instead ("GemmaRMSNorm", named in `dump_norms.rs`'s own doc
comment) would have found it immediately.

## What is verified now

- One token vs runtime prefill logits: cos 0.9994.
- Real text through the streamed walk: mean loss 4.03 on the bf16 source and
  4.03 on the oq8 artifact.
- The random-init fixture still lands at ln(vocab), as it must.
- All five gradchecks and the stacked-MoE byte check still pass.

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
