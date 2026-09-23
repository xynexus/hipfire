# qwen4_exp calibrates WITHOUT a streamed adapter — 543 Hessians in 53 s

**Status:** working, 2026-09-24. This removes the blocker on `oq4.25++` for
Qwen3.8-Flash-Next, and for any future family in the same position.

## The blocker, and why it was the wrong one to accept

`oq4.25++` requires `--hessian`:

    error: --format oq4++/oq8++ or --ldlq requires --hessian <HFHS .hessian.bin>

and the streamed calibrator refuses arch 26 outright:

    InvalidSourcePlan("no native calibration adapter is registered for
                       architecture 26")

Only qwen35, gemma3, gemma4, zaya and cohere2 call
`register_calibration_adapter!`, and those adapters run 1057-2410 lines because
each reimplements its family's forward for layer-by-layer streaming. Reading
that list, the obvious conclusion is "writing one is multi-session work" — which
is what I concluded, twice, without testing it.

**The streaming machinery exists for models that cannot be held resident.**
Qwen3.8-Flash-Next is not one: it loads PAGED in ~6 s and decodes at 3.25 tok/s.
The forward already works, so the question was never "how do I stream this
model" but "how do I tap the forward I already have".

## What it actually took

`weight_gemv` already carries the capture tap, and its own comment advertises the
generality:

> the single chokepoint that makes activation capture work for **every arch that
> routes its linears through `weight_gemv`**

qwen4_exp routes all of them through it. The tap keys on the weight's DEVICE
BUFFER POINTER (`gpu.capture_names: HashMap<usize, String>`), so the one missing
piece was a pointer -> canonical-name map. That is `TrunkWeights::capture_map`,
about 50 lines, plus `examples/calibrate_qwen4exp` to arm it and drive a forward.

    loaded in 6.4s
    capture targets: 543 dense linears
    captured over 24 tokens in 18.3s
    accumulators: 543
    wrote flashnext.calib.hfq (21.99 GiB) in 28.7s

    HessianBf16TrilDiagF32   543 tensors   21.99 GB
    F32 (imatrix/actstats)   543 tensors   10.42 MB

53 seconds, versus a four-figure-line adapter.

## Two things that must stay true

* **The names must match the ARTIFACT's tensor names.** `--hessian` looks them
  up by name at quantize time; a mismatch is silent — the quantizer prints
  `ldlq: skip <tensor> (no Hessian entry for [...])` and produces an
  UNCALIBRATED artifact that still quantizes and still serves.
* **Routed experts take the imatrix path** (`CalibCollector::with_imatrix_only`).
  512 experts per layer, each with a full [K,K] Hessian, does not fit — the same
  split the other families use. Their `ldlq: skip` lines are expected; the dense
  linears' `tiered OBS int4/int8` lines are the ones that must appear.

## Applicability

Any arch whose linears route through `weight_gemv` can calibrate this way, with
only a `capture_map` to write — no adapter. The adapter is only required when
the model genuinely cannot be held resident.

## ⚠️ A process-name trap, since it cost a false alarm

A monitor watching this build reported it DEAD at 0.6% when it was healthy:
`pgrep -x hipfire-quantize` can never match, because the name is 16 characters
and Linux truncates `comm` to 15. `pgrep` warns about it. Use `pgrep -f` against
the command line for anything with a long binary name.

## The first capture was degenerate (2026-09-24)

The first `oq4.25++` build off this path was killed at 6.6%. Its log said why,
per tensor:

    ldlq: DAMPED model.language_model.layers.0.mlp.shared_expert.gate_proj.weight
      [k=2560] factorized only at 100x the requested lambda
      -- the OBS solution is that much closer to RTN

Every captured tensor escalated to the top of the damping ladder. `cli.rs`
already spells out the consequence: `H + lambda*I` is effectively `lambda*I`, so
LDLQ collapses to RTN *while the AWQ scales are rebased against the degenerate
Hessian*. That artifact would have been worse than a plain `oq4.25+`, and its
KLD would have looked like a quantizer result rather than a calibration bug.

Three compounding causes, all in the example, none in the quantizer:

1. **Self-generated tokens.** The capture greedy-argmax'd its own continuation
   from one seed token. Greedy decode falls into a repeating cycle, so `XtX` has
   an effective rank of a handful *no matter how many tokens run*. The comment
   defending this cited the MTP probe — which is the very experiment where a
   synthetic prompt drove the trunk into degenerate repetition and inverted a
   result. Now: tokenize a real corpus with the artifact's own tokenizer and
   teacher-force it.
2. **32 tokens, structurally capped.** `Qwen4ExpBackend::load(.., 256)` capped
   context at 256, so no token count could reach the K a full-rank Hessian needs.
   `max_seq` is now sized from the requested token count.
3. **Prefix sampling.** `calib-multi-8m.txt` is concatenated by language, so the
   first N tokens are one language. Now strides contiguous 512-token chunks
   across the whole file — contiguous so activations stay coherent within a
   chunk, spread so the mix is real.

Captured K distribution (`max_k=8192`): 320 x97, 640 x48, 2560 x195, 6144 x12.
16384 tokens clears the worst case by 2.7x. Cost is ~0.16 s/token, so ~45 min.

### PinAll is what OOMs the capture, not the accumulators

With real varied text the capture died at `l9 e333`, `free=6.6 GiB`. The 352
accumulators are only ~1.8 GB — the earlier "accumulators are too big" read was
wrong. `build_expert_pager` defaults to `ResidencyPolicy::PinAll`, so every
routed expert a diverse token stream touches pins and never evicts. 24576
modules x 4.05 MiB is ~100 GB.

Set a budget for calibration runs:

    HIPFIRE_CALIB_MAX_K=8192 \
    HIPFIRE_QWEN4EXP_EXPERT_CACHE_BYTES=21474836480 \
      calibrate_qwen4exp <model.hfq> benchmarks/calib/calib-multi-8m.txt <out> 16384

`PinAll` is right for serving, where a session's routing is narrow and stable.
It is wrong for calibration, whose whole point is to touch the distribution
widely.

### Routed experts stay imatrix-only

The build logs ~1024 `mlp.experts.N.gate_up_proj` LDLQ skips per two layers, and
that is by design, not a coverage hole: a full [K,K] Hessian per expert across
24576 modules does not fit, which is why `CalibCollector::with_imatrix_only`
exists and why every other family splits the same way.
