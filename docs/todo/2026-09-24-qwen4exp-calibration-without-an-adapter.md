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

## CORRECTION: the experts get no imatrix either, and the ++ is net-harmful

The section above says the routed-expert LDLQ skips are "by design, not a
coverage hole ... which is why `CalibCollector::with_imatrix_only` exists". The
first half is right and the second is WRONG, so correcting it here rather than
editing it away.

The experts were supposed to take the imatrix path. They did not take any path.
The build log counts it:

    imatrix derived from HFQM calibration: 704 keys
      (352 Hessian diagonals, 352 imatrix vectors)

352 against 24576 expert modules. `TrunkWeights::capture_map` names only
resident trunk tensors, and routed experts are PAGED — the `weight_gemv` tap is
keyed on buffer POINTER, and a paged expert's pointer is a rotating pager
buffer, so `with_imatrix_only(["experts."])` collected nothing. 97% of the
weights are uncalibrated.

### What that costs, measured

Four artifacts, same corpus (held-out wikitext2), 4095 scored tokens:

    oq4       canonical Oq4G256, no calib   PPL 4.0400
    oq4.25    tiered OqPlusCompact, no calib PPL 4.0418
    oq4.25++  alpha=0.1                      PPL 4.1090
    oq4.25++  alpha=0.55 (default)           PPL 5.0611

`oq4.25` and `oq4.25++` are FORMAT-IDENTICAL (same histogram, 172.37 GB); the
`++` differs by 96 AWQ sidecars out of 50347. So the tiered 4.25-bit format
costs nothing, and the whole 25% regression is the calibration.

96 tensors do that much because they are `shared_expert.{gate,up}_proj` on all
48 layers, and the shared expert fires on EVERY token. They are also exactly the
tensors logged as `DAMPED ... only at 10x the requested lambda`: LDLQ collapsed
toward RTN while AWQ pre-scaling was still applied against that same
ill-conditioned Hessian, and AWQ weights ship PRE-SCALED, so the bad scale is
permanent.

The alpha sweep is monotonic and alpha=0.1 is still worse than no AWQ, so this
is not a mistuned alpha — the optimum is zero. **Ship `oq4` or `oq4.25`; do not
ship `oq4.25++` for this model.**

The `++` has no upside here until calibration reaches the routed experts, which
needs the pager to name them. Until then it can only add the downside.

## CORRECTION TO THE CORRECTION: experts DO take the imatrix path

The section above says the routed experts "do not take the imatrix path ... they
take no path". That is wrong about the SYSTEM and right only about this run.

Experts do take the imatrix path, by design and in code:

- `CalibCollector::with_imatrix_only(["experts."])` exists to route them there.
- `imatrix_col_weights_for_parent` consumes a per-expert `[K, n_experts]`
  imatrix, one importance column per expert.
- a dedicated expert-coverage admission runs over the package and preserves
  undercovered routed experts at source precision, which only makes sense
  because expert coverage is expected.

What actually happened is narrower and is a defect in
`examples/calibrate_qwen4exp.rs`, not in the design: THIS package contained no
expert entries at all.

    shared_expert 288 | hyper_connection 194 | self_attn 120 | ple 4 | experts. 0

`TrunkWeights::capture_map` names only resident trunk tensors, and the
`weight_gemv` tap is keyed on buffer POINTER. A paged expert's pointer is a
rotating pager slot, so no expert was ever named and the imatrix path they were
routed to received nothing. Fix the capture, not the split.

### The damping is partly self-inflicted by Hessian STORAGE

The same build says so directly, and it was skimmed past the first time:

    LDLQ damping: 4 tensor(s) needed escalated damping ... Conditioning, not
    coverage: bf16 Hessian storage forces ~13% of the mean diagonal on its own,
    and rank deficiency sets the floor (calibrate more sequences, or store the
    Hessian at f16 -- same bytes).

So the 10x damping on `shared_expert` — the thing that made AWQ destructive — is
not purely a token-count problem. Roughly 13% of the mean diagonal is quantisation
noise from storing the off-diagonal triangle as BF16.

`hessian_io` has exactly ONE compact storage dtype today,
`Bf16TrilDiagF32` (qt 130), so the f16 option the message recommends is NOT
implemented. F16 has 10 mantissa bits against BF16's 7 at identical width, which
is the right trade for Hessian entries (narrow dynamic range, precision-critical)
as opposed to weights.

That makes "do not ship oq4.25++" a statement about the CURRENT pipeline, not a
verdict on `++` for this model. Ranked next steps:

1. Name paged experts in the capture so 97% of the weights get their imatrix.
2. Add an F16 tril storage dtype and re-measure the damping.
3. Only then re-judge AWQ alpha.
