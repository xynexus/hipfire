# W4A4 through the daemon: coherent, +13% serving prefill, still opt-in

2026-08-23, halo/gfx1151, Qwen3.8-27B / Qwen3.5-27B / Qwen3.6-27B oq4.25++,
kvarn KV. First validation that could run at all, because it needed the batched
prefill default flip — before that the daemon never reached the compact GEMM.

## What W4A4 is here

Weights stay the SAME compact 4.25-bit blocks; only the ACTIVATION narrows to
int4, collapsing the exact radix-16 pair (`x = 16*x_hi + x_lo`, two iu4 WMMA
passes) into one. **The bits/weight floor is untouched** — no requantization was
needed or done.

## Speed, through the daemon

| model | A4=0 | A4=1 |
|---|---|---|
| Qwen3.8-27B--oq4.25++ | 301.1 | **341.5** tok/s |
| Qwen3.5-27B--oq4.25++ | 300.9 | **341.2** |
| Qwen3.6-27B--oq4.25++ | 301.4 | **340.2** |

Consistently **+13%** serving prefill. (In the isolated bench it was +14.9%;
through the daemon other work dilutes it.)

## Quality

Four checkable questions appended to a ~700-token context, greedy, scored on
answer correctness rather than token equality — A4 is lossy by construction, so
identical text is NOT the right bar:

| model | A4=0 | A4=1 |
|---|---|---|
| Qwen3.8-27B | 4/4 | **4/4** |
| Qwen3.6-27B | 4/4 | **4/4** |
| Qwen3.5-27B | still inside `<think>` at 500 tok | same, both arms |

All nine `coherence_probe` detectors OK in both arms on every model
(attractor, long_state_collapse, ngram_density, loop_guard, special_leak, ...).

Sample, Qwen3.8-27B, A4=1: *"17 × 23 = 391 / Canberra / 2, 3, 5, 7, 11 / A cache
line is the minimum unit of data transferred between memory levels, so writing a
single byte still requires reading and writing the entire cache line."* All
correct. Wording differs from A4=0; content does not.

The Qwen3.5-27B row is a verbose-reasoning artifact, not degradation: it never
closes `<think>` within 500 tokens in EITHER arm, identically.

## Why it stays OPT-IN anyway

Coherent is not the same as lossless, and unlike the kvarn flip (byte-identical
output) **W4A4 genuinely changes the numerics**. What is missing:

1. **No KLD.** `hipfire eval --battery quality` needs a `--reference`, and
   Qwen3.8-27B has no bf16 twin. Qwen3.5/3.6-27B do — that comparison is the
   real gate and has not been run.
2. **Perplexity cannot serve as the metric.** Both the `--battery perplexity`
   binary AND the daemon-independent path leave ppl bit-identical between arms
   (16.737 either way, elapsed_ms 92740 vs 92592) because that binary never
   enters the compact batched GEMM. Same trap as the kvarn battery: check that
   the knob reaches the code under test before believing a null.
3. **Four questions is a smoke test, not an eval.** No GPQA/RULER/long-context.
4. **Greedy only.** Sampling would expose logit drift that argmax hides.

Enable with `HIPFIRE_OQ_COMPACT_A4=1`.

## Next

Run `hipfire eval <oq4.25++> --compare <bf16 twin> --battery quality` with A4 on
and off on Qwen3.5-27B or Qwen3.6-27B. That is the number that decides whether
this becomes default.
