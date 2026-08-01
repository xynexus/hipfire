# Next phase: from reference kernel to production NPU decode

Written 2026-08-02, closing the FLM reverse-engineering phase.

The reference kernel is done: it reproduces FLM's tokens exactly and beats its
throughput. `docs/npu/flm-refe-log.md` is the full record. This document is what
carries forward and what to do next.

## The one number that decides the next phase

**Decode on this NPU is DMA-bound. Bits per weight IS speed.**

With every one of the 32 cores' compute stubbed out, the fused design still runs
731.5 of its 791.5 us/layer — 92.4% — at 52.29 GB/s against lm_head's measured
54.7 GB/s ceiling. All the arithmetic in the model is worth 60.0 +- 3.5 us/layer.

The two-pass lm_head measured the mechanism directly, which is why this is not an
inference from a roofline sketch:

    coarse tier  4.02 bits/weight   131.8 MB   54.7 GB/s
    exact tier   5.00 bits/weight   164.7 MB   55.2 GB/s
    bytes ratio 0.8006 against time ratio 0.8033 — matching to 0.3%

Same GB/s both ways. The kernel is not the lever and never was.

### What that implies, in tok/s

Llama-3.2-1B decode moves 60.8M params/layer x 16 layers plus a 262M-param
lm_head every token. At the measured 54.7 GB/s ceiling:

    format        bits   bytes/token   projected   note
    q4nx (FLM)    5.00     775.7 MB     63.4 tok/s  MEASURED, our fused design
    oq4++        ~4.25     659   MB    ~72   tok/s  projected, +18%
    oq4          ~4.50     698   MB    ~68   tok/s  projected, +11%
    oq8++         8.00    1235   MB    ~44   tok/s  projected, -28% vs FLM

**`oq8++`/W8A8 cannot beat FLM on NPU decode.** It moves 59% more bytes than the
format FLM already ships. W8A8's activation half is free here — activations are
negligible against weights — but the weight half is the entire cost. If that goal
came from GPU work, note that the intuition inverts: on a compute-rich,
bandwidth-poor NPU, the quant format is the performance story.

**So the next phase's headline lever is a TIGHTER format, not a faster kernel.**
`oq4++` (symmetric Opus Quant, clip-search/AWQ, Hessian error feedback) is both
faster AND more accurate than the q4_1-shaped container FLM uses. That is the
rare case where the two goals do not trade against each other.

Sanity-check the projections before building on them: they assume the same 54.7
GB/s and the same dispatch structure, and a format with a different tile shape may
not hit the same ceiling.

## What is established and should not be re-derived

Hard limits, all measured on this hardware:

    dispatch floor        92.9 us       7-point fit, R^2 0.99997
    in-dispatch barrier    6.00 us      fit R^2 0.99904; 80/token = 3.8%
    shim budget           16 in / 16 out    8 shim tiles x 2 each way, HARD
    memtile budget        48 in / 48 out    8 memtiles, ~6 each way
    program memory        16 KB/core
    data memory           64 KB/core
    effective bandwidth   ~54.7 GB/s
    same-build spread     2.6% run to run

Structural findings:

  - FLM issues ~2.5 dispatches per TOKEN. A design issuing one per layer pair
    loses to the dispatch floor alone. Fusion into ONE dispatch is what made the
    reference kernel competitive; interleaved as separate dispatches the same
    kernels run 55.0 tok/s, a 10% loss.
  - Latency-bound slopes DO NOT transfer. `groups_ab` standing alone measured
    123.6 us/layer at 31.9 GB/s; fused, its bytes queue at the DDR ceiling and
    cost 75.2. Bytes add. Slopes measured on a latency-bound design do not.
  - Two-pass output projection works and is worth ~20% of the head: a coarse
    low-bit shortlist on the NPU, an exact rescore of K=32 rows on the host.
    Device recall verified 48/48, worst rank 3.

## Traps that cost real time here

Each of these produced a confident wrong result that survived at least one check.

  - **Position 0 is structurally blind to attention.** Its softmax is over a
    single entry, so no score reaches the output. A misaligned `aie::load_v` left
    pos 0 BIT-EXACT while every later position collapsed to cosine 0.05-0.28.
    The attention scale error hid the same way. Nothing about attention is
    verified until it is verified at pos > 0.
  - **`aie::load_v` on a misaligned pointer does not fault** — it reads the wrong
    bytes, silently.
  - **`iron.jit` caches on the AST.** Comments and compile FLAGS do not bust it.
    Stamp cache-busting values into fifo NAMES. A design whose generator cannot
    even run will be served from cache indefinitely (this happened: `groups_ab`
    at 16 cores has never been able to generate its own design).
  - **A held dispatch that rebinds nothing silently ignores new inputs.** Binding
    once and dispatching many times is only correct if each dispatch reads the
    buffer's CURRENT contents — perturb, dispatch, restore, dispatch, and demand
    the output moved and came back bit for bit. This shipped broken and read as
    "the coarse tier has no recall".
  - **Benchmark by INTERLEAVING, never sequentially.** A sequential A/B called
    chunk=16 a 40% win twice; interleaved, it was a loss. Whichever ran first
    kept the cache.
  - **Verify against something that shares no code.** Five correctness faults all
    passed every check because each check compared a stage against a reference
    computed from the same wrong input. The fp32 forward from
    `consolidated.00.pth` is what eventually caught them.
  - **Stale constants in scripts that re-measure everything else.** A fresh
    lm_head number composed with a layers constant from a different build config
    overstated throughput by 1.1 tok/s. If a constant tracks a build knob, say so
    where it is defined.

## Goals for the next phase

1. **Port the decode path to `oq4++`.** The headline. Target ~72 tok/s on
   Llama-3.2-1B, +18% over the current reference kernel and +17% over FLM.
   Success is measured on the WALL CLOCK against FLM's 61.18, not device time.
2. **Keep the correctness apparatus.** Token-for-token agreement with FLM on its
   own `context` array is the acceptance test; the fp32 oracle is the reference.
   Both exist and work — `decode.py`, `decode_oracle.py`, `sweep.sh`.
3. **Then MoE: `qwen3.6-moe:35b-a3b`.** This is a genuinely different problem,
   not a port. Expert routing, per-expert scales, and a working set that does not
   fit the assumptions above. Per project memory, no MoE model has taken the
   batched-prefill path and it needs per-expert AWQ scales.

## Open questions

  - Does an `oq4++` tile shape hit the same 54.7 GB/s? The projections assume it
    does. Measure a single GEMV at the new format before porting 16 layers.
  - Prefill is UNEXAMINED. Everything here is decode. Prefill is compute-bound
    and batched — a different regime where the format argument above does not
    apply, and where FLM may still have something to teach.
  - The 16.2 us/layer surcharge over projection is attributed to interaction
    (12.2 +- 4.3 us/layer at 2.9 sigma) but not eliminated.
  - Whether the two-pass head's coarse tier should be `oq4++` too, or stay a
    purpose-built 4-bit direction encoding.
