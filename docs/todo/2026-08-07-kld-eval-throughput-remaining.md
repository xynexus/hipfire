# KLD/eval throughput — the two kernel items left

Status 2026-08-07. The structural work is done and committed; what remains is
kernel efficiency only, no plumbing. Stopped here deliberately — picking it back
up needs no rediscovery, just the two items in "Remaining".

## Where it landed

One 2048-token chunk, qwen3.5-0.8b bf16, gfx1151 (`HIPFIRE_KLD_PHASE_TIMING=1`):

| phase | session start | now | floor |
|---|---|---|---|
| body | ~60 s (per-token) | 1.247 s | ~0.23 s |
| head | ~4 s | 0.770 s | ~0.04 s |
| download | — | 0.117 s | ~0 |
| scoring (CPU) | 7.30 s | 0.103 s | ~0.01 s |
| **chunk** | ~70 s | **2.24 s** | **~0.28 s** |

1-chunk `hipfire eval` end-to-end: 138.1 s → 19 s (7.3×). Roughly 12.5% of
roofline. Landing both items below puts a chunk near 0.45 s (~85% of what is
realistically achievable) and a 1175-chunk reference at ~10 minutes rather than
the ~40 h it started at.

Commits: `0bc5d4d5b`, `53af2c0c3`, `16cb54d56`, `22d0d2825`, `c6c214a84`,
`e52c10cbc`, `497ec2424`.

## Remaining

### 1. Body GEMM — 1.247 s, 56% of the chunk

3.28 TFLOP in 1.247 s = **2.6 TFLOPS**, roughly 10% of peak.
`gemm_bf16_x_bf16_wmma_gfx1151_m128`, 1200 launches at 834 µs.

Not a new problem: the tuning recipe that took a W4A4 prefill GEMM to ~50% of
peak on this same chip is wave64 + double-buffered LDS + N-heavy 2×8 + BK64
(see the gfx1151 iu4 GEMM tuning notes). Dead ends recorded there — reg
blocking, `__shfl`, LDS bank padding — are worth not repeating.

Benefits every qwen3.5 workload (serving prefill, calibration), not just eval.

### 2b. LUT3 as the only in-memory bf16 form — blocked on a GEMM rewrite

The idea (worth doing, not yet payable): `Bf16Huff` is always expanded at load
even under `HIPFIRE_BF16L3_RESIDENT`, because Huffman has no in-kernel decoder,
so a Huff-stored model gets the 1.507x on-disk win and then throws away the VRAM
and bandwidth win. Transcoding Huff -> LUT3 during load would close that, and
both primitives are already public (`bf16_huff::decode_par`,
`bf16_lut3::encode`). `.hfa` ingestion feeds the same path: the container is an
index of per-file huff-compressed payloads (NOT 7z), so random access is
feasible without a full restore.

What blocks it is prefill. `bench_bf16l3_vs_bf16_gemm` (validates against a CPU
reference before timing — an earlier version timed two disagreeing kernels and
printed a clean-looking table):

| lut3/bf16, >1 = LUT3 slower | N=1 | N=64 | N=256 |
|---|---|---|---|
| dn_qkv 6144x1024 | 0.87x | 1.05x | 2.16x |
| dn_out 1024x2048 | 0.63x | 6.34x | 8.13x |
| ffn_gate 3584x1024 | 0.92x | 1.42x | 2.39x |
| ffn_down 1024x3584 | 0.60x | 6.38x | 7.92x |
| lm_head 248320x1024 | 0.39x | 1.82x | 2.28x |

LUT3 wins batch-1 (up to 2.6x) and loses prefill; crossover is below N=64. Two
theories about the prefill gap were tested and are FALSE — do not re-run them:

- **Weight re-reads.** At NT=8 an N=256 GEMM makes 32 passes over the whole
  matrix while the WMMA GEMM reads its weights once. Raising NT 8 -> 32
  quartered that traffic, bought ~10% at N=256, and made N=1 worse (NT
  accumulators are allocated even at n_cols=1). Kept at 32 anyway: production
  only calls this GEMM at N=256 windows, worth 1.28x there end-to-end.
- **Strided activation loads.** Transposing x to [K,N] so a tile's columns share
  a cache line made every shape 15-30% SLOWER. The [N,K] layout is coalesced
  ACROSS THE WAVE (adjacent lanes own k0 8 apart, 32 bytes), and that matters
  more than per-thread contiguity.

What is actually left is structural: decode a weight tile into LDS once and run
the same WMMA compute the BF16 path uses, making LUT3 purely a storage format.
Then prefill matches bf16 while reading 1.38x fewer bytes, and the transcode
above becomes a straight win everywhere.

Note the framing that makes this worth doing at all: bf16 is the
calibration/reference path, not production serving (oq8++ is smaller, faster and
effectively lossless), so a modest prefill loss is acceptable — but 2.3-8x is
not modest, and it lands on the reference path this document is about.

### 2. Head GEMM — 0.770 s, 34% of the chunk

`gemm_bf16l3_xf32` (`kernels/src/gemm_bf16l3_xf32.hip`), 128 launches at
5994 µs. It got 2.6× against an 8× weight-traffic reduction, so it is **no
longer bandwidth-bound**: 61 GB/s against the GEMV's 189 GB/s.

Cause is the activation access pattern. Each weight element does `BF16L3_NT`
(8) activation loads at `x[(col_base + c) * K + k]` — K floats, 4 KB, apart, so
one cache line per column per element.

Fix, in order:
1. Transpose activations to `[K, N]` so a tile's 8 values are contiguous — one
   line instead of eight. The producer (`forward_chunk_scored` in
   `qwen35/loading.rs`) has the hidden states in hand and can stage them
   transposed.
2. Then widen `BF16L3_NT` (32–64 via LDS staging rather than registers). Weight
   traffic falls from 47 GB toward the 367 MB a single full pass would read.

## Reference sizing — settled, build at 128 chunks

Measured 2026-08-07, before regenerating anything. The old 1175-chunk refs were
2.48 GB each; the current `RefArchive::encode` bit-packs tokens and
`top_indices` at 18 bits (`ceil(log2(248320))`) and would land at ~1.93 GB, of
which **64% is `top_log_probs` stored as raw f32**. That invited a codec, but
the codec turns out to be unnecessary — the chunk count was the real lever.

Convergence, uniform3 against a bf16 reference over 128 chunks:

| chunks | running mean_kld | error vs 128 | SE |
|---|---|---|---|
| 1 | 0.041320 | **26.5%** | 0.007610 |
| 8 | 0.049702 | 11.6% | 0.002691 |
| 32 | 0.053853 | 4.2% | 0.001345 |
| 128 | 0.056243 | — | 0.000673 |

Per-chunk `mean_kld` has a 13.5% coefficient of variation, so **single-chunk
numbers are worthless** — they are ~26% off and no comparison at n=1 means
anything.

Candidates are compared on the SAME chunks, so the relevant statistic is
paired. Per-chunk KLDs of two candidates correlate at r = 0.78, which shrinks
the noise 2.1× versus an independent-samples model:

| resolve | unpaired | paired |
|---|---|---|
| 5% | 57 chunks | **16** |
| 2% | 352 | **95** |
| 1% | 1407 | **378** |

Real config gaps are much larger than that floor: uniform3 vs down7rest1 is
0.056243 vs 0.063887, a **13.6% difference resolvable at n ≥ 3 chunks**.

**Decision: 128 chunks.** Measured 210,625,187 bytes, ~4.8 min to build, 95% CI
halfwidth 0.000966 (1.7% of the mean). That is 9.2× smaller than a 1175-chunk
ref and resolves everything the outlier-budget sweep compares. Go to 378 chunks
(≈620 MB) only if 1% resolution is ever actually needed. No f16 log-probs, no
delta coding — the format work is moot at this size.

## Caveats to settle, not just perf

### Batched body is not numerically identical to per-token

Accepted deliberately when the BUG-001 guard was lifted, and documented in
`is_batchable_la` (`qwen35/mod.rs`), but **un-root-caused**:

- typical |Δ logit| ~6e-2, max 2.4e-1 (against ~4e-6 for pure reordering)
- only 15% of positions keep the same top-256 set; top-1 argmax agrees 99.36%
- deltas are flat across position (5.99e-2 first half vs 6.60e-2 second), so it
  is a per-position path difference, not accumulating drift

The `mean_kld` 0.0409 → 0.0413 quoted in `22d0d2825`'s message as the metric's
response to batching is **not evidence of anything** — it was an n=1 comparison,
and a single chunk is ~26% off the converged mean (see "Reference sizing").

**Measured properly 2026-08-07 and closed.** Same reference, same chunks, same
candidate, 16 paired chunks, per-token forced via `HIPFIRE_PREFILL_BATCHED=0`:

| forward | mean_kld |
|---|---|
| batched | 0.052349 |
| per-token | 0.052109 |
| difference | **+0.000240 (+0.46%)**, 95% CI [+0.13%, +0.79%] |

The CI excludes zero, so batching carries a small **systematic** positive bias,
not noise — per-chunk correlation between the two paths is 0.9993. But it is
0.46%, against config gaps of ~13.6%, i.e. **3.1% of the signal**, and below the
1.7% CI halfwidth of a 128-chunk reference. Every candidate takes the same path,
so it largely cancels in comparisons.

Verdict: acceptable for ranking quant configs, which is what these references
are for. NOT safe to mix references built on different paths, and worth
re-checking if a future format interacts with the KV tier differently. The
underlying cause (suspected per-tile vs per-token q8 KV scales) is still
un-root-caused, but is now bounded rather than unknown.

Hypothesis, untested: q8 KV scales taken per-tile in the batched attention
versus per-token in the fallback. Worth settling because a KLD *reference* is
the ground truth other artifacts are scored against — a 1% shift in the metric
is small for serving and not obviously acceptable for a reference. A short-context
A/B (where the two paths should converge) plus a check against the duat 3090 HF
oracle would say which one is closer to truth.

Note also that batched prefill *requires* a quantized KV tier: the F4 guard
rejects f32 KV outright (that tier has only `BatchEq(1)`, MissingImpl at
resolve), so references are now built with Q8 KV where they previously used f32.

### tiny-quant gate has 8 pre-existing failures

`hfq4`/`q8f16`/`mq4`/`mq6` across qwen2, gemma3, minimax, qwen3_5, qwen3_5_moe.
Drift values were byte-identical before and after every commit in this series,
so they are not from this work — but while they stand, that gate cannot
distinguish a new regression from the old ones. Worth re-recording baselines.

## Tools worth knowing about

- `HIPFIRE_KLD_PHASE_TIMING=1` splits a chunk into body/head/download/scoring.
  The four sit on different resources; a wall-clock total cannot rank them, and
  three separate wrong guesses were made before this existed.
- Profiling the daemon with rocprofv3: see the recipe in `AGENTS.md`
  (Verification). Drive it over the stdin JSON protocol; `--attach` cannot work
  here (`ptrace_scope=1`).
- `HIPFIRE_QWEN35_BF16_HEAD=1` forces the plain-BF16 head back for A/B against
  the packed LUT3 default (packed measured 2.6× better per position).
