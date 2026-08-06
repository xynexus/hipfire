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

## Caveats to settle, not just perf

### Batched body is not numerically identical to per-token

Accepted deliberately when the BUG-001 guard was lifted, and documented in
`is_batchable_la` (`qwen35/mod.rs`), but **un-root-caused**:

- typical |Δ logit| ~6e-2, max 2.4e-1 (against ~4e-6 for pure reordering)
- only 15% of positions keep the same top-256 set; top-1 argmax agrees 99.36%
- deltas are flat across position (5.99e-2 first half vs 6.60e-2 second), so it
  is a per-position path difference, not accumulating drift
- `mean_kld` moved 0.0409 → 0.0413

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
