# Routing batched prefill to iu4x2 + K-major overlay — +14% prefill

State box: halo, Strix Halo gfx1151, 128 GB UMA, RAM configured 8000 MT/s.
Qwen3.8-27B `OqPlusCompact` (qt 36), G=256, n_out=3, KVarN KV,
`HIPFIRE_KVARN_BATCHED_PREFILL=1`.

## What changed

Batched prefill on compact weights ran `gemm_oq_compact_grouped_wmma` — the iu8
W8A8 core — which sign-extends 4-bit weights into int8 lanes and therefore
spends half its matrix-core throughput on known-zero high bits. The exact W4A8
twin does the same arithmetic with two iu4 WMMA passes (`x = 16*hi + lo`, hi
signed, lo unsigned), and the sparse overlay rides along as a separate K-major
accumulate pass.

Two enabling pieces landed first:

1. `gemm_oq_compact_iu4x2_wmma` now consumes **int8 directly** and splits the
   digit planes while staging into LDS. It previously took two pre-split int4
   planes that no production path could produce. Since K/2 + K/2 == K, the split
   planes are exactly the byte volume of the int8 buffer
   `quantize_act_oq8_batched` already leaves in `oq8_xq_batch`, so this is free
   (147 VGPRs / 9 waves / 0 spills, unchanged; bench medians unchanged).
2. `oq_compact_x8_transpose`, the int8 twin of `oq_compact_x4_transpose`. The
   correction kernel already read `XT` as int8; only the transpose unpacked int4.

## Measured

Two runs each, interleaved, direct daemon stdin protocol.

| prompt | route | prefill tok/s | TTFT ms |
|---|---|---|---|
| 240 tok | iu8 | 193.9 / 192.0 | 1237.8 / 1249.7 |
| 240 tok | **iu4x2** | **220.2 / 220.7** | **1089.7 / 1087.4** |
| 539 tok | iu8 | 175.9 / 176.5 | 3065.1 / 3054.3 |
| 539 tok | **iu4x2** | **201.1 / 201.2** | **2680.7 / 2678.4** |

**+14.6%** and **+14.2%**. Decode is unchanged at 14.5 tok/s, as expected — decode
runs the multicol GEMV, not this GEMM.

Correctness: generated text is byte-identical on both prompts, and identical over
a 453-character generation (sha256 match) on the 539-token prompt.

Under the 1.43x the GEMM microbench predicts for these projection shapes, because
prefill also spends time in attention, rotation, norms and lm_head, and neither
prompt fills the B=256 tile the bench measures.

Opt-in via `HIPFIRE_OQ_COMPACT_IU4X2=1`. Default stays off pending a coherence
battery. `gemm_oq_compact_residual_act_batched` delegates to the same function,
so one edit covers the plain and residual arms both.

## The trap this cost

The first version allocated the K-major scratch (`xt`, `xst`) per GEMM call.
That is roughly 900 `hipMalloc`/`hipFree` pairs per prefill chunk — 64 layers x
~7 projections x 2 — and it took prefill **195.0 -> 132.2 tok/s**, a 32%
REGRESSION that reads exactly like the kernel being wrong. The buffers now live
in `ensure_oq8_scratch_batched` next to `oq8_xq_batch`/`oq8_xs_batch`, and the
same code then measured +14%.

Worth stating because the failure mode is silent: the kernel was correct the
whole time and every microbench still said 1.43x. Only the end-to-end number
moved, and it moved the wrong way.

## Unrelated defect found while validating — since FIXED

`TriAttention eviction only supports Q8, asym2, asym3, asym4 KV modes for now`
(`crates/hipfire-runtime/src/triattn.rs`) — **panicked**, it did not fall back.
Any prompt longer than `physical_cap` (896 at max_seq 8192) killed the daemon
under `kvarn`, which is why the 2k-token arm of this experiment could not be run
at first. Fixed in a follow-up: the load path now declines a TriAttention
sidecar under KVarN and sizes the KV for the full window, and `maybe_evict`
returns an error rather than aborting the process. See
`2026-08-22-triattn-kvarn-eviction.md`.

With that in, the 2059-token arm runs, and the iu4x2 win holds:

| prompt | iu8 | iu4x2 | |
|---|---|---|---|
| 2059 tok | 185.1 / 184.4 | **209.7 / 211.3** | **+13.9%** |

## Gate state

`no-gpu-ci` PASS. `tiny-affected-gate --require-coverage` is at its known
pre-existing 8-cell failure state (qwen2/hfq4, gemma3/q8f16+hfq4, minimax/mq4,
qwen3_5/q8f16, qwen3_5_moe/q8f16+mq6+mq4); the flaky
`qwen3_5_moe_indexed/oq8+(calib)` cell passed. None of the failing cells
exercise qt=36 compact, and this path is default-off.

## Measurement note: `hipfire eval` cached the A/B away

The first attempt to measure this through `hipfire eval` reported *identical*
numbers for both arms and nearly produced a "no effect" conclusion. Cause: eval
caches rows across invocations and its key does not include the environment, so
flipping `HIPFIRE_OQ_COMPACT_IU4X2` replayed the first arm. `--force` fixes it,
and then eval independently reproduces the win:

| route | prefill tok/s (first / reset) |
|---|---|
| iu8 | 194.9 / 199.4 |
| **iu4x2** | **220.3 / 227.6** (+13.5%) |

Written up in `docs/methodology/perf-benchmarking.md`.
