# §M7 wall-clock — measured, and the crossover cannot exist yet

Status: measured 2026-08-22, nix1 / gfx1103, `Qwen3.6-35B-A3B--oq4`, kvarn KV.
Companion to `2026-08-22-m7-amortization-measured.md`, which settled the
weight-byte half.

Unblocked by the executor fix in `m3-drain-before-teardown`: before it, the v2
executor lost every admitted stream when a teardown frame shared the batch, so
no width measurement was possible at all.

## The scaling curve

N concurrent sessions, 16 tokens each, greedy. Decode span is measured
externally from first token to last, so model load is excluded.

| N | v2 aggregate | inline aggregate | v2 advantage | v2 per-stream |
|---|---|---|---|---|
| 1 | 24.01 tok/s | 24.10 tok/s | 1.00× | 24.01 |
| 4 | 22.58 | — | — | 5.64 |
| 16 | 22.43 | 10.86 | **2.07×** | 1.40 |
| 32 | 18.42 | 7.89 | **2.33×** | 0.58 |

Two facts, and the second is the one that matters.

**1. The v2 executor is worth having.** At width it is 2.07–2.33× the inline
path. The inline path *collapses* under concurrency (24.1 → 10.9 → 7.9); v2
holds roughly flat (24.0 → 22.4 → 18.4). That is a real result and it is what
the march loop buys.

**2. Aggregate throughput does not GROW with N under either path.** Sixteen
concurrent streams produce no more tokens per second than one. Concurrency is
being time-sliced, not exploited.

## Why there is no crossover to find

§M7's thesis is that module-major execution beats layer-major past some N,
because distinct experts touched grows sublinearly — measured at 1.74× sharing
at N=16 and 6.59× at N=128 (`2026-08-22-m7-amortization-measured.md`).

**Realising that requires one forward pass to serve N streams' tokens at once.**
Neither path does. The march loop steps one stream per quantum through the
single resident session slot, park/resume between quanta: N sequential forwards,
each amortising nothing across streams. The available sharing is never
collected.

So the crossover is not "not yet reached" — it is **not measurable**, because no
execution mode on the decode path coalesces across streams.

## What is missing is wired-but-uncalled

The batched decode machinery exists. `Qwen35DecodeBatchBackend` carries
`FusedDenseLayerChunked` and `FusedGroupedMoeLayerChunked`, and
`select_qwen35_decode_batch_backend` chooses between them by arch and session
count (`hipfire-generate/src/lib.rs:1476-1523`).

**Nothing in production calls it.** Outside its own unit tests and the daemon's
`generate_batch_prefill_tests.rs`, there is no consumer — the selector is tested
and unreachable. Prefill has its fused multi-session path; decode has the
backends and no caller.

## The next step, precisely

Make `march_streams` dispatch **one batched step across all runnable streams**
rather than one stream per quantum, routing through
`select_qwen35_decode_batch_backend`. That is where §M7's 1.74×–6.59× would
finally be realised, and only then does the crossover become a measurable
quantity.

Note the shape of that change: it does **not** need the per-slot MoE split. Per
`2026-08-22-m4-premise-falsified.md`, per-slot dispatch costs +39% launch count
in a workload measured at >99% launch-bound. Cross-stream batching is the
opposite move — it *reduces* launches per token by serving N streams in one
pass. Those two directions were conflated in §M4's framing; only the second one
serves M7.
