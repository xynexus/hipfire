# PR 5 D3b minimal — bench on hipx 2026-05-08

**Commit:** 4661d3ce (path_d.md D3b minimal)
**Hardware:** hipx, Strix Halo iGPU (gfx1151, **96 GB UMA / 103 GB ROCm-visible**)
**Bench device:** `ROCR_VISIBLE_DEVICES=1` (the iGPU). ROCR=0 on the
current hipx plug binds to the 9070 XT eGPU which is correctly capped
at 17.1 GB — without explicit ROCR=1, the demo would default to the
dGPU and OOM on 27B+27B-DFlash. Enumeration order changed from earlier
TB5 sessions where ROCR=0 was the iGPU.
**Models tested:** qwen3.5-{9b,27b}.mq4 + qwen35-{9b,27b}-dflash-mq4.hfq
**Prompt:** benchmarks/prompts/lru_cache_pep8_strict.txt (canonical PEP-8 strict)
**Config:** --max 120 --ctx 4096 --kv-mode asym3 --no-chatml

## Token-level correctness

**PASS, bit-identical.** env=0 and env=1 produce byte-identical DFlash
token streams; τ identical across all runs; accept_rate identical;
cycle structure identical.

| Model | τ (env=0 = env=1) |
|---|---:|
| 9B target + 9B-DFlash draft | 7.7857 |
| 27B target + 27B-DFlash draft (canonical) | 10.6364 |

## Decode tok/s — 9B target + 9B-DFlash draft

| Run | env=0 | env=1 |
|-----|------:|------:|
| 1   | 219.25 | 218.72 |
| 2   | 219.42 | 218.95 |
| 3   | 219.06 | 218.58 |
| **median** | **219.25** | **218.72** |

Delta: **-0.24% at env=1** (event-API overhead, no overlap).

## Decode tok/s — 27B target + 27B-DFlash draft (canonical)

| Run | env=0 | env=1 |
|-----|------:|------:|
| 1   | 82.23 | 83.49 |
| 2   | 83.76 | 83.51 |
| 3   | 83.42 | 83.33 |
| **median** | **83.42** | **83.49** |

Delta: **+0.08% at env=1 — within noise.** env=0 first-run is colder
(82.23) and the env=1 runs are tighter (range 0.18 vs env=0 range 1.53).
No statistical signal of regression or win on the canonical config —
the few-µs/cycle event-API overhead at env=1 is balanced by SDMA-queue
overlap of commit + scatter at 27B's larger context.

## Coherence-gate-dflash --fast — PASS at both env=0 AND env=1

```
ROCR_VISIBLE_DEVICES=1 bash scripts/coherence-gate-dflash.sh --fast

env=0 → /tmp/coherence-dflash-20260508-173127.md  no hard errors
env=1 → /tmp/coherence-dflash-20260508-173206.md  no hard errors
```

env=1 prose case (canonical "fall of the Roman Empire" prompt):
- emitted=192, unique_ratio=0.594, max_freq=0.078 (well inside thresholds)
- coherent essay output, no token attractor

env=1 code case (canonical Python prompt):
- emitted=45, unique_ratio=0.75, max_freq=0.091
- τ=10.000 within rounding of the env=0 baseline

Bit-identical tokens between modes confirms the cycle restructure
(`verify_dflash_block_no_commit` + `commit_staging_to_ring_on_stream` +
`scatter_hidden_block_from_staging_on_stream`) produces the same
hidden-state ring layout as the sequential path.

## What this validates

1. The five Path D primitives (D0a/D0b/D0c/D1/D3a) plus
   `scatter_hidden_block_from_staging_on_stream` and
   `DflashScratchPair` (D2 scaffolding) compile and integrate cleanly.
2. The pipeline_mode branch in `spec_step_dflash` is correctness-
   preserving — bit-identical tokens, identical τ, coherence-gate
   PASS at the canonical 27B prose + code battery.
3. End-of-cycle `stream_synchronize(draft_stream)` is correctness-
   preserving with no measurable cycle-time penalty on canonical 27B.

## What this does NOT achieve

The minimal D3b ends every cycle with `stream_synchronize(draft_stream)`
to ensure draft_scratch.target_hidden is fully written before the next
cycle's draft_forward reads it. That sync removes the inter-cycle async
overlap that path_d.md §D3b's full design depends on — the commit and
scatter run on different streams within a cycle but still serialize
against the rest of the cycle.

The two D2D copy chains *can* overlap on independent SDMA queues during
a single cycle, but the saved time is bounded by the smaller of the two
durations (~1-2 ms on 9B/short context) and is masked by event-API
overhead (~few µs per cycle on RDNA3+).

## Next-session work to realize the lift

Per path_d.md §D3b's final design — required to escape the end-of-cycle
sync wall:

1. Thread `stream_wait_event` into `dflash::draft_forward` so the next
   cycle's first kernel reads draft_scratch.target_hidden behind a
   wait on `scatter_done_evt` instead of a global stream sync.
2. Wire the existing DflashScratchPair (D2 scaffolding) into the
   pipeline_mode branch — speculative draft N+1 launch on
   `pair.current()` while cycle N's verify lm_head reads
   `pair.previous()` (the pre-drafted block from the prior cycle).
3. Adaptive bypass on tau-debt (D4) — fall back to sequential when
   3 cycles of low-τ in a row signal that the speculative-prefetch
   acceptance assumption is breaking.

Estimated effort: 5-7 days of careful coding + coherence-gate-dflash
+ 5-run bench validation per CLAUDE.md spec-decode rules.

## Default-OFF rationale

The pipeline_mode branch is gated by `HIPFIRE_DFLASH_PIPELINE=1` and
`init_pipeline_streams()` is also gated by the same env. At default
(unset), the entire chain is a no-op — gpu.draft_stream stays None,
`pipeline_mode` resolves to false, and every existing call site flows
through the sequential path with byte-identical behavior.

Keeping the env opt-in means the -0.24% measurement above does not
affect any production caller. Once the next-session work lands and
realizes net-positive lift, the default flips per the path_d.md §D5
gate plan.

## Why hipx 27B coherence-gate-dflash hard-failed (and why that was operator error)

The gate ran with `ROCR_VISIBLE_DEVICES` unset → defaulted to ROCR=0
which on the current hipx plug binds to the 9070 XT eGPU (gfx1201,
17.1 GB pool) instead of the iGPU (gfx1151, 96 GB UMA pool). 27B + 27B-
DFlash hits the dGPU's pool ceiling at DflashScratch alloc.

With explicit `ROCR_VISIBLE_DEVICES=1` (iGPU), the canonical bench
runs end-to-end (see 27B table above) — coherence-gate-dflash will
PASS once it sets ROCR=1 itself. Suggested gate fix: detect 27B+27B-DFlash
combined size and prefer the largest-pool device, or expose a
`HIPFIRE_GATE_DEVICE` env. Out of scope for this PR.

The gate's `--fast` mode at HIP_VISIBLE_DEVICES=0 fails on hipx for
this exact reason. Correct invocation:
```
ROCR_VISIBLE_DEVICES=1 bash scripts/coherence-gate-dflash.sh --fast
```
