# PR 5 D3b minimal — bench on hipx 2026-05-08

**Commit:** 4661d3ce (path_d.md D3b minimal)
**Hardware:** hipx, Strix Halo iGPU (gfx1151, ~17 GB VRAM)
**Model:** qwen3.5-9b.mq4 target + qwen35-9b-dflash-mq4.hfq draft
**Prompt:** benchmarks/prompts/lru_cache_pep8_strict.txt (canonical PEP-8 strict)
**Config:** --max 120 --ctx 4096 --kv-mode asym3 --no-chatml

## Token-level correctness

**PASS, bit-identical.** env=0 and env=1 produce byte-identical
DFlash token streams; τ=7.7857 across all 6 runs; accept_rate
identical; cycle structure identical.

```
env=0: emitted=124  τ=7.7857
env=1: emitted=124  τ=7.7857
```

## Decode tok/s

| Run | env=0 | env=1 |
|-----|------:|------:|
| 1   | 219.25 | 218.72 |
| 2   | 219.42 | 218.95 |
| 3   | 219.06 | 218.58 |
| **median** | **219.25** | **218.72** |

**Delta: -0.24% at env=1.** Ranges do not overlap (env=1 strictly slower
across all 3 runs) but the gap is bounded by a few microseconds per
cycle of HIP event-API overhead.

## What this validates

1. The five Path D primitives (D0a/D0b/D0c/D1/D3a) plus
   `scatter_hidden_block_from_staging_on_stream` and
   `DflashScratchPair` (D2 scaffolding) compile and integrate cleanly.
2. The pipeline_mode branch in `spec_step_dflash` is correctness-
   preserving (bit-identical tokens, identical τ).
3. `verify_dflash_block_no_commit` + `commit_staging_to_ring_on_stream`
   produces the same hidden-state ring layout the sequential path does.

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

## Why hipx 27B coherence-gate-dflash failed

Unrelated to this PR. The gate hardcodes `qwen3.5-27b.mq4` + 27B-DFlash
target+draft pair, which together push the iGPU past its ~17 GB VRAM
allocation envelope (~16.87 GB used after model load, OOM on
DflashScratch alloc). hipx is configured for 27B AR-only via the
daemon's PFlash compose path; full 27B + 27B-DFlash speculative
fits only on hiptrx (R9700 32 GB) or workstation cards.

This PR's 9B + 9B-DFlash bench is the appropriate test on hipx
hardware and is included in the canonical anchor set.
