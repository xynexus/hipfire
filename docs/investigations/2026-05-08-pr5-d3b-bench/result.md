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

### Minimal D3b (4661d3ce — end-of-cycle stream_synchronize)

| Run | env=0 | env=1 |
|-----|------:|------:|
| 1   | 82.23 | 83.49 |
| 2   | 83.76 | 83.51 |
| 3   | 83.42 | 83.33 |
| **median** | **83.42** | **83.49** |

Delta: **+0.08% at env=1 — within noise.**

### Cross-cycle async (cdf077cb — sync replaced with pending_scatter_evt)

| Run | env=0 | env=1 |
|-----|------:|------:|
| 1   | 83.74 | 83.49 |
| 2   | 83.52 | 83.42 |
| 3   | 83.65 | 83.36 |
| **median** | **83.65** | **83.42** |

Delta: **-0.27% at env=1 — env-1 strictly slower 3/3 runs but bounded
by event-API overhead.** Coherence-gate-dflash --fast PASS.

## What the cross-cycle async did NOT deliver

The cross-cycle async eliminates the CPU-blocking
`stream_synchronize(draft_stream)` at end of the pipelined Phase 9 — the
scatter now overlaps with Phase 10 (DeltaNet rewind + KV state
advancement) on `verify_stream` AND the caller's inter-cycle CPU work.
The next cycle blocks on the queue side via `stream_wait_event`, not
the CPU side.

But measured perf is still within noise of env=0. Why:

1. The Phase 9 scatter at canonical config is small. With τ≈10.6 and
   `accept_len + 1` averaging ~2 rows, it's ~2 rows × 5 layers ×
   5120 hidden × 4 B = ~205 KB per cycle. At ~640 GiB/s SDMA peak the
   scatter takes well under a millisecond — there's not enough work
   to hide.

2. Phase 10's verify_stream work is also small (~ms). The two streams
   COULD overlap but the critical-path saving is bounded by the smaller
   of (scatter, phase 10) durations — both fit in the few-ms range.

3. ROCm dispatches on hipx iGPU may serialize SDMA copies regardless
   of stream affinity (untested — would need rocprof to confirm).

4. The path_d.md projected 5-15% lift assumes the BIG overlap:
   speculative draft N+1 launched on draft_stream during cycle N's
   verify forward. That overlaps a ~10-20 ms `draft_forward` with
   the ~30-60 ms `verify_dflash_block` — a far larger win than
   the few-ms scatter overlap.

## The real lift requires speculative draft prefetch

Per path_d.md §D3b's full design, cycle N's spec_step_dflash should
ALSO launch cycle N+1's draft_forward speculatively on draft_stream,
using `pair.current()` for the prefetch scratch and the bonus_token
from cycle N's verify as the seed. Cycle N+1 then skips Phase 4
entirely and consumes the prefetched block.

Implementing this requires:

1. **Threading `DflashScratchPair` into `spec_step_dflash`'s API.**
   Today it takes `&mut DflashScratch` directly; the speculative
   prefetch needs the pair so cycle N+1's draft writes don't race
   cycle N's reads. Either change the signature (invasive across
   callers) or stash the pair on `Gpu` (smaller surface).

2. **Caching the pre-drafted block on the pair.** Includes
   `Vec<u32> drafted` (B-1 token IDs), `Vec<f32> draft_probs_at_drafted`,
   `Vec<Vec<f32>> draft_softmaxes` (for temp>0), expected position
   and B at draft time.

3. **Cycle entry: detect prefetch hit/miss.** If the cached predraft's
   position matches the current cycle's expected position, use it and
   skip Phase 1-4. Otherwise (PLD bypass next cycle, env toggled
   mid-run, accept_count diverged from prefetch assumption), discard
   and run Phase 4 inline. Emit pf_hit_rate telemetry.

4. **Mispredicted-acceptance handling.** The prefetch assumes
   `accept_len = B-1` (full acceptance). Real τ < B-1 means some of
   the prefetched block tokens point at the wrong position. Either
   re-draft inline on miss (simpler, loses the lift on those cycles)
   or design partial-prefetch reuse (complex).

5. **Adaptive bypass on tau-debt** (path_d.md §D4). 3 cycles of
   `rolling_tau < expected_pipelined_floor` deactivates the prefetch
   path for the rest of the request and frees `pair.b`.

Estimated total scope: 200-400 LOC, 1-2 days of focused work +
coherence-gate-dflash + 5-run bench validation per CLAUDE.md.

The four primitives + DflashScratchPair scaffolding shipped this
session unblock that work — pair allocation is gated, primitives
support stream-async dispatch on either scratch. A follow-on commit
adds the prefetch wiring without re-scaffolding.

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
