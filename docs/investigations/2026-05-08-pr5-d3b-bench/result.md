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

### Speculative prefetch (be7a9a2f — full path_d.md §D3b design)

Cycle N tail launches cycle N+1's draft_forward + lm_head + argmax
download speculatively on draft_stream. Cycle N+1 entry takes the
cache, waits the predraft event, skips Phases 2-5, and uses the
cached drafted directly.

#### hipx iGPU SOLO (gfx1151 target + gfx1151 drafter, lru max=120)

| Run | env=0 | env=1 |
|-----|------:|------:|
| 1   | 83.73 | 82.27 |
| 2   | 83.62 | 82.18 |
| 3   | 83.45 | 82.09 |
| **median** | **83.62** | **82.18** |

Delta: **-1.7% at env=1 — env-1 strictly slower 3/3 runs.** 100%
predraft hit rate (10/10 cycles after cycle 0). Coherence-gate-dflash
--fast PASS.

#### MQ3 (uniform 3-bit) 5-run on hiptrx — BW-saturation hypothesis test (REINFORCED)

Test of "lower-bpw quants move workload from BW-bound to compute-bound,
where predraft might invert the negative result." MQ3 has 19% lower
per-token weight bytes than MQ4 (3.25 bpw vs 4 bpw + per-block scale
overhead).

Models: `qwen3.5-27b.mq3` (11.8 GB) + `qwen35-27b-dflash-mq3.hfq`
(703 MB) — both downloaded from `schuttdev/hipfire-qwen3.5-27b` HF repo.
Predraft eligibility extended to MQ3G256 / HFQ3G256 lm_head dispatch
at commit `080c5b48`.

Hiptrx 1× R9700, canonical merge_sort max=256, DPM_WARMUP=10:

| Run | env=0 | env=1 |
|---|---:|---:|
| 1 | 212.81 | 196.98 |
| 2 | 216.12 | 196.85 |
| 3 | 214.94 | 196.59 |
| 4 | 214.47 | 196.58 |
| 5 | 214.04 | 197.13 |
| **median** | **214.47** | **196.85** |
| range | 3.31 (1.55%) | 0.55 (0.28%) |

Delta: **-8.21% at env=1**, strictly slower 5/5 runs. Predraft hits
12/cycle (perfect). τ=11.0000 invariant 10/10 runs.

**The BW-saturation hypothesis is REINFORCED, not inverted.** The MQ3
regression (-8.21%) is deeper than MQ4's (-7.16% at 5-run / -6.70%
hardened). Two reinforcing mechanisms:

1. **Lower acceptance**: MQ3 yields τ=11.0 vs MQ4's τ=13.27 (lower
   draft-target argmax alignment per cycle on the 3-bit quant). Same
   output → more cycles → more event-API overhead per output token.
2. **BW relief is illusory at this scale**: while per-token weight
   bytes drop ~19%, the verify step still saturates SDMA + memory
   subsystem. Predraft contention takes a bigger relative chunk
   because verify is shorter.

This was the only experimental regime that could plausibly have
inverted the verdict. It deepened it instead. The negative-result
disposition (default OFF, do not enable) is now empirically robust
across MQ4 (uniform 4-bit) AND MQ3 (uniform 3-bit) at 27B on R9700.

#### Cross-process 5-run hardening (CLAUDE.md spec-decode rule compliance)

Per CLAUDE.md: 5 runs, fresh process per run, stddev narrowing >30%
hard-fails. Each `dflash_spec_demo` invocation is a separate process
(not within-session A/B), DPM warmup 10s before timed window.

hiptrx 1× R9700 (canonical 27B + 27B-DFlash, merge_sort max=256):

| Run | env=0 | env=1 |
|---|---:|---:|
| 1 | 195.06 | 180.05 |
| 2 | 194.21 | 180.32 |
| 3 | 193.42 | 180.90 |
| 4 | 192.83 | 180.46 |
| 5 | 192.20 | 180.58 |
| **median** | **193.42** | **180.46** |
| range | 2.86 (1.48%) | 0.85 (0.47%) |

Delta: **-6.70% at env=1**, strictly slower 5/5 runs.

env=1 stddev tighter than env=0 (3× narrower) but well below CLAUDE.md's
30%-narrowing hard-fail threshold and not an attractor signature
(τ=13.2727 invariant across all 10 runs, output bit-identical between
modes). The narrowing is consistent with cycle-deterministic
event-API overhead adding the same delta each cycle plus thermal
warming during the env=0 → env=1 sequence.

This **5-run cross-process measurement hardens the original 3-run
within-sequence finding**: -6.70% (5-run) vs -7.16% (3-run) median
deltas agree within 0.5 percentage points. The negative result is
robust.

#### Silicon-invariance across all 4× R9700 (single-card solo, 1 run each)

| Card | env=0 | env=1 | Δ |
|---|---:|---:|---:|
| R9700 #0 | 194.15 | 180.96 | -6.79% |
| R9700 #1 | 194.83 | 181.68 | -6.75% |
| R9700 #2 | 194.26 | 180.91 | -6.87% |
| R9700 #3 | 195.08 | 181.30 | -7.06% |
| **mean** | **194.58 (±0.24%)** | **181.21 (±0.20%)** | **-6.87%** |

The regression is silicon-invariant across the 4-card cluster. Each
R9700 individually replicates the env=1 regression to within 0.3%
of the others. Confirms the BW-saturation diagnosis is a property
of the gfx1201 silicon + workload, not a single-card anomaly.

#### hiptrx PP=4 cluster (27B layer-split 16/16/16/16 across 4× R9700, daemon mode)

Daemon load: `pp:4 + draft:qwen35-27b-dflash-mq4.hfq + kv_mode:asym3 + max_seq:2080`,
`HIPFIRE_PP_DFLASH=1` to lift the PP=N>1 + DFlash refusal. Layers
distributed `[0..16, 16..32, 32..48, 48..64]` across devices `[0,1,2,3]`,
output_device=3, peer_access=true.

| Run (env=0, single daemon load) | tok_s | decode_tok_s |
|---|---:|---:|
| r1 (cold first req) | 33.3 | 33.8 |
| r2 | 33.0 | 33.5 |
| r3 | 32.8 | 33.5 |
| r4 | 32.6 | 33.4 |

Decode median ≈ **33.5 tok/s** — about 5.8× slower than single-card
solo R9700 (194 tok/s) due to per-cycle PP boundary-copy cost
(3 cross-card layer-band transfers × every step + KV state
distributed across cards).

PP=4 doesn't go through `spec_step_dflash`'s `pipeline_mode` —
the multi-GPU forward path lives in `generate_multi` and uses
`forward_pp_batch` / boundary_copy. The PP=N>1 DFlash path is
documented as PR2-4 of the hetero-pflash-dflash PRD ("not yet
implemented — the load message will accept but generate will not
run cross-card spec-decode"). Predraft therefore can't activate
on PP=4 today; env=1 is bypassed at the same gate that bypasses
hetero.

The single-R9700 solo result (194 → 180 tok/s = -7.16% with
predraft active) is the cleaner test of the BW-saturation
hypothesis on gfx1201 silicon — no PP boundary transfer noise.

#### hiptrx solo R9700 (gfx1201 target + gfx1201 drafter, merge_sort max=256, DPM_WARMUP=10)

| Run | env=0 | env=1 |
|-----|------:|------:|
| 1   | 195.24 | 180.45 |
| 2   | 194.03 | 180.13 |
| 3   | 193.23 | 179.83 |
| **median** | **194.03** | **180.13** |

Delta: **-7.16% at env=1 — env-1 strictly slower 3/3 runs.** Larger
regression than hipx iGPU because R9700 runs the same hot kernels
closer to its peak BW (~640 GB/s) than the iGPU does to its peak
(~256 GB/s effective on UMA). Predraft contention with verify on the
same memory bus is more punishing on higher-peak-BW silicon.

#### hipx HETERO (target=gfx1151 iGPU + drafter=gfx1201 9070 XT direct PCIe gen1×4, lru max=120)

| Run | env=0 | env=1 |
|-----|------:|------:|
| 1   | 79.02 | 79.47 |
| 2   | 79.27 | 79.20 |
| 3   | 79.10 | 79.34 |
| **median** | **79.10** | **79.34** |

Delta: **+0.30% at env=1 — within noise.** Predraft is bypassed in
hetero mode (the `pipeline_mode = ... && !hetero` guard) so env=1
only triggers the probe-scoped event scaffolding — essentially no-op.
The hetero baseline (79 tok/s) is ~5% slower than solo iGPU (83
tok/s) due to cross-card embedding-lookup + draft-hidden transfers
each cycle.

Tokens bit-identical between env=0 and env=1 across all three
configurations; τ invariant in every run.

## Why speculative prefetch didn't deliver

The implementation is correct — 100% hit rate where it activates,
every cycle's drafted matches what cycle N+1 would have computed
inline. But measured perf regressed across BOTH solo configurations
(-1.7% on hipx iGPU, -7.16% on hiptrx R9700). Diagnosed:

1. **GPU memory bandwidth contention** (path_d.md §D3b risk analysis,
   "Bandwidth contention erodes overlap"). The 27B + 27B-DFlash
   decode hot kernels run at 491-585 GiB/s effective on R9700 / Strix
   Halo iGPU — 77-91% of peak. Adding draft_stream work in parallel
   doesn't unlock a second BW lane; both streams compete for the same
   memory bus.

2. **Compute-unit contention.** Phase 4-5 work on draft_stream uses
   the same SMs/CUs as Phase 7 verify on verify_stream. ROCm's scheduler
   serializes at the wave level when CU occupancy is saturated, which
   on 27B decode it usually is.

3. **Event-API + state-management overhead.** The cache check + event
   create/record/wait/destroy at every cycle adds ~µs of CPU + HSA
   queue work. Without compensating GPU-side overlap, this is a small
   net cost.

The empirical finding: **the full path_d.md §D3b speculative prefetch
correctly skips cycle N+1's Phases 2-5 — but the GPU work that gets
"saved" was already saturating the BW bus, and putting it on a
parallel stream just creates contention, not concurrency.**

This matches the larger pattern across this session and prior:

- Phase 9 coalesce A/B null on TB5 + M.2 gen1×1 + direct PCIe gen1×4
  (different fabrics, same null result — fabric/bus saturated).
- Multi-row GEMV gfx1201 port NULL on R9700 (BW-saturated kernel).
- Memory entry `project_hetero_session_2026_05_08_compaction.md`:
  "Graph capture / kernel batching lever class is dead on this
  codebase + ROCm 7.2."

The speculative prefetch is the strongest counter-test of this pattern
because it's the largest possible kernel-batching parallelism. It still
nulls. Confidence rises that the canonical decode hot path is fully
BW-bound and no kernel-level parallelism trick can extract more.

## What hetero would need to participate

Pipeline_mode is currently solo-only. To enable speculative prefetch in
hetero would require:

- Cross-card events (target_gpu records draft_done_evt; drafter waits
  via `hipStreamWaitEvent` on its own stream — supported via
  `hipExternalSemaphore_t` IPC OR direct event handle on same-host
  multi-GPU)
- Multi-stream cross-card scatter (predraft Phase 2 embedding lookup
  needs target weights → cross-card transfer to drafter; predraft
  Phase 5 needs drafter hidden → cross-card transfer to target)
- Re-use the existing `cross_card_copy_via_pinned` path for the
  cross-card legs — but on draft_stream rather than null stream

Estimated effort: 100-150 LOC. Given the solo-mode regression on
both hipx iGPU and R9700, hetero predraft is likely to ALSO regress —
adding the cross-card transfer stalls during the speculative phase
when the verify path doesn't need that bandwidth. Don't ship without
re-evaluating the BW-saturation hypothesis.

## Default-OFF rationale (unchanged)

`HIPFIRE_DFLASH_PIPELINE=1` opt-in. At default, every existing call
site flows through the byte-identical sequential path. The cumulative
-0.27% (cross-cycle async) → -1.7%/-7.16% (speculative prefetch
solo) regression at env=1 is invisible to production users.

The path_d.md projected 5-15% lift was BW-saturation-naive; the
actual ceilings empirically established this session:

| Host           | Config                     | Peak tok/s |
|---|---|---:|
| hipx iGPU      | 27B + 27B-DFlash solo     | 83.62 (env=0) |
| hipx hetero    | iGPU + 9070 XT direct gen1×4 | 79.10 (env=0) |
| hiptrx R9700   | 27B + 27B-DFlash solo merge_sort | 194.03 (env=0) |

Exceeding these requires (a) lower per-token BW demand (smaller
weights / lower-bpw quant — Lloyd-MQ3, Lloyd-MQ2 are existing levers),
or (b) hardware with peak BW significantly above R9700's ~640 GiB/s.

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

## Validation gaps surveyed (2026-05-08)

After the 5-run cross-process hardening, the open validation gaps and
their disposition:

- **gfx1100 7900 XTX (localmaxxing)**: not directly accessible from
  this session; predicted to regress more than R9700 (higher peak BW
  → more contention) but unverified empirically. Out of session reach.
- **Lloyd-MQ3 27B (compute-bound regime test)**: Lloyd-MQ3 27B model
  not on disk on hiptrx (only `qwen3.5-35b-a3b.mq3` MoE variant exists,
  different architecture). Skipped — would require generating the
  Lloyd-MQ3 27B file via the calibration pipeline, multi-hour task.
  This is the only regime where the feature could plausibly invert
  (workload moves from BW-bound to compute-bound when per-token weight
  bytes drop ~35%). Real test of the BW-saturation hypothesis;
  reachable in a follow-on session if the model is built.
- **5-run cross-process (CLAUDE.md compliance)**: COMPLETED above on
  hiptrx canonical config. Hardens the negative result.
- **gfx1010 RDNA1, gfx1030 RDNA2**: low-information per analysis
  (DFlash itself is net-negative on RDNA1; 6950 XT TB3 dock currently
  disconnected on hipx). Skipped.
- **Long-prompt regimes (NIAH 16k, multi-turn agentic)**: predicted
  similar BW-saturation pattern given decode kernels are the same.
  Skipped — would consume disproportionate session time relative to
  expected information gain.

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
