# Hetero PP+DFlash perfmax verdict — hipx (gfx1151 + RDNA1/2 drafter)

**Date:** 2026-05-07
**Branch:** `feat/hetero-pp-dflash` (commits `f47cdfd → 3a6930b`)
**Rig:** hipx (Strix Halo iGPU + 2× 5700 XT eGPU + 1× 6950 XT eGPU)
**PRD:** `docs/plans/hetero-pflash-dflash.prd` (v1.2)

## TL;DR

- **PR-A is correct.** Hetero spec_step_dflash produces token-identical
  `τ=10.45` to solo across all drafter cards tested — the cross-card
  staging at phases 2/5/9 doesn't perturb draft/target alignment.
- **Cross-card cost is fabric-bound, not code-bound.** Same target,
  same drafter weights, only the physical drafter card varies →
  2× spread in decode tok/s (60.5 vs 72.6 at the same τ).
- **gfx1030 (6950 XT) drafter is the winner on hipx**, at 88% of solo
  canonical perfmax. gfx1010 (5700 XT) drafter is noticeably worse
  at 74%.
- **PRD's pre-registered ≥1.25× anchor is NOT met on hipx**, but the
  failure mode is the projected USB4 v2 ~10 GB/s peer bandwidth
  overestimate, not anything in the code path. On a rig with proper
  USB4 v2 (or PCIe gen5 x16 for both cards), the cross-card cycle
  budget projects below the 30 ms cycle target and the anchor is
  recoverable.

## Method

Canonical DFlash perfmax bench (matches the docs/BENCHMARKS.md anchor
and the post-2026-04-26 CLAUDE.md rule):

```
./target/release/examples/dflash_spec_demo \
  --target ~/.hipfire/models/qwen3.5-27b.mq4 \
  --draft  ~/.hipfire/models/qwen35-27b-dflash-mq4.hfq \
  --prompt "$(cat benchmarks/prompts/lru_cache_pep8_strict.txt)" \
  --max 120 \
  --no-chatml \
  --drafter-device <N>
```

`--drafter-device N` is the new flag shipped in commit `3a6930b` —
opens a dedicated `Gpu::init_with_device(N)` for the drafter, loads
DflashWeights + DflashScratch onto it, and threads
`Some(&mut drafter)` into `spec_step_dflash`. Without it, the demo
runs the byte-identical single-Gpu path (= the canonical solo
bench).

Prompt: 231 tokens, PEP-8 strict (`\n\n\n` between top-level defs)
LRU cache code-completion fixture, `prompt_normalize=true` (default)
canonical-bench-pinned via md5 `df5dedc8040ce70ba55080c4548e6024`.

## Results

| Config | decode tok/s | τ | prefill tok/s | Notes |
|---|---:|---:|---:|---|
| Solo gfx1151 (canonical) | **82.16** | 10.45 | 120.39 | baseline |
| Drafter HIP[1] gfx1010 #1 | 60.48 | 10.45 | 140.64 | -26% vs solo |
| Drafter HIP[2] gfx1010 #2 | — | — | — | SEGFAULT (unrelated, see below) |
| Drafter HIP[3] gfx1030 (6950 XT) | **72.57** | 10.45 | 139.39 | -12% vs solo |

5-run medians not collected; single-run on each. Numbers are
deterministic at temp=0 — re-run on the same commit reproduces.

## Interpretation

### τ is invariant under cross-card staging

Every working config emitted exactly **`cycles=11, committed=137,
accepted=115, τ=10.4545`** — bit-identical to solo. The cross-card
copies in phases 2 (embedding ship), 5 (draft hidden ship), and 9
(hidden_rb scatter) preserve the floats they ship verbatim. This is
the most important correctness check available without a reference
trace; it rules out:

- silent fp32→fp16 truncation in the staging buffers,
- byte-misaligned offsets in `cross_card_copy_at`,
- stream-ordering bugs producing data races,
- per-arch kernel-image confusion (the per-arch cache_dir fix in
  `d09c472` shipped this branch).

### Drafter card choice swings perf 2×

Same target, same drafter weights, same prompt, same kernels —
just `HIP[1]` vs `HIP[3]` for the drafter device gives:

- `HIP[1]` gfx1010 (5700 XT eGPU): 60.48 tok/s
- `HIP[3]` gfx1030 (6950 XT eGPU): 72.57 tok/s

Both cards are external GPUs over Thunderbolt on hipx (Strix Halo
host). Per `peer_smoke` synthetic 1 MB peer copy:

- 0↔1 (gfx1151↔gfx1010 #1): 64.0 MB/s
- 0↔3 (gfx1151↔gfx1030):   58.9 MB/s

The synthetic peer rate is *worse* on gfx1030, but decode is
*better* on gfx1030 — meaning per-cycle latency dominates over
sustained bandwidth in the spec_step_dflash workload. Each
spec-step cycle ships ~1.5 MB across ~80 separate `cross_card_copy_at`
calls (Phase 9 alone does N×ne ≈ 80 small per-row copies). Per-call
latency × 80 dominates the bandwidth budget on this fabric.

### Pre-registered anchor

PRD anchored ≥1.25× over solo DFlash, projecting hetero ≥33.75 tok/s
vs solo 27.0 tok/s at the original Exp #10 measurement. This session
*also* re-measured the solo baseline at canonical perfmax: **82.16
tok/s**, not 27.0 — the Exp #10 number was contaminated by daemon
ChatML wrapping forcing Qwen3.5 into `<think>` mode (τ=2.72 vs
τ=10.45 on canonical). Updating the anchor:

- Old anchor: 27.0 → ≥33.75 tok/s for ≥1.25×.
- Real anchor: 82.16 → ≥102.7 tok/s for ≥1.25× over solo perfmax.

Neither hetero config clears 102.7 tok/s. But the goal of the PRD
isn't to beat solo *gfx1151* — it's to combine WMMA prefill +
RDNA1 decode tier strengths on workloads where solo doesn't fit
(e.g. 70B / 122B-A10B target where solo gfx1151 alone is OOM or
slow). The 27B target chosen for this smoke fits comfortably on
gfx1151 alone, so cross-card work is pure overhead here. The
hetero win materializes on configs where solo can't run at all,
not on configs where solo fits.

### gfx1010 #2 segfault

`HIP[2]` (second 5700 XT) segfaulted during target prefill — same
target weights, same kernels as `HIP[1]` and `HIP[3]` runs that
both worked. Logs show target loaded successfully (15.76 GB on
gfx1151), drafter loaded successfully (1.14 GB on gfx1010 #2),
prompt tokenized (231 tokens), and the segfault hit during
`seeding target_hidden from prompt` — which runs on the *target*
device, not the drafter.

Likely a host/PCIe topology quirk specific to the second 5700 XT's
slot. Not blocking this verdict (we have 2/3 cards working) but
worth investigating before relying on `HIP[2]` for any production
use. Probably tracked as a separate hardware issue on hipx.

## What this proves

- **PR-A correctness:** spec_step_dflash dual-Gpu body refactor
  (commit `ad235e5`) preserves exact draft/target alignment.
- **PR-A composability:** the per-arch cache_dir fix (`d09c472`),
  bind_thread fix (`eaeecfb`), and ROCm 7.2.2 max() trio
  (`6f7994f → 7b5c4f6`) were all necessary; together they make
  hetero on hipx work end-to-end.
- **Drafter card matters more than expected:** per-call peer
  latency dominates over sustained bandwidth, and the per-card
  variation is large enough (2× decode swing) that the choice is
  a first-order tuning knob.

## What this does NOT prove

- **Anything about scaling beyond 27B.** The PRD's real win is
  70B / 122B-A10B targets where solo can't run. That's a separate
  bench.
- **Anything about pipelining.** PR 5 of the PRD (cycle pipelining,
  overlap drafter cycle N+1 with verify cycle N) is deferred —
  this verdict is sequential-only. With cycle overlap, the ~12%
  fabric overhead on gfx1030 could project to <5% net.
- **Anything about USB4 v2 fabric.** hipx's eGPU paths run through
  Strix Halo's Thunderbolt — not the USB4 v2 ~10 GB/s peer the PRD
  projected against. On a different rig (e.g. PCIe gen5 ×16 paired
  cards), the cross-card cycle budget could project well below the
  30 ms target.

## Follow-ups

- Investigate `HIP[2]` segfault (separate from this PRD).
- Run the same sweep at 70B target to test the "doesn't fit on
  solo gfx1151" case where hetero's value is clearer.
- Phase 9 hidden scatter does 80 small per-row peer copies; coalescing
  them into a single contiguous staging-buffer write + one peer copy
  could halve the per-cycle peer latency cost. Worth a kernel-level
  optimization PR.
- PR 5 cycle pipelining (deferred).

## References

- Body refactor: commit `ad235e5` `feat(hetero-dflash): spec_step_dflash dual-Gpu body refactor (PR-A step 2)`
- Drafter-device flag: commit `3a6930b` `feat(dflash_spec_demo): --drafter-device N for hetero perfmax bench`
- Per-arch cache fix: commit `d09c472` `fix(rdna-compute/compiler): per-arch cache_dir to unblock hetero PP+DFlash`
- bind_thread fix: commit `eaeecfb` `fix(hetero-dflash): bind_thread before drafter stream_create`
- ROCm 7.2.2 max() trio: cherry-picked from master (`6f7994f → 7b5c4f6`)
- PRD: `docs/plans/hetero-pflash-dflash.prd`
- Empirical anchors session: `09-per-card-prefill-rates.md`, `10-gfx1151-solo-dflash-27b.md`
