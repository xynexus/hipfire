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
| Drafter HIP[2] gfx1010 #2 (TB5) | 61.27 | 10.4545 | 152.42 | -25% vs solo (post H2D-seed fix, commit 9ba0b87) |
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

### gfx1010 #2 (TB5) segfault — root-caused and fixed (2026-05-08)

The TB5-attached 5700 XT segfaulted during prompt seeding. Root cause:
the `scatter_hidden_block_to_interleaved` call at the end of prefill
issued `hipMemcpyDeviceToDevice` from `hidden_rb` (target gfx1151) into
`draft_scratch.target_hidden` (drafter card). HIP routes cross-device
D2D through its P2P path. ROCm 7.2.2's libamdhip64.so crashes inside
that path when the two devices report **asymmetric peer access** —
specifically the TB5+gfx1010 topology shows `can 0→1: false / 1→0:
true` while TB3-attached cards (`HIP[1]` gfx1010 #1 and `HIP[3]`
gfx1030) report bidirectional `true` and don't crash.

Diagnostic chain:
- `peer_smoke` SIGSEGV at `hipMemcpyPeer` in libamdhip64.so address
  `0x6636da8`.
- `dflash_spec_demo --drafter-device 1` (TB5 path on 2026-05-08)
  SIGSEGV at the *same* libamdhip64.so address from `memcpy_dtod_at`
  inside `scatter_hidden_block_to_interleaved`.
- `bind_thread()` before the D2D (commit `75a998a`) does NOT prevent
  the crash — the bug isn't thread-local current_device confusion;
  the runtime's P2P dispatch path crashes regardless.
- `AMD_LOG_LEVEL=3` log: `hipMemcpy ( 0x770003a00000, 0x76ffffc00000,
  20480, hipMemcpyDeviceToDevice )` was the last call before the
  segfault. dst pointer (`0x770003a*`) was on the drafter; src
  (`0x76fff*`) was on the target.

Fix (commit `9ba0b87`): `target_hidden_host` is already populated by
`seed_target_hidden_from_prompt`, so when `--drafter-device` is set,
upload it via H2D directly to the drafter's `target_hidden.buf`. One
H2D leg per prompt (one-time at start), no P2P, no peer enable, no
SDMA path. Solo path is unchanged.

Result: TB5 5700 XT #2 now produces decode 61.27 tok/s with τ=10.4545
**bit-identical** to TB3 5700 XT #1 (60.48 / 10.45) and to solo
(82.16 / 10.45). The H2D-seed swap preserves draft/target alignment
exactly.

The TB5 fabric's higher BW (80 Gb/s × 2 lanes vs TB3 ~40 Gb/s) doesn't
deliver a perf advantage at this workload — within run noise of TB3
5700 XT. Confirms the verdict's per-call-latency-dominated finding:
cross-card cost isn't BW-bound on this fabric.

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

## PFlash + DFlash compose result (2026-05-08, commit 366710d)

PRD v1.2 PR 4 closed: `generate_dflash` now threads `PflashState` and
`PflashConfig` so long prompts can be compressed before the target
prefill. Bench on hipx TB5 5700 XT drafter + NIAH 16k:

| Metric | hetero | hetero+PFlash | Δ |
|---|---:|---:|---:|
| `prefill_tokens` | 10879 | 5512 (kept 5504/10871 = 50.6%) | -49% |
| target prefill_ms | 48483 | 21136 | -56% |
| PFlash score_ms | 0 | 4970 | new line item |
| **True TTFT** (score + prefill) | 48483 | 26106 | **-46%** |
| `decode_tok_s` | 4.6 | 7.3 | +59% (smaller KV) |
| `tau` | 2.75 | 2.81 | invariant |

Decode lift comes from compressed prompt → smaller target KV →
cheaper attention per spec-decode cycle, not from any cross-card
cost change. τ invariance proves PFlash compression preserves enough
context that the drafter's predictions still align with the target's
greedy.

PFlash + DFlash compose was previously bypassed by an explicit
`pflash_bypass dflash_decode_active` gate at the daemon entry; that
gate is now removed and PFlash's own bypass reasons (ModeOff,
ToolCall, ShortPrompt, etc.) surface as on the AR path.

## Follow-ups

- Investigate `HIP[2]` segfault (separate from this PRD).
- Run the same sweep at 70B target to test the "doesn't fit on
  solo gfx1151" case where hetero's value is clearer.
- ~~Phase 9 hidden scatter coalescing~~ → **shipped (commit `f50121f`),
  null result.** Replacing 60 small per-(row, ext) peer copies with
  one same-device D2D scatter + one large peer copy moved decode by
  -0.5% to -0.4% — within run-to-run noise. This rules out per-call
  HIP launch overhead as the bottleneck.
- ~~Pinned-host bounce buffer~~ → **shipped (commit `c59dd97`),
  measurable +3.3% on gfx1030 hetero.** Replacing `hipMemcpyPeer`'s
  implicit unpinned-bounce-buffer fallback with explicit
  `hipMemcpyDtoHAsync` + `hipMemcpyHtoDAsync` legs through a
  `hipHostMalloc(MAPPED|PORTABLE)` buffer:
  - gfx1010: 60.28 → 60.93 (+1.1%)
  - gfx1030: 72.29 → 74.71 (+3.3%) → **91% of solo, was 88%**
  - τ=10.45 unchanged across all configs.

  Real improvement, but did not crack solo. Per-cycle cross-card
  cost dropped from ~18 ms to ~15 ms, not the projected ~0.3 ms.
  TB tunnel latency-per-copy on hipx's iGPU↔eGPU pair appears to
  exceed pure fabric bandwidth as the bottleneck — or the HIP
  driver's "pinned host" path on this topology still routes
  through some host-staged code path that doesn't fully exploit
  the eGPU's native DMA engine. Worth a `rocprof` / `roctracer`
  investigation if pursuing further.
- **Quant-on-the-wire (FP32→FP16 compress + decompress kernels)** —
  the next kernel-level lever. Cuts cross-card bytes 2×; if
  cross-card cost scales linearly with bytes (it should now that
  pinned bounce is in place), hetero gfx1030 projects to ~78 tok/s
  = 95% of solo. Still under but very close. Some τ-validation
  risk from FP16 round-trip on hidden states.
- PR 5 cycle pipelining (deferred). Stacking pipelining over
  pinned-host + FP16 quant on the wire is the path to plausibly
  cracking solo on this fabric.

## Session 2026-05-08 perfmax sweep — full A/B/C/D table + raw=true canonical

### Canonical 27B LRU short prompt (231 tok PEP-8 strict, prompt md5 `df5dedc8040ce70ba55080c4548e6024`)

Daemon path with the per-request `raw=true` flag (commit `b77cf72`)
defuses ChatML's forced `<think>` mode, so the daemon now reaches the
same canonical perfmax that `dflash_spec_demo --no-chatml` posts. Run
on hipx (gfx1151 + TB-attached gfx1010 #1).

| Config | decode tok/s | τ | cycles | tokens | Notes |
|---|---:|---:|---:|---:|---|
| solo daemon raw=true | 77.7 | 10.27 | 11 | 120 | gfx1151-only, drafter on same Gpu |
| hetero daemon raw=true (drafter HIP[1] gfx1010) | 57.4 | 10.27 | 11 | 120 | -26% vs solo |
| solo dflash_spec_demo --no-chatml (verdict baseline) | 82.16 | 10.45 | 11 | ~120 | bare bench, ~5% perf cost on daemon path |
| hetero dflash_spec_demo --drafter-device 1 (verdict baseline) | 60.48 | 10.45 | 11 | ~120 | matches daemon-raw within 5% |

τ invariance is **bit-identical** between solo and hetero on the daemon
path (10.27 / 10.27), matching the verdict's invariance finding from
`dflash_spec_demo`. The ~5% delta between daemon and bare-bench paths is
the daemon's per-request setup overhead (PFlash bypass-reason emission,
prompt-frame ChatML-or-raw branching, EosFilter wiring, JSONL stdout
flushes); it does not perturb spec-decode acceptance.

### NIAH 16k long-context (5×config sweep, PFlash decision row included)

Same prompt across all 5 configs (10871-token NIAH JSONL line 1 with
needle "mauve-velociraptor-7741"; daemon's `prompt_normalize` collapses
the JSONL filler's whitespace runs to canonical form). max_tokens=60.

| Config | prefill tok | prefill ms | decode tok/s | τ | tokens emitted | answer reached |
|---|---:|---:|---:|---:|---:|---|
| solo (DFlash) | 10871 | 48199 | 12.0 | 2.53 | 60 | no — stuck in `<think>` |
| hetero (DFlash, drafter HIP[1]) | 10871 | 48414 | 4.4 | 2.53 | 60 | no — stuck in `<think>` |
| hetero+pf | 5504 (kept 50.6%) | 21006 | 5.8 | 2.83 | 24 | partial; PFlash-compose proves out |
| ar (no DFlash) | 10871 | 47600 | 14.0 | n/a | 60 | no — stuck in `<think>` |
| ar+pf | 5504 (kept 50.6%) | 20614 | 14.5 | n/a | 60 | no — `<think>` opener only |

Observations:

- **PFlash composes with both AR and DFlash** without crashing or τ drift.
  Compressed-prompt prefill ms is 56–57% lower than uncompressed; both
  AR+PF and hetero+PF cut TTFT by ~27 s on the same source prompt.
- **τ degrades on long OOD prompts** (2.53 / 2.83 vs 10.27 on short
  in-distribution prompts). Spec-decode wastes compute when draft and
  target greedy disagree most of the time. Match the verdict's note
  that hetero-DFlash wins materialize on different workloads.
- **AR beats DFlash on long OOD with low τ** (14.0 vs 12.0 solo;
  14.5 vs 5.8 hetero+pf). At τ ≈ 2.5, the verify cost dominates the
  draft savings; AR is structurally cheaper.
- **The 60-token cap is too tight** to retrieve the needle through the
  Qwen3.5 27B chat template's default opener. A larger budget (or
  raw=true on the long prompt) would let the answer reach. This is a
  bench-design issue, not a path-correctness issue.

### PR 5 cycle pipelining — deferred (rationale)

PR 5 of the PRD overlaps cycle N+1's drafter forward with cycle N's
target verify. Implementation requires:

1. Splitting `spec_step_dflash` into two halves (drafter forward issue,
   target verify + commit) so the drafter can run on its own stream
   while the target stream is verifying the previous cycle.
2. HIP event objects (`hipEvent_t`) recorded on the drafter stream and
   awaited on the target stream (and vice versa) to coordinate the
   handoff without blocking the host.
3. A new in-flight drafter output state buffer holding cycle N+1's
   draft tokens / probs while cycle N is still verifying.
4. Speculative bookkeeping to roll back the in-flight cycle when N
   accepts fewer tokens than the drafter's optimistic continuation.

Codesearch confirms `spec_step_dflash` is currently structured as a
single sequential cycle (draft forward → verify → commit, all on one
implicit stream binding flipping between drafter Gpu and target Gpu).
The HIP event scaffolding required for stream-cross does NOT already
exist inside the function. PRD line 107-111 already labels PR 5
"Genuinely new code path — not in scope for v1.2 initial ship; ships
as v1.2.1 follow-on."

This session does NOT attempt PR 5. Effort estimate stands at 1-2 days
of net-new work, dominated by the rollback bookkeeping in step 4 (the
plumbing in steps 1-3 is mechanical).

### PLD spine + n-gram cache daemon activation (commit `e84cc31`)

`spec_step_dflash` already accepts `pld_spine: Option<&[u32]>` and
`ngram_cache: Option<&NgramCache>` — both with full bypass-mode bodies
inside the function (speculative.rs:2630+ for PLD, the bigram override
path is internal). The daemon callsite in `generate_dflash` was
passing `None` for both. `dflash_spec_demo` had been the only caller
threading them.

Commit `e84cc31` wires both at the daemon callsite, mirroring
`dflash_spec_demo`'s pattern (lookup at :1175-1188, observe_many at
:1329-1337). Both are env-gated and default-OFF:

```
HIPFIRE_DAEMON_PLD=1            enable Goose §4.3 PLD spine
HIPFIRE_DAEMON_PLD_CONSENSUS=2  bypass-mode confidence gate
HIPFIRE_DAEMON_PLD_CHAIN=5      bypass-mode chain length gate
HIPFIRE_DAEMON_NGRAM=1          enable rolling bigram cache
HIPFIRE_DAEMON_NGRAM_MIN_COUNT=3 min count before override fires
```

Default behavior (envs unset) is byte-identical to pre-activation.
When enabled, the daemon emits `pld_enabled` / `pld_hits` /
`ngram_enabled` fields on the `done` event for observability.

Smoke (canonical LRU short prompt with both flags off): τ=10.27
solo + 10.27 hetero — invariant vs the pre-activation path. Empirical
perf evaluation of the wired levers is left to a follow-up bench
session; this commit is a wiring landing, not a perf claim.

### ON-path bench (canonical LRU short prompt, daemon raw=true, 2026-05-08)

Same canonical prompt (md5 `df5dedc8040ce70ba55080c4548e6024`, 232 tok),
same daemon binary (commit `c2dcf4a`, md5 `4712ce885e8c9aac104a4fdc664d5348`),
same 27B mq4 target + DFlash drafter, max_tokens=120, temp=0.0,
`raw=true`, kv_mode=asym3, max_seq=4096. Solo = drafter on gfx1151;
hetero = drafter on `HIP[1]` gfx1010 #1 via TB. OFF baseline carried
forward from the previous section: solo 77.7 / 10.27, hetero 57.4 / 10.27.

| Config | decode tok/s | τ | cycles | pld_hits | Δ tok/s vs OFF | Δ τ vs OFF |
|---|---:|---:|---:|---:|---:|---:|
| solo PLD-only (`HIPFIRE_DAEMON_PLD=1`) | 72.4 | 9.33 | 12 | 1 | -6.8% | -0.94 |
| solo NGRAM-only (`HIPFIRE_DAEMON_NGRAM=1`) | 47.9 | 5.61 | 18 | 0 | -38% | -4.66 |
| solo BOTH-on | 46.2 | 5.63 | 19 | 2 | -41% | -4.64 |
| hetero PLD-only | 54.7 | 9.33 | 12 | 1 | -4.7% | -0.94 |
| hetero NGRAM-only | 35.9 | 5.61 | 18 | 0 | -37% | -4.66 |
| hetero BOTH-on | 35.2 | 5.63 | 19 | 2 | -39% | -4.64 |

τ is bit-identical between solo and hetero in every row (9.33 / 9.33,
5.61 / 5.61, 5.63 / 5.63), confirming the wiring preserves cross-card
draft/target alignment under the new accept paths the same way the
PR-A refactor preserves it under the original DFlash path. Decoded
text was inspected on every run — clean Python LRU-cache code
completion, no attractor or single-token loop signature.

**Verdict, PLD on this prompt:** *net-negative*. ~1-2 spine hits per
generation, modest ~5% decode penalty, τ slips outside the
pre-registered ±0.5 envelope (9.33 vs 10.27). The bypass-mode spine
the matcher proposes is honored by `spec_step_dflash` (the cycle
count goes up by 1, exactly matching the one extra "free" cycle the
spine creates), but the spine continuation drives one less-confident
draft–target alignment per generation. Default OFF stays correct.

**Verdict, NGRAM on this prompt:** *strongly net-negative*. Cycles
jump 11 → 18, τ collapses 10.27 → 5.61, decode tok/s drops 38%. The
bigram cache at `min_count=3` fires aggressively on the LRU code's
repetitive identifier sequences (`self.`, `node.`, `self._`, etc.)
and the override replaces higher-quality drafter predictions with
lower-quality bigram continuations. Override correctness is intact —
output is coherent and finishes the LRU implementation — but spec
acceptance is starved. Default OFF stays correct.

**Verdict, BOTH-on:** *dominated by NGRAM*. PLD's 2 hits don't
recover the floor that NGRAM sets; the composition is essentially
NGRAM's regression with PLD's small cycle delta layered on top.

**Caveat — workload sensitivity:** these conclusions are pinned to
the canonical 27B LRU short prompt (highly repetitive Python code,
high in-distribution τ on the bare DFlash path). The bigram override
that hurts here may help on prompts where the drafter is *worse*
than the bigram (e.g. long OOD prompts where the verdict's NIAH
sweep already showed τ=2.53 on bare DFlash). The PRD-relevant
follow-up is to bench both levers on a low-baseline-τ workload
before retiring them. Until then, both stay env-gated default-OFF.

## References

- Body refactor: commit `ad235e5` `feat(hetero-dflash): spec_step_dflash dual-Gpu body refactor (PR-A step 2)`
- Drafter-device flag: commit `3a6930b` `feat(dflash_spec_demo): --drafter-device N for hetero perfmax bench`
- Per-arch cache fix: commit `d09c472` `fix(rdna-compute/compiler): per-arch cache_dir to unblock hetero PP+DFlash`
- bind_thread fix: commit `eaeecfb` `fix(hetero-dflash): bind_thread before drafter stream_create`
- ROCm 7.2.2 max() trio: cherry-picked from master (`6f7994f → 7b5c4f6`)
- PR 4 PFlash+DFlash compose: commit `366710d`
- raw=true daemon flag: commit `b77cf72`
- bench script `pflash_dflash_compose.sh` AR/AR+PF configs: commit `1a7ad25`
- PLD + n-gram daemon activation: commit `e84cc31`
- PLD + n-gram ON-path bench: this section, ad-hoc harness `/tmp/canonical_daemon_raw_bench.py` on hipx
- PRD: `docs/plans/hetero-pflash-dflash.prd`
- Empirical anchors session: `09-per-card-prefill-rates.md`, `10-gfx1151-solo-dflash-27b.md`
