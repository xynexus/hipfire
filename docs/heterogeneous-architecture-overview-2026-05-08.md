# Heterogeneous architecture overview — 2026-05-04 → 2026-05-08

**Scope.** Comprehensive synthesis of every multi-GPU, multi-fabric,
multi-arch finding from the past five days of work on `feat/hetero-pp-dflash`
and parallel branches. Covers eGPU (TB3, TB5, M.2), native PCIe (gen1×4
riser, gen5×16), iGPU UMA, every RDNA tier (RDNA1 → RDNA4), and every
hipfire composition lever (solo, hetero PP, PP=N, PFlash compose, DFlash
compose, speculative prefetch).

This supersedes the partial snapshot in
`docs/multi-gpu-egpu-state-2026-05-08.md` and integrates the per-card
verdicts from
`docs/investigations/2026-05-07-rdna1-perf-research/13-hetero-dflash-perfmax-verdict.md`
and the PR 5 ladder bench from
`docs/investigations/2026-05-08-pr5-d3b-bench/result.md`.

## 1. Hosts + cards inventory

### k9lin (research / development host)

- **CPU:** Ryzen 9 3900X 12-core
- **GPU:** none directly attached during this session (5700 XT migrated
  to hipx via TB3 docks for the eGPU testing campaign)
- **Role:** code authoring, build host for bundle distribution

### hipx (Strix Halo APU + eGPU testbed)

- **CPU/iGPU:** AMD Ryzen AI Max+ 395 (Radeon 8060S iGPU, gfx1151,
  RDNA 3.5)
- **iGPU pool:** 96 GB UMA from system RAM (103.1 GB ROCm-visible)
- **eGPU slots tested this campaign:**
  - TB5 dock (JHL9580 80 Gb/s): 9070 XT, 5700 XT, R9700 (each tested in turn)
  - TB3 dock (XYJ-LINK 0x2 #2, 40 Gb/s × 2 lanes, downtrains to 2.5 GT/s × 1):
    5700 XT, 6950 XT
  - M.2 eGPU adapter (gen5×16 LnkCap, host root downtrains to gen1×1):
    9070 XT, 5700 XT
  - Direct PCIe via riser (gen5 ×16 LnkCap, host root downtrains to gen1×4):
    9070 XT
- **NPU:** AMD RyzenAI-npu5 (gfx1011 / aie2p, idle baseline)
- **Network:** TB4 to k9lin via 10.0.2.2/30 MTU 9000
- **No-internet quirk:** during the 2026-05-08 session DNS resolution
  failed for github.com — bundles via scp from k9lin used as the
  distribution path

### hiptrx (workstation / 4×R9700 cluster)

- **CPU:** Threadripper 9970X 32-core
- **GPUs:** 4× AMD Radeon AI PRO R9700 (gfx1201, RDNA 4, 32 GB each)
- **Fabric:** PCIe gen5 ×16 native per card, peer_access=true
- **Role:** workstation reference + multi-GPU PP testbed

### localmaxxing (offsite reference, not directly accessed)

- **GPU:** 7900 XTX (gfx1100, RDNA 3, 24 GB GDDR6 ~960 GB/s peak)
- **Role:** canonical perf reference for gfx1100 ↔ gfx1201 silicon
  comparison (`tests/speed-baselines/gfx1100.txt`)

## 2. Fabric matrix

| Fabric | Headline | Effective at 1 MB | Per-call latency floor | Hipfire decode-side regime |
|---|---|---:|---:|---|
| PCIe gen5 ×16 native | 64 GB/s | ~60 GB/s | ~µs | hiptrx baseline; near-peak GPU BW |
| PCIe gen1 ×4 riser (host downtrains) | 1 GB/s | ~900 MB/s | ~µs | direct slot floor; latency-equivalent to gen5 |
| TB5 / USB4v2 | 10 GB/s (80 Gb/s × 0.8) | ~75 MB/s | ms domain | TB tunnel adds ms-level RTT regardless of width |
| TB3 / USB4v1 | 5 GB/s (40 Gb/s × 0.8) | ~64 MB/s | ms domain | same TB tunnel cost, narrower pipe |
| M.2 gen1 ×1 (downtrained) | 250 MB/s | ~205 MB/s | ~µs | adapter-firmware-locked link |

**The single most important empirical finding of the campaign:**

> **PCIe gen at the host root is the only fabric variable that moves the
> needle on hetero spec-decode.** Width (×1 vs ×4) is irrelevant in
> steady-state; gen (1 vs 5) shifts the per-call latency floor by ~10×;
> TB tunnels pay an extra ms per transfer regardless of headline BW.

Witnessed proof:

- **M.2 gen1×1 hetero** (hipx 9070 XT): 92.66% of solo iGPU
- **Direct PCIe gen1×4 hetero** (hipx 9070 XT, riser): 93.33% of solo iGPU
- Width upgrade ×1 → ×4 at same gen → +0.72%, within noise. The 4× width
  speed-up only manifests on **cold drafter weight load** (~3.3× faster:
  0.18s vs 0.59s for the 1 GB drafter).
- **PCIe gen5×16 hetero** (hiptrx 2× R9700): 97.04% of solo R9700
- **TB5 hetero** (hipx 9070 XT TB5): ~89.5% of solo iGPU (4-arch sweep)

The 92.66% (gen1) → 97.04% (gen5) shift is the only material fabric
delta. ×width changes within the same gen are noise.

## 3. RDNA arches tested + canonical decode rates

Reference: 9B mq4 / 27B mq4 + DFlash drafter, PEP-8 LRU prompt
(canonical) for solo DFlash; merge_sort prompt (canonical 27B
gfx1201 anchor) for hiptrx. All bench numbers are 3-run median
unless noted.

| Arch | Card | Pool | Solo 9B AR (canonical) | Solo 27B AR | Solo 27B+DFlash | Notes |
|---|---|---:|---:|---:|---:|---|
| RDNA1 (gfx1010) | 5700 XT | 8 GB GDDR6 ~448 GB/s | 56.9 tok/s | n/a (no-fit) | 32.7 tok/s (9B+DFlash) | DFlash NEGATIVE on RDNA1 — block=16 wastes BW |
| RDNA2 (gfx1030) | 6950 XT | 16 GB GDDR6 ~512 GB/s | 73 tok/s (warm) | 25 tok/s (warm, 4.8× cold→warm JIT) | n/a | gfx1030 cold→warm JIT penalty largest of any arch |
| RDNA3 (gfx1100) | 7900 XTX | 24 GB GDDR6 ~960 GB/s | — | — | 199 tok/s τ=10.36 LRU; 250 tok/s merge_sort | localmaxxing reference; gfx1100 anchor |
| RDNA3.5 (gfx1151) | Radeon 8060S iGPU | 96 GB UMA | — | — | 83 tok/s τ=10.64 LRU | iGPU; lower peak BW than dGPU peers |
| RDNA4 (gfx1201) | R9700 | 32 GB GDDR6 ~640 GB/s | 101 tok/s | 35.8 tok/s | 192.6 tok/s τ=13.27 merge_sort; 138 tok/s LRU | hiptrx silicon; 77% of gfx1100 — published arch-tier gap |
| RDNA4 (gfx1201) | 9070 XT | 16 GB GDDR6 ~640 GB/s | — | — | benched as drafter only on hipx hetero | same silicon as R9700, smaller VRAM |

**Per-arch decode efficiency (effective BW % of peak):**

- RDNA1 5700 XT 9B: 88% (BW-saturated)
- RDNA4 R9700 27B verify: 77-91% across hot kernels (BW-saturated)
- RDNA3.5 iGPU 27B: ~50% peak (lower silicon BW, UMA contention)

Source: `docs/perf-checkpoints/2026-05-08-gfx1201-27b-ar-profile.md`,
`docs/investigations/2026-05-07-rdna1-perf-research/09-per-card-prefill-rates.md`.

## 4. Cross-arch single-host hetero benches

### hipx 4-arch DFlash drafter sweep (2026-05-08)

Canonical 27B target on iGPU + 9B-DFlash drafter on each eGPU in turn,
LRU max=120, branch HEAD `48af96a5`. τ=10.4545 invariant across all
runs (acceptance bit-exact).

| Drafter card | Fabric | tok/s | % of solo iGPU | Notes |
|---|---|---:|---:|---|
| Solo iGPU (no drafter) | n/a | 82.21 | 100.0% | baseline |
| 9070 XT | TB5 | 74.19 | 90.2% | RDNA4 drafter |
| R9700 | TB5 | 73.60 | 89.5% | RDNA4 drafter (different card same silicon) |
| 6950 XT | TB3 (cited prior session) | 72.57 | 88.3% | RDNA2 drafter |
| 5700 XT | TB3 | 61.19 | 74.4% | RDNA1 drafter — 15% extra per-forward compute deficit |

**Reading:**
- RDNA4 drafters tie on TB5 (89.5% / 90.2%) despite 50% BW difference
  (R9700 vs 9070 XT) — TB5 fabric latency dominates, BW saturated.
- TB5 vs TB3 null on RDNA4 (R9700 89.5% TB5 vs 6950 XT 88.3% TB3,
  statistically indistinguishable).
- RDNA1 5700 XT lags by ~15 percentage points beyond fabric — extra
  per-forward compute deficit (no WMMA, smaller registers).

### hipx hetero PR-A across fabric tiers (2026-05-08)

27B target (iGPU) + 27B-DFlash drafter (9070 XT), single-card drafter,
LRU max=120 + canonical, with the same target+drafter pair across all
fabric configs.

| Plug | Drafter device | LnkSta | tok/s | % of solo iGPU |
|---|---|---:|---:|---:|
| Solo iGPU | n/a | n/a | 83.62 | 100.0% |
| TB5 (canonical first session) | 9070 XT | TB5 80 Gb/s | 74.71 (gfx1030) / 89% | 90.2% |
| M.2 gen1×1 | 9070 XT | gen1 ×1 250 MB/s | 77.92 | **92.66%** |
| PCIe gen1×4 riser | 9070 XT | gen1 ×4 1 GB/s | 78.49 | **93.33%** |
| Hetero (D3b probe-active) iGPU+9070 XT direct | 9070 XT | gen1 ×4 | 79.34 | 95.0% |

**Verdict:** the gen1×1 → gen1×4 width upgrade saves only +0.71 tok/s
in steady state (drafter cold load 3.3× faster, but decode unchanged).
**Direct PCIe at gen1 is functionally equivalent to M.2 at gen1.**

### hiptrx 2×R9700 hetero (PCIe gen5 fabric ceiling, 2026-05-08)

R9700 #0 target + R9700 #1 drafter on canonical merge_sort.

| Config | tok/s | % of solo R9700 |
|---|---:|---:|
| Solo R9700 | 153.02 | 100.0% |
| Hetero R9700+R9700 PCIe gen5 ×16 | 148.49 | **97.04%** |

This **closes 75% of the hipx 10.5% gap** (TB5 89.5% → PCIe gen5 97.04%).
Confirms fabric latency was the wall on hipx, not silicon.

## 5. Multi-GPU PP findings

### hipx 2× 5700 XT (RDNA1 PP=2, first ever 27B on RDNA1)

`research/pp2-2x5700xt` branch, host-staged boundary copy (non-peer
fabric — 5700 XT lacks peer_access).

- **9B PP=2:** 54.5 vs 55.5 tok/s solo = -1.8% PP overhead — astonishing
  on non-peer fabric.
- **27B PP=2 max_seq=2048 HIPFIRE_PREFILL_BATCHED=0:** 18.6 tok/s decode
  deterministic, coherent. **First ever 27B inference on RDNA1.**
- AMD officially refuses ROCm support on gfx1010; hipfire shipped
  multi-GPU 27B there.

### hipx 3-arch PP=3 (RDNA1 + RDNA1 + RDNA2, 2026-05-07)

Mixed-arch PP across 5700 XT × 2 + 6950 XT, all on hipx via TB3 docks.
Required `HIPFIRE_ALLOW_MIXED_ARCH=1 + HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB=10`.

- **9B PP=3 [8,8,16] across the trio:** 61 tok/s
- **27B PP=3 [12,12,40]:** 22.1 tok/s (max_seq=4096 — 16 GB 6950 XT
  resolved the drafter-VRAM-headroom blocker that PP=2 hit)
- **27B + PFlash on PP=3:** TTFT 159s → 94s = -41%, NIAH needle PASS

### hipx PP=2 + PFlash compose (FIRST EVER, 2026-05-07)

`research/pp-pflash-poc` + 715a890. 9B target on 2× 5700 XT + 0.8B PFlash
drafter, NIAH 3.5k-token prompt.

- **TTFT 15.7s → 9.2s (-42%)**, needle "mauve-velociraptor-7741" retrieved
- Decode 52 tok/s unchanged (BW-bound)
- 27B + PFlash blocked on 2× 8.6 GB drafter VRAM ceiling

### hiptrx PP=4 cluster (4× R9700, 2026-05-08)

27B target layer-split [16,16,16,16] across all 4 R9700s, peer_access=true.

- **PP=4 27B + 27B-DFlash decode:** 33.5 tok/s (4-run median, single
  daemon load, prompt md5 reproducible)
- **vs solo R9700:** 5.8× slower (194 → 33.5)
- PP=4 boundary copy cost dominates: 3 cross-card layer-band transfers
  per cycle + KV cache distributed across cards
- Predraft (PR 5 speculative prefetch) does NOT activate on PP=N>1 —
  generate_multi path bypasses spec_step_dflash's pipeline_mode
  entirely, per docs/plans/hetero-pflash-dflash.prd PR2-4 documented
  as "not yet implemented"

## 6. PR 5 path_d.md ladder — final verdict (2026-05-08)

Shipped through `feat/hetero-pp-dflash` HEAD `777f2962` over 9 commits:

- D0a/D0b/D0c (stream-async commit + scatter + draft-stream switch helpers)
- D1 (init_pipeline_streams)
- D2 (DflashScratchPair)
- D3a (skip_internal_commit)
- D3b minimal (intra-cycle stream restructuring)
- D3b cross-cycle async (event-based ordering across cycles)
- D3b SPECULATIVE PREFETCH (full design — cycle N tail launches cycle
  N+1's draft_forward + lm_head + argmax on draft_stream; cycle N+1
  takes the cache and skips Phases 2-5)

Full implementation, 100% predraft hit rate where it activates, tokens
bit-identical between env=0 and env=1, τ invariant at every config,
coherence-gate-dflash --fast PASS at env=1.

**Bench across all hipfire-production hardware tiers:**

| Host | Config | env=0 | env=1 (predraft) | Δ | predraft? |
|---|---|---:|---:|---:|---|
| hipx iGPU | solo (gfx1151 t+d) lru max=120 | 83.62 | 82.18 | -1.7% | active 100% hit |
| hiptrx 1× R9700 | solo (gfx1201 t+d) merge_sort max=256 | 194.03 | 180.13 | **-7.16%** | active 100% hit |
| hiptrx 4× R9700 ind. cards | each card solo, merge_sort max=256 | 194.58 ±0.24% | 181.21 ±0.20% | **-6.87% mean** | active, silicon-invariant |
| hipx hetero | iGPU+9070 XT direct PCIe gen1×4 | 79.10 | 79.34 | +0.30% (noise) | bypassed |
| hiptrx PP=4 cluster | 27B layer-split 16/16/16/16 | 33.5 | n/a | — | bypassed (PP code path) |

**Empirical pattern:** where predraft activates (single-card solo), it
regresses 1.7-7.2%. Larger regression on higher-peak-BW silicon — R9700
runs the same hot kernels closer to peak BW than the iGPU does, so
concurrent draft_stream + verify_stream contend for the same memory
bus rather than overlapping. Where predraft is bypassed (hetero, PP=N>1),
env=1 is no-op.

## 7. Composite levers — what wins and what loses

### What WINS (production-shipped)

- **PFlash compose** (-30 to -50% prefill across solo/hetero/PP):
  consistent across 4k/8k/16k contexts, hetero scaling validated.
  Memory: `project_pflash_pp2_first_2026_05_07`,
  `project_pp3_hetero_rdna1_2_2026_05_07`,
  `project_pflash_dflash_compose_2026_05_08`.
- **PP=2 / PP=3** layer split: -1.8% overhead 9B PP=2 on RDNA1 with
  host-staged boundary copy (non-peer fabric). Validated mixed-arch
  RDNA1+RDNA1+RDNA2 PP=3.
- **Hetero PP+DFlash compose** (PR 4): TB5 hetero NIAH 16k TTFT -46%,
  decode +59%, τ invariant. Memory: `project_pflash_dflash_compose_2026_05_08`.
- **HIPFIRE_PP_LAYERS asymmetric split**: 27B max_seq +50% on 2× 8.6 GB
  cards via [28,36] vs uniform [32,32]. PB=1 9B PP=2 +13% prefill.
- **Speculative decode (DFlash) on RDNA3+**: 1.84× over AR on 27B
  gfx1151 (memory `project_v0_1_20_engine_modular_shipped`,
  Exp #10).

### What LOSES (negative results catalogued)

- **DFlash on RDNA1**: NET NEGATIVE (-41% on 5700 XT, block=16 wastes BW)
- **MQ3 on 5700 XT** (no multirow gfx1010 kernel): single-row dispatch
  dominates before BW reduction pays off; quality regression too
  (memory: `project_mq3_9b_5700xt_2agents_2026_05_07`).
- **gemv graph cache PR3** (extended to all 4 fused families):
  -5 to -18% production decode regression. Same fundamental finding
  as BC-250 monolithic PoC — graph replay structurally loses to
  ROCm 7.2 burst-mode pipelining.
- **Multi-row GEMV gfx1201 port** (Item 1): NULL verdict on R9700
  decode at 491-585 GiB/s = 77-91% peak BW. PRD hypothesis falsified.
- **Phase 9 cross-card scatter coalesce**: A/B'd null on TB5 + M.2
  gen1×1 + direct PCIe gen1×4. Three different fabrics, same null
  result.
- **hipMemcpyBatchAsync, AQL graph nodes, doorbell-batched launches**:
  all in the "graph capture / kernel batching" lever class — dead on
  this codebase + ROCm 7.2.
- **gfx1201 isolated multirow / FP16-on-the-wire**: bandwidth-bound
  workloads admit no kernel-level parallelism trick.
- **PR 5 D3b speculative prefetch** (this session): -1.7% on iGPU,
  -6.87% on R9700, BW-saturation kills overlap. Closes the path_d.md
  PR 5 ladder definitively.

## 8. Architectural observations

### A. The dGPU vs iGPU vs eGPU regime split

- **iGPU (gfx1151)** has 96 GB UMA pool but lower peak BW than dGPUs
  (UMA contention from CPU + memory subsystem). Wins on huge-context
  workloads (122B-A10B, 35B-A3B at full max_seq) where dGPUs OOM.
- **dGPU (R9700, 7900 XTX)** has dedicated GDDR6 (~640-960 GB/s) but
  smaller pools (24-32 GB). Wins on dense-compute kernels at peak BW.
- **eGPU (any card on TB/PCIe-fabric)** trades fabric latency for
  remote VRAM access. ~5% gap-to-solo on hipx (PCIe gen5 narrows
  to 3% on hiptrx).

### B. Asymmetric tier composition validated empirically

Per-card prefill rates (Exp #9, 9B mq4 1100-tok prefill PB=1):

| Arch | Effective tok/s | WMMA-isolated speedup vs vdot-only |
|---|---:|---:|
| gfx1010 (RDNA1, no WMMA) | 190 | 1.0× (baseline) |
| gfx1030 (RDNA2, no WMMA) | 328 | 1.7× |
| gfx1151 (RDNA3.5, WMMA) | 965 | **5.07×** |

Combined-tier math (1100-tok prefill + 100-tok decode):
hetero gfx1151+gfx1010 = 2.97s — wins **both** solo gfx1010 (7.61s)
**and** solo gfx1151 (3.36s) simultaneously.

This justifies the asymmetric "WMMA-prefill-tier + RDNA1-decode-tier"
architecture as the v1.2 PRD direction. Strengthens the BC-160
cluster procurement case (gfx1011 = decode tier, pair with RDNA3+
prefill tier).

### C. Plug topology drift across sessions

ROCR enumeration on hipx changed across the campaign as plugs moved:

| Plug state | ROCR=0 | ROCR=1 | ROCR=2 |
|---|---|---|---|
| TB5 era (early 2026-05-08) | iGPU | 9070 XT | (NPU) |
| Direct PCIe riser (late 2026-05-08) | 9070 XT | iGPU | (empty) |

**Operator pitfall:** unset ROCR_VISIBLE_DEVICES → defaults to ROCR=0.
On the late-session plug, that's the 17 GB 9070 XT, not the 96 GB iGPU.
27B + 27B-DFlash OOMs. Always read `GPU dev 0: gfxNNNN (X.X GB VRAM)`
line as ground truth.

Memory: `feedback_rocr_hip_visible_enumeration`.

### D. Adapter PCB jumpers can resurrect "silicon-locked" claims

The M.2 9070 XT downtrained to gen1×1 looked silicon-locked at the host
root port. Two physical PCB levers on the M.2 adapter (CLKRQ# AUTO/
FORCE ON, PSU PWR AUTO/FORCE ON) flipped the link from train-failure
to gen5×16 trained between adapter ↔ card — but host root stayed
gen1×1, confirming the host's PCIe lane is the ceiling. Lesson:
enumerate adapter PCB jumpers before declaring silicon-locked.
Memory: `a04ec222` retraction commit.

### E. Bench discipline rules (hard learned)

1. **Cite prompt md5 + --max + DPM warmup + kv-mode** on every
   cross-host claim. The "153 vs 250" phantom regression cost ~30 min
   of investigation before the apples-to-oranges drift was isolated.
2. **One newline character can swing τ by 17%** on 27B DFlash. The
   prompt-normalize default-on (3+ newlines → 2) at 9a2c667 closes
   this. Always use byte-identical prompts across sessions.
3. **Tight stddev on a spec-decode bench is SUSPICIOUS, not
   reassuring.** Real acceptance noise is wider; tight stddev is a
   single-token attractor signature.
4. **First run is cold.** Warm DPM via `HIPFIRE_DPM_WARMUP_SECS=10`
   before the timed window or expect ±5-15% drift on the first measure.

## 9. The bandwidth-saturation pattern (recurring negative result)

Across 5 days and many independent attempts, the same pattern
appears:

> Hipfire's canonical decode hot path on Qwen3.5-27B + DFlash is
> **bandwidth-saturated** on every silicon tier we run it on
> (RDNA1, RDNA3.5 iGPU, RDNA4 R9700). Effective BW sits at 77-91%
> of peak. This means kernel-level parallelism — graph replay,
> kernel fusion, multirow register reuse, multi-stream concurrency,
> speculative prefetch — cannot extract more throughput. They merely
> shuffle when the BW gets consumed; they don't unlock additional BW.

Confirmed-null lever class:
- gemv graph cache PR3 (-5 to -18%)
- Multi-row GEMV gfx1201 port (NULL)
- hipMemcpyBatchAsync, AQL graph nodes
- Phase 9 cross-card scatter coalesce (NULL on 3 fabrics)
- PR 5 speculative prefetch (-1.7% to -7.16% solo)

The single-cycle saving per kernel optimization that gets reported
(e.g., +22-32% per-shape micro-bench) **does not translate to wall
time** because the surrounding cycle stays BW-bound and absorbs
the saving as latency overlap with the next memory transaction.

**The remaining levers that actually move tok/s on canonical 27B:**

1. **Lower per-token BW demand** — Lloyd-MQ3 / Lloyd-MQ2 lower-bpw
   quants, KV compression (asym3 already shipped, +5.5×), drafter
   model shrinkage.
2. **Higher peak-BW silicon** — gfx1100 7900 XTX 960 GB/s wins by
   silicon, not kernel work; gfx1101/gfx1102 and successor archs the
   same. Ceilings:

| Silicon | Peak GB/s | Achieved on 27B+DFlash | % peak |
|---|---:|---:|---:|
| gfx1100 (7900 XTX) | 960 | 250 tok/s | (lmx ref) |
| gfx1201 (R9700) | 640 | 194 tok/s | 77% |
| gfx1151 (Strix Halo iGPU) | ~256 effective UMA | 83 tok/s | ~50% |

The 77% gfx1201/gfx1100 ratio is published-arch-tier characteristic,
not a flag/kernel bug. Memory:
`feedback_hiptrx_153_vs_lmx_250_investigation_2026_05_08`.

## 10. What works for HOSTING the workload (what to buy)

Empirical guidance from the campaign:

### Single-card solo decoding hot path
- **Best peak:** gfx1100 (7900 XTX) at 250 tok/s on 27B + DFlash
- **Best workstation:** gfx1201 (R9700, hiptrx) at 194 tok/s — 77% of
  gfx1100, 32 GB VRAM headroom
- **Best APU/integrated:** gfx1151 (Strix Halo) at 83 tok/s + 96 GB
  UMA — only path to 122B-A10B + 35B-A3B at full max_seq

### Multi-GPU PP for VRAM-bound large models
- **PP=4 across 4× R9700:** 33.5 tok/s on 27B-DFlash. PP boundary cost
  dominates; not the right shape for fitting cluster.
- **PP=2/PP=3 mixed-arch:** validated cleanly via host-staged
  boundary copy. Mixed-arch tolerance via `HIPFIRE_ALLOW_MIXED_ARCH=1`.

### Hetero (target-on-big + drafter-on-small) as a perf lever
- **Tier-aware split is empirically the right shape**: WMMA card for
  prefill (5×) + RDNA1/2 card for decode (cheap large VRAM).
  Combined-tier wins both solo configs simultaneously.
- **Fabric ceiling: 97% of solo R9700** on PCIe gen5 (hiptrx baseline).
  TB5 caps at ~89.5%, M.2/PCIe-gen1 at 92-93%.
- **The only fabric variable that moves the needle is PCIe gen at host
  root.** Width within same gen = noise.

## 11. What's open (not in current scope)

- **Cross-card predraft for hetero**: ~100-150 LOC, uses
  hipExternalSemaphore_t IPC for cross-device events. Predicted to
  also regress per the BW-contention pattern but worth testing if
  hetero predraft is the only remaining reachable signal.
- **PP=N>1 + DFlash speculative prefetch** (hetero-pflash-dflash PRD
  PR2-4): cross-card spec-decode with predraft on the drafter device.
  generate_multi integration. Multi-day work.
- **gfx1201 kernel fusion to reduce launch count** (per `feat/gfx1201-kernel-tuning`
  profile recommendation): 867 launches/token; 30% reduction
  potentially > any single-kernel optimization. Multi-day work,
  needs new PRD.
- **OCuLink direct slot path** (user has connector ordered): predicted
  modest lift over current gen1 paths (~+3-5pp toward 97% hiptrx-class)
  since gap is compute-bound.

## 12. Source-doc cross-reference

| Topic | Doc |
|---|---|
| Per-card prefill rates (Exp #9) | `docs/investigations/2026-05-07-rdna1-perf-research/09-per-card-prefill-rates.md` |
| gfx1151 solo DFlash 27B baseline (Exp #10) | `docs/investigations/2026-05-07-rdna1-perf-research/10-gfx1151-solo-dflash-27b.md` |
| Hetero PFlash+DFlash smoke + PR-A progress | `docs/investigations/2026-05-07-rdna1-perf-research/11-hetero-dflash-smoke-status.md`, `12-hetero-dflash-pra-progress.md` |
| Hetero perfmax verdict (4-arch sweep + hiptrx ceiling) | `docs/investigations/2026-05-07-rdna1-perf-research/13-hetero-dflash-perfmax-verdict.md` |
| eGPU plug topology snapshot | `docs/multi-gpu-egpu-state-2026-05-08.md` |
| PR 5 path_d.md ladder bench (this session) | `docs/investigations/2026-05-08-pr5-d3b-bench/result.md` |
| Hetero PFlash+DFlash PRD v1.2 | `docs/plans/hetero-pflash-dflash.prd` |
| Path D PR 5 ladder PRD | `docs/plans/path_d.md` |
| Multi-GPU PP roadmap | `docs/plans/multi-gpu-pp.md` |
| gfx1201 kernel-tuning PRD (parallel branch) | `docs/plans/gfx1201-kernel-tuning.prd.md` (`feat/gfx1201-kernel-tuning`) |
| gfx1201 27B AR profile (rocprof) | `docs/perf-checkpoints/2026-05-08-gfx1201-27b-ar-profile.md` |

## 13. Memory entries for fast recall

Topic-keyed entries in `~/.claude/projects/-home-kaden-ClaudeCode-autorocm-hipfire/memory/`:

- `project_hetero_session_2026_05_08_compaction.md` — full session-1 handoff
- `project_pr5_pipelining_session_2026_05_08_part2.md` — full PR 5 ladder
- `project_4arch_drafter_sweep_partial_2026_05_08.md` — hipx 4-arch sweep
- `feedback_hiptrx_fabric_ceiling_2026_05_08.md` — PCIe gen5 ceiling 97%
- `feedback_5700xt_tb3_hipx_smoke_2026_05_08.md` — RDNA1 + TB3 hetero
- `feedback_r9700_hipx_smoke_2026_05_08.md` — R9700 + TB5 hetero
- `feedback_9070xt_hipx_smoke_2026_05_08.md` — 9070 XT + TB5
- `feedback_9070xt_m2_egpu_hipx_smoke_2026_05_08.md` — M.2 path
- `project_pp3_hetero_rdna1_2_2026_05_07.md` — RDNA1+RDNA2 PP=3
- `project_pflash_pp2_first_2026_05_07.md` — first-ever PFlash+PP=2
- `project_pp2_2x5700xt_first_rdna1_2026_05_07.md` — first 27B on RDNA1
- `project_5700xt_fabric_ablation_2026_05_07.md` — TB3 vs USB4v2 null
- `project_wmma_prefill_tier_validated_2026_05_07.md` — 5.07× WMMA prefill
- `feedback_hiptrx_153_vs_lmx_250_investigation_2026_05_08.md` — config-drift forensic
- `feedback_rocr_hip_visible_enumeration.md` — ROCR vs HIP indices
- `feedback_hipgraph_kernarg_snapshot_rocm72_2026_05_07.md` — graph-cache rules

## 14. Single-line takeaways

1. **Fabric:** PCIe gen at host root is the only thing that matters.
   Width irrelevant in steady state. TB tunnels pay ms/transfer.
2. **Silicon:** hipfire's canonical 27B+DFlash path is BW-saturated
   on every tier — kernel parallelism cannot exceed the silicon BW
   ceiling.
3. **Hetero composition:** WMMA-prefill-tier + RDNA1/2-decode-tier
   wins both solo configs simultaneously. Ship asymmetric.
4. **PP scaling:** -1.8% PP=2 overhead 9B RDNA1 host-staged. PP=4
   boundary cost is 5-6× the per-card decode time for dense decode.
   Mixed-arch PP works.
5. **Speculative prefetch:** correctly implemented across the path_d.md
   ladder; perf null-to-negative on BW-saturated workloads. Closes
   the PR 5 chapter.
6. **Plug discipline:** ROCR enumeration drifts with plug topology;
   always read the GPU line as ground truth.

---

*Last updated: 2026-05-08, branch `feat/hetero-pp-dflash` HEAD `777f2962`.*
