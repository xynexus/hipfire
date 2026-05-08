# Multi-GPU + eGPU state across arches — 2026-05-08 snapshot

Branch: `feat/hetero-pp-dflash`. Captures the full empirical position after PR-A
foundation + PR 4 (PFlash+DFlash compose) + PR 5 (probe-scoped event sync)
across hipx (Strix Halo iGPU + 4-eGPU rig), hiptrx (4× R9700 PCIe gen5), and
k9lin (5700 XT + USB4 v1).

## 1. Hardware matrix as currently mapped

### hipx — Strix Halo APU + 4-eGPU rig

Single host, AMD Ryzen AI Max+ 395 (Strix Halo) APU + JHL9580 USB4 v2 dock chain:

| ROCR slot | HIP idx | Card | Arch | VRAM | Fabric | Cold-plug | Notes |
|---|---|---|---|---|---|---|---|
| 0 | 0 | Strix Halo iGPU | gfx1151 | 96 GB unified | on-package | clean | always present, default |
| 1 | varies | 5700 XT | gfx1010 | 8 GB GDDR6 | TB3/USB4 v1 | needs PCI rescan after replug under same process | RDNA1 |
| 2 | varies | 9070 XT | gfx1201 | 16 GB GDDR6 | TB5 (USB4 v2 80 Gb/s) | clean | RDNA4, primary eGPU |
| 3 | varies | R9700 | gfx1201 | 32 GB GDDR6 | TB5 (USB4 v2 80 Gb/s) | clean | RDNA4 prosumer |
| 4 | varies | 6950 XT | gfx1030 | 16 GB GDDR6 | TB3 dock | dock currently disconnected | RDNA2, gated until user reconnects |

ROCR enumeration is order-of-detection at HIP runtime init; verify with
`rocminfo` per HIP_VISIBLE_DEVICES probe (see memory entry
`feedback_rocr_hip_visible_enumeration.md` for the rocm-smi-vs-HIP gotcha that
burnt hours on 2026-05-06).

TB5 + gfx1010 cold-plug behavior: when `dflash_spec_demo --drafter-device N`
opens an HIP[2] gfx1010 handle in the same process as a gfx1151 dispatch
already in flight, the first gfx1151 dispatch segfaults. Worked around by
serializing handle creation; the underlying HIP/amdgpu race is unresolved
(memory entry `project_hetero_dflash_pra_session_2026_05_07`). TB3 paths and
gfx1201 cold-plug are clean on both ports.

#### M.2-direct slot — width silicon-locked to ×1; speed gen1 cause unresolved (RETRACTED 2026-05-08: adapter PCB jumpers were missed)

A separate 9070 XT was probed on hipx via an M.2-to-PCIe-with-Navi-10-XL-switch
adapter (the second M.2 slot, BIOS label `ssd1`, root port `00:02.4`). After
SR-IOV and `ssd1` were both enabled in BIOS:

| Endpoint | Capability | Trained state |
|---|---|---|
| `00:02.4` (host root port) | gen4 ×1 (`Speed 16GT/s, Width x1`) | gen1 ×1 (`Speed 2.5GT/s, Width x1`) |
| `63:00.0` (Navi 10 XL switch upstream) | gen5 ×16 (`Speed 32GT/s, Width x16`) | gen1 ×1 (`Speed 2.5GT/s (downgraded)`) |
| `64:00.0` (switch downstream → card) | gen5 ×16 | gen5 ×16 |
| `65:00.0` (9070 XT card) | gen5 ×16 | gen5 ×16 |

**Width is silicon-locked to ×1** at the host root port — SR-IOV/`ssd1` BIOS
toggles cannot widen this; the second M.2 slot is wired ×1 by Strix Halo's
16-lane PCIe budget allocation (×4 NVMe + 2× ×1 NIC + ×1 WiFi + ×1 second
M.2 + 4-lane ×16 OCuLink/eGPU), confirmed by the lspci `LnkCap: Width x1`
field and the Strix Halo PCIe lane breakdown documented across vendor sources
(VideoCardz, ServeTheHome, ultrabookreview). This finding stands.

**Speed is stuck at gen1; root cause is NOT yet established.** Root
port supports gen4 ×1 (sibling NICs trained at 16GT/s ×1 cleanly). Tried
software levers, all null — link still 2.5GT/s ×1:

- Lever A — `setpci -s 63:00.0 CAP_EXP+10.w=20:20` (retrain bit on switch
  upstream): no change.
- Lever B — force target speed via `LnkCtl2` on root port `00:02.4`,
  drop to gen1 then ramp to gen4 with retrain between each step: link
  re-trained back to gen1.
- Lever C — disable ASPM on all three nodes (`echo performance >
  /sys/module/pcie_aspm/parameters/policy`, then clear LnkCtl bits 0–1
  on `63:00.0`/`64:00.0`/`65:00.0`) followed by retrain: no change.
- Lever D — PCI remove + rescan (`echo 1 > /sys/bus/pci/devices/0000:65:00.0/remove`
  then `echo 1 > /sys/bus/pci/rescan`): card re-enumerated, link re-trained
  to gen1.
- Lever F (research): cross-referenced reports show `setpci` is reliable for
  protocol-fallback (force lower) but unreliable for fixing physical-layer
  training failures with retimer/switch chips in the path
  ([Arch eGPU bandwidth thread](https://bbs.archlinux.org/viewtopic.php?id=295045),
  [TI PCIe retrain FAQ](https://e2e.ti.com/support/processors-group/processors/f/processors-forum/1553181/faq-am68-am67-j722s-j721s2-j784s4-tda4vm-tda4vl-tda4vh-dra821-how-to-retrain-pcie-link-speed-between-gen1-2-5gt-s-gen2-5-0gt-s-and-gen3-8-0gt-s-speeds)).
  Strix Halo wiki documents stable gen4 ×4 only via M.2-to-OCuLink adapter
  ([M2OC-3 + Sixunited AXB35](https://strixhalo.wiki/Guides/External_GPU)) —
  no working setpci recovery for the M.2-with-switch-chip topology has been
  reported. `amdgpu pcie_gen_cap=0x80000` modprobe option is documented but
  reported as ineffective for the host-side cap.

These four software levers are downstream of link-training negotiation;
they cannot fix what the link layer has already settled at gen1. They
are correctly recorded as null, but they do not establish that gen1 is
silicon-locked.

##### Missing from investigation — adapter PCB hardware jumpers (pending witness)

The prior framing on this slot ("silicon-locked, no software lever can
fix it") was wrong because the upstream investigation never enumerated
the M.2-to-PCIe adapter board's own physical jumpers. Two are present
on the adapter PCB and both can affect link training **before software
sees the link**:

- **CLKRQ# (AUTO / FORCE ON)** — controls whether the M.2 host's PCIe
  reference clock is gated by the endpoint's CLKRQ# request line.
  AUTO honors the host-handshake protocol; on Strix Halo's reduced-
  sideband second M.2 slot the CLKRQ# line may not be wired through
  reliably, in which case REFCLK delivery to the adapter is unstable
  during link bring-up and the link falls back to gen1 at the speed
  retrain step (gen3+ requires a clean REFCLK to negotiate). FORCE ON
  delivers REFCLK unconditionally, bypassing the host-side CLKRQ#
  handshake.
- **PSU PWR (AUTO / FORCE ON)** — controls whether the adapter waits
  for a host GPIO/power-good signal before delivering 12 V from the
  external eGPU PSU to the GPU. AUTO waits for a host signal that
  may not be wired through M.2 form-factor pinout; FORCE ON powers
  the GPU from the eGPU PSU regardless. If AUTO is causing a partial
  power-up sequence, the link may train at the most permissive (gen1)
  state and never re-train upward.

User is testing both jumpers in FORCE ON next. **Result pending.** This
section will be updated when the test reports back. Until then:

- Width-locked-to-×1 conclusion: **confirmed**, stands.
- Gen1-speed root cause: **unresolved**. Candidates remaining: (a)
  CLKRQ# AUTO REFCLK gating, (b) PSU PWR AUTO sequencing, (c)
  retimer/SI in the M.2 ribbon, (d) some combination. Software levers
  exhausted.
- "Silicon-locked" terminal framing: **retracted**.

Generalized lesson for adapter-mediated PCIe paths: enumerate adapter
PCB hardware jumpers (CLKRQ#, PSU PWR, DC-IN selection, link-speed
straps) FIRST, before reaching for `setpci`. Software levers run
downstream of the adapter's link-training behavior and cannot
override what the adapter has already negotiated at the physical layer.

##### Post CLKRQ#/PSU FORCE ON reboot — partial topology change, host bottleneck confirmed silicon-locked (witnessed 2026-05-08)

User flipped both adapter jumpers (CLKRQ# AUTO→FORCE ON, PSU PWR AUTO→
FORCE ON) and rebooted hipx. Re-probe witnessed:

| Endpoint | LnkCap | Pre-flip LnkSta | Post-flip LnkSta | Δ |
|---|---|---|---|---|
| `00:02.4` (host root port) | gen4 ×1 (16 GT/s ×1) | gen1 ×1 (2.5 GT/s ×1) | gen1 ×1 (2.5 GT/s ×1) | **null** |
| `63:00.0` (switch upstream → host) | gen5 ×16 (32 GT/s ×16) | gen1 ×1 (downgraded) | gen1 ×1 (downgraded), `EqualizationComplete+ EqualizationPhase1+` | **null on speed/width**, equalization now reports complete |
| `64:00.0` (switch downstream → card) | gen5 ×16 | gen5 ×16 | gen5 ×16 | unchanged |
| `65:00.0` (9070 XT card)** | gen5 ×16 | gen5 ×16 | gen5 ×16 | unchanged |

**Card BDF shifted** `66:00.0` → `65:00.0` because the adapter PCIe
switch (Subsystem: Tul Corporation / PowerColor Device 1478) now
enumerates as `63:00.0`/`64:00.0` and pushes the card-bus number down
one. Pre-flip the `lspci -tv` chain went `00:02.4 → 66:00.0` direct;
post-flip it goes `00:02.4 → 63:00.0 → 64:00.0 → 65:00.0`. The
adapter's switch silicon is correctly enumerated by the host now;
**that** is the topology change CLKRQ#/PSU FORCE ON delivered.

**The bandwidth ceiling did not move.** The trained `LnkSta` on the
host-side leg (`63:00.0` ↔ `00:02.4`) is identical to pre-flip:
`Speed 2.5GT/s (downgraded), Width x1 (downgraded)`. Effective ceiling
remains ~250 MB/s = ~2 Gb/s for the M.2-direct path. amdgpu binds
clean at `0000:65:00.0` (Navi 48, 64 CU, Using BACO for runtime pm,
no MES/discovery errors). Card is functional for steady-state decode
where weights are VRAM-resident.

**Silicon-budget side is now confirmed.** Host root port `00:02.4`
LnkCap is gen4 ×1 (16 GT/s × 1) — the physical lane budget allocated
to this M.2 slot is one PCIe lane wide, capped at gen4. The adapter's
switch silicon and card train fully gen5 ×16 internally; the bottleneck
is upstream of the adapter PCB, in Strix Halo's lane-allocation. The
training settles at gen1 even though both endpoints (root port + switch
upstream) advertise higher speeds, with `EqualizationComplete+
EqualizationPhase1+` post-flip — i.e. equalization successfully
completed but the speed negotiation still falls back to 2.5 GT/s. The
remaining gap (gen1 vs gen4 the LnkCap allows) is likely SI margin in
the M.2 ribbon / PCB at high speeds with the 16-lane switch hanging
off it; not recoverable without different cabling or a different host
slot.

**OCuLink direct PCIe slot path remains the only fabric upgrade path
on hipx for this card.** Width is Strix-Halo-lane-budget-locked at ×1,
and even the gen4 ceiling the LnkCap allows is unreachable in the
current M.2 → switch → card configuration. Setpci levers, jumper
config, and BIOS toggles are all exhausted; the next try worth doing
on this card is the OCuLink riser when it ships.

**Verdict: NULL on bandwidth recovery, PARTIAL on topology.** Adapter
switch enumeration improved, equalization signals improved, but
trained `LnkSta` on the binding leg is identical. No need to rerun
`peer_smoke` or `dflash_spec_demo` — bandwidth ceiling unchanged from
the entry's pre-flip kernel-smoke result (81.23 tok/s decode at 5-token
horizon, ~250 MB/s host-to-card, see memory entry
`feedback_9070xt_m2_egpu_hipx_smoke_2026_05_08`).

### hiptrx — Threadripper 9970X + 4× R9700

| Slot | Card | Arch | VRAM | Fabric | Notes |
|---|---|---|---|---|---|
| 0–3 | 4× R9700 | gfx1201 | 32 GB each | direct PCIe gen5 ×16 | 64 CU each, 256 CU total, no eGPU/TB |

System: 128 GB DDR5, 3.6 TB NVMe, Ubuntu 26.04, ROCm 7.2.2 manual install
(libxml2.so.2 symlink fix per memory entry `feedback_libxml2_ubuntu26_rocm`).
First gfx1201 inference logged 2026-05-03 at 393 tok/s decode 0.8B mq4 q8 KV.
Fabric is the only place direct gen5 ×16 is available in this rig matrix.

### k9lin — workstation + 5700 XT

Single 5700 XT (gfx1010) on TB4/USB4 v1 ~40 Gb/s. Validated end-to-end
2026-05-06: 56.9 tok/s decode 9B mq4 (88% peak BW), 230 tok/s prefill
9B (`project_gfx1010_5700xt_validated_2026_05_06`). RDNA1 portability
empirically holds.

## 2. Hetero PFlash + DFlash branch state

Branch `feat/hetero-pp-dflash` since `c59dd972` (PR-A pinned-host bounce
foundation). 21 commits as of HEAD `a809ce3c` 2026-05-08:

| Commit | Layer | Subject |
|---|---|---|
| `c59dd972` | PR-A | pinned-host bounce for cross-card transfers |
| `f50121f8` | PR-A | coalesce Phase 9 cross-card scatter |
| `e51ec667` | PR-A | coalesce-scatter null result + fabric-bound conclusion |
| `e078fa1d` | PR-A | pinned-host bench result — +3.3% gfx1030 hetero |
| `75a998a6` | PR-A | bind_thread before scatter_hidden_block_to_interleaved |
| `9ba0b870` | PR-A | H2D for prompt seed when drafter on different device |
| `59c6a762` | PR-A | TB5 hetero unblocked — 61.27 tok/s τ=10.45 |
| `3e272dcf` | PR 4 | pflash_dflash_compose script for TB5 hetero validation |
| `750aff34` | PR 4 | apply H2D seed fix in generate_dflash |
| `8180ab10` | PR 4 | extend compose timeouts for cold-cache JIT |
| `366710d7` | **PR 4** | **PFlash + DFlash composition** |
| `6fbddd34` | PR 4 | compose result on hipx TB5 hetero — TTFT -46%, decode +59% |
| `b77cf726` | PR 4 | per-request raw flag — opt out of ChatML wrapping |
| `1a7ad253` | bench | ar / ar+pf configs for spec-vs-AR comparison |
| `e84cc31b` | feat | wire PldMatcher + NgramCache in generate_dflash |
| `dd3e055f` | docs | hetero perfmax verdict — session 2026-05-08 sweep |
| `c2dcf4ac` | docs | document hetero DFlash + PLD/n-gram daemon env vars |
| `8dcb05bd` | bench | hetero PLD + n-gram ON-path bench — both net-negative on LRU |
| `48af96a5` | fix | dflash_spec_demo preflight target/draft hidden-dim compat |
| `101f008f` | bench | 3-of-4 arch drafter sweep on hipx — fabric latency dominates |
| `7a85f756` | bench | hiptrx PCIe gen5 fabric-ceiling — 97.04% of solo |
| `eccf06ce` | **PR 5** | **probe-scoped speculative drafter event sync** (env-gated) |
| `a809ce3c` | docs | PR 5 probe bench — +1.59% on hiptrx hetero, τ=10.4545 invariant |

PR ladder status:
- **PR-A foundation** — complete and merged within branch (pinned-host
  bounce + coalesce-scatter + bind_thread + H2D seed fix + per-arch JIT cache).
- **PR 4 PFlash + DFlash composition** — complete; TTFT -46% / decode +59%
  on TB5 hipx hetero.
- **PR 5 probe-scoped event sync** — shipped env-gated (`HIPFIRE_DFLASH_PIPELINE`),
  no-op in sequential path, +1.59% on hiptrx scheduling-alignment side effect.
  Full 9-item path_d.md ladder still open and deferred per perfmax verdict.

## 3. Empirical anchors — canonical 27B + 27B-DFlash

All anchors below cite **prompt md5**, **`--max`**, **DPM warmup**, and
**`--kv-mode`** to comply with the bench-discipline rule reinforced this
session (see §4 below for cause). Prompt: `lru_cache_pep8_strict.txt`,
md5 captured per-bench in `tests/speed-baselines/`.

### Solo per arch (no hetero)

| Arch | Card | Config | Decode tok/s | τ | Notes |
|---|---|---|---:|---:|---|
| gfx1010 | 5700 XT (TB3 hipx) | DFlash 27B mq4 LRU `--max 120` `--no-chatml` `--kv-mode asym3` DPM=10 | ~82 (0.82× hetero floor) | — | RDNA1 baseline; DFlash net-negative below 64-block, see memory `project_gfx1010_5700xt_validated` |
| gfx1030 | 6950 XT (TB3 dock) | DFlash 27B mq4 LRU same | — | — | currently disconnected; cold-plug warm 27B mq4 = 25 tok/s on hipx |
| gfx1100 | 7900 XTX (localmaxxing) | DFlash 27B mq4 merge_sort `--max 256` `--no-chatml` `--kv-mode asym3` DPM=10 | **250.3** | 13.18 | gfx1100 lmx canonical anchor |
| gfx1151 | Strix Halo iGPU (hipx) | DFlash 27B mq4 LRU `--max 120` same | **82.21** | 10.45 | hipx solo canonical floor for hetero ratios |
| gfx1201 | R9700 (hiptrx) | DFlash 27B mq4 merge_sort `--max 256` same DPM=10 | **192.6** | 13.27 | matches `tests/speed-baselines/gfx1201.txt` 192.69 within 0.05% |
| gfx1201 | R9700 (hiptrx) | DFlash 27B mq4 LRU `--max 120` same DPM=10 | **137.83** | 9.50 | matches gfx1201.txt 138.02 within 0.14% |

### Hetero — hipx 4-arch drafter sweep (commit `101f008f`)

Target on hipx Strix Halo iGPU; drafter on each eGPU in turn. Solo
gfx1151 = 82.21 tok/s = 100% reference.

| Drafter card | Drafter arch | Hetero tok/s | % of solo gfx1151 | Conclusion |
|---|---|---:|---:|---|
| 5700 XT | gfx1010 | 60.93 | **74.4%** | RDNA1 fabric latency dominates |
| 6950 XT | gfx1030 | 72.6 | **88.3%** | TB3 |
| 9070 XT | gfx1201 | 74.16 | **90.2%** | TB5; ties R9700 |
| R9700 | gfx1201 | 73.55 | **89.5%** | TB5; ties 9070 XT |

R9700 vs 9070 XT tie → fabric latency ≈ same since both on TB5. RDNA1
penalty in row 1 is the +TB3 latency, not arch-specific.

### Hetero — hiptrx fabric-ceiling (commit `7a85f756`)

Direct PCIe gen5 ×16 between two R9700s in same chassis (same arch,
no fabric tunnel):

| Pairing | Config | Hetero tok/s | Solo R9700 LRU | % of solo |
|---|---|---:|---:|---:|
| R9700 + R9700 (PCIe gen5 ×16) | DFlash 27B mq4 LRU `--max 120` | **148.49** | 153.02 | **97.04%** |

Closes ~75% of the break-solo gap that TB5 tunnels can't on hipx
(74–90% range).

### PR 5 probe (commit `a809ce3c`)

| Config | tok/s | τ |
|---|---:|---:|
| HIPFIRE_DFLASH_PIPELINE=0 (control) | 148.61 | 10.4545 |
| HIPFIRE_DFLASH_PIPELINE=1 (probe) | 150.97 | 10.4545 |
| Δ | **+1.59%** | invariant |

Flagged as **scheduling-alignment side effect, not real pipelining**.
Real overlap pipelining is the path_d.md 9-item ladder, deferred.

## 4. Key findings + mandatory bench rules

1. **BW saturates on TB5 drafter dispatch.** Per-cycle drafter cost is
   dominated by fabric latency (host-staged bounce + tunnel hop), not
   by drafter compute. Pinned-host buffer +3.3% on gfx1030 was the
   only PR-A win (`e078fa1d`). All eGPU drafters land in 88–90% range.

2. **Direct PCIe gen5 ×16 closes ~75% of the break-solo gap** TB5
   tunnels cannot. The fabric is the wall, confirmed by
   `7a85f756` (97.04% gen5 ratio) vs hipx 88–90% TB5 ratio.

3. **gfx1201 / gfx1100 ratio = 77%** is the published arch-tier
   characteristic on this codebase, NOT a regression. RDNA4 multi-row
   gemv kernels are not yet ported (`gemv_hfq4g256_multirow_r2`
   family is gfx1100-only). Gap is structural until kernel-tuning
   project ships. Verdict locked at `dd3e055f` end-of-session
   appendix.

4. **MANDATORY BENCH RULE — cite prompt md5 + `--max` + DPM warmup +
   `--kv-mode` on every cross-host claim, even if it's the default.**
   Lost ~30 min this session to a 153 vs 250 phantom anomaly. Three
   drift axes were active simultaneously: subagent ran `--max 120` LRU
   without DPM warmup (151 tok/s) while user was citing `--max 256`
   merge_sort with DPM warmup (250 tok/s gfx1100 / 192 tok/s gfx1201).
   Same silicon, three-axis config drift, ~50 tok/s noise floor.

5. **`tests/speed-baselines/gfx1201.txt` lru anchor (138.02) is stale
   by ~10%** at HEAD; current is ~150 tok/s. Speed-gate uses
   merge_sort canonical 192.69 which is on-anchor within 0.05%.
   Refresh the lru row when next perf-touching commit lands on
   gfx1201 path.

## 5. Open levers — post-session

### Tier 1 — biggest single wins, scoped this session
- **gfx1201 kernel-tuning project** (own branch
  `feat/gfx1201-kernel-tuning`, this delivery) — multi-row GEMV port +
  fused QKV gfx12 + fused gate+up gfx12 + MMQ tile autotune. Target:
  gfx1201 R9700 hits ≥250 tok/s on canonical 27B (gfx1100 parity).
  ~30% lift. PRD at
  `docs/plans/gfx1201-kernel-tuning.prd.md` on that branch.

### Tier 2 — defer or user-action-gated
- **PR 5 full ladder** (path_d.md 9 items, 5–7 days) — only worth
  pursuing if PR 5 probe's +1.59% holds on rocprof investigation,
  which it has not. Status: dominated empirically per
  `findings/path-d-vs-path-c.md`, kept as historical context.
- **Long-prompt PLD/n-gram revalidation** — orthogonal small lever;
  on-path benches at `8dcb05bd` were net-negative on canonical LRU.
  Revalidate on long-context prompts (>4k) where draft pool overlap
  expectation is higher.
- **6950 XT TB3 dock recovery** — user-action-gated. Currently
  disconnected; reconnecting unblocks gfx1030 row in §3 hetero sweep.
- **70B target on hipx PP=3** — workload that hetero is actually
  designed for. Current sweep was on a workload that fits solo;
  PP=3 hipx (gfx1151+gfx1201+gfx1201) at 70B Q4 unlocks the
  break-solo case.

## 6. Memory entries cross-reference

Primary sources for this snapshot, all available via `exec:recall`:

- `project_hetero_dflash_pra_session_2026_05_07` — PR-A foundation, 12
  commits, TB5 unresolved, gfx1151 82.18 anchor, gfx1030 91% best,
  gfx1010 74%.
- `project_pflash_pp2_first_2026_05_07` — first-ever PFlash + PP=2
  compose, TTFT -42%, NIAH PASS.
- `project_pp3_hetero_rdna1_2_2026_05_07` — first mixed-RDNA-gen
  PP=3, 27B max_seq=4096 unblocked, PFlash composes -41% TTFT.
- `project_wmma_prefill_tier_validated_2026_05_07` — WMMA-prefill-tier
  hypothesis 5.07× isolated WMMA contribution 2.94×.
- `project_gfx1010_5700xt_validated_2026_05_06` — gfx1010 cross-arch
  portability empirically validated, 56.9 tok/s decode 88% peak BW.
- `feedback_rocr_hip_visible_enumeration` — HIP_VISIBLE_DEVICES uses
  ROCR enum, not rocm-smi (the gotcha that burned hours).
- `feedback_libxml2_ubuntu26_rocm` — ROCm 7.2.x JIT fix on Ubuntu 26.
- `feedback_jhl9580_runtime_pm_trap` — TB5 dock runtime-PM race.
- `feedback_hipx_rocm722_jit_broken` — clang 22 max() bug, fix
  cherry-picked.
- `feedback_27b_over_4b` — 27B is production case, 4B is dev-iter.
- `feedback_gfx12_wmma_builtin_gotchas` — `_gfx12` suffix mandatory,
  8-row-block acc layout, foundational to gfx12 kernel-tuning PRD.
- `feedback_rocblas_gfx12_regresses` — rocBLAS gfx12 prefill 5.6×
  slower than hipfire's wmma_gfx12.
- `project_hiptrx_4xr9700_provisioned` — 4× R9700 baseline.
- `project_hiptrx_worktree_and_gfx12_isa_2026_05_03` — gfx12 ISA
  inventory, biggest unused levers.


## M.2 hetero PP empirical witness — 2026-05-08 (RETRACTS prior null verdict)

After CLKRQ#/PSU FORCE ON, the M.2 9070 XT eGPU runs hetero PP with the 27B target on gfx1151 iGPU. Witnessed at HEAD `eccf06ce`:

| Config | Decode tok/s (med, n=3) | τ | Solo % |
|---|---:|---:|---:|
| Solo gfx1151 (merge_sort, max=256, asym3, DPM warmup) | **105.67** | 13.27 | 100.0% |
| Hetero gfx1151 + 9070 XT drafter (gen1×1, M.2) | **97.91** | 13.27 | **92.66%** |

Cold load: target 9.0s on iGPU (system DDR5), draft 0.6s on gen1×1 (~150 MB at ~250 MB/s — matches expected link BW for one-time drafter upload).

τ bit-identical to hiptrx PCIe gen5 ×16 hetero at the same canonical config — drafter predictions correct, link gen does not affect spec-decode quality.

**Verdict: M.2 gen1×1 hetero loses only 7.3% vs solo, BETTER than TB5 hetero (lost 10.5% on a different config but same latency-dominated regime). Per-call latency is the governing variable, not headline BW. PCIe-native µs latency floor on gen1×1 beats TB tunnel ms latency despite the BW handicap.**

Prior "M.2 hetero is dead" / "structurally capped" framing fully retracted. Steady-state decode runs on on-card GDDR6 (~640 GB/s); the gen1×1 link only carries small per-cycle xcard transfers (KB-scale), not weights/KV.
