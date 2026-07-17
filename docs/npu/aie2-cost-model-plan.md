# AIE2/AIE2P NPU cost model — scope, design, build plan

**Status:** IMPLEMENTED — all 7 phases, with calibrated NPU1/AIE2 and
NPU2/AIE2P targets. Code in `tools/npu/aiecost/`.
Both §3 gates pass on an uncontaminated set (Kendall tau = +1.000, 6/6 within
±30%). See §10 for results, including two priors this plan got wrong.
**Targets:** XDNA / NPU1 / AIE2 (Phoenix, `RyzenAI-npu1`) on **nix1**, and
XDNA / NPU2 / AIE2P (Strix Halo, `NPU2`) on **halo**.
**Goal:** a Python meta-program that characterises the NPU, then predicts kernel
latency and overheads *ahead of writing the kernel*, well enough to rank
candidate schedules and identify the limiter.

This is a **best-estimate** tool. It is not a simulator and does not aim for high
absolute accuracy. It is expected to converge over several
predict → measure → refit iterations.

## 1. Why this is not a roofline model

`benchmarks/npu_gemm_tuning/findings.md` (rounds R0–R121, on **aie2p**)
repeatedly falsifies the FLOPs/bandwidth framing:

- **R64** — warm production wrapper 1.0292 ms = 241 µs device span + **76.6%
  host prep/submit/sync/deblock**. Output DMA starved 198 µs of the 241 µs.
- **R68** — collapsing three output objects into one padded object cut ~360 DMA
  tasks ~3× → **24% faster**, identical math.
- **R117** — N16 → N32 **doubled useful work and ran 9.8% faster**. Quoted:
  "direct evidence that dispatch/fixed traffic dominates this small consumer".
- **R56** — resident weights consumed at ~0.9 GB/s against a 56.5 GB/s feed roof
  (**1.6%**). Not bandwidth-bound.
- **tilesweep** — output tile area is the lever: 32²=2.4, 64²=8.2, 128×64=11.0,
  64×128=14.3, 128²=15.7 TOPS; L1-capped at area 16384.

A model whose core was arithmetic intensity would have mispredicted every one of
those. **The object of prediction is the overhead structure**: fixed dispatch
cost, DMA task count, feed/drain concurrency, and host wrapper cost.

## 2. What transfers from the aie2p corpus — and what does not

The 95 CSVs in `benchmarks/npu_gemm_tuning/results/` are **aie2p on halo**
(8 columns, 512-bit vector, 58 TOPS, Strix Halo memory). NPU1 differs on every
one of those axes. Therefore:

| Transfers (as prior) | Does **not** transfer (must be measured) |
|---|---|
| Which cost terms exist | Every constant's value |
| That fixed/dispatch cost dominates small consumers | Fixed cost magnitude |
| That output-tile area is the top lever | The L1 cap that sets tile area |
| That task count matters (R68/R119 `repeat_count`) | Per-task issue cost |
| That host wrapper can dominate device time | Wrapper/device split |
| Trace counters worth logging (`mean_receive_stalled`, `device_span_cycles`) | Feed roof, drain roof |

**The aie2p corpus tells us what to measure, and roughly what shape the answer
takes. It cannot calibrate AIE2 and is not a validation set for it.** This is the
single biggest change from the original aie2p-targeted draft: there is no
ready-made back-test corpus, so trust must be built by iteration instead.

### Known AIE2 vs AIE2P deltas that force re-measurement

- **Columns**: `Total Columns = 5` (1 shim + 4 compute); `NPU1().cols = 4` →
  **16 cores** vs halo's 32. Firmware `total_col=5` caps `column_width`.
- **Vector width**: AIE2 256-bit vs AIE2P 512-bit → different cycles/MMUL.
- **Peak**: ~16 TOPS vs 58.
- **Tile SRAM**: **not** a difference — **both are 64 KB/tile**. `NPU1` and
  `NPU2` both derive from `AIE2TargetModel` (`getLocalMemorySize() = 0x10000`).
  The skill's "32 KB for AIE2" is the AIE1 value and is wrong; see §5. So the
  aie2p tile-area findings (L1-capped at area 16384) **do carry over** — one of
  the few quantitative results that does.
- **Memory**: dual-channel LPDDR5 vs Strix Halo. Feed roof unknown.
- **tanh**: LUT (`getTanhBf16`) vs hardware. Affects core body cost only.

## 3. Scope

**Predicts**, for a candidate schedule, before the kernel exists:

1. device span,
2. end-to-end wrapper time,
3. the **dominant limiter**,
4. predicted **receive-stall fraction** — directly comparable to the trace
   counters, so the model's internal story is falsifiable, not just its total.

**Out of scope:**

- Cycle-accurate AIE VLIW simulation. Peano scheduling, software pipelining, and
  the `accum`/`auto` bypass hazards are not tractable to model. The core body is
  a **calibrated** cost, not a simulated one.
- Correctness prediction.
- The **context-transition zero-output** flakiness (R114–R120). That is platform
  nondeterminism. The harness must **detect and exclude** it — never fit to it,
  never average it in.
- Anything in the Rust inference hot path. Python throughout (AGENTS.md allows
  Python for tooling); the model informs kernel design, it does not run in serve.

### Accuracy gates

A best-estimate model still needs a falsification criterion, or it is decoration.

- **Ordinal (primary)**: correctly rank candidate schedules within a family.
  Ordinal accuracy *is* the product — every win in R0–R121 came from correctly
  ranking two candidates, not from a precise number.
- **Magnitude**: ±30% on device span inside the calibrated envelope. Loosened
  from the aie2p draft's ±25% because there is no back-test corpus to fit
  against.
- **Limiter classification** correct on held-out rounds.
- **Refuses to predict** outside the calibrated envelope rather than
  extrapolating.

## 4. Design

`ScheduleSpec` (tiles, DMA task counts, FIFO depths, bytes per role, MMUL counts,
core-text estimate) → `Prediction` (per-term breakdown + limiter + stall
fraction).

### Cost terms

| Term | Form | Calibrated by |
|---|---|---|
| `t_host` | `c_call + c_pack·B_in + c_deblock·B_out` | C7 (pure CPU, no NPU) |
| `t_submit` | `c_cmd + c_bo·n_bos` | C1 null-kernel dispatch sweep |
| `t_feed` | `B_wire / BW_eff(streams, cols)` | C2 feed-only stream |
| `t_task` | `c_issue · n_live_tasks` | C3 task-count sweep |
| `t_core` | `n_mmul·cyc/f_H + stage + align` | C4 compute-only, C8 alignment |
| `t_drain` | `B_out / BW_drain(channels)` | C5 drain-only |

Composition is an **overlap model with an explicit stall term, not a sum**:

```
T_device  ≈ fill + max(t_feed, t_core + t_stage, t_drain) + tail
T_wrapper ≈ t_host + t_submit + T_device      # overlap TBD, measure
```

The `max` is what lets the model reproduce R117-style results (more work, same
fixed cost, less time). A sum cannot.

`limiter = argmax(term)`; predicted stall fraction follows from the gap between
the max term and the others.

### Calibration suite — IRON-generated

Each microbench isolates **one** constant. All kernels generated from Python via
IRON (`ExternalFunction` + `transform_*`), parameterised and compiled per
`.agents/skills/npu-kernel-build/SKILL.md`, so the suite regenerates rather than
carrying hand-written variants.

| ID | Isolates | Bench |
|---|---|---|
| C0 | static facts | → **H-series, §5** (topology, L1, memtile, clock, arg slots) |
| C1 | `c_cmd`, `c_bo` | null/no-op kernel, sweep BO count → **fixed-cost floor** |
| C2 | `BW_eff` | feed-only stream, sweep streams × columns |
| C3 | `c_issue` | fixed payload, sweep DMA task count (+ `repeat_count`) |
| C4 | `cyc_mmul` | resident-data compute only, per dtype/shape |
| C5 | `BW_drain` | drain-only, sweep shim channels |
| C6 | fill/drain | sweep `fifo_depth` |
| C7 | `c_pack`, `c_deblock` | host-side only, no NPU |
| C8 | align penalty | local staging, aligned vs unaligned (cf. R118 64 B) |

The H-series and C1 come first: §5 sets the envelope and the time base, and C1
measures the floor that R64/R117 say dominates.

### The iteration loop (the actual program shape)

This is not build-then-use. The program *is* the loop:

```
predict(spec) → measure(spec via IRON) → residual → refit constants → report
```

Each pass emits a residual report: per-term error, which constant is least
constrained, and what to measure next. Expect several passes before the ordinal
gate holds. The residual report is the deliverable of each round, and it is what
picks the next round's bench.

### Layout

```
tools/npu/aiecost/
  device.py       # C0 probe + static facts
  spec.py         # ScheduleSpec / Prediction dataclasses
  model.py        # cost terms → breakdown + limiter
  benches/        # IRON-generated calibration kernels (C1–C8)
  calib/          # versioned constants JSON, keyed device+XRT+firmware
  refit.py        # measure → residual → refit
  report.py       # residual + envelope reporting
  # CLI: python -m aiecost {probe,calibrate,predict,refit,report}
benchmarks/npu_gemm_tuning/r122+/   # calibration rounds, existing convention
benchmarks/npu_gemm_tuning/results/ # durable CSVs, existing convention
```

Constants are **version-keyed on device + XRT + firmware** so drift is
detectable. The existing CSVs already record `xrt_version`, `firmware_version`,
`amdxdna_version`, `git_commit` — keep that schema.

## 5. Hardware and memory characterisation (H-series)

The cost model is only as good as its static facts, and the repo's existing
sources are **inconsistent, wrong-generation, or self-discrediting**. This
section confirms what we can and marks the rest as measured-or-unknown.

### Source tiers

Facts are ranked by the trust their source earns. This ordering did real work
during the audit — it resolved M1 and rehabilitated M5 without touching hardware.

1. **Probe on silicon** — highest. Settles behaviour.
2. **Toolchain target model** (`mlir_aie/include/aie/Dialect/AIE/IR/AIETargetModel.h`)
   — what the *compiler believes*. Critically, this **bounds what we can build
   regardless of silicon**: if mlir-aie thinks local memory is 64 KB, a 65 KB
   buffer will not compile even if the die has more. For capacity limits, the
   toolchain's belief is frequently the *operative* constraint.
3. **Vendor doc for the correct generation** — AM020 (AIE-ML) for AIE2.
4. **Vendor doc, wrong generation** — UG1079 (AIE1). **Not evidence about AIE2.**
5. **Self-inconsistent metadata** — `rocminfo`. Corroboration only, never alone.

### Audit findings

1. **M1 is resolved: L1 = 64 KB/tile on NPU1.** The toolchain arbitrates what the
   in-repo manual cannot. `AIETargetModel.h` defines `AIE1TargetModel` with
   `getLocalMemorySize() = 0x8000` (32 KB) and `getMemTileSize() = 0` (no
   memtiles), versus `AIE2TargetModel` with `getLocalMemorySize() = 0x10000`
   (**64 KB**) and `getMemTileSize() = 0x80000` (512 KB). And
   `class BaseNPU1TargetModel : public AIE2TargetModel` — **NPU1 inherits the
   AIE2 sizes**. Therefore:
   - `findings.md`'s "L1 is 64 KB/tile" is correct, **and applies to NPU1**, not
     just aie2p.
   - The npu-kernel-build skill's "32 KB SRAM per compute tile" is the **AIE1
     number** — it matches UG1079's text exactly, which is almost certainly where
     it came from. **The skill is wrong for AIE2 and should be corrected.**
   - UG1079's 32 KB / 16 KB program / 128 KB-with-neighbours are all AIE1 and
     carry no weight for this target.
2. **M5 is rehabilitated, not discredited.** `rocminfo` reports `L2: 2048 KB` on
   nix1 alongside junk fields (Cacheline Size 0, Max Clock 0, Compute Unit 0).
   But IRON reports NPU1 as `memtiles=4`, and 4 × 512 KB = **exactly 2 MiB**. Two
   independent sources agree: the L2 figure is the **memtile aggregate**. The
   surrounding junk fields remain junk; corroboration is what promoted this one.
3. **H2 is confirmed.** IRON reports NPU1 as `cols=4 rows=6 compute=16
   memtiles=4 shim=4` — 16 cores, matching the derivation from H1. The 6 rows are
   1 shim + 1 memtile + 4 core rows.
4. **npu1 exposes far less telemetry than halo.** `xrt-smi examine` on nix1
   returns Power Mode and Total Columns only; Estimated Power and Temperature
   are `N/A`, and there is **no `npu_clk_max` and no `npu_tops_max`** — both of
   which halo reports (1800 MHz / 58 TOPS) and which findings.md relied on.

Consequence of (3): **clock cannot be read, so `f_H` and `cyc_mmul` are
entangled** in `t_core`. They must be separated by timing a loop of known
instruction count, or from trace timestamps if the npu1 trace exposes a cycle
counter. Confirming that trace path is a phase-1 gate, because without it the
core term is only ever fitted as a product.

### Claims register

Every static fact is `claim → source → status → probe`. Nothing enters `device.py`
at status *assumed*.

| # | Claim | Source | Status | Probe |
|---|---|---|---|---|
| H1 | 5 total columns, 1 shim + 4 compute | `xrt-smi`, skill | **confirmed on nix1** | — |
| H2 | **16 cores**; rows = 1 shim + 1 memtile + 4 core | IRON `NPU1()` + H1 | **confirmed** (`compute=16`) | — |
| H3 | 16 KB program memory/tile | UG1079 (**AIE1**) | wrong-gen | AM020 (§6), then H3: grow core text to link failure; cf. aie2p observed ≤11,200 B |
| H4 | 5 DPU data-argument slots | findings R59 (aie2p) | likely transfers (command-packet ABI) | H4: sweep BO count to EINVAL |
| **M1** | **L1 data memory = 64 KB/tile** | **`AIE2TargetModel` = `0x10000`; NPU1 inherits it** | **resolved** — skill's 32 KB is the AIE1 value and is **wrong** | M1: confirm buildable ceiling by binary-search to compile failure |
| M2 | 8 banks × (256 w × 128 b) | UG1079 (AIE1) | wrong-gen | AM020 (§6), then M2: stride sweep → bank-conflict signature |
| M3 | 128 KB via 3 neighbours + own | UG1079 (AIE1) | wrong-gen | AM020 (§6), then M3: neighbour-access compile+run test |
| M4 | 3 concurrent ports if different banks | UG1079 (AIE1) | wrong-gen | AM020 (§6), then M4: concurrent load/store throughput |
| M5 | Memtile = 512 KB × 4 = **2 MiB** | `AIE2TargetModel` = `0x80000` × IRON `memtiles=4`; **`rocminfo` L2 agrees exactly** | **corroborated** (two independent sources) | M5: confirm via objectfifo capacity if it ever binds |
| B1 | Per-stream feed roof | halo: 14.4 GB/s | **does not transfer** | C2 |
| B2 | Aggregate feed roof | halo: 56.5 GB/s @8col | **does not transfer** | C2, ×4 cols |
| B3 | Drain roof / shim channel capacity | R61 (aie2p, qualitative) | unknown | C5 |
| K1 | AIE compute clock | **not reported by npu1** | unknown | K1: known-cycle-count loop |
| K2 | Peak ~16 TOPS | marketing | unverified | C4 (derives from K1) |
| K3 | Alignment penalty (64 B loads) | R118 (aie2p) | likely transfers | C8 |
| X1 | NPU is **not cache coherent** | amdxdna driver | documented | X1: explicit-sync omission test |
| X2 | No usable MALL path | R56 (aie2p/Strix Halo) | **N/A** — Phoenix has no MALL | none |

**M1 was the top priority and is now resolved at tier 2** — 64 KB/tile, from the
toolchain rather than from hardware time. It sets the output-tile area cap, and
output-tile area is the single biggest lever in the whole aie2p corpus (2.4 →
15.7 TOPS across the tilesweep, L1-capped at area 16384). Had we inherited the
skill's 32 KB, every tile-area prediction would have been wrong by 2× — the
error would have propagated into every schedule the model ranks. This is the
case for the register: **one afternoon of provenance work removed the plan's
largest single risk without touching the NPU.**

With M1 resolved, **K1 (clock) is now the top unknown**, because it is the time
base for the entire `t_core` term and npu1 reports no clock at all.

### Probe method

- **Capacity probes (M1, M3, M5, H3)** binary-search to a **failure boundary**
  — compile error, allocation failure, or `DRM_AMDXDNA_CREATE_HWCTX` EINVAL.
  Failure boundaries are exact and cheap; they beat inferring capacity from a
  performance knee. IRON-generated, parameterised, same as the C-series.
- **Behavioural probes (M2, M4, K3)** need a throughput signature and are only
  meaningful once K1 gives a time base.
- **K1 first among the behavioural set** — it is the time base for everything
  in `t_core`.
- Findings note `xrt-smi configure --pmode` needs `CAP_SYS_ADMIN` and sudo was
  password-gated on halo. If the same holds on nix1, **pmode is a fixed
  uncontrolled variable**: record it (`default`) with every row and do not claim
  clock-invariance across it.

### Deliverable

A `device.py` fact table where each entry carries `value, source, status,
probe_id, confidence`, plus a **provenance report** that renders the register
above with measured values filled in and conflicts resolved. Anything still
`unknown` narrows the model's declared envelope rather than getting a guess.

The register is also the honest answer to "what do our docs actually tell us".
After the source-tier audit:

- **Confirmed / resolved**: H1, H2 (silicon + toolchain), **M1** (toolchain), M5
  (two-source corroboration).
- **Likely transfers**: H4, K3.
- **Wrong generation, pending AM020**: H3, M2, M3, M4.
- **Do not transfer**: B1, B2 (halo-specific).
- **Unknown, must probe**: K1, K2, B3.

The audit moved four facts from unknown/disputed to confirmed at zero hardware
cost, and **corrected an in-repo doc bug** (the skill's 32 KB). The remaining
wrong-generation cluster is exactly what §6's external references target.

## 6. External references and network access

**Network access is permitted for this work**, specifically to AMD's GitHub
organisations and AMD/Xilinx documentation sites, to fetch reference material
when the in-repo sources are absent, wrong-generation, or disputed.

The audit in §5 is the justification: the single largest risk in the plan (M1)
was retired by reading a source we already had, and the remaining
wrong-generation cluster (H3, M2, M3, M4) is retirable the same way — by
obtaining the **right-generation** vendor manual instead of spending NPU time
inferring what AMD already documents.

### Sanctioned sources

| Source | Answers | Why |
|---|---|---|
| **AM020 — AI Engine-ML Architecture Manual** (docs.amd.com) | H3, M2, M3, M4 | The AIE-ML/AIE2 counterpart to UG1079. **The correct-generation manual for this target.** Highest-value fetch. |
| **`Xilinx/mlir-aie`** (GitHub) | toolchain belief, IRON semantics, `AIETargetModel` history | Upstream of the local install; useful when the installed wheel's behaviour is ambiguous or a constant looks stale. |
| **`amd/xdna-driver`** (GitHub) | H4 (arg slots), hwctx limits, coherency (X1) | The driver is authoritative on the command ABI and `CREATE_HWCTX` validation. |
| **`Xilinx/XRT`** (GitHub) | telemetry surface, why npu1 reports no clock (K1) | May explain the missing `npu_clk_max`, or reveal another path to it. |
| AMD Ryzen AI / XDNA product docs (amd.com) | K2 (peak TOPS), power modes | Marketing-grade; **tier 5**, corroboration only. |

### Rules

- **Fetched docs are tier 3, not ground truth.** UG1079 is precisely how we got
  into this mess: an authoritative manual for the wrong silicon. Any fetched
  claim enters the register with its **generation explicitly recorded**, and a
  vendor doc never outranks a probe or the toolchain on buildable limits.
- **Record provenance**: URL, document ID, version/revision, retrieval date, and
  SHA-256 of the retrieved artifact go into the register beside the claim. The
  repo already SHAs its payloads; docs get the same treatment.
- **Vendoring** follows the existing `docs/npu/ug1079-.../` precedent — extract
  to markdown under `docs/npu/`, keep it to the sections actually cited, and
  retain AMD's attribution. Do not bulk-import PDFs.
- **Offline-first**: nothing in the build may *require* network at run time.
  Fetches are a one-off research step whose output is a checked-in reference plus
  register entries. `aiecost` itself never reaches the network.
- **Public material only** — published manuals and public repos.

### Immediate fetch list

1. **AM020** — retires H3/M2/M3/M4 in one document. Do this first.
2. `amd/xdna-driver` — confirm H4's 5 arg slots on npu1 rather than assuming the
   aie2p ABI transfers.
3. `Xilinx/XRT` — determine whether K1 is obtainable at all on npu1, before
   committing to the known-cycle-count workaround.

## 7. Build phases

| Phase | Deliverable |
|---|---|
| 0 | Cost-term taxonomy + gates, formalised from findings.md. No code. |
| 1 | **H-series (§5)**: claims register + probes. **M1 already resolved (64 KB)**; settles **K1** (clock, since npu1 reports none), and whether the npu1 trace exposes cycles. Emits the provenance report. |
| 2 | C1 + C7 — the fixed-cost floor (dispatch + host). Highest prior value. |
| 3 | Model core: `ScheduleSpec` → breakdown + limiter. |
| 4 | C2–C6, C8 — remaining constants, IRON-generated. |
| 5 | Refit loop + residual reporting. Iterate to the ordinal gate. |
| 6 | Ordinal validation: reproduce a tilesweep-style ranking **on AIE2**. |
| 7 | MLIR extractor: compiled MLIR → `ScheduleSpec`, to audit predictions. |

Phase 2 precedes the model core deliberately: if the AIE2 fixed-cost floor is as
dominant as R64/R117 imply for aie2p, it constrains every later term, and a
schedule below that floor is unbuildable regardless of its dataflow.

Phase 6 replaces the original plan's back-test phase. Without an AIE2 corpus,
trust comes from a *fresh* ordinal sweep whose ranking the model must call
before the sweep runs.

## 8. Risks

- **No AIE2 back-test corpus.** The main one. Mitigation: phase 6 predicts a
  sweep's ranking *before* running it; the model commits first.
- **Context-transition zeros** (R114–R120): fresh-process repetition + parity
  gating; excluded runs reported, never silently dropped.
- **Python host runtime**: `benchmarks/npu_gemm_tuning/README.md` records the
  mlir-aie Python harness segfaulting under Python 3.14 on halo, forcing a
  compiled C++ host. nix1's venv is Python 3.12 and the skill documents
  `XRTHostRuntime` working — but confirm early. C++ host is the fallback.
- **Calibration drift**: version-keyed constants; refuse to predict on a
  device/XRT/firmware key that has no calibration.
- **Overfitting to one schedule family**: hold out whole rounds, not rows.
- **NPU1 is small** (16 cores, ~16 TOPS): absolute headroom is narrow, so fixed
  overheads matter proportionally *more* than on aie2p. Expect the floor to
  dominate more, not less.

## 9. Open decisions

- Confirmed: predict **end-to-end wrapper time**, not device-only. R64 puts
  device at 23.4% of the wrapper on aie2p; a device-only model would predict the
  small half. AIE2's split is unknown and is itself an early measurement (C1+C7).
- Resolved 2026-07-17: npu2/aie2p is a second calibration target. Target
  selection is explicit (`--device auto|npu1|npu2`) and calibration remains
  version-keyed, so constants cannot silently cross NPU generations.

## 10. Results (implemented 2026-07-16)

All seven phases are built in `tools/npu/aiecost/`. Constants are version-keyed
to `RyzenAI-npu1_xrt2.25.0_fw1.5.5.391` and every one carries its evidence and
caveats (`python -m aiecost calib`).

### Gates

Phase 6 ran commit-first on **family B**, which had never informed any fit:

- **Ordinal: Kendall tau = +1.000** (gate ≥0.8) — PASS
- **Magnitude: 6/6 within ±30%**, worst −12.5%, errors mixed-sign — PASS

Family A is **burned**: its residuals were used to fix the model (below), so its
tau is contaminated and it is kept for regression only.

### Measured constants

| Bench | Result |
|---|---|
| **K1** | f_H = **1.015 GHz**. Plateau at 1.015/1.019/1.013 G VMAC/s (chains 2/4/8) proves saturation; disassembly (16 bundles / 16 vmac, unrolled 4×) proves II=1. |
| **C1** | Dispatch floor = **~155 µs of device time for a null kernel**. `c_bo` below noise (sign flipped between runs). |
| **C7** | c_pack 71.7 GB/s, c_deblock 89.1 GB/s, c_call 7.72 µs. **BO alloc = 17.6 ms, size-independent — setup only.** |
| **C2** | Feed **4.03 GB/s/column**, near-linear to 16.1 GB/s at 4 columns. |
| **C5** | Drain **3.98 GB/s/column** — feed and drain are **symmetric** on npu1. |
| **C3** | Task issue **≈5 ns, below noise**. |
| **C6** | FIFO depth 1→4 buys **2.2%**. |

### Priors from the aie2p corpus that did NOT survive

1. **"Task count is the lever" is false on npu1.** R68 cut tasks ~3× on halo for
   24%; here an **8× cut buys ~1.5%**. npu1 feeds at ~4 GB/s/column, so a 2 KiB
   tile takes ~553 ns to move while task issue costs ~5 ns — the overhead hides
   entirely behind the transfer. halo moves bytes ~3.5× faster per column, which
   is what made per-task cost visible there. **Structure transferred; the ranking
   of levers did not.**
2. **Drain is not the weak side.** R64 found the output DMA starved 198 µs of a
   241 µs span; on npu1 drain ≈ feed (15.9 vs 16.1 GB/s).

What *did* transfer is the headline: fixed cost dominates. Device time is
**17.5%** of a predicted wrapper for a QKV-shaped schedule, against R64's 23.4%
on aie2p — different silicon, same story.

### A model bug the refit loop caught

The first phase-6 run passed ordinally (tau=+0.867) but every residual was
negative with a near-constant ~160–260 µs gap. That signature — uniform absolute
offset rather than uniform percentage — meant a **missing fixed term**, not a bad
rate: C1's floor is measured with `npu_time` and is therefore *device* time, but
the model filed it under `t_submit` (wrapper-only). Moving it into `device_s`
took the errors from −0.3…−30.9% to −0.4…−9.0%.

`refit.systematic()` now detects this class of error across residuals, because
the per-point diagnosis had confidently blamed the wrong thing (C2's bandwidth).

### C4 (done) — the "2× on the table" lead was WRONG, and is now closed

The lead was: K1 implies `mmul<4,8,8>` (256 MACs/VMAC) gives 8.3 TOPS while the
~16 TOPS claim needs 512, and aie2p used 8×8×8 — so maybe a wider native shape
exists. **It does not.** `aie::mmul<8,8,8>` compiles on AIE2 but is a **virtual**
instruction: the API decomposes it into two native 256-MAC VMACs.

Two independent checks agree (`benches/c4_mmul.py`):

| shape | vmac/call (ISA) | MACs/vmac | ns/iter (HW) | G MACs/s | verdict |
|---|---|---|---|---|---|
| `<4,8,8>` | 1.00 | **256** | 3.819 | 268.2 | NATIVE, fills the VMAC |
| `<8,8,8>` | 2.00 | 256 | **7.937** | 258.1 | VIRTUAL — 2× the time, same MACs/s |
| `<4,16,8>` | 2.00 | 256 | 7.859 | 260.6 | VIRTUAL |
| `<2,8,8>` | 1.00 | 128 | 3.943 | 129.9 | native but half-fills the VMAC |
| `<4,8,16>` | — | — | — | — | does not compile (static assert) |

Disassembly counts `vmac` in the unrolled chain loop; hardware measures the ITERS
slope. They agree: **every full-width shape lands at ~256 MACs/cycle/core**, and
`<8,8,8>` costs exactly 2× per call for 2× the MACs. A wider `aie::mmul` is
source-level sugar.

**"Virtual" is throughput-neutral, not harmful** — a distinction worth stating
because it is easy to get backwards. `<4,16,8>`=520.09, `<4,32,8>`=524.98,
`<8,16,8>`=522.82 G MACs/s: a virtual shape costs N× the issue slots and returns
N× the MACs. Its real cost is **accumulator registers** (`C_block` holds N
accums), which leaves fewer registers for the independent chains that hide VMAC
latency — exactly why R0 found 2×2 mmul optimal. The genuinely lossy case is
**under-filling**: `<2,8,8>` gets 128 of 256 MACs per issue and throws away half.

**Consequences:**

1. **`mmul<4,8,8>` is already optimal for int8×int8 on AIE2.** No 2× is available
   *within that dtype pair*. `cyc_per_vmac=1.0` is now measured, not assumed.
2. This is why aie2p's 8×8×8 does not carry over: on AIE2P that shape is native
   (512 MACs/VMAC); on AIE2 it is emulated.

Method note: the `C_block<..., N>` template parameter in `mmul_8_4.hpp` looks
like a native-op count and mostly is, but it says `1` for `<4,32,8>` while the
disassembly shows 2 vmac/call. **The ISA is the truth; the header is a hint.**

### The ~16 TOPS figure resolved: it is int8×int4, not dense int8

Dense int8×int8 peaks at 8.31 TOPS, half the marketing number. Two candidate
explanations were tested and only one survived.

**Clock — ruled out.** Unlike halo (password-gated sudo), this box has
passwordless sudo, so `xrt-smi configure --pmode turbo` was testable for the
first time. K1 re-run under turbo: **988.5 MHz vs default's 1019.3 MHz** — a
no-op, 3% slower and within noise. This matches halo's finding exactly (turbo
15.21 vs 15.7 TOPS: "compute clock already maxed under load"). pmode restored to
`default` afterwards.

**Operand width — confirmed.** `aie::mmul` has a separate `mmul_8_4` family for
int8×int4 with its **own shape set** (`4×16×8`, `8×16×8`, `4×32×8` — the int8
shapes are invalid for it, which is why a first probe using `<4,8,8,int8,int4>`
wrongly looked like "int4 unsupported").

| pair | ceiling | optimal shape | HW MACs/cycle | peak |
|---|---|---|---|---|
| int8 × int8 | 256 MACs/VMAC | `<4,8,8>` | 254.9 | 8.31 TOPS |
| **int8 × int4** | **512 MACs/VMAC** | **`<4,16,8>`** | **512.4** | **16.63 TOPS** |

`<4,16,8,int8,int4>` is native — 1 vmac/call for 512 MACs — and hardware confirms
520.09 G MACs/s = 512.4 MACs/cycle, exactly 2× int8×int8. **VMAC/s stays
~1.01–1.03 G across every shape and every dtype**: the issue rate is pinned at
f_H and only MACs-per-VMAC changes. 16 × 512 × 2 × 1.015 GHz = **16.63 TOPS**,
matching the ~16 TOPS claim.

**This matters for hipfire directly**: OQ4/MQ4 are 4-bit weight formats, so the
quant path is exactly the int8×int4 case and gets the 2×. The dense-int8 path
does not. Any plan that assumes 16 TOPS for int8×int8 is 2× optimistic; any plan
that assumes 8.31 for the OQ4 path leaves half the machine unused.

### Not done

- **C8** (alignment penalty) — only affects specs with `aligned_loads=False`.
- **AM020** — §6's fetch list is untouched; H3/M2/M3/M4 remain wrong-generation.
- **int4×int4, bf16, sparsity** — unmeasured; `mmul_bf16_bf16.hpp` and
  `mmul_16_8/16_16/8_16` families exist and would extend the table.

### Model corrections applied

`t_core` assumed 256 MACs/VMAC for everything, which would have mispredicted
every OQ4/MQ4 kernel by 2×. Fixed:

- `ScheduleSpec` now carries `dtype_a`/`dtype_b`, and counts **source-level
  `mmul_calls_per_core`** rather than native VMACs. The old field made the
  *caller* responsible for knowing that `<8,8,8>` costs two VMACs — getting that
  wrong was a silent 2×.
- `model.py` derives `native_vmacs_per_call = ceil(macs_per_call / ceiling)`
  where the ceiling comes from C4 keyed by operand-type pair. That one line
  reproduces every measured C4 row, including `<2,8,8>`'s `ceil(0.5)=1`.
- Predictions now carry **actionable advice**: virtual shapes (with the correct
  neutral framing), under-filled VMACs (real loss), and the int4 2× when a spec
  uses int8×int8 — qualified with "only if `t_core` is the limiter".

Verified: same total MACs, `t_core` 3.027 µs (int8×int8 `<4,8,8>`) → 1.513 µs
(OQ4 `<4,16,8>`), exactly half.

**The honest caveat the model itself surfaces**: on the QKV-shaped schedule
above, int4 changes device time by *nothing* — 191.6 µs either way — because
`t_submit` (155 µs) and `t_feed` (36.6 µs) dominate and `t_core` is ~1.5–3 µs.
The 2× is real but invisible unless a schedule is actually compute-bound. That is
the same lesson as R117, arriving from the other direction.

## 11. NPU2/AIE2P extension (implemented 2026-07-17)

The same harness now resolves IRON's `NPU2` target and the AIE2P runtime library,
uses target-specific compiler flags and cache tags, and records the detected
topology in the calibration key. On this Strix Halo host that is 8 compute
columns / 32 cores, 64 KiB L1 per tile, eight 512 KiB memory tiles, and 16 BDs
and locks per core. The saved calibration is
`tools/npu/aiecost/calib/NPUStrixHalo_xrt2.25.0_fw1.1.2.65.json`.

| Bench | NPU2/AIE2P result |
|---|---|
| **C1** | `c_cmd` **72.55 µs**, `c_bo` **5.51 µs**. |
| **C7** | Pack **63.78 GB/s**, deblock **129.76 GB/s**, `c_call` **6.92 µs**; BO allocation is a **6.58 ms setup-only** cost. |
| **C2** | Feed **12.50 GB/s/column**, **50.01 GB/s aggregate at 4 columns**. |
| **C5** | Drain **9.69 GB/s/column**, **48.46 GB/s aggregate at 5 columns**. |
| **C3** | Task-issue slope is below measurement noise and is stored as zero. |
| **C6** | Depth 2 was best in the refresh sweep: **414.39 µs** versus **463.20 µs** at depth 1 (1.118×). The winner is noisy across runs, so treat this as a schedule hint, not a universal optimum. |
| **K1** | Inadmissible on AIE2P. The independent-chain probe was extended to 16 chains and driven with the small MR=4 accumulator, but the iteration floor is constant (~4 ns for chains 1/2/4 — pure latency-hiding, throughput just scales with chain count) and the register file spills before the pipe saturates, so no II=1 plateau is reachable in this kernel family. The admissible route is the disassembly VMAC-count method C4 used on npu1 (count `vmac` bundles in the loop; a native shape emits one per `mac()`), pending an aie2p-native int8 shape. |

C2 exercises at most four columns because its source plus one output BO per
column reaches the five-data-argument DPU ABI limit; C5 has no source BO and
therefore reaches five drain columns. Their model constants use a measured
per-column rate and scale to the requested active-column count; the 8-column
extrapolation must be validated before it is used as an admission claim. All
raw probe records are under
`benchmarks/npu_gemm_tuning/results/aiecost-aie2p-*-20260717.json`.
