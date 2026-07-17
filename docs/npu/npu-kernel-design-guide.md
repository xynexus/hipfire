# NPU kernel design guide — AIE2 (npu1) and AIE2P (npu2)

Everything measured about these NPUs, and how to measure it without fooling
yourself. Companion to `tools/npu/aiecost/` (the metrics tool) and
`docs/npu/aie2-cost-model-plan.md` (the cost model's design and validation).

**Read part 2 before running a benchmark.** Every number in part 1 that was
initially wrong was wrong because of something in part 2 — and *most* of the
numbers here were initially wrong.

**Two generations, kept separate on purpose.** Parts 1–4 are **AIE2 / npu1 /
Phoenix**, measured on nix1 (16 cores, 1.015 GHz). Part 5 is **AIE2P / npu2 /
Strix Halo**, measured on halo (32 cores, ~1.68 GHz) — it has its own measured
values *and* its own conclusions, several of which point the opposite way. Do
not read a npu1 number as an aie2p number; the vector width, core count, clock,
and even the int4 and task-count levers differ. Calibration constants live in
`tools/npu/aiecost/calib/`, version-keyed to device+XRT+firmware (npu1 =
`RyzenAI-npu1_*`, npu2 = `NPUStrixHalo_*`); run `python -m aiecost calib
--device {npu1,npu2}` for the live set with evidence and caveats.

---

## Part 1 — What the hardware is

### Topology and limits (all confirmed)

| fact | value | source |
|---|---|---|
| compute columns | 4 (of 5 total; 1 is shim) | `xrt-smi`, toolchain |
| cores | **16** (4 cols × 4 rows, rows 2–5) | toolchain, verified by placement |
| L1 per core tile | **64 KB** | `AIE2TargetModel::getLocalMemorySize` = 0x10000 |
| memtile | 512 KB/column, 2 MiB total | toolchain; `rocminfo` L2 agrees |
| BDs per core | 16 | toolchain |
| locks per core | 16 | toolchain |
| DPU data-arg slots | **5** | R59 (aie2p); binds constantly in practice |
| AIE clock (f_H) | **1.015 GHz** | K1; **pmode-invariant** |

**L1 is 64 KB, not 32 KB.** 32 KB is the AIE1/Versal value that UG1079
documents — and UG1079 is vendored in this repo, which is how it got into the
npu-kernel-build skill. `xcvc1902` (Versal) reports 32 KB and *zero* memtiles;
npu1 and npu2 both derive from `AIE2TargetModel` and report 64 KB / 512 KB.
**UG1079 is the wrong generation for anything XDNA.**

**Turbo does nothing.** `xrt-smi configure --pmode turbo` → 988.5 MHz vs
default's 1019.3 MHz: a no-op within noise. Matches halo's finding ("compute
clock already maxed under load"). Passwordless sudo works on nix1, so this is
testable here even though it wasn't on halo.

### Compute: MMUL shapes and rates

| operand pair | native shape | MACs/native VMAC | peak |
|---|---|---|---|
| int8 × int8 | `mmul<4,8,8>` | 256 | 8.3 TOPS |
| **int8 × int4** | **`mmul<4,16,8>`** | **512** | **16.6 TOPS** |

- **Measured at 16 cores: 3.72 T MACs/s = 7.4 TOPS int8 — 89% of theoretical.**
- The VMAC issue rate is pinned at f_H: ~1.01–1.03 G VMAC/s across *every* shape
  and dtype. Only MACs-per-VMAC changes.
- **`mmul<8,8,8>` is VIRTUAL on AIE2** — it emits 2 native VMACs for 2× the
  MACs. Throughput-neutral, *not* harmful, but it burns 2 accumulator registers
  per call, leaving fewer for the independent chains that hide VMAC latency.
- **Under-filling is the lossy case**: `mmul<2,8,8>` gets 128 of 256 MACs per
  issue and throws away half.
- `int8×int4` has its **own shape family** (`mmul_8_4`: 4×16×8, 8×16×8, 4×32×8).
  The int8 shapes are invalid for it — probing with `<4,8,8,int8,int4>` fails and
  looks like "int4 unsupported". It isn't.
- The ~16 TOPS marketing figure is **int8×int4**, not dense int8×int8.

**Rule: use `<4,8,8>` for int8, `<4,16,8>` for int4. Both native, both fill the
VMAC.**

### Bandwidth and the dispatch floor

| quantity | value |
|---|---|
| feed | **~3.95 GB/s per DMA STREAM (fifo)** — linear in stream count |
| feed roof | **≥30.8 GB/s at 8 streams**, still scaling; true ceiling unmeasured |
| drain | ~symmetric with feed |
| **dispatch floor** | **~155 µs of DEVICE time for a null kernel** |
| host per dispatch | 7.7 µs fixed + pack 71.7 GB/s + deblock 89.1 GB/s |
| BO allocation | **17.6 ms**, size-independent — **setup only, never per dispatch** |
| DMA task issue | ~5 ns — **below the noise floor** |
| FIFO depth 1→4 | 2.2% |

**Bandwidth is per STREAM, not per column.** An earlier "4.03 GB/s/column →
16.1 GB/s at 4 columns" was a mislabel: `c2_feed` builds one fifo per column, so
"4 columns" meant *4 streams*. Measured:

    1 column, 1/2/4 fifos:  3.972 / 8.071 / 15.803 GB/s   (span FLAT: 2267/2234/2278 us)
    4 columns, 4/8 streams:      15.40 / 30.79 GB/s

**One column with 4 fifos matches what was reported as the whole device's roof.**
More fifos per column is a real and large lever. 16 streams fails placement, so
the true ceiling is still unknown.

**The 155 µs floor dominates everything.** It is 58% of the best 256×768×1280
GEMM candidate and 37% of a kvarn4 decode token (32 layers × 155 µs = 4.96 ms of
a 13.57 ms token). **Work-per-dispatch is the top lever on this device** — bigger
than any format or tiling choice.

### Energy

**There is no NPU power sensor.** `xrt-smi` reports `Estimated Power : N/A`.
Energy is only observable as a **package delta** on shared APU rails (CPU+GPU+NPU),
and must never be reported as "NPU power".

| quantity | value |
|---|---|
| J/byte (external) | 1.9508e-10 → **5.13 GB/J** |
| J/MAC int8 (16 cores) | 1.0659e-12 → **938 G MACs/J** |
| J/MAC int4 (16 cores) | 4.8671e-13 → **2055 G MACs/J** (2.19× int8) |
| **break-even AI** | **183 MACs/byte (int8), 401 (int4)** |

**On npu1, energy is bytes.** The array is efficient enough relative to DDR that
a kernel needs arithmetic intensity above 183 before arithmetic dominates its
energy at all — and the LLM hot path never gets close:

| phase | AI | energy set by |
|---|---|---|
| prefill (M=256) | 82–101 | **movement** |
| decode (M=1) | 1.0 | **movement** (183×) |

So **int4's 2× MMUL rate is mostly a *speed* lever**; its energy value comes from
halving weight bytes, not from the faster MMUL.

### NPU vs GPU (gfx1103, same die, same rails, same instrument)

| axis | NPU | GPU | verdict |
|---|---|---|---|
| movement | **5.13 GB/J** | 3.73 GB/J | **NPU 1.37×** |
| compute int8 | **938 G MACs/J** | 432 G MACs/J | **NPU 2.15×** |

**On tok/J the NPU wins both regimes.** This is a **tok/J verdict only** — the
NPU's 7.4 TOPS ceiling is far below the GPU's, so a latency-bound prefill may
still belong on the GPU. That comparison is unmeasured.

---

## Part 2 — How to benchmark this NPU without fooling yourself

Every rule below cost a wrong result first.

### 2.1 THE BIG ONE: never bench the NPU under-parallelised

**A 1-core bench is not an NPU bench.** The array is 16 cores, and fixed costs
amortise across them. Measured:

    1 core:   0.487 W → 169 G MACs/J
    16 cores: 1.4 W   → 938 G MACs/J    (2× the power, 16× the MACs)

**The 1-core figure is 5.5× pessimistic.** It produced three wrong conclusions in
one session:

1. "GPU is 2.29× more energy-efficient than the NPU at compute" — actually the
   NPU is 2.15× *better*. The bench was 12 GPU CUs vs **1 NPU core**.
2. "The 16-core NPU extrapolates to 551.6 G MACs/J" — measured 929.
3. **"One byte costs 37 MACs"** — the design law itself. It divided J/byte by the
   1-core J/MAC. The real figure is 183, which *reclassifies prefill* from
   compute-dominated to movement-dominated.

**Check the worker count before trusting any per-core number.** `c4_mmul` built
one `Worker`; `c2_feed` built one per column. Neither is obvious from the output.

**The same bug bit the feed roof, in units.** `c2_feed`'s one-fifo-per-column
shape made a per-STREAM rate look like a per-COLUMN law — "16.1 GB/s at 4
columns" was really 4 streams, and one column with 4 fifos reaches the same
15.8 GB/s. The constant described the *bench topology*, not the device. Note the
aie2p corpus said "one active receive stream is 14.4 GB/s" — it had the units
right; the per-column framing was introduced here.

**Generalised rule: a constant measured on one topology is a property of that
topology until proven otherwise.** Before naming a constant "per X", vary X
*independently* of the bench's incidental structure.

**Reaching all 16 cores** (the 5-BO arg cap is what stops you):
- **inputs broadcast**: one fifo whose `cons()` every worker shares (an
  objectfifo's consumer list is native broadcast). Input DMA stays at 2 tasks
  regardless of core count — R61 found ~32 independent shim transfers exceed
  shim DMA capacity.
- **outputs join per column** via `ObjectFifo.prod().join()`, at that column's
  memtile. A single join across all 16 cores is **unplaceable** ("Failed to find
  a tile matching column 0") — a memtile serves only its own column.
- → 1 shared source + 4 column outputs = **5 BOs for all 16 cores**.

### 2.2 Energy: match the dispatch rate, or measure the CPU

**Free-running, package power tracks dispatch RATE, not work:**

    kernel    dispatch/s   package delta
    null           4326        10.343 W    <- doing NOTHING
    feed            352         4.251 W
    compute         133         1.082 W

A null kernel burns 10× the compute kernel because the host submit loop spins and
RAPL is package-wide. Free-run even **inverts** the compute-vs-feed ordering
(free-run says feed ≫ compute; matched-rate says compute > feed).

**Rules:**
- Pad every kernel to the **same** dispatch/s, then subtract a **null** baseline
  so host cost cancels by construction.
- **Verify the rate was met.** A kernel that overruns its slot silently runs
  slower; dividing by the *target* rate inflates per-launch energy AND breaks the
  null subtraction. `g1_gpu` once ran 311/500 launches at 30.8/s against a 50/s
  target. The harnesses now refuse such results.
- The load must be **sustained**. A first attempt ran ~2.6 s of dispatches then
  sampled a 6 s window that had already gone idle, measuring 0.02 W of nothing.

### 2.3 Instruments

- **RAPL** `/sys/class/powercap/intel-rapl:0/energy_uj` — package-0,
  **accumulating microjoules**, needs root (passwordless sudo here). *Integrate
  the counter; never sample a wattage and multiply.*
- **PPT** (amdgpu hwmon `power1_average`) — **UNUSABLE**. Read 11.0 W idle and
  5.5 W under load: *lower while working*. Rolling average, unclear window.
- **No NPU rail exists.** Every figure is a package delta with the host mixed in.

### 2.4 Noise floor

Marginals below ~1 W sit inside the ~0.3 W idle drift and go **non-monotonic**
(4 cores once read *lower* than 2). The 16-core delta (~1.4 W) clears it and
reproduces to ±6%. **Size the workload so the delta clears the drift**, and
repeat: 3+ runs, report median and range.

### 2.5 Use slopes; fixed cost cancels

Sweep a parameter and fit — the 155 µs floor lands in the intercept, not the
rate. K1 gets f_H from the ITERS slope; C2/C5 get bandwidth from the bytes slope.
An absolute measurement of anything small is mostly floor.

### 2.6 Do not fit across a topology change

The core-power fit `W = 0.358 + 0.1261·cores` was excellent (±0.6% over 1–4
cores) — and its extrapolation to 16 was **1.7× wrong**, because reaching 16
cores *required* changing the topology to broadcast+join, which costs far less
DMA. The fit was fine; the extension was not. **A fit is only valid inside the
topology it was taken on.**

### 2.7 The ISA is the truth

- The `C_block<..., N>` template parameter in `mmul_8_4.hpp` looks like a
  native-op count and mostly is — but says `1` for `<4,32,8>` where the
  disassembly shows 2 vmac/call. **Count `vmac` in the disassembled loop.**
- Verify against hardware anyway: ISA count and measured MACs/s must agree. They
  did for every C4 row.
- Watch the unroll factor (`add rN, rN, #-0x4` ⇒ 4×) — bundles-per-source-iter is
  not what you'd assume.

### 2.8 Sources, ranked

1. **Probe on silicon** — settles behaviour.
2. **Toolchain target model** (`AIETargetModel.h`) — what the compiler believes.
   For capacity limits this is often the *operative* constraint: if mlir-aie
   thinks L1 is 64 KB, 65 KB won't compile whatever the die does.
3. **Vendor doc, correct generation** (AM020 for AIE-ML).
4. **Vendor doc, wrong generation** (UG1079/AIE1) — **not evidence**.
5. **Self-inconsistent metadata** (`rocminfo`: reports L2 alongside Cacheline 0,
   Max Clock 0, CU 0) — corroboration only, never alone.

### 2.9 Platform hazards

- **`@jit` is broken here**: aiecc asserts `targetModel.hasProperty(IsNPU)`
  during CDO generation even for a correct `aie.device(npu1)`. Use the
  skill-documented `compile_mlir_module` + `XRTHostRuntime` path.
  `r0/r0b_run.py` is halo-era and will not run.
- **Context-transition zero output** (R114–R120): fresh contexts sometimes return
  all zeros. Platform nondeterminism — detect and **exclude**, never average in.
- **Cold vs warm**: R63 saw 3.5–4.2 ms cold vs 1.03 ms warm. Warm up, or use
  slopes.
- **`sdot4` doesn't exist on gfx1103** (no `dot1-insts`). The GPU's int8 path is
  `v_wmma_i32_16x16x16_iu8` — what hipfire's own kernels use.
- **py311 f-strings**: nested same-quotes needs 3.12; pyproject targets py311.
- **Duplicated constants across the .hip/.py boundary** silently desync: a
  `constexpr COMPUTE_ITERS` in the kernel and a Python copy for MAC accounting
  drifted apart and produced a 1.6×-wrong number.
- Large fifo counts can **crash XRT** (an 8-fifo/4-column feed variant dumped
  core).
- **There is one NPU, on a shared package.** Two hazards follow. (a) *Serialize*:
  every NPU bench must hold the `hipfire lock` — there is a single hardware queue,
  so a second NPU process (a parallel agent, another bench) contends and skews
  timing. (b) *Isolate*: RAPL/PPT are package-wide, so **any** concurrent CPU/GPU
  load lands in an energy delta. A `graphify` rebuild fired by a commit hook, or a
  second agent's energy run, adds several watts — a compute kernel that should
  read ~1 W read ~6 W with a rebuild running. Check `ps`/lock and quiesce the box
  before an energy window; a single spinning CPU core is **+17 W** of RAPL (§2.3),
  which dwarfs the whole NPU signal.

### 2.10 Retire superseded constants

Five constants in the calibration set are `admissible=False` because they are
artifacts or superseded (`byte_mac_energy_ratio`=36.7, `j_per_mac_int8` 1-core,
`j_per_mac_int8_4core`, `npu_core_power_fit`, and a redundant duplicate). **A
stale constant left admissible is a trap** — the model would silently fall back
to the 37-MACs/byte artifact for any dtype pair without a specific ratio.

---

## Part 3 — Design rules that follow from the numbers

1. **Maximise work per dispatch.** The 155 µs floor is 37–58% of realistic
   kernels. Fusing layers into one dispatch beats any format choice.
2. **Energy is bytes** (AI break-even 183). Narrower weights/KV cut energy; a
   faster MMUL mostly does not.
3. **Speed and energy pick different schedules.** M=256 QKV: `c4` is
   speed-optimal (265.2 µs, 1.940 mJ), `c1` is energy-optimal (484.3 µs,
   1.825 mJ). Broadcasting activations across 4 columns replicates bytes that
   cost 183 MACs each. *You cannot optimise both at once.*
4. **Use native, VMAC-filling MMUL shapes** (part 1). Wider is sugar; narrower
   wastes half the issue.
5. **Task count is NOT a lever on npu1** — an 8× cut buys ~1.5%. This
   *contradicts* aie2p (R68: 3× cut → 24%). npu1 feeds at ~4 GB/s/column, so a
   2 KiB tile takes ~553 ns to move while task issue costs ~5 ns: the overhead
   hides entirely behind the transfer.
6. **Watch the output drain.** For int4 GEMM the int32 accumulator output becomes
   the limiter once weights halve. Emit BF16 — which is what R65 did by hand on
   aie2p.
7. **KV bit-width is the decode lever**: kvarn8→4 gives 1.61× (not 2× — the floor
   doesn't shrink), kvarn2 2.33×.
8. **Allocate BOs once.** 17.6 ms each, size-independent.

### Buildability — check before designing

| limit | value | what it kills |
|---|---|---|
| L1 | 64 KB/core | tile × fifo depth (32 KiB tile × depth 2 = exactly L1 → fails) |
| BDs | 16/core | DMA task budget (R118/R119) |
| locks | 16/core | R61 exhausted these at FIFO depth 1 |
| DPU args | **5 BOs** | binds constantly; forces broadcast+join |
| DMA channels | ~2 in/core tile | `nargs≥3` fails "Resource allocation pipeline failed" |
| columns | 4 | — |

`python -m aiecost predict spec.json` checks all of these and rejects with the
specific limit cited.

---

## Part 4 — The metrics tool

    python -m aiecost probe              # H-series claims register + provenance
    python -m aiecost calib              # live constants, with evidence + caveats
    python -m aiecost predict spec.json  # breakdown, limiter, energy, advice
    python -m aiecost.design gemm --m 256 --k 768 --n 1280 --weight-bits 4
    python -m aiecost.design kv-sweep --context 4096
    python -m aiecost.validate --family c    # commit-first ordinal validation
    python -m aiecost.benches.e1_energy --kernel compute --cores 16 --rate 50
    python -m aiecost.benches.g1_gpu --all --rate 50

The model **refuses to predict** on missing constants and **rejects unbuildable
schedules**, citing the limit. It is a best-estimate tool: ±30% on device span,
validated commit-first at Kendall τ=+1.000 in both the transport-bound and
compute-bound regimes.

---

## Part 5 — AIE2P (npu2 / Strix Halo): measured, and where it diverges

Both are `AIE2TargetModel` (64 KB L1, 512 KB memtile), but AIE2P is a wider,
faster, differently-shaped machine and several *conclusions* flip. Measured on
halo; constants keyed `NPUStrixHalo_xrt2.25.0_fw1.1.2.65`.

### Topology, clock, bandwidth, dispatch

| fact | AIE2 (npu1) | AIE2P (npu2) |
|---|---|---|
| columns / cores | 4 / 16 | **8 / 32** |
| vector width | 256-bit | **512-bit** |
| memtiles / shim | 4 / 4 | **8 / 8** |
| L1 / memtile | 64 KB / 512 KB | same (4 MiB memtile agg) |
| BDs / locks per core | 16 / 16 | same |
| f_H | 1.015 GHz | **~1.68 GHz** (1.676 from K1; the II=1 int4 loop implies ~1.77) |
| feed roof | 16.1 GB/s @4col | **50.0 GB/s @4col** (12.5/col) |
| drain roof | 15.9 GB/s | **48.5 GB/s @5col** (9.69/col) |
| dispatch floor `c_cmd` | ~155 µs device | **72.6 µs** (`c_bo` 5.5 µs) |
| host `c_call` / pack / deblock | 7.7 µs / 71.7 / 89.1 GB/s | 6.9 µs / 63.8 / 129.8 GB/s |

**f_H comes from the ISA bundle count, not a plateau.** K1's throughput-plateau
method is DEAD on AIE2P: the accumulator register file spills at 8 independent
chains *before* the VMAC pipe saturates, so II=1 is never reached (constant ~4 ns
iteration floor for 1/2/4 chains, then collapse at 8). Instead, the AIE core is
statically-scheduled VLIW so loop bundles ARE cycles: the `mmul<4,8,8>` int8 loop
is `add` + `jnz` + a 5-slot branch shadow = 7 bundles/iter, giving f_H = 7 cyc ÷
(ns/iter) = 1.68 GHz — just under the 1800 MHz nominal.

### Compute: shapes, rates, and the int4 question

| operand pair | native shape | MACs / native VMAC | peak |
|---|---|---|---|
| int8 × int8 | `mmul<8,8,8>` | **512** | ~55–58 TOPS |
| int8 × int4 (W4A8) | `mmul<4,16,16>` (`mac_4x16_16x16`) | **512** (2 VMACs per `mac()`) | ~55–58 TOPS |

- **`mmul<8,8,8>` is NATIVE on AIE2P** (512 MACs, 1 VMAC) — the *opposite* of npu1
  where it is VIRTUAL. Use `<8,8,8>` for int8; `<4,8,8>` under-fills (256 of 512).
- **int4 does NOT double the per-VMAC rate on AIE2P.** `mac_4x16_16x16` takes a
  1024-bit B operand (`vector<int4,256>`) and computes 1024 MACs, but the
  disassembly shows it **lowers to 2 native VMACs of 512 MACs each** — the same
  512 MACs/native-VMAC as int8. (npu1's int4 genuinely doubles it, 256→512; the
  levers do not transfer.)
- **But int4 IS ~1.7× faster in the chain microbench**, measured per core:
  int8 `<8,8,8>` = **532 G MACs/s** (317 of 512 MACs/cyc); int4 `<4,16,16>` =
  **904 G MACs/s** (539 MACs/cyc — at peak). This corrects an earlier ISA-only
  "no compute win" claim: the win is real but it is **pipe-filling**, not a
  hardware rate. `<4,16,16>` issues 2 VMACs/call and reaches II=1 with just 4
  chains, whereas int8 `<8,8,8>` is overhead/spill-limited in the same microbench.
  A well-scheduled int8 GEMM can also approach the ~55 TOPS peak — treat 1.7× as a
  *schedule* effect. int4's durable wins: it fills the pipe more easily **and**
  halves weight-DMA bytes.
- **int4 × int4 (W4A4): no `mmul` family** on AIE2P — unsupported natively.
- **The ~126 TOPS figure is NOT the NPU.** It is AMD's Ryzen AI Max+ 395 *system*
  nameplate — NPU (~50) + Radeon 8060S GPU + Zen5 CPU. The NPU alone measures
  ~55–58 TOPS (32 × 512 MACs/native-VMAC × 2 × ~1.68 GHz), int8 and int4 alike.
  **There is no int4×int8 path that doubles NPU compute to ~110+ TOPS**: 126 on
  the NPU alone would need 3.85 GHz, and every shape lowers to 512 MACs/VMAC.

### Conclusions that flip vs npu1

| | AIE2 (npu1) | AIE2P (npu2) |
|---|---|---|
| `mmul<8,8,8>` int8 | VIRTUAL (2 VMACs) | **NATIVE** (512 MACs) |
| int4 (W4A8) per-VMAC rate | **2×** (256→512) | same 512; ~1.7× *sustained* via pipe-fill |
| task count as a lever | no (~1.5% for 8×) | **yes** (R68: 24% for 3×) — feed is ~3.5× faster/col, so per-task cost is no longer hidden behind the transfer |

**Conclusions do not transfer, not just constants.** The int4 and task-count
levers point *opposite ways* on the two generations. What transfers is the
*structure* — which cost terms exist, that fixed cost dominates small consumers,
that output-tile area matters — never the values or the rankings.

---

## Part 6 — Open / unmeasured

- **tok/s device comparison** (NPU 7.4 TOPS vs GPU peak). The speed verdict may
  invert the energy one. This is the biggest gap.
- **The true feed ceiling.** RESOLVED that the roof is per-stream and ≥30.8 GB/s
  at 8 streams (was reported as 16.1 GB/s total), but 16 streams fails
  placement, so the ceiling itself is unmeasured. J/byte and the 183 law are
  unaffected — energy per byte does not depend on how fast the bytes move.
- **CPU as a third target.**
- **Dual-objective (tok/s × tok/J) search** — the objectives provably diverge but
  `design.py` still ranks on time alone.
- **C8** alignment penalty; **AM020** never fetched (H3/M2/M3/M4 remain
  wrong-generation).
- Attention modelling excludes projections and assumes a device-resident cache.
