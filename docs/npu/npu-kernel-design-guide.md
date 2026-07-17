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
| **feed roof** | **~30.8 GB/s at 8 streams** — shim-channel limited, this is the ceiling |
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
More streams is a real and large lever — up to a hard wall at **8**.

### The wall: shim DMA channels

    device.get_num_connections()   shim: in=2  out=2   (x4 tiles)  => 8 streams
                                   memtile: in=6 out=6 (x4)
                                   core:    in=2 out=2 (x16)

    4 streams  OK   15.40 GB/s
    8 streams  OK   30.79 (4col x 2) / 28.31 (2col x 4)  <- both work
    12 streams FAIL placement (shim channels exhausted)
    16 streams FAIL

**Measurement and the toolchain agree exactly: the feed ceiling is 8 concurrent
shim input streams ≈ 30.8 GB/s.** Shim channels are the binding resource — *not*
columns, *not* cores. 2 columns × 4 fifos works as well as 4 columns × 2; only
the total stream count matters.

These channel budgets explain two other things:

- **`core: in=2`** is why C1's `nargs≥3` failed with "Resource allocation
  pipeline failed" — 3 inputs + 1 output needs 4 core in-channels and a core has
  2. More streams therefore needs more **cores**, not more fifos per core.
- **`memtile: in=6`** is why the per-column output join works (≤6 sub-fifos
  gathered per memtile).

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

### NPU vs GPU vs CPU (all on one die, one set of rails, one RAPL counter)

**The two objectives choose opposite devices, and the CPU wins neither.**

| axis | NPU | GPU | CPU |
|---|---|---|---|
| **movement** | 30.8 GB/s · **5.13 GB/J** | **79.5 GB/s** · 3.73 GB/J | 55.8 GB/s · 1.48 GB/J |
| **compute int8** | 7.4 TOPS · **938 G MACs/J** | **15.1 TOPS** · 432 G MACs/J | 4.1 TOPS · 54 G MACs/J |

- **tok/s → GPU.** 2.58× the NPU on movement, 2.03× on compute.
- **tok/J → NPU.** 1.37× the GPU on movement, 2.15× on compute — and **17× the
  CPU** on compute energy.
- **CPU → neither.** Dominated on both axes: slower than the GPU on both, and far
  less efficient than either. Its one advantage over the NPU is movement *speed*
  (55.8 vs 30.8 GB/s), because the NPU is capped by its 8-stream shim ceiling
  while the CPU has the full DDR path — but it pays 3.5× the energy per byte for
  it.

Confidence differs by leg. The **CPU compute figure is at ~100% of Zen 4's peak**
(8 cores × 64 MACs/dpbusd × 4.05 GHz = 2.07 T MACs/s exactly — AVX-512 is
double-pumped, so 1 dpbusd/cycle is the ceiling and SMT cannot help a
port-limited op). The **NPU is at 89%**. The **GPU is at ~42%** and is therefore
the one understated leg.

- **tok/s → GPU.** ~2–2.6× faster on *both* axes.
- **tok/J → NPU.** ~1.4–2.2× more efficient on *both* axes.
- **The GPU buys that speed for ~17× the power**: 24–28 W free-run package delta
  vs the NPU's ~1.4 W at full width.

There is no regime where one device wins both. Placement is a *policy* choice —
latency or battery — not an optimisation the model can make for you.

**The speed gap is understated.** The NPU compute figure is 89% of its
theoretical peak; the GPU's 15.1 TOPS is only ~42% of gfx1103's ~35 TOPS — the
`g1` chain kernel was written as an NPU analogue, not tuned for the GPU. A tuned
GPU kernel widens the gap. By contrast GPU feed (79.5 GB/s ≈ 78% of LPDDR5-6400
dual-channel) is near roofline, and the NPU's 30.8 GB/s is a hard shim-channel
ceiling, not a tuning artifact.

**Never mix the two measurements**: speed is free-run, energy is matched-rate.
Free-run energy tracks dispatch rate (§2.2) and is meaningless.

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

### 2.5 Validate the COMPOSITION, not just the terms

Family B validated `t_feed` (feed-only, trivial core). Family C validated
`t_core` (compute-only, no feed). Both in **isolation** — so the model's central
claim, that terms compose as `max()` and not a sum, went untested for the whole
build. A model can have every constant right and still compose them wrong.

D1 streams tiles *and* computes per tile, sweeping MMULS across the crossover:

    MMULS   t_core/tile   MEASURED   max model   sum model
        0            0n      1044n       1037n       1037n
       64          252n      1057n       1037n       1289n
      256         1009n      1151n       1037n       2046n   <- crossover
      512         2018n      2142n       2018n       3055n
     1024         4035n      4136n       4035n       5072n

    mean |error|:  max 4.1%   sum 33.2%

**Feed and compute genuinely overlap.** The shape is the prediction: flat while
compute hides under the feed, then slope 1. **Known limit:** at the crossover the
hard `max()` under-predicts by ~10% — a real pipeline has a soft knee. Away from
it, 2–6%. Single stream, single core; overlap under 8-stream × 16-core
contention is unmeasured.

### 2.6 Use slopes; fixed cost cancels

Sweep a parameter and fit — the 155 µs floor lands in the intercept, not the
rate. K1 gets f_H from the ITERS slope; C2/C5 get bandwidth from the bytes slope.
An absolute measurement of anything small is mostly floor.

### 2.7 Do not fit across a topology change

The core-power fit `W = 0.358 + 0.1261·cores` was excellent (±0.6% over 1–4
cores) — and its extrapolation to 16 was **1.7× wrong**, because reaching 16
cores *required* changing the topology to broadcast+join, which costs far less
DMA. The fit was fine; the extension was not. **A fit is only valid inside the
topology it was taken on.**

### 2.8 The ISA is the truth

- The `C_block<..., N>` template parameter in `mmul_8_4.hpp` looks like a
  native-op count and mostly is — but says `1` for `<4,32,8>` where the
  disassembly shows 2 vmac/call. **Count `vmac` in the disassembled loop.**
- Verify against hardware anyway: ISA count and measured MACs/s must agree. They
  did for every C4 row.
- Watch the unroll factor (`add rN, rN, #-0x4` ⇒ 4×) — bundles-per-source-iter is
  not what you'd assume.

### 2.9 Sources, ranked

1. **Probe on silicon** — settles behaviour.
2. **Toolchain target model** (`AIETargetModel.h`) — what the compiler believes.
   For capacity limits this is often the *operative* constraint: if mlir-aie
   thinks L1 is 64 KB, 65 KB won't compile whatever the die does.
3. **Vendor doc, correct generation** (AM020 for AIE-ML).
4. **Vendor doc, wrong generation** (UG1079/AIE1) — **not evidence**.
5. **Self-inconsistent metadata** (`rocminfo`: reports L2 alongside Cacheline 0,
   Max Clock 0, CU 0) — corroboration only, never alone.

### 2.10 Platform hazards

- **`@jit` is broken here**: aiecc asserts `targetModel.hasProperty(IsNPU)`
  during CDO generation even for a correct `aie.device(npu1)`. Use the
  skill-documented `compile_mlir_module` + `XRTHostRuntime` path.
  `r0/r0b_run.py` is halo-era and will not run. **19 files use `@jit`.**
- **`tools/npu/oq_gemm_design.py` is bit-rotted** — the repo's only NPU GEMM, and
  it does not run on nix1's toolchain. It needs **five removed `aie.iron` names**:

      In  Out  kernels (kernels.mm)  ceildiv  CompileTime      -> all MISSING
      str_to_dtype, TensorTiler2D                              -> still present

  It also uses `@jit`, so it is dead twice over. Blast radius is contained: it is
  the *only* file importing removed names (`build_qwen35_*.py` etc. use only
  surviving APIs and still build — verified). Its dependents
  `bench_oq_gemm_npu.py` and `test_oq_gemm_npu.py` go with it.

  **This blocks validating `design.py` against an independent kernel** — which is
  what it was worth. Rewriting it against the new API would make it *mine*, and
  destroy exactly the independence that made it valuable evidence. Fixing it is
  a toolchain-pin-vs-rewrite decision for a human.
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

### 2.11 Retire superseded constants

Five constants in the calibration set are `admissible=False` because they are
artifacts or superseded (`byte_mac_energy_ratio`=36.7, `j_per_mac_int8` 1-core,
`j_per_mac_int8_4core`, `npu_core_power_fit`, and a redundant duplicate). **A
stale constant left admissible is a trap** — the model would silently fall back
to the 37-MACs/byte artifact for any dtype pair without a specific ratio.

---

## Part 3 — Design rules that follow from the numbers

0. **Pick the device by objective, not by habit.** GPU for tok/s, NPU for tok/J
   — ~2× either way, on both movement and compute. There is no regime where one
   wins both, and the **CPU wins neither** (dominated on both axes; 17× worse
   than the NPU on compute energy). Its only edge over the NPU is movement speed
   (55.8 vs 30.8 GB/s), at 3.5× the energy per byte.
1. **BATCH — the biggest lever measured.** 64× tok/J and 348× tok/s from B=1 to
   B=512 on a 768×1280 projection, because weights are read once and reused
   across the batch (weight byte share falls 32% → 6%).

   | B | AI | device | tok/s | tok/J | µJ/tok |
   |---|---|---|---|---|---|
   | 1 | 1.0 | 217.9 µs | 4 589 | 5 144 | 194.4 |
   | 64 | 41.7 | 230.1 µs | 278 090 | 177 232 | 5.6 |
   | 512 | 97.2 | 320.9 µs | 1 595 443 | 331 077 | 3.0 |
   | 2048 | 113.4 | 818.7 µs | 2 501 662 | 365 026 | 2.7 |

   **Batching does NOT amortise the output**, which scales with B and comes to
   dominate the byte count (43% → 59% at 4 columns, **80%** at 1 column). So AI
   ceilings at ~97–113 with an int32 accumulator: **batching alone cannot cross
   the 183 break-even.** It also does **not** amortise the KV cache — every
   sequence carries its own — so decode attention stays movement-bound at *any*
   batch size. **Batch the projections; the KV read is irreducible.**

   `python -m aiecost.design batch-sweep --batches 1 64 512 2048`

2. **Emit bf16, not int32 — a DOUBLE win.** (a) 22% on *both* speed and energy
   (B=2048: AI 113.4→160.8, 819→639 µs, 365k→446k tok/J) — nearly free, since the
   f32 accumulator is rounded on the way out anyway. (b) It also **halves the L1
   output tile** (2 B vs 4 B), which lets a *bigger* compute tile fit — and tile
   area drives core efficiency. Concretely: `whole_array` 128³ won't build with
   int32 output (128×128×4 = 64 KB = full L1), capping npu1 at tile 64³ ≈ 0.24
   efficiency; bf16 output frees the room. Only bf16 output *plus* minimal
   activation replication crosses AI 183 (B=512,1col: 187.3; B=2048: 258.2). R65
   reached the movement half of this by hand on aie2p.
3. **Maximise work per dispatch.** The 155 µs floor is 37–58% of realistic
   kernels. Fusing layers into one dispatch beats any format choice.
4. **Energy is bytes** (AI break-even 183) — and per rule 1, that holds at every
   practical batch size. Narrower weights/KV cut energy; a faster MMUL mostly
   does not.
5. **Speed and energy pick different schedules.** M=256 QKV: `c4` is
   speed-optimal (265.2 µs, 1.940 mJ), `c1` is energy-optimal (484.3 µs,
   1.825 mJ). Broadcasting activations across 4 columns replicates bytes that
   cost 183 MACs each. *You cannot optimise both at once* — but check the size of
   the trade: at M=1 it is 0.2% and the "trade" is illusory.
6. **Use native, VMAC-filling MMUL shapes** (part 1). Wider is sugar; narrower
   wastes half the issue.
7. **Task count is NOT a lever on npu1** — an 8× cut buys ~1.5%. This
   *contradicts* aie2p (R68: 3× cut → 24%). npu1 feeds at ~3.95 GB/s/stream, so a
   2 KiB tile takes ~553 ns to move while task issue costs ~5 ns: the overhead
   hides entirely behind the transfer.
8. **Watch the output drain.** For int4 GEMM the int32 accumulator output becomes
   the limiter once weights halve — see rule 2.
9. **KV bit-width is the decode lever**: kvarn8→4 gives 1.61× (not 2× — the floor
   doesn't shrink), kvarn2 2.33×.
10. **Allocate BOs once.** 17.6 ms each, size-independent.

### Buildability — check before designing

| limit | value | what it kills |
|---|---|---|
| L1 | 64 KB/core | tile × fifo depth (32 KiB tile × depth 2 = exactly L1 → fails) |
| BDs | 16/core | DMA task budget (R118/R119) |
| locks | 16/core | R61 exhausted these at FIFO depth 1 |
| DPU args | **5 BOs** | binds constantly; forces broadcast+join |
| core DMA channels | **2 in / 2 out** | `nargs≥3` fails "Resource allocation pipeline failed" |
| **shim streams** | **8** (2 in × 4 tiles) | the feed ceiling; 12 streams fails placement |
| memtile channels | 6 in / 6 out | bounds the per-column join |
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
- **bfp16 (block-float / MX) is implemented, and is int8-rate — not a 2× path.**
  It is a real type (`bfp16ebs8`), with `block_vector.hpp`, the
  `mmul_bfp16_bfp16` family, and an AMD aie2p reference GEMM
  (`aie_kernels/aie2p/mm_bfp.cc`). **Compile-confirmed**: a minimal
  `mac_8x8_8x8T(block_vector<bfp16ebs8,64>, …, accum<accfloat,64>)` builds for
  aie2p and emits **one native `vmac.f` per call = 512 MACs**, same as int8 (the
  loads are `vlda.pop.576` — 64 values × 9 bits = mantissa + shared exponent). The
  "Block FP16 2×" claim is bfp16 *vs true bf16* (which is heavily emulated, ~16
  VMACs per `mac()`), **not** vs int8: bfp16 buys FP16-ish accuracy **at int8
  speed**, not more throughput. **Caveat for building it:** AMD's *full*
  `mm_bfp.cc` fails to compile with the installed peano/llvm-aie backend
  (`unable to legalize <8 x s8> G_BUILD_VECTOR` in the shuffle/exponent *helper*,
  not the matmul; see `BUGS.md`) — a hipfire bfp16 GEMM must call the core
  `mac_8x8_8x8T` directly and avoid that helper. **Hardware-confirmed rate**: the
  `e2_bfp16` resident-chain bench builds and runs on halo at **489 G MACs/s/core**
  — int8-rate (int8 `<8,8,8>` = 532), as predicted. (Numerics — block-float
  packing + a reference compare — still unrun; the rate uses random-byte inputs.)
- **512 MACs/native-VMAC is the datapath ceiling for every dtype on AIE2P.** int8,
  int4, and bfp16 all top out there — the 512-bit VMAC does 512 MACs/cycle and no
  operand width exceeds it (int8 already saturates the MAC units; narrower operands
  don't add MAC lanes, unlike npu1 where int8 leaves half the lanes idle so int4
  doubles). Peak is therefore ~55–58 TOPS for the whole int/bfp family.
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

- ~~tok/s device comparison~~ **RESOLVED**: GPU 2.0–2.6× faster on both axes;
  NPU 1.4–2.2× more efficient on both. The verdicts *do* invert.
- **A tuned GPU compute kernel** — `g1`'s chain reaches only ~42% of gfx1103's
  theoretical, so the measured speed gap is a lower bound.
- **`design.py` validated against an independent kernel — ORDINAL passes,
  MAGNITUDE is ~2.3× optimistic.** With `oq_gemm` runnable again (custom
  toolchain, see AGENTS.local.md), the model was checked against its measured
  single-core batch sweep (M=512,K=512):

      B     pred    meas    err
      32   190 µs  370 µs  −49%
      256  413 µs 1062 µs  −61%

  Ranking is perfect (τ=1.0, both monotonic) — so the model's *product* (ranking
  schedules) holds against a real kernel. But absolute `t_core` under-predicts
  because it assumes a **saturated VMAC pipe** (1 VMAC/cyc), which the tight
  resident chains of families B/C/D reach but a hand-written tiled GEMM does not.
  Fixed by `ScheduleSpec.core_efficiency` (default 1.0 for the saturated chains):
  `t_core /= η`, and with η=0.28 the `oq_gemm` predictions land within ±5%
  (compute-bound), ordinal preserved.

  **η is NOT a constant, though.** The independent multi-core `whole_array`
  matmul measures efficiency climbing with tile area and problem size —
  **0.014 (512³/32³ tile) → 0.24 (2048³/64³)** — the same aie2p tilesweep law
  (32²=2.4 → 128²=15.7 TOPS). So **0.28 is the large-tile asymptote** (reached by
  single-core `oq_gemm` and multi-core 2048³), and `design.py`'s flat value is
  optimistic by up to ~20× for small GEMMs. The real fix is `η = f(output-tile
  area)`, but `design.py` models only MAC counts, not the L1 tile — so a scalar
  stand-in remains. Defensible for prefill-scale (large M → large tiles →
  ~0.24–0.28); optimistic for small compute-bound GEMMs; moot for decode
  (movement-bound, `t_core` doesn't matter).
- ~~The true feed ceiling~~ **RESOLVED**: ~30.8 GB/s at 8 shim input streams,
  confirmed by measurement and the toolchain channel budget independently.
- ~~CPU as a third target~~ **DONE** (P1): dominated on both axes — 55.8 GB/s ·
  1.48 GB/J, 4.1 TOPS · 54 G MACs/J. Measured at ~100% of Zen 4's VNNI peak.
- ~~Dual-objective (tok/s × tok/J) search~~ **DONE**: `design.py` reports the
  Pareto front, both optima, and the cost of choosing tok/J — with a 2%
  tolerance so a sub-noise difference is not announced as a trade.
- **C8** alignment penalty; **AM020** never fetched (H3/M2/M3/M4 remain
  wrong-generation).
- Attention modelling excludes projections and assumes a device-resident cache.
