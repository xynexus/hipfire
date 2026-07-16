# NPU GEMM — path to ~50 TOPS (findings)

**Goal:** reach ~50 TOPS int8 GEMM on Strix Halo NPU (aie2p/npu2), or determine
the true ceiling + limiter with evidence. TOPS = 2·M·K·N / time (matches the
whole_array bench's "gflops").

## Headline conclusions

1. **The NPU is NOT power/clock-throttled.** Hardware ceiling is **58 TOPS int8**
   and, under GEMM load, `default` pmode already boosts to the **full 58-TOPS
   budget with the AIE compute clock maxed at 1800 MHz**. So ~50 TOPS is real
   silicon that is *available*, not gated behind a power mode. Turbo pmode
   expected ~no-op (compute clock already at max) — 1-line confirm pending sudo.
2. **The mlir-aie `whole_array` REFERENCE dataflow caps at ~15.7 TOPS = 27% of
   peak**, and every tunable knob is explored (below). This is a **dataflow
   efficiency** limit, not compute or power.
3. **AMD's PRODUCTION `mladf` kernel does NOT reach 50 either** — built
   DynamicDispatch from source and ran real mladf gemms on the NPU. The shipped
   int4 (w3a16) gemm is a flat **~7 TOPS** across LLM shapes — a *memory-bound
   weight-quant decode* kernel, actually **below** our int8 reference. The
   compute-bound bf16 prefill gemm wasn't runnable in this package
   (`op_version "bfp16gemm"` unregistered).

**Bottom line (revised, evidence-backed):** The hardware genuinely does 58 TOPS
and is un-throttled — but **no real GEMM kernel I could measure approaches it.**
The tuned reference (int8) and AMD's shipped production kernel (int4) both sit in
the **~12–27%-of-peak band (7–16 TOPS)**. ~50/58 TOPS is a theoretical/marketing
peak; real inference GEMM on this AIE dataflow is per-tile-feed/overhead-bound at
these shapes. Our int8 reference at **15.7 TOPS is the highest real number
measured** — beating AMD's shipped int4 decode kernel. Reaching ~50 would need a
compute-bound dataflow that neither the reference nor the shipped kernels
demonstrate (unverified whether one exists; the compute-bound production path was
not runnable here).

## Device facts (xrt-smi examine + hipfire-xdna resource_info)

- AMD RYZEN AI MAX+ 395, NPU Strix Halo **aie2p**, topology 6×8 = shim + memtile
  + **4 compute rows × 8 cols = 32 cores**. XRT 2.25.0, amdxdna 2.25.0, FW 1.1.2.65.
- `npu_clk_max=1800 MHz`, `npu_tops_max=58`.
- Idle (default pmode): tops_curr=25, MP-NPU 396 MHz, H 792 MHz.
- **Under GEMM load (default pmode): tops_curr=58, MP-NPU 1267 MHz, H 1800 MHz (max).**
- Peak check: 32 cores × 512 int8 MAC/mmul (8×8×8) × 1.8 GHz = 59 TMAC ≈ 58 ✓.
  (H = AIE compute clock.)
- pmode set (`xrt-smi configure --pmode`) needs CAP_SYS_ADMIN → root; sudo is
  password-gated on this box.

## Experiments (all i8/i8, 4096³, 8col, warmup5/iters20 unless noted)

| # | Lever | Result | Verdict |
|---|---|---|---|
| E1 | power/clock throttle | default already 58 TOPS + H=1800 under load | NOT throttled |
| — | output tile m×n | 32²=2.4, 64²=8.2, 128×64=11.0, 64×128=14.3, **128×128=15.7** | THE lever; L1-capped (area 16384) |
| — | reduction depth k | k=128 vs 256 at 32² → 2.41 vs 2.28 | irrelevant |
| — | columns | 4c=4.4, 8c=8.2 (1.88×) | maxed at 8 |
| — | OPT_PERF flag | 8228 vs 8211 gflops (flag compiled in) | no-op (Peano ignores chess pragma) |
| — | pre-tiled weights (contiguous B) | 15.62 vs 15.66 | no-op (re-tile is cheap on-chip memtile→core) |
| E2 | fifo_depth | 2→15.3, 3→**15.7** (+2.5%), 4→build-fail | marginal; DMA already hidden |
| — | fifo_depth=1 + 2× tile | control 15.01 (< fd2); 2× tiles build-fail | dead end |
| — | microkernel mm.cc | AMD-documented 2×2 mmul optimal, register-limited | tapped |

**Best config: m128 k32 n128, 8col, i8/i8, fifo_depth=3 → ~15.7 TOPS (27%).**
Every reference knob is explored; ~15.7 is the reference dataflow ceiling.

## Why the reference caps at 27% (mechanism)

Feed/overhead-bound: throughput scales with **output-tile size** (amortizes
per-tile DMA setup + C accumulator load/store + objectfifo acquire/release +
software-pipeline fill/drain), and the tile is L1-capped (64 KB/core) at area
16384. Deeper k, more columns, better weight layout, and deeper buffering do not
help — the cores are starved by per-tile overhead the reference can't shrink
without a different dataflow (e.g. K-resident C, cross-core cascade/systolic, or
a larger effective tile via smarter memtile use — i.e. what mladf does).

## mladf bench — DONE (built DynamicDispatch from source, ran on NPU)

Measured, all PASS (latency reported by the op; TOPS = 2·M·K·N / latency):

| mladf kernel | shape M×K×N | latency | TOPS |
|---|---|---|---|
| int4 w3a16 grp128 | 512×3584×3584 | 1.874 ms | **7.0** |
| int4 w3a16 grp128 | 512×3584×18944 | 10.10 ms | 6.9 |
| int4 w3a16 grp128 | 1024×3584×18944 | 20.20 ms | 6.9 |

Flat ~7 TOPS = a memory-bound weight-quant **decode** kernel (int4 weight, bf16
activation), **below** our int8 reference (15.7). The compute-bound `Bfp16Gemm`
test throws "op version does not exist" (`op_version "bfp16gemm"` not registered
in this package); a16w8/a16a16 gemms are meta.json runners (not driven). So the
compute-bound production path was not runnable here — but the shipped int4 kernel
sitting at the same ~25%-of-peak efficiency as our reference is strong evidence
the ceiling is dataflow-fundamental, not a kernel we're missing.

### DynamicDispatch build recipe (worked; build left in place at ~/build/DynamicDispatch/build)
Deps: nlohmann_json/spdlog/xaiengine auto-fetched; XRT found; system protobuf +
a **CPU-torch venv** (ROCm-torch needs an absent HIP cmake pkg):
```
uv python install 3.12 && uv venv -p 3.12 /tmp/cputorch
uv pip install --python /tmp/cputorch/bin/python torch --index-url https://download.pytorch.org/whl/cpu
# edit DynamicDispatch/CMakeLists.txt:74  find_package(Protobuf CONFIG REQUIRED) -> MODULE
cmake -B build -DENABLE_DD_TESTS=ON -DUNIT_TEST_PERF_EN=ON -DENABLE_DD_PYTHON=OFF \
  -DDD_DISABLE_AIEBU=ON -DCMAKE_BUILD_TYPE=Release -DXRT_DIR=/opt/xilinx/xrt \
  -DTorch_DIR=/tmp/cputorch/lib/python3.12/site-packages/torch/share/cmake/Torch \
  -DCMAKE_CXX_FLAGS="-I/opt/xilinx/xrt/include/xrt -include cstdint -include cstddef"
cmake --build build --target cpp_tests -j$(nproc)
# run: build/tests/cpp/unit_tests/cpp_tests --gtest_filter='Qwen7b_2Testw3a16_high_time.Kernel4mladf_512x3584x3584_int4_grp128_v1'
```
Gotchas fixed: XRT 2.25 moved experimental headers to `xrt/experimental/`
(`-I.../include/xrt`); GCC 15 needs `-include cstdint -include cstddef`
(transitive-include tightening); protobuf CONFIG→MODULE (Ubuntu ships no config pkg).

## Confirmed no-op levers
- **turbo pmode**: 15.21 vs 15.7 TOPS (compute clock already maxed under load).

## Open (only if chasing the last mile)
- Run a compute-bound production gemm (fix `bfp16gemm` op-version, or hand-build a
  a16w8/a16a16 meta.json) to test whether ANY shipped kernel exceeds ~16 TOPS.

## 2026-07-12 external-memory and MALL update

R56 separated the external-feed roof from the much lower throughput of the
current resident model dataflow:

- one active receive stream is 14.4 GB/s; eight columns saturate near 56.5 GB/s;
- shared working sets from 64 KiB through 64 MiB show no 2 MiB or 32 MiB cache
  knee, and shared versus distinct regions have the same roof;
- CPU and over-capacity GPU streaming contend strongly with the NPU, while GPU
  hot sets totaling 16 or 32 MiB do not;
- the installed amdxdna driver states that the NPU is not cache coherent, and
  neither UG1079 nor the XDNA driver documents an NPU path through GPU MALL.

The current amdxdna SHMEM path therefore has no observed usable MALL caching.
More importantly, resident EmbeddingGemma attention and FFN consume their packed
weight payloads at only about 0.9 GB/s, roughly 1.6% of the external-feed roof.
Those phases are dataflow/dispatch/local-scheduling limited, not globally
memory-bandwidth limited. Full methods, raw results, caveats, and the complete
manual audit are in
[`../../docs/npu/npu-memory-bandwidth-cache-characterization.md`](../../docs/npu/npu-memory-bandwidth-cache-characterization.md).

## 2026-07-12 R57 bandwidth-first production ladder

R57 replaced the synthetic extent with R34's exact production schedule: four
active streams, 125 16-KiB blocks per stream, and 8,192,000 wire bytes. Direct
DMA, one-to-one memory-tile forwarding, four-row memory-tile broadcast, and a
real `.rdna2.hfp` payload measure 43.577, 43.536, 43.198, and 43.251 GB/s at the
four-column median. The final stage retains 99.25% of direct DMA; memory-tile
staging and broadcast therefore do not explain the resident model's ~0.9-GB/s
payload rate.

The physical stream contains 8,159,360 non-padding bytes but only 2,558,980
semantically unique bytes after removing R34's cross-column/M-tile replication.
R57 reports all three quantities instead of calling every requested byte useful
bandwidth. Four-way broadcast raises logical semantic delivery to about 54.0
GB/s without increasing external bytes.

The loader-side block layout is now cached in source- and payload-SHA-validated
`.rdna2.hfp` files. The current checkpoint caches the dense-W8/BF16 R34 payload
derived from OQ4; packed-nibble OQ4 remains the next stage and must keep only its
local nibble/lane swizzle in-kernel.

## 2026-07-12 R58 packed Opus HFP + NIBBLE_DECODE

The production loader now writes version-2 Opus HFP artifacts whose W4 payload
stays nibble-packed in the exact eight-column `NpuGemmWholeScaled` block order.
For layer-0 combined QKV (M256/K768/N1280), the file is 192 header bytes plus
2,359,296 payload bytes. A cache-hit load preserved its mtime, size, and SHA.

R58 compares the same real artifact and four-row memory-tile fanout across
feed-only, signed-nibble-decode, and first-MMUL modes. Three trace-timed trials
give 56.173, 56.061, and 55.933 GB/s wire medians respectively. Decode retains
99.80% of feed; compute stage 1 retains 99.77% of decode. Every decode trial passes a 64-lane whole-stream checksum
and byte-exact real-vector low/high/sign oracle. Every compute trial passes an
exact int32 `mmul<4,16,16,int8,int4>` oracle and reaches 2.685 TOPS for the
minimal M=16-equivalent stage. Neither decode nor the first MMUL costs observable
external bandwidth.

Compute stage 2 adds six activation row blocks and real K=256 continuation. It
passes exact parity at 11.497 TOPS, while wire bandwidth falls to 39.919 GB/s
and receive stalls rise to 57.45%. This is the first honest compute-backpressure
point: the 71.37% feed retention is below the nominal 85% gate, but useful TOPS
rises 4.28x and counters identify the consumer bottleneck, satisfying the plan's
documented exception. The next ratchet is per-group scale accumulation and
distinct production output placement, not additional external-bandwidth work.

The first checksum-only scale ratchet failed correctness and was removed. Its
in-loop horizontal float reductions perturbed later virtual-int4 MMUL results;
integer sentinels proved the error preceded scale comparison, and explicit
vector-mode resets plus local activation reloads did not repair it. This is a
compiler/schedule interaction specific to that diagnostic structure, not
evidence against the production epilogue.

The production vector-store epilogue passes on the real model and real HFP:
layer-0 concatenated Q/K/V at M256/K768/N1280 has zero mismatches across 327,680
outputs and `max_abs=2e-7` against the generic Opus CPU reference. Its wrapper
time is 0.8635 ms (0.5829 logical TOPS) including host activation packing,
dispatch/sync, and output deblocking. Keep this number separate from R58's
trace-derived feed/compute ceilings and from end-to-end embedding tok/s.

Durable rows: `results/r58-nibble-decode-20260712.csv`.

## 2026-07-13 generic Opus HFP matrix

`FullKV1` extends the versioned `.rdna2.hfp` contract to the older full-K slab
schedule, including compact mixed `qt=36`. Real AIE2P projection parity now
covers 26 W4/W8/mixed cases, plain/`+`/`++`, overlay counts 1 through 39, and
N=256 through N=2304. Every case passed; mixed direct full-K was exact and
W4/W8 maximum absolute error was 2e-7. A repeated OQ6.5 load proved mtime,
size, and SHA-stable reuse. This removes runtime global slab reordering from
the generic mixed path but does not improve the still-dense R34/R35 resident
schedule by itself. Durable rows:
`results/r58-opus-hfp-format-matrix-20260713.csv`.

## 2026-07-13 R59 resident HFP argument ABI

The DPU command packet has five data-argument slots, so packed R34/R35 cannot
add every role and parameter as a separate BO while retaining their existing
shared I/O. `ResidentContextBundleV1` solves this offline: one HFP BO contains
QKV+O+params for R34 or gate/up+down+params for R35, preserving block order
inside every source role.

R34 and R35 separate-BO controls measure 56.121 and 56.053 GB/s median; their
production bundles measure 55.884 and 56.018 GB/s, at least 99.48% of R58 feed-only.
All guards pass and traced receive stalls are zero. This removes argument/DMA
bandwidth as the next resident blocker; projection compute integration remains.
Durable rows: `results/r59-resident-weight-abi-20260713.csv`.

## 2026-07-13 R60 exact R34 shared-input compute ladder

R60 consumes the admitted R34 `ResidentContextBundleV1` and the existing
2,949,120-byte shared activation layout together. It does not create a second
activation format and does no global weight-block reordering in the kernel. A
new production CPU oracle delegates to the established R30 packer and proves
that R34 uses the same 16-KiB activation blocks byte-for-byte.

Three accumulated hardware stages all pass across eight AIE2P columns:

| stage | median wire GB/s | range | R59 retention | correctness |
|---|---:|---:|---:|---|
| first 4x16x16 MMUL | 53.838 | 53.829-54.112 | 96.34% | exact int32 |
| full K=256, one group | 54.252 | 53.870-54.635 | 97.08% | exact int32 |
| three groups + real scale tails | 50.912 | 50.605-51.092 | 91.10% | f32 tolerance |

The scaled stage performs 48 native int8-by-int4 MMULs for the first 4x16 QKV
output tile and uses the real activation and OQ4 weight scales. Its roughly
9.9% median receive-stall fraction identifies compute/epilogue backpressure,
but it still clears the plan's 85% retention gate. An attempted depth-3
activation FIFO exceeded tile SRAM beside the weight double buffer; depth 1 is
the correct sequential schedule. The first trace also caught and rejected a
683-GB/s channel-selection artifact before admission. Durable rows:
`results/r60-first-shared-input-mmul-20260713.csv`.

## 2026-07-13 R61 complete QKV placement and initial activation-layout hypothesis

R61 expands the resident-bundle OQ4 path to the complete M256/K768/N1280 QKV
projection and scatters the joined core streams directly into padded
M288/N1536 row-major output. All 327,680 real outputs and padded cells pass;
`max_abs=3.8147e-6`. Immutable weights remain in bundle order.

Correctness exposed a new performance boundary. Median NPU time progressed from
6.870 ms with scalar local activation assembly to 4.238 ms with native 32-byte
loads and `interleave_zip(...,8)`. Caching the complete 6-KiB transformed view
in tile SRAM produced 4.114 ms, so it did not remove the remaining cost. The
final median is only 0.122 useful TOPS and about 4.8x slower than the existing
0.8635-ms whole-scaled wrapper control. R61 is therefore not integrated into
the model. R62 later falsified the initial conclusion that activation joining
was the dominant cause; the two values came from different runtime/timing
protocols.

Two topology failures further narrowed the design. Thirty-two independent
output drains exceeded shim DMA capacity. A memory-tile join reduced them to
eight, but the high-level IRON stateful transform exhausted tile locks even at
FIFO depth 1. The equivalent raw-MLIR R16 topology compiled and ran. The next
ratchet is a producer- or DMA-native mutable W4 activation view, not more
in-core caching or any immutable tensor-block reorder. Durable final rows:
`results/r61-full-qkv-rowmajor-20260713.csv`.

## 2026-07-13 R62/R63 native-input and production-wrapper controls

R62 makes mutable activations producer-native W4 while preserving the R34
bundle. Row-major output measured 3.850 ms median; physical output plus an
exact R15 dynamic group loop measured 3.856 ms. Both pass the complete QKV
oracle. The small change from R61's 4.114 ms disproves the claim that the legacy
8+8 activation join was the main remaining cost.

R63 then makes the whole graph identical to R15 and uses the existing compact
2,359,296-byte QKV `.rdna2.hfp` rather than the 3,670,016-byte layer bundle.
Neither compact backing nor interleaved W/C task launch reduces the cold Python
raw-runtime time. A controlled production-wrapper matrix explains why: current
R63 is 1.0292 ms median, the pre-`3db7a1497` spill binary is 1.0288 ms, and the
historical cache is 1.0596 ms. All nine wrapper runs pass 327,680 outputs at
`max_abs=2e-7`.

The 3.5-4.2-ms raw values are cold first-command `XRTHostRuntime.npu_time`
diagnostics and must not be compared with a warmed Rust/C++ wrapper average.
The admitted path is the current compact offline HFP and production executor;
the next ratchet is resident-layer integration and shared-BO handoff, not more
activation swizzle work. Durable rows:
`results/r62-w4-native-full-qkv-20260713.csv`,
`results/r62-w4-native-physical-qkv-20260713.csv`,
`results/r63-w4-native-compact-qkv-20260713.csv`, and
`results/r63-production-wrapper-ab-20260713.csv`.

## 2026-07-13 R64 exact-graph device trace

R64 injects declarative trace configuration into parsed canonical R63 MLIR;
the untraced module must first diff byte-for-byte against R15. Core-tile trace
packet routing is not viable on the fully occupied graph: eight and four flows
remain CPU-bound beyond five minutes after address assignment, and one core
flow exceeds four minutes. Shim-local DMA tracing builds in seconds and avoids
routing through the compute network.

Twelve locked traces across activation-bearing shim columns 0–3 pass the full
327,680-output oracle. The median device input-to-output span is 241.248 us
(240.189–243.356 us), aggregate effective traffic is 19.559 GB/s, and the
output DMA is starved for 198.240 us median. Padded and useful compute rates are
2.817 and 2.086 TOPS. Columns 4–7 lose their final S2MM event at trace stop, so
their spans are not admitted despite full output parity.

The 1.0292-ms warm production wrapper is therefore about 23.4% device work and
76.6% preparation/submit/sync/deblock overhead. R65 should write the mutable
QKV result directly into the verified BF16 attention layouts and chain by
shared BO; more immutable-layout or activation-join work is not indicated.
Durable rows: `results/r64-full-qkv-shim-trace-20260713.csv`.

## 2026-07-13 R65 direct BF16 attention staging

R65 retains R63's compact QKV `.rdna2.hfp`, producer-native 8-KiB activation
records, and unchanged R15 W4 scaled compute. A local finish function splits
each completed 24x96 f32 accumulator tile into three padded 24x32 BF16 records.
DMA scatters only the 40 real 32-column stripes into the five-role R29 raw
attention ABI; padded projection columns are consumed but never emitted.

Three locked fresh-process runs pass all 327,680 BF16 values bit-for-bit,
preserve every preseeded cos/sin/norm tail byte, and leave padding records zero.
Median NPU time is 0.487964 ms (0.485649-0.490328), median host-call time is
0.551481 ms, useful projection rate is 1.0315 TOPS, and maximum core text is
9,280 bytes.

This is mutable output placement, not immutable conversion: HFP block order and
local OQ4 nibble/lane handling are unchanged. R66 should consume this BO with
the existing R29 headnorm/RoPE packers. Durable rows:
`results/r65-w4-bf16-raw-attention-20260713.csv`.

## 2026-07-13 R66 inline-stage pack consumer

R66 consumes R65's exact five-role 10-KiB records and reuses the established
R29 Q/K/V headnorm/RoPE pack functions. It emits the canonical 393,216-byte Q
and 262,144-byte single-replay K/V layouts. The locked oracle matches R28: Q
cosine 0.99999121/max 0.0078125, K cosine 0.99999156/max 0.0078125, and zero V
bit mismatches.

The schedule is rejected for performance. Three fresh 100-command processes
measure 0.9511, 0.9915, and 0.9984 ms (median 0.9915 ms); combined sequentially
with R65 this is about 1.48 ms before attention. R66 broadcasts one record at a
time and serializes the four core-pair packers. R28's joined compact input runs
the four pairs concurrently and historically measures 0.3659 ms (0.4145 ms in
a current ten-command control). R67 should change the mutable producer/staging
layout to feed that joined schedule, not optimize the already-correct pack math.
Maximum R66 core text is 10,432 bytes. Durable rows:
`results/r66-r65-stage-to-qkv-20260713.csv`.

## 2026-07-13 R67 joined mutable staging

R67 replaces R66's serialized physical-record broadcast with a mutable joined
layout. Projection emits 8x32 BF16 tiles into 36 8-KiB records per role; the
first 32 records are M256 data and four are padding. R28's split FIFO can then
activate all four core pairs from one joined input. HFP ordering and pack math
are unchanged.

Projection passes all 327,680 BF16 values bit-for-bit and preserves all
cos/sin/parameter tails. Three fresh warmed processes measure 0.725232,
0.751200, and 1.152343 ms (median 0.751200 ms). The pack consumer retains Q/K
cosine above 0.99999, 0.0078125 maximum error, and bit-exact V; three
100-command runs measure 0.3517, 0.3670, and 0.3687 ms (median 0.3670 ms),
recovering historical R28 throughput.

The sequential median is about 1.1182 ms before attention. R68 should collapse
the projection's three token-group DMA objects per slice into one padded
24-token object with overlapping padding records, reducing roughly 360 output
tasks by about threefold. Durable rows:
`results/r67-w4-joined-stage-20260713.csv` and
`results/r67-joined-stage-to-qkv-20260713.csv`.

## 2026-07-13 R68 overlapping joined producer

R68 collapses R67's three 8-token output objects per slice into one padded
24-token object. DMA uses a three-record core-row stride, so each padding record
is overwritten by the next core row's first real record; a 37th record per role
absorbs the final pad. This cuts projection output tasks about threefold while
preserving R28 joined consumption order.

Projection remains bit-exact across 327,680 BF16 values and preserves all
preseeded tails. Three fresh warmed processes measure 0.465281, 0.494605, and
1.067005 ms (median 0.494605 ms). Pack runs measure 0.3435, 0.3579, and 0.3627
ms (median 0.3579 ms) with unchanged Q/K/V parity. Sequential medians total
about 0.8525 ms before attention, 24% below R67 and 42% below R65+R66.

R69 should measure the real two-context chain by importing one shared stage BO
into both kernels; no host copy or immutable conversion is needed. Durable
rows: `results/r68-w4-overlap-joined-stage-20260713.csv` and
`results/r68-overlap-joined-stage-to-qkv-20260713.csv`.

## 2026-07-13 R69 cross-context rejection

Independent amdgpu-GTT imports are not coherent after producer completion, and
native XDNA SHMEM PRIME export returns `EINVAL`. Sharing the original GEM handle
through one DRM file works only intermittently and takes 5.45-5.66 ms because
both full-array contexts inflate to roughly 2.7-2.9 ms. The BO sync itself is
only 0.028-0.029 ms. UG1079 documents ordered graph streams and graph-owned DMA
buffers/locks, but no cross-context SHMEM visibility fence. The architecture is
rejected. Durable rows: `results/r69-cross-context-shared-qkv-20260713.csv`.

## 2026-07-13 R70 single-context projection and pack

R70 merges W4 projection and headnorm/RoPE packing into one graph. The literal
R68+R67 merge exceeded two input DMA channels; an initial inline merge exceeded
16 KiB of program memory. Reusing the activation channel, assigning K to
columns 0-3 and V to columns 4-7, and merging W4 init/accumulate into one group
function closes both limits. Loader-side activation padding is the only layout
change; immutable weights remain in offline `.rdna2.hfp` order.

Three fresh primed 100-command runs are byte-exact against isolated R65 stage
and R66 Q/KV oracles and measure 1.3076, 1.3108, and 1.3006 ms (median 1.3076
ms). Maximum core text is 13,504 bytes. This is a projection/pack boundary, not
full-model throughput. The next graph should attach attention in this context.
Durable rows: `results/r70-single-context-projection-pack-20260713.csv`.

## 2026-07-13 R71 fused attention: correct, not admitted

R71 attaches R30 attention inside R70's graph without exceeding the five-data-
argument ABI. The attention result is appended to the staging/result BO. A
third core input channel and the first 19-KiB linked programs were rejected.
The fitting graph moves Q/V packing to columns 0-3 and runs attention on
columns 4-7; maximum core text is 15,888 bytes.

All stage, Q, KV, and 393,216 attention bytes match the isolated R70-to-R27
reference. Three primed 100-command runs measure 3.5951, 3.2617, and 3.3118 ms
(median 3.3118 ms). The exact redistributed-pack control measures 1.5446 ms
median and isolated attention 0.9141 ms, leaving about 1.77 ms for fused
attention/feed. A split-direction shim design was rejected because it exceeded
memory-tile input DMA channels. R71 is correctness evidence, not the resident
integration candidate; the next rung must eliminate or locally stream the
external Q/KV round trip. Durable rows:
`results/r71-single-context-projection-pack-attention-20260713.csv`.

## 2026-07-13 R72 scalar direct-Q rejection

R72 removes the 393,216-byte external Q write/read by streaming packed query
pairs from projection columns 0-3 to attention columns 4-7. The receivers cache
six groups in storage shared sequentially with the projection accumulator. The
graph fits through sequential tile-memory allocation at 15,248 bytes maximum
core text and leaves the Q BO unused.

Projection stage, external KV, and final attention remain byte-exact against
the R70/R27 references. Performance regresses: three primed 100-command runs
measure 3.9288, 3.7749, and 3.9272 ms (median 3.9272 ms), 18.6% above R71.
Scalar stream synchronization and cache pressure cost more than the removed Q
traffic. Do not copy this topology to K/V; retain the exact graph-local result
as evidence while the next handoff uses burst/vector DMA or an existing FIFO.
The existing kernel parameter remains the correctness workaround; this result
does not impose LDS/tile-memory avoidance. Durable rows:
`results/r72-direct-q-stream-20260713.csv`.

## 2026-07-13 R73 adjacent-tile Q rejection

R73 replaces R72's scalar per-word stream with one depth-one 24-KiB
ObjectFIFO shared by each adjacent projection/attention core pair. The Q BO is
unused, and projection stage, external KV, and final attention remain
byte-exact. A producer-local cache first failed the 16-KiB program limit at
16,880-16,896 bytes; the adjacent topology fit after reducing the producer
stack from 4 KiB to 2 KiB, with maximum producer and consumer core text of
14,912 and 14,352 bytes respectively.

Three fresh primed 100-command runs measure 3.6449, 3.7165, and 3.7205 ms
(median 3.7165 ms). This is 5.4% faster than scalar R72 but 12.2% slower than
R71's external-Q baseline, so it is not admitted and must not be copied to
K/V. The result proves shared tile memory is functionally viable; the rejected
part is the serial six-group depth-one schedule. The existing kernel parameter,
not tile-memory avoidance, remains the correctness workaround. Durable rows:
`results/r73-adjacent-q-objectfifo-20260713.csv`.

## 2026-07-13 R74 paired-query KV replay rejection

R74 returns to R71's observable Q/KV boundary and keeps two query groups live
per attention core. Four accumulator/stat sets let each 262-KiB KV plane update
both groups, reducing complete KV replays and their DMA tasks from six to three.
The first 4-KiB-stack link exceeded the 64-KiB tile allocation by 1,184 bytes;
a 2-KiB stack fits at 64,672 bytes with 864 bytes spare. Maximum core text is
15,248 bytes.

Projection stage, Q, KV, and attention remain byte-exact. Three fresh primed
100-command runs measure 3.4496, 3.4242, and 3.2867 ms (median 3.4242 ms), 3.4%
slower than R71's 3.3118-ms median. One sample is faster than R71, but the stable
median fails the ratchet. Halving KV replay/task count does not repay the extra
live Q/accumulator state. With R72 and R73, this redirects the next rung toward
phase scheduling/core utilization rather than more tile-resident buffering.
The added kernel parameter remains the correctness workaround; memory placement
is an independent measured constraint. Durable rows:
`results/r74-qgroup2-kv-replay-20260713.csv`.

## 2026-07-13 R75 two-group task-window admission

R75 changes only R71's runtime task schedule. It starts two groups' ordered Q,
KV, and output tasks before awaiting/freeing them, reducing six per-group
completion barriers to three windows. The first six-group attempt exhausted
static BD IDs at group 4. A four-group image linked but failed hardware parity
with 392,405 of 393,216 Q bytes wrong, establishing a runtime queue/order limit.

The two-group window restores exact projection stage, Q, KV, and attention
bytes. Three fresh primed 100-command runs measure 3.2580, 3.2775, and 3.3314
ms (median 3.2775 ms), 1.0% faster than R71's 3.3118-ms median. Kernels, tile
buffers, external traffic, and math are unchanged, so the small gain is
attributable to fewer command-stream completion barriers. R75 is admitted as
the next projection/pack/attention baseline. Durable rows:
`results/r75-attention-window2-20260713.csv`.

## 2026-07-13 R76 three-group task-window admission

R76 tests the only queue window between exact R75 and the corrupt four-group
control. It preserves every R71/R75 kernel, buffer, byte boundary, and math path
while reducing three two-group completion windows to two three-group windows.

Projection stage, Q, KV, and attention remain byte-exact. Three fresh primed
100-command runs measure 3.4199, 3.2222, and 3.2604 ms (median 3.2604 ms), 0.52%
faster than R75 and 1.55% faster than R71. Three groups is the maximum correct
window: four corrupts Q and six exhausts static BDs. Admit R76 as the schedule
to carry into resident-weight integration. Durable rows:
`results/r76-attention-window3-20260713.csv`.

## 2026-07-13 R77 resident R76 HFP admission

R77 removes the extracted `weights.bin` from the fused path. The production
`NpuEmbeddingQkvAttentionOpus` executor validates the real HFP v2 descriptor,
length, and payload SHA-256, allocates the 2,359,296-byte compact weight BO in
the destination R76 context, uploads once, and reuses it across commands. The
offline block order and local nibble/lane work are unchanged.

Projection stage, Q, KV, and attention remain byte-exact. Three fresh primed
100-command runs measure 3.2753, 3.3165, and 3.2137 ms (median 3.2753 ms), only
0.46% above raw R76 and 1.1% below R71. Admit the resident QKV/attention weight
seam. This is not complete-layer or model-level evidence: O projection,
residual/norm, FFN, next-layer handoff, tokens/s, and tokens/J remain open.
Durable rows: `results/r77-resident-hfp-r76-20260713.csv`.

## 2026-07-13 R78 odd-attention role-remap rejection

R78 moves unchanged R76 attention to odd columns and concentrates external
Q/K/V packing on adjacent even columns, establishing the neighbor direction
needed by R32 output projection without adding a graph-local Q buffer. Stage,
Q, KV, and attention remain byte-exact; odd cores fit at 15,888 bytes.

Three fresh primed 100-command runs measure 3.8331, 3.7729, and 3.7959 ms
(median 3.7959 ms), 16.4% slower than R76. Reject it as a standalone schedule.
Even cores still contain compact-W4 projection and packing, so appending R32 O
projection would exceed program memory. The next capacity design must use
R33-style paired compact-W4 projection on odd cores and drop projection code
from even cores. Pair-major block ordering belongs in a loader-created
`.rdna2.hfp`, never in the kernel. Durable rows:
`results/r78-odd-attention-remap-20260713.csv`.

## 2026-07-13 R79 offline paired-HFP checkpoint

R79 adds `PairedWholeScaledV1` and a generic cached loader transform from
`(column, block)` to `(pair, block, lane)`. Complete encoded blocks are copied
without byte changes; encoding, local nibble decode, and lane swizzle are not
altered. The derivative is identified by the complete source artifact and
records its source payload size.

The deterministic unit oracle covers all blocks, exact block bytes, order,
descriptor fields, and cache reuse. This is the immutable-layout prerequisite
for R33-style compact paired projection, not hardware or speed evidence.

## 2026-07-13 R80 paired compact-W4 projection admission

R80 consumes R79 pair-major weights on odd cores. One activation block feeds
two intact adjacent-column weight blocks and two accumulators; both stripes
scatter to the exact R65 inline stage. Even cores have no QKV projection image.

Queuing six odd-shim output tasks per outblock timed out. R65's proven
one-slice-per-channel cadence passes all 327,680 BF16 values bit-for-bit with
zero tail/padding corruption. Three warm medians are 0.818433, 0.833471, and
0.789289 ms (median 0.818433 ms), 67.7% slower than eight-column R65. Maximum
odd-core text is 11,872 bytes, leaving 4,512 bytes; even cores remain empty.
Admit R80 for capacity, not speed. Durable rows:
`results/r80-paired-w4-projection-20260713.csv`.

## 2026-07-13 R81-R83 paired projection/attention capacity tranche

R81 attaches the exact external Q/K/V pack boundary to R80. Every projection,
Q, and KV byte matches; three 100-command runs measure 1.8370, 1.8179, and
1.8397 ms (median 1.8370 ms). Odd/even text reaches 14,032/10,912 bytes. This
admits the topology for capacity but is 40.5% slower than R70.

R82 appends unchanged R76 attention and fails CDO generation. Odd columns 1/3
reach 22,416 bytes, exactly 6,032 over the 16 KiB store. The 8,384-byte growth
is 2,992 bytes of fully unrolled driver, 4,256 bytes of attention functions,
and 1,136 bytes of helpers. No hardware correctness or timing is claimed.

R83 replaces duplicated projection init/accumulate bodies with R70's exact
single-group ABI and obtains attention/slice trip counts through non-LTO
helpers. Peano then retains two attention calls and 12 finish calls instead of
32 and 36. Maximum odd/even text is 15,888/10,912 bytes, so the graph packages
with 496 bytes of odd-core headroom. Stage, Q, KV, and attention are byte-exact.
Three 100-command runs measure 4.1666, 4.0493, and 4.0535 ms (median 4.0535
ms), 6.8% slower than R78 and 24.3% slower than R76. Admit R83 only as the
first fitting paired projection/pack/attention image. Next, use the even-core
program budget for direct O projection/tails; do not append code to odd cores.

These are program-image results. The existing kernel parameter remains the
platform correctness workaround; LDS or tile-memory avoidance is neither the
fix nor a requirement. Durable rows:
`results/r81-paired-w4-projection-pack-20260713.csv`,
`results/r82-program-capacity-20260713.csv`, and
`results/r83-compact-paired-projection-pack-attention-20260713.csv`.

## 2026-07-13 R93 bandwidth-first native FFN activation checkpoint

R93 consumes canonical M256xK768 BF16 pre-FFN-normalized state and emits the
exact R25 resident-W4 input ABI: 108 6,656-byte blocks with a 6,240-byte dynamic
prefix replicated into the three N-macro consumer positions. All 589,824 int8
values match the CPU signed-FWHT oracle exactly, maximum scale error is `7e-9`,
and block/padded-row guards remain zero. Core text is 7,856-9,040 bytes.

The scalar/int8-sign version is rejected. Separate route-only and load-only
images prove the FIFO/DMA graph and source/parameter reads; restoring R47's
noinline BF16-vector sign/post-scale path restores exactness. This is Peano/AIE
code-generation evidence and does not imply LDS avoidance.

Three fresh 100-command runs measure 4.0618, 4.1218, and 4.1117 ms (median
4.1117 ms), only 0.263 GiB/s of physical source-plus-output traffic. Admit the
byte contract but reject a standalone producer context. Fuse preparation into
the first gate/up phase so it overlaps resident weight DMA and does not
materialize the three replicated consumer copies across a context boundary.
The existing kernel parameter remains the platform correctness workaround.
Durable rows: `results/r93-bf16-to-r25-activation-20260713.csv`.

## Tooling added
- `crates/hipfire-xdna/examples/npu_info.rs` — dumps resource_info (max/curr TOPS,
  clocks); source of the power finding. Run: `cargo run -p hipfire-xdna --example npu_info`.
- `tune.sh` extended usage confirmed (DTYPE_IN/OUT, configs). Results CSVs in `results/`.

## 2026-07-13 R94-R97 native-W4 fusion tranche

R94 replaces R93's scalar preparation body with the BF16-vector path while
retaining the exact R25 physical ABI. It has three one-code q differences out
of 589,824 values, `7e-9` maximum scale error, clean padding, and 6,608-7,792
bytes of core text. Three fresh 100-command runs measure 2.1320, 2.4486, and
2.1293 ms (median 2.1320 ms), a 48.1% reduction from R93 but still only 0.507
GiB/s. Admit the vector implementation for fusion; reject it as a standalone
phase. Durable rows: `results/r94-vector-activation-prep-20260713.csv`.

R95 unifies W4 init/accumulate behind one runtime flag. The full FFN retains
the R25 oracle (`0.99999662` cosine, `0.0041370` maximum absolute error) while
reducing every core image from 16,320 to 13,968 bytes. R96 shares the five down
fragment exchanges through one compact ring driver and retains the same oracle
at 12,944 bytes, recovering 3,440 bytes total. These are program-capacity
admissions, not sustained-speed claims.

R97 consumes canonical M256xK768 BF16 directly in the R96 graph. Its 216 DMA
tasks deliver every three-row/group source object and every weight object
byte-exactly. A widened physical oracle over all 256 rows finds one one-code q
difference in group 0, none in groups 1-2, and only float-rounding scale
differences. The initial full graph emitted NaNs because canonical gate packing
reused R25's `own`/`transit` fragment buffers while they still held the spilled
partial down accumulator. Dedicated 784-byte gate `own`/`transit` buffers fix
that state-lifetime alias. The complete hardware oracle then passes with gate
cosine `1.00000000`, final cosine `0.99998228`, maximum absolute error
`0.2597733`, and mean absolute error `0.03750710`; maximum core text is 15,456
bytes. A fresh 6.4095-ms dispatch is 39,941 M256 rows/s. R97 already inherits
R15's required `rounding=floor` and `saturation=none` kernel controls. A
20-command run still reaches the separate four-second command timeout cadence,
but bounded context recycling resolves that execution limit. Three independent
100-command runs with recycling every seven commands preserve the full oracle
at 6.4974, 6.4844, and 6.3388 ms (median 6.4844 ms, 39,479 M256 rows/s). Admit
sustained standalone R97 with this timeout mitigation; do not call it full-layer
or end-to-end encoder throughput.

Terminology is binding: the added kernel parameter is the platform correctness
workaround that stops the issue. It is not LDS avoidance and it is not R15's
rounding/saturation configuration. The R97 buffer separation fixes an ordinary
kernel alias. The command timeout, LDS/tile-memory use or avoidance, context
recycling, and these capacity refactors remain separate measured issues or
choices and none replaces the kernel-parameter workaround.

R98-R100 close the precision-preserving output boundary one operation at a
time. R98 converts each final F32 value in place to an interleaved compensated
BF16 pair without changing the 884,736-byte payload. It fits at 16,032 bytes,
retains `0.99998228` full-FFN cosine, and sustains a 6.5832-ms median (38,886
M256 rows/s). R99 changes only the DMA destination stride so those bytes occupy
the first 3,072 bytes of the resident tail's existing 4,608-byte combined row;
its 6.6000-ms sustained median is indistinguishable from R98.

R100 changes only the split-X tail reader to deinterleave each scalar pair. Its
standalone oracle passes at `0.99999861` cosine and `0.0039062` maximum error.
Early random/stale regions were a verifier bug: explicit synchronization of
host-written split-X input flushed the combined buffer but not the separate
residual. The helper now flushes both. NPU-to-NPU handoffs remain on the
no-host-sync path. No rung performs an immutable tensor-block reorder.

## 2026-07-13 R99/R100 reusable-layer integration checkpoint

The reusable native-W4 executor now selects R99's canonical-BF16 input and
combined-row compensated output ABI, then feeds R100's interleaved split-X
tail. The first layer-level run exposed a wiring bug rather than a kernel
failure: the temporary host pre-FFN normalization bridge wrote normalized H
over the attention buffer that still held architectural X. Preserving X in a
separate 442,368-byte shared BO raises tail cosine from `0.90738614` to
`0.99999873`, FFN cosine is `0.99997024`, and completed-layer cosine is
`0.99998514`.

A complete 24-layer resident-only OQ4 execution now finishes at 256 tokens, but
the current host bridge reaches only 291.6 input tok/s (878.003 ms total). Mean
per-layer phases are roughly 9 ms attention, 9-10 ms FFN, 3-4 ms tail, plus
about 10-12 ms preparation/output. This is correctness integration evidence,
not throughput admission: remove the host normalization/readback bridge before
using the result to judge the bandwidth-first kernel schedule.

The platform-issue workaround is the separately added kernel parameter, not
LDS avoidance and not R15's `rounding=floor`/`saturation=none` numerical
controls. X preservation, R97 fragment buffers, LDS/tile-memory choices, and
context recycling solve distinct problems.

## 2026-07-13 R105/R106 device unit-RMS rejection

R105 moves the mutable direct-X RMS normalization boundary onto AIE2P while
R106 reuses R99 with immutable pre-FFN norm folded into the loader-side W4
activation divisor. Standalone R105 passes at cosine `0.99999122`, maximum
absolute error `0.0078125`, and a 0.1295-ms median. The integrated pair also
passes layer 0: unit-RMS cosine is `0.99999269`, FFN cosine `0.99990930`, tail
cosine `0.99999862`, and completed-layer cosine `0.99996179`.

The full 24-layer result rejects the extra context boundary. Explicit cache
maintenance raises the nominally small R105 phase to 2.35-4.14 ms per layer;
the run takes 901.432 ms or 284.0 input tok/s at about 21 W and 13.5 tok/J,
worse than the R99/R100 host-bridge baseline of 878.003 ms and 291.6 tok/s.
R105/R106 is now diagnostic-only behind
`HIPFIRE_EMBED_UNIT_RMS_BRIDGE=1`. Continue with single-context RMS fusion;
do not infer that the standalone dispatch rate survives a context boundary.
Durable comparison rows:
`results/r106-unit-rms-layer-integration-20260713.csv`.

## 2026-07-13 R104 single-context inline-RMS admission

R104 is no longer a program-memory rejection. Vectorizing the mean multiply,
fusing inverse completion, sharing the runtime-stride FWHT body, and retaining
one full `3 x 768` BF16 X object per core reduce the normal `-O2` image from
18,352 bytes to exactly 16,384 bytes on every core. The full object is scanned
once for RMS and reused by group, reducing input DMA to one 442,368-byte
canonical-X pass instead of a prepass plus nine replays.

The standalone oracle passes at gate cosine `1.00000000`, final cosine
`0.99996707`, maximum absolute error `0.0737100`, and mean absolute error
`0.01499078`. One hundred commands with context recycling every seven preserve
the result at 6.5401 ms. The default layer-0 integration reaches FFN cosine
`0.99991494`, tail cosine `0.99999844`, and completed-layer cosine
`0.99996658`. A current default 24-layer run measures 894.222 ms / 286.3 input
tok/s at 18.07 W / 15.8 tok/J.

Package variance is material, so admission uses paired controls: R99 takes
892.708 and 909.986 ms, while alternating R104 runs take 859.599 and 869.015
ms. The paired means are 901.347 ms versus 864.307 ms, a 4.1% latency reduction
and roughly 4.3% throughput gain. R104 is therefore the native-W4 default when
the artifact is present; R105/R106 remains diagnostic-only. The remaining
9-12 ms per-layer preparation/output boundary is now larger than the removed
pre-FFN host bridge.

Compiler post-link variants at 15,824-15,840 bytes returned all-zero output
after the command timeout and remain rejected. The existing kernel-parameter
workaround remains mandatory and distinct from LDS placement, R97 fragment
lifetime, and context recycling. Durable rows:
`results/r104-inline-rms-full-object-20260713.csv`.

## 2026-07-13 R108/R109 direct residual handoff

R107 first tried to add R48's residual copy to R47's RMS pass, but four new
residual output FIFOs per memory tile exceed the tile DMA-channel budget. R108
instead reuses attention's existing residual FIFO and reads the already-rounded
high BF16 plane directly from the completed-state buffer. A sixth-argument form
then hit amdxdna's five-argument DPU register-map limit. R109 resolves the ABI by
placing completed BF16x2 at the front of the attention input BO and writing R34
activation records into a disjoint suffix through one in-place argument. A
duplicate source/destination import was also rejected with `EALREADY`; the
final graph imports the BO once.

Layer 0 passes at FFN cosine `0.99991644`, tail cosine `0.99999886`, and
completed-layer cosine `0.99996836`. The traced preparation boundary falls from
roughly 9-12 ms to 7-9 ms per layer. Alternating same-lock trials measure R48 at
813.690/816.898 ms and R108/R109 at 801.536/807.908 ms. Their paired means are
815.294 and 804.722 ms, a 1.30% full-model latency reduction. Package-power
samples do not yet show a tokens/J win, so admit the latency/correctness change
without an energy claim. The existing kernel-parameter workaround remains
independent of this DMA/ABI change and of LDS placement. Durable rows:
`results/r108-r109-direct-residual-20260713.csv`.

## 2026-07-13 R110 generic Opus format refresh

The R108/R109 completed-layer path remains format-generic. Locked M256 layer-0
oracles pass for native OQ8, calibrated OQ8+, freshly generated OQ8++, and the
arbitrary compact OQ6.5 mix; completed-layer cosine ranges from `0.99996270` to
`0.99997103`. The OQ8++ artifact was generated offline from BF16 plus the
unified calibration package, with 168/168 LDLQ projection packs successful.

Full 24-layer runs reach `296.0`, `299.1`, `305.1`, and `294.9` input tok/s,
respectively. OQ8-family BF16 embedding cosine is `0.99547863-0.99584466`;
OQ6.5 reaches `0.95821863`, so it proves arbitrary mixed-width execution but
not OQ8-level quantization quality. These rows close the refreshed generic
execution matrix, not the 10k performance target. Durable rows:
`results/r110-generic-opus-formats-20260713.csv`.

## 2026-07-13 R111 one-pass completed-state preparation

R111 keeps R109's in-place completed-prefix/R34-suffix ABI and exact pack math,
but copies one 3,072-byte completed row into tile memory, releases its input
FIFO immediately, and produces all three K256 chunks from that local row. The
884,736-byte completed allocation contains 32 padded rows; each physical sweep
reads only 786,432 active bytes. R111 reduces four sweeps (3,145,728 bytes) to
one, saving 2,359,296 active bytes and 24 completed-input DMA task lifecycles.
Core text remains 9,072-10,592 bytes.

Two failures were useful. Holding the input FIFO across RMS plus all three pack
calls reproduced R54's schedule failure. The first copy-and-release image then
corrupted only packer-owned group-1/group-2 Q bytes because those chunks were
32-byte aligned but the assembly helper loaded 64 bytes at a time. A 32-byte Q
copy matches the allocator guarantee and restores the oracle: five one-code Q
differences, maximum Q delta 1, and `7e-9` maximum scale error.

The standalone paired mean improves only from 5.1424 to 5.0949 ms (0.9%), so
full-model admission used four counterbalanced pairs with three encodes per
process. R109 averages 760.323 ms, 336.7 input tok/s, and 17.83 tok/J; R111
averages 749.409 ms, 341.6 tok/s, and 18.10 tok/J. R111 wins every latency pair
and preserves BF16 embedding cosine `0.92839295`, so it is the default when its
artifact is present, with R109 retained as fallback. The small 1.44% latency
gain despite removing 2.36 MB per layer confirms that this seam is not governed
by external-memory bandwidth alone. The next rung should target context and
route consolidation, with R100 tail fusion the plausible seam; R108 has only
16 bytes of program headroom and is not a viable fusion host.

The added kernel parameter remains the platform workaround. It is not LDS
avoidance, not R15's numerical mode, and not R111's 32-byte alignment fix.
Durable rows: `results/r111-one-pass-next-prep-20260713.csv`.

## 2026-07-13 R112 fusion-ready post-FFN tail topology

R112 transposes only mutable token ownership: each core processes eight
contiguous tokens across four two-token phases, while strided DMA gathers and
scatters canonical rows. The first split-X design added a second memory-tile
broadcast and is rejected because it exceeds the tile's output DMA-channel
budget. The admitted design places canonical token-major X in the third plane
already reserved by R99's 4,608-byte mutable row. Interleaved FFN high/low and
X then share the existing memory-tile route; no immutable block is reordered.

The active input total is unchanged at 1,179,648 bytes: R100 reads 786,432 FFN
bytes plus 393,216 split-X bytes, while R112 reads their joined row state.
Maximum core text falls from 4,208 to 3,696 bytes and all 24 horizontal core
flows disappear. The locked hardware oracle remains cosine `0.99999861` with
`0.0039062` maximum error. R112 wins four counterbalanced 100-command pairs;
mean dispatch falls from `0.324965 ms` to `0.218271 ms`, or 32.84%.

Admit the topology as the R111 fusion seam. It does not yet include next-layer
RMS/pack math and is not a full-model speed claim. The added kernel parameter
remains the platform workaround; this result neither uses nor establishes LDS
avoidance. Durable rows: `results/r112-fusion-ready-tail-20260713.csv`.

## 2026-07-13 R113 fuses next-layer RMS/pack into the tail

R113 preloads each core's three K256 parameter groups, keeps R112's canonical
joined-row DMA and eight contiguous tokens per core, and applies R111's exact
RMS/AWQ/FWHT/int8 pack to each still-local two-row tail output. It removes the
separate 786,432-byte completed-state input pass without adding an immutable
layout conversion or a new memory-tile channel.

The literal eight-row local completed buffer is rejected because its extra
24,576 bytes do not fit bank allocation. Phase-local packing fits all cores at
9,984 text bytes. A second failure exposed the shim output queue: retaining
four completed-output tasks plus three diagnostic tasks leaves every group-2
slot zero. Launching all stripes, retiring one completed task per stripe, and
then queuing diagnostics passes without serializing the array. An intermediate
per-stripe await also passed but took 13.3578 ms and is rejected.

The hardware oracle reaches tail cosine `1.00000000`, maximum error
`0.0000310`, three one-code Q differences, Q delta 1, and `7e-9` scale error.
Four live 50-command samples average 5.056325 ms. Live R112 and R111 controls
average 0.236051 and 5.024850 ms, respectively, or 5.260901 ms unfused. R113
saves 0.204576 ms (3.8886%). This is a useful but modest context/traffic win;
RMS/FWHT pack compute now dominates. R114 tests whether the R34 suffix can be
assembled without restoring the completed-state pass.

One fresh process returned an all-zero tail between otherwise passing runs;
four immediate fresh contexts and the complete repeated series passed. Record
it as a context-transition diagnostic. The added kernel parameter remains the
platform workaround that stops the separate platform issue. It is not LDS
avoidance; LDS placement, output-queue capacity, R111 alignment, and context
lifetime remain independent. Durable rows:
`results/r113-tail-next-pack-fusion-20260713.csv`.

## 2026-07-13 R114 rejects in-context compact R34 assembly

R114 explored four ways to turn R113's per-core chunks into an R34 activation
boundary without restoring the completed-state pass or materializing five
N-macro replicas. Logical-owner stream chains and physical column-major stream
chains each failed after routing search with `Unable to find a legal routing`.
Adjacent neighbor-memory ObjectFIFOs compiled until a new shim output route was
added, then failed with the same routing error. Reusing a completed-output task
with a zero destination stride was also rejected because DMA strides must be
positive.

The final split-plane variant reuses the existing completed-output route and
builds in about nine seconds. It emits a compact 589,824-byte Q/scale ABI with
no 16 KiB padding and no fivefold N-macro replication; maximum core text is
11,200 bytes. Hardware rejects it, however: after a good completed-tail result,
the pack has 107,811 mismatches, maximum Q delta 254, and maximum scale error
0.034057196. Mismatches are spread across all K256 groups and local-memory
owner positions, so the compact assembly/mapping is incorrect but the current
evidence does not isolate one neighbor chunk as the cause. R114 is not admitted.

The bandwidth-first boundary remains useful. R113's already-correct diagnostic
output is itself a compact per-core ABI: 589,824 padded bytes containing 199,680
unique chunk bytes. The next resident R34 GEMM should consume these chunks
directly and reuse each chunk across five N-macros, avoiding both in-tail
assembly and the canonical 2,949,120-byte replicated activation materialization.
That consumer must preserve immutable `.rdna2.hfp` tensor-block order.

One intervening fresh context returned an all-zero completed tail; an immediate
repeat returned the distributed R114 pack failure. This remains separate
context-transition evidence. The added kernel parameter is the workaround that
stops the platform issue. It is not LDS avoidance; local-memory placement,
R111 alignment, routing, and context lifetime are independent variables.
Durable rows: `results/r114-r34-compact-boundary-20260713.csv`.

## 2026-07-13 R115 admits the first direct compact consumer

R115 avoids R114's incorrect assembly entirely. Each of the 32 cores consumes
its admitted R113 eight-token group-0 chunk and computes one scaled K256-by-N16
int8 matrix stage. The same offline-packed weight record is broadcast to every
token owner, and the 32 outputs scatter directly to canonical `[256,16]` f32.
There is no producer-side 24-token assembly and no N-macro activation replica.

The first rung intentionally retains R113's diagnostic padding: it reads
196,608 physical activation bytes containing 66,560 unique chunk bytes. This is
a mapping/math isolation step; later DMA gathering may remove padding only
after preserving the oracle. The image fits at 1,692 maximum core text bytes.

Locked hardware parity reports zero mismatches and `2e-9` maximum absolute
error. Six fresh 1,000-dispatch processes all pass at 0.090826-0.094342 ms,
averaging 0.092506 ms. One preceding fresh process returned mostly zero output;
the immediate six-process series passed, so this remains context-transition
evidence rather than an R115 math, LDS, or platform-workaround diagnosis.

Admit the direct consumer and group-0 compute stage, not full K or N. Next add
groups 1-2 with local f32 accumulation against the same compact ABI. Immutable
weights remain loader/offline ordered with the `.rdna2.hfp` tag. The added
kernel parameter remains the separate platform workaround that stops the
platform issue; it is not LDS avoidance. Durable rows:
`results/r115-direct-compact-group-n16-20260713.csv`.

## 2026-07-13 R116 admits direct compact full-K math, with context caveat

R116 consumes all three R113 K256 chunks per token owner and accumulates one
N16 tile locally in f32. It reads 589,824 physical activation bytes containing
199,680 unique chunk bytes, exactly one fifth of the canonical 2,949,120-byte
five-N-macro materialization. It preserves zero N-macro replicas and performs no
immutable tensor reorder.

The first two-group oracle isolated a deterministic error to columns 0-7 of
group 1; columns 8-15 and group 2 were exact. Padding 4,160-byte weight records
to 4,224 bytes for 128-byte starts did not change it. Splitting group DMA tasks
and unrolling the core loop instead produced all-zero output. All three are
rejected. The working form stages the prior 8x16 output tile into a 512-byte
local array before the next MMUL, making the output reload dependency explicit.
K512 with unit scales is bit-exact; K768 passes with zero mismatches and `4e-9`
maximum error. Maximum core text is 2,220 bytes.

Eight passing fresh 1,000-dispatch processes average 0.096384 ms. Two other
fresh processes returned all-zero output, followed by four passing fresh
processes. Admit the full-K compact-consumer math but not context-stable runtime
selection. Extend N only after retaining this byte oracle. The added kernel
parameter remains the platform workaround that stops the separate platform
issue. The prior-output staging fix is not that workaround, and none of these
results establishes LDS avoidance. Durable rows:
`results/r116-direct-compact-fullk-n16-20260713.csv`.

## 2026-07-13 R117 doubles N work without increasing activation traffic

R117 widens the direct compact consumer from N16 to N32. Every K-tile
activation load feeds four 8-column MMUL halves, while the graph still reads
589,824 physical activation bytes containing 199,680 unique bytes and
materializes zero N-macro replicas. Immutable N32 weight records remain
offline/loader `.rdna2.hfp` data.

Both N16 halves pass: zero mismatches and `3e-9` maximum error. Core text grows
from 2,220 to 3,192 bytes. Eight passing fresh 1,000-dispatch processes average
0.086916 ms, 9.82% below R116's 0.096384 ms passing mean while doubling useful
N work. This is direct evidence that dispatch/fixed traffic dominates this
small consumer and that increasing arithmetic per activation load is the right
direction.

Two fresh contexts returned all-zero output before the eight passes. Admit the
N32 math/ABI but not context-stable selection. The next scalable step is not a
larger monolithic output tile: stage all three 2,080-byte chunks once per core,
then stream multiple N32 weight/output records while activation DMA remains
fixed. The added kernel parameter remains the distinct platform workaround;
neither the wider MMUL body nor local staging is LDS avoidance. Durable rows:
`results/r117-direct-compact-fullk-n32-20260713.csv`.

## 2026-07-13 R118 stages activations once and emits N64

R118 copies each core's three compact chunks into local storage once, releases
the activation FIFO, and streams two full-K N32 weight/output records. The first
6,240-byte stage placed group 1 at offset 2,080, only 32-byte aligned for the
64-byte MMUL load. A 2,112-byte group stride (6,336 bytes total) restores
64-byte alignment: block 0 then passes exactly. This is an explicit local-load
alignment fix, not the platform workaround and not LDS avoidance.

The first single-task N64 output descriptor compiled after treating its outer
dimension as the BD repeat, but it transferred only block 0. Two explicit
output tasks fit the known queue depth and make both blocks pass. Final parity
is zero mismatches with `5e-9` maximum error; maximum core text is 3,736 bytes.
Activation traffic remains one 589,824-byte physical pass carrying 199,680
unique bytes, with zero N-macro replicas.

Nine passing fresh 1,000-dispatch processes average 0.106058 ms, only 22.0%
above R117's 0.086916 ms while doubling useful N work. One other fresh context
returns all-zero output. Admit the activation-once N64 math/topology, not a
context-stable default. Next test the task `repeat_count` attribute together
with the outer DMA tiling dimension so many N32 outputs can be drained without
hundreds of live tasks. The added kernel parameter remains the separate
platform workaround. Durable rows: `results/r118-staged-fullk-n64-20260713.csv`.

## 2026-07-13 R119 admits repeated output-task scheduling

R119 changes no compute, activation staging, weights, or output layout. It adds
`repeat_count=1` to the single output task whose outer DMA tiling dimension is
two. That combination consumes both N32 objects and scatters canonical N64;
the outer dimension alone in R118 consumed only the first object.

Parity is zero mismatches with `5e-9` maximum error. Eight passing fresh
1,000-dispatch processes average 0.102308 ms, 3.54% faster than R118's explicit
two-task mean of 0.106058 ms. Two other contexts return all zeros. Admit the
repeat schedule for growing the N32 block count, not context-stable runtime
selection. Activation remains one 589,824-byte pass and N-macro replicas remain
zero. The platform-workaround kernel parameter, local-memory choices, and this
DMA scheduling rule remain separate. Durable rows:
`results/r119-repeat-output-task-20260713.csv`.

## 2026-07-13 R120 gates four repeated N32 blocks

R120 parameterizes the R119 graph to N128. The output row strides scale with N,
the outer DMA tiling dimension grows to four, and the task repeat count grows to
three. The compute kernel, 6,336-byte aligned local activation stage, and one
589,824-byte activation pass remain unchanged. The W8 diagnostic payload is
798,720 bytes and the f32 output is 131,072 bytes.

All four blocks pass together with zero mismatches and `7e-9` maximum error.
Four passing fresh 1,000-dispatch contexts average 0.115102 ms. Six other fresh
contexts return the same whole-output zero symptom seen in earlier rungs, so
R120 admits the widened topology rather than context-stable selection. Maximum
core text is 4,312 bytes. The added kernel parameter remains the platform
workaround; the local stage and repeated DMA schedule are independent. Durable
rows: `results/r120-staged-fullk-n128-20260713.csv`.

## 2026-07-13 R121 admits the complete N1280 projection schedule

R121 scales the same graph to all 40 N32 blocks of a 768x1280 projection. It
streams 7,987,200 W8 diagnostic weight bytes and scatters 1,310,720 f32 output
bytes while activation traffic remains one 589,824-byte pass containing
199,680 unique bytes. No N-macro activation replicas or kernel-side immutable
tensor reorder are introduced. The output task uses an outer dimension of 40
and `repeat_count=39`; tile-local activation storage is unchanged.

The full 256x768 by 768x1280 byte oracle passes with zero mismatches and `6e-9`
maximum error. All ten fresh 1,000-dispatch contexts pass. Their range is
0.319049-0.325542 ms and mean is 0.320640 ms, equivalent to about 798,402 M256
projection rows/s and 30.84 GB/s over weight, activation, and output bytes.
Maximum core text is 3,848 bytes.

This admits the full-width projection schedule, not an end-to-end encoder
throughput claim. The next rung must feed generic runtime Opus records (OQ4,
arbitrary mixed OQ, OQ8, and +/++ metadata) through this topology and measure
the complete resident layer. The added kernel parameter remains the platform
workaround; LDS placement, output repetition, and payload encoding remain
separate. Durable rows: `results/r121-staged-fullk-n1280-20260713.csv`.
## 2026-07-15 R123 core-stationary FFN rejection

Object-FIFO `iter_count` compile/lowering evidence did not survive hardware:
the repeated memtile MM2S chain consumed a source lock that shim S2MM produced
only once. Normal priming returned zeros; bypassing prime exposed exactly one
valid macro. `repeat_count` exhausted 24 memtile BDs.

A core-stationary replacement is correct. M512 `[X;X]` is bit-exact with M256,
using one 28-record weight sequence per column and a rolling three-output task
window. It is not faster: 20.111 ms at M256 and 33.500 ms at M512 versus the
replicated-weight 10.520/19.929 ms path. The saved weight DMA does not repay the
weight-major routing/accumulation overhead. See
`results/r123-weight-stationary-m-scaling-20260715.csv`.

## 2026-07-15 R124 direct-X M512 contract

Direct-X batching needs document-padded physical rows, not a contiguous M512
buffer. Each M256 document now occupies one M288 slot and has its own physical
inverse-RMS record plane, eliminating selector aliasing at rows 256-287. The
host also scales the canonical input allocation, which previously remained one
M288 buffer at batch two.

The r55 M256/M512 images compile and pass the combined absolute/self-consistency
gate. Direct M256 versus canonical measures cosine 0.99988810 and maximum
absolute error 0.0147171; both M512 documents are bit-exact with direct M256.
Timing is 8.366/16.489 ms, only 1.01x row-throughput gain. This admits the
direct-X ABI seam but confirms that replicated-weight FFN batching remains
row-linear. Continue with tail/next-prep consolidation. See
`results/r124-direct-x-batched-ffn-20260715.csv`.

## 2026-07-15 R125 unpins the tail/next-prep boundary

Doubling the R100 tail's phase tasks exhausted shim BD IDs. Keeping the four
M256 descriptors and adding a document dimension plus `repeat_count` compiles
and is correct: M512 retains cosine 0.99999861, maximum absolute error
0.0039062, and bit-exact duplicated documents. Timing is 0.381335/0.600561 ms,
a 1.27x row-throughput gain.

R111 next-prep repeats its eight-row transaction for each padded document and
writes separate canonical R34 prefix regions. Duplicated outputs are
byte-identical; M512 has ten one-code Q differences, maximum Q delta 1, and
`7e-9` scale error. Its 5.0606/10.0303 ms timing is row-linear. The ABI seam is
admitted, while performance says future work should avoid materializing the
replicated R34 activation surface. See
`results/r125-batched-tail-next-prep-20260715.csv`.

## 2026-07-15 R126 proves block-diagonal attention scheduling

R27 can process multiple independent M256 documents in one command by
resetting online-softmax state per query group and feeding each group only its
document's sixteen K/V blocks. Concatenating packed Q buffers behind one row
descriptor is invalid because the physical layout is row-major outside query
groups; that attempt produced about 0.99867 cosine in both documents.

Explicit Q, K/V, and output tasks per document pass with distinct inputs.
Doc0/doc1 reach 0.99999410/0.99999464 cosine and 0.0002527/0.0002543 maximum
error, ruling out cross-document attention. Timing is 1.4303 ms at M256 and
2.8932 ms at M512, so the math remains row-linear. Apply this descriptor
topology to the fused resident attention image next. See
`results/r126-segmented-bf16-attention-20260715.csv`.

## 2026-07-15 R127/R128 fused and end-to-end M512 batching

R108 now repeats its complete B1 task schedule inside one runtime command, with
private Q/K/V and scratch regions per document. A first per-document R34 image
passed the local attention oracle but failed the downstream direct-X handoff.
The corrected form scatters final padded X rows and inverse metadata into the
compact multi-document prefix consumed by r55 and r46, while keeping transient
R34 scratch disjoint.

Distinct fused documents are bit-exact against their separate M256 hardware
oracles. Fused M256/M512 timing is 6.405/11.615 ms (1.10x row throughput).
The full oq8 encoder now accepts `[0,256,512]` segment offsets, checks matching
batch geometry across every resident component, and pools the two final states
independently. One M512 encode matches separate M256 embeddings at mean/minimum
cosine 0.99999845/0.99999797 and maximum error 0.00025596. Ten reused commands
retain those values.

Matched three-run timing is 774.491 ms per M256 document versus 633.625 ms per
M512 document (1267.250 ms per command), a 1.22x throughput gain. The full path
is admitted, but descriptor replication does not realize the original
fixed-floor estimate; further scaling needs actual weight/dataflow reuse. See
`results/r127-fused-segmented-attention-20260715.csv` and
`results/r128-full-batched-encode-20260715.csv`.
