# NPU bandwidth-first kernel development plan

Date: 2026-07-12
Target: Strix Halo XDNA2/AIE2P
Foundation: R1/R56 external-feed characterization

## Objective

Build production NPU kernels as an incremental bandwidth ratchet. Begin with
the measured host-memory-to-array DMA path, then add one dataflow or compute
stage at a time. Every stage keeps the preceding modes runnable so the first
loss of useful bandwidth, overlap, or correctness is attributable to one
change.

The measured reference points are:

- one active compute-tile receive stream: 14.4 GB/s;
- eight-column aggregate external feed: about 56.5 GB/s;
- current resident EmbeddingGemma attention and FFN useful packed-weight feed:
  about 0.9 GB/s.

The immediate goal is not a complete replacement GEMM. It is to close that gap
without losing the ability to explain where each byte and cycle went.

## Non-negotiable tensor-layout rule

Tensor-block layout conversion is **never part of an NPU kernel**. It runs once:

1. preferably when producing the on-disk architecture-packed weight file; or
2. as a loader operation that converts once, persists/reuses the result, and
   completes before kernel timing begins.

The converted file carries the literal `.rdna2.hfp` filename suffix/tag. It is
an architecture-packed derived tensor-block layout, not the canonical model
container and not a new runtime quantization mode. The loader must bind it to
the source tensor with shape, dtype/quant encoding, layout-version, and source
content hash metadata. A stale or mismatched derivative is rejected rather
than silently regenerated inside dispatch.

Examples of the intended relationship:

```text
canonical model/container              immutable source of truth
<tensor-or-cache-stem>.rdna2.hfp        preconverted kernel-consumption layout
```

If the derivative is absent, policy may choose either an explicit loader-time
conversion or a clear admission failure. It must not fall back to an in-kernel
reordering of tensor blocks. Mutable activations may still use DMA addressing
for placement, but immutable weight blocks arrive in final stream order.

This rule does **not** prohibit representation-local decode inside a kernel.
Packed Opus W4/OQ4 still carries two signed 4-bit values per byte. The kernel
may split low/high nibbles, sign-extend them, interleave them into the vector
lane order required by MMUL, and apply per-group scales. That work is part of
quant decode and must be measured explicitly. It may not change which tensor
block is streamed next, transpose macro tiles, or rebuild a different global
weight traversal order.

## Incremental kernel ladder

Keep each accumulated mode selectable at build time and give every mode the
same production-shaped external arguments:

| step | mode | behavior added | required evidence |
|---:|---|---|---|
| 0 | `FEED_ONLY` | SHMEM -> shim DMA -> trivial drain/checksum | external wire and payload roof |
| 1 | `PRODUCTION_DMA` | real projection extents, padding, BD count, and transfer order | useful bytes versus transferred bytes |
| 2 | `MEMTILE_STAGE` | memory-tile ping-pong buffering and forwarding | external ingress plus memory-tile service rate |
| 3 | `MEMTILE_BROADCAST` | one weight stream fanned out to multiple compute tiles | physical ingress and logical reuse amplification |
| 4 | `PRECONVERTED_LAYOUT` | admit and validate `.rdna2.hfp` weights already arranged in final tensor-block stream order | byte-for-byte layout/version/checksum validation and unchanged feed |
| 5 | `NIBBLE_DECODE` | split, sign-extend, and lane-swizzle packed Opus W4/OQ4 nibbles without changing tensor-block order | decoded checksum, payload rate, and decode backpressure |
| 6 | `COMPUTE_STAGE1` | first vector/MMUL operation with resident partial accumulators | feed/compute overlap and first compute backpressure |
| 7 | `COMPUTE_STAGE2` | K continuation, cascade, or partial reduction | accumulator/cascade stalls and achieved TOPS |
| 8 | `SCALE_AND_EPILOGUE` | per-group scale application and required output conversion | useful TOPS and numerical parity |
| 9 | `OUTPUT_DRAIN` | production output placement and drain | full single-projection latency |
| 10 | `PERSISTENT_PIPELINE` | next projection/stage without an external intermediate | eliminated bytes/dispatches and model-level gain |

Step 4 is an admission test, not a tensor-reordering stage. Its kernel should be
structurally equivalent to step 3 apart from consuming the final packed block
order. Step 5 deliberately adds the required in-core nibble/lane swizzle as a
separate measurable cost. Reordering macro tiles or tensor blocks in an AIE
core, memory-tile program, or dispatch loop is a plan violation.

## Measurements and invariants

Report both physical and useful traffic:

```text
wire_GB/s    = bytes actually requested by DMA / receive span
payload_GB/s = unique useful weight bytes / receive span
duplication  = DMA weight bytes / unique useful weight bytes
logical_GB/s = useful weight bytes consumed across all reuse sites / span
```

Every row also records:

- mode, source commit, `.rdna2.hfp` layout version and source hash;
- M/K/N, quant encoding, padding, columns, tile size, FIFO depth, and BD count;
- receive-port running, stalled, and idle cycles per traced column;
- compute active/stall cycles and cascade events when available;
- external bytes, useful bytes, output bytes, MACs, TOPS, and wall time;
- numerical checksum for feed-only stages and reference error/cosine once
  computation is present;
- H clock, power mode, XRT/driver/firmware, and concurrent CPU/GPU load state.

Bandwidth is not required to remain at 56.5 GB/s after the kernel becomes
compute-bound. The invariant is **no unexplained loss**:

- while feed-bound, retain at least 85% of the prior stage's payload bandwidth;
- when feed bandwidth falls, counters and the roofline must identify the new
  consumer-side bottleneck;
- accept a lower feed rate only when useful TOPS or end-to-end time improves;
- reject a stage that lowers both useful feed and useful compute throughput;
- never hide padding, replication, or rereads inside the logical-byte count.

## First experiment matrix

Use one real EmbeddingGemma projection and run every accumulated mode at:

```text
M = 64, 128, 256, 512, 1024
```

Start with the exact packed weight payload and transfer geometry used by the
resident runtime. Run one, two, four, and eight columns for steps 0-4, then the
production column count for compute stages. Use warm repeated trials and store
all durable rows under `benchmarks/npu_gemm_tuning/results/`.

The first implementation tranche ends at step 4. It must answer:

1. Does production transfer geometry retain the R56 external-feed roof?
2. What is the sustained memory-tile ingress/forwarding rate?
3. How much logical bandwidth does broadcast create per external byte?
4. Does consuming preconverted `.rdna2.hfp` weights preserve step-3 feed?
5. Which padding, duplication, or BD choices reduce payload efficiency?

Only after those answers are durable should `NIBBLE_DECODE`, followed by
`COMPUTE_STAGE1`, be added.

## Implementation work items

1. Extend the R56 harness with production M/K/N and payload accounting without
   changing its trace-derived timing method.
2. Define and version the `.rdna2.hfp` layout manifest: tensor identity, source
   hash, quant encoding, dimensions, padding, tile order, scale order, and byte
   length.
3. Add the offline producer and/or one-time loader tensor-block conversion.
   Test cache hit, cache miss, stale source hash, truncated derivative, and
   wrong layout version.
4. Add memory-tile ping-pong and broadcast stages with checksum consumers.
5. Emit one stable CSV schema shared by every mode.
6. Add and measure `NIBBLE_DECODE` after steps 0-4, then add `COMPUTE_STAGE1`
   only after nibble decode meets its bandwidth and correctness gates.

## Completion gates

- Reproduction script and committed raw CSV are sufficient to recreate every
  claimed transition.
- Step 4 contains no tensor-block reordering in generated core or DMA program.
- Step 5 may perform only representation-local nibble/lane swizzling; decoded
  values and global block traversal must match the `.rdna2.hfp` manifest.
- The `.rdna2.hfp` derivative is reproducible and cryptographically tied to its
  canonical source.
- Feed-bound stages retain at least 85% of their predecessor's payload rate or
  carry a documented, counter-supported failure result.
- Compute stages pass exact/integer parity where applicable and the established
  cosine/error gate where scaling changes representation.
- Model integration requires a measured end-to-end improvement; isolated wire
  bandwidth alone is not admission evidence.

## Steps 0-4 checkpoint (2026-07-12)

The first bandwidth ratchet is implemented under
`benchmarks/npu_gemm_tuning/r57/` and passes on `halo`. Median exact
four-column results are 43.577 GB/s for production DMA, 43.536 GB/s through a
one-to-one memory-tile stage, 43.198 GB/s with four-way memory-tile broadcast,
and 43.251 GB/s while consuming a real preconverted `.rdna2.hfp` file. The
preconverted stage retains 99.25% of direct DMA and 99.88% of broadcast, while
the eight-column control remains near the 56-GB/s R56 roof.

The loader now writes an atomic 128-byte version/hash header plus the exact
8,192,000-byte R34 payload. It rejects source-SHA or payload-SHA mismatch. A
real `EmbeddingGemma-300M.npu.oq4.hfq` run produced 24 layer files under
`~/.hipfire/npu/prepacked/`, completed the resident M256 model, and a second run
consumed the files without changing their size, mtime, or SHA-256.

Scope warning: R34 currently expands an OQ4 source to its dense-W8/BF16 physical
execution payload before the one-time pack. This proves the block-order cache
and bandwidth invariants, but does not satisfy the packed-OQ4 step. The next
tranche must define a new encoding/version whose `.rdna2.hfp` payload retains
two signed nibbles per byte, then add `NIBBLE_DECODE` without changing tensor
block order.

## Packed OQ4 and NIBBLE_DECODE checkpoint (2026-07-12)

R58 closes that scope warning. `hipfire-xdna::opus_hfp` defines version 2 of a
format-generic, source-SHA- and payload-SHA-validated `.rdna2.hfp` container.
Its descriptor records Opus encoding, quant type, M/K/N, column and macro
geometry, tile/data/scale offsets, and payload length. The production
`NpuGemmWholeScaled` loader now creates or reuses these artifacts before device
upload. W4 remains two signed nibbles per byte; only global tensor-block order
is converted offline. OQ8 uses the same API and container with a distinct
encoding value. Mixed W4-plus-overlay has a reserved generic encoding and will
need a segmented full-K payload before it can use this physical schedule.

A real resident-only load of `EmbeddingGemma-300M.npu.oq4.hfq` created 216
packed files (nine individual/combined projection forms for each of 24 layers)
under `~/.hipfire/npu/prepacked/`. The layer-0 combined QKV file is 2,359,488
bytes: a 192-byte header plus 2,359,296 payload bytes for eight columns, six
outblocks, three K groups, and 16-KiB physical blocks. A second complete loader
pass preserved its mtime, size, and SHA-256 exactly. The 216-file count exposes
a packaging inefficiency: fallback individual q/k/v and gate/up forms coexist
with the selected qkv and gate-up forms. Final packaging should retain only the
admitted execution graph's immutable forms rather than shipping every fallback
derivative.

`benchmarks/npu_gemm_tuning/r58/` runs an identical eight-column, four-row
memory-tile-broadcast topology in `PACKED_FEED_ONLY` and `NIBBLE_DECODE` modes.
The decode kernel uses AIE2P's native signed `int4 -> int8` unpack, visits every
packed data vector, returns 64 lane sums, and exposes one decoded real-artifact
vector for byte-exact low/high/sign validation. Three locked trials per mode
produced:

| mode | median wire GB/s | median packed-data GB/s | retention |
|---|---:|---:|---:|
| `PACKED_FEED_ONLY` | 56.173 | 42.130 | reference |
| `NIBBLE_DECODE` | 56.061 | 42.046 | 99.80% |
| `COMPUTE_STAGE1` | 55.933 | 41.949 | 99.77% vs decode |
| `COMPUTE_STAGE2` | 39.919 | 29.939 | 71.37% vs stage 1 |

All decode trials passed both hardware oracles. Four-way fanout corresponds to
about 336.4 GB/s median logical decoded-byte consumption without increasing
external wire bytes. Within run variance, representation-local nibble decode
adds no receive-side backpressure and clears the 85% gate decisively.

`COMPUTE_STAGE1` is also implemented. It performs one native
`mmul<4,16,16,int8,int4>` for every 128 packed weight bytes with resident
all-one activations, accumulates exact int32 results, and passes the CPU MMUL
oracle in every trial. It sustains 55.933 GB/s median wire bandwidth (99.77%
of decode and 99.57% of feed-only) while executing 2.685 TOPS of deliberately
minimal M=16-equivalent work.

`COMPUTE_STAGE2` grows this to the production 6x16 MMUL schedule per core: six
distinct four-row activation blocks reuse each W4 slab across a complete
K=256 group. Exact int32 parity still passes. This is the first observed
feed-to-compute transition: wire bandwidth falls to 39.919 GB/s (71.37% of
stage 1), receive-port stalls rise from zero to 57.45%, and useful compute rises
4.28x to 11.497 TOPS. The bandwidth drop is accepted under the plan because it
is counter-explained compute backpressure and accompanies a large useful-TOPS
gain; it is not evidence that the external memory path regressed. Step 8 is now
admitted: apply per-group scales and preserve distinct output accumulators,
then measure numerical parity and final output placement. Durable raw rows are in
`benchmarks/npu_gemm_tuning/results/r58-nibble-decode-20260712.csv`.

## R60 resident-context compute checkpoint (2026-07-13)

The ladder now crosses the actual resident boundary rather than a standalone
projection wrapper. R60 consumes one R34 `ResidentContextBundleV1` BO and the
unchanged R34 shared activation BO. The production CPU oracle reuses the R30
packer byte-for-byte; immutable QKV/O ordering exists only in `.rdna2.hfp`,
while the kernel performs only the required local 8+8 activation join and
signed-W4 MMUL operand handling.

Three locked trials per stage give:

| accumulated stage | median bundle GB/s | range | retention vs 55.884 GB/s R59 |
|---|---:|---:|---:|
| first 4x16x16 MMUL | 53.838 | 53.829-54.112 | 96.34% |
| full K=256 for one group | 54.252 | 53.870-54.635 | 97.08% |
| all three groups plus f32 scale epilogue | 50.912 | 50.605-51.092 | 91.10% |

All 512 outputs per run pass their exact-int32 or scaled-f32 oracle. The final
stage performs 48 MMULs per checked tile and exposes about 9.9% median receive
stalls, but remains above the 85% feed-retention gate. This proves the first
scaled QKV output tile at the production ABI. It does not yet prove every row,
N tile, output placement, attention, or model throughput.

Resource and measurement failures are part of the checkpoint: depth-3
activation buffering exceeded local tile SRAM beside weight double buffering,
so activations are consumed sequentially at depth 1. Adding the activation
stream moved weights to receive DMA channel 1; a channel-0-only trace measured
the short activation transfer and was rejected. The harness now traces both
channels and attributes bytes only to the long bundle stream.

## R61 full-output checkpoint and rejected integration (2026-07-13)

R61 covers all row stripes, N tiles, and K groups and performs direct padded
row-major output DMA. The raw-MLIR graph passes every real M256/N1280 output and
the M288/N1536 padding at `max_abs=3.8147e-6`. This closes the complete-QKV
correctness and output-placement step for OQ4.

It does not clear the performance gate. Three final trials have a 4.114-ms
median NPU time (3.900-4.476 ms), only 0.122 useful TOPS. Scalar local assembly
was 6.870 ms; native vector interleave reduced it to 4.238 ms; caching the full
6-KiB W4 activation view locally left the median at 4.114 ms. Against the
0.8635-ms whole-scaled wrapper control, this initially suggested that
compatibility with the old R34 W8 activation layout was the dominant cost.
R62/R63 later falsified that inference because the values used different
runtime and timing protocols. Integrating R61 would
move the model away from the 10k tok/s target, so it is rejected as a runtime
replacement despite correctness.

R62 must preserve the immutable `.rdna2.hfp` order and test the mutable side of
the boundary: have the upstream producer write the W4 activation view directly,
or use a measured DMA/memory-tile representation transform before compute.
That conversion may place mutable activations but must not become per-command
immutable tensor-block reordering. Compare the exact same full-output oracle
and report the conversion cost separately.

## R62/R63 measurement-system checkpoint (2026-07-13)

R62 supplies the mutable W4 view directly. It preserves exact full-QKV parity
but measures 3.850 ms with row-major output and 3.856 ms with physical output,
QKV-only bundle traversal, and the canonical R15 dynamic group loop. The small
gain over R61 proves activation compatibility is not the dominant limit.

R63 removes every remaining graph/immutable-input difference: its generated
MLIR matches R15, it uses the existing compact 2,359,296-byte QKV
`.rdna2.hfp`, and W/C DMA tasks use R15 ordering. Its cold Python raw-runtime
timing remains 3.964 ms median. However, the same current cache through the
production Rust/C++ executor measures 1.0292 ms median across three fresh
processes; pre-no-spill and historical controls measure 1.0288 and 1.0596 ms.
All production runs pass the real 327,680-output oracle at `max_abs=2e-7`.

Therefore the apparent ~4-ms bottleneck is first-command Python host-runtime
overhead, not AIE compute or DRAM delivery. Future performance admission must
use the warmed production wrapper (and ultimately end-to-end resident timing),
while the raw runner remains a correctness/output-layout diagnostic. Step 9's
next work is to attach the compact HFP BO and current whole-scaled executor to
the resident layer's shared activation/output chain without global in-kernel
tensor reordering.

## R64 exact-graph device-trace checkpoint (2026-07-13)

R64 injects declarative trace operations into parsed canonical R63 MLIR only
after proving that the untraced module is byte-identical to R15. Core-tile
packet routing is not practical on this fully occupied graph: eight- and
four-flow builds remained CPU-bound beyond five minutes after address
assignment, and a one-flow build exceeded four minutes. Shim-local DMA trace
lowering builds in seconds and does not route trace packets through the compute
network.

Twelve locked fresh-process traces across shim columns 0-3 pass all 327,680
real outputs with `max_abs=3.8147e-6`. The median device input-to-output span is
241.248 us (240.189-243.356 us), effective aggregate traffic is 19.559 GB/s,
and output-DMA starvation occupies 198.240 us median. Padded and useful compute
rates are 2.817 and 2.086 TOPS. Columns 4-7 lose their terminal S2MM event when
trace capture stops, so no timing span is inferred for them despite output
parity.

Compared with the 1.0292-ms warmed production wrapper, the traced device window
is about 23.4% of elapsed time. The remaining roughly 788 us combines wrapper
preparation, submission, synchronization, and output deblocking; it is not
attributed to a single host phase without a further trace. R65 should therefore
preserve the compact W4 `.rdna2.hfp`, produce the mutable BF16 attention handoff
directly, and chain through shared BOs. This does not change the offline-only
rule for immutable tensor-block conversion. Local nibble/lane swizzles remain
kernel work.

## R65 BF16 attention-stage checkpoint (2026-07-13)

R65 implements that mutable boundary without changing the compact W4 HFP or
R15 projection math. Each completed 24x96 f32 accumulator tile is converted
locally into three padded 24x32 BF16 records. Output DMA writes the 40 real
32-column stripes directly into the five-role, 10-KiB-per-record raw layout
consumed by R29. The cos/sin and norm/epsilon regions are preseeded by the
caller and never written; padded sixth-role columns are not emitted.

Three locked fresh-process runs with two warmups and three timed iterations
pass 327,680/327,680 BF16 values bit-for-bit under AIE `floor` rounding,
preserve every attention-tail byte, and leave all padding records zero. Median
NPU time is 0.487964 ms (0.485649-0.490328), median host call is 0.551481 ms,
and useful projection rate is 1.0315 TOPS. Largest linked core text is 9,280
bytes versus the 16-KiB program limit.

R66 is the next isolated bandwidth-first stage: consume this exact BO with the
proven R29 Q/K/V headnorm and RoPE packers and emit the verified Q/KV physical
layouts. If fusion exceeds program memory, split K and V across even/odd column
sets or retain a second shared-BO context; do not change HFP ordering or move
immutable conversion into a kernel.

## R66 pack-consumer checkpoint (2026-07-13)

R66 proves that R65's inline records are a correct input contract for the
existing headnorm/RoPE pack functions. It emits canonical Q (393,216 bytes)
and single-replay K/V (262,144 bytes) with Q cosine 0.99999121/max 0.0078125,
K cosine 0.99999156/max 0.0078125, and bit-exact V.

It is rejected as a performance schedule. Three fresh 100-command runs measure
0.9511, 0.9915, and 0.9984 ms (median 0.9915 ms), making sequential R65+R66
about 1.48 ms before attention. The inline graph broadcasts records one at a
time and serializes four core-pair packers; R28's compact joined input activates
all four pairs concurrently and is roughly 2.4-2.7x faster in current/historical
controls. This is a scheduling/layout result, not a rejection of headnorm/RoPE.

R67 must preserve R65's compact W4 HFP and local BF16 conversion while writing
or presenting the mutable values in the R28 joined consumption order. Measure
the projection-side DMA cost independently, then require total projection plus
pack to beat the R65+R66 sum before resident integration.

## R67 joined-stage checkpoint (2026-07-13)

R67 implements the concurrency correction. The W4 projection emits 8-token x
32-dimension BF16 tiles into 36 8-KiB records per role: 32 consumed M256
records and four padding records. Cos/sin remains in each record's second 4
KiB, and a shared 2-KiB parameter tail follows all roles. The R28 split FIFO
therefore feeds all four core pairs concurrently from one joined input.

Projection correctness is bit-exact across 327,680 values and all preseeded
tails/padding guards pass. Three fresh warmed runs measure 0.725232, 0.751200,
and 1.152343 ms (median 0.751200 ms). The pack consumer reproduces Q/K cosine
above 0.99999 with 0.0078125 max error and bit-exact V; 100-command runs measure
0.3517, 0.3670, and 0.3687 ms (median 0.3670 ms). Thus joined staging recovers
R28 pack speed and lowers the sequential R65/R66-style boundary from about
1.48 ms to about 1.1182 ms.

The remaining producer cost is task granularity: roughly 360 small output DMA
tasks. R68 should emit one padded 24-token x 32-dimension object per slice and
overlap each padding record with the next core row's first real record. A 37th
record per role safely absorbs the final padded write. This preserves the same
joined consumer order while reducing producer output tasks about threefold.

## R68 overlapping-write checkpoint (2026-07-13)

R68 validates that task-count reduction. Each projection core emits one padded
24-token x 32-column BF16 object per slice. DMA uses a three-record stride: the
padding record from one core row is overwritten by the following core row's
first real record, and a 37th record per role absorbs the final pad. The joined
consumer still sees 32 contiguous M256 records.

All 327,680 BF16 values pass bit-for-bit and every position/parameter tail is
preserved. Three warmed projection processes measure 0.465281, 0.494605, and
1.067005 ms (median 0.494605 ms). The pack stage measures 0.3435, 0.3579, and
0.3627 ms (median 0.3579 ms) with unchanged Q/K/V parity. Sequential medians
total about 0.8525 ms before attention, 24% below R67 and 42% below R65+R66.

R69 must now prove the boundary as deployed: allocate one shared 1,517,568-byte
BO, preseed its mutable position/parameter tails once, import it into both NPU
contexts, and time projection followed by pack without CPU synchronization or
copy between stages. Only that chained measurement can be used for resident
integration decisions.

R69 rejects the two-context deployment. Independent dma-buf imports are not
coherent, native XDNA SHMEM PRIME export returns `EINVAL`, and even the direct
single-GEM-handle experiment is intermittently wrong. Its passing runs spend
5.45-5.66 ms per chain because both full-array contexts inflate to about
2.7-2.9 ms; the required BO sync is only about 0.03 ms and is not the bottleneck.
The bandwidth-first ladder therefore continues with an R70 single-context
graph: retain R68's measured producer and consumer schedules, but make their
mutable boundary graph-local so no context switch or online global reorder is
introduced.

R70 proves that single-context boundary with the hardware's actual constraints.
The literal joined merge exceeded two input DMA channels, and the first inline
merge exceeded 16 KiB of core text. Channel reuse, K/V column specialization,
and one generic W4 group function produce a 13,504-byte maximum core and pass
the isolated stage and Q/KV byte oracles in three fresh primed processes. The
1.3076-ms median is the admitted projection/pack boundary. R71 should attach
attention inside the same graph before attempting to recover joined-input
concurrency; a second context would discard the R69 lesson.

R71 now proves the next accumulated mode functionally. Its five-argument graph
appends attention output to the existing result BO, redistributes Q/V packing
to columns 0-3, and runs full-width attention on columns 4-7. All projection
stage, Q, KV, and attention bytes match isolated references; maximum core text
is 15,888 bytes. Three 100-command runs measure a 3.3118-ms median, versus
1.5446 ms for the redistributed pack-only control and 0.9141 ms for isolated
attention. The fused attention/feed phase is therefore about twice the isolated
control and is not admitted.

The failed split-direction experiment is also informative: adding a second
core-to-memory-tile output path exceeds memory-tile input DMA channels. The
next bandwidth-first rung should remove the external Q/KV write/read boundary
with graph-local streams or reuse an existing memory-tile FIFO; it must retain
the observable R71 byte oracle while developing that path. The kernel-parameter
correctness workaround remains mandatory and independent of LDS use.

R72's scalar direct-Q experiment is exact but fails the bandwidth-first speed
ratchet. It removes the 393,216-byte external Q BO path, leaves that ABI
argument unused, and matches projection stage, external KV, and final attention
byte-for-byte. Sequential tile-memory allocation fits with a 15,248-byte
maximum core program. Three 100-command runs have a 3.9272-ms median, 18.6%
slower than R71's 3.3118 ms. The local handoff therefore must preserve
burst/vector transfer efficiency; scalar per-word synchronization is not an
admitted basis for K/V. The separate kernel parameter remains the correctness
workaround. LDS or tile-memory use is still chosen only by measured performance.

R73 then uses tile memory rather than avoiding it: one adjacent depth-one
24-KiB ObjectFIFO carries all six Q groups from each projection core to its
attention neighbor. Projection stage, external KV, and final attention remain
byte-exact with the Q BO unused. The producer-local precursor exceeded the
16-KiB program limit, while the adjacent graph fit after reducing producer
stack from 4 KiB to 2 KiB. Maximum producer/consumer core text is 14,912/14,352
bytes.

The three fresh 100-command times are 3.6449, 3.7165, and 3.7205 ms (median
3.7165 ms): 5.4% faster than scalar R72, but 12.2% slower than R71. Reject the
serialized six-group schedule, not shared tile memory itself. The next rung
must overlap the graph-local handoff or reuse an existing DMA path before K/V
is made local. The added kernel parameter remains the correctness workaround;
LDS/tile-memory choice remains independent and performance-driven.

R74 returns to the exact R71 boundary and halves KV replay rather than making
another operand local. Two query groups, two 4-KiB Q buffers, and four
accumulator/stat sets remain live per attention core, allowing one 262-KiB KV
plane to update both groups. The first 4-KiB-stack build exceeded the 64-KiB
tile allocation by 1,184 bytes; the measured 2-KiB stack fits at 64,672 bytes
with 864 bytes spare and 15,248-byte maximum core text.

Three exact 100-command runs measure 3.4496, 3.4242, and 3.2867 ms (median
3.4242 ms), 3.4% slower than R71. Reject additional query-group residence:
halving KV replays and DMA task count does not improve the stable result. The
next bandwidth-first rung should retain the R71 byte oracle but attack phase
scheduling/core utilization, not add more tile-resident Q/KV state. The kernel
parameter remains the correctness workaround, independent of this capacity and
performance result.

R75 retains R71's kernels and traffic but enqueues two complete query groups
before awaiting tasks, reducing six completion barriers to three. Six groups
exhaust static BD IDs at group 4. Four groups lower and link but fail hardware
Q parity with 392,405 of 393,216 bytes wrong, proving that descriptor allocation
success is weaker than runtime ordering correctness.

Two-group windows pass every byte oracle. Three 100-command runs measure
3.2580, 3.2775, and 3.3314 ms (median 3.2775 ms), 1.0% faster than R71. Admit
this schedule: it improves the complete boundary without changing math,
traffic, or tile residence. The next scheduling window must retain the same
oracle; afterward, port the best admitted schedule to resident-weight layer
integration.

R76 uses three groups per task window and passes every byte oracle. Three fresh
100-command runs measure 3.4199, 3.2222, and 3.2604 ms (median 3.2604 ms), 0.52%
faster than R75 and 1.55% faster than R71. This is the maximum correct window:
four groups corrupt Q and six exhaust BDs. Admit R76, end the queue-width sweep,
and port its two-window schedule into the resident-weight layer graph without
changing offline `.rdna2.hfp` order or the kernel-parameter workaround.

R77 ports the admitted R76 schedule to a destination-context-owned weight BO.
`NpuEmbeddingQkvAttentionOpus` validates and consumes the real QKV HFP payload
directly, uploads it once, and reuses it across commands. The raw extracted
`weights.bin` is retained only in the independent reference oracle.

Three exact 100-command runs measure 3.2753, 3.3165, and 3.2137 ms (median
3.2753 ms), within 0.46% of raw R76. Admit resident QKV/attention weights. The
next bandwidth-first boundary is the O projection plus residual/norm tail;
retain the five-argument ABI, offline layout, and exact output oracle. Full
layer/model throughput and tokens/J are still unmeasured.

R78 remaps attention to odd columns and Q/K/V packing to adjacent even columns
without changing the external oracle. All bytes pass, but three 100-command
runs have a 3.7959-ms median, 16.4% slower than R76. This is not an admitted
speed result. It proves the neighbor direction while showing that even cores
still carry too much compact-W4 projection/pack code to append R32 O projection.

The next bandwidth-first capacity rung is paired compact-W4 projection on odd
cores, allowing even cores to drop QKV projection and specialize in K/V pack,
O projection, and tails. Create the pair-major immutable block order once in a
loader-side `.rdna2.hfp`; do not reorder blocks in the graph or kernel. Measure
the paired projection and exact stage boundary before attaching O projection.

R79 completes the loader-side half. `PairedWholeScaledV1` reorders only intact
whole-scaled blocks into `(pair, block, lane)` order and retains their exact
encoded bytes. The generic cached API is keyed by the source artifact and does
not name a specific fractional OQ format. Unit coverage proves order, bytes,
descriptor metadata, and cache reuse. Next, implement only the paired compact
projection consumer and compare its complete stage output against R65/R70.

R80 implements that consumer. Odd cores reuse each activation block across two
pair-major compact-W4 blocks and emit both stripes into the R65 stage; even
cores have no projection program. The initial six-task output queue timed out,
while the one-slice R65 cadence passes all 327,680 values plus tail/padding
guards exactly.

The three warm times are 0.818433, 0.833471, and 0.789289 ms (median 0.818433
ms), slower than R65 but capacity-admitted. Maximum odd-core text is 11,872
bytes and even cores remain empty. Next add only Q/K/V packing and preserve the
exact R66 oracle before attaching attention or output projection.

R81 adds that Q/K/V pack phase and is exact, with a 1.8370-ms median and
14,032/10,912-byte odd/even images. R82's direct attention append is rejected:
odd columns 1/3 grow to 22,416 bytes, 6,032 over program capacity. This is not
an LDS result and does not replace the existing kernel-parameter workaround.

R83 applies the bandwidth-first ratchet at the code-image boundary. The exact
R70 single-group projection replaces duplicated bodies, while non-LTO loop
bounds prevent Peano from cloning 32 attention and 36 finish calls. The graph
fits at 15,888/10,912 bytes and passes stage, Q, KV, and attention byte-for-byte.
Its 4.0535-ms median is slower than R78 and R76, so retain it only as the
capacity base for an even-core direct O/tail join. The next rung succeeds only
if that join removes enough external movement to beat the current complete
boundary; odd-core code growth is no longer available.

R84 extends the fitting R83 image through direct attention-to-O projection.
The existing kernel parameter remains the correctness workaround; the
bank-aware allocator fallback and tile-memory placement are separate capacity
mechanics. The first hardware oracle found that the inherited R32 scatter had
the two token axes transposed for R83's topology. R84 now writes the produced
`active_column * 32 + core_row * 8` token blocks directly to canonical
token-major output using DMA destination strides. No graph/kernel tensor-block
reorder is performed.

With that scatter corrected, stage and KV are byte-exact and the fused O output
passes the standalone-attention plus deterministic-BF16-O oracle. Three
100-command processes measure 5.9362, 5.8716, and 5.9856 ms (median 5.9362
ms). This admits R84 as a correctness/capacity rung but rejects it for speed:
it is 46.4% slower than R83's 4.0535 ms median. The next bandwidth-first step
must attribute the added 1.8827 ms before fusing residual/norm; likely probes
are O-weight feed only, attention-to-O FIFO only, and O compute/output DMA.

The R84 attribution controls close that question. Their medians are 4.0951 ms
for paired attention handoff plus a 64 KiB completion signal, 4.7160 ms after
adding the complete 4,718,592-byte O-weight stream without MMUL, and 5.7761 ms
after adding every O MMUL and F32 finish into tile-local scratch. Full R84 is
5.9362 ms. Relative to R83's 4.0535 ms, the increment is therefore 0.0416 ms
handoff, 0.6209 ms weight delivery/consumption, 1.0601 ms compute/finish, and
0.1601 ms canonical output DMA. The adjacent FIFO is not the bottleneck.

R85 reuses each O activation tile across all four N tiles while preserving K
accumulation order. It passes the full oracle and measures 5.7945 ms median,
2.4% faster than R84; admit it. R86's two-accumulator compromise also passes
but measures 5.9203 ms and is rejected. Continue from R85, focusing on O
compute scheduling and the serialized 16 KiB weight-block cadence before
attaching residual/norm. None of these controls changes the existing
kernel-parameter correctness workaround or adds kernel-side tensor reorder.

R87 tests O-weight overlap without increasing compute-tile storage: a
depth-two shim FIFO passes and measures 5.7450 ms median. R88 depth three is
rejected as saturated at 5.7343 ms median. The depth-two schedule is retained.

R89 then adds the smallest local tail-storage rung. The even cores retain the
final 10 KiB activation FIFO object after its projection/packing lifetime and
add a 2 KiB tail, enough for one BF16 8x768 O wave. Six existing output-FIFO
chunks use DMA-only canonical scatter with the R83/R84 token mapping. Exact
stage/KV and the BF16 O numerical oracle pass; three 100-command runs measure
5.9044, 5.5931, and 5.7202 ms (median 5.7202 ms). This preserves bandwidth and
proves storage before norm math, while leaving the kernel-parameter workaround
unchanged. Even-core text now has only 1,296 bytes of physical headroom, so the
next step is program compaction followed by post-attention residual and RMS
norm in this same local store—not an output BO round trip.

Step 8 correctness is now established through the production kernel rather
than a reduced checksum surrogate. The checksum attempt was rejected because
horizontal float reductions changed subsequent virtual-int4 MMUL results; two
integer sentinels caught the corruption, and the broken mode was removed. The
production vector-store schedule, driven by `npu_opus_hfp_verify`, loaded the
real layer-0 Q/K/V roles through the real packed HFP and matched all 327,680
M256/K768/N1280 outputs with `max_abs=2e-7`. Its 0.8635-ms wrapper measurement
includes host preparation and deblocking and is not a kernel-only or model-level
throughput result. Next: retain this full output contract while eliminating the
host pack/unpack boundary and admitting it into the resident layer graph.

## Generic offline-layout checkpoint (2026-07-13)

The earlier “mixed encoding reserved” limitation is closed for the existing
full-K schedule. `FullKV1` persists the exact W4/W8/mixed slab order, cache
flags, and geometry in the same version-2 `.rdna2.hfp` envelope. Mixed W4 base
entries remain nibble-packed; their dense W8 overlays occupy a separate
schedule entry. A 26-case locked AIE2P matrix passed all plain/`+`/`++` storage
families, mixed overlay counts 1/3/7/39, and output widths 256 through 2304 with
zero mismatches. Cache reuse was mtime/size/SHA stable.

This is an offline-layout and projection-correctness milestone, not step 10.
R34/R35 still read independently generated dense derivatives. The next
bandwidth-first resident tranche must keep the current shared activation chain,
add destination-context-owned HFP weight BOs plus small parameter BOs, and port
the already-admitted decode/MMUL/output schedule into those resident xclbins.
It must not import a projection BO handle from a different XRT context or
reintroduce tensor-block conversion inside the kernel.

## R59 resident-context ABI checkpoint (2026-07-13)

The destination-context feed ratchet passes. The fixed DPU regmap allows five
data arguments, which rules out adding separate QKV, O, and parameter BOs beside
R34's four shared I/O arguments (and likewise separate gate/up, down, and params
beside R35). The loader-side solution is a generic
`ResidentContextBundleV1`: one checksummed HFP BO per R34/R35 context, with
role-local block order preserved and one padded immutable parameter tile.

R34 and R35 separate-BO controls sustain 56.121 and 56.053 GB/s median. Their
production bundles sustain 55.884 and 56.018 GB/s, at least 99.48% of R58 feed-only, with zero
receive stalls. All guards pass. The rejected eight-column trace configuration,
the six-argument XRT crash, and the nonpersistent read-modify-write guard are
recorded in `benchmarks/npu_gemm_tuning/r59/README.md`; none is hidden as a
successful mode. R60 may now replace the feed guard with nibble decode and the
first R34 QKV MMUL while leaving the shared layer I/O unchanged.

## R101-R104 host-bridge removal checkpoint (2026-07-13)

The first in-record state design is rejected, but it narrows the next rung.
R101's literal sixteen-way 128-byte relay exceeds core program memory at
16,444 bytes. Shim-DMA scattering reduces it to 16,380 bytes, but hardware
reads misaddressed inverse state and layer-0 cosine falls to 0.50248530.
Putting the metadata object behind the even normalized-X output channel fits
at 16,268 bytes but reaches the independent four-second command timeout.

R102 proves that the consumer-side three-row object can fit: reducing only the
15,872-byte weight FIFO from depth two to depth one removes the bank allocator
warning and yields 16,064-byte maximum text. It is not admitted because its
producer is not correct. R104's consumer-side RMS recomputation is also
rejected at 18,352 bytes (`-Oz`: 20,032 bytes). A third metadata FIFO was
separately rejected by the hardware's DMA-channel count.

Proceed with a smaller observable boundary: retain R44 direct architectural X,
retain R99/R100 output and tail contracts, preserve X in place, and move only
X-times-inverse activation preparation off the host. Immutable pre-FFN norm
stays folded into loader-side W4 scaling. Measure the new stage alone before
composition; admission requires correct per-row inverse mapping, no four-second
timeout, and materially less traffic than the current two 5.1-MiB host syncs.
The separately added kernel parameter remains the platform-issue workaround;
it is not LDS avoidance. R15's `rounding=floor` and `saturation=none` remain
independent numerical controls. Gate fragment buffers remain an independent
state-lifetime fix, and context recycling remains only a timeout mitigation.

## R112 route-consolidation checkpoint (2026-07-13)

R112 preserves the bandwidth-first invariant while changing tail topology.
R100's active input is 786,432 bytes of interleaved FFN state plus 393,216
bytes of split architectural X. R112 moves the same canonical X rows into the
third plane already reserved by R99 and transfers one 1,179,648-byte joined
row state. No immutable block order changes and no tensor layout conversion is
performed by the kernel.

A proposed second memory-tile X broadcast fails resource allocation because it
exceeds the tile's output DMA-channel count. Reusing the existing row route
fits, removes all 24 horizontal core flows, and lowers maximum core text from
4,208 to 3,696 bytes. The hardware oracle is unchanged. Four counterbalanced
100-command pairs improve mean tail dispatch from `0.324965 ms` to
`0.218271 ms` (32.84%), despite identical active byte volume. This is evidence
that task/route overhead and ownership schedule matter after bandwidth is
preserved; it is not evidence for LDS avoidance. The separately added kernel
parameter remains the platform workaround.

## R113 tail-local next-pack checkpoint (2026-07-13)

R113 preserves R112's 1,179,648-byte joined canonical row input and performs
next-layer RMS/AWQ/FWHT/int8 packing before each two-row tail output leaves its
core. The extra R111 completed-state input pass falls from 786,432 active bytes
to zero. No immutable tensor block is reordered in the kernel.

Capacity and queue limits were tested first. A separate 24,576-byte eight-row
completed buffer fails bank allocation, so the admitted graph packs each phase
while its 6,144-byte output FIFO object is local. Four tail outputs plus three
diagnostic outputs also exceed the usable shim task window and leave group 2
zero. Launching all stripes, retiring one old output task per stripe, and then
queuing the diagnostics preserves parallelism and passes. A per-stripe wait is
correct but rejected at 13.3578 ms because it serializes the array.

The admitted hardware result is tail cosine `1.00000000`, `0.0000310` maximum
error, three one-code Q differences, and `7e-9` scale error. R113 averages
5.056325 ms against a current 5.260901-ms sum for R112 plus R111, a 3.8886%
win. This shows the bandwidth-first ratchet worked but also exposes the next
limit: exact RMS/FWHT pack compute dominates after the full input pass is
removed. Assemble R34 suffix blocks in-context next, then vectorize the pack;
never reintroduce the completed-state round trip to make compute tuning easier.

One fresh-context transition produced all-zero output while four immediate
reproductions and the repeated timing series passed. Keep context lifetime as
a separate diagnostic. The added kernel parameter remains the platform
workaround; it is not LDS avoidance or this queue/capacity handling.
