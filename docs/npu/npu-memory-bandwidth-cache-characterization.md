# XDNA2 memory bandwidth, cache hierarchy, and MALL characterization

Date: 2026-07-12
Host: `halo`, Ryzen AI Max+ 395 / Strix Halo, 128 GiB LPDDR5X
NPU: XDNA2/AIE2P, eight columns, XRT 2.21.75, amdxdna 2.25.0, firmware 1.1.2.65

## Conclusions

The external NPU feed is bandwidth-limited once enough independent columns are
active, but the current resident EmbeddingGemma execution is **not close to that
bandwidth roof**:

- one compute-tile receive stream sustains 13.05-13.29 GB/s over its full span
  and exactly 14.4 GB/s while active;
- eight columns saturate at about 56.5 GB/s, with per-column receive busy time
  falling to about 49%;
- the ceiling is flat for 1, 4, and 16 KiB DMA tiles and for shared working sets
  from 64 KiB through 64 MiB;
- eight columns reading one shared 1 MiB region reach 56.35 GB/s; eight distinct
  1 MiB regions reach 55.65 GB/s. Shared-region locality provides no useful
  bandwidth amplification;
- CPU DRAM pressure reduces NPU feed to 43.04 GB/s. A 512 MiB GPU streaming
  workload reduces it to 18.21 GB/s. The NPU therefore shares an upstream
  memory/fabric resource with CPU and GPU external-memory traffic;
- GPU copy loops whose total hot sets are 16 or 32 MiB do not reduce NPU feed:
  57.09 and 57.24 GB/s respectively. A MALL-sized GPU workload and the NPU feed
  can proceed without measurable contention, while an over-capacity GPU stream
  strongly contends.

The practical MALL answer is therefore: **the current amdxdna SHMEM path shows
no evidence of a usable NPU MALL cache**. This does not prove that no undocumented
SoC transaction type could ever traverse MALL. It does show that Hipfire's
current NPU buffers receive no observable 32 MiB capacity benefit, no
shared-region amplification, and no interference from a GPU workload occupying
the reported MALL capacity.

## Measurement hierarchy

It is important not to call every byte movement “memory bandwidth.” Four
different resources exist:

1. **Tile-local data memory.** UG1079 documents 32 KiB per tile, eight
   single-port 4 KiB banks, two 256-bit load ports, and one 256-bit store port
   ([memory](ug1079-2026.1-AIE-programming-manual/004-ai-engine-memory.md),
   [kernel bounds](ug1079-2026.1-AIE-programming-manual/229-kernel-coding-bounds.md)).
   The issue ceiling is 64 B/cycle read plus 32 B/cycle write when placement,
   alignment, and scheduling avoid bank conflicts. This is explicit SRAM, not a
   transparent cache.
2. **Neighbor memory and streams.** A core can address neighboring tile memories,
   while normal streams carry 32 bits/cycle and backpressure when a consumer is
   not ready
   ([data movement](ug1079-2026.1-AIE-programming-manual/053-data-movement-between-ai-engines.md),
   [stream access](ug1079-2026.1-AIE-programming-manual/114-stream-based-access.md)).
3. **Memory tiles.** Linux's XDNA documentation describes the memory-tile row as
   software-managed on-chip L2 fed by DMA, not as an automatically filled cache.
   Capacity is device-specific. The programming manual does not establish the
   Strix Halo capacity.
4. **Host/global memory.** amdxdna SHMEM objects are page-backed GEM allocations
   mapped for NPU DMA. The installed driver explicitly states that the NPU is
   not cache coherent and requires explicit CPU-cache synchronization. This is
   the path characterized by R1/R56.

The complete 297-file UG1079 manual was read in three contiguous ranges. It
contains no statement that Ryzen/XDNA can allocate in, hit in, snoop, or bypass
GPU MALL. It also does not specify a Ryzen cache line, transparent cache size,
replacement policy, cache counters, or host/NPU coherency protocol. Much of the
manual is Versal AI Engine material; its local-memory and scheduling rules are
useful priors, but they are not sufficient evidence for Strix Halo fabric
topology.

## External-feed limits

R1's receive-port trace excludes host allocation, JIT, buffer synchronization,
and dispatch setup. `PORT_RUNNING`, `PORT_STALLED`, and `PORT_IDLE` are measured
on the compute tile's S2MM receive port. At the measured 1.8 GHz H clock, one
active stream transfers 8 B/cycle, or 14.4 GB/s.

### Tile-size invariance

All rows transfer 1 MiB through one column:

| DMA tile | span GB/s | active GB/s | receive busy |
|---:|---:|---:|---:|
| 1 KiB | 13.185 | 14.400 | 0.916 |
| 4 KiB | 13.293 | 14.474 | 0.918 |
| 16 KiB | 13.053 | 14.400 | 0.906 |

The same byte rate across a 16x tile-size range rules out descriptor size and
the core touch loop as the long-transfer limiter. The roughly 8-9% gap between
active and span bandwidth is inter-tile idle time.

### Column scaling

The durable R1 trace shows:

| columns | aggregate GB/s | per column | mean receive busy |
|---:|---:|---:|---:|
| 1 | 13.4 | 13.4 | 0.93 |
| 2 | 25.8 | 12.9 | 0.90 |
| 4 | 43.9-45.3 | 11.0-11.3 | 0.77-0.80 |
| 8 | 54.0-56.3 | 6.75-7.04 | 0.47-0.49 |

One to two columns is nearly linear. Above two, the aggregate approaches 56
GB/s while each receive port spends more time idle. Increasing FIFO depth,
using two streams per shim, spreading regions by hundreds of MiB, and changing
DMA tile size do not raise the roof. The bottleneck is upstream of the compute
tile and shared across columns.

## Hidden-capacity search

Eight columns read the same address range. Requested bytes count all eight
reads, so a transparent shared cache should produce either a capacity-dependent
bandwidth regime or a warm/shared advantage over distinct regions.

| shared working set per column | aggregate GB/s | mean receive busy |
|---:|---:|---:|
| 64 KiB | 60.30 | 0.523 |
| 256 KiB | 58.26 | 0.506 |
| 1 MiB | 56.84 | 0.493 |
| 2 MiB | 56.73 | 0.492 |
| 4 MiB | 56.54 | 0.491 |
| 8 MiB | 56.67 | 0.492 |
| 16 MiB | 56.59 | 0.491 |
| 32 MiB | 56.53 | 0.491 |
| 64 MiB | 56.52 | 0.491 |

There is a small-transfer transient below 1 MiB, followed by an exceptionally
flat long-transfer regime. There is no knee at the 2 MiB or 32 MiB capacities
reported for the NPU agent by `rocminfo`, nor when crossing 32 MiB.

Very small 4-32 KiB runs are dominated by pipeline start/drain and trace-event
granularity. They should not be interpreted as cache bandwidth. Shared and
distinct regions are already indistinguishable in that range within those
effects, and converge to the same 56 GB/s roof at useful transfer sizes.

## Does the NPU use MALL?

Evidence must be separated into metadata, documented behavior, and measured
behavior.

### Metadata that suggests a cache

`rocminfo` presents `aie2p` as a DSP agent and reports L2=2 MiB and L3=32 MiB.
The L3 number matches the GPU's reported MALL. However, the same agent reports
cache-line size 0, clock 0, compute units 0, and no ISA. This is not enough to
show that amdxdna DMA requests use that cache; it may describe shared APU
topology or capabilities exposed by the ROCr DSP adapter.

### Documentation and driver evidence

- AMD describes Strix Halo's up-to-32-MiB MALL as amplifying **graphics**
  bandwidth, while separately describing CPU/GPU/NPU unified memory.
- AMDGPU's ISA documentation defines MALL as a cache for GPU memory.
- Linux's [AMD NPU documentation](https://docs.kernel.org/accel/amdxdna/amdnpu.html)
  describes dedicated DMA engines moving data between host DDR and memory tiles;
  it does not place MALL on that path.
- The installed amdxdna 2.25 driver allocates SHMEM with GEM shmem pages and says
  explicitly that the NPU is not cache coherent, requiring explicit CPU cache
  flushing/invalidation.
- UG1079 has no MALL or XDNA cache-path statement.

Relevant external AMD descriptions:

- [AMD workstation memory-wall article](https://www.amd.com/en/blogs/2026/ansys-discovery-on-amd-graphics-jointly-scaling-the-memory-wall.html)
- [AMDGPU backend memory model](https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.2.1/LLVM/llvm/html/AMDGPUUsage.html)

### Interference experiment

The NPU reads a shared 16 MiB region through eight columns while another agent
runs continuously:

| concurrent control | control working set / rate | NPU GB/s | receive busy | delta |
|---|---|---:|---:|---:|
| idle | none | 56.743 | 0.493 | — |
| CPU streaming | 16 threads, 133.6 logical GB/s | 43.042 | 0.374 | -24.1% |
| GPU hot copy | 2x8 MiB, 883.8 logical GB/s | 57.091 | 0.496 | +0.6% |
| GPU full-MALL copy | 2x16 MiB, 922.1 logical GB/s | 57.237 | 0.497 | +0.9% |
| GPU streaming | 2x256 MiB, 198.2 logical GB/s | 18.215 | 0.158 | -67.9% |

Logical GPU bandwidth above physical DRAM bandwidth is expected for a hot set
served from GPU cache. The absence of NPU interference while the GPU cycles a
full 32 MiB hot set, followed by severe interference when the GPU set grows to
512 MiB, is strong behavioral evidence that current NPU SHMEM reads do not
consume useful MALL capacity. It also confirms that both agents meet again at
the external-memory/fabric path.

This is still an interference inference, not a MALL hit counter. Shared package
power, clock policy, trace traffic, cache replacement, and GPU write policy are
possible confounders. A definitive architectural claim requires an AMD fabric
diagram or counters that attribute NPU transactions and MALL hits.

## Current EmbeddingGemma roofline

The completed resident 256-token path is not globally memory-bandwidth limited.
Its packed immutable payload per layer is:

| phase | packed weight payload | mean phase time | payload/time | 56.5-GB/s lower bound |
|---|---:|---:|---:|---:|
| attention | 8.192 MB | 9.050 ms | 0.905 GB/s | 0.145 ms |
| FFN | 12.190 MB | 13.603 ms | 0.896 GB/s | 0.216 ms |

The payload sizes come from the runtime's actual block constants, including
padding and duplication across columns. The timing is the 24-layer mean from
the 2026-07-12 resident phase trace. Even treating every packed byte as one
compulsory external read, both phases achieve only about 1.6% of the measured
array feed roof. Against one 14.4-GB/s active stream, they still have roughly
16x more wall time than the payload-only lower bound.

The same trace averages 3.56 ms for the post-FFN tail and 9.85 ms for
preparation/output per layer. Those stages move far less immutable weight data.
Their cost cannot be explained by saturating the 56 GB/s global feed.

Therefore the user's conditional is correct in principle but false for this
path: once a large streaming transfer saturates the NPU fabric, dispatch latency
and core inefficiency become secondary. The current model path does not saturate
it. Dataflow serialization, per-stage dispatch, stream utilization, local bank
traffic, scalar/vector scheduling, synchronization, and repeated preparation
remain first-order costs.

## Limitations and next measurements

- Event trace measures requested bytes arriving at compute-tile S2MM ports, not
  physical DRAM beats or cache hits.
- The 1.8 GHz trace clock is confirmed under load on this host but should be
  recorded again on another machine or power mode.
- Shared-region tests expose reuse across concurrent columns, not a randomized
  latency pointer chase. The AIE DMA programming model is optimized for streams,
  so latency characterization needs a separate dependent-access kernel/runtime.
- R57 now proves that R34's exact 125x16-KiB per-column schedule retains 43.5
  GB/s across four columns through one-to-one memory-tile forwarding, and 43.2
  GB/s with four-row broadcast. This compute-port trace proves delivery through
  the hop; a dedicated memory-tile port/bank sweep is still required to isolate
  memory-tile capacity and bank bandwidth from the upstream external feed.
- Local-memory load/store tests should force same-bank and distinct-bank
  placement, aligned and misaligned vectors, and one versus two independent
  pointers. Compare against the manual's 64-B/cycle read ceiling.
- If AMD exposes Data Fabric or MALL client counters for this APU, repeat the
  shared/distinct and GPU-pressure matrix while collecting per-client reads,
  misses, and DRAM beats. Current DF PMUs did not attribute XDNA traffic cleanly.

Reproduction sources and raw rows are under
[`benchmarks/npu_gemm_tuning/r56/`](../../benchmarks/npu_gemm_tuning/r56/) and
[`r56-feed-cache-20260712.csv`](../../benchmarks/npu_gemm_tuning/results/r56-feed-cache-20260712.csv).
