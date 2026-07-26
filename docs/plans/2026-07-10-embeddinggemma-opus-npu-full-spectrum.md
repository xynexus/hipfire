# EmbeddingGemma 300M Full-Spectrum Opus NPU Pull-Up

## Goal

Implement and optimize a format-generic, resident AIE2P/XDNA2 execution path
for EmbeddingGemma 300M across the complete Opus quant spectrum:

- pure OQ4 through the W4A8 kernel;
- arbitrary mixed OQ bitwidths through W4A8 plus variable sparse W8 overlays;
- pure OQ8 through the W8A8 kernel;
- `+` variants through activation-aware/AWQ sidecars;
- `++` variants through the same runtime encoding with offline Hessian/LDLQ
  error feedback.

Do not design names or APIs around one mixed precision such as OQ4.25. W8A8 is
the upper-quality/performance anchor, not the architecture of the solution.

The intended performance target on the Ryzen AI Max+ 395 AIE2P NPU is at least
10,000 input embedding tokens/s at M=256, with 15,000+ tokens/s as the stretch
target for W8A8 and competitive mixed formats. Report package tokens/joule from
the shared `amdgpu power1_average` SoC rail.

## Important Correction

The existing 2026-07-10 throughput chart is GPU-only:

- BF16: about 14,120 input tokens/s on the Radeon 8060S (`gfx1151`);
- compressed GPU Opus: about 4,655-4,801 input tokens/s.

Those numbers are valid GPU diagnostics but are not NPU results and must not be
used as evidence of NPU progress.

Historical Phoenix measurements from this codebase were NPU kernel results:

| precision | quality vs BF16 | throughput | package power | package tokens/J |
|---|---:|---:|---:|---:|
| W4A8 | 0.955 | 5,854 tok/s | 6.76 W | 866 |
| W8A8 | 0.9998 | 5,171 tok/s | 6.20 W | 834 |
| GPU BF16 | 1.0 | 3,978 tok/s | 28.4 W | 140 |

The Phoenix NPU benchmark is a GEMM-ceiling estimate derived from measured
GMAC/s divided by 0.11 GMAC/token. It excludes attention, norms, and full-model
dispatch overhead. See
`crates/hipfire-xdna/examples/npu_embeddinggemma_bench.rs`.

## Current AIE2P Baseline

The current Strix Halo W8A8 projection-inventory benchmark at M=256 reports:

- projection time: 146.586 ms per encode;
- projection-only throughput: about 1,746 input tokens/s;
- aggregate throughput: 0.371 TOPS;
- 10,000 tok/s requires 25.6 ms total, a 5.73x reduction;
- 15,000 tok/s requires 17.07 ms total, an 8.59x reduction.

Result:
`benchmarks/npu_gemm_tuning/results/embeddinggemma-aie2p-w8-m256-baseline-20260710.csv`.

The current end-to-end mixed projector is a correctness bridge, not a
performance path. It performs about 144 synchronous GPU-to-host-to-NPU-to-host-
to-GPU projection crossings per encode. Three 16-18-token documents take
roughly 667-849 ms. See
`benchmarks/npu_gemm_tuning/results/embeddinggemma-aie2p-opus-mixed-hybrid-e2e.csv`.

## Existing Implementation

### Quant producer

`crates/hipfire-quantize/src/main.rs` parses generic mixed names from OQ4.125
through OQ7.9375. Mixed storage uses compact `qt=36` blocks with variable
sparse-overlay counts. Plain, `+`, and `++` names are generic and canonical.

### NPU primitives

- `crates/hipfire-xdna/src/gemm_mp.rs`: per-dispatch W4A8/W8A8 GEMM with K=256.
- `crates/hipfire-xdna/src/sparse3_mp.rs`: sparse-three residual dispatch.
- `crates/hipfire-xdna/src/opus.rs`: generic W4, variable-overlay mixed, and W8
  decoder/executor with resident K-group buffers and ordered queued base dispatch.
- `benchmarks/npu_gemm_tuning/r6/r6_gemm_ts_w8m8.cc`: AIE2P W8A8 kernel.
- `benchmarks/npu_gemm_tuning/r6/r6_gen_mp.py`: M-parallel W-broadcast array.

The existing `_rN` whole-GEMM caches fuse multiple M blocks into one dispatch.
They do not fuse K=256 groups, so merely enabling `_rN` does not solve the
EmbeddingGemma bottleneck at M=256, where one M dispatch already covers the
batch.

### EmbeddingGemma bridge

`crates/hipfire-arch-embeddinggemma/src/npu_opus.rs` currently:

- admits pure W4 (`qt=33/34`), compact mixed (`qt=36`), and pure W8 (`qt=35`)
  rank-two tensors through one generic projector;
- downloads each GPU input to host;
- runs the host-driven NPU executor;
- uploads each output back to GPU;
- leaves attention, norms, residuals, pooling, Dense heads, and non-Opus down
  projections on GPU/host.

The generic `NpuOpusProjector` is the correctness bridge. Full-K fused kernels,
resident layer state, and removal of per-projection host crossings remain open.

## Required Architecture

### 1. Unified Opus matrix contract

Introduce a generic packed matrix enum or trait representing:

- pure W4 (`qt=34`, and `qt=33` where applicable);
- compact mixed (`qt=36`, arbitrary overlay count);
- pure W8 (`qt=35`).

Each matrix carries K, N, scales, optional AWQ input scale, and resident packed
weights. `++` must not require a separate runtime path because its difference is
offline error feedback.

### 2. Shared preprocessing contract

Use one activation contract for all formats:

- optional AWQ input scaling;
- the exact existing FWHT sign/order convention;
- per-row/per-group int8 activation quantization;
- format-independent output-scale reconstruction.

Do not duplicate or subtly change the rotation/scaling maths between W4, mixed,
and W8 paths.

### 3. Resident weights and reusable buffers

The timed path currently reloads packed weights for every K group and matrix.
Move packed model weights into persistent XDNA-accessible buffers and reuse
activation/output buffers. Weight upload and xclbin/context creation must be
outside measured forwards.

### 4. Full-K projection accumulation

Current kernels contract one K=256 group. EmbeddingGemma projections require
3, 5, or 12 groups. Add a full-K projection path that accumulates those groups
without returning to the host between groups. The preferred progression is:

1. one XRT dispatch per complete projection;
2. multiple projection roles per layer per dispatch where practical;
3. resident layer or full-encoder scheduling.

The W4 base and sparse residual must accumulate under the same scheduling
contract so arbitrary mixed bitwidths do not multiply host dispatch overhead.

### 5. Remove GPU/host crossings

Use existing dma-buf/shared-buffer support in `NpuGemmMp` as an intermediate
step, but the end target is NPU-resident execution of:

- q/k/v/o projections;
- bidirectional attention;
- RMSNorm and residual operations;
- gate/up/down FFN;
- mean/last/CLS pooling;
- Dense projection heads;
- final normalization.

Do not claim full NPU throughput while attention, norms, or every projection
crosses back to GPU.

## Implementation Order

1. Add tests for generic OQ tensor admission and matrix classification.
2. Refactor the mixed-only projector into a unified Opus projector.
3. Add correctness support for pure OQ4 and pure OQ8 beside mixed `qt=36`.
4. Preserve `+` AWQ and `++` offline-value semantics across every encoding.
5. Add persistent weight and reusable activation/output buffers.
6. Implement full-K accumulation for W4, W8, and sparse overlays.
7. Measure projection-inventory throughput at M=32, 128, 256, and 512.
8. Replace per-projection GPU round trips with shared buffers.
9. Move non-linear layer operations onto AIE2P and build a resident layer path.
10. Run full end-to-end quality, throughput, and package-energy sweeps.

## Format Test Matrix

At minimum test these local artifacts under
`~/.hipfire/models/embeddinggemma-300m/`:

- `EmbeddingGemma-300M--oq4.hfq`
- `EmbeddingGemma-300M--oq4+.hfq`
- `EmbeddingGemma-300M--oq4++.hfq`
- `EmbeddingGemma-300M--oq4.25.hfq`
- `EmbeddingGemma-300M--oq4.25+.hfq`
- `EmbeddingGemma-300M--oq4.25++.hfq`
- `EmbeddingGemma-300M--oq4.5.hfq`
- `EmbeddingGemma-300M--oq8.hfq`
- `EmbeddingGemma-300M--oq8+.hfq`
- `EmbeddingGemma-300M--oq8++.hfq`
- joint-scale and Dense/tail promotion candidates already in that directory.

Also generate/test at least one lower and one higher arbitrary mixed point, for
example OQ4.125 and OQ6.5, to prove the runtime is not hardcoded to OQ4.25.

## Verification

### Correctness

- CPU integer/scaling oracle parity for W4, mixed, and W8 matrices.
- Hardware parity for multiple K/N/M shapes and variable overlay counts.
- GPU-versus-NPU full embedding cosine and max-absolute-error checks.
- STS-Benchmark Spearman-vs-gold admission gate (see the 2026-07-12 checkpoint):
  the `embedding_quality` hipfire-eval battery, gating candidate-vs-reference
  Spearman delta within a ~1.0-point band. Cross-model cosine and rank
  agreement are recorded as evidence but are NOT the admission criterion.
- `./tests/coherence-gate-dflash.sh` after quant/kernel/dispatch changes.

### Performance

- Acquire `hipfire lock` for every NPU/GPU run.
- Separate kernel ceilings, projection-inventory estimates, hybrid correctness
  runs, and fully resident end-to-end results.
- Time only after contexts, weights, and reusable buffers are resident.
- Report median and range across at least three independent runs.
- Use input embedding tokens/s, never decode tokens/s.

### Energy

- Record active package watts from `amdgpu power1_average`.
- Report package tokens/J as the primary cross-backend metric.
- Record idle-subtracted dynamic tokens/J as supplementary evidence.
- Keep workload, host, sequence length, and power rail identical within charts.

## Guardrails

- HIP/ROCm remains the GPU backend; XDNA/MLIR-AIE remains the NPU backend.
- No Python in inference hot paths; Python is allowed for kernel generation and
  plotting.
- Keep generated xclbins, model files, and sidecars under `~/.hipfire`, not in
  the repository.
- Preserve RDNA portability for shared GPU code.
- Preserve unrelated dirty worktree changes.
- Use generic Opus names; do not create `425` files, functions, or types.
- Do not present GPU results as NPU results.
- Do not present GEMM ceilings or projection inventories as full-model results.

## First Concrete Slice

Start with a correctness-preserving unified projector:

1. classify OQ4, compact mixed, and OQ8 tensors from HFQ quant types;
2. load optional AWQ sidecars generically;
3. reuse one activation preprocessing implementation;
4. run all three encodings through current AIE2P primitives;
5. add tests proving arbitrary mixed overlay counts and `+`/`++` admission;
6. establish hardware parity before beginning full-K/residency optimization.

Then optimize the shared path rather than producing separate one-off W4, OQ4.25,
or W8 implementations.

## Progress checkpoint — 2026-07-10

Completed in the full-K projection slice:

- generic W4, arbitrary mixed, and W8 admission with shared AWQ/FWHT/scaling;
- resident complete-projection weights and reusable argument buffers;
- one XRT/AIE dispatch per projection for 3, padded-5, and 12 K groups;
- AIE-side mixed W4+dense-W8 residual accumulation;
- resumable M=256 cache generation under `~/.hipfire/npu`;
- exact CPU-oracle hardware parity across formats and extreme K/N shapes;
- automatic full-K-only cache selection in the EmbeddingGemma projector.

### Whole-array reuse checkpoint

The R14 schedule is now ported from Phoenix to AIE2P and reusable through the
Rust XDNA runtime. It broadcasts four activation stripes across array rows and
four weight stripes down array columns, so all 16 compute tiles share both
operands. Generated caches and xclbins remain under `~/.hipfire/npu`.

- W4 uses native `4x16x16` int8/int4 MMULs with 96x384 macro tiles.
- W8 uses paired `8x8x8` int8 MMULs with 96x192 macro tiles. A first 96-column
  stripe exceeded AIE core memory even at FIFO depth one; 48-column stripes
  restore double buffering and exact execution.
- Patterned signed hardware parity is exact for both modes across the model's
  `K=768/1280` and `N=256/768/1152` projection geometries.
- The hardware gate now uses normal FWHT-derived int8 activations and full W8
  values. This exposed two bugs hidden by the first low-amplitude oracle: long
  native int8/int4 MMUL chains become invalid above about signed-i16 range, and
  multi-megabyte output buffers need explicit pre/post host-cache sync. W4 now
  accumulates exact K=32 segments as int32 vectors.
- Reusable q/o wrapper timing, including host matrix marshaling, explicit cache
  reconciliation, and readback, is about 0.76 ms W4 and 1.39 ms W8 in the
  realistic-range gate. These timings vary enough that a median sweep remains
  required before using them as a performance baseline.
- Raw W4 schedule timing at the same q/o geometry is 0.135 ms (2.51 TOPS), so
  host packing/unpacking—not AIE compute—is now the dominant projection seam.

The shared Opus executor and EmbeddingGemma projector now select resident
whole-array caches automatically for pure OQ4/OQ8, including `+` and `++`.
CPU-scaling-oracle parity is exact for all four runtime cases. The current
GPU/XDNA hybrid still performs every projection crossing and host scaling:
short-document probes measured 68.4 actual input tok/s for OQ4 and 62.2 tok/s
for OQ8. OQ8 matched the GPU path at minimum cosine 0.999893; OQ4 reproduced
the earlier approximately 0.985 GPU/NPU discrepancy and is not a 0.999 parity
result.

These are projection-kernel or hybrid results, not full-model NPU throughput.
Mixed Opus overlays, scaled group accumulation, shared buffers, and resident
layer operations still need to be moved onto the same whole-array scheduling
contract. The next kernel slice should retain a row/column output tile across
all K groups, apply activation and weight scales on AIE, and emit one final f32
tile instead of group-major int32 partials.

### Group-retaining scaled W4 checkpoint

R15 now implements that next W4 slice. Each core retains its output tile across
all K=256 groups, reconstructs each group with the exact activation and weight
scales on AIE, and emits one final f32 tile. The resident Rust runtime and
shared Opus executor select these caches automatically for pure OQ4 artifacts,
including `+` and `++`.

- Appending scale tails directly to the 6,240/12,672-byte A/W payloads corrupted
  integer dots. A separate scale input exceeded shim/core DMA-channel limits.
  The working contract preserves the exact 6,144/12,288-byte R14 prefixes and
  pads their containing payloads to 8 KiB/16 KiB before the scale tails.
- Scaled hardware parity is exact across `K=768/1280` and
  `N=256/768/1152`; maximum absolute error is at most 8e-7.
- M256 dispatch measurements are 0.300 ms for K768/N256, 0.341 ms for
  K768/N768, 0.538 ms for K768/N1152, and 0.532 ms for K1280/N768.
- The resulting 24-layer projection inventory is still about 69.3 ms per
  M256 encode (roughly 3,690 projection-only input tok/s), so it does not meet
  the 25.6 ms / 10,000 tok/s target.
- The short-document GPU/XDNA hybrid improved from 68.4 to 77.2 actual input
  tok/s, but still performs every projection crossing and remains only a
  hybrid correctness/performance diagnostic. Its OQ4 GPU/NPU minimum cosine is
  0.984162, consistent with the previously recorded OQ4 discrepancy.

The next performance lever is projection-role and layer scheduling, not more
host-side batching: q/k/v and gate/up should share activation preprocessing and
submission, followed by resident attention, residual, normalization, and FFN
operations so projection outputs stop crossing back to the GPU.

### Dense W8 and combined-role checkpoint

R15 now applies the same resident, group-retaining scaled contract to dense
W8. W4/W8 share one runtime and cache-selection path; W8, W8+, and W8++ match
the CPU scaling oracle across all model shapes with maximum absolute error
1.7e-5.

For plain artifacts with identical/no AWQ input sidecars, the projector also
concatenates q/k/v into one N=1280 matrix and gate/up into one N=2304 matrix.
This reduces seven projection calls per layer to four while preserving exact
role ordering. Full hybrid checks remain stable:

- OQ4: 92.4 actual input tok/s, minimum GPU/NPU cosine 0.984162;
- OQ8: 80.7 actual input tok/s, minimum GPU/NPU cosine 0.999904.

This is a useful crossing reduction, not a path to the M256 target by itself.
M256 shared-Opus wrapper timing (activation FWHT/quantization, allocations,
packing, dispatch, and readback) is 10.46 ms per W4 layer and 11.09 ms per W8
layer across combined qkv, o, combined gate/up, and down. That is much larger
than the raw scaled-dispatch inventory because activation preprocessing and
host marshalling still repeat for every role. FWHT/quantization must move onto
AIE, and layer intermediates must remain resident, before further role fusion
can translate into 10,000 input tok/s.

Measured status is still below admission: W4 projection inventory is 73.083 ms
at M=256 (about 3.5k input tok/s before non-projection work). The real hybrid
model is 130.9 input tok/s and 5.6 package tok/J, and its GPU cosine gate remains
below threshold at 0.98601860. Attention, norms, residuals, activations, pooling,
Dense heads, and final normalization remain GPU/host resident. Do not promote
this checkpoint as a full-NPU or 10k tok/s result.

### Scaled full-K experiment

The next slice proved that AIE2P int32-to-f32 conversion and f32 multiply are
correct in isolation, then added an experimental `w4-scaled` cache. Row 2 keeps
the exact W4 GEMM unchanged; row 3 applies activation/weight scales and performs
f32 K-group accumulation. Resident weights remain device-backed. The final
eight-column M=256/K=768/N=256 path passed all-ones and AWQ `+/++` CPU-oracle
parity (maximum absolute error 1e-6) and measured 0.3675 ms for the production
projection seam.

It is deliberately not selected by the default projector. A complete local
OQ4++ hybrid run with the scaled cache inventory regressed to 237.632 ms per
short encode, 71.5 input tok/s, and 3.6 package tok/J; minimum GPU cosine was
0.98575091, still below 0.999. The experiment proves scaled projection math but
does not satisfy mixed/OQ8 scaling, full-model residency, quality admission, or
the 10k tok/s target. The next useful boundary is a single resident layer/model
schedule that removes per-projection XRT and GPU round trips, not another
projection-only scale micro-optimization.

### Shared GPU/AIE physical-I/O checkpoint

The production scaled whole-array W4/W8 path now accepts GPU-exported dma-bufs
for both activation and output arguments. A HIP producer applies optional AWQ,
the canonical signed FWHT-256, and per-row/group int8 quantization directly into
the AIE physical activation layout. After the blocking AIE dispatch, a HIP
consumer deblocks the physical f32 output directly into one, two, or three
row-major role tensors. There is no CPU activation packing, output unpacking,
GPU download, or GPU upload in this projection seam. Weight and shared-buffer
imports remain resident after lazy initialization.

The direct GPU -> dma-buf -> AIE2P -> dma-buf -> GPU oracle gate is exact at
M=256 for W4, W8, AWQ `+`/`++`, and a non-multiple-of-256 K tail:

- W4 K768/N768: maximum absolute error 4e-7;
- W8 K768/N1280: maximum absolute error 7.6e-6;
- W4+ K1152/padded-K1280/N768: maximum absolute error 4e-7;
- W8+ K1152/padded-K1280/N768: maximum absolute error 1.34e-5.

The tail test caught a one-code activation discrepancy caused by multiplying by
a reciprocal before rounding. Matching the CPU contract's exact divide-then-
round order removed it; the gate also compares the physical packed activation
bytes and scales against the CPU oracle before timing.

One warmed 20-iteration M256 projection inventory measured:

| mode | combined qkv | o | combined gate/up | down | 24-layer total | projection-only tok/s |
|---|---:|---:|---:|---:|---:|---:|
| W4 | 0.4296 ms | 0.2937 ms | 0.5714 ms | 0.3663 ms | 39.86 ms | about 6,422 |
| W8 | 0.4617 ms | 0.3007 ms | 0.6254 ms | 0.4037 ms | 43.00 ms | about 5,954 |

These are shared-boundary projection inventories, not fully resident model
results, and are still above the 25.6 ms total-model budget for 10,000 input
tokens/s before attention or norms. The short-document OQ8+ hybrid remained
stable at minimum GPU/NPU cosine 0.99991620 and 79.7 input tok/s after warmup;
it still runs attention, norms, residuals, activations, pooling, Dense heads,
and final normalization on GPU/host.

The next required slice is a resident layer schedule. In particular, the FFN
can stream combined gate/up tiles through GeGLU and the down projection without
materializing the full intermediate or crossing engines; the attention block
needs an analogous qkv -> headnorm/RoPE -> bidirectional attention -> o pipeline.
Arbitrary mixed Opus overlays must join that same schedule rather than falling
back to the host or multiplying per-overlay dispatches.

### Row-major and first resident-FFN checkpoint

R16 changes the scaled whole-array output contract from core-block order to a
padded row-major dma-buf. Hardware parity remains exact for W4, W8, AWQ
`+/++`, and padded-K inputs. The M256 projection inventory is effectively
latency-neutral (about 1.70 ms per W4 layer and 1.76 ms per W8 layer), but the
row-major boundary lets another AIE program consume a projection without a GPU
deblock kernel.

R17 adds an all-32-core row-major GeGLU stage. Its standalone M256/I1152 gate
reports cosine 0.99999825, maximum absolute error 0.02594, and 0.2425 ms per
resident dispatch. A shared-dma-buf projection -> GeGLU chain is correct for
both W4+/++ and W8+/++, but separate XDNA contexts spend roughly 5.3-5.6 ms
per chain on imported-buffer cache reconciliation. That chain is a zero-copy
boundary proof, not a performance path.

R18 removes that boundary by interleaving gate/up weight columns within each
projection stripe and applying GeGLU to the retained scaled accumulator before
the tile is released. The full `[256,2304]` gate/up intermediate never leaves
the array. Independent production-packer projection output and a CPU
integer/scaling oracle agree before the nonlinear comparison.

| fused kernel | cosine | max abs | resident dispatch |
|---|---:|---:|---:|
| W4 gate/up + GeGLU | 0.99999978 | 0.0001497 | 0.5808 ms (0.5709-0.5829) |
| W8 gate/up + GeGLU | 0.99995608 | 0.0296386 | 0.6750 ms (0.6667-0.6929) |

W8 uses 24 logical columns in a 32-lane physical output stripe. An 8-lane
compacted store was not stable with the surrounding W8 MMUL program, and
inlined vector-argument nonlinear helpers corrupted live registers. Explicit
padding plus pointer-based `noinline` 16/8-lane helpers is correct. This is a
fused resident kernel result, not a complete FFN or full-model result: the down
projection, its AWQ/FWHT activation preprocessing, arbitrary mixed overlays,
attention, norms/residuals, pooling, Dense heads, end-to-end throughput, and
package tokens/J remain open. The 10k/15k input-token targets are therefore not
yet admitted.

### Canonical down-activation checkpoint

R19 adds an all-32-core AIE2P baseline for the exact activation contract between
GeGLU and the padded `K=1280` Opus down projection: AWQ division, seed-42
pre-signs, five Hadamard-256 transforms, `1/16` normalization with seed-1042
post-signs, five symmetric int8 scales, and divide-then-`roundf` quantization.
Quantized values and scales share a 1312-byte row record because separate
output FIFOs exceed the memory tile input-DMA channel count.

The AIE compiler produced infinities for a naive indexed AWQ division loop and
silently truncated scalar float-to-int8 conversions despite the nearest-round
mode. Fixed-width pointer helpers repair division; an explicit signed half bias
plus toward-zero saturated conversion exactly reproduces `roundf`.

Three independent M256, 100-iteration hardware runs agree with the CPU oracle
for all 327,680 int8 values, with maximum scale error `7e-9`. Dispatches were
6.9405, 6.9061, and 6.8772 ms (median 6.9061 ms), or about 37k rows/s for this
stage alone. This is correctness evidence for standalone preprocessing, not a
resident GeGLU-to-down chain: R18 output still needs an in-array layout bridge,
the scalar FWHT/quant path must be vectorized or fused, and the down projection
has not yet consumed R19 output. Full-FFN, full-model, 10k/15k input-token, and
package tokens/J claims remain open.

### Vector down-activation checkpoint

R20 vectorizes the R19 contract without changing its physical output: 16-lane
AWQ divide/sign, filter/interleave FWHT butterflies for strides 1 through 8,
16-lane add/sub butterflies for strides 16 through 128, vector post-sign/max,
and vector normalization/int8 conversion. The vector reciprocal paths were
accepted only after all 327,680 physical int8 values matched the exact R19 CPU
oracle; maximum scale error is `1.1e-8`.

Three independent 100-iteration M256 runs measure 0.3106, 0.3221, and 0.3246
ms (median 0.3221 ms), 21.4 times faster than R19's 6.9061 ms median and about
795k rows/s for the standalone stage. Summed serially over 24 layers, 0.3221 ms
is about 7.73 ms, so preprocessing alone no longer exhausts the 25.6 ms
full-model budget. The GeGLU-to-down integration gate remains pending; this is
still not a complete FFN, overlap proof, full-model throughput result, or
package tokens/J result.

### First combined pack-to-down checkpoint

R21 combines vector canonical activation preprocessing and the resident scaled
W4 `K=1280, N=768` down MMUL in one AIE2P dispatch. Each 16 KiB weight block
stores its group's 3,072-byte AWQ/sign payload in the unused tail, avoiding an
illegal third input DMA channel. A packer core column produces each 24-row
activation block, consumes it locally, and broadcasts it directly to the other
seven compute columns without using the memory tile's already-full input DMA
budget. The compact group stream also removes R21's initial fivefold redundant
full-row traffic.

The combined hardware gate reports zero mismatches across 196,608 W4 outputs
and maximum absolute error `1.4e-6`. Three 100-iteration runs measure 2.6385,
2.5797, and 2.4033 ms (median 2.5797 ms). A redundant per-core pack variant
measured 2.4386 ms in one 20-iteration run, showing that direct broadcast is
cleaner but serial 24-row packing and synchronization remain the latency
problem. R21 is a combined pack/down projection result, not a complete FFN:
its group stream is still host-arranged rather than emitted directly by R18,
only W4 is integrated, arbitrary mixed overlays are absent, and attention,
full-model throughput, 10k/15k admission, and package tokens/J remain open.
The next schedule must stripe the 24 rows across columns and gather one packed
activation block before the down MMUL.

### Parallel pack-to-down checkpoint

R22 stripes each 24-row activation block across all eight compute columns, so
every core vector-packs three rows rather than one core packing all 24. Each
core keeps a 784-byte fragment and all-gathers the other seven fragments over
direct core streams, leaving both DMA inputs available for activations and
weights. Adjacent columns share six-row memory-tile outputs to remain within
the memory tile's output-DMA limit.

A closed stream ring did not make forward progress: the circuit-switched path
has no cycle-breaking buffer, and Peano also initially bundled blocking send
and receive operations into one VLIW instruction. The admitted schedule uses
eight acyclic token broadcasts over the physical ring. Each owner sends through
seven hops; its predecessor receives without forwarding, so no broadcast
closes the cycle. An acyclic `col0 -> col1` probe first proved direct-stream
payload correctness before enabling the complete all-gather.

Three independent 100-iteration M256 W4 runs agree with the CPU oracle for all
196,608 outputs, with maximum absolute error `1.4e-6`. Dispatches are 0.9967,
0.9974, and 0.9743 ms (median 0.9967 ms), 2.59 times faster than R21's 2.5797
ms median. This is still a combined pack/down projection rather than a complete
FFN: R22 consumes a host-arranged group stream instead of R18's in-array GeGLU
output, and W8, arbitrary mixed overlays, attention, full-model 10k/15k
admission, and package tokens/J remain open.

### W8 combined pack-to-down checkpoint

R23 applies R22's exact three-row vector pack and acyclic token broadcasts to
the W8 down projection. W8 uses `LM=3`, `MR=8`, 48 columns per compute column,
and two N-macros for `N=768`. Linking the original separate W8 init/accumulate
functions with the packer overflowed 128 KiB program memory, so R23 uses one
compact MMUL body with a runtime accumulate flag.

The first correct W8 schedule repeated packing and broadcasts for both
N-macros and measured 3.8961 ms. Replacing byte-at-a-time physical activation
insertion with aligned 32-bit copies reduced that to 1.6915 ms. The admitted
schedule retains each complete activation block and applies both N-macro weight
blocks into two output FIFO slots, packing and broadcasting each group once.

Three independent 100-iteration M256 runs agree with the CPU oracle for all
196,608 outputs, with maximum absolute error `1.4e-6`. Dispatches are 1.1953,
1.2100, and 0.9956 ms (median 1.1953 ms). The shared generator rebuilds R22 W4
with exact parity at 1.0052 ms in a 20-iteration regression run. R23 is combined
OQ8 preprocessing/down evidence, not a complete FFN: R18-to-R23 in-array GeGLU
streaming, arbitrary mixed overlays, attention, full-model 10k/15k admission,
and package tokens/J remain open.

### Arbitrary mixed combined pack-to-down checkpoint

R24 extends the R22 W4 combined schedule with the existing Opus sparse-overlay
contract. Canonical compact blocks still store `(index, W8 replacement)`; the
host decoder already turns these into signed `(index, delta)` chunks, which R24
applies to the packed int8 activations after each resident W4 group MMUL. Cache
specialization accepts every legal mixed count from 1 through 62 and grows the
weight FIFO only when the W4 data, scales, AWQ/sign payload, and overlays exceed
16 KiB.

The common three-overlay gate (`oq4.25`) agrees with the CPU oracle for all
196,608 outputs, with maximum absolute error `5.7e-6`, and averages 8.9671 ms
over 50 dispatches. A maximum-size 62-overlay specialization also compiles,
runs on AIE2P, and passes all outputs with maximum absolute error `1.53e-5`; its
single measured dispatch is 36.1533 ms. This proves the full arbitrary mixed
format range at the combined down seam, but the scalar sparse gathers are a
large performance regression relative to R22's 0.9967 ms W4 median and remain
unadmitted for throughput. R18-to-R24 in-array GeGLU handoff, attention,
full-model 10k/15k admission, and package tokens/J remain open.

### First complete resident W4 FFN checkpoint

R25 keeps the R18 W4 gate/up accumulator on the array, applies GeGLU, performs
the canonical AWQ/FWHT/int8 down-activation transform, and accumulates all five
W4 down groups in one full-M/full-K AIE2P dispatch. The 32 cores exchange gate
tiles and three-row quantized fragments over the acyclic direct-stream schedule;
no gate/up or down intermediate crosses to the host.

The gate accumulator and down accumulator share the only core-local 9 KiB
output tile. R25 therefore spills the partial down result between the three
gate nblocks as BF16 across otherwise idle scratch buffers, then restores it
before the next down groups. A random CPU integer/scaling oracle reports
`0.99999649` cosine, `0.0032624` maximum absolute error, and `0.00033742` mean
absolute error. Identity isolation passes all five down groups with cosine
`0.99999846` through `1.0`. Fresh-context dispatches measured 3.79-4.35 ms.

Sustained submission is not yet admitted. On the installed amdxdna
driver/firmware, a 39- or 42-weight-object stream deterministically reaches the
four-second TDR on the tenth command submitted to one hardware context. A
36-object stream remains stable. Increasing the core stack, using fresh command
BOs, and splitting the host weight DMA into smaller awaited tasks did not remove
the failure; the split-task schedule also broke numerical ordering and was
discarded. Eight fresh contexts completed 32 full parity-checked dispatches
without a TDR, so bounded context recycling was the first evidence-backed
mitigation. It is not the current workaround: the later kernel parameter
prevents the issue without requiring LDS avoidance. Preserve that parameter
when deriving kernels, and choose LDS use or avoidance independently from
measured bandwidth, tile-memory pressure, and compute performance.

R25 is complete-FFN W4 kernel evidence, not arbitrary-mixed/OQ8 FFN support or
full-model NPU evidence. Attention, norms/residuals, pooling, Dense heads,
end-to-end 10k/15k admission, and package tokens/J remain open.

### Real-artifact resident W4 checkpoint

The reusable runtime now uploads all 24 layers of R25 weights once, uses a
compact shared GPU-to-AIE activation layout, and recycles the XDNA hardware
context after seven commands. Twenty synthetic full-FFN submissions completed
without the tenth-command TDR; recycling increased the amortized standalone
dispatch from roughly 3.8-4.0 ms to 4.25 ms.

The offline Opus producer now zero-pads each ragged matrix row independently to
`ceil(K/256)` groups while retaining the logical HFQ shape and AWQ sidecar
length. Its LDLQ path extends the logical Hessian with zero-covariance padded
coordinates before the block FWHT. This removes the previous Q8 fallback for
EmbeddingGemma's `[768,1152]` down projection across OQ4, OQ8, arbitrary mixed
OQ, and their `+`/`++` recipes. A generated
`EmbeddingGemma-300M--npu.oq4.hfq` therefore carries qt=34 for all 24 gate, up,
and down matrices; generated artifacts remain under `~/.hipfire`.

That artifact selected the complete resident FFN in a real M=256 hybrid model
run. A same-process comparison against the established per-projection Opus path
reported embedding cosine `0.99949509` and maximum absolute error `0.00368346`,
which admits the R25 integration as a correctness path. The candidate's
selection cosine against BF16 was only `0.92908514`, so padded OQ4 down weights
are not a quality promotion over the former Q8-down recipe.

Performance is also not admitted: the per-projection fallback took 288.7 ms per
M256 hybrid encode, while the resident-W4 path took 310.5 ms (about 824 input
tokens/s at 24.08 W, or 34.2 package tokens/J). These remain hybrid results:
attention, norms, residuals, pooling, and Dense heads still execute outside the
NPU. R25's dispatch and context-recycle costs exceed the projection calls it
replaces, so the next resident schedule must fuse across layer boundaries and
add W8/mixed execution rather than treating complete-FFN residency alone as a
throughput win.

### Dense-W8 resident contract checkpoint

Native OQ8 and compact mixed OQ now expose one resident compute contract.
Native OQ8 borrows each decoded int8 group directly; compact mixed storage adds
its sparse signed deltas to the W4 base exactly once during weight upload. The
result is the original int8 replacement value, so overlay count never enters
the future AIE dispatch API. OQ4 remains the distinct native-W4 resident mode.

Real padded-down artifacts were generated under `~/.hipfire` for OQ8,
OQ4.125 (one overlay per group), and OQ6.5 (39 overlays per group). Every layer's
gate/up/down tensors retained qt=35 or qt=36 respectively, including logical
`K=1152` down projections. M256 full-K hybrid references against BF16 measured:

| format | BF16 cosine | hybrid ms | input tok/s | package W | package tok/J |
|---|---:|---:|---:|---:|---:|
| OQ8 | 0.99962372 | 293.0 | 873.6 | 27.04 | 32.3 |
| OQ4.125 | 0.92552006 | 370.9 | 690.2 | 28.03 | 24.6 |
| OQ6.5 | 0.95758063 | 365.1 | 701.2 | 30.01 | 23.4 |

These results validate the generic producer and established per-projection
full-K path only. They are not resident-W8/mixed or full-model NPU results. The
next kernel must consume the dense-W8 resident groups for fused gate/up GeGLU
and down execution without materializing the intermediate outside AIE2P.

### Dense-W8 complete-FFN R26 checkpoint

R26 compiles one AIE2P command containing resident W8 gate/up, GeGLU, canonical
AWQ/FWHT activation packing, and the resident W8 down projection. The graph
uses one shared core output FIFO to stay within the memory tile's two inbound
DMA channels. A depth-one 16 KiB memory-tile weight FIFO fits only after the
other core buffers are tightened; the generated xclbin and instruction stream
live under `~/.hipfire/npu` and are reproducible with `r26_cache.sh`.

Hardware isolation found and fixed three independent layout defects. Gate/up
weights require a 48-column `gate16,up16,gate8,up8` interleave rather than the
generic contiguous W8 column order. Group windows beginning 8 or 16 floats into
a gathered 24-wide chunk require `aie::load_unaligned_v`; aligned `load_v`
collapsed group 2. Finally, the padded fifth group needs a 56-chunk physical T
row so its zero tail does not cross into the next token row. With those fixes,
isolated group 3 reports cosine `0.99986195`, group 4 reports `0.99984475`, and
one complete all-group run reports cosine `0.99985511`, maximum absolute error
`0.0089338`, mean absolute error `0.00180048`, and a 3.6269 ms dispatch.

R26 is not admitted yet. Independent fresh-process repeats intermittently lose
part of the down result, with observed cosine ranging from roughly 0.82 to
0.99 while the retained gate/GeGLU result remains stable at `0.99989984`.
Direct shim weight broadcast, scalar output stores, oversized combined output
DMA, raw contiguous output capture, and an exact six-row input object did not
remove the variance; the memory-tile weight FIFO and padded 9216-byte pair
object are the best observed schedule. The current blocker is repeatable
down-stream/ring delivery, not static resource allocation or the mathematical
W8 contract. Runtime integration, real OQ8/mixed artifacts, attention and
remaining model stages, the 10k/15k admission gate, and package tokens/J remain
pending until this repeatability failure is eliminated.

### Dense-W8 R26 runtime-admission update

Fresh-context isolation showed that the apparent down-stream variance was a
hardware-context initialization effect rather than random ring delivery. The
first R26 command in each newly created context can produce an incomplete down
result; discarding that command makes subsequent commands bit-for-bit
repeatable at the established oracle tolerance. Five independent all-group
processes with one prime command all reported cosine `0.99985511`, maximum
absolute error `0.0089338`, and mean absolute error `0.00180048`. The reusable
runtime now counts the prime command, permits at most six commands per context,
and primes again after bounded context recreation.

The production executor uploads every layer's packed weights once, owns the
retained-gate scratch buffer, imports reusable GPU/XDNA input and output
dma-bufs, and accepts either native OQ8 or any compact mixed Opus matrix through
the same dense-int8 upload contract. Its GPU activation producer replicates the
canonical W8 tile into the four `9216`-byte memory-tile consumer windows. A
20-iteration synthetic run, including repeated context recreation and priming,
passed sustained final-output parity at cosine `0.99985511` and averaged
`3.9663 ms` per measured complete-FFN dispatch.

Real 24-layer hybrid checks at 32 input tokens selected the resident FFN for
native OQ8, OQ6.5 (39 overlays per group), calibrated OQ8+, and LDLQ OQ8++.
Same-process comparisons against the established per-projection NPU path were:

| format | resident vs projection cosine | max abs | resident hybrid ms |
|---|---:|---:|---:|
| OQ8 | 0.99981368 | 0.00254846 | 286.408 |
| OQ6.5 | 0.99985284 | 0.00219250 | 262.117 |
| OQ8+ | 0.99986529 | 0.00215597 | 283.084 |
| OQ8++ | 0.99982262 | 0.00257182 | 284.621 |

The OQ8+ and OQ8++ checks used newly generated ragged-padding artifacts under
`~/.hipfire/models/embeddinggemma-300m/`, with all K=1152 down matrices kept in
qt=35 and the existing AWQ sidecars consumed generically. `++` introduces no
runtime branch: its LDLQ-adjusted values use the same OQ8+ resident encoding.

This admits R26 as a reusable complete-FFN correctness path, not as a full-model
NPU or performance result. These measurements still execute attention, norms,
residuals, pooling, Dense heads, and final normalization outside AIE2P. The
32-token OQ8+ hybrid measured only `113.0` input tok/s at `24.01 W`, or `4.7`
package tok/J; the 10k/15k M256 target remains unproven and cannot be evaluated
as fully resident throughput until the remaining encoder stages move onto the
NPU.

### R26 context-lifetime and M256 checkpoint

The initial six-command recycle bound was inherited conservatively from the R25
W4 stream and was not an R26 limit. After the mandatory first-command prime,
R26 completed 20, 100, and 1,000 measured commands in one context with unchanged
final parity (`0.99985511` cosine, `0.0089338` maximum absolute error). Dispatch
averages improved from `3.9663 ms` with frequent recreation to `2.7431 ms`,
`2.6567 ms`, and `2.6309 ms` respectively. The runtime now uses a finite
evidence-backed 1,000-command bound before recreating and priming again.

Ten consecutive real OQ8+ M256 encodes (240 layer-specific weight commands in
one process) retained the same BF16-reference embedding cosine `0.99975753` and
maximum absolute error `0.00293646`. The warmed hybrid average was `283.018 ms`,
or `904.5` input tok/s at `25.43 W` and `35.6` package tok/J.

Three independent one-encode M256 processes measured `294.950-297.011 ms`
(median `296.073 ms`), `861.9-867.9` input tok/s (median `864.7`), and
`31.0-33.2` package tok/J. These remain hybrid full-encoder measurements, not
fully resident NPU results: only each complete FFN is resident, while attention,
norms, residuals, pooling, and Dense heads still execute elsewhere. Even the
isolated `2.6309 ms` FFN command would consume about `63.1 ms` over 24 layers,
so both kernel acceleration and removal of the remaining stage boundaries are
required for the `25.6 ms`/10k target.

### M256 layer-phase attribution and R27 boundary

A synchronized M256 trace now separates the current hybrid layer into qkv,
attention-core, output-projection, FFN-core, and FFN-output phases. For OQ8+
across 24 layers it measured:

| phase | total | per layer |
|---|---:|---:|
| input norm + qkv shared projection | 71.906 ms | 2.996 ms |
| Q/K norm + RoPE + GPU bidirectional attention | 14.264 ms | 0.594 ms |
| o shared projection + post-attention norm/residual | 66.214 ms | 2.759 ms |
| pre-FFN norm + resident R26 command | 138.011 ms | 5.750 ms |
| post-FFN norm/residual | 1.641 ms | 0.068 ms |

The GPU-only reference in the same process spent `2.806 ms`, `3.933 ms`,
`1.686 ms`, `5.028 ms`, and `0.679 ms` on those phases respectively. Internal
GPU event profiling also measured all 145 RMSNorm launches at only `4.573 ms`.
This rules out standalone norm offload as the next optimization: projection
commands and engine-boundary cache reconciliation dominate, while the actual
bidirectional attention core is comparatively small.

R27 should therefore be one resident layer command spanning input norm and
qkv, per-head norm/RoPE, bidirectional attention, o projection, both residual
norms, and the complete FFN. Existing AIE2P headnorm/RoPE and BF16 softmax
kernels under `tools/npu/` are reusable arithmetic references; R27 still needs
resident QK/PV tiling and a layer-level stream schedule. A collection of
standalone elementwise NPU dispatches would preserve the measured boundary
cost and is not an admissible substitute.

### R27 BF16 MMUL attention checkpoint

The first 32-core M256 attention schedule established the complete online-
softmax QK/PV dataflow with four query rows per core and 16-key streamed KV
blocks. Its scalar vector-dot/PV implementation passed the CPU oracle at
`0.99999410` cosine, `0.0002527` maximum absolute error, and `0.00005579` mean
absolute error, but averaged `2.8596 ms` per sustained dispatch.

The admitted arithmetic now uses the AIE2P native BF16 `mmul<4,8,8>` shape for
both QK and PV. Q is packed as 4x8 head-dimension tiles, K as transposed 8x8
dimension/key tiles, and V as 8x8 key/dimension tiles. The blockwise online
maximum, BF16 exponential, running sum, and f32 retained PV accumulator remain
unchanged. Pairing adjacent 8-column PV results into 16-lane accumulator
updates also avoids a Peano legalization failure on a scalar sum/reduction
shape; reducing the concatenated 16 softmax weights in one operation is the
working compiler contract.

Three independent 100-command hardware processes retain exactly the same
oracle metrics and measure `0.9074`, `0.9192`, and `0.9405 ms` per dispatch
(median `0.9192 ms`). This is a 3.11x reduction from the scalar schedule's
`2.8596 ms`, but it is still standalone BF16 attention-kernel evidence. Summed
serially over 24 layers it would consume about `22.1 ms`, leaving essentially
no room in the `25.6 ms`/10k full-model budget for projections, norms, FFN,
pooling, or dispatch. R27 therefore still requires one resident layer command,
direct qkv-to-attention layout production, projection/FFN overlap or further
kernel acceleration, tail stages, end-to-end quality, and package tokens/J.

The direct projection handoff must preserve those MMUL layouts. An experimental
row-major BF16 K/V input used 8x8 gathers plus an in-register K transpose in the
attention core. It retained exact oracle parity but regressed a 20-command run
to `3.9442 ms`, more than four times the packed median, and is rejected. QKV
projection must therefore emit K as dimension-by-key 8x8 tiles and V as
key-by-dimension 8x8 tiles rather than leaving either role row-major for the
attention core to gather.

R27 no longer stores that packed K/V sequence once per six Q groups. Its DMA
task replays one 16-block M256 K/V sequence six times inside the command. The
resident argument falls from `1,572,864` to `262,144` bytes while parity remains
`0.99999410` cosine and `0.0002527` maximum absolute error. Three independent
100-command processes measure `0.8893`, `0.9506`, and `0.9792 ms` (median
`0.9506 ms`), overlapping the earlier packed range. This is the admitted
projection-facing contract: one packed Q tensor, one packed K/V sequence, and
internal K/V replay; it remains attention-kernel rather than full-layer or
full-model evidence.

### R28 Q/K headnorm and RoPE pack checkpoint

R28 consumes the physical output staging contract intended for the combined
QKV projection and produces R27's packed Q plus single-replay packed K/V
arguments directly. All 32 cores normalize and rotate four Q rows at a time;
the 16 even-column cores normalize and rotate eight K rows in two waves, while
V is a bit-exact layout transform. Cosine and sine values are precomputed as
BF16 in the persistent tails of each raw Q/K row object because the installed
AIE2P compiler does not accept the documented `aie::sincos` API for this
target. The future projection can overwrite only the raw prefixes while those
immutable position tails remain resident.

The initial vector implementation exposed an AIE register-lifetime failure:
Q lower halves and early K rows were plausible, but live vector arguments and
two returned rotation vectors corrupted later results. Scalar RMS reduction,
identity RoPE, and a larger core stack did not remove it, while V stayed
bit-exact. Pointer-based `noinline` Q lower/upper stores and a 16-value K row
scratch before the 8x8 transpose eliminated the corruption. Restoring the
vector RMS reduction then preserved parity and improved timing over the scalar
diagnostic.

The real precomputed-RoPE CPU oracle reports Q cosine `0.99999121`, Q maximum
absolute error `0.0078125`, K cosine `0.99999156`, K maximum absolute error
`0.0078125`, and zero V BF16 mismatches. Three independent fresh-process
100-command runs measure `0.3514`, `0.3659`, and `0.3688 ms` (median
`0.3659 ms`). This is a verified projection-output transform and R27 handoff,
not yet a combined QKV projection or resident attention layer: Q/K/V projection
accumulation still has to write R28's raw prefixes in the same command, and the
o projection, residuals, FFN, tail stages, full-model 10k/15k admission, and
package tokens/J remain open.

### R29 resident W8 QKV-to-pack checkpoint

R29 splices the three-group W8 QKV projection into the verified R28
headnorm/RoPE transform and R27 physical pack in one AIE2P command. The
projection's BF16 result never becomes a host-visible output: its output DMA
scatters directly into persistent eight-token records whose immutable tails
hold precomputed position values and Q/K norm parameters, then the same cores
consume those records and emit packed Q plus one single-replay K/V sequence.

The first 48-column projection schedule required a second core-output FIFO and
exceeded the memory tile's input-DMA channel count. A 32-column W8 stripe
(`LN=2`) makes each core's 24x32 result fit the existing 2 KiB attention-output
FIFO. Each core pads to 32 token rows, so the projection DMA writes four
eight-token records and R28 skips the fourth record after every 24 real rows.
This preserves five exact 256-column roles (`Q0`, `Q1`, `Q2`, `K`, `V`) without
format-specific column names or 288-column role padding. Fully unrolling all 15
projection blocks then overflowed program memory; equivalent `scf.for` loops
reduced the graph and linked successfully.

The combined CPU integer/scaling and transform oracle reports projection cosine
`0.99999182` with `0.0000610` maximum absolute error, Q cosine `0.99999193`
with `0.015625` maximum error, K cosine `0.99999198` with `0.015625` maximum
error, and zero V BF16 mismatches. As in R28, direct vector-argument K rotation
helpers corrupted results in the larger linked program; pointer-based
`noinline` lower/upper helpers restore exact handoff behavior. Three independent
100-command processes measure `1.2041`, `1.2118`, and `1.5602 ms` (median
`1.2118 ms`).

This is the first verified complete W8 QKV projection-to-attention-pack command,
not a resident attention layer or full-model result. R27 attention still needs
to be appended to the command, followed by the o projection, residual/norm
operations, FFN, tail stages, generic W4 and dense-mixed upload/dispatch,
end-to-end quality, 10k/15k admission, and package tokens/J.

### R30 resident W8 QKV and attention checkpoint

R30 appends the R27 BF16 MMUL online-softmax attention phase to R29's full-K
W8 QKV projection and pack. Projection activations, raw position records, the
complete 16 KiB packed Q row-join, and each 16 KiB packed K/V block all use the
same phase-ordered activation FIFO; weights retain the other memory-tile input
channel. Packed Q/K/V are command-local intermediates and attention output is
written into a tail of the persistent staging argument, keeping the DPU packet
within its five-argument hardware limit.

Linking projection, Q/K/V transforms, and MMUL attention on every core exceeded
the 16 KiB AIE program-memory limit (`19.9 KiB` at the first attempt). Collapsing
the separate W8 projection init/accumulate specializations removed about
`1.3 KiB` but was not sufficient. The admitted graph partitions code by column:
even cores retain K/V packing, while each odd core loads both adjacent Q tiles,
updates two online-softmax accumulators against one streamed K/V block, and
emits the even result followed by the odd result. Even cores consume the shared
attention stream without linking the attention arithmetic. This fits at `-Os`
without changing R27's BF16 MMUL/softmax contract.

The first six-argument runtime attempt was rejected before execution because
the installed DPU register map accepts at most five arguments. Folding the
393,216-byte attention output into the staging BO tail removes that ABI error.
The resulting hardware oracle preserves R29's projection/Q/K/V metrics and
reports attention cosine `0.99997371` with `0.0001469` maximum absolute error.
Three independent 100-command processes measure `2.8999`, `3.0678`, and
`3.0897 ms` (median `3.0678 ms`). The refactored standalone R29 path also
retains parity and measures `1.2287 ms` in a 20-command regression run.

R30 is resident W8 QKV plus bidirectional attention evidence, not a complete
layer or full model. The o projection, residual/norm operations, complete FFN,
tail stages, W4 and arbitrary dense-mixed dispatch, end-to-end quality,
10k/15k admission, and package tokens/J remain open. Serially repeating the
current `3.0678 ms` command across 24 layers would already take about
`73.6 ms`, so code residency alone is insufficient; subsequent schedules must
fuse the remaining layer roles and recover parallelism or overlap.

### R30 production runtime and generic Opus admission checkpoint

`NpuResidentAttentionDenseW8` now loads the R30 cache as a production runtime,
owns resident layer-specific QKV weights and Q/K norm/RoPE parameters, shares
the GPU-visible M256 input allocation, primes once per context, and keeps a
finite 1,000-command recycle bound. Its upload API accepts an
`OpusPackedMatrix` or format-neutral dense groups and scales. Native OQ4 and
OQ8 groups and arbitrary compact mixed groups all pass through
`group_dense_i8`; mixed sparse overlays are expanded once at upload rather
than becoming format-specific dispatch branches. AWQ metadata is preserved by
the same API, so `+` and LDLQ-adjusted `++` values do not introduce execution
mode names or branches.

The EmbeddingGemma projector concatenates the separately stored Q/K/V roles
into the physical N=1280 R30 groups, loads per-layer norm and local/global RoPE
parameters, and exposes one `project_attention` boundary to the encoder. A
real OQ4 artifact initially failed admission because the loader incorrectly
required the *source* resident mode to be dense W8 even though the R30 upload
contract already densifies every Opus encoding. Removing that contradictory
source-format check, while retaining exact K/N/group and shared-AWQ checks,
admits OQ4 through the same API.

The raw production runtime reproduces the R30 hardware oracle at cosine
`0.99997371` and maximum absolute error `0.0001469`. M256 full-encoder probes
also selected resident attention for every tested source format:

| source format | resident vs established projection fallback cosine | max abs | hybrid ms |
|---|---:|---:|---:|
| OQ4 | 0.99979383 | 0.00238923 | 356.324 |
| OQ4.125 mixed | 0.99954844 | 0.00361437 | 320.116 |
| OQ6.5 mixed | 0.99976164 | 0.00249144 | 317.656 |
| OQ8 | 0.99965537 | 0.00363689 | 341.277 |
| OQ8+ | 0.99975610 | 0.00380161 | 334.099 |
| OQ8++ | 0.99967772 | 0.00416599 | 335.726 |

These are generic runtime-admission and hybrid correctness results, not a
fully resident model or performance admission. The temporary bridge reads
R30's head-major attention output to the CPU, converts it to token-major, and
uploads it to the GPU for the output projection. Consequently the probes reach
only `718.4-805.9` input tok/s and `28.6-32.2` package tok/J, substantially
below both the GPU reference and the target. Low-bit padded NPU artifacts also
differ materially from their unpadded GPU comparison artifacts (OQ4 cosine
`0.97213209`, OQ4.125 `0.97449738`, and OQ6.5 `0.98404664`), so those
cross-artifact comparisons are not evidence against the much tighter
same-artifact resident/fallback comparison. The next admitted boundary must
remove this output crossing and include output projection plus residual/norm;
the 10k/15k and fully resident package-efficiency gates remain open.

### R31 packed attention-to-output-projection checkpoint

R31 extends the resident boundary with a full-array BF16 output projection.
The schedule follows the output-stationary mapping described by Taka et al. in
"Striking the Balance": A is broadcast across each core row, B across each
column, K is reduced in time, one C tile remains resident, and multidimensional
DMA addressing performs layout conversion instead of a separate repack kernel
(https://arxiv.org/html/2512.13282v1). For M256/K768/N768, all 32 cores process
32x32 output tiles in three K=256 groups. Each core retains one 1,024-float C
buffer; the producer and consumer use 16 KiB A/B objects and 2 KiB core output
objects already proven by R30.

The first attempted single-command append is rejected. It wrote R30 attention
to the external staging argument and read the same region back through a later
shim MM2S task. The first command produced 392,563 non-zero attention bytes but
zero projected bytes; the next command was desynchronized. A separate dense-
ones diagnostic also returned zero, ruling out sparse-weight packing. In
addition, a four-dimensional gather initially transferred only one-eighth of
the intended bytes because the highest BD dimension is a repeat dimension on
this toolchain. This is not an admitted fused result.

The accepted producer contract writes attention at argument offset zero in the
exact `mmul<4,8,8>` A layout. Its S2MM scatter uses an explicit four-row task
repeat, while immutable Q/K position and norm staging follows the 393,216-byte
attention region. A second resident AIE2P context consumes those contiguous
blocks and writes canonical token-major F32 output. The AIE objects are kept
in separate translation units: adding unrelated functions changes Peano's
inlining and reproduced the known R30 register-lifetime failure.

The packed producer retains R30 attention cosine `0.99997371` and maximum
absolute error `0.0001469`. A diagonal full-coverage output matrix exercises
every token, K group, and output column; the production
`NpuAttentionOutputBf16` wrapper reports output cosine `0.99995894`, maximum
absolute error `0.0000901`. One hundred sustained output-projection commands
average `0.9076 ms`. The generic upload accepts any `OpusPackedMatrix`, expands
native OQ4/OQ8 or arbitrary mixed groups once, inverse-transforms each stored
signed-FWHT K=256 weight group to the canonical attention basis, and removes
the AWQ sidecar from the recovered weight because this boundary does not divide
its input activation. There is no format-specific dispatch branch.

The production runtime now allocates one 4,325,376-byte GPU GTT staging dma-buf
per layer and imports the same physical pages as R30's output-first staging
argument and R31's input argument. R31 writes F32 to a reusable GPU-visible
dma-buf, followed by a device-to-device copy into the encoder's existing `o`
tensor; there is no CPU read, token/head transpose, or host upload between the
two contexts. The encoder's attention boundary now distinguishes fallback,
attention-only, and output-projected results so the established output GEMM is
skipped when R31 owns it.

On the real OQ8++ artifact, the full 24-layer resident attention+output boundary
matches the established same-artifact projection fallback at cosine
`0.99978751` and maximum absolute error `0.00278265`. Arbitrary mixed OQ4.125
matches at cosine `0.99972636` and maximum absolute error `0.00404532`.
End-to-end OQ8++ versus BF16 reaches cosine `0.99959564`; OQ4 and OQ4.125 retain
their expected lower quantized-model cosines (`0.92913818` and `0.92766410`).

This is a zero-CPU-copy QKV/attention/output-projection boundary, not a complete
resident layer. First-use per-layer dma-buf allocation/import is still visible
in the first iteration. A three-iteration OQ8++ run reaches `709.7` input tok/s
and `29.5` package tok/J; its warm phase traces spend about `198.7 ms` across
the 24 serial R30+R31 boundaries and `131.5 ms` in the still-separate resident
FFN boundary. This remains far below the final throughput gate. Residual/norm,
cross-context scheduling/overlap, FFN fusion, tail stages, 10k/15k admission,
and competitive package tokens/J remain open.

### R32 direct attention-to-output stream checkpoint

R32 removes the R30-to-R31 external staging round trip and hardware-context
switch. The first direct-stream graph joined four odd-column attention
producers in a memory tile and broadcast a 16 KiB M32 activation block to the
even output-projection cores. It was rejected by the allocator because an even
core would need three FIFO inputs (`abc`, `wbc`, and the new attention join),
while this AIE2P routing contract exposes only two. Moving the joins between
memory tiles did not change that compute-tile limit.

The admitted topology uses the programming manual's neighboring-AIE shared
memory path instead: each odd attention core writes one 4 KiB pair containing
eight token rows, and its adjacent even core acquires three such buffers with
locks, one per K=256/head group. No DMA or third input stream is needed between
those cores. Each even core owns eight tokens, streams all 24 N=32 output
slices, and retains only two 256-float accumulators. Two adjacent slices are
emitted together through the existing 2 KiB `oc` FIFO, so the earlier output
DMA channel is reused rather than allocating another one.

Two implementation failures narrowed the command contract. Fully unrolling
the M8 schedule produced about 32.8 KiB of even-core text and overflowed
program memory; retaining the output-pair loop reduced the linked image enough
to load. Starting all twelve shim output BDs at once transferred only the first
six pairs and left subsequent commands with stale locks. Running the BDs in two
six-pair waves restores full coverage and sustained command reuse.

The hardware oracle covers every token, K group, and output column. R32 retains
projection cosine `0.99999182`, Q cosine `0.99999193`, K cosine `0.99999198`,
and bit-exact V. Its final F32 output-projection result reaches cosine
`0.99997223`, maximum absolute error `0.0000901`, and full 196,608-value
coverage. Sustained 20- and 100-command processes average `4.6948 ms` and
`4.5999 ms` per fused QKV/headnorm/RoPE/attention/output command.

R32 is slower than the sum of the isolated raw R30 and R31 command times, but
it eliminates the much larger runtime hardware-context reconciliation seen in
the hybrid trace and establishes a compilable one-command stream boundary.
The M8 ownership is also the required shape for local full-width RMS reductions:
each even core owns all 768 columns for its eight tokens. The next extension is
to retain the output in BF16 tile memory, apply post-attention norm/residual and
pre-FFN norm locally, then stream directly into the resident FFN. R32 is not a
complete layer or 10k/15k performance admission; residual/norm, FFN, tails,
end-to-end generic-format quality, and package tokens/J remain open.

#### Paper-guided R32 scheduling follow-up

Taka et al., [*Striking the Balance: GEMM Performance Optimization Across
Generations of Ryzen AI NPUs*](https://arxiv.org/html/2512.13282v1), provides a
useful independent explanation for two R32 constraints. Compute tiles expose
two S2MM and two MM2S channels, so the rejected three-input output core was not
an allocator accident. The paper also treats shim BD capacity and
reconfiguration as a first-order system cost: it keeps 15 of 16 BDs occupied,
waits only for the corresponding output-completion token, then retires and
reconfigures the associated input, weight, and output BDs while other transfers
continue.

That completion-driven schedule is the next R32 data-movement experiment. The
current two-wave scheduling workaround drains six output-pair tasks before it configures
the next six, introducing an avoidable command-processor bubble. The follow-up
should test a rolling task window that retires an output task and immediately
reuses its completed BD slot. It should also sweep the amount of contiguous
output covered per task, selecting the smallest span at bandwidth saturation
rather than maximizing L2 occupancy. Multi-dimensional shim and memory-tile
addressing should retain the runtime's standard token-major layout and perform
the tiling in flight.

The paper's balanced-point method is more relevant than importing its best
single-core tile literally. For each candidate schedule, measure core compute,
effective DRAM bandwidth, command latency, and wall-clock fused-layer time;
stop increasing reuse when reduced per-core efficiency outweighs transfer
savings. R32's fused M=256 shape and neighbor-memory attention dependency differ
from the paper's independent large-GEMM mapping, so any scheduling change still
requires the existing full-coverage oracle and sustained multi-command test.

The first rolling-window experiment kept the working six-task capacity but,
after each output-pair completion, immediately freed its task and configured the
corresponding task in the next half-wave. It preserved full coverage and all
oracle metrics, but averaged `4.7357 ms` over 100 commands versus the bulk
six-pair schedule's `4.5999 ms`, a `2.95%` regression. At this small fused shape,
per-pair command-processor reconfiguration costs more than the drain/refill
bubble it removes. Keep the bulk schedule for R32. A subsequent scheduling
experiment should reduce the number of output BDs by aggregating more contiguous
columns or rows per transfer; do not retry finer-grained rolling reuse without
trace evidence that reconfiguration can be amortized.

A second experiment placed two 8 KiB output transfers in one chained shim task,
reducing twelve task submissions to six while retaining twelve BDs. The graph
compiled and loaded, but the second BD did not receive the per-object lock
behavior supplied by separate ObjectFIFO tasks: priming produced only `238892`
nonzero output bytes and the measured output remained zero. Multi-BD aggregation
therefore requires explicit lock/BD construction below the current ObjectFIFO
task abstraction. Do not admit this compiled-but-incorrect chain. The resident
layer work returns to the proven bulk schedule; local residual/norm fusion can
remove the final-output transfer entirely, which is a higher-value boundary than
optimizing this temporary oracle readback path.

The first R33 residual/norm topology then tested three 4 KiB neighbor buffers
with explicit lifetime reuse: even cores used them for QKV accumulation, odd
cores used them for attention accumulators and Q scratch, and even cores
reacquired them for the full M8-by-768 output. This solved the 12 KiB storage
problem without DRAM and passed the AIE data-memory allocator. The residual,
post-attention norm, and pre-FFN norm parameter block also fit one 16 KiB weight
FIFO object per core row.

R33 still failed admission at program load. The even-core image grew from
R32's `0x3bb0` bytes (15,280) to `0x5260` bytes (21,088), exceeding the 16 KiB
program store; the odd image remained near the limit at `0x3fc0` bytes. Reducing
the two K-group output accumulators to BF16 made data memory and linking fit but
could not solve program capacity. Therefore do not add norm code to the current
all-cores QKV image. The next topology must specialize roles: odd cores retain
QKV packing and attention, while even cores retain output projection,
residual/norm, and eventually FFN. Paired QKV weights and a 4 KiB two-lane pack
should be streamed to each odd core so the even image can drop the Q/K/V pack and
projection routines before adding layer-tail code.

### R33 paired-QKV role-specialization checkpoint

R33 implements that odd/even split. Each odd core now projects two adjacent
32-column QKV stripes, emits the two raw and Q lanes sequentially through one
2 KiB output object, and retains the R32 attention role. Each even core drops
the QKV projection accumulator and Q-pack path while retaining K/V packing and
the direct output projection. This follows the paper's single-output-buffer
result: output movement is infrequent relative to the reduction, so reclaiming
L1 for larger resident state is more valuable than double-buffering C.

The first paired layout used a 16,896-byte weight object containing two
8,192-byte int8 payloads and two 128-byte scale records. It still failed
bank-aware and sequential L1 allocation after the output object was reduced to
one buffer. The admitted layout keeps the weight FIFO exactly 16 KiB and moves
all four pairs' 1,024 bytes of column scales into the unused tail of each
16 KiB activation object. This is also consistent with the paper's broader
principle of using in-flight layout transformation and contiguous payloads
instead of expanding the core's local-buffer footprint.

The resulting graph compiles without allocation warnings. Even-core text falls
to `0x3470` bytes (13,424), leaving 2,960 bytes for residual/norm work. Odd-core
text is `0x3fa0` bytes (16,288), only 96 bytes below the 16 KiB program limit,
so further functions must remain on the even image. The full hardware oracle
retains projection cosine `0.99999182`, Q cosine `0.99999193`, K cosine
`0.99999198`, bit-exact V, and output cosine `0.99997223` with maximum absolute
error `0.0000901`.

The role split is a capacity checkpoint, not a speed admission. One hundred
sustained commands average `5.1845 ms`, `12.7%` slower than R32's `4.5999 ms`,
because each odd core performs two projections serially. Keep R32 as the faster
standalone boundary. R33 is justified only if the 2,960-byte even-core budget
admits enough residual/norm/FFN work to remove a larger external boundary; that
is the next experiment.

#### FlatAttention implications for the resident M256 graph

Zhang et al., [*FlatAttention: Dataflow and Fabric Collectives Co-Optimization
for Large Attention-Based Model Inference on Tile-Based Accelerators*](https://arxiv.org/html/2604.02110v1),
reinforces the value of treating several tiles' aggregate scratchpad as one
attention working set, but also identifies "over-flattening": for short or
moderate sequences, expanding the cooperating tile group shrinks per-tile work
until fixed synchronization and movement costs dominate. R33's 12.7% regression
is consistent with that warning; do not spread the fixed M256 projection over
more serial cooperation merely to reduce local code or storage.

The directly applicable asynchronous variant is smaller in scope. The paper
notes that two output-row blocks can be scheduled concurrently while sharing
Q/KV blocks, overlapping one block's matrix work with the other's vector
softmax and data movement. Each R33 odd core already owns two four-query lanes,
two attention accumulators, one query pair, and one shared K/V stream. A
replacement paired-attention kernel should therefore load each K/V tile once,
interleave the two score/softmax/PV updates, and replace the two current
single-lane calls. This must replace, not supplement, the existing attention
routine because the odd image has only 96 bytes of program space. Admit it only
if the full oracle passes and it recovers enough of the paired-projection loss;
avoid larger software reductions or multicast groups unless hardware ObjectFIFO
broadcast/reduction semantics eliminate their synchronization cost.

### R34 resident residual and normalization checkpoint

R34 uses the capacity created by R33 to retain each even core's M8-by-768
output as three 4 KiB BF16 blocks. The output projection writes paired
64-column slices into that local store, then one 16 KiB parameter object carries
the eight-token residual, post-attention norm, pre-FFN norm, and epsilon. The
core applies post-attention RMSNorm, adds the residual, applies pre-FFN RMSNorm,
and leaves the resulting BF16 activation resident for the FFN boundary.

The first allocation exceeded 64 KiB by only 48 bytes. R34 removes the unused
paired-graph K inverse scratch and reuses the later output accumulator for that
earlier K-pack phase. The remaining 16-byte allocator bookkeeping required
reducing the declared stack from 4 KiB to 2 KiB; linked stack-size metadata
reports only 64 bytes. With scalar division replaced by multiplication by the
constant reciprocal of 768, the even image avoids the 1,168-byte software
division helper and fits at `0x3f00` bytes (16,128). The odd paired-QKV and
attention image remains `0x3fa0` bytes (16,288).

A compact diagnostic drains the first 128 normalized dimensions for every
token while the full 768 dimensions remain in tile memory. Across 32,768 probed
values, the hardware result reaches cosine `0.99990586` and maximum absolute
error `0.0546875` against a BF16-aware CPU oracle. The larger maximum reflects
two chained AIE reciprocal-square-root approximations; a one-step Newton
refinement did not change the BF16 result and consumed too much program space,
so it is rejected. Projection/Q/K parity remains unchanged and V remains bit
exact. One hundred sustained fused commands average `5.9733 ms`, including the
diagnostic drain, versus `5.1845 ms` for R33 and `4.5999 ms` for R32.

R34 is the first admitted one-command QKV/headnorm/RoPE/attention/output/
post-attention-norm/residual/pre-FFN-norm boundary. It is not yet a full layer:
the next step must consume the resident three-block BF16 activation directly in
the FFN. Both core images are now nearly full, so FFN compute must replace
diagnostic and phase-specific routines or use a separate role image/context;
simply appending FFN code cannot fit.

### R35 canonical-BF16 resident dense-W8 FFN checkpoint

R35 admits a second-context FFN image that consumes token-major BF16 instead of
R26's 7,962,624-byte GPU-prepacked activation representation. Its logical input
is M256-by-768 BF16, physically padded to 288 rows so every 24-row DMA stripe is
in bounds. Each memory-tile row broadcasts one 24-by-256 BF16 group to all eight
cores. Every core quantizes its three owned rows with the existing AWQ/signed-
FWHT contract, exchanges the eight fragments through the proven ring, and uses
the resident dense-int8 gate/up and down kernels. The external output is compact
token-major BF16 with 288 physical rows and 256 logical rows.

The resident Rust executor selects this ABI from
`mode=dense-w8-canonical-bf16` plus `input=token-major-bf16` in `shape.txt`.
Legacy R26 geometry and raw generated MLIR remain byte-identical. A new
format-independent `OpusPackedMatrix::from_payload` path lets fused residents
decode OQ8 or compact mixed Opus groups without allocating a standalone
projection context; compact mixed groups still expand once at resident upload.
The down-weight stream changes only for this ABI, from R26's
`mblock,group,nmacro` order to `mblock,nmacro,group`.

Two hardware details were required for correctness. First, a 48-byte innermost
DMA row delivered only one 32-byte beat. Each local 24-column output is now
encoded as two 32-byte beats: columns 0-15 followed by duplicated columns 8-15
and columns 16-23. A three-dimensional DMA scatter overlaps the beats by 16
bytes and reconstructs the compact 48-byte destination row. The same scheme is
used for GeGLU and both 24-column halves of each 48-column down macro. Second,
the internal GeGLU tensor uses a 1,280-BF16 physical row stride. Its first 1,152
values are logical and its final 128 values remain zero, preventing down K group
4 from spilling into the next token row.

The clean synthetic OQ8 hardware oracle covers every one of the 256-by-768 final
values and BF16-rounds both resident handoffs. It reaches cosine `0.99984681`,
maximum absolute error `0.0048828`, and mean absolute error `0.00072746`.
The retained GeGLU intermediate independently reaches cosine `0.99990001` and
maximum absolute error `0.0156250`. A 20-run process averages `10.2844 ms` per
executor call, including canonical host-buffer fill/synchronization, the AIE
command, output synchronization, and BF16 readback. This is approximately
24,891 M256 FFN input rows/s for this isolated stage, not an end-to-end model
throughput claim. Each core image uses 14,908 bytes of program text.

R35 is a correctness and ABI checkpoint, not the complete layer. R34 still
retains its normalized 768-wide activation in tile-local memory and R35 still
receives a host-visible argument buffer in a separate context. The next step is
to replace R34's diagnostic drain with a full canonical BF16 handoff, then
measure the two-context layer boundary before deciding whether context fusion or
an explicit zero-copy shared buffer is necessary. Full model execution, generic
OQ4/mixed/OQ8 +/++ admission, the 10k/15k end-to-end target, and package
tokens/joule remain open.

### R36 full R34-to-R35 canonical handoff checkpoint

R36 replaces R34's 128-dimension diagnostic probe with a complete
M256-by-768 token-major BF16 drain. Each of the three local 8-by-256 norm
blocks is emitted as two four-row chunks through the existing 2 KiB output
FIFO. The shim DMA scatters each chunk directly into its canonical row stride,
so the transfer preserves complete local row blocks instead of fragmenting the
hidden dimension. This is the small-group, row-local schedule suggested by
FlatAttention's warning against over-flattening fixed moderate-size work.

The six additional output objects initially overflowed the even AIE program
store. Replacing eight statically duplicated norm-parameter FIFO acquisitions
with one row-aware loop reduced the even image to 16,352 bytes, 32 bytes below
the 16 KiB limit, while preserving the same eight-object stream protocol. The
odd image and all projection, attention, and norm arithmetic remain unchanged.

The hardware oracle now covers all 196,608 normalized values. It reports cosine
`0.99990598`, maximum absolute error `0.0625`, no non-finite values, and no
zeros. The full-output maximum is one BF16 step above the old probe envelope
because it observes all dimensions after two AIE reciprocal-square-root
approximations; the gate uses a `0.065` ceiling while retaining the `0.9998`
cosine floor. Twenty sustained R34 commands average `5.9177 ms`, close to the
`5.9733 ms` diagnostic-probe checkpoint despite draining six times as many
values.

The verifier feeds those exact BF16 bits into R35's canonical ABI and compares
the FFN result against the format-independent CPU matrix oracle. The composed
handoff reaches cosine `0.99990868`, maximum absolute error `0.0117188`, and a
three-run R35 average of `10.0665 ms`. This proves the full two-context data
contract, but it still performs a host-visible copy between separately allocated
buffers. A shared dma-buf or fused-context path, the remaining post-FFN layer
tail, full model execution, generic OQ4/mixed/OQ8 +/++ admission, end-to-end
10k/15k throughput, and tokens/joule remain open.

### R37 zero-copy two-context handoff checkpoint

R37 replaces R36's host-visible canonical copy with one PRIME-exported GTT
dma-buf imported by both NPU contexts. R34's large staging/result argument is
the shared backing. Once QKV projection has finished, its prefix is dead; the
full normalized M256-by-768 BF16 result overwrites that prefix. R35 imports the
same fd as a larger-than-minimum input backing and consumes the canonical prefix
at offset zero. No activation bytes are repacked or copied between commands.
The cache advertises this ABI as `handoff=staging-prefix-dmabuf`.

Two rejected layouts establish the hardware constraints. A dedicated sixth
R34 argument cannot work because the amdxdna DPU regmap admits at most five
arguments. Reusing Q keeps five arguments but is unsafe: R34 begins draining
the first normalized wave while later attention groups still consume Q, which
corrupts the result. Delaying that drain breaks the resident FIFO cadence on
the following command. Reusing the dead R staging prefix preserves the original
task schedule and sustained-command behavior.

The shared-buffer hardware oracle retains full parity: Q cosine
`0.99998086`, K cosine `0.99998099`, V cosine `0.99999192`, full normalized
output cosine `0.99990598`, and zero-copy FFN cosine `0.99990868`. Isolated
averages are `5.9156 ms` for R34 and `9.8379 ms` for R35. A three-run loop that
actually alternates the two hardware contexts averages `21.6999 ms`, or
11,797 M256 layer-boundary rows/s. The approximately 5.95 ms gap over the two
isolated averages is context-switch/scheduling overhead, not a memory copy.

This clears the 10k row/s threshold for this one fused attention-plus-FFN layer
boundary, but it is not the requested full-model 10k input-token result. At the
current alternating-context cost, repeating the boundary over all encoder
layers cannot meet the full-model budget. The next execution slice must avoid a
per-layer R34/R35 context alternation, most likely by fusing the schedules into
one image or batching layer phases within longer-lived contexts while retaining
all inter-layer state on shared buffers. The post-FFN residual/tail, full model,
generic OQ4/mixed/OQ8 +/++ hardware matrix, package-energy sweep, and 10k/15k
end-to-end gates remain open.

### R38 reconstructible attention-residual state checkpoint

R38 extends the R34-to-R35 boundary with the missing scalar state needed to
recover the attention residual after pre-FFN normalization. For each token,
R34 now retains the pre-FFN inverse RMS value alongside the canonical BF16
activation. A downstream tail can reconstruct the residual as
`h / (pre_ffn_norm_weight * pre_ffn_inverse)` without draining a second
M256-by-768 tensor from the attention context.

The inverse state follows a deliberately narrow on-fabric path. Each even core
aggregates its two M-wave vectors into one 64-byte ObjectFIFO object and sends
it to its odd neighbor. The odd core relays that object into the first 64 bytes
of its existing 2 KiB output object; the four core-row objects retain the
existing 8 KiB memory-tile join. This avoids both a sixth DPU argument and a
new memory-tile DMA input channel. A separate even-core output stream broke
steady-state cadence, while a direct even-core-to-odd-memory-tile FIFO was
rejected because the memory tile had no remaining input DMA channel. Relaying
one object per M-wave also overflowed the odd program image; aggregating both
waves removes that loop and halves the output-task count.

Two `minsize` annotations on the attention load/init helpers recover the 96
bytes needed for the relay without changing arithmetic. Final program text is
16,368 bytes on even cores and 16,352 bytes on odd cores. The default R29
generator remains byte-identical because every new graph object and task is
gated by `--residual-norm`.

The locked AIE2P oracle restores the established Q/K/V and normalized-output
parity and measures pre-inverse cosine `0.99999994` with maximum absolute error
`0.0028937`. Reconstructing the full attention residual from the emitted BF16
activation reaches cosine `0.99990720` and maximum absolute error `0.0551744`.
Twenty sustained R34 commands average `5.7747 ms`. The alternating R34/R35
chain averages `22.8842 ms`, or 11,186.7 layer-boundary rows/s, with R35 alone
at `10.2926 ms`. This remains above the one-layer 10k row/s threshold but is
slower than R37's 11,797 rows/s; it is not an end-to-end encoder result.

R38 proves the compact state contract only. The next slice is a resident
post-FFN tail that consumes the R35 down-projection, reconstructs the attention
residual, applies post-FFN RMSNorm, and leaves the completed layer state in a
shared buffer. Full-model execution, generic OQ4/mixed/OQ8 `+/++` admission,
package tokens/joule, and the 10k/15k end-to-end gates remain open.

### R39 resident post-FFN layer-tail checkpoint

R39 completes the EmbeddingGemma layer equation on AIE2P. It consumes R34's
canonical pre-FFN-normalized activation and inverse-RMS state plus R35's
canonical BF16 down-projection, reconstructs the attention residual, applies
post-FFN RMSNorm, adds the residual, BF16-rounds the result, and overwrites the
shared R35 output pages with the completed layer state.

Each of the 32 cores owns eight complete 768-wide tokens. This avoids a
cross-core reduction: every core computes all eight post-FFN RMS reductions
locally. To stay within compute-tile DMA channels, metadata, duplicated static
norm parameters, H, and Y travel sequentially through one input ObjectFIFO.
Each core copies the small inverse and parameter state locally, copies H into
its output object, then consumes Y and finishes in place. Each token row is
split across two memory tiles with four consumers apiece, matching the proven
fan-out geometry instead of over-allocating one memory tile's output channels.

R38's inverse table now begins immediately after the logical M256-by-768 BF16 H
tensor at byte 393,216. It uses 32 records of 12,288 bytes, ordered by core row
and combined wave/active-column. R34's odd relay places its two eight-float
vectors at offsets 0 and 1,024 of the 2 KiB row object; one shim DMA scatters
the eight 1 KiB chunks into the R39 records. The table occupies only R35's
padded input rows and dead R34 staging, so no logical activation is displaced.

R35 now accepts a caller-owned shared output dma-buf. R39 imports that buffer
once and supplies the same BO in both Y-input and output DPU slots. The generic
XDNA submission path keeps both command-packet addresses but deduplicates GEM
residency handles, because amdxdna rejects a repeated handle with `EALREADY`.
This is a general in-place dispatch capability rather than an R39-only special
case.

The R39 core image is 3,360 bytes. Its isolated hardware oracle reaches cosine
`0.99999170` and maximum absolute error `0.0078125`. The full locked
R34-to-R35-to-R39 oracle retains Q/K/V, normalized-H, inverse, residual, and FFN
gates, then measures the completed layer at cosine `0.99987080` and maximum
absolute error `0.09375`. The final three-run sample measures R39 at `3.8005 ms`;
the three-context completed-layer chain averages `24.7080 ms`, or 10,361.0 M256
layer rows/s.

This clears 10k rows/s for one complete encoder layer, not the requested full
encoder. Repeating three hardware contexts per layer cannot meet the full-model
target. The next slice must eliminate per-layer context alternation by batching
layer phases across long-lived contexts or fusing compatible schedules, then
wire every encoder layer, final norm, pooling, and output normalization through
the resident path. Generic OQ4/mixed/OQ8 `+/++` end-to-end admission, package
tokens/joule, and the 10k/15k full-model gates remain open.

### R40 real-model resident-layer pull-up and direct-residual correction

R40 wires the R34, canonical-BF16 R35, and a new direct-residual post-FFN tail
through the production `LinearProjector` layer boundary. The loader owns actual
layer-specific QKV, output-projection, FFN, Q/K norm, RoPE, and layer-norm
payloads for all 24 layers. One shared input feeds R34, its H backing feeds R35,
and R35's output is consumed and overwritten by the tail. An opt-in
`HIPFIRE_EMBED_RESIDENT_LAYER=1` keeps this unadmitted path from replacing the
established encoder until its quality gate passes. A layer-limit and
same-process boundary comparison are available for bisection.

The real OQ8 artifact invalidates R39's reconstructible-residual assumption.
Layers 8, 10, and 12 contain an exact zero at pre-FFN norm dimension 39; layers
20, 21, and 23 contain one at dimension 731. Dividing H by the pre-FFN norm
cannot recover those residual components. Packing a BF16 inverse and exception
into the existing metadata word was rejected: the R34 even image grew to
16,576 bytes, 192 bytes beyond program memory. R40 instead retains canonical
BF16 X and computes the exact architectural tail `X + post_ffn_norm(Y)` without
using pre-FFN inverses. Its isolated hardware oracle reaches cosine
`0.99999166` and maximum absolute error `0.0078125`.

The first real 24-layer OQ8 execution proves that every production layer can
select the new resident boundary, but it is not admitted. Against the BF16
reference it measures cosine `0.32072166`, `665.782 ms`, 384.5 input tokens/s,
22.00 W package power, and 17.5 package tokens/J. A layer-0 same-input bisection
localizes the quality failure:

| boundary | cosine | minimum row cosine | max abs |
|---|---:|---:|---:|
| R34 pre-FFN H vs established H | 0.99996548 | 0.99994784 | 0.8271027 |
| canonical R35 Y vs established Y | 0.99995595 | 0.99986730 | 0.0151756 |
| R40 tail vs a CPU oracle using the same resident X/Y | 0.99999334 | — | 4.0000000 |
| completed layer vs established layer | 0.94956975 | 0.79338516 | 62.4840775 |

The direct tail and both upstream boundaries are individually coherent, but
post-FFN normalization plus residual cancellation amplifies the canonical R35
BF16 output error on the real model. The next correction must preserve more
than one BF16 component across R35-to-tail—prefer an F32 or compensated BF16x2
down-projection handoff and consume it before rounding the completed layer.
Do not admit R40's full-model result or treat its package efficiency as meeting
the goal. OQ4/mixed/OQ8 `+/++`, 10k/15k throughput, and admitted package
tokens/J remain open.

### R41 compensated FFN handoff and architectural-X correction

R41 first tested the R40 hypothesis that canonical BF16 rounding at the R35
down-projection boundary caused the real-model layer collapse. A new generic
resident FFN ABI accepts the same canonical token-major BF16 H input but emits
each F32 accumulator as compensated BF16x2 (`high + low`). High and low travel
as two sequential records over the existing core-to-shim ObjectFIFO, so the
image does not consume another memory-tile DMA channel. The tail consumes
BF16x2 in two-token/four-phase tiles; an earlier four-token/two-phase image
compiled with overlapping FIFO allocations and corrupted later token bands,
so it was rejected. The admitted two-token image has no allocator warnings,
uses a 3,440-byte core program, and reaches cosine `0.99999166` with maximum
absolute error `0.0078125` in the locked tail oracle. The R41 FFN reconstructs
against its F32 reference at cosine `0.99985433`, maximum error `0.0044407`, and
`21.3356 ms` per measured synthetic command.

The BF16x2 experiment exposed a more fundamental R40 bug. `project_layer` is
called before attention, so R40's separate residual buffer contains the layer
input, not the architectural post-attention state X required by
`X + post_ffn_norm(Y)`. R34 already exports canonical H and the per-token
pre-FFN inverse RMS. Reconstructing X from that state changes the real layer-0
comparison from cosine `0.94952030` to `0.99997782` (minimum row cosine
`0.99987070`, maximum error `4.5864868`). This proves that the wrong residual,
not FFN BF16 rounding alone, caused the catastrophic R40 result.

Six real-model layers remain non-invertible because their learned pre-FFN norm
contains one exact zero: layers 8, 10, and 12 at dimension 39 and layers 20,
21, and 23 at dimension 731. The layer-limit bisect shows the largest discrete
drop when layer 8 is admitted (`0.99839473` through eight resident layers to
`0.99488682` through nine). R41 therefore refuses the completed-layer shortcut
for those six layers and uses the established path rather than inventing X.
The resulting 18-layer hybrid reaches final embedding cosine `0.99606335`,
maximum error `0.01134668`, `397.6` input tokens/s at M=256, average package
power `23.03 W`, and `17.3` tokens/J. This is a large correction over R40's
all-layer cosine `0.32072166`, but it is still below quality and throughput
admission and is not a fully resident result.

Two follow-up experiments were rejected. Compensating reconstructed X as
BF16x2 did not recover information already lost in canonical BF16 H and reduced
final cosine slightly to `0.99597490`. Replacing zero pre-norm weights with a
temporary nonzero sentinel, reconstructing X, then restoring zero H columns
before R35 admitted all 24 layers but reached only `0.99083066`; the sentinel
perturbed the rounded H/inverse trajectory. The next correctness slice must
export the one missing X exception value per token from R34 without changing
the model arithmetic. Only after all 24 layers pass the real-model gate should
the host reconstruction/copy be moved fully onto fabric and the context and
throughput work resume.

### R42 zero-norm X-exception export checkpoint

R42 removes the six-layer non-invertibility identified by R41 without changing
the learned pre-FFN norm. Fixed-column AIE2P images retain the exact F32
pre-FFN inverse and export one BF16 architectural X value per token for either
dimension 39 or 731 in otherwise unused metadata-record space. Layers without
an exact-zero norm continue to select the standard R34 image; the six affected
layers select the matching fixed-column image through the same resident-layer
API.

Two compiler details were necessary. A scalar load placed after the block
stores was scheduled against stale projection data. Reloading the containing
16-lane X vector creates the required store/load dependence, while
`chess_copy` keeps the selected block pointer distinct from the restricted
three-block array; without it Peano silently substituted block2 for block0.
The extra vector reload initially exceeded AIE program memory by one 16-byte
instruction packet. `-mllvm -aie-bottomup-cycles=0` recovers that packet and
keeps the post-residual/norm function at the prior 2,000-byte size.

Locked M256 same-input comparisons validate both exception families:

| boundary | layer / column | cosine | minimum row cosine | max abs |
|---|---:|---:|---:|---:|
| exported X vs established X | 8 / 39 | 0.99999805 | — | 9.1132812 |
| reconstructed X vs established X | 8 / 39 | 0.99998621 | 0.99997409 | 9.1132812 |
| completed layer vs established layer | 8 / 39 | 0.99996358 | 0.99987305 | 19.6830444 |
| exported X vs established X | 20 / 731 | 0.99999812 | — | 96.6386719 |
| reconstructed X vs established X | 20 / 731 | 0.99999375 | 0.99998402 | 96.6386719 |
| completed layer vs established layer | 20 / 731 | 0.99998723 | 0.99995838 | 144.5478516 |

All 24 layers now execute the completed resident-layer shortcut, so the R41
six-layer functional blocker is resolved. The resulting OQ8-versus-BF16
embedding cosine is only `0.99114025`, however, with `0.01623118` maximum
error. That is better than the rejected zero-sentinel experiment but still
below the real-model quality gate and far below the established OQ8 path.
The measured `355.6` input tokens/s at `21.07 W` (`16.9` package tokens/J) is
also a host-reconstructed hybrid measurement: metadata is read on the host and
final normalization/pooling remain outside the NPU. It is neither fully
resident nor a performance admission. The next correctness slice must reduce
ordinary per-layer BF16 boundary error across R34/R35/R41 before moving the
reconstruction onto fabric and resuming the 10k/15k throughput work.

### R43 compensated completed-layer output checkpoint

R43 isolates the ordinary rounding loss at the R41 tail output. The existing
tail ABI remains backward compatible with canonical BF16 caches, while a new
generic cache emits each completed architectural state as token-major
compensated BF16x2 (`high + low`). The runtime selects the compensated cache
when present, sizes the shared output dma-buf from its manifest-selected output
encoding, and reconstructs F32 without changing the logical residual tensor
view used by the GPU. The first BF16 plane therefore remains usable for input
staging while the larger backing allocation carries both output components.

The AIE2P kernel stores two rows per token in one output ObjectFIFO record. Its
per-core output tile grows from 3,072 to 6,144 bytes; together with the 9,216
byte input tile, parameters, and 2 KiB stack, it remains inside the 32 KiB tile
memory budget. The artifact is generated under `~/.hipfire/npu/` as
`embgemma_aie2p_post_ffn_direct_tail_bf16x2_completed_bf16x2_m256_k768` and
declares `output=shared-completed-bf16x2`.

Locked same-input comparisons show that the compensated output is active and
improves all three inspected layer families:

| boundary | layer | R42 cosine | R43 cosine | R43 max abs |
|---|---:|---:|---:|---:|
| completed layer | 0 | 0.99997782 | 0.99998185 | 2.6809692 |
| completed layer | 8 | 0.99996358 | 0.99997813 | 16.2114258 |
| completed layer | 20 | 0.99998723 | 0.99999170 | 107.0312500 |

Across all 24 completed resident layers, OQ8-versus-BF16 embedding cosine rises
from `0.99114025` to `0.99557614`, while maximum absolute error falls from
`0.01623118` to `0.01181315`. The one-run M256 sample measures `345.1` input
tokens/s at `21.05 W`, or `16.4` package tokens/J. This remains a
host-reconstructed hybrid measurement and is not a throughput or fully
resident admission.

Opt-in component attribution compares the fallback oracle, resident X plus
resident Y, fallback X plus resident Y, and resident X plus fallback Y. Under
R43 the actual tail closely tracks the resident-X-plus-resident-Y oracle, so
the dominant remaining correctness loss is upstream: architectural X export
and resident FFN projection error are comparable, rather than completed-tail
rounding. The next correctness slice should preserve architectural X directly
across R34, preferably as compensated fabric-resident state, before removing
the host reconstruction boundary.

### R44 direct architectural-X handoff checkpoint

R44 replaces the lossy H-plus-inverse reconstruction boundary with a direct
canonical BF16 architectural-X handoff. The first attempted design streamed
both X and H from R34. It was rejected after exposing four independent AIE2P
constraints: twelve simultaneous output epochs exhausted shim buffer
descriptors; interleaved H/X records could not be scattered to distant planes
with the available three-dimensional DMA layout; a separate 1.5 KiB pre-norm
copy overflowed tile SRAM by 48 bytes; and the smallest dual-state core image
still exceeded program memory by about 1.3 KiB. These failures rule out paying
twice for state at the producer.

The accepted design exports only X in the exact canonical prefix previously
used for H. R34 keeps its original ObjectFIFO graph, DMA schedule, output byte
count, and program-memory footprint; a compile-time tail mode simply omits the
final pre-FFN normalization loop after X and the exact F32 inverse have been
computed. The cache is generated under `~/.hipfire/npu/` as
`embgemma_aie2p_resident_w8_qkv_paired_attention_o_norm_x_bf16_m256_k768_n1280`
and declares `output=canonical-token-major-x-bf16`.

The current correctness bridge reconstructs canonical H from direct X, the
exported inverse, and the layer pre-FFN norm on the host, writes H into the same
shared R35 input pages, and explicitly publishes the host write to the NPU.
The tail consumes direct X, so zero learned pre-norm columns no longer require
fixed-column exception images. This bridge is deliberately temporary: the
next slice must fuse X-to-H normalization into the R35 consumer and remove the
host state rewrite.

Locked same-input comparisons improve every inspected layer:

| boundary | layer | R43 cosine | R44 cosine | R44 max abs |
|---|---:|---:|---:|---:|
| architectural X | 0 | 0.99997447 | 0.99998837 | 0.6121483 |
| completed layer | 0 | 0.99998185 | 0.99998702 | 2.7560730 |
| completed layer | 8 | 0.99997813 | 0.99998436 | 15.4559326 |
| completed layer | 20 | 0.99999170 | 0.99999417 | 118.5166016 |

Across all 24 completed resident layers, OQ8-versus-BF16 embedding cosine rises
from `0.99557614` to `0.99807179`, and maximum absolute error falls from
`0.01181315` to `0.00729077`. The one-run M256 sample measures `359.6` input
tokens/s at `21.07 W`, or `17.1` package tokens/J. It remains a hybrid
correctness result, not a fully resident or throughput admission, because the
host still creates H and final normalization/pooling remain outside the NPU.

### R45 consumer-resident pre-FFN normalization checkpoint

R45 removes the temporary R44 host rewrite of architectural X into normalized
H. The R44 producer already leaves direct canonical BF16 X and all 256 exact
F32 inverse-RMS values in one shared 5.1 MiB backing allocation. The R45 R35
consumer declares the portion through the physical inverse records as its input
argument and gathers the first 512 bytes of each of the 32 records into one
16 KiB preload. That preload travels through the existing per-column weight
ObjectFIFO, so it does not shrink the 24-row by 256-column activation tile or
consume a third shim output DMA channel.

Each core selects only the nine inverse values needed by its three owned rows
across the three M macros into 36 bytes of local state, releases the preload,
and then consumes the unchanged resident weight schedule. The learned BF16
pre-FFN norm group occupies 512 bytes of the existing weight record's unused
832-byte tail. The pack kernel applies `BF16(X * norm * inverse)` before the
established AWQ/FWHT/int8 activation transform, preserving the R44 host
bridge's BF16 boundary including exact zero learned norm columns. The generated
artifact lives under `~/.hipfire/npu/` as
`embgemma_aie2p_resident_ffn_dense_w8_direct_x_bf16x2_m256_k768_i1152_o768`.

An initially attempted independent inverse ObjectFIFO was rejected by the
compiler because every shim already uses both output DMA channels for
activation and weight streams. Reusing the weight multicast follows the
FlatAttention principle of preserving per-tile arithmetic intensity before
adding another collective and compiles without changing the gate/down GEMM
geometry.

Locked OQ8++ comparisons against the established fallback report:

| boundary | layer 0 | layer 8 | layer 20 |
|---|---:|---:|---:|
| normalized H cosine | 0.99998035 | 0.99998560 | 0.99998813 |
| resident FFN cosine | 0.99996978 | 0.99994007 | 0.99997241 |
| completed layer cosine | 0.99998346 | 0.99998120 | 0.99999386 |

Across all 24 completed layers, OQ8++ versus BF16 embedding cosine is
`0.99808222` with `0.00816466` maximum absolute error. The one-run M256 sample
measures `362.6` input tokens/s at `22.02 W`, or `16.5` package tokens/J. This
admits the consumer-resident normalization boundary, but not a fully resident
model: the current tail still receives architectural X through a host copy,
and final normalization and pooling remain outside AIE2P. Pure OQ4 and the
generic mixed-format hardware matrix also remain open for the completed-layer
schedule.

### R46 direct architectural-X tail checkpoint

R46 removes the remaining host read, copy, and cache publication of
architectural X between the resident FFN and the post-FFN tail. The tail now
imports the R44 hidden dma-buf as a separate X argument while retaining the R45
BF16x2 Y buffer and the compensated completed-layer output. In non-debug runs,
the runtime neither decodes X nor reads the pre-FFN inverse plane on the host
before dispatching the FFN and tail contexts.

A first two-input ObjectFIFO graph compiled through Peano but failed AIE
resource allocation because each memory tile already spends its two output DMA
channels on the Y input and completed output streams. The accepted graph leaves
that pair unchanged. Each shim uses its spare outbound channel to send eight
canonical X rows directly to the first core in a four-core group. That source
core keeps its two-row 3 KiB slice and forwards the remaining three chunks over
a core-stream chain; each following core retains one chunk and relays the rest.
The source core's Y input, completed output, direct X FIFO, local X slice,
parameters, and stack remain just below the 32 KiB data-memory budget.

The artifact is generated under `~/.hipfire/npu/` as
`embgemma_aie2p_post_ffn_direct_tail_bf16x2_split_x_completed_bf16x2_m256_k768`.
Locked layer-0 attribution is identical to R45, including `0.99998346`
completed-layer cosine. Across all 24 layers, OQ8++ versus BF16 remains exactly
`0.99808222` cosine with `0.00816466` maximum absolute error. The no-readback
M256 sample measures `362.6` input tokens/s at `21.08 W`, or `17.2` package
tokens/J.

R46 makes attention X, pre-FFN normalization, FFN, and the completed residual
tail an NPU-to-NPU shared-buffer path. It is still not the requested fully
resident encoder: the next layer's normalized projection input is produced on
the GPU, and final normalization and pooling remain outside AIE2P.

### R47 resident next-layer activation checkpoint

R47 consumes R46's compensated completed-layer BF16x2 output directly on
AIE2P and writes the dynamic 6,240-byte prefix of every next-layer R34 QKV
activation block. The layer-resident paired QKV weight scales at byte 6,272
and above remain untouched. Layer 0 retains the canonical GPU producer; layers
1 through 23 skip both GPU input RMSNorm and `pack_opus_npu_activations` when
the preceding resident tail has prepared their shared input buffer.

Each compute tile owns eight rows. It first reduces the full K=768 compensated
row, then replays the source for the three 256-value groups, applying the next
layer's learned input norm, optional AWQ scale, seed-42/1042 signed FWHT, and
per-row int8 quantization. Three logical row chunks aggregate over core streams
into R34's exact 24-row physical prefix, which each packer emits five times for
the N-macro consumers. The final partial block retains zero padding.

The first graph used separate X and parameter broadcasts. AIE resource
allocation rejected it because those two memory-tile outbound DMA channels
left no route for packer output. The accepted schedule preloads the three
immutable parameter records over the X ObjectFIFO, then serializes completed
state replays on that same channel. This leaves one memory-tile route in each
direction and compiled across all eleven aggregation chains.

The generated artifact is
`~/.hipfire/npu/embgemma_aie2p_next_layer_prep_w8_bf16x2_m256_k768`. Its locked
standalone gate checks 983,040 physical int8 values across all five replicas:
five values differ from the CPU square-root oracle by one LSB because AIE
`invsqrt` lands on the opposite quantization boundary; maximum scale error is
`7e-9`. Ten measured dispatches average `5.0835 ms`.

A locked layer-1 same-input comparison after an R47 handoff reaches
`0.99998975` completed-layer cosine. Across all 24 completed resident layers,
OQ8++ versus BF16 reaches `0.99818254` cosine with `0.00840785` maximum absolute
error. The one-run M256 sample measures `289.5` input tokens/s at `20.04 W`, or
`14.5` package tokens/J. This is a correctness checkpoint, not a performance
admission: the standalone preprocessing context is intentionally unfused and
adds about 117 ms over 23 boundaries. R34 still receives its architectural
residual through the host-updated parameter records, and final normalization
and pooling remain outside AIE2P. The next cross-layer slice must make R34 read
that residual directly from the shared completed state, then fold this prep
work into a producer or consumer schedule rather than repeating a standalone
context.

### R48 external architectural-residual checkpoint

R48 removes the host-updated residual records from R34. The paired-QKV,
attention, output-projection, and residual/norm graph now reads 32 padded
16 KiB residual records appended to its existing activation argument. It reuses
the even-column weight multicast after output weights and stages the immutable
post-attention norm in dead output-accumulator storage, avoiding a sixth DPU
argument and a third shim stream. The generated artifact is
`~/.hipfire/npu/embgemma_aie2p_resident_w8_qkv_paired_attention_o_norm_external_x_bf16x2_m256_k768_n1280`.

A companion 32-core graph converts R46's token-major compensated BF16x2 output
into R48's `[wave, active-column, core-row]` record order. It reconstructs
`high + low` in float and uses AIE `conv_even` rounding for the final BF16 value,
matching the former host reconstruction plus GPU BF16 cast. Copying only the
high plane passed a structural record test but accumulated end-to-end quality
loss: plain OQ8 fell to `0.99560386` cosine versus BF16. The compensated oracle
caught 4,387 rounding-boundary differences in the old artifact; the rebuilt
graph passes both private-SHMEM and AMDGPU-dma-buf gates with zero mismatches
and zero nonzero padding. Five standalone dispatches average `0.2906 ms`.

The admitted runtime contract has two initialization details:

- layer 0 writes its host-created compensated state to a prep-owned SHMEM
  bootstrap input; layers 1 through 23 consume the NPU-produced shared dma-buf;
- the imported combined activation/residual output BO is zeroed and synchronized
  once after attachment. Omitting that first publication produced roughly
  179k incorrect BF16 payload values even though later mappings appeared valid.

Several alternatives were rejected with hardware evidence. A fused R47
residual route exceeded the memory tile's inbound DMA channels. A separate
third shim stream was unavailable. PRIME-exporting an amdxdna-owned SHMEM BO
returned `EINVAL` on the installed driver. Importing the combined GTT BO at one
full size across HIP/R47/R48 did not repair the uninitialized first-output
failure. A standalone AMDGPU-owned dma-buf gate proves producer, owner, and a
second amdxdna import all see the same 32 records before and after explicit
cache synchronization.

Locked same-input comparisons report:

| boundary | layer 0 | layer 1 |
|---|---:|---:|
| architectural X cosine | 0.99998837 | 0.99999209 |
| normalized H cosine | 0.99998086 | 0.99997304 |
| resident FFN cosine | 0.99996137 | 0.99996234 |
| completed layer cosine | 0.99998506 | 0.99998700 |

The canonical OQ8++ full 24-layer run matches the R47 checkpoint exactly at
`0.99818254` embedding cosine and `0.00840785` maximum absolute error versus
BF16. Its one-run M256 sample is `267.2` input tokens/s at `20.04 W`, or `13.3`
package tokens/J. This is a correctness checkpoint, not performance admission:
R47 remains a separate roughly 5 ms cross-layer preprocessing context, the
outer runtime still materializes completed state, and final normalization,
pooling, and Dense heads are not yet resident. The 10k/15k targets therefore
remain open.

### R49 resident final-norm/mean-pool checkpoint

R49 removes two more host/GPU boundaries. Intermediate resident layers no
longer decode the 256x768 compensated output on the CPU and upload it to the
GPU when R47 and R48 have already prepared both the activation and residual for
the next admitted layer. Terminal, debug-comparison, unavailable-next-layer,
and partial-preparation paths still materialize the canonical output, so the
optimization does not silently hand stale GPU state to a fallback.

The final admitted layer now feeds
`embgemma_aie2p_final_norm_mean_bf16x2_m256_k768` through the same shared
completed-state dma-buf. A single AIE2P core streams all 256 compensated rows,
applies the baked F32 `model.norm` vector and RMS epsilon, and retains the
768-wide mean accumulator locally. Only the final 3 KiB pooled F32 vector is
read by the host. Five fresh-process hardware checks are exact against the CPU
oracle (`1.00000000` cosine, about `3e-8` maximum absolute error); three timed
commands per process measured `2.04-2.19 ms`. The context requires one priming
command because an unprimed fresh hardware context intermittently returned an
all-zero first output.

The full OQ8++ M256 path preserves BF16 quality at `0.99818283` cosine and
`0.00840781` maximum absolute error. A five-iteration run measured `914.595 ms`,
`279.9` input tokens/s, `17.05 W`, and `16.4` package tokens/J. This is a real
resident final-norm/pooling boundary but still not performance admission: the
roughly 120 serialized attention, FFN, tail, and preparation context commands
dominate the two-millisecond final stage. The FlatAttention-style next lever is
therefore command fusion or asynchronous overlap, not further optimization of
the final mean reduction.

Format admission also remains incomplete. Native OQ8 reaches the completed
resident layer and reports `0.99805230` cosine versus BF16. The current native
OQ4 artifact still selects projection-only execution
(`completed_resident_layer=false`) and reports `0.92913818`; the resident FFN
mode gate must be made format-generic before OQ4, mixed OQ, and their `+`/`++`
variants can claim this same full-layer path. Dense heads and final L2
normalization also remain host-resident.

### R50 format-neutral completed-layer admission

R50 removes the stale source-format restriction identified by R49. The
completed FFN already calls `group_dense_i8()` for every gate, up, and down
group, so native W4, compact mixed Opus, and W8 all produce the same dense
signed-byte plus scale execution payload. Its validator nevertheless required
the source label to be `DenseW8`, and the completed-layer selector repeated
that check. A unit test now admits a real native-W4 payload to the dense
execution contract; geometry and group-count validation remain unchanged.

Locked M256 hardware checks now report `completed_resident_layer=true` for:

| artifact | source family | BF16 cosine | hybrid input tok/s |
|---|---|---:|---:|
| `EmbeddingGemma-300M--npu.oq4.hfq` | native OQ4 | 0.92998326 | 265.0 |
| `EmbeddingGemma-300M--npu.oq4.125.hfq` | compact mixed | 0.92795205 | 288.6 |
| `EmbeddingGemma-300M--npu.oq6.5.hfq` | compact mixed | 0.95893502 | 270.0 |
| `EmbeddingGemma-300M--npu.oq8+.hfq` | calibrated OQ8+ | 0.99834824 | 268.4 |

The established OQ8++ result remains the `++` proof at `0.99818283`. Existing
non-`npu` OQ4+/OQ4++/OQ4.25++ artifacts keep their down projections in a
non-Opus storage type and correctly remain projection-only; they are not valid
evidence against the format-neutral kernel contract. Canonical all-projection
OQ4+ and OQ4++ artifacts still need to be produced and gated. The low W4/mixed
cosines are quantization-quality failures, not resident-dispatch failures, and
must not be promoted despite their now-complete NPU execution path.

The canonical suffix artifacts were subsequently generated from BF16 with the
unified `EmbeddingGemma-300M.calib.hfq` package. Both keep the Dense heads F16
while quantizing all 168 backbone projections:

- `EmbeddingGemma-300M--npu.oq4+.hfq` uses AWQ/clip-search calibration and
  reaches the completed resident layer at `0.97610980` BF16 cosine;
- `EmbeddingGemma-300M--npu.oq4++.hfq` reports 168/168 successful LDLQ packs,
  reaches the completed resident layer, and measures `0.97711462` BF16 cosine.

These close the `+`/`++` execution matrix but still fail the quality target.

### R51 resident Dense-head and L2 checkpoint

R51 completes the post-encoder AIE2P graph. R49's 768-element pooled F32 output
is written into the prefix of a shared input/weight dma-buf. A single-core
consumer streams BF16 rows for the two identity Dense heads
`768 -> 3072 -> 768`, retains the 3072-element intermediate in local tile
memory, and applies final L2 normalization before returning the 768-element
embedding. AWQ Dense sidecars, when present, are folded into the uploaded
effective weights, preserving the host equation without an activation-side
buffer. The forward contract now distinguishes a backend-produced final
embedding from a pooled hidden vector, preventing the CPU heads from being
applied a second time.

The standalone hardware oracle reports `1.00000000` cosine and about `1e-8`
maximum absolute error with an `18.1184 ms` five-command average. Full M256
checks report:

| artifact | BF16 cosine | max abs | hybrid ms | input tok/s |
|---|---:|---:|---:|---:|
| OQ8++ | 0.99818963 | 0.00833587 | 959.318 | 266.9 |
| OQ4++ | 0.97710949 | 0.02409978 | 1000.163 | 256.0 |

This is the first checkpoint where encoder layers, final RMSNorm, mean pooling,
both Dense heads, and final L2 normalization all execute on AIE2P. It is still
not a serving or performance admission. The single-core Dense graph is a
correctness schedule, and the full model remains dominated by serialized
hardware contexts. More importantly, opening an all-projection Opus artifact
without `HIPFIRE_EMBED_REFERENCE_MODEL` still invokes the legacy GPU Gemma3
weight loader first; OQ8 fails its `K % 256 == 0` assertion on the K=1152 down
projection before the resident projector can take ownership. A lean NPU-native
weight-loading seam is therefore required before this path is independently
servable.

### R52 resident-only weight-loading checkpoint

R52 removes that legacy loader dependency for the explicit resident path. A
Gemma3 normalization scaffold loads the six norm vectors per layer and retains
logical projection shapes, but allocates only one-element dummy projection
buffers. `EmbeddingGemmaWeights::load_resident_npu` keeps token embeddings and
Dense tensors host-visible, decodes Q8F16 embedding rows directly from their
34-byte blocks without expanding the 262k-row table, and marks the result
resident-only. The forward loop fails closed before any GPU fallback can touch
a dummy projection, including non-M256 inputs or an unavailable resident layer.

The end-to-end probe selects this loader whenever the completed resident path
is explicitly requested without a reference model. It omits GPU parity metrics
in that mode rather than comparing against itself. Locked self-contained M256
smokes now open and run directly from:

| artifact | hybrid ms | input tok/s | package tok/J |
|---|---:|---:|---:|
| OQ8++ | 1008.062 | 254.0 | 12.7 |
| OQ4++ | 997.849 | 256.6 | 13.5 |

This closes the independent-loading correctness blocker for the experimental
M256 path. It does not widen support to other sequence lengths: resident-only
weights intentionally reject those until the NPU layer graphs become
shape-generic. It also does not change the performance conclusion; serialized
context submission remains roughly two orders of magnitude from the 10k target.

### R53 resident phase decomposition

`HIPFIRE_EMBED_TRACE_RESIDENT=1` now records setup, attention, FFN, post-FFN
tail, cross-layer preparation/materialization, and total wall time for every
completed resident layer. A locked self-contained OQ8++ M256 run measured
`932.187 ms`, `274.6` input tokens/s, and `13.1` package tokens/J. Averaged over
all 24 layers:

| phase | mean ms/layer |
|---|---:|
| setup | 0.522 |
| attention + output/residual/pre-FFN norm | 9.050 |
| resident FFN | 13.603 |
| post-FFN norm/residual tail | 3.563 |
| next activation + residual preparation/output | 9.847 |
| total | 36.592 |

This rejects a dispatch-overhead-only explanation. The tail and next-layer
preparation consume `13.410 ms/layer`, repeatedly materializing and rereading
the same compensated token rows, while attention plus FFN consume another
`22.653 ms/layer`. The first FlatAttention-inspired fusion boundary is therefore
FFN down accumulation -> post-FFN norm/residual -> next-layer AWQ/FWHT/quant
packing and residual-record emission on each row's owning core. It should remove
two commands and the BF16x2 round trips per layer. Even eliminating that entire
13.4 ms would leave roughly 22.7 ms/layer, so attention/FFN tile utilization and
eventual whole-layer fusion remain necessary for the 25.6 ms model target.

### R54 rejected local-reuse schedules

Two attempts tested whether R47's four completed-state DMA passes could be
collapsed into one before undertaking a larger fused graph:

1. Retaining all eight compensated BF16x2 rows in each core required `24,576`
   bytes per tile. Together with parameters, scratch, routing chunks, FIFOs, and
   the assembly block, this exceeded AIE tile-local allocation. `aiecc` emitted
   `Failed to allocate buffer` warnings even though the cache wrapper returned
   success, so that artifact was rejected without execution.
2. Keeping each object-FIFO row acquired while packing all three groups reduced
   explicit storage to three `2,080`-byte chunks and compiled without allocation
   warnings. Hardware parity nevertheless produced almost entirely zero output
   (`974,234` mismatches, maximum quantized delta `127`). Holding the FIFO element
   across the three compute calls violates the graph's working producer/consumer
   schedule; it is not a safe substitute for distributed scratchpad ownership.

The admitted four-pass R47 graph was rebuilt after both experiments and passed
the hardware gate with five one-step quantization ties, maximum scale error
`7e-9`, and `5.0422 ms` mean dispatch. The next fusion should therefore follow
FlatAttention's actual requirement: explicitly partition persistent state across
the aggregate tile-group memory and schedule DMA, vector transforms, and fabric
collectives asynchronously. It must not rely on oversized per-core duplication
or prolonged FIFO acquisition.

### R55 retained gate-activation checkpoint

R55 applies the first successful FlatAttention-style distributed-local reuse to
the resident FFN. R45 previously broadcasts each 24-by-256 BF16 token group and
repeats pre-FFN RMSNorm, AWQ scaling, signed FWHT, and row quantization for every
one of six gate/up output blocks. Gate and up are already required to share one
AWQ activation scale, so R55 retains each core's three 784-byte quantized row
fragments for the three K groups and reuses them across all six output blocks.
For each 96-token M block this reduces gate input broadcasts from 18 to 3 and
the corresponding vector preprocessing work by the same factor. A separate
manifest field, `gate-activation=reuse-quantized-local-fragments`, selects the
63-block weight-stream ABI; the admitted R45 generator remains byte-identical.

The implementation exposed both AIE2P storage limits explicitly. An initial
three-path unrolled ring image used 16,748 bytes of program text and was
rejected. A looped selector reduced that to 16,428 bytes, still rejected. The
admitted graph uses one contiguous 2,352-byte fragment bank, dynamic group
offsets in the existing insert/send helpers, and looped preprocessing. Its
largest image is 16,412 bytes and `aiecc` produces a valid xclbin. The generated
artifact lives under `~/.hipfire/npu` and is rebuilt by `r55_cache.sh`.

Locked M256 end-to-end checks preserve the established quality envelope:

| candidate | BF16 cosine | maximum absolute error |
|---|---:|---:|
| OQ8++ | 0.99818963 | 0.00833587 |
| OQ4++ | 0.97710949 | 0.02409978 |

One phase-attributed OQ8++ A/B measured the steady layers (excluding the first
context-prime layer) at `13.793 ms/layer` for R45 and `11.329 ms/layer` for
R55, a 17.9% latency reduction or 1.218x speedup in the resident FFN command.
Three independent self-contained OQ8++ processes measured:

| graph | latency range (ms) | median ms | throughput range (tok/s) | median tok/s | median package tok/J |
|---|---:|---:|---:|---:|---:|
| R45 | 1007.642-1020.227 | 1019.821 | 250.9-254.1 | 251.0 | 13.3 |
| R55 | 946.261-961.962 | 947.893 | 266.1-270.5 | 270.1 | 14.1 |

Thus tile-local activation reuse improves complete resident execution by about
7.1% in latency, 7.6% in throughput, and 6.0% in package tokens/J. It does not
approach the 10k admission threshold: attention, post-FFN tail, next-layer
preparation, and the gate-to-down host-visible intermediate remain separate
bottlenecks. The next FFN step is to retain or transpose one M block of the
GeGLU tensor across aggregate tile memory so the down projection does not drain
and reread the full 288-by-1280 BF16 scratch plane through the shims.

### Quality-admission gate derivation — STS-Benchmark Spearman (2026-07-12)

Every prior checkpoint that called a candidate "below the real-model quality
gate" or a "quality-quality failure" was measuring against a gate that did not
exist. Code inspection confirmed it: the "selection-stability metric against
BF16" was the `quality_compare.rs` example, which computes rich statistics
(top-1 agreement, top-k overlap, regret, Spearman/Pearson) but only prints JSON
with no threshold; nothing consumed it. The hipfire-eval admission engine had no
embedding metric in its vocabulary and could not gate an embedding candidate at
all. The low OQ4 cosines quoted throughout this document (for example the padded
`npu.oq4++` "0.97711") were captured stdout of a CROSS-MODEL raw-vector cosine
field, eyeballed by a human. That is the wrong admission metric for an embedding
model: embeddings live on a hypersphere where a rotated-but-equally-good space
deflates cross-model cosine without any retrieval-quality loss.

The correct, self-consistent metric is Spearman of the model's OWN pair-cosine
scores against human gold labels — how well the model tracks similarity
judgments when it both indexes and queries. Measured on STS-Benchmark dev (1500
pairs, 500 selection queries) with `quality_compare` on the GPU path:

| model | Spearman vs gold | Δ vs BF16 (95% CI) | mean cos vs BF16 |
|---|---:|---|---:|
| BF16 (ref) | 0.86147 | — | 1.0 |
| OQ8 | 0.86150 | +0.00003 (−.0002,+.0003) ns | 0.99979 |
| OQ8++ | 0.86139 | −0.00008 (−.0003,+.0001) ns | 0.99984 |
| OQ4 | 0.86190 | +0.00043 (−.0022,+.0029) ns | 0.95074 |
| OQ4+ | 0.86073 | −0.00075 (−.0034,+.0016) ns | 0.96436 |
| OQ4++ | 0.85764 | −0.00383 (−.0065,−.0013) sig | 0.96475 |

(`ns` = bootstrap CI crosses zero, i.e. statistically indistinguishable from
BF16.) Every OQ4 variant retains >=99.5% of BF16's downstream STS quality despite
cross-model cosine of only ~0.95-0.96, so the earlier "0.977 fails quality"
verdict was measuring the wrong thing by an order of magnitude. Two secondary
findings: OQ8/OQ8++ are indistinguishable from BF16; and among the OQ4 family,
plain OQ4 and OQ4+ tie BF16 while OQ4++ is the ONLY variant with a
statistically significant (but tiny, ~0.44% relative) drop — LDLQ error
feedback minimizes weight reconstruction, not downstream task error, and here it
slightly hurt. For this embedding workload, plain OQ4 or OQ4+ is the better
W4A8 operating point than OQ4++.

Combined with OQ4's ~half-the-bytes memory/bandwidth advantage on this NPU
path, OQ4 sits strictly ahead of OQ8 on the quality-per-resource frontier: there
is no quality-based reason to prefer OQ8 for this model. The GPU-path quant
formats (`oq4`, `oq4+`, `oq4++`, `oq8`, `oq8+`, `oq8++`) all clear a
task-relative gate; this is a property of the encodings, separate from the
still-open NPU throughput work.

This was made enforceable rather than eyeballed. A new `embedding_quality`
hipfire-eval battery (commit `b103908da`) shells out to `quality_compare`, emits
a candidate row and a reference row sharing one comparison key that both carry a
raw `spearman` metric, and lets the existing admission pipeline gate the delta.
The gate is a per-metric admission dead-band: `spearman` gets a 0.01 band (~1.0
STS point) and is registered higher-is-better, quality-trigger, and hard-reject;
`delta >= -0.01` admits, `delta < -0.01` rejects. Cross-model cosine and rank
agreement are recorded as evidence but deliberately NOT gated. Run with
`hipfire-eval --battery embedding_quality --model <cand>.hfq --reference
<bf16>.hfq`; the battery is opt-in (absent from every default tier) and skips
rather than fails when the binary, models, or STS corpus are missing.

This is a quality-admission result, not an NPU throughput result. It does not
change any 10k/15k performance conclusion. Open: retrieval-quality metrics on a
labeled corpus (nDCG/recall@k; no MTEB-style set is in the repo, so STS Spearman
is the only task signal today), and the padded `.npu.` artifacts still differ
from their unpadded GPU comparison artifacts and cannot load through the GPU
`quality_compare` path (K=1152 down projection), so their STS quality was not
measured here.

### OQ4++ anomaly root-caused to calibration domain (2026-07-12)

The only OQ4 variant that significantly dropped (OQ4++, the LDLQ recipe) was
traced to the calibration artifact's INPUT distribution, not to LDLQ itself.
The EmbeddingGemma calib forward was already embedding-correct (bidirectional,
mean-pooled, all q/k/v/o+gate/up/down taps, Dense-head capture), but the
samples it tapped were wrong twice over: the `document_prompt` prefix that
`embed_forward` prepends at inference was never applied, and the corpus was a
383-token slice of wikitext, not retrieval sentences. LDLQ error feedback needs
the full activation Hessian (confirmed full, not diagonal), so it is the recipe
most sensitive to a mismatched input covariance — exactly the observed ordering
(RTN and AWQ, which use no/coarse activation stats, tied BF16; only LDLQ dipped).

A controlled A/B confirmed it. A new embedding-focused calib was collected
(STS-Benchmark train sentences, the `title: none | text: ` document prompt
applied, 4087 tokens), and OQ4++ was requantized off it; a control requantized
off the old wikitext calib with the same current binary. Only the calib differs.
On STS-Benchmark dev (1500 pairs, Spearman vs gold):

| OQ4++ calibration | Spearman | Δ vs BF16 (95% CI) |
|---|---:|---|
| STS + document prompt (new) | 0.86111 | −0.00037 (−.0027,+.0020) *ns* |
| wikitext, no prompt (old) | 0.85764 | −0.00383 (−.0065,−.0013) **sig** |
| plain OQ4 (RTN, reference) | 0.86190 | +0.00043 *ns* |

BF16 reference Spearman = 0.86147. The significant −0.00383 drop shrank ~10x to
a non-significant −0.00037: embedding-focused calibration recovers OQ4++ to
BF16 parity. The control reproduced the original artifact's 0.85764 exactly,
isolating the calibration data as the sole cause. OQ4++ is therefore the
byte-cheapest `++` W4A8 operating point at BF16-parity quality, and the repo's
whole OQ4 family (all calibrated on the 383-token wikitext package) should be
recalibrated embedding-focused.

Caveat: the fix bundled three sub-levers (retrieval domain, document prompt,
10x tokens); the A/B proves the bundle recovers OQ4++ but does not isolate which
sub-lever dominates.

Supporting changes:
- `collect_artifacts` now prepends the config `document_prompt` to each arch-19
  calibration sample (opt out with `--no-calib-prompt`) and records it in the
  calib provenance (`calib_document_prompt`).
- `hipfire-quantize` gained an opt-in `HIPFIRE_OQ_RAGGED_Q8`: ragged-K Opus
  tensors (e.g. EmbeddingGemma down_proj K=1152) fall back to Q8 instead of
  zero-padding to a 256 group, keeping the artifact loadable on the GPU serving
  path. The default stays padded-OQ4 for the NPU-native loader.

Provenance gap: the quantized model `.hfq` did NOT record the calib source or a
calib hash — only the transient quantize log did, so an artifact could not
self-report which calibration produced it. The old-calib-generated `+`/`++`
EmbeddingGemma artifacts were removed to avoid silently serving them.

Follow-up (2026-07-12): gap closed and family regenerated. `hipfire-quantize`
now stamps the consumed calib's signature into the output model metadata
(`"calibration": {source, xxh64, bytes}`, hashed over the full calib file), so
any artifact self-reports its calibration. The `HIPFIRE_OQ_RAGGED_Q8`
GPU-servable fallback was generalized from OQ4 to the whole Opus family
(OQ4/OQ8/OQ6/OQ3/tiered/compact), keeping ragged-K tensors at Q8 so calibrated
artifacts load on the GPU serving path. The STS+prompt calib was promoted to the
canonical `EmbeddingGemma-300M.calib.hfq` (the wikitext calibs were deleted), and
the whole GPU-servable `+`/`++` family was requantized off it and gated through
`embedding_quality` (STS-B dev, 1500 pairs). All admit within the 1.0-point band:

| artifact | Spearman | Δ vs BF16 |
|---|---:|---|
| OQ8++ | 0.86156 | +0.00009 |
| OQ8+ | 0.86153 | +0.00006 |
| OQ4.25+ | 0.86148 | +0.00000 |
| OQ4++ | 0.86111 | −0.00037 |
| OQ4.25++ | 0.85999 | −0.00148 |
| OQ4+ | 0.85891 | −0.00257 |
| OQ4 (RTN) | 0.86190 | +0.00043 |

BF16 = 0.86147. Every regenerated artifact is embedding-focused-calibrated,
GPU-servable (Q8 ragged down_proj), and carries its calib hash in metadata.

NPU padded family + resident-path validation (2026-07-13): the NPU-native padded
`+`/`++` family was regenerated off the same canonical embedding-focused calib.
Recipe: `--format <fmt> --hessian <canonical calib> --tensor-format
"dense.*=f16"` WITHOUT `HIPFIRE_OQ_RAGGED_Q8`, so the down projection pads to
OQ4G256/OQ8G256 for the NPU-native loader while Dense heads stay F16. All four
(`npu.oq4+`, `npu.oq4++`, `npu.oq8+`, `npu.oq8++`) carry the canonical calib hash
`744be90eeb9ac33a`.

`npu.oq4++` and `npu.oq8++` were validated end-to-end on the resident AIE2P path
at M256 via `embed_e2e_npu_opus`, in two modes (resident attn+FFN vs BF16, and
the fully-fused completed-resident-layer self-contained R52 loader):

| artifact | resident attn+FFN vs BF16 cosine | completed_resident_layer | resident tok/s (attn+FFN / fused) | pkg tok/J |
|---|---:|---|---:|---:|
| npu.oq4++ (W4A8) | 0.97526 | true (runs) | 626 / 308 | 25.0 / 16.2 |
| npu.oq8++ (W8A8) | 0.99958 | true (runs) | 645 / 297 | 26.8 / 14.8 |

Both load and execute resident bidirectional attention + resident W4/W8 FFN
across all 24 layers with no non-finite values, and the fully-fused
completed-resident-layer schedule engages cleanly. W8A8 is near-lossless at
0.99958; W4A8 sits at 0.975 (its padded down is OQ4 vs OQ8).

Caveats: (1) the example cannot compare the completed-layer path to BF16 in one
run — resident-only weights carry dummy projection buffers, so that mode
self-references and reports NaN cosine; the completed-layer quality is inferred
from the resident attn+FFN path (same kernels) matching BF16. (2) Raw embedding
cosine is the pessimistic metric; the authoritative quality verdict remains the
GPU `embedding_quality` Spearman gate, which the padded NPU artifacts cannot run
through (the GPU loader asserts K % 256 == 0). (3) Throughput (~300-650 tok/s)
stays far below the 10k target, consistent with the R53 per-layer
context-serialization finding; this was a correctness/coherence validation of the
regenerated artifacts, not a throughput result.

### R57 bandwidth-first kernel-development plan (2026-07-12)

Further NPU kernel work follows the dedicated
[`2026-07-12-npu-bandwidth-first-kernel-development.md`](2026-07-12-npu-bandwidth-first-kernel-development.md)
plan. R1/R56 is the stage-zero oracle: one active receive stream reaches 14.4
GB/s and eight columns reach about 56.5 GB/s, while the current resident
attention and FFN paths expose only about 0.9 GB/s of useful packed-weight
payload. New kernels must therefore grow from the measured DMA path one
observable stage at a time rather than landing as complete GEMMs.

The required sequence is external feed, production transfer geometry,
memory-tile ping-pong, memory-tile broadcast, preconverted-layout admission,
representation-local nibble decode, compute stage 1, compute stage 2/reduction,
scale/epilogue, output drain, and finally a persistent multi-stage boundary.
Every accumulated mode remains runnable and emits the same wire-byte,
useful-byte, trace-cycle, compute, and correctness schema.

Immutable tensor-block reordering is explicitly outside the kernels. It is
performed once by the offline producer or loader, and the derived layout is
identified by the literal `.rdna2.hfp` filename suffix/tag plus a version and
source-content hash. The preconverted-layout step only validates and consumes
blocks in that stream order; it must not transpose or reorder macro tiles on a
compute tile, in a memory-tile program, or in the dispatch loop.

Representation-local decode remains in-kernel. In particular, packed Opus
W4/OQ4 weights still require low/high-nibble separation, signed extension, and
lane swizzling before MMUL. That operation gets its own measured stage and may
not change global tensor-block order. Multidimensional DMA remains valid for
placing mutable activations, but it is no longer the immutable tensor-block
reordering mechanism described by the R31 experiment above.

### R58 packed-W4 admission and nibble-decode result (2026-07-12)

The dense R57 checkpoint has been superseded for projection weights by the
version-2 `hipfire_xdna::opus_hfp` ABI. A real combined layer-0 QKV derivative,
`EmbeddingGemma-300M.npu.oq4.layer-0.qkv.oq4.whole-scaled.rdna2.hfp`, retains
packed W4 through its 2,359,296-byte physical payload. The loader validates
source and payload SHA-256 plus the complete production geometry and reuses the
file without rewriting it. Dense R34 completed-attention caches now use the
unambiguous `.attention-dense.rdna2.hfp` role tag so the two encodings cannot
collide.

On locked `halo` AIE2P trials, the exact same eight-column/four-row QKV stream
measured 56.173 GB/s median in feed-only mode and 56.061 GB/s with every nibble
sign-extended and lane-decoded in each consumer. Retention is 99.80%; hardware
lane-sum and byte-exact vector oracles pass. `COMPUTE_STAGE1` then adds one real
native int8-by-int4 MMUL per 128 packed bytes and passes an exact int32 oracle at
55.933 GB/s median and 2.685 TOPS. `COMPUTE_STAGE2` expands to six activation
row blocks and full K=256 continuation per core. It passes exact int32 parity,
reaches 11.497 TOPS, and reduces wire delivery to 39.919 GB/s with 57.45%
receive stalls. That is the desired, counter-supported transition from
feed-bound to compute-backpressured operation rather than unexplained bandwidth
loss. It does not yet prove a complete scaled projection, end-to-end tok/s, or
the 10k target; scale accumulation, distinct output placement, and resident
model integration remain open.

Follow-up: complete scaled-projection correctness now passes through the
production vector-store kernel. `npu_opus_hfp_verify` loaded the real layer-0
Q/K/V roles through the real QKV HFP and matched all 327,680 outputs at
`max_abs=2e-7`. Three wrapper runs averaged 0.8635 ms (0.5829 logical TOPS), but
that includes CPU activation preparation, dispatch/synchronization, and output
deblocking and must not be presented as a kernel ceiling or end-to-end tok/s.
A checksum-only epilogue experiment was rejected because horizontal float
reductions corrupted later virtual-int4 MMUL results; integer sentinels caught
it before admission. The remaining work is to preserve the proven full-output
contract inside the resident layer graph without host pack/unpack boundaries.

### Generic HFP format checkpoint (2026-07-13)

The offline-layout ABI is no longer pure-W4/W8-only. `OpusHfpLayout::FullKV1`
now persists the existing one-dispatch full-K slab order for W4, W8, and compact
mixed `qt=36`. Mixed HFP payloads keep the W4 base nibble-packed and place the
dense W8 overlay in the exact AIE schedule entry; no runtime model load repeats
global slab ordering. Descriptor flags distinguish direct/scaled and combined
input layouts, while source and payload SHA-256 fail closed on stale/corrupt
files.

Twenty-six locked real-model AIE2P cases passed with zero mismatches. Coverage
includes OQ4/OQ8 plain, `+`, and `++`; OQ4.125, OQ4.25 plain/`+`/`++`, OQ4.5,
and OQ6.5; overlay counts 1/3/7/39; and N=256/768/1152/1280/2304 including
combined QKV and gate/up. Maximum absolute error was 2e-7 for whole-scaled
W4/W8 and zero for the direct mixed full-K path. Reopening OQ6.5 preserved the
HFP mtime, byte size, and SHA. Raw rows are in
`benchmarks/npu_gemm_tuning/results/r58-opus-hfp-format-matrix-20260713.csv`.

This closes generic projection-format/HFP correctness, not resident-model
integration or performance. The completed R34/R35 layer kernels still consume
separate dense derivatives and achieve only about 270 input tok/s end to end.
Their next ABI must load QKV/O and gate-up/down HFP payloads into BOs owned by
the destination hardware context, move norm/FWHT/AWQ metadata into small
parameter BOs, and retain the existing shared activation/hidden/FFN chain. A
standalone projection wrapper or debug-build timing is not evidence for the
10k/15k targets.

### R59 resident HFP argument checkpoint (2026-07-13)

R59 establishes the destination-context ABI before porting R34/R35 compute.
The DPU command regmap accepts at most five data arguments. A combined test with
four weight BOs, one parameter BO, and one output BO exceeded that limit and
made XRT reference a nonexistent BO. Counting the completed graph's shared I/O
shows that separate role BOs would also exceed the limit in production.

The selected ABI is one offline `ResidentContextBundleV1` HFP per destination
context: QKV+O+attention parameters for R34, and gate/up+down+FFN parameters for
R35. The loader places immutable role segments column-major without changing
block order within a role and pads one parameter tile. Source/payload SHA-256,
segment sizes, geometry, and parameter length are validated. The generic Rust
entry point is `NpuOpusExecutor::prepack_resident_context_bundle_cached`.

Three locked trials per mode produced:

| R59 ABI | median wire GB/s | range | packed-data GB/s |
|---|---:|---:|---:|
| R34 separate-BO control | 56.121 | 56.012-56.227 | 42.042 |
| R35 separate-BO control | 56.053 | 55.941-56.114 | 42.009 |
| R34 bundled weight/parameter BO | 55.884 | 55.789-55.898 | 40.416 |
| R35 bundled weight/parameter BO | 56.018 | 55.922-56.062 | 41.037 |

All role and parameter guards pass, with zero traced receive stalls. Both bundles
retain at least 99.48% of R58's 56.173-GB/s feed median, so the required production
argument reduction does not reintroduce the resident path's old ~0.9-GB/s
bottleneck. This remains a feed/ABI result, not resident projection or tok/s.
R60 must port R58 decode/MMUL/output into R34 first while preserving its current
shared input/hidden/query/key-value boundary.

### R60 first resident scaled-output tile (2026-07-13)

R60 now proves that boundary directly. `NpuEmbeddingLayerAttentionDenseW8`
exposes a CPU activation-packing oracle that delegates to the existing R30
layout and verifies the full 2,949,120-byte R34 input byte-for-byte. The AIE2P
graph accepts that shared input plus the R59 QKV/O/parameter bundle, keeping the
command below the five-data-argument ceiling and performing no global block
reorder in-core.

Three locked trials per stage:

| R60 stage | median bundle GB/s | range | R59 retention | result |
|---|---:|---:|---:|---|
| first MMUL | 53.838 | 53.829-54.112 | 96.34% | exact int32 |
| K=256 group reduction | 54.252 | 53.870-54.635 | 97.08% | exact int32 |
| three groups + scale tails | 50.912 | 50.605-51.092 | 91.10% | scaled f32 pass |

The last stage performs the first real 4x16 QKV output tile: 48 W4A8 MMULs,
three group reductions, activation scales from the existing shared blocks, and
weight scales from the HFP tails. It clears the 85% bandwidth gate while
showing about 9.9% receive stalls. The next ratchet is complete row/N-tile
coverage and production output placement, followed by attention; this result
is not yet a complete projection, layer, encoder, tok/s, or tokens/J claim.

### R61 complete QKV result: correct but not admitted (2026-07-13)

R61 expands that tile to the entire M256/K768/N1280 projection and drains
directly to padded row-major output. All 327,680 real values plus padding pass
with `max_abs=3.8147e-6`. The raw R16-style graph respects the three-argument
input/bundle/output ABI and never reorders immutable weight blocks in-core.

The final three-trial median is 4.114 ms (3.900-4.476), or 0.122 useful TOPS.
This is roughly 4.9x slower than the already-proven 0.8635-ms whole-scaled
wrapper control. Vectorizing R34's local 8+8 activation join improved the
scalar 6.870-ms median to 4.238 ms, while caching the complete transformed
6-KiB activation view in tile SRAM did not improve it further. R61 is therefore
correctness/output-placement evidence only and is rejected for model
integration.

R62/R63 complete the required controls. Producer-native W4 activation only
changes the cold raw-runtime median to 3.850 ms; physical output and canonical
dynamic group looping measure 3.856 ms. R63 then uses byte-identical R15 MLIR
and the compact 2,359,296-byte QKV `.rdna2.hfp`, yet its cold Python result is
3.964 ms. These results falsify the claim that the R34 activation join was the
dominant cost.

The discrepancy is measurement-system overhead. Through the production
Rust/C++ executor, current R63 measures 1.0292 ms median across three fresh
processes, versus 1.0288 ms for the pre-`3db7a1497` spill kernel and 1.0596 ms
for the historical cache. Every run passes the real 327,680-output oracle at
`max_abs=2e-7`. The no-spill kernel is not a material regression.

The next required stage is resident integration: attach the current compact
QKV HFP/BO and whole-scaled executor to the shared activation/output chain.
The offline `.rdna2.hfp` writer owns immutable tensor-block ordering; the
kernel retains only local nibble/lane swizzles. Admission must use warmed
production-wrapper and end-to-end resident timings, never cold Python
`XRTHostRuntime.npu_time` as a substitute.

R64 now separates the exact production graph's device window from that wrapper
measurement. Twelve locked shim-DMA traces across columns 0-3 pass every real
output and measure a 241.248-us median input-to-output span, 19.559 GB/s of
effective aggregate traffic, 198.240 us of output-DMA starvation, 2.817 padded
TOPS, and 2.086 useful TOPS. Core-tile trace packet routing is a compiler
scalability dead end on the fully occupied graph, while shim-local tracing
builds quickly. Columns 4-7 lack a terminal S2MM trace event and are excluded
from the timing result rather than estimated.

Only about 23.4% of the 1.0292-ms warm wrapper is inside the traced device
window. The next ratchet is therefore the mutable output boundary: emit the
verified BF16 attention handoff and keep it in shared BOs instead of returning
physical f32 QKV for host deblocking. Immutable role/block ordering remains a
one-time loader or on-disk `.rdna2.hfp` operation; the kernel may still perform
the local nibble and lane swizzles required by OQ4.

R65 closes the first half of that boundary. It retains the compact W4 HFP and
unchanged R15 compute, converts completed f32 accumulator tiles to BF16 locally,
and scatters them directly into the five-role R29 raw attention staging ABI.
Three locked fresh-process runs pass all 327,680 BF16 values bit-for-bit,
preserve every preseeded cos/sin/norm byte, and keep padding records zero.
Median warmed raw-runtime time is 0.487964 ms (0.485649-0.490328), with a
0.551481-ms median host call and 1.0315 useful TOPS. Largest core text is 9,280
bytes.

This is not yet the complete attention projection: R66 must consume the shared
staging BO with R29's headnorm/RoPE packers and prove exact Q/KV layouts before
attention execution is attached. The result establishes that returning and
host-deblocking physical f32 QKV is unnecessary. Immutable role/block
conversion remains offline in `.rdna2.hfp`; only mutable output placement and
local OQ4 nibble/lane work occur on the NPU.

R66 validates the second half functionally but rejects the first inline-record
schedule for speed. The exact R65 records produce canonical Q/K/V with the
established R28 metrics (Q cosine 0.99999121/max 0.0078125, K cosine
0.99999156/max 0.0078125, and bit-exact V). Three fresh 100-command processes
measure 0.9511-0.9984 ms, median 0.9915 ms. Sequential R65+R66 would therefore
cost about 1.48 ms before attention.

The regression is structural: broadcasting one physical record at a time
serializes four core-pair packers, while R28's joined compact input activates
them concurrently. R67 should change only the mutable staging/consumer order
to recover that concurrency. Neither the offline HFP order nor the pack math is
implicated, and the separate kernel-parameter workaround remains independent
of LDS use or avoidance.

This distinction is a design constraint for later rungs: do not carry forward
"no LDS" as a correctness requirement. The added kernel parameter is the
platform-issue workaround and must remain enabled. Local tile-memory use is a
separate capacity/performance variable that must be measured like DMA depth,
FIFO schedule, and swizzle cost; using LDS does not replace or weaken that
parameter workaround.

R67 restores the lost pack concurrency with a mutable joined layout. Each role
has 36 8-KiB records (32 M256 records plus four padding records), and R28's
split FIFO feeds all four core pairs from one joined DMA input. Projection
passes 327,680 BF16 values bit-for-bit and preserves all cos/sin/parameter
tails. Its median is 0.751200 ms across three fresh warmed processes, including
one 1.152343-ms outlier. The pack stage returns to a 0.3670-ms median while
preserving the established Q/K/V oracle. Sequential medians total about 1.1182
ms before attention.

This is an admitted layout/concurrency direction but not yet the resident
handoff: the projection issues roughly 360 small output tasks. R68 should
collapse three token-group writes into one padded 24-token write with safe
record overlap, then compare the complete projection-plus-pack boundary before
attention is attached.

R68 reduces the producer task count about threefold using overlapping padded
writes. A 37th 8-KiB record per role absorbs the terminal pad while the first
32 records remain the exact joined M256 consumer order. Projection stays
bit-exact and measures a 0.494605-ms median; pack stays at 0.3579 ms with the
established Q/K/V oracle. Sequential medians are about 0.8525 ms before
attention, 24% faster than R67 and 42% faster than R65+R66.

This is still a sum of isolated contexts. R69 must import one shared stage BO
into both contexts and measure the actual no-copy chain before the handoff is
attached to resident attention. No immutable conversion or tensor-block reorder
is permitted at that boundary.

R69 rejects that cross-context architecture. Independent imports of an amdgpu
GTT dma-buf are incoherent despite producer/consumer BO sync and dma-buf
barriers, while the installed amdxdna driver rejects PRIME export of native
XDNA SHMEM with `EINVAL`. Sharing one DRM file description and the original GEM
handle does produce exact output after a 0.028-0.029-ms BO sync in two fresh
processes, but a third fails by 384 Q bytes. More importantly, the two passing
100-command runs take 5.4478 and 5.6600 ms: projection and pack each inflate to
roughly 2.7-2.9 ms because two full-array hardware contexts are scheduled around
the boundary. R70 must therefore fuse projection and headnorm/RoPE packing into
one compiled graph/hardware context. The handoff must remain graph-local; it
must not add an online tensor-block conversion.

R70 completes that single-context seam. A literal R68+R67 merge first exceeded
the tile's two input DMA channels; an inline merge then exceeded the 16-KiB
program limit. The admitted build reuses the activation channel, specializes K
and V ownership across columns, and replaces duplicate W4 init/accumulate code
with one generic group function. Three fresh primed 100-command processes match
the isolated R65 projection stage and R66 Q/KV outputs byte-for-byte at a
1.3076-ms median (1.3006-1.3108 ms); maximum core text is 13,504 bytes.
Activation padding is loader-side and weights remain in offline `.rdna2.hfp`
order. R71 must now attach the existing packed-Q/KV attention phase inside this
same context. Do not route R70 output through a second full-array context and
do not interpret 256/1.3076 ms as end-to-end model tokens/s.

R71 attaches attention in that context and closes the correctness seam, but it
is not yet the resident integration candidate. The five-argument graph appends
the attention result to the stage BO, moves Q/V pack ownership to columns 0-3,
and uses columns 4-7 for full-width attention. Maximum core text is 15,888
bytes. The projection stage, Q, KV, and all 393,216 attention bytes match the
isolated R70-to-R27 reference exactly in three fresh primed runs.

Those runs measure 3.5951, 3.2617, and 3.3118 ms (median 3.3118 ms). The exact
redistributed pack-only control is 1.5446 ms median and isolated R27 attention
is 0.9141 ms, leaving about 1.77 ms in the fused attention/feed phase. A
two-column fallback is rejected at 5.8228 ms median, and adding a separate
attention output path is rejected because it exceeds memory-tile input DMA
channels. R72 must eliminate or locally stream the packed-Q/KV external round
trip while preserving R71's byte oracle. The kernel parameter remains the
correctness workaround; LDS use or avoidance remains an independent measured
optimization choice.

R72's first graph-local sub-rung proves the direct-Q dependency but rejects its
scalar-stream implementation. Columns 0-3 stream packed query pairs to columns
4-7, which reuse a 24-KiB projection accumulator as a six-group query cache.
The external Q BO is unused, and projection stage, external KV, and final
attention match the R70/R27 references byte-for-byte. The graph fits at 15,248
bytes maximum core text.

Three fresh primed 100-command runs measure 3.9288, 3.7749, and 3.9272 ms
(median 3.9272 ms), an 18.6% regression from R71. The scalar synchronization
and cache pressure outweigh removal of the 393,216-byte Q round trip. Do not
extend this implementation to K/V. Continue R72 with a burst/vector DMA or
existing-FIFO handoff that preserves the exact final-attention oracle. The
existing kernel parameter remains the correctness workaround; neither this
rejection nor the next design should infer an LDS/tile-memory avoidance rule.

R73 tests that distinction directly by using shared tile memory for a
graph-local Q handoff. One depth-one 24-KiB ObjectFIFO joins each adjacent
projection/attention core pair; the external Q BO is unused and all projection,
KV, and attention bytes remain exact. A producer-local cache exceeded the
16-KiB program limit, and the adjacent graph required a 2-KiB rather than
4-KiB producer stack to fit tile memory. Those are capacity constraints, not
correctness workarounds.

Three fresh primed 100-command runs measure 3.6449, 3.7165, and 3.7205 ms
(median 3.7165 ms). R73 recovers 5.4% from scalar R72 but remains 12.2% slower
than R71, so the depth-one six-group schedule is rejected. The next graph-local
handoff must overlap production and attention or reuse an existing DMA path;
it must not extend this serialization to K/V. The added kernel parameter remains
the workaround that stops the platform issue. LDS/tile-memory use remains a
separate measured optimization choice.

R74 tests the other external attention operand without adding a local handoff.
It keeps two query groups and four accumulator/stat sets live per attention
core, so one 262-KiB KV replay updates both groups and complete KV replays fall
from six to three. The first 4-KiB-stack graph exceeded the 64-KiB active-tile
allocation by 1,184 bytes; the already-hardware-tested 2-KiB stack makes it fit
at 64,672 bytes, leaving 864 bytes. Maximum linked core text is 15,248 bytes.

All projection stage, Q, KV, and attention bytes remain exact. Three fresh
primed 100-command runs measure 3.4496, 3.4242, and 3.2867 ms (median 3.4242
ms), 3.4% slower than R71. Reject paired-group residence as the next integration
topology: halving KV replays/tasks does not recover the stable median. R72-R74
together show that simply moving more attention state into tile memory is not
the missing win. The next rung must change phase scheduling or redistribute
core work while retaining R71's exact observable oracle. The added kernel
parameter remains the platform-issue workaround; no local-memory avoidance
rule follows from this result.

R75 changes no kernel math or local-memory allocation. It batches R71's
runtime tasks into two-group windows, reducing six per-group await/free barriers
to three. A six-group window exhausts static BD IDs at group 4; a four-group
window links but corrupts the observable Q boundary (392,405 of 393,216 bytes
wrong), so queue depth cannot be inferred from successful lowering alone.

The two-group window is byte-exact across projection stage, Q, KV, and final
attention. Three fresh primed 100-command runs measure 3.2580, 3.2775, and
3.3314 ms (median 3.2775 ms), 1.0% faster than R71. Admit R75 as the new
projection/pack/attention scheduling baseline. The result identifies a real,
if modest, command-stream bubble without adding resident state. Continue the
window sweep only under the same exact oracle, then carry the admitted schedule
into the resident-weight layer graph.

R76 tests the only remaining queue point, three groups per window. It remains
byte-exact across projection stage, Q, KV, and attention. Three fresh primed
100-command runs measure 3.4199, 3.2222, and 3.2604 ms (median 3.2604 ms), 0.52%
faster than R75 and 1.55% faster than R71. Three groups is therefore the maximum
correct and best measured task window for this graph: four corrupts Q and six
cannot allocate BDs. Admit R76 and stop widening this queue. The next work is
to preserve its two-window attention schedule while replacing the benchmark
weight argument with destination-context-owned `.rdna2.hfp` resident bundles.

R77 closes the first resident-weight seam in production Rust. The
`NpuEmbeddingQkvAttentionOpus` executor reads the real layer QKV
`.rdna2.hfp`, validates its v2 descriptor and payload SHA-256, allocates the
2,359,296-byte compact BO from the destination R76 hardware context, uploads
once, and reuses it across commands. The fused path no longer depends on the
benchmark's extracted `weights.bin`; immutable block order remains offline and
only local nibble/lane work remains in the kernel.

The layer-0 hardware oracle passes every projection stage, Q, KV, and attention
byte. Three fresh primed 100-command runs measure 3.2753, 3.3165, and 3.2137 ms
(median 3.2753 ms), 0.46% above raw R76 and 1.1% below R71. Admit this resident
QKV/attention weight seam. Do not call it a resident layer or model result:
output projection, residual/norm tail, FFN, next-layer handoff, end-to-end
tokens/s, and package tokens/J remain open. Next, join this executor's output
to the existing resident O/tail path without adding an online block reorder or
a sixth DPU argument.

R78 isolates the role map required for that join before adding O-projection
code. Even columns own the unchanged external Q/K/V packers; adjacent odd
columns run R76 attention and its three-group task window. Projection stage, Q,
KV, and attention remain byte-exact, and odd cores fit at 15,888 bytes. Three
fresh primed 100-command runs measure 3.8331, 3.7729, and 3.7959 ms (median
3.7959 ms), 16.4% slower than R76. Reject the remap as a standalone schedule.

The capacity result is also decisive: even cores still link compact-W4
projection plus pack routines, leaving insufficient program store for the R32
output projection and R34 norm tail. The next graph cannot append those
functions to R78. It must adopt R33's specialization principle: odd cores
project a pair of adjacent compact-W4 stripes and retain attention; even cores
drop QKV projection before receiving the direct attention pair and running O
projection/tails. The pair-major immutable block schedule must be generated
once by the loader and stored as `.rdna2.hfp`; only nibble/lane swizzle remains
local to the kernel.

R79 implements that offline prerequisite as a generic HFP layout rather than a
kernel conversion. `PairedWholeScaledV1` changes complete-block order from
`(column, block)` to `(adjacent-column-pair, block, lane)` while preserving
every encoded block byte-for-byte. Its cache identity includes the complete
source artifact, and the descriptor retains source encoding/geometry plus the
source payload size. A deterministic unit oracle verifies full block coverage,
unchanged bytes, exact pair order, descriptor fields, and cache reuse.

R79 is not hardware evidence. The next rung must feed this pair-major HFP to a
paired compact projection graph, remove QKV projection functions from even
cores, and match R65/R70 stage bytes before Q/K/V packing, attention, or O
projection are attached.

R80 consumes R79's pair-major HFP in the isolated projection graph. Each odd
core holds two accumulators, reuses one activation block across two intact
adjacent-column weight blocks, and scatters both stripes into the exact R65
inline stage. Even cores execute no QKV projection code. A six-task-per-shim
output schedule timed out; restoring R65's one-slice-per-channel cadence passes
all 327,680 BF16 outputs bit-for-bit with zero tail or padding corruption.

Three warm process medians are 0.818433, 0.833471, and 0.789289 ms (median
0.818433 ms), 67.7% slower than R65's eight-column projection. Admit R80 only
as the required capacity topology. Odd-core text is at most 11,872 bytes,
leaving 4,512 bytes, and even cores remain completely free. Next, attach the
existing external Q/K/V pack boundary to this exact stage while retaining
paired weights and measure program fit before attention or O projection.

R81 attaches even-core Q/K/V packing and preserves every stage, Q, and KV byte.
Three 100-command runs have a 1.8370-ms median; odd/even text is at most
14,032/10,912 bytes. It is a capacity checkpoint, not a speed win.

R82 appends R76 attention to the odd cores and is rejected before hardware
execution. Columns 1/3 reach 22,416 bytes, exactly 6,032 bytes beyond the 16 KiB
program store. Attention contributes 2,992 bytes of fully unrolled driver,
4,256 bytes of functions, and 1,136 bytes of helpers. This is program capacity,
not LDS/tile-memory capacity, and it does not change the separate
kernel-parameter correctness workaround.

R83 removes that overflow without changing math or offline layout. It reuses
R70's single projection-group ABI and uses non-LTO trip-count helpers so Peano
retains the 16-block attention and three-slice finish loops. Maximum odd/even
text is 15,888/10,912 bytes; stage, Q, KV, and attention are byte-exact. Three
100-command runs have a 4.0535-ms median, 6.8% slower than R78 and 24.3% slower
than R76. Admit R83 only as the first fitting paired
projection/pack/attention image.

R84 attaches direct O projection to the even cores without growing the odd
projection/attention program. The existing kernel parameter remains the
correctness workaround; allocator placement and tile-memory use remain
independent capacity/performance choices. A first hardware oracle exposed an
R32-to-R83 token-axis transpose, corrected solely in the output DMA scatter as
`active_column * 32 + core_row * 8`. There is no kernel-side tensor-block
reorder, and the paired QKV plus direct-O order remains loader-preconverted.

The corrected graph passes exact stage and KV checks and the fused
attention-to-O numerical oracle. Three 100-command runs measure 5.9362, 5.8716,
and 5.9856 ms (median 5.9362 ms), 46.4% slower than R83. Admit R84 as a
correctness/capacity rung and reject it as a speed baseline. Before adding
residual/norm, split the 1.8827-ms increment into O-weight feed,
attention-to-O FIFO, O compute, and output-DMA probes; retain R76/R77 as the
current speed baseline.

R84 attribution subsequently separates its 1.8827-ms R83 increment into
0.0416 ms paired attention handoff, 0.6209 ms complete O-weight
delivery/consumption, 1.0601 ms O MMUL plus F32 finish, and 0.1601 ms canonical
output DMA. The corresponding control medians are 4.0951, 4.7160, and 5.7761
ms, each synchronized with a 64 KiB completion signal. This rules out the
adjacent FIFO as the dominant cost and identifies O compute/finish first, then
the 16 KiB weight cadence, as the next levers.

R85 reuses each activation load across four O N tiles and passes the full
oracle at a 5.7945-ms median, 2.4% faster than R84. R86's two-accumulator
variant passes but regresses to 5.9203 ms. Admit R85 as the direct-O baseline,
reject R86, and preserve the loader-only layout conversion and independent
kernel-parameter workaround in the next tail-fusion rung.

R87 deepens only the shim-to-memory-tile O-weight FIFO to two objects and
measures 5.7450 ms median. R88 depth three reaches 5.7343 ms, only 0.0107 ms
better, so retain the simpler R87 depth-two feed. Both pass the full oracle;
neither changes compute-tile storage or the correctness parameter.

R89 establishes the post-attention tail's local-storage seam without an
external round trip. Each even core reuses 8 KiB of its final dead 10 KiB
activation FIFO object after packing and adds one 4 KiB buffer, staging one
8x768 O wave as three block-aligned 8x256 BF16 tiles. The DMA scatter alone applies R83's token mapping
`active_column * 32 + core_row * 8`; the kernel performs no tensor-block
reorder. Stage and KV are bit-exact, and O has 0.99999225 cosine with 0.0625
maximum error, one BF16 quantum at the worst magnitude. Three fresh
100-command runs have a 5.7202-ms median. Maximum even/odd text is
14,544/16,048 bytes. Admit this capacity seam, but do not call it residual/norm
execution or a full-layer/model throughput result.

R90 closes that local tail boundary. A first literal image expanded the
18-record projection-drain loop and reached 16,784 bytes. Replacing the Q-pack
sequence with a loop linked at 14,976 bytes but changed attention numerics, so
that rewrite is rejected. R90 instead makes only the existing projection-drain
bound noinline/runtime-stable. This retains all 12 Q-pack calls per even core in
their byte-verified order and fits at 15,952/16,048 bytes maximum even/odd text.
The build now treats missing artifacts and any core above 16 KiB as hard
failures, even when `aiecc` returns success after a CDO overflow.

The full hardware oracle keeps stage and KV bit-exact. All 196,608 normalized
BF16 values are finite and nonzero, with 0.99995399 global cosine, 0.99994058
minimum token-row cosine, and 0.09375 maximum error. The two AIE `invsqrt`
operations produce a bounded row-scale difference rather than a tensor-layout
or parameter-record error; admission requires both cosine measures at least
0.9998 and maximum error at most 0.1. Three fresh 100-command runs measure
6.6044, 6.2915, and 6.3890 ms (median 6.3890 ms), adding 0.6688 ms to R89.
Admit the complete post-attention norm/residual/pre-FFN-norm boundary for
correctness and proceed to direct FFN consumption; it is not a speed or
full-layer admission. The kernel parameter remains the platform workaround,
independent of tile-memory use.

R91 makes that FFN consumption physically zero-copy. A first offset-zero DMA
passed one command but corrupted sustained execution: the paired projection
regenerates only 327,680 stage bytes, while the following 65,536 bytes are
immutable headnorm/RoPE pack state. The normalized 393,216-byte tensor had
overwritten those tails. The accepted layout shifts the complete 2,457,600-byte
stage ABI forward by 393,216 bytes and reserves offset zero for canonical BF16
H. This is a loader/DMA offset change, not a kernel tensor-block reorder.

Sustained SHMEM and PRIME/GTT controls pass the R90 tail gate and exact KV
oracle. The same physical GTT pages then feed the existing canonical-BF16 R35
resident FFN without a host write. The synthetic dense-OQ8 FFN reaches
0.99989925 cosine and 0.0118408 maximum error. Three fresh 100-command runs have
6.3727 ms producer, 9.7654 ms isolated FFN, and 22.1772 ms alternating-chain
medians. The 6.0391-ms residual is context alternation, 27.2% of the chain. One
layer boundary reaches 11,543 M256 rows/s, but this is not end-to-end encoder
input-token throughput. R91 admits the zero-copy contract and rejects per-layer
two-context alternation as the final schedule. Test same-DRM peer contexts next;
if the tax remains, the graph must change phase partitioning rather than memory
layout.

R92 tests that same-DRM alternative. `NpuKernel::load_peer` places R91 and R35
on separate hardware contexts backed by one amdxdna DRM file and device heap;
the physical handoff remains the proven GTT dma-buf because PRIME-exporting an
XDNA-owned SHMEM BO returns `EINVAL` on this driver. All tail/KV/FFN oracles
pass. Three fresh 100-command runs have 6.4109 ms producer, 9.7080 ms isolated
FFN, and 22.1542 ms alternating-chain medians. The residual 6.0353 ms (27.2%)
is indistinguishable from R91's 6.0391 ms. Reject peer ownership as a sustained
win: the tax belongs to hardware-context alternation, not separate DRM files or
the GTT handoff. A one-iteration 19.8480-ms sample is not admission evidence.
Continue with the bandwidth-first native compact FFN phase and a different
phase partition; do not spend another rung changing shared-buffer ownership.

### R93 bandwidth-first canonical-BF16 to resident-W4 checkpoint

R93 implements the next boundary one operation at a time. It consumes R90/R91's
canonical M256xK768 BF16 pre-FFN-normalized state, applies only gate/up AWQ
scaling, the canonical seed-42/1042 signed FWHT-256, and row quantization, then
writes R25's exact 718,848-byte activation ABI. The DMA scatter produces four
row stripes, 27 blocks per stripe, and 6,656 bytes per block. Each 6,240-byte
dynamic prefix is placed directly in the three N-macro consumer positions;
the 416-byte tail remains zero. This is mutable activation preparation, not
immutable tensor-block conversion. Weight order remains an offline/loader
`.rdna2.hfp` contract and OQ4 nibble/lane swizzle remains local to compute.

The physical input BO pads the canonical 256 rows to 288. Each AIE DMA object
reads a two-row 3,072-byte window and consumes only the first row, allowing the
same FIFO to preload the proven 3,072-byte BF16-vector parameter record without
changing canonical row order. Across all three replicas, all 589,824 int8
values match the CPU oracle, maximum scale error is `7e-9`, every block tail
and padded-row guard remains zero, and core text is 7,856-9,040 bytes.

A scalar/int8-sign implementation is rejected. A route-only image writes all
chains and a load-only image reads valid source/parameter data, but the full
scalar image corrupts the scale/quantization loop. Reusing R47's noinline
BF16-vector sign/post-scale path restores exactness. This is compiler/codegen
evidence, not a tile-memory correctness constraint. The added kernel parameter
continues to be the platform workaround; LDS use is independent.

Three fresh 100-command processes measure 4.0618, 4.1218, and 4.1117 ms
(median 4.1117 ms). At only 0.263 GiB/s of physical source-plus-output traffic,
the standalone producer is transform/control limited rather than external
bandwidth limited. Admit the byte contract and reject the separate phase. R94
must fold preparation into the first native gate/up stage, overlap it with
resident weight DMA, and avoid external materialization of the three identical
N-macro activation copies. This remains a rung-level result, not complete-layer
or end-to-end token/s evidence.

### R94-R97 native-W4 preparation and fusion checkpoint

R94's BF16-vector preparation reduces the R93 median from 4.1117 to 2.1320 ms
while retaining the physical ABI (three one-code q differences, `7e-9` maximum
scale error, zero padding corruption). Its 0.507 GiB/s rate still rejects a
separate preparation context.

R95 unifies W4 init/accumulate and R96 compacts the repeated fragment-ring
driver. The complete R25 FFN oracle is unchanged while maximum core text falls
from 16,320 to 13,968 and then 12,944 bytes. These rungs buy fusion capacity;
they do not establish a sustained throughput win.

R97 spends that capacity to consume canonical BF16 inside the first gate/up
phase. Hardware probes prove all 216 source DMA objects, all weight objects,
and the full 256-row quantized activation boundary. Groups 1-2 are q-exact;
group 0 differs by one code in one value, and scale differences are bounded to
normal floating-point rounding. The initial full graph emitted NaNs because
inline gate preparation reused `own` and `transit` while those buffers still
held R25's spilled partial down accumulator. Two dedicated 784-byte gate
fragment buffers remove that state-lifetime alias. The full hardware oracle
then passes with gate cosine `1.00000000`, final cosine `0.99998228`, maximum
absolute error `0.2597733`, and mean absolute error `0.03750710`; maximum core
text is 15,456 bytes. A fresh dispatch measures 6.4095 ms, or 39,941 M256
rows/s. This admits correctness and fusion capacity, not sustained throughput:
R97 already inherits R15's `rounding=floor` and `saturation=none` numerical
controls, but a 20-command run still encounters the independent four-second
command timeout cadence. The added kernel parameter that stops the platform
issue remains active. Bounded context
recycling closes the immediate execution gate: three independent 100-command
runs recycling every seven commands preserve the full oracle at 6.4974, 6.4844,
and 6.3388 ms (median 6.4844 ms, 39,479 M256 rows/s). Admit sustained standalone
R97 with this timeout mitigation. The next step is to replace the reusable W4
runtime's externally packed activation ABI with this canonical-BF16 input and
then join it to the resident layer graph; this row rate is not yet full-layer or
end-to-end encoder throughput.

Terminology is binding here: the added kernel parameter is the platform-issue
workaround. It is not LDS avoidance and it is not the R15 rounding/saturation
configuration. LDS/tile-memory use or avoidance,
the command timeout, context recycling, and program-capacity refactors are
independent issues or performance/capacity decisions; none replaces those
workarounds. R97's dedicated fragment buffers fix its own kernel state
alias and are likewise not that workaround.

R98-R100 complete the next bandwidth-first output ladder. R98 converts the
final F32 values in place to compensated BF16 high/low pairs, retaining the
884,736-byte transfer and fitting at 16,032 bytes. Its 6.5832-ms sustained
median is 38,886 M256 rows/s. R99 changes only the output DMA destination stride
to place those bytes in the first 3,072 bytes of the existing 4,608-byte
post-FFN combined row; it sustains 6.6000 ms or 38,788 rows/s. R100 changes only
the split-X tail reader and passes at `0.99999861` cosine and `0.0039062`
maximum error. These are mutable scalar encoding and DMA placement changes, not
immutable tensor-block conversion. The next integration step is to select the
R99/R100 pair for native W4 layers in the reusable resident executor while
leaving arbitrary mixed and OQ8 layers on the generic dense-W8 executor.

That reusable native-W4 selection is now implemented and compile-checked. The
first layer oracle isolated an integration alias: the temporary host
pre-FFN-normalization bridge overwrote direct architectural X with normalized H
in the attention hidden BO, while R100's split-X tail still consumed that BO.
A separate 442,368-byte shared X buffer fixes the seam. Layer 0 now reaches
`0.99997024` FFN cosine, `0.99999873` tail cosine, and `0.99998514`
completed-layer cosine. All 24 resident-only OQ4 layers complete for M256.

The full run is only 291.6 input tok/s (878.003 ms), so it is not a throughput
admission. The temporary host readback/normalization bridge and per-layer
preparation/output account for roughly 10-12 ms per layer and must be removed.
The next bandwidth-first rung should make attention emit canonical
pre-FFN-normalized H while preserving X directly into the R100 split input,
then remeasure each phase before extending native compact execution beyond W4.
Mixed OQ and OQ8 remain on the generic dense-W8 executor meanwhile.

Do not conflate this integration fix with the platform workaround. The latter
is the separately added kernel parameter that stops the issue, not LDS
avoidance and not R15's `rounding=floor`/`saturation=none` numerical controls.
X preservation, R97's dedicated fragment buffers, context recycling, and
tile-memory placement are separate concerns.

R101-R103 and the first R104 form test the initial direct-X bridge-removal
designs and reject them before runtime selection. The literal per-row inverse
relay overflows at 16,444 bytes.
A shim-DMA scatter fits at 16,380 bytes but misaddresses inverse state and
reduces layer-0 cosine to 0.50248530. Reusing the even normalized-X output FIFO
fits at 16,268 bytes but triggers the independent four-second timeout. The R102
consumer is allocation-clean at 16,064 bytes with weight FIFO depth one, but
has no correct producer; the first R104 in-FFN RMS prepass overflows at 18,352
bytes (`-Oz`: 20,032 bytes). A separate metadata FIFO also exceeds DMA-channel
count.

The next rung is therefore smaller: keep R44's known-good direct-X producer and
R99/R100's admitted W4/tail seam, preserve X, and move only X-times-inverse
activation preparation off host. Fold immutable pre-FFN norm into loader-side
W4 scaling. First admit that boundary alone against exact per-row inverse and
FFN oracles, then integrate it only if it avoids the four-second timeout and
materially reduces the current two 5.1-MiB host synchronizations. Keep both the
R15 numerical controls and the distinct platform-workaround kernel parameter.

R105/R106 completes this smaller boundary as a correctness experiment but
rejects it as the resident topology. R105 alone passes at cosine `0.99999122`
and a 0.1295-ms median; loader-side folding lets R106 preserve the intended
learned pre-FFN norm without another immutable input. Integrated layer 0 reaches
unit-RMS cosine `0.99999269`, FFN cosine `0.99990930`, tail cosine
`0.99999862`, and completed-layer cosine `0.99996179`. The full 24-layer run is
slower than the R99/R100 host bridge: 901.432 ms / 284.0 input tok/s versus
878.003 ms / 291.6 tok/s. Cross-context cache maintenance inflates the nominal
R105 phase to 2.35-4.14 ms per layer and preparation/output remains roughly
9-12 ms. Keep the pair behind `HIPFIRE_EMBED_UNIT_RMS_BRIDGE=1` for diagnostics
and compact RMS into the resident W4 program next. This rejection is unrelated
to LDS; the existing kernel parameter remains the separate platform-issue
workaround.

### R104 single-context RMS is admitted

R104 now fits the resident W4 context under the normal build. Vectorizing the
mean multiply removes the scalar runtime helper; inverse completion is fused
into the scan; four cloned FWHT specializations share one runtime-stride body;
and each core retains one complete `3 x 768` BF16 X object. All 32 core text
sections are exactly 16,384 bytes. The input DMA contract is one padded
442,368-byte canonical-X pass, not a prepass followed by nine tensor replays.

Standalone hardware reaches gate cosine `1.00000000`, final cosine
`0.99996707`, maximum absolute error `0.0737100`, and mean absolute error
`0.01499078`; a 100-command recycle-every-seven run takes 6.5401 ms. Default
layer-0 execution reaches FFN cosine `0.99991494`, tail cosine `0.99999844`, and
completed-layer cosine `0.99996658`. Alternating full-model trials measure R99
at 892.708/909.986 ms and R104 at 859.599/869.015 ms, a 4.1% paired-mean latency
reduction. A separate current default run is 894.222 ms, 286.3 input tok/s,
18.07 W, and 15.8 tok/J.

Select R104 by default for native W4 when its artifact exists; keep R105/R106
behind `HIPFIRE_EMBED_UNIT_RMS_BRIDGE=1`. Reject the smaller post-link Peano
variants because they time out with all-zero hardware output. The next
bandwidth-first rung is no longer pre-FFN normalization: it is the separate
next-layer and residual preparation contexts, which still consume about 9-12
ms per layer. Preserve the existing kernel-parameter workaround independently
of LDS/tile-memory placement.

### R108/R109 remove the residual-copy context

R107's literal R47+R48 graph exceeds memory-tile DMA channels. A separate R108
argument then exceeds amdxdna's five-argument DPU register map, and importing
the same dma-buf twice into R109 returns `EALREADY`. The admitted form uses one
in-place attention argument: its first 884,736 bytes hold completed BF16x2 and
its suffix holds R34 activation records. R109 reads the prefix and prepares the
suffix; R108 strided-DMAs the already-rounded high BF16 residual through the
existing residual FIFO. No tensor block is reordered and the immutable
`.rdna2.hfp` layout is unchanged.

Layer 0 reaches FFN cosine `0.99991644`, tail cosine `0.99999886`, and completed
layer cosine `0.99996836`. Preparation drops from roughly 9-12 ms to 7-9 ms.
Paired full-model means improve from 815.294 ms (R48) to 804.722 ms (R108/R109),
or 1.30%. Tokens/J samples are too variable for an energy claim. Admit R108/R109
as the default direct-residual path when both artifacts exist. The next rung
must attack R47/R109's remaining four full completed-state passes or fold
next-layer preparation into the tail/attention context; the kernel-parameter
workaround and LDS decisions remain separate.

### R110 refreshes generic Opus suffix and mixed-width execution

The current R108/R109 completed-layer chain is not native-W4-specific. Locked
M256 layer-0 oracles pass for native OQ8, calibrated OQ8+, freshly generated
OQ8++, and compact mixed OQ6.5, with completed-layer cosine between
`0.99996270` and `0.99997103`. OQ8++ was produced offline from BF16 and the
unified calibration/Hessian package; all 168 backbone projection LDLQ packs
succeeded. No tensor-block conversion was added to a kernel.

Full 24-layer results are 296.0 tok/s for OQ8, 299.1 for OQ8+, 305.1 for OQ8++,
and 294.9 for OQ6.5. OQ8-family BF16 embedding cosine is
`0.99547863-0.99584466`; OQ6.5 reaches `0.95821863`, so it is arbitrary
mixed-width execution evidence rather than an OQ8-quality promotion. This
closes the refreshed generic API matrix but does not alter the performance
blocker: the best row remains about 33x below 10k. Continue by reducing the
remaining full completed-state passes and serialized context boundaries.
The added kernel parameter remains the platform-issue workaround; LDS use or
avoidance and R15 numerical controls remain independent.

### R111 admits one-pass completed-state preparation

R111 preserves R109's in-place completed-prefix/R34-suffix ABI and all
immutable `.rdna2.hfp` ordering. Each core copies one active 3,072-byte
completed row into tile memory, immediately releases the input FIFO, and then
computes RMS plus all three K256 activation chunks from that explicit local
copy. The allocation remains 884,736 bytes because it includes 32 padded rows,
but one physical sweep reads 786,432 active bytes. R111 reduces four sweeps
(3,145,728 bytes) to one and cuts completed-input shim tasks from 32 to 8.

The first schedule held the FIFO across RMS and three packs and reproduced
R54's failure; it remains rejected. Copy-and-release initially left all scales
exact but corrupted the packer-owned group-1/group-2 Q partitions. Those chunks
were 32-byte aligned while assembly used a 64-byte load. Using the allocator's
guaranteed 32-byte alignment restores the gate: five one-code Q differences,
maximum Q delta 1, and `7e-9` maximum scale error. Core text ranges from 9,072
to 10,592 bytes.

Standalone paired means improve from 5.1424 ms for R109 to 5.0949 ms for R111,
only 0.9%, so the default decision uses four counterbalanced full-model pairs
with three encodes per process. R109 averages 760.323 ms, 336.7 input tok/s,
18.88 W, and 17.83 tok/J. R111 averages 749.409 ms, 341.6 tok/s, 18.87 W, and
18.10 tok/J. It wins every latency pair and preserves BF16 embedding cosine
`0.92839295`. Select R111 when its artifact is present and retain R109 as the
fallback.

This is a clean bandwidth-first admission, but its size is diagnostically
important: removing 2,359,296 active bytes per layer improves end-to-end
latency by only 1.44%. The remaining path is not simply capped by external
memory bandwidth. The next rung should consolidate a context and its routes,
with R100 tail fusion the plausible target. Do not append it to R108, whose
largest core has only 16 bytes of program headroom. The added kernel parameter
continues to be the platform workaround; it is separate from LDS placement,
R15 rounding/saturation, and R111's alignment fix.

### R112 admits the fusion-ready tail topology

R112 changes R100's mutable ownership schedule so each core receives eight
contiguous tokens, while DMA gathers from and scatters to canonical token-major
rows. The first design tried to send split architectural X through a second
memory-tile broadcast; compilation rejects it because the tile's output DMA
channels are already consumed. The admitted form uses the third plane reserved
in R99's 4,608-byte mutable row for canonical BF16 X. This eliminates the
separate X DMA/relay route without reordering immutable tensor blocks or adding
kernel-side layout conversion.

Total active input DMA remains 1,179,648 bytes in both variants. R100 divides
that into 786,432 FFN bytes and 393,216 split-X bytes; R112 carries the same
bytes in one joined row-state route. Maximum core text drops from 4,208 to
3,696 bytes and 24 horizontal core flows become zero. The locked oracle is
unchanged at cosine `0.99999861` and maximum error `0.0039062`. Four
counterbalanced 100-command pairs all favor R112: mean dispatch drops from
`0.324965 ms` to `0.218271 ms`, a 32.84% reduction.

Admit R112 as the topology for the next tail-local RMS/three-group pack rung.
Do not treat this standalone tail win as an end-to-end throughput claim. The
added kernel parameter remains the platform-issue workaround and stays enabled;
it is not LDS avoidance. Tile-memory placement, R15 numerical controls, R97
fragment buffers, and context recycling remain separate decisions or fixes.

### R113 admits phase-local tail-to-next-pack fusion

R113 now executes R111's exact next-layer RMS/AWQ/FWHT/int8 preparation in the
R112 tail context. It preloads one 9,216-byte three-group parameter record per
core and packs each still-local two-row completed output. It does not retain a
separate completed-state tensor and therefore removes the 786,432-byte active
input pass that R111 still requires. Canonical mutable row order and immutable
`.rdna2.hfp` tensor-block order remain unchanged.

The failed literal implementation kept all eight completed rows in a
24,576-byte local buffer and exceeded bank capacity. The phase-local form fits
at 9,984 text bytes on every core. Its first diagnostic schedule kept seven
shim output tasks live; group 2 remained all zero. Retiring one old completed
task per stripe before queuing three diagnostics fixes the queue limit, but
those retirements must occur after every stripe is launched. Doing them inside
the construction loop serialized the array and measured 13.3578 ms.

The final oracle is tail cosine `1.00000000`, maximum error `0.0000310`, three
one-code Q differences, Q delta 1, and `7e-9` maximum scale error. Four live
50-command samples average 5.056325 ms. Current R112 and R111 controls total
5.260901 ms, so fusion saves 0.204576 ms (3.8886%) and one full completed-state
pass. One fresh process returned all-zero output during the first series, but
four immediate fresh contexts and the complete repeated series passed; retain
that as context-transition evidence.

Admit R113 as the next bandwidth-first rung, not yet as the full-layer default.
Do not restore the eliminated completed-state pass. The separately added kernel
parameter remains the platform workaround and is independent of LDS placement,
the shim queue schedule, R111 alignment, and context lifetime.

### R114 rejects assembled compact R34 output; consume R113 chunks directly

R114 tested in-context assembly before adding R34 compute. Logical-owner stream
chains and physical column-major stream chains both exhaust legal routing.
Adjacent neighbor-memory ObjectFIFOs avoid extra core flows, but adding a new
shim output route also exhausts routing. A zero-destination-stride scheme cannot
reuse one output task because the DMA compiler requires positive strides.

The remaining split-plane graph reuses the completed-output route, builds in
about nine seconds, and fits at 11,200 bytes maximum core text. It emits only
589,824 bytes: no 16 KiB record padding and no fivefold N-macro materialization.
Hardware parity rejects it with 107,811 pack mismatches, maximum Q delta 254,
and maximum scale error 0.034057196 even when the completed tail passes. Errors
span all three K256 groups and all local-memory owner positions. This proves the
compact assembly/mapping is wrong, but does not isolate one neighbor link. R114
is not admitted.

Change the next slice from producer-side assembly/replication to consumer-side
reuse. R113 already emits correct per-core chunks in a 589,824-byte padded ABI,
of which 199,680 bytes are unique chunk data. Add the first resident R34 GEMM
consumer directly against that ABI and reuse every chunk across its five
N-macros. Do not materialize the canonical 2,949,120-byte activation replicas,
and do not reorder immutable tensor blocks in the kernel; `.rdna2.hfp` order is
an offline or loader responsibility. Once that consumer passes, optimize the
RMS/FWHT pack body that now dominates this seam.

One fresh context returned an all-zero completed tail between otherwise useful
runs; the immediate repeat returned the distributed pack mismatch. Record it as
context-transition evidence, not as an LDS diagnosis. The added kernel
parameter is the workaround that stops the platform issue. It is not LDS
avoidance; local-memory use, output mapping, R111 alignment, and context
lifetime are separate design variables.

### R115 admits direct compact consumption for K256 x N16

R115 implements the consumer-side plan without trying to fix R114 assembly.
Each core owns eight tokens, reads its R113 group-0 chunk directly, and computes
one scaled int8 K256-by-N16 stage. The same offline-packed weight record reaches
all token owners and 32 core outputs scatter to canonical `[256,16]` f32. No
24-token block is assembled, no N-macro activation replica is materialized, and
no immutable tensor block is reordered in the kernel.

The first mapping gate deliberately keeps R113's slot padding. It reads 196,608
physical activation bytes containing 66,560 unique chunk bytes. The image fits
at 1,692 bytes maximum core text. Locked parity is zero mismatches and `2e-9`
maximum absolute error. Six fresh 1,000-dispatch processes pass at a mean of
0.092506 ms. One preceding fresh context returned mostly zero output and the
immediate six-process series passed; keep that as context-transition evidence.

Admit the direct consumer mapping and group-0 matrix stage only. Next add groups
1 and 2 with local f32 accumulation, preserving this ABI and its zero N-macro
replicas. Then extend N one slice at a time. Only after full-K parity should a
DMA gather remove diagnostic padding. Immutable weights remain an offline or
loader `.rdna2.hfp` responsibility. The added kernel parameter remains the
platform workaround that stops the platform issue; it is not LDS avoidance.

### R116 proves full-K direct consumption at N16

R116 adds groups 1 and 2 without changing the R113 ABI. Every token-owning core
consumes three K256 chunks and accumulates a single N16 output tile locally in
f32. Physical activation input is 589,824 bytes with 199,680 unique bytes,
versus 2,949,120 bytes for the old five-N-macro materialization. N-macro
activation replicas remain zero and immutable tensor ordering stays offline.

The first group-1 build corrupts only columns 0-7. Padding weight records to
128-byte starts does not alter the error. Separate group DMA tasks and an
unrolled core schedule return zeros and are also rejected. The correct design
copies the previous 8x16 f32 tile into a 512-byte local staging array before the
next MMUL. This makes the inter-group output reload dependency explicit. K512
with unit scales is bit-exact; full K768 has zero mismatches and `4e-9` maximum
error at 2,220 bytes maximum core text.

Eight passing fresh 1,000-dispatch processes average 0.096384 ms. Two other
fresh processes return all-zero output, after which fresh processes pass again.
Admit the full-K math/ABI rung, not context-stable production selection. The
next slice adds one more N16 output at a time while reusing the same three input
chunks; do not reintroduce N-macro activation replicas. Keep a byte oracle for
each N extension before considering padding removal.

The added kernel parameter remains the workaround that stops the platform
issue. The R116 prior-output staging is a separate local data-dependency fix,
and local-memory use remains an independent design variable rather than the
workaround.

### R117 admits N32 reuse at unchanged activation traffic

R117 doubles the useful output width by feeding four 8-column MMUL halves from
each K-tile activation load. It retains R116's three-group local accumulation
fix, reads the same 589,824 physical/199,680 unique activation bytes, and still
materializes zero N-macro replicas. Immutable N32 records are loader/offline
`.rdna2.hfp` data; no kernel-side tensor reorder is introduced.

Both N16 halves pass with zero mismatches and `3e-9` maximum error. Maximum core
text is 3,192 bytes. Eight passing fresh 1,000-dispatch processes average
0.086916 ms, 9.82% faster than R116's N16 mean of 0.096384 ms despite twice the
useful N work. Fixed dispatch and activation traffic dominate this small stage,
so wider compute per load is productive.

Two fresh contexts returned all-zero output before the eight passes. Admit N32
math and activation-load reuse, not context-stable production selection. For
the next slice, stage the three 2,080-byte compact chunks once per core and
stream at least two N32 weight/output records from that local state. Activation
DMA must remain 589,824 bytes as N grows. This topology, rather than a monolithic
ever-wider accumulator, is the scalable path to N1280.

The added kernel parameter remains the workaround that stops the platform
issue. Wider MMUL use, per-core activation staging, and LDS placement are
independent design decisions.

### R118 admits activation-once N64

R118 stages each core's three compact activation chunks once, releases the
activation FIFO, and streams two full-K N32 weight/output blocks. Activation
DMA remains one 589,824-byte physical pass containing 199,680 unique bytes as N
grows from 32 to 64. N-macro activation replicas remain zero.

The first 6,240-byte stage is rejected because group 1 starts at byte 2,080,
only 32-byte aligned for a 64-byte MMUL activation load. A 2,112-byte stride
creates a 6,336-byte stage with every group 64-byte aligned. This fixes block 0.
The first repeated output BD then leaves block 1 zero; two explicit output tasks
at queue depth two restore both blocks. Final parity is zero mismatches with
`5e-9` maximum error, and core text is 3,736 bytes.

Nine passing fresh 1,000-dispatch processes average 0.106058 ms, only 22.0%
above R117 N32 while doing twice the useful work. One other fresh process
returns all zeros. Admit activation-once N64 math/topology, not context-stable
production selection.

Before increasing the block count, test an output task with both the outer DMA
tiling dimension and `repeat_count`. If admitted, increase N32 block count while
keeping activation DMA fixed. If not, use bounded two-task waves with await/free
between waves; do not queue all N1280 outputs simultaneously.

The added kernel parameter remains the workaround that stops the platform
issue. R118's group alignment and local activation stage are separate design
constraints and do not imply LDS avoidance.

### R119 admits the scalable repeated S2MM task

R119 keeps R118 compute and local state unchanged. It combines the outer output
tiling dimension of two with task `repeat_count=1`. This consumes and scatters
both N32 output objects using one task per stream; the outer dimension alone
only transferred block 0.

Parity is zero mismatches with `5e-9` maximum error. Eight passing fresh
1,000-dispatch processes average 0.102308 ms, 3.54% below the explicit two-task
R118 mean. Two other fresh processes return all zeros. Admit the repeated-task
schedule for N scaling, with the existing context-stability caveat.

Parameterize the staged consumer by N32 block count. Gate four blocks (N128)
before the full 40-block N1280 projection, keeping activation DMA fixed at one
589,824-byte pass. The task repeat must be `block_count - 1`, and the outer
tiling dimension must equal `block_count`.

The added kernel parameter remains the platform workaround. Output task repeat,
activation staging, and LDS placement are independent.

### R120 admits four repeated N32 blocks

R120 parameterizes the R119 graph to N128. The output row strides scale with N,
the outer tiling dimension is four, and task `repeat_count` is three. The
compute kernel, 6,336-byte aligned local activation stage, and one 589,824-byte
activation pass do not change. All four output blocks pass with zero mismatches
and `7e-9` maximum error.

Four passing fresh 1,000-dispatch contexts average 0.115102 ms. Six other fresh
contexts return the known whole-output zero symptom, so admit the N128 topology
without claiming context-stable production selection. Maximum core text is
4,312 bytes. The added kernel parameter remains the platform workaround; this
result neither requires nor implies LDS avoidance.

### R121 admits the complete N1280 projection schedule

R121 grows the same graph to all 40 N32 blocks of the 768x1280 projection. The
task outer dimension is 40 and `repeat_count=39`. Each core still stages the
three R113 activation chunks exactly once. Physical activation traffic remains
589,824 bytes with 199,680 unique bytes and zero N-macro replicas while the W8
diagnostic weight payload grows to 7,987,200 bytes and f32 output to 1,310,720
bytes. Immutable record layout remains an offline/loader `.rdna2.hfp`
responsibility.

The full M256 K768 N1280 byte oracle passes with zero mismatches and `6e-9`
maximum error. All ten fresh 1,000-dispatch contexts pass at
0.319049-0.325542 ms, mean 0.320640 ms. That is about 798,402 M256 projection
rows/s and 30.84 GB/s over the W8 input, compact activation, and f32 output
bytes. Maximum core text is 3,848 bytes.

This is a full-width projection-schedule result, not full-layer resident NPU or
encoder tok/s evidence. The next concrete slice is to feed this schedule from
the generic runtime Opus packed-record API, preserve the oracle for OQ4,
arbitrary mixed OQ bitwidths, OQ8, and +/++ variants, then replace W8 diagnostic
records with native packed/decode records one measured stage at a time. The
added kernel parameter remains the platform workaround. LDS placement, local
activation staging, DMA repetition, and encoding/decode are separate design
variables.
