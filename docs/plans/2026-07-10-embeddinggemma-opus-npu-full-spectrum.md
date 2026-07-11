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

- `EmbeddingGemma-300M.oq4.hfq`
- `EmbeddingGemma-300M.oq4+.hfq`
- `EmbeddingGemma-300M.oq4++.hfq`
- `EmbeddingGemma-300M.oq4.25.hfq`
- `EmbeddingGemma-300M.oq4.25+.hfq`
- `EmbeddingGemma-300M.oq4.25++.hfq`
- `EmbeddingGemma-300M.oq4.5.hfq`
- `EmbeddingGemma-300M.oq8.hfq`
- `EmbeddingGemma-300M.oq8+.hfq`
- `EmbeddingGemma-300M.oq8++.hfq`
- joint-scale and Dense/tail promotion candidates already in that directory.

Also generate/test at least one lower and one higher arbitrary mixed point, for
example OQ4.125 and OQ6.5, to prove the runtime is not hardcoded to OQ4.25.

## Verification

### Correctness

- CPU integer/scaling oracle parity for W4, mixed, and W8 matrices.
- Hardware parity for multiple K/N/M shapes and variable overlay counts.
- GPU-versus-NPU full embedding cosine and max-absolute-error checks.
- Existing selection-stability metrics against BF16.
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
