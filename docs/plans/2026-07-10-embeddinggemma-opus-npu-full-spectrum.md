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
