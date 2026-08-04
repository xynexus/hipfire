# BF16L3-resident KLD reference artifacts (GPU only)

Status: transcode primitive landed (`storage::transcode_resident`, same commit
series as this doc); batched LUT3 path not started

Owner: quant-format / KLD eval / RDNA kernels

Scope: **GPU only.** KLD references are generated on the GPU. Nothing here
targets XDNA/NPU, and no NPU encode path should be added to satisfy it.

Related: `crates/hipfire-quant-format/src/storage.rs`,
`crates/hipfire-primitives/src/bf16_lut3.rs`,
`crates/hipfire-runtime/src/kld_eval.rs`, `kernels/src/gemv_bf16l3.hip`

## Why a lossless codec at all, given OQ8++

OQ8++ is the better answer whenever memory is the binding constraint:

| encoding | bits/weight | vs BF16 |
|---|---:|---:|
| BF16 raw | 16.00 | 1.00x |
| `bf16_lut3` | ~11.60 | 1.38x |
| `bf16_huff` | ~10.66 | 1.50x |
| `Oq8G256` — `[f16 scale][256 int8]` | 8.06 | 1.98x |

A KLD reference is the exception, and the only reason this work exists. The
artifact a candidate is *scored against* has to be the real weights, or the
measurement moves with the reference. Same for drafter/target logit parity and
for chasing a kernel discrepancy — if the weights themselves differ, the
investigation has no fixed point. So references stay bit-exact even though a
near-lossless format would be smaller.

`bf16_huff` and `bf16_lut3` are both exactly lossless: every input `u16` is
reproduced bit-for-bit including zeros, denormals, infinities and NaN payloads.

## Current state

**Transcode exists.** `storage::transcode_resident` recodes on-disk Huffman to
LUT3, and `storage::resident_type` is the companion predicate.
`huff_transcodes_to_the_same_lut3_bytes_as_direct_encoding` pins the output as
byte-identical to `bf16_lut3::encode` of the original weights, so a resident
Huffman tensor and a resident LUT3 tensor of the same weights decode the same
in-kernel. It is deliberately **not wired into the load path** — see the
"Load-path wiring" section.

**One LUT3 kernel exists.** `gemv_bf16l3` — decode-shaped GEMV, wave32, zero
LDS, F32 accumulate, `K % 256 == 0` asserted by the dispatch caller.

**The KLD forward never reaches it.** `ChunkScoredForward::forward_chunk_scored`
scores a chunk of tokens at a time, so every weight op is batched: GEMM, not
GEMV. The BF16 GEMM paths are `rocblas_gemm_bf16_nt_bf16` / `_nt_f32` plus WMMA
overlays such as `gemm_bf16_moe_grouped_wmma_gfx1151`. A LUT3-stored kldref
therefore loads and works today — `expand_bf16_index` expands it to plain BF16 —
but yields the disk saving and none of the VRAM saving.

**Reference and candidate are never co-resident.** `tiny_harness::run_kld` is
sequential:

```rust
let refs  = run_logits(arch, ref_path,  gpu, &tokens, warmup)?;
let cands = run_logits(arch, cand_path, gpu, &tokens, warmup)?;
```

`run_logits` loads, scores, and drops each model. Peak VRAM is one model.
Chunked scoring and reference streaming are already implemented; neither is a
lever here. The only thing a batched LUT3 path buys is the reference's own
footprint, 16 -> 11.6 b/w on whichever artifacts are stored as LUT3.

## What a native LUT3 GEMM has to solve

1. **Blocking mismatch.** The format blocks the *flattened* tensor at 256
   elements. WMMA wants K-steps of 16 with a fixed fragment lane mapping. A GEMV
   reads one row linearly so block boundaries are trivial; a GEMM tile spans
   rows, and a 16x16 fragment's elements land at scattered positions across LUT3
   blocks. The workable shape is decode a 256-element block into LDS, then have
   fragments read from there — which reintroduces the LDS traffic `gemv_bf16l3`
   deliberately avoids.

2. **`K % 256` becomes a tiling constraint.** GEMV asserts it once. A GEMM tiles
   K, so every tile boundary must stay block-aligned or a tile straddles blocks
   and needs two LUTs and two escape-table lookups in flight.

3. **Escapes make tile cost data-dependent.** `esc_tab[b]` is one indexed read
   per block, amortised fine over a whole row. Per fragment it is a dependent
   load on the critical path, and escape density varies across the matrix, so
   decode cost is not uniform tile to tile.

4. **It takes work off rocBLAS.** BF16 GEMM is vendor-tuned and multi-arch
   today. Owning a LUT3 GEMM means owning its performance across RDNA2/3/4 per
   the portability invariant in `AGENTS.md`.

## Recommended sequence

**Phase 1 — decode-into-scratch.** Decode LUT3 into bounded BF16 tiles in
scratch, feed rocBLAS per tile. Keeps the vendor GEMM; the VRAM ceiling becomes
scratch-sized rather than tensor-sized. Critically for a reference path it is
bit-exact-testable against the expanded path, because both produce identical
BF16 before the GEMM ever runs.

**Phase 2 — native `gemm_bf16l3`.** Only if Phase 1 measures too slow. By then
the tiling and the correctness harness exist, and the native kernel is a
drop-in for the innermost step.

Doing Phase 2 first means debugging a new WMMA kernel and a new memory path at
once, on the one artifact whose whole job is being trustworthy.

**Lead arch: gfx1151 or gfx1201, not gfx1103.** Phase 2 puts decoded blocks in
LDS, and gfx1103 carries the HIP-719 CWSR hazard around multi-wave LDS work (see
`README` and `docs/plans/gfx1103-lds-hip719-investigation.md`). Bring it up where
LDS is not a known hazard, then port.

## Load-path wiring (blocks any phase)

`expand_bf16_index` decides expansion from the index alone, before any payload
is read. LUT3's length is data-dependent — fixed planes plus a *variable* escape
plane — so the resident size is unknown until the payload is transcoded. Two
options, and this is a real decision:

- **Eager transcode at open** fixes sizing but reads every payload page, which
  breaks `open_index_only`'s documented guarantee that a 100GiB+ artifact can be
  inspected without touching them.
- **Lazy transcode at materialisation** requires the upload path to size from
  the returned buffer rather than `info.data_size`. Getting that wrong corrupts
  weights silently instead of failing loudly.

Also needed: **per-artifact residency.** `bf16l3_resident()` is a process-wide
env flag (`HIPFIRE_BF16L3_RESIDENT`). Scoring loads a reference and a candidate
in the same process, and they want different treatment, so residency has to
become a per-load parameter.

And **a `K % 256` fallback**: tensors whose K is not a whole number of blocks
must fall back to expansion, decided per tensor rather than per model. Vocab- and
head-dim-adjacent shapes are the risk.

## Producer side

The kldref producer emits plain BF16. Nothing above is reachable until it can
emit LUT3, or emit Huffman and rely on `transcode_resident`. Emitting Huffman on
disk is the better default — 1.50x rather than 1.38x — with LUT3 materialised at
load.

Note the store itself is not a reliable guide here: `/srv/hipfire/kldrefs/` held
`qwen3.5-{0.8b,2b}*.kldref.hfq` on 2026-08-04 and is **empty** as of 2026-08-21,
with the mount otherwise healthy. Confirm what the producer actually writes
before sizing this work off whatever happens to be on the share.

## Validation

- Bit-exactness against the expanded path is the gate, not a tolerance. A KLD
  reference that is *nearly* right is worse than one that is obviously wrong.
- `gemv_bf16l3`'s existing fixtures cover the decode; Phase 1 needs tile-boundary
  cases, specifically K a multiple of 256 but tiles that split a block, and
  escape-heavy blocks (a flat exponent run gives zero escapes and exercises
  nothing).
- Scoring a known artifact against a LUT3 reference and a BF16 reference must
  produce identical KLD, not merely close.

## Open questions

- Is any kldref large enough that 1.38x decides whether it fits? If not, this is
  optimisation without a forcing constraint and should stay parked.
- Should `bf16_huff` remain disk-only, or is there a case for Huffman-resident
  once a transcode exists? (Analysis says no: the chunk-overhead maths survives
  at 1.43-1.47x for per-lane chunks, but the bit-serial cursor is a dependency
  chain WMMA cannot absorb, and OQ8++ dominates on any pure-capacity argument.)
