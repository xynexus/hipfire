# Native GPU DFLASH Opus Quant

Status: native packed W4A8/W8A8 GPU path implemented; tuned/admission work open

Owner: runtime / RDNA kernels / quantization

Related: [bundled model induction program](2026-07-22-bundled-model-induction-program.md)

## Problem

`dflash_convert` writes three non-rotated, plain-basis formats for the NPU path:

| Quant type | Storage | Canonical meaning |
|---:|---|---|
| 45 | signed int8 + F16 scale per G256 | `oq8+` plain basis |
| 46 | signed int4 bulk + sparse int8 overlays per G256 | mixed `oq4.N+` plain basis |
| 47 | signed int4 + F16 scale per G256 | `oq4+` plain basis |

The `+` is the converter's clip search. These are intentionally not the
rotated primary-model OQ encodings. Before this scope, the GPU path dequantized
all three formats on the CPU, converted them to F16, uploaded the expanded
weights, and dispatched the F16 path. That path is now retained only as an
explicit measurement oracle, not production native packed OQ execution.

## Implementation evidence: 2026-07-22

The current worktree now has distinct `DflashOq8Plain`, `DflashOq4Plain`, and
`DflashOq4MixedPlain` runtime dtypes. The loader validates arch-20 metadata,
plain basis, group size, block bytes, canonical quant token, clip-search recipe,
producer fingerprint, mixed-overlay geometry, payload length, and finite F16
scales before uploading the original interleaved blocks. Production dispatch no
longer expands these weights on the host. The old comparison path is opt-in via
`HIPFIRE_DFLASH_OQ_ORACLE=f16`.

`gemm_dflash_oq_plain_ref.hip` consumes qt=45/46/47 blocks directly and applies
no FWHT. Ragged-K shapes retain a scalar W4A16/W8A16 oracle. Aligned G256 shapes
use an eight-wave kernel: 256 threads cooperatively quantize each activation
tile to signed int8 in LDS, then each wave computes one output row with signed
dot4 while OQ4 weights remain packed. Mixed OQ applies exact sparse int8
last-write-wins corrections over the int4 bulk result. The channel test passes
all three layouts on gfx1103 for both ragged W4A16/W8A16 and aligned W4A8/W8A8,
including three batches, extreme signed codes, and duplicate mixed-overlay
positions. Direct HIP compilation passes for gfx1030, gfx1100, gfx1103,
gfx1151, and gfx1201. `dflash_convert` now accepts canonical
`--format oq4+`, `--format oq8+`, and `--format oq4.25+` tokens; legacy boolean
spellings are rejected.

Promotion remains open: a tuned WMMA/register-tiled path and Atlas evidence do
not exist, and the native kernel is still slower than the expanded F16 oracle.

Real-artifact bring-up now also covers Qwen3.5-9B. The canonical converter read
the safetensors DFLASH snapshot and wrote a 508 MiB `oq4+` sidecar for 1.049B
parameters. The packed GPU path loaded it without host expansion and completed
a five-layer B=16/L=32 draft forward with finite output. Against the explicit
F16-expanded oracle, final hidden-state error was `max_abs=0.02495`,
`mean_abs=0.001844`, and `rmse=0.002347`. The reference path took 528 ms versus
60 ms for the expanded F16 oracle. The aligned DP4A path lowers the repeated
five-layer median to 276 ms (273.81/276.21/280.27 ms), while the same binary's
F16 oracle median is 56.05 ms. Subphase timing attributes 169.6 ms to packed
FFN GEMMs and 39.5 ms to packed attention GEMMs, versus 36.3 ms and 8.7 ms for
F16. This is a 1.91x improvement over the scalar packed kernel but remains
4.92x slower than F16. The additional W4A8 activation approximation changes
the deterministic final hidden state by `max_abs=0.351851`,
`mean_abs=0.048445`, `rmse=0.061944`, with cosine similarity `0.999715` to the
W4A16/F16-activation oracle; all values remain finite.

End-to-end Q8-KV speculative decode with fixed B=16 produced the exact same
16-token stream, five-cycle acceptance histogram, and counters as the explicit
F16-expanded oracle (`accepted=10`, `committed=20`, `tau=2.0`). Native packed
decode reached 4.15 tok/s versus 5.33 tok/s for the F16 oracle and used 6.58 GiB
reported VRAM versus 7.97 GiB. The native route is therefore correctness-ready
and materially more compact, but it must not be promoted as the tuned default
performance claim.

## Staged-activation tuning checkpoint: 2026-07-23

The aligned W4A8/W8A8 path now splits activation quantization from the weight
GEMM. `quantize_dflash_act_g256` quantizes each `(batch, G256)` activation once
into bounded, stream-reused scratch; all output-row blocks consume that staged
payload. The former inline kernel repeated the same quantization
`ceil(M / 8)` times per projection. Runtime dispatch processes at most 64 rows
per scratch chunk and retains the exact F16-activation path for ragged K.

The recovered kernel source is byte-identical to the source embedded in the
release parity binary (SHA-256
`b01584157e00e735df750a21b355be1238da78e9306388277d7334935267b569`).
Direct HIP compilation passes for gfx1030, gfx1100, gfx1103, gfx1151, and
gfx1201. On gfx1103, code-object metadata reports zero spills and zero LDS for
all staged weight kernels: OQ8 uses 14 VGPR/19 SGPR, OQ4 uses 19 VGPR/19 SGPR,
and mixed OQ uses 37 VGPR/42 SGPR. The separate quantizer uses 13 VGPR/13 SGPR
and 36 bytes of LDS. Both aligned/ragged dispatch-policy tests pass, and the
full no-GPU gate passes. Live packed parity, Atlas rows, and end-to-end
acceptance/performance comparison remain required before replacing the
2026-07-22 performance verdict above.

## Goal

Keep DFLASH OQ weights packed on GPU and execute native HIP kernels for pure
OQ4, OQ8, and mixed OQ on RDNA2, RDNA3, and RDNA4. The same non-rotated artifact
must remain consumable by the NPU path; GPU support must not rewrite it into a
rotated primary-model format.

## Format and naming cleanup

Replace boolean format flags with the canonical quant token contract:

```bash
dflash_convert --format oq4+
dflash_convert --format oq8+
dflash_convert --format oq4.25+
```

Canonical sidecar names are:

```text
<Model>.dflash.oq4+.hfq
<Model>.dflash.oq8+.hfq
<Model>.dflash.oq4.25+.hfq
```

Remove legacy spelling/order fallbacks as part of the change. Metadata records
plain basis, group size, block bytes, overlay count, clip-search recipe, and
producer fingerprint. The loader validates metadata against the quant type and
actual payload length.

## Runtime representation

Add explicit packed DFLASH storage variants instead of lying that the loaded
weight is `DType::F16`. Preserve:

- packed device buffer;
- logical matrix shape;
- group size and block stride;
- overlay count for mixed OQ; and
- `rotated = false`.

Dispatch must make the basis distinction explicit. Never call a rotated OQ
kernel on plain-basis DFLASH data and never apply FWHT to its activations.

## Kernel ladder

Implement and validate in this order:

1. **OQ8 W8A16 reference kernel**: F16/BF16 activation, packed signed int8
   weights, F32 accumulation. This establishes indexing and group-scale parity.
2. **OQ4 W4A16 reference kernel**: signed nibble decode in registers, no
   activation quantization.
3. **Mixed OQ W4A16 reference kernel**: OQ4 bulk plus sparse int8 overwrite or
   correction in the same output accumulation.
4. **Batched W8A8/W4A8 paths** for DFLASH block sizes, with per-token activation
   scales and no FWHT.
5. **Tuned WMMA/register-tiled paths** selected by architecture and shape.

Favor register tiling on gfx1103; do not introduce an LDS-heavy default that can
wedge the local APU. Keep a scalar/reference HIP kernel available for golden and
unsupported-shape coverage.

## Dispatch and fallback policy

- Native packed execution is selected from the actual storage variant, shape,
  batch, and architecture capability.
- Ragged K/N shapes use a correct packed reference kernel or an explicit
  high-precision fallback; they must not be mislabeled as native OQ.
- The current CPU-dequantized F16 path remains temporarily as an explicit
  comparison oracle, not an automatic production fallback.
- Kernel logs and Atlas rows report storage and activation contracts separately
  (`oq4+ W4A16`, `oq4+ W4A8`, `oq8+ W8A16`, and so on).

## Correctness gates

1. CPU pack/dequant oracle for every group layout and overlay count.
2. GPU single-matrix parity over edge shapes, partial groups, zero groups,
   extreme scales, duplicate/invalid overlay positions, and non-finite rejection.
3. Full DFLASH layer parity against the F16-expanded oracle.
4. Draft logits: finite, bounded max/mean error, top-k overlap, and top-1
   agreement on a fixed prompt corpus.
5. End-to-end decoded output plus DFLASH tau/acceptance comparison against BF16
   and F16 draft baselines.
6. Compile/fixture coverage for gfx1030, gfx1103/gfx1100, gfx1151, and gfx1201.
7. `./tests/tiny-affected-gate.sh --require-coverage`.

## Performance gates

Collect phase-aware Atlas rows for DFLASH projection and full-draft time. Report:

- packed bytes read versus F16-expanded bytes;
- conversion/upload time and steady-state memory;
- decode/block draft latency;
- accepted tokens per draft call; and
- end-to-end tokens/s.

A kernel is not promoted merely because its isolated GEMM is faster. It must
improve end-to-end DFLASH throughput without failing output or acceptance gates.

## Definition of done

No host-side weight expansion occurs for supported OQ DFLASH artifacts, GPU
memory reflects packed size, all three format families execute natively, and
quality/performance evidence exists for every promoted architecture/path.
