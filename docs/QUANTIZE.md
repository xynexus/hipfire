# Quantizing Models

`hipfire-quantize` converts Hugging Face safetensors directories, GGUF files, or
source-precision `.hfq` files into HipFire `.hfq` artifacts.

The default quant is `oq4.25++`. Omitting `--format` selects its mixed 4.25-bit
Opus storage plus AWQ and Hessian/LDLQ recipe; the corresponding calibration
artifact is therefore required. Pass `--format` explicitly to select another
encoding.

## Basic Usage

```bash
cargo run --release -p hipfire-quantize -- \
  --input /srv/huggingface/models--Qwen--Qwen3.5-9B/snapshots/<snapshot> \
  --output ~/.hipfire/models/Qwen3.5-9B.oq4.hfq \
  --format oq4
```

The output filename should use the canonical artifact convention:

```text
<family>[-]<version>-<size[-effective/active]>[-tag1][-tag2...][.feature1[.feature2...]].<format>[.arch].hfq
```

Examples:

```text
Qwen3.5-9B.oq4.hfq
Qwen3.5-9B.oq4+.hfq
Qwen3.5-9B.oq4++.hfq
Qwen3.5-9B.mq4l.hfq
Qwen3.5-9B.mq4+.gfx1103.hfq
```

## Quant Token Taxonomy

Quant tokens describe weight encoding only:

```text
<family><bitwidth>[l][+][+]
```

- `mq` / `MQ` is affine Magnum Quant.
- `oq` / `OQ` is symmetric Opus Quant.
- `l` after the bitwidth means Lloyd-Max/codebook MQ encoding, for example
  `mq4l`. Do not use `lloyd-mq4` for new artifacts.
- A first `+` means clip-search, SmoothQuant, AWQ, or a comparable
  activation-aware clipping/scaling pass.
- A second `+` means Hessian/LDLQ error feedback.
- Mixed precision includes a decimal place in the bitwidth, for example
  `mq4.5+` or `oq4.25++`.

Do not use `+` for bundled runtime features or sidecars. Encode each feature as
its own dot group before the quant token, for example `Qwen3.5-9B.mtp.vl.oq4.hfq`
or `Gemma-4-8B.dflash.triattn.oq4++.gfx1151.hfq`.

## Public Quant Names

### Magnum Quant

Magnum Quant (`mq`) is the FWHT-rotated weight family used by the main
Qwen3.5-style runtime path. MQ tokens describe stored weight precision and
offline calibration. They do not force a particular activation precision; decode
GEMV commonly consumes full-precision rotated activations, while batched prefill
may use F16 WMMA or Q8_1/MMQ activation paths for throughput.

The public MQ producer grammar is `mqN`, `mqN+`, or `mqNl`:

| Token | Meaning | Current status |
|---|---|---|
| `mq2` | 2-bit affine MQ, 256-element groups | implemented as `MQ2G256`; research-gated for quality |
| `mq3` | 3-bit affine MQ, 256-element groups | implemented as `MQ3G256` |
| `mq4` | 4-bit affine MQ, 256-element groups | implemented as `MQ4G256`; default production MQ shape |
| `mq5` | 5-bit affine MQ, 256-element groups | reserved/planned; do not publish artifacts yet |
| `mq6` | 6-bit affine MQ, 256-element groups | implemented as `MQ6G256` |
| `mq7` | 7-bit affine MQ, 256-element groups | reserved; no current codec/runtime path |
| `mq8` | 8-bit MQ endpoint, symmetric int8 weights | implemented as `MQ8G256` in the runtime/kernel family; not a general CLI default format |
| `mqN+` | same `mqN` weight layout plus clip-search and AWQ/SmoothQuant-style calibration | same base kernels and dtype as `mqN`; only claim quality after eval evidence for that bitwidth |
| `mqNl` | Lloyd-Max/codebook MQ with `2^N` centroids per group | implemented today for `mq2l`, `mq3l`, and `mq4l` only |

`mqN+` is a producer-side quality recipe, not a new runtime format. The
quantizer strips the trailing `+`, enables clip-search, enables AWQ at the
default alpha unless overridden, and writes the same base MQ weight layout plus
any required sidecar. Runtime dispatch therefore reuses the `mqN` kernels. For
example, `mq4+` remains `MQ4G256` in the loader and uses the same MQ4 GEMV/GEMM
families as `mq4`.

`mqN++` is reserved for producer-specific Hessian/error-feedback MQ recipes,
and the model resolver recognizes tokens such as `mq4++`. It is not a generic
`hipfire-quantize --format mqN++` producer today. A valid MQ++ artifact must
still contain legal base `mqN` weights, plus any sidecar needed by the runtime,
so it can reuse the same loader dtype and kernels as `mqN`/`mqN+`.

Lloyd-Max/codebook MQ uses the canonical artifact spelling `mqNl`, for example
`mq4l`. The current quantizer still accepts legacy CLI spellings such as
`--format lloyd-mq2`, `--format lloyd-mq3`, and `--format lloyd-mq4`; new
artifact names should use `mq2l`, `mq3l`, or `mq4l`.

Do not publish `mq5`, `mq7`, or higher Lloyd spellings just because the grammar
allows them. Add the dtype, quantizer arm, loader mapping, kernels, and
coherence/eval evidence first.

### Opus Quant

Opus Quant (`oq`) is the symmetric FWHT-rotated signed-integer family. OQ tokens
describe stored weights and calibration. The runtime can reuse the same OQ
weights with different activation precision paths.

| Format | Stored weights | Calibration recipe |
|---|---|---|
| `oq4` | signed int4 OQ, 256-element groups | plain RTN/clip path |
| `oq4+` | same `oq4` bytes | AWQ/SmoothQuant-style activation-aware scaling |
| `oq4++` | same `oq4` bytes | `oq4+` plus full-Hessian LDLQ/OBS error feedback |
| `oq8` | signed int8 OQ, 256-element groups | plain RTN/clip path |
| `oq8+` | same `oq8` bytes | AWQ/SmoothQuant-style activation-aware scaling |
| `oq8++` | same `oq8` bytes | `oq8+` plus full-Hessian LDLQ/OBS error feedback |

The Rust code still uses internal enum names such as `Oq4G256` and `Oq8G256`.
Those are implementation details; artifact names should use the canonical `oq*`
tokens.

For an OQ8 matrix whose logical `K` is not divisible by 256, the quantizer
stores one independently zero-padded final group per row and tags the tensor as
`Oq8G256RowPadded` (`quant_type=43`). Its payload is
`M * ceil(K/256) * 258` bytes while its HFQ shape remains logical `[M,K]`.
This layout is XDNA-native and must not enter the GPU OQ8 unpacker. Exact-width
OQ8 matrices retain `Oq8G256` (`quant_type=35`) and remain usable by either
backend. Set `HIPFIRE_OQ_RAGGED_Q8` while quantizing when a GPU-compatible
artifact is required; ragged matrices then use `Q8F16` instead.

For a Qwen3 or EmbeddingGemma SentenceTransformers checkpoint intended for the
XDNA2 embedding path, add `--npu-embedding`. This reads `modules.json`, the
pooling/Dense module configs, and `config_sentence_transformers.json`, then
embeds a typed `hipfire.embedding.v1` contract in the HFQ metadata. The contract
records prompt roles, pooling, output dimensions, the 128/256/512/1024/2048
sequence buckets, the 4096-padded-row dispatch ceiling, and the required AIE2P
Opus storage layout. Runtime admission reads that metadata and never infers NPU
execution from the filename.
The compiled manifest and resident buffer contract are specified in
[Qwen3 embedding image ABI](npu/QWEN3-EMBEDDING-IMAGE-ABI.md).

```bash
cargo run --release -p hipfire-quantize -- \
  --input /srv/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/<snapshot> \
  --output ~/.hipfire/models/Qwen3-Embedding-0.6B.npu.oq8+.gfx1151.hfq \
  --format oq8+ \
  --npu-embedding \
  --imatrix <Qwen3-Embedding-0.6B.imatrix.gguf>
```

The plus marks are positional:

- `+` means activation-aware clipping/scaling. The quantizer auto-enables AWQ
  for `oq4+`, `oq8+`, and `mqN+`; provide an imatrix or Hessian-derived imatrix
  so the scaling is real.
- `++` means the same first-stage calibration plus Hessian/LDLQ feedback.
  `oq4++` and `oq8++` are wired to `--hessian`; MQ++ tokens are reserved for
  producer-specific artifacts and need an explicit producer/evidence trail.
- No suffix means the plain weight codec.

For a quality-gated `oq4+` artifact, provide activation calibration inputs:

```bash
cargo run --release -p hipfire-quantize -- \
  --input <source-model> \
  --output ~/.hipfire/models/<name>.oq4+.hfq \
  --format oq4+ \
  --imatrix <model>.imatrix.gguf
```

For a quality-gated `oq4++` artifact, provide full-Hessian calibration inputs:

```bash
target/release/hipfire-coexistence calibrate \
  --model /srv/huggingface/models--Qwen--Qwen3.5-397B-A17B \
  --corpus benchmarks/calib/calib-5m.txt \
  --output ~/.hipfire/calib/Qwen3.5-397B-A17B.calib.hfq \
  --kldref \
  --sequence-batch 64 \
  --time-tile 32 \
  --max-rows 2048 \
  --layer-prefetch-bytes 17179869184
```

The family-neutral native engine resolves a thin model adapter, loads the
embedding once, keeps layer-boundary activations in host memory or an mmap
spool, then loads one layer from the original safetensor shards and runs every
calibration microbatch before releasing that layer. Per-layer
Hessians/imatrices are spooled to disk immediately. The final norm/lm-head load produces
`lm_head.kldref_{idx,logit,logz}` without another BF16 checkpoint sweep. The
native read ledger rejects attempted tensor rereads. Routed experts are
captured separately with per-expert coverage and tile-admission telemetry.
Undercovered experts either fail the strict gate or are explicitly listed for
BF16/F16 preservation.

The CLI's automatic geometry ceiling is 2,048 rows. It probes the adapter's
layer/state/scratch estimate against live VRAM and verifies the candidate with
a real allocation, falling back to a smaller geometry where required. A gfx1151
sweep on the 397B source selected 2,048 rows and sequence batch 64 for the
production recipe; explicit geometry remains part of the artifact fingerprint
and is required when reproducing that run on another host.

Run the command once with `--dry-run` before allocating the artifact. The JSON
report includes the compact/dense Hessian mode, calibration payload estimate,
part-plus-final-assembly peak, mmap boundary bytes, safety margin, filesystem
free bytes, and a sufficiency verdict. A fresh run is refused when this
conservative bound does not fit. Corpus tokenization is windowed according to
the remaining requested sample geometry, so a small smoke does not tokenize a
large concatenated prefix before truncation.

For a bounded real-weight mechanism check, add `--pause-after-layers N`. The
engine commits exactly `N` total layers, the boundary checkpoint, and the
monotonic read ledger, then returns `status: paused` without publishing a
calibration artifact. Continue the same job with `--resume` and either a larger
pause count or no pause count. The pause control changes execution scheduling,
not the run fingerprint or final artifact semantics. Each committed layer also
records load/upload, execution, capture-write, collector-finish, part-sync/hash,
and total pre-checkpoint time; the same timing history is copied into the final
artifact for throughput and resume ETA analysis.

Network-backed sources use a bounded one-layer lookahead by default. After
layer N is GPU-resident, a family-neutral worker reads layer N+1's canonical
physical source ranges into the operating-system page cache while N executes.
The worker uses one fixed 8 MiB staging buffer, never consumes the logical read
ledger, and waits only for any unfinished tail before N+1 uploads. The default
budget is 16 GiB with a live 32 GiB host-memory reserve; use
`--layer-prefetch-bytes 0` to disable it. Checkpoints record prefetched bytes,
background read time, foreground wait time, submission time, and errors, while
older timing checkpoints remain resume-compatible.

On unified-memory systems, completed safetensor ranges receive mapping-level
`MADV_DONTNEED` plus a backing-file cache hint after their synchronous GPU
upload. A canonical range with a declared tied-weight alias is retained until
the alias view is consumed. This bounds layer-stream RSS without turning a tied
embedding/lm-head into an untracked source reread; later access safely refaults
the original read-only bytes.

This is one BF16 source-checkpoint pass. `hipfire-quantize` is the second
source-checkpoint pass. Later KLD/PPL evaluation reads the quantized HFQ, not
the BF16 safetensors. `scripts/collect_hessian.py` remains an explicit
parity/debug oracle; it is not the production model-forward path.

KLDREF is the teacher signal for comparing quant candidates; it is not an
input to the Hessian calculation itself. `hipfire-quantize` consumes the
Hessian/imatrix records for AWQ/LDLQ, while the bundled KLDREF is retained for
matched-corpus candidate scoring and promotion evidence.

```bash
cargo run --release -p hipfire-quantize -- \
  --input <source-model> \
  --output ~/.hipfire/models/<name>.oq4.25++.hfq \
  --format oq4.25++ \
  --awq \
  --ldlq \
  --hessian ~/.hipfire/calib/<Model-Size>.calib.hfq
```

The two commands can be run as one resumable workflow. Extra quantizer flags
go after `--`:

```bash
python3 scripts/two_pass_quantize.py \
  --model /srv/huggingface/models--Qwen--Qwen3.5-397B-A17B \
  --calib ~/.hipfire/calib/Qwen3.5-397B-A17B.calib.hfq \
  --output ~/.hipfire/models/Qwen3.5-397B-A17B.oq4.25++.hfq \
  --format oq4.25++ \
  --batch-size 64 \
  --time-tile 32 \
  --max-rows 2048 \
  -- --awq --ldlq
```

The native calibration command owns the shared GPU lock directly; the wrapper
scopes pass 2 under `hipfire lock run` without nesting locks. It writes an atomic
`*.two-pass.json` manifest containing the native read ledger and both artifact
fingerprints. Use `--skip-calib` to resume pass 2 from an existing artifact,
and `--dry-run` to inspect both commands without loading the model. Geometry is
part of the two-pass recipe fingerprint, so a resume cannot silently change its
sequence batch, time tile, or row budget.

Large quantization runs spill completed tensors to the output filesystem to
bound RAM. During final HFQ assembly on Linux, each spill range is hole-punched
only after its payload has been copied and included in the quantization hash.
This keeps peak temporary storage near one output artifact instead of retaining
a full spill plus a full second copy; filesystems without hole-punch support
retain the conservative two-copy behavior.

For the end-to-end Qwen3.5 397B workflow—including DFlash conversion,
calibration/KLDREF, target quantization, TriAttention/CASK sidecar generation,
resumption, and an admission manifest—use `scripts/induct_model.py`. See
[MODEL-INDUCTION.md](MODEL-INDUCTION.md).

`--hessian` retains its historical flag name but reads the unified HFQM
`*.calib.hfq` package. New workflows must not create legacy HFHS
`*.hessian.bin` sidecars.

## Activation-Precision Reuse

Artifact tokens describe stored weights. `W4A4`, `W4A8`, `W4A16`, `W8A8`, and
`W8A16` describe the runtime compute contract:

```text
W<weight bits>A<activation bits>
```

The same artifact can therefore be reused across multiple runtime qualities when
the loader and dispatch path support it.

| Stored artifact | Runtime quality | Reused weights? | Main path |
|---|---|---:|---|
| `oq4`, `oq4+`, `oq4++` | `W4A4` | yes | rotate/FWHT activation, `quantize_act_oq4`, then `gemm_oq4_grouped_wmma` |
| `oq4`, `oq4+`, `oq4++` | `W4A8` | yes | sign-extend OQ4 weights to the MMQ int8 path and use Q8_1/int8 activations via `gemm_oq4_residual_mmq` |
| `oq4`, `oq4+`, `oq4++` | `W4A16` | yes | consume full-precision rotated activations directly in `gemv_oq4_grouped` or `gemm_oq4_grouped_f16_wmma` |
| `oq8`, `oq8+`, `oq8++` | `W8A8` | yes | rotate/FWHT activation, `quantize_act_oq8`, then `gemm_oq8_grouped_wmma` |
| `oq8`, `oq8+`, `oq8++` | `W8A16` | yes | consume full-precision rotated activations directly in `gemv_oq8_grouped` |
| `mq4`, `mq4+` | `W4A8` prefill | yes | MQ4/HFQ4 MMQ path with Q8_1 activations, same packed MQ4 weights |
| `mq4`, `mq4+` | `W4A16` decode/F16 prefill | yes | `gemv_mq4g256` consumes f32 rotated activations; F16 WMMA prefill dequants weights inline |

This is why quality comparisons should state both the artifact token and the
activation path. `oq4+ W4A16` and `oq4+ W4A4` may use the same `.hfq` weight
bytes and sidecars, but they test different activation damage. The W4A4 path is
useful as a completeness and regression tier, while current production-quality
OQ4+ work prefers W4A16 decode and W4A8/MMQ batched prefill.

For OQ4 batched prefill, `HIPFIRE_OQ4_PREFILL_ACT_BITS` can force a path for
experiments:

| Value | Path |
|---|---|
| `4` | W4A4 int4 activation path |
| `8` | W4A8/MMQ int8 activation path |
| `16` | W4A16 F16-WMMA path |
| unset | runtime default, currently MMQ for larger batches and the coverage path for small batches |

For decode, OQ4 and OQ8 avoid activation quantization in the main GEMV path:
they rotate the activation and multiply directly against dequantized signed
weights (`W4A16` or `W8A16`). This reuses the same stored weights while avoiding
the A4/A8 activation error and extra activation-quantization launch.

## Implementation Notes

- MQ and OQ 256-group formats require `K % 256 == 0`. Ragged tensors fall back
  to safer formats such as Q8 in the quantizer.
- Embeddings, routers, norms, small metadata tensors, and other precision-
  sensitive tensors may stay Q8/F16/BF16 even when the artifact token names an
  MQ or OQ body format.
- `+` and `++` are quality claims only when the calibration inputs were present
  and the runtime path attaches the matching sidecar. Check loader logs and run
  KLD/PPL/coherence gates before promoting an artifact.
- Older plan docs may mention `OQ+`, `Opus Plus`, `op4`, or `op8`; these are
  historical spellings and are not accepted by the current quantizer.

## Useful Flags

| Flag | Use |
|---|---|
| `--chat-template-file <path>` | Override the chat template embedded from the source model. |
| `--threads <n>` | Set Rayon worker threads. Defaults to 80% of host cores. |
| `--imatrix <path>` | Load llama.cpp imatrix GGUF data for activation-aware calibration. |
| `--awq` / `--awq-alpha <f>` | Enable the first `+`: activation-aware weight pre-scaling. Requires imatrix data or a Hessian-derived imatrix. |
| `--ldlq` | Enable the second `+`: full-Hessian error-feedback packing. Requires `--hessian`. |
| `--arch-id <id>` | Override the architecture id stamped in the `.hfq` header. |

After producing a portable OQ4 artifact, use `hipfire optimize` to pre-pack it
for a specific GPU architecture (the `repack` alias is still accepted):

```bash
hipfire optimize ~/.hipfire/models/Qwen3.5-9B.oq4++.hfq --arch gfx1103
```
