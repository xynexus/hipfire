# Bundled model induction program

Status: implementation in progress; two candidate rows rejected, no model row admitted or transferred

Owner: unassigned

This is the coordination document for making the requested induction workflow
real rather than producing bundles that only look complete. The target workflow
is:

1. read a Hugging Face Safetensors snapshot;
2. generate a native calibration artifact;
3. quantize the target to `oq4.25++`;
4. generate CASK/TriAttention calibration data;
5. quantize an optional DFLASH drafter to `oq4+`;
6. embed the DFLASH and CASK components in one HFQ model;
7. load the embedded components directly at runtime; and
8. transfer only the verified bundle.

Artifact construction is not an admission claim. Astrea plans, successful
composition, finite metadata, and a load smoke do not replace KLD/PPL, DFLASH
acceptance, long-context CASK, combined-feature, or Kernel Atlas evidence.

## Implementation checkpoint: 2026-07-22

The shared prerequisites are under active implementation. Compose v2 and its
writer now live in dedicated offline tooling and losslessly namespace DFLASH
HFQM plus raw TRIA. Embedded DFLASH passed real Qwen3.5-9B parity under the
required source precedence. A real MQ4 + arch-20 `oq4+` DFLASH + TRIA bundle
also loaded both embedded roles through the daemon and generated without loose
paths. Plain-basis DFLASH qt=45/46/47 execute from packed GPU weights and pass
output/acceptance parity. The aligned eight-wave DP4A path is 1.91x faster than
the scalar packed kernel and reduces reported DFLASH VRAM, but remains 4.92x
slower than the F16-expanded draft-body oracle and is not promoted.
Activation quantization is now staged once per `(batch, G256)` instead of once
per eight output rows; the zero-spill, zero-LDS staged weight kernels compile
across gfx1030/gfx1100/gfx1103/gfx1151/gfx1201, but live parity and Atlas
performance evidence are still pending.
Gemma 4 native streamed calibration now passes dry plans and one-layer dense/
MoE GPU smokes, emits per-layer half-split HFQM CASK, and has a heterogeneous
F32 CASK runtime. BLS now has safe arch-25 identity, quantizer ingress, a
complete 49-layer minimal calibration/CASK producer smoke, and a layered
full/SWA resident backend that accepts its heterogeneous CASK package;
full-model BF16 parity and admission remain open.

The first product-recipe row is now materially constructed. Qwen3.5-2B used
the frozen 128-sequence, 2048-token corpus recipe to produce an audited 1.7 GiB
calibration artifact (186 Hessians and 186 imatrices), a 77,824-byte CASK with
262,144 samples on each of its six full-attention layers, and a 1.76 GB
`oq4.25++` target. Compose staged the canonical 1.79 GB
`Qwen3.5-2B.triattn.oq4.25++.hfq` bundle under `~/.hipfire/models` with exact
base/CASK SHA-256 records. The first CASK finalization caught a real missing
pre-RoPE tap in streamed Qwen session-batch full attention; both dense and MoE
branches now capture before RoPE, and Qwen's adapter accepts intentionally
empty Hessian parts in `--cask-only` recovery mode. The repaired full-corpus
CASK pass and a two-token real-model regression both pass.

The bundle loads its embedded CASK, generates and unloads through the daemon,
and passes all three formal smoke rows (metadata, finite 64-token decode, and
multi-turn reset recall). The dedicated eval CASK battery is now daemon-backed:
it requires exact needle recovery after a prefill larger than the derived
physical KV cap. On the current release daemon, both the explicit loose HFQM
CASK and embedded compose.v2 form processed 8,159 prompt tokens through an
896-slot physical cache and produced the identical 20-token, 92-byte output
hash `fnv64:c399f73c03a184b6`. Storage-form parity therefore passes, but neither
output recovered the committed `twenty-one` needle. The row is rejected under
the frozen long-context gate and remains local; it was not transferred. No
other matrix row was claimed bundled at that checkpoint.

The second product-recipe row, Qwen3.5-4B, has now completed the same frozen
recipe. Its valid 4,612,795,904-byte calibration artifact contains 248
Hessians, 248 imatrices, 262,016 KLDREF positions, and a complete 427/427
logical read ledger. The 200,704-byte HFQM CASK covers all eight full-attention
layers with 262,144 samples per layer. The 3,185,581,419-byte `oq4.25++` target
has SHA-256 `7e37df9312ad1e8f1ff42196f5a1e768b9115bbd2f76238af988495736018154`;
compose recorded the CASK SHA-256
`b975b35cd255ed086a76bc2a08a8e7a82d3b6fae984ce04553a19105f6b1ee27`
and atomically staged the 3,217,141,120-byte canonical bundle with SHA-256
`3ba3c6da3cb86b5140d7bff7ac00deb6a2503e52246ea04194418ca672d59ec7`.

The current-daemon smoke battery passed metadata, finite 64-token decode at
8.8 tok/s, and multi-turn reset recall. The frozen loose and embedded CASK
runs each prefilled 8,159 tokens past the 896-token physical cap and generated
the identical 128-token, 537-byte output hash `fnv64:58f77d8c1455931e`.
Storage-form parity therefore passes, but neither output recovered the
committed needle. Qwen3.5-4B is rejected and remains local; it was not
transferred. No other matrix row is claimed bundled.

The generic induction wrapper now resolves optional DFLASH sources, emits
canonical `oq4+` DFLASH, requests HFQM CASK from the native calibration pass,
keeps large intermediate artifacts outside the staging model directory,
atomically composes the final role bundle, validates exact roles and component
SHA-256 digests, and preserves admission state across resumes. Reuse is
dependency-aware: target/DFLASH reruns invalidate the bundle, target reuse
requires its two-pass recipe/audit fingerprints, and DFLASH reuse requires the
same immutable snapshot/format recipe plus a matching recorded output SHA-256.
Delivery to
`halo:~/.hipfire/models` is implemented as a verified temporary upload and
atomic remote rename, but fails closed unless the manifest is explicitly
`admitted`. Live preflight succeeds for all 11 source rows and records the
resolved immutable snapshots; this is compatibility evidence, not admission.

The per-row production entry point is now:

```bash
python3 scripts/induct_model.py \
  --target /srv/huggingface/<target-cache-root> \
  --model-name <canonical-model-stem> \
  --artifact-root /srv/huggingface/.hipfire-work \
  --model-dir ~/.hipfire/models \
  [--dflash-source /srv/huggingface/<draft-cache-root>]
```

`--transfer --remote halo` is a separate, fail-closed promotion action on the
same manifest. It refuses `pending`/`rejected` rows and verifies the local,
temporary-remote, and final-remote SHA-256 values.

Workflow verification passes the no-GPU checks (including 168 Python tests),
the full tiny-quant tier, the tiny-spec DFLASH tier, direct HIP compilation for
gfx1030/gfx1100/gfx1103/gfx1151/gfx1201, and gfx1103 qt=45/46/47 parity. The
Qwen capture fix's two-file affected selector passes all 35 dense, VL, and MoE
quant rows (including all three `oq4.25++` rows); its state tier matches dense
and VL but still reports the pre-existing Qwen3.5-MoE hash drift. The broader
frozen tiny-state tier also reports the pre-existing Mamba2 drift reproduced on
clean HEAD. These baselines were not rewritten.

## Initial induction matrix

Treat every row as independently resumable. A blank DFLASH source means the
bundle contains CASK/TriAttention but no fabricated or borrowed drafter.

| Target source under `/srv/huggingface` | DFLASH source under `/srv/huggingface` |
|---|---|
| `models--CohereLabs--BLS-Mini-Code-1.0` | none |
| `Ornith-1.0-9B` | `models--z-lab--Qwen3.5-9B-DFlash` |
| `Ornith-1.0-35B` | `models--z-lab--Qwen3.6-35B-A3B-DFlash` |
| `models--google--gemma-4-26B-A4B-it` | `models--z-lab--gemma-4-26B-A4B-it-DFlash` |
| `models--google--gemma-4-31B-it` | `models--z-lab--gemma-4-31B-it-DFlash` |
| `models--Qwen--Qwen3.5-2B` | none |
| `models--Qwen--Qwen3.5-4B` | none |
| `models--Qwen--Qwen3.5-9B` | `models--z-lab--Qwen3.5-9B-DFlash` |
| `models--Qwen--Qwen3.6-35B-A3B` | `models--z-lab--Qwen3.6-35B-A3B-DFlash` |
| `models--Qwen--Qwen3.6-27B` | `models--z-lab--Qwen3.6-27B-DFlash` |
| `models--Qwen--Qwen3.5-122B-A10B` | `models--z-lab--Qwen3.5-122B-A10B-DFlash` |

Source discovery must resolve each cache root to one immutable snapshot and
record that revision before calibration begins. Ornith's reused Qwen DFLASH
pairings are compatibility hypotheses, not presumed-valid matches; the
component compatibility and DFLASH acceptance gates may reject them.

## Child scopes

- [HFQ compose/decompose for DFLASH and TRIA](2026-07-22-hfq-compose-dflash-tria.md)
- [Embedded DFLASH and CASK runtime loading](2026-07-22-embedded-dflash-cask-runtime.md)
- [Native GPU DFLASH Opus Quant](2026-07-22-dflash-opus-gpu-native.md)
- [BLS Mini Code induction](2026-07-22-bls-mini-code-induction.md)
- [Gemma 4 native calibration and CASK](2026-07-22-gemma4-native-calibration-cask.md)

## Current blockers

| Area | Current state | Required outcome |
|---|---|---|
| HFQ composition | implemented and lossless for arch-20 DFLASH, raw TRIA, and namespaced HFQM roles; large inputs use index-only reads and exact range streaming | retain full no-GPU and real-artifact round-trip coverage |
| Runtime component loading | embedded DFLASH/TRIA plus heterogeneous HFQM CASK parsing and Gemma/BLS layered F32 execution implemented | admitted loose-vs-embedded CASK long-context parity |
| DFLASH Opus on GPU | packed qt=45/46/47 and aligned W4A8/W8A8 paths pass correctness, but DP4A remains slower than F16 | tuned native layout plus Atlas/performance admission |
| BLS | arch-25 identity, layered full/SWA resident serving, quantizer fixture, native calibration/CASK, and bundled-CASK fixture parity pass | full-model BF16 parity and product-quality calibration/admission |
| Gemma 4 | dense/MoE native calibration, per-layer CASK production, heterogeneous F32 runtime, and full 31B/26B-A4B minimal calibration audits pass | product-scale center/long-context evidence and frozen quant admission |
| Qwen3.5-2B row | product calibration, `oq4.25++`, full-corpus CASK, canonical bundle, embedded-CASK daemon generation, formal smoke pass, and exact loose/embedded long-context output parity | rejected: current-daemon long-context CASK output did not recover the committed needle; no transfer |
| Qwen3.5-4B row | product calibration, `oq4.25++`, eight-layer full-corpus CASK, canonical bundle, formal smoke pass, and exact loose/embedded long-context output parity | rejected: current-daemon long-context CASK output did not recover the committed needle; no transfer |

## Frozen packaging outcome

The target bundle name follows the repository naming contract:

```text
<Family>-<Size>[...].dflash.triattn.oq4.25++.hfq
```

The final quant token describes the primary model weights. Each embedded
component records its own encoding in the component manifest; the DFLASH
component therefore records `oq4+` without adding a second terminal quant token
to the bundle name.

Independent artifacts use role-before-quant names:

```text
Qwen3.5-9B.dflash.oq4+.hfq
Qwen3.5-9B.triattn.hfq
Qwen3.5-9B.calib.hfq
```

Do not add fallbacks for legacy quant ordering or `op*` spellings.

## Dependency order

1. Land the component manifest, DFLASH namespace mapping, raw-TRIA wrapping,
   and byte-identical split tests.
2. Land embedded component readers and deterministic source precedence.
3. Land packed DFLASH OQ GPU kernels and dispatch while retaining the F16
   expansion path only as an explicit comparison oracle.
4. Land the family-neutral calibration/CASK producer seam while adding the
   Gemma 4 adapter.
5. Bring up BLS identity and BF16 runtime before adding its calibration/CASK
   adapter.
6. Extend `scripts/induct_model.py` only after every producer and consumer
   contract exists. A script must not advertise a bundled output before the
   runtime can consume it.

Packaging and GPU OQ work may proceed in parallel. BLS calibration must wait
for BLS BF16 forward parity. Gemma 4 calibration must preserve the canonical
Gemma 4 plan's frozen admission gates.

## Program invariants

- Keep conversion and package writing in `hipfire-coexistence` or a dedicated
  offline tooling crate. Runtime owns read-only component views, not conversion.
- Keep the backend HIP/ROCm-direct; do not add Vulkan, wgpu, or Python production
  tooling.
- Preserve RDNA2, RDNA3, and RDNA4 portability.
- Do not extract an embedded component to a temporary file during load.
- Explicit user paths override embedded components; invalid embedded components
  fail closed instead of silently falling back to a sibling file.
- Record source revision, producer commit, calibration corpus, quant recipe,
  component hashes, and engine fingerprint in induction manifests.
- Keep experimental paths opt in until their gates pass.

## Staging and delivery contract

- Write intermediate calibration data and resumable state outside the model
  directory; only complete HFQ artifacts are staged in `~/.hipfire/models`.
- Build to a temporary filename and atomically rename to the canonical bundle
  name after structural, digest, and load checks pass.
- Transfer only the verified bundled model to `halo:~/.hipfire/models`; do not
  send loose calibration, DFLASH, or TRIA inputs unless requested separately.
- Verify the remote byte length and strong digest before marking the row
  delivered. A failed or rejected row remains local with its evidence and is
  not copied under a promotion-looking name.

## Program verification

Each child scope owns its unit and targeted gates. The complete workflow also
requires:

1. `./tests/no-gpu-ci.sh` for tooling and protocol changes.
2. `./tests/tiny-affected-gate.sh --require-coverage` for runtime, dispatch,
   quant-format, or kernel changes.
3. Lossless `compose -> inspect -> decompose` round trips for base, DFLASH, and
   TRIA inputs.
4. Separate and bundled load parity on the same prompt and engine fingerprint.
5. Native-packed versus F16-expanded DFLASH logit and decoded-output evidence.
6. `hipfire-eval` KLD/PPL against an accepted high-precision reference.
7. DFLASH tau/acceptance and decoded-output checks.
8. Long-context CASK recall and combined DFLASH+CASK recall.
9. Kernel Atlas AR and DFLASH rows on every promoted architecture.

## Definition of done

The program is complete only when an induction run produces one canonically
named bundle, `hipfire inspect` reports both embedded roles and their hashes,
the daemon loads both roles without loose files, decompose reproduces every
source component byte-for-byte, and all frozen admission gates have explicit
pass or reject outcomes.
