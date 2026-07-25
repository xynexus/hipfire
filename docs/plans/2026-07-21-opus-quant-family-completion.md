# Opus Quant Family Completion Plan

Date: 2026-07-21

## Goal

Make Opus Quant a complete, admitted quantization path across every hipfire
model family that has a servable runtime, and make unsupported families fail
clearly until they have a runtime path.

Completion means all of the following are true for each applicable family:

- `oq4`, `oq4+`, `oq4++`, `oq8`, `oq8+`, `oq8++`, and the default mixed
  `oq4.25++` producer path either work end-to-end or are explicitly blocked
  with a documented reason.
- The quantizer emits only GPU-servable OQ tensor layouts for GPU artifacts,
  with deliberate fallbacks for ragged shapes.
- The loader maps every emitted OQ quant type to a runtime dtype without
  family-local duplication.
- Decode and prefill are coherent on RDNA3 and RDNA3.5 first, then checked for
  RDNA2/RDNA4 portability where kernels are meant to be portable.
- `tiny_quant`, fixture golden, and model eval evidence exist before a support
  matrix row is promoted from `partial` to `full`.

## Current State

The catalog already registers `oq4`, `oq4+`, `oq8`, and `oq8+` as opt-in formats
in `docs/model-support.toml`. The default quantizer format is `oq4.25++`, but
mixed-precision OQ is not yet covered by `tiny_quant`.

Current strongest coverage is Gemma3 text:

- `gemma3` tiny-quant candidates include `oq4` and `oq8`.
- `gemma3` calibrated tiny-quant includes `oq4++`.
- No family currently exercises mixed OQ, where one artifact contains both OQ4
  and OQ8 tensors.

Several families have loader/runtime paths but are still marked `partial` for
OQ prefill because quality or eval-battery admission is pending:

- `qwen3.5` dense: all OQ prefill variants are gated.
- `qwen3.5` MoE: routed-expert OQ support exists, but automatic family coverage
  is still MQ/QTIP-focused.
- `lfm2-moe`: OQ4/OQ8 paths have smoke/parity evidence only.
- `qwen2`: OQ4/OQ8 load and decode coherently on Qwen2-0.5B, eval admission
  pending; ragged OQ8 requires GPU-compatible fallback.
- `llama`: OQ8 coherent on Llama-3.2-1B, quality admission pending; OQ4 requires
  `K % 256` for quantized linears.
- `nemotron_h`: OQ8 coherent on Nemotron-3-Nano-4B, quality admission pending.

Families without OQ coverage in the automatic tiny matrix today include
`gemma3-vl`, `gemma4`, `deepseek4`, `minimax`, `mamba2`, `dots-ocr`,
`embeddinggemma`, and `zaya`. Diffusion has its own matrix: `flux2` is the only
fully wired text-to-image path, while `krea2` and `qwen-image` remain partial.

## Phase 0 - Freeze Support Semantics

1. Define support levels in `docs/model-support.toml` comments:
   - `none`: no servable runtime path.
   - `partial`: load/decode or smoke works, but prefill, quality, or coverage is
     incomplete.
   - `full`: loader, decode, prefill, tiny/golden, and eval admission are all
     complete for the family and quant.
2. Add an OQ-specific support checklist next to each OQ `[[gate]]` note:
   `producer`, `loader`, `decode`, `prefill`, `tiny`, `golden`, `eval`,
   `artifact`.
3. Regenerate `crates/hipfire-model/src/model_support_generated.rs` after each
   matrix edit with:

```bash
cargo run -p hipfire-cli -- gen-model-support
```

4. Keep OQ rows `partial` until evidence exists. Do not promote based only on a
   successful loader smoke.

## Phase 1 - Shared OQ Loader and Format Contract

1. Centralize OQ quant-type-to-dtype mapping so arch crates do not hand-roll
   OQ decode or dtype lookup.
2. Ensure the shared path covers:
   - canonical OQ4, arch-packed OQ4
   - canonical OQ8
   - OQ plus compact / mixed OQ
   - row-padded OQ8 as XDNA-only unless explicitly converted/fallbacked
3. Add unit tests for byte-length and dtype mapping:
   - OQ4 canonical and arch-packed
   - OQ8 canonical
   - OQ plus compact
   - row-padded OQ8 rejection on GPU loaders
4. Remove family-local OQ fallback code once shared coverage is in place.

Acceptance:

- `cargo test -p hipfire-runtime quant_catalog_matches_derived_gemv_routes`
  passes.
- No family loader panics on valid emitted OQ tensor types.
- GPU-incompatible OQ8 row-padded artifacts fail with an actionable error or
  are quantized with GPU-compatible fallback.

## Phase 2 - Tiny-Quant Coverage Expansion

Add OQ cells to `crates/hipfire-eval/src/executor_tinyquant.rs` by family, in
this order:

1. `gemma3`: add mixed `oq4.25++` first, because the loader already handles OQ4
   and OQ8. This closes the current pure-width-only gap.
2. `qwen3_5`: add `oq4`, `oq8`, and mixed `oq4.25++` dense cells.
3. `qwen3_5_moe`: add `oq4`, `oq8`, and mixed `oq4.25++` MoE cells, including
   routed expert coverage.
4. `qwen2`: add `oq4`, `oq8`, and mixed OQ with the ragged OQ8 fallback path
   exercised.
5. `llama`: add `oq8` first, then `oq4` only for a fixture whose linears satisfy
   the OQ4 alignment contract or preserve ragged tensors as BF16/F16.
6. `nemotron_h`: add `oq8`, then `oq4`, then calibrated/mixed variants after
   the Mamba activation calibration policy is stable.
7. `lfm2_moe`: add `oq4`, `oq8`, and mixed OQ after confirming tiered assignment
   is either generic or intentionally LFM2-specific.
8. `gemma3_vl`, `dots_ocr`: add text-side OQ first; keep vision tensors on a
   proven format until vision OQ loader support is explicitly admitted.
9. `gemma4`: add OQ only after Gemma4 forward support is promoted beyond
   matrix-level `none`.
10. `deepseek4`, `minimax`, `mamba2`: add only after family-specific OQ loader
    and prefill routes exist.
11. `embeddinggemma`: track separately because its primary OQ path is XDNA/NPU
    resident Opus, not the normal autoregressive GPU path.
12. `zaya`: keep blocked until the servable forward path exists.

Baseline procedure:

```bash
HIPFIRE_TINYQUANT_RECORD=1 ./tests/tiny-quant-gate.sh
./tests/fixture-golden-gate.sh
```

Record both gfx1151 and gfx1103 baselines before promotion. Add gfx1201 and
gfx10/RDNA2 evidence for any route claimed portable across those families.

## Phase 3 - Decode and Prefill Admission

For each family/quant pair:

1. Prove single-token decode:
   - finite logits
   - stable greedy answer on a short prompt
   - no dtype fallback unless documented
2. Prove prefill:
   - batched prefill matches per-token reference within family threshold
   - long-prefill prompt stays coherent
   - OQ4 W4A4 paths remain opt-in until end-to-end divergence is resolved
3. Prove mixed OQ dispatch:
   - artifact contains both OQ4 and OQ8 tensors
   - loader dispatches per tensor, not by global quant assumption
   - prefill and decode both cover the mixed tensor set
4. Prove calibrated plus variants:
   - `oq4+` / `oq8+`: imatrix or activation-aware scale provenance exists
   - `oq4++` / `oq8++`: Hessian/LDLQ sidecar is audited
   - no `++` artifact is promoted without calibration audit output

Use the existing coherence gates for prompt-level checks, but move final model
admission into `hipfire-eval` batteries or suites.

## Phase 4 - Family Work Items

### Gemma3

- Add `oq4.25++` tiny cell.
- Verify mixed loader dispatch with a real per-tensor OQ4/OQ8 blend.
- Promote to full only after mixed OQ and calibrated OQ4++ baselines pass.

### Qwen3.5 Dense

- Add dense `oq4`, `oq8`, and `oq4.25++` tiny cells.
- Resolve or retain the OQ4 W4A4 batched-prefill gate based on end-to-end logit
  parity.
- Add eval batteries for 0.8B and 9B before promoting `oq4+`/`oq8+`.

### Qwen3.5 MoE

- Add OQ routed-expert tiny cells.
- Validate uniform OQ experts and mixed full-precision fallback experts.
- Require routed expert coverage telemetry before `oq4++` or mixed OQ
  promotion.

### Qwen2

- Exercise ragged hidden sizes explicitly.
- Keep row-padded OQ8 out of GPU artifacts unless the GPU loader learns that
  layout.
- Promote only after Qwen2-0.5B and at least one non-ragged fixture pass.

### LLaMA / Mistral

- Promote OQ8 first.
- For OQ4, document tensor preservation rules when `K` is not 256-aligned.
- Add a tiny fixture that catches accidental OQ4 quantization of unsupported
  ragged linears.

### Nemotron-H / Mamba2

- Keep Mamba2 pure recurrent coverage separate from Nemotron-H hybrid coverage.
- Validate AWQ alpha policy for Mamba activations before `oq4+`.
- Add prefill GEMM coverage for OQ8 and OQ4.

### LFM2 MoE

- Decide whether mixed tier assignment is LFM2-only or generic.
- If LFM2-only, add the first tiered mixed OQ tiny cell here.
- Promote OQ8 before OQ4 if OQ4 W4A4 quality remains tighter.

### Vision Families

- For Gemma3-VL and dots-ocr, first admit text-side OQ while keeping vision
  towers on q8f16/hfq4.
- Add vision OQ only after image preprocessing, projector, and splice paths have
  golden coverage.

### Gemma4

- Do not claim OQ full support while the support matrix still marks Gemma4
  runtime capabilities as `none`.
- Once Gemma4 forward admission lands, add dense, PLE, and MoE OQ cells
  separately.

### DeepSeek4 and MiniMax

- Add OQ only after family-specific MoE and compressed/auxiliary tensor roles
  have explicit dtype policies.
- Keep source-precision tiny cells as the control until OQ loader routes exist.

### EmbeddingGemma / NPU Opus

- Treat NPU resident Opus as a separate admission path.
- Validate `oq8+` buckets and Dense heads through the XDNA executor.
- Do not use autoregressive prefill gates as a substitute for embedding
  encoder admission.

### Diffusion

- Keep diffusion OQ status in the diffusion matrix, not the autoregressive
  matrix.
- For Flux2, add OQ quality batteries for denoise outputs, sampler stability,
  and VAE decode.
- For Krea2 and Qwen-Image, keep OQ listed as ingest/denoise capability only
  until the t2i loop is wired.

## Phase 5 - Artifact Promotion

For each promoted artifact:

1. Use canonical names, for example:
   - `Qwen3.5-9B--oq4++.hfq`
   - `Gemma-3-4B-it--oq4.25++.hfq`
   - `LFM2.5-8B-A1B--oq8+.hfq`
2. Store calibration sidecars under `/srv/hipfire/{calib,imatrix,hessians}`.
3. Store models under `/srv/hipfire/models`.
4. Record provenance:
   - source checkpoint
   - quantizer command
   - calibration corpus and sidecars
   - arch id
   - git commit
   - eval battery results
5. Update docs and registry entries only after the artifact passes admission.

## Phase 6 - Required Gates

Run no-GPU checks for structural changes:

```bash
./tests/no-gpu-ci.sh
```

Run GPU front-tier gates for runtime or quantizer changes:

```bash
./tests/tiny-affected-gate.sh --require-coverage
./tests/tiny-quant-gate.sh
./tests/fixture-golden-gate.sh
```

Run family-specific eval batteries before matrix promotion. Required minimum:

- tiny KLD baseline
- fixture golden baseline
- short decode coherence
- long prefill coherence where prefill is claimed
- held-out KLD/perplexity or task battery for product artifacts

## Promotion Rule

An OQ family row may move from `partial` to `full` only when:

1. every emitted OQ quant type loads through shared dtype mapping;
2. decode and prefill are both covered, or unsupported prefill is explicitly
   excluded from the family capability;
3. pure OQ4/OQ8 and mixed OQ are represented in automated tests;
4. calibrated plus variants have audited calibration provenance;
5. quality evidence is recorded in `hipfire-eval`;
6. the support matrix, docs, and artifact registry agree.

Anything short of that remains `partial`.
