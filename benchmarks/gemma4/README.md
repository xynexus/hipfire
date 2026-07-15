# Gemma 4 reference fixtures and oracle

This directory owns the offline Phase-0 truth freeze for
`docs/plans/2026-07-15-gemma4-support.md`. Nothing here is imported by an
inference binary, and no committed fixture contains model weights.

Reproduce every config, safetensors-header manifest, official Jinja render, and
token-id fixture from the pinned cache snapshots:

```bash
python3 benchmarks/gemma4/extract_fixtures.py --check
```

Regenerate intentionally with the same pinned inputs:

```bash
python3 benchmarks/gemma4/extract_fixtures.py
```

The 12B unified config is pinned to Hub revision
`1dd69cd087619018c29fbfe2c30c3cd3530479fb` because that official checkpoint is
not present in `/srv/huggingface`. The extractor fetches only `config.json` for
that variant. Real-model 12B gates remain unavailable until its full snapshot is
local.

`capture_transformers_reference.py` is the retained BF16 oracle. It accepts a
fully rendered prompt so prompt rendering and model math can be compared
independently. Example:

```bash
python3 benchmarks/gemma4/capture_transformers_reference.py \
  --model /path/to/pinned/snapshot \
  --prompt '<bos><|turn>user\nHello<turn|>\n<|turn>model\n' \
  --layers 0,5,11,17,59 \
  --output benchmarks/results/gemma4-31B-it-bf16-reference
```

The output records input IDs, selected hidden states, final logits, greedy
generation IDs, software versions, prompt hash, and source revision.

For long exact-token boundary prompts,
`capture_transformers_streaming_reference.py` runs the same upstream Gemma 4
BF16 modules and source tensors one decoder layer at a time. Its streamed path
must first pass `compare_transformers_oracles.py` against a resident capture;
the retained base-short calibration is bit-exact at all 60 captured layers and
final logits.

Comparison contracts:

- `oq8-thresholds.json` is the revised broad OQ8 functional baseline, frozen
  from the first valid exact-prompt OQ8 measurement by explicit plan revision;
- `oq8pp-thresholds.json` is the final narrower OQ8++ promotion gate and retains
  the original strict limits unchanged;
- `bf16-thresholds.json` preserves the original pre-observation BF16 candidate
  contract as historical evidence.

Select the contract explicitly when comparing a capture:

```bash
python3 benchmarks/gemma4/compare_bf16_captures.py \
  --oracle /path/to/pinned-oracle \
  --hipfire /path/to/hipfire-capture \
  --thresholds benchmarks/gemma4/oq8-thresholds.json
```

The complete Phase 5 OQ8 matrix is frozen in
`phase5-oq8-capture-plan.json`. It uses committed exact-token fixtures for the
short prompt and the 1023/1024/1025-token SWA boundary cases, so both sides see
identical IDs independently of prompt-file newline handling. Run it under the
shared GPU lock:

```bash
hipfire lock run gemma4-phase5-oq8 -- \
  python3 benchmarks/gemma4/run_full_model_admission.py \
    --oracle-model /path/to/pinned/google/gemma-4-31B/snapshot \
    --candidate-model ~/.hipfire/models/Gemma-4-31B.oq8.hfq \
    --output ~/.hipfire/evidence/gemma4/phase5-oq8-admission
```

`--case CASE_ID` runs a selected case without discarding prior case records in
the output manifest. The base-short case also sets the candidate capture's
lifecycle mode, which requires an exact second request and exact unload/reload
rerun. The canonical candidate example is `gemma4_capture`; the historical
`bf16_capture` name remains available only so old evidence commands reproduce.
The plan selects the resident oracle for base-short and the validated streamed
oracle for the one-token SWA boundary cases. Stop at the first frozen-gate
failure; do not run later cases or revise thresholds after observing it.
