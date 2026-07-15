# Gemma 4 Phase 0 truth freeze

Date: 2026-07-15. Status: passed.

## Frozen evidence

- Eight locally available standard snapshots resolve to the revisions in the
  canonical plan. Header-only extraction recorded 1,013 tensors for each
  26B-A4B checkpoint, 1,188 for each 31B checkpoint, 2,011 for each E2B
  checkpoint, and 2,130 for each E4B checkpoint.
- The official 12B unified config is pinned to Hub revision
  `1dd69cd087619018c29fbfe2c30c3cd3530479fb`; only its config was fetched
  because the full checkpoint is not local.
- Config assertions distinguish layer count, hidden width, context/SWA size,
  PLE, KV-sharing tail, K=V policy, double-wide tail, and MoE topology.
- Manifest assertions distinguish PLE tables and per-layer weights, dense
  variants, stacked `[expert, 2 * expert_intermediate, hidden]` GeGLU weights,
  and global K=V layers whose source manifest omits V projection tensors.
- Four instruction checkpoints were rendered with their pinned official Jinja
  templates. The committed fixtures contain exact UTF-8 output and token IDs
  for plain, system, thinking on/off, multi-turn, tool declaration, tool call,
  tool response, and assistant continuation cases.
- `bf16-thresholds.json` froze full-model hidden/logit/generation admission
  limits before any Hipfire Gemma 4 whole-model result was observed.

Reproduction gate:

```text
$ python3 benchmarks/gemma4/extract_fixtures.py --check
Gemma 4 fixtures reproduce exactly from /srv/huggingface
```

## Reuse and cleanup ledger

- Existing primitive reused: Transformers' official chat-template renderer and
  fast tokenizer; safetensors header reader; existing offline BF16 capture
  conventions from `scripts/dump_hf_hidden_states.py`.
- Duplicate removed or retained: both released official template families are
  retained because their bytes differ; one fixture generator owns all standard
  variants instead of per-model scripts.
- Generic seam added or changed: none. Phase 0 intentionally changes no runtime
  execution path.
- Generic abstraction consumers: not applicable; no generic abstraction was
  added.
- Stale assumption removed: the June roster no longer labels E2B/E4B as MoE,
  Gemma 4 as Gemma 3 plus MoE, or bring-up as cheap. Pre-release JSON parser
  tests are ignored and explicitly excluded from acceptance evidence.
- Oracle retained: `capture_transformers_reference.py` remains the independent
  BF16 path until the later recorded reference/lowered/fleet parity gates pass.
