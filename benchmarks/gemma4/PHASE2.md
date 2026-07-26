# Gemma 4 Phase 2 shared transformer loader

Date: 2026-07-15. Status: passed.

## Exit-gate evidence

- `hipfire-runtime::transformer_loader` now owns exact logical-shape
  validation, required and optional lookup, F16/F32/BF16 direct widening,
  embedding upload, tied/untied head selection, linear quant upload/repacking,
  BF16 buffer tagging, and AWQ sidecar attachment.
- Family crates still own tensor-name construction, prefix rules, which tensors
  are required or optional, layer structure, paging, PLE, and MoE assembly.
- Gemma 3 migrated from its private copies to the shared loader. The Gemma 3-VL
  and EmbeddingGemma consumers still compile and pass their full unit suites.
- `hipfire-arch-gemma4` is the second consumer. Its Phase 2 core loader uses the
  released `model.language_model.*` names for the embedding and final direct
  norm, then delegates tied/untied output loading to the shared module.
- `rg` finds no `load_weight_tensor`, `load_norm`, `load_embed`, or
  `load_lm_head` implementation in the Gemma 4 crate and none of the removed
  private loader implementations in Gemma 3.

Targeted gates:

```text
$ cargo test -p hipfire-runtime transformer_loader --lib
test result: ok. 5 passed; 0 failed

$ cargo test -p hipfire-arch-gemma3
test result: ok. 7 passed; 0 failed

$ cargo test -p hipfire-arch-gemma4
test result: ok. 1 passed; 0 failed

$ cargo test -p hipfire-arch-gemma3-vl
test result: ok. 7 passed; 0 failed

$ cargo test -p hipfire-arch-embeddinggemma
test result: ok. 16 passed; 0 failed

$ cargo check -p hipfire-serving-core
Finished `dev` profile
```

The shared loader tests separately cover missing, wrong-rank, wrong-shape,
optional absence/present validation, tied versus separate head selection, and
direct BF16 norm widening with no offset.

The committed Gemma 3 tiny-quant regression battery passed all six selected
rows on gfx1103 after the migration: collect, Q8F16, HFQ4, OQ4, OQ8, and
calibrated OQ4+. Observed KLDs remained within their committed baselines and the
battery reported zero findings.

A separately locked BF16 tiny check generated a fresh
`Gemma-3-Tiny.bf16.hfq`, loaded it twice through the migrated path, and compared
the two forwards:

```text
arch: gemma3
mean_kld: 0.00000000
max_kld: 0.00000000
n_scored: 16
finite: true
```

## Reuse and cleanup ledger

- Existing primitives reused: `HfqFile::find_tensor_info`/`tensor_data_vec`,
  canonical quant-type mapping, OQ repackers, AWQ sidecar loader,
  `WeightTensor`, `EmbeddingFormat`, and GPU upload methods.
- Duplicate removed or retained: Gemma 3's private norm, embedding, head, and
  weight loaders were removed. Qwen2 remains on its existing loader because the
  phase contract requires Gemma 3 migration and Gemma 4 as the second consumer;
  widening the migration to Qwen2 would add unrequired bias/path risk.
- Generic seam added or changed: one small `TransformerLoader` plus pure
  validation/decoding helpers. It does not absorb prefill composition,
  Qwen3.5 slab/pager orchestration, or family layer policy.
- Generic abstraction consumers: Gemma 3 and Gemma 4 directly; Gemma 3-VL and
  EmbeddingGemma transitively through the migrated Gemma 3 loader.
- Stale assumption removed: loaders no longer infer validity from element count
  alone; required tensors must match the exact logical rank and shape before
  upload.
- Oracle retained: Gemma 3's committed tiny-quant baselines and independent
  fresh BF16 self-comparison remain available as regression anchors for later
  Gemma 4 loader/forward work.
