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
