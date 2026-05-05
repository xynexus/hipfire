# Calibration corpora

Two distinct corpus types live here:

## TQV calibration tables

`tqv/*.json` are small committed calibration artifacts for the
TurboQuant V-cache table family. They are produced by
`scripts/tqv_fit_tables.py` from captured post-normalization,
post-FWHT/sign V-cache samples. Keep these tables in git when the
runtime embeds or references them; they are calibration data, not local
scratch.

Quick synthetic smoke for the fitter:

```
scripts/tqv_fit_tables.py --synthetic 128,256 --synthetic-samples 4096 \
  --max-iter 20 --bits 2,3,4 --out /tmp/tqv-synth.json
```

## Optimization-bench corpora (NOT sidecar-quality)

`calib-1m.txt` and `calib-5m.txt` are wikitext-103 slices used to measure
**timing speedups** of the calibration pipeline across commits. They are
stable, byte-reproducible inputs for cross-session A/B (per the τ
prompt-shape rule in `CLAUDE.md` — bench inputs must be md5-pinned).

**Do NOT use these for shipping sidecars.** Per
`feedback_wikitext_triattn_sidecar_garbage.md` (2026-04-19), wikitext
calibration produces measurably worse downstream behavior than
representative-distribution corpora; only Hermes/Aureth-trained sidecars
ship to users.

| file | source | bytes | est tokens | md5 |
|---|---|---|---|---|
| `calib-1m.txt` | wikitext-103-raw-v1 test+train shard 0 prefix | 4,798,009 | ~1.2M | `c1879341cb2d4bcf06ead9d1c02ef5fa` |
| `calib-5m.txt` | wikitext-103-raw-v1 train shard 0 prefix | 19,996,814 | ~5.0M | `5dc7dc29676eb591869378b3ddc17815` |

## Sidecar-quality corpora (built on demand)

These are NOT committed (too large; deterministic via fetch script).
Run the corresponding script during a calibration session.

### `hermes-corpus.txt` (~1.1 GB / ~280M tokens)

ChatML-flattened conversations from `lambda/hermes-agent-reasoning-traces`,
configs `kimi` + `glm-5.1` (14,701 traces). Used to calibrate
target-model sidecars for Carnice / Qwen3.6-27B / dense Qwen models.
Generate with:

```
bash scripts/fetch_hermes_corpus.sh benchmarks/calib/hermes-corpus.txt
```

### `aureth-corpus.txt` (~127 MB / ~32M tokens)

Prompt+chosen pairs from `OusiaResearch/Aureth-Corpus-Hermes4.3-Generated`.
Used to calibrate Qwen3.5-A3B / Qwen3.6-A3B sidecars per
`project_carnice_hermes_niche.md`. Generate with:

```
hf download --repo-type dataset OusiaResearch/Aureth-Corpus-Hermes4.3-Generated \
  compiled_corpus.jsonl --local-dir benchmarks/calib/aureth-raw
python3 scripts/aureth_to_corpus.py \
  benchmarks/calib/aureth-raw/compiled_corpus.jsonl \
  benchmarks/calib/aureth-corpus.txt
```
