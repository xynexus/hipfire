# Calibration corpora

Two distinct corpus types live here:

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

### Matched layer-prefetch timing

`prefetch-abba.sh` runs fresh two-layer native-calibration trials in an
order-balanced off/on/on/off sequence. It requires exact layer-part hashes,
run fingerprints, and engine identities across trials before reporting timing,
and records the corpus MD5/SHA256 plus the executable SHA256 in `evidence.json`.
The result proves timing and mechanism parity only; it is not calibration-quality
evidence. The native calibrator coordinates each GPU trial with the shared lock.

```sh
benchmarks/calib/prefetch-abba.sh \
  --model /srv/huggingface/models--Qwen--Qwen3.5-0.8B \
  --corpus benchmarks/calib/calib-1m.txt \
  --artifact-stem Qwen3.5-0.8B \
  --output-dir ~/.hipfire/experiments/calibration-prefetch/Qwen3.5-0.8B
```

### Cross-architecture raw grouped channel

`raw-grouped-channel.sh` builds and lock-runs the F16/BF16 routed-expert channel
test, requires all eight portable CPU-oracle cases, and writes a hashed
`evidence.json` row. On gfx1151 it also runs the WMMA and compact indexed paths;
on RDNA2, other RDNA3 devices, RDNA4, and CDNA it exercises the portable kernel
instead of silently skipping the host.

```sh
benchmarks/calib/raw-grouped-channel.sh \
  --output-dir ~/.hipfire/experiments/raw-grouped-channel/$(hostname)-$(date +%Y%m%d-%H%M%S)
```

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
mkdir -p ~/.hipfire/corpora/aureth/aureth-raw
hf download --repo-type dataset OusiaResearch/Aureth-Corpus-Hermes4.3-Generated \
  compiled_corpus.jsonl --local-dir ~/.hipfire/corpora/aureth/aureth-raw
python3 scripts/adhoc/aureth_to_corpus.py \
  ~/.hipfire/corpora/aureth/aureth-raw/compiled_corpus.jsonl \
  ~/.hipfire/corpora/aureth/aureth-corpus.txt
```
