# Configuration

Two layers:

1. **Global config** at `~/.hipfire/config.json` — applies to every
   model unless overlaid.
2. **Per-model overlay** at `~/.hipfire/per_model_config.json` — sparse
   keys overriding global for a specific tag.

Edit interactively with `hipfire config` (global) or `hipfire config
<tag>` (overlay). Or set non-interactively: `hipfire config set <key>
<value>`.

## Generation

| Key | Default | Range / values | Notes |
|---|---|---|---|
| `temperature` | 0.30 | 0.0–2.0 | 0.0 = greedy. |
| `top_p` | 0.80 | 0.0–1.0 | Nucleus sampling. |
| `repeat_penalty` | 1.05 | 1.0–3.0 | Default kept conservative — 1.3 causes MQ4 gibberish at low temp. |
| `max_tokens` | 512 | 1–131072 | Per-request cap. |
| `max_seq` | 32768 | 512–524288 | KV cache physical capacity. |
| `thinking` | on | on / off | Whether to keep `<think>...</think>` reasoning blocks. |
| `max_think_tokens` | 0 | 0–32768 | 0 = no cap. Caps tokens emitted before `</think>` closes. |

## KV cache

| Key | Default | Values |
|---|---|---|
| `kv_cache` | auto (per arch) | auto / q8 / asym4 / asym3 / asym2 / turbo / turbo4 / turbo3 / turbo2 |

Per-arch defaults: gfx1100 → asym3, gfx1030 → asym3, gfx1010/1013 →
asym2. asym3 is rotated K (Lloyd-Max) + Q8 V — the multi-turn quality
sweet spot. Use `q8` for byte-exact reference behavior at higher VRAM
cost.

## Speculative decode (DFlash)

| Key | Default | Values | Notes |
|---|---|---|---|
| `dflash_mode` | off | on / off / auto | `auto` enables DFlash on dense Qwen 3.5+ targets and skips configs known to lose. |
| `dflash_adaptive_b` | true | true / false | Adaptive draft block size. |
| `dflash_ngram_block` | auto | true / false / auto | n-gram cache prefilling. |

DFlash speedup is genre-conditional: large on code, modest on
instruct, can be a net loss on prose. See [BENCHMARKS.md](BENCHMARKS.md)
for measured speedups. Per-model override is the most common knob:
`hipfire config qwen3.5:9b set dflash_mode off` if your workload is
mostly long-form prose.

## Attention

| Key | Default | Values |
|---|---|---|
| `flash_mode` | auto | auto / always / never |

`auto` enables FlashAttention when the seq len passes the FA-vs-vanilla
crossover for the current arch. `never` is the byte-exact reference;
`always` forces FA even on short prompts.

## CASK (m-folding speculative decode)

| Key | Default | Notes |
|---|---|---|
| `cask` | false | Enable CASK m-folding atop DFlash. |
| `cask_sidecar` | "" | Path to CASK sidecar tape file. Empty = disabled. |
| `cask_budget` | 1024 | 64–65536. |
| `cask_beta` | 0 | 0–65536. |
| `cask_fold_m` | 1 | 1–16. |
| `cask_core_frac` | 0.25 | 0.0–1.0. |

CASK is experimental. Leave defaults unless you've read the
DFLASH/CASK code.

## Prompt processing

| Key | Default | Values | Notes |
|---|---|---|---|
| `prompt_normalize` | true | true / false | Collapse `\n{3,}` → `\n\n` at engine entry. +24% τ on PEP-8-style code prompts; default ON since 2026-04-26. Opt out only when raw whitespace patterns are semantically load-bearing. |

## Server

| Key | Default | Range |
|---|---|---|
| `port` | 11435 | 1–65535 |
| `idle_timeout` | 300 | 0–86400 (seconds) |
| `default_model` | "" (none) | tag or path |

`idle_timeout` evicts the loaded model from VRAM after that many
seconds of no requests; the next request reloads with a 2–5 s cold
start. Set to 0 to keep weights resident forever (useful when you have
spare VRAM and want zero-latency requests).

`default_model` is what `hipfire serve` pre-warms on startup.

## Per-model overlay

```bash
hipfire config qwen3.5:9b
```

Opens the same TUI but writes to the overlay file. Rows show
`(inherited)` if the key matches global and `(overridden)` if it
diverges. A rendered overlay JSON looks like:

```json
{
  "qwen3.5:9b": {
    "dflash_mode": "off",
    "kv_cache": "q8"
  }
}
```

Only keys explicitly set are written; everything else inherits global.
Delete a row's override with the TUI's `d` key.

## One-shot env overrides

For testing without touching the config file:

```
HIPFIRE_KV_MODE=asym3
HIPFIRE_ATTN_FLASH=auto
HIPFIRE_NORMALIZE_PROMPT=0          # opt out of \n{3,} collapse
HIPFIRE_LOCAL=1                     # skip the running daemon
HIPFIRE_HIPCC_EXTRA_FLAGS="-mcumode"
HIPFIRE_PROMPT_TOKEN_HEAT=1         # dump per-position BPE merge ranks
HIPFIRE_PROMPT_HEAT_JSON=1          # the same, machine-readable
HIPFIRE_GRAPH=1                     # hipGraph capture (debug; AR-only, may degrade quality on large models)
```
