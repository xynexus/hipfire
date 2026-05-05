# CLI reference

Every subcommand of the `hipfire` wrapper. Run `hipfire <cmd> --help` for
flag-level detail; this page is the index.

## Model lifecycle

| Command | Purpose |
|---|---|
| `hipfire pull <tag>` | Download a model from HuggingFace into `~/.local/share/hipfire/models/`. |
| `hipfire list [-r]` | Show local models. `-r` adds remotely-available tags from the curated registry. |
| `hipfire ps` | Show running daemons, in-flight quantize jobs, and HuggingFace upload tasks. |
| `hipfire rm <tag>` | Delete a local model file. |

## Inference

| Command | Purpose |
|---|---|
| `hipfire run <tag\|path> [prompt...]` | Generate. Auto-pulls if missing. Routes through the running `serve` daemon if one is up; otherwise spawns a one-shot daemon. |
| `hipfire serve [port] [-d]` | Start the OpenAI-compatible HTTP server. `-d` detaches into the background and writes a pid file. Default port `11435`. |
| `hipfire stop` | Graceful shutdown of the background daemon. |
| `hipfire bench <tag>` | Measure prefill + decode tok/s on a fixed prompt set. |
| `hipfire metrics <tag>` | Build a KV-cache quality/perf dashboard against an fp32 baseline. |

`hipfire run` accepts either a registry tag (`qwen3.5:9b`) or a literal
file path (`./my.mq4`). For a prompt with shell-special characters,
quote it: `hipfire run qwen3.5:9b "What's 2+2?"`.

Thinking controls are budget-based, not prompt-injection based:
`--no-think` caps thinking to one token, `--think` allows uncapped thinking,
and `--max-think-tokens N` sets a per-run reasoning budget.

## Configuration

| Command | Purpose |
|---|---|
| `hipfire config` | Interactive TUI for global config (`~/.config/hipfire/config.json`). |
| `hipfire config <tag>` | Per-model overlay (`~/.config/hipfire/per_model_config.json`). Rows show `(inherited)` vs `(overridden)`. |
| `hipfire config set <key> <val>` | Non-interactive set. |
| `hipfire config view` | Print effective config + all overlays. |

Full key list and tradeoffs in [CONFIG.md](CONFIG.md).

## Quantization

| Command | Purpose |
|---|---|
| `hipfire quantize <hf-id\|local-dir\|file.gguf>` | CPU-side quantize from safetensors or GGUF to MQ4 / MQ6 / HF4 / HF6. Optional `--install` puts the result in `~/.local/share/hipfire/models/` and `--register <tag>` adds an alias. |

The full quantize how-to (formats, when to pick which, GGUF caveats) is
in [QUANTIZE.md](QUANTIZE.md).

## Diagnostics

| Command | Purpose |
|---|---|
| `hipfire diag` | GPU arch, VRAM, HIP version, ROCm version, kernel blob hashes, model directory. First place to check if anything misbehaves. |
| `hipfire update` | `git pull` + rebuild + refresh kernel blobs. Use when upstream pushes a fix. |

## Metrics

`hipfire metrics <model>` runs the quantization gate over selected KV
cache modes and writes a report under `benchmarks/results/` by default.
It is intended for comparing K/V cache quality and speed while keeping
the prompt set and output artifacts together.

Common forms:

```bash
hipfire metrics qwen3.5:2b --skip-build
hipfire metrics qwen3.5:2b --modes fp32,q8,asym4_tqv4,asym4_tqv3
hipfire metrics qwen3.5:2b --modes-k fp32,asym4 --modes-v fp32,q8,tqv4,tqv3,tqv2,tqv1
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--modes <csv>` | Direct runtime modes. |
| `--modes-k <csv>` / `--modes-v <csv>` | Expand supported K/V mode pairs into runtime modes. |
| `--runs <n>` | Perf runs per split. |
| `--full` | Larger prompt/context coverage. |
| `--coherence` | Run the coherence gate per mode. |
| `--strict` | Promote dashboard warnings to hard failures. |
| `--skip-build` | Reuse existing release examples. |

## Where files live

- Models: `~/.local/share/hipfire/models/`
- Config: `~/.config/hipfire/config.json`
- Per-model overlay: `~/.config/hipfire/per_model_config.json`
- Local model aliases: `~/.config/hipfire/models.json`
- Program files and pre-compiled kernels: `~/.local/lib/hipfire/`
- Daemon log: `~/.local/state/hipfire/serve.log`
- Daemon pid file: `~/.local/state/hipfire/serve.pid`

## Environment overrides

Single-invocation overrides bypass the config file:

| Variable | Effect |
|---|---|
| `HIPFIRE_KV_MODE=asym3\|q8\|asym4\|asym2\|asym4_tqv4\|asym4_tqv3\|asym4_tqv2\|asym4_tqv1` | Override KV cache layout. |
| `HIPFIRE_ATTN_FLASH=auto\|always\|never` | Force or disable FlashAttention. |
| `HIPFIRE_NORMALIZE_PROMPT=0` | Opt out of `\n{3,}` → `\n\n` prompt collapse (default ON). |
| `HIPFIRE_LOCAL=1` | `hipfire run` skips the HTTP daemon and spawns a fresh one-shot. |
| `HIPFIRE_HIPCC_EXTRA_FLAGS=...` | Append flags to JIT kernel compilations. |
| `HIPFIRE_PROMPT_TOKEN_HEAT=1` | Dump per-position BPE merge-rank heat to stderr. |
