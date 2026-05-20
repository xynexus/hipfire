# calibration-mix-v1 — deployment-mirror calibration corpus

A mixed-source calibration corpus that replaces wikitext-2 for Tier 1
calibration (imatrix + Hessian) of Qwen3.5/3.6 MQ4 quantization. The goal is
to better match the deployment activation distribution: dense chat dialog with
ChatML framing, agentic tool-call traces, and production code, plus a wiki
prose tail for general-English coverage.

The activation imbalance with wikitext-only calibration is the same class of
problem documented in the project's "Prompt-structure τ sensitivity"
methodology: identical token *count* with a different token *shape* shifts the
prefix-conditioned distribution at every position. A calibration set whose
shape matches deployment yields better-fit MQ4 scales/codebooks for the layers
that fire on dialogue and tool-call inputs.

## Files

| File | Purpose |
|------|---------|
| `calibration-mix-v1.txt` | 8.13 MB concatenated text; tokenizer chunks into 2048-tok windows. **Generated; do not edit by hand.** |
| `calibration-mix-v1.md5` | md5 tripwire (`68a1d2e62117e692e0e04c2811349aaf`) — verify byte stability before runs |
| `calibration-mix-v1.build.log` | Per-bucket assembly statistics from the build run |
| `build/build_calibration_mix_v1.py` | Deterministic generator (seed = 1024); the recipe |
| `build/verify_buckets.py` | Re-runs each bucket assembler standalone to compute per-bucket token counts via `llama-tokenize` |

## Composition

Target composition per task brief, with realized values from
`llama-tokenize -m /mnt/nas/kaden/hipfire/lucebox-quants/Qwen3.5-9B-Q4_K_M.gguf`:

| Source class | Target % | Realized % | Tokens | Bytes |
|---|---:|---:|---:|---:|
| Wikipedia / English prose | 25.0% | 24.60% | 551,156 | 2,410,934 |
| Chat dialogs (ChatML framed) | 30.0% | 29.52% | 661,327 | 2,789,267 |
| Code (Python / Rust / HIP-C++) | 25.0% | 24.73% | 553,940 | 1,791,027 |
| Tool-call / agentic JSON | 20.0% | 21.15% | 473,818 | 1,499,497 |
| **Total** | **100%** | **100%** | **2,240,241** | **8,490,725** |

Total tokens: **2.24 M** (target was 1024 × 2048 = 2.10 M; we slightly over-write
so the chunker has margin past the 1024-window evaluation point, mirroring the
existing wikitext slice which over-writes to ~2.41 M).

The final file shuffles all four source classes' chunks together with a fixed
seed (1024), so the deployment-mirror mix is interleaved rather than block-
segmented. This matches the deployment activation distribution better than
ordering by class.

## Sources

### 1. Wikipedia / English prose (24.6%)

The first ~2.3 MB of `wikitext2-1024s-2048ctx.txt` (the existing slice, md5
`83b0205a304bf4e52172ecdb05f2e895`), clipped at a paragraph boundary.

License: **Creative Commons Attribution-ShareAlike 3.0** (Wikipedia text;
inherited from wikitext-2 source).

### 2. Chat dialogs (29.5%)

Primarily ChatML-framed multi-turn dialogue derived from
`lambda/hermes-agent-reasoning-traces` (Apache-2.0) with `<tool_call>` and
`<tool_response>` blocks stripped — the reasoning + final-answer skeleton
remains. 154 hermes rows used; rows shorter than ~200 B after stripping are
filtered out.

A small synthesized set of 19 helpful-assistant exchanges (concept
explanation, multi-turn coding help, step-by-step how-to, open-ended Q&A) is
appended for genre diversity. These are written from scratch in this
repository and add some lift to the long-form-answer token distribution.

License: **Apache-2.0** (hermes traces) + this-repo (synthesized exchanges,
written for this corpus). All synthesized text is in the build script so it
travels with the recipe.

### 3. Code (24.7%) — Python 35.2% / Rust 49.3% / HIP-C++ 15.5%

Original task brief targeted Python 60% / Rust 25% / HIP 15% but the in-tree
`scripts/` Python corpus is only ~611 KB total. We pull every script we have
(35 files), then top up the byte budget with additional Rust modules so the
overall code share stays at ~25% of the corpus. The 49% Rust share is higher
than originally targeted but is *more* representative of this project's actual
deployment workload (the engine that consumes the MQ4 outputs).

Specific sources:

- `scripts/*.py` — production Python tooling for quantization, profiling,
  benchmarks, calibration. Includes:
  `astrea.py`, `kernel_atlas.py`, `mq4_masked_calib.py`, `test_astrea.py`,
  `dflash_train_poc.py`, `governance/apply_spdx_headers.py`,
  `collect_hessian.py`, `coverage-audit.py`, plus 27 smaller scripts.
- `benchmarks/prompts/humaneval_*.txt`, `benchmarks/prompts/lru_cache_*.txt`,
  `benchmarks/prompts/dirty/*.txt` — curated Python code prompts.
- `crates/*/src/**/*.rs` and `crates/*/examples/*.rs` — production Rust source
  drawn from across the workspace (40 + 41 topped-up = 81 files, mid-sized
  1 KB – 80 KB).
- `kernels/src/*.hip` — 45 HIP kernel sources (mid-sized 1.5 KB – 80 KB);
  exposes `__global__`, `__device__`, dp4a / WMMA / packed-FP16 intrinsics,
  and the macro/typedef patterns common to GPU code.

Each file is emitted with a `# source: <relpath>` / `// source: <relpath>`
comment header for provenance. The tokenizer emits these as ordinary comment
tokens; they don't affect distribution materially.

License: in-repo source. **MIT** (per this project's `LICENSE`).

### 4. Tool-call / agentic JSON (21.2%)

Three layers:

a. Existing fixtures from `benchmarks/prompts/`:
   - `agentic_hermes_system.txt` — Hermes function-calling system prompt
   - `agentic_pi_system.txt` — pi-harness multi-tool system prompt
   - `tool_call_system.txt` — minimal Qwen3.5 tool-call system prompt
   - `agentic_user_read.txt`, `agentic_user_multistep.txt`,
     `tool_call_read_file.txt` — short user-question seeds

b. Hermes traces with tool-call density (13 rows that contain at least one
   `<tool_call>` turn), rendered with full Qwen3.5 ChatML + tools-in-system
   framing (template lifted from
   `/mnt/nas/kaden/models/Qwen3.5-35B-A3B/chat_template.jinja`).

c. 28 synthesized round-trip examples written for this corpus. Coverage:
   - Single round-trips: read_file / write_file / bash / web_search / compute
     across 20 distinct prompts (file IO, computation, search, system info,
     scripts).
   - Multi-step scenarios: 3 examples chaining 2+ tool calls per user turn.
   - Error / recovery scenarios: 3 examples where the tool returns an error
     and the assistant recovers gracefully.
   - Schema edge cases: nested JSON arguments, multi-line string parameters
     (a Python function body inside `<parameter=content>`).

All synthesized payloads follow the canonical Qwen3.5 tool-call format from
the chat template:

```
<tool_call>
<function=NAME>
<parameter=KEY>
VALUE
</parameter>
</function>
</tool_call>
```

Tool responses are wrapped as user turns with `<tool_response>...</tool_response>`.

License: **Apache-2.0** (hermes traces) + this-repo (fixtures + synthesized
examples).

## License summary

The composite corpus carries multiple permissive licenses:

| Bucket | License |
|---|---|
| Wikipedia prose | CC-BY-SA 3.0 |
| Hermes-derived chat + tool-call | Apache-2.0 |
| In-repo source code | MIT (this project) |
| Synthesized exchanges + tool-calls | MIT (written for this corpus) |

No copyleft restrictions apply to using this corpus for calibration. The
corpus file itself is committed to the repository under this project's
license; downstream consumers should preserve the upstream attribution for
the Wikipedia and Hermes-derived material when redistributing the corpus.

## Reproducibility

```sh
# From repo root, no internet access required if NAS path is mounted.
python3 benchmarks/quality-baselines/slice/build/build_calibration_mix_v1.py
md5sum benchmarks/quality-baselines/slice/calibration-mix-v1.txt
```

Inputs the script depends on (must be byte-identical for reproducibility):

| Input | Path | Notes |
|---|---|---|
| Existing wiki slice | `benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt` | md5 `83b0205a304bf4e52172ecdb05f2e895` |
| Hermes GLM parquet | `~/.cache/huggingface/hub/datasets--lambda--hermes-agent-reasoning-traces/snapshots/b92885e4f0161d4b2536512710e004d4892cac6e/data/glm-5.1/train.parquet` | first 4000 rows |
| Hermes Kimi parquet | `~/.cache/huggingface/hub/datasets--lambda--hermes-agent-reasoning-traces/snapshots/b92885e4f0161d4b2536512710e004d4892cac6e/data/kimi/train.parquet` | first 4000 rows |
| In-repo code | `scripts/**/*.py`, `crates/**/*.rs`, `kernels/src/*.hip` | from current HEAD |

Random seed: **1024** (`SEED` constant in `build_calibration_mix_v1.py`).

If any in-repo source file changes, regenerating produces different bytes —
the md5 will mismatch. That's intentional: the calibration corpus tracks the
project's actual deployment surface area, so as code evolves the calibration
should evolve with it (or be re-pinned with a different version tag).

The tokenized count (above) was measured with
`/home/kaden/llama.cpp/build/bin/llama-tokenize` on the Qwen3.5-9B GGUF at
`/mnt/nas/kaden/hipfire/lucebox-quants/Qwen3.5-9B-Q4_K_M.gguf`. Counts will
vary by ±0.5% across Qwen3.5/3.6 model sizes due to BPE merge-table tweaks but
the bucket ratios are stable.

## Versioning

This is **v1**. If we evolve the mix (different ratios, additional source
classes, larger total tokens), the next iteration will be `calibration-mix-v2.txt`
with its own md5 tripwire and README. Prior version files stay committed so
old calibration runs are byte-reproducible.

## See also

- `wikitext2-1024s-2048ctx.txt` + `make_slice.sh` — the original
  wikitext-2-only slice this corpus is designed to supplement (and possibly
  replace) for calibration use.
- `docs/methodology/perf-benchmarking.md` — the "byte-identical prompts"
  rule that motivated putting calibration corpora in git rather than
  regenerating on demand.
- PR #7 / branch `pr-7-tier1-calibration` — the Tier 1 imatrix+Hessian
  calibration pipeline that consumes this corpus.
