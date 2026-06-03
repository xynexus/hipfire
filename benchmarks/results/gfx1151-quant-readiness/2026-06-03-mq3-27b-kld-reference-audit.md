# MQ3 27B KLD Reference Audit

- date: 2026-06-03T08:18:27+08:00
- arch scope: gfx1151
- commit: fab9d2bc88d2de7b5febfa9ec8afed80b6700557
- branch: qwen35-native-mtp
- target fixture: qwen3.5-27b.mq3
- control format: MQ4

## Decision

Qwen3.5 27B MQ3 remains a dense candidate, but its dense promotion gate is
still incomplete because no comparable qwen3.5-27B BF16 or Q8 KLD reference is
available locally or in the pinned reference manifest.

Do not reuse `qwen3.6-27b-bf16.kldref.bin` for this qwen3.5-27B comparison.
The model-family/version mismatch makes the KLD row non-comparable.

## Local Reference State

The pinned manifest at
`benchmarks/quality-baselines/harness/manifest.json` contains:

| reference | manifest hf_repo | local state |
|---|---|---|
| `qwen3.5-0.8b-bf16.kldref.bin` | none | absent |
| `qwen3.5-4b-bf16.kldref.bin` | `hipfire-models/qwen-kldref` | present, sha256 OK |
| `qwen3.5-9b-bf16.kldref.bin` | `hipfire-models/qwen-kldref` | present, sha256 OK |
| `qwen3.6-27b-bf16.kldref.bin` | `hipfire-models/qwen-kldref` | present, sha256 OK |

Verified local reference hashes:

| reference | size bytes | sha256 |
|---|---:|---|
| `qwen3.5-4b-bf16.kldref.bin` | 2480989032 | `d3ba9b5618f86bd9efcf2cd95bc718c41df288612ddc9f8f2592ab7dbb90bb75` |
| `qwen3.5-9b-bf16.kldref.bin` | 2480989032 | `06948cd36bab71fce2df5d9af1be03c9cfb4090637d881056a6937a29caa65a7` |
| `qwen3.6-27b-bf16.kldref.bin` | 2480989032 | `8af83b38710fbc8e5ee46ce2b84b3545381c834f17bf6dfaa15fd817e4734446` |

Machine-readable inventory:
`benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq3-kld-reference-inventory.json`
verifies the manifest refs above and scans:

- `benchmarks/quality-baselines/refs`
- `/home/sadara/Models`
- `/home/sadara/.cache/huggingface/hub`
- `/home/sadara/.hipfire/models`

For the required `qwen3.5-27b` KLD fixture, the inventory reports:

- manifest entry present: `false`
- local manifest ref present: `false`
- local search matches: `0`

The same inventory also records that the Qwen3.5 and Qwen3.6 35B-A3B KLD refs
are absent from the manifest and local roots.

## Remote Reference State

`huggingface_hub.list_repo_files("hipfire-models/qwen-kldref",
repo_type="dataset")` returned:

- `.gitattributes`
- `.gitignore`
- `.gitignore~`
- `README.md`
- `qwen3.5-0.8b-bf16.kldref.bin`
- `qwen3.5-4b-bf16.kldref.bin`
- `qwen3.5-9b-bf16.kldref.bin`
- `qwen3.6-27b-bf16.kldref.bin`

No `qwen3.5-27b-bf16.kldref.bin` file was present in that remote listing.

The `qwen3.5-4b-bf16.kldref.bin` reference is now manifest-pinned and present
locally; it supports the 4B boundary rejection only. It is not a substitute for
the missing qwen3.5-27B reference.

## Reproduction Route

The in-tree producer is `crates/hipfire-runtime/examples/build_kld_ref.rs`.
It requires a BF16 GGUF plus the pinned llama.cpp `llama-perplexity` binary,
then writes an HFKLDR reference. The current local source for Qwen3.5-27B is an
HF safetensors cache, not a BF16 GGUF, so the reference cannot be generated from
the presently discovered local files.

Required next work before a 27B MQ3 KLD row can be treated as promotion
evidence:

1. Obtain a qwen3.5-27B BF16 GGUF compatible with the pinned llama.cpp
   reference path, or add a manifest-pinned uploaded
   `qwen3.5-27b-bf16.kldref.bin`.
2. Record sha256, source GGUF, slice md5, producer command, and HF repo metadata
   in `benchmarks/quality-baselines/harness/manifest.json`.
3. Run bounded `eval_hipfire` rows for qwen3.5-27B MQ4 control and MQ3
   candidate against that exact reference.
4. Reduce the rows with `benchmarks/quality-baselines/harness/kld_reduce.py`
   and only then reconsider dense 27B MQ3 promotion.
