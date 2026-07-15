# Gemma 4 Phase 6 prompt, channel, tool, and sampler correctness

Date: 2026-07-15. Status: implementation exit gates passed; 31B-it remains
unadmitted because Phase 5 is blocked.

## Result

The generic prompt and generation seams now implement the released Gemma 4
instruction contract:

- strict official Jinja rendering remains byte-identical for all committed
  Phase-0 prompt cases;
- non-ByteLevel Hugging Face BPE tokenizers use ranked merges after
  SentencePiece normalization and byte fallback, rather than greedy
  longest-vocabulary matching;
- the offline `gemma4_prompt_token_ids` validator matches all 36 prompt cases
  across 31B-it, 26B-A4B-it, E4B-it, and E2B-it: 1,640 token IDs exactly;
- Gemma 4 serving carries the full metadata EOS set `[1, 106, 50]` through a
  shared multi-terminator decode seam. Terminators are classified before visible
  output and cannot leak marker bytes;
- the released `call:name{...}` tool grammar, multiple calls, malformed output,
  thought/visible channel separation, and tool continuation have unit coverage;
- generic top-k 64 sampling has a fixed-seed CPU reference test.

No Gemma 4 runtime or capture path retains the obsolete `<end_of_turn>`-only or
`<|tool_call|>{json}` assumptions.

## Verification

```text
$ cargo test -p hipfire-prompt gemma4_official_prompt_fixtures_are_byte_identical
PASS

$ cargo run -q -p hipfire-runtime --example gemma4_prompt_token_ids -- \
    /srv/huggingface/models--google--gemma-4-31B-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb/tokenizer.json \
    benchmarks/gemma4/fixtures/prompts/gemma-4-31B-it.json \
    benchmarks/gemma4/fixtures/prompts/gemma-4-26B-A4B-it.json \
    benchmarks/gemma4/fixtures/prompts/gemma-4-E4B-it.json \
    benchmarks/gemma4/fixtures/prompts/gemma-4-E2B-it.json
gemma4_prompt_token_ids: PASS (fixtures=4 cases=36 tokens=1640)

$ cargo test -p hipfire-runtime tool_call
17 passed

$ cargo test -p hipfire-runtime sampler
18 passed

$ cargo test -p hipfire-arch-gemma4 --lib
8 passed

$ cargo check -p hipfire-serving-core
PASS
```

`./tests/no-gpu-ci.sh` passed its Rust checks, no-GPU unit suites, and fixture
round trips. The aggregate script then exited nonzero in its repository-wide
Python Ruff stage on five unrelated pre-existing `benchmarks/npu_gemm_tuning/`
findings (`PLW1510` once and `PLW2901` four times). The Gemma 4 diagnostic tool
passes `ruff check` and `py_compile` directly.

## Reuse and cleanup ledger

- Existing primitive reused: `JinjaChatFrame`, `Gemma4OutputState`, the shared
  sampler, `EosFilter`, and the existing ranked BPE merge engine.
- Duplicate removed or retained: single-EOS entry points remain compatibility
  wrappers; the new multi-terminator variants own the shared implementation.
- Generic seam added or changed: ranked SentencePiece-BPE encoding and
  multi-terminator simple-AR/decode entry points.
- Generic abstraction consumers: Gemma 4 uses both seams; existing GPT-2 BPE
  tokenizers share the same merge engine, and all existing single-EOS
  architectures keep their wrapper behavior.
- Stale assumption removed: Gemma 4 no longer truncates `[1, 106, 50]` to one
  EOS ID or treats non-ByteLevel BPE as greedy SentencePiece vocabulary matching.
- Oracle retained: committed official prompt bytes/token IDs and the pinned
  Hugging Face tokenizer remain independent offline comparison inputs.

This phase does not override the frozen Phase-5 stop. The 31B-it checkpoint is
not advertised or admitted until the dense BF16 model gate passes.
