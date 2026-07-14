# KV & long-context eval buildout

Status: **implemented** (all phases landed; live GPU accuracy validation pending).
Branch: `chaingun`. Date: 2026-07-14.

Turns the scattered, half-wired KV/long-context tests into first-class
`hipfire eval` batteries/suites: human-usable CLI, the NIAH bench driven through
eval, the five long-context suites actually running a model, and a graded
long-context KLD metric. Companion inventory: `docs/kv-long-context-test-inventory.md`.

## Context

- `pflash_niah_bench` (NIAH retrieval) + `perplexity` (PPL/KLD) are external
  example binaries; long-context knobs (`--ctx`, corpus, KV mode) were env-only.
- The five long-context suites (`Ruler`, `Niah`, `NoLiMa`, `NeedleChain`,
  `SequentialNiah`) are declared in `SuiteId` but dead-end in `barrage_rows()`
  ("native barrage runner is not implemented yet", `lib.rs:1610`).
- Long-context KV *quality* had no graded number (NIAH bench = PASS/FAIL + TTFT).

## Governing decisions (confirmed with user; do not re-litigate)

1. **Subprocess orchestration** — `hipfire-eval` stays GPU-independent (no
   `hipfire-runtime`/`hipfire-rdna` link); it resolves + spawns the GPU example
   binaries and parses stdout into rows. The "move into eval" is about the
   battery/CLI surface, not linking the runtime.
2. **Suite datasets are heterogeneous** (discovered by inspecting the real
   sources — the five are *not* five uniform fetch-and-score datasets):
   - **Niah / SequentialNiah / Ruler** have no canonical static HF dataset (they
     are generators) → **vendor generated slices** via in-repo deterministic
     generators, committed as static fixtures. Offline, portable.
   - **NeedleChain** is a real static HF dataset (`hyeonsss/needlechain`,
     `data/k{5,10,20,50,100,200}.parquet`) → fetch + **read parquet in Rust**
     (arrow-rs `parquet` crate added to `hipfire-eval`).
   - **NoLiMa** (`amodaresi/NoLiMa`) ships *components* (haystack books +
     `needlesets/*.json`), so faithful NoLiMa means reimplementing its
     test-assembly + scoring → heaviest, done last.
3. **Include the KLD bridge** — long-context quality gets a graded number.

## Phases

### Phase 0 — human-usable CLI — DONE (`88fec8dfb`), pushed
`--ctx`, `--corpus`, `--kv-mode` (validated against the union of what the
`perplexity`/`run` binaries accept), `--kv-hierarchical` (sets
`HIPFIRE_KV_HIERARCHICAL` via `apply_kv_env`). Perplexity executor reads the
flags (env as fallback). Portable model paths in `run_ppl_baseline.sh` /
`run_lloyd_compare.sh`. Files: `config.rs`, `lib.rs` (EvalConfig), `executor_examples.rs`,
`forward.rs`.

### Phase 1 — flag-drive the NIAH battery — DONE (`79f8fd18b`), pushed
`pflash` battery takes `--kv-mode` (default asym3) + `--fixture <a,b>` filter;
modes the bench can't do (kvarn/f32/hierarchical) skip toward the perplexity
battery (`PFLASH_KV_MODES`). `apply_kv_env` applied. Unit tests for the filter.

### Phase 2a — NIAH-family suites (Niah + SequentialNiah + NeedleChain) — DONE (`1039a4265`), pushed
Shared long-context barrage runner, then the three suites. Template:
`gpqa_materialized_items`/`read_gpqa_item`/`run_examples_gpqa_item`
(`executor_examples.rs`) + `fetch_dataset` (`datasets.rs`).
- Add arrow-rs `parquet` dep + a reader helper to `hipfire-eval`. **NOTE:** a
  follow-up (see memory `project-parquet-crate-training-port`) ports
  `hipfire-train`'s JSONL+QEMB loading (`labels.rs`) onto this same crate — so
  factor the reader to be reusable, not eval-private.
- `run_examples_longctx_item()`: prompt-file → `run` example binary (`--kv`,
  `--max-tokens`) → substring/numeric-recall score → `EvalResult`; match arms in
  `examples_barrage_rows`; drop the `barrage_rows` dead-ends for these suites.
- **Niah** ← local `benchmarks/longctx/niah/niah_*.jsonl` (prompt = `filler_text`
  with the needle already embedded + `question`; score = `expected_answer_substring`).
  `selected_item_ids` aligned to real fixtures (`niah_8k:0`…).
- **SequentialNiah** ← new deterministic generator + vendored slices (ordered
  multi-needle retrieval).
- **NeedleChain** ← parquet materializer (multi-hop numeric reasoning: 4
  orderings parallel/forward/backward/chaotic, numeric `*_total_val` answer);
  numeric-match scoring.
- Per-suite `_materialized_items`, registration touch points, unit tests.

### Phase 2b — NoLiMa — DONE
`nolima_materialized_items` assembles a needle-in-book test from the
`amodaresi/NoLiMa` components: needle template (`{CHAR} lives next to {1}`) +
one-hop question, character chosen by the test key's `_C0N` suffix, inserted at
mid-depth into a haystack book truncated to the ctx budget. Expected answer =
the character name (substring scored via the shared runner). Non-commercial
license carried in the manifest. Real-component assembly validated behind an
`#[ignore]` test.

### Phase 2c — RULER — DONE
Vendored generated slices (`benchmarks/longctx/ruler/generate_ruler.py`) for two
canonical RULER task families: S-NIAH (magic-number retrieval) and
variable-tracking (chained assignments → recover all vars equal to a value), at
4k/8k, in the NIAH multi-needle schema. Wired as a local suite (like Niah) via
`ruler_materialized_items`. Recall-scored by the shared runner. Follow-up: the
remaining RULER tasks (multi-key/value NIAH, CWE/FWE aggregation, QA).

### Phase 3 — graded long-context KLD bridge — DONE
The perplexity battery now accepts a NIAH-family `.jsonl` fixture as `--corpus`:
`longctx_corpus_from_fixture` extracts the haystack text to a plain-text corpus,
so PPL + `KLD/tok` are measured over the long sequence (the graded long-context
KV-quality metric). Plain-text corpora pass through unchanged.

bf16 long-context reference recipe (needs a bf16 model + GPU):
```
# 1. one eval run writes the extracted corpus to
#    <out_dir>/artifacts/perplexity_corpus/<fixture>.txt  (or use any long .txt)
# 2. build the bf16 reference over that corpus:
perplexity <bf16-model.hfq> <long-corpus.txt> --ctx 16384 --dump-ref ref.pkld
# 3. score a quantized model against it at long ctx:
hipfire eval <model.hfq> --battery perplexity \
  --corpus benchmarks/longctx/niah/niah_16k.jsonl --ctx 16384 \
  --kldref ref.pkld --kv-mode kvarn
```
Stretch (deferred): position-windowed (post-needle) KLD needs a small
`perplexity` flag.

## Registration touch points (per suite)

`SuiteId::{hf_repo_id,hf_revision,license}` + `parse`/`as_str` (`lib.rs:249+`);
`datasets.rs::selected_item_ids` + `<suite>_materialized_items`;
`executor_examples.rs::examples_barrage_rows` + `run_examples_<suite>_item`;
`config.rs::usage` / `forward.rs::EVAL_HELP`.

## Verification

- Per phase: `cargo clippy -p hipfire-eval -p hipfire-cli --all-targets` + tests;
  `./tests/no-gpu-ci.sh` (Rust portion — note a pre-existing unrelated ruff red
  in `benchmarks/npu_gemm_tuning/r*/r*_gen.py`).
- Live GPU (nix2/halo, `hipfire lock`): `hipfire eval MODEL --battery barrage
  --suite niah,needle_chain --executor examples --kv-mode asym3` yields real
  Pass/Fail accuracy; `--offline` degrades to a clean skip.
- Pre-commit hook runs the affected-model tiny-quant/state gates on eval changes
  (~5 min) — commit in the background.
