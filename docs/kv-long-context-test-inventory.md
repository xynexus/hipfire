# KV-cache & long-context test inventory

Date: 2026-07-14. Branch: `chaingun`. Status: reference.

What hipfire already has for exercising **KV-cache quality** and **long-context
behaviour**. Written to settle "is there anything to measure the KV levers
with?" — there is: **8 runnable KV-applicable tests, 5 of them long-context**
(single- and multi-needle retrieval from 8K to 128K, plus PPL/KLD at arbitrary
ctx). The gap is not "no long-context test"; it is that the retrieval bench
reports **pass/fail + TTFT, not PPL/KLD**, so KV-*quality* deltas in the
retrieval regime aren't yet a graded number (see §4).

Anchors are `file:line` at time of writing; verify before relying on a line.

---

## 1. Runnable KV-quality tests

Every one below actually executes today.

| Test | Location | Measures | KV knobs | Long-ctx? |
|---|---|---|---|---|
| **perplexity** harness | `crates/hipfire-runtime/examples/perplexity.rs` | PPL, NLL/tok, KLD/tok | `--kv-mode` (all modes) + `HIPFIRE_KV_*` | any ctx via `--ctx` |
| **pflash_niah_bench** | `crates/hipfire-runtime/examples/pflash_niah_bench.rs` | needle-retrieval PASS/FAIL, TTFT breakdown, kept/source tokens | `--asym3` etc.; optional `--pflash` drafter | **yes, 8K–128K** |
| **pflash-gate.sh** | `tests/pflash-gate.sh` | wraps the NIAH bench over 6 fixtures; verdict + ±perf-drift regression | passes KV mode through | **yes, 8K–128K** |
| **coherence-gate-dflash.sh** | `tests/coherence-gate-dflash.sh` | degenerate-output / token-attractor detection over DFlash+DDTree decode | `--kv-mode q8` (hardcoded) | multi-length prose/code |
| **quant_kv_matrix.sh** | `benchmarks/scripts/quant_kv_matrix.sh` (+ `quant_kv_summarize.py`) | speed matrix: model × format × KV mode (tok/s, TTFT, bytes) | sweeps `ONLY_KV="q8 asym3 …"` | prefill/decode speed |
| **run_ppl_baseline.sh** | `benchmarks/run_ppl_baseline.sh` | PPL across {0.8B,4B,9B} × {MQ3,MQ4} | `--kv-mode` via env | ctx=2048 |
| **run_lloyd_compare.sh** | `benchmarks/run_lloyd_compare.sh` | Lloyd-Max vs uniform MQ2/MQ3 PPL | `--kv-mode` via env | ctx=2048 |
| **tiny-quant-gate.sh** | `tests/tiny-quant-gate.sh` | tokenizer-free emit→quantize→KLD-vs-anchor (weight quant, not KV mode) | — (weight-focused) | short |

> `run_ppl_baseline.sh` / `run_lloyd_compare.sh` hardcode model paths under
> `/home/kaden/.hipfire/models/…` — repoint before running on another box.

### KV modes accepted by `perplexity --kv-mode`

From the match in `perplexity.rs` (~L299–404): **`f32` / `fp16`, `q8`,
`asym4`, `asym3`, `asym2`, `kvarn`, `fwht4` / `fwht3` / `fwht2`.**
`hierarchical` is **not** a `--kv-mode` value — the two-tier hot/cold cache is
gated by `HIPFIRE_KV_HIERARCHICAL=1` (it replaces the `kvarn` decode path; see
`crates/hipfire-runtime/src/kv_hier.rs`). `asym2/3/4` are deprecated (see the
KV plans) but still selectable for head-to-head.

### Invocation

```bash
# PPL + KLD at 2K ctx, single KV mode
cargo build --release -p hipfire-runtime --example perplexity
./target/release/examples/perplexity MODEL.hfq CORPUS.txt \
  --ctx 2048 --warmup 8 --offset 0 --kv-mode kvarn --kld-ref REF-bf16.hfq

# Same, hierarchical two-tier cache (env-gated; kvarn decode path)
HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=512 HIPFIRE_KV_FOLD_M=4 \
HIPFIRE_KV_CORE_FRAC=0.125 HIPFIRE_KV_IMPORTANCE=vnorm \
  ./target/release/examples/perplexity MODEL.hfq CORPUS.txt --ctx 16384 --kv-mode kvarn

# Long-context needle retrieval (build with the deltanet feature)
cargo build --release --features deltanet -p hipfire-runtime --example pflash_niah_bench
./target/release/examples/pflash_niah_bench TARGET.hfq \
  benchmarks/longctx/niah/niah_16k.jsonl --maxgen 64 --asym3
```

---

## 2. Long-context tests specifically (≥3 required — there are 5)

Two distinct signals: **PPL/KLD** (quality of next-token distribution) and
**retrieval PASS/FAIL** (did the needle survive into the answer).

| Test | Signal | Context lengths | Single/multi needle |
|---|---|---|---|
| perplexity `--ctx N` | PPL, KLD/tok | any (limited by VRAM) | — (density, not retrieval) |
| pflash_niah_bench (single) | retrieval PASS/FAIL, TTFT | 8K, 16K, 32K, 64K, 128K | single needle @ ~50% depth |
| pflash_niah_bench (multi) | retrieval, ≥2/3 recovered | 16K, 64K | 3 needles @ 25/50/75% |
| longprose / longcode fixtures | retrieval | ~12.6K / ~13K | multi-doc prose; code constant |
| pflash-gate.sh | verdict + perf regression over 6 fixtures | 8K–128K | mixed |

### Fixtures (`benchmarks/longctx/niah/` and `benchmarks/prompts/`)

Single needle: `niah_{8k,16k,32k,64k,128k}.jsonl` (+ `.tok.jsonl` pre-tokenized
Qwen3 BPE). Deterministic (seed `0x5407_FFFF`), needle at ~50% depth. Schema:
`context_tokens, needle, question, expected_answer_substring, filler_text`.
Generator `generate_niah.py`.

Multi needle: `niah_multi_{16k,64k}.jsonl` (+ `.tok`). Seed `0x5407_FEEE`,
3 needles at depths 0.25/0.50/0.75, PASS = ≥2/3 substrings recovered.
Generator `generate_niah_multi.py`.

Prompt fixtures: `benchmarks/prompts/longprose_multidoc.jsonl` (~12.6K tok,
3 self-contained docs, question targets one doc) and `longcode_pflash.jsonl`
(~13K tok, truncated `pflash.rs`, needle = the `0xCAFEf00d` constant).
Generators `generate_longprose.py`, `generate_longcode.py`.

Each is a single deterministic sample — these are **regression fixtures, not
statistical benchmarks**. Position-sensitivity or accuracy-vs-depth curves would
need many samples per length (not currently generated).

---

## 3. Reference: KV-cache modes and `HIPFIRE_KV_*` knobs

Full hierarchical/KVarN knob set lives in `crates/hipfire-runtime/src/kv_hier.rs`
and is documented in `docs/env-vars.md` (generated). Condensed:

**Hierarchical two-tier** (`HIPFIRE_KV_HIERARCHICAL=1`): `HOT_BUDGET`(512),
`MIGRATE_BATCH`(128), `IDLE_KEEP`(0). **Cold compaction**: `FOLD_M`(4),
`CORE_FRAC`(0.125), `IMPORTANCE`(`vnorm`|uniform|knorm|kvnorm|attn|triattn),
`POS_LOCAL`(on), `MERGE`(`similarity` opt-in), `TRIATTN_SIDECAR`(path).
**Precision**: `COLD_BITS`(4), `COLD_V_BITS`(=K), `COLD_V_PERSLOT`(off).
**Layout/opt**: `DEFRAG_SEGMENTS`(0), `PYRAMID`(off)+`PYRAMID_AMP`(0.5).
**Capture** (offline analysis): `CAPTURE_K`, `CAPTURE_V`.
**KVarN**: `HIPFIRE_KVARN_ROTATE`(on), `HIPFIRE_KVARN_SIM`(off).

---

## 4. Gaps (what is *not* runnable / measured)

- **No PPL/KLD in the retrieval regime.** `pflash_niah_bench` reports pass/fail +
  TTFT, not a quality number. The long-context/redundancy regime the KV merge
  levers (CASK similarity-merge, PyramidKV, low-rank V) target therefore has **no
  graded metric** — a lever can pass NIAH while degrading distribution quality,
  and we won't see it. Bridging NIAH fixtures → per-token KLD (feed the fixture
  through `perplexity` machinery, score against a bf16 ref) is the missing piece
  and the highest-value next build for the KV work.
- **eval-crate barrage long-context suites are stubs.** `hipfire-eval` *defines*
  `Ruler`, `Niah`, `NoLiMa`, `NeedleChain`, `SequentialNiah` (4K synthetic, some
  HF-fetched) in its tier lists, but the native runner returns *"native barrage
  runner is not implemented yet"* (`crates/hipfire-eval/src/lib.rs:1610`); several
  daemon-backed anchors are likewise "not implemented yet" (`driver.rs:760–848`).
  Prompt materialization works; **model execution does not** — do not treat
  `--suite ruler` etc. as live. The runnable long-context path is the
  file-fixture NIAH bench in §1–2, not these suites.
- **`hierarchical` cannot be selected via `--kv-mode`** — only via
  `HIPFIRE_KV_HIERARCHICAL=1`. Sweeps that iterate `--kv-mode` values silently
  skip the two-tier cache.
- **Fixtures are single-sample** — good for regression, insufficient for
  accuracy-vs-depth or position-sensitivity curves.

---

## 5. Related docs

- `docs/plans/2026-07-12-hot-cold-hierarchical-kv-implementation.md` — hier-KV master plan.
- `docs/plans/2026-07-13-kv-compression-adoption-plan.md` — the 5 levers + eval-methodology blocker.
- `docs/env-vars.md` — generated full env-knob registry.
- `benchmarks/quality-baselines/` — frozen wikitext PPL slice + KLD-reference harness.
