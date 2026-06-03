# Lloyd KLD Harness Audit - 2026-06-03

Scope: prepare current gfx1151 BF16-referenced KLD harness coverage for
MQ3-Lloyd and MQ4-Lloyd, and record the current bounded MQ4-Lloyd same-run
KLD result without treating invalid zero-KLD/NaN-PPL output as a promotion
signal.

## Live Artifact Search

Command:

```bash
rg --files /home/sadara/.hipfire/models /home/sadara/Models/hipfire-candidates/gfx1151-readiness 2>/dev/null | rg -i 'lloyd|mq3-lloyd|mq4-lloyd|lloyd-mq3|lloyd-mq4'
```

Current result after Lloyd artifact bring-up:

- `/home/sadara/Models/hipfire-candidates/gfx1151-readiness/qwen3.5-9b-lloyd-mq3.hfq`
- `/home/sadara/Models/hipfire-candidates/gfx1151-readiness/qwen3.5-9b-lloyd-mq4.hfq`
- `/home/sadara/.hipfire/models/qwen3.5-9b.mq3-lloyd`
- `/home/sadara/.hipfire/models/qwen3.5-9b.mq4-lloyd`

Conclusion: current 9B MQ3-Lloyd and MQ4-Lloyd artifacts exist. The broader
4B, 27B, and A3B Lloyd artifacts remain missing.

## Prepared Wrapper Cases

`scripts/mq6_kld_compare.py` now exposes guarded Lloyd cases:

- `qwen35-4b-mq3-lloyd` -> `qwen3.5-4b.mq3-lloyd`
- `qwen35-9b-mq3-lloyd` -> `qwen3.5-9b.mq3-lloyd`
- `qwen35-27b-mq3-lloyd` -> `qwen3.5-27b.mq3-lloyd`
- `qwen35-a3b-mq3-lloyd` -> `qwen3.5-35b-a3b.mq3-lloyd`
- `qwen35-9b-mq4-lloyd` -> `qwen3.5-9b.mq4-lloyd`
- `qwen35-27b-mq4-lloyd` -> `qwen3.5-27b.mq4-lloyd`
- `qwen35-a3b-mq4-lloyd` -> `qwen3.5-35b-a3b.mq4-lloyd`

The wrapper checks reference filenames against fixture tokens and rejects
accidental reuse of the wrong BF16/HFKLDR reference unless an experiment
explicitly passes `--allow-ref-mismatch`.

## First Commands Once Artifacts Exist

MQ3-Lloyd 9B boundary rerun, paired with the MQ4 control:

```bash
python3 scripts/mq6_kld_compare.py \
  --case qwen35-9b-mq4 --case qwen35-9b-mq3-lloyd \
  --ref benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin \
  --max-chunks 20 --kv-mode q8 --scoring-mode prefill \
  --out benchmarks/results/gfx1151-quant-readiness/<date>-mq3-lloyd-9b-kld.json \
  --fail-on-missing
```

MQ4-Lloyd same-run cohort against MQ4 and MQ6:

```bash
python3 scripts/mq6_kld_compare.py \
  --case qwen35-9b-mq4 --case qwen35-9b-mq4-lloyd --case qwen35-9b-mq6 \
  --ref benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin \
  --max-chunks 512 --kv-mode q8 --scoring-mode prefill --timeout 12000 \
  --out benchmarks/results/gfx1151-quant-readiness/<date>-mq4-lloyd-9b-kld.json \
  --fail-on-missing
```

This cohort has now been run in bounded c20 form:

```bash
python3 scripts/mq6_kld_compare.py \
  --case qwen35-9b-mq4 --case qwen35-9b-mq4-lloyd --case qwen35-9b-mq6 \
  --ref benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin \
  --max-chunks 20 --kv-mode q8 --scoring-mode prefill --timeout 3600 \
  --out benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq4-lloyd-9b-kld-c20.json \
  --fail-on-missing
```

Evidence:

- JSON:
  `benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq4-lloyd-9b-kld-c20.json`
- Markdown:
  `benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq4-lloyd-9b-kld-c20.md`
- Reducer table:
  `benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq4-lloyd-9b-kld-c20/result-table.md`

Result:

| Variant | Mean KLD | PPL | Interpretation |
|---|---:|---:|---|
| `qwen3.5-9b.mq4-kvq8-c20` | `0.236385` | `9.04742` | current MQ4 control |
| `qwen3.5-9b.mq4-lloyd-kvq8-c20` | `0.0` | `NaN` | invalid zero-KLD/NaN-PPL evidence |
| `qwen3.5-9b.mq6-kvq8-c20` | `0.0568137` | `9.28051` | current MQ6 comparison row |

The MQ4-Lloyd eval log reports `mean NLL = NaN` and `PPL = NaN`. Per Astrea
guardrails this must not be interpreted as a quality win. It is a rejection
signal for the current artifact hash.

A3B Lloyd KLD remains blocked until matching Qwen3.5/Qwen3.6 35B-A3B HFKLDR
references exist and are manifest-pinned. The prepared A3B cases are useful only
after that reference gap is closed.

## Decision

- MQ3-Lloyd: keep `candidate-research-gated`; current 9B BF16-referenced KLD
  loses to MQ4.
- MQ4-Lloyd: keep `candidate-research-gated`; the current 9B artifact is
  coherence-rejected and the bounded same-run KLD cohort produced invalid
  zero-KLD/NaN-PPL evidence.
