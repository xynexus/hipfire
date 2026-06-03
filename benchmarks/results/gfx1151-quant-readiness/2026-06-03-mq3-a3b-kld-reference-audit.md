# MQ3 A3B KLD Reference Audit

- date: 2026-06-03
- arch: gfx1151
- commit: fab9d2bc88d2de7b5febfa9ec8afed80b6700557
- branch: qwen35-native-mtp
- scope: Qwen3.5/Qwen3.6 35B-A3B MQ3 versus MQ4 KLD readiness

This audit records why A3B KLD rows are not currently runnable for the MQ3 MoE
promotion lane. The blocker is the missing HFKLDR reference, not a measured
MQ3 runtime or quality failure.

## Current reference inventory

Machine-readable inventory:
`benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq3-kld-reference-inventory.json`
verifies that the required A3B HFKLDR refs are absent from the manifest and
local roots:

- `qwen3.5-35b-a3b-bf16.kldref.bin`: manifest entry `false`, local matches `0`
- `qwen3.6-35b-a3b-bf16.kldref.bin`: manifest entry `false`, local matches `0`

The local filename search shape is:

```bash
rg --files benchmarks/quality-baselines/refs \
  /home/sadara/Models \
  /home/sadara/.hipfire/models \
  | rg -i 'a3b.*(kldref|bf16|q8)|kldref.*a3b|35b-a3b.*(bf16|q8)'
```

Result: no matches.

The current remote `hipfire-models/qwen-kldref` dataset listing contains:

```text
qwen3.5-0.8b-bf16.kldref.bin
qwen3.5-4b-bf16.kldref.bin
qwen3.5-9b-bf16.kldref.bin
qwen3.6-27b-bf16.kldref.bin
```

It does not contain a Qwen3.5 or Qwen3.6 35B-A3B KLD reference.

## Available A3B candidates

The target/candidate artifacts required for the candidate side are present:

```text
/home/sadara/.hipfire/models/qwen3.5-35b-a3b.mq4
/home/sadara/.hipfire/models/qwen3.5-35b-a3b.mq3
/home/sadara/.hipfire/models/qwen3.6-35b-a3b.mq4
/home/sadara/.hipfire/models/qwen3.6-35b-a3b.mq3
```

The missing piece is a matching reference such as:

```text
benchmarks/quality-baselines/refs/qwen3.5-35b-a3b-bf16.kldref.bin
benchmarks/quality-baselines/refs/qwen3.6-35b-a3b-bf16.kldref.bin
```

Do not use the 9B or qwen3.6-27B references as substitutes for A3B.

## Harness state

`scripts/mq6_kld_compare.py` now includes A3B cases for the MQ4 control and
MQ3 candidate:

```text
qwen35-a3b-mq4
qwen35-a3b-mq3
qwen36-a3b-mq4
qwen36-a3b-mq3
```

It also refuses an accidental fixture mismatch by default. A local guard smoke
with the 9B reference exits before writing output:

```bash
python3 scripts/mq6_kld_compare.py \
  --case qwen35-a3b-mq4 \
  --case qwen35-a3b-mq3 \
  --ref benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin \
  --out /tmp/should-not-write.json
```

Expected error:

```text
reference filename 'qwen3.5-9b-bf16.kldref.bin' does not match selected case(s): qwen35-a3b-mq4, qwen35-a3b-mq3
```

## Next gate

After a matching A3B HFKLDR reference exists and is manifest-pinned, run:

```bash
python3 scripts/mq6_kld_compare.py \
  --case qwen35-a3b-mq4 \
  --case qwen35-a3b-mq3 \
  --ref benchmarks/quality-baselines/refs/qwen3.5-35b-a3b-bf16.kldref.bin \
  --out benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq3-qwen35-a3b-kld.json \
  --max-chunks 20
```

For Qwen3.6 A3B, use the matching `qwen3.6-35b-a3b-bf16.kldref.bin` reference
with `--case qwen36-a3b-mq4 --case qwen36-a3b-mq3`.

Only after those rows exist should A3B KLD count toward an MQ3 MoE promotion
claim.
