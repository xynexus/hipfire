# ParoQ4G256 Contract Audit (2026-06-03)

- repo: `/home/sadara/.hipfire/src`
- branch: `qwen35-native-mtp`
- commit: `fab9d2bc`
- arch target: `gfx1151`
- baseline lane: ParoQ4G128
- structured status: `benchmarks/results/gfx1151-quant-readiness/2026-06-03-paro-q4g256-status.json`

## Current Contract

`docs/plans/paroquant-g256-milestone.md` defines the G256 gate:

- `PARO4G256`: true ParoQuant calibration/export at group size 256.
- `PARO4G256_MQ`: Paro rotation side metadata plus a Hipfire row-major G256 W4
  body so HFQ4/MQ4-style kernel families can be reused downstream.

The document is explicit that a regrouped G128 checkpoint is not proof of true
`PARO4G256` quality. If upstream ParoQuant cannot emit a real
`group_size=256` checkpoint, the quality result must be labeled
`UNVERIFIABLE`.

## Probe Surface

`scripts/paroquant_g256_probe.py` is present and CPU-only. It loads native G128
Paro safetensors, dequantizes the rotated body, and compares:

1. source ParoQ4G128 oracle output,
2. `PARO4G256`-style AWQ regrouping of the same rotated weights,
3. `PARO4G256_MQ`-style row-major HFQ4-G256 body with the same Paro rotation.

The probe reports output NRMSE and payload ratio, but it is a format-loss probe,
not a producer or runtime admission artifact.

The milestone document records the prior six-module probe:

| Variant | Avg output NRMSE vs source Paro | Worst output NRMSE | Avg payload ratio vs source |
|---|---:|---:|---:|
| `PARO4G256` AWQ regroup | 0.0859 | 0.1095 | 0.9817x |
| `PARO4G256_MQ` row-major G256 body + Paro side metadata | 0.0951 | 0.1114 | 1.0220x |

This satisfies the payload-ratio shape for `PARO4G256_MQ` (`1.0220x`, under the
`<=1.03x` target), but it does not satisfy the quality gate because it is based
on regrouped G128 weights rather than true G256 calibration/export.
The structured status artifact records `payload_ratio_gate_passed=true` and
`true_g256_quality_evidence_present=false` for this reason.

## Local Artifact Inventory

Machine-readable inventory:
`benchmarks/results/gfx1151-quant-readiness/2026-06-03-paro-q4g256-source-inventory.json`

The inventory was generated with:

```bash
python3 scripts/paroquant_inventory.py --pretty \
  --out benchmarks/results/gfx1151-quant-readiness/2026-06-03-paro-q4g256-source-inventory.json
```

It scanned the three local source roots below, excluding kernel/cache
directories that cannot be native Paro checkpoints:

- `/home/sadara/Models`
- `/home/sadara/.cache/huggingface/hub`
- `~/.hipfire/models`

Current summary:

- files seen: `861`
- safetensor dirs scanned: `17`
- safetensor files scanned: `151`
- complete native Paro modules: `0`
- complete `group_size=128` Paro modules: `0`
- complete `group_size=256` Paro modules: `0`
- scan errors: `0`
- decision quality state: `UNVERIFIABLE`

The structured status artifact records `native_g256_source_found=false` and
`cpu_probe_runnable_now=false`.
It also records:

```text
contract_state.current_stage=blocked_before_true_group_size_256_source
contract_state.quality_state=UNVERIFIABLE
contract_state.evidence_class=regrouped_g128_format_loss_only
contract_state.runtime_work_allowed=false
```

Therefore the CPU probe cannot be re-run locally against a current native Paro
checkpoint. If a true Paro source appears, the first bounded command is:

```bash
python3 scripts/paroquant_g256_probe.py \
  --model <native-paro-safetensors-dir-or-hf-repo> \
  --local-only \
  --max-modules 6 \
  --samples 4 \
  --pretty
```

No first-class `PARO4G256` or `PARO4G256_MQ` runtime DType/container is wired
in the current matrix. Runtime work must wait for a true producer contract,
oracle quality result, and PPL/KLD evidence against ParoQ4G128.
The structured status artifact records `runtime_dtype_container_ready=false`
and `promotion_allowed=false`; runtime DType/container work is blocked while
`contract_state.runtime_work_allowed=false`.

## Decision

Keep ParoQ4G256 `prototype-only` and mark quality as unverifiable until a
native `group_size=256` checkpoint exists. Do not add runtime DType/container
support from regrouped G128 evidence alone. The next legitimate artifact is a
true G256 source checkpoint or an explicit upstream refusal, followed by the
CPU probe, oracle comparison, payload ratio check, and KLD/PPL against
ParoQ4G128.
