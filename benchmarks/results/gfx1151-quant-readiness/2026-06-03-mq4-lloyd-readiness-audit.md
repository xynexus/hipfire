# MQ4-Lloyd gfx1151 Readiness Audit

- date: 2026-06-03
- arch: gfx1151
- control format: MQ4
- comparison format: MQ6
- scope: current MQ4-Lloyd promotion evidence

MQ4-Lloyd is a quality-over-MQ4 candidate with a larger payload
(`160 B/group` versus MQ4's `136 B/group`). Promotion therefore requires current
same-run evidence that it improves quality enough to justify the extra bytes,
plus coherence and perf evidence on the promoted fixtures.

## Current artifact inventory

A current 9B MQ4-Lloyd candidate artifact now exists:

- artifact: `/home/sadara/Models/hipfire-candidates/gfx1151-readiness/qwen3.5-9b-lloyd-mq4.hfq`
- loader symlink: `/home/sadara/.hipfire/models/qwen3.5-9b.mq4-lloyd`
- size: `6056190976` bytes
- md5: `61359a2af6804da8f16c09614d4692a9`
- sha256: `7e38d72b69e04d43789b4e0b5c99e313d5083224d6aeb236b0f3d9b9d5db00a1`

Producer command:

```bash
./target/release/hipfire-quantize \
  --input /home/sadara/Models/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --output /home/sadara/Models/hipfire-candidates/gfx1151-readiness/qwen3.5-9b-lloyd-mq4.hfq \
  --format mq4-lloyd \
  --allow-mq4-lloyd \
  --threads 8
```

The producer completed and emitted `MQ4G256Lloyd` records for the 2D weights,
with embeddings and conv1d kept on the expected safer paths. Current 27B and
A3B MQ4-Lloyd artifacts are still missing.

## Historical quality context

Historical `gfx1151` KV-Q8 prefill KLD rows from 2026-05-13 are reduced in
`benchmarks/results/gfx1151-quant-readiness/2026-06-03-lloyd-historical-2026-05-13-kld.json`.
Those rows are useful prior evidence, but they are not a current promotion
cohort because they do not include a clean same-run MQ4 control.

| row | chunks | mean KLD | p99 KLD | PPL |
|---|---:|---:|---:|---:|
| historical `qwen3.5-9b.mq4-lloyd-kvq8-c512` | 512 | 0.311426 | 18.6878 | 9.0850 |
| historical `qwen3.5-9b.mq4-lloyd-q8conv1d-kvq8-c512` | 512 | 0.251877 | 17.7834 | 8.8033 |
| historical `qwen3.5-9b.mq6-q8conv1d-kvq8-c512` | 512 | 0.0509511 | 8.67745 | 9.1862 |

Current 2026-06-03 MQ4/MQ6 BF16-referenced c512 context is recorded in
`benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-kld-c512.md`:

| row | chunks | mean KLD | p99 KLD | PPL |
|---|---:|---:|---:|---:|
| current `qwen3.5-9b.mq4-kvq8-c512` | 512 | 0.250007 | 17.3150 | 8.7953 |
| current `qwen3.5-9b.mq6-kvq8-c512` | 512 | 0.0509511 | 8.67745 | 9.1862 |

Do not treat the historical MQ4-Lloyd rows and current MQ4/MQ6 rows as a clean
same-run comparison. As context only, the best historical MQ4-Lloyd row does not
show a decisive quality win over the current MQ4 control, and MQ6 remains much
lower-KLD at the same c512 scale.

## Current decision

Keep MQ4-Lloyd research-gated. The current 9B artifact is no longer
artifact-blocked, but it failed qualitative coherence review.

`./scripts/coherence-gate.sh --full` wrote
`benchmarks/results/gfx1151-quant-readiness/2026-06-03-coherence-full-after-mq4-lloyd.md`.
The coherence battery itself reported no hard errors, but the MQ4-Lloyd row
emitted an obvious token attractor:

```text
!!!!!!!!!!!
```

The exact row was:

- model: `qwen3.5-9b.mq4-lloyd`
- prompt id: `reason-mq4-lloyd-9b`
- prompt md5: `fcfff9c745d58b218fc88be1639d759d`
- status: `OK` at the process/gate level
- stats: `tokens=11`, `prefill_tok_s=219.4`, `decode_tok_s=5.7`

This is a human-review coherence failure even though the daemon did not panic
or emit zero tokens. The combined gate command exited nonzero because the
separate pflash regression stage failed against its old baseline; that pflash
failure is not MQ4-Lloyd evidence, but it is why the command's final exit code
was `1`.

KLD was attempted with the prepared same-run 9B
`MQ4/MQ4-Lloyd/MQ6` cohort at `c512` and then `c64`, but both attempts were
interrupted before writing per-sequence evidence because the MQ4 control row did
not produce output within the turn. No KLD result is claimed from those
attempts. Given the visible attractor, do not spend perf-baseline time on this
artifact; fix producer/calibration first, then rerun coherence before KLD/perf.

## Required next gates

- Treat the current 9B MQ4-Lloyd artifact as coherence-rejected unless a
  producer or calibration fix changes the artifact hash.
- Generate or locate current canonical Qwen3.5 27B and A3B MQ4-Lloyd artifacts
  only after the 9B producer issue is understood.
- Run a same-run MQ4/MQ4-Lloyd/MQ6 BF16-referenced KLD cohort only after a
  coherent 9B candidate exists.
- Add fresh-process gfx1151 AR and DFlash/spec rows only if the quality cohort
  shows MQ4-Lloyd has a real quality reason to exist.
