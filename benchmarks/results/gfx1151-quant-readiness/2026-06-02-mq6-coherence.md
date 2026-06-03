# gfx1151 MQ6 Coherence Evidence — 2026-06-02

Command:

```bash
./scripts/coherence-gate.sh --full
```

Engine state:

- branch: `qwen35-native-mtp`
- commit reported by gate: `fab9d2bc`
- report path: `/tmp/coherence-20260602-231403.md`
- mode: `full`

Result summary:

- Coherence phase: no hard errors.
- Combined command: failed after coherence because `pflash-gate` reported
  `0/12 rows clean` and exited `1`.
- The pflash regression is tracked separately from MQ6 coherence; it does not
  invalidate the MQ6 rows below, but it means the full command was not green.
- A newer md5-stamped full coherence run is recorded at
  `benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-coherence-md5.md`.
  Use that report as the current MQ6 coherence evidence.

MQ6 rows added and exercised:

| row | status | prefill tok/s | decode tok/s | note |
|---|---:|---:|---:|---|
| `qwen3.5-9b.mq6` / `reason-mq6` | OK | 98.8 | 30.5 | answered the sheep riddle with final answer 9 |
| `qwen3.5-27b.mq6` / `cap-mq6-27b` | OK | 15.9 | 9.9 | no hard error; capped at 80 tokens while still in reasoning |
| `qwen3.5-35b-a3b.mq6` / `moe-mq6-sheep` | OK | 227.9 | 51.1 | answered the sheep riddle with final answer 9 |
| `qwen3.6-35b-a3b.mq6` / `moe36-mq6-sheep` | OK | 230.7 | 50.4 | answered the sheep riddle with final answer 9 |

MQ4 control rows from the same report:

| row | status | prefill tok/s | decode tok/s | note |
|---|---:|---:|---:|---|
| `qwen3.5-9b.mq4` / `reason` | OK | 231.4 | 43.3 | same sheep prompt as 9B MQ6 |
| `qwen3.5-35b-a3b.mq4` / `moe-sheep` | OK | 56.0 | 54.3 | same sheep prompt as 35B-A3B MQ6 |
| `qwen3.6-35b-a3b.mq4` / `moe36-sheep` | OK | 55.5 | 53.0 | same sheep prompt as 3.6 A3B MQ6 |

Open MQ6 promotion gaps:

- This is coherence/hard-error evidence, not KLD/PPL evidence.
- This is single-run evidence, not a perf baseline with fresh-process medians.
- Artifact SHA-256 hashes are recorded separately in
  `benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-artifact-provenance.json`.
- Prompt md5 and binary md5 were not recorded by this historical gate run;
  `scripts/coherence-gate.sh` now records those fields for future runs.
- DFlash target-side MQ6 evidence remains open.
- The pflash regression stage failed and should be triaged separately before a
  broad release claim.
