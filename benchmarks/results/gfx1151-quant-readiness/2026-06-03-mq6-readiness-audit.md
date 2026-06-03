# MQ6 gfx1151 Readiness Audit (2026-06-03)

- repo: `/home/sadara/.hipfire/src`
- branch: `qwen35-native-mtp`
- commit: `fab9d2bc`
- arch: `gfx1151`
- control: MQ4

## Candidate Artifacts

`benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-artifact-provenance.json`
records canonical readiness symlinks and SHA-256 hashes for:

- `qwen3.5-9b-mq6.hfq` -> `qwen3.5-9b.mq6`
- `qwen3.5-27b-mq6.hfq` -> `qwen3.5-27b.mq6`
- `qwen3.5-35b-a3b-mq6.hfq` -> `qwen3.5-35b-a3b.mq6`
- `qwen3.6-35b-a3b-mq6.hfq` -> `qwen3.6-35b-a3b.mq6`

## Runtime Surface

| surface | status | evidence |
|---|---|---|
| Dense decode | wired | AR perf rows for 9B and 27B complete with no hard errors. |
| Dense prefill | wired | AR and DFlash rows report prefill throughput for 9B/27B. |
| MoE prefill | wired | `cargo test -p hipfire-arch-qwen35 --lib moe_prefill` covers dtype+arch routing and grouped path-2 policy. |
| MoE decode | wired | A3B AR rows complete for Qwen3.5 and Qwen3.6 MQ6. |
| DFlash/spec target verify | partial | 27B dense target-side rows pass with the MQ4 draft; no MQ6 draft lane is claimed. |

## Quality Evidence

- Full coherence report:
  `benchmarks/results/gfx1151-quant-readiness/2026-06-02-mq6-coherence-md5.md`
  includes MQ6 9B, 27B, Qwen3.5 A3B, and Qwen3.6 A3B rows with no hard runtime
  errors.
- Dense 9B PPL:
  `benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-ppl.json` scored
  MQ6 `PPL=11.5880` versus MQ4 `PPL=12.0679` on the bounded Wikitext2 slice.
- A3B PPL:
  `2026-06-03-mq6-a3b-ppl.json` and
  `2026-06-03-mq6-qwen36-a3b-ppl.json` provide bounded Qwen3.5/Qwen3.6 A3B
  quality rows.
- BF16-referenced 9B KLD:
  `2026-06-03-mq6-kld-c256.json` scored MQ6 mean KLD `0.047336` versus MQ4
  mean KLD `0.238527` at `256` chunks. The c20 and c64 artifacts show the
  same direction.
  `2026-06-03-mq6-kld-c512.json` scored MQ6 mean KLD `0.050951` versus MQ4
  mean KLD `0.250007` at `512` chunks with the same BF16 reference and
  `eval_hipfire` md5.

Current quality gap: no c512 KLD gap remains for the dense 9B checkpoint. The
remaining broad-promotion blocker is dense perf, not BF16-referenced KLD
coverage.

## Perf Evidence

AR medians from `2026-06-02-mq6-ar-perf.json`:

| pair | MQ4 prefill tok/s | MQ6 prefill tok/s | MQ4 decode tok/s | MQ6 decode tok/s | decision |
|---|---:|---:|---:|---:|---|
| Qwen3.5 9B dense | 229.5 | 100.0 | 43.4 | 30.5 | dense perf-blocked |
| Qwen3.5 27B dense | 37.1 | 16.0 | 14.1 | 9.9 | dense perf-blocked |
| Qwen3.5 35B-A3B | 56.1 | 227.9 | 54.4 | 51.3 | A3B-first candidate |
| Qwen3.6 35B-A3B | 55.9 | 230.8 | 53.1 | 50.4 | A3B-first candidate |

DFlash/spec medians from `2026-06-03-mq6-dflash-r3.json`:

| prompt | MQ4 decode tok/s | MQ6 decode tok/s | MQ4 tau | MQ6 tau | decision |
|---|---:|---:|---:|---:|---|
| prose | 6.51 | 3.11 | 1.3875 | 1.3415 | dense DFlash perf-blocked |
| code | 30.24 | 14.84 | 10.0 | 10.0 | dense DFlash perf-blocked |

Acceptance/tau is close to MQ4 on these DFlash rows, but target-side MQ6
throughput is roughly 2x slower than MQ4. This blocks a broad dense promotion
claim even though target-side correctness did not fail.

## Grouped i8 Decision

`2026-06-03-mq6-i8-grouped-decision.md` records the current grouped-path
decision. MQ6 A3B routed-expert prefill already dispatches through the HFQ6
grouped WMMA sister on `gfx1151`; the existing i8 grouped-MMQ kernels are
HFQ4/Paro-shaped research or opt-in surfaces, not a ready MQ6 implementation.

Do not port an MQ6-specific i8 grouped-MMQ path now. The measured MQ6 blocker
is dense throughput: dense 9B/27B AR rows and dense 27B DFlash rows lag MQ4,
while A3B MQ6 prefill is already roughly `4x` faster than MQ4 and decode is
close. Reopen MQ6 i8 grouped-MMQ only if a future A3B profile shows grouped
HFQ6 WMMA as the dominant remaining bottleneck.

## Decision

MQ6 remains the closest non-MQ4 format, but the promotion scope should be
A3B-first. Dense MQ6 has good bounded quality evidence and working runtime
paths, but current dense AR and DFlash performance is materially behind MQ4.
Before any broad promotion claim, either produce a dense perf fix or scope the
claim to A3B/MoE where MQ6 prefill is substantially faster and decode is close
to the MQ4 control.
