# MQ2 / MQ2-Lloyd gfx1151 Readiness Audit

- date: 2026-06-03T07:57:36+08:00
- arch: gfx1151
- commit: fab9d2bc88d2de7b5febfa9ec8afed80b6700557
- branch: qwen35-native-mtp
- control format: MQ4
- benchmark binary: `target/release/examples/bench_mq2g256_lloyd_moe_4w`
- benchmark binary md5: `58b5845b9f507717a331b8bb09931bfb`
- structured 4-warp artifact: `benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq2-lloyd-4w-bench.json`

## MQ2 Dense Decision

Plain MQ2 has current local dense rejection artifacts for Qwen3.5 0.8B, 4B,
and 9B. SHA-256 provenance is recorded in
`benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq2-artifact-provenance.json`.

The bounded sweep in
`benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq2-sweep.md` loaded and
generated on gfx1151 without hard runtime errors, but every dense prompt
collapsed:

- Paris rows did not answer Paris.
- Code rows did not emit a Python function.
- Sheep rows did not produce the final answer.
- Longform rows produced garbled non-answers.

Decision: keep MQ2 quality-rejected for dense text. Runtime OK status is useful
only as a fallback/admission smoke; the sweep output is quality rejection
evidence, not promotion evidence.

## MQ2-Lloyd Artifact State

No current MQ2-Lloyd `.hfq` artifact was found under:

- `/home/sadara/Models/hipfire-candidates/gfx1151-readiness`
- `/home/sadara/.hipfire/models`
- broader `/home/sadara/Models` filename search

DeepSeek V4 is the intended routed-expert specialty fixture, but the local
`deepseek-v4-flash.mq2lloyd` HFQ symlink required by `scripts/coherence-gate.sh`
is not present. Therefore MQ2-Lloyd has no current artifact-backed coherence,
KLD/PPL, or model-level perf evidence in this readiness directory.

Decision: keep MQ2-Lloyd dense-text rejected and routed-expert research-only.
Do not use dense Qwen prompts to promote it. Reconsider only after an A3B or
DeepSeek routed-expert artifact is available and passes coherence.

## 4-Warp Kernel A/B

Command:

```bash
cargo run --release -p rdna-compute --example bench_mq2g256_lloyd_moe_4w
```

Scope: synthetic DeepSeek V4 MoE hot-path shapes. This is kernel-only evidence;
it does not prove model quality or end-to-end model throughput.

| shape | correctness | k2 baseline us | 4-warp us | speedup |
|---|---|---:|---:|---:|
| gate/up B=128, M=2048, K=4096, m_total=768 | OK, max_abs=0, bad=0, nan=0 | 1636.1 | 1806.4 | 0.91x |
| gate/up B=256, M=2048, K=4096, m_total=1536 | OK, max_abs=0, bad=0, nan=0 | 3401.6 | 3672.5 | 0.93x |
| gate/up B=1024, M=2048, K=4096, m_total=6144 | OK, max_abs=0, bad=0, nan=0 | 14724.0 | 15147.7 | 0.97x |
| down B=128, M=4096, K=2048, m_total=768 | OK, max_abs=0, bad=0, nan=0 | 1704.7 | 1825.9 | 0.93x |
| down B=256, M=4096, K=2048, m_total=1536 | OK, max_abs=0, bad=0, nan=0 | 3530.3 | 3793.8 | 0.93x |
| down B=1024, M=4096, K=2048, m_total=6144 | OK, max_abs=0, bad=0, nan=0 | 14875.6 | 15028.8 | 0.99x |

The 4-warp kernel is correctness-clean against the current k2 baseline on these
synthetic shapes, but it is slower across all measured rows. The structured
JSON companion records `all_correct=true`, `all_candidate_slower_than_baseline=true`,
`promote_4w_default=false`, `default_kernel_change_allowed=false`,
`model_level_promotion_allowed=false`, and `bandwidth_bottleneck_proven=false`.
Do not make `HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W=1` a promoted default from this
evidence.

This result also does not prove that the model-level MQ2-Lloyd decode issue is
pure DDR bandwidth. Effective-bandwidth shortcuts overcount inactive experts,
and the current bottleneck can include launch overhead, scalar GEMV occupancy,
codebook unpack/lookup cost, and non-MQ2 decode work. Treat selected-expert
batching across tokens and launch reduction/fusion as the next performance
experiments before more packing-only work.

## Required Next Gates

- Keep MQ2 dense-text rejected unless a new calibration first clears bounded
  coherence and KLD/PPL against MQ4 or Q8.
- Generate or locate a current MQ2-Lloyd A3B or DeepSeek HFQ artifact before
  model-level routed-expert validation.
- Run A3B/DeepSeek coherence before any model-level MQ2-Lloyd speed claim.
- Keep the 4-warp kernel opt-in/research unless a future model-backed benchmark
  beats the current k2 route and preserves coherence.
- Batch selected experts across tokens and reduce/fuse small decode launches
  before treating packing as the main lever.
