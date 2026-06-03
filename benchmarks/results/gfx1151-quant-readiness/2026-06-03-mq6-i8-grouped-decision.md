# MQ6 gfx1151 i8 Grouped-Path Decision

- date: 2026-06-03
- repo: `/home/sadara/.hipfire/src`
- branch: `qwen35-native-mtp`
- commit: `fab9d2bc`
- arch: `gfx1151`
- decision: do not port an MQ6-specific i8 grouped-MMQ path now

## Current MQ6 MoE Path

MQ6 A3B routed-expert prefill is already on the grouped path-2 surface:

- `moe_grouped_gemm_supported_for_dtype(DType::MQ6G256, "gfx1151")` admits MQ6.
- `prefill_moe_ffn_body_batched` dispatches MQ6 gate/up and down through
  `gemm_hfq6g256_moe_grouped_wmma`.
- `cargo test -p hipfire-arch-qwen35 --lib moe_prefill` covers MQ6 admission
  and path-2 routing policy.

The existing i8 grouped-MMQ kernels are HFQ4/Paro-shaped research or opt-in
surfaces. They are not a current MQ6 implementation, and an MQ6 port would need
new HFQ6 layout handling rather than flipping an existing default.

## Evidence

AR medians from `2026-06-02-mq6-ar-perf.json`:

| pair | MQ4 prefill tok/s | MQ6 prefill tok/s | MQ4 decode tok/s | MQ6 decode tok/s |
|---|---:|---:|---:|---:|
| Qwen3.5 9B dense | 229.5 | 100.0 | 43.4 | 30.5 |
| Qwen3.5 27B dense | 37.1 | 16.0 | 14.1 | 9.9 |
| Qwen3.5 35B-A3B | 56.1 | 227.9 | 54.4 | 51.3 |
| Qwen3.6 35B-A3B | 55.9 | 230.8 | 53.1 | 50.4 |

Dense MQ6 is perf-blocked, but that blocker is not the grouped MoE prefill
path. The A3B rows already show MQ6 prefill roughly `4x` faster than MQ4 while
decode remains close to MQ4.

DFlash medians from `2026-06-03-mq6-dflash-r3.json` remain dense target-side
rows only:

| prompt | MQ4 decode tok/s | MQ6 decode tok/s | MQ4 tau | MQ6 tau |
|---|---:|---:|---:|---:|
| prose | 6.51 | 3.11 | 1.3875 | 1.3415 |
| code | 30.24 | 14.84 | 10.0 | 10.0 |

These DFlash rows identify a dense target-side throughput issue, not a routed
expert grouped-path issue.

## Consequence

For MQ6, spend engineering time on dense MQ6 throughput or A3B promotion
packaging/evidence before attempting an MQ6 i8 grouped-MMQ port. Keep i8
grouped-MMQ work in the HFQ4/Paro research lane unless a future MQ6 A3B profile
shows grouped HFQ6 WMMA as the dominant remaining bottleneck.
