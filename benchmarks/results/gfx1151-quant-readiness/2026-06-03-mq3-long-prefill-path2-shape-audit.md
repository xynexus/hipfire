# MQ3 Long-Prefill Path2 Shape Audit

- date: 2026-06-03
- arch: gfx1151
- scope: no-GPU contract for MQ3 A3B grouped MoE long-prefill routing

This audit records the no-GPU contract added for the production-shaped MQ3
A3B batched-prefill path. It does not claim artifact-backed promotion evidence.

## Covered Contract

Rust test:

```text
cargo test -p hipfire-arch-qwen35 --lib moe_prefill
```

New no-GPU test:

```text
qwen35::tests::moe_prefill_mq3_long_prefill_path2_shape_is_production_shaped
```

The test locks the following `gfx1151` MQ3 MoE prefill invariants:

- MQ3 routed experts remain admitted for `gfx1151` when router and scalar gate
  stay on Q8.
- MQ3 forces grouped path2 even when `HIPFIRE_MOE_GROUPED_GEMM=0` because no
  indexed MQ3 fallback is wired.
- For a full long-prefill chunk (`N=256`, `K_TOP=8`, `num_experts=256`), the
  grouped scatter shape is:
  - `total_slots = 2048`
  - `m_total_bound = 5888`
  - `m_total_bound % 16 == 0`
- Gate/up grouped GEMM consumes `x_rot_batch [N x dim]` with `x_row_div = 8`.
- Down grouped GEMM consumes `rot_batch [N*K_TOP x mi]` with `x_row_div = 1`.

## Interpretation

This closes the missing no-GPU route-shape coverage for the MQ3 long-prefill
batched path called out in the readiness matrix. The test specifically guards
the production-shaped `x_row_div=K_TOP` gate/up path that synthetic
single-token tests do not exercise.

This is not a promotion-grade runtime result. MQ3 A3B still needs
artifact-backed long-prefill coherence/perf rows, and the existing A3B KLD and
DFlash/spec lanes remain blocked on missing matching references and paired
A3B draft sidecars.
