# Raw F16/BF16 routed-MoE batched prefill diverges on every arch

**Status:** open defect, reproducer in tree, root cause not yet found.
**Found:** 2026-08-24, while making batched MoE prefill reachable on gfx1103.

## What is wrong

`forward_prefill_batch`'s grouped path-2 produces a decode distribution that
does not match the per-token `forward_scratch` reference when the routed
experts are raw `F16`/`BF16`. It is not close: `max_kld ~0.88`,
`max_abs_diff ~5.3`, `argmax_agree 0/4`.

The divergence is **flat in prefill length** — 0.75..0.94 across
prefill = 2, 4, 8, 16, 32, 64, with prefill = 2 being exactly one 16-slot
tile. That rules out accumulation and points at a structural error.

## Reproduce

```sh
HIPFIRE_TINYQUANT_FAMILIES=qwen3_5_moe_indexed ./tests/tiny-prefill-gate.sh
```

`qwen3_5_moe_indexed` (`moe_indexed_preset`: E=16, k_top=8, moe_inter=768,
2 layers) is the only in-tree fixture that reaches this path. `qwen3_5_moe` is
top-2-of-8 and is refused by `moe_prefill_topk_shape_supported` on every arch.

## What is already ruled out

| Hypothesis | Verdict | Evidence |
|---|---|---|
| gfx1103-specific | **No** | halo/gfx1151 fails identically on unmodified master: 0.87970 / 5.3399 vs nix1 0.87971 / 5.3300 |
| the grouped GEMM kernel | **No** | two independent kernels (gfx1151 WMMA, portable scalar) agree to 4 digits; and `examples/test_moe_grouped_wmma_f16_bf16` passes the portable kernel against a CPU reference on gfx1103 at max_abs ~2e-7, including the `x_row_div=8` and sparse-sentinel cases |
| batched prefill in general | **No** | the dense `qwen3_5` family passes on gfx1103 at max_kld 2e-6 (tol 1e-4) |
| accumulation over tokens | **No** | flat across prefill 2..64 |
| path-2 vs path-1 selection | **N/A** | `moe_grouped_gemm_path2_required_for_dtype` makes F16/BF16 path-2-only; `HIPFIRE_MOE_GROUPED_GEMM=0` does not move it |

## Where to look next

The raw-F16 arm differs from every MQ arm in exactly three places, all in
`prefill_chunk.rs`:

1. routed gate_up reads `pbs.x_norm_batch`, while MQ arms read
   `pbs.x_rot_batch` (correct in principle — raw weights need no rotation);
2. the down pre-activation is a plain `silu_mul_f32(gate, up, rot_batch)`
   instead of one of the fused silu+rotate variants;
3. the grouped GEMM call itself (ruled out above).

Note the shared expert also parks its down output in
`pbs.x_rot_batch.sub_offset(..)` before the routed section, but it accumulates
into `x_batch` immediately, so it is consumed before the routed path rebinds
that buffer.

## Blast radius

Any MoE model served with **unquantized** routed experts through batched
prefill. Quantized routed experts (MQ*/OQ*) take different arms and are
covered by the existing gates. On gfx1151 this fails **silently** — it returns
wrong numbers with no error. gfx1103 used to panic
(`gemm_f16_moe_grouped_wmma_gfx1151: only gfx1151 is supported`) until the
portable route landed, which is why the defect stayed invisible there.
