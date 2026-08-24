# Raw F16/BF16 routed-MoE batched prefill diverges on every arch

**Status:** FIXED in `0bbbfd08f`. Kept as the write-up of the defect, the
hunt, and the two traps it set. See "The fix" below.
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

## ROOT CAUSE (found 2026-08-24)

**`LayerWeights::DeltaNetMoe` never got the full-precision dispatch arms that
`LayerWeights::DeltaNet` has.** It is not a MoE bug at all — the MoE body is
innocent. The defect is in the linear-attention layer that precedes it.

The dense LA branch was migrated to the lowered program, whose matcher
(`prefill_lowered.rs:2327-2328`) carries explicit arms:

```rust
let is_f32 = matches!(layer.wqkv.gpu_dtype, DType::F32);
let is_f16 = matches!(layer.wqkv.gpu_dtype, DType::F16 | DType::BF16);
```

The MoE LA branch still runs the legacy hand-rolled chain in
`prefill_chunk.rs`, which enumerates PARO / 6-bit / Q8 / Q8-wmma /
OqCompactG256 and sends **everything else** — F16 and BF16 included — to
`FusedQkvzaHfq4G256`. Raw halves are then decoded as `[f16 scale][128 nibbles]`
HFQ4 blocks. No error; just wrong numbers. The `wo` chain a few hundred lines
below has the identical shape, defaulting to `GemmHfq4G256Residual`.

This is the third recorded instance of one failure mode. The comment above the
`OqCompactG256` qkv arm documents the second, in the same words: "Two layouts
sharing no field, and no error: just wrong numbers."

### The measurement chain

Dumped via the in-tree `HIPFIRE_DUMP_HIDDEN` instrument (`"*_b"` tags in the
batched path, `"pertoken"` in the reference), fp16 `qwen3_5_moe_indexed`,
prefill 6, tokens `[87,87,87,82,8,14]`:

| tensor | rows across DIFFERENT tokens | verdict |
|---|---|---|
| `x_batch` (layer input) | differ | correct |
| rmsnorm output (`x_rot_batch`) | differ | correct |
| `dn_qkv_batch` (wqkv projection) | **BIT-IDENTICAL** | **broken here** |
| `dn_q/k/v`, `alpha`, `beta` | bit-identical | downstream of the above |
| router logits | bit-identical | downstream |

Every token's prefill therefore produced token 0's activations, which is why
the divergence is flat in prefill length and identical on both archs.

### Two traps this hunt fell into, recorded so the next one does not

1. **`dn_quant=FP32` in the `kernel-trace` line is the DeltaNet STATE quant,
   not the weight dtype.** Reading it as the weight dtype sent this
   investigation after `gemm_f32_register_tiled` (which is correct). The
   fixture has NO F32 tensors — `hipfire inspect` reports F16/BF16/Bf16Lut3
   only. Read dtypes from the artifact, not from that field.
2. **`synthetic_tokens(seed=42)` begins `[87, 87, 87, 82, ...]`.** At
   `--prefill 2` or `3` the prefill tokens are all the SAME id, so identical
   per-row activations are expected and prove nothing. Any row-invariance test
   must use a prefill long enough to reach a distinct token id.

## The fix (landed `0bbbfd08f`)

`DeltaNetMoe` and `FullAttnMoe` were folded onto the same lowered super-ops
their dense siblings use, via borrowed attention-half views (`DnLaWeights` /
`FaAttnWeights`) that both layer variants expose as `.la()` / `.fa()`. Net
-1834 lines. One attention implementation now, so there is nothing left to
drift.

Result on gfx1103, `qwen3_5_moe_indexed` fp16, tolerance 1e-4:

| prefill | max_kld | argmax |
|---|---|---|
| 6 | 3.1e-6 | 4/4 |
| 16 | 2.2e-5 | 4/4 |
| 64 | 5.4e-6 | 4/4 |
| 128 | 2.3e-5 | 4/4 |

from 0.88 / 0-of-4. Dense cells unchanged bit-for-bit. `--corrupt-kv-prefix`
still moves it to 0.34, so the check can still fail.

### An earlier arm-by-arm attempt, and why it failed

Adding full-precision arms to the legacy chains one at a time went 0.88 -> 0.21
(QKVZA) -> 0.51 (plus wo) and never approached tolerance. The coupled sites in
the DeltaNetMoe body alone are four — QKVZA GEMM, wo GEMM, wo input rotation,
and the LA preamble's rmsnorm/FWHT choice — and `FullAttnMoe` is a fifth
through eighth. Patching a subset leaves the activation basis inconsistent
between them. Consolidation was the only tractable shape.

## Gate gap this exposed

`tiny-prefill-gate.sh` reports these cells SKIP rather than OK. Its path check
infers "the batched path never ran" from the batched and reference
recurrent-state hashes being EQUAL, and they now legitimately are — both paths
run the same per-token GDN kernels. That heuristic is evaluated BEFORE the KLD
comparison, so during this hunt it also reported `max_kld 0.377, argmax 0/4` as
INCONCLUSIVE rather than FAIL. It needs a positive probe that the batched path
executed (the `moe_topk_ok` / `[features] moe_routed` trace already proves it)
instead of inferring absence from equal state.

## Correction to the first commit message

`75b718181` says `gemm_raw_moe_grouped_portable` "had no callers". That is
wrong: `hipfire_dispatch::pipeline::run_grouped_moe_gemm` already dispatched it
(with the same `F16 if arch == "gfx1151"` / else-portable split), and that is
the path the per-token DECODE reference takes. The grep that produced the claim
was scoped to `hipfire-arch-qwen35` and `hipfire-runtime` and missed it. The fix
itself stands — the batched PREFILL call sites bypassed that shared dispatcher
and called the gfx1151 entry point directly — but the right long-term shape is
for prefill to delegate to `run_grouped_moe_gemm` too, rather than keep a second
dispatcher (`gemm_raw_moe_grouped`) beside it.

## Blast radius

Any MoE model served with **unquantized** routed experts through batched
prefill. Quantized routed experts (MQ*/OQ*) take different arms and are
covered by the existing gates. On gfx1151 this fails **silently** — it returns
wrong numbers with no error. gfx1103 used to panic
(`gemm_f16_moe_grouped_wmma_gfx1151: only gfx1151 is supported`) until the
portable route landed, which is why the defect stayed invisible there.
