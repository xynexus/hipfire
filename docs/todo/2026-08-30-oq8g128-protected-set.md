# TODO: finish Oq8G128 for the protected set (needs a G128 fused rmsnorm+rotate)

**Status:** open (proposed 2026-08-30). Infrastructure landed; default blocked on
one kernel.

## Why Oq8G128

Under an `oq8` bulk the protected set — routers, attention projections, residual
writers — has nowhere to go. `Oq8G256` *is* the bulk, so promoting there deletes
the protected set; the deprecated `Q8F16` it used to sit on is what this work
removes. That left source precision (bf16) as the only thing above the bulk, at
2.0 B/weight.

`Oq8G128` is the right answer, and measurably so. Reconstruction RMSE on gaussian
weights with 1-in-257 outliers (`codecs.rs::oq8_group_size_reconstruction_report`):

    Oq8G256    4.493e-3   (0.673% of rms)   8.125 b/w
    Oq8G128    4.110e-3   (0.615% of rms)   8.125 b/w   0.915x vs G256
    Q8F16(G32) 5.243e-3   (0.785% of rms)   8.5   b/w   -- the deprecated home

So Oq8G128 is **21.6% better than the Q8F16 it replaces while using FEWER bits**,
and 8.5% better than the G256 bulk. The FWHT rotation more than pays for the
coarser grouping — which is why the naive "G32 beats G128" intuition is wrong
here, and why bf16 at twice the bytes is not the right trade.

## What already landed

- `gemv_oq8_grouped` takes group 128. It was G256-only because its packing assumed
  `group/32 = 8 int8 per lane = two aligned int32 loads`; G128 is 4 per lane and
  one load. The arms are selected by a WAVE-UNIFORM branch, and the G256 arm is
  kept verbatim because the two accumulate in a different ORDER — merging them
  would have re-rounded every recorded oq8 baseline. Verified by
  `parity_gemv_oq8_g128` (G128 rel 2.3e-7, G256 control unchanged, plus a negative
  control proving the test is not vacuous).
- `QuantType::Oq8G128` (54), `DType::Oq8G128`, `quantize_oq8g128`,
  `oq8_combined_g`, the loader arms (shared `transformer_loader` + qwen35),
  `GemvOq8G128Prerotated`, the registry entry, and `weight_gemv` / `weight_gemm`
  arms. `gemm_oq8_grouped_wmma` needed nothing (it already took `group`), nor did
  `quantize_act_oq8` (asserts only `group % 32`), nor `mq_rotate_x_128` (it already
  offsets by `blockIdx.y * K`; only a batched *dispatch* was missing).
- A real bug found on the way: `pipeline/mod.rs` passed `RotationVariant::Plain`
  unconditionally, i.e. always the FWHT-256 tables, regardless of the weight's
  rotation plan. Any G128 dtype reaching that path was rotated by a transform its
  dequant never inverts — coherent-looking garbage, not an error. Now follows
  `dtype_rotation_plan`.

Archs whose weights rotate per-tensor already serve it correctly: gemma4 (via
`weight_gemv`'s direct arm), nemotron, zaya.

## What blocks the default

qwen35's lowered executor fuses rmsnorm + rotate into ONE kernel
(`fused_rmsnorm_rotate_mq`) and every attention projection consumes that single
FWHT-256-rotated activation. Giving one of those weights the 128 basis needs:

1. a **G128 fused rmsnorm+rotate** (`RmsnormRotateMqG128` and its AWQ/batched
   siblings — none of the `RmsnormRotateMq*` keys have a G128 form), and
2. a way to **split the shared activation**, since a layer would then need both a
   256-rotated and a 128-rotated copy of the same input.

Without both, qwen35 serves garbage (KLD 0.83 vs 0.0196). So the emit stays on
bf16 and `Oq8G128` is wired but not selected.

Do NOT work around this by keying the quantizer on `arch_id` — the capability
layer exists so archs declare behaviour (`AGENTS.md`). If a partial rollout is
wanted, it should be a declared cap ("weights rotate per-tensor"), not a family
name in `cli.rs`.

## Also worth knowing

- `rotate_x_mq_128_for` takes the non-AWQ path — there is no AWQ sidecar support
  at G128. Fine while `oq8+`/`oq8++` protected tensors do not carry one; check
  before enabling calibrated paths.
- LDLQ has no G128 form either (`cli.rs` refuses `--ldlq` at group 128, because
  `oqplus_compact_ldlq_pack` emits 256-element blocks).
- The tiny-quant KLD numbers could NOT adjudicate this: the same change scored
  6.9x better on one family and 2x worse on another, on random-init fixtures where
  KLD measures perturbation sensitivity rather than quality. The reconstruction
  test above is the trustworthy signal. See
  `docs/todo/2026-08-30-tiny-fixture-training-and-qat.md`.
