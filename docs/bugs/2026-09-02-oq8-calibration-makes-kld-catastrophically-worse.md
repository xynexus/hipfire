# 8-bit calibration makes KLD *worse* — up to 1250x — and the baselines record it as expected

Status: **OPEN, measured.** Found 2026-09-02 while explaining why two re-baselined
`qwen3_5_moe oq8` rows moved the wrong way.

## The defect

`oqN+` adds a clip-search / SmoothQuant / AWQ activation-aware pass on top of
`oqN`. At 8 bits it should be neutral-to-helpful. On several families it is
catastrophically harmful, and the effect reproduces on two different GPU
architectures — so it is deterministic and algorithmic, not numeric noise:

| family | `oq8` | `oq8+-calib` | factor |
|---|---|---|---|
| qwen2 | 0.00002927 | 0.03554929 | **1214x worse** |
| dots_ocr | 0.00002790 | 0.02493017 | **894x worse** |
| qwen3_5_moe | 0.00169520 | 0.00491353 | 2.9x worse |
| gemma4_ple | 0.00084832 | 0.00158208 | 1.9x worse |
| lfm2_moe | 0.00031654 | 0.00058333 | 1.8x worse |

(gfx1103 rows; gfx1151 agrees to within a few percent on every one, and adds
`qwen3_5_moe_indexed` at 1.7x.)

For reference, on families where the pass behaves, 8-bit calibration lands
between 0.56x and 1.03x — i.e. mildly helpful to neutral, which is what it should
be. `qwen2` at 1214x is three orders of magnitude outside that band.

## Why nobody noticed

`tests/tiny-quant-gate.sh` PASSES all of these. The baselines were recorded with
the defect present, so the gate is asserting that a 1214x regression is the
expected value. It only fires on *drift from* the broken number.

This is the same failure shape as the `--n 32` hidden-probe bug in
`2026-09-02-kvarn-write-path-is-batch-invariant.md`: a gate that measures
faithfully and asserts the wrong thing.

## What this is NOT

**Not** the `+` vs `++` question. Those two are byte-identical at 8 bits for
several families, which looks suspicious and is not:

    qwen3_5_moe oq8+  vs oq8++  ->  artifacts BYTE-IDENTICAL
    qwen3_5_moe oq4+  vs oq4++  ->  artifacts differ

LDLQ demonstrably runs in the `++` build — 40 per-tensor
`ldlq+awq: ... OBS int8 + smooth` lines, `success=40 attempts=40 missing=0` —
and the `+` build runs none. At 8 bits the OBS correction is smaller than half an
LSB, so it rounds to the same codes; at 4 bits the LSB is 16x coarser and the
codes do move. So `oq8+ == oq8++` is expected physics, and the identical KLD in
rows 2 and 3 of a re-baseline is a symptom of the calibration bug above, not of
LDLQ being inert.

## Reproduce

    hipfire-quantize --emit-fixture qwen2 --out $W/src --seed 42
    hipfire-quantize --input $W/src --output $W/anchor.fp16.hfq --format fp16 --arch-id 1
    tiny_quant_probe collect --arch qwen2 --model $W/anchor.fp16.hfq --out $W/calib.hfq --len 128
    # then score oq8 vs oq8+ against the fp16 anchor

or read the two committed baseline rows, which already contain the result.

## Where to look

The `+` pass is the activation-aware clipping/scaling step, not LDLQ. Suspects,
in order:

1. The AWQ pre-scale (`awq_pre_scale_weights`) and its inverse. If the scales are
   applied to the weights but not consistently unwound into the stored scales,
   the error is multiplicative and would scale with how far the per-channel
   scales stray from 1 — which fits a 1000x blow-up on some families and 1.8x on
   others.
2. Clip-search selecting a clip range from calibration statistics that are
   mis-shaped for these architectures (qwen2 and dots_ocr are the two worst, and
   are unrelated families, so a shape assumption is more likely than a
   per-arch quirk).
3. Whether the pass is being applied to tensors it should skip (embeddings,
   lm_head, norms) on these families specifically.

## Do not re-baseline these

Recording the current numbers would make the regression permanent and silent.
The three `gemma4_moe` cells and everything listed here should stay red until the
pass is fixed or explicitly declared not-for-8-bit.
