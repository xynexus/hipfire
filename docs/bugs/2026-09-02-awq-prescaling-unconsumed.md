# AWQ pre-scaling is applied at quantization time and not undone at inference

Status: **ROOT-CAUSED 2026-09-02; guard added, per-arch fixes OPEN.** This is the
mechanism behind
`2026-09-02-oq8-calibration-makes-kld-catastrophically-worse.md`.

## The contract

AWQ multiplies each weight column by `s[j]` at quantization time and REQUIRES the
runtime to divide the activation by the same `s` before the GEMM:
`(W·s)·(x/s) = W·x`. The quantizer writes `s` as a 1-D F16 sidecar named
`<weight_stem>.awq_scale.weight`. If nothing applies it, the model computes
`(W·s)·x`, and the error scales with how far `s` strays from 1 — invisible on flat
activations, catastrophic on peaky ones.

## Proof that AWQ is the whole story

Tiny qwen2 fixture, same weights, AWQ the only variable:

    plain oq8                       mean_kld 0.00002762
    oq8+  (AWQ alpha=0.55, default) mean_kld 0.02524465   914x worse
    oq8+  --awq-alpha 0             mean_kld 0.00002762   EXACTLY plain

Neutralising AWQ restores plain-oq8 KLD to the digit. Nothing else in the `+`
pass contributes.

## It is not only the F2 projections

The quantizer scales all seven projections. `o_proj`/`down_proj` are architecturally
harder — they are not fed by a fused RMSNorm, which is where the `x/s` division
happens — and `HIPFIRE_AWQ_F1_ONLY=1` already exists to exclude them. But
excluding them only halves the damage:

    oq8+ F2 (default, all 7)        0.02524465   914x
    oq8+ F1-only (q,k,v,gate,up)    0.00710044   257x
    plain                           0.00002762

So on qwen2 the scales are not being applied to ANY projection, not just the two
awkward ones.

## Per-arch status

`load_awq_scale` consumption, by architecture:

| arch | consumes sidecars | oq8 calibration effect |
|---|---|---|
| qwen35 | yes — its own `load_awq_scale_for`, and the dispatcher selects the AWQ kernel | fine |
| llama | yes — 3 direct `load_awq_scale` calls | 0.75x (helps) |
| gemma3 / gemma4 | via the shared `transformer_loader` | ~neutral |
| **qwen2** | has the FIELD and reads it at 2 sites, but the scales never arrive | **1214x worse** |
| **dots_ocr** | no AWQ awareness at all | **894x worse** |

## Guard added

`hipfire_runtime::hfq::warn_if_awq_sidecars_unconsumed` compares the number of
`.awq_scale.weight` tensors an artifact carries against the number the runtime
actually applied, and warns loudly when the artifact was pre-scaled but the
inverse never ran. Called at every `LoadedModel` return so no arch path skips it.

It is a warning, not a hard error: artifacts already in the field carry these
sidecars, and refusing to load them would strand working setups on a defect they
did not cause. The message names the escape hatch (`--awq-alpha 0`).

**The first version of this guard fired a FALSE ALARM on
`qwen3.5-0.8b--oq4++.hfq`** — 24 sidecars, "0 applied" — because `qwen35` carries
its own copy of the sidecar lookup and the runtime's counter could not see it.
That model is healthy. Arch-local loaders now call
`hfq::note_awq_sidecar_consumed()`; verified the warning is silent on that model
and the counter reaches 24. Any future arch-local loader must do the same, or it
will be reported as broken while working correctly.

## Still open

- **qwen2 and dots_ocr serve AWQ-scaled weights uncorrected.** Either wire the
  sidecars through their forward paths, or stop emitting AWQ for them.
- The `tiny-quant` baselines still record the broken KLDs as expected, so the
  gate asserts the regression. They must not be re-recorded before the fix; see
  the companion doc.
