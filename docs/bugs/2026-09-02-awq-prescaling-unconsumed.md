# AWQ pre-scaling is applied at quantization time and not undone at inference

Status: **FIXED 2026-09-02** for qwen2 and dots_ocr; guard added; wiring the sidecars through those forward paths remains an option, not a requirement. This is the
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

## Fix: AWQ defaults OFF for architectures that cannot undo it

`AWQ_UNCONSUMED_ARCHS = [ARCH_ID_QWEN2, ARCH_ID_DOTS_OCR]` in the quantizer's
per-arch alpha resolution — the same hook nemotron_h already uses for its own
reason. Those archs default to `alpha = 0`, print why, and an explicit
`--awq-alpha` still overrides so the experiment stays available for whoever wires
the sidecars through those forward paths.

Producing an artifact the runtime can serve correctly beats producing one that is
quietly wrong.

**Verified on the real arch_id 7 path** (the gate routes qwen2 to 7, not the
LLaMA-default 1):

    plain oq8               0.00002762
    oq8+ (new default)      0.00002762   <- identical to plain
    oq8+ --awq-alpha 0.55   0.02524465   <- override still works

Every calibrated cell for both families improved:

| cell | before | after | |
|---|---|---|---|
| qwen2 oq8+ | 0.035549 | 0.000029 | 1226x better |
| qwen2 oq8++ | 0.035552 | 0.000027 | 1317x better |
| qwen2 oq4+ | 0.037011 | 0.004534 | 8.2x |
| qwen2 oq4++ | 0.036999 | 0.004177 | 8.9x |
| qwen2 oq4.25++ | 0.037386 | 0.004185 | 8.9x |
| dots_ocr oq8+ | 0.024930 | 0.000028 | 890x better |
| dots_ocr oq8++ | 0.024944 | 0.000028 | 891x better |
| dots_ocr oq4+ | 0.029493 | 0.007578 | 3.9x |
| dots_ocr oq4++ | 0.029322 | 0.007736 | 3.8x |
| dots_ocr oq4.25++ | 0.028880 | 0.005028 | 5.7x |

Those ten baseline rows are re-recorded — this time legitimately, because the
regression is fixed rather than absorbed. Only the ten `-calib` rows were
applied; `--record` also rewrote the uncalibrated rows for these families and
those were restored from a backup, as with the qwen3.5 re-baseline.

`HIPFIRE_TINYQUANT_FAMILIES=qwen2,dots_ocr ./tests/tiny-quant-gate.sh` -> PASS.

## Still open## Still open

- **The 1.8-1.9x cases are untouched**: `gemma4_ple`, `lfm2_moe`, and
  `qwen3_5_moe` still show 8-bit calibration making KLD modestly worse. Those
  archs DO consume sidecars, so the cause is different and possibly legitimate
  (AWQ is not guaranteed to help at 8 bits). Not gated, not re-baselined.
- Wiring the sidecars through the qwen2 and dots_ocr forward paths would let them
  benefit from AWQ rather than merely avoid its damage. The `--awq-alpha`
  override is the way to test such a build.
