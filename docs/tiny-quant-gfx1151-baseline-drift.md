# gfx1151 tiny-quant baselines are stale after the recoded-tensor fix

**Status:** diagnosed, not acted on — re-recording baselines is a judgment call.

`tests/tiny-quant-gate.sh` fails 8 cells on gfx1151. The failures are stale
baselines, not a regression, and the evidence is arithmetic rather than
argument.

## What the gate reports

```
fail qwen3_5_moe/kld:q8f16  drift 0.179210 vs baseline 0.141306 (budget ±0.035327)
fail qwen3_5_moe/kld:mq6    drift 0.215099 vs baseline 0.154634 (budget ±0.038658)
fail qwen3_5_moe/kld:mq4    drift 0.215099 vs baseline 0.154634 (budget ±0.038658)
fail qwen2/kld:hfq4         drift 0.001790 vs baseline 0.002662
fail gemma3/kld:q8f16       drift 0.000868 vs baseline 0.001592
fail gemma3/kld:hfq4        drift 0.094058 vs baseline 0.158772
fail minimax/kld:mq4        drift 0.000000 vs baseline 0.001042
fail qwen3_5/kld:q8f16      drift 0.000538 vs baseline 0.000843
```

Five of the eight moved DOWN — the quantized model now tracks the reference
better than the recorded baseline says it should. Only `qwen3_5_moe` moved up.

## Why this is a stale baseline

The measured gfx1151 numbers now equal the **gfx1103** baselines exactly:

| cell | measured on gfx1151 | gfx1103 baseline | gfx1151 baseline |
|---|---|---|---|
| `qwen3_5_moe mq4` | 0.215099 | 0.21510008 | 0.15463363 |
| `qwen3_5_moe mq6` | 0.215099 | 0.21510008 | 0.15463363 |
| `qwen3_5_moe q8f16` | 0.179210 | 0.17921833 | 0.14130644 |

Two architectures agreeing to six digits is what a correct computation looks
like. Two architectures disagreeing by 39% was the anomaly.

`tests/tiny-quant-baselines.txt` was last recorded in `753df2b27`
(2026-07-22). The recoded-tensor read sweep landed after it, including
`226bb66b2` — *"qwen35: decode recoded embed/lm_head — one failure mode was
SILENT"*. Before that fix, gfx1151 read losslessly-recoded embed/lm_head
tensors incorrectly and reported a KLD that was too LOW, because the reference
and the quantized run were both wrong in the same direction. The 2026-07-22
gfx1151 numbers therefore measured a broken path; gfx1103 never took it, which
is why its baseline was already the honest one.

The higher `qwen3_5_moe` KLD is the truthful number, not a regression.

## What to do

Re-record the gfx1151 rows (`tiny-quant-gate.sh --record`). Not done here
because re-recording admission evidence bakes in whatever is currently true —
if any of these cells is ALSO carrying a real regression, recording hides it
permanently, and the gate stops being able to tell anyone. The five
improvements are individually plausible as fix effects, but "plausible" is not
the standard for a baseline.

Worth checking before recording: `minimax mq4` now reads exactly 0.000000,
down from 0.00104174. A quantized model matching its own reference to the last
digit is more likely a collapsed or short-circuited eval path than a perfect
quantizer.

## Not caused by hipfire-train

`hipfire-train` is a leaf crate — only `hipfire-eval` and `hipfire-daemon`
depend on it, and neither routes quantization numerics through it. The gate
escalates on it because the affected-family heuristic is coarse, not because
the crate participates.
