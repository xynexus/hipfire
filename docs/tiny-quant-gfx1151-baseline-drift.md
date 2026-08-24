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

## AMENDMENT 2026-08-25: the qwen3_5_moe rows moved again

Re-measured on gfx1151. Six of the eight cells still reproduce this document
exactly. The three `qwen3_5_moe` rows do NOT:

| cell | this doc | measured 2026-08-25 |
|---|---|---|
| `qwen3_5_moe mq4` | 0.215099 | **0.223334** |
| `qwen3_5_moe mq6` | 0.215099 | **0.223334** |
| `qwen3_5_moe q8f16` | 0.179210 | **0.179465** |

So the "measured gfx1151 now equals the gfx1103 baseline to six digits"
argument below no longer holds for mq4/mq6 — that agreement was the evidence
for the stale-baseline diagnosis, and it has decayed. The diagnosis may still
be right, but this pair is no longer proof of it.

NOT a regression from the 2026-08-24/25 spec-decode work: the same three values
were measured at `88ae2b8f3` (before that work) and at `890d350c0` (after) in a
scratch worktree, and they agree to every digit printed. Whatever moved them
landed in the ~1000 commits between this document and `88ae2b8f3` — several MoE
residency changes are in that range (`92e81f22d` routed experts stay compact
resident, `928d0f8cb` indexed-compact feed, `cc532499d` per-expert stride
table). Bisecting that range is the open work; until then do not re-record
these three rows, because recording an unexplained number enshrines it.

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

Re-record the gfx1151 rows (`tiny-quant-gate.sh --record`) — EXCEPT the two
cells below, which should be fixed or dropped rather than recorded. Recording
a vacuous cell enshrines it: it will pass forever and can never report the
regression it names.

## Two cells compare an artifact against itself

Both were traced by rebuilding the fixtures and diffing the artifacts. Neither
is a quantizer defect — both are deliberate policy meeting a gate that assumes
`--format` decides everything.

### `minimax/kld:mq4` measures nothing

`--format q8f16` and `--format mq4` on the minimax fixture emit the SAME dtype
histogram: `{F16: 11, Q8_0: 11, MQ4G256: 48, BF16: 1}`. The 48 expert tensors
are `MQ4G256` in both. `main.rs:9944` says so outright — *"Expert format by
--format: mq2-lloyd, mq6 (oracle check), else mq4 (MQ4G256, validated
baseline)"* — so `q8f16` lands in `else` and gets 4-bit experts.

The two artifacts differ in 18,860 bytes, all of it metadata plus lossless
bf16-recode packing, so they are numerically identical and the KLD is exactly
0. The "near-full-precision anchor" is not near-full-precision: it carries the
same 4-bit experts as the candidate.

The 2026-07-22 baseline of 0.00104174 was NOT signal. It predates the recoded
embed/lm_head fix, when the two runs decoded that tensor differently — the
nonzero reading came from the bug, and fixing the bug revealed the cell had
nothing in it.

### `qwen3_5_moe/kld:mq6` duplicates `kld:mq4`

The two artifacts differ by 18 bytes, all inside the metadata string
(`"quant_format":"mq4"` vs `"mq6"`). Both emit `MQ6G256` for all 59 quantized
tensors, and the mq4 artifact is LARGER than its own q8f16 anchor
(5,106,402 B vs 5,008,100 B).

This is K-map promotion working as designed: `main.rs:10171` promotes routed
experts to MQ6 under `(kmap_promote && use_mq4g256)`. On this fixture every
G256-eligible tensor is a routed expert, so mq4 and mq6 converge completely.
The dense `qwen3_5` fixture behaves correctly — mq4 emits `MQ4G256` and mq6
emits `MQ6G256`, 1.5 MB of differing payload — which is what isolates this to
the MoE split path.

Identical baselines on BOTH arches (gfx1103 and gfx1151 each record
0.21510008 for mq4 and mq6) are the fingerprint: it has always been one cell
wearing two names.

## Not caused by hipfire-train

`hipfire-train` is a leaf crate — only `hipfire-eval` and `hipfire-daemon`
depend on it, and neither routes quantization numerics through it. The gate
escalates on it because the affected-family heuristic is coarse, not because
the crate participates.
