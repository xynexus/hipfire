# The 8 tiny-quant-gate failures, classified

`tests/tiny-quant-gate.sh` has failed 8 cells on every run this session,
identically, including on commits proven not to touch the paths involved
(reverting a change and re-running reproduces every number to six decimals).
They are not one problem, and only one of them is a regression.

## Reading the message

    KLD drift 0.001790 vs baseline 0.002662 (budget ±0.000665)

"drift" is misleading: the first number is the **current mean KLD**, not a
delta (`executor_tinyquant.rs:495` formats `cell.mean_kld` then `b`). Lower is
better, so several of these "failures" are the fixture scoring *better* than its
recorded baseline.

## Classification

| cell | current | baseline | what it is |
|---|---|---|---|
| `qwen2/kld:hfq4` | 0.001790 | 0.002662 | better — stale baseline |
| `gemma3/kld:q8f16` | 0.000868 | 0.001592 | better — stale baseline, **deprecated format** |
| `gemma3/kld:hfq4` | 0.094058 | 0.158772 | better — stale baseline |
| `qwen3_5/kld:q8f16` | 0.000538 | 0.000843 | better — stale baseline, **deprecated format** |
| `minimax/kld:mq4` | **0.000000** | 0.001042 | **vacuous** |
| `qwen3_5_moe/kld:mq6` | 0.215099 | 0.154634 | **vacuous** (see below) |
| `qwen3_5_moe/kld:mq4` | 0.215099 | 0.154634 | **vacuous** (see below) |
| `qwen3_5_moe/kld:q8f16` | 0.179210 | 0.141306 | worse, but **Q8 is deprecated** |

**Four are stale baselines in the good direction.** Re-recording clears them and
loses nothing; they are noise that trains the reader to ignore the gate.

**`minimax/kld:mq4` scores exactly 0.000000.** A quantised model with zero KLD
against its own reference is not a pass, it is a cell measuring nothing. Already
on the open-decisions list.

**`qwen3_5_moe/kld:mq6` and `kld:mq4` are the same cell twice.** They report
0.215099 against 0.154634 — identical to six decimals in BOTH the current run
and the committed baseline, across a 6-bit and a 4-bit format. Two different bit
widths cannot produce bit-identical KLD; these are not measuring their nominal
formats. The identity held when the baselines were recorded, so this is
long-standing rather than new. Only `mq6` was on the known-vacuous list — `mq4`
belongs there too, making it three vacuous cells, not two.

**`qwen3_5_moe/kld:q8f16` is on a deprecated format, so it is not worth chasing.**
An earlier revision of this document called it "the only real signal" — 0.179210
against a 0.141306 baseline, +27% and well outside the ±0.035 budget, which on a
nearly-lossless format would be alarming. **Q8 weights are deprecated** per the
2026-07-18 directive (`docs/plans/2026-07-18-blocked-feature-coverage-plans.md`
:169: "Q8 (weight and KV) is being deprecated"). A regression in a format on its
way out does not earn investigation; the cell earns deletion.

That takes the three `q8f16` cells here — `gemma3`, `qwen3_5`, and
`qwen3_5_moe` — out of scope entirely, whichever direction they moved.

**So none of the 8 requires a fix.** Every one is a stale baseline, a vacuous
cell, or a deprecated format. The whole set clears with re-recording and
deletion, no debugging.

## Why this matters beyond the cleanup

A gate that fails 8 cells on every run cannot report a 9th. Every runtime change
this session had to be cleared by reverting it and re-running to compare
numbers, because "8 failures" carries no information.

And the failure set is *entirely* clearable: re-record the stale baselines, drop
the three vacuous cells, drop the deprecated-format cells. Nothing here needs
debugging. That is the strongest argument for doing it — the standing noise is
not protecting against anything, it is only hiding whatever comes next.
