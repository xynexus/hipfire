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
| `gemma3/kld:q8f16` | 0.000868 | 0.001592 | better — stale baseline |
| `gemma3/kld:hfq4` | 0.094058 | 0.158772 | better — stale baseline |
| `qwen3_5/kld:q8f16` | 0.000538 | 0.000843 | better — stale baseline |
| `minimax/kld:mq4` | **0.000000** | 0.001042 | **vacuous** |
| `qwen3_5_moe/kld:mq6` | 0.215099 | 0.154634 | **vacuous** (see below) |
| `qwen3_5_moe/kld:mq4` | 0.215099 | 0.154634 | **vacuous** (see below) |
| `qwen3_5_moe/kld:q8f16` | 0.179210 | 0.141306 | **worse — the only real signal** |

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

**`qwen3_5_moe/kld:q8f16` is the one worth investigating**: 0.179210 against a
0.141306 baseline, +27%, well outside the ±0.035 budget. Q8F16 is nearly
lossless, so a 27% KLD rise on the MoE fixture is a real signal. It is unrelated
to anything in this session — it reproduces on unmodified checkouts — but it is
the single cell where the gate is doing its job and being ignored because it is
buried among seven that are not.

## Why this matters beyond the cleanup

A gate that fails 8 cells on every run cannot report a 9th. Every runtime change
this session had to be cleared by reverting it and re-running to compare
numbers, because "8 failures" carries no information. Re-recording the four
stale cells and fixing or dropping the three vacuous ones would leave one
genuine failure visible — and make the gate able to catch the next one.
