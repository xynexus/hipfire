# R86: two-accumulator O activation reuse

R86 balances R85's activation-load reuse against accumulator pressure. It
reuses each `4x8` activation tile across two adjacent N tiles instead of four,
while preserving the R84 graph, stream order, K accumulation order, output
scatter, and kernel-parameter correctness workaround.

```bash
./build_r86.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Result

The full hardware oracle passes. Three 100-command processes measure 5.9589,
5.9203, and 5.7962 ms (median 5.9203 ms). This recovers only 0.3% over R84
and is 2.2% slower than R85. Reject R86 and retain R85's four-way activation
reuse.

Durable rows: `../results/r86-output-reuse-a2-20260713.csv`.
