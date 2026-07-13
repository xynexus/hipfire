# R85: reuse each O activation tile across N tiles

R85 changes only the even-core direct-O group kernel. R84 reloads the same
`4x8` BF16 activation tile once for each of four `8`-column weight tiles. R85
keeps four MMUL accumulators live and loads that activation once per K tile,
preserving the K accumulation order and loader-preconverted weight stream.

The graph, DMA schedule, attention handoff, output scatter, and existing
kernel-parameter correctness workaround are unchanged from R84.

```bash
./build_r85.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Result

The full hardware oracle passes exact stage and KV checks and the fused
attention-to-O numerical threshold. Three 100-command processes measure
5.7945, 5.8026, and 5.6847 ms (median 5.7945 ms), 0.1417 ms or 2.4% faster
than R84. The output object grows from 1,248 to 1,760 text bytes but the full
graph still links. Admit R85 as the direct-O speed/capacity baseline.

Durable rows: `../results/r85-output-reuse-a-20260713.csv`.
