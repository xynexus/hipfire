# R91: zero-copy residual/norm to FFN handoff

R91 changes only R90's final DMA destination. Once projection and Q/K/V packing
have completed, the staging prefix is dead; the six canonical BF16 output
chunks overwrite that prefix instead of appending a second tensor. A resident
R35 FFN context can import the same dma-buf and consume token-major BF16 at
offset zero without a host copy or tensor-block reorder.

All kernel math, Q/K/V pack ordering, O weights, residual/norm parameter order,
tile-local 8+4 KiB staging, and the independent kernel-parameter correctness
workaround are unchanged from R90. This is intentionally a two-context
zero-copy boundary: both R90 core roles and the R35 FFN image are already near
their 16 KiB program limits, so claiming an appended one-image FFN would be
physically unsupported.

```bash
./build_r91.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Result

A literal offset-zero output corrupted the next sustained command. The paired
projection rewrites 327,680 stage bytes, but the following 65,536 bytes contain
immutable headnorm/RoPE packing state. Overwriting the complete 393,216-byte
prefix therefore made the next pack phase consume stale tail state. R91 fixes
this with a DMA/loader-only layout: the complete 2,457,600-byte stage ABI moves
forward by 393,216 bytes, while the canonical BF16 FFN handoff remains at
offset zero. All 48 handoff BDs are below 393,216 and all 280 stage BDs are at
or above it. No kernel-side tensor-block reorder is used.

Both ordinary XDNA SHMEM and PRIME-imported GTT controls pass the R90 tail gate
and exact KV oracle across sustained commands. The GTT pages feed the resident
canonical-BF16 R35 FFN directly, with no host write or copy between producer
and consumer. Its synthetic dense-OQ8 oracle reaches cosine 0.99989925 and
maximum absolute error 0.0118408.

Three fresh 100-command processes measure producer medians of 6.3727 ms,
isolated FFN medians of 9.7654 ms, and alternating zero-copy chain medians of
22.1772 ms. The unexplained difference is 6.0391 ms, or 27.2% of the chain,
matching R37's context-alternation penalty. The boundary processes 11,543 M256
rows/s for one layer, but this is not 11,543 end-to-end input tokens/s: repeating
it for the encoder cannot meet the model target. Admit R91 for correctness and
reject the two-context cadence as the end-to-end execution strategy.

Durable rows: `../results/r91-zero-copy-ffn-handoff-20260713.csv`.
