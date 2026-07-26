# R74: two-query-group attention replay

R74 returns to the exact R71 graph and attacks the larger attention feed cost
directly. Each attention core retains two 4-KiB query-pair buffers and four
accumulator/stat sets. It loads two query groups, streams the 262-KiB KV plane
once, updates both groups, and then emits four head outputs. Across six query
groups this reduces complete KV replays and their DMA tasks from six to three.

Q and KV remain externally observable so the R71 byte oracle is unchanged.
Weights retain their offline `.rdna2.hfp` order. The added kernel parameter is
the correctness workaround; tile-memory use here is an independent capacity
and performance decision.

```bash
./build_r74.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Result

The first 4-KiB-stack build exceeded the 64-KiB active-tile allocation by
1,184 bytes. Reusing R73's measured 2-KiB stack setting makes the graph fit at
64,672 bytes, leaving 864 bytes. Bank-aware placement falls back to sequential
allocation. Maximum linked core text is 15,248 bytes.

Projection stage, Q, KV, and final attention match the R70/R27 references
byte-for-byte. Three fresh primed 100-command processes measure 3.4496, 3.4242,
and 3.2867 ms (median 3.4242 ms).

R74 is rejected by the speed ratchet: its median is 3.4% slower than R71's
3.3118 ms, despite one run reaching 3.2867 ms. Halving full KV replays and DMA
task count does not repay the extra live Q/accumulator state. Together with
R72/R73, this points away from additional tile-resident buffering and toward
the projection/pack/attention phase schedule and core utilization.

Durable rows: `../results/r74-qgroup2-kv-replay-20260713.csv`.
