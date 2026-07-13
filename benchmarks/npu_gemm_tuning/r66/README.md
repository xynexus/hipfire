# R66 — R65 staging to canonical Q/KV

R66 isolates the consumer of R65's five-role 10 KiB inline records. It reuses
the proven R29 Q/K/V headnorm and RoPE pack functions and emits the exact R27
physical buffers: 393,216 bytes of Q and 262,144 bytes of single-replay K/V.
The runtime ABI is only staging/Q/KV; no immutable weights or tensor-block
conversion are involved.

```bash
./build_r66.sh
cargo run -p hipfire-xdna --release \
  --example npu_embedding_qkv_pack_verify -- \
  "$HOME/.hipfire/npu/embgemma_r66_r65_stage_to_qkv_m256" 100
```

Acquire `hipfire lock` before hardware runs.

The locked gate passes the established R28 oracle: Q
cosine 0.99999121 and max error 0.0078125, K cosine 0.99999156 and max error
0.0078125, and bit-exact V. Three fresh 100-command processes measure 0.9511,
0.9915, and 0.9984 ms (median 0.9915 ms). This is not performance-admitted:
R65 plus this isolated stage would be about 1.48 ms before attention. The graph
broadcasts each inline record to all columns and serializes the four core-pair
packers, whereas R28's compact joined input activates all four pairs together.
R67 should therefore change only the mutable staging/consumer schedule toward
the R28 joined layout before attempting resident integration.

Maximum linked core text is 10,432 bytes. Durable rows:
`../results/r66-r65-stage-to-qkv-20260713.csv`.
