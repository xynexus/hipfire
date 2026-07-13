# R80: paired compact-W4 projection

R80 consumes R79's pair-major HFP layout. Each odd core acquires one activation
block, consumes the two intact adjacent-column weight blocks, and accumulates
two 24x96 projection stripes. Even cores run no QKV projection code. Both
stripes scatter through the odd column's existing output FIFO into the exact
R65 inline stage ABI.

This rung contains projection only. Q/K/V packing, attention, output projection,
and norms are intentionally absent until every R65 stage byte matches.

```bash
cargo run --release -p hipfire-coexistence -- npu pair-hfp \
  --in "$HOME/.hipfire/npu/prepacked/EmbeddingGemma-300M.npu.oq4.layer-0.qkv.oq4.whole-scaled.rdna2.hfp" \
  --out "$HOME/.hipfire/npu/prepacked/EmbeddingGemma-300M.npu.oq4.layer-0.qkv.oq4.paired-whole-scaled.rdna2.hfp"
./build_r80.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Result

The first output schedule queued six odd-shim tasks per outblock and timed out.
Restoring R65's one-slice-per-channel await/free cadence removes the deadlock.
The paired graph then matches all 327,680 projected BF16 values bit-for-bit,
with zero attention-tail and padding corruption.

Three fresh processes with two warmups and three timed commands measure
0.818433, 0.833471, and 0.789289 ms (median 0.818433 ms). This is 67.7% slower
than R65's eight-column 0.487964-ms median, but below the naive 2x serialization
bound. Maximum odd-core text is 11,872 bytes; even cores have no program image.

R80 is admitted as a capacity topology, not the fastest projection. It leaves
4,512 bytes of odd-core program store and the entire even core for the next
pack/attention and direct-output integration steps.

Durable rows: `../results/r80-paired-w4-projection-20260713.csv`.
