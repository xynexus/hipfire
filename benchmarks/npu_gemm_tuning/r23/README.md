# R23 parallel token pack + W8 down

R23 extends R22's exact three-row packing and acyclic direct-stream token
broadcasts to the scaled W8 down projection. It implements the W8 activation
layout (`LM=3`, `MR=8`, 48 columns per compute column) and retains each packed
activation group while applying both `N=384` macro weight blocks.

Two implementation details are required for the AIE2P limits:

- The original separate W8 init/accumulate functions overflowed 128 KiB
  program memory when linked with the vector packer. R23 uses one compact W8
  MMUL function with a runtime accumulate flag.
- Byte-at-a-time W8 activation insertion measured 3.90 ms. Each aligned
  eight-byte tile is now copied as two 32-bit words, reducing the repeated-pack
  version to 1.69 ms before N-macro reuse.

The final schedule packs and broadcasts each group once per M-macro, then
consumes two resident weight blocks into two output FIFO slots:

```sh
bash benchmarks/npu_gemm_tuning/r23/r23_cache.sh
cargo run -p hipfire-xdna --release --example npu_pack_down_verify -- \
  ~/.hipfire/npu/embgemma_aie2p_token_pack_down_w8_m256_k1152_n768 100
```

Three independent 100-iteration runs report zero mismatches across 196,608
outputs, maximum absolute error `1.4e-6`, and dispatches of 1.1953, 1.2100, and
0.9956 ms (median 1.1953 ms). R23 is combined OQ8 preprocessing/down evidence,
not a complete FFN: R18-to-R23 in-array GeGLU streaming, arbitrary mixed
overlays, attention, full-model throughput, and package energy remain open.
