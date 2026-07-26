# R21 vector activation pack + W4 down projection

R21 is the first single-dispatch AIE2P chain that consumes floating-point
EmbeddingGemma GeGLU rows, applies the canonical Opus AWQ/FWHT/int8 contract,
and immediately feeds the resulting groups into the resident scaled W4 down
MMUL. The 1312-byte R19/R20 row record is no longer exported.

Each 16 KiB resident weight block uses its otherwise-unused tail for that
group's 3,072-byte AWQ/sign payload. This keeps every compute tile within its
two input-DMA-channel limit: group activations plus weights. One core column
packs each 24-row activation stripe, uses the packed block locally, and sends
it directly to the other seven compute columns without entering an already
saturated memory tile.

The current host argument is an internal group stream: each object contains
the active 256-float slice, not the full padded row. A future R18-to-R21 graph
must produce that stream in-array; R21 does not yet prove the complete FFN
boundary.

```sh
bash benchmarks/npu_gemm_tuning/r21/r21_cache.sh
cargo run -p hipfire-xdna --release --example npu_pack_down_verify -- \
  ~/.hipfire/npu/embgemma_aie2p_vector_pack_down_w4_m256_k1152_n768 100
```

Hardware parity is exact within scaled-f32 accumulation tolerance: zero
mismatches across 196,608 W4 outputs and maximum absolute error `1.4e-6`.
Three 100-iteration runs measured 2.6385, 2.5797, and 2.4033 ms (median
2.5797 ms). A redundant per-core pack experiment reached 2.4386 ms in one
20-iteration run; direct sharing is architecturally cleaner but does not solve
the serial 24-row pack/synchronization latency. The next schedule must stripe
those rows across columns and gather one activation block before the MMUL.
