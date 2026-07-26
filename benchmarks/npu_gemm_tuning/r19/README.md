# R19 canonical down-activation preprocessing

R19 is an exact all-32-core AIE2P baseline for the canonical Opus activation
contract immediately before EmbeddingGemma's `K=1152` down projection. For
each M256 row it:

1. pads K to 1280 and divides by the AWQ activation scales;
2. applies the seed-42 signs, five independent Hadamard-256 transforms, and
   the seed-1042 signs with the canonical `1/16` normalization;
3. computes one symmetric int8 scale per 256-value group; and
4. emits 1280 quantized bytes plus five scales in a 1312-byte row record.

The physical row record combines quantized values and scales because separate
output FIFOs exceed the memory tile's input-DMA channel budget. Each compute
tile retains a 256-float scratch buffer and processes eight rows.

Peano does not preserve the scalar division/rounding contract when the AWQ
loop is emitted naively. R19 uses a fixed-width `noinline` helper for exact AWQ
division and makes `roundf` semantics explicit as a signed half bias followed
by a toward-zero saturated conversion.

Build artifacts stay under `~/.hipfire/npu`:

```sh
bash benchmarks/npu_gemm_tuning/r19/r19_cache.sh
cargo run -p hipfire-xdna --release --example npu_fwht_quant_verify -- \
  ~/.hipfire/npu/embgemma_aie2p_resident_awq_fwht_quant_m256_k1280 100
```

Three independent 100-iteration hardware runs produced exact parity for all
327,680 int8 values, maximum scale error `7e-9`, and dispatches of 6.9405,
6.9061, and 6.8772 ms (median 6.9061 ms). This is a standalone canonical
preprocessing kernel, not a fused GeGLU-to-down pipeline or a full FFN result.
At about 37k rows/s it is correct but much too expensive to repeat as a scalar
stage in every layer; the next slice must vectorize/fuse it with R18 output and
the resident down projection.
