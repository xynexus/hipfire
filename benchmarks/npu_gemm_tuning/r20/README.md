# R20 vector canonical down-activation preprocessing

R20 preserves R19's exact M256 `K=1152`/padded-`K=1280` Opus activation
contract while moving its hot loops onto AIE2P vectors:

- 16-lane AWQ divide and pre-sign;
- filter/interleave butterflies for FWHT strides 1, 2, 4, and 8;
- 16-lane add/sub butterflies for strides 16 through 128;
- vector post-sign, absolute maximum reduction, normalization, and int8
  conversion.

The vector reciprocal paths in `aie::div` were admitted only after comparing
all 327,680 physical int8 values and every group scale against R19's exact CPU
oracle. The tested AWQ/activation corpus remains byte-exact, with maximum scale
error `1.1e-8`.

Three independent 100-iteration hardware runs measured 0.3106, 0.3221, and
0.3246 ms (median 0.3221 ms), about 795k rows/s and 21.4 times faster than the
R19 scalar median.

Build artifacts stay under `~/.hipfire/npu`:

```sh
bash benchmarks/npu_gemm_tuning/r20/r20_cache.sh
cargo run -p hipfire-xdna --release --example npu_fwht_quant_verify -- \
  ~/.hipfire/npu/embgemma_aie2p_vector_awq_fwht_quant_m256_k1280 100
```

This remains a standalone preprocessing kernel rather than a fused
GeGLU-to-down FFN. R20 exists to establish that the canonical pack need not be
the scalar bottleneck before its vectors are moved into the down schedule.
