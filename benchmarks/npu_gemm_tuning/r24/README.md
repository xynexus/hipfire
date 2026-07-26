# R24: arbitrary mixed Opus pack/down

R24 extends the exact R22 W4 combined activation-pack/down schedule with the
sparse W8 overlays used by compact mixed Opus blocks. The host-facing Opus
decoder already converts canonical `(index, replacement)` entries to signed
`(index, delta)` chunks; R24 consumes those decoded chunks after each resident
W4 group MMUL and reconstructs the result with the same activation and weight
scales.

The cache is specialized by overlay count because compact Opus formats fix one
count for every 256-weight block. Counts 1 through 62 are accepted, covering
the complete `oq4.125` through `oq7.9375` mixed range. The weight FIFO grows
from 16 KiB only when its W4 data, scales, AWQ/sign parameters, and decoded
overlay slab no longer fit.

```bash
benchmarks/npu_gemm_tuning/r24/r24_cache.sh 3
cargo run --release -p hipfire-xdna --example npu_pack_down_verify -- \
  ~/.hipfire/npu/embgemma_aie2p_token_pack_down_mixed_o3_m256_k1152_n768 50
```

Hardware parity is exact at both a common and the maximum overlay count:

- 3 overlays: zero mismatches over 196,608 outputs, maximum absolute error
  `5.7e-6`, 50-iteration dispatch average `8.9671 ms`.
- 62 overlays: zero mismatches over 196,608 outputs, maximum absolute error
  `1.53e-5`, one measured dispatch `36.1533 ms`.

This closes format coverage at the combined down-projection seam, but it is not
a performance win. Scalar sparse activation gathers dominate even at three
overlays, consistent with the earlier standalone sparse3 result. R18-to-R24
in-array GeGLU streaming, attention, full-model throughput, and package energy
remain open.
