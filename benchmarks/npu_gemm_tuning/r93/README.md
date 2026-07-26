# R93: canonical BF16 to resident R25 W4 activation ABI

R93 is the first native FFN bandwidth-first rung after R91/R92. It consumes
the canonical 256x768 BF16 pre-FFN-normalized state and writes R25's exact
718,848-byte activation payload: four row stripes, 27 blocks per stripe, and
6,656 bytes per block. Each 6,240-byte dynamic prefix is emitted directly to
the three N-macro consumer positions; the 416-byte padding tail remains zero.

This is mutable activation preparation, not immutable tensor-block layout
conversion. Weight block order remains loader/offline `.rdna2.hfp` work. The
kernel performs only layer-dependent AWQ scaling, the canonical signed
FWHT-256, int8 quantization, and physical DMA placement required by R25.
Nibble/lane swizzle remains local to the OQ4 compute kernel.

The physical source BO has 288 BF16 rows so each DMA object can carry a
3,072-byte two-row window; only its first row is consumed. This preserves the
tight canonical row order and lets the proven BF16-vector FWHT/sign path share
the same FIFO with 3,072-byte parameter records. It is an NPU DMA replay, not a
host or on-disk tensor reorder.

The existing kernel parameter remains the platform correctness workaround.
Tile-local storage is used here only as a measured capacity/performance choice.

```bash
./build_r93.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Result

All 589,824 replicated int8 values match the CPU FWHT/quantization oracle
exactly. Maximum scale error is `7e-9`; every 416-byte block tail and every
padded-row location remains zero. Core text is 7,856-9,040 bytes.

A smaller scalar/int8-sign implementation is rejected: a route-only image
wrote every aggregation chain and a load-only image read source/parameters,
but the full scalar image emitted corrupt scales and almost no quantized data.
Restoring R47's proven noinline BF16-vector sign/post-scale path fixes the
oracle. This is a code-generation result, not an LDS restriction.

Three fresh 100-command processes measure 4.0618, 4.1218, and 4.1117 ms
(median 4.1117 ms). The physical source-plus-output rate is only 0.263 GiB/s at
the median, so the standalone rung is transform/control limited, not external
memory-bandwidth limited. Admit the exact R25 input byte contract, but reject a
separate producer context. The next rung must fuse this preparation with the
first resident gate/up stage so its work overlaps weight DMA and no replicated
activation payload crosses an external context boundary.

Durable rows: `../results/r93-bf16-to-r25-activation-20260713.csv`.
