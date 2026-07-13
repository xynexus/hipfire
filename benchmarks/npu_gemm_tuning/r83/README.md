# R83: compact paired projection, pack, and attention

R83 retains R82's single-context topology and exact observable boundaries but
targets the measured 6,032-byte odd-core program overflow. It replaces R15's
duplicated init/accumulate functions with R70's exact single-group ABI and
obtains the 16-block attention trip count through a non-LTO helper so Peano
cannot clone all 32 attention-block calls into the core driver. The same
boundary preserves each three-slice projection-finish loop instead of emitting
36 static finish calls on the full-role cores.

This is a program-image change only. Projection, packing, and attention math,
the offline `.rdna2.hfp` pair order, nibble swizzle, and the separate
kernel-parameter correctness workaround are unchanged.

```bash
./build_r83.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

`build_projection_probe.sh` isolates the single-group and dynamic-finish
changes at R81's pre-attention boundary when the byte oracle needs to identify
which size lever changed stage ordering.

## Result

R83 links and packages. The maximum odd-core image is 15,888 bytes, 6,528
bytes smaller than R82 and 496 bytes below the 16 KiB limit; even cores remain
at 10,912 bytes. Objdump confirms two attention-block call sites and 12
projection-finish call sites on full-role odd cores, versus R82's 32 and 36.

The rebuilt layer-0 verifier passes projection stage, Q, KV, and attention
byte-for-byte. Three fresh 100-command processes measure 4.1666, 4.0493, and
4.0535 ms (median 4.0535 ms). R83 is admitted as the first fitting paired
projection/pack/attention capacity image, but rejected as a speed baseline: it
is 6.8% slower than R78 and 24.3% slower than R76.

Durable rows: `../results/r83-compact-paired-projection-pack-attention-20260713.csv`.
