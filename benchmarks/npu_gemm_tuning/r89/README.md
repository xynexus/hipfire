# R89: local BF16 O staging

R89 is the capacity seam for fusing the post-attention residual and RMS norms
into the admitted R87 graph. It reuses the first 8 KiB of the final dead 10 KiB
activation FIFO object after QKV packing, adds one 4 KiB tail buffer, and stages
each 8x768 O tile in BF16 as three block-aligned 8x256 tiles. Six 2 KiB
block/half chunks then leave through the existing output FIFO in canonical
token-major order.

This rung adds no external tensor round trip and no kernel-side tensor-block
reorder. The loader-preconverted weight order and the separate kernel-parameter
correctness workaround remain unchanged. R89 deliberately adds only BF16
staging and emission; residual and norm math follow after this storage seam
passes its full hardware oracle.

```bash
./build_r89.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Result

The hardware oracle passes bit-exact stage and KV checks. O output has cosine
0.99999225 against the scalar reference and a 0.0625 maximum absolute error,
one BF16 quantum at the worst magnitude. Three fresh 100-command processes
measure 5.9044, 5.5931, and 5.7202 ms (median 5.7202 ms). Maximum even-core
text is 14,544 bytes and odd-core text remains 16,048 bytes. Admit the local storage
seam; this is not yet residual/norm execution.

Durable rows: `../results/r89-bf16-local-o-stage-20260713.csv`.
