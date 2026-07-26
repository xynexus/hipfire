# R87: depth-two O-weight shim staging

R87 keeps the admitted R85 O kernel and complete graph unchanged, but gives
the shim-to-memory-tile O-weight FIFO two 16 KiB slots. The per-core broadcast
FIFO remains depth one, so tile-local memory, compute order, loader-preconverted
layout, and the kernel-parameter correctness workaround are unchanged.

```bash
./build_r87.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Result

The full hardware oracle passes. Three fresh 100-command processes measure
5.6078, 5.8531, and 5.7450 ms (median 5.7450 ms). Depth two is a small,
low-risk improvement over R85 and is admitted as the O-weight staging baseline.

Durable rows: `../results/r87-output-weight-shim-depth2-20260713.csv`.
