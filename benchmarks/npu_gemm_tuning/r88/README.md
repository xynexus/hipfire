# R88: depth-three O-weight shim staging

R88 raises only R87's shim-to-memory-tile O-weight FIFO from two to three
16 KiB slots. Core-local depth, graph math, loader-preconverted layout, output
scatter, and the kernel-parameter correctness workaround remain unchanged.

## Result

The full hardware oracle passes. Three fresh 100-command processes measure
5.7539, 5.6671, and 5.7343 ms (median 5.7343 ms), only 0.0107 ms faster than
R87. Reject depth three as saturated and retain the simpler depth-two staging.

Durable rows: `../results/r88-output-weight-shim-depth3-20260713.csv`.

```bash
./build_r88.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.
